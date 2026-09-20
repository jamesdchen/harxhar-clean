"""Run one session of the afternoon delta-hedged nearest-OTM SPX 0DTE straddle.

The book (audit 2026-09-18, sections 4-7):

    preflight   load the premium ledger (seed + live), read TODAY's regime
                off it -- proposal 50's DELEVERAGING CANDIDATE, full / half /
                flat against the expanding lagged percentiles of the trailing
                realized variance -- and pick TODAY's entry clock from it
                causally, or fall back to the fixed afternoon clock while the
                selector warms up.  A zero multiplier refuses the day here.
    entry       sell the nearest-OTM SPXW 0DTE straddle (the *body*) at that
                clock, sized by the STRESS table -- capital x fraction over
                the dollar loss of one hedged straddle in a 5 % last-bar
                jump, never by margin -- then hedge its package delta with
                ES for the bulk and MES for the remainder
    :30 ..      re-quote the body, re-invert the total volatility from the
    15:30       package mid, CORRECT it by the trailing per-clock
                realized-over-implied factor, recompute the delta, move the
                lots, journal the residual
    16:00:00    flatten ES and MES (passive five seconds, then cross) and let
                the options cash-settle; journal the provisional settlement
                and write ``pending_settlement.json``
    next am     ``--reconcile`` reads the official settlement, finalises the
                day's P&L and appends the session's records to the live
                premium ledger

``--terminal flatten`` keeps the old variant: buy the body back at
``--exit-clock`` and flatten the futures in the same minute.  It is the
optional variant now.

``--date YYYY-MM-DD`` replays a recorded session through ``FakeBroker`` with
a simulated clock and no network at all; that is the smoke test.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from live.ibkr.broker import (
    Broker,
    BrokerError,
    FakeBroker,
    Fill,
    IBBroker,
    QuoteHealth,
)
from live.ibkr.config import Config, parse_clock, typed_live_ack
from live.ibkr.hedge import rebalance_lots, residual_delta_lots, target_lots
from live.ibkr import journal as journal_module
from live.ibkr.journal import DaySummary, Journal
from live.ibkr.premium_ledger import ClockRecord, PremiumLedger
from live.ibkr.pricing import (
    corrected_total_vol,
    hours_to_close,
    invert_total_vol,
    package_delta,
)
from live.ibkr.selector import pick_entry_clock
from live.ibkr.sizing import contracts_for, stress_loss_per_contract
from live.ibkr.strikes import Body, Quote, pick_nearest_otm_reason, pick_wings

NAN = float("nan")

#: A clock reached more than this many seconds late is an alert: the research
#: showed a one-bar lag zeroes the edge.
LATE_ALERT_S = 60.0

PENDING_NAME = "pending_settlement.json"

#: The journal kind the deleveraging rule writes: the regime it read, the
#: thresholds it read it against, and the multiplier it returned.  The
#: journal owns the whitelist of kinds; this registers one more against it
#: rather than editing that whitelist from two places.
REGIME_KIND = "regime"
if REGIME_KIND not in journal_module.EVENT_KINDS:
    journal_module.EVENT_KINDS = tuple(journal_module.EVENT_KINDS) + (REGIME_KIND,)  # type: ignore[assignment]

#: The head of the refusal a zero multiplier raises, so a supervisor grepping
#: the journal or the summary matches one string.
DELEVER_FLAT_REASON = "deleveraging: regime above the "


class Abort(RuntimeError):
    """Preflight refused the session; nothing was traded."""


@dataclass
class PackageQuote:
    bid: float = NAN
    ask: float = NAN
    mid: float = NAN
    legs: list[Quote] = field(default_factory=list)
    ok: bool = False
    health: QuoteHealth = field(default_factory=QuoteHealth)

    @property
    def spread(self) -> float:
        return float(self.ask) - float(self.bid)

    @property
    def tradeable(self) -> bool:
        return bool(self.ok and self.health.ok)


@dataclass
class DayState:
    entry_clock: str = ""
    body: Body | None = None
    wings: tuple[float, float] | None = None
    n_entry: int = 0  # straddles actually sold
    n_open: int = 0  # straddles still open
    entry_fill: float = NAN
    entry_quoted_mid: float = NAN
    total_vol: float = NAN
    hours_prev: float = NAN
    last_delta: float = NAN
    lots: dict[str, int] = field(default_factory=lambda: {"ES": 0, "MES": 0})
    futures_cash: float = 0.0  # dollars; negative = paid out
    entered: bool = False
    killed: bool = False
    closed: bool = False
    #: one row per clock the runner actually observed, for the ledger
    observed: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------- ledger ---


def _records_from_frame(frame: Any) -> list[ClockRecord]:
    """Every row of a ledger frame as a :class:`ClockRecord`."""
    return [
        ClockRecord(
            session=r.session.date() if hasattr(r.session, "date") else r.session,
            clock=str(r.clock),
            implied_rem_var=float(r.implied_rem_var),
            realized_rem_var=float(r.realized_rem_var),
            spot=float(r.spot),
            premium_mid=float(r.premium_mid),
            kc=float(r.kc),
            kp=float(r.kp),
        )
        for r in frame.itertuples()
    ]


def load_premium_ledger(path: str) -> PremiumLedger:
    """``PremiumLedger.load``, but an absent file is an EMPTY ledger."""
    try:
        return PremiumLedger.load(path)
    except FileNotFoundError:
        return PremiumLedger()


def last_session(led: PremiumLedger) -> date | None:
    days = led.sessions()
    return days[-1] if days else None


def previous_trading_day(day: date) -> date:
    """The previous weekday.  Holidays are not in this package's scope."""
    d = day - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def load_ledger(cfg: Config) -> tuple[PremiumLedger, PremiumLedger, dict[str, Any]]:
    """(merged, seed_only, info).  The live file overrides the seed per session."""
    info: dict[str, Any] = {"seed": cfg.ledger_path, "live": cfg.ledger_live_path}
    seed = load_premium_ledger(cfg.ledger_path)
    info["seed_rows"] = int(len(seed.frame))
    info["seed_last"] = last_session(seed)
    merged = PremiumLedger(seed.frame, min_sessions=seed.min_sessions)
    live_rows = 0
    if os.path.exists(cfg.ledger_live_path):
        live = load_premium_ledger(cfg.ledger_live_path)
        rows = _records_from_frame(live.frame)
        live_rows = len(rows)
        if rows:
            # ``append_session`` replaces every touched session wholesale, so
            # the live file wins on overlap by construction.
            merged.append_session(rows)
    info["live_rows"] = live_rows
    info["last_session"] = last_session(merged)
    return merged, seed, info


def make_clock_record(
    session: date,
    clock: str,
    implied_rem_var: float,
    realized_rem_var: float,
    spot: float,
    premium_mid: float,
    kc: float,
    kp: float,
) -> ClockRecord:
    return ClockRecord(
        session=session,
        clock=clock,
        implied_rem_var=implied_rem_var,
        realized_rem_var=realized_rem_var,
        spot=spot,
        premium_mid=premium_mid,
        kc=kc,
        kp=kp,
    )


# ---------------------------------------------------------------- runner ---


class DayRunner:
    """One session, one journal.  Every decision is journaled before it acts."""

    def __init__(self, config: Config, broker: Broker, journal: Journal):
        self.cfg = config
        self.broker = broker
        self.jr = journal
        self.state = DayState()
        self.futures: dict[str, Any] = {}
        self.chain: list[Any] = []
        self.ledger: Any = None
        self.ladder: tuple[str, ...] = ()
        #: proposal 50's deleveraging multiplier and the dict behind it.
        self.multiplier: float = 1.0
        self.delever: dict[str, Any] = {"state": "not_computed", "multiplier": 1.0}

    # -- helpers ---------------------------------------------------------
    def _now(self) -> datetime:
        return self.broker.now_et()

    def _expiry(self) -> str:
        return self.broker.session_date().replace("-", "")

    def _band(self, contracts: Sequence[Any], spot: float) -> list[Any]:
        w = self.cfg.strike_band_pct * spot
        return [c for c in contracts if abs(float(c.strike) - spot) <= w]

    def _package_legs(self) -> list[tuple[Any, int, str]]:
        """Legs of the LONG package: BUY the body, SELL the wings."""
        b = self.state.body
        assert b is not None
        legs: list[tuple[Any, int, str]] = [
            (self.broker.contract_for(b.Kc, "C"), 1, "BUY"),
            (self.broker.contract_for(b.Kp, "P"), 1, "BUY"),
        ]
        if self.state.wings is not None:
            kcw, kpw = self.state.wings
            legs.append((self.broker.contract_for(kcw, "C"), 1, "SELL"))
            legs.append((self.broker.contract_for(kpw, "P"), 1, "SELL"))
        return legs

    def _package_quote(self) -> PackageQuote:
        """Top of book of the traded package at the current stamp."""
        legs = self._package_legs()
        qs = self.broker.quotes([c for (c, _, _) in legs])
        health = self.broker.quote_health()
        bid = 0.0
        ask = 0.0
        for q, (_, ratio, leg_action) in zip(qs, legs):
            if not q.live:
                return PackageQuote(legs=qs, ok=False, health=health)
            if str(leg_action).upper() == "BUY":
                bid += ratio * q.bid
                ask += ratio * q.ask
            else:
                bid -= ratio * q.ask
                ask -= ratio * q.bid
        return PackageQuote(
            bid=bid, ask=ask, mid=0.5 * (bid + ask), legs=qs, ok=True, health=health
        )

    def _futures_ref(self) -> float:
        """The FUTURES price the open hedge is marked at -- never the index."""
        try:
            return float(self.broker.futures_reference("ES"))
        except Exception as exc:  # pragma: no cover - defensive
            self.jr.event(
                "error",
                ts_et=self._now(),
                what="no_futures_reference",
                level="alert",
                detail=repr(exc),
            )
            return NAN

    def _hedge_dollars(self, futures_ref: float) -> float:
        """Realized futures cash plus the open lots marked at ``futures_ref``."""
        cash = self.state.futures_cash
        if not math.isfinite(futures_ref):
            return NAN if any(self.state.lots.values()) else cash
        for sym, lots in self.state.lots.items():
            if lots:
                cash += lots * float(futures_ref) * self.cfg.multiplier(sym)
        return cash

    def _pnl_units(self, buyback_price: float, futures_ref: float) -> float:
        """Mark-to-market day P&L per straddle in units of entry premium."""
        e = self.state.entry_fill
        denom = self.state.n_entry * self.cfg.index_multiplier
        if not (math.isfinite(e) and e > 0) or denom <= 0:
            return NAN
        hedge = self._hedge_dollars(futures_ref)
        if not math.isfinite(hedge):
            return NAN
        dollars = denom * (e - float(buyback_price)) + hedge
        return dollars / (denom * e)

    def _hours(self, now: datetime) -> float:
        """Hours to the close; NaN at/after 16:00 (``hours_to_close`` raises)."""
        try:
            return float(hours_to_close(now))
        except Exception:
            return NAN

    def _factor(self, clock: str) -> float:
        """The trailing realized-over-implied factor for THIS clock (V9)."""
        if not self.cfg.delta_correction or self.ledger is None:
            return NAN
        try:
            return float(
                self.ledger.realized_over_implied_mean(
                    clock, self._session_date(), self.cfg.delta_correction_window
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.jr.event(
                "error",
                ts_et=self._now(),
                what="delta_correction_failed",
                level="warn",
                detail=repr(exc),
                clock=clock,
            )
            return NAN

    def _at_clock(self, clock: str) -> datetime:
        """Reach ``clock``; reconnect and shout if we got there late."""
        target_t = parse_clock(clock)
        now = self.broker.wait_until(clock)
        if not self.broker.is_connected():
            ok = bool(self.broker.ensure_connected())
            now = self._now()
            self.jr.event(
                "error",
                ts_et=now,
                what="disconnect",
                detail="reconnect " + ("succeeded" if ok else "FAILED"),
                level="alert",
                clock=clock,
            )
            if not ok:
                raise BrokerError("disconnected past the grace window at " + clock)
        target = now.replace(
            hour=target_t.hour,
            minute=target_t.minute,
            second=target_t.second,
            microsecond=0,
        )
        late = (now - target).total_seconds()
        if late > LATE_ALERT_S:
            msg = (
                "LATE at " + clock + " by " + f"{late:.0f}" + "s -- the research "
                "shows a one-bar lag zeroes the edge; rebalancing immediately"
            )
            print("\n*** " + msg + " ***\n", flush=True)
            self.jr.event(
                "error",
                ts_et=now,
                what="late_clock",
                detail=msg,
                level="alert",
                clock=clock,
                late_s=late,
            )
        return now

    # -- preflight -------------------------------------------------------
    def preflight(self) -> None:
        cfg = self.cfg
        checks: dict[str, Any] = {}
        reason = ""

        checks["connected"] = self.broker.is_connected()
        if not checks["connected"]:
            reason = "not connected to IB"

        if not reason:
            try:
                acct = self.broker.account_summary()
            except Exception as exc:
                acct = {}
                reason = "account summary failed: " + repr(exc)
            checks["account"] = (
                cfg.account or acct.get("AccountId", "") or acct.get("AccountType", "")
            ) or "unknown"
            if not reason and not acct:
                reason = "no account summary; account not known"
            if not reason and cfg.mode == "paper":
                acct_id = str(acct.get("AccountId", "") or cfg.account or "")
                if acct_id and not acct_id.upper().startswith("DU"):
                    reason = (
                        "paper mode but the account is "
                        + acct_id
                        + " (paper accounts are DU*); refusing"
                    )

        expiry = self._expiry()
        checks["expiry"] = expiry
        if not reason:
            try:
                self.chain = self.broker.spxw_0dte_contracts(expiry)
            except Exception as exc:
                self.chain = []
                reason = "chain request failed: " + repr(exc)
            checks["n_chain"] = len(self.chain)
            if not reason and not self.chain:
                reason = "today is not an SPXW expiry (no 0DTE contracts listed)"

        if not reason:
            try:
                probe = self.chain[0] if self.chain else None
                open_et, close_et = self.broker.liquid_hours(probe)
                checks["liquid_open"] = open_et.strftime("%H:%M")
                checks["liquid_close"] = close_et.strftime("%H:%M")
                if (close_et.hour, close_et.minute) != (16, 0):
                    reason = (
                        "half session: the book drops these days (liquid hours "
                        "close " + close_et.strftime("%H:%M") + " ET, not 16:00)"
                    )
            except Exception as exc:
                reason = "liquid hours unavailable: " + repr(exc)

        if not reason:
            spot = self.broker.spx_spot()
            checks["spot"] = spot
            if not math.isfinite(spot) or spot <= 0:
                reason = "SPX spot is not finite (no live index print)"

        if not reason:
            for sym in ("ES", "MES"):
                try:
                    self.futures[sym] = self.broker.futures_contract(sym)
                    checks["futures_" + sym] = str(
                        getattr(self.futures[sym], "localSymbol", "")
                        or getattr(
                            self.futures[sym], "lastTradeDateOrContractMonth", ""
                        )
                        or sym
                    )
                except Exception as exc:
                    reason = "no qualified " + sym + " front month: " + repr(exc)
                    break

        if not reason:
            # A swallowed exception here is how a crashed session double
            # enters (L-10): an error aborts, it never reads as "flat".
            try:
                pos = self.broker.positions()
            except Exception as exc:
                pos = {}
                reason = "positions() failed; cannot verify we are flat: " + repr(exc)
            if not reason:
                if pos.get("stale"):
                    reason = "position report is stale; reconcile manually"
                checks["open_options"] = len(pos.get("options", {}))
                checks["open_futures"] = dict(pos.get("futures", {}) or {})
                if pos.get("options") or any((pos.get("futures") or {}).values()):
                    reason = "position already open; reconcile manually"

        if not reason and cfg.n_override is None and cfg.capital <= 0:
            reason = (
                "no --capital and no --n: the stress table cannot size the "
                "day and this runner does not guess a size"
            )

        if not reason:
            # The ledger is read INSIDE preflight, because the deleveraging
            # rule is read off it and a zero multiplier is a preflight
            # refusal: the day never reaches an entry clock.
            self._load_ledger()
            self._regime()
            checks["delever"] = dict(self.delever)
            if self.multiplier <= 0.0:
                reason = self._delever_flat_reason()

        checks["book"] = cfg.book_name
        checks["mode"] = cfg.mode
        self.jr.event(
            "preflight",
            ts_et=self._now(),
            ok=not reason,
            reason=reason or None,
            checks=checks,
        )
        if reason:
            raise Abort(reason)

        self.state.entry_clock = self._choose_entry_clock()
        self.ladder = cfg.ladder(self.state.entry_clock)

    def _session_date(self) -> date:
        return date.fromisoformat(self.broker.session_date())

    def _load_ledger(self) -> None:
        cfg = self.cfg
        want = previous_trading_day(self._session_date())
        try:
            merged, seed, info = load_ledger(cfg)
        except Exception as exc:
            self.jr.event(
                "ledger",
                ts_et=self._now(),
                ok=False,
                detail="ledger unreadable: " + repr(exc),
                seed=cfg.ledger_path,
                live=cfg.ledger_live_path,
            )
            print(
                "\n*** the premium ledger is unreadable ("
                + repr(exc)
                + "): the selector cannot run, the fixed clock will be used ***\n",
                flush=True,
            )
            self.ledger = None
            return
        last = info.get("last_session")
        current = last == want
        if not current:
            msg = (
                "the premium ledger's last session is "
                + (str(last) if last else "(empty)")
                + ", not the previous trading session "
                + str(want)
                + " -- falling back to the SEED ledger; today's selector pick "
                "is made on stale information"
            )
            print("\n*** " + msg + " ***\n", flush=True)
            self.ledger = seed
        else:
            self.ledger = merged
        self.jr.event(
            "ledger",
            ts_et=self._now(),
            ok=current,
            used="merged" if current else "seed",
            last_session=str(last) if last else None,
            expected_last=str(want),
            **{k: v for k, v in info.items() if k != "last_session"},
        )

    # -- the deleveraging candidate (proposal 50, part B2) ---------------
    def _regime(self) -> None:
        """Read today's size multiplier off the ledger and journal it.

        Proposal 50's rule: full size while the trailing realized variance sits
        at or below its own expanding lagged 90th percentile, half above it,
        flat above the 97.5th, with a 252-session minimum before the thresholds
        are trusted.  A DESIGN CANDIDATE -- 50 evaluated it on its replay and
        adopted it nowhere -- so it is a flag, ``--no-delever``.
        """
        cfg = self.cfg
        if not cfg.delever:
            self.multiplier = 1.0
            self.delever = {"state": "disabled", "multiplier": 1.0}
        elif self.ledger is None:
            # An unreadable ledger has no regime: it does not silently become
            # a quiet one.  The day runs at full size and says so.
            self.multiplier = 1.0
            self.delever = {"state": "no_ledger", "multiplier": 1.0}
        else:
            mult, info = self.ledger.delever_multiplier(
                self._session_date(),
                window=cfg.delever_window,
                p_half=cfg.delever_half_pct,
                p_zero=cfg.delever_zero_pct,
                min_sessions=cfg.delever_min_sessions,
            )
            self.multiplier = float(mult)
            self.delever = {**dict(info), "multiplier": float(mult)}
        self.jr.event(
            REGIME_KIND,
            ts_et=self._now(),
            rule="proposal 50 part B2, a design candidate",
            window=cfg.delever_window,
            half_pct=cfg.delever_half_pct,
            zero_pct=cfg.delever_zero_pct,
            min_sessions=cfg.delever_min_sessions,
            enabled=bool(cfg.delever),
            **self.delever,
        )
        state = str(self.delever.get("state", ""))
        if state == "warmup":
            print(
                "  the deleveraging rule is still warming up ("
                + str(self.delever.get("n_prior", 0))
                + " of "
                + str(cfg.delever_min_sessions)
                + " prior sessions carry a trailing variance): full size",
                flush=True,
            )
        elif state == "half":
            print(
                "  deleveraging: the trailing realized variance is above its "
                "expanding lagged "
                + f"{cfg.delever_half_pct * 100.0:g}"
                + "th percentile -- HALF size today",
                flush=True,
            )

    def _delever_flat_reason(self) -> str:
        cfg = self.cfg
        rv = float(self.delever.get("rv21", NAN))
        pz = float(self.delever.get("p_zero", NAN))
        return (
            DELEVER_FLAT_REASON
            + f"{cfg.delever_zero_pct * 100.0:g}"
            + "th percentile (RV21 "
            + f"{rv:.4e}"
            + " > "
            + f"{pz:.4e}"
            + " over "
            + str(self.delever.get("n_prior", 0))
            + " prior sessions): the day is FLAT"
        )

    def _choose_entry_clock(self) -> str:
        cfg = self.cfg
        if cfg.entry_mode == "fixed" or self.ledger is None:
            self.jr.event(
                "selector",
                ts_et=self._now(),
                clock=cfg.fixed_entry_clock,
                mode="fixed" if cfg.entry_mode == "fixed" else "fixed_no_ledger",
                warmup=self.ledger is None,
                candidates=list(cfg.candidate_clocks),
            )
            return cfg.fixed_entry_clock
        try:
            picked, scores = pick_entry_clock(
                self.ledger,
                self._session_date(),
                window=cfg.selector_window,
                min_sessions=cfg.selector_min_sessions,
                candidate_clocks=cfg.candidate_clocks,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.jr.event(
                "error",
                ts_et=self._now(),
                what="selector_failed",
                level="alert",
                detail=repr(exc),
            )
            picked, scores = None, {}
        if picked:
            clock, mode = str(picked), "selector"
        else:
            clock, mode = cfg.fixed_entry_clock, "warmup_fixed"
        self.jr.event(
            "selector",
            ts_et=self._now(),
            clock=clock,
            mode=mode,
            warmup=picked is None,
            selector_pick=picked,
            candidates=list(cfg.candidate_clocks),
            scores={str(k): float(v) for k, v in dict(scores or {}).items()},
            window=cfg.selector_window,
            min_sessions=cfg.selector_min_sessions,
        )
        if mode == "warmup_fixed":
            print(
                "  the selector is still warming up (no candidate clock has "
                + str(cfg.selector_min_sessions)
                + " prior sessions): entering at the fixed "
                + clock,
                flush=True,
            )
        return clock

    # -- sizing ----------------------------------------------------------
    def _size(self, spot: float, body: Body, pq: PackageQuote, delta: float) -> int:
        """Contracts from the STRESS table; ``--n`` overrides it, loudly."""
        cfg = self.cfg
        stress = stress_loss_per_contract(
            spot,
            body.Kc,
            body.Kp,
            pq.mid,
            delta,
            jump=cfg.stress_jump,
        )
        loss = float(stress.get("total", NAN))
        sized = contracts_for(cfg.capital, cfg.stress_fraction, loss)
        if cfg.n_override is not None:
            n_stress = int(cfg.n_override)
            note = "--n override"
        else:
            n_stress = min(int(sized), int(cfg.max_straddles))
            note = "stress table" + (
                ", capped at max_straddles" if sized > cfg.max_straddles else ""
            )
        # The deleveraging multiplier scales the FULL-SIZE intent, whichever
        # way that intent was formed: an explicit --n is a full-size intent,
        # not an exemption from the regime filter.
        mult = float(self.multiplier)
        n = int(math.floor(n_stress * mult)) if math.isfinite(mult) else n_stress
        if mult < 1.0:
            note += ", deleveraged x" + f"{mult:g}"
        scale = 1.0 / cfg.capital if cfg.capital > 0 and math.isfinite(loss) else NAN
        implied = n * loss * scale
        implied_stress = n_stress * loss * scale
        self.jr.event(
            "sizing",
            ts_et=self._now(),
            clock=self.state.entry_clock,
            n=n,
            n_stress=n_stress,
            multiplier=mult,
            delever_state=str(self.delever.get("state", "")),
            n_from_table=int(sized),
            n_override=cfg.n_override,
            max_straddles=cfg.max_straddles,
            capital=cfg.capital,
            stress_fraction=cfg.stress_fraction,
            stress_jump=cfg.stress_jump,
            fraction_implied=implied,
            fraction_implied_stress=implied_stress,
            loss_option=float(stress.get("option", NAN)),
            loss_hedge=float(stress.get("hedge", NAN)),
            loss_total=loss,
            loss_side=str(stress.get("side", "")),
            delta_pkg=delta,
            premium_mid=pq.mid,
            note=note,
        )
        if cfg.n_override is not None and math.isfinite(implied_stress):
            print(
                "  --n "
                + str(n_stress)
                + " overrides the stress table (it wanted "
                + str(sized)
                + "): that is "
                + f"{implied_stress:.1%}"
                + " of capital at a "
                + f"{cfg.stress_jump:.0%}"
                + " last-bar jump",
                flush=True,
            )
        if mult < 1.0:
            print(
                "  deleveraging x"
                + f"{mult:g}"
                + ": "
                + str(n_stress)
                + " straddle(s) at full size become "
                + str(n),
                flush=True,
            )
        return n

    # -- entry -----------------------------------------------------------
    def enter(self) -> bool:
        cfg = self.cfg
        clock = self.state.entry_clock
        now = self._at_clock(clock)
        spot = self.broker.spx_spot()
        # Re-read the chain AT the entry clock: preflight ran at a different
        # stamp and the selector may have moved the entry hours away from it.
        self.chain = self.broker.spxw_0dte_contracts(self._expiry()) or self.chain
        band = self._band(self.chain, spot)
        quotes = self.broker.quotes(band)
        health = self.broker.quote_health()
        body, why = pick_nearest_otm_reason(quotes, spot)
        if not health.ok:
            body, why = None, "stale_or_halted_quotes: " + health.detail
        if body is None:
            self.jr.event(
                "entry",
                ts_et=now,
                clock=clock,
                S=spot,
                entered=False,
                reason=why,
                n_band=len(band),
                health=health.as_payload(),
            )
            print("  no entry: " + why + " -- the day ends flat", flush=True)
            return False

        wings: tuple[float, float] | None = None
        if cfg.wings_pct is not None:
            width = float(spot) * float(cfg.wings_pct)
            wings = pick_wings(quotes, body, width)
            if wings is None:
                self.jr.event(
                    "entry",
                    ts_et=now,
                    clock=clock,
                    S=spot,
                    entered=False,
                    reason="wings_not_quoted",
                    wings_pct=cfg.wings_pct,
                    wing_width_pts=width,
                    K_c=body.Kc,
                    K_p=body.Kp,
                )
                print(
                    "  no entry: the fly was requested but no live wing at "
                    + f"{width:.1f}"
                    + " points -- no naked entry; the day ends flat",
                    flush=True,
                )
                return False

        self.state.body = body
        self.state.wings = wings
        pq = self._package_quote()
        if not pq.tradeable:
            self.jr.event(
                "entry",
                ts_et=now,
                clock=clock,
                S=spot,
                entered=False,
                reason="package_not_quoted"
                if not pq.ok
                else "unhealthy_quotes: " + pq.health.detail,
                K_c=body.Kc,
                K_p=body.Kp,
                health=pq.health.as_payload(),
            )
            return False

        # Pre-trade delta, off the quoted mid: it is all there is before the
        # order exists, and it is what the stress table has to be sized on.
        h = self._hours(now)
        tv_quote = invert_total_vol(spot, body.Kc, body.Kp, pq.mid)
        factor = self._factor(clock)
        delta_quote = (
            package_delta(corrected_total_vol(tv_quote, factor), spot, body.Kc, body.Kp)
            if math.isfinite(tv_quote)
            else NAN
        )
        n = self._size(spot, body, pq, delta_quote)
        if n < 1:
            deleveraged = self.multiplier < 1.0
            why_zero = (
                "deleveraging_took_the_size_to_zero_contracts"
                if deleveraged
                else "stress_sizing_is_zero_contracts"
            )
            self.jr.event(
                "entry",
                ts_et=now,
                clock=clock,
                S=spot,
                entered=False,
                reason=why_zero,
                multiplier=float(self.multiplier),
                delever_state=str(self.delever.get("state", "")),
                K_c=body.Kc,
                K_p=body.Kp,
            )
            print(
                "  no entry: "
                + (
                    "the deleveraging multiplier takes this day to zero contracts"
                    if deleveraged
                    else "the stress table sizes this day at zero contracts"
                )
                + " -- the day ends flat",
                flush=True,
            )
            return False

        self.jr.event(
            "entry",
            ts_et=now,
            clock=clock,
            S=spot,
            entered=True,
            K_c=body.Kc,
            K_p=body.Kp,
            same_strike=body.same_strike,
            Kc_wing=None if wings is None else wings[0],
            Kp_wing=None if wings is None else wings[1],
            pkg_bid=pq.bid,
            pkg_ask=pq.ask,
            pkg_mid=pq.mid,
            n_straddles=n,
            hours_to_close=h,
            reason=None,
        )
        self.jr.event(
            "order",
            ts_et=self._now(),
            clock=clock,
            what="body_entry",
            action="SELL",
            quantity=n,
            limit=pq.mid,
            spread=pq.spread,
        )
        fill = self.broker.place_combo(
            self._package_legs(),
            n,
            pq.mid,
            action="SELL",
            spread=pq.spread,
            what="body_entry",
        )
        self.jr.event("fill", ts_et=self._now(), clock=clock, **fill.as_payload())
        filled = abs(int(fill.quantity))
        if filled == 0 or not math.isfinite(fill.price) or fill.price <= 0:
            self.jr.event(
                "error",
                ts_et=self._now(),
                what="entry_unfilled",
                detail=(fill.note or "no fill") + " [status " + fill.status + "]",
                level="alert",
                clock=clock,
            )
            print("  entry did not fill; the day ends flat", flush=True)
            return False
        if filled < n:
            # A partial entry is a real position, not a flat day (L-4).
            msg = (
                "PARTIAL entry: "
                + str(filled)
                + " of "
                + str(n)
                + " straddles filled -- hedging what we have"
            )
            print("\n*** " + msg + " ***\n", flush=True)
            self.jr.event(
                "error",
                ts_et=self._now(),
                what="entry_partial",
                detail=msg,
                level="alert",
                clock=clock,
                requested=n,
                filled=filled,
            )

        self.state.entry_fill = float(fill.price)
        self.state.entry_quoted_mid = pq.mid
        self.state.n_entry = filled
        self.state.n_open = filled
        self.state.entered = True
        # The research inverts the total vol from the traded package price;
        # a fill far from the quoted mid is a bad print, so fall back to it.
        self.hedge_to_target(clock, now, spot, pq, source="entry_fill")
        return True

    # -- hedging ---------------------------------------------------------
    def hedge_to_target(
        self,
        clock: str,
        now: datetime,
        spot: float,
        pq: PackageQuote,
        source: str = "quote_mid",
    ) -> None:
        cfg = self.cfg
        h = self._hours(now)
        px = (
            self.state.entry_fill
            if source == "entry_fill" and math.isfinite(self.state.entry_fill)
            else pq.mid
        )
        body = self.state.body
        assert body is not None
        tv = invert_total_vol(spot, body.Kc, body.Kp, px)
        used = source
        if not math.isfinite(tv) and source == "entry_fill":
            tv = invert_total_vol(spot, body.Kc, body.Kp, pq.mid)
            used = "quote_mid_fallback"
        if not math.isfinite(tv):
            # Carry the last total vol, scaled by the square root of the
            # remaining-time ratio (the research's own decay), and shout.
            if (
                math.isfinite(self.state.total_vol)
                and self.state.hours_prev > 0
                and math.isfinite(h)
                and h > 0
            ):
                tv = self.state.total_vol * math.sqrt(h / self.state.hours_prev)
                used = "carried_sqrt_scaled"
            else:
                used = "none"
            self.jr.event(
                "error",
                ts_et=now,
                what="vol_not_invertible",
                level="alert",
                detail="package mid at/below intrinsic; "
                + (
                    "carried the last total vol scaled by sqrt(h_now/h_prev)"
                    if used == "carried_sqrt_scaled"
                    else "no carryable total vol"
                ),
                clock=clock,
                h_now=h,
                h_prev=self.state.hours_prev,
            )
        if math.isfinite(tv):
            self.state.total_vol = tv
            self.state.hours_prev = h

        factor = self._factor(clock)
        tv_used = corrected_total_vol(tv, factor) if math.isfinite(tv) else NAN
        if math.isfinite(tv_used):
            delta = package_delta(tv_used, spot, body.Kc, body.Kp)
            self.state.last_delta = delta
            delta_source = "inverted"
        else:
            # NEVER a delta of zero: that unwinds the whole hedge on exactly
            # the bar the book is most exposed (L-6).
            delta = self.state.last_delta
            delta_source = "carried_last_delta"
            self.jr.event(
                "error",
                ts_et=now,
                what="delta_not_computable",
                level="alert",
                detail="no invertible total volatility at "
                + clock
                + "; carrying the last delta "
                + f"{delta:+.4f}"
                + " (a zero delta would unwind the hedge)",
                clock=clock,
                delta_pkg=delta,
            )
        self.state.observed.append(
            {
                "clock": clock,
                "S": spot,
                "total_vol": tv if math.isfinite(tv) else NAN,
                "premium_mid": pq.mid,
                "kc": body.Kc,
                "kp": body.Kp,
                "hours_to_close": h,
            }
        )
        self.jr.event(
            "delta",
            ts_et=now,
            clock=clock,
            S=spot,
            total_vol=tv,
            total_vol_corrected=tv_used,
            correction_factor=factor,
            iv_hourly=(tv / math.sqrt(h)) if (math.isfinite(tv) and h > 0) else NAN,
            hours_to_close=h,
            delta_pkg=delta,
            source=used,
            delta_source=delta_source,
            residual_delta=self._residual(delta) if math.isfinite(delta) else NAN,
        )
        if not math.isfinite(delta):
            # Nothing computable and nothing to carry: leave the hedge alone.
            return

        target = target_lots(
            delta,
            self.state.n_open,
            index_multiplier=cfg.index_multiplier,
            es_multiplier=cfg.es_multiplier,
            mes_multiplier=cfg.mes_multiplier,
        )
        move = rebalance_lots(target, self._lots_tuple())
        self.jr.event(
            "target",
            ts_et=now,
            clock=clock,
            target_es=int(target[0]),
            target_mes=int(target[1]),
            current_es=self.state.lots["ES"],
            current_mes=self.state.lots["MES"],
            qty_es=int(move[0]),
            qty_mes=int(move[1]),
        )
        for sym, qty in (("ES", int(move[0])), ("MES", int(move[1]))):
            if qty:
                self._trade_futures(sym, qty, clock, what="futures")
        resid = self._residual(delta)
        self.jr.event(
            "rebalance",
            ts_et=now,
            clock=clock,
            target_es=int(target[0]),
            target_mes=int(target[1]),
            current_es=self.state.lots["ES"],
            current_mes=self.state.lots["MES"],
            residual_delta=resid,
            delta_pkg=delta,
        )
        if abs(resid) > cfg.max_residual_delta:
            one_mes = cfg.mes_multiplier / (
                max(self.state.n_open, 1) * cfg.index_multiplier
            )
            msg = (
                "residual delta "
                + f"{resid:+.3f}"
                + " exceeds "
                + f"{cfg.max_residual_delta:.2f}"
                + " at "
                + clock
                + " (one MES is "
                + f"{one_mes:.3f}"
                + " straddle deltas)"
            )
            print("\n*** " + msg + " ***\n", flush=True)
            self.jr.event(
                "error",
                ts_et=now,
                what="residual_delta",
                detail=msg,
                level="alert",
                clock=clock,
                residual_delta=resid,
            )

    def _lots_tuple(self) -> tuple[int, int]:
        return (int(self.state.lots["ES"]), int(self.state.lots["MES"]))

    def _residual(self, delta: float) -> float:
        """Unhedged package delta left by the current ES + MES lots."""
        cfg = self.cfg
        return residual_delta_lots(
            delta,
            max(int(self.state.n_open), 1),
            int(self.state.lots["ES"]),
            int(self.state.lots["MES"]),
            index_multiplier=cfg.index_multiplier,
            es_multiplier=cfg.es_multiplier,
            mes_multiplier=cfg.mes_multiplier,
        )

    def _trade_futures(
        self,
        sym: str,
        qty: int,
        clock: str,
        what: str,
        passive_wait_s: float | None = None,
    ) -> Fill:
        cfg = self.cfg
        self.jr.event(
            "order",
            ts_et=self._now(),
            clock=clock,
            what=what,
            symbol=sym,
            action="BUY" if qty > 0 else "SELL",
            quantity=abs(qty),
            limit=NAN,
        )
        fill = self.broker.place_futures(
            self.futures[sym], qty, passive_wait_s=passive_wait_s, what=what
        )
        self.jr.event("fill", ts_et=self._now(), clock=clock, **fill.as_payload())
        # A partial fill moved the position: book what filled, by its sign.
        if fill.quantity:
            self.state.lots[sym] += int(fill.quantity)
            self.state.futures_cash -= (
                int(fill.quantity) * float(fill.price) * cfg.multiplier(sym)
            )
        if not fill.ok:
            self.jr.event(
                "error",
                ts_et=self._now(),
                what="futures_unfilled",
                level="alert",
                detail=(fill.note or "")
                + " [status "
                + fill.status
                + ", filled "
                + str(fill.quantity)
                + " of "
                + str(qty)
                + "]",
                clock=clock,
                symbol=sym,
            )
        return fill

    # -- kill / exit -----------------------------------------------------
    def check_kill(
        self, clock: str, now: datetime, spot: float, pq: PackageQuote
    ) -> bool:
        cfg = self.cfg
        if not pq.tradeable or not math.isfinite(self.state.entry_fill):
            return False
        ref = self._futures_ref()
        if not math.isfinite(ref):
            # Without a futures mark the loss is unknown, not zero.
            self.jr.event(
                "error",
                ts_et=now,
                what="kill_check_skipped",
                level="alert",
                detail="no futures reference: the mark-to-market loss cannot "
                "be computed, so the kill switch is not armed at " + clock,
                clock=clock,
            )
            return False
        units = self._pnl_units(pq.ask, ref)
        if not math.isfinite(units) or -units <= cfg.max_loss_units:
            return False
        reason = (
            "mark-to-market loss "
            + f"{-units:.2f}"
            + " premium units exceeds "
            + f"{cfg.max_loss_units:.2f}"
            + " at "
            + clock
        )
        print("\n*** KILL: " + reason + " ***\n", flush=True)
        self.jr.event(
            "kill",
            ts_et=now,
            clock=clock,
            loss_units=-units,
            reason=reason,
            pkg_ask=pq.ask,
            S=spot,
            futures_ref=ref,
        )
        fill = self.close_position(
            clock, now, spot, pq, aggressive=True, what="body_exit"
        )
        if fill is None or not self.state.closed:
            # The body is still open: the hedge stays on and the day keeps
            # running.  Flattening the futures here is L-1.
            return False
        self.state.killed = True
        return True

    def close_position(
        self,
        clock: str,
        now: datetime,
        spot: float,
        pq: PackageQuote,
        aggressive: bool = False,
        what: str = "body_exit",
    ) -> Fill | None:
        """Buy the package back as one combo; flatten the futures ONLY if it filled."""
        limit = pq.ask if aggressive else pq.mid
        self.jr.event(
            "exit",
            ts_et=now,
            clock=clock,
            S=spot,
            pkg_bid=pq.bid,
            pkg_ask=pq.ask,
            pkg_mid=pq.mid,
            aggressive=aggressive,
            n_open=self.state.n_open,
        )
        self.jr.event(
            "order",
            ts_et=self._now(),
            clock=clock,
            what=what,
            action="BUY",
            quantity=self.state.n_open,
            limit=limit,
            spread=pq.spread,
        )
        fill = self.broker.place_combo(
            self._package_legs(),
            self.state.n_open,
            limit,
            action="BUY",
            spread=pq.spread,
            what=what,
        )
        self.jr.event("fill", ts_et=self._now(), clock=clock, **fill.as_payload())
        filled = abs(int(fill.quantity))
        if filled:
            self.state.n_open -= filled
        if self.state.n_open > 0:
            msg = (
                (
                    "exit did not fill"
                    if filled == 0
                    else "exit filled only " + str(filled)
                )
                + " at "
                + clock
                + ": "
                + str(self.state.n_open)
                + " straddle(s) STILL SHORT. The futures hedge stays on -- "
                "stripping it off an open short straddle is what this book "
                "exists to avoid. Work the order by hand."
            )
            print("\n*** " + msg + " ***\n", flush=True)
            self.jr.event(
                "error",
                ts_et=self._now(),
                what="exit_unfilled" if filled == 0 else "exit_partial",
                level="alert",
                detail=msg + " [status " + fill.status + "] " + (fill.note or ""),
                clock=clock,
                n_open=self.state.n_open,
            )
            return fill
        self.state.closed = True
        self.flatten(clock, self._now())
        return fill

    def flatten(self, clock: str, now: datetime) -> None:
        """Flatten ES then MES: passive for five seconds, then cross."""
        if not any(self.state.lots.values()):
            self.jr.event("flatten", ts_et=now, clock=clock, qty=0, note="already flat")
            return
        for sym in ("ES", "MES"):
            qty = -int(self.state.lots[sym])
            if qty:
                self._trade_futures(
                    sym, qty, clock, what="futures_flatten", passive_wait_s=5.0
                )
        self.jr.event(
            "flatten",
            ts_et=self._now(),
            clock=clock,
            residual_es=self.state.lots["ES"],
            residual_mes=self.state.lots["MES"],
        )

    # -- terminal --------------------------------------------------------
    def mark(self, now: datetime) -> None:
        """Journal the futures reference every open lot is marked at (L-11)."""
        ref = self._futures_ref()
        self.jr.event(
            "mark",
            ts_et=now,
            futures_ref=ref,
            es=self.state.lots["ES"],
            mes=self.state.lots["MES"],
            n_open=self.state.n_open,
            hedge_cash=self.state.futures_cash,
        )

    def settle(self, now: datetime) -> dict[str, Any]:
        """Hold variant: journal the PROVISIONAL cash settlement."""
        body = self.state.body
        assert body is not None
        try:
            s = float(self.broker.settlement_spot())
        except Exception as exc:
            s = NAN
            self.jr.event(
                "error",
                ts_et=now,
                what="no_settlement_print",
                level="alert",
                detail=repr(exc),
            )
        payoff = settlement_payoff(s, body.Kc, body.Kp, self.state.wings)
        self.jr.event(
            "settle",
            ts_et=now,
            S_close=s,
            payoff=payoff,
            provisional=True,
            n_settled=self.state.n_open,
            note="provisional: SPXW is PM-settled, so the official settlement "
            "is the official 16:00 SPX close; --reconcile finalises it next "
            "morning",
        )
        # Cash settlement closes the body: nothing is left to flatten.
        self.state.n_open = 0
        self.state.closed = True
        return {"S_close": s, "payoff": payoff}

    def write_pending(self, provisional: dict[str, Any]) -> str:
        """The handover file ``--reconcile`` reads the next morning."""
        body = self.state.body
        assert body is not None
        path = os.path.join(self.cfg.journal_dir, PENDING_NAME)
        blob = {
            "date": self.broker.session_date(),
            "journal": self.jr.path,
            "book": self.cfg.book_name,
            "entry_clock": self.state.entry_clock,
            "n_straddles": self.state.n_entry,
            "K_c": body.Kc,
            "K_p": body.Kp,
            "wings": list(self.state.wings) if self.state.wings else None,
            "entry_fill": self.state.entry_fill,
            "provisional_settlement": provisional.get("S_close", NAN),
            "provisional_payoff": provisional.get("payoff", NAN),
            "observed": self.state.observed,
            "written_at": self._now().isoformat(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True, default=_json_default)
            fh.write("\n")
        print("  wrote " + path + " -- run --reconcile tomorrow morning", flush=True)
        return path

    # -- driver ----------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        self.jr.event(
            "config",
            ts_et=self._now(),
            date=self.broker.session_date(),
            book=cfg.book_name,
            mode=cfg.mode,
            entry_mode=cfg.entry_mode,
            terminal=cfg.terminal,
            hedge_symbols=["ES", "MES"],
            index_multiplier=cfg.index_multiplier,
            es_multiplier=cfg.es_multiplier,
            mes_multiplier=cfg.mes_multiplier,
            n_override=cfg.n_override,
            capital=cfg.capital,
            stress_fraction=cfg.stress_fraction,
            stress_jump=cfg.stress_jump,
            max_straddles=cfg.max_straddles,
            delta_correction=cfg.delta_correction,
            delever=cfg.delever,
            delever_window=cfg.delever_window,
            delever_half_pct=cfg.delever_half_pct,
            delever_zero_pct=cfg.delever_zero_pct,
            delever_min_sessions=cfg.delever_min_sessions,
            candidate_clocks=list(cfg.candidate_clocks),
            fixed_entry_clock=cfg.fixed_entry_clock,
            exit_clock=cfg.exit_clock,
            replay=cfg.is_replay,
        )
        self.preflight()
        if not self.enter():
            return
        # Everything past the entry is wrapped: a crash must not end the
        # process with a position open and no record of where it is (L-9).
        try:
            for clock in self.ladder:
                now = self._at_clock(clock)
                spot = self.broker.spx_spot()
                pq = self._package_quote()
                self.jr.event(
                    "quote",
                    ts_et=now,
                    clock=clock,
                    S=spot,
                    pkg_bid=pq.bid,
                    pkg_ask=pq.ask,
                    pkg_mid=pq.mid,
                    ok=pq.ok,
                    health=pq.health.as_payload(),
                )
                if self.check_kill(clock, now, spot, pq):
                    return
                if not pq.tradeable:
                    self.jr.event(
                        "error",
                        ts_et=now,
                        what="no_package_quote" if not pq.ok else "unhealthy_quotes",
                        level="alert" if not pq.health.ok else "warn",
                        detail="skipping this rebalance: "
                        + (pq.health.detail or "no two-sided package quote"),
                        clock=clock,
                    )
                    continue
                self.hedge_to_target(clock, now, spot, pq)

            if cfg.hold_to_settle:
                now = self._at_clock(cfg.flatten_clock)
                self.flatten(cfg.flatten_clock, now)
                provisional = self.settle(now)
                self.write_pending(provisional)
                return

            now = self._at_clock(cfg.exit_clock)
            spot = self.broker.spx_spot()
            pq = self._package_quote()
            self.jr.event(
                "quote",
                ts_et=now,
                clock=cfg.exit_clock,
                S=spot,
                pkg_bid=pq.bid,
                pkg_ask=pq.ask,
                pkg_mid=pq.mid,
                ok=pq.ok,
                health=pq.health.as_payload(),
            )
            if not pq.ok:
                self.jr.event(
                    "error",
                    ts_et=now,
                    what="no_exit_quote",
                    level="alert",
                    detail="the body is not quoted at the exit clock; crossing "
                    "the ask is the only remaining exit",
                    clock=cfg.exit_clock,
                )
            self.close_position(cfg.exit_clock, now, spot, pq, aggressive=not pq.ok)
        finally:
            self.mark(self._now())
            if self.state.n_open > 0:
                print(
                    "\n*** MANUAL FLATTEN REQUIRED: "
                    + str(self.state.n_open)
                    + " straddle(s) and "
                    + repr(self.state.lots)
                    + " futures are open ***\n",
                    flush=True,
                )
                self.jr.event(
                    "error",
                    what="manual_flatten_required",
                    level="fatal",
                    detail="the session ended with "
                    + str(self.state.n_open)
                    + " straddle(s) open; the futures hedge was NOT removed",
                    es=self.state.lots["ES"],
                    mes=self.state.lots["MES"],
                )
            elif any(self.state.lots.values()) and not cfg.hold_to_settle:
                self.jr.event(
                    "error",
                    what="futures_left_open",
                    level="alert",
                    detail="the options are closed but "
                    + repr(self.state.lots)
                    + " futures are still on",
                )


# ------------------------------------------------------------ reconcile ---


def settlement_payoff(
    s: float, kc: float, kp: float, wings: Sequence[float] | None = None
) -> float:
    """Cash settlement of the SHORT body (plus long wings), in index points."""
    if not math.isfinite(s):
        return NAN
    payoff = max(s - float(kc), 0.0) + max(float(kp) - s, 0.0)
    if wings:
        payoff -= max(s - float(wings[0]), 0.0) + max(float(wings[1]) - s, 0.0)
    return payoff


def clock_records(
    session: date, observed: Sequence[Mapping[str, Any]], settlement: float
) -> list[ClockRecord]:
    """Implied and realized remaining variance at every clock we observed.

    Implied is the square of the total volatility bisected out of that
    stamp's package mid; realized is the sum of squared log returns of the
    observed spot path from that clock through the settlement print -- the
    same two objects proposals 44 and 46 build off the tape.
    """
    rows = [r for r in observed if math.isfinite(float(r.get("S", NAN)))]
    if not rows:
        return []
    path = [float(r["S"]) for r in rows] + [float(settlement)]
    steps: list[float] = []
    for a, b in zip(path[:-1], path[1:]):
        steps.append(math.log(b / a) ** 2 if a > 0 and b > 0 else NAN)
    out: list[ClockRecord] = []
    for k, r in enumerate(rows):
        tail = steps[k:]
        rv = sum(tail) if all(math.isfinite(x) for x in tail) else NAN
        tv = float(r.get("total_vol", NAN))
        iv = tv * tv if math.isfinite(tv) else NAN
        if not (math.isfinite(iv) and iv > 0 and math.isfinite(rv) and rv > 0):
            continue
        out.append(
            make_clock_record(
                session,
                str(r["clock"]),
                iv,
                rv,
                float(r["S"]),
                float(r.get("premium_mid", NAN)),
                float(r.get("kc", NAN)),
                float(r.get("kp", NAN)),
            )
        )
    return out


def append_live_ledger(path: str, records: Sequence[Any]) -> int:
    """Append one session's records to the live ledger; returns the count."""
    if not records:
        return 0
    led = load_premium_ledger(path)
    led.append_session(list(records))
    led.save(path)
    return len(records)


def reconcile_day(cfg: Config, broker: Broker | None = None) -> int:
    """Finalise the pending session against the official settlement."""
    pending_path = os.path.join(cfg.journal_dir, PENDING_NAME)
    if not os.path.exists(pending_path):
        print("  nothing to reconcile: no " + pending_path, flush=True)
        return 2
    with open(pending_path, encoding="utf-8") as fh:
        pending = json.load(fh)
    session = str(pending["date"])
    journal_path = str(pending["journal"])
    if not os.path.exists(journal_path):
        print("  the pending journal is missing: " + journal_path, flush=True)
        return 2

    settlement = NAN
    source = ""
    if cfg.settlement is not None:
        settlement, source = float(cfg.settlement), "manual (--settlement)"
    else:
        own = broker is None
        if own:
            broker = build_broker(cfg)
            broker.connect()
        try:
            assert broker is not None
            settlement = float(broker.official_settlement(session))
            source = "broker statement"
        except Exception as exc:
            print(
                "  the broker could not supply a settlement: " + repr(exc), flush=True
            )
            settlement, source = NAN, "unavailable"
        finally:
            if own and broker is not None:
                broker.disconnect()
    if not (math.isfinite(settlement) and settlement > 0):
        print(
            "  no official settlement for "
            + session
            + " (source: "
            + source
            + "): pass --settlement <value>; nothing was finalised",
            flush=True,
        )
        return 2

    payoff = settlement_payoff(
        settlement, float(pending["K_c"]), float(pending["K_p"]), pending.get("wings")
    )
    jr = Journal(journal_path, tz=cfg.tz, echo=False, roll=False)
    try:
        jr.event(
            "reconcile",
            S_close=settlement,
            payoff=payoff,
            provisional_settlement=pending.get("provisional_settlement"),
            provisional_payoff=pending.get("provisional_payoff"),
            source=source,
            date=session,
        )
        records = clock_records(
            date.fromisoformat(session), pending.get("observed", []), settlement
        )
        n = append_live_ledger(cfg.ledger_live_path, records)
        jr.event(
            "ledger",
            ok=True,
            used="append",
            appended=n,
            live=cfg.ledger_live_path,
            date=session,
        )
    finally:
        jr.close()

    summary = DaySummary.from_journal(journal_path)
    summary.write_beside(journal_path)
    print(summary.render(), flush=True)
    print(
        "  reconciled "
        + session
        + ": settlement "
        + f"{settlement:.2f}"
        + " ("
        + source
        + "), payoff "
        + f"{payoff:.4f}"
        + " pts, "
        + str(n)
        + " ledger record(s) appended to "
        + cfg.ledger_live_path,
        flush=True,
    )
    os.replace(
        pending_path, os.path.splitext(pending_path)[0] + "." + session + ".json"
    )
    return 0 if summary.is_flat else 3


# ----------------------------------------------------------------- main ---


def _json_default(obj: Any) -> Any:  # pragma: no cover - defensive
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def build_broker(cfg: Config) -> Broker:
    if cfg.is_replay:
        assert cfg.replay_date is not None
        return FakeBroker(cfg, cfg.replay_date)
    return IBBroker(cfg)


def main(argv: list[str] | None = None) -> int:
    cfg = Config.from_args(argv)
    if cfg.mode == "live":
        print(cfg.banner(), flush=True)
        if not typed_live_ack():
            print("  live mode not confirmed; nothing was sent", flush=True)
            return 2

    if cfg.reconcile:
        return reconcile_day(cfg)

    broker = build_broker(cfg)
    session = cfg.replay_date or datetime.now(ZoneInfo(cfg.tz)).strftime("%Y-%m-%d")
    jr = Journal(os.path.join(cfg.journal_dir, session + ".jsonl"), tz=cfg.tz)
    rc = 0
    try:
        broker.connect()
        runner = DayRunner(cfg, broker, jr)
        runner.run()
    except Abort as exc:
        print("  PREFLIGHT ABORT: " + str(exc), flush=True)
        rc = 2
    except Exception as exc:  # noqa: BLE001 - the journal must record it
        jr.event("error", what=type(exc).__name__, detail=str(exc), level="fatal")
        print("  FATAL: " + type(exc).__name__ + ": " + str(exc), flush=True)
        rc = 1
    finally:
        try:
            broker.disconnect()
        finally:
            jr.close()

    summary = DaySummary.from_journal(jr.path)
    summary.write_beside(jr.path)
    print(summary.render(), flush=True)
    # A supervisor must not read "success" off a day that left a leg on (L-34).
    if not summary.is_flat or (summary.entered and not summary.exit_kind):
        rc = max(rc, 3)
    return rc


def cli(argv: list[str] | None = None) -> int:
    """Entry point used by the tests: ``main`` plus the journal path."""
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
