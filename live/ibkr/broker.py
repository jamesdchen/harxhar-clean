"""Broker layer: one real (``IBBroker``) and one recorded (``FakeBroker``).

Both satisfy the same :class:`Broker` protocol, so ``run_day`` never branches
on which one it is holding.  Three rules hold everywhere:

* **No market orders, ever.**  Every order starts as a limit resting at the
  touch and is repriced at most ``ceil(spread / tick) + 1`` ticks through it
  (capped by ``config.max_cross_ticks``); if it is still unfilled it is
  cancelled and reported, not chased.
* **Mode gates the wire.**  In ``dry`` mode the connection is read-only and
  the order methods log the order they would have sent and return a
  synthetic fill at the limit; only ``paper`` and ``live`` reach
  ``placeOrder``.
* **A snapshot is data with a timestamp.**  Every ``quotes()`` call records a
  :class:`QuoteHealth` -- age, halt flag, crossed markets -- and the runner
  refuses to act on a sick one.

Combo convention.  ``place_combo`` takes the legs of the **long package**
(BUY the body, SELL the wings) and an explicit bag action.  Shorting the
straddle is therefore ``action="SELL"`` at a **positive** limit equal to the
package price -- the same positive number the research calls ``entry`` --
rather than a negative-debit BUY.  IB reverses every leg on a bag SELL.

Hedge instruments.  The book hedges with ES for the bulk and MES for the
remainder, so every futures call takes the instrument explicitly and every
fill carries its symbol and multiplier into the journal.

Nothing here was exercised against a live socket: no TWS or IB Gateway is
installed on the development machine, so ``IBBroker`` is written from the
ib_async 2.1.0 API surface and must be walked through on a paper connection
before it is trusted.  The places where an IB detail was inferred rather
than observed are marked ``GUESS:``.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

from ib_async import (
    ComboLeg,
    Contract,
    Future,
    IB,
    Index,
    LimitOrder,
    Option,
    TagValue,
)

from live.ibkr.config import FUTURES_MULTIPLIER, Config, parse_clock
from live.ibkr.strikes import Quote

NAN = float("nan")

#: The recorded chain the replay reads (expiration, strike, cp, timestamp,
#: bid, ask, underlying_price, hours_to_expiration).
CHAIN_PARQUET = os.path.join("data", "spxw_chain.parquet")

#: Sessions the recorded chain covers.
REPLAY_FIRST_DATE = "2020-01-03"
REPLAY_LAST_DATE = "2025-12-31"

#: Roll the futures hedge this many calendar days before the contract's own
#: expiration -- the standard rule, and the one that does not mis-count a
#: holiday week the way "3 trading days is at most 5 calendar days" did.
ROLL_DAYS_BEFORE_EXPIRY = 8

#: Seconds any single blocking market-data request may take.
SNAPSHOT_TIMEOUT_S = 10.0

#: GUESS: SMART will not leg a combo without this tag on some accounts.
COMBO_NON_GUARANTEED = True


class BrokerError(RuntimeError):
    """Anything the broker layer refuses to do."""


@dataclass
class Fill:
    """The outcome of one order.  ``price`` is per package / per contract."""

    ok: bool
    price: float
    quantity: int
    action: str
    what: str = ""
    limit: float = NAN
    crossed_ticks: int = 0
    synthetic: bool = False
    note: str = ""
    #: IB's order id (or the replay's counter): the journal's dedupe key.
    order_id: int = 0
    #: ``orderStatus.status`` -- a rejection and a passive miss must not
    #: read the same in the journal (L-28).
    status: str = ""
    symbol: str = ""
    multiplier: float = NAN
    #: The stepping stopped because the caller's deadline passed (the month-end
    #: long's ``deadline_s``), not because the tick allowance ran out.
    deadline_hit: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "price": self.price,
            "quantity": self.quantity,
            "action": self.action,
            "what": self.what,
            "limit": self.limit,
            "crossed_ticks": self.crossed_ticks,
            "synthetic": self.synthetic,
            "note": self.note,
            "order_id": self.order_id,
            "status": self.status,
            "symbol": self.symbol,
            "multiplier": self.multiplier,
            "deadline_hit": self.deadline_hit,
        }


@dataclass
class QuoteHealth:
    """How sick the last snapshot was.  ``ok`` False means: do not trade it."""

    ok: bool = True
    n: int = 0
    stale: int = 0
    halted: int = 0
    crossed: int = 0
    max_age_s: float = 0.0
    detail: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n": self.n,
            "stale": self.stale,
            "halted": self.halted,
            "crossed": self.crossed,
            "max_age_s": self.max_age_s,
            "detail": self.detail,
        }


def round_to_tick(price: float, tick: float) -> float:
    """Round to the nearest whole tick (IB rejects sub-tick limit prices)."""
    if not math.isfinite(price) or tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


class Broker(Protocol):
    """What ``run_day`` needs; both implementations satisfy it."""

    config: Config

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def ensure_connected(self) -> bool: ...
    def now_et(self) -> datetime: ...
    def wait_until(self, clock: str) -> datetime: ...
    def session_date(self) -> str: ...
    def spx_spot(self) -> float: ...
    def spxw_0dte_contracts(self, expiry_yyyymmdd: str) -> list[Any]: ...
    def quotes(self, contracts: Sequence[Any]) -> list[Quote]: ...
    def quote_health(self) -> QuoteHealth: ...
    def contract_for(self, strike: float, right: str) -> Any: ...
    def futures_contract(self, symbol: str) -> Any: ...
    def futures_reference(self, symbol: str = "ES") -> float: ...
    def place_combo(
        self,
        legs: list[tuple[Any, int, str]],
        quantity: int,
        limit_price: float,
        action: str = "SELL",
        passive_wait_s: float | None = None,
        spread: float | None = None,
        what: str = "",
        max_cross_ticks: int | None = None,
        deadline_s: float | None = None,
    ) -> Fill: ...
    def place_futures(
        self,
        contract: Any,
        signed_qty: int,
        passive_wait_s: float | None = None,
        what: str = "",
    ) -> Fill: ...
    def positions(self) -> dict[str, Any]: ...
    def liquid_hours(self, contract: Any) -> tuple[datetime, datetime]: ...
    def account_summary(self) -> dict[str, Any]: ...
    def settlement_spot(self) -> float: ...
    def official_settlement(self, date: str) -> float: ...


# ---------------------------------------------------------------------------
# Interactive Brokers
# ---------------------------------------------------------------------------


class IBBroker:
    """``ib_async`` implementation.  Untested against a socket -- see README."""

    def __init__(self, config: Config, ib: Any | None = None):
        self.config = config
        self.tzinfo = ZoneInfo(config.tz)
        self.ib: Any = ib if ib is not None else IB()
        self._futures: dict[str, Any] = {}
        self._futures_qty: dict[str, int] = {}
        self._contract_cache: dict[tuple[float, str], Any] = {}
        self._indices: dict[str, Any] = {}
        self._health = QuoteHealth()
        self._log: list[str] = []

    # -- connection ------------------------------------------------------
    def connect(self, attempts: int = 3) -> None:
        last: Exception | None = None
        # A dry run must not be able to send an order even by accident (L-22).
        readonly = self.config.mode == "dry"
        for k in range(attempts):
            try:
                self.ib.connect(
                    self.config.host,
                    int(self.config.port or 7497),
                    clientId=self.config.client_id,
                    timeout=20,
                    readonly=readonly,
                )
                return
            except Exception as exc:  # pragma: no cover - needs a socket
                last = exc
                time.sleep(min(5.0 * (k + 1), self.config.disconnect_grace_s))
        raise BrokerError(
            "could not connect to "
            + self.config.host
            + ":"
            + str(self.config.port)
            + " after "
            + str(attempts)
            + " attempts: "
            + repr(last)
        )

    def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:  # pragma: no cover - defensive
            pass

    def is_connected(self) -> bool:
        try:
            return bool(self.ib.isConnected())
        except Exception:  # pragma: no cover - defensive
            return False

    def ensure_connected(self) -> bool:
        """Reconnect within the grace window.  True if we are live again."""
        if self.is_connected():
            return True
        deadline = time.time() + self.config.disconnect_grace_s
        while time.time() < deadline:
            try:
                self.connect(attempts=1)
                if self.is_connected():
                    return True
            except Exception:  # pragma: no cover - needs a socket
                pass
            time.sleep(3.0)
        return self.is_connected()

    # -- clock -----------------------------------------------------------
    def now_et(self) -> datetime:
        return datetime.now(self.tzinfo)

    def session_date(self) -> str:
        return self.now_et().strftime("%Y-%m-%d")

    def wait_until(self, clock: str) -> datetime:
        """Sleep (pumping the IB event loop) until the ET wall clock hits."""
        t = parse_clock(clock)
        now = self.now_et()
        target = now.replace(
            hour=t.hour, minute=t.minute, second=t.second, microsecond=0
        )
        while True:
            now = self.now_et()
            left = (target - now).total_seconds()
            if left <= 0:
                return now
            try:
                self.ib.sleep(min(left, 1.0))
            except Exception:  # pragma: no cover - no loop under test
                time.sleep(min(left, 1.0))

    # -- market data -----------------------------------------------------
    def _req_tickers(self, contracts: Sequence[Any]) -> list[Any]:
        """``reqTickers`` with a hard timeout (L-21).

        ``ib_async.reqTickersAsync`` is a bare ``gather``; one stuck snapshot
        would otherwise hang the runner past its clock with a position on.
        """
        if not contracts:
            return []
        try:
            coro = self.ib.reqTickersAsync(*contracts)
        except AttributeError:  # pragma: no cover - a stub IB in the tests
            return list(self.ib.reqTickers(*contracts))
        try:
            return list(self.ib.run(asyncio.wait_for(coro, SNAPSHOT_TIMEOUT_S)))
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise BrokerError(
                "market-data snapshot timed out after "
                + str(SNAPSHOT_TIMEOUT_S)
                + "s for "
                + str(len(contracts))
                + " contracts"
            ) from exc

    def _index_contract(self, symbol: str) -> Any:
        """A qualified CBOE index (``SPX``, or ``XSP`` for the Mini-SPX book)."""
        if symbol not in self._indices:
            idx = Index(symbol, "CBOE", "USD")
            self.ib.qualifyContracts(idx)
            self._indices[symbol] = idx
        return self._indices[symbol]

    def _spx_index(self) -> Any:
        """The instrument's own index contract (kept under its old name)."""
        return self._index_contract(self.config.spec.index_symbol)

    def _live_print(self, contract: Any) -> float:
        tickers = self._req_tickers([contract])
        if not tickers:
            return NAN
        t = tickers[0]
        for attr in ("last", "markPrice"):
            v = getattr(t, attr, None)
            if callable(v):  # markPrice is a method on ib_async's Ticker
                try:
                    v = v()
                except Exception:  # pragma: no cover - defensive
                    v = None
            if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
                return float(v)
        return NAN

    def index_spot(self) -> float:
        """The live print of the INSTRUMENT'S index.  Never yesterday's close (L-23).

        ``close`` is the previous session's settle; strike selection and the
        volatility inversion key off this number, so serving a stale close
        during RTH would pick the wrong strikes and invert the wrong vol.

        For XSP the print is the XSP index itself (one tenth of the S&P 500);
        if IB serves no XSP print, the SPX print scaled by ``index_scale`` is
        used and the fallback is logged (GUESS: IB lists ``XSP`` as a CBOE
        index; unverified without a socket).
        """
        spec = self.config.spec
        v = self._live_print(self._index_contract(spec.index_symbol))
        if math.isfinite(v):
            return v
        if spec.index_symbol != "SPX":
            spx = self._live_print(self._index_contract("SPX"))
            if math.isfinite(spx):
                self._log.append(
                    "index_spot: no live "
                    + spec.index_symbol
                    + " print; using SPX x "
                    + repr(spec.index_scale)
                )
                return spx * float(spec.index_scale)
        self._log.append("index_spot: no live last/markPrice (refused to use close)")
        return NAN

    def spx_spot(self) -> float:
        """Alias of :meth:`index_spot` (the Protocol's name)."""
        return self.index_spot()

    def spxw_0dte_contracts(self, expiry_yyyymmdd: str) -> list[Any]:
        """Every listed contract of the instrument for ``expiry_yyyymmdd`` (YYYYMMDD).

        SPXW by default; XSP under ``--instrument xsp`` (GUESS: IB lists the
        Mini-SPX weeklys as ``Option(symbol="XSP", tradingClass="XSP")`` on
        the ``XSP`` CBOE index; unverified without a socket).
        """
        spec = self.config.spec
        idx = self._spx_index()
        # Confirms the expiry is a listed expiration (preflight check).
        try:
            params = self.ib.reqSecDefOptParams(
                spec.option_symbol, "", "IND", idx.conId
            )
            listed = set()
            for p in params:
                if getattr(p, "tradingClass", "") == spec.trading_class:
                    listed |= set(p.expirations)
            if listed and expiry_yyyymmdd not in listed:
                return []
        except Exception as exc:  # pragma: no cover - needs a socket
            self._log.append("reqSecDefOptParams failed: " + repr(exc))
        # GUESS: exchange="SMART" routes the chain.  If SMART comes back empty
        # the fault is routing, not the calendar, so try CBOE before reporting
        # "not an expiry" (the mis-diagnosis ranked #4 in the audit).
        out: list[Any] = []
        for exch in ("SMART", "CBOE"):
            tmpl = Option(
                symbol=spec.option_symbol,
                lastTradeDateOrContractMonth=expiry_yyyymmdd,
                exchange=exch,
                currency="USD",
                tradingClass=spec.trading_class,
            )
            details = self.ib.reqContractDetails(tmpl)
            out = [d.contract for d in details if d.contract is not None]
            if out:
                if exch != "SMART":
                    self._log.append(
                        spec.trading_class + " chain came from " + exch + ", not SMART"
                    )
                break
        for c in out:
            self._contract_cache[(float(c.strike), str(c.right)[:1])] = c
        return out

    def contract_for(self, strike: float, right: str) -> Any:
        key = (float(strike), str(right)[:1])
        if key not in self._contract_cache:
            raise BrokerError("no qualified contract for " + repr(key))
        return self._contract_cache[key]

    def quotes(self, contracts: Sequence[Any]) -> list[Quote]:
        if not contracts:
            self._health = QuoteHealth(ok=True, n=0, detail="no contracts requested")
            return []
        tickers = self._req_tickers(contracts)
        now = datetime.now(ZoneInfo("UTC"))
        out: list[Quote] = []
        stale = halted = crossed = 0
        max_age = 0.0
        for t in tickers:
            c = t.contract
            bid = getattr(t, "bid", None)
            ask = getattr(t, "ask", None)
            b = float(bid) if bid is not None and math.isfinite(float(bid)) else 0.0
            a = float(ask) if ask is not None and math.isfinite(float(ask)) else 0.0
            # IB sends -1 for "no quote"; the research sentinel is bid==ask==0.
            if b < 0:
                b = 0.0
            if a < 0:
                a = 0.0
            ts = getattr(t, "time", None)
            if isinstance(ts, datetime):
                age = (
                    now - (ts if ts.tzinfo else ts.replace(tzinfo=now.tzinfo))
                ).total_seconds()
                max_age = max(max_age, age)
                if age > self.config.max_quote_age_s:
                    stale += 1
            h = getattr(t, "halted", None)
            if isinstance(h, (int, float)) and h > 0:
                halted += 1
            if b > 0.0 and a > 0.0 and b > a:
                crossed += 1
            out.append(
                Quote(strike=float(c.strike), right=str(c.right)[:1], bid=b, ask=a)
            )
        bad = []
        if stale:
            bad.append(str(stale) + " stale")
        if halted:
            bad.append(str(halted) + " halted")
        if crossed:
            bad.append(str(crossed) + " crossed")
        self._health = QuoteHealth(
            ok=not bad,
            n=len(out),
            stale=stale,
            halted=halted,
            crossed=crossed,
            max_age_s=max_age,
            detail=", ".join(bad),
        )
        return out

    def quote_health(self) -> QuoteHealth:
        return self._health

    # -- futures ---------------------------------------------------------
    def futures_contract(self, symbol: str = "ES") -> Any:
        """The front month by the contract's REAL expiry, with a roll rule."""
        sym = str(symbol).upper()
        if sym in self._futures:
            return self._futures[sym]
        tmpl = Future(symbol=sym, exchange="CME", currency="USD")
        details = self.ib.reqContractDetails(tmpl)
        today = self.now_et().date()
        cands: list[tuple[Any, Any]] = []
        for d in details:
            c = d.contract
            raw = str(
                getattr(d, "realExpirationDate", "")
                or (c.lastTradeDateOrContractMonth or "")
            )[:8]
            if len(raw) != 8:
                continue
            try:
                exp = datetime.strptime(raw, "%Y%m%d").date()
            except ValueError:
                continue
            # Roll a fixed number of calendar days before the contract's own
            # expiration; a trading-day count is what mis-fires in a holiday
            # week (L-27).
            if (exp - today).days < ROLL_DAYS_BEFORE_EXPIRY:
                continue
            cands.append((exp, c))
        if not cands:
            raise BrokerError(
                "no "
                + sym
                + " expiry more than "
                + str(ROLL_DAYS_BEFORE_EXPIRY)
                + " calendar days out"
            )
        cands.sort(key=lambda kv: kv[0])
        c = cands[0][1]
        self.ib.qualifyContracts(c)
        self._futures[sym] = c
        return c

    def futures_reference(self, symbol: str = "ES") -> float:
        c = self.futures_contract(symbol)
        tickers = self._req_tickers([c])
        if not tickers:
            return NAN
        t = tickers[0]
        bid = getattr(t, "bid", None)
        ask = getattr(t, "ask", None)
        if bid and ask and bid > 0 and ask > 0 and bid <= ask:
            return 0.5 * (float(bid) + float(ask))
        last = getattr(t, "last", None)
        return float(last) if last else NAN

    # -- orders ----------------------------------------------------------
    def _synthetic(
        self,
        action: str,
        quantity: int,
        limit: float,
        what: str,
        symbol: str,
        multiplier: float,
    ) -> Fill:
        note = (
            "DRY: would "
            + action
            + " "
            + str(quantity)
            + " "
            + what
            + " limit "
            + f"{limit:.4f}"
        )
        self._log.append(note)
        print("  " + note, flush=True)
        self._dry_order_id = getattr(self, "_dry_order_id", 0) + 1
        return Fill(
            ok=True,
            price=float(limit),
            quantity=int(quantity),
            action=action,
            what=what,
            limit=float(limit),
            synthetic=True,
            note=note,
            order_id=-self._dry_order_id,
            status="DryFilled",
            symbol=symbol,
            multiplier=multiplier,
        )

    def _work_order(
        self,
        contract: Any,
        action: str,
        quantity: int,
        limit: float,
        tick: float,
        passive_wait_s: float,
        max_cross_ticks: int,
        what: str,
        symbol: str = "",
        multiplier: float = NAN,
        deadline_s: float | None = None,
    ) -> Fill:
        """Rest at the limit, then reprice one tick at a time through it.

        ``deadline_s`` (seconds from now) stops the stepping when it passes --
        the month-end long's clock deadline; the unfilled remainder is then
        cancelled and ``Fill.deadline_hit`` says so.  ``None`` = no deadline
        beyond the tick allowance.
        """
        limit = round_to_tick(limit, tick)
        if self.config.mode == "dry":
            return self._synthetic(action, quantity, limit, what, symbol, multiplier)
        if not self.ensure_connected():
            raise BrokerError("disconnected while placing " + what)

        deadline_at = (
            None if deadline_s is None else time.time() + max(float(deadline_s), 0.0)
        )
        cross_sign = -1.0 if action.upper() == "SELL" else 1.0
        order = LimitOrder(action.upper(), abs(int(quantity)), limit)
        order.tif = "DAY"
        order.outsideRth = False
        if self.config.account:
            # FA / advisor logins reject an order with no account (L-28).
            order.account = self.config.account
        if COMBO_NON_GUARANTEED and getattr(contract, "secType", "") == "BAG":
            # GUESS: SMART refuses to leg some combos without the tag.
            order.smartComboRoutingParams = [TagValue("NonGuaranteed", "1")]
        trade = self.ib.placeOrder(contract, order)

        def _wait(seconds: float) -> None:
            end = time.time() + max(seconds, 0.0)
            if deadline_at is not None:
                end = min(end, deadline_at)
            while time.time() < end and not trade.isDone():
                self.ib.sleep(0.25)

        def _past_deadline() -> bool:
            return deadline_at is not None and time.time() >= deadline_at

        _wait(passive_wait_s)
        crossed = 0
        deadline_hit = False
        step = max(1.0, passive_wait_s / max(1, max_cross_ticks))
        while crossed < max_cross_ticks and not trade.isDone():
            if _past_deadline():
                deadline_hit = True
                break
            crossed += 1
            order.lmtPrice = round_to_tick(limit + cross_sign * crossed * tick, tick)
            self.ib.placeOrder(contract, order)
            _wait(step)
        if not trade.isDone() and _past_deadline():
            deadline_hit = True

        status = str(getattr(trade.orderStatus, "status", "") or "")
        filled = int(getattr(trade.orderStatus, "filled", 0) or 0)
        avg = float(getattr(trade.orderStatus, "avgFillPrice", NAN) or NAN)
        oid = int(getattr(order, "orderId", 0) or 0)
        last_log = ""
        try:
            entries = list(getattr(trade, "log", []) or [])
            if entries:
                last_log = str(getattr(entries[-1], "message", "") or "")
        except Exception:  # pragma: no cover - defensive
            last_log = ""
        if status == "Filled" and filled:
            return Fill(
                ok=True,
                price=avg,
                quantity=filled,
                action=action,
                what=what,
                limit=limit,
                crossed_ticks=crossed,
                order_id=oid,
                status=status,
                note=last_log,
                symbol=symbol,
                multiplier=multiplier,
                deadline_hit=deadline_hit,
            )
        try:
            self.ib.cancelOrder(order)
            self.ib.sleep(1.0)
        except Exception:  # pragma: no cover - defensive
            pass
        status = str(getattr(trade.orderStatus, "status", "") or status)
        filled = int(getattr(trade.orderStatus, "filled", 0) or 0)
        avg = float(getattr(trade.orderStatus, "avgFillPrice", NAN) or NAN)
        return Fill(
            ok=False,
            price=avg if filled else NAN,
            quantity=filled,
            action=action,
            what=what,
            limit=limit,
            crossed_ticks=crossed,
            order_id=oid,
            status=status,
            note=(
                "status="
                + (status or "unknown")
                + "; filled "
                + str(filled)
                + "/"
                + str(abs(int(quantity)))
                + " after "
                + str(crossed)
                + " ticks through; cancelled"
                + ("; deadline passed" if deadline_hit else "")
                + (" | " + last_log if last_log else "")
            ),
            symbol=symbol,
            multiplier=multiplier,
            deadline_hit=deadline_hit,
        )

    def place_combo(
        self,
        legs: list[tuple[Any, int, str]],
        quantity: int,
        limit_price: float,
        action: str = "SELL",
        passive_wait_s: float | None = None,
        spread: float | None = None,
        what: str = "combo",
        max_cross_ticks: int | None = None,
        deadline_s: float | None = None,
    ) -> Fill:
        """``legs`` define the LONG package; ``action`` trades the bag.

        ``max_cross_ticks`` caps the repricing for this one order; ``0`` rests
        at the limit for ``passive_wait_s`` and then cancels.  ``deadline_s``
        stops the stepping after that many seconds (the month-end long's
        clock deadline); ``None`` = the tick allowance alone bounds it.
        """
        cfg = self.config
        bag = Contract(
            secType="BAG",
            symbol=cfg.spec.option_symbol,
            exchange="SMART",
            currency="USD",
            comboLegs=[
                ComboLeg(
                    conId=int(c.conId),
                    ratio=int(ratio),
                    action=str(leg_action).upper(),
                    exchange="SMART",
                    openClose=0,
                )
                for (c, ratio, leg_action) in legs
            ],
        )
        tick = cfg.option_tick_for(limit_price)
        return self._work_order(
            bag,
            action,
            quantity,
            limit_price,
            tick,
            cfg.passive_wait_s if passive_wait_s is None else passive_wait_s,
            cfg.cross_ticks_for(NAN if spread is None else spread, tick)
            if max_cross_ticks is None
            else max(0, int(max_cross_ticks)),
            what,
            symbol=cfg.spec.trading_class,
            multiplier=cfg.index_multiplier,
            deadline_s=deadline_s,
        )

    def place_futures(
        self,
        contract: Any,
        signed_qty: int,
        passive_wait_s: float | None = None,
        what: str = "futures",
    ) -> Fill:
        cfg = self.config
        sym = str(getattr(contract, "symbol", "ES") or "ES").upper()
        mult = cfg.multiplier(sym)
        if signed_qty == 0:
            return Fill(
                True, NAN, 0, "NONE", what, note="no trade", symbol=sym, multiplier=mult
            )
        action = "BUY" if signed_qty > 0 else "SELL"
        ref = self.futures_reference(sym)
        if not math.isfinite(ref):
            raise BrokerError("no " + sym + " reference price")
        # Rest at the touch: buy at the bid side, sell at the ask side.
        limit = (
            ref - cfg.futures_tick / 2.0
            if signed_qty > 0
            else (ref + cfg.futures_tick / 2.0)
        )
        fill = self._work_order(
            contract,
            action,
            abs(int(signed_qty)),
            limit,
            cfg.futures_tick,
            cfg.passive_wait_s if passive_wait_s is None else passive_wait_s,
            cfg.cross_ticks_for(cfg.futures_tick, cfg.futures_tick),
            what,
            symbol=sym,
            multiplier=mult,
        )
        # A PARTIAL fill moved the position too: sign and book it on the
        # filled quantity, not on ``ok`` (L-3).
        if fill.quantity:
            signed = int(math.copysign(abs(fill.quantity), signed_qty))
            fill.quantity = signed
            self._futures_qty[sym] = self._futures_qty.get(sym, 0) + signed
        return fill

    # -- state -----------------------------------------------------------
    def positions(self) -> dict[str, Any]:
        """Open positions.  Raises rather than reporting a false flat (L-10)."""
        opts: dict[int, float] = {}
        fut: dict[str, int] = {}
        raw: list[dict[str, Any]] = []
        try:
            rows = list(self.ib.positions(self.config.account or ""))
        except Exception as exc:
            raise BrokerError("positions() failed: " + repr(exc)) from exc
        for p in rows:
            c = p.contract
            raw.append(
                {
                    "secType": c.secType,
                    "symbol": c.symbol,
                    "strike": getattr(c, "strike", None),
                    "right": getattr(c, "right", None),
                    "expiry": getattr(c, "lastTradeDateOrContractMonth", None),
                    "position": float(p.position),
                }
            )
            if c.secType == "OPT" and c.symbol == self.config.spec.option_symbol:
                opts[int(c.conId)] = float(p.position)
            elif c.secType == "FUT" and str(c.symbol).upper() in FUTURES_MULTIPLIER:
                sym = str(c.symbol).upper()
                fut[sym] = fut.get(sym, 0) + int(p.position)
        return {
            "options": opts,
            "futures": {k: v for k, v in fut.items() if v},
            "raw": raw,
            "stale": False,
        }

    def liquid_hours(self, contract: Any) -> tuple[datetime, datetime]:
        """Today's regular session, in ET, read in the contract's OWN zone."""
        details = self.ib.reqContractDetails(contract)
        if not details:
            raise BrokerError("no contractDetails for liquid hours")
        d = details[0]
        text = str(getattr(d, "liquidHours", "") or "")
        tzname = str(getattr(d, "timeZoneId", "") or "")
        try:
            zone = ZoneInfo(tzname) if tzname else self.tzinfo
        except Exception:
            self._log.append("unknown timeZoneId " + repr(tzname) + "; assuming ET")
            zone = self.tzinfo
        day = self.now_et().strftime("%Y%m%d")
        found: list[tuple[datetime, datetime]] = []
        for seg in text.split(";"):
            seg = seg.strip()
            if not seg or "CLOSED" in seg.upper() or not seg.startswith(day):
                continue
            # "20231124:0930-20231124:1300"
            try:
                lhs, rhs = seg.split("-", 1)
                o = datetime.strptime(lhs.strip(), "%Y%m%d:%H%M")
                cl_txt = rhs.strip()
                if ":" in cl_txt and len(cl_txt) > 5:
                    cl = datetime.strptime(cl_txt, "%Y%m%d:%H%M")
                else:  # pragma: no cover - older TWS format
                    cl = datetime.strptime(day + ":" + cl_txt, "%Y%m%d:%H%M")
            except ValueError:
                continue
            found.append(
                (
                    o.replace(tzinfo=zone).astimezone(self.tzinfo),
                    cl.replace(tzinfo=zone).astimezone(self.tzinfo),
                )
            )
        if not found:
            raise BrokerError(
                "no liquid-hours segment for " + day + " in " + repr(text)
            )
        # A multi-segment day: the regular session is the widest one.
        found.sort(key=lambda oc: oc[1] - oc[0], reverse=True)
        return found[0]

    def account_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            rows = list(self.ib.accountSummary(self.config.account or ""))
        except Exception as exc:
            raise BrokerError("accountSummary failed: " + repr(exc)) from exc
        for row in rows:
            out[row.tag] = row.value
            acct = str(getattr(row, "account", "") or "")
            if acct:
                out.setdefault("AccountId", acct)
        return out

    def settlement_spot(self) -> float:
        """Provisional settlement: the last index print.  Reconciled am."""
        return self.spx_spot()

    def official_settlement(self, date: str) -> float:
        """The official settlement for ``date`` (YYYY-MM-DD), or NaN.

        SPXW and XSP are PM-settled: the settlement is the official 16:00 SPX
        close (times ``index_scale`` for XSP -- Cboe settles XSP to one tenth
        of the SPX close, so the SPX bar is the source for both).  IB
        publishes the close as the daily bar's close.  Anything unexpected
        returns NaN so that ``--reconcile`` falls back to the manual value
        rather than finalising a day on a guess.
        """
        end = date.replace("-", "") + " 23:59:59 " + self.config.tz
        try:
            bars = self.ib.reqHistoricalData(
                self._index_contract("SPX"),
                endDateTime=end,
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
            )
        except Exception as exc:  # pragma: no cover - needs a socket
            self._log.append("official_settlement failed: " + repr(exc))
            return NAN
        for b in reversed(list(bars or [])):
            if str(getattr(b, "date", ""))[:10] == date:
                v = float(getattr(b, "close", NAN))
                return (
                    v * float(self.config.index_scale)
                    if math.isfinite(v) and v > 0
                    else NAN
                )
        return NAN


# ---------------------------------------------------------------------------
# Recorded replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeOption:
    """Stand-in for a qualified ``Option``; carries what the runner reads."""

    strike: float
    right: str
    conId: int
    lastTradeDateOrContractMonth: str = ""
    symbol: str = "SPX"
    secType: str = "OPT"
    tradingClass: str = "SPXW"


#: Fixed conIds the replay hands each hedge future.
FAKE_FUTURE_CONID: dict[str, int] = {"ES": 900001, "MES": 900002, "NES": 900003}


def synthetic_xsp_frame(frame: Any, index_scale: float) -> Any:
    """A SYNTHETIC Mini-SPX tape from the recorded SPXW day: scale by ``index_scale``.

    XSP has no tape in this repository.  To exercise the XSP code path the
    replay takes the SPXW session, keeps the strikes that land on XSP's
    whole-dollar grid (every second 5-point SPX strike: 6,500 -> 650.0,
    6,510 -> 651.0), and divides strikes, quotes and the underlying by ten.
    Quotes are then snapped to XSP's $0.01 tick.  This is a stand-in for the
    CODE PATH -- sizing, the ladder, the ledger scaling, settlement -- and
    says nothing about how XSP actually quotes (its relative spread is
    several times SPXW's); the runner journals ``synthetic_xsp_from_spxw``
    on every such replay.
    """
    scale = float(index_scale)
    step = round(1.0 / scale, 6)  # SPX points per whole instrument point
    df = frame.copy()
    strikes = df["strike"].astype(float)
    on_grid = (strikes / step - (strikes / step).round()).abs() < 1e-9
    df = df[on_grid].copy()
    df["strike"] = (df["strike"].astype(float) * scale).round(6)
    for col in ("bid", "ask"):
        v = df[col].astype(float) * scale
        df[col] = (v / 0.01).round() * 0.01
    df["underlying_price"] = df["underlying_price"].astype(float) * scale
    df.attrs["synthetic_xsp_from_spxw"] = True
    return df.sort_values(["et", "strike", "cp"]).reset_index(drop=True)


@dataclass(frozen=True)
class FakeFuture:
    """Stand-in for a qualified ``Future``."""

    symbol: str = "ES"
    conId: int = 900001
    secType: str = "FUT"
    lastTradeDateOrContractMonth: str = ""


def _cache_name(date: str, chain_path: str, band_pct: float) -> str:
    """Key the cache on the date AND the band it was cut with (L-25)."""
    key = os.path.abspath(chain_path) + "|" + format(float(band_pct), ".6f")
    return date + "_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12] + ".parquet"


def load_replay_day(
    date: str,
    chain_path: str = CHAIN_PARQUET,
    cache_dir: str | None = None,
    band_pct: float = 0.01,
    tz: str = "America/New_York",
) -> Any:
    """Read one session's ATM band out of the 74 MB chain, with a cache.

    ONE filtered read per replay: pyarrow pushes ``expiration == date`` down
    into the file and only the eight columns the runner needs come back.

    The band is the union, over every stamp of the day, of strikes within
    ``band_pct`` of that stamp's underlying; all stamps are then kept for
    those strikes, so an entry at any clock still has a 15:30 and a 16:00
    quote no matter how far spot travelled.  Wing strikes (1.5 % of spot)
    fall outside the band on purpose: the replay cannot quote them, and the
    runner must refuse to enter a fly it cannot hedge.

    The cache is keyed on ``(date, chain_path, band_pct)`` and lives wherever
    the caller puts it -- never in the live journal tree.
    """
    import pandas as pd  # local: keeps `import broker` cheap for config-only use

    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, _cache_name(date, chain_path, band_pct))
    if cache and os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        import pyarrow.parquet as pq

        stamp = pd.Timestamp(date)
        table = pq.read_table(
            chain_path,
            columns=[
                "timestamp",
                "expiration",
                "strike",
                "cp",
                "bid",
                "ask",
                "underlying_price",
                "hours_to_expiration",
            ],
            filters=[("expiration", "==", stamp)],
        )
        df = table.to_pandas()
        if df.empty:
            raise BrokerError(
                "no SPXW chain rows for "
                + date
                + " in "
                + chain_path
                + " (not a 0DTE session on this tape?)"
            )
        keep: set[float] = set()
        for _, g in df.groupby("timestamp"):
            s = float(g["underlying_price"].iloc[0])
            if not math.isfinite(s):
                continue
            w = band_pct * s
            keep |= set(
                g.loc[(g["strike"] >= s - w) & (g["strike"] <= s + w), "strike"]
                .astype(float)
                .tolist()
            )
        df = df[df["strike"].astype(float).isin(keep)].copy()
        if cache:
            df.to_parquet(cache, index=False)
    df = df.copy()
    df["et"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
    df["hhmm"] = df["et"].dt.strftime("%H:%M")
    return df.sort_values(["et", "strike", "cp"]).reset_index(drop=True)


@dataclass
class FakeFaults:
    """What the replay is told to do wrong, per order ``what``.

    * ``refuse``   -- the order rests and never fills (``ok`` False, 0 filled)
    * ``reject``   -- the venue rejects it (``ok`` False, 0 filled, a reason)
    * ``partial``  -- only this many contracts fill (``ok`` False, signed)
    * ``disconnect_at`` -- the socket is down on arrival at these clocks
    * ``stale_at`` -- the snapshot at these clocks is stale/halted
    """

    refuse: frozenset[str] = frozenset()
    reject: Mapping[str, str] = field(default_factory=dict)
    partial: Mapping[str, int] = field(default_factory=dict)
    disconnect_at: frozenset[str] = frozenset()
    stale_at: frozenset[str] = frozenset()
    #: seconds the simulated clock advances while reconnecting
    reconnect_delay_s: float = 0.0
    #: the order fills only after this many ticks through the limit (at
    #: limit + ticks x tick); fewer allowed -- by the cap or by the deadline --
    #: and it rests unfilled and is cancelled (the chase tests)
    needs_ticks: Mapping[str, int] = field(default_factory=dict)


class FakeBroker:
    """Replays one recorded session.  Never opens a socket.

    The clock is simulated: ``wait_until``/``advance_to`` jump to the stamp
    and every quote, spot and fill is taken from that stamp's recorded rows.
    Any stamp the tape carries (10:00 .. 15:30, plus the 16:00 settlement
    print) and any session in the chain can be served, so the afternoon
    entries and the hold-to-settlement terminal both replay.
    """

    def __init__(
        self,
        config: Config,
        date: str | None = None,
        chain_path: str = CHAIN_PARQUET,
        fill_mode: str | None = None,
        serve_band_pct: float = 0.0075,
        liquid_close: str = "16:00",
        futures_slippage_pts: float = 0.0,
        futures_basis_pts: float = 18.0,
        faults: FakeFaults | None = None,
        frame: Any | None = None,
    ):
        self.config = config
        self.date = date or config.replay_date or ""
        if not self.date:
            raise BrokerError("FakeBroker needs a replay date")
        if not REPLAY_FIRST_DATE <= self.date <= REPLAY_LAST_DATE:
            raise BrokerError(
                "the recorded chain covers "
                + REPLAY_FIRST_DATE
                + " .. "
                + REPLAY_LAST_DATE
                + "; "
                + self.date
                + " is outside it"
            )
        self.tzinfo = ZoneInfo(config.tz)
        self.fill_mode: str = str(fill_mode or config.replay_fill or "cross")
        self.serve_band_pct = serve_band_pct
        self.liquid_close = liquid_close
        self.futures_slippage_pts = futures_slippage_pts
        #: The futures trade at a basis to the index; marking the hedge at the
        #: index is exactly the phantom-P&L bug L-5, so the replay must not
        #: quote them at the same number.
        self.futures_basis_pts = futures_basis_pts
        self.faults = faults or FakeFaults()
        self.frame = (
            frame
            if frame is not None
            else load_replay_day(
                self.date, chain_path, config.replay_cache_dir, tz=config.tz
            )
        )
        #: The recorded tape is SPXW.  Any other instrument is served as a
        #: SYNTHETIC rescaling of it (see ``synthetic_xsp_frame``).
        self.synthetic: bool = False
        if config.spec.key != "spxw" and not bool(
            getattr(self.frame, "attrs", {}).get("synthetic_xsp_from_spxw", False)
        ):
            self.frame = synthetic_xsp_frame(self.frame, config.index_scale)
            self.synthetic = True
        elif bool(
            getattr(self.frame, "attrs", {}).get("synthetic_xsp_from_spxw", False)
        ):
            self.synthetic = True
        self.clocks: list[str] = sorted(set(self.frame["hhmm"].tolist()))
        self._listed: set[tuple[float, str]] = {
            (float(k), str(r)[:1])
            for k, r in zip(self.frame["strike"], self.frame["cp"])
        }
        self._clock: str = config.fixed_entry_clock
        self._offset_s: float = 0.0
        self.connected = False
        self._futures_qty: dict[str, int] = {}
        self.futures_fills: list[Fill] = []
        self._conid: dict[tuple[float, str], int] = {}
        self.orders: list[dict[str, Any]] = []
        self._health = QuoteHealth()
        self._order_id = 0
        self._down_seen: set[str] = set()

    # -- connection (no network) ----------------------------------------
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def ensure_connected(self) -> bool:
        if not self.connected:
            self._offset_s += float(self.faults.reconnect_delay_s)
            self.connected = True
        return self.connected

    # -- clock -----------------------------------------------------------
    def _clock_key(self, clock: str) -> str:
        return clock[:5]

    def now_et(self) -> datetime:
        t = parse_clock(self._clock)
        d = datetime.strptime(self.date, "%Y-%m-%d")
        return d.replace(
            hour=t.hour, minute=t.minute, second=t.second, tzinfo=self.tzinfo
        ) + timedelta(seconds=self._offset_s)

    def session_date(self) -> str:
        return self.date

    def advance_to(self, clock: str) -> datetime:
        key = self._clock_key(clock)
        if key not in self.clocks:
            raise BrokerError(
                "the recorded tape has no "
                + key
                + " stamp for "
                + self.date
                + " (30-minute stamps only: "
                + ", ".join(self.clocks)
                + ")"
            )
        self._clock = clock
        if key in self.faults.disconnect_at and key not in self._down_seen:
            self._down_seen.add(key)
            self.connected = False
        return self.now_et()

    def wait_until(self, clock: str) -> datetime:
        return self.advance_to(clock)

    # -- market data -----------------------------------------------------
    def _rows(self, clock: str | None = None) -> Any:
        key = self._clock_key(clock or self._clock)
        return self.frame[self.frame["hhmm"] == key]

    def spx_spot(self) -> float:
        rows = self._rows()
        if rows.empty:
            return NAN
        return float(rows["underlying_price"].iloc[0])

    def settlement_spot(self) -> float:
        """The 16:00 print.  Refuses a tape that stops at 15:30 (L-46)."""
        last = self.clocks[-1]
        if last != "16:00":
            raise BrokerError(
                "the tape for "
                + self.date
                + " ends at "
                + last
                + ": there is no settlement print to serve"
            )
        rows = self._rows(last)
        if rows.empty:
            return NAN
        return float(rows["underlying_price"].iloc[0])

    def official_settlement(self, date: str) -> float:
        """The replay's stand-in for the broker statement: the 16:00 print."""
        if date != self.date:
            raise BrokerError("the replay holds " + self.date + ", not " + date)
        return self.settlement_spot()

    def _conid_for(self, strike: float, right: str) -> int:
        key = (float(strike), right)
        if key not in self._conid:
            self._conid[key] = 100000 + len(self._conid)
        return self._conid[key]

    def spxw_0dte_contracts(self, expiry_yyyymmdd: str) -> list[Any]:
        """Strikes the replay serves: the ATM band at the current stamp.

        Deliberately narrower than a live chain.  Wings at 1.5% of spot are
        not in it, so ``pick_wings`` returns None in replay.
        """
        if expiry_yyyymmdd != self.date.replace("-", ""):
            return []
        rows = self._rows()
        if rows.empty:
            return []
        s = float(rows["underlying_price"].iloc[0])
        w = self.serve_band_pct * s
        sel = rows[(rows["strike"] >= s - w) & (rows["strike"] <= s + w)]
        out: list[Any] = []
        for _, r in sel.iterrows():
            right = str(r["cp"])[:1]
            k = float(r["strike"])
            out.append(
                FakeOption(
                    strike=k,
                    right=right,
                    conId=self._conid_for(k, right),
                    lastTradeDateOrContractMonth=expiry_yyyymmdd,
                    symbol=self.config.spec.option_symbol,
                    tradingClass=self.config.spec.trading_class,
                )
            )
        return out

    def contract_for(self, strike: float, right: str) -> Any:
        """Mirror ``IBBroker``'s refusal: an unlisted strike has no contract."""
        key = (float(strike), str(right)[:1])
        if key not in self._listed:
            raise BrokerError("no qualified contract for " + repr(key))
        return FakeOption(
            strike=key[0],
            right=key[1],
            conId=self._conid_for(key[0], key[1]),
            lastTradeDateOrContractMonth=self.date.replace("-", ""),
            symbol=self.config.spec.option_symbol,
            tradingClass=self.config.spec.trading_class,
        )

    def quotes(self, contracts: Sequence[Any]) -> list[Quote]:
        rows = self._rows()
        table: dict[tuple[float, str], tuple[float, float]] = {}
        for _, r in rows.iterrows():
            b = float(r["bid"])
            a = float(r["ask"])
            table[(float(r["strike"]), str(r["cp"])[:1])] = (
                0.0 if not math.isfinite(b) or b < 0 else b,
                0.0 if not math.isfinite(a) or a < 0 else a,
            )
        out: list[Quote] = []
        for c in contracts:
            key = (float(c.strike), str(c.right)[:1])
            # Not on the tape -> the no-quote sentinel, exactly as the vendor
            # writes an unquoted contract.
            b, a = table.get(key, (0.0, 0.0))
            out.append(Quote(strike=key[0], right=key[1], bid=b, ask=a))
        sick = self._clock_key(self._clock) in self.faults.stale_at
        self._health = QuoteHealth(
            ok=not sick,
            n=len(out),
            stale=len(out) if sick else 0,
            max_age_s=60.0 if sick else 0.0,
            detail="replay: forced stale snapshot" if sick else "",
        )
        return out

    def quote_health(self) -> QuoteHealth:
        return self._health

    # -- futures ---------------------------------------------------------
    def futures_contract(self, symbol: str = "ES") -> Any:
        sym = str(symbol).upper()
        self.config.multiplier(sym)  # rejects anything but ES/MES/NES
        return FakeFuture(symbol=sym, conId=FAKE_FUTURE_CONID[sym])

    def futures_reference(self, symbol: str = "ES") -> float:
        """The FUTURES price: the S&P 500 index plus a basis, never the index (L-5).

        The futures are on the S&P 500 itself, so the instrument's spot is
        taken back to S&P points (``/ index_scale``) before the basis.
        """
        s = self.spx_spot()
        if not math.isfinite(s):
            return NAN
        return s / float(self.config.index_scale) + self.futures_basis_pts

    # -- orders ----------------------------------------------------------
    def package_quote(
        self, Kc: float, Kp: float, wings: tuple[float, float] | None = None
    ) -> tuple[float, float, float]:
        """(bid, ask, mid) of the package at the current stamp."""
        legs = [(Kc, "C", 1), (Kp, "P", 1)]
        if wings is not None:
            legs += [(wings[0], "C", -1), (wings[1], "P", -1)]
        qs = self.quotes([self.contract_for(k, r) for (k, r, _) in legs])
        bid = 0.0
        ask = 0.0
        for q, (_, _, sgn) in zip(qs, legs):
            if not q.live:
                return NAN, NAN, NAN
            if sgn > 0:
                bid += q.bid
                ask += q.ask
            else:
                bid -= q.ask
                ask -= q.bid
        return bid, ask, 0.5 * (bid + ask)

    def _next_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def _fault(self, what: str, quantity: int) -> tuple[str, int, str] | None:
        """(status, filled, note) when a fault is configured for ``what``."""
        if what in self.faults.refuse:
            return "Cancelled", 0, "replay: refused (rested, never filled)"
        if what in self.faults.reject:
            return "Rejected", 0, "replay: " + str(self.faults.reject[what])
        if what in self.faults.partial:
            want = abs(int(quantity))
            got = min(want, abs(int(self.faults.partial[what])))
            if got < want:
                return (
                    "Submitted",
                    got,
                    "replay: partial fill " + str(got) + "/" + str(want),
                )
        return None

    def place_combo(
        self,
        legs: list[tuple[Any, int, str]],
        quantity: int,
        limit_price: float,
        action: str = "SELL",
        passive_wait_s: float | None = None,
        spread: float | None = None,
        what: str = "combo",
        max_cross_ticks: int | None = None,
        deadline_s: float | None = None,
    ) -> Fill:
        """Fill at the package touch (``cross``) or at the limit (``mid``).

        With ``max_cross_ticks=0`` an order that is not marketable at the
        stamp's touch rests and is cancelled, as ``IBBroker`` would cancel it:
        the replay does not fill a limit the live order could not reach.
        ``FakeFaults.needs_ticks`` makes the fill cost that many ticks through
        the limit and refuses it when the allowance -- the cap, or the
        deadline at ``IBBroker``'s own step timing -- is smaller.
        """
        oid = self._next_id()
        pkg_bid = 0.0
        pkg_ask = 0.0
        qs = self.quotes([c for (c, _, _) in legs])
        for q, (_, ratio, leg_action) in zip(qs, legs):
            sgn = 1 if str(leg_action).upper() == "BUY" else -1
            if not q.live:
                return Fill(
                    False,
                    NAN,
                    0,
                    action,
                    what,
                    limit_price,
                    note="leg " + str(q.strike) + q.right + " has no quote",
                    order_id=oid,
                    status="NoQuote",
                    symbol=self.config.spec.trading_class,
                    multiplier=self.config.index_multiplier,
                )
            if sgn > 0:
                pkg_bid += ratio * q.bid
                pkg_ask += ratio * q.ask
            else:
                pkg_bid -= ratio * q.ask
                pkg_ask -= ratio * q.bid
        if self.fill_mode == "mid":
            price: float = float(limit_price)
            crossed = 0
        else:
            price = float(pkg_bid if action.upper() == "SELL" else pkg_ask)
            tick = self.config.option_tick_for(price)
            crossed = int(round(abs(price - float(limit_price)) / max(tick, 1e-9)))
        fault = self._fault(what, quantity)
        deadline_hit = False
        need = int(self.faults.needs_ticks.get(what, 0))
        if fault is None and need > 0:
            tick = self.config.option_tick_for(float(limit_price))
            allowed = (
                self.config.cross_ticks_for(NAN if spread is None else spread, tick)
                if max_cross_ticks is None
                else max(0, int(max_cross_ticks))
            )
            wait = (
                self.config.passive_wait_s
                if passive_wait_s is None
                else float(passive_wait_s)
            )
            if deadline_s is not None and allowed > 0:
                # IBBroker's own timing: rest passive_wait_s, then one step per
                # max(1, passive_wait_s / allowed) seconds until the deadline.
                step = max(1.0, wait / max(1, allowed))
                by_deadline = int(max(float(deadline_s) - wait, 0.0) // step)
                if by_deadline < allowed:
                    allowed, deadline_hit = by_deadline, True
            if allowed >= need:
                # exactly limit + need ticks: the accounting identity the runner's
                # slippage fields assume (IBBroker aligns each step to the tick)
                chase_sign = 1.0 if action.upper() == "BUY" else -1.0
                price = float(limit_price) + chase_sign * float(need) * tick
                crossed = need
                deadline_hit = False
            else:
                fault = (
                    "Cancelled",
                    0,
                    "replay: the fill needs "
                    + str(need)
                    + " ticks through the limit, "
                    + str(allowed)
                    + " allowed"
                    + ("; deadline passed" if deadline_hit else "")
                    + "; cancelled",
                )
        touch = pkg_ask if action.upper() == "BUY" else pkg_bid
        unreachable = (
            max_cross_ticks is not None
            and int(max_cross_ticks) <= 0
            and (
                float(limit_price) < touch
                if action.upper() == "BUY"
                else float(limit_price) > touch
            )
        )
        if fault is None and unreachable:
            fault = (
                "Cancelled",
                0,
                "replay: limit "
                + format(float(limit_price), ".4f")
                + " is not marketable at the touch "
                + format(touch, ".4f")
                + " and crossing is off; cancelled",
            )
        status, filled, note = (
            fault
            if fault is not None
            else ("Filled", abs(int(quantity)), "replay fill (" + self.fill_mode + ")")
        )
        rec = {
            "clock": self._clock,
            "what": what,
            "action": action,
            "quantity": filled,
            "limit": limit_price,
            "fill": price if filled else NAN,
            "status": status,
        }
        self.orders.append(rec)
        return Fill(
            ok=status == "Filled",
            price=float(price) if filled else NAN,
            quantity=int(filled),
            action=action,
            what=what,
            limit=float(limit_price),
            crossed_ticks=crossed,
            note=note,
            order_id=oid,
            deadline_hit=deadline_hit,
            status=status,
            symbol=self.config.spec.trading_class,
            multiplier=self.config.index_multiplier,
        )

    def place_futures(
        self,
        contract: Any,
        signed_qty: int,
        passive_wait_s: float | None = None,
        what: str = "futures",
    ) -> Fill:
        sym = str(getattr(contract, "symbol", "ES") or "ES").upper()
        mult = self.config.multiplier(sym)
        oid = self._next_id()
        if signed_qty == 0:
            return Fill(
                True,
                NAN,
                0,
                "NONE",
                what,
                note="no trade",
                order_id=oid,
                status="NoTrade",
                symbol=sym,
                multiplier=mult,
            )
        ref = self.futures_reference(sym)
        price = ref + math.copysign(self.futures_slippage_pts, signed_qty)
        fault = self._fault(what, signed_qty)
        if fault is None:
            status, got, note = (
                "Filled",
                abs(int(signed_qty)),
                "replay futures fill at the futures reference",
            )
        else:
            status, got, note = fault
        signed = int(math.copysign(got, signed_qty)) if got else 0
        if signed:
            self._futures_qty[sym] = self._futures_qty.get(sym, 0) + signed
        f = Fill(
            ok=status == "Filled",
            price=float(price) if got else NAN,
            quantity=signed,
            action="BUY" if signed_qty > 0 else "SELL",
            what=what,
            limit=float(ref),
            note=note,
            order_id=oid,
            status=status,
            symbol=sym,
            multiplier=mult,
        )
        self.futures_fills.append(f)
        self.orders.append(
            {
                "clock": self._clock,
                "what": what,
                "action": f.action,
                "quantity": signed,
                "fill": f.price,
                "status": status,
                "symbol": sym,
            }
        )
        return f

    def positions(self) -> dict[str, Any]:
        return {
            "options": {},
            "futures": {k: v for k, v in self._futures_qty.items() if v},
            "raw": [],
            "stale": False,
        }

    def liquid_hours(self, contract: Any) -> tuple[datetime, datetime]:
        d = datetime.strptime(self.date, "%Y-%m-%d")
        t = parse_clock(self.liquid_close)
        return (
            d.replace(hour=9, minute=30, tzinfo=self.tzinfo),
            d.replace(hour=t.hour, minute=t.minute, tzinfo=self.tzinfo),
        )

    def account_summary(self) -> dict[str, Any]:
        return {
            "AccountType": "REPLAY",
            "AccountId": "DU0000000",
            "NetLiquidation": "0",
            "note": "FakeBroker: no account, no network",
        }


def replay_sessions(
    chain_path: str = CHAIN_PARQUET, limit: int | None = None
) -> list[str]:
    """Every session the recorded chain can replay, as ``YYYY-MM-DD``."""
    import pyarrow.parquet as pq

    table = pq.read_table(chain_path, columns=["expiration"])
    import pandas as pd

    col: Iterable[Any] = pd.unique(table.column("expiration").to_pandas())
    days = sorted({pd.Timestamp(v).strftime("%Y-%m-%d") for v in col})
    return days[:limit] if limit else days
