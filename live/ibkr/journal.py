"""JSONL journal for one session, and the day summary recomputed from it.

The journal is the only record of what happened: every quote the runner
acted on, every order it sent, every fill it got, and every delta it
computed lands in ``<journal_dir>/<date>.jsonl`` as one JSON object per
line.  ``DaySummary.from_journal`` reads that file back and recomputes the
day's P&L from the *fills alone*, so the morning reconciliation compares two
independent things: the broker's statement and this file.

P&L convention (short body, the research convention):

    pnl_units = (entry_fill - exit_fill + hedge_pnl_points) / entry_fill

where ``entry_fill``/``exit_fill`` are package prices per straddle in index
points and ``hedge_pnl_points`` is the futures P&L expressed in index points
of one straddle (dollars / (n * index_multiplier)).  In the ``hold`` book
``exit_fill`` is the cash-settlement payoff instead of a buy-back fill.

The month-end override BUYS the body (``calendar_guard``), and its sign comes
from the entry fill's own ``action`` -- a ``body_entry`` fill that is a BUY is
a long day -- never from a flag the runner could forget to write:

    pnl_units = (exit_fill - entry_fill + hedge_pnl_points) / entry_fill

which with no hedge is the research's long return, ``payoff / ask - 1``.

Four things this file refuses to do, each of which produced a confident wrong
number in the 2026-09-18 code audit (L-11 .. L-14):

* it never marks an open futures leg at **zero** -- it marks it at the
  futures reference the runner journaled, and NaNs the day when there is
  none;
* it never double counts a re-run: the writer rolls to ``<date>.1.jsonl``
  rather than appending to a finished journal, and fills are deduplicated by
  ``order_id`` inside one file;
* it never defaults a multiplier -- without a ``config`` record the
  multipliers are NaN and no P&L is computed;
* it never counts an order that did not trade: a fill contributes on its
  **filled quantity**, so a rejection (quantity 0) is excluded and a partial
  contributes exactly what it filled.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, TextIO
from zoneinfo import ZoneInfo

NAN = float("nan")

EVENT_KINDS = (
    "config",
    "preflight",
    "calendar",
    "ledger",
    "regime",
    "selector",
    "sizing",
    "entry",
    "quote",
    "delta",
    "target",
    "order",
    "fill",
    "rebalance",
    "exit",
    "flatten",
    "mark",
    "settle",
    "reconcile",
    "error",
    "kill",
)


def _jsonable(obj: Any) -> Any:
    """Make numpy scalars / datetimes / NaN survive ``json.dumps``."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _jsonable(obj.item())
        except Exception:  # pragma: no cover - defensive
            return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def next_journal_path(path: str) -> str:
    """``path`` if free, else ``<stem>.1.jsonl``, ``<stem>.2.jsonl``, ...

    A second run of the same date is a second session, not more lines of the
    first one: appending is how one journal came to report ``hedge$ 500 ->
    1000`` and a still-plausible P&L (L-12).
    """
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(stem + "." + str(k) + ext):
        k += 1
    return stem + "." + str(k) + ext


class Journal:
    """Append-only JSONL writer.  Every line is flushed AND fsynced."""

    def __init__(
        self,
        path: str,
        tz: str = "America/New_York",
        echo: bool = True,
        roll: bool = True,
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        #: the file actually written: callers must read THIS, not their guess.
        self.path = next_journal_path(path) if roll else path
        self.tz = ZoneInfo(tz)
        self.echo = echo
        self._fh: TextIO | None = open(self.path, "a", encoding="utf-8")

    # -- writing ---------------------------------------------------------
    def event(
        self,
        kind: str,
        ts_et: datetime | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        if kind not in EVENT_KINDS:
            # A typo must not take the runner down with a position on (L-31).
            payload = dict(payload)
            payload["bad_kind"] = kind
            payload.setdefault("detail", "unknown journal kind")
            payload.setdefault("what", "journal_kind")
            payload["level"] = "alert"
            kind = "error"
        stamp = ts_et if ts_et is not None else datetime.now(self.tz)
        rec = {
            "ts_et": stamp.isoformat(),
            "kind": kind,
            "payload": _jsonable(payload),
        }
        line = json.dumps(rec, sort_keys=True)
        if self._fh is None:  # pragma: no cover - defensive
            raise RuntimeError("journal is closed")
        self._fh.write(line + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())  # survive a power loss, not just a crash
        except OSError:  # pragma: no cover - not every filesystem can
            pass
        if self.echo:
            print(
                "[" + stamp.strftime("%H:%M:%S") + "] " + kind + " " + line, flush=True
            )
        return rec

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_journal(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _f(value: Any, default: float = NAN) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


@dataclass
class DaySummary:
    """The day recomputed from the journal's fills."""

    date: str = ""
    book: str = ""
    mode: str = ""
    entry_mode: str = ""
    n_straddles: int = 0
    index_multiplier: float = NAN
    es_multiplier: float = NAN
    mes_multiplier: float = NAN

    entered: bool = False
    #: "short" (the book) or "long" (the month-end override), off the entry
    #: fill's action; "" until a body_entry fill is seen.
    side: str = ""
    entry_clock: str = ""
    entry_premium: float = NAN
    K_c: float = NAN
    K_p: float = NAN
    wings: tuple[float, float] | None = None

    exit_clock: str = ""
    exit_price: float = NAN
    exit_kind: str = ""  # "buyback" | "settlement" | ""
    provisional: bool = False

    hedge_pnl_dollars: float = 0.0
    hedge_pnl_points: float = NAN
    futures_open: dict[str, int] = field(default_factory=dict)
    futures_open_qty: int = 0
    futures_mark: float = NAN
    n_futures_fills: int = 0

    option_pnl_points: float = NAN
    pnl_dollars: float = NAN
    pnl_units: float = NAN

    straddles_open: int = 0
    stress_loss_per_contract: float = NAN
    stress_fraction_implied: float = NAN

    n_rebalances: int = 0
    residual_delta: dict[str, float] = field(default_factory=dict)
    killed: bool = False
    flat_reason: str = ""
    incomplete: str = ""
    errors: list[str] = field(default_factory=list)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_journal(cls, path: str) -> DaySummary:
        recs = read_journal(path)
        s = cls(date=os.path.splitext(os.path.basename(path))[0].split(".")[0])

        cash = 0.0  # futures cash flow in dollars (negative = paid)
        qty: dict[str, int] = {}
        entry_fill: float | None = None
        exit_fill: float | None = None
        entry_qty = 0
        exit_qty = 0
        seen_config = False
        seen_orders: set[str] = set()

        for rec in recs:
            kind = rec.get("kind")
            p = rec.get("payload") or {}
            if kind == "config":
                seen_config = True
                s.book = str(p.get("book", ""))
                s.mode = str(p.get("mode", ""))
                s.entry_mode = str(p.get("entry_mode", ""))
                s.index_multiplier = _f(p.get("index_multiplier"))
                s.es_multiplier = _f(p.get("es_multiplier"))
                s.mes_multiplier = _f(p.get("mes_multiplier"))
                if p.get("date"):
                    s.date = str(p["date"])
            elif kind == "selector":
                if p.get("clock"):
                    s.entry_clock = str(p["clock"])
            elif kind == "sizing":
                s.stress_loss_per_contract = _f(p.get("loss_total"))
                s.stress_fraction_implied = _f(p.get("fraction_implied"))
            elif kind == "entry":
                s.entry_clock = str(p.get("clock", s.entry_clock))
                if p.get("entered") is False:
                    s.flat_reason = str(p.get("reason") or "entry refused")
                else:
                    s.entered = True
                if p.get("K_c") is not None:
                    s.K_c = _f(p["K_c"])
                if p.get("K_p") is not None:
                    s.K_p = _f(p["K_p"])
                if p.get("Kc_wing") is not None and p.get("Kp_wing") is not None:
                    s.wings = (_f(p["Kc_wing"]), _f(p["Kp_wing"]))
            elif kind == "fill":
                oid = p.get("order_id")
                if oid is not None:
                    key = str(oid)
                    if key in seen_orders:
                        continue  # a restart re-journaling the same order
                    seen_orders.add(key)
                what = str(p.get("what", ""))
                price = p.get("price")
                filled = int(p.get("quantity") or 0)
                # A rejection trades nothing; a partial trades what it filled.
                if price is None or filled == 0:
                    continue
                if what == "body_entry":
                    entry_fill = float(price)
                    entry_qty += abs(filled)
                    action = str(p.get("action", "") or "").upper()
                    s.side = "long" if action == "BUY" else "short"
                elif what == "body_exit":
                    exit_fill = float(price)
                    exit_qty += abs(filled)
                    s.exit_kind = "buyback"
                    s.exit_clock = str(p.get("clock", s.exit_clock))
                elif what.startswith("futures"):
                    sym = str(p.get("symbol", "") or "").upper()
                    mult = _f(p.get("multiplier"), NAN)
                    if not math.isfinite(mult):
                        mult = {
                            "ES": s.es_multiplier,
                            "MES": s.mes_multiplier,
                        }.get(sym, NAN)
                    cash -= filled * float(price) * mult
                    qty[sym] = qty.get(sym, 0) + filled
                    s.n_futures_fills += 1
            elif kind == "rebalance":
                s.n_rebalances += 1
                if p.get("residual_delta") is not None:
                    s.residual_delta[str(p.get("clock", ""))] = _f(p["residual_delta"])
            elif kind == "delta":
                if p.get("residual_delta") is not None:
                    s.residual_delta.setdefault(
                        str(p.get("clock", "")), _f(p["residual_delta"])
                    )
            elif kind == "mark":
                s.futures_mark = _f(p.get("futures_ref"))
            elif kind == "settle":
                if p.get("payoff") is not None:
                    exit_fill = _f(p["payoff"])
                    exit_qty = entry_qty
                    s.exit_kind = "settlement"
                    s.exit_clock = "settle"
                    s.provisional = bool(p.get("provisional", False))
            elif kind == "reconcile":
                if p.get("payoff") is not None:
                    exit_fill = _f(p["payoff"])
                    exit_qty = entry_qty
                    s.exit_kind = "settlement"
                    s.exit_clock = "settle"
                    s.provisional = False
            elif kind == "kill":
                s.killed = True
                s.flat_reason = str(p.get("reason", "kill switch"))
            elif kind == "error":
                s.errors.append(
                    str(p.get("what", "")) + ": " + str(p.get("detail", ""))
                )
            elif kind == "preflight":
                if p.get("ok") is False:
                    s.flat_reason = str(p.get("reason", "preflight failed"))

        s.n_straddles = entry_qty
        s.straddles_open = entry_qty - exit_qty
        s.futures_open = {k: v for k, v in sorted(qty.items()) if v}
        s.futures_open_qty = sum(s.futures_open.values())
        s.hedge_pnl_dollars = cash

        if not seen_config:
            s.incomplete = "no config record: multipliers unknown, no P&L computed"
            return s

        # An open futures leg is marked at the futures reference, never at
        # zero, and never silently (L-11).
        if s.futures_open:
            if math.isfinite(s.futures_mark):
                for sym, q in s.futures_open.items():
                    mult = {"ES": s.es_multiplier, "MES": s.mes_multiplier}.get(
                        sym, NAN
                    )
                    cash += q * s.futures_mark * mult
                s.hedge_pnl_dollars = cash
                s.incomplete = (
                    "open futures "
                    + repr(s.futures_open)
                    + " marked at "
                    + f"{s.futures_mark:.2f}"
                )
            else:
                s.incomplete = (
                    "open futures " + repr(s.futures_open) + ": journal incomplete"
                )

        denom = s.n_straddles * s.index_multiplier
        if math.isfinite(denom) and denom > 0:
            s.hedge_pnl_points = cash / denom

        if entry_fill is not None:
            s.entry_premium = entry_fill
        if exit_fill is not None:
            s.exit_price = exit_fill
        if s.straddles_open:
            s.incomplete = (
                (s.incomplete + "; " if s.incomplete else "")
                + str(s.straddles_open)
                + " straddle(s) still open"
            )
        computable = (
            entry_fill is not None
            and exit_fill is not None
            and math.isfinite(denom)
            and denom > 0
            and not s.straddles_open
            and not (s.futures_open and not math.isfinite(s.futures_mark))
        )
        if computable:
            assert entry_fill is not None and exit_fill is not None
            # A short earns entry - exit; the month-end long earns exit - entry.
            sign = -1.0 if s.side == "long" else 1.0
            s.option_pnl_points = sign * (entry_fill - exit_fill)
            s.pnl_dollars = denom * s.option_pnl_points + cash
            s.pnl_units = (
                s.pnl_dollars / (denom * entry_fill) if entry_fill > 0 else NAN
            )
        elif not s.entered and not s.flat_reason:
            s.flat_reason = "no entry"
        return s

    # -- output ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))  # type: ignore[no-any-return]

    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        return path

    def write_beside(self, journal_path: str) -> str:
        base = os.path.splitext(journal_path)[0]
        return self.write(base + "_summary.json")

    @property
    def is_flat(self) -> bool:
        """Nothing left open: the morning reconciliation's precondition."""
        return not self.futures_open and not self.straddles_open

    def render(self) -> str:
        lines = [
            "=" * 72,
            "day summary  "
            + self.date
            + "   book="
            + self.book
            + "  mode="
            + self.mode,
            "=" * 72,
        ]
        fmt = "  {:<28s} {:>14s}"
        if not self.entered:
            lines.append("  FLAT: " + (self.flat_reason or "never entered"))
            lines.append("=" * 72)
            return "\n".join(lines)
        if self.side == "long":
            lines.append(fmt.format("side", "LONG (month-end)"))
        lines.append(fmt.format("entry clock", self.entry_clock or "-"))
        lines.append(fmt.format("straddles", str(self.n_straddles)))
        lines.append(fmt.format("strikes (Kc/Kp)", f"{self.K_c:.0f}/{self.K_p:.0f}"))
        if self.wings is not None:
            lines.append(
                fmt.format("wings (Kc/Kp)", f"{self.wings[0]:.0f}/{self.wings[1]:.0f}")
            )
        lines.append(fmt.format("entry premium (pts)", f"{self.entry_premium:.4f}"))
        lines.append(
            fmt.format(
                "exit ("
                + (self.exit_kind or "-")
                + (", provisional" if self.provisional else "")
                + ") pts",
                f"{self.exit_price:.4f}",
            )
        )
        lines.append(fmt.format("option P&L (pts)", f"{self.option_pnl_points:+.4f}"))
        lines.append(fmt.format("hedge P&L (pts)", f"{self.hedge_pnl_points:+.4f}"))
        lines.append(fmt.format("hedge P&L ($)", f"{self.hedge_pnl_dollars:+.2f}"))
        lines.append(fmt.format("futures fills", str(self.n_futures_fills)))
        lines.append(fmt.format("rebalances", str(self.n_rebalances)))
        if math.isfinite(self.stress_loss_per_contract):
            lines.append(
                fmt.format(
                    "stress loss / contract ($)",
                    f"{self.stress_loss_per_contract:,.0f}",
                )
            )
        lines.append(fmt.format("day P&L ($)", f"{self.pnl_dollars:+.2f}"))
        lines.append(fmt.format("day P&L (premium units)", f"{self.pnl_units:+.4f}"))
        # The morning ritual pivots on this line; it must always print (L-35).
        open_txt = (
            ", ".join(k + " " + f"{v:+d}" for k, v in self.futures_open.items())
            if self.futures_open
            else "0"
        )
        lines.append(fmt.format("OPEN futures", open_txt))
        if self.futures_open:
            lines.append(
                "  *** FUTURES STILL OPEN: "
                + open_txt
                + " -- flatten by hand before the next session ***"
            )
        if self.straddles_open:
            lines.append(
                "  *** "
                + str(self.straddles_open)
                + " STRADDLE(S) STILL OPEN -- reconcile by hand ***"
            )
        if self.residual_delta:
            worst = max(self.residual_delta.items(), key=lambda kv: abs(kv[1]))
            lines.append(
                fmt.format("worst residual delta", f"{worst[1]:+.3f} @ {worst[0]}")
            )
        if self.incomplete:
            lines.append("  incomplete: " + self.incomplete)
        if self.killed:
            lines.append("  KILLED: " + self.flat_reason)
        for e in self.errors:
            lines.append("  error: " + e)
        lines.append("=" * 72)
        return "\n".join(lines)
