"""Parity harness: the live engine replayed against the backtest's hedge tape.

The book of record is the EXIT variant: short one nearest-OTM 0DTE SPX
straddle at 11:00, delta-hedge every 30 minutes on the index, buy the
straddle back at 15:30 and flatten the futures at the same stamp.  The HOLD
variant -- same entry and hedge, no buy-back, cash settlement at the official
close -- is kept as the second reference because it is the tape the standalone
writeup gates on.

``replay_day`` walks one expiration day's cached stamps exactly as the live
``run_day`` will: pick the body from the stamp's quotes, take the total
volatility over the remaining session, take the package delta at every stamp,
accumulate ``delta * dS`` over the 30-minute steps, and mark the exit.  It is
pure and scalar; nothing here touches a broker.

``parity_report`` runs that replay over every scored day and compares it with
``writeup/make_dh_causal_standalone_tex.hold_mark_1100`` -- the research's own
11:00 book -- on three axes:

1. the vendor-implied path (the same formula on the same inputs: this must
   agree to floating point),
2. the mid-inverted path, where the total volatility is bisected out of the
   straddle's package midpoint at every stamp, which is what the live engine
   must do because the broker publishes no vendor implied volatility, and
3. the whole-contract futures hedge, which is what the live engine can
   actually hold.

It also prices the 15:30 buy-back three ways -- the Black-76 model mark, the
quoted midpoint and the quoted ask against an entry at the bid -- from the
option chain, because the model mark is not a price anyone can trade.

Run:  python -m live.ibkr.parity
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from math import isfinite, sqrt
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from .hedge import ES_MULTIPLIER, SPX_INDEX_MULTIPLIER, target_futures
from .premium_ledger import MIN_SESSIONS, PremiumLedger, records_from_grids
from .pricing import (
    MARKET_CLOSE_HOUR,
    corrected_total_vol,
    hourly_iv_from_total_vol,
    hours_to_close,
    invert_total_vol,
    package_delta,
    package_price,
    total_vol_from_hourly,
)
from .selector import pick_entry_clock
from .strikes import ATM_MIN_LIVE, Quote, pick_nearest_otm_reason

__all__ = [
    "DAILY_ERA_START",
    "ENTRY_CLOCK",
    "EXIT_CLOCK",
    "PARITY_TOL",
    "SESSION_CLOCKS",
    "ParityError",
    "month_end_override_parity",
    "parity_report",
    "replay_day",
    "require_inputs",
    "research_tape",
    "seed_premium_ledger",
    "sharpe_ann",
    "short_return",
    "validate_strike_selection",
]

ENTRY_CLOCK = "11:00"
EXIT_CLOCK = "15:30"
#: The trade cache's 30-minute session grid (the stamps the book can trade).
SESSION_CLOCKS: tuple[str, ...] = (
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
)
#: atm_straddle_lib.PERIODS_PER_YEAR
PERIODS_PER_YEAR = 252.0
#: The chain turns daily on 2022-05-16; the deck reports this era separately.
DAILY_ERA_START = "2022-05-16"
#: Straddle counts the whole-contract hedge is reported at.
ES_ROUNDING_LOTS = (1, 3, 5)
#: The vendor-implied replay is the same formula on the same inputs, and every
#: reproduction below is asserted against the research at this bar.  It is
#: REFERENCED, not decorative: ``_require_close`` raises on it, on the live
#: path of ``parity_report``, before any report is written.
PARITY_TOL = 1e-9
#: Published Sharpes are quoted to 7 decimals, so the bar on them is half a
#: unit in their last place -- tighter is not a claim the print can support.
PARITY_SHARPE_TOL = 5e-8
#: The research's own selector window (proposal 46, E2, rolling 252).
SELECTOR_WINDOW = 252
#: The entry clock proposal 43 found positive per contract with t > 2 unseen.
AFTERNOON_CLOCK = "13:30"
#: The 11:00 book's stamps: entry, eight rebalances, the 15:30 buy-back.
BOOK_CLOCKS: tuple[str, ...] = SESSION_CLOCKS[2:]
#: Proposal 32's hours-to-close at those stamps.
BOOK_H_REM: tuple[float, ...] = tuple(
    float(MARKET_CLOSE_HOUR) - (11.0 + 0.5 * k) for k in range(len(BOOK_CLOCKS))
)
#: Proposal 36's bootstrap: circular blocks of 21 sessions, 2000 draws, seed 0.
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
CI_PCT: tuple[float, float] = (2.5, 97.5)
#: Proposal 43's own GATE 0 constants for the 11:00 mid-inverted flatten book.
REF_1100_WHOLE_SHARPE = 2.2530773
REF_1100_ERA_SHARPE = 2.5133605

_NAN = float("nan")


class ParityError(RuntimeError):
    """A parity reproduction did not agree with the research it replays."""


def _require_close(
    what: str, got: float, want: float, tol: float = PARITY_TOL
) -> float:
    """Raise unless ``got`` reproduces ``want``.  The gate, not a print."""
    dev = abs(float(got) - float(want))
    if not (dev <= tol):
        raise ParityError(
            f"{what}: engine {got!r} vs research {want!r}, |difference| "
            f"{dev:.6e} > {tol:.0e}"
        )
    return dev


# ------------------------------------------------------------------ utils ---
def sharpe_ann(x: "pd.Series[float]") -> float:
    """Annualised Sharpe of a daily return series (one unit a day, ddof=1)."""
    v = pd.Series(x).dropna()
    if len(v) < 2:
        return _NAN
    sd = float(v.std(ddof=1))
    if not (isfinite(sd) and sd > 0.0):
        return _NAN
    return float(v.mean()) / sd * sqrt(PERIODS_PER_YEAR)


def _h_rem(day: Any, hhmm: str) -> float:
    """Hours from the ET clock ``hhmm`` on ``day`` to the 16:00 close."""
    d = pd.Timestamp(day)
    hh, mm = (int(part) for part in hhmm.split(":"))
    return hours_to_close(
        datetime(int(d.year), int(d.month), int(d.day), hh, mm),
        close_hour=MARKET_CLOSE_HOUR,
    )


def _f(row: "pd.Series[Any]", col: str) -> float:
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return _NAN


# ------------------------------------------------------------- one day ------
def replay_day(
    rows: pd.DataFrame,
    n_straddles: int = 1,
    *,
    entry_clock: str = ENTRY_CLOCK,
    exit_clock: str | None = EXIT_CLOCK,
    clocks: Sequence[str] = SESSION_CLOCKS,
    iv_mode: str = "vendor",
    iv_column: str = "iv_hourly_used",
    exit_price: float | None = None,
    futures_multiplier: float = ES_MULTIPLIER,
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
) -> dict[str, Any]:
    """Replay one expiration day of the short straddle, delta-hedged to the exit.

    ``rows`` are the cached stamps of ONE expiration date.  ``exit_clock`` is
    the buy-back stamp (``"15:30"`` for the book of record); pass ``None`` to
    hold through cash settlement at the official close.  ``iv_mode`` is
    ``"vendor"`` (the stamp's hourly implied volatility, column
    ``iv_column``) or ``"mid"`` (the total volatility bisected out of that
    stamp's own straddle midpoint -- what the live engine does).

    Returns a dict with the entry, the per-stamp hedge tape, the continuous
    and whole-contract hedge P&L in index points, and the short-side return
    in units of the entry premium.  A refused day carries ``ok=False`` and
    the refusal reason.
    """
    grid = list(clocks)
    tab = rows.drop_duplicates(subset=["hhmm"]).set_index("hhmm")
    day = pd.Timestamp(rows["date"].iloc[0])
    out: dict[str, Any] = {
        "date": day,
        "ok": False,
        "reason": "",
        "n_straddles": int(n_straddles),
        "entry_clock": entry_clock,
        "exit_clock": exit_clock,
        "iv_mode": iv_mode,
    }
    if entry_clock not in tab.index or entry_clock not in grid:
        out["reason"] = "no_entry_row"
        return out
    ent = tab.loc[entry_clock]

    # The trade cache keeps only the two legs the backtest picked, so the
    # picker is exercised on those two and the live-contract guard is read
    # off the cache's own n_live count for the stamp.
    quotes = [
        Quote(_f(ent, "K_c"), "C", _f(ent, "bid_c"), _f(ent, "ask_c")),
        Quote(_f(ent, "K_p"), "P", _f(ent, "bid_p"), _f(ent, "ask_p")),
    ]
    body, reason = pick_nearest_otm_reason(quotes, _f(ent, "S"), min_live=0)
    if body is None:
        out["reason"] = reason
        return out
    n_live = int(ent["n_live"]) if "n_live" in tab.columns else ATM_MIN_LIVE
    if n_live < ATM_MIN_LIVE:
        out["reason"] = "few_live"
        return out

    entry_mid = body.entry_mid
    if not (isfinite(entry_mid) and entry_mid > 0.0):
        out["reason"] = "no_entry_premium"
        return out
    s_close = _f(ent, "S_close")
    kc, kp = body.Kc, body.Kp

    j0 = grid.index(entry_clock)
    j_last = len(grid) - 1
    jx = grid.index(exit_clock) if exit_clock is not None else j_last

    hedge_pts = 0.0
    hedge_pts_rounded = 0.0
    stamps: list[dict[str, Any]] = []
    tot_exit = _NAN
    spot_exit = _NAN
    futures_pos = 0
    gap_steps = 0
    no_vol_stamps = 0

    for j in range(j0, jx + 1):
        clock = grid[j]
        has = clock in tab.index
        row = tab.loc[clock] if has else None
        spot = _f(row, "S") if row is not None else _NAN
        h_rem = _h_rem(day, clock)

        if row is None:
            total_vol = _NAN
        elif iv_mode == "mid":
            total_vol = invert_total_vol(
                _f(row, "S"), _f(row, "K_c"), _f(row, "K_p"), _f(row, "entry")
            )
        else:
            iv = (
                _f(row, iv_column) if iv_column in tab.columns else _f(row, "iv_hourly")
            )
            total_vol = total_vol_from_hourly(iv, h_rem) if iv > 0.0 else _NAN

        delta = package_delta(total_vol, spot, kc, kp)
        if not isfinite(total_vol):
            no_vol_stamps += 1

        if j < j_last:
            nxt = grid[j + 1]
            spot_next = float(tab.loc[nxt, "S"]) if nxt in tab.index else _NAN
        else:
            spot_next = s_close
        # Is the delta carried over the step that starts here?  Every step from
        # the entry on, except the exit stamp itself, which the book flattens.
        held = exit_clock is None or j < jx
        if not held:
            d_spot = 0.0
        elif isfinite(spot) and isfinite(spot_next):
            d_spot = spot_next - spot
        else:
            # A missing stamp is a hole in the tape, not a flat market.  Booking
            # it as zero hedge P&L reports a day the engine did not replay; count
            # it and refuse the day below.
            gap_steps += 1
            d_spot = 0.0

        target = target_futures(
            delta, n_straddles, futures_multiplier, index_multiplier
        )
        delta_rounded = (
            target * futures_multiplier / (int(n_straddles) * index_multiplier)
        )
        hedge_pts += delta * d_spot
        hedge_pts_rounded += delta_rounded * d_spot
        stamps.append(
            {
                "hhmm": clock,
                "S": spot,
                "h_rem": h_rem,
                "total_vol": total_vol,
                "delta_pkg": delta,
                "dS": d_spot,
                "target_futures": target,
                "trade_futures": target - futures_pos,
            }
        )
        futures_pos = target
        if j == jx:
            tot_exit = total_vol
            spot_exit = spot

    if gap_steps:
        out["reason"] = "stamp_gap"
        out["n_gap_steps"] = gap_steps
        return out

    if exit_price is not None:
        exit_px = float(exit_price)
        exit_source = "given"
    elif exit_clock is None:
        exit_px = max(s_close - kc, 0.0) + max(kp - s_close, 0.0)
        exit_source = "settlement"
    else:
        exit_px = package_price(tot_exit, spot_exit, kc, kp)
        exit_source = "model_mark"

    opt_short = -(exit_px - entry_mid)
    out.update(
        {
            "ok": True,
            "Kc": kc,
            "Kp": kp,
            "S_entry": body.S,
            "S_close": s_close,
            "same_strike": body.same_strike,
            "n_live": n_live,
            "entry_mid": entry_mid,
            "entry_bid": body.entry_bid,
            "entry_ask": body.entry_ask,
            "exit_price": exit_px,
            "exit_source": exit_source,
            "hedge_pts": hedge_pts,
            "hedge_pts_rounded": hedge_pts_rounded,
            "n_gap_steps": gap_steps,
            "n_no_vol_stamps": no_vol_stamps,
            "one_sided": body.one_sided,
            "delta_entry": stamps[0]["delta_pkg"],
            "r_short": (opt_short + hedge_pts) / entry_mid,
            "r_short_rounded": (opt_short + hedge_pts_rounded) / entry_mid,
            "r_short_crossed": (-(exit_px - body.entry_bid) + hedge_pts) / entry_mid,
            "stamps": stamps,
        }
    )
    return out


def short_return(
    hedge_pts: float, entry_ref: float, exit_px: float, entry_mid: float
) -> float:
    """Short-side return in units of the entry premium.

    ``entry_ref`` is the premium actually received (the midpoint, or the bid
    for a crossed fill); ``entry_mid`` is always the midpoint denominator the
    research reports in.
    """
    return (-(float(exit_px) - float(entry_ref)) + float(hedge_pts)) / float(entry_mid)


# ------------------------------------------------------------- research -----
def repo_root(start: Path | str | None = None) -> Path:
    """The repository root, found from this file (or ``start``) upwards."""
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "notebooks" / "atm_straddle_lib.py").exists():
            return cand
    raise RuntimeError(f"no repository root above {here}")


def trade_cache_candidates(root: Path) -> list[Path]:
    """Every hold-to-close trade cache present, newest first."""
    cache = root / "results" / "atm_straddle_intraday_holdclose" / "cache"
    return sorted(
        cache.glob("trade_*.parquet"), key=lambda q: q.stat().st_mtime, reverse=True
    )


def latest_trade_cache(root: Path) -> Path:
    """The NEWEST hold-to-close trade cache, by modification time.

    The names are content hashes, so ``sorted(...)[-1]`` picks the largest hex
    string, not the newest file: a re-minted cache that happens to sort lower
    would have been ignored in silence.  Modification time is the only ordering
    the file system actually carries here.
    """
    paths = trade_cache_candidates(root)
    if not paths:
        raise FileNotFoundError(
            "no trade cache under "
            f"{root / 'results' / 'atm_straddle_intraday_holdclose' / 'cache'}: "
            "mint it by running the intraday hold-to-close notebook"
        )
    return paths[0]


@contextmanager
def _argv_flag(flag: str) -> Iterator[None]:
    """Append ``flag`` to ``sys.argv`` for the block, then restore argv exactly.

    The writeup modules read their deck flags off ``sys.argv`` at import time.
    Mutating argv permanently leaks the flag into everything that runs later in
    the same process -- pytest, a subsequent ``run_day.main()`` -- so it is
    scoped here.
    """
    saved = list(sys.argv)
    if flag not in sys.argv:
        sys.argv.append(flag)
    try:
        yield
    finally:
        sys.argv[:] = saved


def _import_by_path(path: Path, name: str) -> ModuleType:
    """Import a research script by path, loudly when it is not there."""
    if not path.exists():
        raise FileNotFoundError(f"{name}: no such research module at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}: {path} is not importable")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def research_modules(root: Path) -> tuple[ModuleType, ModuleType]:
    """``(make_dh_causal_standalone_tex, make_rule_by_strategy_intraday_tex)``.

    The deck flag is scoped to this call and the imported rule module is
    CHECKED for it: an earlier import of the same module without the flag
    leaves the non-holdclose constants in ``sys.modules``, and this used to
    pass silently and produce a confidently-numbered wrong report.
    """
    for name in ("notebooks", "writeup"):
        path = str(root / name)
        if path not in sys.path:
            sys.path.insert(0, path)
    with _argv_flag("--dh-holdclose"):
        import make_rule_by_strategy_intraday_tex as rule_mod  # noqa: PLC0415

        if getattr(rule_mod, "DH_HOLDCLOSE", None) is not True:
            raise ParityError(
                "make_rule_by_strategy_intraday_tex was already imported without "
                "--dh-holdclose, so it carries the non-holdclose constants; the "
                "parity harness cannot use it.  Import it after this module, or "
                "run the harness in its own process."
            )
        standalone = _import_by_path(
            root / "writeup" / "make_dh_causal_standalone_tex.py",
            "dh_causal_standalone",
        )
    return standalone, rule_mod


#: Every input the parity report reads, by the name the report prints.
PARITY_INPUTS: tuple[tuple[str, str], ...] = (
    ("repo sentinel", "notebooks/atm_straddle_lib.py"),
    ("rule module", "writeup/make_rule_by_strategy_intraday_tex.py"),
    ("standalone", "writeup/make_dh_causal_standalone_tex.py"),
    ("entry-clock study", "writeup/intraday_proposals/43_causal_entry_over_time.py"),
    ("deck day set", "results/atm_straddle_0dte_1530/daily_blk2.parquet"),
    ("option chain", "data/spxw_chain.parquet"),
    ("spot tape", "data/spxw_spot.parquet"),
    (
        "settlement tape",
        "results/atm_straddle_intraday_holdclose/cache/gspc_ohlc.parquet",
    ),
    (
        "43 clock x year",
        "results/atm_straddle_intraday_holdclose/proposals/43/a_clock_year.csv",
    ),
    ("46 rules", "results/atm_straddle_intraday_holdclose/proposals/46/rules.csv"),
    (
        "46 selector timeline",
        "results/atm_straddle_intraday_holdclose/proposals/46/selector_timeline.csv",
    ),
    (
        "36 bootstrap",
        "results/atm_straddle_intraday_holdclose/proposals/36/bootstrap.csv",
    ),
)


def require_inputs(root: Path) -> dict[str, Path]:
    """Every parity input, or a FileNotFoundError naming all the missing ones.

    Four of these are untracked, so on a clean clone the harness used to SKIP
    and the gate vanished rather than failed.  It now refuses, by name, and the
    caller has to go and fetch them.
    """
    found: dict[str, Path] = {}
    missing: list[str] = []
    for label, rel in PARITY_INPUTS:
        path = root / rel
        if path.exists():
            found[label] = path
        else:
            missing.append(f"{label}: {rel}")
    try:
        found["trade cache"] = latest_trade_cache(root)
    except FileNotFoundError as exc:
        missing.append(f"trade cache: {exc}")
    if missing:
        raise FileNotFoundError(
            "the parity harness cannot run: "
            + str(len(missing))
            + " input(s) missing under "
            + str(root)
            + "\n  - "
            + "\n  - ".join(missing)
        )
    return found


def attach_engine_iv(
    work: pd.DataFrame, rule_mod: ModuleType
) -> tuple[pd.DataFrame, int]:
    """Vendor hourly implied volatility with the censored bars re-inverted here.

    A bar whose call or put leg sits on the vendor's solver bracket node
    carries no volatility; the research recovers the one that reproduces the
    straddle midpoint.  This does the same recovery with the live engine's own
    bisection.  Returns the frame with ``iv_hourly_used`` and the number of
    bars touched.
    """
    out = work.copy()
    capped = rule_mod.on_vendor_node(
        out["impl_volatility_c"]
    ) | rule_mod.on_vendor_node(out["impl_volatility_p"])
    entry = out["entry"].to_numpy(float)
    cap = capped & (entry > 0)
    iv = out["iv_hourly"].astype(float).to_numpy().copy()
    h_rem = out["h_rem"].to_numpy(float)
    spot = out["S"].to_numpy(float)
    kc = out["K_c"].to_numpy(float)
    kp = out["K_p"].to_numpy(float)
    idx = np.flatnonzero(cap)
    for i in idx:
        s_tot = invert_total_vol(spot[i], kc[i], kp[i], entry[i])
        iv[i] = hourly_iv_from_total_vol(s_tot, h_rem[i])
    out["iv_hourly_used"] = iv
    return out, int(idx.size)


def scored_days(pkg: pd.DataFrame, root: Path) -> pd.DataFrame:
    """The deck's days that carry a tradeable 11:00 straddle, as hold_mark_1100 does."""
    deck = pd.read_parquet(
        root / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
    )
    days = pd.to_datetime(deck.index)
    work = pkg.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work[work["date"].isin(days)].copy()
    work["h_rem"] = [
        _h_rem(d, c)
        for d, c in zip(work["date"].to_numpy(), work["hhmm"].to_numpy(), strict=True)
    ]
    entry_rows = (
        work.loc[work["hhmm"] == ENTRY_CLOCK]
        .dropna(subset=["entry", "exit", "K_c", "K_p", "S_close"])
        .drop_duplicates("date")
    )
    entry_rows = entry_rows.loc[entry_rows["entry"].astype(float) > 0]
    keep = pd.DatetimeIndex(sorted(entry_rows["date"].unique()))
    return work[work["date"].isin(keep)].copy()


def load_exit_quotes(
    root: Path, panel: pd.DataFrame, bodies: pd.DataFrame
) -> pd.DataFrame:
    """Quoted 15:30 bid/ask of the 11:00 strikes, straight from the option chain.

    ``bodies`` is indexed by expiration date with columns ``Kc``/``Kp``.  The
    chain is read with a pushdown filter on the 15:30 stamps only.
    """
    close_rows = (
        panel.loc[panel["hhmm"] == EXIT_CLOCK, ["date", "timestamp"]]
        .drop_duplicates("date")
        .set_index("date")["timestamp"]
    )
    stamps = sorted(pd.DatetimeIndex(close_rows.to_numpy()).unique())
    chain = pd.read_parquet(
        root / "data" / "spxw_chain.parquet",
        columns=["expiration", "timestamp", "strike", "cp", "bid", "ask"],
        filters=[("timestamp", "in", stamps)],
    )
    chain["strike"] = chain["strike"].astype(float)
    chain["cp"] = chain["cp"].astype(str)
    chain["expiration"] = chain["expiration"].astype("datetime64[ns]")
    chain["timestamp"] = chain["timestamp"].astype("datetime64[ns, UTC]")

    want = []
    for date, row in bodies.iterrows():
        ts = close_rows.get(date)
        if ts is None:
            continue
        want.append((date, ts, float(row["Kc"]), "C", "c"))
        want.append((date, ts, float(row["Kp"]), "P", "p"))
    key = pd.DataFrame(want, columns=["date", "timestamp", "strike", "cp", "leg"])
    key["expiration"] = key["date"].astype("datetime64[ns]")
    key["timestamp"] = key["timestamp"].astype("datetime64[ns, UTC]")
    got = key.merge(chain, on=["expiration", "timestamp", "strike", "cp"], how="left")
    bid = got["bid"].astype(float)
    ask = got["ask"].astype(float)
    sentinel = (bid == 0.0) & (ask == 0.0)
    got["leg_mid"] = (0.5 * (bid + ask)).mask(sentinel)
    got["leg_ask"] = ask.mask(sentinel)
    legs = {}
    for leg in ("c", "p"):
        one = got.loc[got["leg"] == leg].set_index("date")[["leg_mid", "leg_ask"]]
        legs[leg] = one.loc[~one.index.duplicated()].reindex(bodies.index)
    quoted = pd.DataFrame(index=bodies.index)
    quoted["quoted_mid"] = legs["c"]["leg_mid"] + legs["p"]["leg_mid"]
    quoted["quoted_ask"] = legs["c"]["leg_ask"] + legs["p"]["leg_ask"]
    return quoted


# ------------------------------------------------- the research tape (43) ---
#: The forecast-free entry-clock study whose tape every extended book replays.
RESEARCH_43 = "writeup/intraday_proposals/43_causal_entry_over_time.py"
_P43_NAME = "harxhar_parity_p43"
_P43: ModuleType | None = None


def research_43(root: Path | None = None) -> ModuleType:
    """Proposal 43, imported by path and cached for this process.

    43 owns the objects the extended books need and the engine must not
    re-derive: the one filtered chain read, the guarded nearest-OTM pick at
    every stamp, the mid-inverted package volatility (which it gates against
    ``pricing.invert_total_vol`` cell by cell) and the per-clock hedge tape.
    Re-implementing any of them here would make the parity report a comparison
    of two engines rather than a replay of the research.
    """
    global _P43
    if _P43 is None:
        base = repo_root() if root is None else Path(root)
        _P43 = _import_by_path(base / RESEARCH_43, _P43_NAME)
    return _P43


def _clock_book_job(args: tuple[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """One entry clock's tape, in a worker process.

    Importable by name from a spawned child (which is why it lives here rather
    than being a lambda over ``p43._clock_book``), and it loads 43 itself.
    """
    return research_43()._clock_book(args)


def research_tape(
    root: Path | None = None, *, workers: int | None = None
) -> dict[str, Any]:
    """43's chain tape and the eleven entry clocks' books, built once.

    One pass over the chain, then one worker process per entry clock.  Nothing
    here loops over sessions in Python: the books are numpy over
    ``(sessions x clocks)`` arrays and the clocks run in parallel.
    """
    base = repo_root() if root is None else Path(root)
    p43 = research_43(base)
    stamp, _px = p43.session_stamps()
    half = p43.half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    chain = p43.build_chain(stamp, sessions)
    arrays = {
        k: chain[k]
        for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = chain["S_close"]
    jobs = [(j, arrays) for j in range(len(p43.ENTRIES))]
    n_workers = min(len(jobs), workers or (os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        res = list(pool.map(_clock_book_job, jobs))
    idx = pd.DatetimeIndex(chain["dates"])
    books = {
        clock: {k: pd.Series(v, index=idx) for k, v in res[j].items()}
        for j, clock in enumerate(p43.ENTRIES)
    }
    return {
        "chain": chain,
        "books": books,
        "dates": idx,
        "entries": tuple(p43.ENTRIES),
        "half_sessions": half,
        "n_workers": n_workers,
    }


def clock_series(
    tape: dict[str, Any], clock: str, treat: str, unit: str
) -> "pd.Series[float]":
    """One entry clock's daily P&L: premium units, or index points per contract."""
    book = tape["books"][clock]
    series = book[treat]
    return series * book["entry"] if unit == "pts" else series


def validate_strike_selection(
    tape: dict[str, Any], root: Path | None = None
) -> dict[str, Any]:
    """Re-pick the straddle from the FULL chain and check it against the cache.

    The per-day replay above feeds the picker the two legs the backtest already
    chose, with the live-contract guard switched off, so it can only ever
    agree: that is a check of the picking ARITHMETIC, not of the selection.
    This is the selection check -- 43's tape re-picks the nearest-OTM straddle
    at every stamp of every session from the whole chain, with the research's
    own guards, and its 11:00 strikes must be the trade cache's.
    """
    base = repo_root() if root is None else Path(root)
    cache = pd.read_parquet(latest_trade_cache(base))
    cache["date"] = pd.to_datetime(cache["date"])
    ent = (
        cache.loc[cache["hhmm"] == ENTRY_CLOCK]
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    idx = tape["dates"]
    j = tape["entries"].index(ENTRY_CLOCK)
    mine = pd.DataFrame(
        {"K_c": tape["chain"]["K_c"][:, j], "K_p": tape["chain"]["K_p"][:, j]},
        index=idx,
    )
    common = idx.intersection(ent.index)
    got = mine.reindex(common)
    want = ent.reindex(common)[["K_c", "K_p"]].astype(float)
    same_c = (got["K_c"].to_numpy() == want["K_c"].to_numpy()) | (
        got["K_c"].isna().to_numpy() & want["K_c"].isna().to_numpy()
    )
    same_p = (got["K_p"].to_numpy() == want["K_p"].to_numpy()) | (
        got["K_p"].isna().to_numpy() & want["K_p"].isna().to_numpy()
    )
    n_bad = int((~(same_c & same_p)).sum())
    if n_bad:
        bad = common[~(same_c & same_p)]
        raise ParityError(
            f"chain-level strike selection disagrees with the trade cache on "
            f"{n_bad} of {len(common)} sessions, first {bad[0].date()}"
        )
    return {
        "n_sessions": int(len(common)),
        "n_disagreements": n_bad,
        "refused_stamps": int(len(tape["chain"]["refused"])),
    }


# ------------------------------------------------------ the extended books ---
def fixed_clock_book(
    tape: dict[str, Any], clock: str, treat: str, root: Path | None = None
) -> pd.DataFrame:
    """One fixed entry clock's statistics by period and unit, gated on 43."""
    base = repo_root() if root is None else Path(root)
    p43 = research_43(base)
    masks = p43.period_masks(tape["dates"])
    ref = pd.read_csv(
        base / "results/atm_straddle_intraday_holdclose/proposals/43/a_clock_year.csv"
    )
    rows = []
    for unit, uname in (("units", "premium"), ("pts", "index points per contract")):
        series = clock_series(tape, clock, treat, unit)
        for period, mask in masks.items():
            stats = p43.stats_row(series[mask])
            want = ref.loc[
                (ref["clock"] == clock)
                & (ref["treatment"] == treat)
                & (ref["period"] == period)
                & (ref["unit"] == uname)
            ]
            if len(want) != 1:
                raise ParityError(
                    f"43's a_clock_year.csv has {len(want)} rows for "
                    f"{clock}/{treat}/{period}/{uname}"
                )
            row = want.iloc[0]
            devs = {
                field: _require_close(
                    f"43 {clock} {treat} {period} {uname} {field}",
                    stats[field],
                    float(row[field]),
                )
                for field in ("mean", "sd", "Sharpe_ann", "t_hac", "MaxDD", "worst")
            }
            if int(stats["n"]) != int(row["n"]):
                raise ParityError(
                    f"43 {clock} {treat} {period} {uname}: n {stats['n']} vs "
                    f"{int(row['n'])}"
                )
            rows.append(
                {
                    "clock": clock,
                    "treatment": treat,
                    "period": period,
                    "unit": uname,
                    **stats,
                    "max_abs_diff_vs_43": max(devs.values()),
                }
            )
    return pd.DataFrame(rows)


def selector_picks(
    ledger: PremiumLedger,
    sessions: pd.DatetimeIndex,
    window: int = SELECTOR_WINDOW,
    min_sessions: int = MIN_SESSIONS,
) -> "pd.Series[Any]":
    """The entry clock E2 picks for each session; warm-up sessions are flat."""
    picks = [
        pick_entry_clock(ledger, d.date(), window=window, min_sessions=min_sessions)[0]
        for d in sessions
    ]
    return pd.Series(
        [pd.NA if q is None else q for q in picks], index=sessions, dtype=object
    )


def selector_book(
    tape: dict[str, Any],
    ledger: PremiumLedger,
    treat: str = "hold",
    window: int = SELECTOR_WINDOW,
    root: Path | None = None,
) -> dict[str, Any]:
    """The selector book: enter at ``pick_entry_clock`` each session, then hold.

    Gated twice against proposal 46's E2 rolling-252 rule: the pick sequence
    day for day, and every statistic 46 persists for that rule's daily series
    -- 46 writes the rule's statistics, not the series itself, so its complete
    statistic set (n, mean, sd, Sharpe, HAC t and its lag, MaxDD, worst day and
    its date, in BOTH units) is what pins the series here.
    """
    base = repo_root() if root is None else Path(root)
    p43 = research_43(base)
    idx = tape["dates"]
    picks = selector_picks(ledger, idx, window=window)

    timeline = pd.read_csv(
        base
        / "results/atm_straddle_intraday_holdclose/proposals/46/selector_timeline.csv",
        index_col=0,
        parse_dates=True,
    )
    column = f"pick_E2 relative ratio of sums | rolling {window}"
    if column not in timeline.columns:
        raise ParityError(f"46's selector timeline has no column {column!r}")
    want_pick = timeline[column].reindex(idx)
    got = picks.where(picks.notna(), None).to_numpy()
    ref = want_pick.where(want_pick.notna(), None).to_numpy()
    n_bad = int(sum(1 for a, b in zip(got, ref, strict=True) if a != b))
    if n_bad:
        raise ParityError(
            f"the engine's E2 rolling-{window} picks disagree with proposal 46 "
            f"on {n_bad} of {len(idx)} sessions"
        )

    rules = pd.read_csv(
        base / "results/atm_straddle_intraday_holdclose/proposals/46/rules.csv"
    )
    label = f"E2 relative ratio of sums | rolling {window} | {treat}"
    masks = p43.period_masks(idx)
    fields = (
        ("mean", "mean_{u}"),
        ("sd", "sd_{u}"),
        ("Sharpe_ann", "Sharpe_ann_{u}"),
        ("t_hac", "t_hac_{u}"),
        ("MaxDD", "MaxDD_{u}"),
        ("worst", "worst_{u}"),
    )
    rows = []
    series_by_unit: dict[str, pd.Series] = {}
    for unit, tag in (("units", "units"), ("pts", "pts")):
        series = p43.pick_series(tape["books"], picks, treat, unit, "pnl")
        series_by_unit[unit] = series
        for period in ("deck in-sample", "unseen", "all"):
            stats = p43.stats_row(series[masks[period]])
            want = rules.loc[(rules["book"] == label) & (rules["sample"] == period)]
            if len(want) != 1:
                raise ParityError(
                    f"46's rules.csv has {len(want)} rows for {label} / {period}"
                )
            row = want.iloc[0]
            devs = {
                field: _require_close(
                    f"46 {label} {period} {unit} {field}",
                    stats[field],
                    float(row[key.format(u=tag)]),
                )
                for field, key in fields
            }
            if int(stats["n"]) != int(row["n_sessions"]):
                raise ParityError(
                    f"46 {label} {period}: n {stats['n']} vs {int(row['n_sessions'])}"
                )
            if int(stats["t_lag"]) != int(row[f"t_lag_{tag}"]):
                raise ParityError(
                    f"46 {label} {period} {unit}: HAC lag {stats['t_lag']} vs "
                    f"{int(row[f't_lag_{tag}'])}"
                )
            if str(stats["worst_date"]) != str(row[f"worst_date_{tag}"]):
                raise ParityError(
                    f"46 {label} {period} {unit}: worst day {stats['worst_date']} vs "
                    f"{row[f'worst_date_{tag}']}"
                )
            rows.append(
                {
                    "book": label,
                    "period": period,
                    "unit": unit,
                    **stats,
                    "max_abs_diff_vs_46": max(devs.values()),
                }
            )
    counts = picks.dropna().value_counts().sort_index()
    return {
        "picks": picks,
        "stats": pd.DataFrame(rows),
        "pick_counts": counts,
        "n_flat": int(picks.isna().sum()),
        "series": series_by_unit,
    }


# ------------------------------------ the corrected delta (proposal 36's V9) --
def v9_tape(root: Path | None = None) -> dict[str, Any]:
    """The 11:00 book's implied and realized remaining-window variance grids.

    Proposal 36's own construction: the implied slice is the re-inverted vendor
    hourly volatility squared times the hours remaining, and the realized
    remaining window is the reverse cumulative sum of the forecast panel's own
    bar realized variances.  The V9 factor is then the ledger's
    ``realized_over_implied_mean`` on THAT pair -- the same estimator the live
    engine runs on its own tape, fed the research's numbers so the reproduction
    is a reproduction and not a coincidence.
    """
    base = repo_root() if root is None else Path(root)
    standalone, rule_mod = research_modules(base)
    del standalone
    import atm_straddle_lib as asl  # noqa: PLC0415

    cache = pd.read_parquet(latest_trade_cache(base))
    panel = scored_days(cache, base)
    panel, n_reinverted = attach_engine_iv(panel, rule_mod)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    cols = list(BOOK_CLOCKS)

    def grid(column: str) -> np.ndarray:
        return (
            panel.pivot_table(
                index="date", columns="hhmm", values=column, aggfunc="first"
            )
            .reindex(index=dates, columns=cols)
            .to_numpy(float)
        )

    spot = grid("S")
    iv = grid("iv_hourly_used")
    var0 = (iv * np.sqrt(np.asarray(BOOK_H_REM)[None, :])) ** 2
    ent = (
        panel.loc[panel["hhmm"] == ENTRY_CLOCK]
        .drop_duplicates("date")
        .set_index("date")
        .reindex(dates)
    )
    kc = ent["K_c"].to_numpy(float)
    kp = ent["K_p"].to_numpy(float)
    s_close = ent["S_close"].to_numpy(float)
    entry_mid = ent["entry"].to_numpy(float)
    entry_bid = ent["bid_entry"].to_numpy(float)

    # the forecast panel's realized bar variances, reverse-cumulated
    forecast = asl.load_yhat_panel(asl.yhat_paths(base)["blk2"]).set_index("t")
    frame = forecast.reset_index()
    frame = frame[frame["in_fit"].to_numpy(dtype=bool)].copy()
    bar = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(
        "America/New_York"
    ) - pd.Timedelta(minutes=30)
    frame["pdate"] = bar.dt.normalize().dt.tz_localize(None)
    frame["phhmm"] = bar.dt.strftime("%H:%M")
    prof = (
        frame.pivot_table(
            index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
        )
        .sort_index()
        .reindex(columns=list(SESSION_CLOCKS))
    )
    rv = prof.to_numpy(float)
    remaining = pd.DataFrame(
        rv[:, ::-1].cumsum(axis=1)[:, ::-1], index=prof.index, columns=prof.columns
    )
    realized = remaining.reindex(index=dates, columns=cols).to_numpy(float)
    if not np.isfinite(realized).all():
        raise ParityError("a book stamp has no realized remaining window")

    capped = rule_mod.on_vendor_node(
        panel["impl_volatility_c"]
    ) | rule_mod.on_vendor_node(panel["impl_volatility_p"])
    censored = (
        panel.assign(_cen=np.asarray(capped, dtype=float))
        .pivot_table(index="date", columns="hhmm", values="_cen", aggfunc="max")
        .reindex(index=dates, columns=cols)
        .fillna(1.0)
        .to_numpy(float)
        > 0
    )
    quoted = load_exit_quotes(
        base, panel, pd.DataFrame({"Kc": kc, "Kp": kp}, index=dates)
    )
    return {
        "dates": dates,
        "clocks": cols,
        "spot": spot,
        "var0": var0,
        "realized": realized,
        "censored": censored,
        "Kc": kc,
        "Kp": kp,
        "S_close": s_close,
        "entry_mid": entry_mid,
        "entry_bid": entry_bid,
        "exit_ask": quoted["quoted_ask"].to_numpy(float),
        "n_reinverted": n_reinverted,
    }


def _delta_grid(total_vol: np.ndarray, tape: dict[str, Any]) -> np.ndarray:
    """``package_delta`` at every stamp, the live engine's own scalar formula."""
    spot = tape["spot"]
    kc = tape["Kc"]
    kp = tape["Kp"]
    n, m = total_vol.shape
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = package_delta(total_vol[i, j], spot[i, j], kc[i], kp[i])
    return out


def _crossed_exit_return(delta: np.ndarray, tape: dict[str, Any]) -> np.ndarray:
    """The crossed-quoted 15:30 exit book of the 11:00 straddle, per session."""
    spot = tape["spot"]
    hedge = (delta[:, :-1] * (spot[:, 1:] - spot[:, :-1])).sum(axis=1)
    return (-(tape["exit_ask"] - tape["entry_bid"]) + hedge) / tape["entry_mid"]


def _sharpe_diff_ci(x: np.ndarray, y: np.ndarray, root: Path) -> dict[str, float]:
    """Proposal 36's circular block bootstrap of Sharpe(x) - Sharpe(y)."""
    import atm_straddle_lib as asl  # noqa: PLC0415

    ok = np.isfinite(x) & np.isfinite(y)
    a, b = x[ok], y[ok]
    ann = sqrt(PERIODS_PER_YEAR)
    hat = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * ann
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng(BOOT_SEED), int(a.size), BOOT_BLOCK, BOOT_B
    )
    aa, bb = a[idx], b[idx]
    draws = (
        aa.mean(axis=1) / aa.std(axis=1, ddof=1)
        - bb.mean(axis=1) / bb.std(axis=1, ddof=1)
    ) * ann
    lo, hi = np.percentile(draws, list(CI_PCT))
    return {
        "n": float(a.size),
        "dSharpe": float(hat),
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "pct_draws_positive": float(100.0 * (draws > 0).mean()),
    }


def corrected_delta_book(
    tape: dict[str, Any], root: Path | None = None
) -> dict[str, Any]:
    """The 11:00 book hedged with the V9-corrected implied volatility.

    The correction is ``pricing.corrected_total_vol`` on the ledger's
    ``realized_over_implied_mean``; the fallback is the research's -- a stamp
    whose factor is missing, non-positive, or whose vendor implied sat on the
    solver's bracket node hedges with the uncorrected implied, so both books
    trade the same days.  Gated on proposal 36's published era interval for
    V9 against V0 on the crossed-quoted exit book.
    """
    base = repo_root() if root is None else Path(root)
    dates = tape["dates"]
    clocks = tape["clocks"]
    ledger = PremiumLedger()
    ledger.append_session(
        records_from_grids(
            dates,
            clocks,
            tape["var0"],
            tape["realized"],
            tape["spot"],
            np.repeat(tape["entry_mid"][:, None], len(clocks), axis=1),
            np.repeat(tape["Kc"][:, None], len(clocks), axis=1),
            np.repeat(tape["Kp"][:, None], len(clocks), axis=1),
        )
    )
    factor = np.column_stack(
        [ledger.realized_over_implied_series(c, None).to_numpy(float) for c in clocks]
    )
    tv0 = np.sqrt(tape["var0"])
    tv9 = np.empty_like(tv0)
    n, m = tv0.shape
    for i in range(n):
        for j in range(m):
            tv9[i, j] = (
                tv0[i, j]
                if tape["censored"][i, j]
                else corrected_total_vol(tv0[i, j], factor[i, j])
            )
    r0 = _crossed_exit_return(_delta_grid(tv0, tape), tape)
    r9 = _crossed_exit_return(_delta_grid(tv9, tape), tape)
    era = np.asarray(dates >= pd.Timestamp(DAILY_ERA_START))

    boot = pd.read_csv(
        base / "results/atm_straddle_intraday_holdclose/proposals/36/bootstrap.csv"
    )
    rows = []
    for sample, mask in (("whole", np.ones(len(dates), bool)), ("daily_era", era)):
        got = _sharpe_diff_ci(r9[mask], r0[mask], base)
        want = boot.loc[
            (boot["variant"] == "V9_implied_bc")
            & (boot["book"] == "exit_crossed")
            & (boot["sample"] == sample)
        ]
        if len(want) != 1:
            raise ParityError(
                f"36's bootstrap.csv has {len(want)} V9 rows for {sample}"
            )
        ref = want.iloc[0]
        devs = {
            field: _require_close(
                f"36 V9 exit_crossed {sample} {field}", got[field], float(ref[field])
            )
            for field in ("dSharpe", "boot_lo", "boot_hi")
        }
        if int(got["n"]) != int(ref["n"]):
            raise ParityError(f"36 V9 {sample}: n {int(got['n'])} vs {int(ref['n'])}")
        rows.append(
            {
                "sample": sample,
                **got,
                "Sharpe_V0": sharpe_ann(pd.Series(r0[mask])),
                "Sharpe_V9": sharpe_ann(pd.Series(r9[mask])),
                "max_abs_diff_vs_36": max(devs.values()),
            }
        )
    return {
        "stats": pd.DataFrame(rows),
        "r_v0": pd.Series(r0, index=dates),
        "r_v9": pd.Series(r9, index=dates),
        "factor": pd.DataFrame(factor, index=dates, columns=clocks),
        "n_fallback": int((~np.isfinite(factor)).sum() + int(tape["censored"].sum())),
    }


def _replay_one_day(rows: pd.DataFrame) -> dict[str, Any]:
    """Every replay one expiration day needs, with no reference columns.

    Pure and picklable: the research's own series are joined on in the parent,
    so a worker process never carries them.
    """
    date = pd.Timestamp(rows["date"].iloc[0])
    hold = replay_day(rows, exit_clock=None, iv_mode="vendor")
    if not hold["ok"]:
        return {"date": date, "refused": str(hold["reason"])}
    exit_ = replay_day(rows, exit_clock=EXIT_CLOCK, iv_mode="vendor")
    hold_mid = replay_day(rows, exit_clock=None, iv_mode="mid")
    exit_mid = replay_day(rows, exit_clock=EXIT_CLOCK, iv_mode="mid")
    row: dict[str, Any] = {
        "date": date,
        "refused": "",
        "Kc": hold["Kc"],
        "Kp": hold["Kp"],
        "S_entry": hold["S_entry"],
        "S_close": hold["S_close"],
        "entry_mid": hold["entry_mid"],
        "entry_bid": hold["entry_bid"],
        "entry_cache": float(rows.loc[rows["hhmm"] == ENTRY_CLOCK, "entry"].iloc[0]),
        "delta_entry": hold["delta_entry"],
        "hedge_hold": hold["hedge_pts"],
        "hedge_exit": exit_["hedge_pts"],
        "exit_settle": hold["exit_price"],
        "exit_model_mark": exit_["exit_price"],
        "r_hold": hold["r_short"],
        "r_exit": exit_["r_short"],
        "r_hold_mid": hold_mid["r_short"],
        "r_exit_mid": exit_mid["r_short"],
        "r_exit_crossed": exit_["r_short_crossed"],
        "one_sided": bool(hold["one_sided"]),
        "n_no_vol_stamps": int(hold["n_no_vol_stamps"]),
    }
    for n in ES_ROUNDING_LOTS:
        rh = replay_day(rows, n_straddles=n, exit_clock=None, iv_mode="vendor")
        re_ = replay_day(rows, n_straddles=n, exit_clock=EXIT_CLOCK, iv_mode="vendor")
        row[f"r_hold_es{n}"] = rh["r_short_rounded"]
        row[f"r_exit_es{n}"] = re_["r_short_rounded"]
    return row


def _replay_chunk(chunk: pd.DataFrame) -> list[dict[str, Any]]:
    """One worker's block of expiration days."""
    return [_replay_one_day(g) for _, g in chunk.groupby("date", sort=True)]


def _replay_all(
    by_date: dict[Any, pd.DataFrame],
    dates: pd.DatetimeIndex,
    *,
    workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[pd.Timestamp, str]]]:
    """Replay every day, in worker processes when there are enough of them."""
    n_workers = min(max(int(workers or (os.cpu_count() or 1)), 1), 16)
    blocks = max(1, min(n_workers, len(dates)))
    chunks = [
        pd.concat([by_date[d] for d in part], ignore_index=True)
        for part in np.array_split(np.asarray(dates), blocks)
        if len(part)
    ]
    if n_workers > 1 and len(dates) >= 4 * n_workers:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            out = list(pool.map(_replay_chunk, chunks))
    else:
        out = [_replay_chunk(c) for c in chunks]
    rows = [r for block in out for r in block]
    refused = [
        (pd.Timestamp(r["date"]), str(r["refused"])) for r in rows if r["refused"]
    ]
    kept = [
        {k: v for k, v in r.items() if k != "refused"} for r in rows if not r["refused"]
    ]
    return kept, refused


# ---------------------------------------------------------------- report ----
def _diff_stats(a: "pd.Series[float]", b: "pd.Series[float]") -> dict[str, float]:
    d = (pd.Series(a) - pd.Series(b)).dropna()
    if d.empty:
        return {"max_abs": _NAN, "mean_abs": _NAN, "n": 0.0}
    return {
        "max_abs": float(d.abs().max()),
        "mean_abs": float(d.abs().mean()),
        "n": float(d.size),
    }


def parity_report(
    n_days: int | None = None,
    *,
    root: Path | None = None,
    write: bool = True,
    with_quotes: bool = True,
    extended: bool | None = None,
    workers: int | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Replay every scored day and compare with the research's own books.

    ``n_days`` restricts the 11:00 replay to the first N expiration dates (the
    test uses a small subsample); ``write`` persists the report and the per-day
    tape under ``results/live_parity/``.  ``extended`` adds the four books the
    2026-09-18 audit asks for -- the 11:00 book, the 13:30 hold book, the
    causal selector book and the corrected-delta variant -- each gated against
    the research it replays; it defaults on unless the replay is subsampled,
    because those books are statements about the whole tape.

    Raises rather than skips when an input is missing, and raises rather than
    prints when a reproduction misses by more than ``PARITY_TOL``.
    """
    root = repo_root() if root is None else Path(root)
    inputs = require_inputs(root)
    if extended is None:
        extended = n_days is None
    cache_path = latest_trade_cache(root)
    pkg = pd.read_parquet(cache_path)
    standalone, rule_mod = research_modules(root)

    ref_hold, ref_mark, ref_hold_x, ref_mark_x = standalone.hold_mark_1100(pkg)

    panel = scored_days(pkg, root)
    panel, n_reinverted = attach_engine_iv(panel, rule_mod)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    if n_days is not None:
        dates = dates[: int(n_days)]
    by_date = {d: g for d, g in panel.groupby("date", sort=True)}

    rec, refused = _replay_all(by_date, dates, workers=workers)
    days = pd.DataFrame(rec).set_index("date").sort_index()
    for column, ref in (
        ("r_hold_ref", ref_hold),
        ("r_exit_ref", ref_mark),
        ("r_hold_ref_crossed", ref_hold_x),
        ("r_exit_ref_crossed", ref_mark_x),
    ):
        days[column] = [float(ref.get(d, _NAN)) for d in days.index]

    quoted = pd.DataFrame(
        index=days.index, columns=["quoted_mid", "quoted_ask"], dtype=float
    )
    if with_quotes and len(days):
        quoted = load_exit_quotes(root, panel, days[["Kc", "Kp"]])
    days["quoted_mid"] = quoted["quoted_mid"]
    days["quoted_ask"] = quoted["quoted_ask"]
    days["r_exit_quoted_mid"] = (
        -(days["quoted_mid"] - days["entry_mid"]) + days["hedge_exit"]
    ) / days["entry_mid"]
    days["r_exit_quoted_ask"] = (
        -(days["quoted_ask"] - days["entry_bid"]) + days["hedge_exit"]
    ) / days["entry_mid"]
    days["mark_minus_quoted_mid"] = days["exit_model_mark"] - days["quoted_mid"]

    era = days.index >= pd.Timestamp(DAILY_ERA_START)
    out: dict[str, Any] = {
        "inputs": {k: str(v) for k, v in inputs.items()},
        "cache": cache_path.name,
        "cache_candidates": len(trade_cache_candidates(root)),
        "one_sided_entries": int(days["one_sided"].sum()) if len(days) else 0,
        "n_days": int(len(days)),
        "n_reinverted_bars": n_reinverted,
        "refused": refused,
        "entry_premium_max_abs_diff": float(
            (days["entry_mid"] - days["entry_cache"]).abs().max()
        )
        if len(days)
        else _NAN,
        "exit_vendor": _diff_stats(days["r_exit"], days["r_exit_ref"]),
        "hold_vendor": _diff_stats(days["r_hold"], days["r_hold_ref"]),
        "exit_mid": _diff_stats(days["r_exit_mid"], days["r_exit"]),
        "hold_mid": _diff_stats(days["r_hold_mid"], days["r_hold"]),
        "sharpe": {
            "exit_vendor": sharpe_ann(days["r_exit"]),
            "exit_ref": sharpe_ann(days["r_exit_ref"]),
            "exit_mid": sharpe_ann(days["r_exit_mid"]),
            "hold_vendor": sharpe_ann(days["r_hold"]),
            "hold_ref": sharpe_ann(days["r_hold_ref"]),
            "hold_mid": sharpe_ann(days["r_hold_mid"]),
            "exit_quoted_mid": sharpe_ann(days["r_exit_quoted_mid"]),
            "exit_quoted_ask": sharpe_ann(days["r_exit_quoted_ask"]),
        },
        "sharpe_era": {
            "exit_vendor": sharpe_ann(days.loc[era, "r_exit"]),
            "exit_quoted_mid": sharpe_ann(days.loc[era, "r_exit_quoted_mid"]),
            "exit_quoted_ask": sharpe_ann(days.loc[era, "r_exit_quoted_ask"]),
            "n": int(era.sum()),
        },
        "mark_minus_quoted_mid_median": float(days["mark_minus_quoted_mid"].median())
        if len(days)
        else _NAN,
        "mark_minus_quoted_mid_median_era": float(
            days.loc[era, "mark_minus_quoted_mid"].median()
        )
        if int(era.sum())
        else _NAN,
        "rounding": {
            n: {
                "exit_sharpe": sharpe_ann(days[f"r_exit_es{n}"]),
                "exit_max_abs": _diff_stats(days[f"r_exit_es{n}"], days["r_exit"])[
                    "max_abs"
                ],
                "hold_sharpe": sharpe_ann(days[f"r_hold_es{n}"]),
                "hold_max_abs": _diff_stats(days[f"r_hold_es{n}"], days["r_hold"])[
                    "max_abs"
                ],
            }
            for n in ES_ROUNDING_LOTS
        },
        "days": days,
    }

    # THE GATE.  The vendor-implied replay is the same formula on the same
    # inputs, so it must agree to floating point; PARITY_TOL is what that means
    # and this is where it is enforced, before anything is written.
    out["vendor_parity_dev"] = {
        "exit": _require_close(
            "the 11:00 exit book against the research's vendor-implied series",
            res_exit_dev := float(out["exit_vendor"]["max_abs"]),
            0.0,
        ),
        "hold": _require_close(
            "the 11:00 hold book against the research's vendor-implied series",
            float(out["hold_vendor"]["max_abs"]),
            0.0,
        ),
    }
    del res_exit_dev

    if extended:
        out.update(extended_books(root, workers=workers, ledger_path=ledger_path))

    if write:
        out_dir = root / "results" / "live_parity"
        out_dir.mkdir(parents=True, exist_ok=True)
        days.drop(columns=["Kc", "Kp"], errors="ignore").to_csv(
            out_dir / "parity_days.csv"
        )
        for name, frame in (
            ("parity_clock_books.csv", out.get("clock_books")),
            ("parity_selector_book.csv", out.get("selector_stats")),
            ("parity_corrected_delta.csv", out.get("v9_stats")),
        ):
            if frame is not None:
                frame.to_csv(out_dir / name, index=False)
        if "selector_picks" in out:
            out["selector_picks"].rename("entry_clock").to_csv(
                out_dir / "parity_selector_picks.csv"
            )
        (out_dir / "parity_report.md").write_text(render_report(out), encoding="utf-8")
        out["out_dir"] = out_dir
    return out


def extended_books(
    root: Path,
    *,
    workers: int | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """The four books of the 2026-09-18 redesign, each gated on its research."""
    tape = research_tape(root, workers=workers)
    p43 = research_43(root)
    idx = tape["dates"]
    selection = validate_strike_selection(tape, root)

    # Book 1: the 11:00 mid-inverted book.  43's own gates run here, unchanged.
    p43.gate_zero(tape["books"], tape["chain"])
    p43.gate_one(tape["books"])
    deck = pd.DatetimeIndex(
        pd.read_csv(
            root
            / "results/atm_straddle_intraday_holdclose/proposals/41/a_insample.csv",
            index_col=0,
            parse_dates=True,
        ).index
    ).intersection(idx)
    r_1100 = tape["books"][ENTRY_CLOCK]["flatten"].reindex(deck)
    era = deck >= pd.Timestamp(DAILY_ERA_START)
    book_1100 = {
        "n_whole": int(r_1100.notna().sum()),
        "n_era": int(r_1100[era].notna().sum()),
        "sharpe_whole": p43.sharpe(r_1100),
        "sharpe_era": p43.sharpe(r_1100[era]),
    }
    book_1100["dev_whole"] = _require_close(
        "43's 11:00 mid-inverted flatten book, whole in-sample Sharpe",
        book_1100["sharpe_whole"],
        REF_1100_WHOLE_SHARPE,
        PARITY_SHARPE_TOL,
    )
    book_1100["dev_era"] = _require_close(
        "43's 11:00 mid-inverted flatten book, daily-era Sharpe",
        book_1100["sharpe_era"],
        REF_1100_ERA_SHARPE,
        PARITY_SHARPE_TOL,
    )

    # Book 2: the 13:30 hold book, the redesign's target entry.
    clock_books = pd.concat(
        [
            fixed_clock_book(tape, AFTERNOON_CLOCK, "hold", root),
            fixed_clock_book(tape, ENTRY_CLOCK, "hold", root),
        ],
        ignore_index=True,
    )

    # Book 3: the causal selector book.
    path = (
        root / "results" / "live_seed" / "premium_ledger.parquet"
        if ledger_path is None
        else Path(ledger_path)
    )
    ledger = PremiumLedger.load(path)
    selector = selector_book(tape, ledger, "hold", SELECTOR_WINDOW, root)

    # Book 4: the corrected delta (proposal 36's V9) on the 11:00 book.
    v9 = corrected_delta_book(v9_tape(root), root)

    return {
        "n_sessions": int(len(idx)),
        "n_half_sessions": int(len(tape["half_sessions"])),
        "n_workers": tape["n_workers"],
        "selection": selection,
        "book_1100": book_1100,
        "clock_books": clock_books,
        "selector_stats": selector["stats"],
        "selector_picks": selector["picks"],
        "selector_pick_counts": selector["pick_counts"],
        "selector_flat": selector["n_flat"],
        "v9_stats": v9["stats"],
        "ledger_path": str(path),
        "ledger_sessions": len(ledger.sessions()),
    }


def render_report(res: dict[str, Any]) -> str:
    """The parity report, as the markdown written beside the per-day tape."""
    s = res["sharpe"]
    se = res["sharpe_era"]
    lines = [
        "# Live engine vs backtest: parity report",
        "",
        f"Trade cache: `{res['cache']}` "
        f"(of {res.get('cache_candidates', 1)} in the cache directory, newest by "
        "modification time)  ",
        f"Days replayed at 11:00: {res['n_days']}  ",
        f"Every reproduction below is ASSERTED at PARITY_TOL = {PARITY_TOL:.0e} "
        f"(published Sharpes at {PARITY_SHARPE_TOL:.0e}, half a unit in their "
        "last printed place) and raises `ParityError` on a miss; a missing input "
        "raises `FileNotFoundError` naming it, rather than skipping.  ",
        "",
        "Book of record: short one nearest-OTM 0DTE straddle at 11:00,",
        "delta-hedged every 30 minutes, bought back at 15:30 with the futures",
        "leg flattened at the same stamp (the EXIT book). The HOLD book is the",
        "same entry and hedge carried through cash settlement.",
        "",
        "## 1. Vendor-implied path (must be the same formula on the same inputs)",
        "",
        "| book | max abs diff vs research | Sharpe (engine) | Sharpe (research) |",
        "| --- | --- | --- | --- |",
        f"| exit (buy back at 15:30) | {res['exit_vendor']['max_abs']:.3e} | "
        f"{s['exit_vendor']:.6f} | {s['exit_ref']:.6f} |",
        f"| hold (cash settlement) | {res['hold_vendor']['max_abs']:.3e} | "
        f"{s['hold_vendor']:.6f} | {s['hold_ref']:.6f} |",
        "",
        f"Bars whose vendor implied volatility sat on a solver bracket node and "
        f"was re-inverted from the straddle midpoint by the live engine: "
        f"{res['n_reinverted_bars']}.",
        f"Entry premium rebuilt from the leg quotes, largest disagreement with "
        f"the cached premium: {res['entry_premium_max_abs_diff']:.3e} points.",
        "",
        "## 2. Mid-inverted path (what the live engine must do at every stamp)",
        "",
        "| book | max abs diff | mean abs diff | Sharpe (mid-inverted) | Sharpe (vendor) |",
        "| --- | --- | --- | --- | --- |",
        f"| exit | {res['exit_mid']['max_abs']:.6f} | {res['exit_mid']['mean_abs']:.6f} | "
        f"{s['exit_mid']:.6f} | {s['exit_vendor']:.6f} |",
        f"| hold | {res['hold_mid']['max_abs']:.6f} | {res['hold_mid']['mean_abs']:.6f} | "
        f"{s['hold_mid']:.6f} | {s['hold_vendor']:.6f} |",
        "",
        "This is a finding, not a defect: the broker publishes no vendor implied",
        "volatility, so the live delta is taken from a volatility bisected out of",
        "the straddle's own midpoint.",
        "",
        "## 3. Whole-contract futures hedge (ES, multiplier 50)",
        "",
        "| straddles | exit Sharpe | exit max abs diff | hold Sharpe | hold max abs diff |",
        "| --- | --- | --- | --- | --- |",
    ]
    for n, r in res["rounding"].items():
        lines.append(
            f"| {n} | {r['exit_sharpe']:.6f} | {r['exit_max_abs']:.6f} | "
            f"{r['hold_sharpe']:.6f} | {r['hold_max_abs']:.6f} |"
        )
    lines += [
        "",
        "Continuous hedge for comparison: "
        f"exit {s['exit_vendor']:.6f}, hold {s['hold_vendor']:.6f}.",
        "",
        "## 4. What the 15:30 buy-back actually costs",
        "",
        "| buy-back | Sharpe (whole sample) | Sharpe (daily era) |",
        "| --- | --- | --- |",
        f"| Black-76 model mark | {s['exit_vendor']:.4f} | {se['exit_vendor']:.4f} |",
        f"| quoted midpoint | {s['exit_quoted_mid']:.4f} | {se['exit_quoted_mid']:.4f} |",
        f"| quoted ask, entry at the bid | {s['exit_quoted_ask']:.4f} | "
        f"{se['exit_quoted_ask']:.4f} |",
        "",
        f"Daily era is {DAILY_ERA_START} onward ({se['n']} days).",
        "Median model mark minus quoted midpoint: "
        f"{res['mark_minus_quoted_mid_median']:.4f} points whole sample, "
        f"{res['mark_minus_quoted_mid_median_era']:.4f} points in the daily era.",
        "",
    ]
    if "book_1100" in res:
        lines += _render_extended(res)
    if res["refused"]:
        lines += [
            "## Refused days",
            "",
            *[f"- {d.date()}: {why}" for d, why in res["refused"]],
            "",
        ]
    return "\n".join(lines)


def _render_extended(res: dict[str, Any]) -> list[str]:
    """Sections 5-8: the redesign's books, each against the research it replays."""
    b11 = res["book_1100"]
    sel = res["selector_stats"]
    books = res["clock_books"]
    v9 = res["v9_stats"]
    picks = res["selector_pick_counts"]

    def row(frame: pd.DataFrame, **kw: Any) -> "pd.Series[Any]":
        m = pd.Series(True, index=frame.index)
        for k, v in kw.items():
            m &= frame[k] == v
        return frame.loc[m].iloc[0]

    lines = [
        "## 5. The whole tape, and the strike selection",
        "",
        f"Sessions replayed: {res['n_sessions']} "
        f"(2020-01-03 .. 2025-12-31, {res['n_half_sessions']} half sessions "
        "dropped: the 15:30 row has already expired on them).  ",
        f"Per-clock tapes built on {res['n_workers']} worker processes.  ",
        "Strike-selection parity, at the CHAIN level rather than on the two legs "
        "the backtest already chose: the nearest-OTM straddle is re-picked from "
        "the whole chain at every stamp with the research's guards, and its "
        f"11:00 strikes agree with the trade cache on all "
        f"{res['selection']['n_sessions']} common sessions "
        f"({res['selection']['n_disagreements']} disagreements, "
        f"{res['selection']['refused_stamps']} stamps refused by the guards).  ",
        f"Entries whose package had a leg with no bid: {res['one_sided_entries']}.",
        "",
        "## 6. The 11:00 book, mid-inverted (proposal 43's GATE 0)",
        "",
        "| sample | n | engine Sharpe | research | difference |",
        "| --- | --- | --- | --- | --- |",
        f"| deck in-sample | {b11['n_whole']} | {b11['sharpe_whole']:.7f} | "
        f"{REF_1100_WHOLE_SHARPE:.7f} | {b11['dev_whole']:.2e} |",
        f"| daily era | {b11['n_era']} | {b11['sharpe_era']:.7f} | "
        f"{REF_1100_ERA_SHARPE:.7f} | {b11['dev_era']:.2e} |",
        "",
        "43's own GATE 0 and GATE 1 run inside this harness on the same tape, so "
        "the day-by-day reproduction of proposal 41's chain build and the 413 "
        "unseen sessions' mean and HAC t are asserted too.",
        "",
        "## 7. Fixed entry clocks, held to settlement, on the 413 unseen sessions",
        "",
        "| clock | unit | n | mean | Sharpe | HAC t | MaxDD | worst | vs 43 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for clock in (AFTERNOON_CLOCK, ENTRY_CLOCK):
        for unit in ("index points per contract", "premium"):
            r = row(books, clock=clock, period="unseen", unit=unit)
            lines.append(
                f"| {clock} hold | {unit} | {int(r['n'])} | {r['mean']:+.6f} | "
                f"{r['Sharpe_ann']:.6f} | {r['t_hac']:+.6f} | {r['MaxDD']:.4f} | "
                f"{r['worst']:.4f} | {r['max_abs_diff_vs_43']:.1e} |"
            )
    lines += [
        "",
        "The afternoon entry is the redesign's target: 13:30 held to the "
        "settlement earns +1.173260 index points a day per contract on the "
        "unseen sessions with a HAC t of +3.210432, where fixed 11:00 held is "
        "inside noise. Every cell is asserted against "
        "`proposals/43/a_clock_year.csv`.",
        "",
        "## 8. The causal selector book (proposal 46, estimator E2, rolling 252)",
        "",
        "Entry clock chosen each session by the trailing ratio of sums of "
        "implied over realized remaining-window variance per clock, relative to "
        "that clock's own expanding level, on sessions strictly earlier; held to "
        "the settlement. The pick sequence agrees with proposal 46 on every one "
        f"of the {res['n_sessions']} sessions ({res['selector_flat']} warm-up "
        "sessions flat), and the daily series is pinned through 46's complete "
        "statistic set.",
        "",
        "| sample | unit | n | mean | Sharpe | HAC t | MaxDD | vs 46 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sample in ("deck in-sample", "unseen", "all"):
        for unit in ("pts", "units"):
            r = row(sel, period=sample, unit=unit)
            lines.append(
                f"| {sample} | {unit} | {int(r['n'])} | {r['mean']:+.6f} | "
                f"{r['Sharpe_ann']:.6f} | {r['t_hac']:+.6f} | {r['MaxDD']:.4f} | "
                f"{r['max_abs_diff_vs_46']:.1e} |"
            )
    lines += [
        "",
        "Clocks picked: "
        + ", ".join(f"{k} {int(v)}" for k, v in picks.items())
        + f", flat {res['selector_flat']}.",
        "",
        "Read it as proposal 46 does: the selector tracks the morning-to-"
        "afternoon migration about a quarter late, and it does NOT beat fixed "
        "11:00 with an interval excluding zero, nor fixed 13:30. It is the "
        "defensible causal choice, not a demonstrated improvement.",
        "",
        "## 9. The corrected delta (proposal 36's V9) on the 11:00 book",
        "",
        "The hedge delta is taken from the implied volatility corrected by the "
        "ledger's trailing lagged per-clock mean of realized over implied "
        "variance (`pricing.corrected_total_vol`), with the research's fallback "
        "to the uncorrected implied wherever the factor is missing or the "
        "vendor implied sat on the solver's bracket node. Book: the "
        "crossed-quoted 15:30 exit of the 11:00 straddle.",
        "",
        "| sample | n | Sharpe V0 | Sharpe V9 | dSharpe | 95% CI | vs 36 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in v9.iterrows():
        lines.append(
            f"| {r['sample']} | {int(r['n'])} | {r['Sharpe_V0']:.6f} | "
            f"{r['Sharpe_V9']:.6f} | {r['dSharpe']:+.6f} | "
            f"[{r['boot_lo']:+.5f}, {r['boot_hi']:+.5f}] | "
            f"{r['max_abs_diff_vs_36']:.1e} |"
        )
    lines += [
        "",
        f"Ledger: `{res['ledger_path']}` ({res['ledger_sessions']} sessions). "
        "The V9 factor here is the ledger's own estimator fed proposal 36's "
        "implied and realized grids, so the reproduction is of the estimator as "
        "well as of the book. None of these numbers is charged the 0.5 bp hedge "
        "cost, and V9 raises turnover: proposal 27's reading, that the exit book "
        "nets about Sharpe 1.20 at 0.5 bp and V9 about 1.31, is the one to quote.",
        "",
    ]
    return lines


# ------------------------------------------------------------ the seed ------
#: Where the live engine keeps its premium ledger.
LIVE_SEED = "results/live_seed/premium_ledger.parquet"


def realized_remaining_variance(chain: dict[str, Any]) -> np.ndarray:
    """Realized variance over the remaining window at every stamp.

    Proposal 43's construction: the summed squared 30-minute log spot returns
    from the stamp through the 15:30 stamp plus the last step into the
    settlement print -- the same steps the hedge walks.  A window with a
    missing step is NaN, not a shorter window.
    """
    spot = chain["S"]
    nxt = np.full_like(spot, np.nan)
    nxt[:, :-1] = spot[:, 1:]
    nxt[:, -1] = chain["S_close"]
    ok = np.isfinite(spot) & np.isfinite(nxt)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(ok, np.log(np.where(ok, nxt / spot, 1.0)) ** 2, np.nan)
    out = np.full_like(spot, np.nan)
    for k in range(spot.shape[1]):
        whole = np.isfinite(step[:, k:]).all(axis=1)
        out[:, k] = np.where(whole, np.nansum(step[:, k:], axis=1), np.nan)
    return out


def seed_premium_ledger(
    root: Path | None = None,
    *,
    tape: dict[str, Any] | None = None,
    write: bool = True,
    workers: int | None = None,
) -> dict[str, Any]:
    """Build ``results/live_seed/premium_ledger.parquet`` from the research tape.

    One pass, vectorised: 43's chain tape gives the mid-inverted implied
    remaining variance at every stamp and the spot path gives the realized one,
    and both go into the ledger as they are -- an unusable cell is recorded as
    the NaN it is and the estimators mask it.

    Three gates run before the file is written, each of them the brief's:

    1. ``ratio_of_sums`` per clock over every calendar period reproduces 43's
       ``a_premium_curve``;
    2. ``selector.pick_entry_clock`` at windows 252 and 126 reproduces proposal
       46's E2 pick sequences day for day;
    3. ``realized_over_implied_mean``, fed proposal 36's own implied and
       realized grids, reproduces 36's V9 factor cell by cell.
    """
    base = repo_root() if root is None else Path(root)
    require_inputs(base)
    p43 = research_43(base)
    if tape is None:
        tape = research_tape(base, workers=workers)
    chain = tape["chain"]
    idx = tape["dates"]
    clocks = list(p43.CLOCKS)
    ledger = PremiumLedger()
    ledger.append_session(
        records_from_grids(
            idx,
            clocks,
            chain["tot"] ** 2,
            realized_remaining_variance(chain),
            chain["S"],
            chain["entry"],
            chain["K_c"],
            chain["K_p"],
        )
    )

    # GATE 1 -- the premium curve, clock by clock and period by period.
    curve = p43.premium_curve(chain, idx).pivot(
        index="clock", columns="period", values="implied_over_realised"
    )
    masks = p43.period_masks(idx)
    cells = []
    for period in curve.columns:
        pos = np.flatnonzero(masks[period])
        if not pos.size or not np.array_equal(
            pos, np.arange(pos[0], pos[-1] + 1)
        ):  # pragma: no cover - the research's periods are contiguous
            raise ParityError(f"period {period!r} is not a contiguous block")
        i0, i1 = int(pos[0]), int(pos[-1]) + 1
        asof = (
            idx[i1].date() if i1 < len(idx) else (idx[-1] + pd.Timedelta(days=1)).date()
        )
        for clock in p43.ENTRIES:
            got = ledger.ratio_of_sums(clock, asof, i1 - i0, min_sessions=0)
            cells.append(
                {
                    "period": period,
                    "clock": clock,
                    "ledger": got,
                    "proposal_43": float(curve.loc[clock, period]),
                    "abs_diff": _require_close(
                        f"43's premium curve at {clock} over {period}",
                        got,
                        float(curve.loc[clock, period]),
                    ),
                }
            )
    curve_check = pd.DataFrame(cells)

    # GATE 2 -- the pick sequence, at both of 46's E2 windows.
    timeline = pd.read_csv(
        base
        / "results/atm_straddle_intraday_holdclose/proposals/46/selector_timeline.csv",
        index_col=0,
        parse_dates=True,
    )
    picks: dict[int, pd.Series] = {}
    for window in (SELECTOR_WINDOW, 126):
        mine = selector_picks(ledger, idx, window=window)
        want = timeline[f"pick_E2 relative ratio of sums | rolling {window}"].reindex(
            idx
        )
        a = mine.where(mine.notna(), None).to_numpy()
        b = want.where(want.notna(), None).to_numpy()
        bad = int(sum(1 for x, y in zip(a, b, strict=True) if x != y))
        if bad:
            raise ParityError(
                f"the engine's E2 rolling-{window} picks disagree with proposal 46 "
                f"on {bad} of {len(idx)} sessions"
            )
        picks[window] = mine

    # GATE 3 -- the V9 factor, on proposal 36's own grids.
    v9 = v9_tape(base)
    reference = (
        pd.DataFrame(v9["realized"] / v9["var0"])
        .expanding(min_periods=MIN_SESSIONS)
        .mean()
        .shift(1)
        .to_numpy(float)
    )
    book_ledger = PremiumLedger()
    book_ledger.append_session(
        records_from_grids(
            v9["dates"],
            v9["clocks"],
            v9["var0"],
            v9["realized"],
            v9["spot"],
            np.repeat(v9["entry_mid"][:, None], len(v9["clocks"]), axis=1),
            np.repeat(v9["Kc"][:, None], len(v9["clocks"]), axis=1),
            np.repeat(v9["Kp"][:, None], len(v9["clocks"]), axis=1),
        )
    )
    factor_rows = []
    era = np.asarray(v9["dates"] >= pd.Timestamp(DAILY_ERA_START))
    for j, clock in enumerate(v9["clocks"]):
        mine = book_ledger.realized_over_implied_series(clock, None).to_numpy(float)
        want = reference[:, j]
        both_nan = np.isnan(mine) & np.isnan(want)
        dev = float(np.max(np.where(both_nan, 0.0, np.abs(mine - want))))
        factor_rows.append(
            {
                "clock": clock,
                "n_with_factor": int(np.isfinite(mine).sum()),
                "median_variance_factor_era": float(np.nanmedian(mine[era])),
                "abs_diff": _require_close(f"36's V9 factor at {clock}", dev, 0.0),
            }
        )

    path = base / LIVE_SEED
    if write:
        ledger.save(path)
    return {
        "path": path,
        "n_rows": int(len(ledger.frame)),
        "n_sessions": len(ledger.sessions()),
        "clocks": ledger.clocks(),
        "curve_check": curve_check,
        "curve_max_abs_diff": float(curve_check["abs_diff"].max()),
        "picks": picks,
        "n_flat": {w: int(v.isna().sum()) for w, v in picks.items()},
        "v9_factor": pd.DataFrame(factor_rows),
        "ledger": ledger,
    }


# ------------------------------------------------- the month-end override ----
#: Proposal 54's day file: one row per chain session, with the 15:30 long
#: straddle's at-the-ask return in premium units (``R_ask``) and index points
#: per contract (``pts_ask``), and the month-end flag.
P54_DAILY = "results/atm_straddle_0dte_1530/proposals/54/c_daily.csv"
#: The research's settlement: the official SPX close (``PARITY_INPUTS``).
SETTLEMENT_TAPE = "results/atm_straddle_intraday_holdclose/cache/gspc_ohlc.parquet"
#: The capital the override replay is run at: the README's worked example.  At
#: ``--n 1`` only the loss-budget check reads it, and one straddle's premium is
#: a small fraction of its tenth.
PARITY_CAPITAL = 1_000_000.0


def _month_end_one(job: tuple[str, str, str, str, float]) -> dict[str, Any]:
    """Replay one month-end through the RUNNER in override mode and reconcile.

    One long straddle (``--n 1``) is bought at 15:30 through ``FakeBroker``
    (``--fill cross``: at the package ask) and held; the provisional P&L is
    read at the tape's 16:00 print, the final one after ``--reconcile`` at the
    official close the research settles at.
    """
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    from .broker import FakeBroker, load_replay_day  # noqa: PLC0415
    from .config import Config  # noqa: PLC0415
    from .journal import DaySummary, Journal  # noqa: PLC0415
    from .run_day import Abort, DayRunner, reconcile_day  # noqa: PLC0415

    day, chain, seed, tmp_root, official = job
    jdir = os.path.join(tmp_root, day)
    os.makedirs(jdir, exist_ok=True)
    cfg = Config(
        replay_date=day,
        journal_dir=jdir,
        ledger_path=seed,
        ledger_live_path=os.path.join(jdir, "live_ledger.parquet"),
        replay_cache_dir=os.path.join(jdir, "cache"),
        n_override=1,
        capital=PARITY_CAPITAL,
        month_end_mode="override",
    )
    out: dict[str, Any] = {"date": day}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        broker = FakeBroker(
            cfg, day, frame=load_replay_day(day, chain_path=chain, cache_dir=None)
        )
        broker.connect()
        jr = Journal(os.path.join(jdir, day + ".jsonl"), echo=False)
        try:
            DayRunner(cfg, broker, jr).run()
        except Abort as exc:
            out["refused"] = str(exc)
        finally:
            jr.close()
        prov = DaySummary.from_journal(jr.path)
        out.update(
            side=prov.side,
            n=prov.n_straddles,
            K_c=prov.K_c,
            K_p=prov.K_p,
            fill=prov.entry_premium,
            payoff_provisional=prov.exit_price,
            pts_provisional=prov.option_pnl_points,
            flat_reason=prov.flat_reason,
        )
        cfg.settlement = float(official)
        out["reconcile_rc"] = reconcile_day(cfg, broker)
    final = DaySummary.from_journal(jr.path)
    out.update(
        payoff_official=final.exit_price,
        pts_official=final.option_pnl_points,
        units_official=final.pnl_units,
        provisional_after_reconcile=final.provisional,
    )
    return out


def month_end_override_parity(
    root: Path | None = None,
    dates: Sequence[str] | None = None,
    workers: int | None = None,
    tol: float = PARITY_TOL,
) -> dict[str, Any]:
    """The runner's month-end override against proposal 54's long straddle.

    Every month-end in proposal 54's day file (or ``dates``) is replayed
    through ``DayRunner`` + ``FakeBroker`` in override mode and reconciled at
    the official close; the per-contract P&L must equal 54's ``pts_ask`` and
    the premium-unit return its ``R_ask``, within ``tol``, on every day the
    runner traded.  A day the runner ends flat is reported, not dropped.
    Raises :class:`ParityError` on any day outside the bar.
    """
    import tempfile  # noqa: PLC0415

    root = repo_root() if root is None else Path(root)
    daily = pd.read_csv(root / P54_DAILY, index_col=0, parse_dates=True)
    me = daily[daily["month_end"].astype(bool)]
    days = [str(d.date()) for d in me.index] if dates is None else list(dates)
    close = pd.read_parquet(root / SETTLEMENT_TAPE)["close"]
    close.index = pd.to_datetime(close.index).normalize()
    chain = str(root / "data" / "spxw_chain.parquet")
    seed = str(root / LIVE_SEED)
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(d, chain, seed, tmp, float(close.loc[pd.Timestamp(d)])) for d in days]
        n_workers = min(len(jobs), workers or (os.cpu_count() or 1))
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                rows = list(pool.map(_month_end_one, jobs))
        else:
            rows = [_month_end_one(j) for j in jobs]
    tab = pd.DataFrame(rows).set_index("date")
    tab.index = pd.to_datetime(tab.index)
    tab["pts_research"] = me["pts_ask"].reindex(tab.index)
    tab["units_research"] = me["R_ask"].reindex(tab.index)
    tab["pts_diff"] = tab["pts_official"] - tab["pts_research"]
    tab["units_diff"] = tab["units_official"] - tab["units_research"]
    tab["settle_source_pts"] = tab["pts_provisional"] - tab["pts_official"]
    traded = tab[tab["side"] == "long"]
    worst = float(traded[["pts_diff", "units_diff"]].abs().max().max())
    if len(traded):
        _require_close(
            "month-end override, per-contract P&L at the official close",
            worst,
            0.0,
            tol,
        )
    return {
        "n_days": int(len(tab)),
        "n_traded": int(len(traded)),
        "flat": {
            str(d.date()): str(r["flat_reason"] or r.get("refused", ""))
            for d, r in tab[tab["side"] != "long"].iterrows()
        },
        "max_abs_pts_diff": float(traded["pts_diff"].abs().max()),
        "max_abs_units_diff": float(traded["units_diff"].abs().max()),
        "settle_source_pts_mean_abs": float(traded["settle_source_pts"].abs().mean()),
        "mean_pts": float(traded["pts_official"].mean()),
        "mean_units": float(traded["units_official"].mean()),
        "table": tab,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """``--seed`` rebuilds and re-gates the premium ledger; ``--month-end``
    replays the month-end override against proposal 54; otherwise the report."""
    import time  # noqa: PLC0415

    args = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    if "--month-end" in args:
        me = month_end_override_parity()
        print(
            f"month-end override: {me['n_traded']} of {me['n_days']} month-ends "
            f"traded through the runner; per-contract P&L vs proposal 54 max "
            f"|diff| {me['max_abs_pts_diff']:.3e} pts, premium units "
            f"{me['max_abs_units_diff']:.3e} (bar {PARITY_TOL:.0e}); mean "
            f"{me['mean_pts']:+.4f} pts / {me['mean_units']:+.4f} units; "
            f"settlement print vs official close moves the day by "
            f"{me['settle_source_pts_mean_abs']:.4f} pts on average"
        )
        for day, why in me["flat"].items():
            print(f"  flat {day}: {why}")
        print(f"done in {time.time() - t0:.1f} s")
        return
    if "--seed" in args:
        res = seed_premium_ledger()
        print(
            f"wrote {res['path']}: {res['n_rows']} rows, {res['n_sessions']} "
            f"sessions, clocks {res['clocks'][0]}..{res['clocks'][-1]}"
        )
        print(
            f"GATE 1  ratio_of_sums vs 43's premium curve: "
            f"{len(res['curve_check'])} cells, max abs deviation "
            f"{res['curve_max_abs_diff']:.3e} (bar {PARITY_TOL:.0e})"
        )
        for window, flat in res["n_flat"].items():
            print(
                f"GATE 2  pick_entry_clock(window={window}) reproduces 46's E2 "
                f"pick sequence on all {res['n_sessions']} sessions "
                f"({flat} warm-up sessions flat)"
            )
        print(
            f"GATE 3  realized_over_implied_mean vs 36's V9 factor: max abs "
            f"deviation {float(res['v9_factor']['abs_diff'].max()):.3e}"
        )
        print(res["v9_factor"].to_string(index=False))
        print(f"done in {time.time() - t0:.1f} s")
        return
    res = parity_report()
    print(render_report(res))
    print("wrote", res.get("out_dir"), f"in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
