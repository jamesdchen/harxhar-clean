"""32 - loss anatomy of the 11:00 delta-hedged short straddle, and two counters.

The book of record: every session sells the nearest-OTM SPX 0DTE straddle at
11:00 ET, delta-hedges the package on the vendor spot every 30 minutes, and
buys the straddle back at 15:30 at the QUOTED ask with the entry taken at the
quoted bid (fully crossed), the futures leg flattened at the same stamp.  The
"mid" variant of the same book buys back at the quoted midpoint and sells at
the midpoint.  Returns are one unit a day in units of the 11:00 midpoint
entry premium.

Part A  a per-bar attribution of the day's P&L.  The package's Black-76 value
        is walked stamp by stamp and the walk is split, exactly, into the
        hedge leg, a convexity term (the gamma bill), a theta term and a
        volatility-path term; the quoted buy-back and the quoted entry enter
        as explicit basis terms.  The identity is asserted per day.
Part B  the exit-clock ladder: the same book bought back at 13:00, 13:30,
        14:00, 14:30, 15:00 and 15:30, with hold-to-settlement for reference.
Part C  hedge-to-a-delta-band on the 30-minute tape, with the repo's 0.5 bp
        turnover charge.

Everything is built on the live engine (``live/ibkr/pricing.py`` and
``live/ibkr/parity.py``), which reproduces the research to 2e-13; the three
published gate numbers are asserted before a single new number is computed.

Run:  python writeup/intraday_proposals/32_loss_anatomy.py
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.ibkr.parity import (  # noqa: E402
    DAILY_ERA_START,
    attach_engine_iv,
    latest_trade_cache,
    replay_day,
    research_modules,
    scored_days,
)
from live.ibkr.pricing import (  # noqa: E402
    invert_total_vol,
    package_delta,
    package_price,
)

HOLD = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "32"
CHAIN = ROOT / "data" / "spxw_chain.parquet"

ENTRY = "11:00"
CLOSE = "15:30"
#: The stamps the 11:00 book lives on: entry, eight rebalances, the buy-back.
SESSION: tuple[str, ...] = (
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
#: Hours from each stamp to the 16:00 settlement (live.ibkr.pricing.hours_to_close).
H_REM: tuple[float, ...] = tuple(16.0 - (11.0 + 0.5 * k) for k in range(len(SESSION)))
ERA0 = pd.Timestamp(DAILY_ERA_START)
PERIODS_PER_YEAR = 252.0
ANN = sqrt(PERIODS_PER_YEAR)

#: The published numbers this script reproduces before it computes anything new.
GATE_N_DAYS = 865
GATE_CROSSED_WHOLE = 2.252546754993
GATE_CROSSED_ERA = 2.513132263902
GATE_MARK_MID_WHOLE = 4.695674
#: The gate is a reproduction, not an equality of float paths: the engine's own
#: replay and the research's vectorised tape agree with each other to 1e-12 and
#: with the quoted numbers to this bar.
GATE_TOL = 1e-6
#: The per-day attribution identity, in units of the entry premium.
SUM_TOL = 1e-9

#: |index return in the bar| cut points for the concentration table (Part A1).
BAR_MOVE_GRID: tuple[float, ...] = (0.0025, 0.005, 0.01)
#: The bars the deck calls the last stretch (Part A2).
LATE_BARS: tuple[str, ...] = ("14:00", "14:30", "15:00")
#: The top decile of the 15:30 volatility ratio (Part A4).
TOP_DECILE = 0.90
#: How many worst days the ledger names (Part A5).
N_WORST = 10
#: Exit clocks for the ladder (Part B).
EXIT_CLOCKS: tuple[str, ...] = ("13:00", "13:30", "14:00", "14:30", "15:00", "15:30")
#: Rebalance bands in package delta per straddle (Part C); 0 is the book.
BAND_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.25)
#: The repo's hedge turnover charge, basis points of S x |delta traded|
#: (UNDERLYING_COST_BP, writeup/intraday_proposals/25_dh_holdclose.py).
HEDGE_COST_BP = 0.5


# ------------------------------------------------------------------ stats ---
def sharpe_ann(x: Any) -> float:
    """Annualised Sharpe of a daily return series (one unit a day, ddof=1)."""
    v = pd.Series(x).dropna()
    if v.size < 2:
        return float("nan")
    sd = float(v.std(ddof=1))
    if not (np.isfinite(sd) and sd > 0.0):
        return float("nan")
    return float(v.mean()) / sd * ANN


def maxdd(x: Any) -> float:
    """Worst peak-to-trough of the cumulative SUM path, peak seeded at 0."""
    v = pd.Series(x).dropna().sort_index().to_numpy(float)
    if v.size < 1:
        return float("nan")
    path = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def stats(s: Any, book: str, sample: str, **extra: Any) -> dict[str, Any]:
    """n / mean / sd / Sharpe_ann / t / MaxDD / worst day of a daily series."""
    d = pd.Series(s).dropna().sort_index()
    n = int(d.size)
    mean = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n >= 2 else float("nan")
    ok = bool(np.isfinite(sd)) and sd > 0.0
    row: dict[str, Any] = {
        "book": book,
        "sample": sample,
        "n": n,
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if ok else float("nan"),
        "t": sqrt(n) * mean / sd if ok else float("nan"),
        "MaxDD": maxdd(d),
        "worst_day": float(d.min()) if n else float("nan"),
        "worst_date": str(d.idxmin().date()) if n else "",
    }
    row.update(extra)
    return row


def both_samples(s: Any, book: str, **extra: Any) -> list[dict[str, Any]]:
    """The series scored on the whole sample and on the daily-0DTE era."""
    d = pd.Series(s).dropna()
    return [
        stats(d, book, "whole", **extra),
        stats(d[d.index >= ERA0], book, "daily_era", **extra),
    ]


def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(use.to_string(index=False))


def write(
    df: pd.DataFrame,
    name: str,
    title: str,
    cols: list[str] | None = None,
    head: int | None = None,
) -> None:
    """Persist a table under the proposal's own directory and print it.

    ``head`` prints only the first rows of a table too long to read whole; the
    file on disk always carries every row, and its aggregates are printed in
    full by the blocks that follow.
    """
    df.to_csv(OUT / name, index=False)
    if head is not None and len(df) > head:
        show(df.head(head), f"{title}   [{name}], first {head} of {len(df)} rows", cols)
    else:
        show(df, f"{title}   [{name}]", cols)


# ------------------------------------------------------------- the panel ----
def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """The scored 11:00 days, the engine's implied volatility attached.

    ``scored_days`` is the parity harness's own day filter (the deck's days
    that carry a tradeable 11:00 straddle) and ``attach_engine_iv`` re-inverts
    the bars whose vendor implied volatility sat on the solver's bracket node.
    """
    pkg = pd.read_parquet(latest_trade_cache(ROOT))
    _, rule_mod = research_modules(ROOT)
    panel, n_re = attach_engine_iv(scored_days(pkg, ROOT), rule_mod)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    ent = (
        panel.loc[panel["hhmm"] == ENTRY]
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    print(
        f"panel {panel.shape[0]} stamp-rows on {len(dates)} expirations; "
        f"{n_re} censored vendor bars re-inverted by the live engine"
    )
    return panel, ent, dates


def quotes_at_stamps(
    chain: Path, stamps: tuple[str, ...], keys: pd.DataFrame
) -> pd.DataFrame:
    """Quoted bid/ask of named strikes at named ET stamps, from the option chain.

    Generalises ``27_tradeable_polish.quotes_1530``/``leg_quotes`` to any ET
    clock.  ``keys`` carries one row per (expiration, hhmm, strike, cp) the
    caller wants; the chain's UTC stamp is converted to ET with integer hour
    and minute (not ``strftime``), a 0DTE row is one whose expiration equals
    the ET date of its stamp, and only the asked-for rows survive the scan.
    """
    hh = {int(c[:2]) for c in stamps}
    mm = {int(c[3:]) for c in stamps}
    want = keys.drop_duplicates(["expiration", "hhmm", "strike", "cp"])
    src = pq.ParquetFile(chain)
    cols = ["expiration", "strike", "cp", "bid", "ask", "timestamp"]
    parts: list[pd.DataFrame] = []
    for i in range(src.num_row_groups):
        blk = src.read_row_group(i, columns=cols).to_pandas()
        et = blk["timestamp"].dt.tz_convert("America/New_York")
        hour = et.dt.hour.to_numpy()
        minute = et.dt.minute.to_numpy()
        keep = np.isin(hour, list(hh)) & np.isin(minute, list(mm))
        keep &= (
            blk["expiration"].to_numpy()
            == et.dt.normalize().dt.tz_localize(None).to_numpy()
        )
        sub = blk.loc[keep].copy()
        if sub.empty:
            continue
        sub["hhmm"] = [
            f"{a:02d}:{b:02d}" for a, b in zip(hour[keep], minute[keep], strict=True)
        ]
        sub["strike"] = sub["strike"].astype(float)
        sub["cp"] = sub["cp"].astype(str)
        parts.append(
            want.merge(
                sub[["expiration", "hhmm", "strike", "cp", "bid", "ask"]],
                on=["expiration", "hhmm", "strike", "cp"],
                how="inner",
            )
        )
    got = pd.concat(parts, ignore_index=True)
    assert not got.duplicated(["expiration", "hhmm", "strike", "cp"]).any(), (
        "dup quotes"
    )
    return got


def quote_grids(
    ent: pd.DataFrame, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Package bid / mid / ask of the 11:00 strikes at every 30-minute stamp."""
    rows: list[pd.DataFrame] = []
    for cp, col in (("C", "K_c"), ("P", "K_p")):
        for clock in SESSION:
            rows.append(
                pd.DataFrame(
                    {
                        "expiration": pd.DatetimeIndex(dates),
                        "hhmm": clock,
                        "strike": ent[col].to_numpy(float),
                        "cp": cp,
                    }
                )
            )
    keys = pd.concat(rows, ignore_index=True)
    got = quotes_at_stamps(CHAIN, SESSION, keys).set_index(
        ["expiration", "hhmm", "strike", "cp"]
    )
    n, m = len(dates), len(SESSION)
    bid = np.zeros((n, m))
    ask = np.zeros((n, m))
    dead = np.zeros((n, m), dtype=bool)
    for cp, col in (("C", "K_c"), ("P", "K_p")):
        for j, clock in enumerate(SESSION):
            key = pd.MultiIndex.from_arrays(
                [
                    pd.DatetimeIndex(dates),
                    np.repeat(clock, n),
                    ent[col].to_numpy(float),
                    np.repeat(cp, n),
                ]
            )
            leg = got.reindex(key)
            lb = leg["bid"].to_numpy(float)
            la = leg["ask"].to_numpy(float)
            # the vendor's no-quote sentinel is bid == ask == 0 (asl.quote_mid)
            dead[:, j] |= ~np.isfinite(lb) | ~np.isfinite(la) | ((lb == 0) & (la == 0))
            bid[:, j] += lb
            ask[:, j] += la
    print(
        f"chain quotes on the two 11:00 strikes: {int((~dead).all(axis=1).sum())} of "
        f"{n} days live at all {m} stamps; {int(dead.sum())} dead legs in total"
    )
    return bid, 0.5 * (bid + ask), ask


# ------------------------------------------------------------ the tape ------
def build_tape(
    panel: pd.DataFrame, ent: pd.DataFrame, dates: pd.DatetimeIndex
) -> dict[str, Any]:
    """The 11:00 book's own stamp grid, replayed day by day by the live engine."""
    by_date = {d: g for d, g in panel.groupby("date", sort=True)}
    n, m = len(dates), len(SESSION)
    spot = np.full((n, m), np.nan)
    sig = np.full((n, m), np.nan)
    hedge_ref = np.full(n, np.nan)
    hedge_hold = np.full(n, np.nan)
    settle = np.full(n, np.nan)
    mark = np.full(n, np.nan)
    hedge_clock: dict[str, np.ndarray] = {c: np.full(n, np.nan) for c in EXIT_CLOCKS}
    grid = panel.pivot_table(index="date", columns="hhmm", values="S", aggfunc="first")
    ivg = panel.pivot_table(
        index="date", columns="hhmm", values="iv_hourly_used", aggfunc="first"
    )
    spot[:] = grid.reindex(index=dates, columns=list(SESSION)).to_numpy(float)
    sig[:] = ivg.reindex(index=dates, columns=list(SESSION)).to_numpy(float)
    for i, day in enumerate(dates):
        rows = by_date[day]
        ex = replay_day(rows, exit_clock=CLOSE, iv_mode="vendor")
        hold = replay_day(rows, exit_clock=None, iv_mode="vendor")
        assert ex["ok"] and hold["ok"], (day, ex["reason"], hold["reason"])
        hedge_ref[i] = ex["hedge_pts"]
        hedge_hold[i] = hold["hedge_pts"]
        settle[i] = hold["exit_price"]
        mark[i] = ex["exit_price"]
        for clock in EXIT_CLOCKS:
            hedge_clock[clock][i] = replay_day(
                rows, exit_clock=clock, iv_mode="vendor"
            )["hedge_pts"]
    assert np.isfinite(spot).all(), "a session stamp is missing a spot"
    assert (sig > 0).all(), "a session stamp carries no implied volatility"
    return {
        "spot": spot,
        "sig": sig,
        "hedge_ref": hedge_ref,
        "hedge_hold": hedge_hold,
        "settle": settle,
        "mark": mark,
        "hedge_clock": hedge_clock,
        "Kc": ent["K_c"].to_numpy(float),
        "Kp": ent["K_p"].to_numpy(float),
        "entry_mid": ent["entry"].to_numpy(float),
        "entry_bid": ent["bid_entry"].to_numpy(float),
        "S_close": ent["S_close"].to_numpy(float),
    }


# ------------------------------------------------------------- the gate -----
def gate(
    tape: dict[str, Any], mid: np.ndarray, ask: np.ndarray, dates: pd.DatetimeIndex
) -> None:
    """Reproduce the published numbers; nothing new runs until this passes."""
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    hg = tape["hedge_ref"]
    r_x = pd.Series((-(ask[:, -1] - eb) + hg) / em, index=dates)
    r_mk = pd.Series((-(tape["mark"] - em) + hg) / em, index=dates)
    r_qm = pd.Series((-(mid[:, -1] - em) + hg) / em, index=dates)
    era = r_x.index >= ERA0
    got = {
        "n whole sample": (float(r_x.size), float(GATE_N_DAYS)),
        "exit crossed-quoted Sharpe_ann whole": (sharpe_ann(r_x), GATE_CROSSED_WHOLE),
        "exit crossed-quoted Sharpe_ann era": (
            sharpe_ann(r_x[era]),
            GATE_CROSSED_ERA,
        ),
        "exit model-mark mid Sharpe_ann whole": (
            round(sharpe_ann(r_mk), 6),
            GATE_MARK_MID_WHOLE,
        ),
    }
    print("\nGATE - published reference numbers")
    for label, (have, want) in got.items():
        assert abs(have - want) < GATE_TOL, (label, have, want)
        print(f"  {label:<38s} {have:>16.12f}  reference {want:.12f}  OK")
    print(
        f"  exit quoted-mid Sharpe_ann          {sharpe_ann(r_qm):>16.12f}  "
        f"era {sharpe_ann(r_qm[era]):.12f}  (context, not gated)"
    )
    print(f"  daily-0DTE era ({ERA0.date()} onward): {int(era.sum())} days")
    print("GATE PASSED")


# --------------------------------------------------- Part A: attribution ----
def attribute(
    tape: dict[str, Any], mid: np.ndarray, ask: np.ndarray, dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk the package's Black-76 value stamp by stamp and split the walk.

    Write ``V(s, S)`` for the Black-76 value of the 11:00 package.  At stamp
    ``j`` the book holds the vendor total volatility ``s_j = sigma_j sqrt(h_j)``
    and the delta ``D_j = V_S(s_j, S_j)`` it actually hedged.  Over the bar
    ``j -> j+1`` the value walk splits, exactly, into

        hedge_j   = D_j (S_{j+1} - S_j)                       the hedge leg
        convex_j  = V(s_j, S_{j+1}) - V(s_j, S_j) - hedge_j   the gamma bill
        theta_j   = V(sigma_j sqrt(h_{j+1}), S_{j+1}) - V(s_j, S_{j+1})
        volpath_j = V(s_{j+1}, S_{j+1}) - V(sigma_j sqrt(h_{j+1}), S_{j+1})

    and the four sum to ``V_{j+1} - V_j`` by construction, so the sum over the
    nine bars telescopes to ``V_9 - V_0``.  The terminal volatility ``s_9`` is
    re-inverted from the 15:30 quoted midpoint of the 11:00 strikes (0 when
    that midpoint is at or below forward intrinsic, i.e. the quote carries no
    time value), so the walk ends on the price the book actually pays.

    The short's day is then

        P = theta_earned + convexity_paid + vega + residual

    with ``theta_earned = -sum theta``, ``convexity_paid = -sum convex``,
    ``vega = -[V(s_9, S_9) - V(s_0 sqrt(h_9/h_0), S_9)]`` -- the endpoint
    volatility move at the 15:30 spot -- and a residual that is the volatility
    path beyond its endpoint plus the two quote-versus-mark basis terms.
    """
    spot = tape["spot"]
    sig = tape["sig"]
    kc = tape["Kc"]
    kp = tape["Kp"]
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    n = len(dates)
    nb = len(SESSION) - 1

    stot = sig * np.sqrt(np.asarray(H_REM)[None, :])
    s_end = np.zeros(n)
    n_floor = 0
    for i in range(n):
        s = invert_total_vol(spot[i, -1], kc[i], kp[i], mid[i, -1])
        if not np.isfinite(s):
            s = 0.0
            n_floor += 1
        s_end[i] = s

    bars: list[dict[str, Any]] = []
    hedge_t = np.zeros((n, nb))
    convex_t = np.zeros((n, nb))
    theta_t = np.zeros((n, nb))
    volp_t = np.zeros((n, nb))
    delta_t = np.zeros((n, nb))
    v0 = np.zeros(n)
    v9 = np.zeros(n)
    vega_ref = np.zeros(n)
    for i in range(n):
        v0[i] = package_price(stot[i, 0], spot[i, 0], kc[i], kp[i])
        v9[i] = package_price(s_end[i], spot[i, -1], kc[i], kp[i])
        vega_ref[i] = package_price(
            stot[i, 0] * sqrt(H_REM[-1] / H_REM[0]), spot[i, -1], kc[i], kp[i]
        )
        for j in range(nb):
            s_j = stot[i, j]
            s_decay = sig[i, j] * sqrt(H_REM[j + 1])
            s_nxt = stot[i, j + 1] if j + 1 < nb else s_end[i]
            d_j = package_delta(s_j, spot[i, j], kc[i], kp[i])
            d_s = spot[i, j + 1] - spot[i, j]
            v_now = package_price(s_j, spot[i, j], kc[i], kp[i])
            v_move = package_price(s_j, spot[i, j + 1], kc[i], kp[i])
            v_dec = package_price(s_decay, spot[i, j + 1], kc[i], kp[i])
            v_nxt = package_price(s_nxt, spot[i, j + 1], kc[i], kp[i])
            delta_t[i, j] = d_j
            hedge_t[i, j] = d_j * d_s
            convex_t[i, j] = v_move - v_now - d_j * d_s
            theta_t[i, j] = v_dec - v_move
            volp_t[i, j] = v_nxt - v_dec

    walk = hedge_t.sum(1) + convex_t.sum(1) + theta_t.sum(1) + volp_t.sum(1)
    assert np.max(np.abs(walk - (v9 - v0))) < 1e-9, "the value walk does not telescope"
    assert np.max(np.abs(hedge_t.sum(1) - tape["hedge_ref"])) < 1e-12, "hedge mismatch"

    theta_earned = -theta_t.sum(1)
    convex_paid = -convex_t.sum(1)
    vega = -(v9 - vega_ref)
    volpath_resid = -volp_t.sum(1) - vega
    intrinsic = np.maximum(spot[:, -1] - kc, 0.0) + np.maximum(kp - spot[:, -1], 0.0)

    day = pd.DataFrame(index=pd.DatetimeIndex(dates))
    day["entry_mid"] = em
    day["entry_bid"] = eb
    day["S_entry"] = spot[:, 0]
    day["S_1530"] = spot[:, -1]
    day["move_pts"] = spot[:, -1] - spot[:, 0]
    day["move_pct"] = spot[:, -1] / spot[:, 0] - 1.0
    day["quoted_mid_1530"] = mid[:, -1]
    day["quoted_ask_1530"] = ask[:, -1]
    day["intrinsic_1530"] = intrinsic
    day["tv_mid_1530"] = mid[:, -1] - intrinsic
    day["tv_ask_1530"] = ask[:, -1] - intrinsic
    day["s_1100"] = stot[:, 0]
    day["s_1530_quoted"] = s_end
    day["vol_ratio"] = s_end / (stot[:, 0] * sqrt(H_REM[-1] / H_REM[0]))
    day["hedge_prem"] = tape["hedge_ref"] / em
    day["theta_prem"] = theta_earned / em
    day["convexity_prem"] = convex_paid / em
    day["vega_prem"] = vega / em
    day["volpath_resid_prem"] = volpath_resid / em
    bar_ret = np.divide(spot[:, 1:] - spot[:, :-1], spot[:, :-1])
    day["trend_ratio"] = np.abs(bar_ret.sum(1)) / np.abs(bar_ret).sum(1)

    for book, px, ref in (("mid", mid[:, -1], em), ("crossed", ask[:, -1], eb)):
        pnl = -(px - ref) + tape["hedge_ref"]
        basis_entry = ref - v0
        basis_exit = -(px - v9)
        resid = volpath_resid + basis_entry + basis_exit
        chk = theta_earned + convex_paid + vega + resid
        assert np.max(np.abs((pnl - chk) / em)) < SUM_TOL, f"{book} identity"
        day[f"opt_leg_{book}_prem"] = (-(px - ref)) / em
        day[f"basis_entry_{book}_prem"] = basis_entry / em
        day[f"basis_exit_{book}_prem"] = basis_exit / em
        day[f"residual_{book}_prem"] = resid / em
        day[f"r_{book}"] = pnl / em

    bar_pnl = -(convex_t + theta_t + volp_t)
    for i in range(n):
        for j in range(nb):
            bars.append(
                {
                    "date": dates[i].date(),
                    "clock": SESSION[j],
                    "interval": f"{SESSION[j]}-{SESSION[j + 1]}",
                    "S": spot[i, j],
                    "S_next": spot[i, j + 1],
                    "dS": spot[i, j + 1] - spot[i, j],
                    "bar_ret": bar_ret[i, j],
                    "h_rem": H_REM[j],
                    "total_vol": stot[i, j],
                    "delta": delta_t[i, j],
                    "hedge_pts": hedge_t[i, j],
                    "convexity_pts": -convex_t[i, j],
                    "theta_pts": -theta_t[i, j],
                    "volpath_pts": -volp_t[i, j],
                    "bar_pnl_pts": bar_pnl[i, j],
                    "entry_mid": em[i],
                    "hedge_prem": hedge_t[i, j] / em[i],
                    "convexity_prem": -convex_t[i, j] / em[i],
                    "theta_prem": -theta_t[i, j] / em[i],
                    "volpath_prem": -volp_t[i, j] / em[i],
                    "bar_pnl_prem": bar_pnl[i, j] / em[i],
                    "day_theta_prem": theta_earned[i] / em[i],
                    "day_convexity_prem": convex_paid[i] / em[i],
                    "day_vega_prem": vega[i] / em[i],
                    "day_opt_leg_crossed_prem": day["opt_leg_crossed_prem"].iloc[i],
                    "day_residual_crossed_prem": day["residual_crossed_prem"].iloc[i],
                    "day_trend_ratio": day["trend_ratio"].iloc[i],
                    "day_vol_ratio": day["vol_ratio"].iloc[i],
                    "r_mid": day["r_mid"].iloc[i],
                    "r_crossed": day["r_crossed"].iloc[i],
                }
            )
    attribution = pd.DataFrame(bars)
    print(
        f"\nattribution built: {n} days x {nb} bars; the value walk telescopes to 1e-9 "
        f"and the per-day identity holds to {SUM_TOL:.0e} of the entry premium; "
        f"{n_floor} days whose 15:30 quoted midpoint is at or below forward "
        f"intrinsic carry a terminal total volatility of 0"
    )
    return attribution, day


def part_a(attribution: pd.DataFrame, day: pd.DataFrame) -> list[dict[str, Any]]:
    """A1 concentration, A2 clock, A3 shape, A4 the buy-back volatility."""
    rows: list[dict[str, Any]] = []
    att = attribution.copy()
    att["date"] = pd.to_datetime(att["date"])
    era_bar = att["date"] >= ERA0

    # ---- A0 the day's P&L, averaged ---------------------------------------
    for book in ("crossed", "mid"):
        for sample, cut in (("whole", pd.Timestamp.min), ("daily_era", ERA0)):
            d = day[day.index >= cut]
            for name, ser in (
                ("day return", d[f"r_{book}"]),
                ("option leg (entry premium - buy-back)", d[f"opt_leg_{book}_prem"]),
                ("hedge leg (sum delta dS)", d["hedge_prem"]),
                ("= theta earned", d["theta_prem"]),
                ("+ convexity paid (the gamma bill)", d["convexity_prem"]),
                ("+ vega at the buy-back", d["vega_prem"]),
                ("+ residual", d[f"residual_{book}_prem"]),
                ("   of which volatility path", d["volpath_resid_prem"]),
                ("   of which entry basis", d[f"basis_entry_{book}_prem"]),
                ("   of which exit basis", d[f"basis_exit_{book}_prem"]),
                ("buy-back, intrinsic at 15:30", d["intrinsic_1530"] / d["entry_mid"]),
                (
                    "buy-back, time value at 15:30",
                    (d["tv_ask_1530"] if book == "crossed" else d["tv_mid_1530"])
                    / d["entry_mid"],
                ),
            ):
                rows.append(
                    {
                        "block": "A0_day_means",
                        "metric": f"{book} book, {name}",
                        "sample": sample,
                        "n": int(ser.size),
                        "mean": float(ser.mean()),
                        "median": float(ser.median()),
                        "sum_prem": float(ser.sum()),
                    }
                )

    # ---- A1 loss concentration -------------------------------------------
    for sample, msk in (
        ("whole", pd.Series(True, index=att.index)),
        ("daily_era", era_bar),
    ):
        sub = att.loc[msk]
        tot = float(sub["convexity_prem"].sum())
        for cut in BAR_MOVE_GRID:
            big = sub.loc[sub["bar_ret"].abs() > cut]
            rows.append(
                {
                    "block": "A1_convexity_concentration",
                    "metric": f"|bar return| > {cut:.2%}",
                    "sample": sample,
                    "n_bars": int(big.shape[0]),
                    "share_of_bars": big.shape[0] / sub.shape[0],
                    "sum_prem": float(big["convexity_prem"].sum()),
                    "share": float(big["convexity_prem"].sum()) / tot,
                }
            )
        rows.append(
            {
                "block": "A1_convexity_concentration",
                "metric": "all bars",
                "sample": sample,
                "n_bars": int(sub.shape[0]),
                "share_of_bars": 1.0,
                "sum_prem": tot,
                "share": 1.0,
            }
        )
    worst_bar = att.groupby("date")["bar_pnl_prem"].min()
    for book in ("crossed", "mid"):
        for sample, cut in (("whole", pd.Timestamp.min), ("daily_era", ERA0)):
            r = day[f"r_{book}"]
            r = r[r.index >= cut]
            lose = r[r < 0]
            wb = worst_bar.reindex(lose.index)
            rows.append(
                {
                    "block": "A1_worst_bar_share",
                    "metric": f"{book} book, losing days",
                    "sample": sample,
                    "n": int(lose.size),
                    "share_of_bars": float(lose.size) / float(r.size),
                    "sum_prem": float(lose.sum()),
                    "share": float(wb.sum()) / float(lose.sum()),
                }
            )

    # ---- A2 by clock ------------------------------------------------------
    for sample, msk in (
        ("whole", pd.Series(True, index=att.index)),
        ("daily_era", era_bar),
    ):
        sub = att.loc[msk]
        for clock, grp in sub.groupby("clock"):
            rows.append(
                {
                    "block": "A2_by_clock",
                    "metric": f"{clock}-{SESSION[SESSION.index(clock) + 1]}",
                    "sample": sample,
                    "n_bars": int(grp.shape[0]),
                    "convexity_mean": float(grp["convexity_prem"].mean()),
                    "convexity_sum": float(grp["convexity_prem"].sum()),
                    "hedge_mean": float(grp["hedge_prem"].mean()),
                    "hedge_sum": float(grp["hedge_prem"].sum()),
                    "theta_mean": float(grp["theta_prem"].mean()),
                    "bar_pnl_mean": float(grp["bar_pnl_prem"].mean()),
                    "mean_abs_bar_ret": float(grp["bar_ret"].abs().mean()),
                }
            )
    late = att.loc[att["clock"].isin(LATE_BARS)].groupby("date")["bar_pnl_prem"].sum()
    for book in ("crossed", "mid"):
        for sample, cut in (("whole", pd.Timestamp.min), ("daily_era", ERA0)):
            r = day[f"r_{book}"]
            r = r[r.index >= cut]
            lose = r[r < 0]
            rows.append(
                {
                    "block": "A2_late_share",
                    "metric": f"{book} book, 14:00-15:30 share of losing-day losses",
                    "sample": sample,
                    "n": int(lose.size),
                    "share": float(late.reindex(lose.index).sum()) / float(lose.sum()),
                }
            )

    # ---- A3 trend vs whipsaw ---------------------------------------------
    q1, q2 = day["trend_ratio"].quantile([1 / 3, 2 / 3]).to_numpy()
    lab = pd.Series("2_mixed", index=day.index)
    lab[day["trend_ratio"] <= q1] = "1_whipsaw"
    lab[day["trend_ratio"] > q2] = "3_trend"
    rows.append(
        {
            "block": "A3_shape_terciles",
            "metric": "full-sample tercile cut points of |sum r| / sum |r|",
            "sample": "whole",
            "tercile_lo": float(q1),
            "tercile_hi": float(q2),
        }
    )
    for book in ("crossed", "mid"):
        for sample, cut in (("whole", pd.Timestamp.min), ("daily_era", ERA0)):
            for cls in ("1_whipsaw", "2_mixed", "3_trend"):
                r = day.loc[lab == cls, f"r_{book}"]
                r = r[r.index >= cut]
                rows.append(
                    {
                        "block": "A3_shape",
                        "metric": f"{book} book, {cls}",
                        "sample": sample,
                        "n": int(r.size),
                        "mean": float(r.mean()),
                        "worst": float(r.min()),
                        "Sharpe_ann": sharpe_ann(r),
                        "mean_convexity": float(
                            day.loc[r.index, "convexity_prem"].mean()
                        ),
                        "mean_theta": float(day.loc[r.index, "theta_prem"].mean()),
                    }
                )

    # ---- A4 the buy-back volatility ---------------------------------------
    for sample, cut in (("whole", pd.Timestamp.min), ("daily_era", ERA0)):
        d = day[day.index >= cut]
        vr = d["vol_ratio"]
        qs = vr.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95]).to_numpy()
        rows.append(
            {
                "block": "A4_vol_ratio",
                "metric": "s(15:30 quoted) / s(11:00) sqrt(0.5/5)",
                "sample": sample,
                "n": int(vr.size),
                "p05": float(qs[0]),
                "p25": float(qs[1]),
                "median": float(qs[2]),
                "p75": float(qs[3]),
                "p90": float(qs[4]),
                "p95": float(qs[5]),
                "mean": float(vr.mean()),
                "share_above_one": float((vr > 1.0).mean()),
                "share_at_zero": float((vr == 0.0).mean()),
            }
        )
        hi = d.loc[vr >= vr.quantile(TOP_DECILE)]
        rows.append(
            {
                "block": "A4_vega",
                "metric": f"mean vega term, top {1 - TOP_DECILE:.0%} of the ratio",
                "sample": sample,
                "n": int(hi.shape[0]),
                "mean": float(hi["vega_prem"].mean()),
                "median": float(hi["vol_ratio"].median()),
            }
        )
        rows.append(
            {
                "block": "A4_vega",
                "metric": "mean vega term, all days",
                "sample": sample,
                "n": int(d.shape[0]),
                "mean": float(d["vega_prem"].mean()),
                "median": float(d["vol_ratio"].median()),
            }
        )
        for book in ("crossed", "mid"):
            lose = d.loc[d[f"r_{book}"] < 0]
            rows.append(
                {
                    "block": "A4_vega",
                    "metric": f"{book} book, vega share of losing-day losses",
                    "sample": sample,
                    "n": int(lose.shape[0]),
                    "mean": float(lose["vega_prem"].mean()),
                    "share": float(lose["vega_prem"].sum())
                    / float(lose[f"r_{book}"].sum()),
                }
            )
    return rows


def worst_days(attribution: pd.DataFrame, day: pd.DataFrame) -> pd.DataFrame:
    """A5: the ten worst days of the crossed book, with their splits."""
    att = attribution.copy()
    att["date"] = pd.to_datetime(att["date"])
    pick = att.loc[att.groupby("date")["bar_pnl_prem"].idxmin()].set_index("date")
    bad = day["r_crossed"].nsmallest(N_WORST).index
    out = pd.DataFrame(
        {
            "date": [d.date() for d in bad],
            "r_crossed": day.loc[bad, "r_crossed"].to_numpy(),
            "r_mid": day.loc[bad, "r_mid"].to_numpy(),
            "move_pts": day.loc[bad, "move_pts"].to_numpy(),
            "move_pct": day.loc[bad, "move_pct"].to_numpy(),
            "worst_bar_prem": pick.loc[bad, "bar_pnl_prem"].to_numpy(),
            "worst_bar_clock": pick.loc[bad, "interval"].to_numpy(),
            "worst_bar_ret": pick.loc[bad, "bar_ret"].to_numpy(),
            "trend_ratio": day.loc[bad, "trend_ratio"].to_numpy(),
            "vol_ratio": day.loc[bad, "vol_ratio"].to_numpy(),
            "option_prem": day.loc[bad, "opt_leg_crossed_prem"].to_numpy(),
            "hedge_prem": day.loc[bad, "hedge_prem"].to_numpy(),
            "vega_prem": day.loc[bad, "vega_prem"].to_numpy(),
            "theta_prem": day.loc[bad, "theta_prem"].to_numpy(),
            "convexity_prem": day.loc[bad, "convexity_prem"].to_numpy(),
            "residual_prem": day.loc[bad, "residual_crossed_prem"].to_numpy(),
        }
    )
    return out


# ------------------------------------------------- Part B: exit ladder ------
def part_b(
    tape: dict[str, Any], mid: np.ndarray, ask: np.ndarray, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Buy the 11:00 straddle back at each exit clock; hedging stops there."""
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    ref = {
        "crossed": pd.Series(
            (-(ask[:, -1] - eb) + tape["hedge_ref"]) / em, index=dates
        ),
        "mid": pd.Series((-(mid[:, -1] - em) + tape["hedge_ref"]) / em, index=dates),
    }
    rows: list[dict[str, Any]] = []
    for clock in EXIT_CLOCKS:
        j = SESSION.index(clock)
        hg = tape["hedge_clock"][clock]
        held = 0.5 * j
        for book, px, entry_ref in (
            ("crossed", ask[:, j], eb),
            ("mid", mid[:, j], em),
        ):
            r = pd.Series((-(px - entry_ref) + hg) / em, index=dates)
            for row in both_samples(r, f"exit {clock}, {book}"):
                base = ref[book]
                base = base[
                    base.index
                    >= (ERA0 if row["sample"] == "daily_era" else pd.Timestamp.min)
                ]
                row["exit_clock"] = clock
                row["variant"] = book
                row["hours_held"] = held
                row["mean_per_hour"] = row["mean"] / held
                row["sat_away_premium"] = float(base.mean()) - row["mean"]
                rows.append(row)
    for book, entry_ref in (("crossed", eb), ("mid", em)):
        r = pd.Series(
            (-(tape["settle"] - entry_ref) + tape["hedge_hold"]) / em, index=dates
        )
        for row in both_samples(r, f"hold to settlement, {book}"):
            base = ref[book]
            base = base[
                base.index
                >= (ERA0 if row["sample"] == "daily_era" else pd.Timestamp.min)
            ]
            row["exit_clock"] = "16:00 settle"
            row["variant"] = book
            row["hours_held"] = 5.0
            row["mean_per_hour"] = row["mean"] / 5.0
            row["sat_away_premium"] = float(base.mean()) - row["mean"]
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------- Part C: delta band -----
def part_c(
    tape: dict[str, Any], ask: np.ndarray, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Rebalance only when the package delta has drifted more than a band."""
    spot = tape["spot"]
    sig = tape["sig"]
    kc = tape["Kc"]
    kp = tape["Kp"]
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    n = len(dates)
    nb = len(SESSION) - 1
    stot = sig * np.sqrt(np.asarray(H_REM)[None, :])
    dlt = np.zeros((n, nb))
    for i in range(n):
        for j in range(nb):
            dlt[i, j] = package_delta(stot[i, j], spot[i, j], kc[i], kp[i])
    d_spot = spot[:, 1:] - spot[:, :-1]

    rows: list[dict[str, Any]] = []
    for band in BAND_GRID:
        npos = np.zeros((n, nb))
        npos[:, 0] = dlt[:, 0]
        for j in range(1, nb):
            drift = np.abs(dlt[:, j] - npos[:, j - 1])
            npos[:, j] = np.where(drift > band, dlt[:, j], npos[:, j - 1])
        prev = np.zeros((n, nb + 1))
        prev[:, 1:] = npos
        traded = np.abs(np.concatenate([npos, np.zeros((n, 1))], axis=1) - prev)
        hedge = (npos * d_spot).sum(1)
        cost = (traded * spot * HEDGE_COST_BP * 1e-4).sum(1)
        gross = pd.Series((-(ask[:, -1] - eb) + hedge) / em, index=dates)
        net = pd.Series((-(ask[:, -1] - eb) + hedge - cost) / em, index=dates)
        turn = pd.Series(traded.sum(1), index=dates)
        for row in both_samples(gross, f"band {band:.2f}, no cost"):
            msk = turn.index >= (
                ERA0 if row["sample"] == "daily_era" else pd.Timestamp.min
            )
            row["band"] = band
            row["costed"] = False
            row["mean_turnover"] = float(turn[msk].mean())
            row["mean_cost_prem"] = 0.0
            rows.append(row)
        for row in both_samples(net, f"band {band:.2f}, {HEDGE_COST_BP} bp"):
            msk = turn.index >= (
                ERA0 if row["sample"] == "daily_era" else pd.Timestamp.min
            )
            row["band"] = band
            row["costed"] = True
            row["mean_turnover"] = float(turn[msk].mean())
            row["mean_cost_prem"] = float(
                (pd.Series(cost / em, index=dates))[msk].mean()
            )
            rows.append(row)
        if band == 0.0:
            assert np.max(np.abs(hedge - tape["hedge_ref"])) < 1e-12, (
                "band 0 is the book"
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, ent, dates = load_panel()
    tape = build_tape(panel, ent, dates)
    _, mid, ask = quote_grids(ent, dates)
    assert np.max(np.abs(mid[:, 0] - tape["entry_mid"])) < 1e-12, "11:00 mid mismatch"
    gate(tape, mid, ask, dates)

    attribution, day = attribute(tape, mid, ask, dates)
    write(
        attribution,
        "attribution.csv",
        "Part A  per day x interval attribution (index points and units of entry premium)",
        [
            "date",
            "interval",
            "dS",
            "bar_ret",
            "delta",
            "hedge_prem",
            "convexity_prem",
            "theta_prem",
            "volpath_prem",
            "bar_pnl_prem",
        ],
        head=18,
    )
    summary = pd.DataFrame(part_a(attribution, day))
    summary.to_csv(OUT / "anatomy_summary.csv", index=False)
    show(
        summary[
            summary["block"].isin(["A1_convexity_concentration", "A1_worst_bar_share"])
        ],
        "Part A1  where the convexity loss and the losing days' losses sit"
        "   [anatomy_summary.csv]",
        [
            "block",
            "metric",
            "sample",
            "n",
            "n_bars",
            "share_of_bars",
            "sum_prem",
            "share",
        ],
    )
    show(
        summary[summary["block"] == "A0_day_means"],
        "Part A  the day's P&L, averaged (units of the 11:00 midpoint entry premium)",
        ["metric", "sample", "n", "mean", "median", "sum_prem"],
    )
    show(
        summary[summary["block"] == "A2_by_clock"],
        "Part A2  the convexity term and the hedge leg by interval clock",
        [
            "metric",
            "sample",
            "n_bars",
            "convexity_mean",
            "convexity_sum",
            "hedge_mean",
            "hedge_sum",
            "theta_mean",
            "bar_pnl_mean",
            "mean_abs_bar_ret",
        ],
    )
    show(
        summary[summary["block"] == "A2_late_share"],
        "Part A2  the 14:00-15:30 share of the losing days' losses",
        ["metric", "sample", "n", "share"],
    )
    show(
        summary[summary["block"] == "A3_shape_terciles"],
        "Part A3  the trend / whipsaw terciles",
        ["metric", "tercile_lo", "tercile_hi"],
    )
    show(
        summary[summary["block"] == "A3_shape"],
        "Part A3  the day's shape: |sum of bar returns| / sum |bar returns|",
        [
            "metric",
            "sample",
            "n",
            "mean",
            "worst",
            "Sharpe_ann",
            "mean_convexity",
            "mean_theta",
        ],
    )
    show(
        summary[summary["block"] == "A4_vol_ratio"],
        "Part A4  the 15:30 total volatility against the 11:00 total volatility",
        [
            "metric",
            "sample",
            "n",
            "p05",
            "p25",
            "median",
            "p75",
            "p90",
            "p95",
            "mean",
            "share_above_one",
            "share_at_zero",
        ],
    )
    show(
        summary[summary["block"] == "A4_vega"],
        "Part A4  the vega term at the buy-back",
        ["metric", "sample", "n", "mean", "median", "share"],
    )
    print(f"\n(the full A1-A4 ledger is anatomy_summary.csv, {len(summary)} rows)")

    write(
        worst_days(attribution, day),
        "worst10.csv",
        "Part A5  the ten worst days of the crossed book",
    )

    ladder = part_b(tape, mid, ask, dates)
    write(
        ladder,
        "exit_ladder.csv",
        "Part B  the exit-clock ladder",
        [
            "exit_clock",
            "variant",
            "sample",
            "n",
            "mean",
            "sd",
            "Sharpe_ann",
            "t",
            "MaxDD",
            "worst_day",
            "worst_date",
            "hours_held",
            "mean_per_hour",
            "sat_away_premium",
        ],
    )
    era_x = ladder[(ladder["variant"] == "crossed") & (ladder["sample"] == "daily_era")]
    era_x = era_x[era_x["exit_clock"] != "16:00 settle"]
    best = era_x.loc[era_x["Sharpe_ann"].idxmax()]
    ref = float(era_x.loc[era_x["exit_clock"] == CLOSE, "Sharpe_ann"].iloc[0])
    gain = float(best["Sharpe_ann"]) - ref
    settle_x = ladder[
        (ladder["variant"] == "crossed")
        & (ladder["sample"] == "daily_era")
        & (ladder["exit_clock"] == "16:00 settle")
    ]
    verdict = (
        "PASS (> 0.2)"
        if gain > 0.2
        else "WASH (<= 0.2): no earlier exit clock beats the book of record"
    )
    print(
        f"\nPart B pass bar: of the six exit clocks the era crossed Sharpe_ann is "
        f"maximised at {best['exit_clock']} ({best['Sharpe_ann']:.3f}); the book of "
        f"record exits at {CLOSE} ({ref:.3f}), so sitting the last stretch away is "
        f"worth {gain:+.3f} -> {verdict}"
    )
    print(
        f"  (reference, not an exit clock: hold to settlement "
        f"{float(settle_x['Sharpe_ann'].iloc[0]):.3f})"
    )

    band = part_c(tape, ask, dates)
    write(
        band,
        "band.csv",
        "Part C  hedge-to-a-delta band on the 30-minute tape, exit book crossed-quoted",
        [
            "band",
            "costed",
            "sample",
            "n",
            "mean",
            "sd",
            "Sharpe_ann",
            "t",
            "MaxDD",
            "worst_day",
            "mean_turnover",
            "mean_cost_prem",
        ],
    )
    print(
        "\nPart C caveat: the tape is a 30-minute grid, so a band can only be tested "
        "coarsely - a drift that crossed the band between two stamps and came back is "
        "invisible here, and every rebalance is forced onto a stamp. The real test "
        "needs minute data."
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
