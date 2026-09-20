"""48 - the delta the hedge is taken with: per-leg vol, vendor delta, skew, charm.

Forecast-free.  Every variant here changes ONE thing: the number that goes
into the Black-76 delta of the held 0DTE straddle at each 30-minute rebalance.
The legs, the entry fill, the exit fill, the rebalance clock and the terminal
treatment are the books of record's own, untouched.

The two references
------------------
D0  the book.  On the 11:00 exit book that is the package total volatility the
    book of record carries at the stamp (``iv_hourly_used`` x sqrt(h_rem): the
    mean of the two quoted legs' vendor implied volatilities at the stamp's own
    nearest-OTM straddle, re-inverted from the package midpoint on the bars
    that sat on the vendor's solver node).  On the 43-tape books it is the
    total volatility bisected out of that stamp's own nearest-OTM package
    midpoint.  In BOTH books the volatility is the stamp's at-the-money
    package volatility, applied to the legs that were picked at entry.
D9  V9, proposal 36's per-clock bias-corrected implied: D0's VARIANCE times the
    expanding, one-session-lagged mean of realized over implied remaining-window
    variance at that clock (63-session warm-up, fallback to D0).  This is the
    live default (``live.ibkr.pricing.corrected_total_vol``), and the bar every
    verdict is read against.

The ten causal variants (each read against D9 on two books = 20 gate reads)
---------------------------------------------------------------------------
D0   the book itself.
D1   per-leg volatility: the call at K_c and the put at K_p are each bisected
     out of their OWN midpoint, and the package delta is the sum of the two
     legs' Black-76 deltas at their own volatilities.  At a COMMON volatility
     that sum is the package delta to 1e-12, and the script asserts it.
D1v  the vendor's own per-leg ``delta`` column from the chain, summed.  Sign
     convention checked against D1 and stated.
D2   skew-adjusted (sticky-delta): D1 plus sum over legs of vega_leg x
     dsigma/dS, with the smile slope b = d(total vol)/d ln(K/S) from a linear
     fit of the vendor's per-strike total volatility on ln(K/S) over the five
     nearest listed strikes each side of the spot (OTM leg at each strike,
     vendor solver nodes dropped).  Sticky-delta means sigma is a function of
     ln(K/S) alone, so dsigma/dS|_K = -b/S, and with vega_leg = S phi(d1_leg)
     the correction is exactly -b (phi(d1c) + phi(d1p)).
D3   charm-anticipating: D0's hourly volatility carried to a time-to-close
     reduced by half a rebalance interval (15 minutes) - the delta expected
     mid-interval if the spot does not move - i.e. s' = s0 sqrt((h-0.25)/h).
     The full-interval version (h - 0.5) is the control; at the hold book's
     15:30 stamp it reduces the horizon to zero and the hedge to zero, and the
     script counts those cells.
D4   the combinations: D9+D1 (each leg's variance times the per-clock factor),
     D9+D3, D9+D1+D3, D9+D2+D3.

Reference only, labelled ALPHA and excluded from the gate:
D5   the last held interval's hedge scaled by 1.25 and by 0.75 (15:00-15:30 on
     the exit book, 15:30-16:00 on the hold books), on D0's delta.

The books and the samples
-------------------------
(a) 1100_exit_crossed - proposal 32's book: sell the 11:00 nearest-OTM
    straddle at the quoted bid, hedge every 30 minutes, buy both legs back at
    15:30 at the quoted ask.  865 deck expirations, whole and daily-0DTE era
    (2022-05-16 onward).  This is proposal 36's own design.
(b) 1330_hold - proposal 43's 13:30 tape on all 1279 sessions 2020-01-03 ..
    2025-12-31: deck in-sample (866), UNSEEN (413, 2024-05-01 .. 2025-12-31)
    and all.  Per contract (index points) is the primary unit.
(c) 1100_hold - the same tape at 11:00, for continuity on the unseen sessions.

NONE of these variants reads a forecast: every input is a quote or a spot at
or before the stamp, and the only estimated object (V9's per-clock factor) is
an expanding lagged mean of strictly earlier sessions.  The 413 sessions after
2024-04-30 were never used to build, tune or select anything in this deck, so
the unseen sample is a genuine out-of-sample test of the hedge delta.

The gate
--------
"improves" only where the variant's Sharpe-difference bootstrap interval
against D9 excludes zero on the POSITIVE side on BOTH the 11:00 era exit book
AND the 13:30 hold book on the unseen sessions, per contract.  10 causal
variants x 2 books = 20 gate reads; at the 5% level the expectation under the
null is 1.0 cells.

Reproduced before anything new is computed:
  G1  proposal 32's book: 865 expirations, crossed Sharpe 2.252546754993
      whole / 2.513132263902 era, and the hedge leg against
      live.ibkr.parity.replay_day to 1e-12.
  G2  proposal 32's shape terciles 0.2076 / 0.4782.
  G3  proposal 36's V9 on the primary book, to 1e-12: era dSharpe +0.175552
      [+0.063289, +0.300511], HAC t 2.540922; whole +0.130957
      [+0.052670, +0.219362], HAC t 2.739032.
  G4  the vectorised inversion equals live.ibkr.pricing.invert_total_vol on
      every cell of the 43 tape, exactly.
  G5  the 43 tape rebuilt here reproduces proposal 43's published daily file
      column by column to 1e-12 at 11:00 and at 13:30, and its 13:30 hold book
      on the 413 unseen sessions: +1.173260 index points per contract per day,
      HAC t 3.210432.
  G6  the per-leg deltas at a common volatility equal the package delta to
      1e-12.

Run:  python writeup/intraday_proposals/48_delta_refinements.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import atm_straddle_lib as asl  # noqa: E402
from live.ibkr.pricing import (  # noqa: E402
    INVERT_VOL_HI,
    INVERT_VOL_ITERS,
    INVERT_VOL_LO,
    INVERT_VOL_TOL,
    invert_total_vol,
)

HERE = Path(__file__).resolve().parent
HOLD = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "48"
REF43 = HOLD / "proposals" / "43"
CHAIN = ROOT / "data" / "spxw_chain.parquet"
SPOT = ROOT / "data" / "spxw_spot.parquet"
GSPC = HOLD / "cache" / "gspc_ohlc.parquet"

#: the 30-minute stamps the all-clock tape walks (16:00 is a settlement print).
CLOCKS: tuple[str, ...] = (
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
CLOSE = "15:30"
#: hours from each clock to the 16:00 settlement (live.ibkr.pricing.hours_to_close).
H_OF: dict[str, float] = {c: 16.0 - (int(c[:2]) + int(c[3:]) / 60.0) for c in CLOCKS}
#: the 11:00 book's own stamps (proposal 32's SESSION).
SESSION_1100: tuple[str, ...] = CLOCKS[2:]

ERA0 = pd.Timestamp("2022-05-16")
DECK_END = pd.Timestamp("2024-04-30")
OOS_START = pd.Timestamp("2024-05-01")
OOS_END = pd.Timestamp("2025-12-31")
ANN = float(np.sqrt(252.0))

#: the repo's expanding warm-up, 63 sessions.
WARMUP = 63
#: the repo's hedge turnover charge, basis points of S x |delta traded|.
HEDGE_COST_BP = 0.5
#: half a rebalance interval and a whole one, in hours (D3).
CHARM_HALF = 0.25
CHARM_FULL = 0.5
#: D5's ALPHA lean on the last held interval.
LEAN_GRID: tuple[float, ...] = (1.25, 0.75)
#: the smile fit: this many listed strikes each side of the spot, and the
#: minimum on each side below which the stamp gets no slope.
SMILE_EACH_SIDE = 5
SMILE_MIN_SIDE = 3
#: the strike band retained from the chain, around each session's spot range.
STRIKE_PAD = 0.02

#: the deck's circular block bootstrap: 2000 draws, 21-day blocks, seed 0.
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
CI_PCT: tuple[float, float] = (2.5, 97.5)

#: published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = 865
GATE_CROSSED_WHOLE = 2.252546754993
GATE_CROSSED_ERA = 2.513132263902
GATE_TOL = 1e-6
GATE_TREND_LO = 0.2076
GATE_TREND_HI = 0.4782
TREND_TOL = 5e-5
GATE_V9_DSHARPE_WHOLE = 0.1309569244741494
GATE_V9_CI_WHOLE: tuple[float, float] = (0.05266980309665783, 0.21936214098613388)
GATE_V9_DSHARPE_ERA = 0.17555194627816872
GATE_V9_CI_ERA: tuple[float, float] = (0.06328949647804978, 0.3005113715778494)
GATE_V9_HAC_T_WHOLE = 2.7390318414673698
GATE_V9_HAC_T_ERA = 2.540921506042825
#: proposal 43's tape.
GATE_N_SESSIONS = 1279
GATE_N_DECK = 866
GATE_N_OOS = 413
GATE_1330_OOS_PTS = 1.173260
GATE_1330_OOS_T = 3.210432
GATE_PTS_TOL = 5e-6
GATE_T_TOL = 5e-6
#: float-path bars.
EXACT_TOL = 1e-12

#: the variants, in the order every table prints them.
REF_BOOK = "D0_book"
REF_V9 = "D9_implied_bc"
GATE_VARIANTS: tuple[str, ...] = (
    "D0_book",
    "D1_leg_vol",
    "D1v_vendor_delta",
    "D2_skew_sticky_delta",
    "D3_charm_half",
    "D3_charm_full",
    "D4_D9_D1",
    "D4_D9_D3",
    "D4_D9_D1_D3",
    "D4_D9_D2_D3",
)
ALPHA_VARIANTS: tuple[str, ...] = tuple(f"D5_lastbar_{x:.2f}" for x in LEAN_GRID)
VARIANTS: tuple[str, ...] = (REF_V9, *GATE_VARIANTS, *ALPHA_VARIANTS)

VARIANT_NOTE: dict[str, str] = {
    "D0_book": "the book: the stamp's package total volatility",
    "D9_implied_bc": (
        "V9: D0 variance x the expanding lagged per-clock realized-over-implied "
        "mean (63-session warm-up)"
    ),
    "D1_leg_vol": (
        "each leg bisected out of its own midpoint; delta = call delta + put "
        "delta at their own volatilities"
    ),
    "D1v_vendor_delta": "the chain's own per-leg delta column, summed",
    "D2_skew_sticky_delta": (
        "D1 - b (phi(d1c) + phi(d1p)), b the local smile slope d(total vol)/d ln(K/S)"
    ),
    "D3_charm_half": "D0's hourly volatility at a horizon 15 minutes shorter",
    "D3_charm_full": (
        "D0's hourly volatility at a horizon 30 minutes shorter (control)"
    ),
    "D4_D9_D1": "per-leg volatilities, each variance times V9's per-clock factor",
    "D4_D9_D3": "V9's volatility at a horizon 15 minutes shorter",
    "D4_D9_D1_D3": "per-leg V9 volatilities at a horizon 15 minutes shorter",
    "D4_D9_D2_D3": (
        "D4_D9_D1_D3 plus the sticky-delta skew term at those volatilities"
    ),
    "D5_lastbar_1.25": (
        "ALPHA, not a hedge: D0 with the last held interval's delta scaled by 1.25"
    ),
    "D5_lastbar_0.75": (
        "ALPHA, not a hedge: D0 with the last held interval's delta scaled by 0.75"
    ),
}

BOOK_NAMES: tuple[str, ...] = ("1100_exit_crossed", "1330_hold", "1100_hold")
#: the two books the gate is read on, and the sample it is read on in each.
GATE_CELLS: tuple[tuple[str, str], ...] = (
    ("1100_exit_crossed", "daily_era"),
    ("1330_hold", "unseen"),
)
#: the unit the gate is read in.
GATE_UNIT = "pts"
UNITS: tuple[str, ...] = ("prem", "pts")
UNIT_NAME: dict[str, str] = {
    "prem": "premium units",
    "pts": "index points per contract",
}


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 260, "display.max_columns", 90):
        print(use.to_string(index=False))


def write(df: pd.DataFrame, name: str, title: str) -> None:
    """Persist a table under the proposal's own directory."""
    df.to_csv(OUT / name, index=False)
    print(f"\nwrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


def load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- pricing ----
_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def ncdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, as live.ibkr.pricing writes it."""
    return 0.5 * (1.0 + erf(np.asarray(z, float) / _SQRT2))


def npdf(z: np.ndarray) -> np.ndarray:
    """Standard normal density."""
    zz = np.asarray(z, float)
    return np.exp(-0.5 * zz * zz) / _SQRT2PI


def _d1(s: np.ndarray, f: np.ndarray, k: np.ndarray) -> np.ndarray:
    return (np.log(f / k) + 0.5 * s * s) / s


def pkg_price_vec(
    s: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """Black-76 package price (r = 0, forward = spot) for a strictly positive s."""
    d1c = _d1(s, f, kc)
    d1p = _d1(s, f, kp)
    return f * ncdf(d1c) - kc * ncdf(d1c - s) + kp * ncdf(-(d1p - s)) - f * ncdf(-d1p)


def leg_price_vec(
    s: np.ndarray, f: np.ndarray, k: np.ndarray, is_call: np.ndarray
) -> np.ndarray:
    """Black-76 single-leg price (r = 0, forward = spot) for a positive s."""
    d1 = _d1(s, f, k)
    call = f * ncdf(d1) - k * ncdf(d1 - s)
    put = k * ncdf(-(d1 - s)) - f * ncdf(-d1)
    return np.where(is_call, call, put)


def pkg_delta_vec(
    s: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """Package delta N(d1c) + N(d1p) - 1; zero where s is not positive.

    The branch ``live.ibkr.pricing.package_delta`` takes: at the settlement
    there is nothing left to hedge.
    """
    shape = np.broadcast_shapes(np.shape(s), np.shape(f), np.shape(kc), np.shape(kp))
    ss = np.broadcast_to(np.asarray(s, float), shape)
    ff = np.broadcast_to(np.asarray(f, float), shape)
    kcc = np.broadcast_to(np.asarray(kc, float), shape)
    kpp = np.broadcast_to(np.asarray(kp, float), shape)
    ok = (
        np.isfinite(ss)
        & (ss > 0.0)
        & np.isfinite(ff)
        & (ff > 0.0)
        & np.isfinite(kcc)
        & (kcc > 0.0)
        & np.isfinite(kpp)
        & (kpp > 0.0)
    )
    s1 = np.where(ok, ss, 1.0)
    f1 = np.where(ok, ff, 1.0)
    kc1 = np.where(ok, kcc, 1.0)
    kp1 = np.where(ok, kpp, 1.0)
    return np.where(ok, ncdf(_d1(s1, f1, kc1)) + ncdf(_d1(s1, f1, kp1)) - 1.0, 0.0)


def leg_delta_vec(
    s: np.ndarray, f: np.ndarray, k: np.ndarray, is_call: bool
) -> np.ndarray:
    """Black-76 leg delta: N(d1) for a call, N(d1) - 1 for a put; 0 at s <= 0."""
    shape = np.broadcast_shapes(np.shape(s), np.shape(f), np.shape(k))
    ss = np.broadcast_to(np.asarray(s, float), shape)
    ff = np.broadcast_to(np.asarray(f, float), shape)
    kk = np.broadcast_to(np.asarray(k, float), shape)
    ok = (
        np.isfinite(ss)
        & (ss > 0.0)
        & np.isfinite(ff)
        & (ff > 0.0)
        & np.isfinite(kk)
        & (kk > 0.0)
    )
    s1 = np.where(ok, ss, 1.0)
    f1 = np.where(ok, ff, 1.0)
    k1 = np.where(ok, kk, 1.0)
    n = ncdf(_d1(s1, f1, k1))
    return np.where(ok, n - (0.0 if is_call else 1.0), 0.0)


def leg_vega_vec(s: np.ndarray, f: np.ndarray, k: np.ndarray) -> np.ndarray:
    """d(leg price)/d(total volatility) = F phi(d1); zero at s <= 0."""
    shape = np.broadcast_shapes(np.shape(s), np.shape(f), np.shape(k))
    ss = np.broadcast_to(np.asarray(s, float), shape)
    ff = np.broadcast_to(np.asarray(f, float), shape)
    kk = np.broadcast_to(np.asarray(k, float), shape)
    ok = (
        np.isfinite(ss)
        & (ss > 0.0)
        & np.isfinite(ff)
        & (ff > 0.0)
        & np.isfinite(kk)
        & (kk > 0.0)
    )
    s1 = np.where(ok, ss, 1.0)
    f1 = np.where(ok, ff, 1.0)
    k1 = np.where(ok, kk, 1.0)
    return np.where(ok, f1 * npdf(_d1(s1, f1, k1)), 0.0)


def _bisect(
    price: Any,
    f: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    mid: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """``live.ibkr.pricing.invert_total_vol``'s bracket and halving count, vectorised.

    Same bracket [1e-8, 1], same halving count, same early stop: the bracket
    width halves identically for every cell, so the whole array stops on the
    iteration the scalar engine stops on.
    """
    good = (
        np.isfinite(f)
        & (f > 0.0)
        & np.isfinite(mid)
        & np.isfinite(k1)
        & (k1 > 0.0)
        & np.isfinite(k2)
        & (k2 > 0.0)
        & (mid > intrinsic)
    )
    ff = np.where(good, f, 1.0)
    a = np.where(good, k1, 1.0)
    b = np.where(good, k2, 1.0)
    good = good & (price(np.full(ff.shape, INVERT_VOL_HI), ff, a, b) >= mid)
    lo = np.full(ff.shape, INVERT_VOL_LO)
    hi = np.full(ff.shape, INVERT_VOL_HI)
    for _ in range(INVERT_VOL_ITERS):
        m = 0.5 * (lo + hi)
        below = price(m, ff, a, b) < mid
        lo = np.where(below, m, lo)
        hi = np.where(below, hi, m)
        if float(np.max(hi - lo)) < INVERT_VOL_TOL:
            break
    return np.where(good, 0.5 * (lo + hi), np.nan)


def invert_package_vec(
    f: np.ndarray, kc: np.ndarray, kp: np.ndarray, mid: np.ndarray
) -> np.ndarray:
    """The package total volatility, array at a time (GATE 4 checks it)."""
    ff = np.asarray(f, float)
    kcc = np.asarray(kc, float)
    kpp = np.asarray(kp, float)
    intrinsic = np.maximum(ff - kcc, 0.0) + np.maximum(kpp - ff, 0.0)
    return _bisect(pkg_price_vec, ff, kcc, kpp, np.asarray(mid, float), intrinsic)


def invert_leg_vec(
    f: np.ndarray, k: np.ndarray, mid: np.ndarray, is_call: bool
) -> np.ndarray:
    """One leg's total volatility from its own midpoint, same bracket and count."""
    ff = np.asarray(f, float)
    kk = np.asarray(k, float)
    flag = np.broadcast_to(np.asarray(is_call, bool), ff.shape)

    def price(
        s: np.ndarray, a: np.ndarray, b: np.ndarray, _c: np.ndarray
    ) -> np.ndarray:
        return leg_price_vec(s, a, b, flag)

    intrinsic = np.maximum(ff - kk, 0.0) if is_call else np.maximum(kk - ff, 0.0)
    return _bisect(price, ff, kk, kk, np.asarray(mid, float), intrinsic)


# ------------------------------------------------------------ statistics ----
_BOOT_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    """The deck's circular block bootstrap index, drawn once per sample size.

    Seeded from ``BOOT_SEED`` alone, so a worker process draws exactly the
    index the parent would have drawn: every number here is worker-independent.
    """
    if n not in _BOOT_IDX:
        _BOOT_IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng(BOOT_SEED), n, BOOT_BLOCK, BOOT_B
        )
    return _BOOT_IDX[n]


def maxdd(v: np.ndarray) -> float:
    """Worst peak-to-trough of the cumulative SUM path, in the series' units."""
    x = np.asarray(v, float)
    x = x[np.isfinite(x)]
    if x.size < 1:
        return float("nan")
    path = np.cumsum(x)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def stat_row(x: np.ndarray, dates: np.ndarray) -> dict[str, Any]:
    """n / mean / sd / Sharpe_ann / t / HAC t / MaxDD / worst day of a series."""
    ok = np.isfinite(x)
    v = np.asarray(x, float)[ok]
    d = np.asarray(dates)[ok]
    n = int(v.size)
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n >= 2 else float("nan")
    alive = bool(np.isfinite(sd)) and sd > 0.0
    hac_t, lag = asl.newey_west_t(pd.Series(v))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if alive else float("nan"),
        "t": float(np.sqrt(n)) * mean / sd if alive else float("nan"),
        "hac_t": float(hac_t),
        "hac_lag": int(lag),
        "MaxDD": maxdd(v),
        "worst": float(v.min()) if n else float("nan"),
        "worst_date": str(pd.Timestamp(d[int(np.argmin(v))]).date()) if n else "",
    }


def paired_row(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """The paired daily difference x - y: mean, plain t, HAC t, days improved."""
    ok = np.isfinite(x) & np.isfinite(y)
    d = np.asarray(x, float)[ok] - np.asarray(y, float)[ok]
    n = int(d.size)
    sd = float(d.std(ddof=1)) if n >= 2 else float("nan")
    hac_t, lag = asl.newey_west_t(pd.Series(d))
    return {
        "n": n,
        "mean_diff": float(d.mean()) if n else float("nan"),
        "sd_diff": sd,
        "t_diff": float(np.sqrt(n)) * float(d.mean()) / sd
        if (n and np.isfinite(sd) and sd > 0.0)
        else float("nan"),
        "hac_t_diff": float(hac_t),
        "hac_lag": int(lag),
        "frac_days_improved": float((d > 0).mean()) if n else float("nan"),
        "frac_days_unchanged": float((d == 0).mean()) if n else float("nan"),
    }


def sharpe_diff_ci(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Circular block bootstrap of Sharpe(x) - Sharpe(y) on the common days."""
    ok = np.isfinite(x) & np.isfinite(y)
    a = np.asarray(x, float)[ok]
    b = np.asarray(y, float)[ok]
    n = int(a.size)
    if n < 2:
        return {
            "n": n,
            "dSharpe": float("nan"),
            "boot_lo": float("nan"),
            "boot_hi": float("nan"),
            "pct_draws_positive": float("nan"),
            "ci_excludes_zero": False,
        }
    hat = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * ANN
    idx = boot_idx(n)
    aa = a[idx]
    bb = b[idx]
    ds = (
        aa.mean(axis=1) / aa.std(axis=1, ddof=1)
        - bb.mean(axis=1) / bb.std(axis=1, ddof=1)
    ) * ANN
    lo, hi = np.percentile(ds, list(CI_PCT))
    return {
        "n": n,
        "dSharpe": float(hat),
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "pct_draws_positive": float(100.0 * (ds > 0).mean()),
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def expanding_lagged_mean(a: np.ndarray, min_obs: int) -> np.ndarray:
    """Column-wise expanding mean over strictly earlier rows, NaN before warm-up."""
    return (
        pd.DataFrame(a).expanding(min_periods=min_obs).mean().shift(1).to_numpy(float)
    )


# ------------------------------------------------------- the single read ----
def session_stamps() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(UTC stamp, spot) per (session date, clock) from data/spxw_spot.parquet."""
    spot = pd.read_parquet(SPOT)
    et = pd.to_datetime(spot["timestamp"], utc=True).dt.tz_convert("America/New_York")
    spot = spot.assign(
        date=et.dt.normalize().dt.tz_localize(None), hhmm=et.dt.strftime("%H:%M")
    )
    stamp = spot.pivot_table(
        index="date", columns="hhmm", values="timestamp", aggfunc="first"
    )
    px = spot.pivot_table(
        index="date", columns="hhmm", values="spot", aggfunc="first"
    ).astype(float)
    return stamp, px


def _leg_key(
    date_code: np.ndarray,
    clock_code: np.ndarray,
    strike: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """One int64 key per (session, clock, strike, call/put) - an exact lookup.

    Strikes are quoted on a 0.01 grid at worst, so ``round(100 K)`` is exact
    and the composite key is a bijection on the retained ladder.  A stamp the
    guards refused carries no strike; its key is -1, which matches nothing, so
    the leg simply has no quote there.
    """
    s = np.asarray(strike, float)
    kf = np.where(np.isfinite(s), np.rint(s * 100.0), -1.0)
    ok = (kf >= 0.0) & (kf < 10_000_000.0)
    k = np.where(ok, kf, 0.0).astype(np.int64)
    base = (
        np.asarray(date_code, np.int64) * len(CLOCKS) + np.asarray(clock_code, np.int64)
    ) * 2 + np.asarray(is_call, np.int64)
    return np.where(ok, base * 10_000_000 + k, np.int64(-1))


def read_chain_once(stamp: pd.DataFrame, px: pd.DataFrame) -> dict[str, Any]:
    """ONE pass over data/spxw_chain.parquet for every stamp and strike needed.

    Row group by row group, with the stamp list pushed down: a row survives
    only if it is 0DTE (its expiration is the Eastern date of its stamp), its
    stamp is one of the session's twelve 30-minute clocks, and its strike lies
    inside that SESSION's own band - ``[min spot x (1 - 0.02), max spot x
    (1 + 0.02)]`` over the session's twelve stamps.  The band is built from
    ``data/spxw_spot.parquet`` before the read, so it is known in advance; it
    holds every strike a book can pick at any entry clock and still quote at
    any later stamp, and at least the five listed strikes each side of every
    stamp's spot that the smile fit asks for.

    ``n_live`` - the guard ``asl.pick_nearest_otm_guarded`` reads - is counted
    on the FULL stamp before the band is applied, so the band never turns a
    healthy stamp into a vendor outage.
    """
    dates = pd.DatetimeIndex(stamp.index)
    date_code = {d: i for i, d in enumerate(dates)}
    clock_code = {c: j for j, c in enumerate(CLOCKS)}
    lo = px[list(CLOCKS)].min(axis=1).to_numpy(float) * (1.0 - STRIKE_PAD)
    hi = px[list(CLOCKS)].max(axis=1).to_numpy(float) * (1.0 + STRIKE_PAD)
    assert np.isfinite(lo).all() and np.isfinite(hi).all(), "a session has no spot"
    key = pd.DataFrame(
        {
            "timestamp": pd.concat(
                [stamp[c] for c in CLOCKS], ignore_index=True
            ).to_numpy(),
            "date": np.tile(dates.to_numpy(), len(CLOCKS)),
            "hhmm": np.repeat(np.asarray(CLOCKS, dtype=object), len(dates)),
            "k_lo": np.tile(lo, len(CLOCKS)),
            "k_hi": np.tile(hi, len(CLOCKS)),
        }
    )
    key["timestamp"] = pd.to_datetime(key["timestamp"], utc=True)
    assert key["timestamp"].notna().all(), (
        "a session stamp is missing from the spot file"
    )
    want = set(key["timestamp"].to_numpy())

    src = pq.ParquetFile(CHAIN)
    cols = [
        "expiration",
        "timestamp",
        "strike",
        "cp",
        "bid",
        "ask",
        "underlying_price",
        "impl_volatility",
        "delta",
        "hours_to_expiration",
    ]
    parts: list[pd.DataFrame] = []
    n_live_parts: list[pd.Series] = []
    n_rows = 0
    t0 = time.time()
    for i in range(src.num_row_groups):
        blk = src.read_row_group(i, columns=cols).to_pandas()
        n_rows += len(blk)
        et = blk["timestamp"].dt.tz_convert("America/New_York")
        day = et.dt.normalize().dt.tz_localize(None)
        keep = (blk["expiration"].to_numpy() == day.to_numpy()) & blk["timestamp"].isin(
            want
        ).to_numpy()
        sub = blk.loc[keep].copy()
        if sub.empty:
            continue
        sub["strike"] = sub["strike"].astype(float)
        sub["cp"] = sub["cp"].astype(str)
        sub["mid"] = asl.quote_mid(sub["bid"], sub["ask"]).to_numpy()
        alive = np.isfinite(sub["mid"].to_numpy(float)) & (
            sub["mid"].to_numpy(float) > 0.0
        )
        n_live_parts.append(sub.loc[alive].groupby("timestamp").size())
        sub = sub.merge(key, on="timestamp", how="inner")
        band = (sub["strike"].to_numpy(float) >= sub["k_lo"].to_numpy(float)) & (
            sub["strike"].to_numpy(float) <= sub["k_hi"].to_numpy(float)
        )
        parts.append(sub.loc[band].drop(columns=["k_lo", "k_hi"]))
    tab = pd.concat(parts, ignore_index=True)
    n_live = (
        pd.concat(n_live_parts).groupby(level=0).sum().rename("n_live").reset_index()
    )
    n_live = n_live.merge(
        key[["timestamp", "date", "hhmm"]], on="timestamp", how="left"
    )
    tab["date_code"] = tab["date"].map(date_code).astype(np.int64)
    tab["clock_code"] = tab["hhmm"].map(clock_code).astype(np.int64)
    tab["is_call"] = (tab["cp"].to_numpy() == "C").astype(np.int64)
    tab["S"] = tab["underlying_price"].astype(float)
    print(
        f"chain read once, row group by row group: {n_rows:,} rows scanned, "
        f"{len(tab):,} kept ({len(dates)} sessions x {len(CLOCKS)} clocks, 0DTE, "
        f"strikes inside each session's +-{STRIKE_PAD:.0%} spot band), "
        f"{time.time() - t0:.1f} s"
    )
    return {"tab": tab, "n_live": n_live, "dates": dates, "key": key}


def leg_table(tab: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The retained ladder as a sorted int64 key plus its columns, for lookup."""
    k = _leg_key(
        tab["date_code"].to_numpy(),
        tab["clock_code"].to_numpy(),
        tab["strike"].to_numpy(float),
        tab["is_call"].to_numpy(),
    )
    assert (k >= 0).all(), "a retained ladder row has no usable strike"
    order = np.argsort(k, kind="stable")
    k = k[order]
    assert not (np.diff(k) == 0).any(), "a (session, clock, strike, cp) is duplicated"
    cols = {
        name: tab[name].to_numpy(float)[order]
        for name in ("bid", "ask", "mid", "impl_volatility", "delta", "S")
    }
    return k, cols


def leg_lookup(
    keys: np.ndarray,
    cols: dict[str, np.ndarray],
    date_code: np.ndarray,
    clocks: tuple[str, ...],
    strike: np.ndarray,
    is_call: bool,
) -> dict[str, np.ndarray]:
    """Per (day, clock) quotes of ONE fixed leg, on the (day x clock) grid."""
    n = len(date_code)
    out = {name: np.full((n, len(clocks)), np.nan) for name in cols}
    flag = np.full(n, 1 if is_call else 0, dtype=np.int64)
    for j, c in enumerate(clocks):
        want = _leg_key(
            date_code, np.full(n, CLOCKS.index(c), dtype=np.int64), strike, flag
        )
        pos = np.searchsorted(keys, want)
        pos_ok = (pos < keys.size) & (pos >= 0)
        hit = np.zeros(n, dtype=bool)
        hit[pos_ok] = keys[pos[pos_ok]] == want[pos_ok]
        for name, v in cols.items():
            out[name][hit, j] = v[pos[hit]]
    return out


# --------------------------------------------------------- the smile fit ----
def smile_slopes(tab: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The local smile slope b = d(total vol) / d ln(K/S) at every stamp.

    At each stamp the ladder is read OTM - the call at every listed strike at
    or above the spot, the put at every listed strike below it - the vendor's
    per-strike implied volatility is censored of its solver's bracket nodes
    (``asl.censor_vendor_iv``) and put on the repo's total-volatility scale
    (``iv x sqrt(h_rem)``), and the five strikes nearest the spot on each side
    are kept.  ``sigma = a + b ln(K/S)`` is then fitted by ordinary least
    squares on those (at most ten) points; a stamp with fewer than three
    usable strikes on either side gets no slope and every variant that needs
    one falls back to its own base there.
    """
    v = tab.copy()
    v["iv_ok"] = asl.censor_vendor_iv(v["impl_volatility"]).to_numpy()
    otm = np.where(
        v["is_call"].to_numpy() == 1,
        v["strike"].to_numpy(float) >= v["S"].to_numpy(float),
        v["strike"].to_numpy(float) < v["S"].to_numpy(float),
    )
    live = (
        otm
        & np.isfinite(v["iv_ok"].to_numpy(float))
        & (v["iv_ok"].to_numpy(float) > 0.0)
        & np.isfinite(v["mid"].to_numpy(float))
        & (v["mid"].to_numpy(float) > 0.0)
    )
    v = v.loc[live].copy()
    h = v["hhmm"].map(H_OF).to_numpy(float)
    v["x"] = np.log(v["strike"].to_numpy(float) / v["S"].to_numpy(float))
    v["y"] = v["iv_ok"].to_numpy(float) * np.sqrt(h)
    v["side"] = (v["x"].to_numpy(float) >= 0.0).astype(int)
    v["absx"] = np.abs(v["x"].to_numpy(float))
    v = v.sort_values(["date_code", "clock_code", "side", "absx"])
    rank = v.groupby(["date_code", "clock_code", "side"]).cumcount()
    v = v.loc[rank.to_numpy() < SMILE_EACH_SIDE].copy()
    v["one"] = 1.0
    v["xy"] = v["x"] * v["y"]
    v["xx"] = v["x"] * v["x"]
    v["yy"] = v["y"] * v["y"]
    g = v.groupby(["date_code", "clock_code"])
    agg = g[["one", "x", "y", "xy", "xx", "yy"]].sum()
    side_n = (
        v.groupby(["date_code", "clock_code"])["side"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_right", "count": "n_all"})
    )
    agg = agg.join(side_n)
    n = agg["one"].to_numpy(float)
    sx = agg["x"].to_numpy(float)
    sy = agg["y"].to_numpy(float)
    sxy = agg["xy"].to_numpy(float)
    sxx = agg["xx"].to_numpy(float)
    syy = agg["yy"].to_numpy(float)
    den = n * sxx - sx * sx
    b = np.where(den > 0.0, (n * sxy - sx * sy) / np.where(den > 0.0, den, 1.0), np.nan)
    ss_tot = syy - sy * sy / n
    ss_res = ss_tot - np.where(den > 0.0, (n * sxy - sx * sy) ** 2 / (n * den), np.nan)
    r2 = np.where(
        ss_tot > 0.0, 1.0 - ss_res / np.where(ss_tot > 0.0, ss_tot, 1.0), np.nan
    )
    n_right = agg["n_right"].to_numpy(float)
    n_left = agg["n_all"].to_numpy(float) - n_right
    enough = (n_left >= SMILE_MIN_SIDE) & (n_right >= SMILE_MIN_SIDE)
    shape = (len(dates), len(CLOCKS))
    slope = np.full(shape, np.nan)
    fit_r2 = np.full(shape, np.nan)
    nl = np.zeros(shape)
    nr = np.zeros(shape)
    ii = agg.index.get_level_values(0).to_numpy(np.int64)
    jj = agg.index.get_level_values(1).to_numpy(np.int64)
    slope[ii, jj] = np.where(enough, b, np.nan)
    fit_r2[ii, jj] = np.where(enough, r2, np.nan)
    nl[ii, jj] = n_left
    nr[ii, jj] = n_right
    print(
        f"\nsmile slope: fitted on {int(np.isfinite(slope).sum())} of {slope.size} "
        f"stamp-cells ({len(dates)} sessions x {len(CLOCKS)} clocks); "
        f"{int((~np.isfinite(slope)).sum())} have fewer than {SMILE_MIN_SIDE} usable "
        f"listed strikes on a side and get none; median strikes used "
        f"{np.nanmedian(nl[nl > 0]):.1f} left / {np.nanmedian(nr[nr > 0]):.1f} right; "
        f"median fit R2 {np.nanmedian(fit_r2):.4f}"
    )
    return {"slope": slope, "r2": fit_r2, "n_left": nl, "n_right": nr}


# ------------------------------------------------------- the all-clock tape --
def build_alltape(ch: dict[str, Any]) -> dict[str, Any]:
    """Proposal 43's tape: the nearest-OTM straddle and its volatility at every clock.

    Rebuilt from the single chain read.  The nearest-OTM pick is the repo's
    own guarded picker; its vendor-outage guard (``n_live``) is applied from
    the count taken on the FULL stamp, so the strike band cannot manufacture
    an outage.  Half sessions - those whose 15:30 row has already expired
    (``hours_to_expiration <= 0``) - are dropped, as proposal 43 drops them;
    the chain's own ``early_close`` column is degenerate and is never read.
    """
    tab = ch["tab"]
    dates = ch["dates"]
    close_rows = tab.loc[tab["hhmm"].to_numpy() == CLOSE]
    med = close_rows.groupby("date")["hours_to_expiration"].median()
    half = pd.DatetimeIndex(pd.to_datetime(med.index[med <= 0])).normalize()
    sessions = pd.DatetimeIndex(dates).difference(half)
    print(
        f"\nchain sessions {len(dates)} {dates.min().date()}..{dates.max().date()}; "
        f"half sessions dropped (hours_to_expiration <= 0 at {CLOSE}, {len(half)}): "
        + ", ".join(str(d.date()) for d in half)
        + f"; sessions scored {len(sessions)}"
    )
    work = tab.loc[tab["date"].isin(sessions)].copy()
    alive = np.isfinite(work["mid"].to_numpy(float)) & (
        work["mid"].to_numpy(float) > 0.0
    )
    live = work.loc[
        alive, ["date", "hhmm", "strike", "cp", "bid", "ask", "underlying_price"]
    ]
    sp = asl.stamp_spot(live, ["date", "hhmm"])
    t0 = time.time()
    atm, dropped = asl.pick_nearest_otm_guarded(
        live, sp, keys=("date", "hhmm"), min_live=0
    )
    nl = ch["n_live"]
    bad = nl.loc[nl["n_live"].to_numpy() < asl.ATM_MIN_LIVE]
    outage = set(
        zip(
            pd.DatetimeIndex(bad["date"]).astype("int64").tolist(),
            [str(x) for x in bad["hhmm"].to_numpy()],
            strict=True,
        )
    )
    few = np.array(
        [
            (d, h) in outage
            for d, h in zip(
                pd.DatetimeIndex(atm["date"]).astype("int64").tolist(),
                [str(x) for x in atm["hhmm"].to_numpy()],
                strict=True,
            )
        ],
        dtype=bool,
    )
    refused = pd.concat(
        [
            dropped[["date", "hhmm", "reason"]],
            atm.loc[few, ["date", "hhmm"]].assign(reason="few_live"),
        ],
        ignore_index=True,
    )
    atm = atm.loc[~few].copy()
    print(
        f"nearest-OTM pick with the guards at every stamp: {time.time() - t0:.1f} s; "
        f"stamps refused {len(refused)}"
        + (
            ": "
            + ", ".join(
                f"{pd.Timestamp(r['date']).date()} {r['hhmm']} {r['reason']}"
                for _, r in refused.iterrows()
            )
            if len(refused)
            else ""
        )
    )

    idx = pd.DatetimeIndex(sessions)
    cols = list(CLOCKS)

    def piv(frame: pd.DataFrame, col: str) -> np.ndarray:
        return (
            frame.pivot_table(index="date", columns="hhmm", values=col, aggfunc="first")
            .reindex(index=idx, columns=cols)
            .to_numpy(float)
        )

    s_grid = piv(sp.rename("S_stamp").reset_index(), "S_stamp")
    k_c = piv(atm, "K_c")
    k_p = piv(atm, "K_p")
    entry = piv(atm, "entry")
    bid = piv(atm, "bid_entry")
    tot = invert_package_vec(s_grid, k_c, k_p, entry)

    t0 = time.time()
    ref = np.array(
        [
            [
                invert_total_vol(s_grid[i, k], k_c[i, k], k_p[i, k], entry[i, k])
                for k in range(len(cols))
            ]
            for i in range(len(idx))
        ]
    )
    both_nan = np.isnan(tot) & np.isnan(ref)
    dev = float(np.max(np.where(both_nan, 0.0, np.abs(tot - ref))))
    assert dev == 0.0, dev
    print(
        f"GATE 4  vectorised package-volatility inversion vs "
        f"live.ibkr.pricing.invert_total_vol on all {tot.size} cells: max abs "
        f"difference {dev:.1e}, NaN cells {int(np.isnan(tot).sum())} both sides "
        f"({time.time() - t0:.1f} s for the scalar reference)"
    )

    gspc = pd.read_parquet(GSPC)
    gspc.index = pd.to_datetime(gspc.index)
    s_close = gspc["close"].reindex(idx).to_numpy(float)

    nxt = np.full_like(s_grid, np.nan)
    nxt[:, :-1] = s_grid[:, 1:]
    nxt[:, -1] = s_close
    step2 = np.where(
        np.isfinite(s_grid) & np.isfinite(nxt), np.log(nxt / s_grid) ** 2, np.nan
    )
    rv_rem = np.full_like(s_grid, np.nan)
    for k in range(len(cols)):
        whole = np.isfinite(step2[:, k:]).all(axis=1)
        rv_rem[:, k] = np.where(whole, np.nansum(step2[:, k:], axis=1), np.nan)
    iv_rem = tot**2
    ratio = np.where(
        np.isfinite(iv_rem) & (iv_rem > 0.0) & np.isfinite(rv_rem) & (rv_rem > 0.0),
        rv_rem / np.where(iv_rem > 0.0, iv_rem, 1.0),
        np.nan,
    )
    f9 = expanding_lagged_mean(ratio, WARMUP)
    print(
        f"V9's per-clock factor on the all-clock tape: the expanding lagged "
        f"{WARMUP}-session mean of realized over implied remaining-window variance "
        f"covers {int((np.isfinite(f9) & (f9 > 0.0)).sum())} of {f9.size} "
        f"stamp-cells; the rest hedge with the book's own volatility"
    )
    return {
        "dates": idx,
        "S": s_grid,
        "K_c": k_c,
        "K_p": k_p,
        "entry": entry,
        "bid": bid,
        "tot": tot,
        "S_close": s_close,
        "f9": f9,
        "half": half,
        "refused": refused,
    }


# ------------------------------------------------------------- the books ----
def _charm(h: np.ndarray, cut: float) -> np.ndarray:
    """sqrt((h - cut) / h): the horizon shortened by ``cut`` hours, floored at zero."""
    left = np.maximum(h - cut, 0.0)
    return np.sqrt(left / h)


def _phi_d1(s: np.ndarray, f: np.ndarray, k: np.ndarray) -> np.ndarray:
    """phi(d1) at a leg's own total volatility; zero where the volatility is dead."""
    shape = np.broadcast_shapes(np.shape(s), np.shape(f), np.shape(k))
    ss = np.broadcast_to(np.asarray(s, float), shape)
    ff = np.broadcast_to(np.asarray(f, float), shape)
    kk = np.broadcast_to(np.asarray(k, float), shape)
    ok = (
        np.isfinite(ss)
        & (ss > 0.0)
        & np.isfinite(ff)
        & (ff > 0.0)
        & np.isfinite(kk)
        & (kk > 0.0)
    )
    return np.where(
        ok,
        npdf(_d1(np.where(ok, ss, 1.0), np.where(ok, ff, 1.0), np.where(ok, kk, 1.0))),
        0.0,
    )


def _leg_sum(
    sc: np.ndarray, sp: np.ndarray, s: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """The package delta as the sum of the two legs' own-volatility deltas."""
    return leg_delta_vec(sc, s, kc, True) + leg_delta_vec(sp, s, kp, False)


def _skew_term(
    sc: np.ndarray,
    sp: np.ndarray,
    s: np.ndarray,
    kc: np.ndarray,
    kp: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """sum over legs of vega_leg x dsigma/dS under sticky delta.

    vega_leg = S phi(d1_leg) and dsigma/dS|_K = -b / S with b the slope of the
    total volatility in ln(K/S), so the S cancels and the correction is
    -b (phi(d1c) + phi(d1p)) - a pure number, in the same units as the delta.
    """
    return -slope * (_phi_d1(sc, s, kc) + _phi_d1(sp, s, kp))


def variant_delta(name: str, bk: dict[str, Any]) -> np.ndarray:
    """The hedge delta grid of one variant on one book."""
    s = bk["S"]
    kc = bk["Kc"][:, None]
    kp = bk["Kp"][:, None]
    h = bk["h"][None, :]
    s0 = bk["s0"]
    sc = bk["sC"]
    sp = bk["sP"]
    f9 = bk["f9"]
    slope = bk["slope"]
    if name == "D0_book":
        return pkg_delta_vec(s0, s, kc, kp)
    if name == REF_V9:
        return pkg_delta_vec(np.sqrt(s0 * s0 * f9), s, kc, kp)
    if name == "D1_leg_vol":
        return _leg_sum(sc, sp, s, kc, kp)
    if name == "D1v_vendor_delta":
        return np.asarray(bk["vd"], float)
    if name == "D2_skew_sticky_delta":
        return _leg_sum(sc, sp, s, kc, kp) + _skew_term(sc, sp, s, kc, kp, slope)
    if name == "D3_charm_half":
        return pkg_delta_vec(s0 * _charm(h, CHARM_HALF), s, kc, kp)
    if name == "D3_charm_full":
        return pkg_delta_vec(s0 * _charm(h, CHARM_FULL), s, kc, kp)
    if name == "D4_D9_D1":
        return _leg_sum(np.sqrt(sc * sc * f9), np.sqrt(sp * sp * f9), s, kc, kp)
    if name == "D4_D9_D3":
        return pkg_delta_vec(np.sqrt(s0 * s0 * f9) * _charm(h, CHARM_HALF), s, kc, kp)
    if name == "D4_D9_D1_D3":
        g = _charm(h, CHARM_HALF)
        return _leg_sum(np.sqrt(sc * sc * f9) * g, np.sqrt(sp * sp * f9) * g, s, kc, kp)
    if name == "D4_D9_D2_D3":
        g = _charm(h, CHARM_HALF)
        a = np.sqrt(sc * sc * f9) * g
        b = np.sqrt(sp * sp * f9) * g
        return _leg_sum(a, b, s, kc, kp) + _skew_term(a, b, s, kc, kp, slope)
    if name.startswith("D5_lastbar_"):
        lean = float(name.rsplit("_", 1)[1])
        dlt = pkg_delta_vec(s0, s, kc, kp).copy()
        last = int(np.max(np.flatnonzero(bk["held"] > 0.0)))
        dlt[:, last] = dlt[:, last] * lean
        return dlt
    raise KeyError(name)


def book_pnl(delta: np.ndarray, bk: dict[str, Any]) -> dict[str, np.ndarray]:
    """One delta grid's daily P&L, turnover and hedge charge on one book.

    The position held over the interval that starts at stamp j is
    ``delta[:, j] x held[j]``; the trade at stamp j is the change in that
    position, priced at that stamp's spot, and the book's last position is
    unwound at the terminal price (the 15:30 spot on the exit book, where it
    is already flat, and the settlement print on the hold books).
    """
    pos = delta * bk["held"][None, :]
    hedge = (pos * bk["dS"]).sum(axis=1)
    prev = np.concatenate([np.zeros((pos.shape[0], 1)), pos[:, :-1]], axis=1)
    traded = np.abs(pos - prev)
    turn = traded.sum(axis=1) + np.abs(pos[:, -1])
    cost_pts = (traded * bk["px"]).sum(axis=1) + np.abs(pos[:, -1]) * bk["px_term"]
    cost_pts = cost_pts * (HEDGE_COST_BP * 1e-4)
    pts = bk["pnl_opt"] + hedge
    nan = np.full(pts.shape, np.nan)
    ok = bk["valid"]
    em = bk["entry_mid"]
    return {
        "pts": np.where(ok, pts, nan),
        "prem": np.where(ok, pts / em, nan),
        "hedge_pts": np.where(ok, hedge, nan),
        "turnover": np.where(ok, turn, nan),
        "cost_pts": np.where(ok, cost_pts, nan),
        "cost_prem": np.where(ok, cost_pts / em, nan),
        "delta": delta,
    }


def trend_ratio(path: np.ndarray) -> np.ndarray:
    """|sum of bar returns| / sum |bar returns| over a spot path (proposal 32's)."""
    bar = np.divide(path[:, 1:] - path[:, :-1], path[:, :-1])
    return np.abs(bar.sum(1)) / np.abs(bar).sum(1)


def trend_labels(
    ratio: np.ndarray, cuts: tuple[float, float] | None = None
) -> tuple[np.ndarray, float, float]:
    """Proposal 32's shape terciles of the trend ratio."""
    if cuts is None:
        q1, q2 = (float(x) for x in np.nanquantile(ratio, [1 / 3, 2 / 3]))
    else:
        q1, q2 = cuts
    lab = np.full(ratio.shape, "2_mixed", dtype=object)
    lab[ratio <= q1] = "1_choppy"
    lab[ratio > q2] = "3_trend"
    return lab, q1, q2


def fill_inputs(bk: dict[str, Any]) -> dict[str, int]:
    """Every variant's fallback, applied once and counted once.

    A leg with no usable midpoint hedges that leg at the book's own package
    volatility; a stamp with no vendor delta hedges that leg at the book's own
    delta; a stamp with no smile slope carries no skew term; a clock with no
    V9 factor carries a factor of one.  So a variant differs from the book
    only where it has something to say, and every fallback is on the row.
    """
    s0 = bk["s0"]
    s = bk["S"]
    kc = bk["Kc"][:, None]
    kp = bk["Kp"][:, None]
    bad_c = ~(np.isfinite(bk["sC"]) & (bk["sC"] > 0.0))
    bad_p = ~(np.isfinite(bk["sP"]) & (bk["sP"] > 0.0))
    bk["sC"] = np.where(bad_c, s0, bk["sC"])
    bk["sP"] = np.where(bad_p, s0, bk["sP"])
    vdc = bk["vdC"]
    vdp = bk["vdP"]
    cen_c = ~np.isfinite(
        asl.censor_vendor_iv(pd.Series(bk["ivC"].ravel())).to_numpy(float)
    ).reshape(vdc.shape)
    cen_p = ~np.isfinite(
        asl.censor_vendor_iv(pd.Series(bk["ivP"].ravel())).to_numpy(float)
    ).reshape(vdp.shape)
    bad_vc = ~(np.isfinite(vdc) & (vdc >= 0.0) & (vdc <= 1.0)) | cen_c
    bad_vp = ~(np.isfinite(vdp) & (vdp <= 0.0) & (vdp >= -1.0)) | cen_p
    bk["vd"] = np.where(bad_vc, leg_delta_vec(s0, s, kc, True), vdc) + np.where(
        bad_vp, leg_delta_vec(s0, s, kp, False), vdp
    )
    bad_b = ~np.isfinite(bk["slope"])
    bk["slope"] = np.where(bad_b, 0.0, bk["slope"])
    bad_f = ~(np.isfinite(bk["f9"]) & (bk["f9"] > 0.0))
    bk["f9"] = np.where(bad_f, 1.0, bk["f9"])
    cells = int(s0.size)
    return {
        "n_cells": cells,
        "n_fallback_leg_vol_c": int(bad_c.sum()),
        "n_fallback_leg_vol_p": int(bad_p.sum()),
        "n_fallback_vendor_delta_c": int(bad_vc.sum()),
        "n_fallback_vendor_delta_p": int(bad_vp.sum()),
        "n_fallback_slope": int(bad_b.sum()),
        "n_fallback_v9": int(bad_f.sum()),
    }


def leg_inputs(
    keys: np.ndarray,
    cols: dict[str, np.ndarray],
    date_code: np.ndarray,
    clocks: tuple[str, ...],
    spot: np.ndarray,
    kc: np.ndarray,
    kp: np.ndarray,
) -> dict[str, np.ndarray]:
    """The held legs' own midpoints, own-midpoint volatilities and vendor deltas."""
    qc = leg_lookup(keys, cols, date_code, clocks, kc, True)
    qp = leg_lookup(keys, cols, date_code, clocks, kp, False)
    return {
        "sC": invert_leg_vec(spot, kc[:, None], qc["mid"], True),
        "sP": invert_leg_vec(spot, kp[:, None], qp["mid"], False),
        "vdC": qc["delta"],
        "vdP": qp["delta"],
        "ivC": qc["impl_volatility"],
        "ivP": qp["impl_volatility"],
        "ask_c": qc["ask"],
        "ask_p": qp["ask"],
        "bid_c": qc["bid"],
        "bid_p": qp["bid"],
        "mid_c": qc["mid"],
        "mid_p": qp["mid"],
    }


def hold_book(
    name: str,
    entry_clock: str,
    tape: dict[str, Any],
    slope_grid: np.ndarray,
    keys: np.ndarray,
    cols: dict[str, np.ndarray],
    date_code: np.ndarray,
) -> dict[str, Any]:
    """One entry clock's hold-to-settlement book on the all-clock tape.

    The straddle is sold at the entry clock's quoted bid, hedged on the index
    every 30 minutes from that clock on, and carried into the 4pm cash
    settlement; the last hedge set at 15:30 runs through the settlement print.
    """
    j = CLOCKS.index(entry_clock)
    clocks = CLOCKS[j:]
    idx = pd.DatetimeIndex(tape["dates"])
    n = len(idx)
    m = len(clocks)
    s = tape["S"][:, j:]
    s_close = tape["S_close"]
    nxt = np.full_like(s, np.nan)
    nxt[:, :-1] = s[:, 1:]
    nxt[:, -1] = s_close
    d_s = np.where(np.isfinite(s) & np.isfinite(nxt), nxt - s, 0.0)
    kc = tape["K_c"][:, j]
    kp = tape["K_p"][:, j]
    entry = tape["entry"][:, j]
    bid = tape["bid"][:, j]
    settle = np.maximum(s_close - kc, 0.0) + np.maximum(kp - s_close, 0.0)
    bk: dict[str, Any] = {
        "name": name,
        "entry_clock": entry_clock,
        "clocks": clocks,
        "dates": idx.to_numpy(),
        "h": np.asarray([H_OF[c] for c in clocks], float),
        "S": s,
        "dS": d_s,
        "px": np.where(np.isfinite(s), s, 0.0),
        "px_term": s_close,
        "held": np.ones(m),
        "Kc": kc,
        "Kp": kp,
        "s0": tape["tot"][:, j:],
        "f9": tape["f9"][:, j:],
        "slope": slope_grid[date_code][:, j:],
        "entry_mid": entry,
        "pnl_opt": -(settle - bid),
        "valid": np.isfinite(entry) & (entry > 0.0) & np.isfinite(bid) & (bid > 0.0),
    }
    bk.update(leg_inputs(keys, cols, date_code, clocks, s, kc, kp))
    path = np.concatenate([s, s_close[:, None]], axis=1)
    ratio = trend_ratio(path)
    lab, q1, q2 = trend_labels(ratio)
    bk["trend_lab"] = lab
    bk["trend_cuts"] = (q1, q2)
    bk["samples"] = {
        "deck_in_sample": np.asarray(idx <= DECK_END),
        "unseen": np.asarray((idx >= OOS_START) & (idx <= OOS_END)),
        "all": np.ones(n, dtype=bool),
    }
    return bk


def exit_book_1100(
    p32: Any,
    tape: dict[str, Any],
    dates: pd.DatetimeIndex,
    f9: np.ndarray,
    slope_grid: np.ndarray,
    keys: np.ndarray,
    cols: dict[str, np.ndarray],
    date_code: np.ndarray,
    trend_cuts: tuple[float, float],
) -> dict[str, Any]:
    """Proposal 32's 11:00 crossed-quoted exit book, as proposal 36 hedges it.

    Entry at the 11:00 quoted bid, a rebalance every 30 minutes, both legs
    bought back at the 15:30 quoted ask and the futures flattened there, so
    the last position held is the one set at 15:00.
    """
    clocks = SESSION_1100
    n = len(dates)
    m = len(clocks)
    s = tape["spot"]
    kc = tape["Kc"]
    kp = tape["Kp"]
    d_s = np.zeros((n, m))
    d_s[:, :-1] = s[:, 1:] - s[:, :-1]
    held = np.ones(m)
    held[-1] = 0.0
    legs = leg_inputs(keys, cols, date_code, clocks, s, kc, kp)
    ask_pkg = legs["ask_c"][:, -1] + legs["ask_p"][:, -1]
    assert np.isfinite(ask_pkg).all(), "a 15:30 buy-back leg has no quote"
    h = np.asarray([H_OF[c] for c in clocks], float)
    bk: dict[str, Any] = {
        "name": "1100_exit_crossed",
        "entry_clock": "11:00",
        "clocks": clocks,
        "dates": pd.DatetimeIndex(dates).to_numpy(),
        "h": h,
        "S": s,
        "dS": d_s,
        "px": s,
        "px_term": s[:, -1],
        "held": held,
        "Kc": kc,
        "Kp": kp,
        "s0": tape["sig"] * np.sqrt(h)[None, :],
        "f9": f9,
        "slope": slope_grid[date_code][:, 2:],
        "entry_mid": tape["entry_mid"],
        "pnl_opt": -(ask_pkg - tape["entry_bid"]),
        "valid": np.ones(n, dtype=bool),
        "ask_pkg_1530": ask_pkg,
    }
    bk.update(legs)
    lab, q1, q2 = trend_labels(trend_ratio(s), trend_cuts)
    bk["trend_lab"] = lab
    bk["trend_cuts"] = (q1, q2)
    bk["samples"] = {
        "whole": np.ones(n, dtype=bool),
        "daily_era": np.asarray(pd.DatetimeIndex(dates) >= ERA0),
    }
    return bk


# ----------------------------------------------------------- the workers ----
_BOOKS: dict[str, dict[str, Any]] = {}


def _init_books(payload: dict[str, dict[str, Any]]) -> None:
    """Hand each worker process the books once, not once per task."""
    _BOOKS.clear()
    _BOOKS.update(payload)


def _returns_task(arg: tuple[str, str]) -> tuple[str, str, dict[str, np.ndarray]]:
    """One (book, variant): the delta grid and the daily book it produces."""
    book, variant = arg
    bk = _BOOKS[book]
    return book, variant, book_pnl(variant_delta(variant, bk), bk)


def _sharpe(v: np.ndarray) -> float:
    x = np.asarray(v, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    return float(x.mean()) / sd * ANN if sd > 0.0 else float("nan")


def _stats_task(
    arg: tuple[str, str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """One (book, variant): its rows of variants.csv, pairs.csv and bootstrap.csv."""
    book, variant, pay = arg
    res = pay["res"]
    refs = pay["refs"]
    dates = pay["dates"]
    samples: dict[str, np.ndarray] = pay["samples"]
    lab = pay["trend_lab"]
    fall: dict[str, int] = pay["fallback"]
    vrows: list[dict[str, Any]] = []
    prows: list[dict[str, Any]] = []
    brows: list[dict[str, Any]] = []
    for unit in UNITS:
        gross = res[unit]
        cost = res["cost_pts" if unit == "pts" else "cost_prem"]
        for sample, mask in samples.items():
            row: dict[str, Any] = {
                "book": book,
                "variant": variant,
                "causal": not variant.startswith("D5_"),
                "note": VARIANT_NOTE[variant],
                "sample": sample,
                "unit": UNIT_NAME[unit],
            }
            row.update(stat_row(gross[mask], dates[mask]))
            net = gross - cost
            row["mean_cost_reported"] = float(np.nanmean(cost[mask]))
            row["mean_charged"] = float(np.nanmean(net[mask]))
            row["Sharpe_ann_charged"] = _sharpe(net[mask])
            row["mean_turnover"] = float(np.nanmean(res["turnover"][mask]))
            for ref in (REF_BOOK, REF_V9):
                d = gross - refs[ref][unit]
                tag = "D0" if ref == REF_BOOK else "D9"
                row[f"mean_abs_diff_vs_{tag}"] = float(np.nanmean(np.abs(d[mask])))
                row[f"sd_diff_vs_{tag}"] = float(np.nanstd(d[mask], ddof=1))
            for cls, tg in (("1_choppy", "choppy"), ("3_trend", "trend")):
                sel = mask & (lab == cls)
                row[f"mean_{tg}"] = float(np.nanmean(gross[sel]))
                row[f"Sharpe_{tg}"] = _sharpe(gross[sel])
                row[f"n_{tg}"] = int(sel.sum())
            row.update(fall)
            vrows.append(row)
            for ref in (REF_BOOK, REF_V9):
                if variant == ref:
                    continue
                base = {
                    "book": book,
                    "variant": variant,
                    "reference": ref,
                    "sample": sample,
                    "unit": UNIT_NAME[unit],
                }
                prows.append({**base, **paired_row(gross[mask], refs[ref][unit][mask])})
                brow = {
                    **base,
                    "B": BOOT_B,
                    "block": BOOT_BLOCK,
                    "seed": BOOT_SEED,
                    "ci_pct_lo": CI_PCT[0],
                    "ci_pct_hi": CI_PCT[1],
                }
                brow.update(sharpe_diff_ci(gross[mask], refs[ref][unit][mask]))
                brows.append(brow)
    return {"variants": vrows, "pairs": prows, "bootstrap": brows}


# -------------------------------------------------------------- the gates ---
def remaining_grid(
    p30: Any, dates: pd.DatetimeIndex, cols: tuple[str, ...]
) -> np.ndarray:
    """Proposal 36's realized remaining window, rebuilt by its own recipe."""
    base = asl.load_yhat_panel(asl.yhat_paths(ROOT)["blk2"]).set_index("t")
    clocks = list(CLOCKS)
    prof, _ = p30.panel_profile(base, clocks)
    assert list(prof.columns) == clocks, list(prof.columns)
    rv = prof[clocks].to_numpy(float)
    rem = pd.DataFrame(
        rv[:, ::-1].cumsum(axis=1)[:, ::-1], index=prof.index, columns=clocks
    )
    out = rem.reindex(index=dates, columns=list(cols)).to_numpy(float)
    assert np.isfinite(out).all(), "a book stamp has no realized remaining window"
    print(
        f"\nthe realized remaining window: {len(prof)} panel sessions "
        f"{prof.index.min().date()} .. {prof.index.max().date()} on {len(clocks)} "
        f"trade bars, read on the book's {out.shape[0]} days x {out.shape[1]} stamps"
    )
    return out


def censored_grid(
    panel: pd.DataFrame, dates: pd.DatetimeIndex, cols: tuple[str, ...]
) -> np.ndarray:
    """Proposal 36's censored-implied mask, unchanged."""
    from live.ibkr.parity import research_modules

    _, rule_mod = research_modules(ROOT)
    flag = rule_mod.on_vendor_node(
        panel["impl_volatility_c"]
    ) | rule_mod.on_vendor_node(panel["impl_volatility_p"])
    out = (
        panel.assign(_cen=np.asarray(flag, dtype=float))
        .pivot_table(index="date", columns="hhmm", values="_cen", aggfunc="max")
        .reindex(index=dates, columns=list(cols))
        .fillna(1.0)
        .to_numpy(float)
        > 0
    )
    print(
        f"censored implied volatility: {int(out.sum())} of {out.size} stamp-cells "
        f"on the book's days (those stamps hedge with the book's own volatility)"
    )
    return out


def gate_1100(
    p32: Any,
    p36: Any,
    tape: dict[str, Any],
    mid: np.ndarray,
    ask: np.ndarray,
    dates: pd.DatetimeIndex,
    var0: np.ndarray,
    v9: np.ndarray,
) -> tuple[np.ndarray, float, float, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """GATE 1-3: proposal 32's book, its terciles and proposal 36's V9."""
    dev_entry = float(np.max(np.abs(mid[:, 0] - tape["entry_mid"])))
    assert dev_entry < EXACT_TOL, dev_entry
    print(
        f"\nGATE 1  the chain read reproduces the 11:00 package midpoint of the "
        f"trade cache on all {len(dates)} days to {dev_entry:.3e}"
    )
    p32.gate(tape, mid, ask, dates)

    legs0 = p36.hedge_legs(np.sqrt(var0), tape)
    d_exit = float(np.max(np.abs(legs0["hedge_exit"] - tape["hedge_ref"])))
    d_hold = float(np.max(np.abs(legs0["hedge_hold"] - tape["hedge_hold"])))
    assert d_exit < EXACT_TOL, d_exit
    assert d_hold < EXACT_TOL, d_hold
    print(
        f"GATE 1  the hedge leg rebuilt with the book's implied total volatility "
        f"matches live.ibkr.parity.replay_day to {d_exit:.3e} (exit) and "
        f"{d_hold:.3e} (carried into the settlement)"
    )
    ratio = trend_ratio(tape["spot"])
    q1, q2 = (float(x) for x in np.quantile(ratio, [1 / 3, 2 / 3]))
    assert abs(q1 - GATE_TREND_LO) < TREND_TOL, q1
    assert abs(q2 - GATE_TREND_HI) < TREND_TOL, q2
    lab, _, _ = trend_labels(ratio, (q1, q2))
    print(
        f"GATE 2  proposal 32's shape terciles of |sum of bar returns| / sum |bar "
        f"returns|: {q1:.6f} / {q2:.6f}  reference {GATE_TREND_LO:.4f} / "
        f"{GATE_TREND_HI:.4f}  OK; {int((lab == '1_choppy').sum())} choppy, "
        f"{int((lab == '2_mixed').sum())} mixed, {int((lab == '3_trend').sum())} trend"
    )

    legs9 = p36.hedge_legs(np.sqrt(v9), tape)
    r0 = p36.book_returns(tape, mid, ask, legs0)
    r9 = p36.book_returns(tape, mid, ask, legs9)
    era = np.asarray(pd.DatetimeIndex(dates) >= ERA0, dtype=bool)
    print("GATE 3  proposal 36's V9 on the primary book")
    for sample, m, d_want, ci_want, hac_want in (
        (
            "whole",
            np.ones(era.size, dtype=bool),
            GATE_V9_DSHARPE_WHOLE,
            GATE_V9_CI_WHOLE,
            GATE_V9_HAC_T_WHOLE,
        ),
        ("daily_era", era, GATE_V9_DSHARPE_ERA, GATE_V9_CI_ERA, GATE_V9_HAC_T_ERA),
    ):
        got = sharpe_diff_ci(r9["exit_crossed"][m], r0["exit_crossed"][m])
        pair = paired_row(r9["exit_crossed"][m], r0["exit_crossed"][m])
        assert abs(got["dSharpe"] - d_want) < EXACT_TOL, (sample, got["dSharpe"])
        assert abs(got["boot_lo"] - ci_want[0]) < EXACT_TOL, (sample, got["boot_lo"])
        assert abs(got["boot_hi"] - ci_want[1]) < EXACT_TOL, (sample, got["boot_hi"])
        assert abs(pair["hac_t_diff"] - hac_want) < EXACT_TOL, (
            sample,
            pair["hac_t_diff"],
        )
        print(
            f"  {sample:<10s} n {got['n']:>4d}  dSharpe vs D0 {got['dSharpe']:+.12f} "
            f"[{got['boot_lo']:+.12f}, {got['boot_hi']:+.12f}]  HAC t "
            f"{pair['hac_t_diff']:+.12f}  reference {d_want:+.6f} "
            f"[{ci_want[0]:+.6f}, {ci_want[1]:+.6f}] / {hac_want:+.6f}  OK"
        )
    return lab, q1, q2, r0, r9


def gate_43(
    tape: dict[str, Any],
    keys: np.ndarray,
    cols: dict[str, np.ndarray],
    date_code: np.ndarray,
) -> None:
    """GATE 5: proposal 43's published daily file, column by column, at two clocks."""
    ref = pd.read_csv(REF43 / "a_daily_by_clock.csv", index_col=0, parse_dates=True)
    idx = pd.DatetimeIndex(tape["dates"])
    assert len(idx) == GATE_N_SESSIONS, len(idx)
    common = idx.intersection(pd.DatetimeIndex(ref.index))
    assert len(common) == GATE_N_SESSIONS, len(common)
    s_close = tape["S_close"]
    for clock in ("11:00", "13:30"):
        j = CLOCKS.index(clock)
        s = tape["S"][:, j:]
        kc = tape["K_c"][:, j]
        kp = tape["K_p"][:, j]
        entry = tape["entry"][:, j]
        bid = tape["bid"][:, j]
        nxt = np.full_like(s, np.nan)
        nxt[:, :-1] = s[:, 1:]
        nxt[:, -1] = s_close
        d_s = np.where(np.isfinite(s) & np.isfinite(nxt), nxt - s, 0.0)
        d_flat = d_s.copy()
        d_flat[:, -1] = 0.0
        dlt = pkg_delta_vec(tape["tot"][:, j:], s, kc[:, None], kp[:, None])
        settle = np.maximum(s_close - kc, 0.0) + np.maximum(kp - s_close, 0.0)
        legs = leg_lookup(keys, cols, date_code, (CLOSE,), kc, True)
        legp = leg_lookup(keys, cols, date_code, (CLOSE,), kp, False)
        a_c = legs["ask"][:, 0]
        a_p = legp["ask"][:, 0]
        b_c = legs["bid"][:, 0]
        b_p = legp["bid"][:, 0]
        dead = ((b_c == 0.0) & (a_c == 0.0)) | ((b_p == 0.0) & (a_p == 0.0))
        buyback = np.where(dead, np.nan, a_c + a_p)
        ok = np.isfinite(entry) & (entry > 0.0) & np.isfinite(bid) & (bid > 0.0)
        nan = np.full(len(idx), np.nan)
        mine = pd.DataFrame(
            {
                "hold": np.where(
                    ok, (-(settle - bid) + (dlt * d_s).sum(axis=1)) / entry, nan
                ),
                "flatten": np.where(
                    ok, (-(buyback - bid) + (dlt * d_flat).sum(axis=1)) / entry, nan
                ),
                "entry": np.where(ok, entry, nan),
                "bid": np.where(ok, bid, nan),
                "K_c": kc,
                "K_p": kp,
            },
            index=idx,
        )
        devs = {
            col: float(
                (mine.loc[common, col] - ref.loc[common, f"{clock}|{col}"]).abs().max()
            )
            for col in mine.columns
        }
        assert max(devs.values()) < EXACT_TOL, (clock, devs)
        print(
            f"GATE 5  the {clock} tape reproduces proposal 43's published daily file "
            f"on {len(common)} sessions, column by column: "
            + ", ".join(f"{k} {v:.1e}" for k, v in devs.items())
            + f" (tolerance {EXACT_TOL:.0e})"
        )
        if clock == "13:30":
            pts = mine["hold"] * mine["entry"]
            m = (idx >= OOS_START) & (idx <= OOS_END)
            v = pts[m].dropna()
            t, lag = asl.newey_west_t(v)
            assert int(v.size) == GATE_N_OOS, int(v.size)
            assert abs(float(v.mean()) - GATE_1330_OOS_PTS) < GATE_PTS_TOL, float(
                v.mean()
            )
            assert abs(t - GATE_1330_OOS_T) < GATE_T_TOL, t
            n_deck = int(mine["hold"][idx <= DECK_END].dropna().size)
            assert n_deck == GATE_N_DECK, n_deck
            print(
                f"GATE 5  the 13:30 hold book on the {int(v.size)} UNSEEN sessions: "
                f"{float(v.mean()):+.6f} index points per contract per day "
                f"(reference {GATE_1330_OOS_PTS:+.6f}), HAC t {t:+.6f} (lag {lag}, "
                f"reference {GATE_1330_OOS_T:+.6f}); deck in-sample {n_deck} "
                f"(reference {GATE_N_DECK}), all {len(idx)} (reference "
                f"{GATE_N_SESSIONS})"
            )


def gate_leg_identity(bk: dict[str, Any]) -> None:
    """GATE 6: the two legs' deltas at a COMMON volatility are the package delta."""
    s0 = bk["s0"]
    s = bk["S"]
    kc = bk["Kc"][:, None]
    kp = bk["Kp"][:, None]
    dev = float(
        np.max(np.abs(_leg_sum(s0, s0, s, kc, kp) - pkg_delta_vec(s0, s, kc, kp)))
    )
    assert dev < EXACT_TOL, dev
    print(
        f"GATE 6  on {bk['name']}: call delta + put delta at a common total "
        f"volatility equals live.ibkr.pricing.package_delta's N(d1c) + N(d1p) - 1 on "
        f"all {s0.size} stamp-cells to {dev:.3e}"
    )


def vendor_delta_check(bk: dict[str, Any]) -> pd.DataFrame:
    """What the chain's own ``delta`` column is, checked leg by leg.

    Run BEFORE any fallback is applied: the comparison is against this
    script's own Black-76 leg delta at the leg's own midpoint-inverted total
    volatility, on the cells where both exist.
    """
    s = bk["S"]
    kc = bk["Kc"][:, None]
    kp = bk["Kp"][:, None]
    rows: list[dict[str, Any]] = []
    for tag, vd, own, k in (
        ("call at K_c", bk["vdC"], leg_delta_vec(bk["sC"], s, kc, True), kc),
        ("put at K_p", bk["vdP"], leg_delta_vec(bk["sP"], s, kp, False), kp),
    ):
        ok = np.isfinite(vd) & np.isfinite(bk["sC" if "call" in tag else "sP"])
        ok = ok & np.isfinite(
            asl.censor_vendor_iv(
                pd.Series(bk["ivC" if "call" in tag else "ivP"].ravel())
            )
            .to_numpy(float)
            .reshape(vd.shape)
        )
        a = vd[ok]
        b = own[ok]
        rows.append(
            {
                "book": bk["name"],
                "leg": tag,
                "n_cells": int(vd.size),
                "n_comparable": int(ok.sum()),
                "vendor_min": float(np.nanmin(vd)),
                "vendor_max": float(np.nanmax(vd)),
                "own_min": float(np.nanmin(own)),
                "own_max": float(np.nanmax(own)),
                "max_abs_diff": float(np.max(np.abs(a - b)))
                if a.size
                else float("nan"),
                "mean_abs_diff": float(np.mean(np.abs(a - b)))
                if a.size
                else float("nan"),
                "corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan"),
                "share_same_sign": float(np.mean(np.sign(a) == np.sign(b)))
                if a.size
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def slope_table(
    sm: dict[str, np.ndarray],
    spot: np.ndarray,
    dates: pd.DatetimeIndex,
    date_code: np.ndarray,
) -> pd.DataFrame:
    """The smile slope's distribution, clock by clock, on the scored sessions."""
    b = sm["slope"][date_code]
    r2 = sm["r2"][date_code]
    nl = sm["n_left"][date_code]
    nr = sm["n_right"][date_code]
    dsig = np.where(np.isfinite(b) & np.isfinite(spot), -b / spot, np.nan)
    masks = {
        "deck_in_sample": np.asarray(dates <= DECK_END),
        "unseen": np.asarray((dates >= OOS_START) & (dates <= OOS_END)),
        "all": np.ones(len(dates), dtype=bool),
    }
    rows: list[dict[str, Any]] = []
    for sample, m in masks.items():
        for j, clock in enumerate(CLOCKS):
            v = b[m, j]
            v = v[np.isfinite(v)]
            d = dsig[m, j]
            d = d[np.isfinite(d)]
            pct = np.percentile(v, [10.0, 50.0, 90.0]) if v.size else np.full(3, np.nan)
            rows.append(
                {
                    "sample": sample,
                    "clock": clock,
                    "h_rem": H_OF[clock],
                    "n_sessions": int(m.sum()),
                    "n_fitted": int(v.size),
                    "n_no_slope": int(m.sum()) - int(v.size),
                    "slope_p10": float(pct[0]),
                    "slope_median": float(pct[1]),
                    "slope_p90": float(pct[2]),
                    "slope_mean": float(v.mean()) if v.size else float("nan"),
                    "share_slope_negative": float((v < 0.0).mean())
                    if v.size
                    else float("nan"),
                    "dsigma_dS_median_per_point": float(np.median(d))
                    if d.size
                    else float("nan"),
                    "fit_r2_median": float(np.nanmedian(r2[m, j])),
                    "strikes_left_median": float(np.median(nl[m, j])),
                    "strikes_right_median": float(np.median(nr[m, j])),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 90)
    pd.set_option("display.max_rows", 600)
    t_start = time.time()

    p30 = load_module(HERE / "30_skip_day_rule.py", "p48_p30")
    p32 = load_module(HERE / "32_loss_anatomy.py", "p48_p32")
    p36 = load_module(HERE / "36_remaining_window_honest.py", "p48_p36")

    stamp, px = session_stamps()
    ch = read_chain_once(stamp, px)
    keys, cols = leg_table(ch["tab"])
    sm = smile_slopes(ch["tab"], ch["dates"])
    tape43 = build_alltape(ch)
    code_of = {pd.Timestamp(d): i for i, d in enumerate(ch["dates"])}
    dc43 = np.asarray(
        [code_of[pd.Timestamp(d)] for d in tape43["dates"]], dtype=np.int64
    )
    gate_43(tape43, keys, cols, dc43)

    # ---- book (a): proposal 32's 11:00 crossed exit book -------------------
    panel, ent, dates = p32.load_panel()
    assert len(dates) == GATE_N_DAYS, (len(dates), GATE_N_DAYS)
    tape = p32.build_tape(panel, ent, dates)
    dc_a = np.asarray([code_of[pd.Timestamp(d)] for d in dates], dtype=np.int64)
    kc_a = tape["Kc"]
    kp_a = tape["Kp"]
    qc = leg_lookup(keys, cols, dc_a, SESSION_1100, kc_a, True)
    qp = leg_lookup(keys, cols, dc_a, SESSION_1100, kp_a, False)
    bid_a = qc["bid"] + qp["bid"]
    ask_a = qc["ask"] + qp["ask"]
    mid_a = 0.5 * (bid_a + ask_a)
    assert np.isfinite(mid_a).all(), "a stamp of the 11:00 book has no package quote"

    h_a = np.asarray([H_OF[c] for c in SESSION_1100], float)
    var0_a = (tape["sig"] * np.sqrt(h_a)[None, :]) ** 2
    realized = remaining_grid(p30, dates, SESSION_1100)
    censored = censored_grid(panel, dates, SESSION_1100)
    scale = expanding_lagged_mean(realized / var0_a, WARMUP)
    ok9 = np.isfinite(scale) & (scale > 0.0) & ~censored
    f9_a = np.where(ok9, scale, 1.0)
    print(
        f"V9's per-clock factor on the 11:00 book: the expanding lagged {WARMUP}-day "
        f"realized-over-implied scale covers {int(ok9.sum())} of {ok9.size} "
        f"stamp-cells; the other {int((~ok9).sum())} hedge with the book's own"
    )
    lab_a, q1, q2, r0_ref, r9_ref = gate_1100(
        p32, p36, tape, mid_a, ask_a, dates, var0_a, var0_a * f9_a
    )

    # ---- the three books ---------------------------------------------------
    books: dict[str, dict[str, Any]] = {
        "1100_exit_crossed": exit_book_1100(
            p32, tape, dates, f9_a, sm["slope"], keys, cols, dc_a, (q1, q2)
        ),
        "1330_hold": hold_book(
            "1330_hold", "13:30", tape43, sm["slope"], keys, cols, dc43
        ),
        "1100_hold": hold_book(
            "1100_hold", "11:00", tape43, sm["slope"], keys, cols, dc43
        ),
    }
    books["1100_exit_crossed"]["trend_lab"] = lab_a
    vend = pd.concat(
        [vendor_delta_check(books[b]) for b in BOOK_NAMES], ignore_index=True
    )
    fallback = {b: fill_inputs(books[b]) for b in BOOK_NAMES}
    for b in BOOK_NAMES:
        gate_leg_identity(books[b])
    dev0 = float(
        np.max(
            np.abs(
                book_pnl(
                    variant_delta(REF_BOOK, books["1100_exit_crossed"]),
                    books["1100_exit_crossed"],
                )["prem"]
                - r0_ref["exit_crossed"]
            )
        )
    )
    dev9 = float(
        np.max(
            np.abs(
                book_pnl(
                    variant_delta(REF_V9, books["1100_exit_crossed"]),
                    books["1100_exit_crossed"],
                )["prem"]
                - r9_ref["exit_crossed"]
            )
        )
    )
    assert dev0 < EXACT_TOL and dev9 < EXACT_TOL, (dev0, dev9)
    print(
        f"GATE 6  this script's vectorised D0 and D9 books equal proposal 36's own "
        f"scalar-engine books on all {len(dates)} days to {dev0:.3e} / {dev9:.3e}"
    )
    print("\nGATES PASSED - nothing above this line is new")

    show(
        vend,
        "the chain's own delta column, leg by leg, before any fallback: the sign "
        "convention is call in [0, 1] and put in [-1, 0], and the package delta is "
        "their sum",
    )
    write(vend, "vendor_delta_check.csv", "the vendor delta against our own")

    for b in BOOK_NAMES:
        bk = books[b]
        f = fallback[b]
        print(
            f"\n{b}: {len(bk['dates'])} sessions x {len(bk['clocks'])} stamps "
            f"({bk['clocks'][0]}..{bk['clocks'][-1]}), {int(bk['valid'].sum())} priced; "
            f"fallbacks on {f['n_cells']} stamp-cells - leg volatility "
            f"{f['n_fallback_leg_vol_c']} call / {f['n_fallback_leg_vol_p']} put, "
            f"vendor delta {f['n_fallback_vendor_delta_c']} call / "
            f"{f['n_fallback_vendor_delta_p']} put, smile slope "
            f"{f['n_fallback_slope']}, V9 factor {f['n_fallback_v9']}; shape terciles "
            f"{bk['trend_cuts'][0]:.6f} / {bk['trend_cuts'][1]:.6f}"
        )

    # ---- the smile slope ---------------------------------------------------
    slopes = slope_table(sm, tape43["S"], pd.DatetimeIndex(tape43["dates"]), dc43)
    write(slopes, "smile_slope_by_clock.csv", "the local smile slope, clock by clock")
    for sample in ("deck_in_sample", "unseen", "all"):
        show(
            slopes[slopes["sample"] == sample],
            f"smile_slope_by_clock.csv  |  sample {sample}  |  b = d(total vol) / "
            f"d ln(K/S), OLS on the {SMILE_EACH_SIDE} nearest listed strikes each "
            f"side of the spot; dsigma/dS = -b / S per index point",
        )

    # ---- stage 1: every (book, variant) book --------------------------------
    jobs = [(b, v) for b in BOOK_NAMES for v in VARIANTS]
    n_workers = max(1, min(len(jobs), (os.cpu_count() or 1)))
    t0 = time.time()
    res: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_books, initargs=(books,)
    ) as pool:
        for b, v, out in pool.map(_returns_task, jobs):
            res[(b, v)] = out
    print(
        f"\n{len(jobs)} (book, variant) books built on {n_workers} worker processes "
        f"({len(BOOK_NAMES)} books x {len(VARIANTS)} volatility/delta grids): "
        f"{time.time() - t0:.1f} s"
    )

    # ---- how far each variant moves the hedge ------------------------------
    drows: list[dict[str, Any]] = []
    for b in BOOK_NAMES:
        bk = books[b]
        base = res[(b, REF_BOOK)]["delta"]
        for v in VARIANTS:
            d = res[(b, v)]["delta"] - base
            for j, clock in enumerate(bk["clocks"]):
                sel = bk["valid"]
                drows.append(
                    {
                        "book": b,
                        "variant": v,
                        "clock": clock,
                        "h_rem": H_OF[clock],
                        "held": bool(bk["held"][j] > 0.0),
                        "n_days": int(sel.sum()),
                        "mean_abs_delta_D0": float(np.mean(np.abs(base[sel, j]))),
                        "mean_delta_diff": float(np.mean(d[sel, j])),
                        "mean_abs_delta_diff": float(np.mean(np.abs(d[sel, j]))),
                        "max_abs_delta_diff": float(np.max(np.abs(d[sel, j]))),
                    }
                )
    ddiff = pd.DataFrame(drows)
    write(ddiff, "delta_diff_by_clock.csv", "how far each variant moves the hedge")
    for b in BOOK_NAMES:
        show(
            ddiff[ddiff["book"] == b],
            f"delta_diff_by_clock.csv  |  book {b}  |  the hedge delta of each "
            f"variant minus the book's, in index units per straddle",
        )

    # ---- stage 2: the statistics -------------------------------------------
    t0 = time.time()
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for b in BOOK_NAMES:
        bk = books[b]
        refs = {ref: {u: res[(b, ref)][u] for u in UNITS} for ref in (REF_BOOK, REF_V9)}
        for v in VARIANTS:
            tasks.append(
                (
                    b,
                    v,
                    {
                        "res": res[(b, v)],
                        "refs": refs,
                        "dates": bk["dates"],
                        "samples": bk["samples"],
                        "trend_lab": bk["trend_lab"],
                        "fallback": fallback[b],
                    },
                )
            )
    vrows: list[dict[str, Any]] = []
    prows: list[dict[str, Any]] = []
    brows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for blk in pool.map(_stats_task, tasks):
            vrows.extend(blk["variants"])
            prows.extend(blk["pairs"])
            brows.extend(blk["bootstrap"])
    print(
        f"\n{len(tasks)} (book, variant) statistics blocks on {n_workers} worker "
        f"processes: {time.time() - t0:.1f} s"
    )

    order = {v: i for i, v in enumerate(VARIANTS)}
    variants_df = pd.DataFrame(vrows)
    variants_df = variants_df.sort_values(
        by=["book", "sample", "unit", "variant"],
        key=lambda s: s.map(order) if s.name == "variant" else s,
    ).reset_index(drop=True)
    pairs_df = pd.DataFrame(prows)
    boot_df = pd.DataFrame(brows)
    for df in (pairs_df, boot_df):
        df["_o"] = df["variant"].map(order)
    pairs_df = (
        pairs_df.sort_values(["book", "reference", "sample", "unit", "_o"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )
    boot_df = (
        boot_df.sort_values(["book", "reference", "sample", "unit", "_o"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )

    write(variants_df, "variants.csv", "every variant, scored on every book")
    write(pairs_df, "pairs.csv", "the paired daily difference against D0 and D9")
    write(boot_df, "bootstrap.csv", "the bootstrap interval of the Sharpe difference")

    for b in BOOK_NAMES:
        for sample in books[b]["samples"]:
            for unit in UNITS:
                sub = variants_df[
                    (variants_df["book"] == b)
                    & (variants_df["sample"] == sample)
                    & (variants_df["unit"] == UNIT_NAME[unit])
                ].drop(columns=["book", "sample", "unit", "note"])
                show(
                    sub,
                    f"variants.csv  |  book {b}  |  sample {sample}  |  "
                    f"{UNIT_NAME[unit]}",
                )
    print(
        f"\nmean_cost_reported is the repo's {HEDGE_COST_BP} bp charge on "
        f"S x |delta traded| over the entry hedge, every rebalance and the unwind.  "
        f"It is REPORTED on every row and also CHARGED, in mean_charged and "
        f"Sharpe_ann_charged; every other number in these tables is gross of it, "
        f"exactly as the published books are."
    )

    for b in BOOK_NAMES:
        for ref in (REF_BOOK, REF_V9):
            for sample in books[b]["samples"]:
                for unit in UNITS:
                    sub = pairs_df[
                        (pairs_df["book"] == b)
                        & (pairs_df["reference"] == ref)
                        & (pairs_df["sample"] == sample)
                        & (pairs_df["unit"] == UNIT_NAME[unit])
                    ].drop(columns=["book", "reference", "sample", "unit"])
                    show(
                        sub,
                        f"pairs.csv  |  book {b}  |  against {ref}  |  sample "
                        f"{sample}  |  {UNIT_NAME[unit]}",
                    )
    for b in BOOK_NAMES:
        for ref in (REF_BOOK, REF_V9):
            for sample in books[b]["samples"]:
                for unit in UNITS:
                    sub = boot_df[
                        (boot_df["book"] == b)
                        & (boot_df["reference"] == ref)
                        & (boot_df["sample"] == sample)
                        & (boot_df["unit"] == UNIT_NAME[unit])
                    ].drop(columns=["book", "reference", "sample", "unit"])
                    show(
                        sub,
                        f"bootstrap.csv  |  book {b}  |  against {ref}  |  sample "
                        f"{sample}  |  {UNIT_NAME[unit]}  |  {BOOT_B} circular block "
                        f"draws, block {BOOT_BLOCK}, seed {BOOT_SEED}",
                    )

    # ---- the gate ----------------------------------------------------------
    grows: list[dict[str, Any]] = []
    for v in GATE_VARIANTS:
        cells: list[dict[str, Any]] = []
        for b, sample in GATE_CELLS:
            row = boot_df[
                (boot_df["book"] == b)
                & (boot_df["reference"] == REF_V9)
                & (boot_df["sample"] == sample)
                & (boot_df["unit"] == UNIT_NAME[GATE_UNIT])
                & (boot_df["variant"] == v)
            ]
            assert len(row) == 1, (v, b, sample, len(row))
            r = row.iloc[0]
            cells.append(
                {
                    "variant": v,
                    "book": b,
                    "sample": sample,
                    "unit": UNIT_NAME[GATE_UNIT],
                    "n": int(r["n"]),
                    "dSharpe_vs_D9": float(r["dSharpe"]),
                    "boot_lo": float(r["boot_lo"]),
                    "boot_hi": float(r["boot_hi"]),
                    "pct_draws_positive": float(r["pct_draws_positive"]),
                    "cell_positive_excludes_zero": bool(float(r["boot_lo"]) > 0.0),
                }
            )
        both = all(c["cell_positive_excludes_zero"] for c in cells)
        for c in cells:
            c["verdict"] = "improves on D9" if both else "no evidence"
            grows.append(c)
    gate_df = pd.DataFrame(grows)
    write(gate_df, "gate.csv", "the verdict, cell by cell")
    show(
        gate_df,
        f"gate.csv  |  a variant improves on {REF_V9} only where its Sharpe-difference "
        f"interval excludes zero on the POSITIVE side on BOTH the 11:00 era exit book "
        f"AND the 13:30 hold book on the unseen sessions, "
        f"{UNIT_NAME[GATE_UNIT]}",
    )
    n_cells = int(len(gate_df))
    n_pos = int(gate_df["cell_positive_excludes_zero"].sum())
    n_improve = int(
        gate_df.loc[gate_df["verdict"] == "improves on D9", "variant"].nunique()
    )
    alpha = 1.0 - (CI_PCT[1] - CI_PCT[0]) / 100.0
    print(
        f"\nCELLS TRIED: {len(GATE_VARIANTS)} causal variants x {len(GATE_CELLS)} "
        f"books = {n_cells} gate reads ({REF_V9} is the reference, not a test; the "
        f"{len(ALPHA_VARIANTS)} D5 rows are ALPHA, reported and excluded).  "
        f"{n_pos} of {n_cells} cells exclude zero on the positive side; "
        f"{n_improve} of {len(GATE_VARIANTS)} variants do so on BOTH books.  "
        f"At the {100.0 * alpha:.0f}% level the expectation under the null is "
        f"{n_cells * alpha:.1f} cells, and at least one positive cell arrives with "
        f"probability {1.0 - (1.0 - alpha / 2.0) ** n_cells:.3f}."
    )

    # ---- the family note ---------------------------------------------------
    ins = variants_df[
        (variants_df["book"] == "1330_hold")
        & (variants_df["sample"] == "deck_in_sample")
        & (variants_df["unit"] == UNIT_NAME["pts"])
        & (variants_df["variant"].isin(GATE_VARIANTS))
    ]
    pick = str(ins.loc[ins["Sharpe_ann"].idxmax(), "variant"])
    oos = boot_df[
        (boot_df["book"] == "1330_hold")
        & (boot_df["reference"] == REF_V9)
        & (boot_df["sample"] == "unseen")
        & (boot_df["unit"] == UNIT_NAME["pts"])
        & (boot_df["variant"] == pick)
    ].iloc[0]
    oos_pair = pairs_df[
        (pairs_df["book"] == "1330_hold")
        & (pairs_df["reference"] == REF_V9)
        & (pairs_df["sample"] == "unseen")
        & (pairs_df["unit"] == UNIT_NAME["pts"])
        & (pairs_df["variant"] == pick)
    ].iloc[0]
    print(
        f"\nFAMILY  chosen on the deck in-sample per-contract Sharpe of the 13:30 "
        f"hold book ({ins['Sharpe_ann'].max():.6f}, over "
        f"{len(GATE_VARIANTS)} causal variants): {pick}.  On the UNSEEN sessions "
        f"against {REF_V9}, per contract: dSharpe {float(oos['dSharpe']):+.6f} "
        f"[{float(oos['boot_lo']):+.6f}, {float(oos['boot_hi']):+.6f}], paired mean "
        f"{float(oos_pair['mean_diff']):+.6f} index points per day, HAC t "
        f"{float(oos_pair['hac_t_diff']):+.6f}, days improved "
        f"{float(oos_pair['frac_days_improved']):.4f}, days unchanged "
        f"{float(oos_pair['frac_days_unchanged']):.4f}."
    )

    # ---- one paragraph of numbers ------------------------------------------
    print("\n--- the paragraph, numbers only")
    for b in BOOK_NAMES:
        sub = ddiff[(ddiff["book"] == b) & ddiff["held"]]
        own = float(sub.loc[sub["variant"] == REF_BOOK, "mean_abs_delta_D0"].mean())
        sizes = (
            sub[sub["variant"] != REF_BOOK]
            .groupby("variant")["mean_abs_delta_diff"]
            .mean()
            .reindex([v for v in VARIANTS if v != REF_BOOK])
        )
        big = str(sizes.idxmax())
        print(
            f"  {b}: mean |delta| of the book over the held stamps {own:.6f}; "
            "mean |delta difference| vs D0 by variant "
            + ", ".join(f"{k} {v:.6f}" for k, v in sizes.items())
            + f"; largest {big} {float(sizes.max()):.6f} "
            f"({100.0 * float(sizes.max()) / own:.2f}% of the book's own delta)"
        )
        per_clock = (
            sub[sub["variant"] == big]
            .set_index("clock")["mean_abs_delta_diff"]
            .reindex([c for c in books[b]["clocks"]])
        )
        print(
            f"    {big} by clock: "
            + ", ".join(f"{c} {float(per_clock[c]):.6f}" for c in per_clock.index)
        )
    print(
        f"  unseen-sample verdicts: {n_pos} of {n_cells} gate cells exclude zero on "
        f"the positive side, {n_improve} variants on both books; on the 13:30 hold "
        f"book's unseen sessions per contract the cells read "
        + ", ".join(
            f"{r['variant']} {r['dSharpe_vs_D9']:+.6f} "
            f"[{r['boot_lo']:+.6f}, {r['boot_hi']:+.6f}]"
            for _, r in gate_df[gate_df["book"] == "1330_hold"].iterrows()
        )
    )

    print(f"\nwrote {OUT}")
    print(f"total runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
