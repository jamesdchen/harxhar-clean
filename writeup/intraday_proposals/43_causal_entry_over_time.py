"""43 - the premium by entry clock over time, and a causal entry clock.

Forecast-free.  The instrument is the book of record's own leg: sell the
nearest-OTM SPX 0DTE straddle at an entry clock c, Black-76 delta-hedge on
the index every 30 minutes, entry at the quoted bid.  Two terminal
treatments:

  hold      carry to the 4pm cash settlement; the settlement costs nothing
            to receive, and the hedge runs through the last bar.
  flatten   buy BOTH legs back at 15:30 at the quoted ask and flatten the
            futures at 15:30.  That is the crossed-quoted EXIT book of
            live/ibkr/parity.py.

Part A maps the premium: every entry clock in 10:00..15:00 by calendar
period (2024 split at 04-30, the deck's last scored day), in premium units
and in index points per contract, with the implied-over-realised
remaining-window variance ratio at each clock.

Part B asks whether the entry clock can be CHOSEN causally: proposal 26's
selector (each session, the daytime clock with the best trailing Sharpe of
the hold-to-close always-short book on PRIOR sessions, shift 1, min 63,
warm-up flat), at three lookbacks, against fixed 11:00, fixed 13:30 and the
ex-post best fixed clock of each year.

Part C is one paragraph of numbers.

Gates, in order:
  GATE V  the vectorised package-volatility inversion equals
          live.ibkr.pricing.invert_total_vol on every cell of the tape.
  GATE 0  the builder, run at the 11:00 clock, reproduces proposal 41's
          chain-built book day by day on its 413 unseen sessions (strikes,
          spot, entry, bid, 15:30 asks, both hedges, both books) and its
          in-sample mid-inverted Sharpes 2.2530773 whole / 2.5133605 era.
  GATE 1  that book's 413 unseen sessions: mean +0.025014 premium units,
          HAC t 1.243, per-contract mean -0.0205 index points.
  GATE 2  (cross-check, reported not asserted) the per-clock hold-to-close
          always-short Sharpes against the vendor-implied index CSVs
          results/atm_straddle_intraday_holdclose/rule_by_strategy_dh/
          <hhmm>/rule_by_strategy_always_short.csv; the gap is the
          mid-inversion difference.
  GATE B  a lookback "improves" only if the circular-block bootstrap CI of
          its paired daily difference against fixed 11:00 excludes zero on
          the positive side on the UNSEEN sample, for the per-contract
          series.  3 lookbacks x 2 treatments = 6 cells.

Run:  python writeup/intraday_proposals/43_causal_entry_over_time.py
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import atm_straddle_lib as asl  # noqa: E402
from live.ibkr.pricing import (  # noqa: E402
    INVERT_VOL_HI,
    INVERT_VOL_ITERS,
    INVERT_VOL_LO,
    INVERT_VOL_TOL,
    invert_total_vol,
)

REPO = asl.find_repo(Path(__file__).resolve().parent)
HOLD = REPO / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "43"
REF41 = HOLD / "proposals" / "41"
GROK = HOLD / "rule_by_strategy_dh"
CHAIN = REPO / "data" / "spxw_chain.parquet"
SPOT = REPO / "data" / "spxw_spot.parquet"
GSPC = HOLD / "cache" / "gspc_ohlc.parquet"

#: the 30-minute stamps the tape walks; 16:00 is a settlement print, never a quote.
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
#: the entry clocks the study maps: 10:00..15:00.
ENTRIES: tuple[str, ...] = CLOCKS[:-1]
REF_CLOCK = "11:00"  # proposal 41's clock, and part B's fixed benchmark
ALT_CLOCK = "13:30"  # part B's second fixed benchmark
TREATMENTS: tuple[str, ...] = ("hold", "flatten")
UNITS: tuple[tuple[str, str], ...] = (
    ("units", "premium"),
    ("pts", "index points per contract"),
)

DECK_START = "2020-01-03"
DECK_END = "2024-04-30"  # the deck's last scored day
OOS_START = "2024-05-01"
OOS_END = "2025-12-31"

ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))

# Stated grids.
MIN_DAYS = 63  # proposal 26's minimum trailing sample
LOOKBACKS: tuple[tuple[str, int], ...] = (
    ("expanding", 0),  # 0 = expanding, proposal 26's own
    ("rolling 126", 126),
    ("rolling 252", 252),
)
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5
HEDGE_COST_BP = 0.5e-4  # 0.5 bp of S on every index unit traded
T_GATE = 2.0  # part C asks for t > 2
MORNING: tuple[str, ...] = ("10:00", "10:30", "11:00", "11:30", "12:00")
AFTERNOON: tuple[str, ...] = ("12:30", "13:00", "13:30", "14:00", "14:30", "15:00")

# GATE targets: proposal 41's own prints and its saved daily file.
GATE_INS_WHOLE = 2.2530773
GATE_INS_ERA = 2.5133605
GATE_ERA_START = "2022-05-16"
GATE_OOS_N = 413
GATE_OOS_MEAN = 0.025014
GATE_OOS_T = 1.243
GATE_OOS_PTS = -0.0205
GATE_SHARPE_TOL = 1e-6
GATE_DAY_TOL = 1e-12
GATE_MEAN_TOL = 5e-7
GATE_T_TOL = 5e-4
GATE_PTS_TOL = 5e-5


# ----------------------------------------------------------------- pricing --
def _cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(np.asarray(z, float) / np.sqrt(2.0)))


def pkg_price_vec(
    s: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """Black-76 package price (r = 0, forward = spot) for a strictly positive s."""
    d1c = (np.log(f / kc) + 0.5 * s * s) / s
    d1p = (np.log(f / kp) + 0.5 * s * s) / s
    return f * _cdf(d1c) - kc * _cdf(d1c - s) + kp * _cdf(-(d1p - s)) - f * _cdf(-d1p)


def pkg_delta_vec(
    s: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """Black-76 package delta N(d1c) + N(d1p) - 1; zero where s is not positive.

    The branch live.ibkr.pricing.package_delta takes: at the settlement there
    is nothing left to hedge.
    """
    s = np.asarray(s, float)
    f = np.asarray(f, float)
    kc = np.broadcast_to(np.asarray(kc, float), np.broadcast_shapes(s.shape, f.shape))
    kp = np.broadcast_to(np.asarray(kp, float), np.broadcast_shapes(s.shape, f.shape))
    ok = (
        np.isfinite(s)
        & (s > 0.0)
        & np.isfinite(f)
        & (f > 0.0)
        & np.isfinite(kc)
        & (kc > 0.0)
        & np.isfinite(kp)
        & (kp > 0.0)
    )
    ss = np.where(ok, s, 1.0)
    ff = np.where(ok, f, 1.0)
    kcc = np.where(ok, kc, 1.0)
    kpp = np.where(ok, kp, 1.0)
    d1c = (np.log(ff / kcc) + 0.5 * ss * ss) / ss
    d1p = (np.log(ff / kpp) + 0.5 * ss * ss) / ss
    return np.where(ok, _cdf(d1c) + _cdf(d1p) - 1.0, 0.0)


def invert_total_vol_vec(
    f: np.ndarray, kc: np.ndarray, kp: np.ndarray, mid: np.ndarray
) -> np.ndarray:
    """live.ibkr.pricing.invert_total_vol, array at a time.

    Same bracket [1e-8, 1], same halving count, same early stop: the bracket
    width halves identically for every cell, so the whole array stops on the
    iteration the scalar engine stops on.  GATE V asserts the two agree on
    every cell of the tape, exactly.
    """
    f = np.asarray(f, float)
    kc = np.asarray(kc, float)
    kp = np.asarray(kp, float)
    mid = np.asarray(mid, float)
    good = (
        np.isfinite(f)
        & (f > 0.0)
        & np.isfinite(mid)
        & np.isfinite(kc)
        & (kc > 0.0)
        & np.isfinite(kp)
        & (kp > 0.0)
    )
    ff = np.where(good, f, 1.0)
    kcc = np.where(good, kc, 1.0)
    kpp = np.where(good, kp, 1.0)
    intrinsic = np.maximum(ff - kcc, 0.0) + np.maximum(kpp - ff, 0.0)
    good &= mid > intrinsic
    hi_price = pkg_price_vec(np.full(ff.shape, INVERT_VOL_HI), ff, kcc, kpp)
    good &= hi_price >= mid
    lo = np.full(ff.shape, INVERT_VOL_LO)
    hi = np.full(ff.shape, INVERT_VOL_HI)
    for _ in range(INVERT_VOL_ITERS):
        m = 0.5 * (lo + hi)
        below = pkg_price_vec(m, ff, kcc, kpp) < mid
        lo = np.where(below, m, lo)
        hi = np.where(below, hi, m)
        if float(np.max(hi - lo)) < INVERT_VOL_TOL:
            break
    return np.where(good, 0.5 * (lo + hi), np.nan)


# ------------------------------------------------------------- statistics ---
def sharpe(x: "pd.Series[float] | np.ndarray") -> float:
    v = pd.Series(x).dropna()
    if len(v) < 2:
        return float("nan")
    sd = float(v.std(ddof=1))
    return float(v.mean()) / sd * ANN if sd > 0 else float("nan")


def maxdd(x: "pd.Series[float] | np.ndarray") -> float:
    """Worst peak-to-trough of the cumulative SUM path, in the series' units."""
    v = pd.Series(x).dropna().sort_index().to_numpy(float)
    if v.size < 1:
        return float("nan")
    path = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def stats_row(r: "pd.Series[float]") -> dict[str, Any]:
    v = pd.Series(r).dropna().sort_index()
    if v.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "Sharpe_ann": float("nan"),
            "t_hac": float("nan"),
            "t_lag": 0,
            "MaxDD": float("nan"),
            "worst": float("nan"),
            "worst_date": "",
        }
    t, lag = asl.newey_west_t(v)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size >= 2 else float("nan"),
        "Sharpe_ann": sharpe(v),
        "t_hac": t,
        "t_lag": lag,
        "MaxDD": maxdd(v),
        "worst": float(v.min()),
        "worst_date": str(pd.Timestamp(v.idxmin()).date()),
    }


def boot_ci(d: "pd.Series[float]") -> tuple[float, float]:
    """Circular-block bootstrap percentile CI of the mean of d."""
    v = pd.Series(d).dropna().to_numpy(float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    idx = asl.circular_block_bootstrap_idx(rng, len(v), BOOT_BLOCK, BOOT_B)
    draws = v[idx].mean(axis=1)
    return (
        float(np.percentile(draws, CI_LO_PCT)),
        float(np.percentile(draws, CI_HI_PCT)),
    )


def paired_row(
    label: str, a: "pd.Series[float]", b: "pd.Series[float]"
) -> dict[str, Any]:
    """a - b: mean, HAC t and the bootstrap CI of the mean difference."""
    d = (pd.Series(a) - pd.Series(b)).dropna()
    t, lag = asl.newey_west_t(d)
    lo, hi = boot_ci(d)
    return {
        "difference": label,
        "n": int(d.size),
        "mean_diff": float(d.mean()) if d.size else float("nan"),
        "t_hac": t,
        "t_lag": lag,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero_positive": bool(np.isfinite(lo) and lo > 0.0),
        "d_Sharpe": sharpe(a) - sharpe(b),
    }


# ------------------------------------------------------------ the sessions --
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


def half_sessions(stamp: pd.DataFrame) -> pd.DatetimeIndex:
    """Expirations whose 15:30 row has already expired (hours_to_expiration <= 0).

    The chain's own ``early_close`` column is degenerate and is never used.
    """
    hte = pd.read_parquet(
        CHAIN,
        columns=["expiration", "timestamp", "hours_to_expiration"],
        filters=[("timestamp", "in", sorted(pd.DatetimeIndex(stamp[CLOSE]).unique()))],
    )
    med = hte.groupby("expiration")["hours_to_expiration"].median()
    return pd.DatetimeIndex(pd.to_datetime(med.index[med <= 0])).normalize()


# ------------------------------------------------------ chain-based build ---
def build_chain(stamp: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """The straddle, the spot and the mid-inverted volatility at EVERY clock.

    One pyarrow-filtered read of data/spxw_chain.parquet covers every stamp of
    every scored session.  At each stamp the nearest-OTM straddle is re-picked
    with the research's guards (asl.pick_nearest_otm_guarded) and its midpoint
    is bisected for the total volatility over the remaining session.  That is
    live.ibkr.parity.replay_day with iv_mode="mid", generalised from proposal
    41's 11:00 entry to every entry clock: the volatility at a stamp prices the
    delta of whatever legs are held, and the legs held from entry clock c are
    the ones picked at c.

    Also returns, per (session, entry clock), the 15:30 quoted ask of the two
    legs picked at that clock -- the flatten treatment's fill.
    """
    ts_of = {(d, hh): pd.Timestamp(stamp.loc[d, hh]) for d in dates for hh in CLOCKS}
    back = {v: k for k, v in ts_of.items()}
    ts_list = sorted(back)
    t0 = time.time()
    ch = pd.read_parquet(
        CHAIN,
        columns=[
            "expiration",
            "timestamp",
            "strike",
            "cp",
            "bid",
            "ask",
            "underlying_price",
        ],
        filters=[("timestamp", "in", ts_list)],
    )
    ch["strike"] = ch["strike"].astype(float)
    ch["cp"] = ch["cp"].astype(str)
    ch["mid"] = asl.quote_mid(ch["bid"], ch["ask"]).to_numpy()
    print(
        f"chain read once: {len(ts_list)} stamps ({len(dates)} sessions x "
        f"{len(CLOCKS)} clocks), {len(ch):,} rows, {time.time() - t0:.1f} s"
    )
    key_ts = pd.DataFrame(
        {"timestamp": pd.Series(ts_list, dtype=ch["timestamp"].dtype)}
    )
    key_ts["date"] = [back[pd.Timestamp(t)][0] for t in key_ts["timestamp"]]
    key_ts["hhmm"] = [back[pd.Timestamp(t)][1] for t in key_ts["timestamp"]]
    key_ts["expiration"] = key_ts["date"].astype(ch["expiration"].dtype)
    cols = ["expiration", "timestamp", "date", "hhmm"]

    live = ch[np.isfinite(ch["mid"]) & (ch["mid"] > 0.0)]
    sp = asl.stamp_spot(live, ["expiration", "timestamp"])
    t0 = time.time()
    atm, dropped = asl.pick_nearest_otm_guarded(
        live[["expiration", "timestamp", "strike", "cp", "bid", "ask", "mid"]],
        sp,
        keys=("expiration", "timestamp"),
    )
    print(f"nearest-OTM pick with the guards at every stamp: {time.time() - t0:.1f} s")
    atm = atm.merge(key_ts[cols], on=["expiration", "timestamp"], how="left")
    spot_all = (
        sp.rename("S_stamp")
        .reset_index()
        .merge(key_ts[cols], on=["expiration", "timestamp"], how="inner")
    )
    refused = (
        dropped.merge(key_ts[cols], on=["expiration", "timestamp"], how="left")[
            ["date", "hhmm", "reason", "n_live"]
        ]
        if len(dropped)
        else pd.DataFrame(columns=["date", "hhmm", "reason", "n_live"])
    )
    idx = pd.DatetimeIndex(dates)

    def piv(frame: pd.DataFrame, col: str) -> np.ndarray:
        return (
            frame.pivot_table(index="date", columns="hhmm", values=col, aggfunc="first")
            .reindex(index=idx, columns=list(CLOCKS))
            .to_numpy(float)
        )

    s_grid = piv(spot_all, "S_stamp")
    k_c = piv(atm, "K_c")
    k_p = piv(atm, "K_p")
    entry = piv(atm, "entry")
    bid = piv(atm, "bid_entry")
    tot = invert_total_vol_vec(s_grid, k_c, k_p, entry)

    # GATE V: the vectorised inversion against the live engine's scalar one
    t0 = time.time()
    ref = np.array(
        [
            [
                invert_total_vol(s_grid[i, k], k_c[i, k], k_p[i, k], entry[i, k])
                for k in range(len(CLOCKS))
            ]
            for i in range(len(idx))
        ]
    )
    both_nan = np.isnan(tot) & np.isnan(ref)
    dev = float(np.max(np.where(both_nan, 0.0, np.abs(tot - ref))))
    assert dev == 0.0, dev
    print(
        f"GATE V  vectorised package-volatility inversion vs "
        f"live.ibkr.pricing.invert_total_vol on all {tot.size} cells: max abs "
        f"difference {dev:.1e}, NaN cells {int(np.isnan(tot).sum())} both sides "
        f"({time.time() - t0:.1f} s for the scalar reference)"
    )

    # 15:30 quoted asks of every entry clock's legs, from the 15:30 slice
    close_ts = {ts_of[(d, CLOSE)] for d in dates}
    ch15 = ch[ch["timestamp"].isin(close_ts)][
        ["expiration", "timestamp", "strike", "cp", "bid", "ask"]
    ].copy()
    b15 = ch15["bid"].astype(float)
    a15 = ch15["ask"].astype(float)
    ch15["leg_ask"] = a15.mask((b15 == 0.0) & (a15 == 0.0))  # bid == ask == 0: no quote
    ch15 = ch15.merge(key_ts[cols], on=["expiration", "timestamp"], how="left")
    ask_map = (
        ch15.dropna(subset=["leg_ask"])
        .drop_duplicates(["date", "strike", "cp"])
        .set_index(["date", "strike", "cp"])["leg_ask"]
    )
    n_e = len(ENTRIES)
    ask_c = np.full((len(idx), n_e), np.nan)
    ask_p = np.full((len(idx), n_e), np.nan)
    for j in range(n_e):
        ask_c[:, j] = ask_map.reindex(
            pd.MultiIndex.from_arrays([idx, k_c[:, j], ["C"] * len(idx)])
        ).to_numpy(float)
        ask_p[:, j] = ask_map.reindex(
            pd.MultiIndex.from_arrays([idx, k_p[:, j], ["P"] * len(idx)])
        ).to_numpy(float)

    gspc = pd.read_parquet(GSPC)
    gspc.index = pd.to_datetime(gspc.index)
    s_close = gspc["close"].reindex(idx).to_numpy(float)
    return {
        "dates": idx,
        "S": s_grid,
        "K_c": k_c,
        "K_p": k_p,
        "entry": entry,
        "bid": bid,
        "tot": tot,
        "ask_c": ask_c,
        "ask_p": ask_p,
        "S_close": s_close,
        "refused": refused,
    }


def _clock_book(args: tuple[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """One entry clock's tape.  Pure numpy over (sessions x 12) arrays.

    The hedge holds +delta index units per short straddle over each 30-minute
    step from the entry clock on; the delta is the Black-76 delta of the HELD
    legs priced with the stamp's own mid-inverted total volatility.  The hold
    treatment carries the 15:30 hedge into the settlement print; the flatten
    treatment unwinds it at 15:30, which is what zeroing the last step means.
    """
    j, a = args
    s = a["S"]
    s_close = a["S_close"]
    n_d, n_k = s.shape
    kc = a["K_c"][:, j]
    kp = a["K_p"][:, j]
    entry = a["entry"][:, j]
    bid = a["bid"][:, j]
    tot = a["tot"].copy()
    tot[:, :j] = np.nan
    dlt = pkg_delta_vec(tot, s, kc[:, None], kp[:, None])
    dlt[:, :j] = 0.0
    nxt = np.full_like(s, np.nan)
    nxt[:, :-1] = s[:, 1:]
    nxt[:, -1] = s_close
    d_s = np.where(np.isfinite(s) & np.isfinite(nxt), nxt - s, 0.0)
    d_s[:, :j] = 0.0
    d_s_flat = d_s.copy()
    d_s_flat[:, -1] = 0.0
    hedge_hold = (dlt * d_s).sum(axis=1)
    hedge_flat = (dlt * d_s_flat).sum(axis=1)
    settle = np.maximum(s_close - kc, 0.0) + np.maximum(kp - s_close, 0.0)
    buyback = a["ask_c"][:, j] + a["ask_p"][:, j]
    ok = np.isfinite(entry) & (entry > 0.0) & np.isfinite(bid) & (bid > 0.0)
    nan = np.full(n_d, np.nan)

    pos_flat = dlt.copy()
    pos_flat[:, -1] = 0.0
    prev_h = np.zeros(n_d)
    prev_f = np.zeros(n_d)
    turn_h = np.zeros(n_d)
    turn_f = np.zeros(n_d)
    for k in range(j, n_k):
        turn_h += np.abs(dlt[:, k] - prev_h) * s[:, k]
        turn_f += np.abs(pos_flat[:, k] - prev_f) * s[:, k]
        prev_h = dlt[:, k]
        prev_f = pos_flat[:, k]
    turn_h += np.abs(prev_h) * s_close  # unwound against the settlement print
    turn_f += np.abs(prev_f) * s[:, -1]  # already flat at 15:30
    return {
        "hold": np.where(ok, (-(settle - bid) + hedge_hold) / entry, nan),
        "flatten": np.where(ok, (-(buyback - bid) + hedge_flat) / entry, nan),
        "hold_mid": np.where(ok, (-(settle - entry) + hedge_hold) / entry, nan),
        "entry": np.where(ok, entry, nan),
        "bid": np.where(ok, bid, nan),
        "K_c": kc,
        "K_p": kp,
        "settle_pts": settle,
        "buyback_pts": np.where(ok, buyback, nan),
        "hedge_hold_pts": np.where(ok, hedge_hold, nan),
        "hedge_flatten_pts": np.where(ok, hedge_flat, nan),
        "turnover_hold_pts": np.where(ok, turn_h, nan),
        "turnover_flatten_pts": np.where(ok, turn_f, nan),
        "cost_hold_pts": np.where(ok, HEDGE_COST_BP * turn_h, nan),
        "cost_flatten_pts": np.where(ok, HEDGE_COST_BP * turn_f, nan),
    }


def clock_books(ch: dict[str, Any]) -> dict[str, dict[str, "pd.Series[float]"]]:
    """_clock_book for all 11 entry clocks, one worker per clock."""
    idx = ch["dates"]
    arrays = {
        k: ch[k] for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = ch["S_close"]
    jobs = [(j, arrays) for j in range(len(ENTRIES))]
    t0 = time.time()
    n_workers = min(len(ENTRIES), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        res = list(pool.map(_clock_book, jobs))
    print(
        f"per-clock tapes built on {n_workers} worker processes "
        f"({len(ENTRIES)} clocks): {time.time() - t0:.1f} s"
    )
    return {
        clock: {k: pd.Series(v, index=idx) for k, v in res[j].items()}
        for j, clock in enumerate(ENTRIES)
    }


# ------------------------------------------------------------------ gates ---
def gate_zero(books: dict[str, Any], ch: dict[str, Any]) -> None:
    """Reproduce proposal 41's chain-built book at the 11:00 clock."""
    ref = pd.read_csv(REF41 / "c_oos_daily.csv", index_col=0, parse_dates=True)
    ins = pd.read_csv(REF41 / "a_insample.csv", index_col=0, parse_dates=True)
    idx = ch["dates"]
    j = ENTRIES.index(REF_CLOCK)
    b = books[REF_CLOCK]
    mine = pd.DataFrame(
        {
            "K_c": ch["K_c"][:, j],
            "K_p": ch["K_p"][:, j],
            "S_1530": ch["S"][:, -1],
            "S_close": ch["S_close"],
            "entry_mid": ch["entry"][:, j],
            "entry_bid": ch["bid"][:, j],
            "ask_c_1530": ch["ask_c"][:, j],
            "ask_p_1530": ch["ask_p"][:, j],
            "hedge_exit_pts": b["hedge_flatten_pts"].to_numpy(float),
            "hedge_hold_pts": b["hedge_hold_pts"].to_numpy(float),
            "r_book": b["flatten"].to_numpy(float),
            "r_hold": b["hold"].to_numpy(float),
        },
        index=idx,
    )
    common = idx.intersection(pd.DatetimeIndex(ref.index))
    assert len(common) == GATE_OOS_N, len(common)
    devs = {
        col: float((mine.loc[common, col] - ref.loc[common, col]).abs().max())
        for col in mine.columns
    }
    assert max(devs.values()) < GATE_DAY_TOL, devs
    print(
        f"GATE 0  the 11:00 clock reproduces proposal 41's chain build on "
        f"{len(common)} unseen sessions, column by column: "
        + ", ".join(f"{k} {v:.1e}" for k, v in devs.items())
        + f" (tolerance {GATE_DAY_TOL:.0e})"
    )
    deck = pd.DatetimeIndex(ins.index).intersection(idx)
    r = b["flatten"].reindex(deck)
    era = deck >= pd.Timestamp(GATE_ERA_START)
    s_whole = sharpe(r)
    s_era = sharpe(r[era])
    assert abs(s_whole - GATE_INS_WHOLE) < GATE_SHARPE_TOL, s_whole
    assert abs(s_era - GATE_INS_ERA) < GATE_SHARPE_TOL, s_era
    print(
        f"GATE 0  in-sample mid-inverted 11:00 flatten book Sharpe_ann "
        f"{s_whole:.7f} (target {GATE_INS_WHOLE}, n={len(deck)}) / era from "
        f"{GATE_ERA_START} {s_era:.7f} (target {GATE_INS_ERA}, "
        f"n={int(era.sum())}); hold crossed "
        f"{sharpe(b['hold'].reindex(deck)):.7f}, hold mid "
        f"{sharpe(b['hold_mid'].reindex(deck)):.7f}"
    )


def gate_one(books: dict[str, Any]) -> None:
    """The 413 unseen sessions of the 11:00 flatten book, both units."""
    b = books[REF_CLOCK]
    r = b["flatten"]
    pts = r * b["entry"]
    mask = (r.index >= pd.Timestamp(OOS_START)) & (r.index <= pd.Timestamp(OOS_END))
    v = r[mask].dropna()
    p = pts[mask].dropna()
    t, lag = asl.newey_west_t(v)
    assert int(v.size) == GATE_OOS_N, int(v.size)
    assert abs(float(v.mean()) - GATE_OOS_MEAN) < GATE_MEAN_TOL, float(v.mean())
    assert abs(t - GATE_OOS_T) < GATE_T_TOL, t
    assert abs(float(p.mean()) - GATE_OOS_PTS) < GATE_PTS_TOL, float(p.mean())
    print(
        f"GATE 1  the 11:00 flatten book on the unseen sessions: n {int(v.size)} "
        f"(target {GATE_OOS_N}), mean {float(v.mean()):+.6f} premium units "
        f"(target {GATE_OOS_MEAN}), HAC t {t:+.3f} (lag {lag}, target "
        f"{GATE_OOS_T}), per-contract mean {float(p.mean()):+.4f} index points "
        f"(target {GATE_OOS_PTS})"
    )


def gate_two(books: dict[str, Any]) -> pd.DataFrame:
    """Cross-check the per-clock hold Sharpes against the vendor-implied CSVs."""
    rows = []
    for clock in ENTRIES:
        path = GROK / clock.replace(":", "") / "rule_by_strategy_always_short.csv"
        ref = pd.read_csv(path, index_col=0).loc["all models"]
        h = books[clock]["hold"]
        hm = books[clock]["hold_mid"]
        deck = (h.index >= pd.Timestamp(DECK_START)) & (
            h.index <= pd.Timestamp(DECK_END)
        )
        s_mid = sharpe(hm[deck])
        s_x = sharpe(h[deck])
        rows.append(
            {
                "clock": clock,
                "n_index_vendor_iv": int(ref["n"]),
                "n_mine_mid_inverted": int(hm[deck].dropna().size),
                "Sharpe_mid_index": float(ref["Sharpe_ann"]),
                "Sharpe_mid_mine": s_mid,
                "gap_mid": s_mid - float(ref["Sharpe_ann"]),
                "Sharpe_crossed_index": float(ref["Sharpe_crossed"]),
                "Sharpe_crossed_mine": s_x,
                "gap_crossed": s_x - float(ref["Sharpe_crossed"]),
                "mean_index": float(ref["mean"]),
                "mean_mine_mid": float(hm[deck].mean()),
            }
        )
    out = pd.DataFrame(rows)
    print(
        "\nGATE 2 (cross-check, reported not asserted)  hold-to-close always-short "
        f"Sharpe by entry clock on the deck window {DECK_START}..{DECK_END}: this "
        "build is mid-inverted, the index CSVs are vendor-implied, and the gap is "
        "the mid-inversion difference"
    )
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    return out


# ------------------------------------------------------------ the series ----
def clock_series(
    books: dict[str, Any], clock: str, treat: str, unit: str, kind: str
) -> "pd.Series[float]":
    """One clock's daily series: P&L or hedge cost, premium units or points."""
    b = books[clock]
    s = b[treat] if kind == "pnl" else b[f"cost_{treat}_pts"] / b["entry"]
    return s * b["entry"] if unit == "pts" else s


def period_masks(idx: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """2020, 2021, 2022, 2023, 2024H1 (to the deck's last day), 2024H2, 2025."""
    y = idx.year
    out: dict[str, np.ndarray] = {}
    for yr in (2020, 2021, 2022, 2023):
        out[str(yr)] = np.asarray(y == yr)
    out["2024H1"] = np.asarray((y == 2024) & (idx <= pd.Timestamp(DECK_END)))
    out["2024H2"] = np.asarray((y == 2024) & (idx > pd.Timestamp(DECK_END)))
    out["2025"] = np.asarray(y == 2025)
    out["deck in-sample"] = np.asarray(idx <= pd.Timestamp(DECK_END))
    out["unseen"] = np.asarray(idx >= pd.Timestamp(OOS_START))
    out["all"] = np.ones(len(idx), bool)
    return out


# ----------------------------------------------------------------- part A ---
def part_a_clock_year(books: dict[str, Any], idx: pd.DatetimeIndex) -> pd.DataFrame:
    masks = period_masks(idx)
    rows = []
    for clock in ENTRIES:
        for treat in TREATMENTS:
            for unit, uname in UNITS:
                s = clock_series(books, clock, treat, unit, "pnl")
                for pname, m in masks.items():
                    rows.append(
                        {
                            "clock": clock,
                            "treatment": treat,
                            "period": pname,
                            "unit": uname,
                            **stats_row(s[m]),
                        }
                    )
    return pd.DataFrame(rows)


def part_a_matrix(tab: pd.DataFrame, uname: str, field: str) -> pd.DataFrame:
    sub = tab.loc[tab["unit"] == uname]
    return pd.concat(
        {
            treat: sub.loc[sub["treatment"] == treat]
            .pivot(index="clock", columns="period", values=field)
            .reindex(index=list(ENTRIES))
            for treat in TREATMENTS
        },
        names=["treatment"],
    )


def premium_curve(ch: dict[str, Any], idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Implied over realised remaining-window VARIANCE, by clock and period.

    Implied at clock c is the square of the total volatility bisected out of
    that stamp's straddle midpoint: the market's variance over the whole
    remaining session.  Realised is the sum of squared log returns of the
    30-minute spot tape from c through the 15:30 stamp plus the last step from
    the 15:30 spot to the settlement print -- the same steps the hedge walks.
    The ratio is the mean implied over the mean realised, so each period's
    ratio is the one the short actually earns on, not the mean of noisy
    per-day ratios.
    """
    s = ch["S"]
    nxt = np.full_like(s, np.nan)
    nxt[:, :-1] = s[:, 1:]
    nxt[:, -1] = ch["S_close"]
    step2 = np.where(np.isfinite(s) & np.isfinite(nxt), np.log(nxt / s) ** 2, np.nan)
    rv_rem = np.full_like(s, np.nan)
    for k in range(len(CLOCKS)):
        whole = np.isfinite(step2[:, k:]).all(axis=1)
        rv_rem[:, k] = np.where(whole, np.nansum(step2[:, k:], axis=1), np.nan)
    iv_rem = ch["tot"] ** 2
    rows = []
    for pname, m in period_masks(idx).items():
        for k, clock in enumerate(ENTRIES):
            a = iv_rem[m, k]
            b = rv_rem[m, k]
            ok = np.isfinite(a) & np.isfinite(b)
            mi = float(np.mean(a[ok])) if ok.any() else float("nan")
            mr = float(np.mean(b[ok])) if ok.any() else float("nan")
            rows.append(
                {
                    "period": pname,
                    "clock": clock,
                    "n": int(ok.sum()),
                    "implied_var_rem_mean": mi,
                    "realised_var_rem_mean": mr,
                    "implied_over_realised": mi / mr if mr > 0 else float("nan"),
                    "implied_vol_rem_pct": 100.0 * float(np.sqrt(mi)),
                    "realised_vol_rem_pct": 100.0 * float(np.sqrt(mr)),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- part B ---
def trailing_pick(hold: pd.DataFrame, window: int) -> "pd.Series[Any]":
    """Proposal 26's selector, at one lookback.

    Each session ranks the entry clocks by the trailing Sharpe of the
    hold-to-close always-short book (premium units, proposal 26's own series)
    on PRIOR sessions -- shift 1, at least MIN_DAYS of them.  window == 0 is
    proposal 26's expanding lookback; a positive window is a rolling one with
    the same minimum, so the three lookbacks warm up on the same session and
    are comparable day for day.  Ties go to the earlier clock, which is what
    DataFrame.idxmax does.
    """
    trail = pd.DataFrame(index=hold.index, columns=list(ENTRIES), dtype=float)
    for c in ENTRIES:
        s = hold[c]
        roll = (
            s.expanding(min_periods=MIN_DAYS)
            if window <= 0
            else s.rolling(window, min_periods=MIN_DAYS)
        )
        trail[c] = roll.mean().shift(1) / roll.std(ddof=1).shift(1) * ANN
    arr = trail.to_numpy(float)
    all_nan = np.isnan(arr).all(axis=1)
    out: list[Any] = []
    for i in range(len(arr)):
        out.append(pd.NA if all_nan[i] else ENTRIES[int(np.nanargmax(arr[i]))])
    return pd.Series(out, index=trail.index, dtype=object)


def pick_series(
    books: dict[str, Any],
    pick: "pd.Series[Any]",
    treat: str,
    unit: str,
    kind: str,
) -> "pd.Series[float]":
    """The rule's daily series; warm-up and a refused pick are FLAT (0.0)."""
    idx = pd.DatetimeIndex(pick.index)
    out = pd.Series(0.0, index=idx)
    for c in ENTRIES:
        m = (pick == c).to_numpy()
        if m.any():
            v = clock_series(books, c, treat, unit, kind).reindex(idx).to_numpy(float)
            out.iloc[np.where(m)[0]] = np.nan_to_num(v[m], nan=0.0)
    return out


def fixed_series(
    books: dict[str, Any], clock: str, treat: str, unit: str, kind: str
) -> "pd.Series[float]":
    """A fixed clock's daily series, flat (0.0) when the day cannot be priced."""
    return clock_series(books, clock, treat, unit, kind).fillna(0.0)


def ex_post_best(
    books: dict[str, Any], idx: pd.DatetimeIndex, treat: str, unit: str
) -> tuple["pd.Series[float]", dict[int, str]]:
    """The best fixed clock of EACH year, chosen with that year's own numbers.

    In-sample by construction: a ceiling, never a rule.
    """
    out = pd.Series(0.0, index=idx)
    chosen: dict[int, str] = {}
    for yr in sorted(set(idx.year)):
        m = np.asarray(idx.year == yr)
        best, best_s = "", -np.inf
        for c in ENTRIES:
            s = sharpe(fixed_series(books, c, treat, unit, "pnl")[m])
            if np.isfinite(s) and s > best_s:
                best, best_s = c, s
        chosen[int(yr)] = best
        out.iloc[np.where(m)[0]] = fixed_series(books, best, treat, unit, "pnl")[
            m
        ].to_numpy(float)
    return out, chosen


def cost_row(
    label: str,
    treat: str,
    gross: dict[str, "pd.Series[float]"],
    cost: dict[str, "pd.Series[float]"],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "book": label,
        "treatment": treat,
        "n": int(gross["pts"].size),
    }
    for unit, uname in UNITS:
        g = gross[unit]
        c = cost[unit]
        row[f"mean_gross_{unit}"] = float(g.mean())
        row[f"mean_cost_{unit}"] = float(c.mean())
        row[f"mean_charged_{unit}"] = float((g - c).mean())
        row[f"Sharpe_gross_{unit}"] = sharpe(g)
        row[f"Sharpe_charged_{unit}"] = sharpe(g - c)
        row[f"unit_{unit}"] = uname
    return row


# ----------------------------------------------------------------- part C ---
def part_c(a_tab: pd.DataFrame, p_tab: pd.DataFrame, curve: pd.DataFrame) -> None:
    """One paragraph of numbers: where the premium sits, and what is left."""
    pts = a_tab.loc[a_tab["unit"] == "index points per contract"]

    def m(clock: str, treat: str, period: str, field: str = "mean") -> float:
        s = pts.loc[
            (pts["clock"] == clock)
            & (pts["treatment"] == treat)
            & (pts["period"] == period),
            field,
        ]
        return float(s.iloc[0]) if len(s) else float("nan")

    print("\n--- C1. per-contract premium by clock block and period (index points)")
    for treat in TREATMENTS:
        for period in ("2020", "2021", "2022", "2023", "2024H1", "2024H2", "2025"):
            mm = [m(c, treat, period) for c in MORNING]
            aa = [m(c, treat, period) for c in AFTERNOON]
            print(
                f"  {treat:8s} {period:7s} morning 10:00-12:00 mean "
                f"{np.nanmean(mm):+.4f} pts (range {np.nanmin(mm):+.4f}.."
                f"{np.nanmax(mm):+.4f}), afternoon 12:30-15:00 "
                f"{np.nanmean(aa):+.4f} pts (range {np.nanmin(aa):+.4f}.."
                f"{np.nanmax(aa):+.4f})"
            )

    print(
        f"\n--- C2. every fixed clock on the unseen sample, per contract, with the "
        f"HAC t against zero (part C's question: any clock with t > {T_GATE})"
    )
    un = pts.loc[pts["period"] == "unseen"]
    joined = pd.concat(
        {
            "mean_pts": un.pivot(index="clock", columns="treatment", values="mean"),
            "t_hac": un.pivot(index="clock", columns="treatment", values="t_hac"),
            "Sharpe_ann": un.pivot(
                index="clock", columns="treatment", values="Sharpe_ann"
            ),
        },
        axis=1,
    ).reindex(index=list(ENTRIES))
    print(joined.to_string(float_format=lambda v: f"{v:,.4f}"))
    winners = [
        (c, t)
        for c in ENTRIES
        for t in TREATMENTS
        if m(c, t, "unseen") > 0.0 and m(c, t, "unseen", "t_hac") > T_GATE
    ]
    print(
        f"clocks positive per contract with HAC t > {T_GATE} on the unseen sample: "
        + (", ".join(f"{c} {t}" for c, t in winners) if winners else "NONE")
    )

    print("\n--- C3. 2025 per contract, clock by clock (index points)")
    print(
        "  morning:   "
        + ", ".join(
            f"{c} hold {m(c, 'hold', '2025'):+.4f} / flatten "
            f"{m(c, 'flatten', '2025'):+.4f}"
            for c in MORNING
        )
    )
    print(
        "  afternoon: "
        + ", ".join(
            f"{c} hold {m(c, 'hold', '2025'):+.4f} / flatten "
            f"{m(c, 'flatten', '2025'):+.4f}"
            for c in AFTERNOON
        )
    )
    r20 = curve.loc[curve["period"] == "2020"].set_index("clock")[
        "implied_over_realised"
    ]
    r25 = curve.loc[curve["period"] == "2025"].set_index("clock")[
        "implied_over_realised"
    ]
    print(
        "  implied over realised remaining-window variance, 2020 -> 2025: "
        + ", ".join(
            f"{c} {float(r20.get(c, np.nan)):.3f} -> {float(r25.get(c, np.nan)):.3f}"
            for c in ENTRIES
        )
    )
    g = p_tab.loc[
        (p_tab["sample"] == "unseen") & (p_tab["unit"] == "index points per contract")
    ]
    print(
        f"  causal selector on the unseen sample, per contract, against fixed "
        f"{REF_CLOCK}: "
        + "; ".join(
            f"{r['treatment']} {r['difference']} {r['mean_diff']:+.4f} pts "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            for _, r in g.iterrows()
        )
    )


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 400)
    t_start = time.time()

    stamp, px_file = session_stamps()
    half = half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    print(
        f"chain sessions {len(stamp)} {pd.Timestamp(stamp.index.min()).date()}.."
        f"{pd.Timestamp(stamp.index.max()).date()}; half sessions dropped "
        f"(hours_to_expiration <= 0 at {CLOSE}, {len(half)}): "
        + ", ".join(str(d.date()) for d in half)
        + f"; sessions scored {len(sessions)}"
    )
    ch = build_chain(stamp, sessions)
    idx = ch["dates"]
    dev_spot = float(
        np.nanmax(np.abs(ch["S"] - px_file.loc[idx, list(CLOCKS)].to_numpy(float)))
    )
    ref = ch["refused"]
    print(
        f"entry clocks {len(ENTRIES)} ({ENTRIES[0]}..{ENTRIES[-1]}); stamps refused "
        f"by the guards {len(ref)}"
        + (
            ": "
            + ", ".join(
                f"{pd.Timestamp(r['date']).date()} {r['hhmm']} {r['reason']}"
                for _, r in ref.iterrows()
            )
            if len(ref)
            else ""
        )
        + f"; chain underlying_price vs data/spxw_spot.parquet max abs diff "
        f"{dev_spot:.0e}"
    )
    books = clock_books(ch)
    print(
        "sessions priced per clock (hold / flatten): "
        + ", ".join(
            f"{c} {int(books[c]['hold'].notna().sum())}/"
            f"{int(books[c]['flatten'].notna().sum())}"
            for c in ENTRIES
        )
    )
    no_ask = {
        c: int((books[c]["hold"].notna() & books[c]["flatten"].isna()).sum())
        for c in ENTRIES
    }
    print(
        "days the held legs have no 15:30 quote (bid == ask == 0 or absent): "
        + (
            ", ".join(f"{c} {n}" for c, n in no_ask.items() if n)
            if any(no_ask.values())
            else "none"
        )
    )

    gate_zero(books, ch)
    gate_one(books)
    gate2 = gate_two(books)

    # ------------------------------------------------------------- part A --
    a_tab = part_a_clock_year(books, idx)
    print(
        "\n--- A1. always-short delta-hedged straddle by entry clock, the three "
        "headline samples (the full clock x period table is a_clock_year.csv)"
    )
    print(
        a_tab.loc[a_tab["period"].isin(["deck in-sample", "unseen", "all"])].to_string(
            index=False, float_format=lambda v: f"{v:,.6f}"
        )
    )
    for uname, field, title in (
        (
            "index points per contract",
            "Sharpe_ann",
            "A2. per-contract Sharpe_ann, clock x period",
        ),
        ("premium", "Sharpe_ann", "A3. premium-unit Sharpe_ann, clock x period"),
        (
            "index points per contract",
            "mean",
            "A4. per-contract mean (index points), clock x period",
        ),
        ("premium", "mean", "A5. premium-unit mean, clock x period"),
        (
            "index points per contract",
            "t_hac",
            "A6. per-contract HAC t, clock x period",
        ),
        (
            "index points per contract",
            "MaxDD",
            "A7. per-contract MaxDD (index points), clock x period",
        ),
        ("premium", "MaxDD", "A8. premium-unit MaxDD, clock x period"),
        ("index points per contract", "worst", "A9. worst day (index points)"),
        ("index points per contract", "n", "A10. n, clock x period"),
    ):
        print(f"\n--- {title}")
        print(
            part_a_matrix(a_tab, uname, field).to_string(
                float_format=lambda v: f"{v:,.4f}"
            )
        )
    curve = premium_curve(ch, idx)
    print("\n--- A11. implied over realised remaining-window variance, by clock")
    print(curve.to_string(index=False, float_format=lambda v: f"{v:,.8f}"))
    print("\n--- A12. implied over realised, clock x period")
    print(
        curve.pivot(index="clock", columns="period", values="implied_over_realised")
        .reindex(index=list(ENTRIES))
        .to_string(float_format=lambda v: f"{v:,.4f}")
    )

    # ------------------------------------------------------------- part B --
    hold_df = pd.DataFrame({c: books[c]["hold"] for c in ENTRIES})
    picks = pd.DataFrame(
        {name: trailing_pick(hold_df, w) for name, w in LOOKBACKS}, index=idx
    )
    counts = pd.DataFrame(
        {
            name: [int((picks[name] == c).sum()) for c in ENTRIES]
            + [int(picks[name].isna().sum())]
            for name, _ in LOOKBACKS
        },
        index=list(ENTRIES) + ["flat (warm-up)"],
    )
    print("\n--- B1. chosen clock: session counts by lookback")
    print(counts.to_string())
    q = idx.to_period("Q").astype(str)
    modal = pd.DataFrame(index=sorted(set(q)))
    for name, _ in LOOKBACKS:
        col = []
        for qq in modal.index:
            v = picks.loc[np.asarray(q == qq), name].dropna()
            col.append(
                f"{v.mode().iloc[0]} ({int((v == v.mode().iloc[0]).sum())}/{len(v)})"
                if len(v)
                else "flat (warm-up)"
            )
        modal[name] = col
    print("\n--- B2. modal chosen clock per quarter (count of that clock / sessions)")
    print(modal.to_string())
    print(
        f"warm-up sessions held flat (no clock has {MIN_DAYS} prior sessions): "
        + ", ".join(f"{name} {int(picks[name].isna().sum())}" for name, _ in LOOKBACKS)
    )

    b_rows = []
    daily: dict[str, "pd.Series[float]"] = {}
    ceilings: dict[str, dict[int, str]] = {}
    for treat in TREATMENTS:
        for unit, uname in UNITS:
            series: dict[str, "pd.Series[float]"] = {
                f"causal, {name}": pick_series(books, picks[name], treat, unit, "pnl")
                for name, _ in LOOKBACKS
            }
            for c in (REF_CLOCK, ALT_CLOCK):
                series[f"fixed {c}"] = fixed_series(books, c, treat, unit, "pnl")
            best, chosen = ex_post_best(books, idx, treat, unit)
            series["ex-post best fixed clock per year (IN-SAMPLE CEILING)"] = best
            ceilings[f"{treat}|{uname}"] = chosen
            for label, s in series.items():
                daily[f"{treat}|{uname}|{label}"] = s
                for pname, m in period_masks(idx).items():
                    b_rows.append(
                        {
                            "treatment": treat,
                            "unit": uname,
                            "book": label,
                            "period": pname,
                            **stats_row(s[m]),
                        }
                    )
    print("\n--- B3. the ex-post best fixed clock of each year (in-sample ceiling)")
    for k, chosen in ceilings.items():
        print(f"  {k:36s} " + ", ".join(f"{y} {c}" for y, c in chosen.items()))
    b_tab = pd.DataFrame(b_rows)
    print(
        "\n--- B4. the causal rule against the fixed clocks and the ceiling, by "
        "period (warm-up and unpriceable sessions are flat, so every book is "
        "scored on all 1279 sessions)"
    )
    for treat in TREATMENTS:
        for _, uname in UNITS:
            sub = b_tab.loc[(b_tab["treatment"] == treat) & (b_tab["unit"] == uname)]
            print(f"\n  {treat}, {uname}: Sharpe_ann")
            print(
                sub.pivot(
                    index="book", columns="period", values="Sharpe_ann"
                ).to_string(float_format=lambda v: f"{v:,.4f}")
            )
            print(f"  {treat}, {uname}: mean")
            print(
                sub.pivot(index="book", columns="period", values="mean").to_string(
                    float_format=lambda v: f"{v:,.6f}"
                )
            )
            print(f"  {treat}, {uname}: HAC t")
            print(
                sub.pivot(index="book", columns="period", values="t_hac").to_string(
                    float_format=lambda v: f"{v:,.3f}"
                )
            )

    samples = {
        "deck in-sample": np.asarray(idx <= pd.Timestamp(DECK_END)),
        "unseen": np.asarray(idx >= pd.Timestamp(OOS_START)),
        "all": np.ones(len(idx), bool),
    }
    p_rows = []
    for treat in TREATMENTS:
        for unit, uname in UNITS:
            base = fixed_series(books, REF_CLOCK, treat, unit, "pnl")
            for name, _ in LOOKBACKS:
                s = pick_series(books, picks[name], treat, unit, "pnl")
                for sname, m in samples.items():
                    p_rows.append(
                        {
                            "treatment": treat,
                            "unit": uname,
                            "sample": sname,
                            **paired_row(
                                f"causal {name} - fixed {REF_CLOCK}", s[m], base[m]
                            ),
                        }
                    )
    p_tab = pd.DataFrame(p_rows)
    print(
        f"\n--- B5. paired daily differences against fixed {REF_CLOCK}, HAC t and a "
        f"circular-block bootstrap CI (B={BOOT_B}, block {BOOT_BLOCK}, seed "
        f"{BOOT_SEED})"
    )
    print(p_tab.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    gate_b = p_tab.loc[
        (p_tab["sample"] == "unseen") & (p_tab["unit"] == "index points per contract")
    ]
    n_pass = int(gate_b["excludes_zero_positive"].sum())
    print(
        f"\nGATE B  {len(gate_b)} cells ({len(LOOKBACKS)} lookbacks x "
        f"{len(TREATMENTS)} treatments), per-contract, unseen sample; a cell "
        "improves only if its CI excludes zero on the positive side.  Under the "
        f"null the expectation at the 5% level is {0.05 * len(gate_b):.1f} cells."
    )
    for _, r in gate_b.iterrows():
        print(
            f"  {r['treatment']:8s} {r['difference']:34s} mean "
            f"{r['mean_diff']:+.6f} pts, HAC t {r['t_hac']:+.3f}, CI "
            f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]  -> "
            + ("IMPROVES" if r["excludes_zero_positive"] else "does not improve")
        )
    print(
        f"GATE B  {n_pass} of {len(gate_b)} cells improve -> "
        + ("PASS" if n_pass == len(gate_b) else "FAIL")
    )

    c_rows = []
    for treat in TREATMENTS:
        for name, _ in LOOKBACKS:
            c_rows.append(
                cost_row(
                    f"causal, {name}",
                    treat,
                    {
                        u: pick_series(books, picks[name], treat, u, "pnl")
                        for u, _ in UNITS
                    },
                    {
                        u: pick_series(books, picks[name], treat, u, "cost")
                        for u, _ in UNITS
                    },
                )
            )
        for c in (REF_CLOCK, ALT_CLOCK):
            c_rows.append(
                cost_row(
                    f"fixed {c}",
                    treat,
                    {u: fixed_series(books, c, treat, u, "pnl") for u, _ in UNITS},
                    {u: fixed_series(books, c, treat, u, "cost") for u, _ in UNITS},
                )
            )
    c_tab = pd.DataFrame(c_rows)
    print(
        f"\n--- B6. hedge cost at {HEDGE_COST_BP * 1e4:.1f} bp of S on every index "
        "unit traded (the entry hedge, every rebalance and the unwind), on all "
        "1279 sessions: reported, then charged"
    )
    print(c_tab.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    turn = pd.DataFrame(
        {
            f"{t}": [float(books[c][f"turnover_{t}_pts"].mean()) for c in ENTRIES]
            for t in TREATMENTS
        },
        index=list(ENTRIES),
    )
    print("\n--- B7. mean index units traded x spot, per clock (index points)")
    print(turn.to_string(float_format=lambda v: f"{v:,.3f}"))

    # ------------------------------------------------------------- part C --
    part_c(a_tab, p_tab, curve)

    a_tab.to_csv(OUT / "a_clock_year.csv", index=False)
    curve.to_csv(OUT / "a_premium_curve.csv", index=False)
    gate2.to_csv(OUT / "a_index_crosscheck.csv", index=False)
    pd.DataFrame(
        {
            f"{c}|{k}": books[c][k]
            for c in ENTRIES
            for k in ("hold", "flatten", "entry", "bid", "K_c", "K_p")
        }
    ).to_csv(OUT / "a_daily_by_clock.csv")
    picks.to_csv(OUT / "b_picks.csv")
    modal.to_csv(OUT / "b_modal_clock_by_quarter.csv")
    counts.to_csv(OUT / "b_pick_counts.csv")
    b_tab.to_csv(OUT / "b_rule_period.csv", index=False)
    p_tab.to_csv(OUT / "b_paired_vs_fixed_1100.csv", index=False)
    c_tab.to_csv(OUT / "b_hedge_cost.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "b_rule_daily.csv")
    print(
        f"\nwrote {OUT}: a_clock_year.csv, a_premium_curve.csv, "
        "a_index_crosscheck.csv, a_daily_by_clock.csv, b_picks.csv, "
        "b_modal_clock_by_quarter.csv, b_pick_counts.csv, b_rule_period.csv, "
        "b_paired_vs_fixed_1100.csv, b_hedge_cost.csv, b_rule_daily.csv"
    )
    print(f"total runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
