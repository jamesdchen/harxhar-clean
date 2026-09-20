"""41 - riding the out-of-the-money leg past the 15:30 buy-back.

Book of record (unchanged): sell the nearest-OTM SPX 0DTE straddle at 11:00,
Black-76 delta-hedge on the index every 30 minutes, buy BOTH legs back at
15:30 at the quoted ask against an entry at the bid, futures flat at 15:30.
That is the crossed-quoted EXIT book of live/ibkr/parity.py.

The proposed rule, pre-registered and with no free parameter: at 15:30 buy
back only the leg the spot has passed (call if S > K_c, put if S < K_p) and
leave the other leg to cash settlement; when the spot sits between the two
strikes both legs are out of the money, so buy back the NEARER strike and
ride the farther.  Two variants: futures flat at 15:30 (ride-flat), and
futures rebalanced once at 15:30 to the ridden leg's own Black-76 delta and
flattened at 16:00 against the settlement print (ride-hedged).

Part A prices the rule on the 865 scored in-sample days (the deck's days) on
the book's own vendor-implied hedge tape.  Part B reprices proposal 28's
last-bar jump grid for the two ride variants.  Part C rebuilds the whole
book from data/spxw_chain.parquet with the live engine's mid-inverted
volatility and reports the 2024-05-01..2025-12-31 sessions, which no part of
this research has seen.

Gates, in order:
  GATE 0   the in-sample book reproduces n = 865, crossed-quoted Sharpe
           2.2525467 whole / 2.5131322 era, hold crossed 3.582391, and the
           hedge tape agrees with live.ibkr.parity.replay_day to 1e-12.
  GATE C0  the chain-based builder reproduces the cache's strikes and entry
           premium on every in-sample day, and the book's Sharpe to within
           the mid-inversion difference of results/live_parity/parity_report.md.
  GATE A   both era bootstrap CIs (ride-flat, ride-hedged, each minus the
           book) exclude zero on the positive side.
  GATE C   the ride-vs-book era CI on the unseen sessions excludes zero on
           the positive side.

Run:  python writeup/intraday_proposals/41_otm_leg_ride.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import atm_straddle_lib as asl  # noqa: E402
from live.ibkr import parity as par  # noqa: E402
from live.ibkr.pricing import (  # noqa: E402
    invert_total_vol,
    package_delta,
    package_price,
)

REPO = asl.find_repo(Path(__file__).resolve().parent)
HOLD = REPO / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "41"
CHAIN = REPO / "data" / "spxw_chain.parquet"
SPOT = REPO / "data" / "spxw_spot.parquet"
GSPC = HOLD / "cache" / "gspc_ohlc.parquet"

ENTRY = "11:00"
CLOSE = "15:30"
#: the 30-minute stamps the hedge tape walks, entry stamp first.
CLOCKS: tuple[str, ...] = (
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
ERA_START = "2022-05-16"  # the chain turns daily; the deck reports this era apart
OOS_START = "2024-05-01"  # first session after the deck's last scored day
OOS_END = "2025-12-31"  # last session in data/spxw_chain.parquet
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
CONTRACT_MULTIPLIER = 100.0  # SPX index option, dollars per index point

# Stated grids.
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5
JUMPS = (0.01, 0.02, 0.03, 0.05)
LEG_TOL = 1e-12  # call + put must equal the package to this

# GATE 0 targets (results/live_parity/parity_report.md and
# results/atm_straddle_intraday_holdclose/rule_by_strategy_dh/1100/).
GATE_N = 865
GATE_BOOK_WHOLE = 2.2525467
GATE_BOOK_ERA = 2.5131322
GATE_HOLD_CROSSED = 3.582391
GATE_SHARPE_TOL = 1e-6
GATE_HEDGE_TOL = 1e-12
# parity_report.md section 2: the mid-inverted exit book differs from the
# vendor-implied one by at most this in premium units on a day.
PARITY_MID_MAX_ABS = 0.014758
PARITY_MID_MEAN_ABS = 0.000333


# ----------------------------------------------------------------- pricing --
def _cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(np.asarray(z, float) / np.sqrt(2.0)))


def _d1(s: np.ndarray, f: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d1, ok) for Black-76 with r = 0 and the forward taken as the spot."""
    s = np.asarray(s, float)
    f = np.asarray(f, float)
    k = np.asarray(k, float)
    ok = np.isfinite(s) & (s > 0.0) & np.isfinite(f) & (f > 0.0) & (k > 0.0)
    ss = np.where(ok, s, 1.0)
    ff = np.where(ok, f, 1.0)
    kk = np.where(ok, k, 1.0)
    return (np.log(ff / kk) + 0.5 * ss * ss) / ss, ok


def leg_price(s: np.ndarray, f: np.ndarray, k: np.ndarray, right: str) -> np.ndarray:
    """Black-76 price of ONE leg; call + put reproduces pricing.package_price."""
    d1, ok = _d1(s, f, k)
    ff = np.asarray(f, float)
    kk = np.asarray(k, float)
    ss = np.asarray(s, float)
    if right == "C":
        live = ff * _cdf(d1) - kk * _cdf(d1 - ss)
        dead = np.maximum(ff - kk, 0.0)
    else:
        live = kk * _cdf(-(d1 - ss)) - ff * _cdf(-d1)
        dead = np.maximum(kk - ff, 0.0)
    return np.where(ok, live, dead)


def leg_delta(s: np.ndarray, f: np.ndarray, k: np.ndarray, right: str) -> np.ndarray:
    """Black-76 delta of ONE leg; call + put reproduces pricing.package_delta.

    Zero when the total volatility is not finite and strictly positive, the
    same branch ``package_delta`` takes: at the settlement there is nothing
    left to hedge.
    """
    d1, ok = _d1(s, f, k)
    nd1 = _cdf(d1)
    return np.where(ok, nd1 if right == "C" else nd1 - 1.0, 0.0)


def package_delta_vec(
    s: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    return leg_delta(s, f, kc, "C") + leg_delta(s, f, kp, "P")


def assert_legs_add_up(
    s: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> tuple[float, float]:
    """Largest call + put minus package, for price (relative) and delta.

    The package price is tens of index points, so it is checked relative to
    its own size; the delta is of order one and is checked absolutely.
    """
    pr = leg_price(s, f, kc, "C") + leg_price(s, f, kp, "P")
    dl = leg_delta(s, f, kc, "C") + leg_delta(s, f, kp, "P")
    ref_p = np.array(
        [package_price(a, b, c, d) for a, b, c, d in zip(s, f, kc, kp, strict=True)]
    )
    ref_d = np.array(
        [package_delta(a, b, c, d) for a, b, c, d in zip(s, f, kc, kp, strict=True)]
    )
    dp = float(np.nanmax(np.abs(pr - ref_p) / np.maximum(np.abs(ref_p), 1.0)))
    dd = float(np.nanmax(np.abs(dl - ref_d)))
    assert dp < LEG_TOL and dd < LEG_TOL, (dp, dd)
    return dp, dd


# ------------------------------------------------------------- statistics ---
def sharpe(x: "pd.Series[float] | np.ndarray") -> float:
    v = pd.Series(x).dropna()
    if len(v) < 2:
        return float("nan")
    sd = float(v.std(ddof=1))
    return float(v.mean()) / sd * ANN if sd > 0 else float("nan")


def maxdd(x: "pd.Series[float]") -> float:
    """Worst peak-to-trough of the cumulative SUM path, in the series' units."""
    v = pd.Series(x).dropna().sort_index().to_numpy(float)
    if v.size < 1:
        return float("nan")
    path = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def stats_row(name: str, r: "pd.Series[float]") -> dict[str, Any]:
    v = pd.Series(r).dropna().sort_index()
    t, lag = asl.newey_west_t(v)
    return {
        "book": name,
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
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
    rng = np.random.default_rng(BOOT_SEED)
    idx = asl.circular_block_bootstrap_idx(rng, len(v), BOOT_BLOCK, BOOT_B)
    draws = v[idx].mean(axis=1)
    return float(np.percentile(draws, CI_LO_PCT)), float(
        np.percentile(draws, CI_HI_PCT)
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
        "mean_diff": float(d.mean()),
        "t_hac": t,
        "t_lag": lag,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero_positive": bool(lo > 0.0),
        "d_Sharpe": sharpe(a) - sharpe(b),
    }


# ------------------------------------------------------------- the rule -----
def ride_choice(
    kc: np.ndarray, kp: np.ndarray, s15: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(ride_call, case) at 15:30 under the pre-registered rule.

    Buy back the leg the spot has passed and ride the other one; between the
    strikes both are out of the money, so buy back the nearer strike and ride
    the farther.  Equidistant (only possible when the two strikes straddle the
    spot exactly) rides the call.
    """
    dist_c = kc - s15
    dist_p = s15 - kp
    call_itm = s15 > kc
    put_itm = s15 < kp
    ride_call = np.where(call_itm, False, np.where(put_itm, True, dist_c >= dist_p))
    case = np.where(
        call_itm, "call in the money", np.where(put_itm, "put in the money", "between")
    )
    return ride_call, case


def ride_frame(
    idx: pd.DatetimeIndex,
    kc: np.ndarray,
    kp: np.ndarray,
    entry_mid: np.ndarray,
    entry_bid: np.ndarray,
    s15: np.ndarray,
    s_close: np.ndarray,
    ask_c: np.ndarray,
    ask_p: np.ndarray,
    tot15_hedge: np.ndarray,
    tot15_mid: np.ndarray,
    hedge_exit: np.ndarray,
    hedge_hold: np.ndarray,
) -> pd.DataFrame:
    """The four books and the ride's anatomy, one row a day.

    ``tot15_hedge`` is the total volatility the book's own hedge uses at
    15:30 (it prices the residual delta); ``tot15_mid`` is the total
    volatility bisected out of the 15:30 package midpoint, which is the
    yardstick the ridden leg's moneyness is quoted in.
    """
    ride_call, case = ride_choice(kc, kp, s15)
    k_ride = np.where(ride_call, kc, kp)
    ask_ride = np.where(ride_call, ask_c, ask_p)
    ask_buy = np.where(ride_call, ask_p, ask_c)
    pay_ride = np.where(
        ride_call, np.maximum(s_close - kc, 0.0), np.maximum(kp - s_close, 0.0)
    )
    pay_buy = np.where(
        ride_call, np.maximum(kp - s_close, 0.0), np.maximum(s_close - kc, 0.0)
    )
    dlt_ride = np.where(
        ride_call,
        leg_delta(tot15_hedge, s15, kc, "C"),
        leg_delta(tot15_hedge, s15, kp, "P"),
    )
    resid = dlt_ride * (s_close - s15)
    dist = np.where(ride_call, kc - s15, s15 - kp)
    implied_move = s15 * tot15_mid

    opt_book = -(ask_c + ask_p - entry_bid)
    opt_ride = -(ask_buy + pay_ride - entry_bid)
    exit_settle = np.maximum(s_close - kc, 0.0) + np.maximum(kp - s_close, 0.0)
    out = pd.DataFrame(
        {
            "K_c": kc,
            "K_p": kp,
            "S_1530": s15,
            "S_close": s_close,
            "entry_mid": entry_mid,
            "entry_bid": entry_bid,
            "ask_c_1530": ask_c,
            "ask_p_1530": ask_p,
            "hedge_exit_pts": hedge_exit,
            "hedge_hold_pts": hedge_hold,
            "ride_leg": np.where(ride_call, "call", "put"),
            "case": case,
            "K_ride": k_ride,
            "dist_ride_pts": dist,
            "total_vol_1530_mid": tot15_mid,
            "implied_last_bar_pts": implied_move,
            "dist_in_implied_moves": dist / implied_move,
            "ask_ride_pts": ask_ride,
            "ask_buyback_pts": ask_buy,
            "pay_ride_pts": pay_ride,
            "pay_buyback_pts": pay_buy,
            "saved_pts": ask_ride - pay_ride,
            "delta_ride_1530": dlt_ride,
            "resid_hedge_pts": resid,
            "settle_move_pts": s_close - s15,
            "r_book": (opt_book + hedge_exit) / entry_mid,
            "r_ride_flat": (opt_ride + hedge_exit) / entry_mid,
            "r_ride_hedged": (opt_ride + hedge_exit + resid) / entry_mid,
            "r_hold": (-(exit_settle - entry_bid) + hedge_hold) / entry_mid,
        },
        index=idx,
    )
    for col in ("r_book", "r_ride_flat", "r_ride_hedged", "r_hold"):
        out[col + "_pts"] = out[col] * out["entry_mid"]
    return out


# ------------------------------------------------ in-sample (cache) tape ----
def _standalone() -> ModuleType:
    path = REPO / "writeup" / "make_dh_causal_standalone_tex.py"
    spec = importlib.util.spec_from_file_location("mk_dh_causal_41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cache_path() -> Path:
    return sorted((HOLD / "cache").glob("trade_*.parquet"))[-1]


def insample_grids(mk: ModuleType) -> dict[str, Any]:
    """The arrays hold_mark_1100 prices the 11:00 book from, per stamp."""
    pkg = pd.read_parquet(cache_path())
    m = mk._rule_mod()
    deck = pd.read_parquet(m.DECK / "daily_blk2.parquet")
    days = pd.to_datetime(deck.index)
    p = pkg.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"].isin(days)].copy()
    clocks = sorted(p["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    p["h_rem"] = p["hhmm"].map(n_rem).astype(float) * 0.5
    p["iv_hourly_used"] = m.iv_hourly_reinverted(p).to_numpy()
    dh = m.attach_long_dh(p)
    sl = (
        dh.loc[dh["hhmm"] == ENTRY]
        .dropna(subset=["entry", "exit", "K_c", "K_p", "S_close", "R"])
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    sl = sl.loc[sl["entry"].astype(float) > 0]
    dates = pd.DatetimeIndex(sl.index)

    def grid(col: str) -> np.ndarray:
        return (
            p.pivot_table(index="date", columns="hhmm", values=col, aggfunc="first")
            .reindex(index=dates, columns=clocks)
            .to_numpy(float)
        )

    j = clocks.index(ENTRY)
    j15 = clocks.index(CLOSE)
    sg = grid("S")
    ivg = grid("iv_hourly_used")
    h_row = np.array([n_rem[c] * 0.5 for c in clocks], float)
    tot = np.where(ivg > 0, ivg * np.sqrt(h_row[None, :]), np.nan)
    tot[:, :j] = np.nan
    kc = sl["K_c"].to_numpy(float)
    kp = sl["K_p"].to_numpy(float)
    st = sl["S_close"].to_numpy(float)
    dlt = package_delta_vec(tot, sg, kc[:, None], kp[:, None])
    nxt = np.full_like(sg, np.nan)
    nxt[:, :-1] = sg[:, 1:]
    nxt[:, -1] = st
    d_s = np.where(np.isfinite(sg) & np.isfinite(nxt), nxt - sg, 0.0)
    d_s[:, :j] = 0.0
    d_s_flat = d_s.copy()
    d_s_flat[:, j15] = 0.0
    # The 15:30 package midpoint re-inverted, the live engine's own volatility.
    k_c15 = grid("K_c")[:, j15]
    k_p15 = grid("K_p")[:, j15]
    e15 = grid("entry")[:, j15]
    tot15_mid = np.array(
        [
            invert_total_vol(a, b, c, d)
            for a, b, c, d in zip(sg[:, j15], k_c15, k_p15, e15, strict=True)
        ]
    )
    return {
        "pkg": pkg,
        "panel": p,
        "clocks": clocks,
        "j": j,
        "j15": j15,
        "dates": dates,
        "sl": sl,
        "Sg": sg,
        "IVg": ivg,
        "h_row": h_row,
        "tot": tot,
        "dlt": dlt,
        "dS": d_s,
        "dS_flat": d_s_flat,
        "hedge_exit": (dlt * d_s_flat).sum(axis=1),
        "hedge_hold": (dlt * d_s).sum(axis=1),
        "K_c": kc,
        "K_p": kp,
        "entry": sl["entry"].to_numpy(float),
        "bid": sl["bid_entry"].to_numpy(float),
        "S_close": st,
        "tot15_mid": tot15_mid,
    }


def exit_leg_quotes(panel: pd.DataFrame, g: dict[str, Any]) -> pd.DataFrame:
    """Quoted 15:30 bid/ask of the ELEVEN-O'CLOCK strikes, from the chain."""
    dates = g["dates"]
    close_ts = (
        panel.loc[panel["hhmm"] == CLOSE, ["date", "timestamp"]]
        .drop_duplicates("date")
        .set_index("date")["timestamp"]
    )
    stamps = sorted(pd.DatetimeIndex(close_ts.reindex(dates).to_numpy()).unique())
    chain = pd.read_parquet(
        CHAIN,
        columns=["expiration", "timestamp", "strike", "cp", "bid", "ask"],
        filters=[("timestamp", "in", stamps)],
    )
    chain["strike"] = chain["strike"].astype(float)
    chain["cp"] = chain["cp"].astype(str)
    want = []
    for i, d in enumerate(dates):
        want.append((d, close_ts.get(d), g["K_c"][i], "C", "c"))
        want.append((d, close_ts.get(d), g["K_p"][i], "P", "p"))
    key = pd.DataFrame(want, columns=["date", "timestamp", "strike", "cp", "leg"])
    key["expiration"] = key["date"].astype(chain["expiration"].dtype)
    key["timestamp"] = key["timestamp"].astype(chain["timestamp"].dtype)
    got = key.merge(chain, on=["expiration", "timestamp", "strike", "cp"], how="left")
    bid = got["bid"].astype(float)
    ask = got["ask"].astype(float)
    sentinel = (bid == 0.0) & (ask == 0.0)
    got["leg_ask"] = ask.mask(sentinel)
    out = pd.DataFrame(index=dates)
    for leg, col in (("c", "ask_c"), ("p", "ask_p")):
        one = got.loc[got["leg"] == leg].set_index("date")["leg_ask"]
        out[col] = one.loc[~one.index.duplicated()].reindex(dates)
    return out


# ------------------------------------------------------ chain-based build ---
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
    """Expirations whose 15:30 row has already expired (hours_to_expiration <= 0)."""
    hte = pd.read_parquet(
        CHAIN,
        columns=["expiration", "timestamp", "hours_to_expiration"],
        filters=[("timestamp", "in", sorted(pd.DatetimeIndex(stamp[CLOSE]).unique()))],
    )
    med = hte.groupby("expiration")["hours_to_expiration"].median()
    return pd.DatetimeIndex(pd.to_datetime(med.index[med <= 0])).normalize()


def build_from_chain(stamp: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """The whole book from the chain, with the live engine's mid-inverted vol.

    At every stamp the nearest-OTM straddle is re-picked with the research's
    guards and its midpoint is bisected for the total volatility over the
    remaining session; that volatility prices the delta of the legs actually
    held (the 11:00 strikes).  This is exactly live.ibkr.parity.replay_day
    with iv_mode="mid", run off the chain rather than off the trade cache.
    """
    picks: list[pd.DataFrame] = []
    spots: list[pd.Series] = []
    refused: list[pd.DataFrame] = []
    legs: list[pd.DataFrame] = []
    bodies: dict[pd.Timestamp, pd.Series] = {}
    ts_of = {(d, hh): pd.Timestamp(stamp.loc[d, hh]) for d in dates for hh in CLOCKS}
    back = {v: k for k, v in ts_of.items()}
    for year in sorted(dates.year.unique()):
        d_yr = dates[dates.year == year]
        ts_list = sorted({ts_of[(d, hh)] for d in d_yr for hh in CLOCKS})
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
        ch["mid"] = asl.quote_mid(ch["bid"], ch["ask"]).to_numpy()
        live = ch[np.isfinite(ch["mid"]) & (ch["mid"] > 0.0)].copy()
        sp = asl.stamp_spot(live, ["expiration", "timestamp"])
        atm, dropped = asl.pick_nearest_otm_guarded(
            live[["expiration", "timestamp", "strike", "cp", "bid", "ask", "mid"]],
            sp,
            keys=("expiration", "timestamp"),
        )
        stamps_frame = pd.DataFrame(
            {"timestamp": pd.Series(list(back), dtype=ch["timestamp"].dtype)}
        )
        stamps_frame = stamps_frame[stamps_frame["timestamp"].isin(ts_list)]
        stamps_frame["date"] = [
            back[pd.Timestamp(t)][0] for t in stamps_frame["timestamp"]
        ]
        stamps_frame["hhmm"] = [
            back[pd.Timestamp(t)][1] for t in stamps_frame["timestamp"]
        ]
        stamps_frame["expiration"] = stamps_frame["date"].astype(ch["expiration"].dtype)
        atm = atm.merge(
            stamps_frame[["expiration", "timestamp", "date", "hhmm"]],
            on=["expiration", "timestamp"],
            how="left",
        )
        picks.append(
            atm[
                [
                    "date",
                    "hhmm",
                    "S",
                    "K_c",
                    "K_p",
                    "entry",
                    "bid_entry",
                    "ask_entry",
                    "n_live",
                ]
            ]
        )
        sp_frame = (
            sp.rename("S_stamp")
            .reset_index()
            .merge(
                stamps_frame[["expiration", "timestamp", "date", "hhmm"]],
                on=["expiration", "timestamp"],
                how="inner",
            )
        )
        spots.append(sp_frame[["date", "hhmm", "S_stamp"]])
        if len(dropped):
            dr = dropped.merge(
                stamps_frame[["expiration", "timestamp", "date", "hhmm"]],
                on=["expiration", "timestamp"],
                how="left",
            )
            refused.append(dr[["date", "hhmm", "reason", "n_live"]])
        # the held legs' 15:30 quotes need the 11:00 body first
        ent = atm.loc[atm["hhmm"] == ENTRY].set_index("date").sort_index()
        for d, row in ent.iterrows():
            bodies[pd.Timestamp(d)] = row
        keys = []
        for d, row in ent.iterrows():
            ts = ts_of[(pd.Timestamp(d), CLOSE)]
            keys.append((d, ts, float(row["K_c"]), "C", "c"))
            keys.append((d, ts, float(row["K_p"]), "P", "p"))
        key = pd.DataFrame(keys, columns=["date", "timestamp", "strike", "cp", "leg"])
        key["expiration"] = key["date"].astype(ch["expiration"].dtype)
        key["timestamp"] = key["timestamp"].astype(ch["timestamp"].dtype)
        ch["strike"] = ch["strike"].astype(float)
        ch["cp"] = ch["cp"].astype(str)
        got = key.merge(
            ch[["expiration", "timestamp", "strike", "cp", "bid", "ask"]],
            on=["expiration", "timestamp", "strike", "cp"],
            how="left",
        )
        b = got["bid"].astype(float)
        a = got["ask"].astype(float)
        got["leg_ask"] = a.mask((b == 0.0) & (a == 0.0))
        legs.append(got[["date", "leg", "leg_ask"]])
        del ch, live

    atm_all = pd.concat(picks, ignore_index=True)
    atm_all["date"] = pd.to_datetime(atm_all["date"])
    spot_all = pd.concat(spots, ignore_index=True)
    spot_all["date"] = pd.to_datetime(spot_all["date"])
    leg_all = pd.concat(legs, ignore_index=True)
    leg_all["date"] = pd.to_datetime(leg_all["date"])
    body = atm_all.loc[atm_all["hhmm"] == ENTRY].set_index("date").sort_index()
    idx = pd.DatetimeIndex(body.index)

    def piv(frame: pd.DataFrame, col: str) -> np.ndarray:
        return (
            frame.pivot_table(index="date", columns="hhmm", values=col, aggfunc="first")
            .reindex(index=idx, columns=list(CLOCKS))
            .to_numpy(float)
        )

    s_grid = piv(spot_all, "S_stamp")
    k_c_atm = piv(atm_all, "K_c")
    k_p_atm = piv(atm_all, "K_p")
    e_atm = piv(atm_all, "entry")
    tot = np.array(
        [
            [
                invert_total_vol(
                    s_grid[i, k], k_c_atm[i, k], k_p_atm[i, k], e_atm[i, k]
                )
                for k in range(len(CLOCKS))
            ]
            for i in range(len(idx))
        ]
    )
    gspc = pd.read_parquet(GSPC)
    gspc.index = pd.to_datetime(gspc.index)
    s_close = gspc["close"].reindex(idx).to_numpy(float)
    kc = body["K_c"].to_numpy(float)
    kp = body["K_p"].to_numpy(float)
    dlt = package_delta_vec(tot, s_grid, kc[:, None], kp[:, None])
    nxt = np.full_like(s_grid, np.nan)
    nxt[:, :-1] = s_grid[:, 1:]
    nxt[:, -1] = s_close
    d_s = np.where(np.isfinite(s_grid) & np.isfinite(nxt), nxt - s_grid, 0.0)
    d_s_flat = d_s.copy()
    d_s_flat[:, -1] = 0.0
    ask = leg_all.pivot_table(
        index="date", columns="leg", values="leg_ask", aggfunc="first"
    ).reindex(idx)
    return {
        "dates": idx,
        "body": body,
        "S": s_grid,
        "tot": tot,
        "S_close": s_close,
        "K_c": kc,
        "K_p": kp,
        "entry": body["entry"].to_numpy(float),
        "bid": body["bid_entry"].to_numpy(float),
        "ask_c": ask["c"].to_numpy(float),
        "ask_p": ask["p"].to_numpy(float),
        "hedge_exit": (dlt * d_s_flat).sum(axis=1),
        "hedge_hold": (dlt * d_s).sum(axis=1),
        "refused": pd.concat(refused, ignore_index=True)
        if refused
        else pd.DataFrame(columns=["date", "hhmm", "reason", "n_live"]),
    }


# ------------------------------------------------------------------ gates ---
def gate_zero(g: dict[str, Any], quotes: pd.DataFrame, mk: ModuleType) -> pd.DataFrame:
    """Reproduce the book of record before anything new is computed."""
    dates = g["dates"]
    book = ride_frame(
        dates,
        g["K_c"],
        g["K_p"],
        g["entry"],
        g["bid"],
        g["Sg"][:, g["j15"]],
        g["S_close"],
        quotes["ask_c"].to_numpy(float),
        quotes["ask_p"].to_numpy(float),
        g["tot"][:, g["j15"]],
        g["tot15_mid"],
        g["hedge_exit"],
        g["hedge_hold"],
    )
    era = dates >= pd.Timestamp(ERA_START)
    s_whole = sharpe(book["r_book"])
    s_era = sharpe(book.loc[era, "r_book"])
    s_hold = sharpe(book["r_hold"])
    assert int(len(book)) == GATE_N, len(book)
    assert abs(s_whole - GATE_BOOK_WHOLE) < GATE_SHARPE_TOL, s_whole
    assert abs(s_era - GATE_BOOK_ERA) < GATE_SHARPE_TOL, s_era
    assert abs(s_hold - GATE_HOLD_CROSSED) < GATE_SHARPE_TOL, s_hold
    # the research's own hold tape, independently
    r_hold_ref = mk.hold_mark_1100(g["pkg"])[2].dropna()
    dev_hold = float((book["r_hold"] - r_hold_ref.reindex(dates)).abs().max())
    # and the live engine's replay, day by day
    panel = par.scored_days(g["pkg"], REPO)
    panel, _ = par.attach_engine_iv(panel, mk._rule_mod())
    dev_x = 0.0
    dev_h = 0.0
    for date, rows in panel.groupby("date", sort=True):
        i = int(dates.get_loc(pd.Timestamp(date)))
        ex = par.replay_day(rows, exit_clock=CLOSE, iv_mode="vendor")
        hd = par.replay_day(rows, exit_clock=None, iv_mode="vendor")
        dev_x = max(dev_x, abs(float(ex["hedge_pts"]) - float(g["hedge_exit"][i])))
        dev_h = max(dev_h, abs(float(hd["hedge_pts"]) - float(g["hedge_hold"][i])))
    assert dev_x < GATE_HEDGE_TOL and dev_h < GATE_HEDGE_TOL, (dev_x, dev_h)
    dp, dd = assert_legs_add_up(
        g["tot"][:, g["j15"]], g["Sg"][:, g["j15"]], g["K_c"], g["K_p"]
    )
    print(
        f"GATE 0  n={len(book)}  book crossed-quoted Sharpe_ann {s_whole:.7f} "
        f"(target {GATE_BOOK_WHOLE})  era {s_era:.7f} (target {GATE_BOOK_ERA}, "
        f"n={int(era.sum())})  hold crossed {s_hold:.7f} "
        f"(target {GATE_HOLD_CROSSED})"
    )
    print(
        f"GATE 0  hedge tape vs live.ibkr.parity.replay_day: exit {dev_x:.3e}, "
        f"hold {dev_h:.3e} index points (tolerance {GATE_HEDGE_TOL:.0e}); hold "
        f"return vs make_dh_causal_standalone_tex.hold_mark_1100 {dev_hold:.3e}"
    )
    print(
        f"GATE 0  leg split: max |call + put - package| price {dp:.3e} "
        f"(relative), delta {dd:.3e} (absolute); tolerance {LEG_TOL:.0e}"
    )
    return book


def gate_c0(chain: dict[str, Any], g: dict[str, Any], book: pd.DataFrame) -> None:
    """The chain-based builder reproduces the cache and the book's Sharpe."""
    pkg = g["pkg"].copy()
    pkg["date"] = pd.to_datetime(pkg["date"])
    cached = (
        pkg.loc[pkg["hhmm"] == ENTRY]
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    idx = chain["dates"]
    common = idx.intersection(pd.DatetimeIndex(cached.index))
    got = chain["body"].loc[common]
    ref = cached.loc[common]
    devs = {
        col: float((got[col].astype(float) - ref[col].astype(float)).abs().max())
        for col in ("S", "K_c", "K_p", "entry", "bid_entry", "ask_entry", "n_live")
    }
    assert max(devs.values()) == 0.0, devs
    print(
        f"GATE C0  chain-built 11:00 straddle vs the trade cache on "
        f"{len(common)} sessions (of {len(idx)} built, {len(cached)} cached): "
        + ", ".join(f"{k} {v:.0e}" for k, v in devs.items())
    )
    ins = g["dates"]
    r = pd.Series(
        (-(chain["ask_c"] + chain["ask_p"] - chain["bid"]) + chain["hedge_exit"])
        / chain["entry"],
        index=idx,
    ).reindex(ins)
    exit_settle = np.maximum(chain["S_close"] - chain["K_c"], 0.0) + np.maximum(
        chain["K_p"] - chain["S_close"], 0.0
    )
    r_hold = pd.Series(
        (-(exit_settle - chain["bid"]) + chain["hedge_hold"]) / chain["entry"],
        index=idx,
    ).reindex(ins)
    era = ins >= pd.Timestamp(ERA_START)
    d = (r - book["r_book"]).dropna()
    assert float(d.abs().max()) <= PARITY_MID_MAX_ABS, float(d.abs().max())
    assert float(d.abs().mean()) <= PARITY_MID_MEAN_ABS, float(d.abs().mean())
    print(
        f"GATE C0  chain (mid-inverted) book Sharpe {sharpe(r):.7f} whole / "
        f"{sharpe(r[era]):.7f} era against the cache (vendor-implied) book "
        f"{sharpe(book['r_book']):.7f} / {sharpe(book.loc[era, 'r_book']):.7f}; "
        f"hold {sharpe(r_hold):.7f} against {sharpe(book['r_hold']):.7f}"
    )
    print(
        f"GATE C0  per-day |mid-inverted - vendor| max {float(d.abs().max()):.6f}, "
        f"mean {float(d.abs().mean()):.6f} premium units, against the "
        f"parity_report.md budget max {PARITY_MID_MAX_ABS}, mean "
        f"{PARITY_MID_MEAN_ABS}"
    )


# ----------------------------------------------------------------- part A ---
def part_a(book: pd.DataFrame) -> pd.DataFrame:
    era = book.index >= pd.Timestamp(ERA_START)
    books = ("r_book", "r_ride_flat", "r_ride_hedged", "r_hold")
    label = {
        "r_book": "book (both legs bought back)",
        "r_ride_flat": "ride, futures flat at 15:30",
        "r_ride_hedged": "ride, futures held to 16:00",
        "r_hold": "hold to cash settlement",
    }
    rows = []
    for col in books:
        rows.append({"sample": "whole", **stats_row(label[col], book[col])})
        rows.append({"sample": "era", **stats_row(label[col], book.loc[era, col])})
    summary = pd.DataFrame(rows)
    print("\n--- A1. the four books, 11:00 entry, premium units")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    pairs = []
    for col in ("r_ride_flat", "r_ride_hedged"):
        pairs.append(
            {
                "sample": "whole",
                **paired_row(label[col] + " - book", book[col], book["r_book"]),
            }
        )
        pairs.append(
            {
                "sample": "era",
                **paired_row(
                    label[col] + " - book", book.loc[era, col], book.loc[era, "r_book"]
                ),
            }
        )
    diffs = pd.DataFrame(pairs)
    print(
        f"\n--- A2. paired difference against the book, HAC t and a circular-block "
        f"bootstrap CI (B={BOOT_B}, block {BOOT_BLOCK}, seed {BOOT_SEED})"
    )
    print(diffs.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    dec = []
    for name, mask in (("whole", np.ones(len(book), bool)), ("era", era)):
        b = book.loc[mask]
        dec.append(
            {
                "sample": name,
                "n": int(len(b)),
                "saved_ask_units": float((b["ask_ride_pts"] / b["entry_mid"]).mean()),
                "settlement_payoff_units": float(
                    (b["pay_ride_pts"] / b["entry_mid"]).mean()
                ),
                "net_saved_units": float((b["saved_pts"] / b["entry_mid"]).mean()),
                "resid_hedge_units": float(
                    (b["resid_hedge_pts"] / b["entry_mid"]).mean()
                ),
                "mean_ride_flat_minus_book": float(
                    (b["r_ride_flat"] - b["r_book"]).mean()
                ),
                "mean_ride_hedged_minus_book": float(
                    (b["r_ride_hedged"] - b["r_book"]).mean()
                ),
                "ask_ride_pts": float(b["ask_ride_pts"].mean()),
                "pay_ride_pts": float(b["pay_ride_pts"].mean()),
                "resid_hedge_pts": float(b["resid_hedge_pts"].mean()),
            }
        )
    decomp = pd.DataFrame(dec)
    print("\n--- A3. where the difference comes from (means, premium units and points)")
    print(decomp.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    case = book.groupby("case").agg(
        n=("r_book", "size"),
        share=("r_book", lambda s: len(s) / len(book)),
        ride_call=("ride_leg", lambda s: float((s == "call").mean())),
        dist_ride_pts=("dist_ride_pts", "mean"),
        dist_in_implied_moves=("dist_in_implied_moves", "mean"),
        saved_pts=("saved_pts", "mean"),
        mean_flat_minus_book=("r_ride_flat", "mean"),
    )
    case["mean_flat_minus_book"] = (
        book.assign(d=book["r_ride_flat"] - book["r_book"]).groupby("case")["d"].mean()
    )
    print("\n--- A4. which leg is ridden")
    print(case.to_string(float_format=lambda v: f"{v:,.6f}"))
    print(
        f"call ridden {int((book['ride_leg'] == 'call').sum())} days, put ridden "
        f"{int((book['ride_leg'] == 'put').sum())} days; ridden-leg 15:30 ask "
        f"{float((book['ask_ride_pts'] / book['entry_mid']).mean()):.4f} of entry "
        f"premium on average, its settlement payoff "
        f"{float((book['pay_ride_pts'] / book['entry_mid']).mean()):.4f}; ridden "
        f"leg expires worthless on "
        f"{100.0 * float((book['pay_ride_pts'] == 0).mean()):.1f}% of days"
    )
    return summary


def worst_ten(book: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ride_leg",
        "case",
        "K_ride",
        "S_1530",
        "S_close",
        "settle_move_pts",
        "dist_ride_pts",
        "dist_in_implied_moves",
        "ask_ride_pts",
        "pay_ride_pts",
        "saved_pts",
        "resid_hedge_pts",
        "entry_mid",
        "r_book",
        "r_ride_flat",
        "r_ride_hedged",
        "r_hold",
    ]
    return book.sort_values("r_ride_hedged").head(10)[cols]


# ----------------------------------------------------------------- part B ---
def part_b(book: pd.DataFrame, g: dict[str, Any]) -> pd.DataFrame:
    """Proposal 28's last-bar jump grid, for the ride variants.

    The representative day is proposal 28's: the sample median of entry
    premium over the 11:00 spot among the scored days.  A last-bar jump moves
    only the settlement print; the 15:30 hedge and the 15:30 leg identity are
    already fixed, so the book (both legs bought back at 15:30) is flat by
    construction and only the ridden leg and the residual hedge are exposed.
    """
    ratio = pd.Series(g["entry"] / g["Sg"][:, g["j"]], index=g["dates"]).sort_values()
    rep_day = ratio.index[len(ratio) // 2]
    row = book.loc[rep_day]
    i = int(g["dates"].get_loc(rep_day))
    print(
        f"\n--- B. last-bar jump stress on the representative day {rep_day.date()} "
        f"(median entry premium / spot among the {len(ratio)} scored days: "
        f"{100.0 * float(ratio.iloc[len(ratio) // 2]):.4f}%)"
    )
    print(
        f"S(11:00) {g['Sg'][i, g['j']]:.2f}  K_c {row['K_c']:.0f}  K_p "
        f"{row['K_p']:.0f}  entry mid {row['entry_mid']:.2f} pts  entry bid "
        f"{row['entry_bid']:.2f} pts  S(15:30) {row['S_1530']:.2f}  settlement "
        f"{row['S_close']:.2f}  ridden leg: {row['ride_leg']} at "
        f"{row['K_ride']:.0f} ({row['case']}, {row['dist_ride_pts']:.2f} pts = "
        f"{row['dist_in_implied_moves']:.2f} implied last-bar moves)  ridden-leg "
        f"delta at 15:30 {row['delta_ride_1530']:+.4f}"
    )
    ride_call = row["ride_leg"] == "call"
    rows = []
    for x in JUMPS:
        for sign in (1.0, -1.0):
            st = float(row["S_1530"]) * (1.0 + sign * x)
            pay_ride = (
                max(st - float(row["K_c"]), 0.0)
                if ride_call
                else max(float(row["K_p"]) - st, 0.0)
            )
            opt_ride = -(
                float(row["ask_buyback_pts"]) + pay_ride - float(row["entry_bid"])
            )
            resid = float(row["delta_ride_1530"]) * (st - float(row["S_1530"]))
            settle = max(st - float(row["K_c"]), 0.0) + max(float(row["K_p"]) - st, 0.0)
            r_flat = (opt_ride + float(row["hedge_exit_pts"])) / float(row["entry_mid"])
            r_hedg = (opt_ride + float(row["hedge_exit_pts"]) + resid) / float(
                row["entry_mid"]
            )
            r_hold = (
                -(settle - float(row["entry_bid"])) + float(row["hedge_hold_pts"])
            ) / float(row["entry_mid"])
            toward = (sign > 0) == ride_call
            rows.append(
                {
                    "date": rep_day.date().isoformat(),
                    "ride_leg": row["ride_leg"],
                    "jump_pct": 100.0 * sign * x,
                    "direction": "toward the ridden leg" if toward else "away from it",
                    "S_settle_stressed": st,
                    "pay_ride_pts": pay_ride,
                    "resid_hedge_pts": resid,
                    "r_book_units": float(row["r_book"]),
                    "r_ride_flat_units": r_flat,
                    "r_ride_hedged_units": r_hedg,
                    "r_hold_units": r_hold,
                    "d_book_units": 0.0,
                    "d_ride_flat_units": r_flat - float(row["r_ride_flat"]),
                    "d_ride_hedged_units": r_hedg - float(row["r_ride_hedged"]),
                    "d_hold_units": r_hold - float(row["r_hold"]),
                    "d_book_dollars": 0.0,
                    "d_ride_flat_dollars": (r_flat - float(row["r_ride_flat"]))
                    * float(row["entry_mid"])
                    * CONTRACT_MULTIPLIER,
                    "d_ride_hedged_dollars": (r_hedg - float(row["r_ride_hedged"]))
                    * float(row["entry_mid"])
                    * CONTRACT_MULTIPLIER,
                    "d_hold_dollars": (r_hold - float(row["r_hold"]))
                    * float(row["entry_mid"])
                    * CONTRACT_MULTIPLIER,
                }
            )
    out = (
        pd.DataFrame(rows).sort_values(["direction", "jump_pct"]).reset_index(drop=True)
    )
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    return out


# ----------------------------------------------------------------- part C ---
def oos_books(chain: dict[str, Any], g: dict[str, Any]) -> pd.DataFrame:
    idx = chain["dates"]
    tot15_mid = chain["tot"][:, -1]
    frame = ride_frame(
        idx,
        chain["K_c"],
        chain["K_p"],
        chain["entry"],
        chain["bid"],
        chain["S"][:, -1],
        chain["S_close"],
        chain["ask_c"],
        chain["ask_p"],
        tot15_mid,
        tot15_mid,
        chain["hedge_exit"],
        chain["hedge_hold"],
    )
    frame["in_sample"] = frame.index.isin(g["dates"])
    return frame


def part_c(oos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = {
        "book (both legs bought back)": "r_book",
        "ride, futures flat at 15:30": "r_ride_flat",
        "ride, futures held to 16:00": "r_ride_hedged",
        "hold to cash settlement": "r_hold",
    }
    periods = {
        "2024-05-01..2025-12-31": (OOS_START, OOS_END),
        "2024H2 (2024-05-01..2024-12-31)": (OOS_START, "2024-12-31"),
        "2025": ("2025-01-01", OOS_END),
    }
    rows = []
    for pname, (lo, hi) in periods.items():
        sub = oos.loc[(oos.index >= pd.Timestamp(lo)) & (oos.index <= pd.Timestamp(hi))]
        for label, col in books.items():
            rows.append(
                {"period": pname, "unit": "premium", **stats_row(label, sub[col])}
            )
            rows.append(
                {
                    "period": pname,
                    "unit": "index points",
                    **stats_row(label, sub[col + "_pts"]),
                }
            )
    summary = pd.DataFrame(rows)
    print("\n--- C2. the unseen sessions, by period and unit")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    pairs = []
    for pname, (lo, hi) in periods.items():
        sub = oos.loc[(oos.index >= pd.Timestamp(lo)) & (oos.index <= pd.Timestamp(hi))]
        for label, col in (
            ("ride-flat - book", "r_ride_flat"),
            ("ride-hedged - book", "r_ride_hedged"),
        ):
            pairs.append(
                {
                    "period": pname,
                    "unit": "premium",
                    **paired_row(label, sub[col], sub["r_book"]),
                }
            )
            pairs.append(
                {
                    "period": pname,
                    "unit": "index points",
                    **paired_row(label, sub[col + "_pts"], sub["r_book_pts"]),
                }
            )
    diffs = pd.DataFrame(pairs)
    print(
        f"\n--- C3. paired difference against the book on the unseen sessions "
        f"(B={BOOT_B}, block {BOOT_BLOCK}, seed {BOOT_SEED})"
    )
    print(diffs.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    return summary, diffs


def premium_curve(chain: dict[str, Any], g: dict[str, Any]) -> pd.DataFrame:
    """Implied over realised remaining-window move, by clock and period."""
    idx = chain["dates"]
    s = chain["S"]
    tot = chain["tot"]
    st = chain["S_close"]
    log_rem = np.log(st[:, None] / s)
    periods = {
        "in sample (scored days 2020-01-03..2024-04-30)": idx.isin(g["dates"]),
        "unseen 2024-05-01..2025-12-31": (idx >= pd.Timestamp(OOS_START))
        & (idx <= pd.Timestamp(OOS_END)),
        "unseen 2024H2": (idx >= pd.Timestamp(OOS_START))
        & (idx <= pd.Timestamp("2024-12-31")),
        "unseen 2025": (idx >= pd.Timestamp("2025-01-01"))
        & (idx <= pd.Timestamp(OOS_END)),
    }
    rows = []
    for pname, mask in periods.items():
        for k, clock in enumerate(CLOCKS):
            imp = tot[mask, k]
            rea = log_rem[mask, k]
            ok = np.isfinite(imp) & np.isfinite(rea)
            imp_rms = float(np.sqrt(np.mean(imp[ok] ** 2)))
            rea_rms = float(np.sqrt(np.mean(rea[ok] ** 2)))
            rows.append(
                {
                    "period": pname,
                    "clock": clock,
                    "n": int(ok.sum()),
                    "implied_rem_mean": float(np.mean(imp[ok])),
                    "implied_rem_rms": imp_rms,
                    "realised_rem_rms": rea_rms,
                    "implied_over_realised": imp_rms / rea_rms,
                }
            )
    out = pd.DataFrame(rows)
    print("\n--- C4. implied over realised remaining-window move, by clock")
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    return out


def premium_by_year(chain: dict[str, Any]) -> pd.DataFrame:
    idx = chain["dates"]
    ratio = pd.Series(chain["entry"] / chain["S"][:, 0], index=idx)
    out = ratio.groupby(idx.year).agg(
        n="size",
        mean_pct=lambda s: 100.0 * s.mean(),
        median_pct=lambda s: 100.0 * s.median(),
    )
    out["mean_premium_pts"] = (
        pd.Series(chain["entry"], index=idx).groupby(idx.year).mean()
    )
    print("\n--- C5. 11:00 entry premium over spot, by calendar year")
    print(out.to_string(float_format=lambda v: f"{v:,.6f}"))
    return out


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    mk = _standalone()
    g = insample_grids(mk)
    quotes = exit_leg_quotes(g["panel"], g)
    n_missing = int(quotes.isna().to_numpy().sum())
    print(
        f"15:30 quotes of the 11:00 strikes read from data/spxw_chain.parquet: "
        f"{n_missing} missing or no-quote (bid == ask == 0) legs"
    )
    book = gate_zero(g, quotes, mk)

    stamp, px_file = session_stamps()
    half = half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    print(
        f"\nchain sessions {len(stamp)} 2020-01-03..{pd.Timestamp(stamp.index.max()).date()}; "
        f"half sessions dropped ({len(half)}): "
        + ", ".join(str(d.date()) for d in half)
        + f"; sessions offered to the picker {len(sessions)}"
    )
    chain = build_from_chain(stamp, sessions)
    dev_spot = float(
        np.nanmax(
            np.abs(
                chain["S"] - px_file.loc[chain["dates"], list(CLOCKS)].to_numpy(float)
            )
        )
    )
    print(
        f"chain sessions built {len(chain['dates'])}; stamps refused by the guards "
        f"{len(chain['refused'])}"
        + (
            ": "
            + ", ".join(
                f"{pd.Timestamp(r['date']).date()} {r['hhmm']} {r['reason']}"
                for _, r in chain["refused"].iterrows()
            )
            if len(chain["refused"])
            else ""
        )
        + f"; chain underlying_price vs data/spxw_spot.parquet max abs diff {dev_spot:.0e}"
    )
    gate_c0(chain, g, book)

    a_summary = part_a(book)
    era = book.index >= pd.Timestamp(ERA_START)
    gate_a = {
        name: paired_row(name, book.loc[era, col], book.loc[era, "r_book"])
        for name, col in (
            ("ride-flat - book", "r_ride_flat"),
            ("ride-hedged - book", "r_ride_hedged"),
        )
    }
    ok_a = all(v["excludes_zero_positive"] for v in gate_a.values())
    print(
        "\nGATE A  era paired differences: "
        + "; ".join(
            f"{k} mean {v['mean_diff']:+.6f}, HAC t {v['t_hac']:+.2f}, CI "
            f"[{v['ci_lo']:+.6f}, {v['ci_hi']:+.6f}]"
            for k, v in gate_a.items()
        )
        + f"  -> {'PASS' if ok_a else 'FAIL'}"
    )

    w10 = worst_ten(book)
    print("\n--- A5. the ten worst ride days (ranked on the hedged variant)")
    print(w10.to_string(float_format=lambda v: f"{v:,.4f}"))

    stress = part_b(book, g)

    oos_all = oos_books(chain, g)
    oos = oos_all.loc[
        (oos_all.index >= pd.Timestamp(OOS_START))
        & (oos_all.index <= pd.Timestamp(OOS_END))
    ]
    print(
        f"\n--- C1. unseen sessions {OOS_START}..{OOS_END}: "
        f"{int(((pd.DatetimeIndex(stamp.index) >= pd.Timestamp(OOS_START)) & (pd.DatetimeIndex(stamp.index) <= pd.Timestamp(OOS_END))).sum())} "
        f"in the chain, {int(((half >= pd.Timestamp(OOS_START)) & (half <= pd.Timestamp(OOS_END))).sum())} "
        f"half sessions dropped, {len(oos)} scored"
    )
    c_summary, c_diffs = part_c(oos)
    curve = premium_curve(chain, g)
    by_year = premium_by_year(chain)

    era_oos = c_diffs.loc[
        (c_diffs["period"] == "2024-05-01..2025-12-31") & (c_diffs["unit"] == "premium")
    ]
    ok_c = bool(era_oos["excludes_zero_positive"].all())
    book_row = stats_row("book", oos["r_book"])
    print(
        "\nGATE C  unseen-session paired differences (premium units): "
        + "; ".join(
            f"{r['difference']} mean {r['mean_diff']:+.6f}, HAC t {r['t_hac']:+.2f}, "
            f"CI [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]"
            for _, r in era_oos.iterrows()
        )
        + f"  -> {'PASS' if ok_c else 'FAIL'}"
    )
    print(
        f"GATE C  the book itself out of sample: n {book_row['n']}, mean "
        f"{book_row['mean']:+.6f}, HAC t {book_row['t_hac']:+.3f} (lag "
        f"{book_row['t_lag']}), Sharpe_ann {book_row['Sharpe_ann']:.4f}"
    )

    book.to_csv(OUT / "a_insample.csv")
    w10.to_csv(OUT / "a_worst10.csv")
    stress.to_csv(OUT / "b_stress.csv", index=False)
    oos.to_csv(OUT / "c_oos_daily.csv")
    c_summary.to_csv(OUT / "c_oos_summary.csv", index=False)
    curve.to_csv(OUT / "c_premium_curve.csv", index=False)
    a_summary.to_csv(OUT / "a_insample_summary.csv", index=False)
    c_diffs.to_csv(OUT / "c_oos_paired.csv", index=False)
    by_year.to_csv(OUT / "c_premium_by_year.csv")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
