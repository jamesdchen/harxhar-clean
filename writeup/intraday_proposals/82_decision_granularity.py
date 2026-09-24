"""Study 82 -- is 30 minutes the right granularity for the close card, now that ES 1-minute data exists?

QUESTION (the operator's).  The research line and the live card run on the
30-minute panel (bar-END stamps, naive ET): the card forecasts the variance of
the 15:30-16:00 ES bar with data through the 15:30 stamp and at 15:30 buys the
nearest-OTM 0DTE pair iff ask <= P* = ``live.ibkr.pricing.package_price``
(sqrt(rv_hat), S, Kc, Kp), held to the 16:00 cash settlement; month-ends buy
unconditionally.  Study 80: forecasts from 15:00 or earlier lose the edge.
Does finer resolution near the decision matter?

WHAT CAN BE ANSWERED NOW (data verified in the run log):
  ES.v.0 ohlcv-1m, Databento, 2024-03-31 .. 2026-09-23 (~617 sessions);
  XSP 0DTE cbbo-1m 15:20-16:16 ET, 2023-03-28 .. 2026-09-22 (all sessions);
  SPXW 0DTE cbbo-1m 15:20-16:16 ET on 42 month-ends; the vendor 30-minute panel
  (data/core_stats.parquet, to 2024-04-30) and live/close_signal/state/
  panel_free.parquet (Databento-derived 30-minute rows after it).

(a) FEATURES.  Target: the 15:30-16:00 ES realized variance (sum of squared
    1-minute log returns, = the panel's sumret2 at the 16:00 stamp).  Models:
    OLS of log RV on a HAR-style design, forecast = exp(x'b) * mean(exp(e)) over
    the training residuals (smearing, so the level is the conditional mean QLIKE
    wants), refit every session on strictly prior sessions.
      BASE (30-minute): log sumret2 at 15:30 and 15:00, log mean of the target
        over the prior 5 and 22 sessions, log RTH RV today to 15:30, log RTH RV
        of the prior session, log VIX^2 at the 15:00 stamp, month-end dummy.
      MOM  (the panel's other 30-minute moments at 15:30, bounded ratios):
        sumbipow/sumret2, sumpret2/sumret2, sumret/sqrt(sumret2),
        numobs*sumret4/sumret2^2.
      FINE (1-minute, inside 15:00-15:30): share of the bar's RV in its last 10
        and last 5 minutes (in [0, 1]), log 5-minute-sampled RV, log Parkinson
        range RV from the 1-minute highs/lows, max squared 1-minute return / RV.
    S-models are fitted on the ES 1-minute span only (expanding window, first
    forecast after MIN_TRAIN sessions; 22 more for the slot means).  L0 is BASE
    fitted on the LONG 30-minute panel (vendor 2004 .. 2024-04, Databento rows
    after; rolling 2000 sessions), and L0r / L4r re-fit log RV on log f_L0
    (+ FINE) on the ES span: the parsimonious way to use 2.5 years of 1-minute
    data (6 incremental coefficients instead of 14).  Scored by QLIKE on the
    common out-of-sample sessions; Diebold-Mariano (Newey-West, lag 5) and a
    circular block bootstrap (20 sessions) of the mean QLIKE difference.

(b) DECISION TIME.  For tau in 15:25 .. 15:50 the target is the ES RV from tau
    to 16:00 and the design is BASE moved to tau (log RV of [tau-30, tau) and
    [tau-60, tau-30), the target's own slot means, RTH RV to tau, ...) = T0, and
    T0 + FINE at tau = T1.  At tau = 15:30, T0 IS S0 and T1 IS S4 (gated).
    The card rule at tau on the XSP book: the nearest-OTM pair on the listed
    grid (call at/above, put at/below the put-call-parity spot of the book at
    tau), buy iff ask <= P*(sqrt(f_tau)), payoff at the official SPX close / 10;
    R = payoff / ask - 1, per-day P&L = R on a traded day, 0 otherwise; the same
    rule at the mid (mid <= P*, R at the mid) isolates the spread.  Month-ends
    are excluded from the conditional rule (the card buys them unconditionally)
    and reported as the unconditional long by tau on XSP and on SPXW.

GATES (the run stops on a failed hard gate):
  G1  the 1-minute ES aggregated to 30 minutes here (independent code, same
      SESSION_BREAK and within-contract rules) vs panel_free's databento_es rows
      on every stamp: max relative error per moment column.
  G2  the 30-minute baseline is sane: L0 on the deck's 866 days vs the deck's
      QLIKE (0.1054, same target: the vendor sumret2 at 16:00), ratio in
      [0.8, 1.3] and below the naive 22-session slot mean.
  G3  T0(15:30) == S0 and T1(15:30) == S4 forecast for forecast (1e-9 rel).
  G4  LEAK TEST: every 1-minute return at or after the decision minute on a
      test session is multiplied by sqrt(50); every feature of that session and
      all earlier sessions must be bit-identical (only the target moves);
      positive control: the minute before the decision must move the features.
  G5  XSP book: the parity spot at 15:30 vs the SPXW chain / deck 15:30 spot;
      the SPX close (yfinance ^GSPC) vs the deck's S_close; the 15:30 pair and
      ask vs study 79's daily table; the book's clock: the parity-spot move
      15:30 -> tau against the ES move over the same minutes shifted -2..+2,
      zero shift must fit best (median |gap| < 1 bp).

OUTPUTS (results/atm_straddle_intraday_holdclose/proposals/82/):
  gate.json, gate_es_30min_parity.csv, leak_test.csv   the gates (+ (c) cost estimates)
  features_1530.parquet, forecasts_1530.parquet, qlike_1530.csv   part (a)
  forecasts_tau.parquet, qlike_tau.csv                 part (b) forecasts
  books_tau.parquet                                    XSP / SPXW books at every tau
  per_day_pnl_tau.parquet/.csv, summary_tau.csv, bootstrap_tau.csv,
  rule_1530_by_model.csv, month_end_tau.csv            part (b) trading
  spx_close_yf.csv                                     ^GSPC closes (cache)
  summary.txt, run.log                                 the printed report

    C:/Users/james/miniconda3/envs/285J/python.exe writeup/intraday_proposals/82_decision_granularity.py
    (about 2 minutes; the option books run in a process pool, --workers 8)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.ibkr.calendar_guard import is_last_session_of_month, is_session  # noqa: E402
from live.ibkr.pricing import package_price  # noqa: E402

ET = "America/New_York"
OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "82"
ES_CSV = REPO / "data" / "archive" / "es_v0_ohlcv1m_databento.csv"
PANEL_FREE = REPO / "live" / "close_signal" / "state" / "panel_free.parquet"
VENDOR_CORE = REPO / "data" / "core_stats.parquet"
VENDOR_CBOE = REPO / "data" / "vix_and_voldemand.parquet"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
CHAIN_SPOT = REPO / "data" / "spxw_spot.parquet"
XSP_DIR = REPO / "data" / "archive" / "xsp_opra"
SPXW_DIR = REPO / "data" / "archive" / "spxw_opra"
S79_TABLE = (
    REPO
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "79"
    / "xsp_close_book_daily.csv"
)
S79_SRC = REPO / "writeup" / "intraday_proposals" / "79_xsp_close_book.py"

CORE_COLS = (
    "sumret",
    "sumabsret",
    "sumret2",
    "sumret3",
    "sumret4",
    "sumpret2",
    "sumbipow",
    "sumautocov",
    "sumvolume",
    "numobs",
)
SESSION_BREAK = pd.Timedelta(minutes=60)  # live/close_signal/features.py's rule
VENDOR_LAST_DAY = pd.Timestamp("2024-04-30")
LONG_FIRST_DAY = pd.Timestamp("2004-01-01")
RTH0, RTH1 = 570, 960  # minute of day: 09:30 .. 16:00 (bar STARTS 09:30 .. 15:59)
RTH_STAMPS = tuple(f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30))[:13]
MIN_RTH_MINUTES = 385  # of 390: an early close or a data hole drops the session
TAUS = ("15:25", "15:30", "15:35", "15:40", "15:45", "15:50")
REF_TAU = "15:30"
MIN_TRAIN = 150
MIN_TRAIN_ALT = 250
SLOT_SHORT, SLOT_LONG = 5, 22
LONG_WIN, LONG_MIN = 2000, 1000
NW_LAG = 5
BLOCK = 20
N_BOOT = 10_000
SEED = 82
LEAK_FACTOR = 50.0
# numpy's pairwise sums of a copied array can differ in the last bit (alignment),
# so "unchanged" is 1e-12 relative (measured deviations are ~1e-16) and "moved"
# is 1e-6 relative (a x50 variance perturbation moves a log feature by O(1)).
LEAK_TOL = 1e-12
MOVE_TOL = 1e-6
DECK_QLIKE_BAND = (0.8, 1.3)
XSP_SCALE = 0.1

BASE = ["lrv_a", "lrv_b", "lslot5", "lslot22", "lrth", "lrth_prev", "lvix2", "me"]
MOM = ["mom_bipow", "mom_pret2", "mom_zret", "mom_quart"]
FINE = ["f_share10", "f_share5", "f_lrv5", "f_lpark", "f_jump"]
S_MODELS: dict[str, list[str]] = {
    "S0_base30": BASE,
    "S0m_base30+mom30": BASE + MOM,
    "S1_+recency": BASE + ["f_share10", "f_share5"],
    "S2_+rv5+park": BASE + ["f_lrv5", "f_lpark"],
    "S3_+jump": BASE + ["f_jump"],
    "S4_+fine": BASE + FINE,
    "S5_+mom30+fine": BASE + MOM + FINE,
}
L_MODELS: dict[str, list[str]] = {
    "L0r_recal": ["lf_L0"],
    "L4r_recal+fine": ["lf_L0"] + FINE,
}


_LOG: list[str] = []


def log(msg: str) -> None:
    _LOG.append(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------- ES 1-minute
def es_minutes(path: Path = ES_CSV) -> pd.DataFrame:
    """ES.v.0 1-minute bars, naive-ET bar START, returns within contract segments.

    r_i = ln(close_i / close_{i-1}) inside a run of one instrument_id; a gap of
    SESSION_BREAK or more (CME maintenance hour, weekend) carries no return.
    """
    df = pd.read_csv(
        path,
        usecols=[
            "ts_event",
            "instrument_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        ],
    )
    df = df[df["symbol"].astype(str) == "ES.v.0"]
    t = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    df = df.assign(t=t.to_numpy())
    df = df.drop_duplicates(["t", "instrument_id"], keep="last")
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    if df["t"].duplicated().any():
        raise AssertionError("two instruments on one minute in a continuous file")
    seg = (df["instrument_id"] != df["instrument_id"].shift()).cumsum()
    r = np.log(df["close"].astype(float)).groupby(seg).diff()
    gap = df["t"].groupby(seg).diff()
    r = r.mask(gap >= SESSION_BREAK)
    df["seg"] = seg.to_numpy()
    df["r"] = r.to_numpy()
    df["end"] = df["t"].dt.floor("30min") + pd.Timedelta(minutes=30)
    return df


def moments_30(df: pd.DataFrame) -> pd.DataFrame:
    """CORE_COLS per 30-minute bar END (this study's own implementation)."""
    r_prev_raw = df["r"].groupby(df["seg"]).shift(1)
    g = df[df["r"].notna()].copy()
    g["r_prev_raw"] = r_prev_raw.loc[g.index].to_numpy()
    prev_end = g["end"].groupby(g["seg"]).shift(1)
    same = (g["end"] == prev_end).to_numpy()
    rp = np.where(same, g["r_prev_raw"].to_numpy(), np.nan)
    r = g["r"].to_numpy()
    fin = np.isfinite(rp)
    g["absret"] = np.abs(r)
    g["ret2"] = r**2
    g["ret3"] = r**3
    g["ret4"] = r**4
    g["pret2"] = np.where(r > 0, r**2, 0.0)
    g["bipow"] = np.where(fin, np.abs(r) * np.abs(np.nan_to_num(rp)), 0.0)
    g["autocov"] = np.where(fin, r * np.nan_to_num(rp), 0.0)
    g["volume"] = g["volume"].astype(float)
    agg = g.groupby("end").agg(
        sumret=("r", "sum"),
        sumabsret=("absret", "sum"),
        sumret2=("ret2", "sum"),
        sumret3=("ret3", "sum"),
        sumret4=("ret4", "sum"),
        sumpret2=("pret2", "sum"),
        sumbipow=("bipow", "sum"),
        sumautocov=("autocov", "sum"),
        sumvolume=("volume", "sum"),
        numobs=("r", "size"),
    )
    agg["numobs"] = agg["numobs"].astype(float)
    agg.index.name = "endbartime"
    return agg


def gate_parity(mine: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """G1: this study's 30-minute moments vs panel_free's databento_es rows."""
    p = pd.read_parquet(PANEL_FREE)
    p["endbartime"] = pd.to_datetime(p["endbartime"])
    st = p[p["source"] == "databento_es"].set_index("endbartime")
    both = mine.index.intersection(st.index)
    lo, hi = both.min(), both.max()
    m_span = mine.index[(mine.index >= lo) & (mine.index <= hi)]
    rows = []
    for c in CORE_COLS:
        a = mine.loc[both, c].to_numpy(float)
        b = st.loc[both, c].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        den = np.maximum(np.abs(a[ok]), np.abs(b[ok]))
        rel = np.where(den > 0, np.abs(a[ok] - b[ok]) / np.where(den > 0, den, 1), 0.0)
        rows.append(
            {
                "column": c,
                "stamps": int(ok.sum()),
                "nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
                "median_rel": float(np.median(rel)),
                "p99_rel": float(np.quantile(rel, 0.99)),
                "max_rel": float(rel.max()),
                "n_rel_gt_1e-9": int((rel > 1e-9).sum()),
            }
        )
    rep = pd.DataFrame(rows)
    info = {
        "store_databento_rows": int(len(st)),
        "mine_rows": int(len(mine)),
        "overlap_stamps": int(len(both)),
        "stamps_only_in_mine_within_span": int((~m_span.isin(st.index)).sum()),
        "stamps_only_in_store": int((~st.index.isin(mine.index)).sum()),
        "max_rel_any_column": float(rep["max_rel"].max()),
    }
    info["passed"] = bool(
        info["max_rel_any_column"] <= 1e-9
        and info["stamps_only_in_store"] == 0
        and int(rep["nan_mismatch"].sum()) == 0
    )
    return rep, info


def minute_matrices(df: pd.DataFrame) -> dict:
    """Session x minute (09:30 .. 15:59 bar starts) matrices of r and Parkinson terms."""
    mi = df["t"].dt.hour * 60 + df["t"].dt.minute
    rth = df[(mi >= RTH0) & (mi < RTH1)].copy()
    rth["mi"] = mi[rth.index].to_numpy()
    rth["day"] = rth["t"].dt.normalize()
    rth = rth[[is_session(d.date()) for d in rth["day"]]]
    cols = np.arange(RTH0, RTH1)
    R = rth.pivot(index="day", columns="mi", values="r").reindex(columns=cols)
    hl = np.log(rth["high"].astype(float) / rth["low"].astype(float)) ** 2 / (
        4.0 * np.log(2.0)
    )
    PK = (
        rth.assign(pk=hl.to_numpy())
        .pivot(index="day", columns="mi", values="pk")
        .reindex(columns=cols)
    )
    n_valid = R.notna().sum(1)
    keep = n_valid >= MIN_RTH_MINUTES
    return {
        "days": pd.DatetimeIndex(R.index[keep]),
        "R": R[keep].to_numpy(float),
        "PK": PK[keep].to_numpy(float),
        "dropped": [str(d.date()) for d in R.index[~keep]],
        "n_valid_min": int(n_valid[keep].min()),
    }


def vix_1500() -> pd.Series:
    """VIX at the 15:00 stamp by day: the vendor to its last print, panel_free after.

    The vendor's Cboe feed ends before its ES rows do (last finite VIX
    2024-02-12); panel_free's Cboe rows start the next day.  A source switch
    at the seam, not a fill: a day neither source has stays NaN and is dropped.
    """
    v = pd.read_parquet(VENDOR_CBOE, columns=["endbartime", "vix"])
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    v = v[v["endbartime"].dt.strftime("%H:%M") == "15:00"]
    v = pd.Series(
        v["vix"].to_numpy(float), index=v["endbartime"].dt.normalize().to_numpy()
    ).dropna()
    last_v = v.index.max()
    p = pd.read_parquet(PANEL_FREE, columns=["endbartime", "vix"])
    p["endbartime"] = pd.to_datetime(p["endbartime"])
    p = p[p["endbartime"].dt.strftime("%H:%M") == "15:00"]
    p = pd.Series(
        p["vix"].to_numpy(float), index=p["endbartime"].dt.normalize().to_numpy()
    ).dropna()
    p = p[p.index > last_v]
    s = pd.concat([v, p]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.attrs["seam"] = (
        f"vendor to {last_v.date()}, panel_free from {p.index.min().date()}"
    )
    return s


# --------------------------------------------------------------------------- features
def slot_means(y: np.ndarray, k: int) -> np.ndarray:
    """Mean of y over the k previous rows (sessions); NaN until k exist."""
    return pd.Series(y).shift(1).rolling(k, min_periods=k).mean().to_numpy()


def tau_features(
    R: np.ndarray, PK: np.ndarray, days: pd.DatetimeIndex, vix: pd.Series, m: int
) -> pd.DataFrame:
    """Every feature at decision minute m (data = minutes starting before m) + the target."""
    R0 = np.nan_to_num(R)
    R2 = R0**2
    PK0 = np.nan_to_num(PK)

    def rv(a: int, b: int) -> np.ndarray:
        return R2[:, a - RTH0 : b - RTH0].sum(1)

    n = len(days)
    y = rv(m, RTH1)
    a30 = rv(m - 30, m)
    b30 = rv(m - 60, m - 30)
    w = slice(m - 30 - RTH0, m - RTH0)
    blk = R0[:, w].reshape(n, 6, 5).sum(2)
    rth_full = rv(RTH0, RTH1)
    out = pd.DataFrame(
        {
            "y": y,
            "lrv_a": np.log(a30),
            "lrv_b": np.log(b30),
            "lslot5": np.log(slot_means(y, SLOT_SHORT)),
            "lslot22": np.log(slot_means(y, SLOT_LONG)),
            "lrth": np.log(rv(RTH0, m)),
            "lrth_prev": np.log(pd.Series(rth_full).shift(1).to_numpy()),
            "lvix2": 2.0 * np.log(vix.reindex(days).to_numpy(float)),
            "me": np.array([float(is_last_session_of_month(d.date())) for d in days]),
            "f_share10": rv(m - 10, m) / a30,
            "f_share5": rv(m - 5, m) / a30,
            "f_lrv5": np.log((blk**2).sum(1)),
            "f_lpark": np.log(PK0[:, w].sum(1)),
            "f_jump": R2[:, w].max(1) / a30,
        },
        index=days,
    )
    out.index.name = "day"
    return out


def s_features(
    mom: pd.DataFrame, mm: dict, vix: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a)'s table: BASE + MOM from the 30-minute rows, FINE from the minutes, at 15:30."""
    days = mm["days"]
    wide = {}
    for c in ("sumret2", "sumret", "sumbipow", "sumpret2", "sumret4", "numobs"):
        s = mom[c]
        s = s[s.index.normalize().isin(days)]
        wide[c] = (
            pd.DataFrame(
                {
                    "d": s.index.normalize(),
                    "k": s.index.strftime("%H:%M"),
                    "v": s.values,
                }
            )
            .pivot(index="d", columns="k", values="v")
            .reindex(index=days)
        )
    s2 = wide["sumret2"]
    y = s2["16:00"].to_numpy(float)
    rth_to = s2[list(RTH_STAMPS[:12])].sum(1, min_count=12).to_numpy(float)
    rth_full = s2[list(RTH_STAMPS)].sum(1, min_count=13).to_numpy(float)
    a = s2["15:30"].to_numpy(float)
    t = tau_features(mm["R"], mm["PK"], days, vix, 930)
    out = pd.DataFrame(
        {
            "y": y,
            "lrv_a": np.log(a),
            "lrv_b": np.log(s2["15:00"].to_numpy(float)),
            "lslot5": np.log(slot_means(y, SLOT_SHORT)),
            "lslot22": np.log(slot_means(y, SLOT_LONG)),
            "lrth": np.log(rth_to),
            "lrth_prev": np.log(pd.Series(rth_full).shift(1).to_numpy()),
            "lvix2": t["lvix2"].to_numpy(),
            "me": t["me"].to_numpy(),
            "mom_bipow": wide["sumbipow"]["15:30"].to_numpy(float) / a,
            "mom_pret2": wide["sumpret2"]["15:30"].to_numpy(float) / a,
            "mom_zret": wide["sumret"]["15:30"].to_numpy(float) / np.sqrt(a),
            "mom_quart": wide["numobs"]["15:30"].to_numpy(float)
            * wide["sumret4"]["15:30"].to_numpy(float)
            / a**2,
        },
        index=days,
    )
    for c in FINE:
        out[c] = t[c].to_numpy()
    out.index.name = "day"
    return out, t


def long_table(mom: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """BASE on the long 30-minute panel: vendor 2004 .. 2024-04-30, this study's rows after."""
    v = pd.read_parquet(VENDOR_CORE, columns=["endbartime", "sumret2"])
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    v = v[
        (v["endbartime"] >= LONG_FIRST_DAY)
        & (v["endbartime"] < VENDOR_LAST_DAY + pd.Timedelta(days=1))
    ]
    mine = mom["sumret2"].reset_index()
    mine = mine[mine["endbartime"] >= VENDOR_LAST_DAY + pd.Timedelta(days=1)]
    s = pd.concat([v, mine], ignore_index=True)
    s["d"] = s["endbartime"].dt.normalize()
    s["k"] = s["endbartime"].dt.strftime("%H:%M")
    s = s[s["k"].isin(RTH_STAMPS)]
    w = s.pivot_table(index="d", columns="k", values="sumret2", aggfunc="last")[
        list(RTH_STAMPS)
    ]
    w = w[[is_session(d.date()) for d in w.index]]
    w = w[(w > 0).all(1)]
    days = pd.DatetimeIndex(w.index)
    y = w["16:00"].to_numpy(float)
    lt = pd.DataFrame(
        {
            "y": y,
            "lrv_a": np.log(w["15:30"].to_numpy(float)),
            "lrv_b": np.log(w["15:00"].to_numpy(float)),
            "lslot5": np.log(slot_means(y, SLOT_SHORT)),
            "lslot22": np.log(slot_means(y, SLOT_LONG)),
            "lrth": np.log(w[list(RTH_STAMPS[:12])].sum(1).to_numpy(float)),
            "lrth_prev": np.log(
                pd.Series(w[list(RTH_STAMPS)].sum(1).to_numpy(float))
                .shift(1)
                .to_numpy()
            ),
            "lvix2": 2.0 * np.log(vix.reindex(days).to_numpy(float)),
            "me": np.array([float(is_last_session_of_month(d.date())) for d in days]),
        },
        index=days,
    )
    lt.index.name = "day"
    return lt


# --------------------------------------------------------------------------- models
def ols_forecasts(
    X: np.ndarray, ly: np.ndarray, min_train: int, window: int | None = None
) -> np.ndarray:
    """Session-by-session refit on strictly prior rows; exp(x'b) * mean(exp(e)) smearing."""
    n = len(ly)
    Xc = np.column_stack([np.ones(n), X])
    f = np.full(n, np.nan)
    for t in range(n):
        lo = 0 if window is None else max(0, t - window)
        if t - lo < min_train:
            continue
        A, b = Xc[lo:t], ly[lo:t]
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        e = b - A @ beta
        f[t] = float(np.exp(Xc[t] @ beta) * np.mean(np.exp(e)))
    return f


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def nw_t(d: np.ndarray, lag: int = NW_LAG) -> float:
    d = np.asarray(d, float)
    n = len(d)
    u = d - d.mean()
    s = u @ u / n
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1)) * (u[k:] @ u[:-k] / n)
    return float(d.mean() / np.sqrt(s / n))


def block_boot_idx(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, nb * block)[:, :n]


def boot_ci(d: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bm = d[block_boot_idx(len(d), BLOCK, N_BOOT, rng)].mean(1)
    return (
        float(np.quantile(bm, 0.025)),
        float(np.quantile(bm, 0.975)),
        float((bm < 0).mean()),
    )


def stats(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    n = len(r)
    sd = r.std(ddof=1) if n > 1 else np.nan
    return {
        "n": n,
        "mean_R": float(r.mean()) if n else np.nan,
        "t": float(r.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
        "hit": float((r > 0).mean()) if n else np.nan,
    }


# --------------------------------------------------------------------------- option books
_S79 = None


def _s79():
    global _S79
    if _S79 is None:
        spec = importlib.util.spec_from_file_location("s79_xsp_close_book", S79_SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _S79 = mod
    return _S79


def books_for_day(args: tuple[str, str, str, str]) -> list[dict]:
    """The nearest-OTM pair at the BBO standing at each tau (study 79's helpers)."""
    qpath, dpath, day_s, venue = args
    s79 = _s79()
    day = pd.Timestamp(day_s)
    defs = pd.read_parquet(dpath)
    if defs.empty:
        return []
    strikes = np.sort(defs["strike_price"].astype(float).unique())
    q = s79.load_quotes(Path(qpath))
    rows = []
    for tau in TAUS:
        stamp = pd.Timestamp(f"{day.date()} {tau}:00")
        b = s79.book_at(q, day, f"{tau}:00")
        if b.empty:
            continue
        bt = b.index.get_level_values(0)  # strike level; book time from q
        s = s79.parity_spot(b)
        kc, kp = s79.nearest_otm(strikes, s) if np.isfinite(s) else (np.nan, np.nan)
        bc, ac, bsc, asc = s79.quote(b, kc, "C")
        bp, ap, bsp, asp = s79.quote(b, kp, "P")
        bid, ask = bc + bp, ac + ap
        t_book = q.loc[q["t"] <= stamp, "t"].max()
        rows.append(
            {
                "day": day,
                "venue": venue,
                "tau": tau,
                "book_lag_s": float((stamp - t_book).total_seconds()),
                "S_book": s,
                "kc": kc,
                "kp": kp,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
                "ask_size": min(asc, asp),
                "n_strikes_quoted": int(len(np.unique(bt))),
            }
        )
    return rows


def spx_closes(first: pd.Timestamp, last: pd.Timestamp) -> pd.Series:
    """Official SPX closes (^GSPC daily, yfinance), cached in this study's folder."""
    cache = OUT / "spx_close_yf.csv"
    if cache.exists():
        c = pd.read_csv(cache, index_col=0, parse_dates=True)["close"]
        if c.index.min() <= first and c.index.max() >= last - pd.Timedelta(days=3):
            return c
    import yfinance as yf

    h = yf.Ticker("^GSPC").history(
        start=first.strftime("%Y-%m-%d"),
        end=(last + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
    )
    c = pd.Series(
        h["Close"].to_numpy(float),
        index=pd.DatetimeIndex(h.index.tz_localize(None).normalize()),
        name="close",
    )
    c.to_frame().to_csv(cache)
    return c


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gate: dict[str, object] = {}

    # ---- data
    df = es_minutes()
    log(
        f"ES 1-minute: {len(df)} bars {df['t'].min()} .. {df['t'].max()} (naive ET bar start), "
        f"{df['instrument_id'].nunique()} contracts, {int(df['seg'].max())} segments"
    )
    mom = moments_30(df)

    # ---- G1: 30-minute parity with panel_free
    rep, g1 = gate_parity(mom)
    rep.to_csv(OUT / "gate_es_30min_parity.csv", index=False)
    gate["G1_es_30min_vs_panel_free"] = g1
    log(
        "G1 own 30-min moments vs panel_free databento_es rows:\n"
        + rep.to_string(index=False)
    )
    log(json.dumps(g1))
    if not g1["passed"]:
        log("G1 FAILED")
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        return 2

    mm = minute_matrices(df)
    vix = vix_1500()
    log(f"VIX at 15:00: {vix.attrs['seam']}; {len(vix)} days")
    log(
        f"sessions with >= {MIN_RTH_MINUTES}/390 RTH minutes: {len(mm['days'])} "
        f"({mm['days'].min().date()} .. {mm['days'].max().date()}); dropped {len(mm['dropped'])}: "
        f"{', '.join(mm['dropped'])}"
    )

    # ---- (a) the 15:30 table
    S, T1530 = s_features(mom, mm, vix)
    # internal: the 30-minute rows vs the minute matrix at 15:30
    d_int = float(
        np.nanmax(
            np.abs(np.exp(S["lrv_a"]) - np.exp(T1530["lrv_a"])) / np.exp(S["lrv_a"])
        )
    )
    d_y = float(np.nanmax(np.abs(S["y"] - T1530["y"]) / S["y"]))
    log(
        f"internal: sumret2(15:30) vs minute sums max rel {d_int:.1e}; target {d_y:.1e}"
    )
    L = long_table(mom, vix)
    L = L[L.notna().all(1)]
    fL = ols_forecasts(
        L[BASE].to_numpy(), np.log(L["y"].to_numpy()), LONG_MIN, window=LONG_WIN
    )
    L["f_L0"] = fL
    L["f_naive22"] = np.exp(L["lslot22"])
    log(
        f"long table: {len(L)} sessions {L.index.min().date()} .. {L.index.max().date()}"
    )
    S["lf_L0"] = np.log(L["f_L0"].reindex(S.index))
    # (b)'s tables; ONE session list for (a) and (b): every feature finite, no fill
    tabs = {}
    for tau in TAUS:
        hh, mn = (int(x) for x in tau.split(":"))
        tabs[tau] = tau_features(mm["R"], mm["PK"], mm["days"], vix, hh * 60 + mn)
    need = sorted(set(sum(S_MODELS.values(), []))) + ["lf_L0"]
    ok_rows = S[need + ["y"]].notna().all(1) & np.isfinite(S[need].to_numpy()).all(1)
    common = S.index[ok_rows]
    for tau, t in tabs.items():
        ok = t[BASE + FINE + ["y"]].notna().all(1) & np.isfinite(
            t[BASE + FINE].to_numpy()
        ).all(1)
        common = common.intersection(t.index[ok])
    log(
        f"sessions with every feature at every tau: {len(common)} of {len(S)} "
        f"({common.min().date()} .. {common.max().date()}; the first {SLOT_LONG} lack the "
        f"slot mean; VIX missing on {int(S['lvix2'].isna().sum())})"
    )
    S = S.loc[common].copy()

    # ---- G2: the baseline vs the deck
    deck = pd.read_parquet(DECK)
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    j = (
        deck[["rv_hat", "S_close"]]
        .join(L[["y", "f_L0", "f_naive22"]], how="inner")
        .dropna()
    )
    q_deck = float(qlike(j["y"].to_numpy(), j["rv_hat"].to_numpy()).mean())
    q_l0 = float(qlike(j["y"].to_numpy(), j["f_L0"].to_numpy()).mean())
    q_nv = float(qlike(j["y"].to_numpy(), j["f_naive22"].to_numpy()).mean())
    apr = j[j.index >= pd.Timestamp("2024-04-01")]
    g2 = {
        "deck_days_matched": int(len(j)),
        "qlike_deck": q_deck,
        "qlike_L0": q_l0,
        "qlike_naive_slot22": q_nv,
        "ratio_L0_over_deck": q_l0 / q_deck,
        "april2024_days": int(len(apr)),
        "april2024_qlike_deck": float(
            qlike(apr["y"].to_numpy(), apr["rv_hat"].to_numpy()).mean()
        ),
        "april2024_qlike_L0": float(
            qlike(apr["y"].to_numpy(), apr["f_L0"].to_numpy()).mean()
        ),
    }
    g2["passed"] = bool(
        DECK_QLIKE_BAND[0] <= g2["ratio_L0_over_deck"] <= DECK_QLIKE_BAND[1]
        and q_l0 < q_nv
    )
    gate["G2_baseline_vs_deck"] = g2
    log("G2 baseline vs deck: " + json.dumps(g2))
    if not g2["passed"]:
        log("G2 FAILED")
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        return 2

    # ---- (a) models on the ES span
    ly = np.log(S["y"].to_numpy())
    fc_rows = {}
    fc_all = {}
    qa_rows = []
    for mt in (MIN_TRAIN, MIN_TRAIN_ALT):
        F = pd.DataFrame(index=S.index)
        for name, cols in {**S_MODELS, **L_MODELS}.items():
            F[name] = ols_forecasts(S[cols].to_numpy(), ly, mt)
        F["L0_long"] = L["f_L0"].reindex(S.index).to_numpy()
        fc_all[mt] = F.copy()
        # the common out-of-sample sessions: every model has a forecast
        F = F.dropna()
        y = S.loc[F.index, "y"].to_numpy()
        me = S.loc[F.index, "me"].to_numpy() > 0
        fc_rows[mt] = F
        Q = {k: qlike(y, F[k].to_numpy()) for k in F.columns}
        for k in F.columns:
            ref = "L0r_recal" if k.startswith("L4r") else "S0_base30"
            dq = Q[k] - Q[ref]
            lo, hi, p_neg = boot_ci(dq) if k != ref else (np.nan, np.nan, np.nan)
            qa_rows.append(
                {
                    "min_train": mt,
                    "model": k,
                    "n_params": 1 + len({**S_MODELS, **L_MODELS}.get(k, BASE)),
                    "oos_days": int(len(y)),
                    "oos_first": str(F.index.min().date()),
                    "oos_last": str(F.index.max().date()),
                    "qlike": float(Q[k].mean()),
                    "qlike_ex_month_end": float(Q[k][~me].mean()),
                    "ref": ref,
                    "dqlike_vs_ref": float(dq.mean()),
                    "dqlike_pct": float(dq.mean() / Q[ref].mean()),
                    "dm_t_nw5": nw_t(dq) if k != ref else np.nan,
                    "boot_ci_lo": lo,
                    "boot_ci_hi": hi,
                    "p_boot_better": p_neg,
                }
            )
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(OUT / "qlike_1530.csv", index=False)
    S.to_parquet(OUT / "features_1530.parquet")
    fc_rows[MIN_TRAIN].join(S["y"]).to_parquet(OUT / "forecasts_1530.parquet")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    log("\n(a) QLIKE of the 15:30-16:00 ES RV forecast, common out-of-sample sessions:")
    log(
        qa[
            [
                "min_train",
                "model",
                "n_params",
                "oos_days",
                "qlike",
                "qlike_ex_month_end",
                "dqlike_pct",
                "dm_t_nw5",
                "boot_ci_lo",
                "boot_ci_hi",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    # ---- (b) forecasts at every tau
    fT = {}
    for tau in TAUS:
        t = tabs[tau].loc[common]
        lyt = np.log(t["y"].to_numpy())
        fT[(tau, "T0")] = ols_forecasts(t[BASE].to_numpy(), lyt, MIN_TRAIN)
        fT[(tau, "T1")] = ols_forecasts(t[BASE + FINE].to_numpy(), lyt, MIN_TRAIN)

    # ---- G3: T0(15:30) is S0, T1(15:30) is S4 (on the same sessions)
    f_s0 = fc_all[MIN_TRAIN]["S0_base30"].to_numpy()
    f_s4 = fc_all[MIN_TRAIN]["S4_+fine"].to_numpy()
    g3 = {
        "sessions": int(len(common)),
        "max_rel_T0_vs_S0": float(np.nanmax(np.abs(fT[(REF_TAU, "T0")] / f_s0 - 1))),
        "max_rel_T1_vs_S4": float(np.nanmax(np.abs(fT[(REF_TAU, "T1")] / f_s4 - 1))),
    }
    g3["passed"] = bool(max(g3["max_rel_T0_vs_S0"], g3["max_rel_T1_vs_S4"]) <= 1e-9)
    gate["G3_tau1530_is_a"] = g3
    log("G3 " + json.dumps(g3))
    if not g3["passed"]:
        log("G3 FAILED")
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        return 2

    # ---- G4: leak test on the minute matrices
    k = len(mm["days"]) // 2
    lt_rows = []
    for tau in TAUS:
        hh, mn = (int(x) for x in tau.split(":"))
        m = hh * 60 + mn
        ref_t = tau_features(mm["R"], mm["PK"], mm["days"], vix, m)
        for case, cols_lo, expect_same in (
            ("after_decision", m, True),
            ("minute_before_decision", m - 1, False),
        ):
            Rp = mm["R"].copy()
            PKp = mm["PK"].copy()
            if expect_same:
                Rp[k, cols_lo - RTH0 :] *= np.sqrt(LEAK_FACTOR)
                PKp[k, cols_lo - RTH0 :] *= LEAK_FACTOR
            else:
                Rp[k, cols_lo - RTH0] *= np.sqrt(LEAK_FACTOR)
                PKp[k, cols_lo - RTH0] *= LEAK_FACTOR
            pt = tau_features(Rp, PKp, mm["days"], vix, m)
            feats = BASE + FINE
            upto = slice(0, k + 1)
            a_ = ref_t.iloc[upto][feats].to_numpy()
            b_ = pt.iloc[upto][feats].to_numpy()
            fin = np.isfinite(a_) & np.isfinite(b_)
            nan_same = bool((np.isfinite(a_) == np.isfinite(b_)).all())
            dev = float(
                (np.abs(a_[fin] - b_[fin]) / np.maximum(np.abs(a_[fin]), 1e-300)).max()
            )
            dev_day = float(
                np.nanmax(np.abs(a_[-1] - b_[-1]) / np.maximum(np.abs(a_[-1]), 1e-300))
            )
            y_dev = float(abs(pt["y"].iloc[k] / ref_t["y"].iloc[k] - 1))
            lt_rows.append(
                {
                    "tau": tau,
                    "case": case,
                    "day": str(mm["days"][k].date()),
                    "max_rel_dev_features_through_day": dev,
                    "max_rel_dev_features_on_day": dev_day,
                    "rel_dev_target_on_day": y_dev,
                    "pass": (
                        nan_same and dev <= LEAK_TOL and y_dev > MOVE_TOL
                        if expect_same
                        else dev_day > MOVE_TOL
                    ),
                }
            )
    lt = pd.DataFrame(lt_rows)
    lt.to_csv(OUT / "leak_test.csv", index=False)
    gate["G4_leak_test_pass"] = bool(lt["pass"].all())
    log("G4 leak test:\n" + lt.to_string(index=False))
    if not lt["pass"].all():
        log("G4 FAILED")
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        return 2

    ftab = []
    for tau in TAUS:
        t = tabs[tau].loc[common]
        for mdl in ("T0", "T1"):
            ftab.append(
                pd.DataFrame(
                    {
                        "day": common,
                        "tau": tau,
                        "model": mdl,
                        "y": t["y"].to_numpy(),
                        "f": fT[(tau, mdl)],
                        "me": t["me"].to_numpy() > 0,
                    }
                )
            )
    ftab = pd.concat(ftab, ignore_index=True)
    ftab.to_parquet(OUT / "forecasts_tau.parquet", index=False)
    qb = []
    for tau in TAUS:
        a0 = ftab[(ftab.tau == tau) & (ftab.model == "T0")].dropna()
        a1 = ftab[(ftab.tau == tau) & (ftab.model == "T1")].dropna()
        q0 = qlike(a0["y"].to_numpy(), a0["f"].to_numpy())
        q1 = qlike(a1["y"].to_numpy(), a1["f"].to_numpy())
        lo, hi, _ = boot_ci(q1 - q0)
        qb.append(
            {
                "tau": tau,
                "horizon_min": RTH1 - (int(tau[:2]) * 60 + int(tau[3:])),
                "oos_days": int(len(a0)),
                "qlike_T0": float(q0.mean()),
                "qlike_T1": float(q1.mean()),
                "dqlike_pct": float((q1 - q0).mean() / q0.mean()),
                "dm_t_nw5": nw_t(q1 - q0),
                "boot_ci_lo": lo,
                "boot_ci_hi": hi,
            }
        )
    qb = pd.DataFrame(qb)
    qb.to_csv(OUT / "qlike_tau.csv", index=False)
    log(
        "\n(b) QLIKE of the tau->16:00 RV forecast, T0 (30-min design at tau) vs T1 (+FINE):"
    )
    log(qb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- option books at every tau (process pool over day files)
    jobs = []
    for venue, d_ in (("XSP", XSP_DIR), ("SPXW", SPXW_DIR)):
        for qp in sorted(d_.glob("cbbo1m_*.parquet")):
            ds = qp.stem.split("_")[1]
            dp = d_ / f"defs_{ds}.parquet"
            if dp.exists():
                jobs.append((str(qp), str(dp), ds, venue))
    t_b = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(books_for_day, jobs, chunksize=8))
    books = pd.DataFrame([r for rr in res for r in rr])
    books.to_parquet(OUT / "books_tau.parquet", index=False)
    log(
        f"books: {len(jobs)} day files -> {len(books)} (day, tau) rows in {time.time() - t_b:.0f}s; "
        f"XSP days {books[books.venue == 'XSP'].day.nunique()}, SPXW month-ends "
        f"{books[books.venue == 'SPXW'].day.nunique()}; book lag > 0 s on "
        f"{int((books.book_lag_s > 0).sum())} rows"
    )
    closes = spx_closes(books["day"].min() - pd.Timedelta(days=5), books["day"].max())

    # ---- G5: spot, close, and study 79's 15:30 pair
    x30 = books[(books.venue == "XSP") & (books.tau == REF_TAU)].set_index("day")
    cs = pd.read_parquet(CHAIN_SPOT)
    tt = pd.to_datetime(cs["timestamp"])
    if getattr(tt.dt, "tz", None) is not None:
        tt = tt.dt.tz_convert(ET).dt.tz_localize(None)
    cs = cs.assign(t=tt)
    cs = cs[cs["t"].dt.strftime("%H:%M") == "15:30"]
    c1530 = pd.Series(
        cs["spot"].astype(float).to_numpy(), index=cs["t"].dt.normalize().to_numpy()
    )
    c1530 = c1530[~c1530.index.duplicated(keep="last")]
    ref_s = pd.concat([deck["S"], c1530[~c1530.index.isin(deck.index)]])
    sp = (x30["S_book"] / XSP_SCALE / ref_s.reindex(x30.index) - 1).dropna().abs()
    cl = (closes.reindex(deck.index) / deck["S_close"] - 1).dropna().abs()
    s79t = pd.read_csv(S79_TABLE, index_col=0, parse_dates=True)
    jj = x30.join(s79t[["kc", "kp", "ask"]], rsuffix="_79", how="inner").dropna(
        subset=["kc", "kc_79"]
    )
    same_pair = (jj.kc == jj.kc_79) & (jj.kp == jj.kp_79)
    g5: dict[str, object] = {
        "parity_spot_1530_days": int(len(sp)),
        "parity_spot_1530_median_rel": float(sp.median()),
        "parity_spot_1530_p99_rel": float(sp.quantile(0.99)),
        "close_vs_deck_days": int(len(cl)),
        "close_vs_deck_max_rel": float(cl.max()),
        "pair_vs_s79_days": int(len(jj)),
        "pair_vs_s79_same": int(same_pair.sum()),
        "ask_vs_s79_max_abs_on_same_pair": float(
            (jj.ask - jj.ask_79)[same_pair].abs().max()
        ),
    }
    # ES-consistency of the parity spot across tau
    xs = books[books.venue == "XSP"].pivot(index="day", columns="tau", values="S_book")
    d_es: list[dict] = []
    worst_gap_bp = 0.0
    all_on_time = True
    Rm = pd.DataFrame(
        np.nan_to_num(mm["R"]), index=mm["days"], columns=np.arange(RTH0, RTH1)
    )
    for tau in TAUS:
        if tau == REF_TAU:
            continue
        hh, mn = (int(x) for x in tau.split(":"))
        m = hh * 60 + mn
        lo_, hi_ = sorted((930, m))
        bk_move = np.log(xs[tau] / xs[REF_TAU])
        # timing: the ES window shifted by s minutes; the book is on time iff
        # s = 0 fits best (Pearson is not the criterion: two April 2025 crash
        # days, 62-66 bp apart, carry it at the 5-minute moves -- measured
        # 2026-09-24, Spearman 0.965 / median gap 0.30 bp at 15:35)
        med: dict[int, float] = {}
        for sh in (-2, -1, 0, 1, 2):
            es_s = Rm.loc[:, lo_ + sh : hi_ - 1 + sh].sum(1) * (1 if m > 930 else -1)
            z = pd.concat([es_s, bk_move], axis=1, join="inner").dropna()
            z.columns = ["es", "book"]
            med[sh] = float((z.es - z.book).abs().median() * 1e4)
            if sh == 0:
                z0 = z
        best = min(med, key=lambda k_: med[k_])
        worst_gap_bp = max(worst_gap_bp, med[0])
        all_on_time = all_on_time and best == 0
        d_es.append(
            {
                "tau": tau,
                "days": len(z0),
                "pearson": float(z0.corr().iloc[0, 1]),
                "spearman": float(z0.corr("spearman").iloc[0, 1]),
                "median_abs_diff_bp_by_shift": med,
                "best_shift": best,
            }
        )
    g5["parity_spot_move_vs_es_move"] = d_es
    g5["passed"] = bool(
        float(sp.median()) < 2e-4
        and float(cl.max()) < 1e-4
        and same_pair.mean() > 0.95
        and all_on_time
        and worst_gap_bp < 1.0
    )
    gate["G5_books"] = g5
    log("G5 " + json.dumps(g5, default=str))
    if not g5["passed"]:
        log("G5 FAILED")
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        return 2

    # ---- (b) the card rule at every tau on XSP
    books["close"] = closes.reindex(books["day"]).to_numpy()
    sc = np.where(books["venue"] == "XSP", XSP_SCALE, 1.0)
    books["payoff"] = np.maximum(books["close"] * sc - books["kc"], 0) + np.maximum(
        books["kp"] - books["close"] * sc, 0
    )
    books["R_ask"] = books["payoff"] / books["ask"] - 1
    books["R_mid"] = books["payoff"] / books["mid"] - 1
    books["rel_spread"] = (books["ask"] - books["bid"]) / books["mid"]
    books["month_end"] = [is_last_session_of_month(d.date()) for d in books["day"]]
    xb = books[books.venue == "XSP"]
    pnl = ftab.merge(xb, on=["day", "tau"], how="inner").dropna(
        subset=["f", "ask", "payoff"]
    )
    pnl = pnl[pnl["ask"] > 0]
    pnl["P_star"] = [
        package_price(np.sqrt(f), s, kc, kp)
        for f, s, kc, kp in zip(pnl["f"], pnl["S_book"], pnl["kc"], pnl["kp"])
    ]
    pnl["trade_ask"] = pnl["ask"] <= pnl["P_star"]
    pnl["trade_mid"] = pnl["mid"] <= pnl["P_star"]
    pnl["pnl_ask"] = np.where(pnl["trade_ask"], pnl["R_ask"], 0.0)
    pnl["pnl_mid"] = np.where(pnl["trade_mid"], pnl["R_mid"], 0.0)
    # the same days at every tau and model
    cnt = pnl.groupby("day").size()
    full = cnt.index[cnt == 2 * len(TAUS)]
    pnl = pnl[pnl.day.isin(full)]
    pnl.to_parquet(OUT / "per_day_pnl_tau.parquet", index=False)
    pnl.to_csv(OUT / "per_day_pnl_tau.csv", index=False, float_format="%.6g")
    nm = pnl[~pnl.month_end]
    sb = []
    for (mdl, tau), g in nm.groupby(["model", "tau"]):
        sa = stats(g.loc[g.trade_ask, "R_ask"])
        sm = stats(g.loc[g.trade_mid, "R_mid"])
        sb.append(
            {
                "model": mdl,
                "tau": tau,
                "days": int(len(g)),
                "traded_ask": sa["n"],
                "mean_R_ask": sa["mean_R"],
                "t_ask": sa["t"],
                "hit_ask": sa["hit"],
                "pnl_per_day_ask": float(g.pnl_ask.mean()),
                "traded_mid": sm["n"],
                "mean_R_mid": sm["mean_R"],
                "t_mid": sm["t"],
                "pnl_per_day_mid": float(g.pnl_mid.mean()),
                "uncond_R_ask": float(g.R_ask.mean()),
                "rel_spread_median": float(g.rel_spread.median()),
            }
        )
    sb = pd.DataFrame(sb)
    sb.to_csv(OUT / "summary_tau.csv", index=False)
    log(
        f"\n(b) card rule on XSP, non-month-end, {nm.day.nunique()} sessions "
        f"({nm.day.min().date()} .. {nm.day.max().date()}), same days at every tau:"
    )
    log(sb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    bb = []
    for mdl in ("T0", "T1"):
        base = nm[(nm.model == mdl) & (nm.tau == REF_TAU)].set_index("day").sort_index()
        for tau in TAUS:
            if tau == REF_TAU:
                continue
            o = nm[(nm.model == mdl) & (nm.tau == tau)].set_index("day").loc[base.index]
            for col in ("pnl_ask", "pnl_mid"):
                d = (o[col] - base[col]).to_numpy(float)
                lo, hi, _ = boot_ci(d)
                bb.append(
                    {
                        "model": mdl,
                        "tau_minus_1530": tau,
                        "stat": col,
                        "days": int(len(d)),
                        "point": float(d.mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
    bb = pd.DataFrame(bb)
    bb.to_csv(OUT / "bootstrap_tau.csv", index=False)
    log("\n(b) per-day P&L difference tau - 15:30 (block bootstrap 20 sessions):")
    log(bb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # (a) -> trade: the 15:30 rule with the long-anchored forecasts on XSP
    fa = fc_rows[MIN_TRAIN][["S0_base30", "S4_+fine", "L0r_recal", "L4r_recal+fine"]]
    xa = xb[xb.tau == REF_TAU].set_index("day")
    ta = []
    days_a = fa.index.intersection(xa.index)
    days_a = days_a[[not is_last_session_of_month(d.date()) for d in days_a]]
    for mdl in fa.columns:
        g = xa.loc[days_a].copy()
        g["P_star"] = [
            package_price(np.sqrt(f), s, kc, kp)
            for f, s, kc, kp in zip(fa.loc[days_a, mdl], g.S_book, g.kc, g.kp)
        ]
        tr = g.ask <= g.P_star
        s_ = stats(g.loc[tr, "R_ask"])
        ta.append(
            {
                "model": mdl,
                "days": int(len(g)),
                "traded_ask": s_["n"],
                "mean_R_ask": s_["mean_R"],
                "t_ask": s_["t"],
                "pnl_per_day_ask": float(np.where(tr, g.R_ask, 0).mean()),
            }
        )
    ta = pd.DataFrame(ta)
    ta.to_csv(OUT / "rule_1530_by_model.csv", index=False)
    log("\n(a)->trade: 15:30 XSP rule, non-month-end, by forecast model:")
    log(ta.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # month-ends: the unconditional long by tau, XSP and SPXW
    me_rows = []
    for venue in ("XSP", "SPXW"):
        g = books[(books.venue == venue) & books.month_end].dropna(subset=["R_ask"])
        g = g[g.ask > 0]
        cnt_ = g.groupby("day").size()
        g = g[g.day.isin(cnt_.index[cnt_ == len(TAUS)])]
        wR = g.pivot(index="day", columns="tau", values="R_ask")
        for tau, h in g.groupby("tau"):
            s_ = stats(h["R_ask"])
            dd = (wR[tau] - wR[REF_TAU]).dropna()
            me_rows.append(
                {
                    "venue": venue,
                    "tau": tau,
                    "month_ends": s_["n"],
                    "mean_R_ask": s_["mean_R"],
                    "t_ask": s_["t"],
                    "hit_ask": s_["hit"],
                    "mean_R_mid": float(h["R_mid"].mean()),
                    "rel_spread_median": float(h["rel_spread"].median()),
                    "minus_1530_R_ask": float(dd.mean()),
                    "minus_1530_paired_t": float(
                        dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
                    )
                    if tau != REF_TAU
                    else np.nan,
                }
            )
    me_t = pd.DataFrame(me_rows)
    me_t.to_csv(OUT / "month_end_tau.csv", index=False)
    log(
        "\nmonth-ends: unconditional long by tau (same month-ends at every tau per venue):"
    )
    log(me_t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # (c) what a decisive test would need: costs EXTRAPOLATED from measured pulls
    es_per_day = 3.21 / (pd.Timestamp("2026-09-23") - pd.Timestamp("2024-04-01")).days
    spxw_per_session = 1.02 / 42
    n_pillar = sum(
        is_session(d.date())
        for d in pd.date_range("2023-03-28", "2026-09-22", freq="D")
    )
    est = {
        "ES_1m_per_calendar_day_measured": es_per_day,
        "ES_1m_2023-03-28..2024-03-31_est": es_per_day
        * (pd.Timestamp("2024-03-31") - pd.Timestamp("2023-03-28")).days,
        "ES_1m_2020-01-01..2024-03-31_est": es_per_day
        * (pd.Timestamp("2024-03-31") - pd.Timestamp("2020-01-01")).days,
        "ES_1m_2010-06-06..2024-03-31_est": es_per_day
        * (pd.Timestamp("2024-03-31") - pd.Timestamp("2010-06-06")).days,
        "SPXW_cbbo1m_close_window_per_session_measured": spxw_per_session,
        "OPRA_PILLAR_sessions_2023-03-28..2026-09-22": int(n_pillar),
        "SPXW_cbbo1m_close_window_all_sessions_est": spxw_per_session * n_pillar,
    }
    gate["c_cost_estimates_usd"] = est
    log(
        "\n(c) cost ESTIMATES (USD, linear extrapolation of measured pulls): "
        + json.dumps(est)
    )

    gate["wall_s"] = round(time.time() - T0, 1)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    log(f"\nwall clock {gate['wall_s']} s")
    (OUT / "summary.txt").write_text("\n".join(_LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
