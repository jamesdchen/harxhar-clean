"""Study 89 -- two rescue ideas for the 15:30 card after 2024-04, strictly walk-forward.

THE CARD.  At 15:30 ET buy the SPXW 0DTE nearest-OTM strangle (call at/above the
spot, put at/below, on the listed strikes) iff ask <= P* =
live.ibkr.pricing.package_price(sqrt(rv_hat), S, Kc, Kp); hold to the 16:00
settlement.  Month-ends (bought at any price) and early closes are excluded
here.  rv_hat is the card's per-bar ridge (specs/causal_tune_linear.py through
live/close_signal/arms_shared, bucket free_vix_only, LAG_SCOPE global,
TRAIN_WIN 2000 sessions) on the EXTENDED panel, through the deck's causal
second-order MZ map (notebooks/atm_straddle_lib.load_yhat_1530_mz_cached).

IDEA 1 -- RECENT TRAINING WINDOW.  The same pipeline, the same data files (the
card's own ext root of 2026-09-24, live/close_signal/.scratch/fast_root/data),
with TRAIN_WIN in {250, 500, 1000}; TRAIN_WIN 2000 re-run as the reproduction
gate.  What TRAIN_WIN moves (read off the frozen code): the ridge's rolling
window AND the rolling robust scaler's window (src.backtest.executor passes
train_win_periods = TRAIN_WIN x 1 row/session to rolling_robust_scale).  What it
does not: the spec's tuning constants (TUNE_PER 250, VAL_TAIL 125, EMBARGO 25 --
at TRAIN_WIN 250 the penalty's fit block is 100 rows), the 240-row winsor
window, the HAR ladder, and the MZ map (flat 250-session window from session 63,
atm_straddle_lib.WINDOW_DAYS / MZ_START_DAY -- not tied to TRAIN_WIN, kept).

IDEA 2 -- FORECAST THE TRADE.  Target z_d = log(RV_d / IVask_d): RV_d the
15:30-16:00 realized variance (the panel's sumret2 at the 16:00 stamp = the
card's rv_raw), IVask_d = invert_total_vol(S, Kc, Kp, ask)^2 at the 15:30 ASK of
the traded pair.  Features, all known at 15:30:
  x_card      log(rv_hat_card / IVask)          (card forecast vs the price paid)
  log_ivmid   log(invert_total_vol(S, Kc, Kp, mid)^2)
  log_ask_mid log(ask / mid)
  zbar5/22    mean z of the previous 5 / 22 universe days
  log_vix     log VIX at the 15:30 stamp (the card's panel cell, last print)
  dvix        that minus the previous session's 15:30 cell
  rv_day_rel  log(RV 09:30-15:30 of d / mean of it over the previous 22 sessions)
  dow_0..3    Monday..Thursday dummies (Friday base)
  fomc        FOMC statement day (the vendor's fomc release flag to 2023-11-01,
              live/close_signal/state/fomc_statement_dates.csv after)
  CPI flags are NOT used: the repository's CPI calendar (releases.parquet)
  ends 2024-04-30.
Universe days: SPX book days, non-month-end, non-early-close, finite IVask and
IVmid, RV_d > 0.  Walk-forward, refit every session on the previous N universe
days (N in {250, 500}), features standardized on the training window:
  ridge_N   ridge on all features, penalty by exact leave-one-out (PRESS) on
            the training window over alpha in logspace(-2, 4, 13) (standardized
            units); buy iff zhat > 0
  ols3_N    OLS on the 3 features with the largest |corr(x, z)| in the
            training window (re-selected every session); buy iff zhat > 0
  logit_N   logistic P(R > 0), sklearn default L2 (C = 1); buy iff p > 0.5
A variant is live from the first day with N prior universe days whose
features are complete (zbar22 needs 22 prior z).

DATA FOR THE TRADE.  2020-01 .. 2025-12: data/spxw_chain.parquet (timestamp is
TRUE UTC -> America/New_York), same-day expiration, the 15:30 rows with
ask > 0: S = median underlying_price, nearest-OTM on the listed strikes, leg
bid / ask; the 16:00 close = the median underlying_price of the day's 16:00
rows (the chain's tape; the official ^GSPC close is a reported sensitivity).
2026: data/archive/spxw_opra/cbbo1m_<day>.parquet (study 79's quote readers,
re-implemented here so no working-tree code is imported): the book at 15:30:00,
S = the SPXW put-call parity spot, pair on the strikes quoted with ask > 0; the
close = ^GSPC daily close from studies 79 / 82 / 88's caches (yfinance if
missing, cached here).  R = payoff / ask - 1; per-day P&L = R on a buy day,
0 otherwise (per unit of premium).

GATES (hard gates stop the run before any P&L):
  G1  TRAIN_WIN 2000 re-run == the card's table (live/close_signal/.scratch/
      yhat_close_signal.parquet): same stamps, yhat / baseline max rel <= 1e-9;
      its rv_hat for 2026-09-24 == the card's loader output (2.049719e-06) to
      1e-9 rel; P* on the card's spot (^GSPC 15:29 close) = 6.60 on 7705P/7710C.
  G2  implied-variance inversion: invert_total_vol(S, Kc, Kp, mid)^2 at the
      deck's own legs vs the deck's iv_var, median |log ratio| < 1e-3 on the
      deck days; this study's chain book == the deck's pair / ask / S.
  G3  OPRA leg: on 2025 days both sources hold, the OPRA 15:30 ask at the
      chain's pair == the chain's ask (half a cent), parity-spot pair == chain
      pair; the ^GSPC caches agree with each other.
  G4  F0 from the card's table == F0 from the gated re-run (rv_hat 1e-9).

REPORT per variant (F0 = TRAIN_WIN 2000, ask <= P*): live days, buys, share,
mean R on buy days (t), per-day P&L (t), by calendar year; the paired circular
20-day-block bootstrap (10,000 draws) of per-day P&L variant - F0 on the
variant's live days, separately 2020-01..2024-04 and 2024-05..2026-09, and the
variant's own per-day P&L CI there; a table on the days common to every
variant.  Ten models are scored (F0 + 3 + 6): no multiplicity correction is
applied, so read single t's accordingly.

OUTPUTS (results/atm_straddle_intraday_holdclose/proposals/89/): gate.json,
book.parquet, gate_opra_2025.csv, forecasts_card_table.parquet,
forecasts_tw<N>.parquet, features.parquet, ratio_predictions.parquet,
decisions.parquet (per day, every variant), summary.csv (incl. the bootstrap
CIs), by_year.csv, common_days.csv, forecast_diagnostics.csv, summary.txt,
run_<phase>.log.

RESULT (2026-09-24 run; F0 = +0.0635/day, t 2.27 on 814 research days;
-0.0109/day, t -0.34 on 568 days 2024-05..2026-09): neither idea restores an
edge after 2024-04.  Shorter windows forecast worse in both periods (post
QLIKE 0.1847 at 2000 vs 0.1887 / 0.1889 / 0.1934 at 1000 / 500 / 250; the ask's
implied variance 0.1423) and lose post (-0.0181 / -0.0250 / -0.0124 per day);
500 and 250 cost 0.038 / 0.033 per day in the research years (CIs exclude 0).
The trade-aligned models buy 2-18 % of days and earn +0.001..+0.003 per day
post (all |t| < 0.25, CIs straddle 0): they avoid the card's small loss by not
trading, not by finding an edge, and cost 0.04..0.10 per day vs F0 in the
research years on their live days.  See summary.txt.

RUN (phases are sequential, one engine process at a time, <= 3 workers):
    PY=C:/Users/james/miniconda3/envs/285J/python.exe
    $PY writeup/intraday_proposals/89_recent_and_ratio_models.py --phase book
    $PY writeup/intraday_proposals/89_recent_and_ratio_models.py --phase f0
    $PY writeup/intraday_proposals/89_recent_and_ratio_models.py --phase engine --tw 2000
    $PY ... --phase engine --tw 1000 ; --tw 500 ; --tw 250
    $PY ... --phase ratio
    $PY ... --phase report
  (--work DIR holds the frozen code, the engine root and arm CSVs; default
   $TMP/p89_work.  --workers 2.)
"""

from __future__ import annotations

from typing import Any

import argparse
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROPOSALS = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals"
OUT = PROPOSALS / "89"
CHAIN = REPO / "data" / "spxw_chain.parquet"
CHAIN_SPOT = REPO / "data" / "spxw_spot.parquet"
SPXW_OPRA = REPO / "data" / "archive" / "spxw_opra"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
CARD_SCRATCH = REPO / "live" / "close_signal" / ".scratch"
CARD_DATA = CARD_SCRATCH / "fast_root" / "data"
CARD_TABLE = CARD_SCRATCH / "yhat_close_signal.parquet"
CARD_MZ_GLOB = "yhat1530mz_close_signal_mean_*.parquet"
CARD_GSPC_1M = PROPOSALS / "88" / "gspc_1m_2026-09-24.csv"
YF_CACHES = (
    PROPOSALS / "79" / "spx_close_yf.csv",
    PROPOSALS / "82" / "spx_close_yf.csv",
    PROPOSALS / "88" / "spx_close_yf.csv",
)
FROZEN_PATHS = ("live", "src", "specs", "notebooks/atm_straddle_lib.py")
VENDOR_FILES = (
    "core_stats.parquet",
    "vix_and_voldemand.parquet",
    "releases.parquet",
    "time_categories.parquet",
    "extension.json",
)

ET = "America/New_York"
BUCKET = "free_vix_only"
ESTIMATOR = "ridge"
TW_CARD = 2000
TW_ALL = (2000, 1000, 500, 250)
BARS = tuple(
    f"bar{h:02d}{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
)
CARD_DAY = pd.Timestamp("2026-09-24")
CARD_PAIR = (7710.0, 7705.0)  # (Kc, Kp) on the card of 2026-09-24
CARD_PSTAR = 6.60  # as printed (2 decimals)
FIRST_BOOK = pd.Timestamp("2020-01-01")
LAST_CHAIN_YEAR = 2025
RESEARCH = (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-04-30"))
POST = (pd.Timestamp("2024-05-01"), pd.Timestamp("2026-09-30"))
N_WF = (250, 500)
ALPHAS = np.logspace(-2, 4, 13)
FEATS = (
    "x_card",
    "log_ivmid",
    "log_ask_mid",
    "zbar5",
    "zbar22",
    "log_vix",
    "dvix",
    "rv_day_rel",
    "dow_0",
    "dow_1",
    "dow_2",
    "dow_3",
    "fomc",
)
RV_DAY_LOOKBACK = 22
BLOCK = 20
N_BOOT = 10_000
SEED = 89
GATE_REL = 1e-9
IV_GATE = 1e-3
HALF_CENT = 0.005

_LOG: list[str] = []


def log(msg: str = "") -> None:
    _LOG.append(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------- frozen code
def freeze(work: Path) -> Path:
    """git archive HEAD of the live/src/specs code (+ the deck loader) under work/frozen."""
    dst = work / "frozen"
    sha = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    tag = dst / "FROZEN_SHA"
    if not (tag.exists() and tag.read_text().strip() == sha):
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        tar = subprocess.run(
            ["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", *FROZEN_PATHS],
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            tf.extractall(dst, filter="data")
        tag.write_text(sha)
    if str(dst) not in sys.path:
        sys.path.insert(0, str(dst))
    return dst


def frozen_modules(frozen: Path):
    from live.close_signal import arms_shared as A
    from live.close_signal import forecast as F
    from live.ibkr import calendar_guard as CG
    from live.ibkr import pricing as PR

    for m in (A, F, CG, PR):
        assert Path(str(m.__file__)).resolve().is_relative_to(frozen.resolve()), (
            m.__file__
        )
    assert F.BUCKET == BUCKET and F.TRAIN_WIN == TW_CARD and F.ESTIMATOR == ESTIMATOR
    return A, F, CG, PR


def load_gate() -> dict:
    p = OUT / "gate.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_gate(g: dict) -> None:
    (OUT / "gate.json").write_text(json.dumps(g, indent=1, default=str))


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- quotes
def nearest_otm(strikes: np.ndarray, spot: float) -> tuple[float, float]:
    """(Kc, Kp): the listed call strike at/above the spot, the put strike at/below (study 79)."""
    above = strikes[strikes >= spot]
    below = strikes[strikes <= spot]
    if len(above) == 0 or len(below) == 0:
        return float("nan"), float("nan")
    return float(above.min()), float(below.max())


def load_quotes(path: Path) -> pd.DataFrame:
    """A Databento cbbo-1m file with ET stamps, strike and right (study 79's load_quotes)."""
    q = pd.read_parquet(
        path,
        columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"],
    )
    q["t"] = (
        pd.to_datetime(q["ts_recv"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    )
    sym = q["symbol"].astype(str)
    q["strike"] = sym.str.slice(13).astype(float) / 1000.0
    q["cp"] = sym.str.slice(12, 13)
    return q.drop(columns=["ts_recv", "symbol"])


def book_at(q: pd.DataFrame, day: pd.Timestamp, hhmmss: str) -> pd.DataFrame:
    """The last sample at or before the stamp (study 79's book_at)."""
    t = pd.Timestamp(f"{day.date()} {hhmmss}")
    b = q[q["t"] == t]
    if b.empty:
        b = q[q["t"] <= t]
        b = b[b["t"] == b["t"].max()]
    return b.set_index(["strike", "cp"]).sort_index()


def opra_quote(b: pd.DataFrame, k: float, cp: str) -> tuple[float, float]:
    """(bid, ask) of one contract; NaN when absent or not quoted (study 79's quote)."""
    try:
        r = b.loc[(k, cp)]
    except KeyError:
        return float("nan"), float("nan")
    if isinstance(r, pd.DataFrame):
        r = r.iloc[-1]
    bid, ask = float(r["bid_px_00"]), float(r["ask_px_00"])
    if not np.isfinite(ask) or ask <= 0:
        return float("nan"), float("nan")
    return (bid if np.isfinite(bid) else 0.0), ask


def parity_spot(b: pd.DataFrame) -> float:
    """K + C_mid - P_mid at the 3 strikes nearest C = P (study 79's parity_spot)."""
    m = (b["bid_px_00"] + b["ask_px_00"]) / 2
    ok = (b["ask_px_00"] > 0) & np.isfinite(m)
    m = m[ok].unstack("cp")
    if not {"C", "P"} <= set(m.columns):
        return float("nan")
    d = (m["C"] - m["P"]).dropna()
    if d.empty:
        return float("nan")
    near = d.abs().nsmallest(3).index
    return float(np.median(near.to_numpy() + d.loc[near].to_numpy()))


def chain_rows(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Same-day-expiry chain rows at 15:30 and 16:00 ET for expirations in [lo, hi].

    TRUE UTC stamps: pre-filtered in arrow to the four UTC clocks 15:30 / 16:00 ET
    can be (EDT / EST), then converted and kept at 15:30 / 16:00 ET exactly.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(
        CHAIN,
        columns=[
            "expiration",
            "strike",
            "cp",
            "timestamp",
            "bid",
            "ask",
            "underlying_price",
            "hours_to_expiration",
        ],
        filters=[("expiration", ">=", lo), ("expiration", "<=", hi)],
    )
    ms = t.column("timestamp").cast(pa.int64()).to_numpy()
    mod = (ms // 60000) % 1440
    keep = np.isin(mod, [19 * 60 + 30, 20 * 60 + 30, 20 * 60, 21 * 60])
    t = t.filter(pa.array(keep))
    c = t.to_pandas()
    del t, ms, mod, keep
    et = pd.to_datetime(c["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None)
    c["clk"] = et.dt.strftime("%H:%M")
    c["day"] = et.dt.normalize()
    c = c[
        (c["day"] == pd.to_datetime(c["expiration"]))
        & c["clk"].isin(["15:30", "16:00"])
    ]
    return c.drop(columns=["timestamp"])


def chain_book(c: pd.DataFrame) -> pd.DataFrame:
    """The chain's 15:30 nearest-OTM pair per day (study 79's spxw_book + bids, hte, the 16:00 tape)."""
    rows = []
    c16 = c[c["clk"] == "16:00"].groupby("day")["underlying_price"].median()
    for day, g in c[c["clk"] == "15:30"].groupby("day"):
        hte = float(g["hours_to_expiration"].median())
        ga = g[g["ask"] > 0]
        if ga.empty:
            continue
        s = float(ga["underlying_price"].median())
        gi = ga.set_index(["strike", "cp"]).sort_index()
        ks = np.sort(gi.index.get_level_values(0).unique().to_numpy(dtype=float))
        kc, kp = nearest_otm(ks, s)
        try:
            rc = gi.loc[(kc, "C")]
            rp = gi.loc[(kp, "P")]
        except KeyError:
            continue
        if isinstance(rc, pd.DataFrame):
            rc = rc.iloc[-1]
        if isinstance(rp, pd.DataFrame):
            rp = rp.iloc[-1]
        rows.append(
            {
                "day": pd.Timestamp(day),
                "src": "chain",
                "S": s,
                "kc": kc,
                "kp": kp,
                "bid_c": float(rc["bid"]),
                "ask_c": float(rc["ask"]),
                "bid_p": float(rp["bid"]),
                "ask_p": float(rp["ask"]),
                "hte_1530": hte,
                "S_close_tape": float(c16.get(day, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def yf_closes(need: list[pd.Timestamp]) -> tuple[pd.Series, dict]:
    """^GSPC daily closes: studies 79 / 82 / 88's caches (first non-NaN wins), yfinance for the rest."""
    parts = []
    for p in YF_CACHES:
        if p.exists():
            s = pd.read_csv(p, index_col=0, parse_dates=True)["close"].astype(float)
            s.index = pd.DatetimeIndex(s.index).normalize()
            parts.append(s[~s.index.duplicated(keep="last")].rename(p.parent.name))
    own = OUT / "spx_close_yf.csv"
    if own.exists():
        s = pd.read_csv(own, index_col=0, parse_dates=True)["close"].astype(float)
        parts.append(s.rename("89"))
    j = pd.concat(parts, axis=1)
    agree: dict[str, Any] = {}
    for a_ in j.columns:
        for b_ in j.columns:
            if a_ < b_:
                d = (j[a_] - j[b_]).abs().dropna()
                agree[f"{a_}_vs_{b_}"] = {
                    "days": int(len(d)),
                    "max_abs": float(d.max()) if len(d) else None,
                }
    have = j.bfill(axis=1).iloc[:, 0].dropna()
    miss = [d for d in need if d not in have.index]
    if miss:
        import yfinance as yf

        h = yf.Ticker("^GSPC").history(
            start=min(miss).strftime("%Y-%m-%d"),
            end=(max(miss) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        if len(h) == 0:  # Yahoo has no row (2026-09-22 is missing from ^GSPC daily)
            agree["no_official_close"] = [str(d.date()) for d in miss]
            return have, agree
        got = pd.Series(
            h["Close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(h.index.tz_localize(None).normalize()),
        )
        got = got[got.index.isin(miss)]
        agree["no_official_close"] = [str(d.date()) for d in miss if d not in got.index]
        if len(got):
            got.rename("close").to_frame().to_csv(own)
        have = pd.concat([have, got]).sort_index()
        have = have[~have.index.duplicated(keep="first")]
    return have, agree


def opra_day(
    path: Path, day: pd.Timestamp, chain_pair: tuple[float, float] | None
) -> dict:
    q = load_quotes(path)
    b = book_at(q, day, "15:30:00")
    del q
    stamp = b["t"].max() if len(b) else pd.NaT
    s = parity_spot(b) if len(b) else float("nan")
    quoted = b[b["ask_px_00"] > 0]
    ks = np.sort(quoted.index.get_level_values(0).unique().to_numpy(dtype=float))
    kc, kp = nearest_otm(ks, s) if np.isfinite(s) else (float("nan"), float("nan"))
    bc, ac = opra_quote(b, kc, "C")
    bp, ap = opra_quote(b, kp, "P")
    out = {
        "day": day,
        "src": "opra",
        "stamp_ok": bool(stamp == pd.Timestamp(f"{day.date()} 15:30:00")),
        "S": s,
        "kc": kc,
        "kp": kp,
        "bid_c": bc,
        "ask_c": ac,
        "bid_p": bp,
        "ask_p": ap,
    }
    if chain_pair is not None:
        out["ask_c_at_chain_pair"] = opra_quote(b, chain_pair[0], "C")[1]
        out["ask_p_at_chain_pair"] = opra_quote(b, chain_pair[1], "P")[1]
    return out


def phase_book(work: Path) -> int:
    frozen = freeze(work)
    _, _, CG, PR = frozen_modules(frozen)
    sys.path.insert(0, str(frozen / "notebooks"))
    import atm_straddle_lib as asl  # type: ignore

    assert Path(asl.__file__).resolve().is_relative_to(frozen.resolve())
    gate = load_gate()
    parts = []
    for y in range(FIRST_BOOK.year, LAST_CHAIN_YEAR + 1):
        t0 = time.time()
        c = chain_rows(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))
        bk = chain_book(c)
        del c
        gc.collect()
        parts.append(bk)
        log(f"chain {y}: {len(bk)} days ({time.time() - t0:.0f}s)")
    chain = pd.concat(parts, ignore_index=True)
    del parts
    # the chain's 16:00 tape vs the spot file's 16:00 (studies 79 / 85's close)
    sp = pd.read_parquet(CHAIN_SPOT)
    t = pd.to_datetime(sp["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None)
    sp16 = sp[t.dt.strftime("%H:%M") == "16:00"].assign(day=t.dt.normalize())
    sp16 = sp16.set_index("day")["spot"].astype(float)
    dd = (chain.set_index("day")["S_close_tape"] - sp16).abs().dropna()
    gate["book_close_tape_vs_spotfile_16"] = {
        "days": int(len(dd)),
        "max_abs": float(dd.max()),
    }
    # 2026 from the SPXW OPRA minute files
    files = sorted(
        p
        for p in SPXW_OPRA.glob("cbbo1m_*.parquet")
        if not p.name.startswith("cbbo1m-")
    )
    by_day = {pd.Timestamp(p.stem.split("_")[1]): p for p in files}
    chain_i = chain.set_index("day")
    g3 = []
    for d, p in by_day.items():
        if d.year == 2025 and d in chain_i.index:
            r = chain_i.loc[d]
            o = opra_day(p, d, (float(r["kc"]), float(r["kp"])))
            g3.append(
                {
                    "day": d,
                    "stamp_ok": o["stamp_ok"],
                    "chain_S": float(r["S"]),
                    "parity_S": o["S"],
                    "chain_pair": f"{r['kc']:.0f}/{r['kp']:.0f}",
                    "opra_pair": f"{o['kc']:.0f}/{o['kp']:.0f}",
                    "chain_ask": float(r["ask_c"] + r["ask_p"]),
                    "opra_ask_at_chain_pair": o["ask_c_at_chain_pair"]
                    + o["ask_p_at_chain_pair"],
                    "opra_ask_own_pair": o["ask_c"] + o["ask_p"],
                }
            )
    g3 = pd.DataFrame(g3)
    g3["ask_match"] = (
        g3["opra_ask_at_chain_pair"] - g3["chain_ask"]
    ).abs() <= HALF_CENT
    g3["pair_match"] = g3["chain_pair"] == g3["opra_pair"]
    g3.to_csv(OUT / "gate_opra_2025.csv", index=False)
    gate["G3_opra_2025"] = {
        "days": int(len(g3)),
        "stamp_ok": int(g3["stamp_ok"].sum()),
        "ask_match_half_cent": float(g3["ask_match"].mean()),
        "pair_match": float(g3["pair_match"].mean()),
        "median_abs_spot_diff": float((g3["parity_S"] - g3["chain_S"]).abs().median()),
        "own_pair_ask_minus_chain_ask_median": float(
            (g3["opra_ask_own_pair"] - g3["chain_ask"]).median()
        ),
    }
    log(f"G3 OPRA vs chain on 2025: {json.dumps(gate['G3_opra_2025'])}")
    opra = [opra_day(p, d, None) for d, p in by_day.items() if d.year > LAST_CHAIN_YEAR]
    opra = pd.DataFrame(opra)
    n_bad = int((~opra["stamp_ok"]).sum())
    opra = opra[opra["stamp_ok"]].drop(columns=["stamp_ok"])
    log(f"OPRA 2026: {len(opra)} days with a 15:30:00 sample ({n_bad} without)")
    book = pd.concat([chain, opra], ignore_index=True).sort_values("day")
    book = book.reset_index(drop=True)
    closes, agree = yf_closes(list(book["day"]))
    gate["G3_yf_cache_agreement"] = agree
    book["S_close_official"] = book["day"].map(closes).astype(float)
    book["S_close"] = np.where(
        book["src"] == "chain", book["S_close_tape"], book["S_close_official"]
    )
    book["close_src"] = np.where(book["src"] == "chain", "chain_16:00", "gspc_close")
    book["bid"] = book["bid_c"] + book["bid_p"]
    book["ask"] = book["ask_c"] + book["ask_p"]
    book["mid"] = 0.5 * (book["bid_c"] + book["ask_c"]) + 0.5 * (
        book["bid_p"] + book["ask_p"]
    )
    for col, px in (("iv_var_ask", "ask"), ("iv_var_mid", "mid")):
        v = [
            PR.invert_total_vol(s, kc, kp, x)
            for s, kc, kp, x in zip(book["S"], book["kc"], book["kp"], book[px])
        ]
        book[col] = np.asarray(v, float) ** 2
    for col, close in (("R", "S_close"), ("R_official", "S_close_official")):
        pay = np.maximum(book[close] - book["kc"], 0.0) + np.maximum(
            book["kp"] - book[close], 0.0
        )
        book[col.replace("R", "payoff")] = pay
        book[col] = pay / book["ask"] - 1.0
    early = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    book["early_close"] = book["day"].isin(early) | (
        book["hte_1530"].notna() & (book["hte_1530"] <= 0)
    )
    book["month_end"] = [CG.is_last_session_of_month(d.date()) for d in book["day"]]
    book["session"] = [CG.is_session(d.date()) for d in book["day"]]
    book.to_parquet(OUT / "book.parquet", index=False)
    log(
        f"book: {len(book)} days {book['day'].min().date()} .. {book['day'].max().date()} "
        f"(chain {int((book['src'] == 'chain').sum())}, OPRA {int((book['src'] == 'opra').sum())}); "
        f"month-ends {int(book['month_end'].sum())}, early closes {int(book['early_close'].sum())}, "
        f"no close {int(book['S_close'].isna().sum())}, IVask NaN {int(book['iv_var_ask'].isna().sum())}"
    )

    # G2: the inversion at the deck's own legs vs its iv_var; this book vs the deck
    deck = pd.read_parquet(DECK).sort_index()
    deck.index = pd.DatetimeIndex(deck.index).normalize()
    inv = np.array(
        [
            PR.invert_total_vol(s, kc, kp, mc + mp)
            for s, kc, kp, mc, mp in zip(
                deck["S"], deck["K_c"], deck["K_p"], deck["mid_c"], deck["mid_p"]
            )
        ]
    )
    lr = np.log(inv**2 / deck["iv_var"].to_numpy(float))
    ok = np.isfinite(lr)
    bi = book.set_index("day")
    bk = bi[["S", "kc", "kp", "mid", "iv_var_mid"]].add_prefix("b_")
    j = deck[["S", "K_c", "K_p", "entry", "iv_var"]].join(bk, how="inner")
    same = (j["K_c"] == j["b_kc"]) & (j["K_p"] == j["b_kp"])
    lr_b = np.log(j["b_iv_var_mid"] / j["iv_var"])
    gate["G2_iv_inversion"] = {
        "deck_days": int(len(deck)),
        "finite": int(ok.sum()),
        "median_abs_log_ratio_deck_legs": float(np.median(np.abs(lr[ok]))),
        "p90_abs_log_ratio_deck_legs": float(np.quantile(np.abs(lr[ok]), 0.9)),
        "book_deck_common_days": int(len(j)),
        "book_pair_equals_deck_pair": int(same.sum()),
        # the deck's entry is the package MID (mid_c + mid_p)
        "book_mid_equals_deck_entry_half_cent": int(
            ((j["b_mid"] - j["entry"]).abs() <= HALF_CENT).sum()
        ),
        "book_S_max_abs_diff": float((j["b_S"] - j["S"]).abs().max()),
        "median_abs_log_ratio_book_mid": float(np.nanmedian(np.abs(lr_b[same]))),
    }
    gate["G2_passed"] = bool(
        gate["G2_iv_inversion"]["median_abs_log_ratio_deck_legs"] < IV_GATE
    )
    log(f"G2 inversion vs deck iv_var: {json.dumps(gate['G2_iv_inversion'])}")
    save_gate(gate)
    return 0


# --------------------------------------------------------------------------- forecasts
def card_rv_hat() -> float:
    cands = sorted((CARD_SCRATCH / "mz_cache").glob(CARD_MZ_GLOB))
    assert len(cands) == 1, cands
    m = pd.read_parquet(cands[0])
    m.index = pd.DatetimeIndex(pd.to_datetime(m.index)).normalize()
    return float(m.loc[CARD_DAY, "rv_hat"])


def card_spot() -> float:
    g = pd.read_csv(CARD_GSPC_1M)
    t = pd.to_datetime(g["Datetime"].str.slice(0, 19))
    return float(
        g.loc[t == pd.Timestamp(f"{CARD_DAY.date()} 15:29:00"), "Close"].iloc[0]
    )


def forecast_days(table: pd.DataFrame) -> list[pd.Timestamp]:
    et = pd.DatetimeIndex(table["t"]).tz_convert(ET)
    at16 = (et.hour == 16) & (et.minute == 0)
    days = et[at16].tz_localize(None).normalize().unique()
    return [d for d in days if d >= FIRST_BOOK - pd.Timedelta(days=10)]


def recal(F, frozen: Path, table: pd.DataFrame, work: Path, tag: str) -> pd.DataFrame:
    px = F.recalibrate(table, forecast_days(table), frozen, work / f"mz_{tag}", tag=tag)
    px = px[["yhat", "baseline", "rv_raw", "rv_hat", "m", "s2"]].copy()
    px.index.name = "day"
    return px


def phase_f0(work: Path) -> int:
    frozen = freeze(work)
    _, F, _, _ = frozen_modules(frozen)
    gate = load_gate()
    tab = pd.read_parquet(CARD_TABLE)
    px = recal(F, frozen, tab, work, "p89_card")
    px.reset_index().to_parquet(OUT / "forecasts_card_table.parquet", index=False)
    rv0 = card_rv_hat()
    gate["F0_card_table"] = {
        "rows": int(len(tab)),
        "first": str(tab["t"].min()),
        "last": str(tab["t"].max()),
        "sha1": sha1(CARD_TABLE),
        "days_with_rv_hat": int(px["rv_hat"].notna().sum()),
        "rv_hat_card_day": float(px.loc[CARD_DAY, "rv_hat"]),
        "rv_hat_card_loader": rv0,
        "rel_dev": float(abs(px.loc[CARD_DAY, "rv_hat"] - rv0) / rv0),
    }
    log(f"F0 from the card's table: {json.dumps(gate['F0_card_table'])}")
    save_gate(gate)
    return 0


def engine_root(work: Path, frozen: Path) -> Path:
    root = work / "root"
    if not (root / "data" / "core_stats.parquet").exists():
        for d in ("src", "specs"):
            if (root / d).exists():
                shutil.rmtree(root / d)
            shutil.copytree(frozen / d, root / d)
        (root / "data").mkdir(parents=True, exist_ok=True)
        for f in VENDOR_FILES:
            shutil.copy2(CARD_DATA / f, root / "data" / f)
    return root


def phase_engine(work: Path, tw: int, workers: int) -> int:
    frozen = freeze(work)
    A, F, _, PR = frozen_modules(frozen)
    gate = load_gate()
    root = engine_root(work, frozen)
    gate.setdefault(
        "engine_data_sha1", {f: sha1(root / "data" / f) for f in VENDOR_FILES}
    )
    gate["engine_data_equals_card_data"] = all(
        sha1(root / "data" / f) == sha1(CARD_DATA / f) for f in VENDOR_FILES
    )
    res = work / f"arms_tw{tw}"
    want = res / "causal_tune_linear" / ESTIMATOR / BUCKET
    csvs = {b: want / f"results_{b}.csv" for b in BARS}
    if not all(p.exists() for p in csvs.values()):
        if res.exists():
            shutil.rmtree(res)
        env = A.arm_env(BUCKET, ESTIMATOR, tw, res)
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(frozen), env["PYTHONPATH"]
        t0 = time.time()
        with open(work / f"arms_tw{tw}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "live.close_signal.arms_shared",
                    "--root",
                    str(root),
                    "--workers",
                    str(workers),
                    "--out-json",
                    str(work / f"arms_tw{tw}.json"),
                ],
                cwd=str(root),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            ).returncode
        log(f"engine TRAIN_WIN {tw}: exit {rc} in {time.time() - t0:.0f}s")
        if rc != 0:
            return rc
    tab = F.assemble_yhat_table(csvs, root)
    tab.to_parquet(work / f"yhat_tw{tw}.parquet", index=False)
    px = recal(F, frozen, tab, work, f"p89_tw{tw}")
    px.reset_index().to_parquet(OUT / f"forecasts_tw{tw}.parquet", index=False)
    info: dict[str, Any] = {
        "table_rows": int(len(tab)),
        "first_stamp": str(tab["t"].min()),
        "first_1600_day_with_rv_hat": str(px["rv_hat"].dropna().index.min().date()),
        "rv_hat_card_day": float(px.loc[CARD_DAY, "rv_hat"])
        if CARD_DAY in px.index
        else None,
    }
    if tw == TW_CARD:
        card = pd.read_parquet(CARD_TABLE)
        dev = F.table_deviation(tab, card)
        rv0 = card_rv_hat()
        rv1 = float(px.loc[CARD_DAY, "rv_hat"])
        spot = card_spot()
        kc, kp = nearest_otm(np.arange(7000.0, 8500.0, 5.0), spot)
        pstar = PR.package_price(math.sqrt(rv1), spot, kc, kp)
        f0 = pd.read_parquet(OUT / "forecasts_card_table.parquet").set_index("day")
        jj = f0[["rv_hat"]].join(px[["rv_hat"]], rsuffix="_run", how="outer")
        rel = (jj["rv_hat"] - jj["rv_hat_run"]).abs() / jj["rv_hat"].abs()
        info.update(
            {
                "vs_card_table": dev,
                "rv_raw_max_rel": float(
                    (
                        (
                            tab.set_index("t")["rv_raw"] - card.set_index("t")["rv_raw"]
                        ).abs()
                        / card.set_index("t")["rv_raw"].abs()
                    ).max()
                ),
                "rv_hat_card_loader": rv0,
                "rv_hat_rel_dev": abs(rv1 - rv0) / rv0,
                "card_spot_gspc_1529": spot,
                "pair": [kc, kp],
                "P_star": pstar,
                "P_star_2dp": round(pstar, 2),
                "G4_rv_hat_vs_card_table_days": int(rel.notna().sum()),
                "G4_rv_hat_vs_card_table_unmatched": int(rel.isna().sum()),
                "G4_rv_hat_vs_card_table_max_rel": float(rel.max()),
            }
        )
        gate["G1_passed"] = bool(
            dev["rows_only_a"] == 0
            and dev["rows_only_b"] == 0
            and dev["max_rel_yhat"] <= GATE_REL
            and dev["max_rel_baseline"] <= GATE_REL
            and info["rv_hat_rel_dev"] <= GATE_REL
            and (kc, kp) == CARD_PAIR
            and round(pstar, 2) == CARD_PSTAR
        )
        gate["G4_passed"] = bool(
            info["G4_rv_hat_vs_card_table_unmatched"] == 0
            and info["G4_rv_hat_vs_card_table_max_rel"] <= GATE_REL
        )
    gate[f"engine_tw{tw}"] = info
    log(f"engine TRAIN_WIN {tw}: {json.dumps(info, default=str)}")
    save_gate(gate)
    return 0


# --------------------------------------------------------------------------- idea 2
def panel_features(days: pd.DatetimeIndex, fomc_csv: Path) -> pd.DataFrame:
    """RV_d, the day's 09:30-15:30 RV vs its trailing mean, VIX at 15:30 and its change, FOMC."""
    core = pd.read_parquet(
        CARD_DATA / "core_stats.parquet", columns=["endbartime", "sumret2"]
    )
    core["endbartime"] = pd.to_datetime(core["endbartime"])
    core = core[core["endbartime"] >= FIRST_BOOK - pd.Timedelta(days=90)]
    core = core.set_index("endbartime")["sumret2"].astype(float)
    t = core.index
    mins = t.hour * 60 + t.minute
    rv16 = core[mins == 16 * 60]
    rv16.index = rv16.index.normalize()
    sess = rv16.dropna().index  # sessions: days with a 16:00 row
    rth = core[(mins >= 10 * 60) & (mins <= 15 * 60 + 30)]
    rv_day = rth.groupby(rth.index.normalize()).sum(min_count=1).reindex(sess)
    trail = rv_day.shift(1).rolling(RV_DAY_LOOKBACK, min_periods=RV_DAY_LOOKBACK).mean()
    vx = pd.read_parquet(
        CARD_DATA / "vix_and_voldemand.parquet", columns=["endbartime", "vix"]
    )
    vx["endbartime"] = pd.to_datetime(vx["endbartime"])
    vx = vx[vx["endbartime"] >= FIRST_BOOK - pd.Timedelta(days=90)]
    vx = vx.set_index("endbartime")["vix"].astype(float).ffill()  # the last print
    v1530 = vx.reindex(sess + pd.Timedelta("15:30:00"))
    v1530.index = sess
    rel = pd.read_parquet(
        CARD_DATA / "releases.parquet", columns=["endbartime", "fomc release"]
    )
    rel["endbartime"] = pd.to_datetime(rel["endbartime"])
    fomc_v = set(rel.loc[rel["fomc release"] == 1, "endbartime"].dt.normalize())
    fd = pd.read_csv(fomc_csv, comment="#")[
        "date"
    ]  # forecast.fomc_release_rows reads it so
    fomc_c = set(pd.to_datetime(fd.astype(str).str.strip()))
    f = pd.DataFrame(index=sess)
    f["RV_d"] = rv16.reindex(sess)
    f["rv_day"] = rv_day
    f["rv_day_rel"] = np.log(rv_day / trail)
    f["vix_1530"] = v1530
    f["log_vix"] = np.log(v1530)
    f["dvix"] = v1530 - v1530.shift(1)
    f["fomc"] = [float(d in fomc_v or d in fomc_c) for d in sess]
    return f.reindex(days)


def ridge_loo_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ridge (standardized X, free intercept) with alpha by exact leave-one-out over ALPHAS."""
    n = len(y)
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uy = U.T @ yc
    best: tuple[float, float] | None = None
    for a in ALPHAS:
        f = s**2 / (s**2 + a)
        fit = U @ (f * Uy)
        h = (U**2) @ f + 1.0 / n
        loo = (yc - fit) / (1.0 - h)
        mse = float(np.mean(loo**2))
        if best is None or mse < best[0]:
            best = (mse, a)
    assert best is not None
    a = best[1]
    beta = Vt.T @ ((s / (s**2 + a)) * Uy)
    return beta, float(ym - xm @ beta), float(a)


def walk_forward(ff: pd.DataFrame, n_win: int) -> pd.DataFrame:
    """ridge / ols3 / logit predictions for every universe day with n_win complete prior days."""
    from sklearn.linear_model import LogisticRegression

    X_all = ff[list(FEATS)].to_numpy(float)
    z_all = ff["z"].to_numpy(float)
    win_all = (ff["R"].to_numpy(float) > 0).astype(float)
    complete = np.isfinite(X_all).all(1) & np.isfinite(z_all)
    out = []
    for i in range(len(ff)):
        if not np.isfinite(X_all[i]).all():
            continue
        prev = np.flatnonzero(complete[:i])
        if len(prev) < n_win:
            continue
        tr = prev[-n_win:]
        X, z, w = X_all[tr], z_all[tr], win_all[tr]
        mu, sd = X.mean(0), X.std(0)
        live = sd > 0
        Xs = (X[:, live] - mu[live]) / sd[live]
        xs = (X_all[i, live] - mu[live]) / sd[live]
        beta, b0, alpha = ridge_loo_fit(Xs, z)
        zr = float(b0 + xs @ beta)
        # OLS on the 3 features most correlated with z in the window
        cz = np.abs(
            np.array([np.corrcoef(Xs[:, j], z)[0, 1] for j in range(Xs.shape[1])])
        )
        top = np.argsort(-np.nan_to_num(cz, nan=-1.0))[:3]
        A3 = np.column_stack([np.ones(n_win), Xs[:, top]])
        b3, *_ = np.linalg.lstsq(A3, z, rcond=None)
        zo = float(b3[0] + xs[top] @ b3[1:])
        if 0 < w.sum() < n_win:
            lg = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, w)
            p = float(lg.predict_proba(xs[None, :])[0, 1])
        else:
            p = float(w.mean())
        names = np.array(FEATS)[live]
        out.append(
            {
                "day": ff.index[i],
                f"zhat_ridge_{n_win}": zr,
                f"alpha_ridge_{n_win}": alpha,
                f"zhat_ols3_{n_win}": zo,
                f"ols3_feats_{n_win}": "|".join(names[top]),
                f"p_logit_{n_win}": p,
            }
        )
    return pd.DataFrame(out).set_index("day")


def universe(book: pd.DataFrame) -> pd.DataFrame:
    b = book[book["session"] & ~book["month_end"] & ~book["early_close"]].copy()
    b = b[np.isfinite(b["ask"]) & (b["ask"] > 0) & np.isfinite(b["S_close"])]
    return b.set_index("day").sort_index()


def phase_ratio(work: Path) -> int:
    frozen = freeze(work)
    gate = load_gate()
    book = pd.read_parquet(OUT / "book.parquet")
    f0 = pd.read_parquet(OUT / "forecasts_tw2000.parquet").set_index("day")
    u = universe(book)
    pf = panel_features(
        u.index, frozen / "live" / "close_signal" / "state" / "fomc_statement_dates.csv"
    )
    ff = u[
        ["S", "kc", "kp", "ask", "mid", "iv_var_ask", "iv_var_mid", "R", "R_official"]
    ].join(pf)
    ff["rv_hat_card"] = f0["rv_hat"].reindex(ff.index)
    ok = (
        np.isfinite(ff["iv_var_ask"])
        & (ff["iv_var_ask"] > 0)
        & np.isfinite(ff["iv_var_mid"])
        & (ff["iv_var_mid"] > 0)
        & (ff["RV_d"] > 0)
        & np.isfinite(ff["rv_hat_card"])
    )
    gate["ratio_universe"] = {
        "book_universe_days": int(len(ff)),
        "dropped_incomplete": int((~ok).sum()),
        "dropped_iv_ask_nan": int((~np.isfinite(ff["iv_var_ask"])).sum()),
        "dropped_iv_mid_nan": int((~np.isfinite(ff["iv_var_mid"])).sum()),
        "dropped_rv_d": int((~(ff["RV_d"] > 0)).sum()),
        "dropped_no_rv_hat": int((~np.isfinite(ff["rv_hat_card"])).sum()),
    }
    ff = ff[ok].copy()
    ff["z"] = np.log(ff["RV_d"] / ff["iv_var_ask"])
    ff["x_card"] = np.log(ff["rv_hat_card"] / ff["iv_var_ask"])
    ff["log_ivmid"] = np.log(ff["iv_var_mid"])
    ff["log_ask_mid"] = np.log(ff["ask"] / ff["mid"])
    ff["zbar5"] = ff["z"].shift(1).rolling(5, min_periods=5).mean()
    ff["zbar22"] = ff["z"].shift(1).rolling(22, min_periods=22).mean()
    dow = ff.index.dayofweek
    for k in range(4):
        ff[f"dow_{k}"] = (dow == k).astype(float)
    ff.reset_index().to_parquet(OUT / "features.parquet", index=False)
    log(
        f"idea 2 universe: {len(ff)} days {ff.index.min().date()} .. {ff.index.max().date()}; "
        f"{json.dumps(gate['ratio_universe'])}"
    )
    preds = []
    for n in N_WF:
        t0 = time.time()
        preds.append(walk_forward(ff, n))
        log(f"walk-forward N={n}: {len(preds[-1])} days, {time.time() - t0:.0f}s")
    pr = pd.concat(preds, axis=1)
    pr.reset_index().to_parquet(work / "ratio_preds.parquet", index=False)
    pr.reset_index().to_parquet(OUT / "ratio_predictions.parquet", index=False)
    save_gate(gate)
    return 0


# --------------------------------------------------------------------------- report
def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def block_boot_means(x: np.ndarray, rng) -> np.ndarray:
    """Circular 20-day-block bootstrap draws of mean(x) (N_BOOT draws, in chunks)."""
    n = len(x)
    nb = int(np.ceil(n / BLOCK))
    out = np.empty(N_BOOT)
    step = 1000
    for s0 in range(0, N_BOOT, step):
        k = min(step, N_BOOT - s0)
        starts = rng.integers(0, n, size=(k, nb))
        idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(
            k, -1
        )[:, :n]
        out[s0 : s0 + k] = x[idx].mean(1)
    return out


def score(pnl: pd.Series, buy: pd.Series, R: pd.Series) -> dict:
    rb = R[buy]
    return {
        "days": int(len(pnl)),
        "buys": int(buy.sum()),
        "share": float(buy.mean()) if len(buy) else float("nan"),
        "mean_R_buy": float(rb.mean()) if len(rb) else float("nan"),
        "t_R_buy": tstat(rb.to_numpy()),
        "pnl_day": float(pnl.mean()) if len(pnl) else float("nan"),
        "t_pnl": tstat(pnl.to_numpy()),
    }


def phase_report(work: Path) -> int:
    frozen = freeze(work)
    _, _, _, PR = frozen_modules(frozen)
    gate = load_gate()
    hard = {k: gate.get(k) for k in ("G1_passed", "G2_passed", "G4_passed")}
    if not all(hard.values()):
        log(f"hard gates not passed: {hard}: no P&L")
        return 2
    book = pd.read_parquet(OUT / "book.parquet")
    u = universe(book)
    ff = pd.read_parquet(OUT / "features.parquet").set_index("day")
    pr = pd.read_parquet(OUT / "ratio_predictions.parquet").set_index("day")
    dec = u[["S", "kc", "kp", "ask", "R", "R_official", "src"]].copy()
    first_live = {}
    for tw in TW_ALL:
        fc = pd.read_parquet(OUT / f"forecasts_tw{tw}.parquet").set_index("day")
        rv = fc["rv_hat"].reindex(dec.index)
        ps = np.array(
            [
                PR.package_price(math.sqrt(r), s, kc, kp) if np.isfinite(r) else np.nan
                for r, s, kc, kp in zip(rv, dec["S"], dec["kc"], dec["kp"])
            ]
        )
        name = "F0" if tw == TW_CARD else f"TW{tw}"
        dec[f"pstar_{name}"] = ps
        dec[f"live_{name}"] = np.isfinite(ps)
        dec[f"buy_{name}"] = np.isfinite(ps) & (dec["ask"] <= ps)
        first_live[name] = str(dec.index[dec[f"live_{name}"]].min().date())
    for n in N_WF:
        for m in ("ridge", "ols3", "logit"):
            name = f"{m}_{n}"
            col = f"p_logit_{n}" if m == "logit" else f"zhat_{m}_{n}"
            v = pr[col].reindex(dec.index)
            dec[f"live_{name}"] = v.notna()
            dec[f"buy_{name}"] = (v > (0.5 if m == "logit" else 0.0)) & v.notna()
            first_live[name] = str(dec.index[dec[f"live_{name}"]].min().date())
    names = ["F0", "TW1000", "TW500", "TW250"] + [
        f"{m}_{n}" for n in N_WF for m in ("ridge", "ols3", "logit")
    ]
    dec.reset_index().to_parquet(OUT / "decisions.parquet", index=False)
    gate["first_live"] = first_live
    periods = {"research": RESEARCH, "post": POST}

    def in_period(idx, per):
        return (idx >= per[0]) & (idx <= per[1])

    # forecast diagnostics on the universe days: QLIKE of each forecast of RV_d
    def qlike(y, f):
        r = np.asarray(y, float) / np.asarray(f, float)
        return float(np.mean(r - np.log(r) - 1.0))

    y = ff["RV_d"].reindex(dec.index)
    diag = []
    fcs = {
        f"rv_hat_{'F0' if tw == TW_CARD else f'TW{tw}'}": pd.read_parquet(
            OUT / f"forecasts_tw{tw}.parquet"
        )
        .set_index("day")["rv_hat"]
        .reindex(dec.index)
        for tw in TW_ALL
    }
    fcs["IVmid"] = ff["iv_var_mid"].reindex(dec.index)
    fcs["IVask"] = ff["iv_var_ask"].reindex(dec.index)
    for per_name, per in periods.items():
        m = in_period(dec.index, per) & np.isfinite(y) & (y > 0)
        for k, f in fcs.items():
            mm = m & np.isfinite(f) & (f > 0)
            diag.append(
                {
                    "period": per_name,
                    "forecast": k,
                    "days": int(mm.sum()),
                    "qlike": qlike(y[mm], f[mm]),
                    "median_f_over_IVask": float(np.median(f[mm] / fcs["IVask"][mm])),
                    "median_RV_over_f": float(np.median(y[mm] / f[mm])),
                    "corr_log": float(np.corrcoef(np.log(y[mm]), np.log(f[mm]))[0, 1]),
                }
            )
    diag = pd.DataFrame(diag)
    diag.to_csv(OUT / "forecast_diagnostics.csv", index=False)

    rows, yrows = [], []
    rng = np.random.default_rng(SEED)
    for rcol, tag in (("R", "chain_close"), ("R_official", "official_close")):
        for nm in names:
            live = dec[f"live_{nm}"]
            buy = dec[f"buy_{nm}"]
            R = dec[rcol]
            pnl = pd.Series(np.where(buy, R, 0.0), index=dec.index)
            pnl0 = pd.Series(np.where(dec["buy_F0"], R, 0.0), index=dec.index)
            for per_name, per in periods.items():
                m = live & in_period(dec.index, per) & dec["live_F0"]
                sc = score(pnl[m], buy[m], R[m])
                sc0 = score(pnl0[m], dec["buy_F0"][m], R[m])
                row = {
                    "close": tag,
                    "variant": nm,
                    "period": per_name,
                    "first_day": str(dec.index[m].min().date()) if m.any() else None,
                    **sc,
                    "F0_pnl_same_days": sc0["pnl_day"],
                    "F0_t_same_days": sc0["t_pnl"],
                    "F0_buys_same_days": sc0["buys"],
                    "agree_F0": float((buy[m] == dec["buy_F0"][m]).mean()),
                }
                if m.sum() >= 2 * BLOCK:
                    own = block_boot_means(pnl[m].to_numpy(), rng)
                    row["pnl_ci_lo"], row["pnl_ci_hi"] = np.quantile(
                        own, [0.025, 0.975]
                    )
                    if nm != "F0":
                        d = (pnl[m] - pnl0[m]).to_numpy()
                        bd = block_boot_means(d, rng)
                        row["diff_vs_F0"] = float(d.mean())
                        row["diff_ci_lo"], row["diff_ci_hi"] = np.quantile(
                            bd, [0.025, 0.975]
                        )
                        row["diff_boot_share_le0"] = float((bd <= 0).mean())
                rows.append(row)
            if tag == "chain_close":
                for y in range(2020, 2027):
                    m = live & (dec.index.year == y)
                    if m.sum() == 0:
                        continue
                    sc = score(pnl[m], buy[m], R[m])
                    yrows.append({"variant": nm, "year": y, **sc})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False)
    byy = pd.DataFrame(yrows)
    byy.to_csv(OUT / "by_year.csv", index=False)
    # the days every variant is live on
    all_live = np.logical_and.reduce([dec[f"live_{nm}"].to_numpy() for nm in names])
    crow = []
    for nm in names:
        buy = dec[f"buy_{nm}"]
        pnl = pd.Series(np.where(buy, dec["R"], 0.0), index=dec.index)
        for per_name, per in periods.items():
            m = all_live & in_period(dec.index, per)
            crow.append(
                {
                    "variant": nm,
                    "period": per_name,
                    "first_day": str(dec.index[m].min().date()),
                    **score(pnl[m], buy[m], dec["R"][m]),
                }
            )
    common = pd.DataFrame(crow)
    common.to_csv(OUT / "common_days.csv", index=False)
    # study 85 cross-check: F0 on 2024-05 .. 2025-12
    m85 = dec["live_F0"] & in_period(
        dec.index, (pd.Timestamp("2024-05-01"), pd.Timestamp("2025-12-31"))
    )
    p85 = pd.Series(np.where(dec["buy_F0"], dec["R"], 0.0), index=dec.index)[m85]
    gate["F0_2024_05_2025_12"] = {
        "days": int(m85.sum()),
        "buys": int(dec["buy_F0"][m85].sum()),
        "pnl_day": float(p85.mean()),
        "t": tstat(p85.to_numpy()),
    }
    save_gate(gate)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    lines = []
    lines.append(
        "STUDY 89 -- recent training windows (idea 1) and a trade-aligned target (idea 2)"
    )
    lines.append(
        f"gates: {json.dumps({k: gate.get(k) for k in ('G1_passed', 'G2_passed', 'G4_passed')})}"
    )
    lines.append(f"first live day: {json.dumps(first_live)}")
    lines.append(
        f"study 85 cross-check (F0, SPX, 2024-05..2025-12): {json.dumps(gate['F0_2024_05_2025_12'])}"
    )
    cols = [
        "variant",
        "period",
        "first_day",
        "days",
        "buys",
        "share",
        "mean_R_buy",
        "t_R_buy",
        "pnl_day",
        "t_pnl",
        "pnl_ci_lo",
        "pnl_ci_hi",
        "F0_pnl_same_days",
        "diff_vs_F0",
        "diff_ci_lo",
        "diff_ci_hi",
        "diff_boot_share_le0",
        "agree_F0",
    ]
    for tag in ("chain_close", "official_close"):
        lines.append(
            f"\n== per variant, on its live days (F0 on the same days), close = {tag} =="
        )
        lines.append(
            summ[summ["close"] == tag][cols].to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            )
        )
    lines.append(
        "(pnl_ci / diff_ci: 95 % percentile CI of the circular 20-day-block bootstrap, "
        "10,000 draws; diff_boot_share_le0 = share of draws of the mean difference <= 0)"
    )
    lines.append("\n== forecast diagnostics on the universe days (QLIKE vs RV_d) ==")
    lines.append(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("\n== by calendar year (chain close) ==")
    lines.append(
        byy.pivot(index="variant", columns="year", values="pnl_day")
        .reindex(names)
        .to_string(float_format=lambda v: f"{v:+.4f}")
    )
    lines.append("\n(t of per-day P&L by year)")
    lines.append(
        byy.pivot(index="variant", columns="year", values="t_pnl")
        .reindex(names)
        .to_string(float_format=lambda v: f"{v:+.2f}")
    )
    lines.append("\n(buys by year)")
    lines.append(
        byy.pivot(index="variant", columns="year", values="buys")
        .reindex(names)
        .to_string()
    )
    lines.append("\n== the days common to every variant (chain close) ==")
    lines.append(common.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase", required=True, choices=["book", "f0", "engine", "ratio", "report"]
    )
    ap.add_argument("--tw", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "p89_work"))
    a = ap.parse_args(argv)
    assert a.workers <= 3
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.phase == "book":
        rc = phase_book(work)
    elif a.phase == "f0":
        rc = phase_f0(work)
    elif a.phase == "engine":
        assert a.tw in TW_ALL, a.tw
        rc = phase_engine(work, a.tw, a.workers)
    elif a.phase == "ratio":
        rc = phase_ratio(work)
    else:
        rc = phase_report(work)
    with open(
        OUT / f"run_{a.phase}{'' if a.tw is None else a.tw}.log", "w", encoding="utf-8"
    ) as fh:
        fh.write("\n".join(_LOG) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
