"""Study 88 -- does a SELF-REFITTING, strictly walk-forward buy bar restore the card's edge?

QUESTION.  The daily card buys the SPXW 0DTE nearest-OTM strangle at 15:30 iff
its ask <= P* = ``live.ibkr.pricing.package_price(sqrt(rv_hat), S, Kc, Kp)``
and holds it to the 16:00 cash settlement (month-ends are a separate book and
are excluded here).  In the research years it made +0.0455/day at the SPXW
ask buying on 34 % of days (the deck, ``daily_sub_live_ridge``); after
2024-04 the live pipeline buys on ~54 % of days and loses (study 85: -0.0129
/day on SPX 2024-05 .. 2025-12).  The option market now prices the close
closer to realized, so the FIXED bar ``ask <= P*`` is probably too loose.  Is
a bar that refits itself on PAST days only enough to bring the edge back?

ONE MODEL ACROSS THE SPAN.  The live card's own forecast table -- bucket
free_vix_only on the extended panel, the 13 regular-hours ridge arms, built
2026-09-24 by the live pipeline (``live/close_signal/.scratch/
yhat_close_signal.parquet``; copied to ``OUT/inputs/`` on the first run and
read from there afterwards, since the card rewrites the scratch copy every
day) -- through the card's own causal MZ loader (``forecast.recalibrate`` ->
``asl.load_yhat_1530_mz_cached``, method "mean": coefficients from strictly
prior sessions) for EVERY session date.  The deck's own rv_hat (research years
only) is scored alongside as a labelled second model.  All live code is
imported from a ``git archive HEAD`` copy (frozen), read-only.

BOOKS (every non-month-end session with a book):
  2020-01 .. 2025-12  data/spxw_chain.parquet through study 79's ``spxw_book``
                      (TRUE UTC -> ET, 15:30 rows of the 0DTE expiry, ask > 0,
                      nearest OTM on the listed strikes, S = the chain's
                      underlying), one calendar year per call (streamed);
  2026-01 .. 2026-09  data/archive/spxw_opra/cbbo1m_<day>.parquet: the NBBO
                      standing at 15:30:00, S = the book's put-call-parity spot
                      (study 79 ``parity_spot``), strikes = the day's listed
                      strikes (defs_<day>.parquet).
  Settlement at the official SPX close (^GSPC daily, cached in OUT) -- the
  deck's S_close exactly (gated).  Pre-2022 the chain lists 0DTE SPXW only on
  Monday / Wednesday / Friday, so a "session" below is an ELIGIBLE day
  (non-month-end, finite rv_hat, two-leg ask, a close), not a calendar one.

PER DAY: P*, R = payoff / ask - 1 (EVERY day, bought or not), z = log(P*/ask),
m = log(ask/S), iv = the package's mid-implied total variance
(``invert_total_vol``), rv = the 16:00 row's rv_raw (the ES 15:30-16:00
realized variance, the model's target).

RULES.  W_i = the N eligible days strictly before day i (day j's outcome is
known at 16:00 of day j, before the 15:30 decision of day j+1):
  F0   buy iff ask <= P*                                   (the card)
  F1   c_i = argmax over c in 0.70, 0.72, .., 1.30 of the W_i mean per-day P&L
       of "buy iff ask <= c P*", among the c with >= 40 buys in W_i (ties: the
       c nearest 1, then the smaller); no such c -> c = 1; buy iff ask_i <=
       c_i P*_i
  F2   OLS on W_i of R on (1, z); buy iff the fitted R at z_i > 0
  F2x  OLS on W_i of R on (1, z, m, z m) -- the interaction WITH its main
       effect; buy iff fitted R > 0 (secondary: four parameters)
  F3   k_i = ref_i / median_{W_i}(rv / iv); buy iff ask_i <= k_i P*_i, where
       ref_i = median(rv / iv) over the research-period (2020-01 .. 2024-04)
       days before i: expanding inside the research years, the full-research
       value after 2024-04-30 -- so no future day is ever used.  Direction:
       k < 1 (a tighter bar) when the market prices closer to realized, the
       stated hypothesis.  The request's literal wording ("scale P* by the
       trailing median of realized/implied divided by its research value")
       scales the OTHER way (k > 1, a looser bar, when realized/implied rises);
       it is run as F3lit, a labelled sensitivity, never a candidate.
N = 250 primary, 125 and 500 sensitivities.  A rule is live from the (N+1)-th
eligible day; the rules of one N are compared on the same days; the
N-sensitivity table uses the days where N = 500 is live.

SCORING.  Per-day P&L = R on a buy day, 0 otherwise (premium units, at the
ask).  Per rule and span: days, buys, share, mean R on buys, per-day P&L and
its iid t; by calendar year; the paired circular 20-day block bootstrap
(10,000 draws, one index set per span shared by every rule) of the per-day
difference vs F0, research span (first live day .. 2024-04-30) and post span
(2024-05-01 .. 2026-09-23) separately, one-sided p = share of draws with mean
difference <= 0; the rule's own P&L CI likewise.  Holm over the three primary
rules (F1, F2, F3 at N = 250).

GATES.
  G1  the card's rv_hat from the table: 2026-09-24 against the card's cache of
      the SAME table (exact), and 2026-08-31 / 09-23 / 09-24 against the
      fastpath_check as-of runs (panel cut at 15:30 of that day);
  G2  P* on 2026-09-24 at S = the ^GSPC 15:29 bar close, 7710C / 7705P (the
      card printed 6.60);
  G3  the book and scoring: F0 on the DECK's rv_hat reproduces +0.0455/day and
      34 % buys on the 814 non-month-end research days; pair / ask / S vs deck;
  G4  OPRA vs chain at 15:30 on the overlap (2023-03-28 .. 2025-12-31): the
      same pair's ask, the parity spot vs the chain spot, the pair chosen;
  G5  study 85's 15:30 card (SPX 2024-05 .. 2025-12): decision agreement and
      its P&L at its own settlement (the 16:00 spot print).

OUTPUTS (results/atm_straddle_intraday_holdclose/proposals/88/):
  inputs/yhat_close_signal.parquet (the frozen model table), spx_close_yf.csv,
  gspc_1m_2026-09-24.csv, days.parquet/.csv (the eligible-day table),
  decisions.parquet (per day x rule), summary.csv, by_year.csv, bootstrap.csv,
  n_sensitivity.csv, c_path.csv, c_path_halfyear.csv, deck_model.csv,
  gate.json, summary.txt, run.log

REPRODUCE (~3 min, 4 processes, < 1 GB each):
    C:/Users/james/miniconda3/envs/285J/python.exe writeup/intraday_proposals/88_walkforward_threshold.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROPOSALS = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals"
OUT = PROPOSALS / "88"
CHAIN = REPO / "data" / "spxw_chain.parquet"
SPOT = REPO / "data" / "spxw_spot.parquet"
OPRA_DIR = REPO / "data" / "archive" / "spxw_opra"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
CARD_TABLE = REPO / "live" / "close_signal" / ".scratch" / "yhat_close_signal.parquet"
CARD_SCRATCH = REPO / "live" / "close_signal" / ".scratch"
S85 = PROPOSALS / "85"
ET = "America/New_York"

GRID = np.round(0.70 + 0.02 * np.arange(31), 2)  # 0.70 .. 1.30
NS = (125, 250, 500)
N_PRIMARY = 250
MIN_BUYS = 40
BLOCK = 20
N_BOOT = 10_000
SEED = 88
RESEARCH_END = pd.Timestamp("2024-04-30")
POST_START = pd.Timestamp("2024-05-01")
CHAIN_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
OPRA_FIRST_ONLY = pd.Timestamp("2026-01-01")  # the chain ends 2025-12-31
DECK_PNL, DECK_SHARE = 0.0455, 0.34  # the research deck's card (the request)
CARD_0924 = {"rv_hat": 2.049e-06, "P_star": 6.60, "Kc": 7710.0, "Kp": 7705.0}
PRIMARY_RULES = ("F1", "F2", "F3")
RULES = ("F1", "F2", "F2x", "F3", "F3lit")

_LOG: list[str] = []


def log(msg: str = "") -> None:
    _LOG.append(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------- frozen code
def freeze(work: Path) -> tuple[Path, str]:
    """``git archive HEAD`` of the code this study imports, into ``work/frozen``."""
    sha = (
        subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"])
        .decode()
        .strip()
    )
    dst = work / "frozen"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    tar = work / "frozen.tar"
    subprocess.check_call(
        [
            "git",
            "-C",
            str(REPO),
            "archive",
            "--format=tar",
            "-o",
            str(tar),
            "HEAD",
            "live",
            "notebooks/atm_straddle_lib.py",
            "writeup/intraday_proposals/79_xsp_close_book.py",
        ]
    )
    with tarfile.open(tar) as tf:
        tf.extractall(dst)
    tar.unlink()
    return dst, sha


def _frozen_path(frozen: str) -> None:
    if frozen not in sys.path:
        sys.path.insert(0, frozen)


def _p79(frozen: str) -> Any:
    _frozen_path(frozen)
    p = Path(frozen) / "writeup" / "intraday_proposals" / "79_xsp_close_book.py"
    spec = importlib.util.spec_from_file_location("p79_frozen_88", p)
    assert spec is not None and spec.loader is not None
    mod: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CHAIN = CHAIN  # the data live in the working tree, not in the archive
    mod.SPOT = SPOT
    return mod


# --------------------------------------------------------------------------- books (workers)
def chain_year(args: tuple[str, int]) -> pd.DataFrame:
    """Study 79's spxw_book on one calendar year of the chain (a filtered read)."""
    frozen, year = args
    p79 = _p79(frozen)
    b = p79.spxw_book(pd.DatetimeIndex([f"{year}-01-01", f"{year}-12-31"]))
    b.index = pd.DatetimeIndex(b.index).normalize()
    return b


def opra_day(args: tuple[str, str, str, str]) -> dict:
    """The SPXW NBBO standing at 15:30:00, its parity spot and the listed strikes."""
    frozen, qpath, dpath, day_s = args
    p79 = _p79(frozen)
    day = pd.Timestamp(day_s)
    defs = pd.read_parquet(dpath, columns=["strike_price", "expiration"])
    exp = (
        pd.to_datetime(defs["expiration"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    strikes = np.sort(defs.loc[exp == day, "strike_price"].astype(float).unique())
    q = p79.load_quotes(Path(qpath))
    q = q[q["symbol"].astype(str).str.slice(6, 12) == day.strftime("%y%m%d")]
    stamp = pd.Timestamp(f"{day.date()} 15:30:00")
    out = {
        "day": day,
        "strikes": strikes,
        "S_par": np.nan,
        "lag_s": np.nan,
        "book": None,
    }
    if q.empty:
        return out
    b = p79.book_at(q, day, "15:30:00")
    if b.empty:
        return out
    out["lag_s"] = float((stamp - q.loc[q["t"] <= stamp, "t"].max()).total_seconds())
    out["S_par"] = float(p79.parity_spot(b))
    out["book"] = b[["bid_px_00", "ask_px_00"]].reset_index()
    return out


def leg(book: pd.DataFrame, k: float, cp: str) -> tuple[float, float]:
    """bid, ask of one contract as study 79's ``quote``: NaN when absent or unquoted."""
    if not np.isfinite(k):
        return np.nan, np.nan
    r = book[(book["strike"] == k) & (book["cp"] == cp)]
    if r.empty:
        return np.nan, np.nan
    r = r.iloc[-1]
    bid, ask = float(r["bid_px_00"]), float(r["ask_px_00"])
    if not (np.isfinite(bid) and np.isfinite(ask)) or ask <= 0:
        return np.nan, np.nan
    return bid, ask


# --------------------------------------------------------------------------- stats
def block_boot_idx(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """Circular block bootstrap indices (study 85's)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, nb * block)[:, :n]


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=lambda k: p[k])
    m, run, out = len(keys), 0.0, {}
    for j, k in enumerate(keys):
        run = max(run, min(1.0, (m - j) * p[k]))
        out[k] = run
    return out


def stats(buy: np.ndarray, R: np.ndarray) -> dict:
    buy = np.asarray(buy, bool)
    pnl = np.where(buy, R, 0.0)
    Rb = R[buy]
    return {
        "days": int(len(buy)),
        "buys": int(buy.sum()),
        "share": float(buy.mean()) if len(buy) else np.nan,
        "mean_R_buy": float(Rb.mean()) if len(Rb) else np.nan,
        "t_R_buy": tstat(Rb),
        "hit_buy": float((Rb > 0).mean()) if len(Rb) else np.nan,
        "pnl_day": float(pnl.mean()) if len(pnl) else np.nan,
        "t_pnl": tstat(pnl),
    }


# --------------------------------------------------------------------------- rules
def run_rules(
    d: pd.DataFrame, N: int, ref_ratio: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Walk-forward decisions for every rule at window N; NaN / False before day N.

    ``ref_ratio[i]`` is F3's reference median (research days before i).
    """
    ask, ps, R = (
        d["ask"].to_numpy(float),
        d["P_star"].to_numpy(float),
        d["R"].to_numpy(float),
    )
    z, m = d["z"].to_numpy(float), d["m"].to_numpy(float)
    rr = d["rv_over_iv"].to_numpy(float)
    n = len(d)
    out: dict[str, dict[str, np.ndarray]] = {
        r: {
            "buy": np.zeros(n, bool),
            "live": np.zeros(n, bool),
            "param": np.full(n, np.nan),
        }
        for r in RULES
    }
    out["F1"]["default"] = np.zeros(n, bool)
    out["F1"]["best_trailing_pnl"] = np.full(n, np.nan)
    out["F2"]["slope"] = np.full(n, np.nan)
    X2 = np.column_stack([np.ones(n), z])
    X2x = np.column_stack([np.ones(n), z, m, z * m])
    for i in range(N, n):
        w = slice(i - N, i)
        # F1 -- trailing-optimal multiplier
        B = ask[w][None, :] <= GRID[:, None] * ps[w][None, :]
        nb = B.sum(1)
        pn = np.where(B, R[w][None, :], 0.0).sum(1) / N
        ok = nb >= MIN_BUYS
        if ok.any():
            best = pn[ok].max()
            cand = GRID[ok & (pn >= best - 1e-12)]
            c = float(cand[np.lexsort((cand, np.abs(cand - 1.0)))][0])
            out["F1"]["best_trailing_pnl"][i] = best
        else:
            c = 1.0
            out["F1"]["default"][i] = True
        out["F1"]["param"][i] = c
        out["F1"]["buy"][i] = ask[i] <= c * ps[i]
        # F2 / F2x -- trailing regressions of R
        for name, X in (("F2", X2), ("F2x", X2x)):
            beta, *_ = np.linalg.lstsq(X[w], R[w], rcond=None)
            pred = float(X[i] @ beta)
            out[name]["param"][i] = pred
            out[name]["buy"][i] = pred > 0
            if name == "F2":
                out["F2"]["slope"][i] = beta[1]
        # F3 / F3lit -- trailing premium adjustment
        tr = float(np.nanmedian(rr[w]))
        k = ref_ratio[i] / tr
        out["F3"]["param"][i] = k
        out["F3"]["buy"][i] = ask[i] <= k * ps[i]
        out["F3lit"]["param"][i] = 1.0 / k
        out["F3lit"]["buy"][i] = ask[i] <= ps[i] / k
        for r in RULES:
            out[r]["live"][i] = True
    return out


def research_reference(d: pd.DataFrame) -> np.ndarray:
    """median(rv/iv) over the research days strictly before each day (no future day)."""
    rr = d["rv_over_iv"].to_numpy(float)
    res = np.asarray(d.index <= RESEARCH_END)
    ref = np.full(len(d), np.nan)
    for i in range(len(d)):
        sel = res[:i]
        x = rr[:i][sel]
        x = x[np.isfinite(x)]
        if len(x):
            ref[i] = float(np.median(x))
    return ref


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--work", default=str(Path(tempfile.gettempdir()) / "p88_walkforward")
    )
    a = ap.parse_args(argv)
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs").mkdir(exist_ok=True)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    frozen, sha = freeze(work)
    fz = str(frozen)
    _frozen_path(fz)
    from live.close_signal import forecast as F  # noqa: E402  (frozen)
    from live.ibkr.calendar_guard import is_last_session_of_month  # noqa: E402
    from live.ibkr.pricing import invert_total_vol, package_price  # noqa: E402

    gate: dict[str, Any] = {"frozen_sha": sha, "frozen_dir": fz}
    log(f"frozen code: git archive HEAD {sha} -> {fz}")

    # ---- 1. the model table (frozen copy of the card's)
    tab_p = OUT / "inputs" / "yhat_close_signal.parquet"
    if not tab_p.exists():
        shutil.copy2(CARD_TABLE, tab_p)
        log(f"copied the card's table {CARD_TABLE} -> {tab_p}")
    tab = pd.read_parquet(tab_p)
    gate["model_table"] = {
        "path": str(tab_p),
        "sha1": hashlib.sha1(tab_p.read_bytes()).hexdigest(),
        "rows": int(len(tab)),
        "first_t": str(tab["t"].min()),
        "last_t": str(tab["t"].max()),
    }
    et = pd.DatetimeIndex(tab["t"]).tz_convert(ET)
    sess = pd.DatetimeIndex(
        et[(et.hour == 16) & (et.minute == 0)].tz_localize(None).normalize()
    ).unique()
    t1 = time.time()
    mz = F.recalibrate(tab, sess, frozen, work / "mz", tag="p88")
    mz.index = pd.DatetimeIndex(mz.index).normalize()
    log(
        f"MZ (card loader) on {len(sess)} session dates: {len(mz)} rv_hat, {mz.index.min().date()} .. "
        f"{mz.index.max().date()} ({time.time() - t1:.0f}s)"
    )

    # ---- G1 the card's rv_hat
    g1: dict[str, Any] = {"same_table_cache": [], "asof_fastpath": []}
    for f in sorted(glob.glob(str(CARD_SCRATCH / "mz_cache" / "*.parquet"))):
        c = pd.read_parquet(f)
        c.index = pd.DatetimeIndex(c.index).normalize()
        for dd, r in c.iterrows():
            mine = float(mz.at[dd, "rv_hat"]) if dd in mz.index else np.nan
            g1["same_table_cache"].append(
                {
                    "file": Path(f).name,
                    "date": str(dd.date()),
                    "card_rv_hat": float(r["rv_hat"]),
                    "rv_hat": mine,
                    "rel": abs(mine / float(r["rv_hat"]) - 1),
                }
            )
    for f in sorted(
        glob.glob(
            str(
                CARD_SCRATCH
                / "fastpath_check"
                / "*"
                / "full"
                / "mz_cache"
                / "*.parquet"
            )
        )
    ):
        c = pd.read_parquet(f)
        c.index = pd.DatetimeIndex(c.index).normalize()
        for dd, r in c.iterrows():
            mine = float(mz.at[dd, "rv_hat"]) if dd in mz.index else np.nan
            g1["asof_fastpath"].append(
                {
                    "date": str(dd.date()),
                    "card_rv_hat": float(r["rv_hat"]),
                    "rv_hat": mine,
                    "rel": abs(mine / float(r["rv_hat"]) - 1),
                }
            )
    g1["same_table_max_rel"] = max(
        (x["rel"] for x in g1["same_table_cache"]), default=np.nan
    )
    g1["asof_max_rel"] = max((x["rel"] for x in g1["asof_fastpath"]), default=np.nan)
    g1["passed"] = bool(g1["same_table_max_rel"] <= 1e-12)
    gate["G1_card_rv_hat"] = g1
    log(
        "G1 card rv_hat: same table max rel %.2e on %d; as-of fastpath runs max rel %.2e on %d -> %s"
        % (
            g1["same_table_max_rel"],
            len(g1["same_table_cache"]),
            g1["asof_max_rel"],
            len(g1["asof_fastpath"]),
            "PASS" if g1["passed"] else "FAIL",
        )
    )
    for x in g1["asof_fastpath"]:
        log(
            f"   as-of {x['date']}: card {x['card_rv_hat']:.6e}  table {x['rv_hat']:.6e}  rel {x['rel']:.2e}"
        )
    # G1b: how much the table's PAST 16:00 forecasts move with the sample end (the as-of tables
    # of the fastpath runs vs the model table) -- the inherited full-sample-scalar dependence
    g1b = []
    for f in sorted(
        glob.glob(
            str(
                CARD_SCRATCH
                / "fastpath_check"
                / "*"
                / "full"
                / "yhat_close_signal.parquet"
            )
        )
    ):
        a_ = pd.read_parquet(f)
        j = a_.merge(tab, on="t", suffixes=("_asof", ""))
        e_ = pd.DatetimeIndex(j["t"]).tz_convert(ET)
        j = j[(e_.hour == 16) & (e_.minute == 0)]
        e_ = pd.DatetimeIndex(j["t"]).tz_convert(ET).tz_localize(None).normalize()
        cut = e_.max()
        j = j[(e_ < cut) & (e_ >= pd.Timestamp("2020-01-01"))]
        rel = (j["yhat"] / j["yhat_asof"] - 1).abs().to_numpy(float)
        g1b.append(
            {
                "asof_table": Path(f).parents[1].name,
                "days_before_cut": int(len(rel)),
                "share_rel_gt_1e-3": float((rel > 1e-3).mean()),
                "share_rel_gt_1e-2": float((rel > 1e-2).mean()),
                "share_rel_gt_5e-2": float((rel > 5e-2).mean()),
                "median_rel": float(np.median(rel)),
                "max_rel": float(rel.max()),
            }
        )
    gate["G1b_sample_end_sensitivity_1600_yhat"] = g1b
    log(
        "G1b past 16:00 yhat vs the as-of tables (2020-01 .. the day before each cut): "
        + json.dumps(g1b)
    )

    # ---- G2 P* on 2026-09-24
    m1 = pd.read_csv(OUT / "gspc_1m_2026-09-24.csv", index_col=0)
    m1.index = pd.to_datetime(m1.index, utc=True).tz_convert(ET)
    s0924 = float(m1.loc[m1.index.strftime("%H:%M") == "15:29", "Close"].iloc[0])
    rv0924 = float(mz.at[pd.Timestamp("2026-09-24"), "rv_hat"])
    ps0924 = package_price(np.sqrt(rv0924), s0924, CARD_0924["Kc"], CARD_0924["Kp"])
    g2 = {
        "rv_hat": rv0924,
        "S_gspc_1529_close": s0924,
        "Kc": CARD_0924["Kc"],
        "Kp": CARD_0924["Kp"],
        "P_star": ps0924,
        "card_P_star": CARD_0924["P_star"],
        "card_rv_hat": CARD_0924["rv_hat"],
        "passed": bool(
            round(ps0924, 2) == CARD_0924["P_star"]
            and abs(rv0924 / CARD_0924["rv_hat"] - 1) < 5e-4
        ),
    }
    gate["G2_pstar_0924"] = g2
    log(
        f"G2 2026-09-24: rv_hat {rv0924:.4e} (card {CARD_0924['rv_hat']:.3e}), S {s0924:.2f} -> P* {ps0924:.4f} "
        f"on 7710C/7705P (card 6.60) -> {'PASS' if g2['passed'] else 'FAIL'}"
    )

    # ---- 2. books
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=min(2, a.workers)) as ex:
        parts = list(ex.map(chain_year, [(fz, y) for y in CHAIN_YEARS]))
    chain = pd.concat(parts).sort_index()
    chain = chain[~chain.index.duplicated(keep="first")]
    log(
        f"chain books: {len(chain)} days {chain.index.min().date()} .. {chain.index.max().date()} ({time.time() - t1:.0f}s)"
    )
    files = sorted(glob.glob(str(OPRA_DIR / "cbbo1m_*.parquet")))
    jobs = []
    for f in files:
        ds = Path(f).stem.split("_")[1]
        dp = OPRA_DIR / f"defs_{ds}.parquet"
        if dp.exists():
            jobs.append((fz, f, str(dp), ds))
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        opra = list(ex.map(opra_day, jobs, chunksize=8))
    log(f"OPRA books: {len(opra)} days ({time.time() - t1:.0f}s)")
    orow = []
    for o in opra:
        day, bk = o["day"], o["book"]
        rec = {
            "day": day,
            "S_par": o["S_par"],
            "lag_s": o["lag_s"],
            "o_kc": np.nan,
            "o_kp": np.nan,
            "o_bid": np.nan,
            "o_ask": np.nan,
            "c_pair_o_ask": np.nan,
            "c_pair_o_bid": np.nan,
        }
        if bk is not None and np.isfinite(o["S_par"]) and len(o["strikes"]):
            above, below = (
                o["strikes"][o["strikes"] >= o["S_par"]],
                o["strikes"][o["strikes"] <= o["S_par"]],
            )
            if len(above) and len(below):
                kc, kp = float(above.min()), float(below.max())
                bc, ac = leg(bk, kc, "C")
                bp, ap_ = leg(bk, kp, "P")
                rec.update(o_kc=kc, o_kp=kp, o_bid=bc + bp, o_ask=ac + ap_)
            if day in chain.index:
                ckc, ckp = (
                    float(chain.at[day, "spx_kc"]),
                    float(chain.at[day, "spx_kp"]),
                )
                bc, ac = leg(bk, ckc, "C")
                bp, ap_ = leg(bk, ckp, "P")
                rec.update(c_pair_o_ask=ac + ap_, c_pair_o_bid=bc + bp)
        orow.append(rec)
    ob = pd.DataFrame(orow).set_index("day").sort_index()

    # ---- G4 OPRA vs chain on the overlap
    ov = chain.join(ob, how="inner")
    ov = ov[ov.index < OPRA_FIRST_ONLY]
    okq = np.isfinite(ov["c_pair_o_ask"])
    dask = (ov["c_pair_o_ask"] - ov["spx_ask"])[okq]
    g4 = {
        "overlap_days": int(len(ov)),
        "chain_pair_quoted_in_opra": int(okq.sum()),
        "ask_exact_share": float((dask.abs() < 1e-6).mean()),
        "ask_within_0.05_share": float((dask.abs() <= 0.05 + 1e-9).mean()),
        "ask_within_0.10_share": float((dask.abs() <= 0.10 + 1e-9).mean()),
        "ask_median_abs_diff": float(dask.abs().median()),
        "ask_mean_diff_opra_minus_chain": float(dask.mean()),
        "bid_exact_share": float(
            ((ov["c_pair_o_bid"] - ov["spx_bid"])[okq].abs() < 1e-6).mean()
        ),
        "S_parity_minus_chain_median_abs_pts": float(
            (ov["S_par"] - ov["spx_S"]).abs().median()
        ),
        "S_parity_minus_chain_p95_abs_pts": float(
            (ov["S_par"] - ov["spx_S"]).abs().quantile(0.95)
        ),
        "S_parity_minus_chain_mean_pts": float((ov["S_par"] - ov["spx_S"]).mean()),
        "pair_same_share": float(
            ((ov["o_kc"] == ov["spx_kc"]) & (ov["o_kp"] == ov["spx_kp"])).mean()
        ),
        "book_lag_s_max": float(ob["lag_s"].max()),
    }
    gate["G4_opra_vs_chain"] = g4
    log("G4 OPRA vs chain 15:30 on the overlap: " + json.dumps(g4))

    # ---- 3. the eligible-day table
    closes = pd.read_csv(OUT / "spx_close_yf.csv", index_col=0, parse_dates=True)[
        "close"
    ]
    closes.index = pd.DatetimeIndex(closes.index).normalize()
    b_chain = pd.DataFrame(
        {
            "src": "chain",
            "S": chain["spx_S"],
            "Kc": chain["spx_kc"],
            "Kp": chain["spx_kp"],
            "bid": chain["spx_bid"],
            "ask": chain["spx_ask"],
        }
    )
    o26 = ob[ob.index >= OPRA_FIRST_ONLY]
    b_opra = pd.DataFrame(
        {
            "src": "opra",
            "S": o26["S_par"],
            "Kc": o26["o_kc"],
            "Kp": o26["o_kp"],
            "bid": o26["o_bid"],
            "ask": o26["o_ask"],
        }
    )
    bk = pd.concat([b_chain, b_opra]).sort_index()
    bk["month_end"] = [
        is_last_session_of_month(date(x.year, x.month, x.day)) for x in bk.index
    ]
    bk["S_close"] = closes.reindex(bk.index)
    bk = bk.join(mz[["yhat", "baseline", "rv_raw", "rv_hat"]], how="left")
    n0 = len(bk)
    drop = {
        "month_end": int(bk["month_end"].sum()),
        "no_rv_hat": int((~bk["month_end"] & ~np.isfinite(bk["rv_hat"])).sum()),
        "no_two_leg_ask": int(
            (~bk["month_end"] & np.isfinite(bk["rv_hat"]) & ~(bk["ask"] > 0)).sum()
        ),
        "no_close": int(
            (
                ~bk["month_end"]
                & np.isfinite(bk["rv_hat"])
                & (bk["ask"] > 0)
                & ~np.isfinite(bk["S_close"])
            ).sum()
        ),
    }
    d = bk[
        ~bk["month_end"]
        & np.isfinite(bk["rv_hat"])
        & (bk["ask"] > 0)
        & np.isfinite(bk["S_close"])
    ].copy()
    d = d[d.index <= pd.Timestamp("2026-09-23")]
    d["P_star"] = [
        package_price(np.sqrt(v), s, kc, kp)
        for v, s, kc, kp in zip(d["rv_hat"], d["S"], d["Kc"], d["Kp"])
    ]
    d["payoff"] = np.maximum(d["S_close"] - d["Kc"], 0.0) + np.maximum(
        d["Kp"] - d["S_close"], 0.0
    )
    d["R"] = d["payoff"] / d["ask"] - 1.0
    d["z"] = np.log(d["P_star"] / d["ask"])
    d["m"] = np.log(d["ask"] / d["S"])
    d["mid"] = (d["bid"] + d["ask"]) / 2
    d["iv_var"] = [
        invert_total_vol(s, kc, kp, mm) ** 2
        for s, kc, kp, mm in zip(d["S"], d["Kc"], d["Kp"], d["mid"])
    ]
    d["rv_over_iv"] = d["rv_raw"] / d["iv_var"]
    d.index.name = "date"
    gate["days"] = {
        "book_days": n0,
        "dropped": drop,
        "eligible": int(len(d)),
        "first": str(d.index.min().date()),
        "last": str(d.index.max().date()),
        "by_src": d["src"].value_counts().to_dict(),
        "iv_nan": int((~np.isfinite(d["iv_var"])).sum()),
    }
    log(
        f"eligible days: {len(d)} ({d.index.min().date()} .. {d.index.max().date()}; "
        f"{json.dumps(d['src'].value_counts().to_dict())}); dropped {json.dumps(drop)}; iv NaN {gate['days']['iv_nan']}"
    )
    d.reset_index().to_parquet(OUT / "days.parquet", index=False)
    d.reset_index().to_csv(OUT / "days.csv", index=False)

    # ---- G3 book + scoring against the deck's own card
    deck = pd.read_parquet(DECK)
    deck.index = pd.DatetimeIndex(deck.index).normalize()
    dj = deck.join(chain, how="left")
    me_deck = np.array(
        [is_last_session_of_month(date(x.year, x.month, x.day)) for x in deck.index]
    )
    nme = ~me_deck & dj["spx_ask"].notna().to_numpy()
    dk = dj[nme]
    ps_deck = np.array(
        [
            package_price(np.sqrt(v), s, kc, kp)
            for v, s, kc, kp in zip(
                dk["rv_hat"], dk["spx_S"], dk["spx_kc"], dk["spx_kp"]
            )
        ]
    )
    buy_deck = dk["spx_ask"].to_numpy(float) <= ps_deck
    R_deck = (
        (
            np.maximum(dk["S_close"] - dk["spx_kc"], 0)
            + np.maximum(dk["spx_kp"] - dk["S_close"], 0)
        )
        / dk["spx_ask"]
        - 1.0
    ).to_numpy(float)
    sd = stats(buy_deck, R_deck)
    g3 = {
        "deck_days": int(len(deck)),
        "non_month_end_with_book": int(nme.sum()),
        "pair_equal_deck": int(
            ((dk["spx_kc"] == dk["K_c"]) & (dk["spx_kp"] == dk["K_p"])).sum()
        ),
        "S_equal_deck": int((np.abs(dk["spx_S"] - dk["S"]) < 1e-3).sum()),
        "ask_equal_deck": int(
            (np.abs(dk["spx_ask"] - (dk["ask_c"] + dk["ask_p"])) < 1e-6).sum()
        ),
        "S_close_equal_yf": int(
            (np.abs(deck["S_close"] - closes.reindex(deck.index)) < 1e-6).sum()
        ),
        **{f"deck_card_{k}": v for k, v in sd.items()},
    }
    g3["passed"] = bool(
        g3["non_month_end_with_book"] == 814
        and round(sd["pnl_day"], 4) == DECK_PNL
        and round(sd["share"], 2) == DECK_SHARE
    )
    gate["G3_deck_card"] = g3
    log(
        "G3 the deck's card on this book: "
        + json.dumps(g3)
        + (" PASS" if g3["passed"] else " FAIL")
    )

    # ---- G5 study 85's 15:30 card
    p85 = pd.read_parquet(S85 / "per_day_pnl.parquet")
    p85 = p85[
        (p85.venue == "SPX")
        & (p85.iset == "idealized")
        & (p85.w == "15:30")
        & (p85.variant == "i")
        & (p85.scheme == "profile")
    ].copy()
    p85.index = pd.DatetimeIndex(pd.to_datetime(p85["date"])).normalize()
    b85 = pd.read_parquet(S85 / "books.parquet")
    b85 = b85[b85.venue == "SPX"].copy()
    b85.index = pd.DatetimeIndex(pd.to_datetime(b85["date"])).normalize()
    j5 = d.join(
        p85[["rv_hat", "buy", "pnl", "ask", "kc", "kp", "S"]],
        how="inner",
        rsuffix="_85",
    )
    j5 = j5.join(b85[["S_close"]].rename(columns={"S_close": "S_close_85"}), how="left")
    buy_mine = (j5["ask"] <= j5["P_star"]).to_numpy()
    pay85 = np.maximum(j5["S_close_85"] - j5["Kc"], 0) + np.maximum(
        j5["Kp"] - j5["S_close_85"], 0
    )
    pnl_mine_85close = np.where(buy_mine, pay85 / j5["ask"] - 1.0, 0.0)
    g5 = {
        "days_85": int(len(p85)),
        "days_joined": int(len(j5)),
        "pair_equal": int(((j5["Kc"] == j5["kc"]) & (j5["Kp"] == j5["kp"])).sum()),
        "ask_equal": int((np.abs(j5["ask"] - j5["ask_85"]) < 1e-6).sum()),
        "rv_hat_median_abs_rel": float(
            np.median(np.abs(j5["rv_hat"] / j5["rv_hat_85"] - 1))
        ),
        "rv_hat_max_abs_rel": float(np.max(np.abs(j5["rv_hat"] / j5["rv_hat_85"] - 1))),
        "decision_agree": int((buy_mine == j5["buy"].to_numpy(bool)).sum()),
        "buys_mine": int(buy_mine.sum()),
        "buys_85": int(j5["buy"].sum()),
        "pnl_day_85": float(j5["pnl"].mean()),
        "pnl_day_mine_at_85_settlement": float(pnl_mine_85close.mean()),
        "pnl_day_mine_at_official_close": float(
            np.where(buy_mine, j5["R"], 0.0).mean()
        ),
    }
    gate["G5_study85"] = g5
    log("G5 vs study 85's 15:30 SPX card: " + json.dumps(g5))

    # ---- 4. the market vs realized, by period
    per = pd.Series(
        np.where(d.index <= RESEARCH_END, "research", "post"), index=d.index
    )
    iv_rv = d["iv_var"] / d["rv_raw"]
    mkt = []
    for p_, sel in (
        ("research 2020-01..2024-04", per == "research"),
        ("post 2024-05..2026-09", per == "post"),
        ("2024-05..2025-12", (d.index >= POST_START) & (d.index <= "2025-12-31")),
        ("2026", d.index >= OPRA_FIRST_ONLY),
    ):
        x = d[sel]
        mkt.append(
            {
                "period": p_,
                "days": int(len(x)),
                "median_iv_over_rv": float(np.nanmedian(iv_rv[sel])),
                "median_ask_over_Pstar": float(np.median(x["ask"] / x["P_star"])),
                "share_ask_le_Pstar": float((x["ask"] <= x["P_star"]).mean()),
                "median_rv_hat_over_iv": float(np.nanmedian(x["rv_hat"] / x["iv_var"])),
                "median_rv_over_rv_hat": float(np.nanmedian(x["rv_raw"] / x["rv_hat"])),
                "mean_R_all_days": float(x["R"].mean()),
                "t_R_all_days": tstat(x["R"].to_numpy()),
            }
        )
    mkt = pd.DataFrame(mkt)
    mkt.to_csv(OUT / "market_vs_model.csv", index=False)

    # ---- 5. rules
    ref = research_reference(d)
    gate["F3_reference_full_research"] = float(
        ref[np.searchsorted(d.index, POST_START)]
    )
    dec = {N: run_rules(d, N, ref) for N in NS}
    first_live = {N: d.index[N] for N in NS}
    gate["first_live"] = {str(N): str(first_live[N].date()) for N in NS}
    log("first live day: " + json.dumps(gate["first_live"]))
    R = d["R"].to_numpy(float)
    f0 = (d["ask"] <= d["P_star"]).to_numpy()

    rows = []
    for N in NS:
        for r in RULES:
            x = dec[N][r]
            rows.append(
                pd.DataFrame(
                    {
                        "date": d.index,
                        "N": N,
                        "rule": r,
                        "live": x["live"],
                        "buy": x["buy"],
                        "param": x["param"],
                        "R": R,
                        "pnl": np.where(x["buy"], R, 0.0),
                    }
                )
            )
    rows.append(
        pd.DataFrame(
            {
                "date": d.index,
                "N": 0,
                "rule": "F0",
                "live": True,
                "buy": f0,
                "param": 1.0,
                "R": R,
                "pnl": np.where(f0, R, 0.0),
            }
        )
    )
    decisions = pd.concat(rows, ignore_index=True)
    decisions.to_parquet(OUT / "decisions.parquet", index=False)

    spans = {
        "research": lambda idx, N: (idx >= first_live[N]) & (idx <= RESEARCH_END),
        "post": lambda idx, N: idx >= POST_START,
        "post_2024-05..2025-12": lambda idx, N: (
            (idx >= POST_START) & (idx <= "2025-12-31")
        ),
        "post_2026": lambda idx, N: idx >= OPRA_FIRST_ONLY,
        "all_live": lambda idx, N: idx >= first_live[N],
        # where the research-span difference comes from: the first live months vs the rest
        "research_to_2021-12": lambda idx, N: (
            (idx >= first_live[N]) & (idx <= pd.Timestamp("2021-12-31"))
        ),
        "research_2022-01..2024-04": lambda idx, N: (
            (idx >= max(first_live[N], pd.Timestamp("2022-01-01")))
            & (idx <= RESEARCH_END)
        ),
    }
    boot_idx: dict[tuple, np.ndarray] = {}

    def summarize_on(
        sel: np.ndarray, key: tuple, label: dict, rule_buys: dict[str, np.ndarray]
    ) -> list[dict]:
        n = int(sel.sum())
        if n == 0:  # a span before this N's first live day
            return []
        if key not in boot_idx:
            boot_idx[key] = block_boot_idx(
                n, BLOCK, N_BOOT, np.random.default_rng(SEED + len(boot_idx))
            )
        idx = boot_idx[key]
        pnl0 = np.where(f0[sel], R[sel], 0.0)
        out = []
        for r, b in rule_buys.items():
            s = stats(b[sel], R[sel])
            pnl = np.where(b[sel], R[sel], 0.0)
            bm = pnl[idx].mean(1)
            dm = (pnl - pnl0)[idx].mean(1)
            s.update(label)
            s.update(
                {
                    "rule": r,
                    "first_day": str(d.index[sel].min().date()),
                    "last_day": str(d.index[sel].max().date()),
                    "agree_F0": float((b[sel] == f0[sel]).mean()),
                    "pnl_ci_lo": float(np.quantile(bm, 0.025)),
                    "pnl_ci_hi": float(np.quantile(bm, 0.975)),
                    "p_pnl_le0": float(((bm <= 0).sum() + 1) / (N_BOOT + 1)),
                    "d_vs_F0": float((pnl - pnl0).mean()),
                    "d_ci_lo": float(np.quantile(dm, 0.025)),
                    "d_ci_hi": float(np.quantile(dm, 0.975)),
                    "p_d_le0": float(((dm <= 0).sum() + 1) / (N_BOOT + 1))
                    if r != "F0"
                    else np.nan,
                }
            )
            out.append(s)
        return out

    summ = []
    for N in NS:
        buys = {"F0": f0, **{r: dec[N][r]["buy"] for r in RULES}}
        for sp_name, fn in spans.items():
            sel = np.asarray(fn(d.index, N))
            summ += summarize_on(
                sel,
                (N, sp_name),
                {"N": N, "span": sp_name, "days_basis": f"live at N={N}"},
                buys,
            )
    summ = pd.DataFrame(summ)
    # Holm over the three primary rules at N = 250
    for sp_name in ("post", "research"):
        m_ = (
            (summ["N"] == N_PRIMARY)
            & (summ["span"] == sp_name)
            & summ["rule"].isin(PRIMARY_RULES)
        )
        for col, hcol in (("p_d_le0", "p_d_holm"), ("p_pnl_le0", "p_pnl_holm")):
            h = holm(dict(zip(summ.loc[m_, "rule"], summ.loc[m_, col])))
            summ.loc[m_, hcol] = summ.loc[m_, "rule"].map(h)
    cols = [
        "N",
        "span",
        "rule",
        "first_day",
        "last_day",
        "days",
        "buys",
        "share",
        "mean_R_buy",
        "t_R_buy",
        "hit_buy",
        "pnl_day",
        "t_pnl",
        "pnl_ci_lo",
        "pnl_ci_hi",
        "p_pnl_le0",
        "p_pnl_holm",
        "agree_F0",
        "d_vs_F0",
        "d_ci_lo",
        "d_ci_hi",
        "p_d_le0",
        "p_d_holm",
    ]
    summ = summ[cols]
    summ.to_csv(OUT / "summary.csv", index=False)
    summ[
        ["N", "span", "rule", "d_vs_F0", "d_ci_lo", "d_ci_hi", "p_d_le0", "p_d_holm"]
    ].to_csv(OUT / "bootstrap.csv", index=False)

    # N-sensitivity on the days where N = 500 is live
    sens = []
    for sp_name in ("research", "post", "all_live"):
        sel = np.asarray(spans[sp_name](d.index, 500))
        for N in NS:
            buys = {"F0": f0, **{r: dec[N][r]["buy"] for r in RULES}}
            if N != NS[0]:
                buys.pop("F0")
            sens += summarize_on(
                sel,
                ("sens", sp_name),
                {"N": N, "span": sp_name, "days_basis": "live at N=500"},
                buys,
            )
    sens = pd.DataFrame(sens)
    sens[
        [c for c in cols if c in sens.columns and c not in ("p_pnl_holm", "p_d_holm")]
    ].to_csv(OUT / "n_sensitivity.csv", index=False)

    # by calendar year: F0 on every eligible day; the N = 250 rules on their live days
    by = []
    yrs = d.index.year
    for y in sorted(set(yrs)):
        sel = yrs == y
        by.append(
            {
                "year": y,
                "rule": "F0",
                "N": 0,
                "basis": "all eligible days",
                **stats(f0[sel], R[sel]),
            }
        )
        live = sel & dec[N_PRIMARY]["F1"]["live"]
        if live.any():
            by.append(
                {
                    "year": y,
                    "rule": "F0",
                    "N": N_PRIMARY,
                    "basis": f"live at N={N_PRIMARY}",
                    **stats(f0[live], R[live]),
                }
            )
            for r in RULES:
                by.append(
                    {
                        "year": y,
                        "rule": r,
                        "N": N_PRIMARY,
                        "basis": f"live at N={N_PRIMARY}",
                        **stats(dec[N_PRIMARY][r]["buy"][live], R[live]),
                    }
                )
    by = pd.DataFrame(by)
    by.to_csv(OUT / "by_year.csv", index=False)

    # F1's c path
    cp = []
    for N in NS:
        x = dec[N]["F1"]
        cp.append(
            pd.DataFrame(
                {
                    "date": d.index,
                    "N": N,
                    "c": x["param"],
                    "default_c1": x["default"],
                    "best_trailing_pnl": x["best_trailing_pnl"],
                    "F3_k": dec[N]["F3"]["param"],
                    "F2_pred": dec[N]["F2"]["param"],
                    "F2_slope": dec[N]["F2"]["slope"],
                }
            )[x["live"]]
        )
    cp = pd.concat(cp, ignore_index=True)
    cp.to_csv(OUT / "c_path.csv", index=False)
    cp["half"] = [
        f"{t.year}H{1 if t.month <= 6 else 2}" for t in pd.to_datetime(cp["date"])
    ]
    ch = (
        cp.groupby(["N", "half"])
        .agg(
            days=("c", "size"),
            c_median=("c", "median"),
            c_min=("c", "min"),
            c_max=("c", "max"),
            share_c_lt_1=("c", lambda s: float((s < 1).mean())),
            share_default=("default_c1", "mean"),
            F3_k_median=("F3_k", "median"),
            F2_slope_median=("F2_slope", "median"),
            F2_share_pred_gt0=("F2_pred", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    ch.to_csv(OUT / "c_path_halfyear.csv", index=False)

    # ---- 6. the deck's model (research years only), same rules
    dd_ = d[d.index <= RESEARCH_END].copy()
    dr = deck["rv_hat"].reindex(dd_.index)
    dd_ = dd_[np.isfinite(dr)]
    dd_["rv_hat"] = dr.reindex(dd_.index)
    dd_["P_star"] = [
        package_price(np.sqrt(v), s, kc, kp)
        for v, s, kc, kp in zip(dd_["rv_hat"], dd_["S"], dd_["Kc"], dd_["Kp"])
    ]
    dd_["z"] = np.log(dd_["P_star"] / dd_["ask"])
    ref_d = research_reference(dd_)
    dec_d = run_rules(dd_, N_PRIMARY, ref_d)
    Rd = dd_["R"].to_numpy(float)
    f0d = (dd_["ask"] <= dd_["P_star"]).to_numpy()
    live_d = dec_d["F1"]["live"]
    idx_d = block_boot_idx(
        int(live_d.sum()), BLOCK, N_BOOT, np.random.default_rng(SEED + 1000)
    )
    deck_rows = [
        {
            "model": "deck rv_hat",
            "span": "all research days",
            "rule": "F0",
            **stats(f0d, Rd),
        }
    ]
    pnl0 = np.where(f0d[live_d], Rd[live_d], 0.0)
    for r in ("F0",) + RULES:
        b = f0d if r == "F0" else dec_d[r]["buy"]
        s = stats(b[live_d], Rd[live_d])
        pnl = np.where(b[live_d], Rd[live_d], 0.0)
        dm = (pnl - pnl0)[idx_d].mean(1)
        s.update(
            {
                "model": "deck rv_hat",
                "span": f"live at N={N_PRIMARY}: {dd_.index[live_d].min().date()}..2024-04-30",
                "rule": r,
                "d_vs_F0": float((pnl - pnl0).mean()),
                "d_ci_lo": float(np.quantile(dm, 0.025)),
                "d_ci_hi": float(np.quantile(dm, 0.975)),
                "p_d_le0": float(((dm <= 0).sum() + 1) / (N_BOOT + 1))
                if r != "F0"
                else np.nan,
            }
        )
        deck_rows.append(s)
    deck_tab = pd.DataFrame(deck_rows)
    deck_tab.to_csv(OUT / "deck_model.csv", index=False)

    # ---- 7. write-up
    res_all = np.asarray(d.index <= RESEARCH_END)
    gate["F0_live_model_all_research_days"] = stats(f0[res_all], R[res_all])
    log(
        "\nF0 (live model) on ALL research days 2020-01..2024-04: "
        + json.dumps(gate["F0_live_model_all_research_days"])
    )
    gate["wall_s"] = round(time.time() - T0, 1)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    fmt = {
        c: "{:.4f}".format
        for c in (
            "share",
            "mean_R_buy",
            "hit_buy",
            "pnl_day",
            "pnl_ci_lo",
            "pnl_ci_hi",
            "p_pnl_le0",
            "p_pnl_holm",
            "agree_F0",
            "d_vs_F0",
            "d_ci_lo",
            "d_ci_hi",
            "p_d_le0",
            "p_d_holm",
        )
    }
    fmt.update({c: "{:.2f}".format for c in ("t_R_buy", "t_pnl")})
    log(
        "\nMARKET vs MODEL by period (median iv/rv = mid-implied package variance / ES 15:30-16:00 realized):"
    )
    log(mkt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    show = [
        "span",
        "rule",
        "first_day",
        "days",
        "buys",
        "share",
        "mean_R_buy",
        "pnl_day",
        "t_pnl",
        "pnl_ci_lo",
        "pnl_ci_hi",
        "p_pnl_holm",
        "agree_F0",
        "d_vs_F0",
        "d_ci_lo",
        "d_ci_hi",
        "p_d_le0",
        "p_d_holm",
    ]
    for N in NS:
        log(
            f"\nSUMMARY N = {N} (live from {first_live[N].date()}; every rule on the same days per span):"
        )
        log(summ[summ["N"] == N][show].to_string(index=False, formatters=fmt))
    log("\nN SENSITIVITY on the days where N = 500 is live:")
    log(
        sens[
            [
                "span",
                "N",
                "rule",
                "days",
                "buys",
                "share",
                "pnl_day",
                "t_pnl",
                "d_vs_F0",
                "d_ci_lo",
                "d_ci_hi",
                "p_d_le0",
            ]
        ].to_string(index=False, formatters=fmt)
    )
    log(
        f"\nBY YEAR (F0 on all eligible days; F0 and the N = {N_PRIMARY} rules on their live days):"
    )
    log(
        by[
            [
                "year",
                "rule",
                "basis",
                "days",
                "buys",
                "share",
                "mean_R_buy",
                "pnl_day",
                "t_pnl",
            ]
        ].to_string(index=False, formatters=fmt)
    )
    log("\nF1 c PATH and F3 k by half-year:")
    log(ch.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("\nDECK's rv_hat (research years only), the same rules at N = 250:")
    log(
        deck_tab[
            [
                "span",
                "rule",
                "days",
                "buys",
                "share",
                "mean_R_buy",
                "pnl_day",
                "t_pnl",
                "d_vs_F0",
                "d_ci_lo",
                "d_ci_hi",
                "p_d_le0",
            ]
        ].to_string(index=False, formatters=fmt)
    )
    log(f"\nwall {time.time() - T0:.0f}s")
    (OUT / "summary.txt").write_text("\n".join(_LOG) + "\n")
    return 0


if __name__ == "__main__":
    rc = main()
    (OUT / "run.log").write_text("\n".join(_LOG) + "\n")
    sys.exit(rc)
