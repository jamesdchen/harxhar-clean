"""The main economics table of Section 7, the closing-interval HAR forecast's own Sharpe ratio, the Sharpe ratio by
calendar year, and the break-even spreads (2026-10-25).

SCOPE
  Rows: the 12 pooled intraday forecasts of the final rerun (ridge, lasso, elastic net, LightGBM, XGBoost, random
  forest x {HAR benchmark set, full set}), the closing-interval HAR forecast (CI-HAR: the closing-interval OLS on the
  HAR benchmark set, master-table key sub_ols_baseline, the "HAR benchmark" of Section 6), and the unconditional short
  straddle.  Strategy: the sign strategy of Section 7, q_t = sign(sigma2_t - IV_t^2), ties short, held 15:30 to the
  close, at mid-quotes and at quoted bid/ask prices; every statistic on the 614 post-warmup days of T_opt.

REUSE (nothing of the strategy is re-implemented)
  writeup/make_rehaul_economics.py (imported as me): load() (the master table's per-day frame, the quote book, the
  short straddle), add_pooled_linear() and add_pooled_trees() (the final rerun's pooled linear rows and full-set trees
  at the 16:00 bar, through pooled_linear_13.py / pooled_trees_13.py), tree_16() (the HAR-set trees, traded exactly as
  add_pooled_trees trades the full-set ones; cross-checked against me.pooled_rank_series), sharpe() (the paper's D6
  Sharpe ratio), nw_t() / hac_p() (the LLSW t with fixed-b p, bandwidth 33), boot_p() (the paired circular block
  bootstrap of the D6 difference), the table cell helpers and the figure rcParams.  writeup/hac_inference.py:
  d6_sharpe() (the per-year terms SR_y, N_y, A_y).  master_table_gui.rows._crossed_return (bid/ask returns).
  2026-10-26 (author decision D19): sharpe() is now the paper's pooled ratio (sqrt(q) mean / sd, q = T / sum_y c_y;
  hac_inference.paper_sharpe), every p of a Delta SR is the Ledoit-Wolf (2008) test (me.sr_p), the paired block
  bootstrap of the pooled difference (me.boot_p) is a check in generated/sharpe_tests_d19_econ_main.csv, the break-even
  of the Sharpe ratio is that of the pooled ratio, D6 (me.sharpe_days_d6) is a robustness macro (rhemShDSix<Row>), and
  the by-year terms SR_y stay the descriptive by-year view.

GATE (the run stops if any fails)
  Every pooled Sharpe-ratio macro (\\rhrSh<Est>Pool<Set>, \\rhrShX..., \\rhrPoolSh{Har,Full}{Min,Max} and the bid/ask
  ranges), \\rhrOlsSharpe, \\rheShortSh, \\rheShortShX, \\rheShortT, \\rheShortP, \\rheSzDays, \\rheHACS, \\rheDSix{Year,N}*
  and \\cmSvCostStraddle are reproduced as printed; Panels A, B and F of Table tab:rh_app_hac_robust (mean
  differential, Delta SR, fixed-b t and p, bootstrap p) are reproduced as printed; the closing-interval OLS row of
  Table E1 (Delta SR vs the short straddle and its bootstrap p) likewise.

BREAK-EVEN SPREAD
  The quoted half-spread of the package h_t = (ask_t - bid_t)/2 around the mid P_t = (bid_t + ask_t)/2 is scaled by a
  factor k: a purchase pays P_t + k h_t and a sale receives P_t - k h_t (k = 0: mid-quotes; k = 1: quoted bid/ask,
  reproducing the bid/ask returns exactly).  k*_mean and k*_SR are the smallest k >= 0 at which the post-warmup mean
  daily return and the D6 Sharpe ratio reach zero (bisection on a grid bracket).  They are reported as multiples of the
  quoted half-spread and in percent of the premium, k* times the median of h_t / P_t over the 614 post-warmup days.
  A strategy whose statistic is not positive at mid has no break-even ("--").

X8: THE HAR + RM + FOMC FORECASTS (2026-10-29; Q-2026-10-29-21, the author: "X8 yes")
  PROPOSED AFTER THE AUTHOR'S 27 OCT REMARK; NOT PRE-REGISTERED. Rule: results/econ_rmfomc_2026-10-29/RULE.txt,
  committed (6acaee1) before any of these statistics was computed.  The sign strategy above on each class's forecast
  from HAR + RM + FOMC (the ablation arm add_rm_fomc, results/ablations_2014start_w3000/RULE.txt addendum F: the HAR
  benchmark set, the futures' realized moments and the FOMC group, 132 columns of the FOMC-patched final design; the
  OLS, ridge, lasso, elastic net, LightGBM, XGBoost and the forest).  The 16:00 row of each day, plain = pred_adj^2 x B
  (B and the bar's realized variance from the final linear run's 16:00 rows, as pooled_linear_13.at16), traded by
  sign_row (the lines of add_ols_har) and scored by row_stats and pair, unchanged.  Comparisons at mid (bid/ask
  alongside): each class's full set minus its HAR + RM + FOMC (the question: do the other series add trading value?),
  and HAR + RM + FOMC minus the class's HAR set, the OLS-HAR and the short straddle.  Gates: the input md5s of the rule;
  B of the linear and tree runs equal; every option day has a finite positive forecast whose bar's realized variance is
  the OLS row's; the 16:00 QLIKE on the ablations' 2,579 evaluation days reproduces scores_fomcfix/metrics.csv (c16)
  to 1e-10.  Outputs: the \\rhem<Est>RmFomc... macros appended to the numbers file (its other lines unchanged), the
  table below, the new pairs appended to sharpe_tests_d19_econ_main.csv, and results/econ_rmfomc_2026-10-29/stats.csv
  and pairs.csv.  The table note carries no disclosure label (author, 28 Oct); the appendix prose does.

OUTPUTS (new files; nothing else is written)
  writeup/generated/rehaul_econ_main_numbers.tex   \\rhem... macros (macro-only)
  writeup/generated/rehaul_econ_main_table.tex     Table tab:rh_econ_main
  writeup/generated/rehaul_econ_byyear.pdf         the Sharpe ratio by calendar year
  writeup/generated/rehaul_econ_byyear_fig.tex     its float, fig:rh_econ_byyear
  writeup/generated/rehaul_econ_rmfomc_table.tex   2026-10-29 (X8): Table tab:rh_econ_rmfomc
  results/econ_rmfomc_2026-10-29/stats.csv, pairs.csv   2026-10-29 (X8): every statistic of the rule, unrounded

Run (Git Bash, repository root):  python writeup/make_rehaul_econ_main.py [--preview DIR]
REHAUL_FINAL_RUN is set to 1 here (the paper's pooled forecasts are the final rerun).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path

os.environ["REHAUL_FINAL_RUN"] = "1"  # before final_run is imported (through make_rehaul_economics)

WRITEUP = Path(__file__).resolve().parent
if str(WRITEUP) not in sys.path:
    sys.path.insert(0, str(WRITEUP))

import final_run as fr  # noqa: E402
import hac_inference as hi  # noqa: E402
import make_rehaul_economics as me  # noqa: E402  (load, pooled rows, Sharpe ratio, tests, cell helpers, rcParams)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

GEN = fr.GEN
HEADER = "% AUTO-GENERATED by writeup/make_rehaul_econ_main.py -- do not edit; re-run the script.\n"
NUMBERS = GEN / "rehaul_econ_main_numbers.tex"
TABLE = GEN / "rehaul_econ_main_table.tex"
FIG_PDF = GEN / "rehaul_econ_byyear.pdf"
FIG_TEX = GEN / "rehaul_econ_byyear_fig.tex"
PREVIEW: Path | None = None

EST = ("ridge", "lasso", "enet", "lgbm", "xgb", "rf")
LINEAR = ("ridge", "lasso", "enet")
TREES = ("lgbm", "xgb", "rf")
EST_MAC = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Enet", "lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
EST_NAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
SETS = ("all_features", "baseline")
SET_MAC = {"all_features": "Full", "baseline": "Har"}
CI = me.OLS  # sub_ols_baseline: the closing-interval OLS on the HAR benchmark set (CI-HAR)
# 2026-10-26 (integration plan D2): the primary benchmark is the RTH13 pooled OLS on the HAR set (Table 1's OLS row);
# its strategy is a row of its own (add_ols_har), the reference of the main table's Delta SR column. The CI-HAR leaves
# the main table (D3: closing-interval models in the final appendix section); its macros are still written.
OLSH = "poolpt13_ols_baseline"
SHORT = me.SHORT
YEAR_WORD = ("One", "Two", "Three", "Four", "Five", "Six")
WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
# 2026-10-29 (X8, Q-2026-10-29-21; the author: "X8 yes"): proposed after the author's 27 Oct remark; not
# pre-registered. The sign strategy on the HAR + RM + FOMC forecasts (rule results/econ_rmfomc_2026-10-29/RULE.txt,
# commit 6acaee1, written before any computation; the module docstring, X8).
RMF_LABEL = "proposed after the author's 27 Oct remark; not pre-registered"
RMF_ABL = me.ROOT / "results" / "ablations_2014start_w3000"
RMF_FC = {
    "linear": RMF_ABL / "forecasts" / "linear_wide_rmfomc.parquet",
    "trees": RMF_ABL / "forecasts" / "trees_wide_rmfomc.parquet",
}
RMF_MD5 = {"linear": "aced527588e68a01dc07595adbf7d644", "trees": "e1741bd4aa99a229206e4e38afc9d47a"}  # RULE.txt 2
RMF_METRICS = RMF_ABL / "scores_fomcfix" / "metrics.csv"
RMF_OUT = me.ROOT / "results" / "econ_rmfomc_2026-10-29"
RMF_TABLE = GEN / "rehaul_econ_rmfomc_table.tex"
RMF_CLS = ("ols", *EST)  # the OLS on HAR + RM + FOMC (no full-set twin) and the six classes of the main table
RMF_COL = {e: f"{'rf_skx' if e == 'rf' else e}|add_rm_fomc" for e in RMF_CLS}  # columns of the two wide files
RMF_N_EVAL = 2579  # the ablations' evaluation days (experiments/ablate_score.py N_DAYS; metrics.csv c16 n_days)
X8_SIG_LEVEL = 0.05  # 2026-10-29 (paper-prose V2): the level of the X8 Ledoit-Wolf counts, printed as \rhemSigPct


def key(est: str, pset: str) -> str:
    """Series key of a pooled forecast (me's keys; the HAR-set trees are added by add_har_trees)."""
    return f"poolpt13_{est}_{pset}" if est in LINEAR else f"pooltree_{est}_{pset}"


def rkey(est: str) -> str:
    """2026-10-29 (X8): series key of a class's HAR + RM + FOMC strategy (added by add_rmfomc)."""
    return f"rmfomc_{est}"


def tag(k: str) -> str:
    """Macro stem of a row: <Est><Set>, CiHar or Short; 2026-10-29 (X8): <Est>RmFomc (Est = Ols, ...)."""
    if k.startswith("rmfomc_"):
        return {"ols": "Ols", **EST_MAC}[k.removeprefix("rmfomc_")] + "RmFomc"
    if k == CI:
        return "CiHar"
    if k == OLSH:
        return "OlsHar"
    if k == SHORT:
        return "Short"
    for e in EST:
        for s in SETS:
            if key(e, s) == k:
                return EST_MAC[e] + SET_MAC[s]
    raise KeyError(k)


# ============================================================================================ data
def add_har_trees(series: dict) -> None:
    """The three pooled trees on the HAR benchmark set, traded exactly as me.add_pooled_trees trades the full-set ones
    (sign of plain - iv_var on the OLS row's days, the quote book's straddle, bid/ask by gui._crossed_return)."""
    daily = pd.read_parquet(me.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"])
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    b = series[CI]
    days = b.index
    ref = daily[daily["key"] == CI].set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    assert np.isfinite(iv).all() and np.isfinite(rv).all()
    for est in TREES:
        p = me.tree_16(est, "baseline", days, rv)
        q = np.where(p > iv, 1.0, -1.0)
        k = key(est, "baseline")
        assert k not in series, k
        series[k] = pd.DataFrame(
            {
                "q": q,
                "mid": q * b["R"].to_numpy(float),
                "x": gui._crossed_return(q, b["ex"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy()),
                "s": p - iv,
                "R": b["R"].to_numpy(float),
                "ex": b["ex"].to_numpy(float),
                "bid": b["bid"].to_numpy(float),
                "ask": b["ask"].to_numpy(float),
            },
            index=days,
        )
    # cross-check: me.pooled_rank_series builds the same HAR-set trees (mid only) for the rank correlations
    rk = me.pooled_rank_series(series)
    for est in TREES:
        k = key(est, "baseline")
        assert np.array_equal(rk[k]["mid"].to_numpy(float), series[k]["mid"].to_numpy(float)), k


def add_ols_har(series: dict) -> None:
    """2026-10-26 (D2): the RTH13 pooled OLS on the HAR set as a strategy row, traded exactly as
    me.add_pooled_linear trades the pooled linear rows (sign of plain - iv_var on the OLS row's days, the quote book's
    straddle, bid/ask by gui._crossed_return)."""
    daily = pd.read_parquet(me.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"])
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    b = series[CI]
    days = b.index
    ref = daily[daily["key"] == CI].set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    assert np.isfinite(iv).all() and np.isfinite(rv).all()
    p = me.lin_16("ols", "baseline", days, rv)
    assert OLSH not in series
    series[OLSH] = sign_row(b, p, iv, rv)  # 2026-10-29 (X8): the frame's lines moved to sign_row, unchanged


def sign_row(b: pd.DataFrame, p: np.ndarray, iv: np.ndarray, rv: np.ndarray) -> pd.DataFrame:
    """The sign strategy of one 16:00 forecast p on the option days of b (the CI row's frame: the quote book's
    straddle), as add_ols_har trades the OLS-HAR: q = sign(p - iv_var), ties short; mid q R; bid/ask by
    gui._crossed_return; QLIKE against the bar's realized variance rv.  2026-10-29 (X8): taken out of add_ols_har
    unchanged, so the HAR + RM + FOMC rows (add_rmfomc) are traded by the same lines."""
    q = np.where(p > iv, 1.0, -1.0)
    ratio = rv / p
    return pd.DataFrame(
        {
            "q": q,
            "mid": q * b["R"].to_numpy(float),
            "x": gui._crossed_return(q, b["ex"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy()),
            "s": p - iv,
            "qlike": ratio - np.log(ratio) - 1.0,
            "R": b["R"].to_numpy(float),
            "ex": b["ex"].to_numpy(float),
            "bid": b["bid"].to_numpy(float),
            "ask": b["ask"].to_numpy(float),
        },
        index=b.index,
    )


# ============================================================================================ X8: HAR + RM + FOMC
def rmfomc_16() -> dict[str, pd.DataFrame]:
    """2026-10-29 (X8): the 16:00 rows of each class's HAR + RM + FOMC forecast (the ablation arm add_rm_fomc), one row
    a trading day: plain = pred_adj^2 x B and rv, the bar's realized variance.  B and rv are the final linear run's at
    the same bar (pooled_linear_13's frame, the rows pl13.at16 reads); B is asserted equal to the final tree run's.
    The 16:00 row is the forecast of the 15:30-16:00 variance made at 15:30, the one the closing trade uses."""
    lin = me.pl13._frame()
    g = lin[(lin["model"] == "ridge") & (lin["predictor_set"] == "all_features")]
    tg = pd.DatetimeIndex(pd.to_datetime(g["date"]))
    at = (tg.hour == 16) & (tg.minute == 0)
    ref = pd.DataFrame({"B": g["B"].to_numpy(float)[at], "rv": g["rv_actual"].to_numpy(float)[at]}, index=tg[at])
    tr = me.pt13._frame()
    h = tr[(tr["model"] == "lgbm") & (tr["predictor_set"] == "all_features")]
    b_tree = pd.Series(h["B"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(h["date"])))
    out: dict[str, pd.DataFrame] = {}
    for kind, path in RMF_FC.items():
        assert hashlib.md5(path.read_bytes()).hexdigest() == RMF_MD5[kind], (path, "md5 is not the rule's")
        w = pd.read_parquet(path)
        d = pd.DatetimeIndex(w["date"])
        at16 = (d.hour == 16) & (d.minute == 0)
        ts = d[at16]
        bb = ref["B"].reindex(ts).to_numpy(float)
        assert np.isfinite(bb).all() and np.array_equal(bb, b_tree.reindex(ts).to_numpy(float)), (kind, "B")
        for e in RMF_CLS:
            if RMF_COL[e] in w.columns:
                v = w[RMF_COL[e]].to_numpy(float)[at16]
                out[e] = pd.DataFrame(
                    {"plain": v**2 * bb, "rv": ref["rv"].reindex(ts).to_numpy(float)},
                    index=pd.DatetimeIndex(ts.normalize()).as_unit("ns"),
                )
    assert set(out) == set(RMF_CLS), sorted(out)
    for e, a in out.items():
        assert not a.index.duplicated().any(), e
    return out


def rmfomc_qlike_gate(f16: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """2026-10-29 (X8), RULE.txt 6 (c): on the ablations' 2,579 evaluation days (16:00 rows with a positive realized
    variance, early closes dropped, 2014-01-02 .. 2024-04-30) the mean 16:00 QLIKE of each class's HAR + RM + FOMC
    forecast reproduces the ablation scorer's (scores_fomcfix/metrics.csv, arm add_rm_fomc, span c16) to 1e-10."""
    m = pd.read_csv(RMF_METRICS)
    m = m[(m["arm"] == "add_rm_fomc") & (m["span"] == "c16")].set_index("cls")
    early = pd.DatetimeIndex(pd.to_datetime(list(me.asl.EARLY_CLOSE_DATES))).normalize()
    a = f16["ols"]
    rv = a["rv"].to_numpy(float)
    keep = (rv > 0) & np.isfinite(rv) & ~a.index.isin(early) & (a.index >= fr.FIRST_DAY) & (a.index <= fr.LAST_DAY)
    days = a.index[keep]
    assert len(days) == RMF_N_EVAL, len(days)
    out = {}
    for e in RMF_CLS:
        g = f16[e].reindex(days)
        r = (g["rv"] / g["plain"]).to_numpy(float)
        ql = float(np.mean(r - np.log(r) - 1.0))
        assert int(m.loc[e, "n_days"]) == RMF_N_EVAL, (e, m.loc[e, "n_days"])
        want = float(m.loc[e, "qlike"])
        assert abs(ql / want - 1.0) <= 1e-10, (e, ql, want)
        out[e] = (ql, want)
    return out


def add_rmfomc(series: dict) -> dict[str, tuple[float, float]]:
    """2026-10-29 (X8): each class's HAR + RM + FOMC forecast as a strategy row rmfomc_<est>, traded by sign_row on the
    OLS row's days, implied variance and realized variance (as add_ols_har), with the gates of me.lin_16: a finite
    positive forecast on every option day, and the bar's realized variance equal to the OLS row's rv_unclipped.
    Returns the QLIKE gate's values (rmfomc_qlike_gate)."""
    daily = pd.read_parquet(me.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"])
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    b = series[CI]
    days = b.index
    ref = daily[daily["key"] == CI].set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    assert np.isfinite(iv).all() and np.isfinite(rv).all()
    f16 = rmfomc_16()
    qgate = rmfomc_qlike_gate(f16)
    for e in RMF_CLS:
        a = f16[e]
        a.index = a.index.astype(days.dtype)
        a = a.reindex(days)
        p = a["plain"].to_numpy(float)
        assert np.isfinite(p).all() and (p > 0).all(), (e, int((~np.isfinite(p)).sum()))
        assert np.allclose(a["rv"].to_numpy(float), rv, rtol=1e-12, atol=0.0), e
        k = rkey(e)
        assert k not in series, k
        series[k] = sign_row(b, p, iv, rv)
    return qgate


def har_linear_check() -> dict:
    """The three pooled linear forecasts on the HAR benchmark set coincide (penalties near zero): the largest absolute
    and relative difference of their 16:00 forecasts among themselves and against the pooled OLS, over every 16:00
    row of the run."""
    pl = {e: me.pl13.at16(e, "baseline")["plain"] for e in (*LINEAR, "ols")}
    idx = pl["ols"].index
    for e in LINEAR:
        assert pl[e].index.equals(idx), e
    vals = np.vstack([pl[e].to_numpy(float) for e in (*LINEAR, "ols")])
    fa = me.pl13._frame()
    dif = vals.max(axis=0) - vals.min(axis=0)
    return {
        "max_abs": float(dif.max()),
        "max_rel": float((dif / vals.min(axis=0)).max()),
        "n_rows": int(len(idx)),
        "alpha": {
            e: float(fa.loc[(fa["model"] == e) & (fa["predictor_set"] == "baseline"), "alpha"].iloc[0]) for e in LINEAR
        },
    }


# ============================================================================================ statistics
def r_k(g: pd.DataFrame, k: float) -> np.ndarray:
    """Daily return of position q at half-spread multiple k: buy at P + k h, sell at P - k h (P mid, h half-spread)."""
    q = g["q"].to_numpy(float)
    ex, bid, ask = g["ex"].to_numpy(float), g["bid"].to_numpy(float), g["ask"].to_numpy(float)
    p_mid, h = (bid + ask) / 2.0, (ask - bid) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        long_ = np.abs(q) * (ex / (p_mid + k * h) - 1.0)
        short_ = np.abs(q) * (1.0 - ex / (p_mid - k * h))
    return np.where(q > 0, long_, np.where(q < 0, short_, 0.0))


def break_even(g: pd.DataFrame, stat) -> float:
    """Smallest k >= 0 with stat(r_k[post-warmup]) = 0 (NaN when stat <= 0 at mid; inf when no root below the largest
    k at which every sale price stays positive)."""
    pw = slice(me.WARMUP, None)
    q = g["q"].to_numpy(float)[pw]
    bid, ask = g["bid"].to_numpy(float)[pw], g["ask"].to_numpy(float)[pw]
    ratio = ((bid + ask) / 2.0) / ((ask - bid) / 2.0)
    k_max = 0.999 * float(ratio[q < 0].min()) if (q < 0).any() else 50.0

    def f(k: float) -> float:
        return float(stat(r_k(g, k)[pw]))

    if not f(0.0) > 0:
        return float("nan")
    grid = np.linspace(0.0, k_max, int(math.ceil(k_max / 0.01)) + 1)
    lo = 0.0
    for k in grid[1:]:
        if f(k) <= 0:
            hi_ = float(k)
            break
        lo = float(k)
    else:
        return float("inf")
    for _ in range(80):  # bisection to machine precision
        mid = 0.5 * (lo + hi_)
        if f(mid) > 0:
            lo = mid
        else:
            hi_ = mid
    return 0.5 * (lo + hi_)


def row_stats(series: dict, k: str) -> dict:
    g, sh = series[k], series[SHORT]
    mid, x = g["mid"].to_numpy(float), g["x"].to_numpy(float)
    pw = me.pw_days()
    # the k-scaled returns reproduce the bid/ask returns at k = 1 and the mid returns at k = 0 (R is stored in single
    # precision: |ex / P - 1 - R| <= 5e-7)
    assert np.allclose(r_k(g, 1.0), x, rtol=0, atol=1e-12), k
    assert np.allclose(r_k(g, 0.0), mid, rtol=0, atol=5e-7), k
    out: dict = {
        "sh_mid": me.sharpe(mid),
        "sh_x": me.sharpe(x),
        "mean_mid": float(mid[me.WARMUP :].mean()),
        "mean_x": float(x[me.WARMUP :].mean()),
        "be_mean": break_even(g, lambda r: r.mean()),
        # 2026-10-26 (D19): the break-even of the paper's pooled ratio. Was: hi.d6_sharpe(r, dates=pw)["sharpe"]
        "be_sh": break_even(g, lambda r: hi.paper_sharpe(r, dates=pw)["sharpe"]),
        # the two alternative ratios of Table tab:rh_econ_sharpe_defs (D1: per-year, sqrt(N_y), N_y-weighted; D2: the
        # pooled daily returns, sqrt(252)), at mid
        "sh_d1": me.sharpe_days_d1(mid[me.WARMUP :]),
        "sh_d2": me.sharpe_days_252(mid[me.WARMUP :]),
        "sh_d6": me.sharpe_days_d6(mid[me.WARMUP :]),  # 2026-10-26 (D19): D6, the robustness column
    }
    by = hi.d6_sharpe(mid[me.WARMUP :], dates=pw)["by_year"]
    # 2026-10-26 (D19): the SR_y average to D6, no longer to the paper's ratio. Was: ... - out["sh_mid"]) < 1e-12, k
    assert abs(np.mean([v[2] for v in by.values()]) - out["sh_d6"]) < 1e-12, k
    out["by_year"] = {y: v[2] for y, v in by.items()}
    out["n_y"] = {y: v[0] for y, v in by.items()}
    out["a_y"] = {y: v[1] for y, v in by.items()}
    if k != SHORT:
        for col, sfx in (("mid", ""), ("x", "_x")):
            d = g[col].to_numpy(float) - sh[col].to_numpy(float)
            t = me.nw_t(d)
            out[f"dsh_short{sfx}"] = me.sharpe(g[col]) - me.sharpe(sh[col])
            # 2026-10-26 (D19): the Ledoit-Wolf p (me.sr_p); the bootstrap p is the check pb_*. Was: me.boot_p(...)
            out[f"p_short{sfx}"] = me.sr_p(series, k, SHORT, col)
            out[f"pb_short{sfx}"] = me.boot_p(series, k, SHORT, col)
            out[f"t_short{sfx}"] = t
            out[f"pt_short{sfx}"] = me.hac_p(t)
            out[f"dmean_short{sfx}"] = float(d[me.WARMUP :].mean())
    return out


def pair(series: dict, a: str, b: str) -> dict:
    """Delta SR (a minus b) at mid and bid/ask with the Ledoit-Wolf p (D19; the bootstrap p pb_* is a check), and the
    fixed-b t of the mean mid differential."""
    out = {}
    for col, sfx in (("mid", ""), ("x", "_x")):
        out[f"dsh{sfx}"] = me.sharpe(series[a][col]) - me.sharpe(series[b][col])
        # 2026-10-26 (D19): was out[f"p{sfx}"] = me.boot_p(series, a, b, col)
        out[f"p{sfx}"] = me.sr_p(series, a, b, col)
        out[f"pb{sfx}"] = me.boot_p(series, a, b, col)
    d = series[a]["mid"].to_numpy(float) - series[b]["mid"].to_numpy(float)
    out["t"] = me.nw_t(d)
    out["pt"] = me.hac_p(out["t"])
    out["dmean"] = float(d[me.WARMUP :].mean())
    return out


# ============================================================================================ gate
def _macros(path: Path) -> dict[str, str]:
    """Macro name -> value of a generated numbers file (the last definition of each name; balanced braces)."""
    out: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"\\(?:providecommand|newcommand|renewcommand)\{\\([A-Za-z]+)\}\{", text):
        i, depth = m.end(), 1
        j = i
        while depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out[m.group(1)] = text[i : j - 1]
    return out


def _norm(v: str) -> str:
    return v.replace("\\ensuremath{-}", "-").replace("$", "").strip()


def f2(x: float, dp: int = 2) -> str:
    return me._fmt(x, dp)


def gate(series: dict, st: dict, pairs: dict, spread: dict) -> None:
    rr = _macros(GEN / "rehaul_results_numbers.tex")
    ee = _macros(GEN / "rehaul_econ_numbers.tex")
    cm = _macros(GEN / "close_main_numbers.tex")
    checks: list[tuple[str, str, str]] = []
    for e in EST:
        for s in SETS:
            k = key(e, s)
            m = f"{EST_MAC[e]}Pool{SET_MAC[s]}"
            checks += [
                (f"rhrSh{m}", _norm(rr[f"rhrSh{m}"]), f2(st[k]["sh_mid"])),
                (f"rhrShX{m}", _norm(rr[f"rhrShX{m}"]), f2(st[k]["sh_x"])),
            ]
    for s in SETS:
        v = [st[key(e, s)]["sh_mid"] for e in EST]
        vx = [st[key(e, s)]["sh_x"] for e in EST]
        S = SET_MAC[s]
        checks += [
            (f"rhrPoolSh{S}Min", _norm(rr[f"rhrPoolSh{S}Min"]), f2(min(v))),
            (f"rhrPoolSh{S}Max", _norm(rr[f"rhrPoolSh{S}Max"]), f2(max(v))),
            (f"rhrPoolShX{S}Min", _norm(rr[f"rhrPoolShX{S}Min"]), f2(min(vx))),
            (f"rhrPoolShX{S}Max", _norm(rr[f"rhrPoolShX{S}Max"]), f2(max(vx))),
        ]
    checks += [
        ("rhrOlsSharpe", _norm(rr["rhrOlsSharpe"]), f2(st[CI]["sh_mid"])),
        ("rheShortSh", _norm(ee["rheShortSh"]), f2(st[SHORT]["sh_mid"])),
        ("rheShortShX", _norm(ee["rheShortShX"]), f2(st[SHORT]["sh_x"])),
        ("rheShortT", _norm(ee["rheShortT"]), f2(me.nw_t(series[SHORT]["mid"]))),
        ("rheShortP", _norm(ee["rheShortP"]), me.p_digits(me.hac_p(me.nw_t(series[SHORT]["mid"])))),
        ("rheSzDays", _norm(ee["rheSzDays"]), str(len(me.pw_days()))),
        ("rheHACS", _norm(ee["rheHACS"]), str(me.HAC_S)),
        ("cmSvCostStraddle", _norm(cm["cmSvCostStraddle"]), f"{100 * spread['med_all']:.1f}"),
    ]
    years = sorted(st[SHORT]["n_y"])
    assert _norm(ee["rheNYears"]) == str(len(years)), ee["rheNYears"]
    for i, y in enumerate(years):
        w = YEAR_WORD[i]
        checks += [
            (f"rheDSixYear{w}", _norm(ee[f"rheDSixYear{w}"]), str(y)),
            (f"rheDSixN{w}", _norm(ee[f"rheDSixN{w}"]), str(st[SHORT]["n_y"][y])),
            (f"rheDSixA{w}", _norm(ee[f"rheDSixA{w}"]), f"{st[SHORT]['a_y'][y]:.1f}"),
        ]
    bad = [c for c in checks if c[1] != c[2]]
    for c in checks:
        print(f"  gate {c[0]:24s} printed {c[1]:>8s}  recomputed {c[2]:>8s}")
    assert not bad, f"GATE FAILED: {bad}"

    # Table tab:rh_app_hac_robust, Panels A (vs short), B (Full vs HAR) and F (the short straddle's mean)
    def pcell(p: float) -> str:
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    rob = (GEN / "rehaul_app_hac_robust.tex").read_text(encoding="utf-8")
    panel, n_rob = "", 0
    for line in rob.splitlines():
        mp = re.search(r"Panel ([A-F]):", line)
        if mp:
            panel = mp.group(1)
            continue
        if panel not in ("A", "B", "F") or "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [_norm(x) for x in line.rstrip()[:-2].split("&")]
        label = cells[0]
        name = {v: k_ for k_, v in EST_NAME.items()}
        if panel == "A":
            nm, sname = label.split(", ")
            k = key(name[nm], "baseline" if sname == "HAR" else "all_features")
            r = st[k]
            want = [
                f2(100 * r["dmean_short"]),
                f2(r["dsh_short"]),
                f2(r["t_short"]),
                pcell(r["pt_short"]),
                pcell(r["p_short"]),
            ]
        elif panel == "B":
            pr = pairs[("har", name[label])]
            want = [f2(100 * pr["dmean"]), f2(pr["dsh"]), f2(pr["t"]), pcell(pr["pt"]), pcell(pr["p"])]
        else:
            t = me.nw_t(series[SHORT]["mid"])
            want = [f2(100 * st[SHORT]["mean_mid"]), "--", f2(t), pcell(me.hac_p(t)), "--"]
        got = [cells[1], cells[2], cells[7], cells[8], cells[9]]
        assert got == want, (panel, label, got, want)
        n_rob += 1
    assert n_rob == 12 + 6 + 1, n_rob
    print(f"  gate tab:rh_app_hac_robust Panels A, B, F: {n_rob} rows reproduced")

    # Table E1 (generated, parked): the closing-interval OLS row's Delta SR vs the short straddle and its bootstrap p
    e1 = (GEN / "rehaul_econ_e1_control.tex").read_text(encoding="utf-8").splitlines()
    i = next(j for j, ln in enumerate(e1) if ln.startswith("OLS, HAR benchmark"))
    dsh = re.findall(r"\$(-?[0-9.]+)\^", e1[i])[-1]
    pb = re.findall(r"\[([0-9.]+)\]", e1[i + 1])[-1]
    want_e1 = (f2(st[CI]["dsh_short"]), f"{st[CI]['p_short']:.3f}")
    assert (dsh, pb) == want_e1, ((dsh, pb), want_e1)
    print(f"  gate tab:rh_econ_control CI-HAR row: Delta SR {dsh} [{pb}] reproduced")


# ============================================================================================ LaTeX
def num(x: float, dp: int = 2, plus: bool = False) -> str:
    """A macro value: \\ensuremath{-} for a minus sign (as rehaul_econ_numbers.tex)."""
    if x is None or not np.isfinite(x):
        return "--"
    s = me._fmt(x, dp)
    if s.startswith("-"):
        return "\\ensuremath{-}" + s[1:]
    return ("+" if plus and float(s) > 0 else "") + s


def pval(p: float) -> str:
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def be_val(k: float, dp: int = 2) -> str:
    if np.isnan(k):
        return "--"
    if np.isinf(k):
        return "\\ensuremath{\\infty}"
    return f"{k:.{dp}f}"


def sci(x: float) -> str:
    """x as m \\times 10^{e}, one significant decimal."""
    e = int(math.floor(math.log10(x)))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"\\ensuremath{{{m:.1f}\\times 10^{{{e}}}}}"


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", fr.rel(path))


def numbers_tex(st: dict, pairs: dict, spread: dict, harlin: dict, yr: dict) -> str:
    lines = [
        HEADER.rstrip("\n"),
        "% \\rhem... macros for Section 7's main economics table (tab:rh_econ_main), the by-year figure",
        "% (fig:rh_econ_byyear) and the break-even spreads. Sample: the 614 post-warmup days of T_opt; the sign strategy.",
        "% Row stems: <Est><Set> (Est = Ridge, Lasso, Enet, Lgbm, Xgb, Rf; Set = Full, Har: the pooled intraday",
        "% forecasts of the final rerun), CiHar (the closing-interval OLS on the HAR benchmark set), Short (q = -1).",
    ]

    def mac(name: str, val: str, note: str) -> None:
        assert re.fullmatch(r"rhem[A-Za-z]+", name), name
        lines.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{val}}}  % {note}")

    order = [key(e, "all_features") for e in EST] + [key(e, "baseline") for e in EST] + [CI, SHORT]
    lines.append(
        "% ---- per row: Sharpe ratio (D19: pooled, sqrt(q)), mean daily return (% of premium), at mid and at quoted bid/ask"
    )
    for k in order:
        r, t = st[k], tag(k)
        mac(f"rhemSh{t}", num(r["sh_mid"]), f"{k}: SR, mid")
        mac(f"rhemShX{t}", num(r["sh_x"]), "... at quoted bid/ask")
        mac(f"rhemMean{t}", num(100 * r["mean_mid"], 1), "... mean daily return, %, mid")
        mac(f"rhemMeanX{t}", num(100 * r["mean_x"], 1), "... mean daily return, %, bid/ask")
        mac(f"rhemShDOne{t}", num(r["sh_d1"]), "... SR mid, per-year sqrt(N_y), N_y-weighted (tab:rh_econ_sharpe_defs)")
        mac(f"rhemShDTwo{t}", num(r["sh_d2"]), "... SR mid, pooled daily returns, sqrt(252)")
        mac(f"rhemShDSix{t}", num(r["sh_d6"]), "... SR mid, D6: equal-year mean of SR_y (robustness, D19)")
    lines.append(
        "% ---- against the short straddle: Delta SR with its Ledoit-Wolf p; fixed-b t (and p) of the mean differential"
    )
    for k in order[:-1]:
        r, t = st[k], tag(k)
        mac(f"rhemDSh{t}VsShort", num(r["dsh_short"]), f"{k} minus the short straddle: Delta SR, mid")
        mac(f"rhemPDSh{t}VsShort", pval(r["p_short"]), "... its Ledoit-Wolf p (D19)")
        mac(f"rhemDShX{t}VsShort", num(r["dsh_short_x"]), "... Delta SR at bid/ask")
        mac(f"rhemPDShX{t}VsShort", pval(r["p_short_x"]), "... its Ledoit-Wolf p (D19)")
        mac(f"rhemT{t}VsShort", num(r["t_short"]), "... fixed-b t of the mean mid return differential (S = 33)")
        mac(f"rhemPT{t}VsShort", pval(r["pt_short"]), "... its fixed-b p")
        mac(f"rhemTX{t}VsShort", num(r["t_short_x"]), "... fixed-b t of the mean bid/ask return differential")
        mac(f"rhemPTX{t}VsShort", pval(r["pt_short_x"]), "... its fixed-b p")
    lines.append("% ---- Full minus its own HAR row (same model class); every pooled row minus CI-HAR")
    for e in EST:
        t = EST_MAC[e]
        p = pairs[("har", e)]
        mac(f"rhemDSh{t}FullVsHar", num(p["dsh"]), f"{e}: Full minus HAR, Delta SR, mid")
        mac(f"rhemPDSh{t}FullVsHar", pval(p["p"]), "... its Ledoit-Wolf p (D19)")
        mac(f"rhemDShX{t}FullVsHar", num(p["dsh_x"]), "... Delta SR at bid/ask")
        mac(f"rhemPDShX{t}FullVsHar", pval(p["p_x"]), "... its Ledoit-Wolf p (D19)")
    for k in order[:12]:
        t = tag(k)
        p = pairs[("ci", k)]
        mac(f"rhemDSh{t}VsCiHar", num(p["dsh"]), f"{k} minus CI-HAR: Delta SR, mid")
        mac(f"rhemPDSh{t}VsCiHar", pval(p["p"]), "... its Ledoit-Wolf p (D19)")
        mac(f"rhemDShX{t}VsCiHar", num(p["dsh_x"]), "... Delta SR at bid/ask")
        mac(f"rhemPDShX{t}VsCiHar", pval(p["p_x"]), "... its Ledoit-Wolf p (D19)")
        mac(f"rhemT{t}VsCiHar", num(p["t"]), "... fixed-b t of the mean mid return differential")
        mac(f"rhemPT{t}VsCiHar", pval(p["pt"]), "... its fixed-b p")

    lines.append(
        "% ---- 2026-10-26 (D2): every Full forecast against the RTH13 pooled OLS-HAR strategy (primary benchmark)"
    )
    r_o = st[OLSH]
    mac("rhemShOlsHar", num(r_o["sh_mid"]), f"{OLSH}: SR, mid (the OLS-HAR strategy)")
    mac("rhemShXOlsHar", num(r_o["sh_x"]), "... at quoted bid/ask")
    mac("rhemMeanOlsHar", num(100 * r_o["mean_mid"], 1), "... mean daily return, %, mid")
    mac("rhemDShOlsHarVsShort", num(r_o["dsh_short"]), "... minus the short straddle: Delta SR, mid")
    mac("rhemPDShOlsHarVsShort", pval(r_o["p_short"]), "... its Ledoit-Wolf p (D19)")
    for e in EST:
        t = EST_MAC[e]
        p_ = pairs[("ols", e)]
        mac(f"rhemDSh{t}FullVsOls", num(p_["dsh"]), f"{e}: Full minus OLS-HAR, Delta SR, mid")
        mac(f"rhemPDSh{t}FullVsOls", pval(p_["p"]), "... its Ledoit-Wolf p (D19)")
        mac(f"rhemDShX{t}FullVsOls", num(p_["dsh_x"]), "... Delta SR at bid/ask")
        mac(f"rhemPDShX{t}FullVsOls", pval(p_["p_x"]), "... its Ledoit-Wolf p (D19)")
    d_o = [pairs[("ols", e)]["dsh"] for e in EST]
    p_o = [pairs[("ols", e)]["p"] for e in EST]
    n_sig_o = sum(pairs[("ols", e)]["p"] < 0.05 and pairs[("ols", e)]["dsh"] > 0 for e in EST)
    n_sig_ox = sum(pairs[("ols", e)]["p_x"] < 0.05 and pairs[("ols", e)]["dsh_x"] > 0 for e in EST)
    mac("rhemFullVsOlsMin", num(min(d_o)), "Full minus OLS-HAR, Delta SR mid: min over the six")
    mac("rhemFullVsOlsMax", num(max(d_o)), "... max")
    mac("rhemFullVsOlsMed", num(float(np.median(d_o))), "... median")
    mac("rhemFullVsOlsPMin", pval(min(p_o)), "... smallest Ledoit-Wolf p")
    mac("rhemFullVsOlsPMax", pval(max(p_o)), "... largest Ledoit-Wolf p")
    mac("rhemFullVsOlsPos", str(sum(d > 0 for d in d_o)), "... Delta SR > 0, mid")
    mac("rhemFullVsOlsSig", str(n_sig_o), "... significantly positive (Ledoit-Wolf, 5 %), mid")
    mac("rhemFullVsOlsSigWord", WORDS[n_sig_o], "... in words")
    mac("rhemFullVsOlsSigX", str(n_sig_ox), "... significantly positive at bid/ask")
    mac("rhemFullVsOlsSigXWord", WORDS[n_sig_ox], "... in words")
    mac("rhemFullVsOlsHarSig", str(n_sig_o), "= rhemFullVsOlsSig (FRONT's name)")

    lines.append(
        "% ---- counts over the six Full forecasts (5 %; Ledoit-Wolf for Delta SR, fixed-b for mean differentials)"
    )
    full = [key(e, "all_features") for e in EST]
    har4 = [key("ridge", "baseline")] + [key(e, "baseline") for e in TREES]

    def cnt(name: str, n: int, note: str, word: bool = True) -> None:
        mac(name, str(n), note)
        if word:
            mac(name + "Word", WORDS[n], "... in words")

    cnt("rhemFullVsShortSig", sum(st[k]["p_short"] < 0.05 for k in full), "Full forecasts: Delta SR vs short sig., mid")
    cnt("rhemFullVsShortSigX", sum(st[k]["p_short_x"] < 0.05 for k in full), "... at bid/ask")
    cnt(
        "rhemFullTVsShortSig",
        sum(st[k]["pt_short"] < 0.05 and st[k]["t_short"] > 0 for k in full),
        "Full forecasts: mean mid gain over short sig. (fixed-b)",
    )
    cnt(
        "rhemFullTXVsShortSig",
        sum(st[k]["pt_short_x"] < 0.05 and st[k]["t_short_x"] > 0 for k in full),
        "... mean bid/ask gain",
    )
    cnt("rhemFullVsHarPos", sum(pairs[("har", e)]["dsh"] > 0 for e in EST), "Full minus own HAR: Delta SR > 0, mid")
    cnt("rhemFullVsHarSig", sum(pairs[("har", e)]["p"] < 0.05 for e in EST), "... significant (Ledoit-Wolf)")
    cnt("rhemFullVsCiHarPos", sum(pairs[("ci", k)]["dsh"] > 0 for k in full), "Full minus CI-HAR: Delta SR > 0, mid")
    cnt("rhemFullVsCiHarSig", sum(pairs[("ci", k)]["p"] < 0.05 for k in full), "... significant (Ledoit-Wolf)")
    cnt("rhemFullVsCiHarPosX", sum(pairs[("ci", k)]["dsh_x"] > 0 for k in full), "... Delta SR > 0 at bid/ask")
    cnt("rhemFullVsCiHarSigX", sum(pairs[("ci", k)]["p_x"] < 0.05 for k in full), "... significant at bid/ask")
    cnt(
        "rhemFullAboveCiHarDOne",
        sum(st[k]["sh_d1"] > st[CI]["sh_d1"] for k in full),
        "Full forecasts whose D1 SR (mid) exceeds CI-HAR's",
    )
    cnt(
        "rhemFullAboveCiHarDTwo",
        sum(st[k]["sh_d2"] > st[CI]["sh_d2"] for k in full),
        "Full forecasts whose D2 SR (mid) exceeds CI-HAR's",
    )
    cnt(
        "rhemHarVsShortSig",
        sum(st[k]["p_short"] < 0.05 for k in har4),
        "distinct HAR forecasts: Delta SR vs short sig.",
    )
    cnt("rhemHarVsCiHarNeg", sum(pairs[("ci", k)]["dsh"] < 0 for k in har4), "distinct HAR forecasts below CI-HAR, mid")
    cnt("rhemHarVsCiHarSig", sum(pairs[("ci", k)]["p"] < 0.05 for k in har4), "... significant (Ledoit-Wolf)")
    d_ci = [pairs[("ci", k)]["dsh"] for k in full]
    mac("rhemFullVsCiHarMin", num(min(d_ci)), "Full minus CI-HAR, Delta SR mid: min over the six")
    mac("rhemFullVsCiHarMax", num(max(d_ci)), "... max")
    mac("rhemFullVsCiHarMed", num(float(np.median(d_ci))), "... median")
    p_ci = [pairs[("ci", k)]["p"] for k in full]
    mac("rhemFullVsCiHarPMin", pval(min(p_ci)), "... smallest Ledoit-Wolf p")
    mac("rhemFullVsCiHarPMax", pval(max(p_ci)), "... largest Ledoit-Wolf p")
    for nm, ks in (("Full", full), ("Har", har4)):
        v = [st[k]["sh_mid"] for k in ks]
        vx = [st[k]["sh_x"] for k in ks]
        mac(f"rhem{nm}ShMed", num(float(np.median(v))), f"median SR mid over the {len(ks)} distinct {nm} forecasts")
        mac(f"rhem{nm}ShXMed", num(float(np.median(vx))), "... at bid/ask")

    lines.append("% ---- the pooled linear HAR forecasts coincide (penalties near zero; they are the pooled OLS)")
    mac(
        "rhemHarLinMaxAbsDiff",
        sci(harlin["max_abs"]),
        "largest |difference| of the 16:00 forecasts, ridge/lasso/enet/OLS",
    )
    mac("rhemHarLinMaxRelDiff", sci(harlin["max_rel"]), "... largest relative difference")
    mac("rhemHarLinNRows", f"{harlin['n_rows']:,}".replace(",", "{,}"), "... over these 16:00 rows of the run")
    mac("rhemHarDistinct", "4", "distinct HAR-set forecasts among the six pooled HAR rows")
    mac("rhemHarDistinctWord", "four", "... in words")

    lines.append("% ---- break-even half-spread: multiple k* of the quoted half-spread at which the post-warmup mean")
    lines.append(
        "% ---- (BeMean) or the paper's Sharpe ratio (BeSh; D19: pooled) reaches zero; ...Pct: k* x the median "
        "half-spread, % of premium"
    )
    mac(
        "rhemHalfSpreadMed",
        f"{100 * spread['med_pw']:.1f}",
        "median quoted half-spread, % of premium, 614 post-warmup days",
    )
    mac("rhemHalfSpreadMedAll", f"{100 * spread['med_all']:.1f}", "... on all 866 option days (= \\cmSvCostStraddle)")
    mac("rhemHalfSpreadMeanPw", f"{100 * spread['mean_pw']:.1f}", "... mean, 614 post-warmup days")
    for k in order:
        r, t = st[k], tag(k)
        mac(f"rhemBeMean{t}", be_val(r["be_mean"]), f"{k}: k* at which the mean daily return is zero")
        mac(f"rhemBeMeanPct{t}", be_val(100 * r["be_mean"] * spread["med_pw"], 1), "... in % of premium")
        mac(f"rhemBeSh{t}", be_val(r["be_sh"]), f"{k}: k* at which SR is zero")
        mac(f"rhemBeShPct{t}", be_val(100 * r["be_sh"] * spread["med_pw"], 1), "... in % of premium")
    for nm, ks in (("Full", full), ("Har", har4)):
        for stat in ("mean", "sh"):
            be_all = np.array([st[k][f"be_{stat}"] for k in ks])
            v_be = be_all[np.isfinite(be_all)]
            S = "Mean" if stat == "mean" else "Sh"
            mac(f"rhemBe{S}{nm}Min", be_val(v_be.min()), f"k* ({stat}): min over the distinct {nm} forecasts with one")
            mac(f"rhemBe{S}{nm}Max", be_val(v_be.max()), "... max")
            mac(f"rhemBe{S}{nm}Med", be_val(float(np.median(v_be))), "... median")
            mac(f"rhemBe{S}{nm}MinPct", be_val(100 * v_be.min() * spread["med_pw"], 1), "... min, % of premium")
            mac(f"rhemBe{S}{nm}MaxPct", be_val(100 * v_be.max() * spread["med_pw"], 1), "... max, % of premium")

    lines.append(
        "% ---- by calendar year (mid): SR_y, each year annualized by sqrt(A_y) (descriptive; their mean is D6); years "
        "One.. as \\rheDSixYearOne.."
    )
    years = yr["years"]
    for i, y in enumerate(years):
        w = YEAR_WORD[i]
        mac(f"rhemYear{w}", str(y), f"calendar year {w} of the post-warmup days")
        mac(f"rhemNy{w}", str(yr["n_y"][y]), "... N_y")
        mac(f"rhemByYearFullMed{w}", num(yr["full_med"][i]), f"{y}: median SR_y of the six Full forecasts")
        mac(f"rhemByYearFullMin{w}", num(yr["full_min"][i]), "... min")
        mac(f"rhemByYearFullMax{w}", num(yr["full_max"][i]), "... max")
        mac(f"rhemByYearHarMed{w}", num(yr["har_med"][i]), f"{y}: median SR_y of the four distinct HAR forecasts")
        mac(f"rhemByYearHarMin{w}", num(yr["har_min"][i]), "... min")
        mac(f"rhemByYearHarMax{w}", num(yr["har_max"][i]), "... max")
        mac(f"rhemByYearCiHar{w}", num(yr["ci"][i]), f"{y}: SR_y of CI-HAR")
        mac(f"rhemByYearShort{w}", num(yr["short"][i]), f"{y}: SR_y of the short straddle")
    cnt("rhemNYears", len(years), "calendar years of the post-warmup days")
    cnt("rhemYearsFullAllPos", yr["full_all_pos"], "years in which all six Full forecasts have SR_y > 0")
    cnt("rhemYearsFullAllAboveShort", yr["full_all_above_short"], "years in which all six Full SR_y exceed the short's")
    cnt("rhemYearsFullMedAboveCiHar", yr["full_med_above_ci"], "years in which the Full median SR_y exceeds CI-HAR's")
    cnt(
        "rhemYearsFullMedAboveHarMed", yr["full_med_above_har"], "years in which the Full median exceeds the HAR median"
    )
    cnt("rhemYearsFullMedAboveShort", yr["full_med_above_short"], "years in which the Full median exceeds the short")
    cnt("rhemYearsCiHarAboveAllFull", yr["ci_above_all_full"], "years in which CI-HAR's SR_y exceeds every Full one")
    cnt("rhemYearsCiHarPos", yr["ci_pos"], "years in which CI-HAR has SR_y > 0")
    n_part = sum(yr["n_y"][y] for y in yr["partial"])
    mac("rhemNPartialDays", str(n_part), "post-warmup days in the partial first and last calendar years")
    mac("rhemPartialDaysPct", f"{100 * n_part / sum(yr['n_y'].values()):.0f}", "... in % of the post-warmup days")
    cnt("rhemYearsShortNeg", yr["short_neg"], "years in which the short straddle has SR_y < 0")
    return "\n".join(lines) + "\n"


def table_tex(st: dict, pairs: dict) -> str:
    ncol = 9
    body = ""
    for pi, (pset, title) in enumerate(
        (
            # 2026-10-28 (author: prune "pooled"). Was: "Panel A: Pooled intraday models, full predictor set\\tnote{a}",
            # "Panel B: Pooled intraday models, HAR benchmark set\\tnote{a}"
            ("all_features", "Panel A: Intraday models, full predictor set\\tnote{a}"),
            ("baseline", "Panel B: Intraday models, HAR benchmark set\\tnote{a}"),
        )
    ):
        body += me.panel_head(title, ncol, first=pi == 0)
        # 2026-10-26 (D2): Panel B's three coinciding linear HAR rows are one row, the pooled OLS (note b); the last
        # column is Delta SR against the OLS-HAR strategy (was: against the closing-interval HAR, note c)
        ests = EST if pset == "all_features" else ("ols", *TREES)
        for e in ests:
            k = OLSH if e == "ols" else key(e, pset)
            r = st[k]
            vs_h = pairs.get(("har", e)) if pset == "all_features" else None
            vs_o = pairs.get(("ols", e)) if pset == "all_features" else None
            label = "OLS\\tnote{b}" if e == "ols" else EST_NAME[e]
            top = [
                me.cell(r["sh_mid"], 2, starred=False),
                me.cell(r["sh_x"], 2, starred=False),
                me.cell(100 * r["mean_mid"], 1, starred=False),
                me.cell(100 * r["mean_x"], 1, starred=False),
                me.cell(r["dsh_short"], 2, me.n_stars_p(r["p_short"])),
                me.cell(r["t_short"], 2, me.n_stars_p(r["pt_short"])),
                me.cell(vs_h["dsh"], 2, me.n_stars_p(vs_h["p"])) if vs_h else "--$^{\\phantom{***}}$",
                me.cell(vs_o["dsh"], 2, me.n_stars_p(vs_o["p"])) if vs_o else "--$^{\\phantom{***}}$",
            ]
            under = [
                "",
                "",
                "",
                "",
                me.pbrack(r["p_short"]),
                "",
                me.pbrack(vs_h["p"]) if vs_h else "",
                me.pbrack(vs_o["p"]) if vs_o else "",
            ]
            body += me.rows2(label, top, under)
    # 2026-10-26 (D2/D3): parked -- the closing-interval HAR row of Panel C ("Closing-interval HAR\\tnote{c}"); it is
    # in the closing-interval appendix section now
    body += me.panel_head("Panel C: Benchmark", ncol)
    r = st[SHORT]
    body += me.rows2(
        me.SHORT_LABEL,
        [
            me.cell(r["sh_mid"], 2, starred=False),
            me.cell(r["sh_x"], 2, starred=False),
            me.cell(100 * r["mean_mid"], 1, starred=False),
            me.cell(100 * r["mean_x"], 1, starred=False),
            "--$^{\\phantom{***}}$",
            "--$^{\\phantom{***}}$",
            "--$^{\\phantom{***}}$",
            "--$^{\\phantom{***}}$",
        ],
        [""] * 8,
    )
    return (
        HEADER
        + "% Table tab:rh_econ_main: the sign strategy of the pooled intraday forecasts (the HAR-set linear rows as the\n"
        "% OLS) and the short straddle, 614 post-warmup days. Macros: \\rhem... (rehaul_econ_main_numbers.tex).\n"
        "\\ifdefined\\rhemShOlsHar\\else\\input{generated/rehaul_econ_main_numbers}\\fi\n"
        "\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{threeparttable}\n"
        "\\caption{Economic value of the forecasts: the sign strategy}\n"
        "\\label{tab:rh_econ_main}\n"
        "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{$\\SR$} & \\multicolumn{2}{c}{Mean return (\\%)} & "
        "\\multicolumn{2}{c}{vs.\\ short straddle} & \\multicolumn{2}{c}{$\\Delta\\SR$ vs.} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n"
        " & Mid & Bid/ask & Mid & Bid/ask & $\\Delta\\SR$ & $t(\\bar\\Delta)$ & HAR set & OLS-HAR \\\\\n"
        "\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}[flushleft]\n\\footnotesize\n"
        "\\item \\textit{Notes:} The sign strategy $q_t = \\sign(\\hat\\sigma^2_t - \\mathrm{IV}^2_t)$ on the "
        "\\rheSzDays{} post-warmup days of $\\mathcal T_{\\mathrm{opt}}$; the short straddle holds $q_t \\equiv -1$. "
        "$\\SR$ is the annualized Sharpe ratio of Section~\\ref{sec:rh_econ_measures}, at mid-quotes and at quoted "
        "bid/ask prices; mean return is the mean daily return in percent of the premium. Every comparison is at "
        "mid-quotes. $\\Delta\\SR$ is the difference of the Sharpe ratios: against the short straddle; against the "
        "same model class on the HAR benchmark set and against the OLS on the HAR benchmark set (full-set rows "
        # 2026-10-26 (D19): the Ledoit-Wolf test. Was: "only). Under each $\\Delta\\SR$, in brackets, the $p$-value of
        # the paired block-bootstrap test of equal Sharpe ratios (block length \\rheBootBlock{}, \\rheBootN{}
        # resamples); ..."
        "only). Under each $\\Delta\\SR$, in brackets, the $p$-value of the \\citet{LedoitWolf2008} test of equal "
        "Sharpe ratios (block length \\rheLWBlock{}, \\rheLWBoot{} resamples); $t(\\bar\\Delta)$ is the "
        "$t$-statistic of the mean daily return differential against the short straddle (Newey--West, bandwidth "
        "$S = \\rheHACS$, fixed-$b$ critical values; Section~\\ref{sec:rh_methods_tests}). "
        + me.STAR_NOTE
        # 2026-10-28 (author: prune "pooled"). Was: "of each day (the pooled intraday scheme); each trades ..."
        + "\n\\item[a] Panels A and B: models estimated on the \\rhePoolNBars{} regular-hours 30-minute intervals "
        "of each day (the intraday scheme); each trades its 15:30 forecast of the 15:30--16:00 ET variance.\n"
        "\\item[b] The OLS on the HAR benchmark set, the primary benchmark (Table~\\ref{tab:rh_oos}); the ridge, "
        "lasso and elastic net forecasts on that set coincide with it (largest relative difference "
        "\\rhemHarLinMaxRelDiff{} over \\rhemHarLinNRows{} forecasts; the positions agree on every day), so Panel B "
        "holds \\rhemHarDistinctWord{} distinct forecasts.\n"
        "\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n"
    )


# ============================================================================================ X8 statistics and LaTeX
def x8_stats(series: dict) -> tuple[dict, dict]:
    """2026-10-29 (X8), RULE.txt 5: row_stats of each HAR + RM + FOMC row (SR, mean, against the short straddle) and
    the pairs (pair, unchanged): ("full", e) the class's full set minus its HAR + RM + FOMC (the primary comparison);
    ("har", e) HAR + RM + FOMC minus the class's HAR set; ("ols", e) minus the OLS-HAR. The OLS's own HAR set is the
    OLS-HAR, so its ("har", "ols") and ("ols", "ols") are one pair."""
    xs = {rkey(e): row_stats(series, rkey(e)) for e in RMF_CLS}
    xp: dict = {}
    for e in EST:
        xp[("full", e)] = pair(series, key(e, "all_features"), rkey(e))
        xp[("har", e)] = pair(series, rkey(e), key(e, "baseline"))
        xp[("ols", e)] = pair(series, rkey(e), OLSH)
    xp[("har", "ols")] = xp[("ols", "ols")] = pair(series, rkey("ols"), OLSH)
    return xs, xp


def _mac_line(name: str, val: str, note: str) -> str:
    """One macro line in the numbers file's form (numbers_tex.mac)."""
    assert re.fullmatch(r"rhem[A-Za-z]+", name), name
    return f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{val}}}  % {note}"


def x8_numbers(xs: dict, xp: dict) -> str:
    """2026-10-29 (X8): the \\rhem<Est>RmFomc... macros, appended to the numbers file after its existing lines."""
    lines = [
        "% ---- 2026-10-29 (X8, Q-2026-10-29-21): the sign strategy on the HAR + RM + FOMC forecasts (ablation arm",
        f"% add_rm_fomc); {RMF_LABEL}. Rule: results/econ_rmfomc_2026-10-29/RULE.txt.",
        "% Row stems <Est>RmFomc (Est = Ols, Ridge, Lasso, Enet, Lgbm, Xgb, Rf). Delta SR at mid (X: at quoted bid/ask)",
        "% with its Ledoit-Wolf p (D19); T: fixed-b t of the mean daily mid-return differential (S = 33).",
    ]

    def mac(name: str, val: str, note: str) -> None:
        lines.append(_mac_line(name, val, note))

    def cnt(name: str, n: int, note: str) -> None:
        mac(name, str(n), note)
        mac(name + "Word", WORDS[n], "... in words")

    # 2026-10-29 (paper-prose V2): the significance level the X8 counts use, as a macro, so the text types no digit
    mac("rhemSigPct", f"{100 * X8_SIG_LEVEL:g}", "significance level of the X8 Ledoit-Wolf counts, %")
    for e in RMF_CLS:
        k = rkey(e)
        r, t = xs[k], tag(k)
        mac(f"rhemSh{t}", num(r["sh_mid"]), f"{k}: SR, mid")
        mac(f"rhemShX{t}", num(r["sh_x"]), "... at quoted bid/ask")
        mac(f"rhemMean{t}", num(100 * r["mean_mid"], 1), "... mean daily return, %, mid")
        mac(f"rhemMeanX{t}", num(100 * r["mean_x"], 1), "... mean daily return, %, bid/ask")
        mac(f"rhemDSh{t}VsShort", num(r["dsh_short"]), "... minus the short straddle: Delta SR, mid")
        mac(f"rhemPDSh{t}VsShort", pval(r["p_short"]), "... its Ledoit-Wolf p")
        mac(f"rhemDShX{t}VsShort", num(r["dsh_short_x"]), "... Delta SR at bid/ask")
        mac(f"rhemPDShX{t}VsShort", pval(r["p_short_x"]), "... its Ledoit-Wolf p")
        mac(f"rhemT{t}VsShort", num(r["t_short"]), "... fixed-b t of the mean mid return differential")
        mac(f"rhemPT{t}VsShort", pval(r["pt_short"]), "... its fixed-b p")
        for nm, what in (("har", "Har"), ("ols", "Ols")):
            p_ = xp[(nm, e)]
            ref = "the class's HAR set" if nm == "har" else "the OLS-HAR"
            mac(f"rhemDSh{t}Vs{what}", num(p_["dsh"]), f"... minus {ref}: Delta SR, mid")
            mac(f"rhemPDSh{t}Vs{what}", pval(p_["p"]), "... its Ledoit-Wolf p")
            mac(f"rhemDShX{t}Vs{what}", num(p_["dsh_x"]), "... Delta SR at bid/ask")
            mac(f"rhemPDShX{t}Vs{what}", pval(p_["p_x"]), "... its Ledoit-Wolf p")
    lines.append("% ---- the primary comparison: each class's full set minus its HAR + RM + FOMC (RULE.txt 5 (P))")
    for e in EST:
        p_, t = xp[("full", e)], EST_MAC[e]
        mac(f"rhemDSh{t}FullVsRmFomc", num(p_["dsh"]), f"{e}: full set minus HAR + RM + FOMC, Delta SR, mid")
        mac(f"rhemPDSh{t}FullVsRmFomc", pval(p_["p"]), "... its Ledoit-Wolf p")
        mac(f"rhemDShX{t}FullVsRmFomc", num(p_["dsh_x"]), "... Delta SR at bid/ask")
        mac(f"rhemPDShX{t}FullVsRmFomc", pval(p_["p_x"]), "... its Ledoit-Wolf p")
        mac(f"rhemT{t}FullVsRmFomc", num(p_["t"]), "... fixed-b t of the mean mid return differential")
        mac(f"rhemPT{t}FullVsRmFomc", pval(p_["pt"]), "... its fixed-b p")
    lines.append(
        "% ---- over the six classes (Ols excluded); 5 % level; Sig = Delta SR > 0 with p < 0.05, SigNeg = < 0"
    )
    cnt("rhemRmFomcNCls", len(EST), "model classes with a full-set and a HAR + RM + FOMC forecast")
    groups = {
        "FullVsRmFomc": [xp[("full", e)] for e in EST],
        "RmFomcVsHar": [xp[("har", e)] for e in EST],
        "RmFomcVsOls": [xp[("ols", e)] for e in EST],
        "RmFomcVsShort": [
            {
                "dsh": xs[rkey(e)]["dsh_short"],
                "p": xs[rkey(e)]["p_short"],
                "dsh_x": xs[rkey(e)]["dsh_short_x"],
                "p_x": xs[rkey(e)]["p_short_x"],
            }
            for e in EST
        ],
    }
    for g, ps in groups.items():
        d, p = [x["dsh"] for x in ps], [x["p"] for x in ps]
        mac(f"rhem{g}Min", num(min(d)), f"{g}: Delta SR mid, min over the six")
        mac(f"rhem{g}Max", num(max(d)), "... max")
        mac(f"rhem{g}PMin", pval(min(p)), "... smallest Ledoit-Wolf p")
        mac(f"rhem{g}PMax", pval(max(p)), "... largest Ledoit-Wolf p")
        cnt(f"rhem{g}Pos", sum(x > 0 for x in d), "... Delta SR > 0, mid")
        cnt(f"rhem{g}Neg", sum(x < 0 for x in d), "... Delta SR < 0, mid")
        cnt(f"rhem{g}Sig", sum(x["dsh"] > 0 and x["p"] < X8_SIG_LEVEL for x in ps), "... significantly positive, mid")
        cnt(
            f"rhem{g}SigNeg", sum(x["dsh"] < 0 and x["p"] < X8_SIG_LEVEL for x in ps), "... significantly negative, mid"
        )
        cnt(f"rhem{g}PosX", sum(x["dsh_x"] > 0 for x in ps), "... Delta SR > 0 at bid/ask")
        cnt(
            f"rhem{g}SigX",
            sum(x["dsh_x"] > 0 and x["p_x"] < X8_SIG_LEVEL for x in ps),
            "... significantly positive, bid/ask",
        )
    sh = [xs[rkey(e)]["sh_mid"] for e in EST]
    shx = [xs[rkey(e)]["sh_x"] for e in EST]
    mac("rhemRmFomcShMin", num(min(sh)), "SR mid of the six HAR + RM + FOMC forecasts: min")
    mac("rhemRmFomcShMax", num(max(sh)), "... max")
    mac("rhemRmFomcShXMin", num(min(shx)), "SR at bid/ask: min")
    mac("rhemRmFomcShXMax", num(max(shx)), "... max")
    return "\n".join(lines) + "\n"


def x8_table_tex(xs: dict, xp: dict) -> str:
    """2026-10-29 (X8): Table tab:rh_econ_rmfomc. Rows: the OLS and the six classes on HAR + RM + FOMC; Delta SR at
    mid with the Ledoit-Wolf p in brackets, as Table tab:rh_econ_main. The note carries no disclosure label (author,
    28 Oct); the appendix prose discloses that the analysis was proposed after the author's 27 Oct remark."""
    ncol = 7
    dash = "--$^{\\phantom{***}}$"
    body = ""
    for e in RMF_CLS:
        r = xs[rkey(e)]
        cmp_ = [xp.get(("full", e)), None, xp[("har", e)], xp[("ols", e)]]
        cmp_[1] = {"dsh": r["dsh_short"], "p": r["p_short"]}
        top = [me.cell(r["sh_mid"], 2, starred=False), me.cell(r["sh_x"], 2, starred=False)]
        under = ["", ""]
        for c in cmp_:
            top.append(me.cell(c["dsh"], 2, me.n_stars_p(c["p"])) if c else dash)
            under.append(me.pbrack(c["p"]) if c else "")
        body += me.rows2("OLS\\tnote{a}" if e == "ols" else EST_NAME[e], top, under)
    return (
        HEADER + f"% Table tab:rh_econ_rmfomc (2026-10-29, X8; {RMF_LABEL}; rule\n"
        "% results/econ_rmfomc_2026-10-29/RULE.txt): the sign strategy on the HAR + RM + FOMC forecasts, 614 post-warmup\n"
        "% days. Macros: \\rhem<Est>RmFomc... (rehaul_econ_main_numbers.tex).\n"
        "\\ifdefined\\rhemShOlsRmFomc\\else\\input{generated/rehaul_econ_main_numbers}\\fi\n"
        "\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{threeparttable}\n"
        "\\caption{Economic value of the HAR\\,+\\,RM\\,+\\,FOMC forecasts}\n"
        "\\label{tab:rh_econ_rmfomc}\n"
        f"\\begin{{tabular}}{{@{{}}l{'r' * (ncol - 1)}@{{}}}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{$\\SR$} & \\multicolumn{4}{c}{$\\Delta\\SR$} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(l){4-7}\n"
        " & Mid & Bid/ask & Full set & Short straddle & HAR set & OLS-HAR \\\\\n"
        "\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}[flushleft]\n\\footnotesize\n"
        "\\item \\textit{Notes:} The sign strategy of Table~\\ref{tab:rh_econ_main} on the forecasts from "
        "HAR\\,+\\,RM\\,+\\,FOMC, the HAR benchmark set with the futures' realized moments and the FOMC calendar "
        "(Table~\\ref{tab:rh_ablation}, Panel~D), on the \\rheSzDays{} post-warmup days of $\\mathcal T_{\\mathrm{opt}}$. "
        "Each model keeps the full predictor set's frozen hyperparameters. $\\SR$ is the annualized Sharpe ratio of "
        "Section~\\ref{sec:rh_econ_measures}, at mid-quotes and at quoted bid/ask prices. $\\Delta\\SR$ is a "
        "difference of Sharpe ratios at mid-quotes. In column Full set it is the same model class on the full "
        "predictor set minus the row. In the other three columns it is the row minus the short straddle, minus the "
        "same model class on the HAR benchmark set, and minus the OLS on the HAR benchmark set. Under each "
        "$\\Delta\\SR$, in brackets, the $p$-value of the \\citet{LedoitWolf2008} test of equal Sharpe ratios (block "
        "length \\rheLWBlock{}, \\rheLWBoot{} resamples). " + me.STAR_NOTE + "\n"
        "\\item[a] The OLS on HAR\\,+\\,RM\\,+\\,FOMC. Its HAR benchmark set is the OLS-HAR, and it has no forecast on "
        "the full predictor set.\n"
        "\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n"
    )


def x8_csv(series: dict, xs: dict, xp: dict, qgate: dict) -> None:
    """2026-10-29 (X8), RULE.txt 7: results/econ_rmfomc_2026-10-29/stats.csv (one row per strategy) and pairs.csv (one
    row per comparison), unrounded, each headed by the label and the rule."""
    head = (
        f"# X8 (Q-2026-10-29-21): {RMF_LABEL}. Rule: results/econ_rmfomc_2026-10-29/RULE.txt. 614 post-warmup days.\n"
    )
    rows = []
    for e in RMF_CLS:
        r = xs[rkey(e)]
        rows.append(
            {
                "key": rkey(e),
                "class": e,
                **{c: r[c] for c in ("sh_mid", "sh_x", "mean_mid", "mean_x", "sh_d6", "sh_d1", "be_mean", "be_sh")},
                **{c: r[c] for c in ("dsh_short", "p_short", "pb_short", "t_short", "pt_short", "dmean_short")},
                **{c: r[c] for c in ("dsh_short_x", "p_short_x", "pb_short_x")},
                "long_days": int((series[rkey(e)]["q"].to_numpy(float)[me.WARMUP :] > 0).sum()),
                "qlike16_eval": qgate[e][0],
                "qlike16_ablation_metrics": qgate[e][1],
            }
        )
    out_p = []
    names = {
        "full": ("full set", "HAR + RM + FOMC"),
        "har": ("HAR + RM + FOMC", "HAR set"),
        "ols": ("HAR + RM + FOMC", "OLS-HAR"),
    }
    for (nm, e), p_ in xp.items():
        if (nm, e) == ("ols", "ols"):
            continue  # the same pair as ("har", "ols")
        a, b = names[nm]
        out_p.append(
            {
                "comparison": nm,
                "class": e,
                "a": a,
                "b": b,
                **{c: p_[c] for c in ("dsh", "p", "pb", "dsh_x", "p_x", "pb_x")},
                "t": p_["t"],
                "pt": p_["pt"],
                "dmean": p_["dmean"],
            }
        )
    for e in RMF_CLS:
        r = xs[rkey(e)]
        out_p.append(
            {
                "comparison": "short",
                "class": e,
                "a": "HAR + RM + FOMC",
                "b": "short straddle",
                "dsh": r["dsh_short"],
                "p": r["p_short"],
                "pb": r["pb_short"],
                "dsh_x": r["dsh_short_x"],
                "p_x": r["p_short_x"],
                "pb_x": r["pb_short_x"],
                "t": r["t_short"],
                "pt": r["pt_short"],
                "dmean": r["dmean_short"],
            }
        )
    RMF_OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in (("stats.csv", pd.DataFrame(rows)), ("pairs.csv", pd.DataFrame(out_p))):
        with open(RMF_OUT / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(head)
            frame.to_csv(fh, index=False, float_format="%.10g", lineterminator="\n")
        print("wrote", fr.rel(RMF_OUT / name))


# ============================================================================================ figure
MAG, BLUE, INK = me.FAM_COLOR["pooled"], me.FAM_COLOR["linear"], me.FAM_COLOR["short"]


def by_year_frame(st: dict) -> dict:
    years = sorted(st[SHORT]["by_year"])
    full = [key(e, "all_features") for e in EST]
    har4 = [key("ridge", "baseline")] + [key(e, "baseline") for e in TREES]
    for e in LINEAR:  # the three linear HAR rows coincide year by year too
        for y in years:
            assert abs(st[key(e, "baseline")]["by_year"][y] - st[key("ridge", "baseline")]["by_year"][y]) < 1e-9
    fv = np.array([[st[k]["by_year"][y] for y in years] for k in full])
    hv = np.array([[st[k]["by_year"][y] for y in years] for k in har4])
    ci = np.array([st[CI]["by_year"][y] for y in years])
    sh = np.array([st[SHORT]["by_year"][y] for y in years])
    for k in [*full, *har4, CI]:
        assert st[k]["n_y"] == st[SHORT]["n_y"], k
    fmed, hmed = np.median(fv, axis=0), np.median(hv, axis=0)
    return {
        "years": years,
        "n_y": st[SHORT]["n_y"],
        "full": fv,
        "har": hv,
        "full_keys": full,
        "har_keys": har4,
        "full_med": fmed.tolist(),
        "full_min": fv.min(axis=0).tolist(),
        "full_max": fv.max(axis=0).tolist(),
        "har_med": hmed.tolist(),
        "har_min": hv.min(axis=0).tolist(),
        "har_max": hv.max(axis=0).tolist(),
        "har_med6": np.median(
            np.array([[st[key(e, "baseline")]["by_year"][y] for y in years] for e in EST]), axis=0
        ).tolist(),
        "ci": ci.tolist(),
        "short": sh.tolist(),
        "full_all_pos": int((fv > 0).all(axis=0).sum()),
        "full_all_above_short": int((fv > sh).all(axis=0).sum()),
        "full_med_above_ci": int((fmed > ci).sum()),
        "full_med_above_har": int((fmed > hmed).sum()),
        "full_med_above_short": int((fmed > sh).sum()),
        "ci_above_all_full": int((ci > fv.max(axis=0)).sum()),
        "partial": [y for y in years if st[SHORT]["a_y"][y] > st[SHORT]["n_y"][y] + 1e-9],
        "ci_pos": int((ci > 0).sum()),
        "short_neg": int((sh < 0).sum()),
    }


def fig_byyear(yr: dict, first_day: pd.Timestamp, last_day: pd.Timestamp) -> None:
    years, n_y = yr["years"], yr["n_y"]
    xs = np.arange(len(years), dtype=float)
    fig, ax = plt.subplots(figsize=(me.FIG_W, 3.1), layout="constrained")
    ax.axhline(0, color="#bdbdbd", lw=0.8, zorder=0)
    half = 0.36
    off_full = np.linspace(-0.27, -0.05, 6)
    off_har = np.linspace(0.07, 0.25, 4)
    marker_full = ["o", "o", "o", "s", "s", "s"]
    marker_har = ["o", "s", "s", "s"]
    for i, x in enumerate(xs):
        ax.hlines(yr["short"][i], x - half, x + half, colors=INK, linestyles=(0, (4, 2)), lw=1.0, zorder=2)
        # 2026-10-26 (D2/D3): parked -- the closing-interval HAR line. Was:
        # ax.hlines(yr["ci"][i], x - half, x + half, colors=BLUE, lw=1.4, zorder=2)
        ax.hlines(yr["full_med"][i], x - 0.31, x - 0.01, colors=MAG, lw=1.8, zorder=3)
        ax.hlines(yr["har_med"][i], x + 0.03, x + 0.29, colors=MAG, lw=1.8, zorder=3)
        for j in range(6):
            ax.plot(
                x + off_full[j],
                yr["full"][j, i],
                ls="",
                marker=marker_full[j],
                ms=4.6,
                mfc=MAG,
                mec="white",
                mew=0.5,
                zorder=4,
            )
        for j in range(4):
            ax.plot(
                x + off_har[j],
                yr["har"][j, i],
                ls="",
                marker=marker_har[j],
                ms=4.6,
                mfc="white",
                mec=MAG,
                mew=1.0,
                zorder=4,
            )
    labs = [f"{y}\n({n_y[y]} days)" for y in years]
    if first_day > pd.Timestamp(f"{years[0]}-01-02"):
        labs[0] = f"{years[0]}, {first_day.strftime('%b.')}–Dec.\n({n_y[years[0]]} days)"
    if last_day < pd.Timestamp(f"{years[-1]}-12-30"):
        labs[-1] = f"{years[-1]}, Jan.–{last_day.strftime('%b.')}\n({n_y[years[-1]]} days)"
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(years) - 0.5)
    ax.set_ylabel("Sharpe ratio within the year, mid-quotes")
    me._style(ax)
    handles = [
        Line2D([], [], ls="", marker="o", ms=4.6, mfc=MAG, mec="white", mew=0.5),
        Line2D([], [], ls="", marker="s", ms=4.6, mfc=MAG, mec="white", mew=0.5),
        Line2D([], [], color=MAG, lw=1.8),
        Line2D([], [], ls="", marker="o", ms=4.6, mfc="white", mec=MAG, mew=1.0),
        Line2D([], [], ls="", marker="s", ms=4.6, mfc="white", mec=MAG, mew=1.0),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2))),
    ]
    labels = [
        "Intraday linear, full set",
        "Intraday trees, full set",
        "Median (left: full set; right: HAR set)",
        "Intraday linear, HAR set",
        "Intraday trees, HAR set",
        me.SHORT_LABEL,
    ]
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, handlelength=2.2, columnspacing=1.4)
    w, _ = fig.get_size_inches()
    assert abs(w - me.FIG_W) < 1e-9
    fig.savefig(FIG_PDF)
    if PREVIEW is not None:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / "rehaul_econ_byyear.png", dpi=200)
    plt.close(fig)
    print("wrote", fr.rel(FIG_PDF))


def fig_float_tex(yr: dict) -> str:
    words = YEAR_WORD[: len(yr["years"])]
    seq = lambda stem: ", ".join(f"\\{stem}{w}{{}}" for w in words)  # noqa: E731
    return (
        HEADER + "\\ifdefined\\rhemShOlsHar\\else\\input{generated/rehaul_econ_main_numbers}\\fi\n"
        "\\begin{figure}[tbp]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{generated/rehaul_econ_byyear.pdf}\n"
        "\\caption{Sharpe ratio of the sign strategy by calendar year, at mid-quotes, on the \\rheSzDays{} "
        # 2026-10-26 (D19): parked -- "Each point is $\\SR_y$ of Section~\\ref{sec:rh_econ_measures}, the year's term
        # of the annualized Sharpe ratio $\\SR$, which is their simple mean." (D6)
        "post-warmup days of $\\mathcal T_{\\mathrm{opt}}$. Each point is $\\SR_y$, the Sharpe ratio of the year's "
        "daily returns annualized by the square root of that year's option days per calendar year "
        "(Section~\\ref{sec:rh_econ_measures}); the paper's $\\SR$ pools the days of all years. "
        # 2026-10-28 (author: prune "pooled"; "pools the days of all years" above is kept: pooled over days). Was:
        # "Within each year, the left group shows the six pooled intraday forecasts on the full predictor "
        # "set (filled) and the right group the \\rhemHarDistinctWord{} distinct pooled forecasts on the HAR benchmark "
        "Within each year, the left group shows the six intraday forecasts on the full predictor "
        "set (filled) and the right group the \\rhemHarDistinctWord{} distinct forecasts on the HAR benchmark "
        "set (open; the ridge, lasso and elastic net forecasts coincide, Table~\\ref{tab:rh_econ_main}); circles are "
        "linear models, squares tree ensembles, and the short magenta lines the group medians "
        f"(full set: {seq('rhemByYearFullMed')}; HAR set: {seq('rhemByYearHarMed')}). "
        # 2026-10-26 (D2/D3): parked -- the closing-interval HAR line. Was: "The long lines mark the closing-interval
        # HAR forecast (solid; {seq('rhemByYearCiHar')}) and the short straddle (dashed; ...)."
        f"The long dashed line marks the short straddle ({seq('rhemByYearShort')}). "
        f"The years hold {seq('rhemNy')} option days, and the first and last are partial.}}\n"
        "\\label{fig:rh_econ_byyear}\n\\end{figure}\n"
    )


# ============================================================================================ draft claims
def draft_claims(st: dict, pairs: dict, yr: dict) -> None:
    """DRAFT CLAIMS: the words of writeup/drafts/2026-10-25_econ_main.tex that no macro carries ("none", "all",
    "every"); a rerun whose data no longer bear one out stops here, naming the sentence."""
    full = [key(e, "all_features") for e in EST]
    har4 = [key("ridge", "baseline")] + [key(e, "baseline") for e in TREES]
    claims = [
        (
            all(pairs[("ci", k)]["dsh"] > 0 for k in full),
            "every full-set forecast has a higher mid SR than CI-HAR (the range \\rhemFullVsCiHarMin..Max is positive)",
        ),
        (
            all(pairs[("ci", k)]["p"] >= 0.05 and pairs[("ci", k)]["p_x"] >= 0.05 for k in full),
            "none of the full-set minus CI-HAR differences is significant, at mid or at bid/ask",
        ),
        (all(pairs[("ci", k)]["dsh"] < 0 for k in har4), "all four distinct HAR-set forecasts are below CI-HAR"),
        (all(st[k]["p_short"] >= 0.05 for k in har4), "no distinct HAR-set forecast beats the short significantly"),
        (yr["full_all_pos"] == len(yr["years"]), "every full-set forecast has SR_y > 0 in every year"),
        (all(st[k]["sh_x"] < st[k]["sh_mid"] for k in st), "bid/ask lowers every strategy's SR"),
        (st[SHORT]["be_mean"] < 1.0, "the short straddle's mean reaches zero inside the quoted spread (k* < 1)"),
        (st[SHORT]["sh_mid"] < 0, "the short straddle's SR is negative at mid"),
        (yr["partial"] == [yr["years"][0], yr["years"][-1]], "the first and the last year (only) are partial"),
    ]
    # 2026-10-26 (D2/D3): the three CI-HAR claims (the first three) left the main text: reported, not asserted
    reported = {c for _, c in claims[:3]}
    # 2026-10-26 (author decision D19): FAILS under the pooled ratio and made report-only so the run can finish (the
    # sentence needs the author): the short straddle's mid SR was -0.12 (D6, equal-year mean of SR_y) and is +0.32
    # (the pooled ratio, q = 225.5); its D6 is still -0.12 (rhemShDSixShort)
    reported.add("the short straddle's SR is negative at mid")
    bad = [c for ok, c in claims if not ok and c not in reported]
    for ok, c in claims:
        print(f"  draft claim {'holds' if ok else 'FAILS'}{' (reported only)' if c in reported else ''}: {c}")
    assert not bad, f"DRAFT CLAIMS no longer hold: {bad}"


# ============================================================================================ main
def main() -> None:
    assert fr.FINAL and not fr.PENDING_TREES and fr.pool_n() == fr.POOL_N_ALL == 12, fr.tag()
    print(fr.tag())
    mt, model_set, series, fam = me.load()
    me.gate_against_master(mt, series, [SHORT, CI])
    me.add_pooled_linear(series, fam)
    me.add_pooled_trees(series, fam)
    add_har_trees(series)
    harlin = har_linear_check()
    print(
        f"pooled linear HAR forecasts: max |diff| {harlin['max_abs']:.3e}, max rel {harlin['max_rel']:.3e} over "
        f"{harlin['n_rows']} 16:00 rows; penalties {harlin['alpha']}"
    )
    q0 = series[key("ridge", "baseline")]["q"].to_numpy(float)
    for e in LINEAR:
        assert np.array_equal(series[key(e, "baseline")]["q"].to_numpy(float), q0), e
    add_ols_har(series)  # 2026-10-26 (D2): the primary benchmark's strategy
    assert np.array_equal(series[OLSH]["q"].to_numpy(float), q0), "the OLS-HAR trades as the HAR-set ridge"

    rows = [key(e, s) for s in SETS for e in EST] + [CI, SHORT]
    st = {k: row_stats(series, k) for k in rows}
    st[OLSH] = row_stats(series, OLSH)  # 2026-10-26 (D2)
    pairs: dict = {}
    for e in EST:
        pairs[("har", e)] = pair(series, key(e, "all_features"), key(e, "baseline"))
    for k in rows[:12]:
        pairs[("ci", k)] = pair(series, k, CI)
    for e in EST:  # 2026-10-26 (D2): every full-set strategy against the OLS-HAR strategy
        pairs[("ols", e)] = pair(series, key(e, "all_features"), OLSH)

    b = series[CI]
    bid, ask = b["bid"].to_numpy(float), b["ask"].to_numpy(float)
    hp = ((ask - bid) / 2.0) / ((ask + bid) / 2.0)
    spread = {
        "med_all": float(np.median(hp)),
        "med_pw": float(np.median(hp[me.WARMUP :])),
        "mean_pw": float(np.mean(hp[me.WARMUP :])),
    }
    print(
        f"median half-spread / premium: all 866 days {100 * spread['med_all']:.3f} %, post-warmup "
        f"{100 * spread['med_pw']:.3f} % (mean {100 * spread['mean_pw']:.3f} %)"
    )

    gate(series, st, pairs, spread)
    yr = by_year_frame(st)
    pw = me.pw_days()
    draft_claims(st, pairs, yr)

    # 2026-10-29 (X8; proposed after the author's 27 Oct remark; not pre-registered): the HAR + RM + FOMC strategies,
    # after every existing statistic, so the test log gains rows at its end and the existing outputs are unchanged.
    # Their statistics stay out of st (draft_claims and the gate read st).
    qgate = add_rmfomc(series)
    for e, (got, want) in qgate.items():
        print(f"  gate X8 16:00 QLIKE {e:5s} on {RMF_N_EVAL} days: {got:.12f} (ablation metrics.csv {want:.12f})")
    xs, xp = x8_stats(series)

    # 2026-10-29 (X8): the X8 macros appended after the existing ones. Was: write(NUMBERS, numbers_tex(st, pairs,
    # spread, harlin, yr))
    write(NUMBERS, numbers_tex(st, pairs, spread, harlin, yr) + x8_numbers(xs, xp))
    me.write_test_log("econ_main")  # 2026-10-26 (D19): Ledoit-Wolf p and the bootstrap check p of every tested pair
    write(TABLE, table_tex(st, pairs))
    write(RMF_TABLE, x8_table_tex(xs, xp))  # 2026-10-29 (X8)
    x8_csv(series, xs, xp, qgate)  # 2026-10-29 (X8)
    fig_byyear(yr, pw[0], pw[-1])
    write(FIG_TEX, fig_float_tex(yr))

    # console summary
    print(
        f"\n{'row':34s} {'SRmid':>6s} {'SRba':>6s} {'mean':>6s} {'meanX':>6s} {'dSRsh':>6s} {'p':>6s} {'t':>6s} "
        f"{'dSRhar':>6s} {'p':>6s} {'dSRci':>6s} {'p':>6s} {'kMean':>6s} {'kSR':>6s}"
    )
    for k in rows:
        r = st[k]
        est_ = next((e for e in EST if k == key(e, "all_features")), None)
        h = pairs.get(("har", est_)) if est_ else None
        c = pairs.get(("ci", k))
        print(
            f"{k:34s} {r['sh_mid']:6.2f} {r['sh_x']:6.2f} {100 * r['mean_mid']:6.2f} {100 * r['mean_x']:6.2f} "
            f"{r.get('dsh_short', np.nan):6.2f} {r.get('p_short', np.nan):6.3f} {r.get('t_short', np.nan):6.2f} "
            f"{h['dsh'] if h else np.nan:6.2f} {h['p'] if h else np.nan:6.3f} "
            f"{c['dsh'] if c else np.nan:6.2f} {c['p'] if c else np.nan:6.3f} {r['be_mean']:6.2f} {r['be_sh']:6.2f}"
        )
    print("\nby year:", yr["years"], "N_y", yr["n_y"])
    for nm in ("full_med", "full_min", "full_max", "har_med", "har_med6", "har_min", "har_max", "ci", "short"):
        print(f"  {nm:9s}", [round(v, 2) for v in yr[nm]])
    # 2026-10-29 (X8): the HAR + RM + FOMC rows
    print(f"\nX8 ({RMF_LABEL}): HAR + RM + FOMC, Delta SR mid [Ledoit-Wolf p]")
    print(
        f"{'class':6s} {'SRmid':>6s} {'SRba':>6s} {'Full-RMF':>15s} {'RMF-short':>15s} {'RMF-HAR':>15s} {'RMF-OLSH':>15s}"
    )
    for e in RMF_CLS:
        r = xs[rkey(e)]
        f = xp.get(("full", e))
        cells = [(f["dsh"], f["p"]) if f else (np.nan, np.nan), (r["dsh_short"], r["p_short"])]
        cells += [(xp[(nm, e)]["dsh"], xp[(nm, e)]["p"]) for nm in ("har", "ols")]
        print(f"{e:6s} {r['sh_mid']:6.2f} {r['sh_x']:6.2f} " + " ".join(f"{d:7.2f} [{p:5.3f}]" for d, p in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, default=None, help="directory for a PNG preview of the figure")
    PREVIEW = ap.parse_args().preview
    main()
