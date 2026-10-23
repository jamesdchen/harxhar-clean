"""Tables and figures for the rehauled paper's economic section (writeup/rehaul/07_economics.tex).

Outline: writeup/PAPER_REHAUL_OUTLINE_2026-10-15.md, section 7 (E1-E4 and the figures).

DATA
  The paper model set (results/close_master_table/paper_model_set.csv, 112 forecasts) and its per-day frame
  results/close_master_table/master_table_daily.parquet (q, ret_mid, ret_crossed, plain, iv_var, qlike_raw on 866
  trade days); master_table.csv gates every recomputed number.  The straddle book (R, exit, bids, asks) is the GUI's
  quote book (results/atm_straddle_0dte_1530/daily_a0.parquet).  The per-bar MinGRU (ridge-initialised shortcut,
  QLIKE-tuned; results/puffer_forecast_2026-10-12/qlike_study/score_rinitq/forecasts.csv) is read and scored by the
  GUI's own loader (master_table_gui.rows._nn_daily), so its rows are the GUI's rows.
  2026-10-22 (user, E34: base 5 across the paper): the pooled intraday trees (LightGBM, XGBoost, random forest, on the
  HAR benchmark and on the full predictor set) are the rerun on the design of Table 1's pooled linear 13-bar rerun
  (13 regular-hours bars, base-5 HAR horizons, FOMC series in, 1,000-day window, refit at every bar), read at the
  16:00 bar through writeup/pooled_trees_13.py (results/pooled_trees_rerun13_2026-10-20/w1000).  They were the base-2
  trees of results/pooled_trees_2026-10-07 and, on the full set, the earlier draft's results/spxw_pnl/yhat_tree00,
  yhat_tree10 (the master table's "lgbm" / "xgb" rows, which stay in the model set).
  2026-10-22 (user, E34: one design for the 12 pooled forecasts): the six pooled linear rows (poolpt_{enet,lasso,
  ridge}_{baseline,all_features}) are the like-for-like rerun on the trees' design (13 regular-hours bars, 1,000-day
  window, evaluation from 3 January 2012), read at the 16:00 bar through writeup/pooled_linear_13.py
  (results/pooled_linear_rerun_2026-10-20_13_2012_w1000), in place of the master table's 48-interval runs (2,000-day
  window).  They are rows of their own, poolpt13_<est>_<set> (add_pooled_linear, after gate_against_master), in every
  pooled row of the tables and figures and in the 12-forecast pooled statistics; the master rows stay in the model set
  and its statistics as they are (as the master table's lgbm / xgb for the trees).

TRADE (experiments/master_table_close.py:1168-1187, 1293-1312)
  q = sign(s), s = plain - iv_var, ties short; mid r = q R; crossed: a buy pays the ask, a sale gets the bid, spread
  paid at entry only.  Always short: q = -1.  Sharpe = mean / sd(ddof 1) x sqrt(252), per trade day.  Every t is
  atm_straddle_lib.newey_west_t (Bartlett, lag floor(1.5 n^(1/3)) = 14 on 866 days); dSharpe's t is the HAC t of the
  daily return difference (it tests the mean difference, not the Sharpe difference).
  2026-10-21 (user, decision #2): parked -- the two sentences above. Every test of strategy returns is now
  writeup/hac_inference.py on the 614 post-warmup days: means (mean return, mean return differential) by the
  Lazarus-Lewis-Stock-Watson (2018) Newey-West t with bandwidth S = ceil(1.3 T^(1/2)) = 33 and fixed-b p-values
  (Kiefer-Vogelsang 2005); Sharpe-ratio differences by the Ledoit-Wolf (2008) bootstrap test.  DM (E3 only):
  src.evaluation.diebold_mariano.dm_test on the daily QLIKE difference, the master table's setting (automatic lag).
  2026-10-22 (user, D6): parked -- the per-year ratio of new-notes 15 (asl.sharpe_ann_by_year,
  N_y-weighted SR_y = mean_y / sd_y x sqrt(N_y)) and the Ledoit-Wolf test of the Sharpe-ratio differences. Every
  printed Sharpe ratio is now hac_inference.d6_sharpe (SR_y = mean_y / sd_y x sqrt(A_y), A_y = N_y over the fraction
  of year y the sample covers; the simple mean over the years), and every significance on a Sharpe-ratio difference
  (stars, p, counts) is hac_inference.d6_diff_test (paired circular block bootstrap of the D6 difference, block
  hi.LW_BLOCK, hi.LW_BOOT resamples, seed hi.D6_SEED).  The Ledoit-Wolf p stays computed (lw_p) for the robustness
  table of the Sharpe-ratio definitions (rehaul_econ_sharpe_defs.tex: D6, the per-year ratio "D1", the pooled
  sqrt(252) ratio "D2", and D6 before / after 16 May 2022, for the six pooled classes, full set minus HAR).

SIZING (writeup/master_table_gui/rows.py:88-134 _position_sizes, :1140-1180 _score_sizes)
  The GUI's functions are called directly: sizes use only earlier days; scored on days 253-866 (614).

OUTPUTS
  writeup/generated/rehaul_econ_e1_control.tex     Table E1  (tab:rh_econ_control)
  writeup/generated/rehaul_econ_e2_allfeat.tex     Table E2  (tab:rh_econ_allfeat)
  writeup/generated/rehaul_econ_e3_pooled.tex      Table E3  (tab:rh_econ_pooled)
  writeup/generated/rehaul_econ_e4_sizing.tex      Table E4  (tab:rh_econ_sizing)
  writeup/generated/rehaul_econ_fig_cumpnl.tex     float for figures/rehaul_econ_cumpnl.pdf  (fig:rh_econ_cumpnl)
  writeup/generated/rehaul_econ_fig_sizing.tex     float for figures/rehaul_econ_sizing.pdf  (fig:rh_econ_sizing)
  writeup/generated/rehaul_econ_fig_byyear.tex     float for figures/rehaul_econ_byyear.pdf  (fig:rh_econ_byyear)
  writeup/generated/rehaul_econ_numbers.tex        \\rhe... macros for the section's prose (not \\input by the floats)
  writeup/generated/rehaul_econ_sharpe_defs.tex    2026-10-22 (user, D6): robustness table of the Sharpe-ratio
                                                   definitions (tab:rh_econ_sharpe_defs)

TABLE LEGIBILITY (2026-10-15)
  Tables follow writeup/NOTATION_GUIDE_2026-10-15.md section 8 and TABLE_FIXES_FROM_FLOATS_2026-10-15.md: short
  captions, Notes of about five sentences, stars for the 10/5/1 % levels on the estimate with its HAC t (or DM) in
  parentheses underneath, at most 7 columns, Panel A/B, the guide's vocabulary.  The numbers the notes and the
  cumulative-return caption cite are \\rhe... macros, so every table and the figure float \\input
  rehaul_econ_numbers.tex when \\rheNdays is not yet defined.  The sizing and by-year figures are parked by the floats
  file (guide section 7); their code here is unchanged.

NN PARKED (2026-10-15, user: "park all of the NN stuff for now")
  The closing-interval MinGRU leaves every table and figure: its E1 row and note b, its E2 row and note a, its E4 rows,
  its line in the cumulative-return figure (and the parked by-year and sizing figures) and its mentions in the figure
  captions.  It is still loaded and gated (the outline's Sharpe ratios), and its \\rhe... macros (rheNNSh, rheNNShX,
  rheNNBaseSh, rheNNBaseShX, rheGainNN, rheGainExNN) stay defined with unchanged values but unused by the floats.
  Parked code is marked "2026-10-15 (user): parked -- NN content parked for now".

Run:  python writeup/make_rehaul_economics.py [--preview DIR]
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup"
for _p in (ROOT, ROOT / "notebooks", WRITEUP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import hac_inference as hi  # noqa: E402  (2026-10-21, decision #2: every test of strategy returns)
import matplotlib  # noqa: E402
import pooled_linear_13 as pl13  # noqa: E402  (2026-10-22, E34: the pooled linear rows on the trees' design)
import pooled_trees_13 as pt13  # noqa: E402  (2026-10-22, E34: the base-5 pooled trees at the 16:00 bar)
import final_run as fr  # noqa: E402  (2026-10-23: the switch to the final pooled rerun, REHAUL_FINAL_RUN)

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

RES = ROOT / "results" / "close_master_table"
# 2026-10-23: final_run.GEN / FIG (REHAUL_GEN_DIR / REHAUL_FIG_DIR; defaults writeup/generated, writeup/figures).
# Was: GEN = WRITEUP / "generated"; FIG = WRITEUP / "figures"
GEN = fr.GEN
FIG = fr.FIG
ANN = np.sqrt(252.0)  # 2026-10-18 (user, new-notes 15): now only the legacy sharpe_days_252 (master-table / levels.csv gates)
OPT_DAYS = None  # 2026-10-18 (user, new-notes 15): the 866 option-data days, set by load(); dates for the per-year Sharpe ratio
SIG = 1.959963984540054  # two-sided 5 %
WARMUP = gui._SIZING_WARMUP  # 252
# 2026-10-20 (user, third-list 2): HAC t, mean, hit rate and long days on the 614 post-warmup days (the Sharpe sample);
# lag floor(1.5 * 614^(1/3)) = 12. Was: HAC_LAG = 14 (866 days); kept as HAC_LAG_ALL for the master-table gate.
# 2026-10-21 (user, decision #2): parked -- HAC_LAG = 12, the lag floor(1.5 T^(1/3)) of the paper's tests; the tests
# now use hac_inference (LLSW bandwidth HAC_S, fixed-b p). HAC_LAG_ALL stays: the 866-day master-table gate only.
# HAC_LAG = 12
HAC_LAG_ALL = 14
N_PW = 614  # post-warmup days, the sample of every test (asserted against N_OPT - WARMUP in nw_t)
HAC_S = hi.llsw_bandwidth(N_PW)  # 33
HAC_B = HAC_S / N_PW

# ----------------------------------------------------------------------------- forecasts of record
SHORT = "always_short"
OLS = "sub_ols_baseline"
NN_ALL, NN_BASE = "mingru_rinitq_all_features", "mingru_rinitq_baseline"
PERBAR_ALL = (
    "subpt_enet_all_features",
    "subpt_lasso_all_features",
    "subpt_ridge_all_features",
)
# 2026-10-22 (user, E34: one design for the 12 pooled forecasts): parked -- the master table's 48-interval pooled
# linear rows; the pooled rows are now the 13-bar rerun on the trees' design (add_pooled_linear, poolpt13_*). Was:
# POOLED_ALL = (
#     "poolpt_enet_all_features",
#     "poolpt_lasso_all_features",
#     "poolpt_ridge_all_features",
# )
POOLED_ALL = (
    "poolpt13_enet_all_features",
    "poolpt13_lasso_all_features",
    "poolpt13_ridge_all_features",
)
TREE_EST = (("lgbm", "LightGBM"), ("xgb", "XGBoost"), ("rf", "Random forest"))
# 2026-10-17 (user): the tree rows of Table E1 / E4 and the figures' LightGBM are the POOLED INTRADAY trees (30-minute
# grid), no longer closing-interval trees.  Forecast tables as in make_rehaul_results.py (load_all) and
# experiments/sharpe_tests_pooled.py (load_pooled_trees): the full-set LightGBM / XGBoost are the earlier draft's pooled
# trees (results/spxw_pnl/yhat_tree00, yhat_tree10 = the master table's "lgbm" / "xgb" rows); the full-set forest is the
# INTERIM table of its 16:00 forecasts (the every-interval walk is still running).  Optuna-selected hyperparameters,
# re-estimated at every interval on a rolling window of 24,000 intervals (Appendix sec:rh_exp_schemes).
# 2026-10-22 (user, E34): parked -- the base-2 trees (and the earlier draft's full-set LightGBM / XGBoost), their
# levels.csv / master-table gates and the printed Sharpe ratios of record; the trees are now read through pt13
# (pooled_trees_13.py: the 13-bar rerun, every arm complete, so no interim forest). Was:
# POOLTREE_DIR = ROOT / "results" / "pooled_trees_2026-10-07"
# POOLTREE_ALL = {
#     "lgbm": ROOT / "results" / "spxw_pnl" / "yhat_tree00.parquet",
#     "xgb": ROOT / "results" / "spxw_pnl" / "yhat_tree10.parquet",
#     "rf": POOLTREE_DIR / "yhat_pooltree_rf_all_features_INTERIM1600.parquet",
# }
# POOLTREE_LEVELS = {"lgbm": "score_lgbm", "xgb": "score_xgb", "rf": "score_rf_interim1600"}
# POOLTREE_MT = {"lgbm": "lgbm", "xgb": "xgb"}  # master-table keys of the same forecasts (gate)
# POOLTREE_SR = {"lgbm": 1.36, "xgb": 1.08, "rf": 0.97}  # printed SR (mid, days 253-866) = \rhrSh<Est>PoolFull
# TREE_MARK = {"lgbm": "a,b", "xgb": "a,b", "rf": "a,$\\dagger$"}
# INTERIM_TREES = {"rf"}  # full-set pooled forest: interim 16:00 table
TREE_MARK = {"lgbm": "a", "xgb": "a", "rf": "a"}
INTERIM_TREES: set[str] = set()
POOL_LIN_MARK = "a"  # 2026-10-22 (user, E34): E1 note a now describes every pooled row (one design), the linear ones too
# 2026-10-22 (user, E34): parked -- "30-minute grid" (the 48-interval grid of the earlier pooled linear rows); every
# pooled row is now on the 13 regular-hours bars. Was: PANEL_POOL = "Pooled intraday models (30-minute grid)"
PANEL_POOL = r"Pooled intraday models (\rhePoolNBars{} regular-hours intervals)"
PANEL_CLOSE = "Closing-interval models (15:30--16:00 ET half hour)"
EST_NAME = {"enet": "Elastic net", "lasso": "Lasso", "ridge": "Ridge"}

# ----------------------------------------------------------------------------- palette (one hue per family)
# Validated with the dataviz skill's validate_palette.js on a white surface, all pairs (lines cross): lightness band,
# chroma floor, CVD (worst deutan dE 8.2) and normal-vision (worst dE 19.8) floors and 3:1 contrast all pass.  Always
# short is the neutral ink.  Every family also has its own marker, so identity is never colour alone.
FAM_COLOR = {
    "linear": "#2a78d6",
    "tree": "#d55e00",
    "nn": "#009e73",
    "pooled": "#b9479b",
    "short": "#222222",
}
FAM_MARKER = {"linear": "o", "tree": "s", "nn": "^", "pooled": "D", "short": "x"}
# 2026-10-15 (table legibility): parked -- code terms (per-bar, 48 bars); the guide's model classes below
# FAM_LABEL = {
#     "linear": "Per-bar linear",
#     "tree": "Per-bar trees",
#     "nn": "Per-bar MinGRU",
#     "pooled": "Pooled (48 bars)",
#     "short": "Always short",
# }
FAM_LABEL = {
    "linear": "Closing-interval linear",
    "tree": "Closing-interval trees",
    "nn": "Closing-interval MinGRU",
    "pooled": "Intraday",
    "short": "Short straddle",  # 2026-10-15 (consolidation): was "Always short" (guide section 6b)
}
TOP_DAYS = (
    10  # the tail check: each forecast's 10 days of largest gain over always short
)
GRID = "#d9d9d9"
INK2 = "#52514e"

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.edgecolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": "#111111",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
FIG_W = 6.5  # \linewidth of paper_rehaul.tex (letter, 1in margins), so the PDF is placed at 100 %


# ============================================================================= data
# 2026-10-17 (user, E17): parked -- the Sharpe ratio ran over all 866 option days. Every printed ratio is now on the
# days t > WARMUP (253-866, 614 days), the sample of the sizing rules, so the sign strategy and the sizing rules share
# one sample (07_economics.tex, Performance measures). HAC t, mean, hit rate stay on all 866 days. Was:
# def sharpe(x) -> float:
#     x = np.asarray(x, float)
#     return float(x.mean() / x.std(ddof=1) * ANN)
N_OPT = 866


def _days_of(x, dates=None):
    """2026-10-22 (user, D6): the dates of a return series: given, or the 866 option days, or the 614 post-warmup days
    (by length)."""
    if dates is not None:
        return dates
    if len(x) == len(OPT_DAYS):
        return OPT_DAYS
    if len(x) == len(OPT_DAYS) - WARMUP:
        return OPT_DAYS[WARMUP:]
    raise ValueError(f"sharpe_days: {len(x)} returns, no dates")


def pw_days() -> pd.DatetimeIndex:
    """2026-10-22 (user, D6): the 614 post-warmup option days t > WARMUP (OPT_DAYS is set by load())."""
    assert OPT_DAYS is not None, "load() sets OPT_DAYS"
    return OPT_DAYS[WARMUP:]


# 2026-10-18 (user, new-notes 15): Sharpe ratio annualized year by year (asl.sharpe_ann_by_year, N_y-weighted
# average of SR_y = mean_y / sd_y * sqrt(N_y), N_y = option days present in year y). Was (fixed factor):
# def sharpe_days(x) -> float:
#     """sqrt(252) mean / sd(ddof 1) on exactly the days given (master-table gate, sizing, by-year)."""
#     x = np.asarray(x, dtype=float)
#     return float(x.mean() / x.std(ddof=1) * ANN)
# 2026-10-22 (user, D6): parked -- the N_y-weighted per-year ratio (new-notes 15). Was:
#     return float(asl.sharpe_ann_by_year(x, dates=dates)["sharpe"])
def sharpe_days(x, dates=None) -> float:
    """The paper's Sharpe ratio (D6, hac_inference.d6_sharpe) on exactly the days given (sizing, printed ratios): the
    simple mean over calendar years of SR_y = mean_y / sd_y * sqrt(A_y), A_y = the year's days over the fraction of
    the year the days span.

    dates: the date of each return; when omitted, x must span the 866 option days or the 614 post-warmup days.
    """
    x = np.asarray(x, dtype=float)
    return float(hi.d6_sharpe(x, dates=_days_of(x, dates))["sharpe"])


def sharpe_days_d1(x, dates=None) -> float:
    """2026-10-22 (user, D6): the per-year ratio of new-notes 15 (asl.sharpe_ann_by_year, N_y-weighted SR_y =
    mean_y / sd_y * sqrt(N_y)), the paper's ratio from 2026-10-18 to 2026-10-22: robustness table only ("D1")."""
    x = np.asarray(x, dtype=float)
    return float(asl.sharpe_ann_by_year(x, dates=_days_of(x, dates), combine="weighted")["sharpe"])


def sharpe_days_252(x) -> float:
    """2026-10-18 (user, new-notes 15): the old fixed-factor ratio, kept only to gate against stored files (master table, levels.csv)."""
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * ANN)


def sharpe(x) -> float:
    """The paper's Sharpe ratio: a full option-sample series (866 days) scored on its days t > WARMUP."""
    x = np.asarray(x, float)
    assert len(x) == N_OPT, len(x)
    return sharpe_days(x[WARMUP:])


# 2026-10-21 (user, decision #2): parked -- the lag-12 Newey-West t (normal p). Was:
# def nw_t(x) -> float:
#     """HAC t on the post-warmup days t > WARMUP (third-list 2). A full 866-day series is sliced to its 614 days."""
#     x = np.asarray(x, float)
#     if len(x) == N_OPT:
#         x = x[WARMUP:]
#     t, lag = asl.newey_west_t(x)
#     if len(x) == N_OPT - WARMUP:
#         assert lag == HAC_LAG, lag
#     return float(t)
def post_warmup(x) -> np.ndarray:
    """A full 866-day series sliced to its 614 post-warmup days t > WARMUP; a 614-day series as given."""
    x = np.asarray(x, float)
    if len(x) == N_OPT:
        x = x[WARMUP:]
    assert len(x) == N_PW == N_OPT - WARMUP, len(x)
    return x


def nw_t(x) -> float:
    """The paper's t of a mean (hac_inference: LLSW bandwidth HAC_S = 33) on the 614 post-warmup days."""
    r = hi.llsw_test(post_warmup(x))
    assert r["S"] == HAC_S == 33, r["S"]
    return float(r["t"])


def hac_p(t: float) -> float:
    """Two-sided fixed-b p of a 614-day nw_t (b = 33/614)."""
    return hi.fixedb_p(t, HAC_B)


LW_CACHE: dict[tuple, float] = {}


def lw_p(series, key_a: str, key_b: str, col: str = "mid", block: int = hi.LW_BLOCK) -> float:
    """Ledoit-Wolf (2008) p of equal Sharpe ratios, forecast key_a against key_b, on the 614 post-warmup days."""
    k = (key_a, key_b, col, block)
    if k not in LW_CACHE:
        LW_CACHE[k] = hi.lw_sharpe_test(
            post_warmup(series[key_a][col]), post_warmup(series[key_b][col]), block=block
        )["p"]
    return LW_CACHE[k]


BOOT_CACHE: dict[tuple, float] = {}


def boot_p(series, key_a: str, key_b: str, col: str = "mid") -> float:
    """2026-10-22 (user, D6): p of the paired circular block bootstrap of the D6 Sharpe-ratio difference, forecast
    key_a minus key_b, on the 614 post-warmup days (hac_inference.d6_diff_test: block hi.LW_BLOCK, hi.LW_BOOT
    resamples, seed hi.D6_SEED). The significance of every printed Sharpe-ratio difference (stars, p, counts)."""
    k = (key_a, key_b, col)
    if k not in BOOT_CACHE:
        a, b = post_warmup(series[key_a][col]), post_warmup(series[key_b][col])
        r = hi.d6_diff_test(a, b, pw_days())
        assert abs(r["delta"] - (sharpe_days(a) - sharpe_days(b))) < 1e-10, (
            k,
            r["delta"],
        )
        BOOT_CACHE[k] = r["p"]
    return BOOT_CACHE[k]


def nw_t_all(x) -> float:
    """The 866-day HAC t of the master table (gate only). Was nw_t before third-list 2."""
    t, lag = asl.newey_west_t(np.asarray(x, float))
    assert len(x) != N_OPT or lag == HAC_LAG_ALL, lag
    return float(t)


def family_of(key: str, mt_family: str | None) -> str:
    if key.startswith("mingru_"):
        return "nn"
    if key == SHORT:
        return "short"
    if mt_family == "per-bar linear":
        return "linear"
    if (mt_family or "").startswith("per-bar tree"):
        return "tree"
    if mt_family in (
        "pooled twin",
        "paper",
    ):  # the paper's nine forecasts are pooled 48-bar fits too
        return "pooled"
    raise ValueError(f"no figure family for {key} ({mt_family})")


def load():
    mt = pd.read_csv(RES / "master_table.csv").set_index("key")
    ps = pd.read_csv(RES / "paper_model_set.csv")
    model_set = list(ps["key"])
    assert len(model_set) == 112, len(model_set)
    daily = pd.read_parquet(RES / "master_table_daily.parquet")
    daily = daily[daily["key"].isin(model_set)].copy()
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()

    quotes = pd.read_parquet(
        gui._QUOTE_BOOK, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
    )
    quotes.index = pd.to_datetime(quotes.index).normalize().astype("datetime64[ns]")
    book = pd.DataFrame(
        {
            "R": quotes["R"].astype(float),
            "ex": quotes["exit"].astype(float),
            "bid": (quotes["bid_c"] + quotes["bid_p"]).astype(float),
            "ask": (quotes["ask_c"] + quotes["ask_p"]).astype(float),
        }
    )

    series: dict[str, pd.DataFrame] = {}
    for key, g in daily.groupby("key", sort=False):
        g = g.set_index("day").sort_index().join(book, how="inner")
        assert len(g) == 866, (key, len(g))
        assert np.allclose(g["q"] * g["R"], g["ret_mid"], rtol=0, atol=1e-10), key
        cx = gui._crossed_return(
            g["q"].to_numpy(),
            g["ex"].to_numpy(),
            g["bid"].to_numpy(),
            g["ask"].to_numpy(),
        )
        assert np.allclose(cx, g["ret_crossed"], rtol=0, atol=1e-10), key
        f_s = (g["plain"] / g["iv_var"]).to_numpy(float)
        y_s = (g["rv_unclipped"] / g["iv_var"]).to_numpy(float)
        es1 = np.abs(y_s - 1.0) * (
            (np.minimum(f_s, y_s) <= 1.0) & (1.0 < np.maximum(f_s, y_s))
        )
        series[key] = pd.DataFrame(
            {
                "q": g["q"],
                "mid": g["ret_mid"],
                "x": g["ret_crossed"],
                "s": g["plain"] - g["iv_var"],
                "qlike": g["qlike_raw"],
                "es1": es1,
                "R": g["R"],
                "ex": g["ex"],
                "bid": g["bid"],
                "ask": g["ask"],
            }
        )
    days = next(iter(series.values())).index
    for key, g in series.items():
        assert g.index.equals(days), key
    global OPT_DAYS  # 2026-10-18 (user, new-notes 15): dates for sharpe_days
    OPT_DAYS = days

    # the MinGRU rows, exactly as the GUI builds them (rows._nn_daily: plain = f^2 B, the quote book's straddle)
    nn = gui._nn_daily()
    assert nn is not None, "the GUI's MinGRU loader found no forecasts"
    for key in (NN_BASE, NN_ALL):
        g = nn[nn["key"] == key].copy()
        g["day"] = pd.to_datetime(g["day"]).dt.normalize().astype("datetime64[ns]")
        g = g.set_index("day").sort_index().join(book, how="inner")
        assert g.index.equals(days), key
        # ratio = (g["target"] / g["plain"]).to_numpy(float)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        ratio = (g["rv_unclipped"] / g["plain"]).to_numpy(float)
        series[key] = pd.DataFrame(
            {
                "q": g["q"],
                "mid": g["ret_mid"],
                "x": g["ret_crossed"],
                "s": g["plain"] - g["iv_var"],
                "qlike": ratio - np.log(ratio) - 1.0,
                "R": g["R"],
                "ex": g["ex"],
                "bid": g["bid"],
                "ask": g["ask"],
            }
        )

    b = series[OLS]
    series[SHORT] = pd.DataFrame(
        {
            "q": -1.0,
            "mid": -b["R"],
            "x": 1.0 - b["ex"] / b["bid"],
            "R": b["R"],
            "ex": b["ex"],
            "bid": b["bid"],
            "ask": b["ask"],
        },
        index=days,
    )
    fam = {k: family_of(k, mt["family"].get(k)) for k in series}
    return mt, model_set, series, fam


def plain_16(path: Path) -> pd.Series:
    """16:00 ET rows of a yhat table: f^2 x B (make_rehaul_results.py plain_16; writeup/master_table_gui/rows.py)."""
    y = pd.read_parquet(path, columns=["t", "yhat", "baseline"])
    t = pd.to_datetime(y["t"], utc=True).dt.tz_convert("America/New_York")
    at16 = ((t.dt.hour == 16) & (t.dt.minute == 0)).to_numpy()
    v = y["yhat"].to_numpy(float)[at16] ** 2 * y["baseline"].to_numpy(float)[at16]
    day = pd.DatetimeIndex(t[at16].dt.tz_localize(None)).normalize()
    out = pd.Series(v, index=day)
    assert not out.index.duplicated().any(), path
    return out


def tree_16(est: str, pset: str, days: pd.DatetimeIndex, rv: np.ndarray) -> np.ndarray:
    """2026-10-22 (user, E34): the 16:00 plain of one base-5 pooled tree arm (pt13) on the trade days.  Gates: a
    forecast on every day, and the run's realized variance of the bar equals the OLS row's rv_unclipped (the same
    15:30-16:00 print)."""
    a = pt13.at16(est, pset)
    a.index = a.index.astype(days.dtype)
    a = a.reindex(days)
    p = a["plain"].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all(), (
        est,
        pset,
        int((~np.isfinite(p)).sum()),
    )
    assert np.allclose(a["rv"].to_numpy(float), rv, rtol=1e-12, atol=0.0), (est, pset)
    return p


def lin_16(est: str, pset: str, days: pd.DatetimeIndex, rv: np.ndarray) -> np.ndarray:
    """2026-10-22 (user, E34): the 16:00 plain of one pooled linear model of the 13-bar rerun (pl13) on the trade
    days, with the gates of tree_16."""
    a = pl13.at16(est, pset)
    a.index = a.index.astype(days.dtype)
    a = a.reindex(days)
    p = a["plain"].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all(), (
        est,
        pset,
        int((~np.isfinite(p)).sum()),
    )
    assert np.allclose(a["rv"].to_numpy(float), rv, rtol=1e-12, atol=0.0), (est, pset)
    return p


def add_pooled_linear(series, fam) -> dict:
    """2026-10-22 (user, E34: one design for the 12 pooled forecasts): the six pooled linear models of the 13-bar
    rerun on the trees' design (lin_16) as rows of their own, poolpt13_<est>_<set>, traded like add_pooled_trees (sign
    of plain - iv_var on the OLS row's days, the quote book's straddle).  They take the place of the master table's
    poolpt_<est>_<set> in every pooled row of the tables, figures and pooled statistics; the master rows stay in the
    model set as they are (as the master table's lgbm / xgb for the trees).  Gates on each master row they stand in
    for: its position, gap and QLIKE follow from its plain and rv_unclipped by the rules used here (so the new rows
    are traded and scored the same way), and its iv_var and rv_unclipped are the OLS row's.  Returns the master rows'
    Sharpe ratios (console and report only)."""
    daily = pd.read_parquet(
        RES / "master_table_daily.parquet",
        columns=["key", "day", "plain", "iv_var", "rv_unclipped"],
    )
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    b = series[OLS]
    days = b.index

    def on_days(key: str) -> pd.DataFrame:
        m = daily[daily["key"] == key].set_index("day").sort_index()
        m.index = m.index.astype(days.dtype)
        return m.reindex(days)

    ref = on_days(OLS)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    assert np.isfinite(iv).all() and np.isfinite(rv).all()
    old = {}
    for e in pl13.EST:
        for s in pl13.SETS:
            key = f"poolpt_{e}_{s}"
            g, m = series[key], on_days(key)
            assert np.array_equal(m["iv_var"].to_numpy(float), iv) and np.array_equal(
                m["rv_unclipped"].to_numpy(float), rv
            ), key
            p0 = m["plain"].to_numpy(float)
            r0 = rv / p0
            assert np.array_equal(
                g["q"].to_numpy(float), np.where(p0 > iv, 1.0, -1.0)
            ), key
            assert np.allclose(g["s"].to_numpy(float), p0 - iv, rtol=1e-12, atol=0.0), (
                key
            )
            assert np.allclose(
                g["qlike"].to_numpy(float),
                r0 - np.log(r0) - 1.0,
                rtol=1e-10,
                atol=1e-12,
            ), key
            old[key] = {
                "sh_mid": sharpe(g["mid"]),
                "sh_x": sharpe(g["x"]),
                "qlike": float(g["qlike"].mean()),
            }
            p = lin_16(e, s, days, rv)
            q = np.where(p > iv, 1.0, -1.0)
            ratio = rv / p
            f_s, y_s = p / iv, rv / iv
            new = f"poolpt13_{e}_{s}"
            assert new not in series, new
            series[new] = pd.DataFrame(
                {
                    "q": q,
                    "mid": q * b["R"].to_numpy(float),
                    "x": gui._crossed_return(
                        q, b["ex"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy()
                    ),
                    "s": p - iv,
                    "qlike": ratio - np.log(ratio) - 1.0,
                    "es1": np.abs(y_s - 1.0)
                    * ((np.minimum(f_s, y_s) <= 1.0) & (1.0 < np.maximum(f_s, y_s))),
                    "R": b["R"].to_numpy(float),
                    "ex": b["ex"].to_numpy(float),
                    "bid": b["bid"].to_numpy(float),
                    "ask": b["ask"].to_numpy(float),
                },
                index=days,
            )
            fam[new] = "pooled"
            n = series[new]
            print(
                f"E34 pooled linear {new} (13 bars, w1000, pl13) for {key}: SR mid {old[key]['sh_mid']:.2f} -> {sharpe(n['mid']):.2f}, "
                f"bid/ask {old[key]['sh_x']:.2f} -> {sharpe(n['x']):.2f}, long days {int((g['q'] > 0).sum())} -> "
                f"{int((q > 0).sum())} of {len(q)}"
            )
    return old


def add_pooled_trees(series, fam) -> dict:
    """2026-10-17 (user): the pooled intraday trees on the full predictor set, traded like every other row (sign of
    plain - iv_var on the OLS row's days, the quote book's straddle).  Gates: the 866-day mid Sharpe equals
    experiments/pooled_trees_score.py's levels.csv (as experiments/sharpe_tests_pooled.py checks), the LightGBM and
    XGBoost returns equal the master table's "lgbm" / "xgb" rows, and the printed Sharpe ratios equal \\rhrSh...PoolFull.
    2026-10-22 (user, E34): the base-5 trees (tree_16); the three gates above belong to the base-2 trees and are parked."""
    daily = pd.read_parquet(
        RES / "master_table_daily.parquet",
        columns=["key", "day", "iv_var", "rv_unclipped", "ret_mid", "ret_crossed"],
    )
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    b = series[OLS]
    days = b.index
    ref = daily[daily["key"] == OLS].set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    assert np.isfinite(iv).all() and np.isfinite(rv).all()
    out = {}
    # 2026-10-22 (user, E34): parked -- the base-2 / earlier-draft tables. Was:
    # for est, path in POOLTREE_ALL.items():
    #     pl = plain_16(path)
    #     pl.index = pl.index.astype(days.dtype)
    #     p = pl.reindex(days).to_numpy(float)
    #     assert np.isfinite(p).all() and (p > 0).all(), (est, int((~np.isfinite(p)).sum()))
    for est, _ in TREE_EST:
        p = tree_16(est, "all_features", days, rv)
        q = np.where(p > iv, 1.0, -1.0)
        mid = q * b["R"].to_numpy(float)
        x = gui._crossed_return(q, b["ex"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy())
        ratio = rv / p
        f_s, y_s = p / iv, rv / iv
        es1 = np.abs(y_s - 1.0) * ((np.minimum(f_s, y_s) <= 1.0) & (1.0 < np.maximum(f_s, y_s)))
        key = f"pooltree_{est}_all_features"
        series[key] = pd.DataFrame(
            {
                "q": q,
                "mid": mid,
                "x": x,
                "s": p - iv,
                "qlike": ratio - np.log(ratio) - 1.0,
                "es1": es1,
                "R": b["R"].to_numpy(float),
                "ex": b["ex"].to_numpy(float),
                "bid": b["bid"].to_numpy(float),
                "ask": b["ask"].to_numpy(float),
            },
            index=days,
        )
        fam[key] = "pooled"
        # 2026-10-22 (user, E34): parked -- the gates of the base-2 trees (levels.csv, the master table's lgbm / xgb rows,
        # POOLTREE_SR). Was:
        # lv = pd.read_csv(POOLTREE_DIR / POOLTREE_LEVELS[est] / "levels.csv")
        # want = float(
        #     lv[(lv["bucket"] == "all_features") & (lv["rung"] == "pooled") & lv["forecast"].str.startswith("pooled")][
        #         "sharpe_mid"
        #     ].iloc[0]
        # )
        # got = sharpe_days_252(mid)  # 2026-10-18 (user, new-notes 15): levels.csv stores the fixed-factor ratio. Was: sharpe_days(mid)
        # assert abs(got - want) < 1e-8, (key, got, want)
        # if est in POOLTREE_MT:
        #     m = daily[daily["key"] == POOLTREE_MT[est]].set_index("day").sort_index()
        #     m.index = m.index.astype(days.dtype)
        #     m = m.reindex(days)
        #     assert np.allclose(m["ret_mid"].to_numpy(float), mid, rtol=0, atol=1e-10), key
        #     assert np.allclose(m["ret_crossed"].to_numpy(float), x, rtol=0, atol=1e-10), key
        # assert round(sharpe_days_252(mid[WARMUP:]) + 0.0, 2) == POOLTREE_SR[est], (key, sharpe_days_252(mid[WARMUP:]))
        # print(f"pooled tree {key}: 866-day SR {got:.4f} = {POOLTREE_LEVELS[est]}/levels.csv; printed SR {sharpe(mid):.2f}")
        print(
            f"pooled tree {key} (base 5, pt13): printed SR {sharpe(mid):.2f} mid, {sharpe(x):.2f} bid/ask"
        )
        out[est] = key
    return out


def pooled_rank_series(series) -> dict:
    """2026-10-17 (user, E25): the \\rhrPoolN pooled intraday forecasts for the accuracy-vs-value rank correlations
    (\\rheRankCorrQPool, \\rheRankCorrESPool): the six pooled linear rows of the model set, the three full-set pooled
    trees of add_pooled_trees, and the three HAR-benchmark pooled trees (results/pooled_trees_2026-10-07, as
    make_rehaul_results.py loads them), the last traded exactly like add_pooled_trees (sign of plain - iv_var on the
    OLS row's days).  The HAR-benchmark trees are kept out of `series`, so no table or other macro sees them.
    2026-10-22 (user, E34): the HAR-benchmark trees are the base-5 arms of pt13 (tree_16), as the full-set ones; the
    six pooled linear rows are the 13-bar rerun on the trees' design (poolpt13_*, add_pooled_linear), no longer the
    model set's 48-interval rows."""
    daily = pd.read_parquet(
        RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"]
    )
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    b = series[OLS]
    days = b.index
    ref = daily[daily["key"] == OLS].set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    # 2026-10-22 (user, E34: one design for the 12 pooled forecasts): parked -- the master table's 48-interval rows. Was:
    # out = {f"poolpt_{e}_{s}": series[f"poolpt_{e}_{s}"] for e in ("enet", "lasso", "ridge") for s in ("baseline", "all_features")}
    out = {
        f"poolpt13_{e}_{s}": series[f"poolpt13_{e}_{s}"]
        for e in ("enet", "lasso", "ridge")
        for s in ("baseline", "all_features")
    }
    out.update({f"pooltree_{e}_all_features": series[f"pooltree_{e}_all_features"] for e in ("lgbm", "xgb", "rf")})
    for est in ("lgbm", "xgb", "rf"):
        # 2026-10-22 (user, E34): parked -- the base-2 HAR-benchmark trees. Was:
        # pl = plain_16(POOLTREE_DIR / f"yhat_pooltree_{est}_baseline.parquet")
        # pl.index = pl.index.astype(days.dtype)
        # p = pl.reindex(days).to_numpy(float)
        # assert np.isfinite(p).all() and (p > 0).all(), (est, int((~np.isfinite(p)).sum()))
        p = tree_16(est, "baseline", days, rv)
        q = np.where(p > iv, 1.0, -1.0)
        ratio = rv / p
        f_s, y_s = p / iv, rv / iv
        out[f"pooltree_{est}_baseline"] = pd.DataFrame(
            {
                "mid": q * b["R"].to_numpy(float),
                "qlike": ratio - np.log(ratio) - 1.0,
                "es1": np.abs(y_s - 1.0) * ((np.minimum(f_s, y_s) <= 1.0) & (1.0 < np.maximum(f_s, y_s))),
            },
            index=days,
        )
    assert len(out) == 12, len(out)
    return out


# ============================================================================= statistics
def trade_stats(series, key) -> dict:
    g, sh = series[key], series[SHORT]
    mid, x = g["mid"].to_numpy(float), g["x"].to_numpy(float)
    # 2026-10-20 (user, third-list 2): mean, hit, buys on the post-warmup days (were all 866 days; the
    # 866-day values stay as *_all for the master-table gate).
    q = g["q"].to_numpy(float)
    out = {
        "mean": float(mid[WARMUP:].mean()),
        "t_mean": nw_t(mid),
        "sh_mid": sharpe(mid),
        "sh_x": sharpe(x),
        "hit": float((mid[WARMUP:] > 0).mean()),
        "buys": float((q[WARMUP:] > 0).mean()),
        "mean_all": float(mid.mean()),
        "hit_all": float((mid > 0).mean()),
        "buys_all": float((q > 0).mean()),
    }
    if key != SHORT:
        d = mid - sh["mid"].to_numpy(float)
        out["dsh"] = out["sh_mid"] - sharpe(sh["mid"])
        out["t_dsh"] = nw_t(d)
        out["t_dsh_all"] = nw_t_all(d)
    return out


def gate_against_master(mt, series, keys):
    """Every recomputed E1 statistic must equal the master table's stored value.

    2026-10-17 (user, E17): the master table stores the Sharpe ratios on all 866 days, so the gate recomputes those
    with sharpe_days on the full series; the printed ratios (trade_stats) are on days t > WARMUP.
    """
    for key in keys:
        if key not in mt.index:
            continue
        st, row = trade_stats(series, key), mt.loc[key]
        g, sh = series[key], series[SHORT]
        # 2026-10-18 (user, new-notes 15): the master table stores fixed-factor ratios; the gate uses sharpe_days_252. Was: sharpe_days
        sh_mid_all = sharpe_days_252(g["mid"])
        pairs = [
            (sh_mid_all, row["Sharpe_mid"]),
            (sharpe_days_252(g["x"]), row["Sharpe_crossed"]),
            (st["mean_all"], row["mean_mid"]),  # third-list 2: was st["mean"] (now post-warmup)
            (st["hit_all"], row["hit_rate_mid"]),  # was st["hit"]
            (100 * st["buys_all"], row["pct_buy"]),  # was st["buys"]
        ]
        if key != SHORT:
            pairs += [
                (sh_mid_all - sharpe_days_252(sh["mid"]), row["dSharpe_mid_vs_short"]),  # 2026-10-18 (user, new-notes 15): was sharpe_days
                (st["t_dsh_all"], row["t_hac_mid_vs_short"]),  # third-list 2: was st["t_dsh"]
            ]
        for got, want in pairs:
            assert abs(got - float(want)) < 1e-8, (key, got, want)


def sizing_stats(series, key) -> dict:
    """The GUI's own sizing scorer (rows._score_sizes) on this forecast's gap, scored on days 253-866."""
    g = series[key]
    return gui._score_sizes(
        g["s"].to_numpy(float),
        g["R"].to_numpy(float),
        g["ex"].to_numpy(float),
        g["bid"].to_numpy(float),
        g["ask"].to_numpy(float),
        sharpe_fn=sharpe_days,  # 2026-10-18 (user, new-notes 15): per-year annualization; was the GUI default sqrt(252)
    )


def pair_stats(series, base, new) -> dict:
    a, b = series[new], series[base]
    return {
        "sh_b": sharpe(b["mid"]),
        "sh_a": sharpe(a["mid"]),
        "d_mid": sharpe(a["mid"]) - sharpe(b["mid"]),
        "t_mid": nw_t(a["mid"].to_numpy() - b["mid"].to_numpy()),
        "d_x": sharpe(a["x"]) - sharpe(b["x"]),
        "t_x": nw_t(a["x"].to_numpy() - b["x"].to_numpy()),
        # 2026-10-21 (user, decision #2): Ledoit-Wolf p of the Sharpe-ratio difference (stars and counts on dSharpe)
        # 2026-10-22 (user, D6): the stars and counts now use the bootstrap of the D6 difference (bt_*); the
        # Ledoit-Wolf p stays (lw_*), for the robustness table of the Sharpe-ratio definitions
        "lw_mid": lw_p(series, new, base, "mid"),
        "lw_x": lw_p(series, new, base, "x"),
        "bt_mid": boot_p(series, new, base, "mid"),
        "bt_x": boot_p(series, new, base, "x"),
    }


def tree_est(key: str) -> str:
    """lgbm / xgb / rf: the token sits at the end (subtree_tuned_all_features_lgbm) or after subtree_ (subtree_xgb_baseline)."""
    hits = [e for e, _ in TREE_EST if key.endswith(f"_{e}") or f"_{e}_" in key]
    assert len(hits) == 1, (key, hits)
    return hits[0]


def best_tree(series, fam, model_set, est) -> str:
    cands = [
        k
        for k in model_set
        if fam[k] == "tree" and "_all_features" in k and tree_est(k) == est
    ]
    return max(cands, key=lambda k: sharpe(series[k]["mid"]))


# 2026-10-15 (table legibility): parked -- code terms (sessions, refit, MSE-selected); replaced by tree_spec below
# def tree_rung(key: str) -> str:
#     """Plain-words configuration of a per-bar tree key (the master table's family plus the key's selection)."""
#     sel = "QLIKE-selected" if ("tunedq" in key or "optunaq" in key) else "MSE-selected"
#     if key.startswith("subtree_daily_"):
#         return "default settings, refit daily"
#     if key.startswith("subtree_tuned_daily_") or key.startswith(
#         "subtree_tunedq_daily_"
#     ):
#         return f"random search, refit daily, {sel}"
#     if key.startswith("subtree_tuned") and "optuna" not in key:
#         return f"random search, refit every 10 sessions, {sel}"
#     if key.startswith("subtree_optuna"):
#         tp = key.split("_tp")[1].split("_")[0]
#         k = (
#             key.split("_k")[1].split("_")[0]
#             if "_k" in key.split("_tp")[1][:8]
#             else "50"
#         )
#         every = "every session" if tp == "1" else f"every {tp} sessions"
#         return f"Optuna, retuned {every}, best of {k} trials, {sel}"
#     if key.startswith("subtree_"):
#         return "default settings, refit every 10 sessions"
#     raise ValueError(key)


def tree_spec(key: str) -> str:
    """Plain-words specification of a closing-interval tree key, in the guide's terms (trading days, re-estimated,
    hyperparameters selected on squared-error loss or QLIKE)."""
    loss = "QLIKE" if ("tunedq" in key or "optunaq" in key) else "squared-error loss"
    if key.startswith("subtree_daily_"):
        return "fixed hyperparameters, re-estimated daily"
    if key.startswith("subtree_tuned_daily_") or key.startswith(
        "subtree_tunedq_daily_"
    ):
        return f"random search, re-estimated daily, selected on {loss}"
    if key.startswith("subtree_tuned") and "optuna" not in key:
        return f"random search, re-estimated every 10 trading days, selected on {loss}"
    if key.startswith("subtree_optuna"):
        tp = key.split("_tp")[1].split("_")[0]
        k = (
            key.split("_k")[1].split("_")[0]
            if "_k" in key.split("_tp")[1][:8]
            else "50"
        )
        every = "daily" if tp == "1" else f"every {tp} trading days"
        return f"Optuna search ({k} trials), re-tuned {every}, selected on {loss}"
    if key.startswith("subtree_"):
        return "fixed hyperparameters, re-estimated every 10 trading days"
    raise ValueError(key)


# ============================================================================= LaTeX helpers
def num(x: float, dp: int = 2, plus: bool = False, bold: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    v = round(float(x), dp)
    if v == 0:
        v = 0.0  # no negative zero
    body = f"{abs(v):.{dp}f}"
    sign = "-" if v < 0 else ("+" if plus and v > 0 else "")
    s = f"${sign}{body}$"
    return f"{{\\bfseries\\boldmath {s}}}" if bold else s


def pct(x: float, dp: int = 1, bold: bool = False) -> str:
    return num(100 * x, dp, bold=bold)


# 2026-10-21 (user, decision #2): parked -- |t| > 1.96. Was: return bool(np.isfinite(t) and abs(t) > SIG)
def sig(t: float) -> bool:
    """A 614-day nw_t significant at 5 % (fixed-b)."""
    return bool(np.isfinite(t) and hac_p(t) < 0.05)


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", fr.rel(path))  # 2026-10-23: a test run writes outside the repository. Was: path.relative_to(ROOT)


HEADER = "% AUTO-GENERATED by writeup/make_rehaul_economics.py -- do not edit; re-run the script.\n"

# 2026-10-15 (table legibility): parked -- notes and panel helper of the old tables; replaced by the finance-style helpers below
# NOTE_FILLS = (
#     r"Mid: the straddle is bought or sold at the 15:30 midpoint; crossed: a buy pays the ask and a sale receives "
#     r"the bid, the spread paid at entry only (settlement is cash at intrinsic)."
# )
# NOTE_SHARPE = (
#     r"Sharpe ratio $=\sqrt{252}\,\bar r/\mathrm{sd}(r)$, annualised per trade day (the trade runs about 200 days a "
#     r"year)."
# )
# NOTE_HAC = (
#     r"Every $t$ is Newey--West (Bartlett kernel, lag $\lfloor 1.5\,n^{1/3}\rfloor = 14$ at $n = 866$); the $t$ of a "
#     r"$\Delta$Sharpe is that of the mean daily return difference, so it tests the mean difference, not the Sharpe "
#     r"difference. Bold: $|t| > 1.96$ (5\,\%, two-sided), on the estimate and its $t$."
# )
#
#
# def panel(title: str, ncol: int, first: bool = False) -> str:
#     return (
#         "" if first else "\\addlinespace[3pt]\n"
#     ) + f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n"


# ----------------------------------------------------------------------------- 2026-10-15 (table legibility)
# Finance-journal table style (writeup/NOTATION_GUIDE_2026-10-15.md section 8, TABLE_FIXES_FROM_FLOATS_2026-10-15.md):
# short caption; Notes of about five sentences below; stars for the 10/5/1 % levels on the estimate, its test statistic
# in parentheses underneath; at most about 7 columns; Panel A/B; the guide's vocabulary; numbers in the notes are the
# \rhe... macros of generated/rehaul_econ_numbers.tex, which every table \inputs if they are not yet defined.
LEVELS = (0.10, 0.05, 0.01)
STAR_NOTE = (
    r"$^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\%, 5\% and 1\% levels."
)
NUMBERS_GUARD = (
    "\\ifdefined\\rheNdays\\else\\input{generated/rehaul_econ_numbers}\\fi\n"
)
# 2026-10-15 (consolidation): the Newey-West lag is q_NW (q_t is the position; NOTATION_GUIDE section 6b).
# 2026-10-21 (user, decision #2): parked -- the lag-12 HAC t under each dSharpe. Was:
# HAC_NOTE = (
#     r"HAC $t$-statistic on the daily return differential (Newey--West, $q_{\mathrm{NW}} = \rheHACLag$; "
#     r"Section~\ref{sec:rh_methods_tests})"
# )
# 2026-10-22 (user, D6): parked -- the Ledoit-Wolf p under each dSharpe. Was:
# HAC_NOTE = (
#     r"$p$-value of the \citet{LedoitWolf2008} bootstrap test of equal Sharpe ratios on the \rheSzDays{} post-warmup "
#     r"days, in brackets (Section~\ref{sec:rh_methods_tests})"
# )
HAC_NOTE = (
    r"$p$-value of the paired block-bootstrap test of equal Sharpe ratios on the \rheSzDays{} post-warmup days, in "
    r"brackets (Section~\ref{sec:rh_methods_tests})"
)
MEAN_NOTE = (
    r"$t$-statistic of the mean return (Newey--West with bandwidth $S = \rheHACS$ and fixed-$b$ critical values, "
    r"\citealp{LazarusLewisStockWatson2018}; Section~\ref{sec:rh_methods_tests})"
)
# 2026-10-15 (consolidation): one name for the q_t = -1 benchmark (guide section 6b): "the short-straddle benchmark" in
# prose and notes, "Short straddle" as the row and legend label.
SHORT_LABEL = "Short straddle"


def p_two(t: float) -> float:
    """Two-sided normal p-value of a t or DM statistic (as dm_test and the paper's HAC t use)."""
    return math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else float("nan")


def p_digits(p: float) -> str:
    """2026-10-21 (user): the bare digits of a two-sided p for "$p = ...$" in prose (\\rheEnetP); 2 decimals from
    0.10, else 3, 4 when a cut would blur (make_rehaul_results.mpval's rule). Below 0.001 the digits would read 0.000,
    so that case stops the run: the sentence then needs "$p < 0.001$" instead."""
    assert np.isfinite(p) and 0.001 <= p <= 1.0, f"p = {p}: print p < 0.001 in the sentence instead"
    s = f"{p:.2f}" if p >= 0.10 else f"{p:.3f}"
    if s in ("0.10", "0.100", "0.050", "0.010"):  # keep the 10 / 5 / 1 % cuts visible
        s = f"{p:.4f}"
    return s


# 2026-10-21 (user, decision #2): parked -- stars from the normal p of the t. Was:
# def n_stars(t: float) -> int:
#     p = p_two(t)
#     return int(sum(p < a for a in LEVELS)) if np.isfinite(p) else 0
def n_stars(t: float) -> int:
    """Stars of a 614-day mean t, from its fixed-b p."""
    return n_stars_p(hac_p(t)) if np.isfinite(t) else 0


def n_stars_p(p: float) -> int:
    return int(sum(p < a for a in LEVELS)) if np.isfinite(p) else 0


def pbrack(p: float) -> str:
    """A p-value in brackets under its estimate, padded like a starred cell."""
    if p is None or not np.isfinite(p):
        return ""
    return f"$[{p:.3f}]^{{\\phantom{{***}}}}$"


def _fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0  # no negative zero
    return f"{'-' if v < 0 else ''}{abs(v):.{dp}f}"


def cell(x: float, dp: int = 2, stars: int = 0, starred: bool = True) -> str:
    """An estimate in math mode. In a starred column every cell carries a three-star superscript, the unused stars as
    a phantom, so the decimals line up in an r column."""
    if x is None or not np.isfinite(x):
        return "--$^{\\phantom{***}}$" if starred else "--"
    body = _fmt(x, dp)
    if not starred:
        return f"${body}$"
    pad = "*" * (3 - stars)
    sup = "*" * stars + (f"\\phantom{{{pad}}}" if pad else "")
    return f"${body}^{{{sup}}}$"


def tstat(t: float, dp: int = 2) -> str:
    """A test statistic in parentheses under its estimate, padded like a starred cell."""
    if t is None or not np.isfinite(t):
        return ""
    return f"$({_fmt(t, dp)})^{{\\phantom{{***}}}}$"


def count(k: int, n: int) -> str:
    """Number of pairs significant at the 5 % level, in brackets under a median (padded like a starred cell)."""
    return f"$[{k}/{n}]^{{\\phantom{{***}}}}$"


def rows2(label: str, top: list[str], under: list[str]) -> str:
    """One estimate row, then the row of test statistics under it (skipped when there is none)."""
    out = f"{label} & " + " & ".join(top) + " \\\\\n"
    if any(under):
        out += " & " + " & ".join(under) + " \\\\[2pt]\n"
    return out


def panel_head(title: str, ncol: int, first: bool = False) -> str:
    return (
        ("" if first else "\\addlinespace[4pt]\n")
        + f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n"
        + "\\addlinespace[2pt]\n"
    )


def sub_head(title: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{{title}}} \\\\\n"


def float_head(label: str, what: str) -> str:
    return HEADER + f"% {label} (outline 7.4): {what}\n" + NUMBERS_GUARD


# ============================================================================= E1
# 2026-10-15 (table legibility): parked -- 9-column E1 with bold significance; replaced by table_e1 below (stars, t under the estimate, panels)
# def table_e1(series, fam, model_set, trees) -> tuple[str, dict]:
#     st = {
#         k: trade_stats(series, k)
#         for k in [SHORT, OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]
#     }
#
#     def row(label: str, key: str) -> str:
#         s = st[key]
#         tm = sig(s["t_mean"])
#         cells = [
#             pct(s["mean"], 1, bold=tm),
#             num(s["t_mean"], bold=tm),
#             num(s["sh_mid"]),
#             num(s["sh_x"]),
#             pct(s["hit"]),
#             pct(s["buys"]),
#         ]
#         if key == SHORT:
#             cells += ["--", "--"]
#         else:
#             td = sig(s["t_dsh"])
#             cells += [num(s["dsh"], plus=True, bold=td), num(s["t_dsh"], bold=td)]
#         return f"{label} & " + " & ".join(cells) + " \\\\\n"
#
#     n_trees = {
#         e: sum(
#             1
#             for k in model_set
#             if fam[k] == "tree" and "_all_features" in k and tree_est(k) == e
#         )
#         for e, _ in TREE_EST
#     }
#     body = row(r"Always short ($q_t = -1$)", SHORT)
#     body += panel("Per-bar linear (16:00 bar)", 9)
#     body += row("OLS, HAR + calendar", OLS)
#     for k in PERBAR_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#     body += panel("Per-bar trees, all features", 9)
#     for est, name in TREE_EST:
#         body += row(f"{name}\\tnote{{a}}", trees[est])
#     body += panel("Per-bar neural net", 9)
#     body += row(r"MinGRU, all features\tnote{b}", NN_ALL)
#     body += panel("Pooled linear (all 48 bars)", 9)
#     for k in POOLED_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#
#     assert len(set(n_trees.values())) == 1, n_trees
#     tree_note = "; ".join(
#         f"{name.lower() if e == 'rf' else name}: {tree_rung(trees[e])}"
#         for e, name in TREE_EST
#     )
#     tex = (
#         HEADER
#         + rf"""% Table E1 (outline 7.4): always short and the key forecasts under sign(s), 866 trade days.
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{4.5pt}}
# \begin{{threeparttable}}
# \caption{{The closing trade under $q_t = \sign(s_t)$: always short and the key forecasts, 866 trade days.}}
# \label{{tab:rh_econ_control}}
# \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{2}}{{c}}{{Mid return}} & \multicolumn{{2}}{{c}}{{Sharpe ratio}} & & & \multicolumn{{2}}{{c}}{{vs always short}} \\
# \cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(l){{8-9}}
# Forecast & Mean (\%) & $t$ & Mid & Crossed & Hit (\%) & Buys (\%) & $\Delta$Sharpe & $t$ \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} One 0DTE SPXW straddle (a one-strike strangle when spot is between strikes) is bought
# ($q_t = +1$) or sold ($q_t = -1$) at 15:30 ET and held to the close; $r_t = q_t R_t$, with $R_t$ the return on the
# premium; 3 January 2020 to 30 April 2024. {NOTE_FILLS} Mean: the mean mid return, in \% of the premium. {NOTE_SHARPE}
# Hit: share of days with $r_t > 0$ at mid; Buys: share of days with $q_t = +1$. $\Delta$Sharpe: mid Sharpe ratio minus
# always short's ({num(st[SHORT]["sh_mid"])}). {NOTE_HAC}
# \item[a] Of each estimator's {n_trees["lgbm"]} configurations on all features in the model set, the one with the
# highest mid Sharpe ratio, so selected on the evaluation sample. {tree_note}.
# \item[b] Ridge-initialised linear shortcut plus a MinGRU branch, trained and selected on QLIKE; the mean of 5 seeds.
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     return tex, st


def tree_specs(fam, model_set) -> int:
    """Specifications per tree estimator on the full predictor set in the model set (equal across estimators)."""
    n_trees = {
        e: sum(
            1
            for k in model_set
            if fam[k] == "tree" and "_all_features" in k and tree_est(k) == e
        )
        for e, _ in TREE_EST
    }
    assert len(set(n_trees.values())) == 1, n_trees
    return n_trees["lgbm"]


# 2026-10-17 (user): parked -- note a of the closing-interval fixed daily rung (E1 tree rows, replaced by the pooled
# intraday trees). Was: "\item[a] Fixed hyperparameters from the pre-sample search, re-estimated every trading day,
# the specification of the closing-interval tree rows of the main results table; no ex-post selection."
# 2026-10-22 (user, E34): parked -- Table E1's Panel A sentence and notes a, b and dagger of the base-2 trees (now
# the base-5 trees of pt13: the 13 regular-hours bars, every arm complete, FOMC series in on the full set). Was:
# "Panel A: models estimated on every 30-minute interval of the day (the pooled intraday scheme); Panel B: ..."
# "\item[a] Pooled intraday tree ensembles: hyperparameters from an Optuna search, one configuration per method, not
# re-tuned; re-estimated at every 30-minute interval on a rolling window of $W = \rhdmPooledWindow$ intervals
# (Appendix~\ref{sec:rh_exp_schemes}). No ex-post selection."
# "\item[b] Full predictor set of the earlier draft's intraday specification, which lacks the FOMC predictors."
# "\item[$\dagger$] Interim: only the 16:00 forecasts of the full-set random forest's every-interval walk have been
# computed; the rest of the walk is still running."
# 2026-10-17 (user, E22/E15): parked -- ex-post note a of Table E1. Was: "\item[a] The specification with the highest
# $\SR$ at mid-quotes among the estimator's \rheTreeSpecs{{}} on the full predictor set, so selected ex post on the
# out-of-sample period. {tree_note[0].upper() + tree_note[1:]}."
def table_e1(series, fam, model_set, trees) -> tuple[str, dict]:
    st = {
        k: trade_stats(series, k)
        # for k in [SHORT, OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]  # 2026-10-15 (user): parked -- NN content parked for now
        for k in [SHORT, OLS, *PERBAR_ALL, *trees.values(), *POOLED_ALL]
    }

    def row(label: str, key: str) -> str:
        s = st[key]
        top = [
            cell(100 * s["mean"], 1, n_stars(s["t_mean"])),
            cell(s["sh_mid"], starred=False),
            cell(s["sh_x"], starred=False),
            cell(100 * s["hit"], 1, starred=False),
            cell(100 * s["buys"], 1, starred=False),
        ]
        under = [tstat(s["t_mean"]), "", "", "", ""]
        if key == SHORT:
            top.append(cell(float("nan")))
            under.append("")
        else:
            # 2026-10-21 (user, decision #2): Ledoit-Wolf p on dSharpe. Was: n_stars(s["t_dsh"]) and tstat(s["t_dsh"])
            # 2026-10-22 (user, D6): parked -- the Ledoit-Wolf p; now the bootstrap of the D6 difference. Was:
            # p_lw = lw_p(series, key, SHORT, "mid")
            # top.append(cell(s["dsh"], 2, n_stars_p(p_lw)))
            # under.append(pbrack(p_lw))
            p_bt = boot_p(series, key, SHORT, "mid")
            top.append(cell(s["dsh"], 2, n_stars_p(p_bt)))
            under.append(pbrack(p_bt))
        return rows2(label, top, under)

    tree_specs(
        fam, model_set
    )  # asserts the estimators have equally many specifications
    # 2026-10-17 (user): parked -- Panel A closing-interval (OLS, linear, closing trees), Panel B intraday (linear).
    # Now Panel A = the pooled intraday models, trees included; Panel B = the closing-interval models. Was:
    # body = row(SHORT_LABEL, SHORT)
    # body += panel_head("Panel A: Closing-interval models", 7)
    # body += row("OLS, HAR benchmark", OLS)
    # for k in PERBAR_ALL:
    #     body += row(EST_NAME[k.split("_")[1]], k)
    # for e, name in TREE_EST:
    #     body += row(f"{name}\\tnote{{a}}", trees[e])
    # body += panel_head("Panel B: Intraday models", 7)
    # for k in POOLED_ALL:
    #     body += row(EST_NAME[k.split("_")[1]], k)
    body = row(SHORT_LABEL, SHORT)
    body += panel_head(f"Panel A: {PANEL_POOL}", 7)
    for k in reversed(POOLED_ALL):  # ridge, lasso, elastic net
        # body += row(EST_NAME[k.split("_")[1]], k)  # 2026-10-22 (user, E34): parked -- no note mark (note a was the trees')
        body += row(f"{EST_NAME[k.split('_')[1]]}\\tnote{{{POOL_LIN_MARK}}}", k)
    for e, name in TREE_EST:
        body += row(f"{name}\\tnote{{{TREE_MARK[e]}}}", trees[e])
    body += panel_head(f"Panel B: {PANEL_CLOSE}", 7)
    body += row("OLS, HAR benchmark", OLS)
    for k in reversed(PERBAR_ALL):
        body += row(EST_NAME[k.split("_")[1]], k)
    # 2026-10-15 (user): parked -- NN content parked for now (the MinGRU row and its note b:
    # "\item[b] MinGRU network (ridge-initialized, trained and selected on QLIKE; Section~\ref{sec:rh_methods_models}).")
    # body += row(r"MinGRU\tnote{b}", NN_ALL)
    # 2026-10-17 (user): parked -- tree_spec() describes closing-interval keys only; the pooled trees' note is fixed text.
    # tree_note = "; ".join(
    #     f"{name if e != 'rf' else name.lower()}: {tree_spec(trees[e])}"
    #     for e, name in TREE_EST
    # )
    # 2026-10-22 (user, E34: one design for the 12 pooled forecasts): parked -- the Panel A sentence and note a of the
    # mixed designs. Was: "Panel A: models estimated on the 30-minute intervals of the day (the pooled intraday scheme;
    # the tree ensembles on the \rheTreeNBars{} regular-hours intervals, note a); ..." and "\item[a] Pooled intraday tree
    # ensembles, fitted on the design of the pooled linear models of Table~\ref{tab:rh_oos}: the \rheTreeNBars{}
    # regular-hours 30-minute intervals of each day, re-estimated at every interval on a rolling window of
    # \rheTreeWindowDays{} trading days ($W = \rheTreeWindowBars$ intervals); hyperparameters from an Optuna search, one
    # configuration per method, not re-tuned (Appendix~\ref{sec:rh_exp_schemes}). No ex-post selection."
    tex = (
        float_head("Table E1", "the sign strategy and the short-straddle benchmark")
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Returns of the sign strategy}}
\label{{tab:rh_econ_control}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
 & Mean & \multicolumn{{2}}{{c}}{{$\SR$}} & Hit & Long & \\
\cmidrule(lr){{3-4}}
 & return (\%) & Mid-quotes & Bid--ask & rate (\%) & days (\%) & $\Delta\SR$ \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Daily returns of the straddle position over the out-of-sample period, \rheSampleStart{{}} to
\rheSampleEnd{{}} ($T_{{\mathrm{{oos}}}} = \rheNdays$ trading days). The sign strategy holds
$q_t = \sign(\hat\sigma^2_t - \mathrm{{IV}}^2_t)$ straddles; the short-straddle benchmark (Short straddle) holds
$q_t \equiv -1$; every model except the
OLS uses the full predictor set. Mean return is at mid-quotes, in percent of the premium; $\SR$ is the annualized
Sharpe ratio at mid-quotes and at quoted bid/ask prices; hit rate and long days are the shares of days with
$r_t > 0$ at mid-quotes and with $q_t = +1$. $\Delta\SR$ is $\SR$ at mid-quotes minus that of the short-straddle
benchmark (\rheShortSh); in parentheses, the {MEAN_NOTE}; under $\Delta\SR$, the {HAC_NOTE}.
{STAR_NOTE}
Panel A: models estimated on the \rhePoolNBars{{}} regular-hours 30-minute intervals of each day (the pooled intraday
scheme, note a); Panel B: models
estimated on the closing interval only (one observation a day). Every model row trades the forecast of the 15:30--16:00 ET
variance made at 15:30.
\item[a] Pooled intraday models on one design: the \rhePoolNBars{{}} regular-hours 30-minute intervals of each day,
re-estimated at every interval on a rolling window of \rhePoolWindowDays{{}} trading days ($W = \rhePoolWindowBars$
intervals). The linear models and the tree ensembles are those of Table~\ref{{tab:rh_oos}}: the linear penalties are
selected on a pre-sample and frozen; the tree ensembles take their hyperparameters from an Optuna search, one
configuration per method, not re-tuned (Appendix~\ref{{sec:rh_exp_schemes}}). No ex-post selection.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    return tex, st


# ============================================================================= E2
# 2026-10-15 (table legibility): parked -- E2 with counts in the t columns and code terms; replaced by table_e2 below
# def table_e2(series, fam, model_set) -> tuple[str, dict]:
#     pairs = [
#         (k, k.replace("_baseline", "_all_features"))
#         for k in model_set
#         if "_baseline" in k
#     ]
#     pairs = [(b, a) for b, a in pairs if a in series]
#     ps = {b: pair_stats(series, b, a) for b, a in pairs}
#     nn = pair_stats(series, NN_BASE, NN_ALL)
#
#     def row(label: str, s: dict) -> str:
#         tm, tx = sig(s["t_mid"]), sig(s["t_x"])
#         return (
#             f"{label} & {num(s['sh_b'])} & {num(s['sh_a'])} & {num(s['d_mid'], plus=True, bold=tm)} & "
#             f"{num(s['t_mid'], bold=tm)} & {num(s['d_x'], plus=True, bold=tx)} & {num(s['t_x'], bold=tx)} \\\\\n"
#         )
#
#     def summary(label: str, keys: list[str]) -> str:
#         v = pd.DataFrame([ps[k] for k in keys])
#         n = len(v)
#         return (
#             f"{label} ({n}) & {num(v.sh_b.median())} & {num(v.sh_a.median())} & {num(v.d_mid.median(), plus=True)} & "
#             f"{int((v.t_mid > SIG).sum())}/{n} & {num(v.d_x.median(), plus=True)} & {int((v.t_x > SIG).sum())}/{n} \\\\\n"
#         )
#
#     body = panel("Per-bar linear (16:00 bar)", 7, first=True)
#     for est in ("enet", "lasso", "ridge"):
#         body += row(EST_NAME[est], ps[f"subpt_{est}_baseline"])
#     body += panel("Per-bar neural net", 7)
#     body += row("MinGRU", nn)
#     body += panel("Pooled linear (all 48 bars)", 7)
#     for est in ("enet", "lasso", "ridge"):
#         body += row(EST_NAME[est], ps[f"poolpt_{est}_baseline"])
#     body += panel(
#         r"Per-bar trees: medians over configurations; the $t$ columns count $t > 1.96$",
#         7,
#     )
#     tree_keys = [b for b in ps if fam[b] == "tree"]
#     for est, name in TREE_EST:
#         body += summary(name, [b for b in tree_keys if tree_est(b) == est])
#     body += summary("All trees", tree_keys)
#     body += "\\midrule\n"
#     body += summary("Model set, all pairs", list(ps))
#
#     for est in (
#         "enet",
#         "lasso",
#         "ridge",
#     ):  # the note says the HAR + calendar per-bar fits trade as the OLS
#         assert (series[f"subpt_{est}_baseline"]["q"] == series[OLS]["q"]).all(), est
#     v = pd.DataFrame(ps.values())
#     info = {
#         "n": len(v),
#         "up": int((v.d_mid > 0).sum()),
#         "sig": int((v.t_mid > SIG).sum()),
#         "sig_neg": int((v.t_mid < -SIG).sum()),
#         "up_x": int((v.d_x > 0).sum()),
#         "med": float(v.d_mid.median()),
#         "med_x": float(v.d_x.median()),
#         "n_tree": len(tree_keys),
#         "sig_keys": [b for b in ps if ps[b]["t_mid"] > SIG],
#         "pairs": ps,
#         "nn": nn,
#     }
#     tex = (
#         HEADER
#         + rf"""% Table E2 (outline 7.4): Sharpe change from HAR + calendar to all features, within each family.
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{5pt}}
# \begin{{threeparttable}}
# \caption{{Adding the exogenous inputs: the Sharpe ratio from HAR + calendar to all features, within each family
# ($q_t = \sign(s_t)$, 866 trade days).}}
# \label{{tab:rh_econ_allfeat}}
# \begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{2}}{{c}}{{Sharpe ratio, mid}} & \multicolumn{{4}}{{c}}{{All features minus HAR + calendar}} \\
# \cmidrule(lr){{2-3}}\cmidrule(l){{4-7}}
#  & HAR + cal. & All & $\Delta$Sharpe mid & $t$ & $\Delta$Sharpe crossed & $t$ \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} Each pair is one model fitted on HAR + calendar (\texttt{{baseline}}) and on all features,
# with nothing else changed; the change is all features minus HAR + calendar on the same days. The per-bar linear
# models are pre-tuned; on HAR + calendar all three take the OLS's position
# on every day, so they share its Sharpe ratio. The tree rows summarise
# {len(tree_keys)} pairs (every tuning configuration of the model set, {len(tree_keys) // 3} per estimator): the Sharpe
# and $\Delta$Sharpe columns are medians, and each $t$ column gives the number of pairs with $t > 1.96$. The model-set
# row covers the {info["n"]} pairs of the 112-forecast model set (the MinGRU is not in it): $\Delta$Sharpe mid
# $> 0$ in {info["up"]} of {info["n"]}, none significantly negative. {NOTE_FILLS} {NOTE_SHARPE} {NOTE_HAC}
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     assert info["sig_neg"] == 0, info["sig_neg"]
#     return tex, info


def table_e2(series, fam, model_set) -> tuple[str, dict]:
    pairs = [
        (k, k.replace("_baseline", "_all_features"))
        for k in model_set
        if "_baseline" in k
    ]
    pairs = [(b, a) for b, a in pairs if a in series]
    ps = {b: pair_stats(series, b, a) for b, a in pairs}
    # 2026-10-15 (user): NN content parked for now -- the MinGRU pair is off the table (console summary only)
    nn = pair_stats(series, NN_BASE, NN_ALL)

    def row(label: str, s: dict) -> str:
        top = [
            cell(s["sh_b"], starred=False),
            cell(s["sh_a"], starred=False),
            # 2026-10-21 (user, decision #2): Ledoit-Wolf p on dSharpe. Was: n_stars(s["t_mid"]), n_stars(s["t_x"]) and
            # tstat(s["t_mid"]), tstat(s["t_x"]) underneath
            # 2026-10-22 (user, D6): parked -- the Ledoit-Wolf p (lw_mid, lw_x); now the D6 bootstrap p (bt_mid, bt_x)
            cell(s["d_mid"], 2, n_stars_p(s["bt_mid"])),
            cell(s["d_x"], 2, n_stars_p(s["bt_x"])),
        ]
        return rows2(label, top, ["", "", pbrack(s["bt_mid"]), pbrack(s["bt_x"])])

    def summary(label: str, keys: list[str]) -> str:
        v = pd.DataFrame([ps[k] for k in keys])
        n = len(v)
        top = [
            cell(v.sh_b.median(), starred=False),
            cell(v.sh_a.median(), starred=False),
            cell(v.d_mid.median()),
            cell(v.d_x.median()),
        ]
        under = [
            "",
            "",
            # 2026-10-21 (user, decision #2): Ledoit-Wolf. Was: (v.t_mid > SIG), (v.t_x > SIG)
            # 2026-10-22 (user, D6): parked -- the Ledoit-Wolf p (v.lw_mid, v.lw_x); now the D6 bootstrap p
            count(int(((v.bt_mid < 0.05) & (v.d_mid > 0)).sum()), n),
            count(int(((v.bt_x < 0.05) & (v.d_x > 0)).sum()), n),
        ]
        return rows2(label, top, under)

    tree_keys = [b for b in ps if fam[b] == "tree"]
    n_each = {e: sum(tree_est(b) == e for b in tree_keys) for e, _ in TREE_EST}
    assert len(set(n_each.values())) == 1, (
        n_each
    )  # the notes give one count per estimator
    body = panel_head("Panel A: Closing-interval models", 5, first=True)
    for e in ("enet", "lasso", "ridge"):
        body += row(EST_NAME[e], ps[f"subpt_{e}_baseline"])
    # 2026-10-15 (user): parked -- NN content parked for now (the MinGRU row and its note a:
    # "\item[a] MinGRU network (ridge-initialized, trained and selected on QLIKE); not in the model set.")
    # body += row(r"MinGRU\tnote{a}", nn)
    body += panel_head("Panel B: Intraday models", 5)
    # 2026-10-22 (user, E34): parked -- the master table's 48-interval pooled linear pairs (model-set pairs). Was:
    # for e in ("enet", "lasso", "ridge"):
    #     body += row(EST_NAME[e], ps[f"poolpt_{e}_baseline"])
    # The pooled rows are the 13-bar rerun on the trees' design (poolpt13_*), outside the model set (its summary row
    # below keeps the master rows).
    ps13 = {
        f"poolpt13_{e}_baseline": pair_stats(
            series, f"poolpt13_{e}_baseline", f"poolpt13_{e}_all_features"
        )
        for e in ("enet", "lasso", "ridge")
    }
    for e in ("enet", "lasso", "ridge"):
        body += row(EST_NAME[e], ps13[f"poolpt13_{e}_baseline"])
    body += panel_head("Panel C: Medians across pairs", 5)
    for e, name in TREE_EST:
        body += summary(
            f"{name} (\\rheTreePairsEach)", [b for b in tree_keys if tree_est(b) == e]
        )
    body += summary(r"All trees (\rheTreePairs)", tree_keys)
    body += summary(r"Model set (\rhePairs)", list(ps))

    for e in (
        "enet",
        "lasso",
        "ridge",
    ):  # the note says the HAR-benchmark closing-interval fits trade as the OLS
        assert (series[f"subpt_{e}_baseline"]["q"] == series[OLS]["q"]).all(), e
    v = pd.DataFrame(ps.values())
    info = {
        "n": len(v),
        "up": int((v.d_mid > 0).sum()),
        # 2026-10-21 (user, decision #2): Sharpe-ratio differences significant by the Ledoit-Wolf test (were HAC t > 1.96
        # and < -1.96 of the mean return differential): \rhePairsSig, \rhePairsSigNeg
        # 2026-10-22 (user, D6): parked -- the Ledoit-Wolf p (v.lw_mid, v.lw_x); now the D6 bootstrap p (bt_*). The
        # Ledoit-Wolf counts stay as sig_lw / sig_neg_lw (console)
        "sig": int(((v.bt_mid < 0.05) & (v.d_mid > 0)).sum()),
        "sig_neg": int(((v.bt_mid < 0.05) & (v.d_mid < 0)).sum()),
        "sig_neg_x": int(((v.bt_x < 0.05) & (v.d_x < 0)).sum()),
        "sig_lw": int(((v.lw_mid < 0.05) & (v.d_mid > 0)).sum()),
        "sig_neg_lw": int(((v.lw_mid < 0.05) & (v.d_mid < 0)).sum()),
        "up_x": int((v.d_x > 0).sum()),
        "med": float(v.d_mid.median()),
        "med_x": float(v.d_x.median()),
        "n_tree": len(tree_keys),
        "n_tree_each": n_each["lgbm"],
        # was ps[b]["t_mid"] > SIG; 2026-10-22 (user, D6): was ps[b]["lw_mid"] < 0.05
        "sig_keys": [b for b in ps if ps[b]["bt_mid"] < 0.05 and ps[b]["d_mid"] > 0],
        "pairs": ps,
        "pairs13": ps13,  # 2026-10-22 (user, E34): Panel B, the pooled linear pairs on the trees' design
        "nn": nn,
    }
    neg_clause = (
        "none is significantly negative at that level"
        if info["sig_neg"] == 0 and info["sig_neg_x"] == 0
        else f"significantly negative: {info['sig_neg']} at mid-quotes, {info['sig_neg_x']} at quoted bid/ask prices"
    )
    tex = (
        float_head("Table E2", "the full predictor set against the HAR benchmark")
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Sharpe ratios: HAR benchmark versus full predictor set}}
\label{{tab:rh_econ_allfeat}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{$\SR$ at mid-quotes}} & \multicolumn{{2}}{{c}}{{$\Delta\SR$}} \\
\cmidrule(lr){{2-3}}\cmidrule(l){{4-5}}
 & HAR benchmark & Full set & Mid-quotes & Bid--ask \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Each pair is one model estimated on the HAR benchmark and on the full predictor set, with
everything else unchanged, over the out-of-sample period ($T_{{\mathrm{{oos}}}} = \rheNdays$ trading days). $\SR$ is the
annualized Sharpe ratio of the sign strategy; $\Delta\SR$ is the full-set minus the HAR-benchmark $\SR$, at
mid-quotes and at quoted bid/ask prices, with the {HAC_NOTE}. On the HAR benchmark the three
closing-interval linear models hold the same position as the closing-interval OLS on every day, so they share its
$\SR$. Panel C gives medians across the closing-interval tree pairs (every hyperparameter specification in the model
set) and across all pairs of the \rheModelSetN{{}} forecasts of the model set, with the number of pairs whose
Sharpe ratio difference is significantly positive at the 5\% level in brackets ({neg_clause}). {STAR_NOTE}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    # 2026-10-21 (user, decision #2): parked -- the note's "none is significantly negative" asserted on the HAC t; the
    # note now prints the Ledoit-Wolf counts. Was: assert info["sig_neg"] == 0 and int((v.t_x < -SIG).sum()) == 0
    print(
        f"E2: dSharpe significantly negative (D6 bootstrap, 5 %): {info['sig_neg']} at mid, {info['sig_neg_x']} at bid/ask"
    )
    print(
        f"D6 E2: significant at mid (bootstrap / Ledoit-Wolf): gains {info['sig']} / {info['sig_lw']}, losses "
        f"{info['sig_neg']} / {info['sig_neg_lw']}"
    )
    return tex, info


# ============================================================================= E3
# 2026-10-15 (table legibility): parked -- 9-column E3; replaced by table_e3 below (t and DM under the estimates)
# def table_e3(series) -> tuple[str, list]:
#     rows_ = []
#     body = ""
#     for set_, set_label in (
#         ("all_features", "All features"),
#         ("baseline", "HAR + calendar"),
#     ):
#         body += panel(set_label, 9, first=set_ == "all_features")
#         for est in ("enet", "lasso", "ridge"):
#             p, b = series[f"poolpt_{est}_{set_}"], series[f"subpt_{est}_{set_}"]
#             s = pair_stats(series, f"poolpt_{est}_{set_}", f"subpt_{est}_{set_}")
#             dq = 100 * (b["qlike"].mean() / p["qlike"].mean() - 1)
#             dm = dm_test(b["qlike"].to_numpy(float), p["qlike"].to_numpy(float))["dm"]
#             tm, tx, td = sig(s["t_mid"]), sig(s["t_x"]), sig(dm)
#             body += (
#                 f"{EST_NAME[est]} & {num(s['sh_b'])} & {num(s['sh_a'])} & {num(s['d_mid'], plus=True, bold=tm)} & "
#                 f"{num(s['t_mid'], bold=tm)} & {num(s['d_x'], plus=True, bold=tx)} & {num(s['t_x'], bold=tx)} & "
#                 f"{num(dq, 1, plus=True, bold=td)} & {num(dm, bold=td)} \\\\\n"
#             )
#             rows_.append(dict(est=est, set=set_, **s, dq=dq, dm=dm))
#     tex = (
#         HEADER
#         + rf"""% Table E3 (outline 7.4): pooled (48 bars) vs per-bar (16:00 bar) linear fits, the trade and QLIKE.
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{4.5pt}}
# \begin{{threeparttable}}
# \caption{{Per-bar against pooled linear fits: the trade and accuracy ($q_t = \sign(s_t)$, 866 trade days).}}
# \label{{tab:rh_econ_pooled}}
# \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{2}}{{c}}{{Sharpe ratio, mid}} & \multicolumn{{6}}{{c}}{{Per-bar minus pooled}} \\
# \cmidrule(lr){{2-3}}\cmidrule(l){{4-9}}
#  & Pooled & Per-bar & $\Delta$Sharpe mid & $t$ & $\Delta$Sharpe crossed & $t$ & $\Delta$QLIKE (\%) & DM \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} Pooled: one model fitted on all 48 half-hour bars of the day, read at the 16:00 bar; per-bar:
# fitted on the 16:00 bar alone; both pre-tuned, on the same inputs and days. $\Delta$QLIKE: per-bar mean QLIKE over
# pooled, minus 1, in \%; DM: Diebold--Mariano statistic of the daily QLIKE difference (Harvey--Leybourne--Newbold
# corrected, automatic Newey--West lag), negative when the per-bar fit is more accurate. {NOTE_FILLS} {NOTE_SHARPE}
# {NOTE_HAC} Bold DM: $|\mathrm{{DM}}| > 1.96$.
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     return tex, rows_


def table_e3(series) -> tuple[str, list]:
    rows_ = []
    body = ""
    for set_, set_label in (
        ("all_features", "Panel A: Full predictor set"),
        ("baseline", "Panel B: HAR benchmark"),
    ):
        body += panel_head(set_label, 6, first=set_ == "all_features")
        for e in ("enet", "lasso", "ridge"):
            # p, b = series[f"poolpt_{e}_{set_}"], series[f"subpt_{e}_{set_}"]  # 2026-10-22 (user, E34): parked -- 48-interval rows
            p, b = series[f"poolpt13_{e}_{set_}"], series[f"subpt_{e}_{set_}"]
            # 2026-10-15 (consolidation): parked -- closing-interval minus pooled intraday; the paper's one convention
            # is now pooled intraday minus closing-interval, as in Table T2 (tab:rh_axis2) and Results.
            # s = pair_stats(series, f"poolpt_{e}_{set_}", f"subpt_{e}_{set_}")
            # dq = 100 * (b["qlike"].mean() / p["qlike"].mean() - 1)
            # dmr = dm_test(b["qlike"].to_numpy(float), p["qlike"].to_numpy(float))
            # s = pair_stats(series, f"subpt_{e}_{set_}", f"poolpt_{e}_{set_}")  # 2026-10-22 (user, E34): parked
            s = pair_stats(series, f"subpt_{e}_{set_}", f"poolpt13_{e}_{set_}")
            dq = 100 * (p["qlike"].mean() / b["qlike"].mean() - 1)
            dmr = dm_test(p["qlike"].to_numpy(float), b["qlike"].to_numpy(float))
            dm = dmr["dm"]
            top = [
                cell(
                    s["sh_a"], starred=False
                ),  # pooled intraday (the "new" leg of the pair)
                cell(s["sh_b"], starred=False),  # closing interval
                # 2026-10-21 (user, decision #2): Ledoit-Wolf p on dSharpe (was n_stars / tstat of the HAC t)
                # 2026-10-22 (user, D6): parked -- the Ledoit-Wolf p (lw_mid, lw_x); now the D6 bootstrap p
                cell(s["d_mid"], 2, n_stars_p(s["bt_mid"])),
                cell(s["d_x"], 2, n_stars_p(s["bt_x"])),
                cell(dq, 1, int(sum(dmr["p"] < a for a in LEVELS))),
            ]
            under = ["", "", pbrack(s["bt_mid"]), pbrack(s["bt_x"]), tstat(dm)]
            body += rows2(EST_NAME[e], top, under)
            rows_.append(
                dict(est=e, set=set_, **s, dq=dq, dm=dm, dm_lag=dmr["hac_lag"])
            )
    assert len({r["dm_lag"] for r in rows_}) == 1, rows_  # one lag in the notes
    tex = (
        float_head("Table E3", "closing-interval against intraday linear models")
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Closing-interval versus intraday models}}
\label{{tab:rh_econ_pooled}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{$\SR$ at mid-quotes}} & \multicolumn{{3}}{{c}}{{Intraday minus closing-interval}} \\
\cmidrule(lr){{2-3}}\cmidrule(l){{4-6}}
 & Pooled & Closing & $\Delta\SR$, & $\Delta\SR$, & \\
 & intraday & interval & mid-quotes & bid--ask & $\Delta\QLIKE$ (\%) \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Each row compares the closing-interval model with the intraday model of the same
estimator and predictor set, both with hyperparameters selected on a pre-sample, over the out-of-sample period
($T_{{\mathrm{{oos}}}} = \rheNdays$ trading days); the intraday model is evaluated on the closing interval. $\SR$
is the annualized Sharpe ratio of the sign strategy; $\Delta\SR$ is the intraday minus the closing-interval
$\SR$, at mid-quotes and at quoted bid/ask prices, with the {HAC_NOTE}. $\Delta\QLIKE$ is the
intraday model's mean QLIKE relative to the closing-interval model's, in percent, with the Diebold--Mariano
statistic (Harvey--Leybourne--Newbold correction, Newey--West $q_{{\mathrm{{NW}}}} = \rheDMLag$) in parentheses; positive values favor
the closing-interval model, as in Table~\ref{{tab:rh_axis2}}. {STAR_NOTE}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    return tex, rows_


# ============================================================================= E4
RULES = (
    ("sign", r"$\sign(s)$"),
    ("clip", "Clip"),
    ("lfss", "Long/scaled"),
    ("rank", r"Rank $|s|$"),
)


# 2026-10-15 (table legibility): parked -- 9-column E4 (mid and crossed side by side); replaced by table_e4 below (Panel A mid, Panel B bid-ask)
# def table_e4(series, model_set, trees) -> tuple[str, dict]:
#     sz = {k: sizing_stats(series, k) for k in model_set}
#     for k in [OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]:
#         sz.setdefault(k, sizing_stats(series, k))
#     short_mid = sharpe(series[SHORT]["mid"].to_numpy()[WARMUP:])
#     short_x = sharpe(series[SHORT]["x"].to_numpy()[WARMUP:])
#     n_days = len(series[SHORT]) - WARMUP
#
#     def row(label: str, key: str) -> str:
#         s = sz[key]
#         mids = [num(s[r]["sharpe_mid"]) for r, _ in RULES]
#         xs = [num(s[r]["sharpe_crossed"]) for r, _ in RULES]
#         return f"{label} & " + " & ".join(mids + xs) + " \\\\\n"
#
#     body = (
#         rf"Always short ($q_t = -1$) & \multicolumn{{4}}{{c}}{{{num(short_mid)} under every rule}} & "
#         rf"\multicolumn{{4}}{{c}}{{{num(short_x)} under every rule}} \\" + "\n"
#     )
#     body += panel("Per-bar linear (16:00 bar)", 9)
#     body += row("OLS, HAR + calendar", OLS)
#     for k in PERBAR_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#     body += panel(
#         "Per-bar trees, all features (as in Table~\\ref{tab:rh_econ_control})", 9
#     )
#     for est, name in TREE_EST:
#         body += row(name, trees[est])
#     body += panel("Per-bar neural net", 9)
#     body += row("MinGRU, all features", NN_ALL)
#     body += panel("Pooled linear (all 48 bars)", 9)
#     for k in POOLED_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#
#     ms = {r: [sz[k][r] for k in model_set] for r, _ in RULES}
#     med_mid = {r: float(np.median([v["sharpe_mid"] for v in ms[r]])) for r, _ in RULES}
#     med_x = {
#         r: float(np.median([v["sharpe_crossed"] for v in ms[r]])) for r, _ in RULES
#     }
#     beat_mid = {
#         r: int(sum(v["d_vs_sign"] > 0 for v in ms[r])) for r, _ in RULES if r != "sign"
#     }
#     beat_x = {
#         r: int(sum(v["d_vs_sign_x"] > 0 for v in ms[r]))
#         for r, _ in RULES
#         if r != "sign"
#     }
#     med_gain = {
#         r: float(np.median([v["d_vs_sign"] for v in ms[r]]))
#         for r, _ in RULES
#         if r != "sign"
#     }
#     n = len(model_set)
#     body += "\\midrule\n"
#     body += (
#         f"Model set ({n}): median & "
#         + " & ".join(
#             [num(med_mid[r]) for r, _ in RULES] + [num(med_x[r]) for r, _ in RULES]
#         )
#         + " \\\\\n"
#     )
#     body += (
#         "Model set: above $\\sign(s)$ & -- & "
#         + " & ".join(f"{beat_mid[r]}" for r, _ in RULES if r != "sign")
#         + " & -- & "
#         + " & ".join(f"{beat_x[r]}" for r, _ in RULES if r != "sign")
#         + " \\\\\n"
#     )
#     info = dict(
#         med_mid=med_mid,
#         med_x=med_x,
#         beat_mid=beat_mid,
#         beat_x=beat_x,
#         med_gain=med_gain,
#         n=n,
#         short_mid=short_mid,
#         short_x=short_x,
#         n_days=n_days,
#         sz=sz,
#     )
#     head = " & ".join(lab for _, lab in RULES)
#     tex = (
#         HEADER
#         + rf"""% Table E4 (outline 7.4): sizing rules, scored on days 253-866 exactly as the GUI (rows._score_sizes).
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{4pt}}
# \begin{{threeparttable}}
# \caption{{Sizing rules: Sharpe ratio on days {WARMUP + 1}--{WARMUP + n_days} ({n_days} days).}}
# \label{{tab:rh_econ_sizing}}
# \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{4}}{{c}}{{Mid}} & \multicolumn{{4}}{{c}}{{Crossed}} \\
# \cmidrule(lr){{2-5}}\cmidrule(l){{6-9}}
# Forecast & {head} & {head} \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} Position $q_t$ in straddles, from the gap $s_t = \hat\sigma^2_t - (IV^{{30}}_t)^2$ and the
# earlier days only: $\sign(s)$, $q_t = \pm 1$; Clip (robust clip), $q_t = \clip(s_t/m_t, -1, 1)$ with $m_t$ the median
# $|s_u|$ over $u < t$; Long/scaled (full long, scaled short), $q_t = 1$ if $s_t > 0$, else $\max(s_t/m_t, -1)$; Rank
# $|s|$, $q_t = \sign(s_t)$ times the share of earlier days with $|s_u| \le |s_t|$. The first {WARMUP} days only build the
# scale, so every rule, $\sign(s)$ included, is scored on the last {n_days} of the 866 days. Crossed: the spread is paid
# on $|q_t|$ straddles. Model-set rows: the median over the {n} forecasts of the model set (the MinGRU is not in it), and
# the number of them whose Sharpe ratio is above their own $\sign(s)$ Sharpe ratio. Point estimates; no test.
# {NOTE_SHARPE}
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     return tex, info


# 2026-10-17 (user): parked -- E4 note "the trees are the specifications of Table~\ref{tab:rh_econ_control}" (closing-interval
# trees); the tree rows are now the pooled intraday trees.
def table_e4(series, model_set, trees) -> tuple[str, dict]:
    sz = {k: sizing_stats(series, k) for k in model_set}
    # for k in [OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]:  # 2026-10-15 (user): parked -- NN content parked for now
    for k in [OLS, *PERBAR_ALL, *trees.values(), *POOLED_ALL]:
        sz.setdefault(k, sizing_stats(series, k))
    # 2026-10-17 (user, E17): sharpe() already slices [WARMUP:]; use sharpe_days() here on the explicit slice
    short_mid = sharpe_days(series[SHORT]["mid"].to_numpy()[WARMUP:])
    short_x = sharpe_days(series[SHORT]["x"].to_numpy()[WARMUP:])
    n_days = len(series[SHORT]) - WARMUP

    ms = {r: [sz[k][r] for k in model_set] for r, _ in RULES}
    med_mid = {r: float(np.median([v["sharpe_mid"] for v in ms[r]])) for r, _ in RULES}
    med_x = {
        r: float(np.median([v["sharpe_crossed"] for v in ms[r]])) for r, _ in RULES
    }
    beat_mid = {
        r: int(sum(v["d_vs_sign"] > 0 for v in ms[r])) for r, _ in RULES if r != "sign"
    }
    beat_x = {
        r: int(sum(v["d_vs_sign_x"] > 0 for v in ms[r]))
        for r, _ in RULES
        if r != "sign"
    }
    med_gain = {
        r: float(np.median([v["d_vs_sign"] for v in ms[r]]))
        for r, _ in RULES
        if r != "sign"
    }
    n = len(model_set)

    def block(field: str, short_v: float, med: dict, beat: dict) -> str:
        def row(label: str, key: str) -> str:
            cells = [cell(sz[key][r][field], starred=False) for r, _ in RULES]
            return f"\\quad {label} & " + " & ".join(cells) + " \\\\\n"

        out = (
            rf"{SHORT_LABEL} & \multicolumn{{4}}{{c}}{{{cell(short_v, starred=False)} under every rule}} \\"
            + "\n"
        )
        # 2026-10-17 (user): parked -- closing trees under "Closing-interval models"; the tree rows are now the pooled
        # intraday trees, listed with the intraday models (first, as in Table E1). Was:
        # out += sub_head("Closing-interval models", 5)
        # out += row("OLS, HAR benchmark", OLS)
        # for k in PERBAR_ALL:
        #     out += row(EST_NAME[k.split("_")[1]], k)
        # for e, name in TREE_EST:
        #     out += row(name, trees[e])
        # out += sub_head("Intraday models", 5)
        # for k in POOLED_ALL:
        #     out += row(EST_NAME[k.split("_")[1]], k)
        # out += row("MinGRU", NN_ALL)  # 2026-10-15 (user): parked -- NN content parked for now
        out += sub_head(PANEL_POOL, 5)
        for k in reversed(POOLED_ALL):
            out += row(EST_NAME[k.split("_")[1]], k)
        for e, name in TREE_EST:
            out += row(name + ("$^\\dagger$" if e in INTERIM_TREES else ""), trees[e])
        out += sub_head(PANEL_CLOSE, 5)
        out += row("OLS, HAR benchmark", OLS)
        for k in reversed(PERBAR_ALL):
            out += row(EST_NAME[k.split("_")[1]], k)
        out += "\\addlinespace[2pt]\n"
        out += (
            r"Model set, median & "
            + " & ".join(cell(med[r], starred=False) for r, _ in RULES)
            + " \\\\\n"
        )
        out += (
            r"Model set, number above sign & -- & "
            + " & ".join(f"{beat[r]}" for r, _ in RULES if r != "sign")
            + " \\\\\n"
        )
        return out

    body = panel_head("Panel A: $\\SR$ at mid-quotes", 5, first=True)
    body += block("sharpe_mid", short_mid, med_mid, beat_mid)
    body += panel_head("Panel B: $\\SR$ at quoted bid/ask prices", 5)
    body += block("sharpe_crossed", short_x, med_x, beat_x)

    info = dict(
        med_mid=med_mid,
        med_x=med_x,
        beat_mid=beat_mid,
        beat_x=beat_x,
        med_gain=med_gain,
        n=n,
        short_mid=short_mid,
        short_x=short_x,
        n_days=n_days,
        sz=sz,
    )
    # 2026-10-15 (user): parked -- NN content parked for now: the note's "(the MinGRU is not in it)" after "forecasts of
    # the model set"
    # 2026-10-22 (user, E34): parked -- the interim forest (the base-5 walk is complete). Was: "... of
    # Table~\ref{tab:rh_econ_control} ($^\dagger$interim: only the forest's 16:00 forecasts; see that table)."
    # 2026-10-22 (user, E34: one design for the 12 pooled forecasts): parked -- "the trees are the pooled intraday trees
    # of Table~\ref{tab:rh_econ_control} (note a there)"; note a there now describes every pooled row.
    tex = (
        float_head(
            "Table E4",
            "position-sizing rules, scored exactly as the GUI (rows._score_sizes)",
        )
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Sharpe ratios by position-sizing rule}}
\label{{tab:rh_econ_sizing}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
 & & Robust & Full long, & \\
 & Sign & clip & scaled short & Rank \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Annualized Sharpe ratios of the straddle position $q_t$ under four sizing rules
(Section~\ref{{sec:rh_econ_strategy}}), each a function of $\hat\sigma^2_t - \mathrm{{IV}}^2_t$ and of its values on
earlier days $t' < t$ only; the sign rule is the sign strategy, and the short-straddle benchmark (Short straddle)
holds $q_t \equiv -1$ under every rule. The first \rheSzWarmup{{}} trading days of the out-of-sample period only estimate the
scale of $|\hat\sigma^2_{{t'}} - \mathrm{{IV}}^2_{{t'}}|$, so every rule is evaluated on the remaining \rheSzDays{{}} of
the \rheNdays{{}} trading days; at quoted bid/ask prices the bid--ask spread is paid on $|q_t|$ straddles. Every model except the OLS uses the full predictor set; the pooled
intraday models are those of Table~\ref{{tab:rh_econ_control}} (note a there). The last two rows of each panel give the median across the
\rheModelSetN{{}} forecasts of the model set and the number of them whose $\SR$ is above that of
their own sign strategy. Point estimates; no test is reported.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    return tex, info


# ============================================================================= 2026-10-22 (user, D6): definitions
def table_sharpe_defs(dfd: pd.DataFrame) -> str:
    """Robustness of the Sharpe-ratio definition (main(): the six pooled classes, full set minus HAR, mid-quotes, 614
    post-warmup days): D6 with its bootstrap p, D1, D2 with the Ledoit-Wolf p, and D6 on each side of 16 May 2022."""
    body = ""
    for _, r in dfd.iterrows():
        top = [
            cell(r["d6"], 2, n_stars_p(r["bt"])),
            cell(r["d1"], starred=False),
            cell(r["d2"], 2, n_stars_p(r["lw"])),
            cell(r["pre"], starred=False),
            cell(r["post"], starred=False),
        ]
        body += rows2(r["name"], top, [pbrack(r["bt"]), "", pbrack(r["lw"]), "", ""])
    return (
        float_head(
            "Table (D6 robustness)",
            "the Sharpe-ratio change of the pooled models under other definitions",
        )
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Sharpe ratio change under alternative definitions of the Sharpe ratio}}
\label{{tab:rh_econ_sharpe_defs}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
 & \multicolumn{{3}}{{c}}{{All \rheSzDays{{}} post-warmup days}} & \multicolumn{{2}}{{c}}{{$\SR$ by subsample}} \\
\cmidrule(lr){{2-4}}\cmidrule(l){{5-6}}
 & $\SR$ & Per-year, & Pooled, & Before & From \\
 & (paper) & $N_y$-weighted & $\sqrt{{252}}$ & \rheSplitDate & \rheSplitDate \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Each row is a pooled intraday model class; each entry is the Sharpe ratio of the sign strategy
at mid-quotes on the full predictor set minus that on the HAR benchmark. $\SR$ is the paper's Sharpe ratio
(Section~\ref{{sec:rh_econ_measures}}): the equal-weight mean over calendar years of $\SR_y$, each year annualized by
the square root of its option days per calendar year; in brackets, the $p$-value of the paired block-bootstrap test
of equal Sharpe ratios (Section~\ref{{sec:rh_methods_tests}}). Per-year, $N_y$-weighted: each year's ratio scaled by
the square root of its number of option days $N_y$, the years weighted by $N_y$. Pooled, $\sqrt{{252}}$: the ratio of
the mean to the standard deviation of all \rheSzDays{{}} daily returns, times $\sqrt{{252}}$; in brackets, the
$p$-value of the \citet{{LedoitWolf2008}} test, which tests the equality of these pooled ratios. The last two columns
give $\SR$ on the \rheSplitPreN{{}} post-warmup days before \rheSplitDate{{}}, the first week with an expiration on
every weekday, and on the \rheSplitPostN{{}} days from that date, each subsample annualized as a sample of its own.
{STAR_NOTE}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )


# ============================================================================= figures
PREVIEW_DIR: Path | None = (
    None  # --preview DIR: also write PNG previews there (never into writeup/figures)
)


def _preview(fig, stem: str) -> None:
    if PREVIEW_DIR is not None:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_DIR / f"{stem}.png", dpi=200)


def legend_handle(fam: str, marker: bool = True):
    from matplotlib.lines import Line2D

    short = fam == "short"
    return Line2D(
        [],
        [],
        color=FAM_COLOR[fam],
        lw=1.1,
        ls="--" if short else "-",
        marker=FAM_MARKER[fam] if marker else None,
        ms=4.5 if short else 4,
        mfc=FAM_COLOR[fam],
        mec=FAM_COLOR[fam] if short else "white",
        mew=1.0 if short else 0.5,
    )


def _style(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, lw=0.4, alpha=0.8)
    ax.set_axisbelow(True)


def _save(fig, stem: str) -> None:
    w, h = fig.get_size_inches()
    assert abs(w - FIG_W) < 1e-9, (
        stem,
        w,
    )  # placed at \linewidth = 100 %, so 9 pt stays 9 pt
    fig.savefig(FIG / f"{stem}.pdf")
    _preview(fig, stem)
    plt.close(fig)
    # 2026-10-23: the directory written (final_run.FIG). Was: f"wrote writeup/figures/{stem}.pdf ..."
    print(f"wrote {fr.rel(FIG / f'{stem}.pdf')} ({w:.2f} x {h:.2f} in)")


def fig_cumpnl(series, key4: dict) -> dict:
    """Top: cumulative mid return for always short and one forecast per family. Bottom: cumulative gain over always
    short, each forecast's 10 best days of gain marked."""
    from matplotlib.lines import Line2D

    days = series[SHORT].index
    short = series[SHORT]["mid"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, 4.4), sharex=True, layout="constrained"
    )
    for ax in (ax1, ax2):
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        _style(ax)
    ax1.plot(days, short.cumsum(), color=FAM_COLOR["short"], lw=1.1, ls="--", zorder=3)
    info = {}
    for fam_, (key, _label) in key4.items():
        g = series[key]
        c = FAM_COLOR[fam_]
        ax1.plot(
            days,
            g["mid"].cumsum(),
            color=c,
            lw=1.1,
            zorder=4,
            marker=FAM_MARKER[fam_],
            markevery=(40, 120),
            ms=3.5,
            mec="white",
            mew=0.4,
        )
        edge = g["mid"] - short
        ce = edge.cumsum()
        ax2.plot(days, ce, color=c, lw=1.1, zorder=4)
        top = edge.nlargest(TOP_DAYS).index
        assert (g.loc[top, "q"] > 0).all(), key  # a gain over always short needs a buy
        ax2.plot(
            top,
            ce.loc[top],
            ls="none",
            marker=FAM_MARKER[fam_],
            ms=4.5,
            mfc=c,
            mec="white",
            mew=0.5,
            zorder=5,
        )
        info[key] = {
            "edge": float(edge.sum()),
            "edge_ex10": float(edge.sum() - edge.nlargest(TOP_DAYS).sum()),
            "final": float(g["mid"].sum()),
            "minR_top10": float(g.loc[top, "R"].min()),
        }
    info[SHORT] = {"final": float(short.sum())}
    # ax1.set_ylabel("Cumulative return\n(premia, mid)")  # 2026-10-15 (table legibility): parked -- guide terms
    # ax2.set_ylabel("Cumulative gain over\nalways short (premia)")  # 2026-10-15 (table legibility): parked -- ditto
    ax1.set_ylabel("Cumulative return\n(straddle premia, mid-quotes)")
    # 2026-10-15 (consolidation): parked -- one name for the q_t = -1 benchmark (guide section 6b)
    # ax2.set_ylabel("Cumulative gain over always\nshort (straddle premia)")
    ax2.set_ylabel("Cumulative gain over the short\nstraddle (straddle premia)")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(days[0] - pd.Timedelta(days=10), days[-1] + pd.Timedelta(days=10))
    handles = [legend_handle(f, marker=f != "short") for f in key4] + [
        legend_handle("short", marker=False)
    ]
    labels = [lab for _, (_, lab) in key4.items()] + [SHORT_LABEL]
    # 2026-10-15 (table legibility): one row of five no longer fits the longer model-class labels; three columns,
    # read row by row (matplotlib fills legends by column, so interleave)
    # order_ = [0, 3, 1, 4, 2]  # 2026-10-15 (user): parked -- NN content parked for now (five entries with the MinGRU)
    # three models and the short straddle (five entries with the MinGRU); row 1 the models, row 2 the short straddle
    assert len(handles) == 4, len(handles)
    order_ = [0, 3, 1, 2]
    fig.legend(
        [handles[i] for i in order_],
        [labels[i] for i in order_],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    ax2.legend(
        [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=4.5,
                mfc="#888888",
                mec="white",
                mew=0.5,
            )
        ],
        # ["each forecast's 10 best days of gain"],  # 2026-10-15 (table legibility): parked -- wording
        [f"each model's {TOP_DAYS} days of largest gain"],
        loc="upper left",
        frameon=False,
        handletextpad=0.3,
    )
    fig.align_ylabels((ax1, ax2))
    _save(fig, "rehaul_econ_cumpnl")
    return info


def fig_sizing(
    series, fam, model_set, sz: dict, short_mid: float, short_x: float, med_mid, med_x
) -> None:
    from matplotlib.lines import Line2D

    rule_names = [
        "$\\mathrm{sign}(s)$",
        "Robust clip",
        "Full long,\nscaled short",
        "Rank of $|s|$",
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, 2.9), sharey=True, layout="constrained"
    )
    rng = np.random.default_rng(20261015)
    jit = {k: rng.uniform(-0.26, 0.26) for k in model_set}
    order = [
        "tree",
        "pooled",
        "linear",
    ]  # trees (90) first, so the 22 linear and pooled forecasts stay visible
    for ax, field, shv, med in (
        (axes[0], "sharpe_mid", short_mid, med_mid),
        (axes[1], "sharpe_crossed", short_x, med_x),
    ):
        _style(ax, "x")
        for i, (r, _) in enumerate(RULES):
            y0 = len(RULES) - 1 - i
            for f in order:
                ks = [k for k in model_set if fam[k] == f]
                ax.scatter(
                    [sz[k][r][field] for k in ks],
                    [y0 + jit[k] for k in ks],
                    s=12,
                    color=FAM_COLOR[f],
                    marker=FAM_MARKER[f],
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=0.9,
                    zorder=3,
                )
            # 2026-10-15 (user): parked -- NN content parked for now
            # v = sz[NN_ALL][r][field]
            # ax.scatter(
            #     [v],
            #     [y0],
            #     s=30,
            #     color=FAM_COLOR["nn"],
            #     marker=FAM_MARKER["nn"],
            #     edgecolors="white",
            #     linewidths=0.6,
            #     zorder=6,
            # )
            ax.plot(
                [med[r], med[r]],
                [y0 - 0.34, y0 + 0.34],
                color="#111111",
                lw=1.8,
                zorder=5,
                solid_capstyle="butt",
            )
            ax.text(
                med[r],
                y0 + 0.37,
                f"{med[r]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111111",
                zorder=7,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9),
            )
        ax.axvline(shv, color=FAM_COLOR["short"], lw=0.9, ls="--", zorder=2)
        ax.set_ylim(-0.5, len(RULES) - 0.2)
    axes[0].set_yticks(range(len(RULES)))
    axes[0].set_yticklabels(rule_names[::-1])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    axes[0].set_xlabel("Sharpe ratio, mid")
    axes[1].set_xlabel("Sharpe ratio, crossed")
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker=FAM_MARKER[f],
            ms=4.5,
            mfc=FAM_COLOR[f],
            mec="white",
            mew=0.35,
        )
        for f in ("linear", "tree", "pooled")
    ] + [
        # 2026-10-15 (user): parked -- NN content parked for now
        # Line2D(
        #     [],
        #     [],
        #     ls="none",
        #     marker=FAM_MARKER["nn"],
        #     ms=5.5,
        #     mfc=FAM_COLOR["nn"],
        #     mec="white",
        #     mew=0.6,
        # ),
        Line2D([], [], color="#111111", lw=1.8),
        Line2D([], [], color=FAM_COLOR["short"], lw=0.9, ls="--"),
    ]
    labels = [
        FAM_LABEL["linear"],
        FAM_LABEL["tree"],
        FAM_LABEL["pooled"],
        # "Per-bar MinGRU (not in the set)",  # 2026-10-15 (user): parked -- NN content parked for now
        "Model-set median",
        SHORT_LABEL,
    ]
    # three columns, read row by row: matplotlib fills legends by column, so interleave
    # order_ = [0, 3, 1, 4, 2, 5]  # 2026-10-15 (user): parked -- NN content parked for now (six entries)
    # row 1: the three classes; row 2: the median and the short straddle
    order_ = [0, 3, 1, 4, 2]
    fig.legend(
        [handles[i] for i in order_],
        [labels[i] for i in order_],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    _save(fig, "rehaul_econ_sizing")


def fig_byyear(series, model_set, key4: dict) -> dict:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    days = series[SHORT].index
    years = sorted(set(days.year))
    yr = days.year

    # 2026-10-22 (user, D6): parked -- one year: SR_y = mean_y / sd_y * sqrt(N_y). Was:
    #     return [sharpe_days(np.asarray(x)[yr == y], dates=days[yr == y]) for y in years]
    def by_year(x: pd.Series) -> list[float]:
        # 2026-10-18 (user, new-notes 15): one year: SR_y = mean_y / sd_y * sqrt(N_y). Was: [sharpe_days(x[yr == y]) for y in years]
        # 2026-10-22 (user, D6): one year's term of D6 on the figure's sample (the 866 option days): SR_y = mean_y /
        # sd_y * sqrt(A_y), A_y = N_y over the fraction of year y the sample covers
        by = hi.d6_sharpe(np.asarray(x, float), dates=days)["by_year"]
        assert sorted(by) == list(years), (sorted(by), years)
        return [by[y][2] for y in years]

    set_by = np.array([by_year(series[k]["mid"]) for k in model_set])
    q25, q50, q75 = np.percentile(set_by, [25, 50, 75], axis=0)
    fig, ax = plt.subplots(figsize=(FIG_W, 2.9), layout="constrained")
    xs = np.arange(len(years))
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.fill_between(xs, q25, q75, color="#e8e8e8", lw=0, zorder=1)
    ax.plot(xs, q50, color="#9a9a9a", lw=0.9, zorder=1.5)
    out = {
        "years": years,
        "set_med": q50.tolist(),
        "set_q25": q25.tolist(),
        "set_q75": q75.tolist(),
    }
    fams = list(key4)
    offs = np.linspace(-0.16, 0.16, len(fams) + 1)
    sh_short = by_year(series[SHORT]["mid"])
    out[SHORT] = sh_short
    ax.plot(
        xs + offs[-1],
        sh_short,
        color=FAM_COLOR["short"],
        lw=1.0,
        ls="--",
        marker=FAM_MARKER["short"],
        ms=4.5,
        mew=1.0,
        zorder=3,
    )
    for j, f in enumerate(fams):
        key = key4[f][0]
        v = by_year(series[key]["mid"])
        out[key] = v
        ax.plot(
            xs + offs[j],
            v,
            color=FAM_COLOR[f],
            lw=1.0,
            marker=FAM_MARKER[f],
            ms=4.5,
            mec="white",
            mew=0.5,
            zorder=4,
        )
    n_by = [int((yr == y).sum()) for y in years]
    labs = [f"{y}\n({n} days)" for y, n in zip(years, n_by)]
    labs[-1] = f"{years[-1]}, Jan.–Apr.\n({n_by[-1]} days)"
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(years) - 0.5)
    ax.set_ylabel("Sharpe ratio, mid, within year")
    _style(ax)
    handles = (
        [legend_handle(f) for f in fams]
        + [legend_handle("short")]
        + [Patch(color="#e8e8e8", lw=0), Line2D([], [], color="#9a9a9a", lw=0.9)]
    )
    labels = [key4[f][1] for f in fams] + [
        SHORT_LABEL,
        "Model set: interquartile range",
        "Model set: median",
    ]
    # four columns, read row by row (matplotlib fills by column): row 1 the forecasts, row 2 the references
    # order_ = [0, 4, 1, 5, 2, 6, 3]  # 2026-10-15 (user): parked -- NN content parked for now (four forecasts, ncol=4)
    # three forecasts and three references (seven entries with the MinGRU); three columns: row 1 the forecasts, row 2
    # the references
    assert len(handles) == 6, len(handles)
    order_ = [0, 3, 1, 4, 2, 5]
    fig.legend(
        [handles[i] for i in order_],
        [labels[i] for i in order_],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    out["n_by"] = n_by
    _save(fig, "rehaul_econ_byyear")
    return out


# ============================================================================= main
def main() -> None:
    mt, model_set, series, fam = load()
    gate_against_master(mt, series, [SHORT, *model_set])
    # 2026-10-22 (user, E34): the pooled linear rows on the trees' design (poolpt13_*), after the gate of the master rows
    # they stand in for
    add_pooled_linear(series, fam)

    # 2026-10-17 (user, E22/E15): parked -- ex-post best-Sharpe tree selection (one rule: the tree rows use the
    # fixed-hyperparameter, daily re-estimation rung of the main results table, make_rehaul_results.py FAMILY_ROWS).
    # Was: trees = {est: best_tree(series, fam, model_set, est) for est, _ in TREE_EST}
    # 2026-10-17 (user): parked -- closing-interval fixed daily rung in the tree rows; the rows are now the pooled
    # intraday trees (add_pooled_trees). Was:
    # trees = {est: f"subtree_daily_all_features_{est}" for est, _ in TREE_EST}
    # assert all(k in model_set and k in series for k in trees.values()), trees
    trees = add_pooled_trees(series, fam)
    close_fixed = {est: f"subtree_daily_all_features_{est}" for est, _ in TREE_EST}  # for \\rheLgbmCloseFixedSh
    assert all(k in model_set and k in series for k in close_fixed.values()), close_fixed
    e1, st = table_e1(series, fam, model_set, trees)
    e2, e2i = table_e2(series, fam, model_set)
    e3, e3r = table_e3(series)
    e4, e4i = table_e4(series, model_set, trees)

    # ---- the asserts the outline fixes (rounded as printed)
    # 2026-10-17 (user, E17): Sharpe ratios recomputed on the 614 post-warmup days (t > WARMUP = 252).
    # Pre-E17 (866-day) values parked: SHORT (0.20, -0.27); enet (1.92, 1.43); NN_ALL (1.53, 1.05);
    # NN_BASE (1.04, 0.57); e2i (51, 50, 4, 0.54).
    r2 = lambda x: round(x + 0.0, 2)  # noqa: E731
    # 2026-10-18 (user, new-notes 15): the Sharpe values of record below are fixed-factor (sqrt 252); the ratios are per-year now,
    # so these gates report instead of assert (soft). Was:
    # assert (r2(st[SHORT]["sh_mid"]), r2(st[SHORT]["sh_x"])) == (0.34, -0.03), st[SHORT]
    # assert (
    #     r2(st["subpt_enet_all_features"]["sh_mid"]),
    #     r2(st["subpt_enet_all_features"]["sh_x"]),
    # ) == (1.93, 1.56)
    def soft(got, was, what):
        print(("NN15 holds: " if got == was else "NN15 CHANGED: ") + f"{what} {got} (was {was})")

    soft((r2(st[SHORT]["sh_mid"]), r2(st[SHORT]["sh_x"])), (0.34, -0.03), "SHORT sh_mid/sh_x")
    soft((r2(st["subpt_enet_all_features"]["sh_mid"]), r2(st["subpt_enet_all_features"]["sh_x"])), (1.93, 1.56), "enet sh_mid/sh_x")
    # 2026-10-15 (user): NN content parked for now -- the MinGRU is off E1 (st), so its stats are computed here, for
    # the gate and its \rheNN... macros (kept defined, unused by the floats)
    # assert (r2(st[NN_ALL]["sh_mid"]), r2(st[NN_ALL]["sh_x"])) == (1.53, 1.05), st[NN_ALL]
    assert NN_ALL not in st
    nn_st = trade_stats(series, NN_ALL)
    # assert (r2(nn_st["sh_mid"]), r2(nn_st["sh_x"])) == (1.59, 1.22), nn_st  # 2026-10-18 (user, new-notes 15): soft
    soft((r2(nn_st["sh_mid"]), r2(nn_st["sh_x"])), (1.59, 1.22), "NN_ALL sh_mid/sh_x")
    nb = trade_stats(series, NN_BASE)
    # assert (r2(nb["sh_mid"]), r2(nb["sh_x"])) == (1.07, 0.71), nb  # 2026-10-18 (user, new-notes 15): soft
    soft((r2(nb["sh_mid"]), r2(nb["sh_x"])), (1.07, 0.71), "NN_BASE sh_mid/sh_x")
    meds = tuple(r2(e4i["med_mid"][r]) for r, _ in RULES)
    # assert meds == (0.92, 1.34, 1.12, 1.34), meds  # 2026-10-18 (user, new-notes 15): soft
    soft(meds, (0.92, 1.34, 1.12, 1.34), "E4 sizing medians")
    # outline cross-checks
    # 2026-10-18 (user, new-notes 15): soft. Was: assert (e2i["n"] == 51 and e2i["up"] == 47 and e2i["sig"] == 4
    #     and r2(e2i["med"]) == 0.62), e2i; assert e4i["beat_mid"]["clip"] == 98; assert r2(e4i["short_mid"]) == 0.34
    assert e2i["n"] == 51, e2i
    soft((e2i["up"], e2i["sig"], r2(e2i["med"])), (47, 4, 0.62), "E2 up/sig/median dSharpe")
    soft(e4i["beat_mid"]["clip"], 98, "E4 clip beats sign")
    soft(r2(e4i["short_mid"]), 0.34, "E4 short_mid")
    # 2026-10-21 (user, decision #2): parked -- t > 1.96 (lag-12 t, normal). Now the LLSW t with fixed-b p < 0.05 and a
    # gain (t > 0). Was: sig_vs_short = sum(trade_stats(series, k)["t_dsh"] > SIG for k in model_set)
    def gain_sig(k: str) -> bool:
        t_ = trade_stats(series, k)["t_dsh"]
        return bool(t_ > 0 and hac_p(t_) < 0.05)

    sig_vs_short = sum(gain_sig(k) for k in model_set)
    # 2026-10-20 (user, third-list 2): parked -- 20 on the 866-day HAC t (lag 14); now on the 614 post-warmup days. Was:
    # assert sig_vs_short == 20, sig_vs_short
    print(f"decision #2: model-set forecasts whose mean gain over short is significant (fixed-b p < 0.05): {sig_vs_short}"
          " (20 on the 866-day lag-14 t)")
    # 2026-10-21 (user): of those, the forecasts on the full predictor set (\rheSigVsShortFull; Section 7 and Appendix F
    # print both counts, no longer "all ... use the full predictor set")
    sig_vs_short_full = sum(
        gain_sig(k) and re.search(r"_all_features(?=_|$)", k) is not None for k in model_set
    )
    assert sig_vs_short_full <= sig_vs_short, (sig_vs_short_full, sig_vs_short)
    print(f"model-set forecasts with a significant mean gain over short on the full predictor set: {sig_vs_short_full} of {sig_vs_short}")
    above_vs_short = sum(trade_stats(series, k)["dsh"] > 0 for k in model_set)
    # assert above_vs_short == 94, above_vs_short  # 2026-10-18 (user, new-notes 15): soft
    soft(above_vs_short, 94, "model-set dsh vs short > 0")

    # one forecast per family for the figures
    # 2026-10-15 (table legibility): parked -- code terms (per-bar); the guide's model classes below
    # key4 = {
    #     "linear": ("subpt_enet_all_features", "Per-bar elastic net"),
    #     "tree": (trees["lgbm"], "Per-bar LightGBM"),
    #     "nn": (NN_ALL, "Per-bar MinGRU"),
    #     "pooled": ("poolpt_enet_all_features", "Pooled elastic net"),
    # }
    key4 = {
        "linear": ("subpt_enet_all_features", "Closing-interval elastic net"),
        # "tree": (trees["lgbm"], "Closing-interval LightGBM"),  # 2026-10-17 (user): parked -- now the pooled tree
        "tree": (trees["lgbm"], "Intraday LightGBM"),
        # "nn": (NN_ALL, "Closing-interval MinGRU"),  # 2026-10-15 (user): parked -- NN content parked for now
        # "pooled": ("poolpt_enet_all_features", "Intraday elastic net"),  # 2026-10-22 (user, E34): parked -- 48-interval row
        "pooled": ("poolpt13_enet_all_features", "Intraday elastic net"),
    }
    # 2026-10-18 (user, new-notes 15): under the per-year Sharpe ratio the pooled XGBoost (1.26) is above the LightGBM
    # (1.21); the figure keeps the LightGBM until the user picks (reported, not raised). Was:
    # if sharpe(series[trees["xgb"]]["mid"]) > sharpe(series[trees["lgbm"]]["mid"]):
    #     raise ValueError("the best tree is not the LightGBM: relabel key4['tree']")
    if sharpe(series[trees["xgb"]]["mid"]) > sharpe(series[trees["lgbm"]]["mid"]):
        print("NN15 CHANGED: the best pooled tree is not the LightGBM (figure keeps key4['tree'] = LightGBM)")
    cum = fig_cumpnl(series, key4)
    # 2026-10-15 (user): NN content parked for now -- the MinGRU is off the figure; its gain over the short straddle
    # (\rheGainNN, \rheGainExNN: kept defined, unused) as fig_cumpnl computes it
    nn_edge = series[NN_ALL]["mid"] - series[SHORT]["mid"]
    cum[NN_ALL] = {
        "edge": float(nn_edge.sum()),
        "edge_ex10": float(nn_edge.sum() - nn_edge.nlargest(TOP_DAYS).sum()),
    }
    fig_sizing(
        series,
        fam,
        model_set,
        e4i["sz"],
        e4i["short_mid"],
        e4i["short_x"],
        e4i["med_mid"],
        e4i["med_x"],
    )
    yrs = fig_byyear(series, model_set, key4)

    # tail: the gain over always short without each forecast's 10 best days, over the model set
    neg_ex10 = 0
    for k in model_set:
        edge = series[k]["mid"] - series[SHORT]["mid"]
        neg_ex10 += int(edge.sum() - edge.nlargest(TOP_DAYS).sum() < 0)

    # ---- figure floats
    # 2026-10-15 (table legibility): parked -- the caption's numbers are now \rheGain... macros (see below)
    # def ex10(f):
    #     return num(cum[key4[f][0]]["edge_ex10"], 1, plus=True)
    #
    # def tot(f):
    #     return num(cum[key4[f][0]]["edge"], 1, plus=True)

    # fam_order = ("linear", "tree", "nn", "pooled")  # 2026-10-15 (user): parked -- NN content parked for now
    fam_order = ("linear", "tree", "pooled")
    min_r = min(cum[key4[f][0]]["minR_top10"] for f in fam_order)
    # 2026-10-15 (table legibility): parked -- code terms (closing trade, sign(s), at mid, per-bar, family) and typed
    # numbers; the caption below uses the guide's terms and the \rhe... macros
    # lab = {
    #     "linear": "per-bar elastic net",
    #     "tree": "per-bar LightGBM",
    #     "nn": "per-bar MinGRU",
    #     "pooled": "pooled elastic net",
    # }
    # totals = ", ".join(f"{lab[f]} {tot(f)}" for f in fam_order)
    # ex10s = ", ".join(f"{ex10(f)}" for f in fam_order)
    # fig1 = (
    #     HEADER
    #     + rf"""\begin{{figure}}[tbp]
    # \centering
    # \includegraphics[width=\linewidth]{{figures/rehaul_econ_cumpnl.pdf}}
    # \caption{{The closing trade over time, $q_t = \sign(s_t)$ at mid, for one forecast per family (all features) and
    # always short. Top: cumulative return $\sum_{{u \le t}} r_u$, in units of the straddle premium. Bottom: cumulative gain
    # over always short, $\sum_{{u \le t}} \bigl(r_u - r^{{\mathrm{{short}}}}_u\bigr)$; the markers are each forecast's 10
    # days of largest gain, each a day on which it bought and $R_t \ge {min_r:.2f}$. The gain over the 866 days is
    # {totals}; without each forecast's 10 best days it is {ex10s}. For {neg_ex10} of the 112 forecasts of the model set the
    # gain is negative without their 10 best days.}}
    # \label{{fig:rh_econ_cumpnl}}
    # \end{{figure}}
    # """
    # )
    # 2026-10-17 (user): parked -- caption "the closing-interval elastic net and LightGBM (the specification of
    # Table~\ref{tab:rh_econ_control}), and the intraday elastic net": the LightGBM is now the pooled intraday tree.
    # 2026-10-15 (user): parked -- NN content parked for now; the caption's MinGRU clauses were
    # "one model of each class on the full predictor set: the closing-interval elastic net, LightGBM (the specification of
    # Table~\ref{tab:rh_econ_control}) and MinGRU, and the pooled intraday elastic net." and "the gain is \rheGainEnet,
    # \rheGainLgbm, \rheGainNN{} and \rheGainPoolEnet{} premia, in the order above, and without each model's
    # \rheTopDays{} best days it is \rheGainExEnet, \rheGainExLgbm, \rheGainExNN{} and \rheGainExPoolEnet."
    fig1 = (
        HEADER
        + NUMBERS_GUARD
        + r"""\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{figures/rehaul_econ_cumpnl.pdf}
\caption{Cumulative returns of the sign strategy. The top panel plots the cumulative return at mid-quotes,
$\sum_{t' \le t} r_{t'}$, in units of the straddle premium, for the short-straddle benchmark ($q_t \equiv -1$) and for
one model of each class on the full predictor set: the closing-interval elastic net, the intraday LightGBM (the
specification of Table~\ref{tab:rh_econ_control}), and the intraday elastic net. The bottom panel plots the
cumulative gain over the short-straddle benchmark,
$\sum_{t' \le t} \bigl(r_{t'} + R_{t'}\bigr)$; the markers are each model's \rheTopDays{} days of largest gain, all
of them days with $q_t = +1$ and $R_t \ge \rheTopRMin$. Over the \rheNdays{} trading days of the out-of-sample period
the gain is \rheGainEnet, \rheGainLgbm{} and \rheGainPoolEnet{} premia, in the order above, and without
each model's \rheTopDays{} best days it is \rheGainExEnet, \rheGainExLgbm{} and \rheGainExPoolEnet.
Without their \rheTopDays{} best days, \rheTailNeg{} of the \rheModelSetN{} forecasts of the model set have a negative
gain.}
\label{fig:rh_econ_cumpnl}
\end{figure}
"""
    )
    mm = e4i["med_mid"]
    # 2026-10-15 (user): parked -- NN content parked for now; the caption's sentence "The closing-interval MinGRU on
    # the full predictor set (triangle) is drawn but not in the median."
    fig2 = (
        HEADER
        + rf"""\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figures/rehaul_econ_sizing.pdf}}
\caption{{Sharpe ratio by sizing rule over the {e4i["n"]} forecasts of the model set, mid (left) and crossed (right),
on days {WARMUP + 1}--{WARMUP + e4i["n_days"]}. One marker per forecast, coloured by family and jittered vertically;
the bar and its label mark the model-set median (mid: {", ".join(f"{mm[r]:.2f}" for r, _ in RULES)} for
the sign strategy, robust clip, full long / scaled short and rank of the absolute gap).
Dashed: the short-straddle benchmark ({num(e4i["short_mid"])} mid, {num(e4i["short_x"])} crossed). The
robust clip is above the sign strategy for {e4i["beat_mid"]["clip"]} of the {e4i["n"]} forecasts at mid (median gain
{num(e4i["med_gain"]["clip"], plus=True)}). Rules and scoring as in Table~\ref{{tab:rh_econ_sizing}}.}}
\label{{fig:rh_econ_sizing}}
\end{{figure}}
"""
    )
    ys = yrs[SHORT]
    # 2026-10-18 (user, new-notes 15): caption -- each year's ratio is SR_y = sqrt(N_y) mean_y / sd_y. Was: "Each year's
    # ratio is annualised per trade day, as in the tables."
    # 2026-10-22 (user, D6): parked -- "scaled by the square root of that year's number of option days"; each year is
    # now annualized by sqrt(A_y). Was: "..., scaled by the square root of that year's number of option days."
    fig3 = (
        HEADER
        + rf"""\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figures/rehaul_econ_byyear.pdf}}
\caption{{Sharpe ratio by calendar year at mid, $q_t = \sign(\hat\sigma^2_t - \mathrm{{IV}}^2_t)$, for the forecasts of
Figure~\ref{{fig:rh_econ_cumpnl}} and the short-straddle benchmark (2020 to 2024: {", ".join(num(v) for v in ys)}); 2024 runs from
January to April. Each year's ratio is $\SR_y$ of Section~\ref{{sec:rh_econ_measures}}, annualized by the square root of $A_y$, that year's option days per calendar year. Grey: the interquartile range
and median of the 112 forecasts of the model set (median {", ".join(num(v) for v in yrs["set_med"])}).}}
\label{{fig:rh_econ_byyear}}
\end{{figure}}
"""
    )

    # ---- prose macros (\rhe...), for the section text; not \input by the floats
    def mac(name: str, val: str, note: str) -> str:
        return f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{val}}}  % {note}\n"

    def m2(x: float, plus: bool = False) -> str:
        v = round(float(x), 2) + 0.0
        s = f"{abs(v):.2f}"
        return (
            r"\ensuremath{-}"
            if v < 0
            else (r"\ensuremath{+}" if plus and v > 0 else "")
        ) + s

    nums = (
        HEADER
        + "% \\rhe... macros for writeup/rehaul/07_economics.tex; values as in the rehaul_econ_* floats.\n"
    )
    # 2026-10-18 (user, second-list N2): parked -- the comment said 866 days; trade_stats' Sharpe is on the post-warmup
    # days t > WARMUP (= \rheSzDays, 614; the same 0.77 as \rheSzShort). Was: "always short, Sharpe mid, 866 days"
    nums += mac(
        "rheShortSh", m2(st[SHORT]["sh_mid"]), "always short, Sharpe mid, 614 post-warmup days (t > WARMUP)"
    )
    nums += mac("rheShortShX", m2(st[SHORT]["sh_x"]), "always short, Sharpe crossed")
    nums += mac(
        "rheShortT",
        m2(st[SHORT]["t_mean"]),
        "always short, t of the mean mid return (LLSW bandwidth, 614 days)",
    )
    # 2026-10-21 (user, decision #2): its fixed-b p; "statistically indistinguishable from zero" (07_economics.tex) needs
    # p >= 0.05, so the run stops otherwise
    short_p = hac_p(st[SHORT]["t_mean"])
    assert short_p >= 0.05, f"short straddle mean: fixed-b p = {short_p:.4f} < 0.05: reword 'indistinguishable from zero'"
    nums += mac("rheShortP", p_digits(short_p), "... its two-sided fixed-b p")
    nums += mac(
        "rheShortHit", f"{100 * st[SHORT]['hit']:.1f}", "always short, hit rate %"
    )
    nums += mac(
        "rheEnetSh",
        m2(st["subpt_enet_all_features"]["sh_mid"]),
        "per-bar enet all features, Sharpe mid",
    )
    nums += mac("rheEnetShX", m2(st["subpt_enet_all_features"]["sh_x"]), "... crossed")
    nums += mac(
        "rheEnetT",
        m2(st["subpt_enet_all_features"]["t_dsh"]),
        "... t of the mean mid-return gain over always short (LLSW bandwidth, 614 days)",
    )
    # 2026-10-21 (user): its two-sided normal p (master_table_close.hac_p = p_two), printed beside \rheEnetT
    # 2026-10-21 (user, decision #2): now the fixed-b p. Was: p_digits(p_two(st["subpt_enet_all_features"]["t_dsh"]))
    nums += mac(
        "rheEnetP",
        p_digits(hac_p(st["subpt_enet_all_features"]["t_dsh"])),
        "... its two-sided fixed-b p",
    )
    # 2026-10-15 (user): NN content parked for now -- the \rheNN... macros stay defined (unused by the floats), from
    # nn_st (the MinGRU is off E1): same values as st[NN_ALL] before
    # nums += mac("rheNNSh", m2(st[NN_ALL]["sh_mid"]), "MinGRU all features, Sharpe mid")
    # nums += mac("rheNNShX", m2(st[NN_ALL]["sh_x"]), "... crossed")
    nums += mac(
        "rheNNSh", m2(nn_st["sh_mid"]), "MinGRU all features, Sharpe mid (parked)"
    )
    nums += mac("rheNNShX", m2(nn_st["sh_x"]), "... crossed")
    nums += mac("rheNNBaseSh", m2(nb["sh_mid"]), "MinGRU HAR + calendar, Sharpe mid")
    nums += mac("rheNNBaseShX", m2(nb["sh_x"]), "... crossed")
    nums += mac(
        "rheSigVsShort",
        str(sig_vs_short),
        "model-set forecasts whose mean mid-return gain over always short has t > 0 and fixed-b p < 0.05",
    )
    nums += mac(
        "rheSigVsShortFull",
        str(sig_vs_short_full),
        "... of which on the full predictor set (all_features)",
    )
    nums += mac(
        "rhePairs", str(e2i["n"]), "HAR + cal. -> all features pairs in the model set"
    )
    nums += mac("rhePairsUp", str(e2i["up"]), "... Sharpe mid higher with all features")
    # 2026-10-21 (user, decision #2): Ledoit-Wolf. Was: mac("rhePairsSig", str(e2i["sig"]), "... HAC t > 1.96")
    # 2026-10-22 (user, D6): the D6 bootstrap. Was: "... Sharpe mid higher and Ledoit-Wolf p < 0.05" (and "lower")
    nums += mac(
        "rhePairsSig",
        str(e2i["sig"]),
        "... Sharpe mid higher and D6 bootstrap p < 0.05",
    )
    nums += mac(
        "rhePairsSigNeg",
        str(e2i["sig_neg"]),
        "... Sharpe mid lower and D6 bootstrap p < 0.05",
    )
    nums += mac(
        "rhePairsSigLW",
        str(e2i["sig_lw"]),
        "... Sharpe mid higher and Ledoit-Wolf p < 0.05 (robustness)",
    )
    nums += mac("rhePairsSigNegLW", str(e2i["sig_neg_lw"]), "... Sharpe mid lower and Ledoit-Wolf p < 0.05 (robustness)")
    nums += mac("rhePairsMed", m2(e2i["med"], plus=True), "... median dSharpe mid")
    for r, nm in (
        ("sign", "Sign"),
        ("clip", "Clip"),
        ("lfss", "Lfss"),
        ("rank", "Rank"),
    ):
        nums += mac(
            f"rheSz{nm}Med",
            m2(e4i["med_mid"][r]),
            f"sizing {r}: model-set median Sharpe mid, 614 days",
        )
    nums += mac(
        "rheSzShort",
        m2(e4i["short_mid"]),
        "always short, Sharpe mid on the 614 sizing days",
    )
    nums += mac(
        "rheSzClipBeats",
        str(e4i["beat_mid"]["clip"]),
        "clip above sign(s), of 112 (mid)",
    )
    nums += mac(
        "rheSzClipGain",
        m2(e4i["med_gain"]["clip"], plus=True),
        "median clip - sign(s) Sharpe mid",
    )
    nums += mac(
        "rheTailNeg",
        str(neg_ex10),
        "model-set forecasts whose gain over short is < 0 without 10 best days",
    )
    # 2026-10-15 (consolidation): numbers 07_economics.tex typed (lasso 1.88; elastic net +0.55, lasso +0.50; the
    # short straddle's losing years), now macros; values as printed in Tables E1 and E2.
    lasso_all = "subpt_lasso_all_features"
    nums += mac(
        "rheLassoSh",
        m2(st[lasso_all]["sh_mid"]),
        "per-bar lasso all features, Sharpe mid",
    )
    nums += mac("rheLassoShX", m2(st[lasso_all]["sh_x"]), "... crossed")
    nums += mac(
        "rheAfEnetDSh",
        m2(e2i["pairs"]["subpt_enet_baseline"]["d_mid"], plus=True),
        "per-bar enet: dSharpe mid, all features minus HAR + calendar (E2)",
    )
    nums += mac(
        "rheAfLassoDSh",
        m2(e2i["pairs"]["subpt_lasso_baseline"]["d_mid"], plus=True),
        "per-bar lasso: dSharpe mid, all features minus HAR + calendar (E2)",
    )
    # 2026-10-21 (user): the count behind "(neither statistically significant at the 5% level)" (07_economics.tex,
    # 10_appendix.tex): the closing-interval elastic net and lasso, HAC t of the daily mid-return differential, all
    # features minus HAR + calendar; asserted 0 so the run stops if the sentence is wrong
    # 2026-10-21 (user, decision #2): the sentence is about the Sharpe ratio change, so the test is Ledoit-Wolf. Was:
    # af_enet_lasso_sig = sum(sig(e2i["pairs"][k]["t_mid"]) for k in ("subpt_enet_baseline", "subpt_lasso_baseline"))
    # 2026-10-22 (user, D6): parked -- the Ledoit-Wolf p; the test of the Sharpe change is now the D6 bootstrap. Was:
    # af_enet_lasso_sig = sum(
    #     e2i["pairs"][k]["lw_mid"] < 0.05 for k in ("subpt_enet_baseline", "subpt_lasso_baseline")
    # )
    af_enet_lasso_sig = sum(
        e2i["pairs"][k]["bt_mid"] < 0.05
        for k in ("subpt_enet_baseline", "subpt_lasso_baseline")
    )
    print(
        "closing-interval enet / lasso, full vs HAR, D6 bootstrap p / Ledoit-Wolf p:",
        [
            (round(e2i["pairs"][k]["bt_mid"], 4), round(e2i["pairs"][k]["lw_mid"], 4))
            for k in ("subpt_enet_baseline", "subpt_lasso_baseline")
        ],
    )
    assert af_enet_lasso_sig == 0, f"\\rheAfEnetLassoSig = {af_enet_lasso_sig}: reword 'neither statistically significant'"
    nums += mac(
        "rheAfEnetLassoSig",
        str(af_enet_lasso_sig),
        "... of those two, D6 bootstrap p < 0.05 on the Sharpe mid change (2026-10-22, D6; was Ledoit-Wolf)",
    )
    # 2026-10-18 (user, second-list N2): parked -- the losing-year count ran over all 866 option days (2020-2024, 5
    # years, yrs[SHORT] of fig_byyear), while the Sharpe ratio it sits next to (07_economics.tex) is on the 614
    # post-warmup days. Now counted on the post-warmup days t > WARMUP (2021-08-11 to 2024-04-30: 4 calendar years,
    # 64/219/248/83 days); a year loses money when its mean mid return is negative. Was:
    # short_years = [float(v) for v in yrs[SHORT]]
    # nums += mac("rheShortLossYears", str(sum(v < 0 for v in short_years)),
    #             "calendar years in which always short has a negative Sharpe mid (negative mean return)")
    # nums += mac("rheNYears", str(len(short_years)), "calendar years in the out-of-sample period")
    _pw_r = series[SHORT]["mid"].iloc[WARMUP:]
    _pw_y = _pw_r.index.year
    assert len(_pw_r) == 614, len(_pw_r)
    _pw_n = {int(y): int((_pw_y == y).sum()) for y in sorted(set(_pw_y))}
    assert _pw_n == {2021: 64, 2022: 219, 2023: 248, 2024: 83}, _pw_n
    _pw_mean = {y: float(_pw_r[_pw_y == y].mean()) for y in _pw_n}
    print("N2 always short, post-warmup mean mid return by year:", {y: round(v, 4) for y, v in _pw_mean.items()})
    nums += mac(
        "rheShortLossYears",
        str(sum(v < 0 for v in _pw_mean.values())),
        "calendar years in which always short has a negative mean mid return, 614 post-warmup days (t > WARMUP)",
    )
    nums += mac(
        "rheNYears", str(len(_pw_n)), "calendar years spanned by the 614 post-warmup days (2021-2024)"
    )
    # 2026-10-17 (user, E17): post-warmup counterparts for best LightGBM SR, vs-short count, and rank correlations
    sh_ms = pd.Series({k: sharpe(series[k]["mid"]) for k in model_set})
    ql_ms = pd.Series({k: float(series[k]["qlike"].mean()) for k in model_set})
    # 2026-10-20 (user, third-list 2): mean ES_1 on the post-warmup days. Was: float(series[k]["es1"].mean())
    es_ms = pd.Series({k: float(series[k]["es1"].iloc[WARMUP:].mean()) for k in model_set})
    rho_q = float(ql_ms.rank().corr(sh_ms.rank()))
    rho_es = float(es_ms.rank().corr(sh_ms.rank()))
    nums += mac(
        "rheLgbmBestSh",
        # 2026-10-17 (user, E22/E15): trees["lgbm"] is now the fixed daily rung; this macro keeps its meaning (the
        # ex-post best LightGBM, quoted as such in 07_economics.tex). Was: m2(st[trees["lgbm"]]["sh_mid"]),
        m2(trade_stats(series, best_tree(series, fam, model_set, "lgbm"))["sh_mid"]),
        "best closing-interval LightGBM specification by post-warmup Sharpe mid on all features (ex post; unused by the text since 2026-10-17)",
    )
    nums += mac(
        "rheLgbmCloseFixedSh",
        m2(trade_stats(series, close_fixed["lgbm"])["sh_mid"]),
        "closing-interval LightGBM, fixed hyperparameters re-estimated daily, all features: post-warmup Sharpe mid (2026-10-17)",
    )
    nums += mac(
        "rheVsShortAbove",
        str(above_vs_short),
        "model-set forecasts with post-warmup Sharpe mid above always short",
    )
    nums += mac(
        "rheRankCorrQ",
        m2(rho_q),
        "Spearman rank correlation of mean QLIKE with post-warmup Sharpe mid across the model set",
    )
    nums += mac(
        "rheRankCorrES",
        m2(rho_es),
        "Spearman rank correlation of mean ES_1 with post-warmup Sharpe mid across the model set",
    )    # 2026-10-17 (user, E25): the same correlations on the pooled intraday forecasts only (new macros; the two above
    # are unchanged).
    pser = pooled_rank_series(series)
    sh_p = pd.Series({k: sharpe(v["mid"]) for k, v in pser.items()})
    ql_p = pd.Series({k: float(v["qlike"].mean()) for k, v in pser.items()})
    # 2026-10-20 (user, third-list 2): post-warmup ES_1. Was: float(v["es1"].mean())
    assert all(len(v) == N_OPT for v in pser.values())
    es_p = pd.Series({k: float(v["es1"].iloc[WARMUP:].mean()) for k, v in pser.items()})
    # 2026-10-22 (user, E34): parked -- the base-2 trees' Sharpe ratios of record; the full-set rows of pser are the
    # series of add_pooled_trees (asserted). Was:
    # for k in ("pooltree_lgbm_all_features", "pooltree_xgb_all_features", "pooltree_rf_all_features"):
    #     # 2026-10-18 (user, new-notes 15): POOLTREE_SR is fixed-factor; gate with sharpe_days_252. Was:
    #     # assert round(sh_p[k] + 0.0, 2) == POOLTREE_SR[k.split("_")[1]], (k, sh_p[k])
    #     old_ = sharpe_days_252(pser[k]["mid"].to_numpy(float)[WARMUP:])
    #     assert round(old_ + 0.0, 2) == POOLTREE_SR[k.split("_")[1]], (k, old_)
    for k in ("pooltree_lgbm_all_features", "pooltree_xgb_all_features", "pooltree_rf_all_features"):
        assert pser[k] is series[k], k
    print("E25 pooled Sharpe mid:", {k: round(float(v), 2) for k, v in sh_p.items()})
    nums += mac(
        "rheRankCorrQPool",
        m2(float(ql_p.rank().corr(sh_p.rank()))),
        "Spearman rank correlation of mean QLIKE with post-warmup Sharpe mid across the 12 pooled intraday forecasts",
    )
    nums += mac(
        "rheRankCorrESPool",
        m2(float(es_p.rank().corr(sh_p.rank()))),
        "Spearman rank correlation of mean ES_1 with post-warmup Sharpe mid across the 12 pooled intraday forecasts",
    )
    # 2026-10-22 (user, D6): the two macros above use the D6 Sharpe ratio (sharpe()); the same correlations under the
    # per-year ratio D1 (N_y-weighted sqrt(N_y)) and the pooled sqrt(252) ratio D2, robustness (new macros)
    for nm_, fn_ in (("DOne", sharpe_days_d1), ("DTwo", sharpe_days_252)):
        sh_alt = pd.Series({k: fn_(post_warmup(v["mid"])) for k, v in pser.items()})
        for what_, loss_ in (("Q", ql_p), ("ES", es_p)):
            rho_ = float(loss_.rank().corr(sh_alt.rank()))
            print(
                f"D6 robustness: rank correlation of mean {what_} loss with the {nm_} Sharpe ratio, 12 pooled: {rho_:+.3f}"
            )
            nums += mac(
                f"rheRankCorr{what_}Pool{nm_}",
                m2(rho_),
                f"... as rheRankCorr{what_}Pool, with the {nm_} Sharpe ratio (robustness)",
            )
    # 2026-10-18 (user, second-list N10): the share of positive return that comes from days on which the straddle
    # expires worthless (R_t = -1 exactly: X_t = 0), for the 12 pooled intraday forecasts (pser, E25) on the 614
    # post-warmup days t > WARMUP, sign strategy at mid. For each forecast: sum of r_t over days with r_t > 0 and
    # R_t = -1, divided by the sum of r_t over all days with r_t > 0. Replaces the untraced "account for the bulk of
    # positive returns" in 07_economics.tex.
    _R = series[OLS]["R"].to_numpy(float)[WARMUP:]
    _worth = _R == -1.0
    assert len(_R) == 614 and _R.min() >= -1.0, (len(_R), _R.min())
    _wsh = {}
    for k, v in pser.items():
        _r = v["mid"].to_numpy(float)[WARMUP:]
        assert np.allclose(np.abs(_r), np.abs(_R)), k  # |q_t| = 1 for the sign strategy
        _pos = _r > 0
        _wsh[k] = 100.0 * float(_r[_pos & _worth].sum() / _r[_pos].sum())
    _wsh = pd.Series(_wsh)
    assert len(_wsh) == 12, len(_wsh)
    print("N10 worthless-day share of positive return (%):", _wsh.round(1).to_dict())
    nums += mac("rheWorthlessDays", str(int(_worth.sum())), "post-warmup option days with R_t = -1 (straddle expires worthless)")
    nums += mac("rhePoolWorthlessShareMed", f"{_wsh.median():.0f}", "median % of positive mid return from R_t = -1 days, 12 pooled intraday forecasts, 614 days")
    nums += mac("rhePoolWorthlessShareMin", f"{_wsh.min():.0f}", "lowest % of positive mid return from R_t = -1 days, 12 pooled intraday forecasts, 614 days")
    nums += mac("rhePoolWorthlessShareMax", f"{_wsh.max():.0f}", "highest % of positive mid return from R_t = -1 days, 12 pooled intraday forecasts, 614 days")

    # 2026-10-15 (table legibility): the numbers the table notes and the figure caption used to type, as macros
    def mdp(x: float, dp: int, plus: bool = False) -> str:
        v = round(float(x), dp) + 0.0
        sgn = (
            r"\ensuremath{-}"
            if v < 0
            else (r"\ensuremath{+}" if plus and v > 0 else "")
        )
        return sgn + f"{abs(v):.{dp}f}"

    # 2026-10-18 (user, new-notes 19): the May 2022 subsample split of the 15:30 sign strategy on the \rhrPoolN pooled
    # intraday forecasts (pser, E25), post-warmup days t > WARMUP (614, from 2021-08-11). Split at 2022-05-16, the first
    # week with all five weekday expirations: 125 days before (through 2022-05-13), 489 from 2022-05-16. Per-year Sharpe
    # ratio (item 15, asl.sharpe_ann_by_year, weighted) on each side separately, so a partial year is scaled by its own
    # day count within that side. Mid and crossed fills; always short on the same days. New macros \rheSplit*; the
    # per-forecast table also goes to %TEMP%\item19_split.csv.
    SPLIT = pd.Timestamp("2022-05-16")
    pdays = series[SHORT].index[WARMUP:]
    pre_m = np.asarray(pdays < SPLIT)
    assert len(pdays) == 614 and pd.Timestamp(pdays[0]).date().isoformat() == "2021-08-11", (len(pdays), pdays[0])
    assert int(pre_m.sum()) == 125 and int((~pre_m).sum()) == 489, (int(pre_m.sum()), int((~pre_m).sum()))
    assert pd.Timestamp(pdays[pre_m][-1]).date().isoformat() == "2022-05-13", pdays[pre_m][-1]
    assert pd.Timestamp(pdays[~pre_m][0]).date().isoformat() == "2022-05-16", pdays[~pre_m][0]
    side_days = {"Pre": pdays[pre_m], "Post": pdays[~pre_m]}
    want_ny = {"Pre": {2021: 64, 2022: 61}, "Post": {2022: 158, 2023: 248, 2024: 83}}

    # 2026-10-22 (user, D6): parked -- the N_y-weighted per-year ratio on each side. Now D6 on each side's own sample
    # (A_y from the fraction of year y between the side's first and last day; the simple mean over the side's years).
    # Was:
    #     out = asl.sharpe_ann_by_year(np.asarray(r, float)[WARMUP:][m], dates=side_days[side], combine="weighted")
    def split_sr(r, side: str) -> float:
        m = pre_m if side == "Pre" else ~pre_m
        out = hi.d6_sharpe(np.asarray(r, float)[WARMUP:][m], dates=side_days[side])
        assert {int(y): int(v[0]) for y, v in out["by_year"].items()} == want_ny[side], (side, out["by_year"])
        return float(out["sharpe"])

    bS = series[OLS]
    R_S = bS["R"].to_numpy(float)
    # 2026-10-22 (user, E34): + rv_unclipped, the truth tree_16 gates the trees' bar variance against. Was: columns=["key", "day", "iv_var"]
    _dly = pd.read_parquet(
        RES / "master_table_daily.parquet",
        columns=["key", "day", "iv_var", "rv_unclipped"],
    )
    _dly["day"] = pd.to_datetime(_dly["day"]).dt.normalize()
    _ref = _dly[_dly["key"] == OLS].set_index("day").sort_index()
    _ref.index = _ref.index.astype(bS.index.dtype)
    IV_S = _ref.reindex(bS.index)["iv_var"].to_numpy(float)

    def crossed_of(k, v) -> np.ndarray:
        if "x" in v.columns:
            return v["x"].to_numpy(float)
        # the HAR-set pooled trees store mid only: q = sign(plain - iv_var) as in pooled_rank_series
        # 2026-10-22 (user, E34): parked -- the base-2 HAR-benchmark trees. Was:
        # pl = plain_16(POOLTREE_DIR / f"yhat_pooltree_{k.split('_')[1]}_baseline.parquet")
        # pl.index = pl.index.astype(bS.index.dtype)
        # q = np.where(pl.reindex(bS.index).to_numpy(float) > IV_S, 1.0, -1.0)
        RV_S = _ref.reindex(bS.index)["rv_unclipped"].to_numpy(float)
        q = np.where(
            tree_16(k.split("_")[1], "baseline", bS.index, RV_S) > IV_S, 1.0, -1.0
        )
        assert np.array_equal(q * R_S, v["mid"].to_numpy(float)), k
        return gui._crossed_return(q, bS["ex"].to_numpy(), bS["bid"].to_numpy(), bS["ask"].to_numpy())

    assert len(pser) == 12, len(pser)
    split_rows = []
    for k, v in pser.items():
        mid, x = v["mid"].to_numpy(float), crossed_of(k, v)
        assert len(mid) == 866 and len(x) == 866, k
        split_rows.append(
            dict(forecast=k, pre_mid=split_sr(mid, "Pre"), post_mid=split_sr(mid, "Post"),
                 pre_x=split_sr(x, "Pre"), post_x=split_sr(x, "Post"), full_mid=sharpe(mid))
        )
    sp = pd.DataFrame(split_rows)
    shS = {f"{side}{f}": split_sr(series[SHORT][c].to_numpy(float), side)
           for side in ("Pre", "Post") for f, c in (("", "mid"), ("X", "x"))}
    import os as _os

    sp.assign(short_pre_mid=shS["Pre"], short_post_mid=shS["Post"], short_pre_x=shS["PreX"], short_post_x=shS["PostX"]).to_csv(
        Path(_os.environ.get("TEMP", "/tmp")) / "item19_split.csv", index=False, float_format="%.6f"
    )
    print("item 19 split (D6 SR, mid / crossed):\n" + sp.round(3).to_string(index=False))  # 2026-10-22 (D6): was "per-year SR"
    print("item 19 always short:", {k_: round(v_, 3) for k_, v_ in shS.items()})
    nums += mac("rheSplitDate", "16~May~2022", "first day of the post-split subsample (first week with five weekday expirations)")
    nums += mac("rheSplitPreLast", "13~May~2022", "last day of the pre-split subsample")
    nums += mac("rheSplitPreN", str(int(pre_m.sum())), "post-warmup option days before the split")
    nums += mac("rheSplitPostN", str(int((~pre_m).sum())), "post-warmup option days from the split on")
    for side in ("Pre", "Post"):
        for f, col in (("", "mid"), ("X", "x")):
            c = sp[f"{side.lower()}_{col}"]
            # 2026-10-22 (user, D6): "per-year SR" -> "D6 SR" in the macro comments
            what = f"{'before' if side == 'Pre' else 'from'} the split, {'mid-quote' if col == 'mid' else 'crossed'} fills, 12 pooled intraday forecasts, D6 SR"
            nums += mac(f"rheSplit{side}Med{f}", mdp(c.median(), 2), "median Sharpe ratio " + what)
            nums += mac(f"rheSplit{side}Min{f}", mdp(c.min(), 2), "lowest Sharpe ratio " + what)
            nums += mac(f"rheSplit{side}Max{f}", mdp(c.max(), 2), "highest Sharpe ratio " + what)
            nums += mac(f"rheSplit{side}Pos{f}", str(int((c > 0).sum())), "forecasts with a positive Sharpe ratio " + what)
            nums += mac(
                f"rheSplit{side}Short{f}",
                mdp(shS[side + f], 2),
                "always-short straddle Sharpe ratio "
                + what.split(", 12")[0]
                + ", D6 SR",
            )

    # 2026-10-22 (user, D6): robustness of the Sharpe-ratio definition. The six pooled intraday classes, full predictor
    # set minus HAR benchmark, sign strategy at mid-quotes on the 614 post-warmup days: D6 (the paper's ratio) with the
    # p of its paired block bootstrap; D1, the per-year ratio of new-notes 15 (N_y-weighted, sqrt(N_y)), the paper's
    # ratio from 2026-10-18 to 2026-10-22; D2, the ratio of the pooled daily returns, mean / sd x sqrt(252), with the
    # Ledoit-Wolf p (the test of that ratio, and the paper's Sharpe test before D6); and D6 on each side of 16 May 2022
    # (split_sr). Table rehaul_econ_sharpe_defs.tex and the \rheDef... macros. The D6 differences equal the results
    # generator's \rhrAf<M>PoolDSh (the same forecasts and days; checked by the D6 rerun, not here).
    def_classes = (
        ("Ridge", "Ridge", "poolpt13_ridge"),
        ("Lasso", "Lasso", "poolpt13_lasso"),
        ("Elastic net", "Enet", "poolpt13_enet"),
        ("LightGBM", "Lgbm", "pooltree_lgbm"),
        ("XGBoost", "Xgb", "pooltree_xgb"),
        ("Random forest", "Rf", "pooltree_rf"),
    )
    pw_days_ = pw_days()
    def_rows = []
    for name, tok_, k in def_classes:
        h_, f_ = (
            pser[f"{k}_baseline"]["mid"].to_numpy(float),
            pser[f"{k}_all_features"]["mid"].to_numpy(float),
        )
        hp, fp = post_warmup(h_), post_warmup(f_)
        bt_ = hi.d6_diff_test(fp, hp, pw_days_)
        row_ = dict(
            name=name,
            tok=tok_,
            d6=sharpe(f_) - sharpe(h_),
            bt=bt_["p"],
            d1=sharpe_days_d1(fp) - sharpe_days_d1(hp),
            d2=sharpe_days_252(fp) - sharpe_days_252(hp),
            lw=hi.lw_sharpe_test(fp, hp)["p"],
            pre=split_sr(f_, "Pre") - split_sr(h_, "Pre"),
            post=split_sr(f_, "Post") - split_sr(h_, "Post"),
        )
        assert abs(bt_["delta"] - row_["d6"]) < 1e-10, (k, bt_["delta"], row_["d6"])
        def_rows.append(row_)
    dfd = pd.DataFrame(def_rows).set_index("tok")
    print(
        "D6 robustness of the Sharpe-ratio definition (full minus HAR, mid, 614 days):\n"
        + dfd.round(4).to_string()
    )

    def p_mac(p: float) -> str:
        return p_digits(p) if p >= 0.001 else r"\ensuremath{<}0.001"

    for tok_, r_ in dfd.iterrows():
        nums += mac(
            f"rheDefDSix{tok_}",
            m2(r_["d6"], plus=True),
            f"pooled {r_['name']}: D6 Sharpe mid, full minus HAR (614 days)",
        )
        nums += mac(
            f"rheDefBootP{tok_}", p_mac(r_["bt"]), "... its D6 bootstrap p (two-sided)"
        )
        nums += mac(
            f"rheDefDOne{tok_}",
            m2(r_["d1"], plus=True),
            "... D1 (per-year, N_y-weighted sqrt(N_y)) Sharpe mid change",
        )
        nums += mac(
            f"rheDefDTwo{tok_}",
            m2(r_["d2"], plus=True),
            "... D2 (pooled, mean / sd x sqrt(252)) Sharpe mid change",
        )
        nums += mac(
            f"rheDefLWP{tok_}",
            p_mac(r_["lw"]),
            "... Ledoit-Wolf p (equal Sharpe ratios of the daily returns)",
        )
        nums += mac(
            f"rheDefPre{tok_}",
            m2(r_["pre"], plus=True),
            "... D6 change before 16 May 2022 (125 days)",
        )
        nums += mac(
            f"rheDefPost{tok_}",
            m2(r_["post"], plus=True),
            "... D6 change from 16 May 2022 (489 days)",
        )
    nums += mac(
        "rheDefN",
        str(len(dfd)),
        "pooled classes in the Sharpe-definition robustness table",
    )
    for col, nm in (
        ("d6", "DSix"),
        ("d1", "DOne"),
        ("d2", "DTwo"),
        ("pre", "Pre"),
        ("post", "Post"),
    ):
        nums += mac(
            f"rheDef{nm}Pos",
            str(int((dfd[col] > 0).sum())),
            f"pooled classes (of 6) with {nm} Sharpe change > 0",
        )
        nums += mac(
            f"rheDef{nm}Min", m2(dfd[col].min(), plus=True), f"... {nm} change: min"
        )
        nums += mac(
            f"rheDef{nm}Max", m2(dfd[col].max(), plus=True), f"... {nm} change: max"
        )
    nums += mac(
        "rheDefBootSig",
        str(int((dfd["bt"] < 0.05).sum())),
        "pooled classes (of 6) with D6 bootstrap p < 0.05",
    )
    nums += mac(
        "rheDefLWSig",
        str(int((dfd["lw"] < 0.05).sum())),
        "pooled classes (of 6) with Ledoit-Wolf p < 0.05",
    )
    defs_tex = table_sharpe_defs(dfd)

    days = series[SHORT].index
    assert len(days) == 866 and len({r["dm_lag"] for r in e3r}) == 1
    nums += mac(
        "rheNdays", str(len(days)), "T_oos: trading days of the out-of-sample period"
    )
    nums += mac(
        "rheSampleStart", f"{days[0].day} {days[0]:%B %Y}", "first out-of-sample day"
    )
    nums += mac(
        "rheSampleEnd", f"{days[-1].day} {days[-1]:%B %Y}", "last out-of-sample day"
    )
    # 2026-10-21 (user, decision #2): parked -- \rheHACLag (12, floor(1.5 T^(1/3))). Was:
    # nums += mac("rheHACLag", str(HAC_LAG), "Newey-West lag q of every HAC t (floor(1.5 T^(1/3)))")
    hac_cv = hi.fixedb_cv(HAC_B)
    print(f"decision #2: T = {N_PW}, S = {HAC_S}, b = {HAC_B:.4f}, fixed-b 5% cv = {hac_cv:.3f} "
          f"(Kiefer-Vogelsang cubic {hi.kv_cubic_cv975(HAC_B):.3f})")
    assert abs(hac_cv - hi.kv_cubic_cv975(HAC_B)) < 0.03, (hac_cv, hi.kv_cubic_cv975(HAC_B))
    nums += mac("rheHACS", str(HAC_S), "bandwidth S = ceil(1.3 T^(1/2)) of every t of a mean (LLSW 2018), T = 614")
    nums += mac("rheHACCV", f"{hac_cv:.2f}", "two-sided 5 % fixed-b critical value at b = S/T (simulated)")
    nums += mac("rheLWBlock", str(hi.LW_BLOCK), "block length of the Ledoit-Wolf bootstrap")
    nums += mac("rheLWBlockLo", str(hi.LW_BLOCK_ALT[0]), "... sensitivity: shorter block")
    nums += mac("rheLWBlockHi", str(hi.LW_BLOCK_ALT[1]), "... sensitivity: longer block")
    nums += mac("rheLWBoot", f"{hi.LW_BOOT:,}".replace(",", "{,}"), "bootstrap resamples")
    # 2026-10-22 (user, D6): the bootstrap of the D6 Sharpe-ratio difference (hac_inference.d6_diff_test)
    nums += mac(
        "rheBootBlock",
        str(hi.LW_BLOCK),
        "block length of the D6 bootstrap (= the Ledoit-Wolf block)",
    )
    nums += mac(
        "rheBootN", f"{hi.LW_BOOT:,}".replace(",", "{,}"), "D6 bootstrap resamples"
    )
    _yy = hi.d6_years(pw_days())
    assert _yy["N"] == {2021: 64, 2022: 219, 2023: 248, 2024: 83}, _yy["N"]
    for y_, nm_ in zip(sorted(_yy["A"]), ("One", "Two", "Three", "Four")):
        nums += mac(
            f"rheDSixYear{nm_}",
            str(y_),
            f"D6: calendar year {nm_} of the 614 post-warmup days",
        )
        nums += mac(
            f"rheDSixN{nm_}", str(_yy["N"][y_]), "... N_y, its post-warmup option days"
        )
        nums += mac(
            f"rheDSixCov{nm_}",
            f"{_yy['coverage'][y_]:.3f}",
            "... c_y, the fraction of the calendar year covered",
        )
        nums += mac(f"rheDSixA{nm_}", f"{_yy['A'][y_]:.1f}", "... A_y = N_y / c_y, option days per calendar year")
    nums += mac(
        "rheDMLag",
        str(e3r[0]["dm_lag"]),
        "Newey-West lag q of the DM statistics in Table E3",
    )
    nums += mac("rheModelSetN", str(len(model_set)), "forecasts in the model set")
    nums += mac(
        "rheTreeSpecs",
        str(tree_specs(fam, model_set)),
        "tree specifications per estimator on the full predictor set (E1 note a)",
    )
    # 2026-10-22 (user, E34): the design of the base-5 pooled trees (pt13, forecasts.json), for E1 note a and the text
    tm = pt13.meta()
    nums += mac(
        "rheTreeNBars",
        str(tm["bars_day"]),
        "pooled trees: regular-hours 30-minute bars a day (10:00..16:00 ET bar ends)",
    )
    nums += mac(
        "rheTreeWindowDays",
        f"{tm['window_days']:,}".replace(",", "{,}"),
        "pooled trees: rolling window, trading days",
    )
    nums += mac(
        "rheTreeWindowBars",
        f"{tm['window_bars']:,}".replace(",", "{,}"),
        "pooled trees: rolling window, bars (days x bars a day)",
    )
    nums += mac(
        "rheTreeRunStart",
        f"{tm['span'][0].day} {tm['span'][0]:%B %Y}",
        "pooled trees: first forecast day of the base-5 walk",
    )
    # 2026-10-22 (user, E34: one design for the 12 pooled forecasts): the pooled linear rerun (pl13) has the trees'
    # bars, window and first evaluation day (asserted), so one set of macros describes every pooled row
    lm = pl13.meta()
    assert (lm["bars_day"], lm["window_days"], lm["window_bars"], lm["span"][0]) == (
        tm["bars_day"],
        tm["window_days"],
        tm["window_bars"],
        tm["span"][0],
    ), (lm, tm)
    nums += mac(
        "rhePoolNBars",
        str(lm["bars_day"]),
        "pooled intraday models (linear and trees): regular-hours 30-minute bars a day (10:00..16:00 ET bar ends)",
    )
    nums += mac(
        "rhePoolWindowDays",
        f"{lm['window_days']:,}".replace(",", "{,}"),
        "pooled intraday models: rolling window, trading days",
    )
    nums += mac(
        "rhePoolWindowBars",
        f"{lm['window_bars']:,}".replace(",", "{,}"),
        "pooled intraday models: rolling window, bars (days x bars a day)",
    )
    nums += mac(
        "rhePoolRunStart",
        f"{lm['span'][0].day} {lm['span'][0]:%B %Y}",
        "pooled intraday models: first evaluation day of the rolling-window walk",
    )
    nums += mac(
        "rheTreePairs", str(e2i["n_tree"]), "HAR -> full-set tree pairs (E2 Panel C)"
    )
    nums += mac("rheTreePairsEach", str(e2i["n_tree_each"]), "... per estimator")
    nums += mac("rheSzWarmup", str(WARMUP), "sizing: days that only build the scale")
    nums += mac("rheSzDays", str(e4i["n_days"]), "sizing: days scored")
    nums += mac(
        "rheTopDays", str(TOP_DAYS), "the tail check: days of largest gain dropped"
    )
    nums += mac(
        "rheTopRMin",
        f"{min_r:.2f}",
        # "smallest straddle return R_t among the 4 models' top days",  # 2026-10-15 (user): parked -- NN content parked for now
        "smallest straddle return R_t among the 3 models' top days (MinGRU parked)",
    )
    # 2026-10-15 (user): NN content parked for now -- \rheGainNN / \rheGainExNN stay defined (unused), from cum[NN_ALL]
    keys4 = {f: key for f, (key, _) in key4.items()} | {"nn": NN_ALL}
    names4 = {f: lab for f, (_, lab) in key4.items()} | {
        "nn": "Closing-interval MinGRU (parked)"
    }
    for f, nm in (
        ("linear", "Enet"),
        ("tree", "Lgbm"),
        ("nn", "NN"),
        ("pooled", "PoolEnet"),
    ):
        # c = cum[key4[f][0]]  # 2026-10-15 (user): parked -- NN content parked for now
        c = cum[keys4[f]]
        nums += mac(
            f"rheGain{nm}",
            mdp(c["edge"], 1, plus=True),
            # f"{key4[f][1]}: cumulative gain over always short, premia",  # 2026-10-15: parked (NN)
            f"{names4[f]}: cumulative gain over always short, premia",
        )
        nums += mac(
            f"rheGainEx{nm}",
            mdp(c["edge_ex10"], 1, plus=True),
            "... without its 10 best days",
        )

    write(GEN / "rehaul_econ_e1_control.tex", e1)
    write(GEN / "rehaul_econ_e2_allfeat.tex", e2)
    write(GEN / "rehaul_econ_e3_pooled.tex", e3)
    write(GEN / "rehaul_econ_e4_sizing.tex", e4)
    write(GEN / "rehaul_econ_fig_cumpnl.tex", fig1)
    write(GEN / "rehaul_econ_fig_sizing.tex", fig2)
    write(GEN / "rehaul_econ_fig_byyear.tex", fig3)
    write(GEN / "rehaul_econ_numbers.tex", nums)
    write(GEN / "rehaul_econ_sharpe_defs.tex", defs_tex)  # 2026-10-22 (user, D6)

    # ---- console summary
    print("\nE1:")
    for k, s in st.items():
        print(
            f"  {k:45s} mean {100 * s['mean']:5.2f}% t {s['t_mean']:5.2f} Sh {s['sh_mid']:5.2f}/{s['sh_x']:5.2f} "
            f"hit {100 * s['hit']:4.1f} buys {100 * s['buys']:4.1f} dSh {s.get('dsh', np.nan):+.2f} "
            f"t {s.get('t_dsh', np.nan):.2f}"
        )
    print("trees:", trees)
    print(
        "E2: n",
        e2i["n"],
        "up",
        e2i["up"],
        "sig",
        e2i["sig"],
        "med",
        round(e2i["med"], 3),
        "med_x",
        round(e2i["med_x"], 3),
        "sig keys",
        e2i["sig_keys"],
    )
    for est in ("enet", "lasso", "ridge"):
        for pre in ("subpt", "poolpt", "poolpt13"):
            p = (e2i["pairs13"] if pre == "poolpt13" else e2i["pairs"])[f"{pre}_{est}_baseline"]
            print(
                f"  {pre} {est}: {p['sh_b']:.2f} -> {p['sh_a']:.2f} d {p['d_mid']:+.2f} (t {p['t_mid']:.2f}) "
                f"x {p['d_x']:+.2f} (t {p['t_x']:.2f})"
            )
    p = e2i["nn"]
    print(
        f"  MinGRU: {p['sh_b']:.2f} -> {p['sh_a']:.2f} d {p['d_mid']:+.2f} (t {p['t_mid']:.2f})"
    )
    print("E3:")
    for r in e3r:
        print(
            f"  {r['est']:5s} {r['set']:12s} pooled {r['sh_a']:.2f} perbar {r['sh_b']:.2f} d {r['d_mid']:+.2f} "
            f"(t {r['t_mid']:.2f}) x {r['d_x']:+.2f} (t {r['t_x']:.2f}) dQ {r['dq']:+.1f}% DM {r['dm']:.2f}"
        )
    print(
        "E4 medians mid",
        {r: round(v, 3) for r, v in e4i["med_mid"].items()},
        "crossed",
        {r: round(v, 3) for r, v in e4i["med_x"].items()},
    )
    print(
        "   beat sign mid",
        e4i["beat_mid"],
        "crossed",
        e4i["beat_x"],
        "short",
        round(e4i["short_mid"], 3),
        round(e4i["short_x"], 3),
    )
    print(
        "cum:",
        {k: {a: round(b, 2) for a, b in v.items()} for k, v in cum.items()},
        "neg_ex10",
        neg_ex10,
    )
    print(
        "by year:",
        {
            k: [round(x, 2) for x in v] if isinstance(v, list) else v
            for k, v in yrs.items()
        },
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="directory for PNG previews of the figures",
    )
    PREVIEW_DIR = ap.parse_args().preview
    main()
