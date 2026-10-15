"""Master table for the closing strategy (checklist A4): every 16:00-bar forecast, ONE scorer, the same days.

WHAT IS SCORED.  Every forecast of the 15:30-16:00 bar's realized variance on disk
(the row stamped 16:00: naive ET, bar-END labelled, issued at 15:30):

  forecast tables   results/spxw_pnl/yhat_*.parquet, discovered by GLOB on every run
                    (the paper's eight, the per-bar linear family, the pooled twins,
                    the VIX-only family, the untuned per-bar trees, and whatever
                    lands later: yhat_subtree_tuned_*, yhat_lstm_*, ...).  The
                    16:00 rows of yhat_restofday_* are the direct rest-of-day model
                    at the 15:30 clock -- the same regression as the per-bar
                    16:00 arm by construction -- scored as CHECK rows.
  arm files         16:00 forecasts that were never stacked into a table: the
                    per-bar OLS incumbent (HAR + calendar), and the bucket studies
                    of compare_mfiv_harlag.FAMILIES (vixonly / free / ivrep at the
                    2000-session window; mfiv / ivslice / ivrep_slice_slope at 500
                    sessions, scored from their own comparison-window start) plus
                    the HAR-ladder variants of the live_feasible bucket (2000
                    sessions).  A (bucket, estimator) that also has a table is
                    read from the table (the gate checks both carry the same
                    forecast).  Training-window ablations (the 500-session twins of
                    2000-session arms) are not scored.

THE SCORER (2026-10-06: recalibration removed, user decision -- every reported number
uses the PLAIN back-transform; the notebook's 13-bar map is NOT used either, so no
number here may be set beside a notebook number).  The research scorers' own
functions are imported, never re-implemented:

  * target     the per-bar spec's 16:00 target: true_adj = sqrt(RV/B), winsorized
               by the spec (rolling 5/95), and true_raw = true_adj^2 B.  Every
               2000-session per-bar arm carries the IDENTICAL 16:00 target (gated
               below), so one frame -- TARGET_ARM -- serves every forecast; a table
               that is not a per-bar arm (the paper's pooled forecasts) is scored
               against the same target on the same stamps.
  * forecast   the plain back-transform  plain = f^2 B  (f = the model's forecast on
               the adjusted scale sqrt(RV/B), B the diurnal baseline; no
               second-moment term, no level/slope map).  QLIKE (column qlike_raw),
               the Diebold-Mariano statistic, the trade and every Sharpe use it.
  * parked     the research scorer's 16:00-bar recalibration pred_clock = (f^2 + s) B,
               s = the mean of the forecast's own squared adjusted-scale errors at
               the 16:00 label over the previous SMEAR_W = 250 sessions (at least
               SMEAR_MIN = 63), lagged one session
               (score_linear_subsection_causal.causal_forecasts).  It is computed
               only as the DIAGNOSTIC column pred_clock: the scorer-change record
               recal_vs_plain.csv and the reported-only gates on stored numbers that
               were written with it.  No reported number uses it.
  * trade      the deck's 15:30 sign(s) on the straddle (nearest out-of-the-money
               call + nearest out-of-the-money put, same-day expiry, one position):
               q = +1 (buy) if plain > the 15:30 implied variance, else -1;
               mid return q R; crossed return exit/ask - 1 (buy) or 1 - exit/bid
               (sell) -- score_linear_subsection.trade_1530 on
               results/atm_straddle_0dte_1530/daily_blk2.parquet (866 days,
               2020-01-03 .. 2024-04-30).  Per-day returns are needed for the
               paired statistics, so trade_days() restates trade_1530's lines and a
               gate checks its aggregates against trade_1530 itself for every row.
  * no intervals  (user, 2026-10-06) the circular block bootstrap is disabled
               (commit b761b28: every draw is the original sample), so every
               interval would equal its point estimate; the interval columns are
               commented out and every number is a point estimate.  The DM
               statistic (QLIKE) and the Newey-West HAC t of the daily mid-return
               difference are statistics, not intervals, and stay.

THE SAME DAYS.  Table A holds every forecast that covers all 866 trade days
(the intersection is printed and must equal 866); a forecast that drops days, or
that is only valid from a later comparison-window start, goes to table B, scored
against the reference and always short on ITS OWN days (still paired).

REFERENCE (fixed): the block-diagonal ridge on the panel of record
(yhat_blk2_fomc1.parquet, tag blk2) -- the paper's headline forecast: HAR ladder
(penalty 1) plus the exogenous block (penalty 100), the forecast the paper's 15:30
deck trades (daily_blk2.parquet).  Every row answers "does this forecast improve on
the paper's?".  Second reference: always short (q = -1 every day, no forecast).

COLUMNS per forecast (point estimates only): 16:00 QLIKE of the plain forecast
(qlike_raw, against the spec's target; qlike_raw_unclipped against the unclipped
realized variance of the production table); % difference vs the reference and the
Diebold-Mariano HAC statistic on the daily QLIKE difference
(src.evaluation.diebold_mariano.dm_test, the research scorers' test); sign(s) Sharpe
mid and crossed, mean return, hit rate (share of days the position earns > 0), buy
days; the Sharpe difference vs the reference and vs always short on the same days and
the Newey-West HAC t of the daily mid-return difference vs the reference;
calibration (mean realized / mean plain forecast).  The paired bootstrap intervals
(day_block_ci, paired_sharpe's percentiles) are commented out (2026-10-06).

EXTREME BACK-TRANSFORM VALUES (named, explicit; no silent clip).  PLAIN_FLOOR(d) =
the smallest 16:00 target of the previous FLOOR_W sessions, lagged one session
(FLOOR_W, FLOOR_MIN = the parked recalibration's 250-session window and 63-session
warm-up, no new constant).  A plain forecast below it predicts a quieter 15:30-16:00
half hour than any in the past year: it is flagged, listed in
master_table_extremes.csv, and the QLIKE is reported both as is (qlike_raw, the
reported number) and with the flagged forecasts raised to the floor
(qlike_raw_floored, a diagnostic).  Counted on the trade days and on every 16:00
session of the arm span (2018-06-25 .. 2024-04-30, early closes included).

ARM ROOT (--arm-root, or the environment variable MASTER_TABLE_ARM_ROOT; the CLI wins).
The arm-only rows, the per-bar OLS incumbent and the common 16:00 target are read from
results/<arm root>/ (LS, TARGET_ARM and compare_mfiv_harlag.PULLED_BASE all follow it):
  linear_subsection        (default) the arms of the first pass, the per-bar design with the
                           HAR x open/close session-edge interactions -- the run of commit
                           6177b74 stays reproducible on its inputs.
  linear_subsection_dedup  the same arms re-run on the de-duplicated per-bar design (the
                           16:00 campaign of 2026-09-29, writeup/CAMPAIGN_16H_2026-09-29.md,
                           agent A; same layout, results/linear_subsection_dedup/arm_list.csv).
A non-default root is gated: its TARGET_ARM equals the default root's bit for bit.  The
forecast TABLES (results/spxw_pnl/yhat_*) do not depend on the arm root: they are whatever
is on disk (after the campaign: the de-duplicated linear arms, the window-masked trees and
LSTMs pinned to one CPU class, the Optuna trees, the intraday-sequence LSTM; see FAMILY_PROV).

GATES (a failure of an ASSERTED gate is printed and recorded in master_table_gates.csv;
the tables are still written; exit status 1): the 16:00 target of every arm file equals
TARGET_ARM's (GATE_REL); every table's baseline equals the target's B (GATE_REL), its
rv_raw equals the production table's and its 16:00 stamps are exactly the target's on the
arm span; every arm file's plain forecast equals the research reader's pred_adj^2 x
baseline (compare_mfiv_harlag._prep) on the trade days; every row's trade aggregates equal
trade_1530's on the plain forecast; every row of master_table_daily.parquet has
q = sign(plain - iv_var) (+1 when plain > iv_var, else -1); a table and the arm file of the
same forecast carry the same pred_adj (the per-bar linear tables against
<arm root>/arms_hoffman2); no forecast falls into "other table"; always short equals the
paper table's always-short row.  STORED RESEARCH NUMBERS: every stored number written with
the recalibrated forecast (2026-10-06: recalibration removed, user decision) is now
REPORTED, not asserted (asserted = False), against the same quantity recomputed here from
the diagnostic pred_clock -- the first-pass numbers
(results/linear_subsection_trees/rescore_local/trees_trade_1530.csv, the compare_* CSVs of
the bucket studies; stored_design "pre-campaign design") and the campaign's own numbers
(STORED_CAMPAIGN: agent A's change_by_arm.csv for every linear forecast, agent H's
mask_levels.csv for the tree rungs T10/T1/RS10/RS1 and the per-bar LSTM, agent C's Optuna
report levels.csv, agent I's lstmi_levels.csv, agent E's check_1530.csv for the rest-of-day
check rows; stored_design "16:00 campaign, recalibrated scorer"), and the committed
recalibrated master table (RECAL_SNAPSHOT).  Stored PLAIN numbers (agent C's qlike_raw) stay
asserted with the de-dup root (reported with the default one).  With the de-dup root the
families the campaign did not touch (paper, pooled twins) must also equal the pre-campaign
master table (BEFORE_DIR) bit for bit under the plain scorer.

BEFORE / AFTER (when BEFORE_DIR holds the pre-campaign master_table.csv and
master_table_daily.parquet): before_after.csv, one row per forecast key -- QLIKE and sign(s)
Sharpe mid / crossed before and after, BOTH under the plain scorer (the before side is
re-scored from the pre-campaign daily frame's plain column and the deck), the differences,
the HAC t of the daily mid-return difference, positions changed, and what changed for the
family (FAMILY_PROV, from the campaign file); keys new in this run are listed with status
"new".  SCORER CHANGE: recal_vs_plain.csv, one row per forecast -- QLIKE, Sharpe mid /
crossed and buy days under the parked recalibration (diagnostic pred_clock) and under the
plain forecast on the same days, positions changed.

Parallelism: loading and the per-row statistics run in a process pool of MAX_WORKERS = 4
(the brief's local cap).

Outputs (results/close_master_table/): master_table.csv, master_table_days.csv,
master_table_extremes.csv, master_table_gates.csv, master_table_rankcorr.csv,
master_table_vs_headline.csv, before_after.csv, recal_vs_plain.csv,
master_table_provenance.csv, SUMMARY.md; then writeup/make_master_table_close_tex.py renders
writeup/generated/table_master_close.tex and writeup/master_table_close.pdf (--no-pdf skips
it).

Run:  python experiments/master_table_close.py [--arm-root linear_subsection_dedup] [--no-pdf] [--workers 4]
"""

from __future__ import annotations

import argparse
import hashlib  # 2026-10-07: the switched paper row's table hash (PAPER_SWITCHED)
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import compare_mfiv_harlag as cmh  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402
from build_subsection_yhat import GATE_REL  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

SPXW = ROOT / "results" / "spxw_pnl"
OUT = ROOT / "results" / "close_master_table"
PAPER_OUT = OUT  # 2026-10-13: the paper's run (PERBAR_SWITCH is active only when OUT is this directory)
DEFAULT_ARM_ROOT = (
    "linear_subsection"  # the first-pass arms (the run of commit 6177b74)
)
DEDUP_ARM_ROOT = "linear_subsection_dedup"  # the 16:00 campaign's de-duplicated re-run
ARM_ROOT_ENV = "MASTER_TABLE_ARM_ROOT"
OLD_LS = (
    ROOT / "results" / DEFAULT_ARM_ROOT
)  # where the first pass's stored numbers live


def _target_arm(ls: Path) -> Path:
    """One 2000-session per-bar arm: its 16:00 target is the target of every per-bar arm."""
    return (
        ls / "arms_hoffman2" / "baseline" / "ridge" / "tw2000" / "results_bar1600.csv"
    )


ARM_ROOT = os.environ.get(ARM_ROOT_ENV, DEFAULT_ARM_ROOT)
LS = ROOT / "results" / ARM_ROOT
TARGET_ARM = _target_arm(LS)
cmh.PULLED_BASE = LS


def set_arm_root(name: str) -> None:
    """Point LS, TARGET_ARM and compare_mfiv_harlag.PULLED_BASE at results/<name> (and the workers, by env)."""
    global ARM_ROOT, LS, TARGET_ARM
    ARM_ROOT = name
    LS = ROOT / "results" / name
    TARGET_ARM = _target_arm(LS)
    cmh.PULLED_BASE = LS
    os.environ[ARM_ROOT_ENV] = name  # spawned pool workers re-import this module


def design_of_root() -> str:
    return (
        "pre-campaign design"
        if ARM_ROOT == DEFAULT_ARM_ROOT
        else "de-duplicated design (16:00 campaign)"
    )


# the unclipped realized variance (every production table carries the same rv_raw)
PRODUCTION = SPXW / "yhat_blk2_fomc1.parquet"
REFERENCE = "blk2"  # key of the fixed reference forecast (the paper's headline)
ALWAYS_SHORT = "always_short"
TRADE_BAR = "16:00"
TW_MAIN = 2000  # the production training window of the per-bar arms
MAX_WORKERS = 4  # local process cap per agent (overnight brief)
BOOT_B, BOOT_BLOCK, BOOT_SEED = base.BOOT_B, base.BOOT_BLOCK, base.BOOT_SEED
ANN = base.ANN
TREES_STORED = (
    OLD_LS.parent / "linear_subsection_trees" / "rescore_local" / "trees_trade_1530.csv"
)
PAPER_ALWAYS_SHORT = (
    ROOT / "results" / "atm_straddle_0dte_1530" / "rule_by_strategy_always_short.csv"
)
# the 16:00 campaign's own stored numbers (every one written by the research scorer on these 866 days)
_R = ROOT / "results"
STORED_CAMPAIGN = {
    "linear (agent A)": _R / DEDUP_ARM_ROOT / "change_by_arm.csv",
    "trees T10/T1/RS10/RS1 + per-bar LSTM, masked (agent H)": _R
    / "trees_mask_1600"
    / "mask_levels.csv",
    "Optuna trees, single-class (agent C)": _R
    / "linear_subsection_trees_optuna_mask"
    / "report"
    / "levels.csv",
    "intraday-sequence LSTM (agent I)": _R
    / "linear_subsection_lstm_intraday"
    / "score"
    / "lstmi_levels.csv",
    "rest-of-day check rows (agent E)": _R
    / "linear_subsection_restofday_dedup"
    / "check_1530.csv",
}
BEFORE_DIR = (
    SPXW / "pre_dedup_2026-09-29"
)  # the pre-campaign tables + master table (local snapshot)
REPRO_TOL = 1e-9  # stored research numbers were written as full-precision floats
RANK_CHUNK = 40  # bootstrap draws per worker task in the rank-correlation stage (memory, not a statistic)
# The recommended headline forecast of the closing strategy (see SUMMARY.md for why).
HEADLINE = "sub_ridge_live_feasible"
# 2026-10-06: recalibration removed, user decision.  Every reported number uses the plain
# back-transform f^2 B; the recalibrated (f^2 + s) B is the diagnostic column pred_clock only.
RECAL_REMOVED = "2026-10-06"
# the committed master table of the recalibrated scorer (the run of 2026-09-30, commit e4b83a0),
# kept as the "before" of the scorer-change record (reported, never asserted)
RECAL_SNAPSHOT = OUT / "recal_2026-09-30" / "master_table.csv"
# PLAIN_FLOOR's look-back: the parked recalibration's window and warm-up (no new constant)
FLOOR_W, FLOOR_MIN = slc.SMEAR_W, slc.SMEAR_MIN
# significance level of the DM and HAC t counts in SUMMARY.md (statistics, not intervals)
SIG_LEVEL = 0.05
STORED_RECAL = "16:00 campaign, recalibrated scorer"  # stored_design of the campaign's recal numbers

EST_LONG = {"ridge": "ridge", "lasso": "lasso", "enet": "elastic net", "ols": "OLS"}
EST_SHORT = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
TREE_LONG = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
# 2026-10-06 (user decision): the dev-tuned random forest (results/rf_devmenu_2026-10-06/rf_config.json,
# rf_opt_00; results/trees_rfdev_1600/) as its own model token of the T1 / T10 rows
TREE_LONG["rfdev"] = "random forest (dev-tuned)"
# 2026-10-13 (user; writeup/MASTER_TABLE_REVIEW_2026-10-01.md items 76-79): the per-bar trees on their top 1998-2002
# entries go into the PAPER.  Every per-bar XGBoost key (T10, T1, the random search, every Optuna rung) and every
# per-bar random-forest key with the token ``rf`` keeps its key and label and reads the replacement table:
#   xgb -> results/trees_xgbtop_1600/ (xgb_opt_00, the top XGBoost entry; was xgb_opt_06, 7th-ranked)
#   rf  -> results/trees_rf400_1600/  (trial 255 of the 400-trial search; was sklearn defaults for the tuned rungs and
#          rf_opt_00 of the 87-trial search for T1 / T10 under the separate token rfdev)
# The rule is the GUI's (writeup/master_table_gui/rows.py _tree_swap_table, read only): T10 subtree_<est>_<set> ->
# yhat_subtree_<tok>_<set>; every other key subtree_<...>_<est> -> yhat_subtree_<...>_<tok>; input sets baseline,
# all_features, lean_vol_vix.  PAPER RUN ONLY: active when OUT is the paper's results/close_master_table (the GUI's
# own run redirects OUT and swaps in memory itself, so it is untouched).  The replaced tables stay in results/spxw_pnl/.
PERBAR_SWITCH = {
    "xgb": (ROOT / "results" / "trees_xgbtop_1600", "xgbtop"),
    "rf": (ROOT / "results" / "trees_rf400_1600", "rf400"),
}
PERBAR_SWITCH_SETS = ("baseline", "all_features", "lean_vol_vix")
PERBAR_SWITCH_DECIDED = "2026-10-13 (user; review items 76-79)"
VIX_BUCKETS = (
    "vix_only",
    "live_vix_only",
    "free_vix_only",
    "vix_rvol",
    "live_vix_rvol",
    "free_feasible",
    "free_feasible_vol",
)
OPTUNA_TP = (1, 5, 25, 250)  # TUNE_PER of the Optuna rung (campaign decision 4)
OPTUNA_K = (50, 25, 10)  # best of the first k of the 50 trials


def optuna_family(tp: int, k: int) -> str:
    return f"per-bar tree (Optuna, TUNE_PER={tp}, best-of-{k})"


FAM_T1 = "per-bar tree (untuned, daily refit)"
FAM_RS1 = "per-bar tree (random search, daily)"
FAM_LSTMI = "LSTM (intraday sequence)"
FAMILY_ORDER = [
    "paper",
    "per-bar linear",
    "pooled twin",
    "VIX-only family",
    "per-bar tree (untuned)",
    FAM_T1,
    "per-bar tree (tuned)",
    FAM_RS1,
    *[optuna_family(tp, k) for k in OPTUNA_K for tp in OPTUNA_TP],
    "LSTM",
    FAM_LSTMI,
    "direct rest-of-day at 15:30 (check)",
    "implied-vol representations",
    "HAR-ladder variants",
    "chain-period buckets",
    "other table",
]
TREE_FAMILIES = tuple(f for f in FAMILY_ORDER if f.startswith("per-bar tree"))
NEW_IN_CAMPAIGN = (FAM_T1, FAM_RS1, FAM_LSTMI) + tuple(
    f for f in TREE_FAMILIES if f.startswith("per-bar tree (Optuna")
)
_LIN = {
    "design": "de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped)",
    "mask": "identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns)",
    "refit": "every session",
    "tuning": "penalty every 250 sessions",
    "cpu": "Hoffman2 (agent A)",
    "changed": "design de-dup (the mask had already dropped the session-edge columns: float path only)",
}
_TREE = {
    "design": "de-duplicated per-bar design",
    "mask": "per-window mask (src/models/window_mask.py, WINDOW_MASK=1)",
    "cpu": "CARC, pinned to one CPU class (epyc-7513)",
}
# What each family IS in this run, and what changed against the pre-campaign master table
# (writeup/CAMPAIGN_16H_2026-09-29.md and writeup/PROGRESS_2026-09-29.md section I).
FAMILY_PROV: dict[str, dict[str, str]] = {
    "paper": {
        "design": "pooled 48-bar models (session-edge interactions kept, campaign decision 1)",
        "mask": "n/a",
        "refit": "as the paper",
        "tuning": "as the paper",
        "cpu": "not re-run",
        "changed": "nothing",
    },
    "pooled twin": {
        "design": "pooled 48-bar twins (session-edge interactions kept)",
        "mask": "identifiability mask",
        "refit": "every session",
        "tuning": "penalty every 250 sessions",
        "cpu": "not re-run",
        "changed": "nothing",
    },
    "per-bar linear": _LIN,
    "VIX-only family": _LIN,
    "implied-vol representations": _LIN,
    "HAR-ladder variants": _LIN,
    "chain-period buckets": _LIN,
    "per-bar tree (untuned)": _TREE
    | {
        "refit": "every 10 sessions (T10)",
        "tuning": "none (shipped configuration)",
        "changed": "design de-dup + per-window mask + CPU-class pinning (refit cadence unchanged, every 10)",
    },
    FAM_T1: _TREE
    | {
        "refit": "every session (T1)",
        "tuning": "none (shipped configuration)",
        "changed": "new",
    },
    "per-bar tree (tuned)": _TREE
    | {
        "refit": "every 10 sessions (RS10)",
        "tuning": "random search, 32 candidates, every 250 sessions (MSE rule; tunedq = QLIKE rule)",
        "changed": "design de-dup + per-window mask + CPU-class pinning (refit / tuning cadence unchanged)",
    },
    FAM_RS1: _TREE
    | {
        "refit": "every session (RS1)",
        "tuning": "random search, 32 candidates, every 250 sessions (MSE rule; tunedq = QLIKE rule)",
        "changed": "new",
    },
    **{
        optuna_family(tp, k): _TREE
        | {
            "refit": "every session",
            "tuning": f"Optuna TPE, 50 trials, best of the first {k}, every {tp} session(s)"
            + (" (MSE rule; optunaq = QLIKE rule)" if (tp, k) == (25, 50) else ""),
            "cpu": "CARC, single-class re-run on epyc-7513 (agent C; canonical)",
            "changed": "new",
        }
        for tp in OPTUNA_TP
        for k in OPTUNA_K
    },
    "LSTM": _TREE
    | {
        "mask": "per-window mask, recomputed at every refit",
        "refit": "every session (was every 10)",
        "tuning": "16 configurations x 5 seeds, every 250 sessions (MSE rule; qsel = QLIKE rule)",
        "changed": "design de-dup + per-window mask + refit cadence 10 -> 1 + CPU-class pinning",
    },
    FAM_LSTMI: {
        "design": "last N half-hour bars to 15:30 (N tuned in {13, 48, 96}), bar-level inputs",
        "mask": "per-window mask on the step columns",
        "refit": "every session",
        "tuning": "every 250 sessions (MSE rule; qsel = QLIKE rule)",
        "cpu": "CARC, pinned to epyc-7513 (agent I)",
        "changed": "new",
    },
    "direct rest-of-day at 15:30 (check)": {
        "design": "de-duplicated per-bar design, 15:30 clock of the rest-of-day model",
        "mask": "identifiability mask",
        "refit": "every session",
        "tuning": "penalty every 250 sessions",
        "cpu": "Hoffman2, each 15:30 arm pinned to its per-bar twin's CPU architecture (agent E)",
        "changed": "design de-dup + CPU-architecture pinning to the per-bar twin",
    },
}


# --------------------------------------------------------------------------- forecasts
@dataclass
class Spec:
    key: str
    label: str
    family: str
    source: str  # "table" | "arm"
    path: str = ""  # table path, or the arm's results CSV
    lo: str | None = None  # comparison-window start (None: all trade days)
    twin_key: str | None = None  # the forecast a check row must reproduce
    note: str = ""
    meta: dict = field(default_factory=dict)


def _paper_tags() -> dict[str, str]:
    """table file name -> library tag, for the tags the notebook reads."""
    return {p.name: t for t, p in asl.yhat_paths(ROOT).items()}


# 2026-10-06 (user decision: "use the pre-tuned version for all linear models"): the per-bar
# ridge / lasso / elastic net of the paper are arm (f) of results/linear_pretune_2026-10-06/
# (experiments/close_linear_pretune.py: penalty pre-tuned before 2020, HAR + calendar backbone at
# 1/100 of the exogenous penalty, retuned every 250 sessions only past the margin), stacked into
# results/spxw_pnl/yhat_subpt_<est>_<bucket>.parquet by experiments/stack_pretune_yhat.py (the
# one-penalty yhat_sub_* tables are untouched).  Key subpt_<est>_<bucket>, family "per-bar linear";
# gate (asserted, load_one): the table's pred_adj equals the study's f-arm export bit for bit.
PRETUNE_EXPORT = ROOT / "results" / "linear_pretune_2026-10-06" / "forecasts.parquet"
# 2026-10-06 (user decision): the pooled linear rows are pre-tuned too -- arm (f) of
# results/pool_linear_pretune_2026-10-06/ (Hoffman2 run of record), stacked into
# results/spxw_pnl/yhat_poolpt_<est>_<bucket>.parquet by experiments/stack_pool_pretune_yhat.py;
# key poolpt_<est>_<bucket>, family "pooled twin"; the same asserted export gate as subpt.
POOL_PRETUNE_EXPORT = (
    ROOT / "results" / "pool_linear_pretune_2026-10-06" / "forecasts.parquet"
)
_PRETUNE: dict[str, pd.Series] = {}


def pretune_series(est: str, bucket: str) -> str:
    """The f-arm export series (backbone 1/100; on baseline every column is backbone)."""
    arm = "f_retune_margin" if bucket == "baseline" else "f_retune_margin_bb"
    return f"linear_pretune/{est}_{bucket}_{arm}"


def pretune_export(sid: str) -> pd.Series:
    """One f-arm series of the study's export, pred_adj by naive-ET 16:00 stamp (cached)."""
    if not _PRETUNE:
        ex = pd.concat(
            [
                pd.read_parquet(x, columns=["series_id", "date", "pred_adj"])
                for x in (PRETUNE_EXPORT, POOL_PRETUNE_EXPORT)
                if x.is_file()
            ]
        )
        for s, g in ex.groupby("series_id"):
            if "_f_retune_margin" in str(s):
                idx = pd.DatetimeIndex(pd.to_datetime(g["date"]).to_numpy()).as_unit(
                    "ns"
                )
                _PRETUNE[str(s)] = pd.Series(
                    g["pred_adj"].to_numpy(), index=idx
                ).sort_index()
    return _PRETUNE[sid]


def describe_table(p: Path, paper: dict[str, str]) -> Spec:
    stem = p.stem.removeprefix("yhat_")
    tag = paper.get(p.name)
    if tag is not None and tag in asl.MODEL_ORDER:
        return Spec(tag, asl.YHAT_LABEL[tag], "paper", "table", str(p))
    if p.name == "yhat_b2lasso.parquet":
        return Spec(
            stem,
            "lasso (fixed 1e-4), earlier panel [on disk, unused by the paper]",
            "paper",
            "table",
            str(p),
        )
    # 2026-10-06: the pre-tuned per-bar linear tables (arm f; see PRETUNE_EXPORT)
    m = re.fullmatch(r"subpt_(ridge|lasso|enet)_(\w+)", stem)
    if m:
        est, bucket = m.groups()
        return Spec(
            stem,
            f"per-bar {EST_LONG[est]} [{bucket}], pre-tuned",
            "per-bar linear",
            "table",
            str(p),
            meta={
                "est": est,
                "bucket": bucket,
                "pretune_series": pretune_series(est, bucket),
            },
        )
    # 2026-10-06: the pre-tuned pooled linear tables (arm f; see POOL_PRETUNE_EXPORT)
    m = re.fullmatch(r"poolpt_(ridge|lasso|enet)_(\w+)", stem)
    if m:
        est, bucket = m.groups()
        arm = "f_retune_margin" if bucket == "baseline" else "f_retune_margin_bb"
        return Spec(
            stem,
            f"pooled {EST_LONG[est]} [{bucket}], pre-tuned",
            "pooled twin",
            "table",
            str(p),
            meta={
                "est": est,
                "bucket": bucket,
                "pretune_series": f"pool_linear_pretune/{est}_{bucket}_{arm}",
            },
        )
    m = re.fullmatch(r"sub_(ridge|lasso|enet)_(\w+)", stem)
    if m:
        est, bucket = m.groups()
        fam = "VIX-only family" if bucket in VIX_BUCKETS else "per-bar linear"
        return Spec(
            stem,
            f"per-bar {EST_LONG[est]} [{bucket}]",
            fam,
            "table",
            str(p),
            meta={"est": est, "bucket": bucket},
        )
    m = re.fullmatch(r"pool_(ridge|lasso|enet)_(\w+)", stem)
    if m:
        est, bucket = m.groups()
        return Spec(
            stem,
            f"pooled {EST_LONG[est]} [{bucket}], same spec",
            "pooled twin",
            "table",
            str(p),
            meta={"est": est, "bucket": bucket},
        )
    m = re.fullmatch(r"restofday_(ridge|lasso|enet)_(\w+)", stem)
    if m:
        est, bucket = m.groups()
        return Spec(
            stem,
            f"direct rest-of-day {EST_LONG[est]} [{bucket}], 15:30 clock",
            "direct rest-of-day at 15:30 (check)",
            "table",
            str(p),
            twin_key=f"sub_{est}_{bucket}",
            meta={"est": est, "bucket": bucket},
        )
    # bk = r"(all_features|baseline|live_feasible)"
    # 2026-10-06: the moments_vix input set (HAR + calendar + return moments + VIX level; src/data/loading.py)
    # bk = r"(all_features|baseline|live_feasible|moments_vix)"
    # + moments, ES volume, VIX; lean_vol_vix = signed / absolute / quartic return, autocovariance, volume, VIX
    bk = r"(all_features|baseline|live_feasible|moments_vix|moments_vol_vix|lean_vol_vix)"
    # experiments/build_subsection_tree_tuned_yhat.py: tuned = MSE-selected configuration (the arm of record),
    # tunedq = QLIKE-selected configuration (the second selection rule); _daily = refit every session (RS1)
    m = re.fullmatch(rf"subtree_tuned(q?)_(daily_)?{bk}_(lgbm|xgb|rf)", stem) or (
        re.fullmatch(rf"subtree_tuned(q?)_()(lgbm|xgb|rf)_{bk}", stem)
    )
    if m:
        q, daily, a1, a2 = m.groups()
        model, bucket = (a2, a1) if a2 in TREE_LONG else (a1, a2)
        rule = "QLIKE-selected" if q else "MSE-selected"
        if daily:
            return Spec(
                stem,
                f"tuned per-bar {TREE_LONG[model]}, daily refit [{bucket}], {rule}",
                FAM_RS1,
                "table",
                str(p),
                meta={"model": model, "bucket": bucket, "rung": "RS1" + q},
            )
        return Spec(
            stem,
            f"tuned per-bar {TREE_LONG[model]} [{bucket}], {rule}",
            "per-bar tree (tuned)",
            "table",
            str(p),
            meta={"model": model, "bucket": bucket, "rung": "RS10" + q},
        )
    # specs/causal_tune_trees_optuna.py (agent C): TUNE_PER = N, best of the first k of 50 trials
    m = re.fullmatch(
        rf"subtree_optuna(q?)_tp(\d+)(?:_k(\d+))?_{bk}_(lgbm|xgb|rf)", stem
    )
    if m:
        q, tp, k, bucket, model = m.groups()
        kk = int(k) if k else 50
        rule = ", QLIKE-selected" if q else ""
        return Spec(
            stem,
            f"Optuna per-bar {TREE_LONG[model]}, TUNE_PER {tp}, best of {kk}{rule} [{bucket}]",
            optuna_family(int(tp), kk),
            "table",
            str(p),
            meta={"model": model, "bucket": bucket, "tp": int(tp), "k": kk, "q": q},
        )
    # the untuned trees refit every session (T1, agent B / H)
    m = re.fullmatch(
        rf"subtree_daily_{bk}_(lgbm|xgb|rfdev|rf)", stem
    )  # 2026-10-06: + rfdev
    if m:
        bucket, model = m.groups()
        return Spec(
            stem,
            f"per-bar {TREE_LONG[model]}, daily refit [{bucket}]",
            FAM_T1,
            "table",
            str(p),
            meta={"model": model, "bucket": bucket, "rung": "T1"},
        )
    m = re.fullmatch(r"subtree_(lgbm|xgb|rfdev|rf)_(\w+)", stem)  # 2026-10-06: + rfdev
    if m:
        model, bucket = m.groups()
        return Spec(
            stem,
            f"per-bar {TREE_LONG[model]} [{bucket}]",
            "per-bar tree (untuned)",
            "table",
            str(p),
            meta={"model": model, "bucket": bucket, "rung": "T10"},
        )
    # the intraday-sequence LSTM (agent I) and the per-bar LSTM (daily refit after the campaign)
    m = re.fullmatch(rf"lstm_intraday(_qsel)?_{bk}", stem)
    if m:
        q, bucket = m.groups()
        rule = ", QLIKE-selected" if q else ""
        return Spec(
            stem,
            f"intraday-sequence LSTM{rule} [{bucket}]",
            FAM_LSTMI,
            "table",
            str(p),
            meta={"bucket": bucket, "q": bool(q)},
        )
    m = re.fullmatch(rf"lstm(_qsel)?_{bk}", stem)
    if m:
        q, bucket = m.groups()
        rule = ", QLIKE-selected" if q else ""
        return Spec(
            stem,
            f"per-bar LSTM{rule} [{bucket}]",
            "LSTM",
            "table",
            str(p),
            meta={"bucket": bucket, "q": bool(q)},
        )
    return Spec(stem, stem, "other table", "table", str(p))


def arm_path(pulled_dir: str, root: str, bucket: str, est: str, tw: int) -> Path:
    """The bucket study's nested layout, exactly as compare_mfiv_harlag.pulled reads it."""
    seg = "bar1600"
    return (
        cmh.PULLED_BASE
        / pulled_dir
        / root
        / bucket
        / seg
        / est
        / f"tw{tw}"
        / "causal_tune_linear"
        / est
        / bucket
        / f"results_{seg}.csv"
    )


def discover() -> tuple[list[Spec], list[dict]]:
    """Every forecast to score, and the list of what was looked at but not scored."""
    paper = _paper_tags()
    specs: list[Spec] = []
    skipped: list[dict] = []
    for p in sorted(SPXW.glob("yhat_*.parquet")):
        # 2026-10-07: a pooled tree-bank dump the paper does not read (see PAPER_SWITCHED) is not scored
        if re.fullmatch(r"yhat_tree\d\d\.parquet", p.name) and p.name not in paper:
            skipped.append(
                {
                    "key": p.stem.removeprefix("yhat_"),
                    "source": str(p.relative_to(ROOT)),
                    "reason": "pooled tree-bank dump not read by the paper (2026-10-07: xgb = bank arm 10; "
                    "arm 16 = xgb_opt_06 parked)",
                }
            )
            continue
        specs.append(describe_table(p, paper))
    have = {s.key for s in specs}
    # the per-bar linear tables were stacked from <arm root>/arms_hoffman2 (build_subsection_yhat.py):
    # the 16:00 arm file of each is a gate on the table (same pred_adj, same recalibrated forecast)
    est_dir = {v: k for k, v in EST_SHORT.items()}
    for s in specs:
        if s.meta.get(
            "pretune_series"
        ):  # 2026-10-06: gated against the study's export instead
            continue
        if s.family == "per-bar linear" and s.meta.get("bucket") in (
            "all_features",
            "baseline",
            # "live_feasible",
        ):
            a = (
                LS
                / "arms_hoffman2"
                / s.meta["bucket"]
                / est_dir[s.meta["est"]]
                / f"tw{TW_MAIN}"
                / "results_bar1600.csv"
            )
            if a.is_file():
                s.meta["arm_twin"] = str(a)
    # the per-bar OLS incumbent (HAR + calendar), carried inside every baseline arm dir
    hits = sorted(
        (LS / "baseline").glob(
            f"bar1600/*/tw{TW_MAIN}/causal_tune_linear/incumbent_ols/results_bar1600.csv"
        )
    )
    if hits:
        specs.append(
            Spec(
                "sub_ols_baseline",
                "per-bar OLS [baseline]",
                "per-bar linear",
                "arm",
                str(hits[0]),
            )
        )
    # the bucket studies (compare_mfiv_harlag.FAMILIES), 16:00 arm of each bucket x estimator
    fam_of = {
        "vixonly": "VIX-only family",
        "free": "VIX-only family",
        "ivrep": "implied-vol representations",
        "mfiv": "chain-period buckets",
        "ivslice": "chain-period buckets",
    }
    for fname, fam in cmh.FAMILIES.items():
        for bucket, tw in fam["tw"].items():
            lo = fam["windows"].get(bucket, fam["windows"]["*"])[0][1]
            for est, short in EST_SHORT.items():
                root = (
                    "linear_subsection_lassofix"
                    if est == "reclasso"
                    else "linear_subsection"
                )
                path = arm_path(fam["pulled"], root, bucket, est, tw)
                key = f"sub_{short}_{bucket}" + ("" if tw == TW_MAIN else f"_tw{tw}")
                if not path.is_file():
                    skipped.append(
                        {
                            "key": key,
                            "source": str(path.relative_to(ROOT)),
                            "reason": "arm file not on disk",
                        }
                    )
                    continue
                family = "chain-period buckets" if lo else fam_of[fname]
                if key in have:
                    # the table holds this forecast: the arm file is only a gate
                    for s in specs:
                        if s.key == key:
                            s.meta["arm_twin"] = str(path)
                    continue
                window = (
                    "" if tw == TW_MAIN else f", {tw} sessions"
                )  # the window start is in the table-B heading
                specs.append(
                    Spec(
                        key,
                        f"per-bar {EST_LONG[short]} [{bucket}]{window}",
                        family,
                        "arm",
                        str(path),
                        lo=lo,
                        meta={"study": fname, "bucket": bucket, "est": short, "tw": tw},
                    )
                )
                have.add(key)
    # the HAR-ladder variants of the live_feasible bucket (compare_mfiv_harlag part B)
    for v in cmh.HAR_VARIANTS:
        for est, short in EST_SHORT.items():
            for tw in (500, TW_MAIN):
                path = arm_path(
                    cmh.FAMILIES["mfiv"]["pulled"],
                    "linear_subsection_har/" + v,
                    "live_feasible",
                    est,
                    tw,
                )
                key = f"sub_{short}_live_feasible_har_{v}" + (
                    "" if tw == TW_MAIN else f"_tw{tw}"
                )
                if tw != TW_MAIN:
                    if path.is_file():
                        skipped.append(
                            {
                                "key": key,
                                "source": str(path.relative_to(ROOT)),
                                "reason": "training-window ablation (500 sessions) of a 2000-session arm",
                            }
                        )
                    continue
                if not path.is_file():
                    skipped.append(
                        {
                            "key": key,
                            "source": str(path.relative_to(ROOT)),
                            "reason": "arm file not on disk",
                        }
                    )
                    continue
                specs.append(
                    Spec(
                        key,
                        f"per-bar {EST_LONG[short]} [live_feasible], HAR ladder {v}",
                        "HAR-ladder variants",
                        "arm",
                        str(path),
                        meta={"study": "harlag", "variant": v, "est": short},
                    )
                )
    # 2026-10-13 (user): the per-bar trees on their top entries, paper run only (see PERBAR_SWITCH)
    if perbar_switch_active():
        for s in specs:
            q = perbar_switch_path(s.key)
            if (
                q is None
                or s.source != "table"
                or not s.family.startswith("per-bar tree")
            ):
                continue
            assert q.is_file(), f"per-bar switch: {s.key} -> {q} not on disk"
            s.meta["switched_from"] = s.path
            s.meta["perbar_switch"] = PERBAR_SWITCH[s.meta["model"]][1]
            s.path = str(q)
    return specs, skipped


def perbar_switch_active() -> bool:
    """2026-10-13: the per-bar tree switch applies to the paper's run only (OUT not redirected)."""
    return OUT.resolve() == PAPER_OUT.resolve()


def perbar_switch_path(key: str) -> Path | None:
    """The replacement table of a per-bar XGBoost / random-forest key (PERBAR_SWITCH), else None."""
    if sum(f"_{b}_" in f"_{key}_" for b in PERBAR_SWITCH_SETS) != 1:
        return None
    m = re.fullmatch(r"subtree_(xgb|rf)_(\w+)", key)  # T10
    if m:
        d, tok = PERBAR_SWITCH[m.group(1)]
        return d / f"yhat_subtree_{tok}_{m.group(2)}.parquet"
    m = re.fullmatch(r"(subtree_.+_)(xgb|rf)", key)  # T1, RS10 / RS1, Optuna
    if m:
        d, tok = PERBAR_SWITCH[m.group(2)]
        return d / f"yhat_{m.group(1)}{tok}.parquet"
    return None


# --------------------------------------------------------------------------- the common target
def load_target() -> pd.DataFrame:
    """The 16:00 target frame shared by every forecast, indexed by the naive-ET 16:00 stamp."""
    r = pd.read_csv(TARGET_ARM, parse_dates=["date"]).set_index("date").sort_index()
    r = r[(r["true_adj"] > 0) & (r["true_raw"] > 0)]
    assert (
        r.index.strftime("%H:%M") == TRADE_BAR
    ).all() and not r.index.duplicated().any()
    t = pd.DataFrame({"true_adj": r["true_adj"], "true_raw": r["true_raw"]})
    t["B"] = t["true_raw"] / t["true_adj"] ** 2
    prod = read_table_1600(PRODUCTION, ["rv_raw"])
    t["rv_unclipped"] = prod["rv_raw"].reindex(t.index)
    # PLAIN_FLOOR: the smallest 16:00 target of the previous FLOOR_W sessions, lagged one session
    t["plain_floor"] = (
        t["true_raw"].rolling(FLOOR_W, min_periods=FLOOR_MIN).min().shift(1)
    )
    t["day"] = t.index.normalize()
    early = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    t["early_close"] = t["day"].isin(early)
    return t


def read_table_1600(path: Path | str, cols: list[str]) -> pd.DataFrame:
    """A forecast table's 16:00 rows, indexed by the naive-ET stamp."""
    d = pd.read_parquet(path)
    missing = [c for c in ["t", *cols] if c not in d.columns]
    if missing:
        raise ValueError(f"columns missing: {missing}")
    t = pd.to_datetime(d["t"])
    if t.dt.tz is None:
        et = t  # a naive stamp is the panel's naive-ET convention
    else:
        et = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
    out = d.loc[m, cols].copy()
    out.index = pd.DatetimeIndex(et[m].to_numpy()).as_unit("ns")
    out.index.name = "date"
    if out.index.duplicated().any():
        raise ValueError("duplicated 16:00 stamps")
    return out.sort_index()


# --------------------------------------------------------------------------- stage 1: load + plain back-transform (worker)
_TGT: pd.DataFrame | None = None
_DECK_DAYS: pd.DatetimeIndex | None = None
_IDX: dict[int, np.ndarray] = {}


def _init(tgt: pd.DataFrame, deck_days: pd.DatetimeIndex) -> None:
    global _TGT, _DECK_DAYS
    _TGT, _DECK_DAYS = tgt, deck_days


def load_one(spec: Spec) -> dict:
    """16:00 forecast -> the plain back-transform f^2 B on the common target.  Returns frame + gates.

    The frame also carries pred_clock, the parked recalibration (2026-10-06: recalibration removed,
    user decision), as a DIAGNOSTIC column only -- no reported number uses it.
    """
    assert _TGT is not None and _DECK_DAYS is not None
    tgt = _TGT
    gates: list[dict] = []
    res: dict = {
        "key": spec.key,
        "frame": None,
        "gates": gates,
        "error": "",
        "n16": 0,
        "first16": "",
        "last16": "",
    }
    research_plain = None
    try:
        if spec.source == "table":
            has_rv = "rv_raw" in pq.read_schema(spec.path).names
            x = read_table_1600(
                spec.path, ["yhat", "baseline"] + (["rv_raw"] if has_rv else [])
            ).rename(columns={"yhat": "pred_adj"})
        else:
            r = cmh._prep(Path(spec.path))  # the research reader
            assert r is not None
            r = r[r["hhmm"] == TRADE_BAR]
            x = pd.DataFrame(
                {
                    "pred_adj": r["pred_adj"],
                    "baseline": r["baseline"],
                    "true_raw_own": r["true_raw"],
                }
            )
            # research_clock = r["pred_clock"]  # 2026-10-06: recalibration removed, user decision
            research_plain = r["pred_adj"] ** 2 * r["baseline"]
    except Exception as e:  # noqa: BLE001 -- an unreadable or malformed file is reported, not fatal
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    all_stamps = x.index  # every 16:00 stamp the file carries (finite forecast or not)
    x = x[np.isfinite(x["pred_adj"])]
    res["n16"] = len(x)
    if len(x):
        res["first16"], res["last16"] = (
            str(x.index.min().date()),
            str(x.index.max().date()),
        )
    j = x.join(tgt, how="inner")
    res["n_outside_target"] = int(len(x) - len(j))
    if not len(j):
        res["error"] = "no 16:00 stamp in the target span"
        return res
    if spec.source == "table":
        # stamps: on the arm span the table's 16:00 forecasts sit on exactly the target's stamps
        span = all_stamps[
            (all_stamps >= tgt.index.min()) & (all_stamps <= tgt.index.max())
        ]
        n_miss = len(tgt.index.difference(all_stamps))
        n_extra = len(span.difference(tgt.index))
        gates.append(
            {
                "gate": "table 16:00 stamps = the target's on the arm span (missing + extra)",
                "forecast": spec.key,
                "n": len(tgt),
                "value": float(n_miss + n_extra),
                "bound": 0.0,
                "ok": n_miss + n_extra == 0,
            }
        )
        if "rv_raw" in j:
            relR = float((j["rv_raw"] / j["rv_unclipped"] - 1.0).abs().max())
            gates.append(
                {
                    "gate": "table rv_raw equals the production rv_raw (16:00)",
                    "forecast": spec.key,
                    "n": len(j),
                    "value": relR,
                    "bound": GATE_REL,
                    "ok": relR <= GATE_REL,
                }
            )
    relB = float((j["baseline"] / j["B"] - 1.0).abs().max())
    gates.append(
        {
            "gate": "baseline equals the target's B",
            "forecast": spec.key,
            "n": len(j),
            "value": relB,
            "bound": GATE_REL,
            "ok": relB <= GATE_REL,
        }
    )
    if "true_raw_own" in j:
        relT = float((j["true_raw_own"] / j["true_raw"] - 1.0).abs().max())
        gates.append(
            {
                "gate": "arm target equals TARGET_ARM",
                "forecast": spec.key,
                "n": len(j),
                "value": relT,
                "bound": GATE_REL,
                "ok": relT <= GATE_REL,
            }
        )
    if relB > GATE_REL:
        res["error"] = (
            f"baseline differs from the target's B by {relB:.2e} (different scale)"
        )
        return res
    r = pd.DataFrame(
        {
            "true_adj": j["true_adj"],
            "pred_adj": j["pred_adj"],
            "true_raw": j["true_raw"],
            "baseline": j["B"],
        },
        index=j.index,
    )
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    # 2026-10-06: recalibration removed, user decision.  causal_forecasts is kept ONLY to fill the
    # diagnostic column pred_clock (recal_vs_plain.csv and the reported-only stored-number gates);
    # every reported number is computed from "plain" below.
    cf = slc.causal_forecasts(r)
    f = pd.DataFrame(
        {
            "pred_adj": cf["pred_adj"],
            "plain": cf["pred_adj"] ** 2 * cf["baseline"],  # THE forecast: f^2 B
            "pred_clock": cf["pred_clock"],  # diagnostic only: (f^2 + s) B, parked
            "true_raw": cf["true_raw"],
            "rv_unclipped": j["rv_unclipped"],
            "plain_floor": j["plain_floor"],
            "early_close": j["early_close"],
        },
        index=cf.index,
    )
    if research_plain is not None:
        # the plain forecast equals the research reader's pred_adj^2 x baseline on the trade days
        on = f.index[f.index.normalize().isin(_DECK_DAYS) & f["plain"].notna()]
        rp = research_plain.reindex(on)
        rel = float((f.loc[on, "plain"] / rp - 1.0).abs().max()) if len(on) else np.nan
        ok = bool(len(on) and rp.notna().all() and rel <= 1e-12)
        gates.append(
            {
                "gate": "plain forecast equals the research reader's pred_adj^2 x baseline (trade days)",
                "forecast": spec.key,
                "n": len(on),
                "value": rel,
                "bound": 1e-12,
                "ok": ok,
            }
        )
        # 2026-10-06: recalibration removed, user decision -- the pred_clock gate is parked:
        # on = f.index[f.index.normalize().isin(_DECK_DAYS) & f["pred_clock"].notna()]
        # rc = research_clock.reindex(on)
        # rel = float((f.loc[on, "pred_clock"] / rc - 1.0).abs().max()) if len(on) else np.nan
        # gates.append({"gate": "pred_clock equals the research reader's (trade days)", ...})
    arm_twin = spec.meta.get("arm_twin")
    if arm_twin:
        a = cmh._prep(Path(arm_twin))
        assert a is not None
        a = a[a["hhmm"] == TRADE_BAR]
        a = a.assign(plain=a["pred_adj"] ** 2 * a["baseline"])
        jj = f[["pred_adj", "plain"]].join(
            a[["pred_adj", "plain"]], how="inner", rsuffix="_arm"
        )
        d_adj = float((jj["pred_adj"] - jj["pred_adj_arm"]).abs().max())
        on = jj.index.normalize().isin(_DECK_DAYS)
        d_pl = float((jj.loc[on, "plain"] / jj.loc[on, "plain_arm"] - 1.0).abs().max())
        gates.append(
            {
                "gate": "table equals its arm file (pred_adj, abs)",
                "forecast": spec.key,
                "n": len(jj),
                "value": d_adj,
                "bound": 0.0,
                "ok": d_adj == 0.0 and len(jj) == len(f),
            }
        )
        gates.append(
            {
                "gate": "table plain forecast equals the research reader's on its arm file",
                "forecast": spec.key,
                "n": int(on.sum()),
                "value": d_pl,
                "bound": 1e-12,
                "ok": d_pl <= 1e-12,
            }
        )
        # 2026-10-06: recalibration removed, user decision -- parked:
        # gates.append({"gate": "table pred_clock equals the research reader on its arm file", ...})
    psid = spec.meta.get("pretune_series")
    if psid:
        # 2026-10-06: a pre-tuned table is the study's f-arm export, bit for bit
        ex = pretune_export(psid)
        e_pt = ex.reindex(f.index)
        d_pt = (
            float((f["pred_adj"] - e_pt).abs().max()) if e_pt.notna().all() else np.inf
        )
        gates.append(
            {
                "gate": "pre-tuned table equals the study's f-arm export (pred_adj, abs)",
                "forecast": spec.key,
                "n": len(f),
                "value": d_pt,
                "bound": 0.0,
                "ok": d_pt == 0.0,
            }
        )
    res["frame"] = f
    return res


# --------------------------------------------------------------------------- the trade
def trade_days(pred: pd.Series, deck: pd.DataFrame) -> pd.DataFrame:
    """score_linear_subsection.trade_1530, line for line, returning the per-day series.

    pred: forecast indexed by day (normalized).  Rows = the trade days the forecast covers.
    """
    days = pd.DatetimeIndex(pd.to_datetime(deck.index))
    f = pred.reindex(days)
    ok = f.notna().to_numpy()
    q = np.where(f.to_numpy(float) > deck["iv_var"].to_numpy(float), 1.0, -1.0)[ok]
    dk = deck[ok]
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)
    ex, r = dk["exit"].to_numpy(float), dk["R"].to_numpy(float)
    mid = q * r
    crossed = q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0)
    return pd.DataFrame({"q": q, "mid": mid, "crossed": crossed}, index=days[ok])


def sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * ANN)


def boot_idx(n: int) -> np.ndarray:
    """One set of resampled days per sample size: the days base.day_block_ci draws."""
    if n not in _IDX:
        _IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
        )
    return _IDX[n]


def boot_sharpe(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = x[idx]
    return d.mean(axis=1) / d.std(axis=1, ddof=1) * ANN


def paired_sharpe(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    idx = boot_idx(len(a))
    boot = boot_sharpe(a, idx) - boot_sharpe(b, idx)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return sharpe(a) - sharpe(b), float(lo), float(hi)


# --------------------------------------------------------------------------- stage 2: metrics (worker)
def score_one(args: tuple) -> dict:
    """Every column of one row, on the day set ``days`` (paired with the reference and always short)."""
    spec, f, ref, days, deck, is_ref = args
    fd = f.set_index(f.index.normalize()).loc[days]
    rd = ref.set_index(ref.index.normalize()).loc[days]
    dk = deck.loc[days]
    out: dict = {
        "key": spec.key,
        "n_days": len(days),
        "first_day": str(days.min().date()),
        "last_day": str(days.max().date()),
    }
    # ---- forecast accuracy: QLIKE of the PLAIN forecast f^2 B against the spec's target (also vs
    # the unclipped RV).  2026-10-06: recalibration removed, user decision -- every number below
    # uses fd["plain"]; the recalibrated pred_clock lines are parked (commented out).
    y = fd["true_raw"]
    q_plain = slc.qlike(y, fd["plain"])
    floor = fd["plain_floor"]
    flag = fd["plain"] < floor
    q_plain_fl = slc.qlike(y, fd["plain"].where(~flag, floor))
    q_ref = slc.qlike(rd["true_raw"], rd["plain"])
    q_plain_u = slc.qlike(fd["rv_unclipped"], fd["plain"])
    q_ref_u = slc.qlike(rd["rv_unclipped"], rd["plain"])
    # q_rec = slc.qlike(y, fd["pred_clock"])
    # q_ref = slc.qlike(rd["true_raw"], rd["pred_clock"])
    # q_rec_u = slc.qlike(fd["rv_unclipped"], fd["pred_clock"])
    # q_ref_u = slc.qlike(rd["rv_unclipped"], rd["pred_clock"])
    out.update(
        {
            "qlike_raw": float(q_plain.mean()),  # THE reported QLIKE (plain f^2 B)
            "qlike_raw_floored": float(q_plain_fl.mean()),
            "n_extreme_days": int(flag.sum()),
            "qlike_raw_unclipped": float(q_plain_u.mean()),
            # "qlike_recal": float(q_rec.mean()),
            # "qlike_recal_unclipped": float(q_rec_u.mean()),
            # "n_recal_below_floor": int((fd["pred_clock"] < floor).sum()),
        }
    )
    d = q_plain - q_ref
    out["qlike_pct_vs_ref"] = float(100.0 * d.mean() / q_ref.mean())
    if is_ref:
        out.update({"dm_vs_ref": np.nan, "dm_p_vs_ref": np.nan, "dm_hac_lag": np.nan})
    else:
        # 2026-10-06: intervals commented out (user): the day-block interval equals the point
        # lo, hi = base.day_block_ci(pd.Series(d.to_numpy(), index=fd.index))
        dm = dm_test(q_plain.to_numpy(), q_ref.to_numpy())
        out.update(
            {
                "dm_vs_ref": float(dm["dm"]),
                "dm_p_vs_ref": float(dm["p"]),
                "dm_hac_lag": dm["hac_lag"],
            }
        )
    out["qlike_pct_vs_ref_unclipped"] = float(
        100.0 * (q_plain_u.mean() / q_ref_u.mean() - 1.0)
    )
    out["calib_mean_realized_over_forecast"] = float(y.mean() / fd["plain"].mean())
    out["share_realized_above_forecast"] = float((y > fd["plain"]).mean())
    # ---- the last-30-min trade: q = sign(plain - iv_var)
    t = trade_days(fd["plain"], dk)
    tr = trade_days(rd["plain"], dk)
    # t = trade_days(fd["pred_clock"], dk)  # 2026-10-06: recalibration removed, user decision
    # tr = trade_days(rd["pred_clock"], dk)
    assert len(t) == len(days) == len(tr)
    short_mid = -dk["R"].to_numpy(float)
    short_x = (1.0 - dk["exit"] / (dk["bid_c"] + dk["bid_p"])).to_numpy(float)
    mid, crossed = t["mid"].to_numpy(), t["crossed"].to_numpy()
    out.update(
        {
            "Sharpe_mid": sharpe(mid),
            "Sharpe_crossed": sharpe(crossed),
            "mean_mid": float(mid.mean()),
            "mean_crossed": float(crossed.mean()),
            "hit_rate_mid": float((mid > 0).mean()),
            "hit_rate_crossed": float((crossed > 0).mean()),
            "n_buy": int((t["q"] > 0).sum()),
            "pct_buy": float(100.0 * (t["q"] > 0).mean()),
            "same_position_as_ref": float(
                (t["q"].to_numpy() == tr["q"].to_numpy()).mean()
            ),
        }
    )
    for fill, a, b, s in (
        ("mid", mid, tr["mid"].to_numpy(), short_mid),
        ("crossed", crossed, tr["crossed"].to_numpy(), short_x),
    ):
        # 2026-10-06: intervals commented out (user) -- point estimates only; the paired
        # bootstrap's percentiles (paired_sharpe's lo, hi) equal the point since b761b28.
        out[f"dSharpe_{fill}_vs_ref"] = 0.0 if is_ref else sharpe(a) - sharpe(b)
        out[f"dSharpe_{fill}_vs_short"] = sharpe(a) - sharpe(s)
        # dh, lo, hi = paired_sharpe(a, b)
        # out.update({f"dSharpe_{fill}_vs_ref_lo": lo, f"dSharpe_{fill}_vs_ref_hi": hi})
        # dh, lo, hi = paired_sharpe(a, s)
        # out.update({f"dSharpe_{fill}_vs_short_lo": lo, f"dSharpe_{fill}_vs_short_hi": hi})
    t_s, _lag_s = asl.newey_west_t(mid - short_mid)
    out["t_hac_mid_vs_short"] = t_s
    if not is_ref:
        t_hac, lag = asl.newey_west_t(mid - tr["mid"].to_numpy())
        out.update({"t_hac_mid_vs_ref": t_hac, "hac_lag": lag})
    else:
        out.update({"t_hac_mid_vs_ref": np.nan, "hac_lag": np.nan})
    return out


def score_short(days: pd.DatetimeIndex, deck: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """The always-short row (no forecast) on ``days``."""
    dk = deck.loc[days]
    rd = ref.set_index(ref.index.normalize()).loc[days]
    mid = -dk["R"].to_numpy(float)
    crossed = (1.0 - dk["exit"] / (dk["bid_c"] + dk["bid_p"])).to_numpy(float)
    tr = trade_days(rd["plain"], dk)  # the reference's plain sign(s) trade
    # tr = trade_days(rd["pred_clock"], dk)  # 2026-10-06: recalibration removed, user decision
    out = {
        "key": ALWAYS_SHORT,
        "n_days": len(days),
        "first_day": str(days.min().date()),
        "last_day": str(days.max().date()),
        "Sharpe_mid": sharpe(mid),
        "Sharpe_crossed": sharpe(crossed),
        "mean_mid": float(mid.mean()),
        "mean_crossed": float(crossed.mean()),
        "hit_rate_mid": float((mid > 0).mean()),
        "hit_rate_crossed": float((crossed > 0).mean()),
        "n_buy": 0,
        "pct_buy": 0.0,
        "same_position_as_ref": float((tr["q"].to_numpy() == -1.0).mean()),
    }
    for fill, a, b in (
        ("mid", mid, tr["mid"].to_numpy()),
        ("crossed", crossed, tr["crossed"].to_numpy()),
    ):
        # 2026-10-06: intervals commented out (user) -- point estimates only
        # dh, lo, hi = paired_sharpe(a, b)
        out.update(
            {
                f"dSharpe_{fill}_vs_ref": sharpe(a) - sharpe(b),
                f"dSharpe_{fill}_vs_short": 0.0,
            }
        )
    t_hac, lag = asl.newey_west_t(mid - tr["mid"].to_numpy())
    out.update({"t_hac_mid_vs_ref": t_hac, "hac_lag": lag})
    return out


# --------------------------------------------------------------------------- stage 3: QLIKE vs Sharpe rank correlation (worker)
def _ranks(a: np.ndarray) -> np.ndarray:
    """Ranks along axis 0 (forecasts)."""
    return np.argsort(np.argsort(a, axis=0), axis=0).astype(float)


def _spearman_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean(axis=0)
    ry -= ry.mean(axis=0)
    return (rx * ry).sum(axis=0) / np.sqrt(
        (rx * rx).sum(axis=0) * (ry * ry).sum(axis=0)
    )


def rankcorr_chunk(args: tuple) -> np.ndarray:
    """Spearman(QLIKE mean, Sharpe) across forecasts on each resampled day set of the chunk.

    2026-10-06: no longer called (intervals commented out, user): the bootstrap is disabled, so
    every draw would return the point estimate.
    """
    qmat, rmat, lo, hi = args
    idx = boot_idx(qmat.shape[1])[lo:hi]
    q = qmat[:, idx].mean(axis=2)  # (F, draws)
    r = rmat[:, idx]
    s = r.mean(axis=2) / r.std(axis=2, ddof=1)
    return _spearman_cols(q, s)


# --------------------------------------------------------------------------- reproduction gates
def repro_gates(
    tab: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    deck_days: pd.DatetimeIndex,
    rvp: pd.DataFrame,
) -> list[dict]:
    """The stored research numbers, recomputed here, must agree -- scorer- and design-aware.

    2026-10-06: recalibration removed, user decision.  Every stored number written with the
    recalibrated forecast is REPORTED (asserted = False) against the same quantity recomputed
    here from the diagnostic pred_clock (``rvp``, the scorer-change frame indexed by key); stored
    plain numbers (agent C's qlike_raw) stay asserted with the de-dup root (reported with the
    default one); the paper's always-short row (no forecast) is asserted always.
    """
    g: list[dict] = []
    by = tab.set_index("key")
    dedup = ARM_ROOT != DEFAULT_ARM_ROOT
    PRE, CAMP = "pre-campaign design", STORED_RECAL
    CAMP_PLAIN = "16:00 campaign, plain forecast"

    def chk(
        name: str,
        key: str,
        col: str,
        want: float,
        design: str,
        asserted: bool,
        got: float | None = None,
        n: int | None = None,
        frame: pd.DataFrame | None = None,
    ) -> None:
        src = by if frame is None else frame
        if got is None and key not in src.index:
            g.append(
                {
                    "gate": name,
                    "forecast": key,
                    "n": 0,
                    "value": np.nan,
                    "bound": REPRO_TOL,
                    "ok": False,
                    "asserted": asserted,
                    "stored_design": design,
                }
            )
            return
        if got is None:
            got = float(src.loc[key, col])
            n = int(src.loc[key, "n_days"])
        g.append(
            {
                "gate": name,
                "forecast": key,
                "n": n,
                "value": abs(got - want),
                "bound": REPRO_TOL,
                "ok": bool(abs(got - want) <= REPRO_TOL),
                "asserted": asserted,
                "stored_design": design,
            }
        )

    # ---- (1) the first pass's stored numbers (recalibrated scorer, pre-campaign design): reported
    if TREES_STORED.is_file():
        st = pd.read_csv(TREES_STORED)
        for _, r in st.iterrows():
            if r["stamps"] == "own":
                key = f"subtree_{r['model']}_{r['bucket']}"
            else:
                est = r["forecast"].removeprefix("linear ")
                key = f"sub_{EST_SHORT.get(est, est)}_{r['bucket']}"
            for col in ("Sharpe_mid", "Sharpe_crossed"):
                chk(
                    f"trees_trade_1530.csv {col} (recalibrated; vs the diagnostic pred_clock)",
                    key,
                    col + "_recal",
                    float(r[col]),
                    PRE,
                    False,  # was: not dedup -- 2026-10-06: recalibration removed, user decision
                    frame=rvp,
                )
    for fname in ("vixonly", "free", "ivrep", "mfiv", "ivslice"):
        f = OLD_LS / cmh.FAMILIES[fname]["pulled"] / f"compare_{fname}.csv"
        if not f.is_file():
            continue
        st = pd.read_csv(f)
        st = st[
            (st["arm"] == "bar1600")
            & ~st["bucket"].str.startswith("live_feasible (incumbent")
        ]
        for _, r in st.iterrows():
            tw = cmh.FAMILIES[fname]["tw"][r["bucket"]]
            key = f"sub_{r['estimator']}_{r['bucket']}" + (
                "" if tw == TW_MAIN else f"_tw{tw}"
            )
            for col, scol in (
                ("Sharpe_mid", "trade_Sharpe_mid"),
                ("Sharpe_crossed", "trade_Sharpe_crossed"),
            ):
                chk(
                    f"compare_{fname}.csv {scol} (recalibrated; vs the diagnostic pred_clock)",
                    key,
                    col + "_recal",
                    float(r[scol]),
                    PRE,
                    False,  # was: not dedup -- 2026-10-06: recalibration removed, user decision
                    frame=rvp,
                )
    # ---- (2) the paper's always-short row (no forecast: the same under every design and scorer)
    if PAPER_ALWAYS_SHORT.is_file():
        p = pd.read_csv(PAPER_ALWAYS_SHORT, index_col=0).iloc[0]
        chk(
            "paper always-short Sharpe (rule_by_strategy_always_short.csv)",
            ALWAYS_SHORT,
            "Sharpe_mid",
            float(p["Sharpe_ann"]),
            "paper",
            True,
        )
        chk(
            "paper always-short mean",
            ALWAYS_SHORT,
            "mean_mid",
            float(p["mean"]),
            "paper",
            True,
        )
    # ---- (3) the 16:00 campaign's own stored numbers (recalibrated scorer): reported
    f = STORED_CAMPAIGN["linear (agent A)"]
    if f.is_file():
        # A: trade_1530 on each arm's recalibrated forecast over every trade day it covers
        st = pd.read_csv(f)
        for _, r in st[~st["forecast"].str.startswith("lstm")].iterrows():
            key = r["forecast"]
            if key not in frames:
                chk(f"{f.name} Sharpe_mid_new", key, "Sharpe_mid", np.nan, CAMP, False)
                continue
            pc = frames[key]["pred_clock"]  # the diagnostic recalibrated forecast
            pc = pc[pc.index.normalize().isin(deck_days)].dropna()
            t = base.trade_1530(pc)
            for col in ("Sharpe_mid", "Sharpe_crossed"):
                chk(
                    f"{f.name} {col}_new (recalibrated, every trade day the arm covers)",
                    key,
                    col,
                    float(r[f"{col}_new"]),
                    CAMP,
                    False,  # was: dedup -- 2026-10-06: recalibration removed, user decision
                    got=float(t[col]),
                    n=int(t["deck_days"]),
                )
    cols_h = {
        "qlike": "qlike_recal",
        "sharpe_mid": "Sharpe_mid_recal",
        "sharpe_crossed": "Sharpe_crossed_recal",
        "pct_buy": "pct_buy_recal",
    }
    f = STORED_CAMPAIGN["trees T10/T1/RS10/RS1 + per-bar LSTM, masked (agent H)"]
    if f.is_file():
        st = pd.read_csv(f)
        canon = {"T10", "T1", "RS10", "RS1", "RS10q", "RS1q", "LSTM", "LSTMq"}
        st = st[
            ((st["variant"] == "mask") & st["rung"].isin(canon))
            | (st["variant"] == "ridge")
        ]
        for _, r in st.iterrows():
            key = str(r["table"]).removeprefix("yhat_")
            for sc, col in cols_h.items():
                chk(
                    f"{f.parent.name}/{f.name} {sc} (recalibrated)",
                    key,
                    col,
                    float(r[sc]),
                    CAMP,
                    False,  # was: dedup -- 2026-10-06: recalibration removed, user decision
                    frame=rvp,
                )
    f = STORED_CAMPAIGN["Optuna trees, single-class (agent C)"]
    if f.is_file():
        st = pd.read_csv(f)
        st = st[st["path"].str.contains("yhat_subtree_optuna", regex=False)]
        for _, r in st.iterrows():
            key = Path(r["path"]).stem.removeprefix("yhat_")
            # the plain QLIKE is the reported number: asserted with the de-dup root
            # 2026-10-13: reported, not asserted, for a key switched to its top-entry table (PERBAR_SWITCH)
            switched = perbar_switch_active() and perbar_switch_path(key) is not None
            chk(
                f"linear_subsection_trees_optuna_mask/report/{f.name} qlike_raw (plain)",
                key,
                "qlike_raw",
                float(r["qlike_raw"]),
                CAMP_PLAIN,
                dedup and not switched,
            )
            for col in ("qlike_recal", "Sharpe_mid", "Sharpe_crossed", "pct_buy"):
                chk(
                    f"linear_subsection_trees_optuna_mask/report/{f.name} {col} (recalibrated)",
                    key,
                    col if col == "qlike_recal" else col + "_recal",
                    float(r[col]),
                    CAMP,
                    False,  # was: dedup -- 2026-10-06: recalibration removed, user decision
                    frame=rvp,
                )
            # 2026-10-06: intervals commented out (user) -- the stored interval ends are not compared:
            # for col in ("dSharpe_mid_vs_ref_lo", "dSharpe_mid_vs_ref_hi"): chk(...)
    f = STORED_CAMPAIGN["intraday-sequence LSTM (agent I)"]
    if f.is_file():
        st = pd.read_csv(f)
        st = st[st["forecast"].isin(["intraday", "intraday_qsel"])]
        for _, r in st.iterrows():
            key = f"lstm_{r['forecast']}_{r['bucket']}"
            for sc, col in cols_h.items():
                chk(
                    f"{f.name} {sc} (recalibrated)",
                    key,
                    col,
                    float(r[sc]),
                    CAMP,
                    False,  # was: dedup -- 2026-10-06: recalibration removed, user decision
                    frame=rvp,
                )
    f = STORED_CAMPAIGN["rest-of-day check rows (agent E)"]
    if f.is_file():
        st = pd.read_csv(f)
        for _, r in st[st["phase"] == "after"].iterrows():
            key = f"restofday_{r['estimator']}_{r['bucket']}"
            if key not in rvp.index or not np.isfinite(
                rvp.loc[key, "twin_max_rel_pred_clock"]
            ):
                chk(f"{f.name} max_rel_pred_clock", key, "", np.nan, CAMP, False)
                continue
            x = rvp.loc[key]
            chk(
                f"{f.name} max_rel_pred_clock (vs the per-bar twin; recalibrated)",
                key,
                "",
                float(r["max_rel_pred_clock"]),
                CAMP,
                False,  # was: dedup -- 2026-10-06: recalibration removed, user decision
                got=float(x["twin_max_rel_pred_clock"]),
                n=int(x["n_days"]),
            )
            chk(
                f"{f.name} same_position (days; recalibrated)",
                key,
                "",
                float(r["same_position"]),
                CAMP,
                False,  # was: dedup -- 2026-10-06: recalibration removed, user decision
                got=float(x["twin_same_position_recal"]) * int(x["n_days"]),
                n=int(x["n_days"]),
            )
    # ---- (4) the committed recalibrated master table (RECAL_SNAPSHOT): reported, never asserted
    if RECAL_SNAPSHOT.is_file():
        snap = pd.read_csv(RECAL_SNAPSHOT, float_precision="round_trip").set_index(
            "key"
        )
        SNAP = "committed recalibrated master table (2026-09-30)"
        for key in snap.index:
            if key == ALWAYS_SHORT:
                continue
            # the old plain QLIKE column is the new reported QLIKE: same forecast, same days
            chk(
                f"{RECAL_SNAPSHOT.parent.name}/master_table.csv qlike_raw (plain) = this run's qlike_raw",
                key,
                "qlike_raw",
                float(snap.loc[key, "qlike_raw"]),
                SNAP,
                False,
            )
            for col in ("qlike_recal", "Sharpe_mid", "Sharpe_crossed"):
                chk(
                    f"{RECAL_SNAPSHOT.parent.name}/master_table.csv {col} (recalibrated) = the diagnostic",
                    key,
                    col if col == "qlike_recal" else col + "_recal",
                    float(snap.loc[key, col]),
                    SNAP,
                    False,
                    frame=rvp,
                )
    return g


def recal_vs_plain(
    tab: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    day_sets: dict[str, pd.DatetimeIndex],
    deck: pd.DataFrame,
    common: pd.DatetimeIndex,
) -> pd.DataFrame:
    """The scorer change, per forecast on its own scored days: the parked recalibration (diagnostic
    pred_clock) against the plain forecast -- QLIKE, Sharpe mid / crossed, buy share, positions.

    2026-10-06: recalibration removed, user decision.  Nothing here is a reported number of the
    master table; it records what removing the recalibration changed.
    """
    rows = []
    t_ = tab.set_index("key")
    for k, f in frames.items():
        days = day_sets[k]
        fd = f.set_index(f.index.normalize()).loc[days]
        ok = fd["pred_clock"].notna()
        n_rec = int(ok.sum())
        row: dict = {
            "key": k,
            "label": t_.loc[k, "label"],
            "family": t_.loc[k, "family"],
            "table": t_.loc[k, "table"],
            "n_days": len(days),
            "n_days_recal_defined": n_rec,
            "qlike_plain": float(t_.loc[k, "qlike_raw"]),
            "Sharpe_mid_plain": float(t_.loc[k, "Sharpe_mid"]),
            "Sharpe_crossed_plain": float(t_.loc[k, "Sharpe_crossed"]),
            "pct_buy_plain": float(t_.loc[k, "pct_buy"]),
        }
        if n_rec == len(days):
            dk = deck.loc[days]
            tp = trade_days(fd["plain"], dk)
            tc = trade_days(fd["pred_clock"], dk)
            row.update(
                {
                    "qlike_recal": float(
                        slc.qlike(fd["true_raw"], fd["pred_clock"]).mean()
                    ),
                    "Sharpe_mid_recal": sharpe(tc["mid"].to_numpy()),
                    "Sharpe_crossed_recal": sharpe(tc["crossed"].to_numpy()),
                    "pct_buy_recal": float(100.0 * (tc["q"] > 0).mean()),
                    "positions_changed": int(
                        (tp["q"].to_numpy() != tc["q"].to_numpy()).sum()
                    ),
                    "buy_to_sell": int(((tc["q"] > 0) & (tp["q"] < 0)).sum()),
                    "sell_to_buy": int(((tc["q"] < 0) & (tp["q"] > 0)).sum()),
                    "mean_s_over_plain": float(
                        ((fd["pred_clock"] - fd["plain"]) / fd["plain"]).mean()
                    ),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("qlike", "Sharpe_mid", "Sharpe_crossed", "pct_buy"):
        if f"{c}_recal" not in out:
            out[f"{c}_recal"] = np.nan
        out[f"d_{c}_plain_minus_recal"] = out[f"{c}_plain"] - out[f"{c}_recal"]
    out["qlike_pct_change"] = (
        100.0 * out["d_qlike_plain_minus_recal"] / out["qlike_recal"]
    )
    # the rest-of-day check rows against their per-bar twin, on the recalibrated forecast
    out["twin_max_rel_pred_clock"] = np.nan
    out["twin_same_position_recal"] = np.nan
    for i, r in out.iterrows():
        tw = t_.loc[r["key"], "twin_key"]
        if isinstance(tw, str) and tw and tw in frames:
            a_, b_ = frames[r["key"]], frames[tw]
            on = a_.index[a_.index.normalize().isin(common)]
            out.loc[i, "twin_max_rel_pred_clock"] = float(
                (a_.loc[on, "pred_clock"] / b_.loc[on, "pred_clock"] - 1.0).abs().max()
            )
            ta = trade_days(
                a_.loc[on, "pred_clock"].set_axis(on.normalize()), deck.loc[common]
            )
            tb = trade_days(
                b_.loc[on, "pred_clock"].set_axis(on.normalize()), deck.loc[common]
            )
            out.loc[i, "twin_same_position_recal"] = float(
                (ta["q"].to_numpy() == tb["q"].to_numpy()).mean()
            )
    return out


# --------------------------------------------------------------------------- summary
# 2026-10-06 (user): "remove adj_ in the names of any variable being presented" -- design column
# names are SHOWN without the leading adj_ (adj_vix_ma_* -> vix_ma_*; har_ma_* unchanged); data files,
# keys and code identifiers keep their names.  Applied to every `code span` of SUMMARY.md.
ADJ_PREFIX = re.compile(r"(?<![A-Za-z0-9_])adj_(?=[A-Za-z])")


def display_name(name: str) -> str:
    """A design column name as presented: the leading adj_ dropped."""
    return ADJ_PREFIX.sub("", name)


def display_code_spans(line: str) -> str:
    """Every `code span` of a SUMMARY line through display_name."""
    return re.sub(r"`([^`]+)`", lambda m: "`" + display_name(m.group(1)) + "`", line)


def fmt_ci(v: float, lo: float, hi: float, nd: int = 2) -> str:
    """2026-10-06: no longer used (intervals commented out, user); kept for the parked lines."""
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def hac_p(t: float) -> float:
    """Two-sided normal p-value of a HAC t statistic (a statistic, not an interval)."""
    return float(math.erfc(abs(t) / math.sqrt(2.0))) if np.isfinite(t) else np.nan


def fmt_t(t: float) -> str:
    """A HAC t with its two-sided normal p-value."""
    return f"t {t:+.2f}, p {hac_p(t):.3f}" if np.isfinite(t) else "t n/a"


def write_summary(
    tab: pd.DataFrame,
    cov: pd.DataFrame,
    rk: pd.DataFrame,
    ext: pd.DataFrame,
    extra: dict,
    gates: pd.DataFrame,
) -> None:
    A = tab[(tab["table"] == "A") & (tab["key"] != ALWAYS_SHORT)].set_index("key")
    ref = A.loc[REFERENCE]
    short = tab.set_index("key").loc[ALWAYS_SHORT]
    L: list[str] = []
    L.append("# Master table for the closing strategy (A4) — summary")
    L.append("")
    L.append(
        f"Written by `experiments/master_table_close.py --arm-root {ARM_ROOT}` on {time.strftime('%Y-%m-%d %H:%M')}; every number below is read from"
    )
    L.append(
        "`master_table.csv` (and `before_after.csv`, `recal_vs_plain.csv`) of the same run. Full table: `writeup/master_table_close.pdf`; tex: "
        "`writeup/generated/table_master_close.tex`. Every forecast is the plain back-transform f²·B; every number is a point estimate."
    )
    L.append("")
    L.append("## Provenance: design, window mask, CPU class")
    L.append("")
    for line in extra["provenance"]:
        L.append(line)
    L.append("")
    L.append("## Scorer, days, reference")
    L.append("")
    L.append(
        "- **Scorer: the plain back-transform.** Every forecast is used as f²·B (f = the model's forecast on the adjusted scale "
        "sqrt(RV/B), B = the diurnal baseline; the `yhat`² · `baseline` of `results/spxw_pnl/yhat_*.parquet`); QLIKE (`qlike_raw`) "
        "against the per-bar spec's 16:00 target; the trade is the deck's 15:30 sign(s) on the straddle (`trade_1530`): buy when "
        "f²·B exceeds the 15:30 implied variance, sell otherwise. The research scorer's 16:00-bar recalibration (f² + s)·B "
        f"(s = the forecast's own trailing 250-session mean squared adjusted-scale error, lagged one) was removed on {RECAL_REMOVED} "
        "(user decision); it survives only as the diagnostic column `pred_clock` and in `recal_vs_plain.csv` (section below). "
        "The notebook's 13-bar Mincer–Zarnowitz map is **not** used either: no number here may be set beside a notebook number."
    )
    L.append(
        "- **Point estimates only.** The circular block bootstrap is disabled (commit b761b28: every draw is the original sample), "
        "so no interval is reported. The inference shown is the Diebold–Mariano statistic on the daily QLIKE difference (`dm_test`, "
        "Newey–West variance with the Harvey–Leybourne–Newbold correction) and the Newey–West HAC t of the daily mid-return "
        f"difference (`newey_west_t`); counts 'significant' use the two-sided {100 * SIG_LEVEL:.0f} % level of those statistics."
    )
    nA = int(extra["n_intersection_A"])
    fc_ = A[A["family"] != "direct rest-of-day at 15:30 (check)"]
    L.append(
        f"- **Days:** table A = {len(fc_)} forecasts in {fc_['family'].nunique()} families (+ "
        f"{int((A['family'] == 'direct rest-of-day at 15:30 (check)').sum())} check rows) on the same {nA} trade days "
        f"({extra['first_day']} .. {extra['last_day']}); intersection over table A = {nA} of {extra['n_deck']} trade days. "
        f"Every family that covers the {extra['n_deck']} trade days is in table A. "
        f"Table B = {extra['n_B']} forecasts scored on their own days (paired with the reference on those days): "
        f"{extra['B_list'] or 'none'}. Forecasts that failed to load: {extra['failed'] or 'none'}."
    )
    L.append(
        f"- **Which forecasts drop days:** {extra['drop_line']} Coverage per forecast: `master_table_days.csv`."
    )
    L.append(
        f"- **Reference:** the block-diagonal ridge (`{REFERENCE}`, `yhat_blk2_fomc1.parquet`) — the paper's headline forecast "
        "(HAR ladder, penalty 1, plus the exogenous block, penalty 100, on the panel of record), the forecast the paper's 15:30 deck trades. "
        f"Under this scorer it has QLIKE {ref['qlike_raw']:.4f} and sign(s) Sharpe {ref['Sharpe_mid']:.2f} mid / "
        f"{ref['Sharpe_crossed']:.2f} crossed (the deck's own map gives it 1.34 mid; the difference is the scorer). "
        f"Second reference: always short, Sharpe {short['Sharpe_mid']:.2f} mid / {short['Sharpe_crossed']:.2f} crossed."
    )
    L.append("")
    L.append("## (a) Recommended headline forecast")
    L.append("")
    h = A.loc[HEADLINE] if HEADLINE in A.index else None
    if h is not None:
        L.append(
            f"**{h['label']}** (`{HEADLINE}`), scored with the plain forecast. On the {int(h['n_days'])} days: "
            f"QLIKE {h['qlike_raw']:.4f} ({h['qlike_pct_vs_ref']:+.1f} % vs the reference, DM {h['dm_vs_ref']:+.2f}, p {h['dm_p_vs_ref']:.3f}); "
            f"sign(s) Sharpe {h['Sharpe_mid']:.2f} mid / {h['Sharpe_crossed']:.2f} crossed, mean {h['mean_mid']:+.3f} per unit premium "
            f"(crossed {h['mean_crossed']:+.3f}), hit rate {100 * h['hit_rate_mid']:.1f} %, buys on {int(h['n_buy'])} days ({h['pct_buy']:.1f} %). "
            f"Sharpe difference vs the reference: mid {h['dSharpe_mid_vs_ref']:+.2f}, crossed {h['dSharpe_crossed_vs_ref']:+.2f} "
            f"(HAC {fmt_t(h['t_hac_mid_vs_ref'])} on the daily mid-return difference); vs always short: mid {h['dSharpe_mid_vs_short']:+.2f}, "
            f"crossed {h['dSharpe_crossed_vs_short']:+.2f} (HAC {fmt_t(h['t_hac_mid_vs_short'])})."
        )
        L.append("")
        for line in extra["beats_headline"]:
            L.append(line)
        L.append("")
        for line in extra["headline_reasons"]:
            L.append(line)
    L.append("")
    L.append(f"## Scorer change: recalibrated -> plain ({RECAL_REMOVED})")
    L.append("")
    for line in extra["scorer_change"]:
        L.append(line)
    L.append("")
    L.append("## Before / after the 16:00 campaign (both sides plain)")
    L.append("")
    for line in extra["before_after"]:
        L.append(line)
    L.append("")
    L.append("## (b) QLIKE vs Sharpe across models")
    L.append("")
    for line in extra["pattern_lines"]:
        L.append(line)
    L.append("")
    L.append(
        "Rank correlation across table-A forecasts (check rows and exact duplicates excluded), point estimate "
        "(a negative value = lower QLIKE goes with higher Sharpe):"
    )
    L.append("")
    L.append("| set | forecasts | QLIKE | Sharpe | Spearman |")
    L.append("|---|---:|---|---|---:|")
    for _, r in rk.iterrows():
        L.append(
            f"| {r['set']} | {int(r['n_forecasts'])} | {r['qlike']} | {r['sharpe']} | {r['spearman']:+.2f} |"
        )
    L.append("")
    for line in extra["rank_reading"]:
        L.append(line)
    L.append("")
    L.append("## (c) Extreme back-transform values")
    L.append("")
    for line in extra["extreme_lines"]:
        L.append(line)
    L.append("")
    L.append("## Gates")
    L.append("")
    asr = gates[gates["asserted"]]
    bad = asr[~asr["ok"]]
    rep = gates[~gates["asserted"]]
    L.append(
        f"{len(asr)} gates checked, {len(bad)} failed (`master_table_gates.csv`, column `asserted`). Among them: every arm file's 16:00 "
        "target equals the common target"
        + (
            f" and the common target of `{ARM_ROOT}` equals `{DEFAULT_ARM_ROOT}`'s bit for bit"
            if ARM_ROOT != DEFAULT_ARM_ROOT
            else ""
        )
        + "; every table's baseline equals its B, its rv_raw the production table's, and its 16:00 stamps the target's; every arm "
        "file's plain forecast equals the research reader's f²·B on the trade days; every per-bar linear and VIX-only table equals "
        "its arm file; every row's trade aggregates equal `trade_1530`'s on the plain forecast; every row of "
        "`master_table_daily.parquet` has q = sign(plain - iv_var) and the daily frame reproduces every row's Sharpe, QLIKE and buys; "
        "no table falls into 'other table'; the paper's always-short row is reproduced"
        + (
            "; the families the campaign did not touch equal the pre-campaign table under the plain scorer"
            if ARM_ROOT != DEFAULT_ARM_ROOT
            else ""
        )
        + "; stored plain numbers ("
        + "; ".join(
            f"{src}: {int((asr['stored_design'] == src).sum())}"
            for src in sorted(set(asr["stored_design"]) - {""})
        )
        + ")."
    )
    if len(rep):
        L.append("")
        by_src = rep.groupby("stored_design")
        L.append(
            f"Reported, not asserted ({len(rep)}): stored numbers written with the recalibrated forecast (compared with the same "
            "quantity recomputed here from the diagnostic `pred_clock`) or on another design — "
            + "; ".join(
                f"{src}: {int(g['ok'].sum())} of {len(g)} agree to {REPRO_TOL:g}"
                for src, g in by_src
            )
            + ". A difference there is a design change or an input that changed since the stored number was written, not a failure."
        )
    if len(bad):
        L.append("")
        L.append("Failed gates:")
        for _, r in bad.iterrows():
            L.append(f"- {r['gate']} — {r['forecast']}: {r['value']}")
    L.append("")
    L.append("## Top of table A by sign(s) Sharpe (mid)")
    L.append("")
    L.append(
        "| forecast | QLIKE | Δ% vs ref | DM | Sharpe mid | Sharpe crossed | ΔSharpe mid vs ref | HAC t vs ref | ΔSharpe mid vs short | buy days |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    top = (
        A[
            (A["family"] != "direct rest-of-day at 15:30 (check)")
            & (A["duplicate_of"] == "")
        ]
        .sort_values("Sharpe_mid", ascending=False)
        .head(12)
    )
    for k, r in top.iterrows():
        dm = "" if k == REFERENCE else f"{r['dm_vs_ref']:+.2f}"
        th = "" if k == REFERENCE else f"{r['t_hac_mid_vs_ref']:+.2f}"
        L.append(
            f"| {r['label']} | {r['qlike_raw']:.4f} | {r['qlike_pct_vs_ref']:+.1f} | {dm} | {r['Sharpe_mid']:.2f} | "
            f"{r['Sharpe_crossed']:.2f} | {r['dSharpe_mid_vs_ref']:+.2f} | {th} | {r['dSharpe_mid_vs_short']:+.2f} | {int(r['n_buy'])} |"
        )
    L.append("")
    L.append(
        "Wording: sign(s) = buy the straddle when the plain 16:00 forecast f²·B exceeds the 15:30 implied variance, sell otherwise; "
        "straddle = nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position; the last-30-min trade "
        "enters at 15:30 and settles at the close. Buckets: `baseline` = HAR + calendar (`har_ma_*`, calendar dummies), `all_features` = "
        "the full design, `live_feasible` = the 16 series a 15:30 forecaster rebuilds live (ES return moments and liquidity, VIX/VVIX/VIX3M "
        "(`adj_vix_ma_*` …), FOMC calendar), `vix_only` = HAR + calendar + VIX level, `live_vix_only` = live_feasible minus VVIX and VIX3M, "
        "`free_vix_only` = the free feed (ES 1-min bars + VIX + FOMC calendar). Tree rungs: T10 / T1 = the shipped configuration refit "
        "every 10 sessions / every session; RS10 / RS1 = random search (32 candidates, every 250 sessions) refit every 10 / every session; "
        "Optuna = TPE, 50 trials per tuning point, tuning every TUNE_PER sessions, best of the first k trials, refit every session."
    )
    L = [
        display_code_spans(x) for x in L
    ]  # presented names: no leading adj_ (user, 2026-10-06)
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument(
        "--no-pdf",
        action="store_true",
        help="skip writeup/make_master_table_close_tex.py",
    )
    ap.add_argument(
        "--arm-root",
        default=os.environ.get(ARM_ROOT_ENV, DEFAULT_ARM_ROOT),
        help=f"results/<arm root> of the arm-only rows and the target (default {DEFAULT_ARM_ROOT}; "
        f"the 16:00 campaign: {DEDUP_ARM_ROOT})",
    )
    ap.add_argument(
        "--before-dir",
        default=str(BEFORE_DIR),
        help="the pre-campaign master_table.csv + master_table_daily.parquet (before/after)",
    )
    a = ap.parse_args()
    set_arm_root(a.arm_root)
    assert LS.is_dir(), f"arm root {LS} not on disk"
    workers = max(1, min(a.workers, MAX_WORKERS))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"arm root: {LS.relative_to(ROOT)} ({design_of_root()})")

    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    deck_days = deck.index
    tgt = load_target()
    gates: list[dict] = []
    if ARM_ROOT != DEFAULT_ARM_ROOT:
        # the target frame comes from the chosen root: it must be the default root's, bit for bit
        t_old = pd.read_csv(_target_arm(OLD_LS), parse_dates=["date"]).set_index("date")
        t_new = pd.read_csv(TARGET_ARM, parse_dates=["date"]).set_index("date")
        same_idx = t_old.index.equals(t_new.index)
        nbad = (
            int(
                (
                    (t_old["true_adj"].to_numpy() != t_new["true_adj"].to_numpy())
                    | (t_old["true_raw"].to_numpy() != t_new["true_raw"].to_numpy())
                ).sum()
            )
            if same_idx
            else len(t_new)
        )
        gates.append(
            {
                "gate": f"TARGET_ARM of {ARM_ROOT} equals {DEFAULT_ARM_ROOT}'s bit for bit (rows differing)",
                "forecast": "(target)",
                "n": len(t_new),
                "value": float(nbad),
                "bound": 0.0,
                "ok": same_idx and nbad == 0,
            }
        )
    specs, skipped = discover()
    print(
        f"{len(specs)} forecasts discovered ({sum(s.source == 'table' for s in specs)} tables, {sum(s.source == 'arm' for s in specs)} arm files); "
        # f"{len(skipped)} arm files looked at and not scored"
        f"{len(skipped)} arm files / parked tables looked at and not scored"
    )
    by_key = {s.key: s for s in specs}
    assert len(by_key) == len(specs), "duplicate forecast keys"
    assert REFERENCE in by_key, f"reference {REFERENCE} not on disk"
    other = [s.key for s in specs if s.family == "other table"]
    gates.append(
        {
            "gate": "no forecast falls into 'other table' (every table named by the parser)",
            "forecast": ", ".join(other) or "(all)",
            "n": len(specs),
            "value": float(len(other)),
            "bound": 0.0,
            "ok": not other,
        }
    )
    # 2026-10-07: a paper row switched by decision reads the verified table (see PAPER_SWITCHED)
    for k, sw in PAPER_SWITCHED.items():
        src = Path(by_key[k].path) if k in by_key else None
        h = (
            hashlib.sha256(src.read_bytes()).hexdigest()
            if src is not None and src.is_file()
            else ""
        )
        gates.append(
            {
                "gate": f"paper row {k} reads {sw['table']}, the verified dump (sha256 as recorded)",
                "forecast": k,
                "n": 1,
                "value": float(
                    src is None or src.name != sw["table"] or h != sw["sha256"]
                ),
                "bound": 0.0,
                "ok": src is not None and src.name == sw["table"] and h == sw["sha256"],
            }
        )

    # 2026-10-13 (user): every per-bar tree key switched to its top-entry table (PERBAR_SWITCH), paper run only
    sw_specs = [s for s in specs if s.meta.get("perbar_switch")]
    if perbar_switch_active():
        want = [
            s
            for s in specs
            if s.source == "table"
            and s.family.startswith("per-bar tree")
            and perbar_switch_path(s.key) is not None
        ]
        gates.append(
            {
                "gate": "per-bar switch (2026-10-13): every per-bar xgb / rf key on baseline / all_features / "
                "lean_vol_vix reads its xgbtop / rf400 table (count not switched)",
                "forecast": ", ".join(
                    sorted({s.meta.get("perbar_switch", "?") for s in sw_specs})
                )
                or "(none)",
                "n": len(want),
                "value": float(len(want) - len(sw_specs)),
                "bound": 0.0,
                "ok": len(sw_specs) == len(want) and len(sw_specs) > 0,
            }
        )
    print(
        f"per-bar switch: {len(sw)} per-bar tree keys read the xgbtop / rf400 tables (paper run: {perbar_switch_active()})"
    )

    # ---- stage 1: load + plain back-transform (pred_clock: the parked recalibration, diagnostic only)
    frames: dict[str, pd.DataFrame] = {}
    cov_rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(tgt, deck_days)
    ) as pool:
        loaded = list(pool.map(load_one, specs))
    for s, res in zip(specs, loaded):
        gates += res["gates"]
        row = {
            "key": s.key,
            "label": s.label,
            "family": s.family,
            "source": str(Path(s.path).relative_to(ROOT)).replace("\\", "/"),
            "rows_16h": res["n16"],
            "first_16h": res["first16"],
            "last_16h": res["last16"],
            "window_start": s.lo or "",
            "load_error": res["error"],
        }
        if res["frame"] is not None and not res["error"]:
            f = res["frame"]
            frames[s.key] = f
            # 2026-10-06: recalibration removed, user decision -- coverage of the PLAIN forecast
            has = f.index[f["plain"].notna()].normalize()
            want = deck_days if not s.lo else deck_days[deck_days >= s.lo]
            missing = want.difference(has)
            row.update(
                {
                    "rows_outside_target_span": res.get("n_outside_target", 0),
                    "first_forecast_day": str(has.min().date()) if len(has) else "",
                    "trade_days_in_window": len(want),
                    "trade_days_covered": len(want) - len(missing),
                    "trade_days_missing": len(missing),
                    "missing_days": " ".join(str(d.date()) for d in missing[:40])
                    + (" ..." if len(missing) > 40 else ""),
                }
            )
        cov_rows.append(row)
    cov = pd.DataFrame(cov_rows)
    print(f"[{time.time() - t0:.0f} s] stage 1 (load + plain back-transform) done")
    failed = cov[cov["load_error"] != ""]
    for _, r in failed.iterrows():
        print(f"NOT SCORED {r['key']}: {r['load_error']}")

    # ---- the day sets (on the plain forecast)
    def plain_days(fr: pd.DataFrame) -> pd.DatetimeIndex:
        return fr.index[fr["plain"].notna()].normalize()

    full = [
        k
        for k in frames
        if not by_key[k].lo
        and int(cov.set_index("key").loc[k, "trade_days_missing"]) == 0
    ]
    common = deck_days
    for k in full:
        common = common.intersection(plain_days(frames[k]))
    all_int = deck_days
    for k in frames:
        all_int = all_int.intersection(plain_days(frames[k]))
    print(
        f"table A: {len(full)} forecasts, intersection {len(common)} of {len(deck_days)} trade days; "
        f"intersection over every scored forecast (incl. table B): {len(all_int)}"
    )
    assert REFERENCE in full, "the reference must cover every trade day"
    cov["table"] = [
        "A" if k in full else ("B" if k in frames else "not scored") for k in cov["key"]
    ]
    for sk in skipped:
        cov = pd.concat(
            [
                cov,
                pd.DataFrame(
                    [
                        {
                            "key": sk["key"],
                            "source": sk["source"].replace("\\", "/"),
                            "load_error": sk["reason"],
                            "table": "not scored",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # ---- stage 2: every row's columns (point estimates), in the pool
    ref = frames[REFERENCE]
    jobs = []
    day_sets: dict[str, pd.DatetimeIndex] = {}
    for k, f in frames.items():
        s = by_key[k]
        if k in full:
            days = common
        else:
            days = plain_days(f).intersection(deck_days)
            if s.lo:
                days = days[days >= s.lo]
            days = days.intersection(plain_days(ref))
        day_sets[k] = days
        jobs.append((s, f, ref, days, deck, k == REFERENCE))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(tgt, deck_days)
    ) as pool:
        rows = list(pool.map(score_one, jobs, chunksize=4))
    rows.append(score_short(common, deck, ref))
    print(f"[{time.time() - t0:.0f} s] stage 2 (columns) done")

    # gate: every row's trade aggregates equal trade_1530's on the plain forecast (the research function itself)
    for (s, f, _r, days, _d, _i), row in zip(jobs, rows):
        pc = f["plain"]
        # pc = f["pred_clock"]  # 2026-10-06: recalibration removed, user decision
        pc = pc[pc.index.normalize().isin(days)]
        t = base.trade_1530(pc)
        dv = max(
            abs(t["Sharpe_mid"] - row["Sharpe_mid"]),
            abs(t["Sharpe_crossed"] - row["Sharpe_crossed"]),
            abs(t["pct_buy"] - row["pct_buy"]),
        )
        ok = t["deck_days"] == row["n_days"] and dv <= 1e-12
        gates.append(
            {
                "gate": "trade aggregates equal trade_1530 (plain forecast)",
                "forecast": s.key,
                "n": row["n_days"],
                "value": dv,
                "bound": 1e-12,
                "ok": ok,
            }
        )

    tab = pd.DataFrame(rows)
    meta = pd.DataFrame(
        [
            {
                "key": s.key,
                "label": s.label,
                "family": s.family,
                "source": s.source,
                "window_start": s.lo or "",
                "twin_key": s.twin_key or "",
                "perbar_switch": s.meta.get(
                    "perbar_switch", ""
                ),  # 2026-10-13: xgbtop / rf400 (PERBAR_SWITCH)
            }
            for s in specs
        ]
        + [
            {
                "key": ALWAYS_SHORT,
                "label": "always short (no forecast)",
                "family": "rule",
                "source": "",
                "window_start": "",
                "twin_key": "",
                "perbar_switch": "",
            }
        ]
    )
    tab = meta.merge(tab, on="key", how="inner")
    tab["table"] = [
        "A" if (k in full or k == ALWAYS_SHORT) else "B" for k in tab["key"]
    ]
    tab["is_reference"] = tab["key"] == REFERENCE
    tab["is_headline"] = tab["key"] == HEADLINE

    # check rows: the direct rest-of-day forecast at 15:30 against its per-bar twin (plain forecasts)
    tab["twin_max_rel_plain"] = np.nan
    tab["twin_same_position"] = np.nan
    for i, r in tab.iterrows():
        tw = r["twin_key"]
        if tw and tw in frames and r["key"] in frames:
            a_, b_ = frames[r["key"]], frames[tw]
            on = a_.index[a_.index.normalize().isin(common)]
            rel = float((a_.loc[on, "plain"] / b_.loc[on, "plain"] - 1.0).abs().max())
            ta = trade_days(
                a_.loc[on, "plain"].set_axis(on.normalize()), deck.loc[common]
            )
            tb = trade_days(
                b_.loc[on, "plain"].set_axis(on.normalize()), deck.loc[common]
            )
            tab.loc[i, "twin_max_rel_plain"] = rel
            tab.loc[i, "twin_same_position"] = float(
                (ta["q"].to_numpy() == tb["q"].to_numpy()).mean()
            )

    # order: family, then Sharpe within family (reference first in its family)
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER + ["rule"])}
    tab["_f"] = tab["family"].map(lambda v: fam_rank.get(v, len(fam_rank)))
    tab["_t"] = tab["table"].map({"A": 0, "B": 1})
    paper_rank = {t: i for i, t in enumerate(asl.MODEL_ORDER)}
    tab["_p"] = tab["key"].map(lambda k: paper_rank.get(k, 99))
    tab = tab.sort_values(
        ["_t", "_f", "_p", "Sharpe_mid"], ascending=[True, True, True, False]
    ).drop(columns=["_f", "_t", "_p"])

    # the scorer change (diagnostic): the parked recalibration against the plain forecast, per key
    rvp = recal_vs_plain(tab, frames, day_sets, deck, common)
    gates += repro_gates(tab, frames, deck_days, rvp.set_index("key"))
    print(f"[{time.time() - t0:.0f} s] scorer change (recal -> plain) done")

    # exact duplicates: the same plain forecast, bit for bit, on the same days as an earlier row
    tab["duplicate_of"] = ""
    seen: dict[bytes, str] = {}
    for i, r in tab.iterrows():
        k = r["key"]
        if k not in frames:
            continue
        on = frames[k].index[frames[k].index.normalize().isin(day_sets[k])]
        sig = on.asi8.tobytes() + frames[k].loc[on, "plain"].to_numpy(float).tobytes()
        if sig in seen:
            tab.loc[i, "duplicate_of"] = seen[sig]
        else:
            seen[sig] = k

    # every table-A forecast against the HEADLINE (the same machinery, the headline as the reference)
    vs_h = pd.DataFrame()
    if HEADLINE in full:
        hjobs = [
            (by_key[k], frames[k], frames[HEADLINE], common, deck, k == HEADLINE)
            for k in full
        ]
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init, initargs=(tgt, deck_days)
        ) as pool:
            hrows = list(pool.map(score_one, hjobs, chunksize=4))
        keep = [
            "key",
            "qlike_raw",
            "qlike_pct_vs_ref",
            "dm_vs_ref",
            "dm_p_vs_ref",
            "Sharpe_mid",
            "Sharpe_crossed",
            "dSharpe_mid_vs_ref",
            "dSharpe_crossed_vs_ref",
            "t_hac_mid_vs_ref",
            "same_position_as_ref",
            # 2026-10-06: intervals commented out (user):
            # "qlike_diff_ci_lo", "qlike_diff_ci_hi", "dSharpe_mid_vs_ref_lo", "dSharpe_mid_vs_ref_hi",
            # "dSharpe_crossed_vs_ref_lo", "dSharpe_crossed_vs_ref_hi",
        ]
        vs_h = pd.DataFrame(hrows)[keep].rename(
            columns=lambda c: c.replace("_ref", "_headline")
        )
        vs_h = tab[tab["table"] == "A"][["key", "label", "family"]].merge(
            vs_h, on="key", how="inner"
        )

    print(f"[{time.time() - t0:.0f} s] vs-headline done")

    # ---- extremes: the plain back-transform below PLAIN_FLOOR, on every 16:00 session of the arm span
    ext_rows = []
    allsess = []
    for k, f in frames.items():
        m = f["plain_floor"].notna()
        g = f[m]
        fl = g["plain"] < g["plain_floor"]
        qp = slc.qlike(g["true_raw"], g["plain"])
        qpf = slc.qlike(g["true_raw"], g["plain"].where(~fl, g["plain_floor"]))
        qne = qp[~g["early_close"]]
        allsess.append(
            {
                "key": k,
                "sessions": int(len(g)),
                "early_close_sessions": int(g["early_close"].sum()),
                "n_extreme_all_sessions": int(fl.sum()),
                "n_extreme_early_close": int((fl & g["early_close"]).sum()),
                "qlike_raw_all_sessions": float(qp.mean()),
                "qlike_raw_floored_all_sessions": float(qpf.mean()),
                "qlike_raw_all_sessions_no_early_close": float(qne.mean()),
            }
        )
        for d, rr in g[fl].iterrows():
            ext_rows.append(
                {
                    "key": k,
                    "stamp": str(d),
                    "trade_day": d.normalize() in deck_days,
                    "early_close": bool(rr["early_close"]),
                    "yhat": rr["pred_adj"],
                    "plain_forecast": rr["plain"],
                    "plain_floor": rr["plain_floor"],
                    "target_true_raw": rr["true_raw"],
                    "qlike_raw_this_day": float(
                        slc.qlike(
                            pd.Series([rr["true_raw"]]), pd.Series([rr["plain"]])
                        ).iloc[0]
                    ),
                    # "recal_forecast": rr["pred_clock"],  # 2026-10-06: recalibration removed, user decision
                }
            )
    ext = pd.DataFrame(
        ext_rows,
        columns=[
            "key",
            "stamp",
            "trade_day",
            "early_close",
            "yhat",
            "plain_forecast",
            "plain_floor",
            "target_true_raw",
            "qlike_raw_this_day",
        ],
    )
    tab = tab.merge(pd.DataFrame(allsess), on="key", how="left")

    # ---- stage 3: QLIKE vs Sharpe across forecasts (rank correlation, point estimate only)
    core = tab[
        (tab["table"] == "A")
        & tab["family"].isin(FAMILY_ORDER)
        & (tab["family"] != "direct rest-of-day at 15:30 (check)")
    ]
    core = core[~core["key"].isin([ALWAYS_SHORT]) & (core["duplicate_of"] == "")]
    sets = {
        "all table-A forecasts": core["key"].tolist(),
        "per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies)": core[
            ~core["family"].isin(["paper", "pooled twin"])
        ]["key"].tolist(),
        "per-bar linear + VIX-only family": core[
            core["family"].isin(["per-bar linear", "VIX-only family"])
        ]["key"].tolist(),
        "per-bar nonlinear (every tree rung, LSTM)": core[
            core["family"].isin(list(TREE_FAMILIES) + ["LSTM", FAM_LSTMI])
        ]["key"].tolist(),
        "48-bar forecasts (paper + pooled twins)": core[
            core["family"].isin(["paper", "pooled twin"])
        ]["key"].tolist(),
    }
    q_of = {}
    r_of = {}
    cm = common
    for k in core["key"]:
        f = frames[k].set_index(frames[k].index.normalize()).loc[cm]
        q_of[("plain", k)] = slc.qlike(f["true_raw"], f["plain"]).to_numpy(float)
        # q_of[("recal", k)] = slc.qlike(f["true_raw"], f["pred_clock"]).to_numpy(float)  # 2026-10-06: removed
        t = trade_days(f["plain"], deck.loc[cm])
        r_of[("mid", k)] = t["mid"].to_numpy()
        r_of[("crossed", k)] = t["crossed"].to_numpy()
    rk_rows = []
    # 2026-10-06: intervals commented out (user) -- the day-block bootstrap of the rank correlation
    # (rankcorr_chunk over RANK_CHUNK-draw chunks, BOOT_B draws) is not run: every draw is the
    # original sample since b761b28, so its percentile interval would equal the point.
    for sname, keys in sets.items():
        if len(keys) < 4:
            continue
        for fill in ("mid", "crossed"):
            qmat = np.vstack([q_of[("plain", k)] for k in keys])
            rmat = np.vstack([r_of[(fill, k)] for k in keys])
            point = float(
                _spearman_cols(
                    qmat.mean(axis=1)[:, None],
                    (rmat.mean(axis=1) / rmat.std(axis=1, ddof=1))[:, None],
                )[0]
            )
            rk_rows.append(
                {
                    "set": sname,
                    "n_forecasts": len(keys),
                    "qlike": "plain",
                    "sharpe": fill,
                    "spearman": point,
                }
            )
    rk = pd.DataFrame(rk_rows)
    print(f"[{time.time() - t0:.0f} s] stage 3 (rank correlation) done")

    # ---- per-day long table for the P&L and loss-vs-P&L work (regenerable; not committed)
    daily = []
    for k, f in frames.items():
        dd = f.set_index(f.index.normalize()).loc[day_sets[k]]
        t = trade_days(dd["plain"], deck.loc[dd.index])
        daily.append(
            pd.DataFrame(
                {
                    "key": k,
                    "day": dd.index,
                    "plain": dd["plain"].to_numpy(),  # THE forecast: f^2 B
                    "pred_clock": dd[
                        "pred_clock"
                    ].to_numpy(),  # diagnostic only (parked recalibration)
                    "target": dd["true_raw"].to_numpy(),
                    "rv_unclipped": dd["rv_unclipped"].to_numpy(),
                    "iv_var": deck.loc[dd.index, "iv_var"].to_numpy(float),
                    "qlike_raw": slc.qlike(dd["true_raw"], dd["plain"]).to_numpy(),
                    # "qlike_recal": slc.qlike(dd["true_raw"], dd["pred_clock"]).to_numpy(),  # 2026-10-06: removed
                    "q": t["q"].to_numpy(),
                    "ret_mid": t["mid"].to_numpy(),
                    "ret_crossed": t["crossed"].to_numpy(),
                }
            )
        )
    daily_df = pd.concat(daily, ignore_index=True)
    # gate: the daily position is the plain sign(s) on every row
    q_want = np.where(daily_df["plain"] > daily_df["iv_var"], 1.0, -1.0)
    n_bad = int((daily_df["q"].to_numpy() != q_want).sum())
    gates.append(
        {
            "gate": "master_table_daily.parquet: q = sign(plain - iv_var) on every row (rows differing)",
            "forecast": "(all)",
            "n": len(daily_df),
            "value": float(n_bad),
            "bound": 0.0,
            "ok": n_bad == 0 and bool(daily_df["plain"].notna().all()),
        }
    )
    # gate: the daily frame reproduces the master table, key by key (Sharpe mid / crossed, QLIKE, buys)
    tk = tab.set_index("key")
    for k, g in daily_df.groupby("key", sort=False):
        dv = max(
            abs(sharpe(g["ret_mid"].to_numpy()) - tk.loc[k, "Sharpe_mid"]),
            abs(sharpe(g["ret_crossed"].to_numpy()) - tk.loc[k, "Sharpe_crossed"]),
            abs(float(g["qlike_raw"].mean()) - tk.loc[k, "qlike_raw"]),
            abs(float((g["q"] > 0).sum()) - tk.loc[k, "n_buy"]),
        )
        gates.append(
            {
                "gate": "master_table_daily.parquet reproduces master_table.csv (Sharpe mid / crossed, QLIKE, buys)",
                "forecast": k,
                "n": len(g),
                "value": dv,
                "bound": 1e-12,
                "ok": len(g) == int(tk.loc[k, "n_days"]) and dv <= 1e-12,
            }
        )

    # ---- before / after: the pre-campaign master table against this run, key by key (both plain)
    before_dir = Path(a.before_dir)
    ba, ba_gates = before_after(tab, daily_df, before_dir, deck)
    gates += ba_gates
    print(f"[{time.time() - t0:.0f} s] before/after done ({len(ba)} keys)")

    # ---- write
    gtab = pd.DataFrame(gates)
    gtab["asserted"] = gtab["asserted"].astype("boolean").fillna(True).astype(bool)
    gtab["stored_design"] = gtab["stored_design"].fillna("")
    gtab["ok"] = gtab["ok"].astype(bool)
    prov = provenance_table(tab)
    tab.to_csv(OUT / "master_table.csv", index=False, lineterminator="\n")
    cov.to_csv(OUT / "master_table_days.csv", index=False, lineterminator="\n")
    ext.to_csv(OUT / "master_table_extremes.csv", index=False, lineterminator="\n")
    gtab.to_csv(OUT / "master_table_gates.csv", index=False, lineterminator="\n")
    rk.to_csv(OUT / "master_table_rankcorr.csv", index=False, lineterminator="\n")
    prov.to_csv(OUT / "master_table_provenance.csv", index=False, lineterminator="\n")
    rvp.to_csv(OUT / "recal_vs_plain.csv", index=False, lineterminator="\n")
    if len(vs_h):
        vs_h.to_csv(
            OUT / "master_table_vs_headline.csv", index=False, lineterminator="\n"
        )
    if len(ba):
        ba.to_csv(OUT / "before_after.csv", index=False, lineterminator="\n")
    daily_df.to_parquet(OUT / "master_table_daily.parquet", index=False)

    # ---- summary prose (numbers from this run)
    A = tab[(tab["table"] == "A") & (tab["key"] != ALWAYS_SHORT)].set_index("key")
    extra = {
        "n_intersection_A": len(common),
        "n_deck": len(deck_days),
        "first_day": str(common.min().date()),
        "last_day": str(common.max().date()),
        "n_B": int((tab["table"] == "B").sum()),
        "B_list": "; ".join(
            f"{len(g)} {fam} forecasts from {first} on {n} trade days"
            for (fam, first, n), g in tab[tab["table"] == "B"].groupby(
                ["family", "first_day", "n_days"]
            )
        ),
        "failed": ", ".join(
            f"{r['key']} ({r['load_error']})" for _, r in failed.iterrows()
        ),
        "before_dir": before_dir,
    }
    extra["headline_reasons"] = headline_reasons(A, vs_h)
    extra["beats_headline"] = beats_headline_lines(A, vs_h)
    extra["provenance"] = provenance_lines(prov, gtab)
    extra["before_after"] = before_after_lines(ba, before_dir)
    extra["scorer_change"] = scorer_change_lines(rvp, A)
    drop = cov[
        (cov["table"] != "not scored") & (cov["trade_days_missing"].fillna(0) > 0)
    ]
    Bt = tab[tab["table"] == "B"]
    extra["drop_line"] = (
        f"no table-A forecast misses a trade day; {len(drop)} forecasts miss trade days inside their own window"
        + (f" ({', '.join(drop['key'])})" if len(drop) else "")
        + f". The {len(Bt)} table-B forecasts start at their comparison window"
        + (
            f" ({', '.join(sorted(set(Bt['first_day'])))}), so each omits the {len(deck_days) - int(Bt['n_days'].max())} earlier trade days by design"
            if len(Bt)
            else ""
        )
        + " (their implied-vol inputs are real chain data only from then; compare_mfiv_harlag.CHAIN_START)."
    )
    extra["pattern_lines"] = pattern_lines(A)
    extra["rank_reading"] = rank_reading(A, rk)
    extra["extreme_lines"] = extreme_lines(tab, ext)
    write_summary(tab, cov, rk, ext, extra, gtab)

    # ---- print
    show = [
        "label",
        "n_days",
        "qlike_raw",
        "qlike_pct_vs_ref",
        "dm_vs_ref",
        "Sharpe_mid",
        "Sharpe_crossed",
        "mean_mid",
        "hit_rate_mid",
        "n_buy",
        "dSharpe_mid_vs_ref",
        "t_hac_mid_vs_ref",
        "dSharpe_mid_vs_short",
    ]
    for tb in ("A", "B"):
        print(f"\n=== table {tb} ===")
        print(tab[tab["table"] == tb][show].round(3).to_string(index=False))
    print("\nrank correlation QLIKE vs Sharpe:")
    print(rk.round(3).to_string(index=False))
    asr = gtab[gtab["asserted"]]
    bad = asr[~asr["ok"]]
    rep = gtab[~gtab["asserted"]]
    print(
        f"\ngates: {len(asr)} checked, {len(bad)} failed "
        f"(+ {len(rep)} reported, not asserted: {int(rep['ok'].sum())} agree, {int((~rep['ok']).sum())} differ)"
    )
    if len(bad):
        print(bad.to_string(index=False))
    print(
        f"extreme plain values: {len(ext)} rows ({int(ext['trade_day'].sum()) if len(ext) else 0} on trade days)"
    )
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")
    if not a.no_pdf:
        sys.stdout.flush()
        for extra_args in ([], ["--per-bar-only"]):
            rc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "writeup" / "make_master_table_close_tex.py"),
                    *extra_args,
                ],
                check=False,
            ).returncode
            if rc:
                print(
                    f"WARNING: writeup/make_master_table_close_tex.py {' '.join(extra_args)} exited {rc}; "
                    "the CSVs and SUMMARY.md are written"
                )
    return 1 if len(bad) else 0


# --------------------------------------------------------------------------- summary prose helpers
def _row(A: pd.DataFrame, k: str) -> pd.Series | None:
    return A.loc[k] if k in A.index else None


NEIGHBOURS = (  # the headline's closest alternatives, compared with it directly in SUMMARY.md
    "sub_lasso_live_feasible",
    "sub_enet_live_feasible",
    "sub_ridge_all_features",
    "sub_ridge_baseline",
    "sub_lasso_free_vix_only",
    "sub_ridge_free_vix_only",
    "subtree_lgbm_live_feasible",
    "subtree_xgb_live_feasible",
)


def headline_reasons(A: pd.DataFrame, vs_h: pd.DataFrame) -> list[str]:
    """Why the headline: stated as rules, the numbers filled in from this run (point estimates)."""
    h = _row(A, HEADLINE)
    if h is None:
        return [f"(the headline `{HEADLINE}` is not in table A on this run)"]
    lines = ["Why this one:"]
    fc = A[
        (A["family"] != "direct rest-of-day at 15:30 (check)")
        & (A["duplicate_of"] == "")
    ]
    live = fc[
        fc["label"].str.contains(
            r"\[(?:live_feasible|live_vix_only|free_vix_only|vix_only)\]", regex=True
        )
        & ~fc["family"].isin(["pooled twin"])
    ]
    rank_s = int((fc["Sharpe_mid"] > h["Sharpe_mid"]).sum()) + 1
    rank_q = int((fc["qlike_raw"] < h["qlike_raw"]).sum()) + 1
    lines.append(
        f"- It is a 15:30-specific model on the inputs a 15:30 forecaster can rebuild live (`live_feasible`), so the trade it scores can be run. "
        f"It ranks {rank_s} of {len(fc)} table-A forecasts on sign(s) Sharpe (mid) and {rank_q} of {len(fc)} on QLIKE (plain forecast)."
    )
    best = fc.sort_values("Sharpe_mid", ascending=False)
    top = best.iloc[0]
    if top.name != HEADLINE:
        lines.append(
            f"- The highest Sharpe in table A is {top['label']} ({top['Sharpe_mid']:.2f} mid / {top['Sharpe_crossed']:.2f} crossed). Choosing "
            f"the maximum of {len(fc)} Sharpe ratios would select on the trade's own noise; the headline is chosen on feasibility and "
            "on being the per-bar model of record (the per-bar ridge is the estimator the 15:30 study defined first), "
            "not on the maximum."
        )
    v = vs_h.set_index("key") if len(vs_h) else pd.DataFrame()
    late = []  # the nonlinear groups: each one's best by Sharpe and by QLIKE
    groups = [
        [f]
        for f in (
            "per-bar tree (untuned)",
            FAM_T1,
            "per-bar tree (tuned)",
            FAM_RS1,
        )
    ]
    groups += [[f for f in TREE_FAMILIES if f.startswith("per-bar tree (Optuna")]]
    groups += [["LSTM"], [FAM_LSTMI]]
    for fams in groups:
        g = fc[fc["family"].isin(fams)]
        if len(g):
            late += list(g.sort_values("Sharpe_mid", ascending=False).index[:1]) + [
                g["qlike_raw"].idxmin()
            ]
    ex = [k for k in dict.fromkeys(NEIGHBOURS + tuple(late)) if k in v.index]
    if ex:
        lines.append(
            "- Paired against the headline itself on the same days (`master_table_vs_headline.csv`; a negative QLIKE % / DM = the "
            "alternative forecasts better, a positive Sharpe difference = the alternative trades better; point estimates, HAC t of the "
            "daily mid-return difference):"
        )
        lines.append("")
        lines.append(
            "  | alternative | QLIKE | % vs headline | DM | Sharpe mid / crossed | ΔSharpe mid vs headline | ΔSharpe crossed vs headline | HAC t (mid) | same position |"
        )
        lines.append("  |---|---:|---:|---:|---|---:|---:|---:|---:|")
        for k in ex:
            r = v.loc[k]
            lines.append(
                f"  | {r['label']} | {r['qlike_raw']:.4f} | {r['qlike_pct_vs_headline']:+.1f} | {r['dm_vs_headline']:+.2f} | "
                f"{r['Sharpe_mid']:.2f} / {r['Sharpe_crossed']:.2f} | {r['dSharpe_mid_vs_headline']:+.2f} | "
                f"{r['dSharpe_crossed_vs_headline']:+.2f} | {r['t_hac_mid_vs_headline']:+.2f} | "
                f"{100 * r['same_position_as_headline']:.1f} % |"
            )
        lines.append("")
        others = v.drop(index=HEADLINE)
        others = others[others.index.isin(fc.index)]
        sig_t = others["t_hac_mid_vs_headline"].map(hac_p) < SIG_LEVEL
        sig_q = others["dm_p_vs_headline"] < SIG_LEVEL
        up, dn = (
            others["dSharpe_mid_vs_headline"] > 0,
            others["dSharpe_mid_vs_headline"] < 0,
        )
        lines.append(
            f"- Over all {len(others)} other table-A forecasts (point estimates): {int(up.sum())} trade better than the headline at the "
            f"mid fill ({int((others['dSharpe_crossed_vs_headline'] > 0).sum())} crossed), {int((up & sig_t).sum())} of them with the HAC t "
            f"of the daily mid-return difference significant at {100 * SIG_LEVEL:.0f} %; {int(dn.sum())} trade worse "
            f"({int((dn & sig_t).sum())} significant); {int((others['qlike_pct_vs_headline'] < 0).sum())} have a lower QLIKE "
            f"({int(((others['qlike_pct_vs_headline'] < 0) & sig_q).sum())} with DM p < {SIG_LEVEL:g}), "
            f"{int((others['qlike_pct_vs_headline'] > 0).sum())} a higher one ({int(((others['qlike_pct_vs_headline'] > 0) & sig_q).sum())} with DM p < {SIG_LEVEL:g})."
        )
    rest = fc.drop(index=REFERENCE, errors="ignore")
    sig_r = rest["t_hac_mid_vs_ref"].map(hac_p) < SIG_LEVEL
    sig_s = fc["t_hac_mid_vs_short"].map(hac_p) < SIG_LEVEL
    lines.append(
        f"- Against the reference, {int((rest['dSharpe_mid_vs_ref'] > 0).sum())} of {len(rest)} table-A forecasts have a higher mid-fill "
        f"Sharpe ({int((rest['dSharpe_crossed_vs_ref'] > 0).sum())} at the crossed fill); {int(((rest['dSharpe_mid_vs_ref'] > 0) & sig_r).sum())} "
        f"of them with the HAC t of the daily mid-return difference significant at {100 * SIG_LEVEL:.0f} %. Against always short, "
        f"{int((fc['dSharpe_mid_vs_short'] > 0).sum())} (mid) and {int((fc['dSharpe_crossed_vs_short'] > 0).sum())} (crossed) of {len(fc)} "
        f"have a higher Sharpe; {int(((fc['dSharpe_mid_vs_short'] > 0) & sig_s).sum())} with a significant HAC t (mid)."
    )
    lq = live.sort_values("qlike_raw")
    if len(lq):
        b = lq.iloc[0]
        lines.append(
            f"- On forecast accuracy alone the best live-feasible forecast is {b['label']} (QLIKE {b['qlike_raw']:.4f}, "
            f"Sharpe {b['Sharpe_mid']:.2f} / {b['Sharpe_crossed']:.2f}); a QLIKE-first choice would take it. The trade does not rank "
            "forecasts the way QLIKE does (section b), which is why the headline names the scorer AND the trade numbers."
        )
    return lines


def pattern_lines(A: pd.DataFrame) -> list[str]:
    fc = A[
        (A["family"] != "direct rest-of-day at 15:30 (check)")
        & (A["duplicate_of"] == "")
    ]
    out = []
    for fam in FAMILY_ORDER:
        g = fc[fc["family"] == fam]
        if not len(g):
            continue
        bq = g.sort_values("qlike_raw").iloc[0]
        bs = g.sort_values("Sharpe_mid", ascending=False).iloc[0]
        out.append(
            f"- **{fam}** ({len(g)}): QLIKE {g['qlike_raw'].min():.4f} .. {g['qlike_raw'].max():.4f}, Sharpe mid {g['Sharpe_mid'].min():.2f} .. "
            f"{g['Sharpe_mid'].max():.2f}; best QLIKE {bq['label']} ({bq['qlike_raw']:.4f}, Sharpe {bq['Sharpe_mid']:.2f}); best Sharpe "
            f"{bs['label']} ({bs['Sharpe_mid']:.2f}, QLIKE {bs['qlike_raw']:.4f})."
        )
    fq = fc.sort_values("qlike_raw").iloc[0]
    fs = fc.sort_values("Sharpe_mid", ascending=False).iloc[0]
    out.append(
        f"- Over all of table A the lowest QLIKE is {fq['label']} ({fq['qlike_raw']:.4f}; Sharpe {fq['Sharpe_mid']:.2f}, rank "
        f"{int((fc['Sharpe_mid'] > fq['Sharpe_mid']).sum()) + 1} of {len(fc)} on Sharpe) and the highest Sharpe is {fs['label']} "
        f"({fs['Sharpe_mid']:.2f}; QLIKE {fs['qlike_raw']:.4f}, rank {int((fc['qlike_raw'] < fs['qlike_raw']).sum()) + 1} of {len(fc)} on QLIKE). "
        f"Median over table A: QLIKE {fc['qlike_raw'].median():.4f}, Sharpe mid {fc['Sharpe_mid'].median():.2f}, crossed {fc['Sharpe_crossed'].median():.2f}."
    )
    rest = fc.drop(index=REFERENCE, errors="ignore")
    sig = rest["dm_p_vs_ref"] < SIG_LEVEL
    lo, hi = rest["qlike_pct_vs_ref"] < 0, rest["qlike_pct_vs_ref"] > 0
    out.append(
        f"- Against the reference's QLIKE, {int(lo.sum())} forecasts have a lower point estimate (better), {int((lo & sig).sum())} of them "
        f"with DM p < {SIG_LEVEL:g}; {int(hi.sum())} a higher one, {int((hi & sig).sum())} with DM p < {SIG_LEVEL:g}."
    )
    return out


def rank_reading(A: pd.DataFrame, rk: pd.DataFrame) -> list[str]:
    """What the rank-correlation table says, with the group medians behind it (point estimates)."""
    fc = A[
        (A["family"] != "direct rest-of-day at 15:30 (check)")
        & (A["duplicate_of"] == "")
    ]
    g48 = fc[fc["family"].isin(["paper", "pooled twin"])]
    gpb = fc[~fc["family"].isin(["paper", "pooled twin"])]
    glin = fc[fc["family"].isin(["per-bar linear", "VIX-only family"])]

    def get(sname: str) -> pd.Series | None:
        r = rk[
            (rk["set"] == sname) & (rk["qlike"] == "plain") & (rk["sharpe"] == "mid")
        ]
        return r.iloc[0] if len(r) else None

    def cc(r: pd.Series) -> str:
        return f"{r['spearman']:+.2f}"  # 2026-10-06: interval commented out (user)

    ra = get("all table-A forecasts")
    rp = get("per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies)")
    rl = get("per-bar linear + VIX-only family")
    if ra is None or not len(g48) or not len(gpb):
        return []
    txt = (
        f"Reading: across all of table A the rank correlation of QLIKE with Sharpe is {cc(ra)}. The {len(g48)} "
        f"forecasts fitted on all 48 bars have median QLIKE {g48['qlike_raw'].median():.4f} and median Sharpe {g48['Sharpe_mid'].median():.2f}; "
        f"the {len(gpb)} per-bar forecasts median QLIKE {gpb['qlike_raw'].median():.4f} and median Sharpe {gpb['Sharpe_mid'].median():.2f}"
    )
    if rp is not None:
        both = gpb[
            (gpb["qlike_raw"] >= gpb["qlike_raw"].quantile(0.75))
            & (gpb["Sharpe_mid"] <= gpb["Sharpe_mid"].quantile(0.25))
        ]
        txt += f"; among the per-bar forecasts alone it is {cc(rp)}"
        if len(both):
            txt += (
                f"; {len(both)} per-bar forecasts sit in the worst quarter on both counts ("
                + "; ".join(
                    both.sort_values("qlike_raw", ascending=False)["label"].head(6)
                )
                + ("; ..." if len(both) > 6 else "")
                + ")"
            )
    if rl is not None and len(glin):
        txt += (
            f"; within the per-bar linear and VIX-only family (QLIKE {glin['qlike_raw'].min():.4f} .. {glin['qlike_raw'].max():.4f}, "
            f"Sharpe {glin['Sharpe_mid'].min():.2f} .. {glin['Sharpe_mid'].max():.2f}) it is {cc(rl)}"
        )
    return [txt + "."]


def extreme_lines(tab: pd.DataFrame, ext: pd.DataFrame) -> list[str]:
    A = tab[tab["key"] != ALWAYS_SHORT]
    out = [
        f"- **Rule (named):** PLAIN_FLOOR(d) = the smallest 16:00 target of the previous {FLOOR_W} sessions (FLOOR_W, the parked "
        "recalibration's window), lagged one session. A plain forecast f²·B below it forecasts a quieter 15:30–16:00 half hour than "
        "any of the past year; it is flagged, listed in `master_table_extremes.csv`, and the QLIKE is reported as is (`qlike_raw`, the "
        "reported number) and, as a diagnostic, with flagged forecasts raised to the floor (`qlike_raw_floored`). Nothing is clipped."
    ]
    on_trade = int(ext["trade_day"].sum()) if len(ext) else 0
    out.append(
        f"- **On the trade days:** {on_trade} flagged forecasts over all forecasts, so `qlike_raw_floored` = `qlike_raw` for "
        f"{int((A['n_extreme_days'] == 0).sum())} of {len(A)} rows."
    )
    if len(ext):
        ne = ext.groupby("key").size().sort_values(ascending=False)
        ec = int(ext["early_close"].sum())
        days_ext = ext.loc[ext["early_close"], "stamp"].str[:10].unique()
        out.append(
            f"- **On every 16:00 session of the arm span** ({int(A['sessions'].max())} sessions, the arm span 2018-06-25 .. 2024-04-30 after the "
            f"floor's {FLOOR_MIN}-session warm-up, early closes included): {len(ext)} flagged rows over {len(ne)} forecasts, {ec} of them on "
            f"{len(days_ext)} early-close sessions ({', '.join(sorted(days_ext))}): 13:00 closes, where the 15:30–16:00 bar lies after the cash "
            "close and the trade frame never enters. Negative adjusted-scale forecasts occur there too (the plain back-transform squares them)."
        )
        chk = set(A.loc[A["family"] == "direct rest-of-day at 15:30 (check)", "key"])
        worst = (
            ext[~ext["key"].isin(chk)]
            .sort_values("qlike_raw_this_day", ascending=False)
            .head(3)
        )
        for _, r in worst.iterrows():
            t = A.set_index("key").loc[r["key"]]
            out.append(
                f"  - {t['label']} on {r['stamp'][:10]}{' (early close)' if r['early_close'] else ''}: f = {r['yhat']:.4f} on the adjusted scale, "
                f"plain forecast {r['plain_forecast']:.3g} vs floor {r['plain_floor']:.3g} and target {r['target_true_raw']:.3g}; that one day's "
                f"QLIKE is {r['qlike_raw_this_day']:.1f}. Over all {int(t['sessions'])} sessions the QLIKE is {t['qlike_raw_all_sessions']:.4f} as is, "
                f"{t['qlike_raw_floored_all_sessions']:.4f} floored, {t['qlike_raw_all_sessions_no_early_close']:.4f} without early closes; on the trade "
                f"days {t['qlike_raw']:.4f}."
            )
    rod = LS.parent / "linear_subsection_restofday" / "score_rest_of_day.csv"
    if rod.is_file():
        r = pd.read_csv(rod, dtype={"clock": str})
        r = r[
            (r["bucket"] == "all_features")
            & (r["estimator"] == "ridge")
            & (r["clock"].str.zfill(4) == "1530")
        ]
        if len(r):
            out.append(
                f"- The handoff's '16:00 ridge QLIKE 2.65 unrecalibrated' is this same early-close day: `score_rest_of_day.csv` reports the plain "
                f"QLIKE {float(r['qlike_plain'].iloc[0]):.4f} for the all_features ridge at the 15:30 clock on {int(r['n'].iloc[0])} sessions "
                "against the unclipped variance; the trade frame excludes early closes, so it does not enter any number of this table."
            )
    return out


def scorer_change_lines(rvp: pd.DataFrame, A: pd.DataFrame) -> list[str]:
    """What removing the recalibration changed (recal_vs_plain.csv), table A without check rows and duplicates."""
    fc = A[
        (A["family"] != "direct rest-of-day at 15:30 (check)")
        & (A["duplicate_of"] == "")
    ]
    v = rvp.set_index("key")
    v = v[v.index.isin(fc.index) & v["Sharpe_mid_recal"].notna()]
    if not len(v):
        return [
            "(no forecast has the diagnostic recalibrated column on every scored day)"
        ]
    n = int(v["n_days"].max())

    def rng(c: str, fmt: str) -> str:
        return f"median {format(v[c].median(), fmt)} ({format(v[c].min(), fmt)} .. {format(v[c].max(), fmt)})"

    L = [
        f"The recalibration was removed on {RECAL_REMOVED} (user decision): every number in this file uses the plain forecast f²·B. "
        "`recal_vs_plain.csv` records, per forecast on its own scored days, the parked recalibrated forecast (`pred_clock` = (f² + s)·B, "
        f"diagnostic) against the plain one. Over the {len(v)} table-A forecasts (check rows and exact duplicates left out, {n} days):",
        f"- QLIKE, plain minus recalibrated: {rng('qlike_pct_change', '+.1f')} %; Sharpe mid {rng('d_Sharpe_mid_plain_minus_recal', '+.2f')}, "
        f"crossed {rng('d_Sharpe_crossed_plain_minus_recal', '+.2f')}; buy share {rng('d_pct_buy_plain_minus_recal', '+.1f')} percentage points.",
        f"- Positions changed: {rng('positions_changed', '.0f')} of {n} days; buy -> sell {int(v['buy_to_sell'].sum())}, sell -> buy "
        f"{int(v['sell_to_buy'].sum())} over all of them (s·B ≥ 0, so removing it can only lower a forecast: mean s·B / f²·B "
        f"{rng('mean_s_over_plain', '.2f')}).",
    ]
    for k, name in ((HEADLINE, "Headline"), (REFERENCE, "Reference")):
        if k in v.index:
            r = v.loc[k]
            L.append(
                f"- **{name}** (`{k}`): QLIKE {r['qlike_recal']:.4f} -> {r['qlike_plain']:.4f}; Sharpe mid {r['Sharpe_mid_recal']:.2f} -> "
                f"{r['Sharpe_mid_plain']:.2f}, crossed {r['Sharpe_crossed_recal']:.2f} -> {r['Sharpe_crossed_plain']:.2f}; buys "
                f"{r['pct_buy_recal']:.1f} % -> {r['pct_buy_plain']:.1f} %; positions changed on {int(r['positions_changed'])} days."
            )
    ch = v.sort_values("d_Sharpe_mid_plain_minus_recal")
    L.append(
        "- Largest Sharpe (mid) falls: "
        + "; ".join(
            f"{r['label']} {r['Sharpe_mid_recal']:.2f} -> {r['Sharpe_mid_plain']:.2f}"
            for _, r in ch.head(3).iterrows()
        )
        + ". Largest rises: "
        + "; ".join(
            f"{r['label']} {r['Sharpe_mid_recal']:.2f} -> {r['Sharpe_mid_plain']:.2f}"
            for _, r in ch.tail(3).iloc[::-1].iterrows()
        )
        + "."
    )
    rq = v["qlike_recal"].rank()
    rs = v["Sharpe_mid_recal"].rank()
    L.append(
        f"- Rank agreement across these forecasts, recalibrated vs plain: Spearman {float(rq.corr(v['qlike_plain'].rank())):+.2f} on QLIKE, "
        f"{float(rs.corr(v['Sharpe_mid_plain'].rank())):+.2f} on Sharpe mid."
    )
    return L


# --------------------------------------------------------------------------- provenance + before / after
LINEAR_FAMILIES = (
    "per-bar linear",
    "VIX-only family",
    "implied-vol representations",
    "HAR-ladder variants",
    "chain-period buckets",
)
UNTOUCHED = ("paper", "pooled twin")  # the families the 16:00 campaign did not re-run
# 2026-10-07 (user decision, writeup/MASTER_TABLE_REVIEW_2026-10-01.md item 76): the paper's pooled XGBoost (tag
# xgb) reads bank arm 10 (xgb_opt_00, the TOP entry of the 1998-2002 dev-span search, the rule LightGBM and the
# random forest follow) instead of bank arm 16 (xgb_opt_06, 7th-ranked, hand-picked after post-2002 scoring).  Its
# row differs from the pre-campaign table by design: the 'untouched' gate is REPORTED for it (not asserted), and an
# ASSERTED gate checks that the table read is the verified arm-10 dump (sha256 recorded in
# results/atm_straddle_0dte_1530/TREE10_DONE).  Other pooled tree-bank dumps on disk (yhat_tree<NN>.parquet not
# named by atm_straddle_lib.yhat_paths, e.g. the parked arm-16 table) are looked at and not scored.
PAPER_SWITCHED = {
    "xgb": {
        "table": "yhat_tree10.parquet",
        "sha256": "ebd540db48768c80e9426116449db6399fcb5d090f40b3140cfeb85ac65a2cde",
        "changed": "nothing (the campaign did not re-run it; 2026-10-07: pooled XGBoost switched from bank arm 16 "
        "(xgb_opt_06, yhat_tree16) to bank arm 10 (xgb_opt_00, yhat_tree10), user decision, MASTER_TABLE_REVIEW item 76; "
        "before side = the switched table)",
    }
}


def _arm_clusters() -> dict[str, tuple[str, str]]:
    """key -> (cluster before, cluster after) of agent A's re-run (the de-dup root's arm list)."""
    f = ROOT / "results" / DEDUP_ARM_ROOT / "arm_list.csv"
    if ARM_ROOT == DEFAULT_ARM_ROOT or not f.is_file():
        return {}
    a = pd.read_csv(f)
    return {
        r["key"]: (str(r["cluster_old"]), str(r["cluster_new"]))
        for _, r in a.iterrows()
    }


def what_changed(key: str, fam: str, source: str, clusters: dict) -> str:
    if fam == "paper" and key in PAPER_SWITCHED:  # 2026-10-07: see PAPER_SWITCHED
        return PAPER_SWITCHED[key]["changed"]
    if fam in LINEAR_FAMILIES and ARM_ROOT == DEFAULT_ARM_ROOT and source == "arm":
        return "nothing (arm root = the pre-campaign arms)"
    w = FAMILY_PROV.get(fam, {}).get("changed", "unknown family")
    if fam in LINEAR_FAMILIES and key in clusters:
        old, new = clusters[key]
        w += f"; re-run on {new}" + (f" (was {old})" if old != new else " (as before)")
    if (
        perbar_switch_active() and perbar_switch_path(key) is not None
    ):  # 2026-10-13: PERBAR_SWITCH
        w += (
            "; then switched to the top-entry table "
            + perbar_switch_path(key).parent.name  # type: ignore[union-attr]  # not None: checked above
        )
    return w


def provenance_table(tab: pd.DataFrame) -> pd.DataFrame:
    """One row per family of this run: counts in table A / B and what the family is (FAMILY_PROV)."""
    rows = []
    t = tab[tab["key"] != ALWAYS_SHORT]
    for fam in FAMILY_ORDER:
        g = t[t["family"] == fam]
        if not len(g):
            continue
        p = FAMILY_PROV.get(fam, {})
        rows.append(
            {
                "family": fam,
                "n_forecasts": len(g),
                "n_table_A": int((g["table"] == "A").sum()),
                "n_table_B": int((g["table"] == "B").sum()),
                "arm_root": f"results/{ARM_ROOT}"
                if (g["source"] == "arm").any()
                else "",
                "design": p.get("design", ""),
                "window_mask": p.get("mask", ""),
                "refit": p.get("refit", ""),
                "tuning": p.get("tuning", ""),
                "cpu_class": p.get("cpu", ""),
                "vs_pre_campaign": p.get("changed", ""),
            }
        )
    return pd.DataFrame(rows)


def before_after(
    tab: pd.DataFrame, daily: pd.DataFrame, before_dir: Path, deck: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """Every forecast key before (the pre-campaign master table) and after (this run), both under the plain scorer.

    2026-10-06: recalibration removed, user decision -- the before side is re-scored from the pre-campaign
    daily frame's ``plain`` column and the deck (its stored q / returns / qlike_recal are the recalibrated
    trade and are not used); its plain QLIKE is gated against the pre-campaign table's qlike_raw.
    """
    bt_p, bd_p = (
        before_dir / "master_table.csv",
        before_dir / "master_table_daily.parquet",
    )
    if not (bt_p.is_file() and bd_p.is_file()):
        print(
            f"before/after skipped: {before_dir} has no master_table.csv + master_table_daily.parquet"
        )
        return pd.DataFrame(), []
    was = pd.read_csv(bt_p, float_precision="round_trip").set_index("key")
    now = tab.set_index("key")
    bd = pd.read_parquet(bd_p)
    db: dict[str, pd.DataFrame] = {}
    for k, g in bd.groupby("key"):
        g = g.set_index("day").sort_index()
        t = trade_days(g["plain"], deck.loc[g.index]).reindex(g.index)
        db[k] = pd.DataFrame(
            {
                "plain": g["plain"],
                "qlike_raw": slc.qlike(g["target"], g["plain"]),
                "q": t["q"],
                "ret_mid": t["mid"],
                "ret_crossed": t["crossed"],
            },
            index=g.index,
        )
    dn = {k: g.set_index("day").sort_index() for k, g in daily.groupby("key")}
    # 2026-10-07: a paper row switched by decision (PAPER_SWITCHED) is not a change the 16:00 campaign made (the
    # campaign did not re-run the paper's forecasts).  Its campaign before side is the switched table itself,
    # re-scored exactly as every before side is; the pre-campaign table's own row (the replaced forecast) is
    # compared with it in a REPORTED gate below, for the record.
    for k in PAPER_SWITCHED:
        if k in dn and k in db:
            g = dn[k]
            t = trade_days(g["plain"], deck.loc[g.index]).reindex(g.index)
            db[k] = pd.DataFrame(
                {
                    "plain": g["plain"],
                    "qlike_raw": slc.qlike(g["target"], g["plain"]),
                    "q": t["q"],
                    "ret_mid": t["mid"],
                    "ret_crossed": t["crossed"],
                },
                index=g.index,
            )
    clusters = _arm_clusters()
    dedup = ARM_ROOT != DEFAULT_ARM_ROOT
    gates: list[dict] = []
    rows = []
    keys = list(now.index) + [k for k in was.index if k not in now.index]
    for k in keys:
        status = (
            "both"
            if (k in now.index and k in was.index)
            else ("new" if k in now.index else "dropped")
        )
        r_ = now.loc[k] if k in now.index else was.loc[k]
        fam = str(r_["family"])
        src = str(r_["source"]) if "source" in r_ else ""
        row: dict = {
            "key": k,
            "status": status,
            "family": fam,
            "label": r_["label"],
            "what_changed": (
                "new"
                if status == "new"
                else (
                    "dropped (not on disk in this run)"
                    if status == "dropped"
                    else (
                        "nothing (no forecast)"
                        if k == ALWAYS_SHORT
                        else what_changed(k, fam, src, clusters)
                    )
                )
            ),
            "table_after": now.loc[k, "table"] if k in now.index else "",
            # 2026-10-13: the after side reads the top-entry table (PERBAR_SWITCH): not a campaign change alone
            "perbar_switch": str(now.loc[k, "perbar_switch"])
            if k in now.index
            and "perbar_switch" in now.columns
            and isinstance(now.loc[k, "perbar_switch"], str)
            else "",
            "n_days_after": int(now.loc[k, "n_days"]) if k in now.index else np.nan,
            "qlike_raw_after": now.loc[k, "qlike_raw"] if k in now.index else np.nan,
            "Sharpe_mid_after": now.loc[k, "Sharpe_mid"] if k in now.index else np.nan,
            "Sharpe_crossed_after": now.loc[k, "Sharpe_crossed"]
            if k in now.index
            else np.nan,
        }
        if status != "new":
            b_ = was.loc[k]
            row.update(
                {
                    "label_before": b_["label"],
                    "family_before": b_["family"],
                    "table_before": b_["table"],
                    "n_days_before": int(b_["n_days"]),
                }
            )
            if k in db:
                bb_all = db[k]
                row.update(
                    {
                        "qlike_raw_before": float(bb_all["qlike_raw"].mean()),
                        "Sharpe_mid_before": sharpe(bb_all["ret_mid"].to_numpy()),
                        "Sharpe_crossed_before": sharpe(
                            bb_all["ret_crossed"].to_numpy()
                        ),
                    }
                )
                if (
                    k in PAPER_SWITCHED
                ):  # 2026-10-07: the replaced forecast's stored row, for the record
                    dv = abs(row["qlike_raw_before"] - float(b_["qlike_raw"]))
                    gates.append(
                        {
                            "gate": "before/after: switched paper row -- the switched table's plain QLIKE vs the replaced "
                            "forecast's pre-campaign qlike_raw (differs by design; see PAPER_SWITCHED)",
                            "forecast": k,
                            "n": len(bb_all),
                            "value": dv,
                            "bound": 1e-12,
                            "ok": dv <= 1e-12,
                            "asserted": False,
                        }
                    )
                elif len(bb_all) == int(b_["n_days"]):
                    dv = abs(row["qlike_raw_before"] - float(b_["qlike_raw"]))
                    gates.append(
                        {
                            "gate": "before/after: the pre-campaign daily frame's plain QLIKE reproduces the pre-campaign table's qlike_raw",
                            "forecast": k,
                            "n": len(bb_all),
                            "value": dv,
                            "bound": 1e-12,
                            "ok": dv <= 1e-12,
                        }
                    )
        if status == "both" and "qlike_raw_before" in row:
            row["qlike_diff"] = row["qlike_raw_after"] - row["qlike_raw_before"]
            row["qlike_pct_change"] = (
                100.0 * row["qlike_diff"] / row["qlike_raw_before"]
            )
            row["dSharpe_mid"] = row["Sharpe_mid_after"] - row["Sharpe_mid_before"]
            row["dSharpe_crossed"] = (
                row["Sharpe_crossed_after"] - row["Sharpe_crossed_before"]
            )
            if k in dn:
                days = dn[k].index.intersection(db[k].index)
                aa, bb = dn[k].loc[days], db[k].loc[days]
                row["n_days_paired"] = len(days)
                row["positions_changed"] = int(
                    (aa["q"].to_numpy() != bb["q"].to_numpy()).sum()
                )
                row["max_rel_plain_change"] = float(
                    (aa["plain"] / bb["plain"] - 1.0).abs().max()
                )
                # 2026-10-06: intervals commented out (user) -- point estimates and the HAC t only:
                # lo, hi = base.day_block_ci(pd.Series(qd, index=days)); paired_sharpe(...) lo / hi
                row["t_hac_mid_after_vs_before"] = asl.newey_west_t(
                    aa["ret_mid"].to_numpy() - bb["ret_mid"].to_numpy()
                )[0]
                for fill in ("mid", "crossed"):
                    row[f"dSharpe_{fill}_paired"] = sharpe(
                        aa[f"ret_{fill}"].to_numpy()
                    ) - sharpe(bb[f"ret_{fill}"].to_numpy())
                if fam in UNTOUCHED:
                    dv = max(
                        row["max_rel_plain_change"],
                        abs(row["dSharpe_mid"]),
                        abs(row["dSharpe_crossed"]),
                        abs(row["qlike_diff"]),
                    )
                    gates.append(
                        {
                            "gate": "untouched by the campaign: equals the pre-campaign table under the plain scorer (plain rel, Sharpe, QLIKE)",
                            "forecast": k,
                            "n": len(days),
                            "value": dv,
                            "bound": 0.0,
                            "ok": dv == 0.0,
                            # "asserted": dedup,
                            # 2026-10-07: reported, not asserted, for a paper row switched by decision
                            "asserted": dedup and k not in PAPER_SWITCHED,
                            "stored_design": "pre-campaign master table (plain)",
                        }
                    )
        rows.append(row)
    cols = [
        "key",
        "status",
        "family",
        "label",
        "what_changed",
        "table_before",
        "table_after",
        "n_days_before",
        "n_days_after",
        "n_days_paired",
        "qlike_raw_before",
        "qlike_raw_after",
        "qlike_diff",
        "qlike_pct_change",
        "Sharpe_mid_before",
        "Sharpe_mid_after",
        "dSharpe_mid",
        "Sharpe_crossed_before",
        "Sharpe_crossed_after",
        "dSharpe_crossed",
        "t_hac_mid_after_vs_before",
        "positions_changed",
        "max_rel_plain_change",
        "dSharpe_mid_paired",
        "dSharpe_crossed_paired",
        "family_before",
        "label_before",
        "perbar_switch",  # 2026-10-13: PERBAR_SWITCH token of the after side ('' = not switched)
    ]
    out = pd.DataFrame(rows).reindex(columns=cols)
    return out, gates


def provenance_lines(prov: pd.DataFrame, gtab: pd.DataFrame) -> list[str]:
    tg = gtab[gtab["gate"].str.startswith("TARGET_ARM of")]
    tline = (
        f"its 16:00 target equals the default root's bit for bit ({int(tg['n'].iloc[0])} rows, "
        f"{int(tg['value'].iloc[0])} differ; gate {'passed' if bool(tg['ok'].iloc[0]) else 'FAILED'})"
        if len(tg)
        else "the default root (the target is its own)"
    )
    L = [
        f"- **Arm root:** `results/{ARM_ROOT}/` ({design_of_root()}); it supplies the arm-only rows, the per-bar OLS incumbent, "
        f"the arm files the per-bar linear and VIX-only tables are gated against, and the common 16:00 target; {tline}.",
        "- **Forecast tables:** every `results/spxw_pnl/yhat_*.parquet` on disk at run time (glob). After the 16:00 campaign "
        "(`writeup/CAMPAIGN_16H_2026-09-29.md`) the per-bar tables are the de-duplicated design (the 12 HAR x open/close "
        "session-edge columns dropped from the per-bar design; the pooled 48-bar models keep them); every tree and LSTM fit uses "
        "the per-window mask (`src/models/window_mask.py`: drop columns constant on the fit's training window and exact copies "
        "of an earlier kept column) and ran pinned to one CPU class (CARC epyc-7513), because tree and LSTM numbers depend on "
        "the CPU class (agent D's finding).",
        "",
        "| family | forecasts (A / B) | design | window mask | refit | tuning | CPU class | vs the pre-campaign table |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for _, r in prov.iterrows():
        L.append(
            f"| {r['family']} | {int(r['n_table_A'])} / {int(r['n_table_B'])} | {r['design']} | {r['window_mask']} | "
            f"{r['refit']} | {r['tuning']} | {r['cpu_class']} | {r['vs_pre_campaign']} |"
        )
    return L


def before_after_lines(ba: pd.DataFrame, before_dir: Path) -> list[str]:
    if not len(ba):
        return [f"(no before/after: `{before_dir}` holds no pre-campaign master table)"]
    try:
        bdir = str(before_dir.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        bdir = str(before_dir)
    b = ba[ba["status"] == "both"].set_index("key")
    L = [
        f"Before = the pre-campaign master table (commit 6177b74; snapshot `{bdir}/`), re-scored here with the plain forecast from its "
        "daily frame's `plain` column; after = this run; every key present in both is paired on the same days (`before_after.csv`; "
        "difference = after minus before; point estimates, HAC t of the daily mid-return difference)."
    ]

    def one(k: str, name: str) -> str:
        if k not in b.index:
            return f"- **{name}** (`{k}`): not in both tables."
        r = b.loc[k]
        return (
            f"- **{name}** (`{k}`; {r['what_changed']}): QLIKE {r['qlike_raw_before']:.4f} -> {r['qlike_raw_after']:.4f} "
            f"({r['qlike_pct_change']:+.2f} %); Sharpe mid {r['Sharpe_mid_before']:.2f} -> {r['Sharpe_mid_after']:.2f} "
            f"({r['dSharpe_mid']:+.2f}, HAC {fmt_t(r['t_hac_mid_after_vs_before'])}), crossed {r['Sharpe_crossed_before']:.2f} -> "
            f"{r['Sharpe_crossed_after']:.2f} ({r['dSharpe_crossed']:+.2f}); positions changed on {int(r['positions_changed'])} of "
            f"{int(r['n_days_paired'])} days; largest relative change of the plain forecast {r['max_rel_plain_change']:.2e}."
        )

    L.append(one(HEADLINE, "Headline"))
    L.append(one(REFERENCE, "Reference"))
    L.append("")
    L.append(
        "| family | keys in both | what changed | QLIKE % change, median (min .. max) | ΔSharpe mid, median (min .. max) | "
        "ΔSharpe crossed, median (min .. max) | HAC t significant, up / down | positions changed, median (max) |"
    )
    L.append("|---|---:|---|---|---|---|---|---|")
    fams = [f for f in FAMILY_ORDER if f in set(b["family"])]
    for fam in fams:
        g = b[b["family"] == fam]
        wc = g["what_changed"].str.split(";").str[0].value_counts()
        pos = g["positions_changed"].dropna()
        sig = g["t_hac_mid_after_vs_before"].map(hac_p) < SIG_LEVEL
        L.append(
            f"| {fam} | {len(g)} | {'; '.join(wc.index)} | {g['qlike_pct_change'].median():+.2f} "
            f"({g['qlike_pct_change'].min():+.2f} .. {g['qlike_pct_change'].max():+.2f}) | {g['dSharpe_mid'].median():+.2f} "
            f"({g['dSharpe_mid'].min():+.2f} .. {g['dSharpe_mid'].max():+.2f}) | {g['dSharpe_crossed'].median():+.2f} "
            f"({g['dSharpe_crossed'].min():+.2f} .. {g['dSharpe_crossed'].max():+.2f}) | "
            f"{int((sig & (g['dSharpe_mid'] > 0)).sum())} / {int((sig & (g['dSharpe_mid'] < 0)).sum())} | "
            + (f"{pos.median():.0f} ({pos.max():.0f})" if len(pos) else "")
            + " |"
        )
    new = ba[ba["status"] == "new"]
    if len(new):
        L.append("")
        L.append(
            f"New in this run ({len(new)} keys, `status = new` in `before_after.csv`): "
            + "; ".join(
                f"{fam} {len(g)}" for fam, g in new.groupby("family", sort=False)
            )
            + "."
        )
    gone = ba[ba["status"] == "dropped"]
    if len(gone):
        L.append(
            f"Dropped (in the pre-campaign table, not in this run): {', '.join(gone['key'])}."
        )
    moved = b[b["t_hac_mid_after_vs_before"].map(hac_p) < SIG_LEVEL]
    if len(moved):
        L.append("")
        L.append(
            f"Keys whose own mid-fill return moved with the HAC t significant at {100 * SIG_LEVEL:.0f} % (Sharpe before -> after): "
            + "; ".join(
                f"{r['label']} {r['Sharpe_mid_before']:.2f} -> {r['Sharpe_mid_after']:.2f} ({fmt_t(r['t_hac_mid_after_vs_before'])})"
                for _, r in moved.sort_values("dSharpe_mid").iterrows()
            )
            + "."
        )
    return L


def beats_headline_lines(A: pd.DataFrame, vs_h: pd.DataFrame) -> list[str]:
    """Recorded, never acted on: any forecast that now beats the headline (point estimates; DM / HAC t significance)."""
    if not len(vs_h) or HEADLINE not in A.index:
        return []
    fc = A[
        (A["family"] != "direct rest-of-day at 15:30 (check)")
        & (A["duplicate_of"] == "")
    ].index
    v = vs_h.set_index("key")
    v = v[v.index.isin(fc) & (v.index != HEADLINE)]
    sig_t = v["t_hac_mid_vs_headline"].map(hac_p) < SIG_LEVEL

    def names(m: pd.Series, col: str) -> str:
        g = v[m].sort_values(col, ascending=False)
        return (
            "; ".join(
                f"{r['label']} ({r[col]:+.2f}, HAC {fmt_t(r['t_hac_mid_vs_headline'])})"
                for _, r in g.iterrows()
            )
            or "none"
        )

    bm = v["dSharpe_mid_vs_headline"] > 0
    bx = v["dSharpe_crossed_vs_headline"] > 0
    bq = v["qlike_pct_vs_headline"] < 0
    bqs = bq & (v["dm_p_vs_headline"] < SIG_LEVEL)
    L = [
        f"- **Does any forecast now beat the headline?** Of the {len(v)} other table-A forecasts (check rows and exact duplicates "
        f"left out), {int(bm.sum())} have a higher Sharpe point estimate at the mid fill ({int(bx.sum())} crossed); with the HAC t of the "
        f"daily mid-return difference significant at {100 * SIG_LEVEL:.0f} %: {int((bm & sig_t).sum())} ({names(bm & sig_t, 'dSharpe_mid_vs_headline')}). "
        f"{int(bq.sum())} have a lower QLIKE point estimate; with DM p < {SIG_LEVEL:g}: {int(bqs.sum())}"
        + (
            " ("
            + "; ".join(
                f"{r['label']} {r['qlike_pct_vs_headline']:+.1f} %"
                for _, r in v[bqs].sort_values("qlike_pct_vs_headline").iterrows()
            )
            + ")"
            if bqs.any()
            else ""
        )
        + "."
    ]
    if bm.any():
        top = v[bm].sort_values("dSharpe_mid_vs_headline", ascending=False).head(5)
        L.append(
            "  Highest Sharpe differences over the headline (mid, point estimates): "
            + "; ".join(
                f"{r['label']} {r['dSharpe_mid_vs_headline']:+.2f} ({fmt_t(r['t_hac_mid_vs_headline'])})"
                for _, r in top.iterrows()
            )
            + "."
        )
    if (bm & sig_t).any() or bqs.any():
        L.append(
            f"  RECORDED, not acted on: the headline constant stays `{HEADLINE}` (it is chosen on feasibility and as the per-bar "
            "model of record, not on the maximum of many Sharpe ratios); the forecasts above are the candidates to look at."
        )
    return L


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
