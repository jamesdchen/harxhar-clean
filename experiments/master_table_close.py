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

THE SCORER (the RESEARCH convention; the notebook's 13-bar map is NOT used, so no
number here may be set beside a notebook number).  Exactly what
experiments/compare_mfiv_harlag.py and experiments/score_trees_subsection.py do,
with their own functions imported, never re-implemented:

  * target     the per-bar spec's 16:00 target: true_adj = sqrt(RV/B), winsorized
               by the spec (rolling 5/95), and true_raw = true_adj^2 B.  Every
               2000-session per-bar arm carries the IDENTICAL 16:00 target (gated
               below), so one frame -- TARGET_ARM -- serves every forecast; a table
               that is not a per-bar arm (the paper's pooled forecasts) is scored
               against the same target on the same stamps.
  * recalibration on the 16:00 bar ALONE:
               pred_clock = (f^2 + s) B,  s = the mean of the forecast's own squared
               adjusted-scale errors (true_adj - f)^2 at the 16:00 label over the
               previous SMEAR_W = 250 sessions (at least SMEAR_MIN = 63), lagged one
               session -- score_linear_subsection_causal.causal_forecasts.  No
               level/slope map: a forecast fitted on all 48 bars keeps whatever
               16:00 level bias it has (the notebook's Mincer-Zarnowitz map would
               remove it), so the paper's pooled forecasts read lower here than in
               the deck.
  * raw        the plain back-transform f^2 B (no second-moment term, no map).
  * trade      the deck's 15:30 sign(s) on the straddle (nearest out-of-the-money
               call + nearest out-of-the-money put, same-day expiry, one position):
               q = +1 (buy) if pred_clock > the 15:30 implied variance, else -1;
               mid return q R; crossed return exit/ask - 1 (buy) or 1 - exit/bid
               (sell) -- score_linear_subsection.trade_1530 on
               results/atm_straddle_0dte_1530/daily_blk2.parquet (866 days,
               2020-01-03 .. 2024-04-30).  Per-day returns are needed for the
               paired intervals, so trade_days() restates trade_1530's lines and a
               gate checks its aggregates against trade_1530 itself for every row.

THE SAME DAYS.  Table A holds every forecast that covers all 866 trade days
(the intersection is printed and must equal 866); a forecast that drops days, or
that is only valid from a later comparison-window start, goes to table B, scored
against the reference and always short on ITS OWN days (still paired).

REFERENCE (fixed): the block-diagonal ridge on the panel of record
(yhat_blk2_fomc1.parquet, tag blk2) -- the paper's headline forecast: HAR ladder
(penalty 1) plus the exogenous block (penalty 100), the forecast the paper's 15:30
deck trades (daily_blk2.parquet).  Every row answers "does this forecast improve on
the paper's?".  Second reference: always short (q = -1 every day, no forecast).

COLUMNS per forecast: 16:00 QLIKE raw and recalibrated (against the spec's target;
the recalibrated one also against the unclipped realized variance of the production
table); % difference vs the reference with the day-block 95 % interval and the
Diebold-Mariano HAC statistic (src.evaluation.diebold_mariano.dm_test, the research
scorers' test); sign(s) Sharpe mid and crossed, mean return, hit rate (share of
days the position earns > 0), buy days; paired block-bootstrap 95 % intervals of
the Sharpe difference vs the reference and vs always short (circular blocks of
BOOT_BLOCK = 21 days, BOOT_B = 2000 draws, seed [BOOT_SEED, n] -- the constants of
score_linear_subsection, whose day_block_ci draws the same days); calibration
(mean realized / mean forecast).

EXTREME BACK-TRANSFORM VALUES (named, explicit; no silent clip).  PLAIN_FLOOR(d) =
the smallest 16:00 target of the previous SMEAR_W sessions, lagged one session (the
recalibration's own window, no new constant).  A plain forecast below it predicts a
quieter 15:30-16:00 half hour than any in the past year: it is flagged, listed in
master_table_extremes.csv, and the raw QLIKE is reported both as is and with the
flagged forecasts raised to the floor ("floored").  Counted on the trade days and on
every 16:00 session of the arm span (2018-06-25 .. 2024-04-30, early closes
included).  The recalibrated forecast needs no treatment (it is >= s B > 0).

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
arm span; every arm file's pred_clock equals the research reader's
(compare_mfiv_harlag._prep) on the trade days; every row's trade aggregates equal
trade_1530's; a table and the arm file of the same forecast carry the same pred_adj (the
per-bar linear tables against <arm root>/arms_hoffman2); no forecast falls into "other
table"; always short equals the paper table's always-short row.  STORED RESEARCH NUMBERS
are design-aware: the numbers written on the first-pass design
(results/linear_subsection_trees/rescore_local/trees_trade_1530.csv, the compare_* CSVs of
the bucket studies) are asserted with the default arm root and REPORTED (asserted = False,
stored_design "pre-campaign design") with the de-dup root; the campaign's own stored
numbers (STORED_CAMPAIGN: agent A's change_by_arm.csv for every linear forecast, agent H's
mask_levels.csv for the tree rungs T10/T1/RS10/RS1 and the per-bar LSTM, agent C's Optuna
report levels.csv, agent I's lstmi_levels.csv, agent E's check_1530.csv for the rest-of-day
check rows) are asserted with the de-dup root and reported with the default one.  With the
de-dup root the families the campaign did not touch (paper, pooled twins) must also equal
the pre-campaign master table (BEFORE_DIR) bit for bit.

BEFORE / AFTER (when BEFORE_DIR holds the pre-campaign master_table.csv and
master_table_daily.parquet): before_after.csv, one row per forecast key -- QLIKE (recal)
and sign(s) Sharpe mid / crossed before and after, the differences, the paired day-block
interval of each Sharpe difference from the two daily frames (same days, the same draws as
paired_sharpe), the day-block interval of the daily QLIKE difference, positions changed,
and what changed for the family (FAMILY_PROV, from the campaign file); keys new in this
run are listed with status "new".

Parallelism: loading + recalibration and the bootstraps run in a process pool of
MAX_WORKERS = 4 (the brief's local cap).

Outputs (results/close_master_table/): master_table.csv, master_table_days.csv,
master_table_extremes.csv, master_table_gates.csv, master_table_rankcorr.csv,
master_table_vs_headline.csv, before_after.csv, master_table_provenance.csv, SUMMARY.md;
then writeup/make_master_table_close_tex.py renders writeup/generated/table_master_close.tex
and writeup/master_table_close.pdf (--no-pdf skips it).

Run:  python experiments/master_table_close.py [--arm-root linear_subsection_dedup] [--no-pdf] [--workers 4]
"""

from __future__ import annotations

import argparse
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

EST_LONG = {"ridge": "ridge", "lasso": "lasso", "enet": "elastic net", "ols": "OLS"}
EST_SHORT = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
TREE_LONG = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
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
    bk = r"(all_features|baseline|live_feasible)"
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
    m = re.fullmatch(rf"subtree_daily_{bk}_(lgbm|xgb|rf)", stem)
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
    m = re.fullmatch(r"subtree_(lgbm|xgb|rf)_(\w+)", stem)
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
        specs.append(describe_table(p, paper))
    have = {s.key for s in specs}
    # the per-bar linear tables were stacked from <arm root>/arms_hoffman2 (build_subsection_yhat.py):
    # the 16:00 arm file of each is a gate on the table (same pred_adj, same recalibrated forecast)
    est_dir = {v: k for k, v in EST_SHORT.items()}
    for s in specs:
        if s.family == "per-bar linear" and s.meta.get("bucket") in (
            "all_features",
            "baseline",
            "live_feasible",
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
    return specs, skipped


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
    # PLAIN_FLOOR: the smallest 16:00 target of the previous SMEAR_W sessions, lagged one session
    t["plain_floor"] = (
        t["true_raw"].rolling(slc.SMEAR_W, min_periods=slc.SMEAR_MIN).min().shift(1)
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


# --------------------------------------------------------------------------- stage 1: load + recalibrate (worker)
_TGT: pd.DataFrame | None = None
_DECK_DAYS: pd.DatetimeIndex | None = None
_IDX: dict[int, np.ndarray] = {}


def _init(tgt: pd.DataFrame, deck_days: pd.DatetimeIndex) -> None:
    global _TGT, _DECK_DAYS
    _TGT, _DECK_DAYS = tgt, deck_days


def load_one(spec: Spec) -> dict:
    """16:00 forecast -> the research recalibration on the common target.  Returns frame + gates."""
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
    research_clock = None
    try:
        if spec.source == "table":
            has_rv = "rv_raw" in pq.read_schema(spec.path).names
            x = read_table_1600(
                spec.path, ["yhat", "baseline"] + (["rv_raw"] if has_rv else [])
            ).rename(columns={"yhat": "pred_adj"})
        else:
            r = cmh._prep(Path(spec.path))  # the research reader and back-transform
            assert r is not None
            r = r[r["hhmm"] == TRADE_BAR]
            x = pd.DataFrame(
                {
                    "pred_adj": r["pred_adj"],
                    "baseline": r["baseline"],
                    "true_raw_own": r["true_raw"],
                }
            )
            research_clock = r["pred_clock"]
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
    cf = slc.causal_forecasts(r)
    f = pd.DataFrame(
        {
            "pred_adj": cf["pred_adj"],
            "plain": cf["pred_adj"] ** 2 * cf["baseline"],
            "pred_clock": cf["pred_clock"],
            "true_raw": cf["true_raw"],
            "rv_unclipped": j["rv_unclipped"],
            "plain_floor": j["plain_floor"],
            "early_close": j["early_close"],
        },
        index=cf.index,
    )
    if research_clock is not None:
        on = f.index[f.index.normalize().isin(_DECK_DAYS) & f["pred_clock"].notna()]
        rc = research_clock.reindex(on)
        rel = (
            float((f.loc[on, "pred_clock"] / rc - 1.0).abs().max())
            if len(on)
            else np.nan
        )
        ok = bool(len(on) and rc.notna().all() and rel <= 1e-12)
        gates.append(
            {
                "gate": "pred_clock equals the research reader's (trade days)",
                "forecast": spec.key,
                "n": len(on),
                "value": rel,
                "bound": 1e-12,
                "ok": ok,
            }
        )
    arm_twin = spec.meta.get("arm_twin")
    if arm_twin:
        a = cmh._prep(Path(arm_twin))
        assert a is not None
        a = a[a["hhmm"] == TRADE_BAR]
        jj = f[["pred_adj", "pred_clock"]].join(
            a[["pred_adj", "pred_clock"]], how="inner", rsuffix="_arm"
        )
        d_adj = float((jj["pred_adj"] - jj["pred_adj_arm"]).abs().max())
        on = jj.index.normalize().isin(_DECK_DAYS)
        d_clk = float(
            (jj.loc[on, "pred_clock"] / jj.loc[on, "pred_clock_arm"] - 1.0).abs().max()
        )
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
                "gate": "table pred_clock equals the research reader on its arm file",
                "forecast": spec.key,
                "n": int(on.sum()),
                "value": d_clk,
                "bound": 1e-12,
                "ok": d_clk <= 1e-12,
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
    # ---- forecast accuracy (QLIKE against the spec's target; recal also vs the unclipped RV)
    y = fd["true_raw"]
    q_plain = slc.qlike(y, fd["plain"])
    floor = fd["plain_floor"]
    flag = fd["plain"] < floor
    q_plain_fl = slc.qlike(y, fd["plain"].where(~flag, floor))
    q_rec = slc.qlike(y, fd["pred_clock"])
    q_ref = slc.qlike(rd["true_raw"], rd["pred_clock"])
    q_rec_u = slc.qlike(fd["rv_unclipped"], fd["pred_clock"])
    q_ref_u = slc.qlike(rd["rv_unclipped"], rd["pred_clock"])
    out.update(
        {
            "qlike_raw": float(q_plain.mean()),
            "qlike_raw_floored": float(q_plain_fl.mean()),
            "n_extreme_days": int(flag.sum()),
            "qlike_recal": float(q_rec.mean()),
            "qlike_recal_unclipped": float(q_rec_u.mean()),
            "n_recal_below_floor": int((fd["pred_clock"] < floor).sum()),
        }
    )
    d = q_rec - q_ref
    out["qlike_pct_vs_ref"] = float(100.0 * d.mean() / q_ref.mean())
    if is_ref:
        out.update(
            {
                "qlike_diff_ci_lo": np.nan,
                "qlike_diff_ci_hi": np.nan,
                "dm_vs_ref": np.nan,
                "dm_p_vs_ref": np.nan,
                "dm_hac_lag": np.nan,
            }
        )
    else:
        lo, hi = base.day_block_ci(pd.Series(d.to_numpy(), index=fd.index))
        dm = dm_test(q_rec.to_numpy(), q_ref.to_numpy())
        out.update(
            {
                "qlike_diff_ci_lo": lo,
                "qlike_diff_ci_hi": hi,
                "dm_vs_ref": float(dm["dm"]),
                "dm_p_vs_ref": float(dm["p"]),
                "dm_hac_lag": dm["hac_lag"],
            }
        )
    out["qlike_pct_vs_ref_unclipped"] = float(
        100.0 * (q_rec_u.mean() / q_ref_u.mean() - 1.0)
    )
    out["calib_mean_realized_over_forecast"] = float(y.mean() / fd["pred_clock"].mean())
    out["share_realized_above_forecast"] = float((y > fd["pred_clock"]).mean())
    # ---- the last-30-min trade
    t = trade_days(fd["pred_clock"], dk)
    tr = trade_days(rd["pred_clock"], dk)
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
        if is_ref:
            out.update(
                {
                    f"dSharpe_{fill}_vs_ref": 0.0,
                    f"dSharpe_{fill}_vs_ref_lo": np.nan,
                    f"dSharpe_{fill}_vs_ref_hi": np.nan,
                }
            )
        else:
            dh, lo, hi = paired_sharpe(a, b)
            out.update(
                {
                    f"dSharpe_{fill}_vs_ref": dh,
                    f"dSharpe_{fill}_vs_ref_lo": lo,
                    f"dSharpe_{fill}_vs_ref_hi": hi,
                }
            )
        dh, lo, hi = paired_sharpe(a, s)
        out.update(
            {
                f"dSharpe_{fill}_vs_short": dh,
                f"dSharpe_{fill}_vs_short_lo": lo,
                f"dSharpe_{fill}_vs_short_hi": hi,
            }
        )
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
    tr = trade_days(rd["pred_clock"], dk)
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
        dh, lo, hi = paired_sharpe(a, b)
        out.update(
            {
                f"dSharpe_{fill}_vs_ref": dh,
                f"dSharpe_{fill}_vs_ref_lo": lo,
                f"dSharpe_{fill}_vs_ref_hi": hi,
            }
        )
        out.update({f"dSharpe_{fill}_vs_short": 0.0})
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
    """Spearman(QLIKE mean, Sharpe) across forecasts on each resampled day set of the chunk."""
    qmat, rmat, lo, hi = args
    idx = boot_idx(qmat.shape[1])[lo:hi]
    q = qmat[:, idx].mean(axis=2)  # (F, draws)
    r = rmat[:, idx]
    s = r.mean(axis=2) / r.std(axis=2, ddof=1)
    return _spearman_cols(q, s)


# --------------------------------------------------------------------------- reproduction gates
def repro_gates(
    tab: pd.DataFrame, frames: dict[str, pd.DataFrame], deck_days: pd.DatetimeIndex
) -> list[dict]:
    """The stored research numbers, recomputed here, must agree -- design-aware.

    First-pass numbers (trees_trade_1530.csv, compare_*.csv) are asserted with the default arm
    root and reported (asserted = False) with the de-dup root; the campaign's own numbers
    (STORED_CAMPAIGN) the other way round; the paper's always-short row is asserted always.
    """
    g: list[dict] = []
    by = tab.set_index("key")
    dedup = ARM_ROOT != DEFAULT_ARM_ROOT
    PRE, CAMP = "pre-campaign design", "16:00 campaign"

    def chk(
        name: str,
        key: str,
        col: str,
        want: float,
        design: str,
        asserted: bool,
        got: float | None = None,
        n: int | None = None,
    ) -> None:
        if got is None and key not in by.index:
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
            got = float(by.loc[key, col])
            n = int(by.loc[key, "n_days"])
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

    # ---- (1) the first pass's stored numbers
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
                    f"trees_trade_1530.csv {col}",
                    key,
                    col,
                    float(r[col]),
                    PRE,
                    not dedup,
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
                    f"compare_{fname}.csv {scol}",
                    key,
                    col,
                    float(r[scol]),
                    PRE,
                    not dedup,
                )
    # ---- (2) the paper's always-short row (no forecast: the same under every design)
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
    # ---- (3) the 16:00 campaign's own stored numbers
    f = STORED_CAMPAIGN["linear (agent A)"]
    if f.is_file():
        # A: trade_1530 on each arm's recalibrated forecast over every trade day it covers
        st = pd.read_csv(f)
        for _, r in st[~st["forecast"].str.startswith("lstm")].iterrows():
            key = r["forecast"]
            if key not in frames:
                chk(f"{f.name} Sharpe_mid_new", key, "Sharpe_mid", np.nan, CAMP, dedup)
                continue
            pc = frames[key]["pred_clock"]
            pc = pc[pc.index.normalize().isin(deck_days)].dropna()
            t = base.trade_1530(pc)
            for col in ("Sharpe_mid", "Sharpe_crossed"):
                chk(
                    f"{f.name} {col}_new (every trade day the arm covers)",
                    key,
                    col,
                    float(r[f"{col}_new"]),
                    CAMP,
                    dedup,
                    got=float(t[col]),
                    n=int(t["deck_days"]),
                )
    cols_h = {
        "qlike": "qlike_recal",
        "sharpe_mid": "Sharpe_mid",
        "sharpe_crossed": "Sharpe_crossed",
        "pct_buy": "pct_buy",
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
                    f"{f.parent.name}/{f.name} {sc}",
                    key,
                    col,
                    float(r[sc]),
                    CAMP,
                    dedup,
                )
    f = STORED_CAMPAIGN["Optuna trees, single-class (agent C)"]
    if f.is_file():
        st = pd.read_csv(f)
        st = st[st["path"].str.contains("yhat_subtree_optuna", regex=False)]
        for _, r in st.iterrows():
            key = Path(r["path"]).stem.removeprefix("yhat_")
            for col in (
                "qlike_raw",
                "qlike_recal",
                "Sharpe_mid",
                "Sharpe_crossed",
                "pct_buy",
                "dSharpe_mid_vs_ref_lo",
                "dSharpe_mid_vs_ref_hi",
            ):
                chk(
                    f"linear_subsection_trees_optuna_mask/report/{f.name} {col}",
                    key,
                    col,
                    float(r[col]),
                    CAMP,
                    dedup,
                )
    f = STORED_CAMPAIGN["intraday-sequence LSTM (agent I)"]
    if f.is_file():
        st = pd.read_csv(f)
        st = st[st["forecast"].isin(["intraday", "intraday_qsel"])]
        for _, r in st.iterrows():
            key = f"lstm_{r['forecast']}_{r['bucket']}"
            for sc, col in cols_h.items():
                chk(f"{f.name} {sc}", key, col, float(r[sc]), CAMP, dedup)
    f = STORED_CAMPAIGN["rest-of-day check rows (agent E)"]
    if f.is_file():
        st = pd.read_csv(f)
        for _, r in st[st["phase"] == "after"].iterrows():
            key = f"restofday_{r['estimator']}_{r['bucket']}"
            if key not in by.index:
                chk(f"{f.name} max_rel_pred_clock", key, "", np.nan, CAMP, dedup)
                continue
            x = by.loc[key]
            chk(
                f"{f.name} max_rel_pred_clock (vs the per-bar twin)",
                key,
                "",
                float(r["max_rel_pred_clock"]),
                CAMP,
                dedup,
                got=float(x["twin_max_rel_pred_clock"]),
                n=int(x["n_days"]),
            )
            chk(
                f"{f.name} same_position (days)",
                key,
                "",
                float(r["same_position"]),
                CAMP,
                dedup,
                got=float(x["twin_same_position"]) * int(x["n_days"]),
                n=int(x["n_days"]),
            )
    return g


# --------------------------------------------------------------------------- summary
def fmt_ci(v: float, lo: float, hi: float, nd: int = 2) -> str:
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


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
        "`master_table.csv` (and `before_after.csv`) of the same run. Full table: `writeup/master_table_close.pdf`; tex: `writeup/generated/table_master_close.tex`."
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
        "- **Scorer: the research convention** (`compare_mfiv_harlag.py` / `score_trees_subsection.py`, their functions imported): "
        "the 16:00 bar is recalibrated on its own, forecast = (f² + s)·B with s = the forecast's own mean squared adjusted-scale error "
        "at 16:00 over the previous 250 sessions (≥ 63), lagged one session; QLIKE against the per-bar spec's 16:00 target; the trade is "
        "the deck's 15:30 sign(s) on the straddle (`trade_1530`). The notebook's 13-bar Mincer–Zarnowitz map is **not** used: no number "
        "here may be set beside a notebook number."
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
        f"Under this scorer it has QLIKE {ref['qlike_recal']:.4f} and sign(s) Sharpe {ref['Sharpe_mid']:.2f} mid / "
        f"{ref['Sharpe_crossed']:.2f} crossed (the deck's own map gives it 1.34 mid; the difference is the scorer). "
        f"Second reference: always short, Sharpe {short['Sharpe_mid']:.2f} mid / {short['Sharpe_crossed']:.2f} crossed."
    )
    L.append("")
    L.append("## (a) Recommended headline forecast")
    L.append("")
    h = A.loc[HEADLINE] if HEADLINE in A.index else None
    if h is not None:
        L.append(
            f"**{h['label']}** (`{HEADLINE}`), scored by the research convention above. On the {int(h['n_days'])} days: "
            f"QLIKE {h['qlike_recal']:.4f} ({h['qlike_pct_vs_ref']:+.1f} % vs the reference, day-block interval on the "
            f"daily difference [{h['qlike_diff_ci_lo']:+.4f}, {h['qlike_diff_ci_hi']:+.4f}], DM {h['dm_vs_ref']:+.2f}, p {h['dm_p_vs_ref']:.3f}); "
            f"sign(s) Sharpe {h['Sharpe_mid']:.2f} mid / {h['Sharpe_crossed']:.2f} crossed, mean {h['mean_mid']:+.3f} per unit premium "
            f"(crossed {h['mean_crossed']:+.3f}), hit rate {100 * h['hit_rate_mid']:.1f} %, buys on {int(h['n_buy'])} days ({h['pct_buy']:.1f} %). "
            f"Sharpe difference vs the reference: mid {fmt_ci(h['dSharpe_mid_vs_ref'], h['dSharpe_mid_vs_ref_lo'], h['dSharpe_mid_vs_ref_hi'])}, "
            f"crossed {fmt_ci(h['dSharpe_crossed_vs_ref'], h['dSharpe_crossed_vs_ref_lo'], h['dSharpe_crossed_vs_ref_hi'])}; "
            f"vs always short: mid {fmt_ci(h['dSharpe_mid_vs_short'], h['dSharpe_mid_vs_short_lo'], h['dSharpe_mid_vs_short_hi'])}, "
            f"crossed {fmt_ci(h['dSharpe_crossed_vs_short'], h['dSharpe_crossed_vs_short_lo'], h['dSharpe_crossed_vs_short_hi'])}."
        )
        L.append("")
        for line in extra["beats_headline"]:
            L.append(line)
        L.append("")
        for line in extra["headline_reasons"]:
            L.append(line)
    L.append("")
    L.append("## Before / after the 16:00 campaign")
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
        "Rank correlation across table-A forecasts (check rows and exact duplicates excluded), point and 95 % day-block bootstrap interval "
        "(the same resampled days as every other interval; a negative value = lower QLIKE goes with higher Sharpe):"
    )
    L.append("")
    L.append("| set | forecasts | QLIKE measure | Sharpe | Spearman | 95 % interval |")
    L.append("|---|---:|---|---|---:|---|")
    for _, r in rk.iterrows():
        L.append(
            f"| {r['set']} | {int(r['n_forecasts'])} | {r['qlike']} | {r['sharpe']} | {r['spearman']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] |"
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
        "file's recalibrated forecast equals the research reader's on the trade days; every per-bar linear and VIX-only table equals "
        "its arm file; every row's trade aggregates equal `trade_1530`'s; no table falls into 'other table'; the paper's always-short "
        "row is reproduced; the stored research numbers of the design in use are reproduced ("
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
            f"Reported, not asserted ({len(rep)}): stored numbers written on another design, compared for the record — "
            + "; ".join(
                f"{src}: {int(g['ok'].sum())} of {len(g)} agree to {REPRO_TOL:g}"
                for src, g in by_src
            )
            + ". A difference there is the design change (see the before/after), not a failure."
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
        "| forecast | QLIKE raw | QLIKE recal | Δ% vs ref | DM | Sharpe mid | Sharpe crossed | ΔSharpe mid vs ref [95 %] | ΔSharpe mid vs short [95 %] | buy days |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---|---|---:|")
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
        L.append(
            f"| {r['label']} | {r['qlike_raw']:.4f} | {r['qlike_recal']:.4f} | {r['qlike_pct_vs_ref']:+.1f} | {dm} | {r['Sharpe_mid']:.2f} | "
            f"{r['Sharpe_crossed']:.2f} | {fmt_ci(r['dSharpe_mid_vs_ref'], r['dSharpe_mid_vs_ref_lo'], r['dSharpe_mid_vs_ref_hi'])} | "
            f"{fmt_ci(r['dSharpe_mid_vs_short'], r['dSharpe_mid_vs_short_lo'], r['dSharpe_mid_vs_short_hi'])} | {int(r['n_buy'])} |"
        )
    L.append("")
    L.append(
        "Wording: sign(s) = buy the straddle when the recalibrated 16:00 forecast exceeds the 15:30 implied variance, sell otherwise; "
        "straddle = nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position; the last-30-min trade "
        "enters at 15:30 and settles at the close. Buckets: `baseline` = HAR + calendar (`har_ma_*`, calendar dummies), `all_features` = "
        "the full design, `live_feasible` = the 16 series a 15:30 forecaster rebuilds live (ES return moments and liquidity, VIX/VVIX/VIX3M "
        "(`adj_vix_ma_*` …), FOMC calendar), `vix_only` = HAR + calendar + VIX level, `live_vix_only` = live_feasible minus VVIX and VIX3M, "
        "`free_vix_only` = the free feed (ES 1-min bars + VIX + FOMC calendar). Tree rungs: T10 / T1 = the shipped configuration refit "
        "every 10 sessions / every session; RS10 / RS1 = random search (32 candidates, every 250 sessions) refit every 10 / every session; "
        "Optuna = TPE, 50 trials per tuning point, tuning every TUNE_PER sessions, best of the first k trials, refit every session."
    )
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


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
        f"{len(skipped)} arm files looked at and not scored"
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

    # ---- stage 1: load + recalibrate
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
            has = f.index[f["pred_clock"].notna()].normalize()
            want = deck_days if not s.lo else deck_days[deck_days >= s.lo]
            missing = want.difference(has)
            row.update(
                {
                    "rows_outside_target_span": res.get("n_outside_target", 0),
                    "first_recal_day": str(has.min().date()) if len(has) else "",
                    "trade_days_in_window": len(want),
                    "trade_days_covered": len(want) - len(missing),
                    "trade_days_missing": len(missing),
                    "missing_days": " ".join(str(d.date()) for d in missing[:40])
                    + (" ..." if len(missing) > 40 else ""),
                }
            )
        cov_rows.append(row)
    cov = pd.DataFrame(cov_rows)
    print(f"[{time.time() - t0:.0f} s] stage 1 (load + recalibrate) done")
    failed = cov[cov["load_error"] != ""]
    for _, r in failed.iterrows():
        print(f"NOT SCORED {r['key']}: {r['load_error']}")

    # ---- the day sets
    full = [
        k
        for k in frames
        if not by_key[k].lo
        and int(cov.set_index("key").loc[k, "trade_days_missing"]) == 0
    ]
    common = deck_days
    for k in full:
        common = common.intersection(
            frames[k].index[frames[k]["pred_clock"].notna()].normalize()
        )
    all_int = deck_days
    for k in frames:
        all_int = all_int.intersection(
            frames[k].index[frames[k]["pred_clock"].notna()].normalize()
        )
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

    # ---- stage 2: every row's columns (the bootstraps), in the pool
    ref = frames[REFERENCE]
    jobs = []
    for k, f in frames.items():
        s = by_key[k]
        if k in full:
            days = common
        else:
            days = f.index[f["pred_clock"].notna()].normalize().intersection(deck_days)
            if s.lo:
                days = days[days >= s.lo]
            days = days.intersection(ref.index[ref["pred_clock"].notna()].normalize())
        jobs.append((s, f, ref, days, deck, k == REFERENCE))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(tgt, deck_days)
    ) as pool:
        rows = list(pool.map(score_one, jobs, chunksize=4))
    rows.append(score_short(common, deck, ref))
    print(f"[{time.time() - t0:.0f} s] stage 2 (columns + bootstraps) done")

    # gate: every row's trade aggregates equal trade_1530's (the research function itself)
    for (s, f, _r, days, _d, _i), row in zip(jobs, rows):
        pc = f["pred_clock"]
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
                "gate": "trade aggregates equal trade_1530",
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
            }
        ]
    )
    tab = meta.merge(tab, on="key", how="inner")
    tab["table"] = [
        "A" if (k in full or k == ALWAYS_SHORT) else "B" for k in tab["key"]
    ]
    tab["is_reference"] = tab["key"] == REFERENCE
    tab["is_headline"] = tab["key"] == HEADLINE

    # check rows: the direct rest-of-day forecast at 15:30 against its per-bar twin
    tab["twin_max_rel_pred_clock"] = np.nan
    tab["twin_same_position"] = np.nan
    for i, r in tab.iterrows():
        tw = r["twin_key"]
        if tw and tw in frames and r["key"] in frames:
            a_, b_ = frames[r["key"]], frames[tw]
            on = a_.index[a_.index.normalize().isin(common)]
            rel = float(
                (a_.loc[on, "pred_clock"] / b_.loc[on, "pred_clock"] - 1.0).abs().max()
            )
            ta = trade_days(
                a_.loc[on, "pred_clock"].set_axis(on.normalize()), deck.loc[common]
            )
            tb = trade_days(
                b_.loc[on, "pred_clock"].set_axis(on.normalize()), deck.loc[common]
            )
            tab.loc[i, "twin_max_rel_pred_clock"] = rel
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
    gates += repro_gates(tab, frames, deck_days)

    # exact duplicates: the same recalibrated forecast, bit for bit, on the same days as an earlier row
    tab["duplicate_of"] = ""
    seen: dict[bytes, str] = {}
    for i, r in tab.iterrows():
        k = r["key"]
        if k not in frames:
            continue
        f = frames[k]
        on = f.index[f["pred_clock"].notna() & f.index.normalize().isin(deck_days)]
        if by_key[k].lo:
            on = on[on >= by_key[k].lo]
        sig = on.asi8.tobytes() + f.loc[on, "pred_clock"].to_numpy(float).tobytes()
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
            "qlike_recal",
            "qlike_pct_vs_ref",
            "qlike_diff_ci_lo",
            "qlike_diff_ci_hi",
            "dm_vs_ref",
            "dm_p_vs_ref",
            "Sharpe_mid",
            "Sharpe_crossed",
            "dSharpe_mid_vs_ref",
            "dSharpe_mid_vs_ref_lo",
            "dSharpe_mid_vs_ref_hi",
            "dSharpe_crossed_vs_ref",
            "dSharpe_crossed_vs_ref_lo",
            "dSharpe_crossed_vs_ref_hi",
            "t_hac_mid_vs_ref",
            "same_position_as_ref",
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
                    "recal_forecast": rr["pred_clock"],
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
            "recal_forecast",
        ],
    )
    tab = tab.merge(pd.DataFrame(allsess), on="key", how="left")

    # ---- stage 3: QLIKE vs Sharpe across forecasts (day-block bootstrap of the rank correlation)
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
        q_of[("recal", k)] = slc.qlike(f["true_raw"], f["pred_clock"]).to_numpy(float)
        q_of[("raw", k)] = slc.qlike(f["true_raw"], f["plain"]).to_numpy(float)
        t = trade_days(f["pred_clock"], deck.loc[cm])
        r_of[("mid", k)] = t["mid"].to_numpy()
        r_of[("crossed", k)] = t["crossed"].to_numpy()
    rk_rows = []
    chunks = [(lo, min(lo + RANK_CHUNK, BOOT_B)) for lo in range(0, BOOT_B, RANK_CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sname, keys in sets.items():
            if len(keys) < 4:
                continue
            for qm in ("recal", "raw"):
                for fill in ("mid", "crossed"):
                    qmat = np.vstack([q_of[(qm, k)] for k in keys])
                    rmat = np.vstack([r_of[(fill, k)] for k in keys])
                    point = float(
                        _spearman_cols(
                            qmat.mean(axis=1)[:, None],
                            (rmat.mean(axis=1) / rmat.std(axis=1, ddof=1))[:, None],
                        )[0]
                    )
                    boots = np.concatenate(
                        list(
                            pool.map(
                                rankcorr_chunk,
                                [(qmat, rmat, lo, hi) for lo, hi in chunks],
                            )
                        )
                    )
                    lo_, hi_ = np.percentile(boots, [2.5, 97.5])
                    rk_rows.append(
                        {
                            "set": sname,
                            "n_forecasts": len(keys),
                            "qlike": qm,
                            "sharpe": fill,
                            "spearman": point,
                            "ci_lo": float(lo_),
                            "ci_hi": float(hi_),
                        }
                    )
    rk = pd.DataFrame(rk_rows)
    print(f"[{time.time() - t0:.0f} s] stage 3 (rank correlation) done")

    # ---- per-day long table for the P&L and loss-vs-P&L work (regenerable; not committed)
    daily = []
    for k, f in frames.items():
        s_ = by_key[k]
        g = f[f["pred_clock"].notna() & f.index.normalize().isin(deck_days)]
        if s_.lo:
            g = g[g.index >= s_.lo]
        dd = g.set_axis(g.index.normalize())
        t = trade_days(dd["pred_clock"], deck.loc[dd.index])
        daily.append(
            pd.DataFrame(
                {
                    "key": k,
                    "day": dd.index,
                    "pred_clock": dd["pred_clock"].to_numpy(),
                    "plain": dd["plain"].to_numpy(),
                    "target": dd["true_raw"].to_numpy(),
                    "rv_unclipped": dd["rv_unclipped"].to_numpy(),
                    "iv_var": deck.loc[dd.index, "iv_var"].to_numpy(float),
                    "qlike_recal": slc.qlike(
                        dd["true_raw"], dd["pred_clock"]
                    ).to_numpy(),
                    "q": t["q"].to_numpy(),
                    "ret_mid": t["mid"].to_numpy(),
                    "ret_crossed": t["crossed"].to_numpy(),
                }
            )
        )
    daily_df = pd.concat(daily, ignore_index=True)

    # ---- before / after: the pre-campaign master table against this run, key by key
    before_dir = Path(a.before_dir)
    ba, ba_gates = before_after(tab, daily_df, before_dir)
    gates += ba_gates
    print(f"[{time.time() - t0:.0f} s] before/after done ({len(ba)} keys)")

    # ---- write
    gtab = pd.DataFrame(gates)
    gtab["asserted"] = gtab["asserted"].astype("boolean").fillna(True).astype(bool)
    gtab["stored_design"] = gtab["stored_design"].fillna("")
    gtab["ok"] = gtab["ok"].astype(bool)
    prov = provenance_table(tab)
    tab.to_csv(OUT / "master_table.csv", index=False)
    cov.to_csv(OUT / "master_table_days.csv", index=False)
    ext.to_csv(OUT / "master_table_extremes.csv", index=False)
    gtab.to_csv(OUT / "master_table_gates.csv", index=False)
    rk.to_csv(OUT / "master_table_rankcorr.csv", index=False)
    prov.to_csv(OUT / "master_table_provenance.csv", index=False)
    if len(vs_h):
        vs_h.to_csv(OUT / "master_table_vs_headline.csv", index=False)
    if len(ba):
        ba.to_csv(OUT / "before_after.csv", index=False)
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
        "qlike_recal",
        "qlike_pct_vs_ref",
        "dm_vs_ref",
        "Sharpe_mid",
        "Sharpe_crossed",
        "mean_mid",
        "hit_rate_mid",
        "n_buy",
        "dSharpe_mid_vs_ref",
        "dSharpe_mid_vs_ref_lo",
        "dSharpe_mid_vs_ref_hi",
        "dSharpe_mid_vs_short_lo",
        "dSharpe_mid_vs_short_hi",
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
        rc = subprocess.run(
            [sys.executable, str(ROOT / "writeup" / "make_master_table_close_tex.py")],
            check=False,
        ).returncode
        if rc:
            print(
                f"WARNING: writeup/make_master_table_close_tex.py exited {rc}; the CSVs and SUMMARY.md are written"
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
    """Why the headline: stated as rules, the numbers filled in from this run."""
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
    rank_q = int((fc["qlike_recal"] < h["qlike_recal"]).sum()) + 1
    lines.append(
        f"- It is a 15:30-specific model on the inputs a 15:30 forecaster can rebuild live (`live_feasible`), so the trade it scores can be run. "
        f"It ranks {rank_s} of {len(fc)} table-A forecasts on sign(s) Sharpe (mid) and {rank_q} of {len(fc)} on the recalibrated QLIKE."
    )
    best = fc.sort_values("Sharpe_mid", ascending=False)
    top = best.iloc[0]
    if top.name != HEADLINE:
        lines.append(
            f"- The highest Sharpe in table A is {top['label']} ({top['Sharpe_mid']:.2f} mid / {top['Sharpe_crossed']:.2f} crossed). Choosing "
            f"the maximum of {len(fc)} Sharpe ratios would select on the trade's own noise; the headline is chosen on feasibility and "
            "on being the per-bar model of record of the research scorer (the per-bar ridge is the estimator the 15:30 study defined first), "
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
                g["qlike_recal"].idxmin()
            ]
    ex = [k for k in dict.fromkeys(NEIGHBOURS + tuple(late)) if k in v.index]
    if ex:
        lines.append(
            "- Paired against the headline itself on the same days (`master_table_vs_headline.csv`; a negative QLIKE % / DM = the "
            "alternative forecasts better, a positive Sharpe difference = the alternative trades better; 95 % intervals):"
        )
        lines.append("")
        lines.append(
            "  | alternative | QLIKE | % vs headline | DM | Sharpe mid / crossed | ΔSharpe mid vs headline | ΔSharpe crossed vs headline | same position |"
        )
        lines.append("  |---|---:|---:|---:|---|---|---|---:|")
        for k in ex:
            r = v.loc[k]
            lines.append(
                f"  | {r['label']} | {r['qlike_recal']:.4f} | {r['qlike_pct_vs_headline']:+.1f} | {r['dm_vs_headline']:+.2f} | "
                f"{r['Sharpe_mid']:.2f} / {r['Sharpe_crossed']:.2f} | "
                f"{fmt_ci(r['dSharpe_mid_vs_headline'], r['dSharpe_mid_vs_headline_lo'], r['dSharpe_mid_vs_headline_hi'])} | "
                f"{fmt_ci(r['dSharpe_crossed_vs_headline'], r['dSharpe_crossed_vs_headline_lo'], r['dSharpe_crossed_vs_headline_hi'])} | "
                f"{100 * r['same_position_as_headline']:.1f} % |"
            )
        lines.append("")
        others = v.drop(index=HEADLINE)
        others = others[others.index.isin(fc.index)]
        lines.append(
            f"- Over all {len(others)} other table-A forecasts, {int((others['dSharpe_mid_vs_headline_lo'] > 0).sum())} trade better than the headline "
            f"with an interval above zero (mid; {int((others['dSharpe_crossed_vs_headline_lo'] > 0).sum())} crossed) and "
            f"{int((others['dSharpe_mid_vs_headline_hi'] < 0).sum())} trade worse with an interval below zero (mid; "
            f"{int((others['dSharpe_crossed_vs_headline_hi'] < 0).sum())} crossed); {int((others['qlike_diff_ci_hi'] < 0).sum())} forecast better "
            f"on QLIKE with the day-block interval below zero, {int((others['qlike_diff_ci_lo'] > 0).sum())} worse."
        )
    n_res_ref = int((fc["dSharpe_mid_vs_ref_lo"] > 0).sum())
    lines.append(
        f"- Against the reference, {n_res_ref} of {len(fc) - 1} table-A forecasts have a mid-fill Sharpe-difference interval wholly above zero; "
        f"{int((fc['dSharpe_crossed_vs_ref_lo'] > 0).sum())} at the crossed fill. Against always short, "
        f"{int((fc['dSharpe_mid_vs_short_lo'] > 0).sum())} (mid) and {int((fc['dSharpe_crossed_vs_short_lo'] > 0).sum())} (crossed) of {len(fc)}."
    )
    lq = live.sort_values("qlike_recal")
    if len(lq):
        b = lq.iloc[0]
        lines.append(
            f"- On forecast accuracy alone the best live-feasible forecast is {b['label']} (QLIKE {b['qlike_recal']:.4f}, "
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
        bq = g.sort_values("qlike_recal").iloc[0]
        bs = g.sort_values("Sharpe_mid", ascending=False).iloc[0]
        out.append(
            f"- **{fam}** ({len(g)}): QLIKE {g['qlike_recal'].min():.4f} .. {g['qlike_recal'].max():.4f}, Sharpe mid {g['Sharpe_mid'].min():.2f} .. "
            f"{g['Sharpe_mid'].max():.2f}; best QLIKE {bq['label']} ({bq['qlike_recal']:.4f}, Sharpe {bq['Sharpe_mid']:.2f}); best Sharpe "
            f"{bs['label']} ({bs['Sharpe_mid']:.2f}, QLIKE {bs['qlike_recal']:.4f})."
        )
    fq = fc.sort_values("qlike_recal").iloc[0]
    fs = fc.sort_values("Sharpe_mid", ascending=False).iloc[0]
    out.append(
        f"- Over all of table A the lowest QLIKE is {fq['label']} ({fq['qlike_recal']:.4f}; Sharpe {fq['Sharpe_mid']:.2f}, rank "
        f"{int((fc['Sharpe_mid'] > fq['Sharpe_mid']).sum()) + 1} of {len(fc)} on Sharpe) and the highest Sharpe is {fs['label']} "
        f"({fs['Sharpe_mid']:.2f}; QLIKE {fs['qlike_recal']:.4f}, rank {int((fc['qlike_recal'] < fs['qlike_recal']).sum()) + 1} of {len(fc)} on QLIKE)."
    )
    n_q = int((fc["qlike_diff_ci_hi"] < 0).sum())
    n_qw = int((fc["qlike_diff_ci_lo"] > 0).sum())
    gap = (fc["qlike_raw"] - fc["qlike_recal"]).sort_values(ascending=False)
    big = "; ".join(
        f"{fc.loc[k, 'label']} {fc.loc[k, 'qlike_raw']:.4f} raw vs {fc.loc[k, 'qlike_recal']:.4f} recalibrated"
        for k in gap.index[:3]
    )
    out.append(
        f"- Against the reference's QLIKE, {n_q} forecasts have a day-block interval wholly below zero (better) and {n_qw} wholly above (worse). "
        f"The raw (plain back-transform) QLIKE ranks forecasts differently from the recalibrated one (Spearman across table A "
        f"{float(fc['qlike_raw'].rank().corr(fc['qlike_recal'].rank())):+.2f}); the recalibration's term s lowers the loss most for {big}."
    )
    return out


def rank_reading(A: pd.DataFrame, rk: pd.DataFrame) -> list[str]:
    """What the rank-correlation table says, with the group medians behind it."""
    fc = A[
        (A["family"] != "direct rest-of-day at 15:30 (check)")
        & (A["duplicate_of"] == "")
    ]
    g48 = fc[fc["family"].isin(["paper", "pooled twin"])]
    gpb = fc[~fc["family"].isin(["paper", "pooled twin"])]
    glin = fc[fc["family"].isin(["per-bar linear", "VIX-only family"])]

    def get(sname: str) -> pd.Series | None:
        r = rk[
            (rk["set"] == sname) & (rk["qlike"] == "recal") & (rk["sharpe"] == "mid")
        ]
        return r.iloc[0] if len(r) else None

    def cc(r: pd.Series) -> str:
        return f"{r['spearman']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"

    ra = get("all table-A forecasts")
    rp = get("per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies)")
    rl = get("per-bar linear + VIX-only family")
    if ra is None or not len(g48) or not len(gpb):
        return []
    txt = (
        f"Reading: across all of table A a lower QLIKE goes with a higher Sharpe ({cc(ra)}). Part of it is the split between the {len(g48)} "
        f"forecasts fitted on all 48 bars (median QLIKE {g48['qlike_recal'].median():.4f}, median Sharpe {g48['Sharpe_mid'].median():.2f}) "
        f"and the {len(gpb)} per-bar forecasts (median QLIKE {gpb['qlike_recal'].median():.4f}, median Sharpe {gpb['Sharpe_mid'].median():.2f})"
    )
    if rp is not None:
        both = gpb[
            (gpb["qlike_recal"] >= gpb["qlike_recal"].quantile(0.75))
            & (gpb["Sharpe_mid"] <= gpb["Sharpe_mid"].quantile(0.25))
        ]
        txt += f"; among the per-bar forecasts alone it is {cc(rp)}"
        if len(both):
            txt += (
                f", carried by the {len(both)} per-bar forecasts in the worst quarter on both counts ("
                + "; ".join(
                    both.sort_values("qlike_recal", ascending=False)["label"].head(6)
                )
                + ("; ..." if len(both) > 6 else "")
                + ")"
            )
    if rl is not None and len(glin):
        txt += (
            f"; within the per-bar linear and VIX-only family (QLIKE {glin['qlike_recal'].min():.4f} .. {glin['qlike_recal'].max():.4f}, "
            f"Sharpe {glin['Sharpe_mid'].min():.2f} .. {glin['Sharpe_mid'].max():.2f}) it is {cc(rl)}: among forecasts of similar accuracy, "
            "QLIKE does not order the trade"
        )
    return [txt + "."]


def extreme_lines(tab: pd.DataFrame, ext: pd.DataFrame) -> list[str]:
    A = tab[tab["key"] != ALWAYS_SHORT]
    out = [
        "- **Rule (named):** PLAIN_FLOOR(d) = the smallest 16:00 target of the previous 250 sessions (the recalibration window SMEAR_W), "
        "lagged one session. A plain back-transform f²·B below it forecasts a quieter 15:30–16:00 half hour than any of the past year; it is flagged, "
        "listed in `master_table_extremes.csv`, and the raw QLIKE is reported as is (`qlike_raw`) and with flagged forecasts raised to the floor "
        "(`qlike_raw_floored`). Nothing is clipped silently; the recalibrated forecast (≥ s·B) needs no treatment and is never below the floor "
        f"on a trade day in this run (max count {int(A['n_recal_below_floor'].max())})."
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
            f"floor's {slc.SMEAR_MIN}-session warm-up, early closes included): {len(ext)} flagged rows over {len(ne)} forecasts, {ec} of them on "
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
                f"plain forecast {r['plain_forecast']:.3g} vs floor {r['plain_floor']:.3g} and target {r['target_true_raw']:.3g}; that one day's raw "
                f"QLIKE is {r['qlike_raw_this_day']:.1f}. Over all {int(t['sessions'])} sessions the raw QLIKE is {t['qlike_raw_all_sessions']:.4f} as is, "
                f"{t['qlike_raw_floored_all_sessions']:.4f} floored, {t['qlike_raw_all_sessions_no_early_close']:.4f} without early closes; on the trade "
                f"days {t['qlike_raw']:.4f} (recalibrated {t['qlike_recal']:.4f})."
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
                "against the unclipped variance; the recalibration (its `qlike_causal` "
                f"{float(r['qlike_causal'].iloc[0]):.4f}) and the trade frame's exclusion of early closes both remove it."
            )
    return out


# --------------------------------------------------------------------------- provenance + before / after
LINEAR_FAMILIES = (
    "per-bar linear",
    "VIX-only family",
    "implied-vol representations",
    "HAR-ladder variants",
    "chain-period buckets",
)
UNTOUCHED = ("paper", "pooled twin")  # the families the 16:00 campaign did not re-run


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
    if fam in LINEAR_FAMILIES and ARM_ROOT == DEFAULT_ARM_ROOT and source == "arm":
        return "nothing (arm root = the pre-campaign arms)"
    w = FAMILY_PROV.get(fam, {}).get("changed", "unknown family")
    if fam in LINEAR_FAMILIES and key in clusters:
        old, new = clusters[key]
        w += f"; re-run on {new}" + (f" (was {old})" if old != new else " (as before)")
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
    tab: pd.DataFrame, daily: pd.DataFrame, before_dir: Path
) -> tuple[pd.DataFrame, list[dict]]:
    """Every forecast key before (the pre-campaign master table) and after (this run), paired on the same days."""
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
    db = {k: g.set_index("day").sort_index() for k, g in bd.groupby("key")}
    dn = {k: g.set_index("day").sort_index() for k, g in daily.groupby("key")}
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
            "n_days_after": int(now.loc[k, "n_days"]) if k in now.index else np.nan,
            "qlike_recal_after": now.loc[k, "qlike_recal"]
            if k in now.index
            else np.nan,
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
                    "qlike_recal_before": b_["qlike_recal"],
                    "Sharpe_mid_before": b_["Sharpe_mid"],
                    "Sharpe_crossed_before": b_["Sharpe_crossed"],
                }
            )
        if status == "both":
            row["qlike_diff"] = row["qlike_recal_after"] - row["qlike_recal_before"]
            row["qlike_pct_change"] = (
                100.0 * row["qlike_diff"] / row["qlike_recal_before"]
            )
            row["dSharpe_mid"] = row["Sharpe_mid_after"] - row["Sharpe_mid_before"]
            row["dSharpe_crossed"] = (
                row["Sharpe_crossed_after"] - row["Sharpe_crossed_before"]
            )
            if k in dn and k in db:
                days = dn[k].index.intersection(db[k].index)
                aa, bb = dn[k].loc[days], db[k].loc[days]
                row["n_days_paired"] = len(days)
                row["positions_changed"] = int(
                    (aa["q"].to_numpy() != bb["q"].to_numpy()).sum()
                )
                row["max_rel_pred_clock_change"] = float(
                    (aa["pred_clock"] / bb["pred_clock"] - 1.0).abs().max()
                )
                qd = aa["qlike_recal"].to_numpy() - bb["qlike_recal"].to_numpy()
                lo, hi = base.day_block_ci(pd.Series(qd, index=days))
                row.update({"qlike_diff_ci_lo": lo, "qlike_diff_ci_hi": hi})
                for fill in ("mid", "crossed"):
                    x, x_lo, x_hi = paired_sharpe(
                        aa[f"ret_{fill}"].to_numpy(), bb[f"ret_{fill}"].to_numpy()
                    )
                    row.update(
                        {
                            f"dSharpe_{fill}_paired": x,
                            f"dSharpe_{fill}_lo": x_lo,
                            f"dSharpe_{fill}_hi": x_hi,
                        }
                    )
                if len(days) == row["n_days_after"] == row["n_days_before"]:
                    dv = max(
                        abs(sharpe(aa["ret_mid"].to_numpy()) - row["Sharpe_mid_after"]),
                        abs(
                            sharpe(bb["ret_mid"].to_numpy()) - row["Sharpe_mid_before"]
                        ),
                        abs(float(aa["qlike_recal"].mean()) - row["qlike_recal_after"]),
                        abs(
                            float(bb["qlike_recal"].mean()) - row["qlike_recal_before"]
                        ),
                    )
                    gates.append(
                        {
                            "gate": "before/after: the two daily frames reproduce both master tables (Sharpe mid, QLIKE)",
                            "forecast": k,
                            "n": len(days),
                            "value": dv,
                            "bound": 1e-12,
                            "ok": dv <= 1e-12,
                        }
                    )
                if fam in UNTOUCHED:
                    dv = max(
                        row["max_rel_pred_clock_change"],
                        abs(row["dSharpe_mid"]),
                        abs(row["dSharpe_crossed"]),
                        abs(row["qlike_diff"]),
                    )
                    gates.append(
                        {
                            "gate": "untouched by the campaign: equals the pre-campaign master table (pred_clock rel, Sharpe, QLIKE)",
                            "forecast": k,
                            "n": len(days),
                            "value": dv,
                            "bound": 0.0,
                            "ok": dv == 0.0,
                            "asserted": dedup,
                            "stored_design": "pre-campaign master table",
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
        "qlike_recal_before",
        "qlike_recal_after",
        "qlike_diff",
        "qlike_pct_change",
        "qlike_diff_ci_lo",
        "qlike_diff_ci_hi",
        "Sharpe_mid_before",
        "Sharpe_mid_after",
        "dSharpe_mid",
        "dSharpe_mid_lo",
        "dSharpe_mid_hi",
        "Sharpe_crossed_before",
        "Sharpe_crossed_after",
        "dSharpe_crossed",
        "dSharpe_crossed_lo",
        "dSharpe_crossed_hi",
        "positions_changed",
        "max_rel_pred_clock_change",
        "dSharpe_mid_paired",
        "dSharpe_crossed_paired",
        "family_before",
        "label_before",
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
        f"Before = the pre-campaign master table (commit 6177b74; snapshot `{bdir}/`), after = this run; every key present "
        "in both is paired on the same days with the same resampled days as every other interval (`before_after.csv`; "
        "difference = after minus before)."
    ]

    def one(k: str, name: str) -> str:
        if k not in b.index:
            return f"- **{name}** (`{k}`): not in both tables."
        r = b.loc[k]
        return (
            f"- **{name}** (`{k}`; {r['what_changed']}): QLIKE {r['qlike_recal_before']:.4f} -> {r['qlike_recal_after']:.4f} "
            f"({r['qlike_pct_change']:+.2f} %, interval on the daily difference [{r['qlike_diff_ci_lo']:+.5f}, {r['qlike_diff_ci_hi']:+.5f}]); "
            f"Sharpe mid {r['Sharpe_mid_before']:.2f} -> {r['Sharpe_mid_after']:.2f} "
            f"({fmt_ci(r['dSharpe_mid'], r['dSharpe_mid_lo'], r['dSharpe_mid_hi'])}), crossed {r['Sharpe_crossed_before']:.2f} -> "
            f"{r['Sharpe_crossed_after']:.2f} ({fmt_ci(r['dSharpe_crossed'], r['dSharpe_crossed_lo'], r['dSharpe_crossed_hi'])}); "
            f"positions changed on {int(r['positions_changed'])} of {int(r['n_days_paired'])} days; largest relative change of "
            f"the recalibrated forecast {r['max_rel_pred_clock_change']:.2e}."
        )

    L.append(one(HEADLINE, "Headline"))
    L.append(one(REFERENCE, "Reference"))
    L.append("")
    L.append(
        "| family | keys in both | what changed | QLIKE % change, median [min, max] | ΔSharpe mid, median [min, max] | "
        "ΔSharpe mid interval above / below 0 | ΔSharpe crossed interval above / below 0 | positions changed, median [max] |"
    )
    L.append("|---|---:|---|---|---|---|---|---|")
    fams = [
        f for f in FAMILY_ORDER if f in set(b["family"])
    ]  # always short: no forecast, unchanged
    for fam in fams:
        g = b[b["family"] == fam]
        wc = g["what_changed"].str.split(";").str[0].value_counts()
        pos = g["positions_changed"].dropna()
        L.append(
            f"| {fam} | {len(g)} | {'; '.join(wc.index)} | {g['qlike_pct_change'].median():+.2f} "
            f"[{g['qlike_pct_change'].min():+.2f}, {g['qlike_pct_change'].max():+.2f}] | {g['dSharpe_mid'].median():+.2f} "
            f"[{g['dSharpe_mid'].min():+.2f}, {g['dSharpe_mid'].max():+.2f}] | {int((g['dSharpe_mid_lo'] > 0).sum())} / "
            f"{int((g['dSharpe_mid_hi'] < 0).sum())} | {int((g['dSharpe_crossed_lo'] > 0).sum())} / "
            f"{int((g['dSharpe_crossed_hi'] < 0).sum())} | "
            + (f"{pos.median():.0f} [{pos.max():.0f}]" if len(pos) else "")
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
    moved = b[(b["dSharpe_mid_lo"] > 0) | (b["dSharpe_mid_hi"] < 0)]
    if len(moved):
        L.append("")
        L.append(
            "Keys whose own Sharpe (mid) moved with the paired interval excluding zero: "
            + "; ".join(
                f"{r['label']} {r['Sharpe_mid_before']:.2f} -> {r['Sharpe_mid_after']:.2f} "
                f"({fmt_ci(r['dSharpe_mid'], r['dSharpe_mid_lo'], r['dSharpe_mid_hi'])})"
                for _, r in moved.sort_values("dSharpe_mid").iterrows()
            )
            + "."
        )
    return L


def beats_headline_lines(A: pd.DataFrame, vs_h: pd.DataFrame) -> list[str]:
    """Recorded, never acted on: any forecast that now beats the headline with an interval clear of zero."""
    if not len(vs_h) or HEADLINE not in A.index:
        return []
    fc = A[
        (A["family"] != "direct rest-of-day at 15:30 (check)")
        & (A["duplicate_of"] == "")
    ].index
    v = vs_h.set_index("key")
    v = v[v.index.isin(fc) & (v.index != HEADLINE)]

    def names(m: pd.Series, col: str) -> str:
        g = v[m].sort_values(col, ascending=False)
        return (
            "; ".join(
                f"{r['label']} ({fmt_ci(r[col], r[col + '_lo'], r[col + '_hi'])})"
                for _, r in g.iterrows()
            )
            or "none"
        )

    bm = v["dSharpe_mid_vs_headline_lo"] > 0
    bx = v["dSharpe_crossed_vs_headline_lo"] > 0
    bq = v["qlike_diff_ci_hi"] < 0
    L = [
        f"- **Does any forecast now beat the headline?** Of the {len(v)} other table-A forecasts (check rows and exact duplicates "
        f"left out), with the paired Sharpe-difference interval wholly above zero: mid {int(bm.sum())} "
        f"({names(bm, 'dSharpe_mid_vs_headline')}); crossed {int(bx.sum())} ({names(bx, 'dSharpe_crossed_vs_headline')}). "
        f"With the QLIKE-difference interval wholly below zero (better accuracy): {int(bq.sum())}"
        + (
            " ("
            + "; ".join(
                f"{r['label']} {r['qlike_pct_vs_headline']:+.1f} %"
                for _, r in v[bq].sort_values("qlike_pct_vs_headline").iterrows()
            )
            + ")"
            if bq.any()
            else ""
        )
        + "."
    ]
    if bm.any() or bx.any():
        L.append(
            f"  RECORDED, not acted on: the headline constant stays `{HEADLINE}` (it is chosen on feasibility and as the per-bar "
            "model of record, not on the maximum of many Sharpe ratios); the forecasts above are the candidates to look at."
        )
    return L


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
