"""Murphy / threshold analysis and sample splits of the HEADLINE forecast, plain forecast.

2026-10-06 (user decision): the paper has NO headline model and no live-feasible forecasts;
D.17 is re-based on the paper's MODEL SET (results/close_master_table/paper_model_set.csv):
main() scores every model-set forecast (plain f^2 B) by QLIKE, the elementary score at the
trade's threshold, the economic loss and the hit shares, draws the Murphy grid of the whole
set, reports Spearman of minus each loss with the Sharpe ratio across the set, and the
sample splits and the sign split as counts and medians across the set.  Outputs:
scores_modelset.csv, murphy_grid_modelset.csv, murphy_leader_by_threshold_modelset.csv,
rank_vs_sharpe_modelset.csv, splits_modelset.csv, splits_summary_modelset.csv,
sign_split_modelset.csv, gates.csv, murphy_modelset.pdf (vector; 2026-10-15) and .png, SUMMARY.md,
writeup/generated/appendix_close_murphy_splits.tex.  The headline run is kept as
main_headline_parked() (not called); its description follows.

2026-10-06: recalibration removed, user decision.  Every forecast is its PLAIN
back-transform f^2 * B (the per-day frame's column ``plain``), never the research
scorer's (f^2 + s) * B (``pred_clock``); the trade is q = sign(plain - iv_var) and QLIKE,
the elementary scores and the Murphy grid are all of the plain forecast.  No intervals
(the repo bootstrap is disabled, commit b761b28): the Sharpe intervals of the splits and
the bootstrap intervals of the rank correlations are parked; point estimates only.  The
committed-output gates become reported-only comparisons for every row that depends on a
forecast (the numbers change by design); the rows that read no forecast (the two oracle
rules, always short's splits) are still asserted equal to the committed copy.
The text below is the pre-2026-10-06 description, kept for the record.

Two of the diagnostics parked from Section 5.4 (writeup/sections/appendix_running_parked.tex,
computed there for the paper's eight forecasts under the deck's session-bar recalibration),
re-run for the headline under the 16:00-bar recalibration of the master table
(experiments/master_table_close.py).  Nothing is re-implemented that the master table
already computes: the per-day frame results/close_master_table/master_table_daily.parquet
(the recalibrated forecast pred_clock, the scorer's target, the unclipped realized
variance, the 15:30 implied variance, the position q and the mid / crossed returns of
every table-A forecast on the 866 deck days) is the input, and the headline's Sharpe
and QLIKE are gated against master_table.csv.

PART 1 -- the decision-aligned scores (writeup/intraday_proposals/59_trade_aligned_loss.py,
mirrored):
  QLIKE      y/f - log(y/f) - 1, y the scorer's target, f = pred_clock.
  ES (c)     the elementary score of the mean at the threshold c x slice, slice = the
             15:30 implied variance: |y/slice - c| 1{f and y on opposite sides of c slice}.
             QLIKE = integral ES_c c^-2 dc (gated day by day); the trade reads c = 1.
  EL         |R| 1{q != sign(R)}: mean(q R) = mean|R| - 2 mean(EL) (gated).
  hit_var    share of days f and y fall on the same side of the slice; hit_pay: share of
             days q has the sign of the settlement return R.
  tail       long on the 10 / 20 / 50 best long-straddle days.
  Murphy     mean ES_c on the grid c = 2^(k/8), k = -16..16, one line per forecast.
  oracles    a rule told y (q = sign(y - slice)) and a rule told sign(R).
  ranks      Spearman of minus each loss with the crossed Sharpe over the comparison set
             and over every table-A forecast of the master table's rank set (table A
             without the rest-of-day check rows, exact duplicates and always short: 220
             after the 16:00 campaign), with the day-block bootstrap interval.  The
             "overall" ranks of a forecast are over the same rank set (competition ranks,
             ties share the lowest); the two oracle rules are never ranked.

PART 2 -- sample splits of the headline's sign(s) return (writeup/fill_close_option_numbers.py
and the notebook's sign split, mirrored), beside the reference (blk2 under THIS scorer)
and always short:
  COVID      drop the deck days through 2020-03-31 (n, Sharpe, t = sqrt(n) mean / sd).
  era        before / from 2022-05-16 (SPXW listings daily).
  pins       R = -1 exactly: count, share, mean return on / off pins, pin share of P&L.
  sign       mean R when s > 0 vs s <= 0, the difference with an HC0 t (statsmodels).
  Sharpes carry the day-block bootstrap 95 % interval of the master table (block 21,
  2000 draws, seed [0, n]); the difference between two disjoint periods is not paired.

Comparison set (I4c, 2026-09-30): the eight forecasts of the first run plus one canonical
row per family the 16:00 campaign added -- the untuned LightGBM refit every session (T1),
the LightGBM tuned by Optuna at every session, best of 50 trials (tp1), and the
intraday-sequence LSTM -- each on the live-feasible inputs and MSE-selected; its tree and
LSTM tables are the campaign's masked runs (the per-bar LSTM refit every session).

Committed-output gate (I4c): the headline's forecast and positions are unchanged by the
campaign and the reference, always short and the oracles are untouched, so splits.csv,
sign_split.csv, the headline's, the reference's and the oracles' rows of
scores_by_forecast.csv (every column but the ranks) and the headline's and the reference's
rows of murphy_grid.csv must equal their committed copies (git show PREV_REV:<path>;
PREV_REV = HEAD unless the environment variable CLOSE_PREV_REV names another revision),
cell by cell, to |new - old| <= 1e-9 max(1, |old|).  Every other forecast present in both
is compared for the record in before_after_committed.csv (not asserted).

Outputs: results/close_murphy_splits/*.csv, murphy_headline.png, SUMMARY.md (from the CSVs),
writeup/generated/appendix_close_murphy_splits.tex (tables + number macros).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402

OUT = ROOT / "results" / "close_murphy_splits"
TEX = ROOT / "writeup" / "generated" / "appendix_close_murphy_splits.tex"
# 2026-10-15 (number macros): macro-only file for the rehauled paper (paper_rehaul.tex loads it in its preamble; TEX
# above mixes tables and macros, so it is not loaded there). Written by main() and by --numbers-only (from the
# stored CSVs, no re-scoring).
NUMBERS_TEX = ROOT / "writeup" / "generated" / "close_murphy_splits_numbers.tex"
# the values 07_economics.tex typed on 2026-10-15; write_numbers() stops if a stored value differs
# 2026-10-23 (user, E32): the model set loses its 18 ten-day tree rungs (112 -> 94 forecasts), which moves the rank
# correlation from -0.55 to -0.52; the text no longer prints it (parked 2026-10-17 in writeup/rehaul/07_economics.tex
# for the post-warmup \rheRankCorrES). Was: NUMBERS_TYPED = {"hmurRhoESLossMid": r"\ensuremath{-}0.55"}
NUMBERS_TYPED = {"hmurRhoESLossMid": r"\ensuremath{-}0.52"}
DAILY = mtc.OUT / "master_table_daily.parquet"
MASTER = mtc.OUT / "master_table.csv"
HEADLINE = mtc.HEADLINE
REFERENCE = mtc.REFERENCE
# the comparison set: the headline, the reference, the incumbent OLS, the headline's
# lasso twin, the all-features ridge, an untuned and a tuned tree, the LSTM (the first
# eight), and one canonical row per family the 16:00 campaign added (live-feasible,
# MSE-selected): the untuned LightGBM refit every session (T1), the LightGBM tuned by
# Optuna at every session (tp1, best of 50 trials), the intraday-sequence LSTM
COMPARE = [
    HEADLINE,
    REFERENCE,
    "a0",
    "sub_lasso_live_feasible",
    "sub_ridge_all_features",
    "subtree_lgbm_live_feasible",
    "subtree_tuned_all_features_lgbm",
    "lstm_live_feasible",
    "subtree_daily_live_feasible_lgbm",
    "subtree_optuna_tp1_live_feasible_lgbm",
    "lstm_intraday_live_feasible",
]
CHECK_FAMILY = "direct rest-of-day at 15:30 (check)"  # the master table's check rows
ORACLES = ["oracle: told RV (sign(RV - slice))", "oracle: told the payoff's sign"]
RANK_COLS = [
    "QLIKE_rank_all",
    "ES_rank_all",
    "Sharpe_crossed_rank_all",
    "QLIKE_rank_set",
    "ES_rank_set",
]
# the committed outputs the headline rows are gated against (the headline is unchanged by the campaign)
PREV_REV = os.environ.get("CLOSE_PREV_REV", "HEAD")
COMMITTED_TOL = 1e-9  # |new - old| <= COMMITTED_TOL * max(1, |old|), cell by cell
# 2026-10-06: recalibration removed, user decision -- the forecast column of the per-day
# frame every score and position is read from (was "pred_clock", (f^2 + s) B)
FCOL = "plain"
PLAIN_HEADLINE_SHARPE_MID = "1.79"  # the GUI's independent recomputation, 866 days
# the master table's column holding the mean QLIKE of the plain forecast (first present)
QLIKE_PLAIN_COLS = ("qlike_plain", "qlike_raw")
# 2026-10-06: no intervals (bootstrap disabled, commit b761b28) -- the interval macros the
# appendix text still names are written as visible ?? stubs so the paper compiles
PARKED_MACROS = (
    "hmurRhoQSetLo",
    "hmurRhoQSetHi",
    "hmurRhoESSetLo",
    "hmurRhoESSetHi",
    "hmurRhoQAllLo",
    "hmurRhoQAllHi",
    "hmurRhoESAllLo",
    "hmurRhoESAllHi",
    "hsplHPostCovidLo",
    "hsplHPostCovidHi",
    "hsplHPreLo",
    "hsplHPreHi",
    "hsplHPostLo",
    "hsplHPostHi",
)
MURPHY_C = np.array([2.0 ** (k / 8.0) for k in range(-16, 17)])  # 1/4 .. 4, c = 1 on it
TOP_K = (10, 20, 50)
COVID_END = pd.Timestamp("2020-03-31")
ERA_CUT = pd.Timestamp("2022-05-16")
# 2026-10-06: 4001 -> 20001 points, integrated in chunks of days.  The plain forecast sits
# farther from the target than the recalibrated one on some days (a wider [lo, hi]), and the
# trapezoid's error on 4001 points reached 2.4e-6 against the 1e-6 bound; the identity itself
# is unchanged (the error falls as 1 / grid^2)
# G3_GRID = 4001
G3_GRID = 20001
G3_CHUNK = 64  # days per chunk (memory: 64 x 20001 floats per array)
GATE_REL = 1e-9
GATE_ABS = 1e-10


def rank_set(mt: pd.DataFrame) -> set[str]:
    """The master table's rank set: table A, no check rows, no exact duplicates, no always short."""
    dup = mt["duplicate_of"].fillna("").astype(str)
    keep = (
        (mt["table"] == "A")
        & (mt["family"] != CHECK_FAMILY)
        & (dup == "")
        & (mt.index != mtc.ALWAYS_SHORT)
    )
    return set(mt.index[keep])


def committed_csv(path: Path, **kw) -> tuple[pd.DataFrame, str]:
    """The committed copy of an output CSV (git show PREV_REV:<path>) and the revision's short sha."""
    rel = path.relative_to(ROOT).as_posix()
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", PREV_REV], cwd=ROOT, text=True
    ).strip()
    raw = subprocess.check_output(["git", "show", f"{PREV_REV}:{rel}"], cwd=ROOT)
    return pd.read_csv(io.BytesIO(raw), **kw), sha


def committed_diff(new: pd.DataFrame, old: pd.DataFrame) -> float:
    """Largest |new - old| / max(1, |old|) over the committed frame's cells (NaN = NaN);
    inf when a committed row or column is missing or a text cell differs."""
    if not (old.index.isin(new.index).all() and old.columns.isin(new.columns).all()):
        return float("inf")
    n = new.loc[old.index, old.columns]
    worst = 0.0
    for c in old.columns:
        a, b = n[c], old[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a_, b_ = a.to_numpy(float), b.to_numpy(float)
            same_nan = np.isnan(a_) & np.isnan(b_)
            d = np.where(same_nan, 0.0, np.abs(a_ - b_) / np.maximum(1.0, np.abs(b_)))
            if np.isnan(d).any():
                return float("inf")
            worst = max(worst, float(d.max(initial=0.0)))
        elif not (a.astype(str) == b.astype(str)).all():
            return float("inf")
    return worst


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def elementary(f_over_s: np.ndarray, y_over_s: np.ndarray, c: float) -> np.ndarray:
    lo = np.minimum(f_over_s, y_over_s)
    hi = np.maximum(f_over_s, y_over_s)
    return np.abs(y_over_s - c) * ((lo <= c) & (c < hi))


def stats(x: np.ndarray) -> tuple[int, float, float]:
    """(n, t = sqrt(n) mean / sd, annualized Sharpe) -- fill_close_option_numbers.stats."""
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    if not sd > 0:  # every return identical (always short on pin days): no t, no Sharpe
        return len(x), np.nan, np.nan
    return len(x), float(np.sqrt(len(x)) * x.mean() / sd), mtc.sharpe(x)


def fm(v: float, spec: str) -> str:
    """A number for a table cell, '--' when it does not exist."""
    return "--" if not np.isfinite(v) else format(v, spec)


# 2026-10-06: no intervals (bootstrap disabled, commit b761b28) -- parked
# def sharpe_ci(x: np.ndarray) -> tuple[float, float]:
#     if not x.std(ddof=1) > 0:
#         return np.nan, np.nan
#     idx = mtc.boot_idx(len(x))
#     lo, hi = np.percentile(mtc.boot_sharpe(x, idx), [2.5, 97.5])
#     return float(lo), float(hi)


def spearman_point(loss: np.ndarray, pnl_rows: np.ndarray) -> float:
    """Spearman of minus the mean loss with the crossed Sharpe across forecasts (rows);
    loss and pnl_rows are (d, n).  2026-10-06: point estimate only -- the day-block
    bootstrap over the columns (days) is parked (bootstrap disabled, commit b761b28)."""
    from scipy.stats import spearmanr

    # idx = mtc.boot_idx(pnl_rows.shape[1])
    point = float(spearmanr(-loss.mean(axis=1), _sharpe_rows(pnl_rows))[0])
    # draws = []
    # for b in range(idx.shape[0]):
    #     cols = idx[b]
    #     draws.append(
    #         spearmanr(-loss[:, cols].mean(axis=1), _sharpe_rows(pnl_rows[:, cols]))[0]
    #     )
    # lo, hi = np.percentile(np.array(draws, float), [2.5, 97.5])
    return point


def common_diff(new: pd.DataFrame, old: pd.DataFrame) -> float:
    """committed_diff on the rows and columns both frames carry (a reported comparison:
    the run without intervals has fewer columns than the committed one)."""
    rows = old.index.intersection(new.index)
    cols = old.columns.intersection(new.columns)
    if not len(rows) or not len(cols):
        return float("nan")
    return committed_diff(new.loc[rows, cols], old.loc[rows, cols])


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(x)
    return a.mean(axis=1) / a.std(axis=1, ddof=1) * mtc.ANN


# --------------------------------------------------------------------------- the paper's model set
# 2026-10-06 (user decision): the paper has NO headline model and no live-feasible forecasts;
# this diagnostic is re-based on the paper's MODEL SET.  The exact list is the macros agent's
# results/close_master_table/paper_model_set.csv (column key).  Without that file the script
# stops, unless MODELSET_PROVISIONAL=1 (development only): then the set is read off the
# master table's families by the rule below.
MODEL_SET_CSV = mtc.OUT / "paper_model_set.csv"
SET_BUCKETS = ("baseline", "all_features", "lean_vol_vix")
# checked before the shorter names that they contain
BUCKET_NAMES = (
    "lean_vol_vix",
    "moments_vol_vix",
    "moments_vix",
    "all_features",
    "baseline",
    "live_feasible",
)
GROUPS = ("paper", "pooled linear", "per-bar linear", "per-bar trees")
GROUP_COLOURS = {
    "paper": "#1b1b1b",
    "pooled linear": "#2a78d6",
    "per-bar linear": "#1baf7a",
    "per-bar trees": "#eb6834",
}
# families the user excluded from the model set (asserted absent)
EXCLUDED_FAMILY_WORDS = ("LSTM", "VIX-only", "implied-vol", "chain-period", "check")
EXCLUDED_BUCKETS = ("live_feasible", "moments_vix", "moments_vol_vix")


def bucket_of(key: str) -> str:
    """The input set named in a master-table key ('' for the paper's own forecasts)."""
    return next((b for b in BUCKET_NAMES if b in key), "")


def group_of(family: str) -> str:
    """The model-set group of a master-table family."""
    if family == "paper":
        return "paper"
    # 2026-10-06: the pre-tuned pooled rows (poolpt_*) may carry their own family name
    if family == "pooled twin" or family.startswith("pooled"):
        return "pooled linear"
    if family == "per-bar linear":
        return "per-bar linear"
    if family.startswith("per-bar tree"):
        return "per-bar trees"
    return "other"


def model_set(mt: pd.DataFrame) -> tuple[list[str], str]:
    """The model set's keys, in master-table order, and where they came from."""
    if MODEL_SET_CSV.exists():
        keys = pd.read_csv(MODEL_SET_CSV)["key"].astype(str).tolist()
        src = MODEL_SET_CSV.relative_to(ROOT).as_posix()
    elif os.environ.get("MODELSET_PROVISIONAL") == "1":
        a = mt[mt["table"] == "A"]
        grp = a["family"].map(group_of)
        bk = pd.Series([bucket_of(k) for k in a.index], index=a.index)
        keep = (grp == "paper") | (grp.isin(GROUPS) & bk.isin(SET_BUCKETS))
        keys = a.index[keep].tolist()
        if any(
            k.startswith("subpt_") for k in keys
        ):  # the pre-tuned per-bar linear tables
            keys = [k for k in keys if not k.startswith("sub_")]
        src = "PROVISIONAL rule on the master table's families (MODELSET_PROVISIONAL=1)"
    else:
        raise SystemExit(
            f"{MODEL_SET_CSV} is missing (MODELSET_PROVISIONAL=1 for a dry run)"
        )
    assert len(keys) == len(set(keys)), "duplicated keys in the model set"
    missing = [k for k in keys if k not in mt.index]
    assert not missing, f"model-set keys not in the master table: {missing}"
    fam = mt.loc[keys, "family"].astype(str)
    bad = [
        k
        for k in keys
        if any(w in fam[k] for w in EXCLUDED_FAMILY_WORDS)
        or bucket_of(k) in EXCLUDED_BUCKETS
        or group_of(fam[k]) == "other"
    ]
    assert not bad, f"excluded forecasts in the model set: {bad}"
    order = {k: i for i, k in enumerate(mt.index)}
    return sorted(keys, key=order.__getitem__), src


# 2026-10-06: the headline macros, parked (no headline model; user decision).  main() writes
# each one that it does not redefine as a visible ?? stub so the paper still compiles.
HEADLINE_MACROS_PARKED = (
    "hmurHQLIKE",
    "hmurHES",
    "hmurHQLIKERankSet",
    "hmurHESRankSet",
    "hmurHQLIKERankAll",
    "hmurHESRankAll",
    "hmurHSharpeRankAll",
    "hmurNAll",
    "hmurNSet",
    "hmurHHitVar",
    "hmurHHitPay",
    "hmurHTop",
    "hmurOracleAgree",
    "hmurOracleMiss",
    "hmurOracleShMid",
    "hmurOracleShX",
    "hmurPayoffOracleShX",
    "hmurLeaderAtOne",
    "hmurHRankAtOne",
    "hmurNGrid",
    "hmurNLeadH",
    "hmurNLeadLin",
    "hmurGridMin",
    "hmurGridMax",
    "hmurRhoQSet",
    "hmurRhoESSet",
    "hmurRhoQAll",
    "hmurRhoESAll",
    "hsplHAllSh",
    "hsplHAllT",
    "hsplNCovid",
    "hsplHPostCovidN",
    "hsplHPostCovidSh",
    "hsplHPostCovidT",
    "hsplHPreN",
    "hsplHPreSh",
    "hsplHPreT",
    "hsplHPostN",
    "hsplHPostSh",
    "hsplHPostT",
    "hsplRPreSh",
    "hsplRPostSh",
    "hsplNPin",
    "hsplPinPct",
    "hsplHPinMean",
    "hsplHOffPinMean",
    "hsplHPinShare",
    "hsplASOffPinMean",
    "hsplASPinMean",
    "hsplSignBuyMean",
    "hsplSignSellMean",
    "hsplSignDiff",
    "hsplSignT",
    "hsplSignNBuy",
    "hsplSignNSell",
) + PARKED_MACROS


def main_headline_parked() -> None:
    """2026-10-06: PARKED (no headline model; user decision) -- the headline run, kept for
    the record and never called; main() below runs the paper's model set."""
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    def gate(name: str, value: float, bound: float, n: int) -> None:
        gates.append(
            {
                "gate": name,
                "value": float(value),
                "bound": bound,
                "n": n,
                "ok": bool(value <= bound),
            }
        )

    if not DAILY.exists():
        raise SystemExit(
            f"{DAILY} is missing: run experiments/master_table_close.py first"
        )
    D = pd.read_parquet(DAILY)
    D["day"] = pd.to_datetime(D["day"])
    mt = pd.read_csv(MASTER).set_index("key")
    # 2026-10-06: recalibration removed, user decision -- the master table's QLIKE of the
    # plain forecast (was the first qlike*recal* column, the recalibrated forecast's)
    # QLIKE_COL = [
    #     c
    #     for c in mt.columns
    #     if c.lower().startswith("qlike")
    #     and "recal" in c.lower()
    #     and "unclip" not in c.lower()
    #     and "floor" not in c.lower()
    # ][0]
    QLIKE_COL = next(c for c in QLIKE_PLAIN_COLS if c in mt.columns)
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866

    # every table-A forecast on all 866 days, as (d, n) arrays in the deck's day order
    keys = [
        k
        for k in D["key"].unique()
        if (D["key"] == k).sum() == len(days) and k in mt.index
    ]
    frames = {k: D[D["key"] == k].set_index("day").reindex(days) for k in keys}
    # 2026-10-06: recalibration removed, user decision -- FCOL = "plain" (was "pred_clock")
    assert all(f[FCOL].notna().all() for f in frames.values())
    in_set = rank_set(mt)
    assert in_set <= set(keys), sorted(in_set - set(keys))
    assert set(COMPARE) <= in_set, sorted(set(COMPARE) - in_set)
    iset = [i for i, k in enumerate(keys) if k in in_set]
    F = np.vstack([frames[k][FCOL].to_numpy(float) for k in keys])
    y = frames[HEADLINE]["target"].to_numpy(float)
    yu = frames[HEADLINE]["rv_unclipped"].to_numpy(float)
    s = frames[HEADLINE]["iv_var"].to_numpy(float)
    R = deck["R"].to_numpy(float)
    for k in keys:  # the shared target, slice and returns
        gate(
            f"target shared ({k})",
            np.abs(frames[k]["target"].to_numpy(float) / y - 1).max(),
            GATE_REL,
            len(days),
        )
    gate(
        "slice = the deck's iv_var",
        np.abs(s / deck["iv_var"].to_numpy(float) - 1).max(),
        GATE_REL,
        len(days),
    )
    ex = deck["exit"].to_numpy(float)
    ask = (deck["ask_c"] + deck["ask_p"]).to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    cl, cs = ex / ask - 1.0, ex / bid - 1.0

    def pnl(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.atleast_2d(q)
        return a * R, a * np.where(a > 0, cl, cs)

    Q = np.where(F > s, 1.0, -1.0)
    MID, CR = pnl(Q)
    for i, k in enumerate(keys):  # the master table's own positions and returns
        gate(
            f"q = master table ({k})",
            np.abs(Q[i] - frames[k]["q"].to_numpy(float)).max(),
            0,
            len(days),
        )
        gate(
            f"ret_mid = master table ({k})",
            np.abs(MID[i] - frames[k]["ret_mid"].to_numpy(float)).max(),
            GATE_ABS,
            len(days),
        )
        gate(
            f"ret_crossed = master table ({k})",
            np.abs(CR[i] - frames[k]["ret_crossed"].to_numpy(float)).max(),
            GATE_ABS,
            len(days),
        )
    ih = keys.index(HEADLINE)
    gate(
        "headline Sharpe mid = master_table.csv",
        abs(mtc.sharpe(MID[ih]) / float(mt.loc[HEADLINE, "Sharpe_mid"]) - 1),
        GATE_REL,
        len(days),
    )
    gate(
        "headline Sharpe crossed = master_table.csv",
        abs(mtc.sharpe(CR[ih]) / float(mt.loc[HEADLINE, "Sharpe_crossed"]) - 1),
        GATE_REL,
        len(days),
    )
    gate(
        f"headline Sharpe mid = {PLAIN_HEADLINE_SHARPE_MID} to two decimals (the GUI's recomputation)",
        float(f"{mtc.sharpe(MID[ih]):.2f}" != PLAIN_HEADLINE_SHARPE_MID),
        0,
        len(days),
    )

    # ---------------------------------------------------------------- part 1: the scores
    QL = qlike(y[None, :], F)
    gate(
        # 2026-10-06: recalibration removed -- was "headline QLIKE (recal) = master_table.csv"
        f"headline QLIKE (plain) = master_table.csv {QLIKE_COL}",
        abs(QL[ih].mean() / float(mt.loc[HEADLINE, QLIKE_COL]) - 1),
        GATE_REL,
        len(days),
    )
    ES1 = elementary(F / s, (y / s)[None, :], 1.0)
    sgnR = np.where(R > 0, 1.0, -1.0)
    EL = np.abs(R)[None, :] * (Q != sgnR)
    gate(
        "mean(qR) = mean|R| - 2 mean(EL), every forecast",
        np.abs(MID.mean(axis=1) - (np.abs(R).mean() - 2 * EL.mean(axis=1))).max(),
        1e-12,
        len(keys),
    )
    # G3: QLIKE equals the c^-2 mixture of elementary scores, day by day (headline)
    fs, ys = F[ih] / s, y / s
    lo, hi = np.minimum(fs, ys), np.maximum(fs, ys)
    u = np.linspace(0.0, 1.0, G3_GRID)
    # c = lo[:, None] * (hi / lo)[:, None] ** u[None, :]
    # integral = np.trapezoid(np.abs(ys[:, None] - c) / c**2, c, axis=1)
    integral = np.empty(len(lo))
    for a in range(0, len(lo), G3_CHUNK):
        sl = slice(a, a + G3_CHUNK)
        c = lo[sl, None] * (hi[sl] / lo[sl])[:, None] ** u[None, :]
        integral[sl] = np.trapezoid(np.abs(ys[sl, None] - c) / c**2, c, axis=1)
    gate(
        "QLIKE = integral ES_c c^-2 dc, day by day (headline)",
        np.abs(integral - QL[ih]).max(),
        1e-6,
        len(days),
    )
    hit_var = (F > s) == (y > s)[None, :]
    hit_pay = Q == sgnR[None, :]
    order = np.argsort(R)[::-1]
    rows = []
    for i, k in enumerate(keys):
        rows.append(
            {
                "key": k,
                "label": mt.loc[k, "label"],
                "in_comparison_set": k in COMPARE,
                "in_rank_set": k in in_set,
                "QLIKE": QL[i].mean(),
                "ES_at_threshold": ES1[i].mean(),
                "ES_at_threshold_unclipped": elementary(F[i] / s, yu / s, 1.0).mean(),
                "EL": EL[i].mean(),
                "hit_var_pct": 100 * hit_var[i].mean(),
                "hit_pay_pct": 100 * hit_pay[i].mean(),
                **{
                    f"long_on_top{kk}": int((Q[i, order[:kk]] > 0).sum())
                    for kk in TOP_K
                },
                "pct_buy": 100 * (Q[i] > 0).mean(),
                "Sharpe_mid": mtc.sharpe(MID[i]),
                "Sharpe_crossed": mtc.sharpe(CR[i]),
            }
        )
    # the oracles
    q_var = np.where(y > s, 1.0, -1.0)
    q_pay = sgnR.copy()
    for name, q in (
        ("oracle: told RV (sign(RV - slice))", q_var),
        ("oracle: told the payoff's sign", q_pay),
    ):
        m, cr = pnl(q)
        rows.append(
            {
                "key": name,
                "label": name,
                "in_comparison_set": True,
                "in_rank_set": False,
                "QLIKE": np.nan,
                "ES_at_threshold": np.nan,
                "ES_at_threshold_unclipped": np.nan,
                "EL": (np.abs(R) * (q != sgnR)).mean(),
                "hit_var_pct": 100 * ((q > 0) == (y > s)).mean(),
                "hit_pay_pct": 100 * (q == sgnR).mean(),
                **{f"long_on_top{kk}": int((q[order[:kk]] > 0).sum()) for kk in TOP_K},
                "pct_buy": 100 * (q > 0).mean(),
                "Sharpe_mid": mtc.sharpe(m[0]),
                "Sharpe_crossed": mtc.sharpe(cr[0]),
            }
        )
    scores = pd.DataFrame(rows).set_index("key")
    # "overall" ranks over the rank set only (the oracles, check rows and duplicates are not ranked)
    rs = scores[scores["in_rank_set"]]
    for col, v in (
        ("QLIKE_rank_all", rs["QLIKE"]),
        ("ES_rank_all", rs["ES_at_threshold"]),
        ("Sharpe_crossed_rank_all", -rs["Sharpe_crossed"]),
    ):
        scores[col] = np.nan
        scores.loc[rs.index, col] = v.rank(method="min")
    sub = scores.loc[COMPARE]
    scores["QLIKE_rank_set"] = np.nan
    scores.loc[COMPARE, "QLIKE_rank_set"] = sub["QLIKE"].rank(method="min")
    scores["ES_rank_set"] = np.nan
    scores.loc[COMPARE, "ES_rank_set"] = sub["ES_at_threshold"].rank(method="min")
    scores.to_csv(OUT / "scores_by_forecast.csv")

    # the Murphy diagram: mean ES_c, one row per forecast in the comparison set, and the headline/reference on the unclipped y
    ic = [keys.index(k) for k in COMPARE]
    grid = pd.DataFrame(
        {
            f"c={c_:.4f}": [elementary(F[i] / s, y / s, c_).mean() for i in ic]
            for c_ in MURPHY_C
        },
        index=[mt.loc[k, "label"] for k in COMPARE],
    )
    grid.index.name = "forecast"
    grid.to_csv(OUT / "murphy_grid.csv")
    # which forecast leads at each threshold, and the headline's rank
    lead = grid.idxmin(axis=0)
    hl = mt.loc[HEADLINE, "label"]
    rank_h = grid.rank(axis=0).loc[hl]
    murphy_lead = pd.DataFrame(
        {
            "c": MURPHY_C,
            "leader": lead.to_numpy(),
            "headline_rank_of_set": rank_h.to_numpy(),
            "headline_score": grid.loc[hl].to_numpy(),
            "best_score": grid.min(axis=0).to_numpy(),
        }
    )
    murphy_lead.to_csv(OUT / "murphy_leader_by_threshold.csv", index=False)

    # rank agreement of minus each loss with the crossed Sharpe: the comparison set and every table-A forecast
    rk_rows = []
    for setname, idxs in (
        ("comparison set", ic),
        ("all table-A forecasts", iset),
    ):
        for lname, L in (
            ("QLIKE", QL),
            ("ES at the threshold", ES1),
            ("EL (affine in the mid P&L)", EL),
        ):
            # 2026-10-06: no intervals (bootstrap disabled, commit b761b28): point only
            # pt, lo_, hi_ = spearman_ci(L[idxs], CR[idxs])
            pt = spearman_point(L[idxs], CR[idxs])
            rk_rows.append(
                {
                    "set": setname,
                    "n_forecasts": len(idxs),
                    "loss": lname,
                    "spearman_minus_loss_vs_crossed_sharpe": pt,
                    # "lo": lo_,
                    # "hi": hi_,
                }
            )
    ranks = pd.DataFrame(rk_rows)
    ranks.to_csv(OUT / "rank_vs_sharpe.csv", index=False)

    # ---------------------------------------------------------------- part 2: sample splits
    ir = keys.index(REFERENCE)
    series = {
        "headline sign(s)": MID[ih],
        "reference sign(s) (this scorer)": MID[ir],
        "always short": -R,
    }
    series_cr = {
        "headline sign(s)": CR[ih],
        "reference sign(s) (this scorer)": CR[ir],
        "always short": cs * -1.0 * -1.0 * 0 + (1.0 - ex / bid),
    }
    covid = days <= COVID_END
    pre = days < ERA_CUT
    pin = R <= -1 + 1e-12
    gate(
        "always short collects +1 on pin days",
        abs((-R)[pin].mean() - 1.0),
        1e-9,
        int(pin.sum()),
    )
    sp_rows = []
    for name, x in series.items():
        xc = series_cr[name]
        for split, mask in (
            ("all days", np.ones(len(days), bool)),
            ("through 2020-03-31 (the COVID quarter)", covid),
            ("after 2020-03-31", ~covid),
            ("before 2022-05-16", pre),
            ("from 2022-05-16 (daily SPXW listings)", ~pre),
            ("settlement pins (R = -1)", pin),
            ("off pins", ~pin),
        ):
            n, t, sh = stats(x[mask])
            # 2026-10-06: no intervals (bootstrap disabled, commit b761b28)
            # lo_, hi_ = sharpe_ci(x[mask]) if mask.sum() >= 30 else (np.nan, np.nan)
            nc, tc, shc = stats(xc[mask])
            sp_rows.append(
                {
                    "series": name,
                    "split": split,
                    "n": n,
                    "mean_mid": float(x[mask].mean()),
                    "t_mid": t,
                    "Sharpe_mid": sh,
                    # "Sharpe_mid_lo": lo_,
                    # "Sharpe_mid_hi": hi_,
                    "share_of_total_pnl_mid_pct": 100 * x[mask].sum() / x.sum(),
                    "mean_crossed": float(xc[mask].mean()),
                    "t_crossed": tc,
                    "Sharpe_crossed": shc,
                }
            )
    splits = pd.DataFrame(sp_rows)
    splits.to_csv(OUT / "splits.csv", index=False)
    # the sign split of the settlement return by the headline's and the reference's signal
    sg_rows = []
    for name, i in (("headline", ih), ("reference (this scorer)", ir)):
        ind = (Q[i] > 0).astype(float)
        fit = sm.OLS(R, sm.add_constant(ind)).fit(cov_type="HC0")
        sg_rows.append(
            {
                "forecast": name,
                "mean_R_when_s_le_0": float(R[ind == 0].mean()),
                "mean_R_when_s_gt_0": float(R[ind == 1].mean()),
                "diff": float(fit.params[1]),
                "t_HC0": float(fit.tvalues[1]),
                "n_sell": int((ind == 0).sum()),
                "n_buy": int((ind == 1).sum()),
            }
        )
    signsplit = pd.DataFrame(sg_rows)
    signsplit.to_csv(OUT / "sign_split.csv", index=False)

    # ---- the committed copies
    # 2026-10-06: recalibration removed, user decision -- the committed copies are the
    # recalibrated run, so the rows that read a forecast (the headline, the reference) are
    # no longer asserted equal; they are compared for the record (`reported`, SUMMARY.md).
    # The rows that read no forecast are still identities and stay asserted: the two oracle
    # rules of scores_by_forecast.csv and always short's rows of splits.csv (point columns;
    # the committed interval columns are gone).
    # for fname, idx, keep in (
    #     ("splits.csv", ["series", "split"], None),
    #     ("sign_split.csv", ["forecast"], None),
    #     ("scores_by_forecast.csv", ["key"], [HEADLINE, REFERENCE, *ORACLES]),
    #     ("murphy_grid.csv", ["forecast"], [hl, rl]),
    # ):
    #     ... gate(f"{fname} = the committed copy ...", committed_diff(new, old), COMMITTED_TOL, ...)
    hl, rl = mt.loc[HEADLINE, "label"], mt.loc[REFERENCE, "label"]
    reported: list[dict] = []
    for fname, idx, keep in (
        ("splits.csv", ["series", "split"], None),
        ("sign_split.csv", ["forecast"], None),
        ("scores_by_forecast.csv", ["key"], [HEADLINE, REFERENCE]),
        ("murphy_grid.csv", ["forecast"], [hl, rl]),
    ):
        old, sha = committed_csv(OUT / fname)
        new = pd.read_csv(OUT / fname)
        old, new = old.set_index(idx), new.set_index(idx)
        if keep is not None:
            old = old.loc[
                old.index.isin(keep), [c for c in old.columns if c not in RANK_COLS]
            ]
        reported.append(
            {
                "file": fname,
                "rows": "every row" if keep is None else f"{len(old)} rows",
                "max_rel_diff": common_diff(new, old),
                "n_cells": int(old.size),
            }
        )
    for fname, idx, col, keep in (
        ("scores_by_forecast.csv", ["key"], None, ORACLES),
        ("splits.csv", ["series", "split"], "series", ["always short"]),
    ):
        old, sha = committed_csv(OUT / fname)
        new = pd.read_csv(OUT / fname)
        if col is not None:
            old, new = old[old[col].isin(keep)], new[new[col].isin(keep)]
        old, new = old.set_index(idx), new.set_index(idx)
        if col is None:
            old = old.loc[old.index.isin(keep)]
        old = old[[c for c in old.columns if c not in RANK_COLS and c in new.columns]]
        gate(
            f"{fname} = the committed copy ({PREV_REV} = {sha}), {len(old)} rows that read "
            "no forecast (ranks and intervals left out), max |new - old| / max(1, |old|)",
            committed_diff(new, old),
            COMMITTED_TOL,
            int(old.size),
        )
    # every other forecast present in both runs, compared for the record (not asserted)
    old_sc, prev_sha = committed_csv(OUT / "scores_by_forecast.csv", index_col=0)
    num = [
        c
        for c in old_sc.columns
        if c not in RANK_COLS and pd.api.types.is_numeric_dtype(old_sc[c])
    ]
    ba_rows = []
    for k in scores.index:
        if k not in old_sc.index:
            ba_rows.append({"key": k, "label": scores.loc[k, "label"], "status": "new"})
            continue
        o, n_ = old_sc.loc[k, num].astype(float), scores.loc[k, num].astype(float)
        rel = (n_ - o).abs() / np.maximum(1.0, o.abs())
        ba_rows.append(
            {
                "key": k,
                "label": scores.loc[k, "label"],
                "status": "in both",
                # 2026-10-06: only the oracle rules are still asserted unchanged
                # "gated": k in (HEADLINE, REFERENCE, *ORACLES),
                "gated": k in ORACLES,
                "in_comparison_set_before": bool(old_sc.loc[k, "in_comparison_set"]),
                "in_comparison_set_after": bool(scores.loc[k, "in_comparison_set"]),
                **{f"{c}_before": o[c] for c in ("QLIKE", "ES_at_threshold", "EL")},
                **{f"{c}_after": n_[c] for c in ("QLIKE", "ES_at_threshold", "EL")},
                "Sharpe_crossed_before": o["Sharpe_crossed"],
                "Sharpe_crossed_after": n_["Sharpe_crossed"],
                "max_rel_diff": float(rel.fillna(0.0).max()),
            }
        )
    for k in old_sc.index.difference(scores.index):
        ba_rows.append({"key": k, "label": old_sc.loc[k, "label"], "status": "dropped"})
    before_after = pd.DataFrame(ba_rows)
    before_after.to_csv(OUT / "before_after_committed.csv", index=False)
    old_rk, _ = committed_csv(OUT / "rank_vs_sharpe.csv")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False)
    assert all(g["ok"] for g in gates), [g for g in gates if not g["ok"]]

    # ---------------------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # eleven lines: the headline black and thick, the other ten one tab10 colour each;
    # line style by model class (linear solid, tree dashed, LSTM dash-dot); legend below
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    colours = iter(f"C{i}" for i in range(10))  # the tab10 colours
    for k, (lab, row) in zip(COMPARE, grid.iterrows()):
        head = k == HEADLINE
        style = {
            "color": "k" if head else next(colours),
            "lw": 2.4 if head else 1.1,
            "ls": "-"
            if (k.startswith("sub_") or k in (REFERENCE, "a0"))
            else ("--" if k.startswith("subtree_") else "-."),
            "zorder": 3 if head else 2,
        }
        axes[0].plot(MURPHY_C, row.to_numpy(), label=lab, **style)
        axes[1].plot(
            MURPHY_C, row.to_numpy() - grid.mean(axis=0).to_numpy(), label=lab, **style
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.axvline(1.0, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel("threshold c, in units of the 15:30 implied variance")
    axes[0].set_ylabel("mean elementary score ES_c (lower is better)")
    axes[1].set_ylabel("ES_c minus the mean of the set at c")
    axes[0].set_title(
        # 2026-10-06: recalibration removed, user decision (was "16:00-bar recalibration")
        "(a) Murphy diagram, 866 deck days, plain forecast (no recalibration)",
        fontsize=9,
    )
    axes[1].set_title("(b) each line minus the mean of the set", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    fig.savefig(OUT / "murphy_headline.png", dpi=130, bbox_inches="tight")

    # ---------------------------------------------------------------- tex: tables + macros
    def t3(v: float) -> str:
        return f"{v:.3f}"

    # 2026-10-06: recalibration removed (user decision) and no intervals (bootstrap
    # disabled, commit b761b28): captions say "plain forecast", the splits table carries no
    # Sharpe interval, and the interval macros are parked (?? stubs, PARKED_MACROS)
    L = [
        "% AUTO-GENERATED by experiments/close_murphy_splits_headline.py from results/close_murphy_splits/*.csv -- do not edit.",
        "% 2026-10-06: plain 16:00 forecast f^2 B (recalibration removed, user decision); point estimates only (bootstrap disabled, b761b28).",
    ]
    L += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{p{0.27\textwidth}rrrrrrr}",
        r"\caption{The comparison set scored two ways on the 866 deck days, each forecast read as its plain forecast $\hat y^2B$ (no recalibration): QLIKE against the scorer's target, "
        r"the elementary score at the trade's threshold (the 15:30 implied variance), the economic loss $|R|\,\mathbf 1\{q\neq\mathrm{sign}(R)\}$, "
        r"the share of days the forecast falls on the same side of the threshold as the realized variance (same side as $RV$) and the share on which the "
        r"position has the sign of the settlement return (same side as $R$), and the $\mathrm{sign}(s)$ "
        r"Sharpe at both fills. The last two rows are rules told the realized variance, and the payoff's sign, in advance.}\label{tab:app_murphy_headline}\\",
        r"\toprule",
        r"forecast & QLIKE & ES($c{=}1$) & EL & side of $RV$ \% & side of $R$ \% & Sharpe mid & Sharpe crossed \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"forecast & QLIKE & ES($c{=}1$) & EL & side of $RV$ \% & side of $R$ \% & Sharpe mid & Sharpe crossed \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for k in COMPARE + [
        "oracle: told RV (sign(RV - slice))",
        "oracle: told the payoff's sign",
    ]:
        r_ = scores.loc[k]
        lab = str(r_["label"]).replace("_", r"\_").replace("&", r"\&")
        if k == HEADLINE:
            lab = r"\textbf{" + lab + "}"
        ql_ = "--" if np.isnan(r_["QLIKE"]) else t3(r_["QLIKE"])
        es_ = "--" if np.isnan(r_["ES_at_threshold"]) else t3(r_["ES_at_threshold"])
        L.append(
            f"\\raggedright {lab} & {ql_} & {es_} & {t3(r_['EL'])} & {r_['hit_var_pct']:.1f} & {r_['hit_pay_pct']:.1f} & "
            f"{r_['Sharpe_mid']:.2f} & {r_['Sharpe_crossed']:.2f} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]
    # splits table
    L += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{llrrrrr}",
        r"\caption{Sample splits of the headline's $\mathrm{sign}(s)$ return (midpoint fill) on its plain forecast $\hat y^2B$ (no recalibration), beside the paper's "
        r"block-diagonal ridge scored the same way and always short. $t=\sqrt{n}\,\bar x/s_x$; the Sharpe ratio is annualized (point estimate); "
        r"the P\&L share is the split's sum of daily returns over the full-sample sum.}"
        r"\label{tab:app_splits_headline}\\",
        r"\toprule",
        r"series & split & $n$ & mean & $t$ & Sharpe & P\&L share \% \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"series & split & $n$ & mean & $t$ & Sharpe & P\&L share \% \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r_ in splits.iterrows():
        # 2026-10-06: no intervals -- the "[lo, hi]" after the Sharpe ratio is parked
        # ci = (
        #     f"[{r_['Sharpe_mid_lo']:+.2f}, {r_['Sharpe_mid_hi']:+.2f}]"
        #     if np.isfinite(r_["Sharpe_mid_lo"])
        #     else ""
        # )
        L.append(
            f"{r_['series']} & {r_['split']} & {int(r_['n'])} & {r_['mean_mid']:+.3f} & {fm(r_['t_mid'], '+.2f')} & {fm(r_['Sharpe_mid'], '.2f')} & "
            f"{r_['share_of_total_pnl_mid_pct']:.0f} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]

    def sp(series_name: str, split_name: str) -> pd.Series:
        return splits[
            (splits["series"] == series_name) & (splits["split"] == split_name)
        ].iloc[0]

    h = scores.loc[HEADLINE]
    ov, op = (
        scores.loc["oracle: told RV (sign(RV - slice))"],
        scores.loc["oracle: told the payoff's sign"],
    )
    rk = ranks.set_index(["set", "loss"])
    lead_at_1 = murphy_lead.loc[np.isclose(murphy_lead["c"], 1.0)].iloc[0]
    n_lead_h = int((murphy_lead["leader"] == hl).sum())
    lin_labels = {
        mt.loc[k, "label"]
        for k in COMPARE
        if k.startswith("sub_") or k in ("blk2", "a0")
    }
    n_lead_lin = int(murphy_lead["leader"].isin(lin_labels).sum())
    sg = signsplit.set_index("forecast").loc["headline"]
    ha, hc, hb, hd = (
        sp("headline sign(s)", "all days"),
        sp("headline sign(s)", "after 2020-03-31"),
        sp("headline sign(s)", "before 2022-05-16"),
        sp("headline sign(s)", "from 2022-05-16 (daily SPXW listings)"),
    )
    hp, ho = (
        sp("headline sign(s)", "settlement pins (R = -1)"),
        sp("headline sign(s)", "off pins"),
    )
    ap_, ao = (
        sp("always short", "settlement pins (R = -1)"),
        sp("always short", "off pins"),
    )
    rb, rd = (
        sp("reference sign(s) (this scorer)", "before 2022-05-16"),
        sp("reference sign(s) (this scorer)", "from 2022-05-16 (daily SPXW listings)"),
    )
    macros = {
        "hmurHQLIKE": t3(h["QLIKE"]),
        "hmurHES": t3(h["ES_at_threshold"]),
        "hmurHQLIKERankSet": f"{int(h['QLIKE_rank_set'])}",
        "hmurHESRankSet": f"{int(h['ES_rank_set'])}",
        "hmurHQLIKERankAll": f"{int(h['QLIKE_rank_all'])}",
        "hmurHESRankAll": f"{int(h['ES_rank_all'])}",
        "hmurHSharpeRankAll": f"{int(h['Sharpe_crossed_rank_all'])}",
        "hmurNAll": f"{len(iset)}",
        "hmurNSet": f"{len(COMPARE)}",
        "hmurHHitVar": f"{h['hit_var_pct']:.0f}",
        "hmurHHitPay": f"{h['hit_pay_pct']:.0f}",
        "hmurHTop": f"{int(h['long_on_top20'])}",
        "hmurOracleAgree": f"{ov['hit_pay_pct']:.0f}",
        "hmurOracleMiss": f"{100 - ov['hit_pay_pct']:.0f}",
        "hmurOracleShMid": f"{ov['Sharpe_mid']:.2f}",
        "hmurOracleShX": f"{ov['Sharpe_crossed']:.2f}",
        "hmurPayoffOracleShX": f"{op['Sharpe_crossed']:.2f}",
        "hmurLeaderAtOne": str(lead_at_1["leader"]).replace("_", r"\_"),
        "hmurHRankAtOne": f"{int(lead_at_1['headline_rank_of_set'])}",
        "hmurNGrid": f"{len(MURPHY_C)}",
        "hmurNLeadH": f"{n_lead_h}",
        "hmurNLeadLin": f"{n_lead_lin}",
        "hmurGridMin": f"{MURPHY_C.min():.2f}",
        "hmurGridMax": f"{MURPHY_C.max():.0f}",
        "hmurRhoQSet": f"{rk.loc[('comparison set', 'QLIKE'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        # 2026-10-06: no intervals (bootstrap disabled, commit b761b28) -- the interval
        # macros below are parked (PARKED_MACROS, written as ?? stubs)
        # "hmurRhoQSetLo": f"{rk.loc[('comparison set', 'QLIKE'), 'lo']:+.2f}",
        # "hmurRhoQSetHi": f"{rk.loc[('comparison set', 'QLIKE'), 'hi']:+.2f}",
        "hmurRhoESSet": f"{rk.loc[('comparison set', 'ES at the threshold'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        # "hmurRhoESSetLo": f"{rk.loc[('comparison set', 'ES at the threshold'), 'lo']:+.2f}",
        # "hmurRhoESSetHi": f"{rk.loc[('comparison set', 'ES at the threshold'), 'hi']:+.2f}",
        "hmurRhoQAll": f"{rk.loc[('all table-A forecasts', 'QLIKE'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        # "hmurRhoQAllLo": f"{rk.loc[('all table-A forecasts', 'QLIKE'), 'lo']:+.2f}",
        # "hmurRhoQAllHi": f"{rk.loc[('all table-A forecasts', 'QLIKE'), 'hi']:+.2f}",
        "hmurRhoESAll": f"{rk.loc[('all table-A forecasts', 'ES at the threshold'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        # "hmurRhoESAllLo": f"{rk.loc[('all table-A forecasts', 'ES at the threshold'), 'lo']:+.2f}",
        # "hmurRhoESAllHi": f"{rk.loc[('all table-A forecasts', 'ES at the threshold'), 'hi']:+.2f}",
        "hsplHAllSh": f"{ha['Sharpe_mid']:.2f}",
        "hsplHAllT": f"{ha['t_mid']:.2f}",
        "hsplNCovid": f"{int(sp('headline sign(s)', 'through 2020-03-31 (the COVID quarter)')['n'])}",
        "hsplHPostCovidN": f"{int(hc['n'])}",
        "hsplHPostCovidSh": f"{hc['Sharpe_mid']:.2f}",
        "hsplHPostCovidT": f"{hc['t_mid']:.2f}",
        # "hsplHPostCovidLo": f"{hc['Sharpe_mid_lo']:.2f}",
        # "hsplHPostCovidHi": f"{hc['Sharpe_mid_hi']:.2f}",
        "hsplHPreN": f"{int(hb['n'])}",
        "hsplHPreSh": f"{hb['Sharpe_mid']:.2f}",
        "hsplHPreT": f"{hb['t_mid']:.2f}",
        # "hsplHPreLo": f"{hb['Sharpe_mid_lo']:.2f}",
        # "hsplHPreHi": f"{hb['Sharpe_mid_hi']:.2f}",
        "hsplHPostN": f"{int(hd['n'])}",
        "hsplHPostSh": f"{hd['Sharpe_mid']:.2f}",
        "hsplHPostT": f"{hd['t_mid']:.2f}",
        # "hsplHPostLo": f"{hd['Sharpe_mid_lo']:.2f}",
        # "hsplHPostHi": f"{hd['Sharpe_mid_hi']:.2f}",
        "hsplRPreSh": f"{rb['Sharpe_mid']:.2f}",
        "hsplRPostSh": f"{rd['Sharpe_mid']:.2f}",
        "hsplNPin": f"{int(hp['n'])}",
        "hsplPinPct": f"{100 * pin.mean():.1f}",
        "hsplHPinMean": f"{hp['mean_mid']:+.2f}",
        "hsplHOffPinMean": f"{ho['mean_mid']:+.2f}",
        "hsplHPinShare": f"{hp['share_of_total_pnl_mid_pct']:.0f}",
        "hsplASOffPinMean": f"{ao['mean_mid']:+.2f}",
        "hsplASPinMean": f"{ap_['mean_mid']:+.2f}",
        "hsplSignBuyMean": f"{sg['mean_R_when_s_gt_0']:+.2f}",
        "hsplSignSellMean": f"{sg['mean_R_when_s_le_0']:+.2f}",
        "hsplSignDiff": f"{sg['diff']:.2f}",
        "hsplSignT": f"{sg['t_HC0']:.2f}",
        "hsplSignNBuy": f"{int(sg['n_buy'])}",
        "hsplSignNSell": f"{int(sg['n_sell'])}",
    }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    L += [
        "% 2026-10-06: interval macros removed (bootstrap disabled, commit b761b28; point "
        "estimates only). Each is a visible ?? stub until sections/appendix_close_murphy_splits.tex "
        "stops using it.",
    ]
    L += [f"\\newcommand{{\\{k}}}{{\\textbf{{??}}}}" for k in PARKED_MACROS]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---------------------------------------------------------------- SUMMARY.md from the CSVs
    # 2026-10-06: recalibration removed (user decision) and no intervals (bootstrap disabled,
    # commit b761b28): "plain forecast" throughout, point estimates only, and the committed
    # (recalibrated) run set beside this one for the record
    S = [
        "# Murphy / threshold analysis and sample splits of the headline forecast (plain 16:00 forecast, no recalibration)",
        "",
        "Written by `experiments/close_murphy_splits_headline.py` from the CSVs in this folder (`scores_by_forecast.csv`, `murphy_grid.csv`, "
        "`murphy_leader_by_threshold.csv`, `rank_vs_sharpe.csv`, `splits.csv`, `sign_split.csv`, `gates.csv`). Input: the master table's per-day frame "
        "(`results/close_master_table/master_table_daily.parquet`, regenerated by `experiments/master_table_close.py`); the headline's Sharpe, QLIKE, positions "
        "and returns are gated against `master_table.csv`. Forecast: every forecast is its plain back-transform f²·B (column `plain`; no recalibration, "
        "2026-10-06 user decision); the deck's 15:30 sign(s) straddle trade q = sign(f²·B − implied variance); 866 days 2020-01-03 .. 2024-04-30. "
        "Point estimates only (the bootstrap is disabled, commit b761b28). The parked versions of these diagnostics (Appendix D.15) are on the deck's "
        "session-bar recalibration and are not set beside these.",
        "",
        f"## Part 1 — decision-aligned scores (comparison set of {len(COMPARE)}; all {len(keys)} table-A forecasts in `scores_by_forecast.csv`, "
        f"ranked over the master table's rank set of {len(iset)}: check rows, exact duplicates and always short left out, `in_rank_set`)",
        "",
        "| forecast | QLIKE | ES(c=1) | EL | same side as RV % | same side as R % | long on top 10/20/50 | buy % | Sharpe mid | Sharpe crossed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in COMPARE + [
        "oracle: told RV (sign(RV - slice))",
        "oracle: told the payoff's sign",
    ]:
        r_ = scores.loc[k]
        S.append(
            f"| {r_['label']} | {r_['QLIKE']:.4f} | {r_['ES_at_threshold']:.4f} | {r_['EL']:.4f} | {r_['hit_var_pct']:.1f} | {r_['hit_pay_pct']:.1f} | "
            f"{int(r_['long_on_top10'])}/{int(r_['long_on_top20'])}/{int(r_['long_on_top50'])} | {r_['pct_buy']:.0f} | {r_['Sharpe_mid']:.2f} | {r_['Sharpe_crossed']:.2f} |"
        )
    S += [
        "",
        f"- Headline ranks: QLIKE {int(h['QLIKE_rank_set'])} of {len(COMPARE)} in the set ({int(h['QLIKE_rank_all'])} of {len(iset)} overall); ES at the threshold "
        f"{int(h['ES_rank_set'])} of {len(COMPARE)} ({int(h['ES_rank_all'])} overall); crossed Sharpe {int(h['Sharpe_crossed_rank_all'])} of {len(iset)} overall.",
        f"- Murphy diagram (`murphy_grid.csv`, c = 2^(k/8) from {MURPHY_C.min():.2f} to {MURPHY_C.max():.0f}): the headline has the lowest score of the set at {n_lead_h} of {len(MURPHY_C)} thresholds; "
        f"a linear forecast leads at {n_lead_lin}; at c = 1 the leader is {lead_at_1['leader']} and the headline ranks {int(lead_at_1['headline_rank_of_set'])} of {len(COMPARE)}.",
        f"- Oracles: a rule told RV takes the payoff's side on {ov['hit_pay_pct']:.1f} % of days (Sharpe {ov['Sharpe_mid']:.2f} mid / {ov['Sharpe_crossed']:.2f} crossed); "
        f"a rule told the payoff's sign: {op['Sharpe_crossed']:.2f} crossed. The headline takes the payoff's side on {h['hit_pay_pct']:.1f} % and the side of RV − slice on {h['hit_var_pct']:.1f} %.",
        "- Rank agreement of minus each loss with the crossed Sharpe (Spearman, point estimate):",
    ]
    for _, r_ in ranks.iterrows():
        S.append(
            f"  - {r_['set']} (n = {int(r_['n_forecasts'])}), {r_['loss']}: {r_['spearman_minus_loss_vs_crossed_sharpe']:+.2f}"
        )
    S += [
        "",
        f"## Before / after: this run against the committed outputs ({PREV_REV} = {prev_sha}, the 16:00-bar recalibration (f² + s)·B; reported, not asserted)",
        "",
        f"- Comparison set: {len(old_sc[old_sc['in_comparison_set'].astype(bool) & ~old_sc.index.isin(ORACLES)])} -> {len(COMPARE)} forecasts.",
        f"- Forecasts in `scores_by_forecast.csv`: {int((~old_sc.index.isin(ORACLES)).sum())} -> {len(keys)} "
        f"({int((before_after['status'] == 'new').sum())} new, {int((before_after['status'] == 'dropped').sum())} dropped); rank set {len(iset)}.",
        "- Still asserted unchanged (the rows that read no forecast: the two oracle rules, always short's splits; `gates.csv`): max relative difference "
        + f"{before_after.loc[before_after['gated'].eq(True), 'max_rel_diff'].max():.2g}.",
        "- Reported, not asserted (max |new − old| / max(1, |old|) over the cells both runs carry): "
        + "; ".join(
            f"`{r_['file']}` ({r_['rows']}) {r_['max_rel_diff']:.3g}" for r_ in reported
        )
        + ".",
        "- Comparison-set members, before -> after (`before_after_committed.csv`):",
    ]
    ba = before_after[before_after["status"] == "in both"].set_index("key")
    for k in COMPARE:
        if k in ba.index and not bool(ba.loc[k, "gated"]):
            b_ = ba.loc[k]
            S.append(
                f"  - {b_['label']}: QLIKE {b_['QLIKE_before']:.4f} -> {b_['QLIKE_after']:.4f}, ES(c=1) {b_['ES_at_threshold_before']:.4f} -> "
                f"{b_['ES_at_threshold_after']:.4f}, Sharpe crossed {b_['Sharpe_crossed_before']:.2f} -> {b_['Sharpe_crossed_after']:.2f} "
                f"(max relative difference {b_['max_rel_diff']:.2g})"
            )
    S += [
        "- Rank agreement, before [committed interval] -> after (Spearman, point):",
    ]
    ork = old_rk.set_index(["set", "loss"])
    for _, r_ in ranks.iterrows():
        o_ = ork.loc[(r_["set"], r_["loss"])]
        S.append(
            f"  - {r_['set']}, {r_['loss']}: n {int(o_['n_forecasts'])} -> {int(r_['n_forecasts'])}; "
            f"{o_['spearman_minus_loss_vs_crossed_sharpe']:+.2f} [{o_['lo']:+.2f}, {o_['hi']:+.2f}] -> "
            f"{r_['spearman_minus_loss_vs_crossed_sharpe']:+.2f}"
        )
    old_sp, _ = committed_csv(OUT / "splits.csv")
    old_sp = old_sp.set_index(["series", "split"])
    old_sg, _ = committed_csv(OUT / "sign_split.csv")
    old_sg = old_sg.set_index("forecast")
    S += ["- Headline splits, Sharpe mid (t) before -> after:"]
    for _, r_ in splits[splits["series"] == "headline sign(s)"].iterrows():
        o_ = old_sp.loc[(r_["series"], r_["split"])]
        S.append(
            f"  - {r_['split']}: {fm(o_['Sharpe_mid'], '.2f')} ({fm(o_['t_mid'], '+.2f')}) -> "
            f"{fm(r_['Sharpe_mid'], '.2f')} ({fm(r_['t_mid'], '+.2f')}), n {int(o_['n'])} -> {int(r_['n'])}"
        )
    og = old_sg.loc["headline"]
    S.append(
        f"- Headline sign split: buy {og['mean_R_when_s_gt_0']:+.3f} / sell {og['mean_R_when_s_le_0']:+.3f} (t {og['t_HC0']:.2f}, "
        f"{int(og['n_sell'])} / {int(og['n_buy'])}) -> buy {sg['mean_R_when_s_gt_0']:+.3f} / sell {sg['mean_R_when_s_le_0']:+.3f} "
        f"(t {sg['t_HC0']:.2f}, {int(sg['n_sell'])} / {int(sg['n_buy'])})."
    )
    S += [
        "",
        "## Part 2 — sample splits (midpoint fill; `splits.csv` also carries the crossed fill)",
        "",
        "| series | split | n | mean | t | Sharpe | P&L share % |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r_ in splits.iterrows():
        S.append(
            f"| {r_['series']} | {r_['split']} | {int(r_['n'])} | {r_['mean_mid']:+.3f} | {fm(r_['t_mid'], '+.2f')} | {fm(r_['Sharpe_mid'], '.2f')} | {r_['share_of_total_pnl_mid_pct']:.0f} |"
        )
    S += [
        "",
        "Sign split of the settlement return R by the signal (HC0 t of the difference):",
        "",
        "| forecast | mean R, s ≤ 0 | mean R, s > 0 | difference | t | n sell / n buy |",
        "|---|---|---|---|---|---|",
    ]
    for _, r_ in signsplit.iterrows():
        S.append(
            f"| {r_['forecast']} | {r_['mean_R_when_s_le_0']:+.3f} | {r_['mean_R_when_s_gt_0']:+.3f} | {r_['diff']:.3f} | {r_['t_HC0']:.2f} | {int(r_['n_sell'])} / {int(r_['n_buy'])} |"
        )
    S += ["", "## Gates"]
    S += [
        f"- {g['gate']}: {g['value']:.3g} ≤ {g['bound']:.3g} on n = {g['n']} — {'PASS' if g['ok'] else 'FAIL'}"
        for g in gates
        if not g["gate"].startswith(("target shared", "q = master", "ret_"))
    ]
    n_shared = sum(
        1
        for g in gates
        if g["gate"].startswith(("target shared", "q = master", "ret_"))
    )
    S.append(
        f"- per-forecast identity gates vs the master table (target shared, q, ret_mid, ret_crossed): {n_shared} checked, all pass"
    )
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8")
    print(
        scores.loc[
            COMPARE,
            [
                "QLIKE",
                "ES_at_threshold",
                "EL",
                "hit_var_pct",
                "hit_pay_pct",
                "long_on_top20",
                "Sharpe_mid",
                "Sharpe_crossed",
            ],
        ]
        .round(4)
        .to_string()
    )
    print(ranks.round(3).to_string(index=False))
    print(
        splits[splits["series"] == "headline sign(s)"][
            [
                "split",
                "n",
                "mean_mid",
                "t_mid",
                "Sharpe_mid",
                # 2026-10-06: no intervals
                # "Sharpe_mid_lo",
                # "Sharpe_mid_hi",
                "share_of_total_pnl_mid_pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(signsplit.round(3).to_string(index=False))
    print("reported (not asserted):", reported)
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)


SPLITS = (  # (split name, macro tag)
    ("all days", "All"),
    ("through 2020-03-31 (the COVID quarter)", "Covid"),
    ("after 2020-03-31", "PostCovid"),
    ("before 2022-05-16", "Pre"),
    ("from 2022-05-16 (daily SPXW listings)", "Post"),
    ("settlement pins (R = -1)", "Pin"),
    ("off pins", "OffPin"),
)
T_CRIT = (
    1.96  # |t| beyond which a t statistic is counted (a statistic, not an interval)
)

# 2026-10-15 (table legibility): the legend names the groups in the paper's terms (writeup/NOTATION_GUIDE_2026-10-15.md
# section 5); the group keys themselves (GROUPS, the CSV columns) are unchanged
GROUP_LABELS = {
    "paper": "Earlier draft",
    "pooled linear": "Pooled intraday linear",
    "per-bar linear": "Closing-interval linear",
    "per-bar trees": "Closing-interval trees",
}
MURPHY_FIG = (
    "murphy_modelset"  # OUT / f"{MURPHY_FIG}.pdf" (vector, for the paper) and .png
)


def plot_murphy_modelset(grid: pd.DataFrame, out: Path = OUT) -> None:
    """2026-10-15 (table legibility): the model set's Murphy diagram (Ehm, Gneiting, Jordan and Krueger 2016) from its
    grid alone -- murphy_grid_modelset.csv: one row per forecast (index key), its group, and the mean elementary score
    at each threshold column 'c=...' (theta in the paper; the column names are unchanged) -- so it redraws without
    re-scoring (python experiments/close_murphy_splits_headline.py --figure-only).  Drawn at the text width (6.5 in),
    written as a vector PDF and as the PNG the float used before."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    cols = [f"c={c_:.4f}" for c_ in MURPHY_C]
    assert [c for c in grid.columns if c.startswith("c=")] == cols, (
        "the grid's thresholds are not MURPHY_C"
    )
    g_only = grid[cols].astype(float)
    grp = grid["group"].astype(str)
    keys = list(grid.index)
    mean_c = g_only.mean(axis=0).to_numpy()
    rc = {
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
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
        for k in keys:
            row = g_only.loc[k].to_numpy()
            col = GROUP_COLOURS[grp[k]]
            axes[0].plot(MURPHY_C, row, lw=0.6, alpha=0.6, color=col)
            axes[1].plot(MURPHY_C, row - mean_c, lw=0.6, alpha=0.6, color=col)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.axvline(1.0, color="0.5", ls=":", lw=0.8)
            ax.set_xlabel(r"Threshold $\theta$ (in units of $\mathrm{IV}^2_t$)")
        # 2026-10-15 (notation, final consolidation): the daily score is ES_{theta,t}, its mean over the out-of-sample
        # days \overline{ES}_theta (07_economics.tex; the fig:rh_app_murphy caption). Was: "Mean elementary score
        # $\mathrm{ES}_\theta$" and "$\mathrm{ES}_\theta$ minus the model-set mean".
        axes[0].set_ylabel(r"Mean elementary score $\overline{\mathrm{ES}}_\theta$")
        axes[1].set_ylabel(r"$\overline{\mathrm{ES}}_\theta$ minus the model-set mean")
        axes[0].set_title("(a) Murphy diagram")
        axes[1].set_title("(b) Relative to the model-set mean")
        handles = [
            plt.Line2D(
                [],
                [],
                color=GROUP_COLOURS[g],
                lw=1.4,
                label=f"{GROUP_LABELS[g]} ({int((grp == g).sum())})",
            )
            for g in GROUPS
            if (grp == g).any()
        ]
        assert sum(int((grp == g).sum()) for g in GROUPS) == len(keys), (
            "a forecast outside the four groups"
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
            handlelength=1.5,
            columnspacing=1.2,
        )
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        fig.savefig(out / f"{MURPHY_FIG}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(
            out / f"{MURPHY_FIG}.png", dpi=200, bbox_inches="tight", pad_inches=0.02
        )
        plt.close(fig)
    print("wrote", out / f"{MURPHY_FIG}.pdf", "and .png")


def write_numbers() -> None:
    """2026-10-15 (number macros): NUMBERS_TEX, macro definitions only, from the stored rank_vs_sharpe_modelset.csv
    (no re-scoring; python experiments/close_murphy_splits_headline.py --numbers-only).

    \\hmurRhoESLossMid: Spearman's rank correlation, across the model set, of the mean elementary score at the
    trade's threshold (theta = 1) with the annualized Sharpe ratio at mid-quotes, rho_S(ES-bar_1, SR) -- the loss
    itself, so the sign convention matches \\cmMasterRankCorr (QLIKE against SR). The CSV holds the correlation of
    minus the loss (\\hmurRhoESMid), and rho_S(-x, y) = -rho_S(x, y)."""
    ranks = pd.read_csv(OUT / "rank_vs_sharpe_modelset.csv")
    keys = set(pd.read_csv(MODEL_SET_CSV)["key"].astype(str))
    scored = set(pd.read_csv(OUT / "scores_modelset.csv")["key"].astype(str)) - set(
        ORACLES
    )
    assert scored == keys, (
        "scores_modelset.csv is not the current paper_model_set.csv: re-run main()"
    )
    r_ = ranks[(ranks["loss"] == "ES at the threshold") & (ranks["fill"] == "mid")]
    assert len(r_) == 1 and int(r_["n_forecasts"].iloc[0]) == len(keys), r_
    rho = -float(r_["spearman_minus_loss_vs_sharpe"].iloc[0])
    macros = {"hmurRhoESLossMid": f"{rho:.2f}".replace("-", r"\ensuremath{-}")}
    assert macros == NUMBERS_TYPED, (
        f"{macros} != the values the text typed {NUMBERS_TYPED}: update writeup/rehaul/07_economics.tex and "
        "NUMBERS_TYPED together"
    )
    L = [
        "% AUTO-GENERATED by experiments/close_murphy_splits_headline.py (write_numbers) from",
        "% results/close_murphy_splits/rank_vs_sharpe_modelset.csv -- do not edit. Macro definitions only (no tables);",
        "% loaded once by the preamble of writeup/paper_rehaul.tex.",
        f"% Spearman across the {len(keys)} forecasts of the model set; elementary score at theta = 1 (the 15:30 implied",
        "% variance) against the actual realized variance; Sharpe ratio at mid-quotes; point estimate.",
    ]
    L.append(
        rf"\newcommand{{\hmurRhoESLossMid}}{{{macros['hmurRhoESLossMid']}}}"
        "  % rho_S(mean ES_1, SR mid), the loss itself (= -\\hmurRhoESMid)"
    )
    NUMBERS_TEX.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("wrote", NUMBERS_TEX)


def main() -> None:
    """Threshold score, QLIKE and sample splits across the paper's model set.

    2026-10-06 (user decision): no headline; every model-set forecast is scored as its
    plain forecast f^2 B: QLIKE, the elementary score at the trade's threshold (the 15:30
    implied variance), the economic loss and the hit shares; the Murphy grid; Spearman of
    minus each loss with the Sharpe ratio across the set; the sample splits and the
    same-day sign split per forecast, reported as counts and medians across the set.
    2026-10-15 (user, data audit #10): every score is against the ACTUAL realized variance (the
    per-day frame's rv_unclipped), not the winsorized training target (its ``target`` column).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    def gate(name: str, value: float, bound: float, n: int) -> None:
        gates.append(
            {
                "gate": name,
                "value": float(value),
                "bound": bound,
                "n": n,
                "ok": bool(value <= bound),
            }
        )

    if not DAILY.exists():
        raise SystemExit(
            f"{DAILY} is missing: run experiments/master_table_close.py first"
        )
    mt = pd.read_csv(MASTER).set_index("key")
    keys, set_src = model_set(mt)
    QLIKE_COL = next(c for c in QLIKE_PLAIN_COLS if c in mt.columns)
    D = pd.read_parquet(DAILY)
    D["day"] = pd.to_datetime(D["day"])
    D = D[D["key"].isin(keys)]
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866
    frames = {k: D[D["key"] == k].set_index("day").reindex(days) for k in keys}
    assert all(f[FCOL].notna().all() for f in frames.values()), (
        "a model-set forecast misses a day"
    )
    grp = {k: group_of(str(mt.loc[k, "family"])) for k in keys}
    F = np.vstack([frames[k][FCOL].to_numpy(float) for k in keys])
    # y = frames[keys[0]]["target"].to_numpy(float)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
    # yu = frames[keys[0]]["rv_unclipped"].to_numpy(float)
    # 2026-10-15 (data audit #10): y, the truth of every score here (QLIKE, ES, the Murphy grid, hit_var and the
    # RV oracle), is the ACTUAL realized variance; yt, the winsorized training target, is kept for one diagnostic
    y = frames[keys[0]]["rv_unclipped"].to_numpy(float)
    yt = frames[keys[0]]["target"].to_numpy(float)
    s = frames[keys[0]]["iv_var"].to_numpy(float)
    R = deck["R"].to_numpy(float)
    gate(
        "slice = the deck's iv_var",
        np.abs(s / deck["iv_var"].to_numpy(float) - 1).max(),
        GATE_REL,
        len(days),
    )
    gate(
        # "target shared by every forecast",  # 2026-10-15: the scoring truth is rv_unclipped
        "realized variance (rv_unclipped) shared by every forecast",
        max(
            np.abs(frames[k]["rv_unclipped"].to_numpy(float) / y - 1).max()
            for k in keys
        ),
        GATE_REL,
        len(keys),
    )
    ex = deck["exit"].to_numpy(float)
    ask = (deck["ask_c"] + deck["ask_p"]).to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    cl, cs = ex / ask - 1.0, ex / bid - 1.0

    def pnl(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.atleast_2d(q)
        return a * R, a * np.where(a > 0, cl, cs)

    Q = np.where(F > s, 1.0, -1.0)
    MID, CR = pnl(Q)
    QM = np.vstack([frames[k]["q"].to_numpy(float) for k in keys])
    RM = np.vstack([frames[k]["ret_mid"].to_numpy(float) for k in keys])
    RX = np.vstack([frames[k]["ret_crossed"].to_numpy(float) for k in keys])
    gate(
        "q = the master table's, every forecast and day",
        np.abs(Q - QM).max(),
        0,
        Q.size,
    )
    gate("ret_mid = the master table's", np.abs(MID - RM).max(), GATE_ABS, Q.size)
    gate("ret_crossed = the master table's", np.abs(CR - RX).max(), GATE_ABS, Q.size)
    for col, x in (("Sharpe_mid", MID), ("Sharpe_crossed", CR)):
        gate(
            f"{col} = master_table.csv, every forecast",
            max(
                abs(mtc.sharpe(x[i]) / float(mt.loc[k, col]) - 1)
                for i, k in enumerate(keys)
            ),
            GATE_REL,
            len(keys),
        )
    QL = qlike(y[None, :], F)
    gate(
        f"QLIKE (plain) = master_table.csv {QLIKE_COL}, every forecast",
        max(
            abs(QL[i].mean() / float(mt.loc[k, QLIKE_COL]) - 1)
            for i, k in enumerate(keys)
        ),
        GATE_REL,
        len(keys),
    )
    ES1 = elementary(F / s, (y / s)[None, :], 1.0)
    sgnR = np.where(R > 0, 1.0, -1.0)
    EL = np.abs(R)[None, :] * (Q != sgnR)
    gate(
        "mean(qR) = mean|R| - 2 mean(EL), every forecast",
        np.abs(MID.mean(axis=1) - (np.abs(R).mean() - 2 * EL.mean(axis=1))).max(),
        1e-12,
        len(keys),
    )
    # QLIKE = integral ES_c c^-2 dc, day by day, every forecast (chunks of days)
    u = np.linspace(0.0, 1.0, G3_GRID)
    worst = 0.0
    for i in range(len(keys)):
        fs, ys = F[i] / s, y / s
        lo, hi = np.minimum(fs, ys), np.maximum(fs, ys)
        integral = np.empty(len(lo))
        for a in range(0, len(lo), G3_CHUNK):
            sl = slice(a, a + G3_CHUNK)
            c = lo[sl, None] * (hi[sl] / lo[sl])[:, None] ** u[None, :]
            integral[sl] = np.trapezoid(np.abs(ys[sl, None] - c) / c**2, c, axis=1)
        worst = max(worst, float(np.abs(integral - QL[i]).max()))
    gate(
        "QLIKE = integral ES_c c^-2 dc, day by day, every forecast",
        worst,
        1e-6,
        len(keys) * len(days),
    )

    hit_var = (F > s) == (y > s)[None, :]
    hit_pay = Q == sgnR[None, :]
    order = np.argsort(R)[::-1]
    rows = []
    for i, k in enumerate(keys):
        rows.append(
            {
                "key": k,
                "label": mt.loc[k, "label"],
                "family": mt.loc[k, "family"],
                "group": grp[k],
                "inputs": bucket_of(k),
                "QLIKE": QL[i].mean(),
                "ES_at_threshold": ES1[i].mean(),
                # "ES_at_threshold_unclipped": elementary(F[i] / s, yu / s, 1.0).mean(),  # 2026-10-15: = ES_at_threshold now
                "ES_at_threshold_trimmed": elementary(
                    F[i] / s, yt / s, 1.0
                ).mean(),  # diagnostic, vs the trimmed target
                "EL": EL[i].mean(),
                "hit_var_pct": 100 * hit_var[i].mean(),
                "hit_pay_pct": 100 * hit_pay[i].mean(),
                **{
                    f"long_on_top{kk}": int((Q[i, order[:kk]] > 0).sum())
                    for kk in TOP_K
                },
                "pct_buy": 100 * (Q[i] > 0).mean(),
                "Sharpe_mid": mtc.sharpe(MID[i]),
                "Sharpe_crossed": mtc.sharpe(CR[i]),
            }
        )
    q_var = np.where(y > s, 1.0, -1.0)
    for name, q in ((ORACLES[0], q_var), (ORACLES[1], sgnR.copy())):
        m, cr = pnl(q)
        rows.append(
            {
                "key": name,
                "label": name,
                "family": "",
                "group": "oracle",
                "inputs": "",
                "QLIKE": np.nan,
                "ES_at_threshold": np.nan,
                # "ES_at_threshold_unclipped": np.nan,  # 2026-10-15
                "ES_at_threshold_trimmed": np.nan,
                "EL": (np.abs(R) * (q != sgnR)).mean(),
                "hit_var_pct": 100 * ((q > 0) == (y > s)).mean(),
                "hit_pay_pct": 100 * (q == sgnR).mean(),
                **{f"long_on_top{kk}": int((q[order[:kk]] > 0).sum()) for kk in TOP_K},
                "pct_buy": 100 * (q > 0).mean(),
                "Sharpe_mid": mtc.sharpe(m[0]),
                "Sharpe_crossed": mtc.sharpe(cr[0]),
            }
        )
    scores = pd.DataFrame(rows).set_index("key")
    ms = scores.loc[keys]
    for col, v in (
        ("QLIKE_rank_set", ms["QLIKE"]),
        ("ES_rank_set", ms["ES_at_threshold"]),
        ("Sharpe_crossed_rank_set", -ms["Sharpe_crossed"]),
    ):
        scores[col] = np.nan
        scores.loc[keys, col] = v.rank(method="min")
    scores.to_csv(OUT / "scores_modelset.csv")

    # the Murphy grid: mean ES_c of every model-set forecast
    grid = pd.DataFrame(
        {
            f"c={c_:.4f}": [
                elementary(F[i] / s, y / s, c_).mean() for i in range(len(keys))
            ]
            for c_ in MURPHY_C
        },
        index=pd.Index(keys, name="key"),
    )
    grid.insert(0, "group", [grp[k] for k in keys])
    grid.insert(0, "label", [mt.loc[k, "label"] for k in keys])
    grid.to_csv(OUT / "murphy_grid_modelset.csv")
    g_only = grid.drop(columns=["label", "group"])
    lead = g_only.idxmin(axis=0)
    murphy_lead = pd.DataFrame(
        {
            "c": MURPHY_C,
            "leader_key": lead.to_numpy(),
            "leader_label": [mt.loc[k, "label"] for k in lead],
            "leader_group": [grp[k] for k in lead],
            "best_score": g_only.min(axis=0).to_numpy(),
            "median_score": g_only.median(axis=0).to_numpy(),
        }
    )
    murphy_lead.to_csv(OUT / "murphy_leader_by_threshold_modelset.csv", index=False)

    # rank agreement of minus each loss with the Sharpe ratio across the model set (point)
    rk_rows = []
    for lname, Lm in (
        ("QLIKE", QL),
        ("ES at the threshold", ES1),
        ("EL (affine in the mid P&L)", EL),
    ):
        for fill, P in (("crossed", CR), ("mid", MID)):
            rk_rows.append(
                {
                    "set": "model set",
                    "n_forecasts": len(keys),
                    "loss": lname,
                    "fill": fill,
                    "spearman_minus_loss_vs_sharpe": spearman_point(Lm, P),
                }
            )
    ranks = pd.DataFrame(rk_rows)
    ranks.to_csv(OUT / "rank_vs_sharpe_modelset.csv", index=False)

    # ---------------------------------------------------------------- sample splits
    covid = days <= COVID_END
    pre = days < ERA_CUT
    pin = R <= -1 + 1e-12
    gate(
        "always short collects +1 on pin days",
        abs((-R)[pin].mean() - 1.0),
        1e-9,
        int(pin.sum()),
    )
    masks = {
        "all days": np.ones(len(days), bool),
        "through 2020-03-31 (the COVID quarter)": covid,
        "after 2020-03-31": ~covid,
        "before 2022-05-16": pre,
        "from 2022-05-16 (daily SPXW listings)": ~pre,
        "settlement pins (R = -1)": pin,
        "off pins": ~pin,
    }
    sp_rows = []
    series = [(k, grp[k], MID[i], CR[i]) for i, k in enumerate(keys)]
    series.append(("always short", "always short", -R, 1.0 - ex / bid))
    for name, g, x, xc in series:
        for split, mask in masks.items():
            n, t, sh = stats(x[mask])
            _, tc, shc = stats(xc[mask])
            sp_rows.append(
                {
                    "series": name,
                    "group": g,
                    "split": split,
                    "n": n,
                    "mean_mid": float(x[mask].mean()),
                    "t_mid": t,
                    "Sharpe_mid": sh,
                    "share_of_total_pnl_mid_pct": 100 * x[mask].sum() / x.sum(),
                    "mean_crossed": float(xc[mask].mean()),
                    "t_crossed": tc,
                    "Sharpe_crossed": shc,
                }
            )
    splits = pd.DataFrame(sp_rows)
    splits.to_csv(OUT / "splits_modelset.csv", index=False)
    sm_rows = []
    for split, _ in SPLITS:
        x = splits[(splits["split"] == split) & (splits["series"] != "always short")]
        a_ = splits[
            (splits["split"] == split) & (splits["series"] == "always short")
        ].iloc[0]
        sm_rows.append(
            {
                "split": split,
                "n_days": int(a_["n"]),
                "n_forecasts": len(x),
                "n_Sharpe_mid_pos": int((x["Sharpe_mid"] > 0).sum()),
                "Sharpe_mid_median": float(x["Sharpe_mid"].median()),
                "Sharpe_mid_min": float(x["Sharpe_mid"].min()),
                "Sharpe_mid_max": float(x["Sharpe_mid"].max()),
                "t_mid_median": float(x["t_mid"].median()),
                "n_t_mid_above_crit": int((x["t_mid"] > T_CRIT).sum()),
                "n_Sharpe_crossed_pos": int((x["Sharpe_crossed"] > 0).sum()),
                "Sharpe_crossed_median": float(x["Sharpe_crossed"].median()),
                "mean_mid_median": float(x["mean_mid"].median()),
                "pnl_share_median_pct": float(x["share_of_total_pnl_mid_pct"].median()),
                "always_short_Sharpe_mid": float(a_["Sharpe_mid"]),
                "always_short_mean_mid": float(a_["mean_mid"]),
            }
        )
    split_sum = pd.DataFrame(sm_rows).set_index("split")
    split_sum.to_csv(OUT / "splits_summary_modelset.csv")
    sg_rows = []
    for i, k in enumerate(keys):
        ind = (Q[i] > 0).astype(float)
        fit = sm.OLS(R, sm.add_constant(ind)).fit(cov_type="HC0")
        sg_rows.append(
            {
                "key": k,
                "group": grp[k],
                "mean_R_when_s_le_0": float(R[ind == 0].mean()),
                "mean_R_when_s_gt_0": float(R[ind == 1].mean()),
                "diff": float(fit.params[1]),
                "t_HC0": float(fit.tvalues[1]),
                "n_sell": int((ind == 0).sum()),
                "n_buy": int((ind == 1).sum()),
            }
        )
    signsplit = pd.DataFrame(sg_rows)
    signsplit.to_csv(OUT / "sign_split_modelset.csv", index=False)

    # ---- the rows that read no forecast still equal their committed copies
    # 2026-10-15 (data audit #10): the RV oracle and every row's hit_var_pct read the realized variance, now the ACTUAL
    # RV (on 2023-12-20 it sits above the 15:30 implied variance, the winsorized target below), so the RV oracle row and
    # the hit_var_pct column are left out of this gate; the payoff oracle and always short's splits are still compared.
    for fname, idx, col, keep, newframe in (
        # ("scores_by_forecast.csv", ["key"], None, ORACLES, scores.reset_index()),  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        ("scores_by_forecast.csv", ["key"], None, [ORACLES[1]], scores.reset_index()),
        ("splits.csv", ["series", "split"], "series", ["always short"], splits),
    ):
        old, sha = committed_csv(OUT / fname)
        new = newframe.copy()
        if col is not None:
            old, new = old[old[col].isin(keep)], new[new[col].isin(keep)]
        old, new = old.set_index(idx), new.set_index(idx)
        if col is None:
            old = old.loc[old.index.isin(keep)]
        old = old[
            [
                c
                for c in old.columns
                if c not in RANK_COLS
                and c in new.columns
                and c not in ("group", "hit_var_pct")
            ]
        ]
        gate(
            f"{fname} = the committed copy ({PREV_REV} = {sha}), {len(old)} rows that read no forecast",
            committed_diff(new, old),
            COMMITTED_TOL,
            int(old.size),
        )
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False)
    assert all(g["ok"] for g in gates), [g for g in gates if not g["ok"]]

    # ---------------------------------------------------------------- figure
    # 2026-10-15 (table legibility): parked -- labels threshold c / ES_c and "plain f²B", raster only, drawn at 11 in and
    # shrunk to the text width; replaced by plot_murphy_modelset(grid) below (theta / ES_theta after Ehm, Gneiting, Jordan
    # and Krueger 2016, vector PDF plus the PNG, drawn at the text width; the same grid, so no value changes)
    # import matplotlib
    #
    # matplotlib.use("Agg")
    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    # mean_c = g_only.mean(axis=0).to_numpy()
    # for k in keys:
    #     row = g_only.loc[k].to_numpy()
    #     col = GROUP_COLOURS[grp[k]]
    #     axes[0].plot(MURPHY_C, row, lw=0.7, alpha=0.6, color=col)
    #     axes[1].plot(MURPHY_C, row - mean_c, lw=0.7, alpha=0.6, color=col)
    # for ax in axes:
    #     ax.set_xscale("log", base=2)
    #     ax.axvline(1.0, color="0.5", ls=":", lw=0.8)
    #     ax.set_xlabel("threshold c, in units of the 15:30 implied variance")
    # axes[0].set_ylabel("mean elementary score ES_c (lower is better)")
    # axes[1].set_ylabel("ES_c minus the mean of the set at c")
    # axes[0].set_title(
    #     f"(a) Murphy diagram, the model set ({len(keys)} forecasts, plain f²B), 866 days",
    #     fontsize=9,
    # )
    # axes[1].set_title("(b) each line minus the mean of the set", fontsize=9)
    # handles = [
    #     plt.Line2D(
    #         [],
    #         [],
    #         color=GROUP_COLOURS[g],
    #         lw=1.4,
    #         label=f"{g} ({sum(grp[k] == g for k in keys)})",
    #     )
    #     for g in GROUPS
    #     if any(grp[k] == g for k in keys)
    # ]
    # fig.legend(
    #     handles=handles,
    #     loc="lower center",
    #     ncol=len(handles),
    #     fontsize=8,
    #     frameon=False,
    #     bbox_to_anchor=(0.5, 0.0),
    # )
    # fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    # fig.savefig(OUT / "murphy_modelset.png", dpi=130, bbox_inches="tight")
    # plt.close(fig)
    assert list(grid.index) == keys and grid["group"].tolist() == [grp[k] for k in keys]
    plot_murphy_modelset(grid)

    # ---------------------------------------------------------------- tex: tables + macros
    def t3(v: float) -> str:
        return f"{v:.3f}"

    L = [
        "% AUTO-GENERATED by experiments/close_murphy_splits_headline.py (main: the paper's model set) from",
        "% results/close_murphy_splits/*_modelset.csv -- do not edit.",
        f"% 2026-10-06: the paper's model set ({set_src}); no headline; plain forecast f^2 B; point estimates only.",
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{p{0.2\textwidth}rrrrrrrr}",
        rf"\caption{{The paper's model set ({len(keys)} forecasts, each its plain forecast $\hat y^2B$, no recalibration) "
        # r"scored two ways on the 866 deck days, by group: the median QLIKE against the scorer's target and its range, the "  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        r"scored two ways on the 866 deck days, by group: the median QLIKE against the actual (not winsorized) realized "
        r"variance and its range, the "
        r"median elementary score at the trade's threshold (the 15:30 implied variance), the median economic loss "
        r"$|R|\,\mathbf 1\{q\neq\mathrm{sign}(R)\}$, the median shares of days on the realized variance's side of the "
        r"threshold and on the payoff's side, and the median $\mathrm{sign}(s)$ Sharpe ratio at both fills. The last two "
        r"rows are rules told the realized variance, and the payoff's sign, in advance.}\label{tab:app_murphy_headline}\\",
        r"\toprule",
        r"group & $n$ & QLIKE & range & ES($c{=}1$) & EL & $RV$ side \% & $R$ side \% & Sharpe mid/crossed \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"group & $n$ & QLIKE & range & ES($c{=}1$) & EL & $RV$ side \% & $R$ side \% & Sharpe mid/crossed \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    by_grp: list[tuple[str, pd.DataFrame]] = [
        (g, ms[ms["group"] == g]) for g in GROUPS if (ms["group"] == g).any()
    ]
    by_grp.append(("model set", ms))
    for g, x in by_grp:
        lab = r"\textbf{model set}" if g == "model set" else g
        L.append(
            f"{lab} & {len(x)} & {t3(x['QLIKE'].median())} & {t3(x['QLIKE'].min())} to {t3(x['QLIKE'].max())} & "
            f"{t3(x['ES_at_threshold'].median())} & {t3(x['EL'].median())} & {x['hit_var_pct'].median():.1f} & "
            f"{x['hit_pay_pct'].median():.1f} & {x['Sharpe_mid'].median():.2f} / {x['Sharpe_crossed'].median():.2f} \\\\"
        )
    for o in ORACLES:
        r_ = scores.loc[o]
        L.append(
            f"\\raggedright {o.split(' (')[0]} & & -- & & -- & {t3(r_['EL'])} & {r_['hit_var_pct']:.1f} & {r_['hit_pay_pct']:.1f} & "
            f"{r_['Sharpe_mid']:.2f} / {r_['Sharpe_crossed']:.2f} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]
    L += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{p{0.27\textwidth}rrrrrrr}",
        rf"\caption{{Sample splits of the $\mathrm{{sign}}(s)$ return (midpoint fill) across the paper's model set "
        rf"({len(keys)} forecasts, plain forecast): per split the number of forecasts with a positive Sharpe ratio, the "
        r"median, smallest and largest Sharpe ratio, the median $t=\sqrt{n}\,\bar x/s_x$ and the number of forecasts with "
        rf"$t>{T_CRIT:.2f}$, and always short on the same days.}}\label{{tab:app_splits_headline}}\\",
        r"\toprule",
        rf"split & days & Sharpe $>0$ & median & range & median $t$ & $t>{T_CRIT:.2f}$ & always short \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        rf"split & days & Sharpe $>0$ & median & range & median $t$ & $t>{T_CRIT:.2f}$ & always short \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for split, r_ in split_sum.iterrows():
        L.append(
            f"\\raggedright {split} & {int(r_['n_days'])} & {int(r_['n_Sharpe_mid_pos'])} & {fm(r_['Sharpe_mid_median'], '.2f')} & "
            f"{fm(r_['Sharpe_mid_min'], '.2f')} to {fm(r_['Sharpe_mid_max'], '.2f')} & {fm(r_['t_mid_median'], '.2f')} & "
            f"{int(r_['n_t_mid_above_crit'])} & {fm(r_['always_short_Sharpe_mid'], '.2f')} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]

    ov, op = scores.loc[ORACLES[0]], scores.loc[ORACLES[1]]
    rk = ranks.set_index(["loss", "fill"])["spearman_minus_loss_vs_sharpe"]
    lead_at_1 = murphy_lead.loc[np.isclose(murphy_lead["c"], 1.0)].iloc[0]
    n_lead = murphy_lead["leader_group"].value_counts()
    best_q = ms["QLIKE"].idxmin()
    best_es = ms["ES_at_threshold"].idxmin()
    sgs = signsplit
    ap_ = splits[
        (splits["series"] == "always short")
        & (splits["split"] == "settlement pins (R = -1)")
    ].iloc[0]
    ao = splits[
        (splits["series"] == "always short") & (splits["split"] == "off pins")
    ].iloc[0]
    fpin = splits[
        (splits["series"] != "always short")
        & (splits["split"] == "settlement pins (R = -1)")
    ]
    foff = splits[
        (splits["series"] != "always short") & (splits["split"] == "off pins")
    ]

    def tex_label(k: str) -> str:
        return str(mt.loc[k, "label"]).replace("_", r"\_").replace("&", r"\&")

    macros = {
        "hmurNSet": f"{len(keys)}",
        "hmurNPaper": f"{sum(grp[k] == 'paper' for k in keys)}",
        "hmurNPooled": f"{sum(grp[k] == 'pooled linear' for k in keys)}",
        "hmurNPerbarLinear": f"{sum(grp[k] == 'per-bar linear' for k in keys)}",
        "hmurNTrees": f"{sum(grp[k] == 'per-bar trees' for k in keys)}",
        "hmurQLIKEMed": t3(ms["QLIKE"].median()),
        "hmurQLIKEMin": t3(ms["QLIKE"].min()),
        "hmurQLIKEMax": t3(ms["QLIKE"].max()),
        "hmurESMed": t3(ms["ES_at_threshold"].median()),
        "hmurESMin": t3(ms["ES_at_threshold"].min()),
        "hmurESMax": t3(ms["ES_at_threshold"].max()),
        "hmurBestQLIKELabel": tex_label(best_q),
        "hmurBestESLabel": tex_label(best_es),
        "hmurHitVarMed": f"{ms['hit_var_pct'].median():.0f}",
        "hmurHitPayMed": f"{ms['hit_pay_pct'].median():.0f}",
        "hmurHitPayMin": f"{ms['hit_pay_pct'].min():.0f}",
        "hmurHitPayMax": f"{ms['hit_pay_pct'].max():.0f}",
        "hmurSharpeMidMed": f"{ms['Sharpe_mid'].median():.2f}",
        "hmurSharpeXMed": f"{ms['Sharpe_crossed'].median():.2f}",
        "hmurTopMed": f"{ms['long_on_top20'].median():.0f}",
        "hmurOracleAgree": f"{ov['hit_pay_pct']:.0f}",
        "hmurOracleMiss": f"{100 - ov['hit_pay_pct']:.0f}",
        "hmurOracleShMid": f"{ov['Sharpe_mid']:.2f}",
        "hmurOracleShX": f"{ov['Sharpe_crossed']:.2f}",
        "hmurPayoffOracleShX": f"{op['Sharpe_crossed']:.2f}",
        "hmurLeaderAtOne": tex_label(lead_at_1["leader_key"]),
        "hmurLeaderAtOneGroup": str(lead_at_1["leader_group"]),
        "hmurNGrid": f"{len(MURPHY_C)}",
        "hmurGridMin": f"{MURPHY_C.min():.2f}",
        "hmurGridMax": f"{MURPHY_C.max():.0f}",
        "hmurNLeadPaper": f"{int(n_lead.get('paper', 0))}",
        "hmurNLeadPooled": f"{int(n_lead.get('pooled linear', 0))}",
        "hmurNLeadPerbarLinear": f"{int(n_lead.get('per-bar linear', 0))}",
        "hmurNLeadTrees": f"{int(n_lead.get('per-bar trees', 0))}",
        "hmurRhoQX": f"{rk[('QLIKE', 'crossed')]:+.2f}",
        "hmurRhoESX": f"{rk[('ES at the threshold', 'crossed')]:+.2f}",
        "hmurRhoQMid": f"{rk[('QLIKE', 'mid')]:+.2f}",
        "hmurRhoESMid": f"{rk[('ES at the threshold', 'mid')]:+.2f}",
        "hsplNCovid": f"{int(covid.sum())}",
        "hsplNPin": f"{int(pin.sum())}",
        "hsplPinPct": f"{100 * pin.mean():.1f}",
        "hsplASPinMean": f"{ap_['mean_mid']:+.2f}",
        "hsplASOffPinMean": f"{ao['mean_mid']:+.2f}",
        "hsplPinMeanMed": f"{fpin['mean_mid'].median():+.2f}",
        "hsplOffPinMeanMed": f"{foff['mean_mid'].median():+.2f}",
        "hsplPinShareMed": f"{fpin['share_of_total_pnl_mid_pct'].median():.0f}",
        "hsplSignDiffMed": f"{sgs['diff'].median():.2f}",
        "hsplSignNPos": f"{int((sgs['diff'] > 0).sum())}",
        "hsplSignTMed": f"{sgs['t_HC0'].median():.2f}",
        "hsplSignNT": f"{int((sgs['t_HC0'] > T_CRIT).sum())}",
    }
    for split, tag in SPLITS:
        r_ = split_sum.loc[split]
        macros |= {
            f"hspl{tag}N": f"{int(r_['n_days'])}",
            f"hspl{tag}NPos": f"{int(r_['n_Sharpe_mid_pos'])}",
            f"hspl{tag}Med": fm(r_["Sharpe_mid_median"], ".2f"),
            f"hspl{tag}Min": fm(r_["Sharpe_mid_min"], ".2f"),
            f"hspl{tag}Max": fm(r_["Sharpe_mid_max"], ".2f"),
            f"hspl{tag}TMed": fm(r_["t_mid_median"], ".2f"),
            f"hspl{tag}NT": f"{int(r_['n_t_mid_above_crit'])}",
            f"hspl{tag}AS": fm(r_["always_short_Sharpe_mid"], ".2f"),
        }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    stubs = [m for m in HEADLINE_MACROS_PARKED if m not in macros]
    L += [
        "% 2026-10-06: the headline macros are parked (no headline model; user decision). Each is a",
        "% visible ?? stub until sections/appendix_close_murphy_splits.tex stops using it.",
    ]
    L += [f"\\newcommand{{\\{m}}}{{\\textbf{{??}}}}" for m in stubs]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---------------------------------------------------------------- SUMMARY.md
    S = [
        "# Threshold score, QLIKE and sample splits across the paper's model set (plain forecast f²·B)",
        "",
        "Written by `experiments/close_murphy_splits_headline.py` (`main`, 2026-10-06: no headline model, user decision) from "
        "`scores_modelset.csv`, `murphy_grid_modelset.csv`, `murphy_leader_by_threshold_modelset.csv`, `rank_vs_sharpe_modelset.csv`, "
        "`splits_modelset.csv`, `splits_summary_modelset.csv`, `sign_split_modelset.csv` and `gates.csv`. Input: the master table's "
        f"per-day frame. Model set: {len(keys)} forecasts ({set_src}). Every forecast is its plain f²·B; the deck's 15:30 sign(s) "
        "trade; 866 days 2020-01-03 .. 2024-04-30; point estimates only.",
        "",
        "## Scores by group (medians)",
        "",
        "| group | n | QLIKE | QLIKE range | ES(c=1) | EL | same side as RV % | same side as R % | Sharpe mid | Sharpe crossed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g, x in by_grp:
        S.append(
            f"| {g} | {len(x)} | {x['QLIKE'].median():.4f} | {x['QLIKE'].min():.4f} to {x['QLIKE'].max():.4f} | "
            f"{x['ES_at_threshold'].median():.4f} | {x['EL'].median():.4f} | {x['hit_var_pct'].median():.1f} | "
            f"{x['hit_pay_pct'].median():.1f} | {x['Sharpe_mid'].median():.2f} | {x['Sharpe_crossed'].median():.2f} |"
        )
    S += [
        "",
        f"- Lowest QLIKE: {mt.loc[best_q, 'label']}; lowest score at the threshold: {mt.loc[best_es, 'label']}.",
        f"- Murphy grid ({len(MURPHY_C)} thresholds, {MURPHY_C.min():.2f} to {MURPHY_C.max():.0f}): thresholds led by group — "
        + ", ".join(f"{g} {int(n)}" for g, n in n_lead.items())
        + f"; at c = 1 the leader is {lead_at_1['leader_label']} ({lead_at_1['leader_group']}).",
        f"- Oracles: told RV, the payoff's side on {ov['hit_pay_pct']:.1f} % of days (Sharpe {ov['Sharpe_mid']:.2f} / "
        f"{ov['Sharpe_crossed']:.2f}); told the payoff's sign, {op['Sharpe_crossed']:.2f} crossed.",
        "- Spearman of minus the loss with the Sharpe ratio across the set (point): "
        + "; ".join(f"{lo_} vs {fl} {v:+.2f}" for (lo_, fl), v in rk.items())
        + ".",
        "",
        "## Sample splits across the set (midpoint)",
        "",
        "| split | days | Sharpe > 0 | median Sharpe | range | median t | t > 1.96 | always short |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, r_ in split_sum.iterrows():
        S.append(
            f"| {split} | {int(r_['n_days'])} | {int(r_['n_Sharpe_mid_pos'])} of {len(keys)} | {fm(r_['Sharpe_mid_median'], '.2f')} | "
            f"{fm(r_['Sharpe_mid_min'], '.2f')} to {fm(r_['Sharpe_mid_max'], '.2f')} | {fm(r_['t_mid_median'], '.2f')} | "
            f"{int(r_['n_t_mid_above_crit'])} | {fm(r_['always_short_Sharpe_mid'], '.2f')} |"
        )
    S += [
        "",
        f"- Sign split (mean R on buy days minus sell days, HC0 t): median {sgs['diff'].median():.3f}, positive for "
        f"{int((sgs['diff'] > 0).sum())} of {len(keys)}, median t {sgs['t_HC0'].median():.2f}, t > {T_CRIT} for "
        f"{int((sgs['t_HC0'] > T_CRIT).sum())}.",
        "",
        "## Gates",
    ]
    S += [
        f"- {g['gate']}: {g['value']:.3g} ≤ {g['bound']:.3g} on n = {g['n']} — {'PASS' if g['ok'] else 'FAIL'}"
        for g in gates
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8", newline="\n")
    print(
        pd.DataFrame(
            {
                g: x[
                    ["QLIKE", "ES_at_threshold", "Sharpe_mid", "Sharpe_crossed"]
                ].median()
                for g, x in by_grp
            }
        ).T.round(4)
    )
    print(ranks.round(3).to_string(index=False))
    print(split_sum.round(3).to_string())
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)
    write_numbers()  # 2026-10-15 (number macros): the macro-only file, from the CSVs just written


if __name__ == "__main__":
    # 2026-10-15 (table legibility): --figure-only redraws the Murphy figure from the stored grid (no re-scoring)
    if "--figure-only" in sys.argv[1:]:
        plot_murphy_modelset(
            pd.read_csv(OUT / "murphy_grid_modelset.csv", index_col="key")
        )
    # 2026-10-15 (number macros): --numbers-only writes NUMBERS_TEX from the stored CSVs (no re-scoring)
    elif "--numbers-only" in sys.argv[1:]:
        write_numbers()
    else:
        main()
