"""har_base_sweep — is the HAR ladder's base-5 geometry the right one?

PURPOSE: sweep the GEOMETRY of the HAR lag ladder itself — the one structural
constant every model in this repo inherits unquestioned (resolve_har_lags:
powers of 5 up to 3125). Two axes:

* ladder-arm — the geometric base b in {2, 3, 4, 5, 6, 8}, ladder
  {1, b, b^2, ... <= cap}; PLUS one explicit decaying-ratio arm
  {1, 8, 48, 240, 960} (consecutive ratios 8 -> 6 -> 5 -> 4), which no
  geometric base reproduces. b=2 is the densest ladder (12 rungs at cap
  3125, most flexible / most collinear); b=8 the sparsest (4 rungs); b=5 is
  the incumbent geometry, so the production ladder is a POINT ON THE CURVE,
  not just an external anchor.
* rungs-cap — the ladder's maximum-memory rung: rungs are generated UP TO a
  cap C in {240, 960, 3125} (5 days / 20 days / the incumbent's 65 days at
  48 bars per day). The cap axis is DELIBERATELY bounded by 3125: the
  executor's HAR burn-in drop is fixed at resolve_har_lags()[-1] = 3125 rows
  for EVERY arm (arm-invariant index space), so all arms share the
  incumbent's exact bar grid and per-bar losses align bar-for-bar for DM. A
  rung beyond 3125 would either be under-warmed at the leading edge or force
  a burn-in change that breaks that alignment — run_executor now REFUSES
  har_lags beyond the burn-in rather than silently mis-warming. The explicit
  arm is truncated to the cap ({1,8,48,240} at C=240; complete at C=960);
  at C=3125 it is IDENTICAL to C=960 (top rung 960), so that duplicate cell
  is skipped locally — the grid is 6x3 geometric + 2 explicit = 20 arms.

ESTIMATOR HELD FIXED (the ladder is the only treatment): OLS on HAR +
calendar with NO exog — the production src.models.ridge.fit_predict_ridge at
alpha=0.0 (RollingLeastSquares rank-1 is EXACT OLS at zero penalty, per-bar
refit), on the empty-exog `baseline` bucket, i.e. the SAME shared OLS-HAR
incumbent every audited verdict in this repo compares against
(specs/causal_tune_linear.py, specs/causal_tune_tree.py,
specs/spectral_knn_ols.py). The incumbent arm here is that estimator at
har_lags=None — the UNTOUCHED production code path (base-5 ladder), the
baseline the user is familiar with. KNOWN-ANSWER CHECK: the swept arm
(b=5, cap=3125) must reproduce the incumbent's predictions EXACTLY
(byte-identical ladder -> byte-identical features -> byte-identical OLS);
the baseline section asserts array equality.

THE SEAM (gated addition to the shared engine, OFF by default): run_executor
gained `har_lags: list[int] | None = None`, threaded through
_iter_TOD_segment -> _build_har_and_calendar -> generate_har_features(lags=).
har_lags=None is the byte-identical production path (resolve_har_lags()),
so every existing caller is unchanged. The har_ma_{lag}_x_{open,close}
session-edge interactions are built off the arm's own har_names, so the
distilled intraday sign-flip channel scales with the ladder automatically.

MACHINERY (reuse ruling): data prep, walk-forward dispatch, scale guards,
and the raw-space Duan reconstruction ALL come from the production scaffold
src.backtest.executor.run_executor, invoked BYTE-IDENTICALLY to the audited
causal_tune_linear.py / spectral_knn_ols.py arms (calendar on, diurnal
target, winsor 240, impute_indicate=True + overnight_fill=True +
dropna_with_exog=False, prescale=True, train_window=500 days, horizon 1,
seed 42) so the target series and index space match the incumbent's exactly.
Metrics: src.evaluation.metrics.forecast_metrics (the full beat4-table
spread — qlike / mse / rmse / mae / hmse / hmae / oos_r2 / mz_beta / mz_r2 /
n) plus the Diebold-Mariano arm-vs-incumbent test on per-bar QLIKE losses
(src.evaluation.diebold_mariano.qlike_per_bar + dm_test, HAC lag auto) —
the SAME citable columns as the causal_tune_tree table.

SCOPE: ladder-arm x rungs-cap x chunk. 20 arms x 100 chunks -> 2000 tasks.

PARALLELISM: cluster MAP-REDUCE through the hpc-agent submit pipeline. Map
axes are ladder-arm x rungs-cap x chunk: LADDER_ARM ("2".."8" | "explicit")
+ RUNGS_CAP select one arm per task, and run_executor's start, end, halo —
the SliceSpec chunk seam fed via HPC_KW_* env — slice the walk-forward
series per task. The training window is warm-up HALO (BoundedHalo):
train_window=500 days = 24000 bars, so a mid-series chunk replays 24000
warm-up rows without emitting and reproduces the whole-series arm exactly.
Each task writes its results table plus the per-chunk reduce piece
(save_chunk_reduce); the run-level reducer pools the chunks per arm (the
reduce_causal_tune_linear pattern). The incumbent is re-run per task on the
same chunk so per-bar DM losses align within every task. Local defaults
(no env) = all 20 arms serially, whole series. DRY-RUN: the source honors
HPC_NOTEBOOK_SAMPLE_N — when set (and no arm is env-pinned) it trims the
grid to 3 representative arms (b2@240 densest, b5@3125 known-answer,
explicit@960 contract) and caps END at TRAIN_BARS + sample_n, so each arm
emits sample_n OOS rows. The contract arrays (pred_raw, true_raw,
baseline_pred_raw) are the explicit@960 arm — the user's named ladder.
"""

# %%
# Path bootstrap (PREAMBLE — outside every audit section by design; interactive
# kernels default cwd to this file's directory, and `src.*` imports need the
# repo root). Walks up from the file (or cwd when __file__ is absent, e.g. an
# interactive cell) to the first directory containing src/.
import sys
from pathlib import Path

import os

_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _START
while not (_ROOT / "src").is_dir() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if (_ROOT / "src").is_dir():  # only normalize when the repo root was found
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)  # relative data literals ("data") resolve from the root,
    # and stay bare literals for the executes-live lint's surface.

# %%
# hpc-audit-section: data-selection
# [PINNED — copy verbatim; auto-clears. The pinned loader, path literal
# under input_roots, shape shown as evidence.]
from src.data.loading import load_raw_data

DATA_PATH = "data"
df = load_raw_data(DATA_PATH, allow_missing=True)
print(f"loaded: rows={len(df)} cols={len(df.columns)}")

# %%
# hpc-audit-section: target-construction
# [PINNED — copy verbatim; auto-clears. Mirrors the production call
# (src/backtest/executor.py prepare step): the invariant transform CALLED,
# never re-derived — diurnal -> sqrt -> winsorize, is_target=True; the
# multiplicative baseline kept for raw-space reconstruction.]
from src.features.transforms.target import robust_transform

adj_rv, rv_baseline = robust_transform(df, "RV", is_target=True)
df["adj_RV"] = adj_rv
df["baseline"] = rv_baseline
print(f"target adj_RV: non-null={int(adj_rv.notna().sum())} (diurnal->sqrt->winsorize, is_target=True)")

# %%
# hpc-audit-section: feature-construction
# [VARIABLE — the HAR-ladder arms, one OLS-HAR backtest per (base, cap) cell,
# run through the PRODUCTION scaffold: src.backtest.executor.run_executor
# does the data prep (load_and_transform, HAR + calendar build, burn-in drop,
# horizon shift, scale guards + prescale, Duan raw-space reconstruction,
# results table) — nothing re-derived here. The pinned cells above
# demonstrate the loader + target; run_executor re-runs that same prep
# internally (the same functions). The ONLY moving part per arm is the
# ladder handed to the NEW har_lags seam (run_executor ->
# _build_har_and_calendar -> generate_har_features(lags=...), gated OFF by
# default so har_lags=None keeps every existing caller byte-identical; the
# har_ma_{lag}_x_{open,close} session-edge interactions follow the arm's own
# ladder automatically because they are built off har_names). Ladder
# construction: geometric arms {1, b, b^2, ... <= cap} for b in
# {2,3,4,5,6,8}; the explicit decaying-ratio arm {1,8,48,240,960} truncated
# to the cap; caps {240, 960, 3125}. Cap bounded by 3125 BY CONSTRUCTION:
# the executor's burn-in drop stays fixed at resolve_har_lags()[-1] = 3125
# rows for every arm (arm-invariant index space -> per-bar losses align
# bar-for-bar with the incumbent for DM), and run_executor REFUSES a rung
# beyond it (under-warmed rolling mean) rather than silently mis-warming.
# The (explicit, 3125) cell duplicates (explicit, 960) byte-for-byte (top
# rung 960) and is skipped locally — 20 unique arms.
# ESTIMATOR HELD FIXED so the ladder is the only treatment: the production
# fit_predict_ridge at alpha=0.0 (RollingLeastSquares rank-1 = EXACT OLS,
# per-bar refit, incremental) on the empty-exog `baseline` bucket (HAR +
# calendar only) — the SAME estimator as the shared OLS-HAR incumbent of
# specs/causal_tune_linear.py / spectral_knn_ols.py.
# DATA-PREP INVARIANTS: the run_executor invocation is byte-identical to
# those audited arms (calendar on, diurnal target, winsor 240,
# impute_indicate=True + overnight_fill=True + dropna_with_exog=False,
# prescale=True, train_window=500 days, horizon 1, seed 42).
# PARALLEL: cluster map-reduce — tasks map over ladder-arm x rungs-cap x
# chunk (LADDER_ARM + RUNGS_CAP + the SliceSpec start, end, halo fed as
# HPC_KW_* env; run_executor's train_window is in DAYS: 500 days = 24000
# bars, so a mid-series chunk's halo is 24000 warm-up bars). Each task's
# save_chunk_reduce piece is combined by the framework's reduce. Local
# defaults (no env) = all 20 arms serially, whole series. DRY-RUN honor:
# HPC_NOTEBOOK_SAMPLE_N (set, with no arm env-pinned) trims the grid to 3
# representative arms and caps END at TRAIN_BARS + sample_n. Output tables
# land under RESULTS_ROOT — $HPC_RESULT_DIR under the dispatcher, a write
# target the executes-live lint cannot verify (no output_roots are recorded
# for this audit; disclosed, not masked).
# END STATE: per-arm executor results tables in `arm_results`, keyed
# "b{base}_c{cap}" / "explicit_c{cap}", each with its ladder in `arm_lags`.]
import pandas as pd

from src.backtest.executor import run_executor
from src.data.loading import get_bucket
from src.features.extractors.har import resolve_har_lags
from src.models.ridge import fit_predict_ridge

HORIZON = 1
TRAIN_WIN = 500        # days; = 24000 bars at 48 bars/day (the audited default)
TRAIN_BARS = TRAIN_WIN * 48
SEED = 42
BASES = [2, 3, 4, 5, 6, 8]
CAPS = [240, 960, 3125]
EXPLICIT = [1, 8, 48, 240, 960]   # decaying consecutive ratios 8 -> 6 -> 5 -> 4
LADDER_BUCKET = "baseline"        # empty-exog: HAR + calendar only
# alpha=0.0 -> RollingLeastSquares(alpha=0.0): exact OLS, zero penalty,
# per-bar refit — the shared-incumbent estimator, held fixed across arms.
OLS_HP = dict(alpha=0.0, _refit_frequency=1, _incremental=True)


def ladder(arm: str, cap: int) -> list[int]:
    """Rungs-to-a-cap: geometric {1, b, b^2, ... <= cap}, or EXPLICIT truncated."""
    if arm == "explicit":
        return [r for r in EXPLICIT if r <= cap]
    return resolve_har_lags(max_lag=cap, base=int(arm))


# MAP-REDUCE seam: the hpc-agent dispatcher fans tasks out over ladder-arm x
# rungs-cap x chunk — resolve() kwargs arrive as HPC_KW_* env, and start,
# end, halo are run_executor's SliceSpec chunk fields.
def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


LADDER_ARM = _env("LADDER_ARM", "")     # "2".."8" | "explicit" | "" = all
RUNGS_CAP = int(_env("RUNGS_CAP", "0"))  # 240 | 960 | 3125 | 0 = all
START = int(_env("START", "0"))
END = int(_env("END", "-1"))
HALO = int(_env("HALO", "0"))
SAMPLE_N = int(os.environ.get("HPC_NOTEBOOK_SAMPLE_N", "0") or "0")

if LADDER_ARM:
    ARM_GRID = [(LADDER_ARM, RUNGS_CAP)]
else:
    ARM_GRID = [(str(b), c) for b in BASES for c in CAPS]
    ARM_GRID += [("explicit", 240), ("explicit", 960)]  # explicit@3125 ==
    # explicit@960 byte-for-byte (top rung 960): duplicate cell skipped.
    if SAMPLE_N:  # dry-run: 3 representative arms — densest geometry,
        # the known-answer cell, and the contract arm.
        ARM_GRID = [("2", 240), ("5", 3125), ("explicit", 960)]
if SAMPLE_N and END < 0:
    END = TRAIN_BARS + SAMPLE_N  # each arm emits sample_n OOS rows

CONTRACT_ARM = "explicit_c960"

RESULTS_ROOT = os.path.join(
    os.environ.get("HPC_RESULT_DIR", "results"), "har_base_sweep"
)  # write target ($HPC_RESULT_DIR under the dispatcher).

arm_results: dict = {}
arm_lags: dict = {}
for arm, cap in ARM_GRID:
    label = (f"b{arm}" if arm != "explicit" else "explicit") + f"_c{cap}"
    lags = ladder(arm, cap)
    out_csv = os.path.join(RESULTS_ROOT, label, "results.csv")
    # the run_executor invocation of the audited incumbent (causal_tune_linear
    # / spectral_knn_ols), production OLS fit_predict, ONLY har_lags varies.
    run_executor(
        method_name=f"har_ladder_{label}",
        fit_predict=fit_predict_ridge,
        hyperparams=dict(OLS_HP),
        data_path=DATA_PATH,
        output_file=out_csv,
        horizon=HORIZON,
        train_window=TRAIN_WIN,
        start=START,
        end=END,
        halo=HALO,
        exog_cols=get_bucket(LADDER_BUCKET),
        segment=None,
        lag_scope="global",
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=SEED,
        har_lags=lags,
    )
    arm_results[label] = pd.read_csv(out_csv)
    arm_lags[label] = lags
    print(f"har_ladder/{label}: rungs={lags} ({len(lags)}); "
          f"{len(arm_results[label])} OOS rows")

# %%
# hpc-audit-section: baseline
# [VARIABLE-BY-CONTENT, PINNED-BY-IDENTITY — will diff; human sign-off
# expected. DELIBERATE DEVIATION from the template prose (which cites the
# deployed ridge config): the comparator is the OLS-HAR base-5 anchor "as
# we've been doing" (user-ruled) — the production fit_predict_ridge at
# alpha=0.0 on the empty-exog `baseline` bucket at har_lags=None, i.e. the
# UNTOUCHED production ladder code path (resolve_har_lags(), base-5 to
# 3125), through the run_executor invocation copied VERBATIM from the
# audited specs — reproduced LIVE here, never a number quoted from results
# tables on disk. Same index space as every arm (fixed 3125 burn-in), so
# per-bar losses align bar-for-bar.
# KNOWN-ANSWER CHECK (asserted): the swept (b=5, cap=3125) arm hands
# har_lags=[1,5,25,125,625,3125] through the NEW seam and must reproduce the
# har_lags=None incumbent EXACTLY — byte-identical ladder -> byte-identical
# features -> byte-identical rank-1 OLS. np.array_equal, no tolerance: any
# drift means the seam is not the gated no-op it claims to be.
# THE CITABLE TABLE: per arm (ladder x cap), the FULL beat4-writeup spread
# from src.evaluation.metrics.forecast_metrics (qlike primary, mse / rmse /
# mae, hmse / hmae, oos_r2 vs the incumbent, MZ calibration mz_beta /
# mz_r2, n) PLUS the Diebold-Mariano arm-vs-incumbent test on per-bar QLIKE
# losses (src.evaluation.diebold_mariano.qlike_per_bar + dm_test, HAC lag
# auto) — the SAME citable columns as the causal_tune_tree table, units
# from src, never inline; persisted to RESULTS_ROOT/metrics_table.csv
# (per-bar preds live in the executor results tables, kept per the
# persist-raw-outputs ruling). END STATE: pred_raw, true_raw,
# baseline_pred_raw — three aligned 1-D raw-variance arrays read from the
# executor's own Duan-reconstructed results tables — the explicit@960 arm
# (the user's named ladder) vs the base-5 anchor, per the module header.]
import numpy as np

from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.evaluation.metrics import forecast_metrics as _fm

incumbent_csv = os.path.join(RESULTS_ROOT, "incumbent_base5", "results.csv")
run_executor(
    method_name="ols_har_incumbent",
    fit_predict=fit_predict_ridge,
    hyperparams=dict(OLS_HP),
    data_path=DATA_PATH,
    output_file=incumbent_csv,
    horizon=HORIZON,
    train_window=TRAIN_WIN,
    start=START,
    end=END,
    halo=HALO,
    exog_cols=get_bucket(LADDER_BUCKET),
    segment=None,
    lag_scope="global",
    add_calendar=True,
    target_use_diurnal=True,
    target_winsor_window=240,
    dropna_with_exog=False,
    overnight_fill=True,
    impute_indicate=True,
    diurnal_mode="divide",
    prescale=True,
    seed=SEED,
    har_lags=None,  # the untouched production base-5 code path — the anchor
)
incumbent = pd.read_csv(incumbent_csv)
p_i = incumbent["pred_raw"].values

# KNOWN-ANSWER: the in-sweep (b=5, cap=3125) cell IS the incumbent ladder
# through the new seam — exact equality or the seam is not a gated no-op.
if "b5_c3125" in arm_results:
    ka = arm_results["b5_c3125"]["pred_raw"].values
    assert np.array_equal(ka, p_i), (
        "har_lags seam drift: (b=5, cap=3125) != har_lags=None incumbent"
    )
    print("known-answer PASS: b5_c3125 == incumbent (exact)")

rows = []
for label, arm in arm_results.items():
    p_t, t_t = arm["pred_raw"].values, arm["true_raw"].values
    m = _fm(p_t, t_t, benchmark=p_i)
    dm = dm_test(qlike_per_bar(p_t, t_t), qlike_per_bar(p_i, t_t), h=HORIZON)
    rows.append(dict(arm=label, rungs=len(arm_lags[label]),
                     ladder=" ".join(map(str, arm_lags[label])), **m,
                     incumbent_qlike=_fm(p_i, t_t)["qlike"],
                     dm=dm["dm"], dm_p=dm["p"], dm_mean_diff=dm["mean_diff"],
                     dm_better=dm.get("better", "")))
metrics_table = pd.DataFrame(rows)
os.makedirs(RESULTS_ROOT, exist_ok=True)
metrics_table.to_csv(os.path.join(RESULTS_ROOT, "metrics_table.csv"), index=False)
print(metrics_table.to_string(index=False))

contract_label = CONTRACT_ARM if CONTRACT_ARM in arm_results else next(iter(arm_results))
pred_raw = arm_results[contract_label]["pred_raw"].values
true_raw = arm_results[contract_label]["true_raw"].values
baseline_pred_raw = p_i
print(f"contract arrays = {contract_label} arm vs OLS-HAR base-5 anchor")

# %%
# hpc-audit-section: metrics
# [PINNED — copy verbatim; auto-clears GIVEN the interface contract.
# Claimed units computed by src/evaluation/metrics.py, never inline:
# QLIKE raw-space (primary), hmse/oos_r2 (level-robust), MZ calibration.
# A NameError below means the variable sections above were not filled —
# DELIBERATE: an unfilled draft must die here, never emit a vacuous report.]
from src.evaluation.metrics import forecast_metrics

report = forecast_metrics(pred_raw, true_raw, benchmark=baseline_pred_raw)
print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in report.items()})
