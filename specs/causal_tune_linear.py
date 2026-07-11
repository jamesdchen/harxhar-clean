"""causal_tune_linear — does CAUSAL hyperparameter tuning improve metrics?

PURPOSE: per-bucket, per-estimator test of causally optimized hyperparameters
against the fixed-HP incumbent. For each of the 9 data buckets
(src.data.loading.SUBGROUPS — including `baseline`, the empty-exog
no-buckets arm, and `all_features`) and each of THREE estimators run as
SEPARATE arms — ridge, reclasso (recursive lasso), reclasticnet (recursive
elastic net) — the penalty strength is re-selected PERIODICALLY (every
TUNE_PER solves) by a log-scale grid search scored on a leakage-clean
val_tail holdout carved from the trailing training window
(forward_window_split: fit block, EMBARGO gap, VAL_TAIL tail), so every OOS
forecast's hyperparameters were chosen without seeing its future. The
comparator is the FIXED-HP INCUMBENT: OLS on the no-bucket arm — the
production fit_predict_ridge at alpha=0.0 (RollingLeastSquares rank-1 is
exact OLS at zero penalty) on the empty-exog `baseline` bucket (HAR +
calendar only), ONE shared comparator for every bucket x estimator arm.

MACHINERY (reuse ruling): data prep, walk-forward dispatch, per-bucket scale
guards, and the raw-space Duan reconstruction ALL come from the production
scaffold src.backtest.executor.run_executor with the exact data-prep
invariants of src.models.ridge.run. The rolling-tuned fit_predict is the
verified rolling_linear_tune machinery specialized to ONE estimator kind per
arm: between tunings the solution slides WARM every bar — reclasso and
reclasticnet through TWO src.models.reclasso_har.enet_online calls
(Garrigues online homotopy, add + remove per bar, bit-exact), ridge through
the exact Sherman-Morrison rank-1 inverse update (the verified winablate_r1
every-bar machinery). Metrics: src.evaluation.metrics.forecast_metrics (the
full beat4-table spread — qlike / mse / rmse / mae / hmse / hmae / oos_r2 /
mz_beta / mz_r2 / n) plus the Diebold-Mariano tuned-vs-incumbent test on
per-bar QLIKE losses (src.evaluation.diebold_mariano.dm_test, HAC lag
auto).

SCOPE: bucket x estimator x chunk. 9 buckets x 3 estimators, backtest
chunks split across a task-array (chunks are the homogeneous axis).

PARALLELISM: cluster MAP-REDUCE through the hpc-agent submit pipeline. Map
axes are bucket x estimator x chunk: EXOG_BUCKET and ESTIMATOR select one
arm per task, and run_executor's start, end, halo — the SliceSpec chunk
seam fed via HPC_KW_* env — slice the walk-forward series per task. The
training window + VAL_TAIL holdout is warm-up HALO (BoundedHalo): halo-sized
rows are replayed without emitting, so a mid-series chunk reproduces the
whole-series arm exactly; prescale=True keeps chunks fungible. Each task
writes its results table plus the per-chunk reduce piece (save_chunk_reduce);
the framework's reduce combines the pieces. Local defaults (no env) = whole
series, all buckets x all estimators serially. The contract arrays
(pred_raw, true_raw, baseline_pred_raw) are the all_features arm of the
task's estimator (first estimator when running all).
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
# [VARIABLE — the causally-tuned CANDIDATE arms, one estimator kind per arm,
# run through the PRODUCTION scaffold: src.backtest.executor.run_executor
# does the data prep (load_and_transform, HAR + calendar build, burn-in
# drop, horizon shift, scale guards + prescale, Duan raw-space
# reconstruction, results table) with the EXACT invariants of
# src.models.ridge.run — nothing re-derived here. The pinned cells above
# demonstrate the loader + target; run_executor re-runs that same prep
# internally (the same functions). The ONLY new code: RollingTunedLinear, a
# fit_predict regressor exposing the engine's rank-1 rolling protocol
# (init_window + roll + solve) so MultiStageBacktest's incremental fast path
# drives it — the verified rolling_linear_tune machinery SPECIALIZED to one
# estimator per arm. Between tunings the solution slides WARM every bar:
# reclasso / reclasticnet through TWO src.models.reclasso_har.enet_online
# calls (Garrigues online homotopy: add the entering row, remove the leaving
# row; bit-exact, no fallback), ridge through the Sherman-Morrison rank-1
# update of the ridged inverse. Tuning is PERIODIC AND CAUSAL: every TUNE_PER
# solves, a leakage-clean forward split of the CURRENT training window
# (forward_window_split: fit block, EMBARGO gap, VAL_TAIL holdout tail —
# only rows inside the training window are ever read) scores the estimator's
# LOG-SCALE alpha grid — ridge closed-form, reclasso / reclasticnet via the
# FWL batch seed (enet_coef) — and the argmin alpha holds until the next
# tuning, whose COLD RESEED is also the exact re-anchor bounding float
# drift. The intercept rides as a locked, unpenalized augmented column.
# EMBARGO=25 covers the har_ma_25 window; longer HAR lags (125+) cannot be
# embargoed inside a 500-bar window — stated honestly, they are slow-moving
# levels. PARALLEL: cluster map-reduce — tasks map over bucket x estimator x
# chunk (EXOG_BUCKET + ESTIMATOR + the SliceSpec start, end, halo fed as
# HPC_KW_* env into run_executor's chunk seam; the training window +
# VAL_TAIL holdout is the warm-up halo, replayed without emitting; prescale
# keeps chunks fungible), the framework's chunk-reduce combines the
# per-chunk pieces save_chunk_reduce writes. Output tables land under
# RESULTS_ROOT — $HPC_RESULT_DIR under the dispatcher, a write target the
# executes-live lint cannot verify (no output_roots are recorded for this
# audit; disclosed, not masked).
# END STATE: per-bucket-per-estimator executor results tables + selection
# traces in `arm_results` for `baseline`.]
import numpy as np
import pandas as pd

from src.models.reclasso_har import enet_coef, enet_online, forward_window_split
from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.data.loading import SUBGROUPS, get_bucket
from src.features.transforms.residualizer import IdentityResidualizer

HORIZON = 1
TRAIN_WIN = 500        # deployed default (src.models.ridge.run train_window=500)
REFIT_FREQUENCY = 1    # coefficients re-solved EVERY BAR (rank-1 fast path)
TUNE_PER = 250         # penalty re-selected every 250 solves (the periodic cadence)
VAL_TAIL = 125         # forward validation holdout inside each training window
EMBARGO = 25           # kills har_ma_25 fit-to-val bleed (see section note)
SEED = 42

# One estimator kind per arm; alpha grids on a LOG SCALE (sklearn units:
# Ridge(alpha=...) for ridge, ElasticNet(alpha=..., l1_ratio=...) for the
# recursive lasso / elastic net).
ESTIMATOR_GRIDS: dict = {
    "ridge": [("ridge", float(a), 0.0) for a in np.logspace(-2, 3, 6)],
    "reclasso": [("lasso", float(a), 1.0) for a in np.logspace(-6, -2, 5)],
    "reclasticnet": [("enet", float(a), 0.5) for a in np.logspace(-6, -2, 5)],
}

# MAP-REDUCE seam: the hpc-agent dispatcher fans tasks out over bucket x
# estimator x chunk — resolve() kwargs arrive as HPC_KW_* env, and start,
# end, halo are run_executor's SliceSpec chunk fields. NOTE run_executor's
# train_window is in DAYS: TRAIN_WIN=500 days = 24000 bars, so a mid-series
# chunk's halo is 24000 warm-up bars (the canary-caught unit fix; the
# VAL_TAIL holdout lives INSIDE the training window, so the same halo covers
# it). Each task's save_chunk_reduce piece is combined by the framework's
# reduce. Local defaults (no env) = whole series, all buckets x estimators.
def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


EXOG_BUCKET = _env("EXOG_BUCKET", "")
ESTIMATOR = _env("ESTIMATOR", "")
START = int(_env("START", "0"))
END = int(_env("END", "-1"))
HALO = int(_env("HALO", "0"))
BUCKETS = [EXOG_BUCKET] if EXOG_BUCKET else list(SUBGROUPS)  # all 9, incl. the
# empty-exog `baseline` (no-buckets) arm and `all_features`
ESTIMATORS = [ESTIMATOR] if ESTIMATOR else list(ESTIMATOR_GRIDS)
CONTRACT_BUCKET = "all_features" if "all_features" in BUCKETS else BUCKETS[0]
CONTRACT_ESTIMATOR = ESTIMATORS[0]

RESULTS_ROOT = os.path.join(
    os.environ.get("HPC_RESULT_DIR", "results"), "causal_tune_linear"
)  # write target ($HPC_RESULT_DIR under the dispatcher)


def _batch_theta(Xa, yr, locked, a, l1):
    """Cold seed: FWL on the locked (intercept) block + enet_coef on the rest
    — the verified winablate_r1 batch_theta pattern, sklearn ElasticNet units."""
    exog = ~locked
    Hh, Ee = Xa[:, locked], Xa[:, exog]
    Hp = np.linalg.pinv(Hh)
    bHy, bHE = Hp @ yr, Hp @ Ee
    Eres, yres = Ee - Hh @ bHE, yr - Hh @ bHy
    bE = enet_coef(Eres.T @ Eres, Eres.T @ yres, len(yr), a, l1)
    th = np.zeros(Xa.shape[1])
    th[locked] = bHy - bHE @ bE
    th[exog] = bE
    return th


class RollingTunedLinear:
    """Rolling linear model, ONE estimator kind: periodic causal alpha
    re-selection with WARM rank-1 per-bar updates between tunings (the
    verified rolling_linear_tune / winablate_r1 every-bar machinery).

    Implements the MultiStageBacktest incremental protocol (init_window,
    roll, solve, predict_one). Between tunings every bar slides the solution
    exactly: reclasso / reclasticnet through TWO enet_online calls
    (Garrigues warm homotopy — add the entering row, remove the leaving
    row), ridge through the Sherman-Morrison rank-1 update of the ridged
    inverse. Every TUNE_PER solves, forward_window_split re-selects alpha
    from the arm's log-scale grid on the leakage-clean fit-EMBARGO-VAL_TAIL
    split of the current window (only rows inside the training window are
    ever read) and the warm state is COLD-RESEEDED on the full window — the
    exact re-anchor that also bounds float drift. The intercept rides as a
    locked, unpenalized augmented column.
    """

    reinit_every = 0  # drift handled by the cold reseed at every tuning
    grid: list = []   # the active arm's (kind, alpha, l1) log-scale grid
    trace: list = []  # class-level record of selections (evidence)

    def init_window(self, X_win, y_win):
        X = np.asarray(X_win, dtype=np.float64)
        self._X = np.hstack([X, np.ones((X.shape[0], 1))])  # augmented intercept
        self._y = np.asarray(y_win, dtype=np.float64).copy()
        self._ptr = 0
        self._n_solve = 0
        self._locked = np.zeros(self._X.shape[1], dtype=bool)
        self._locked[-1] = True  # intercept: unpenalized, never exits the active set
        self._tune()
        return self

    def _window(self):
        k = self._ptr
        return (
            np.vstack([self._X[k:], self._X[:k]]),
            np.concatenate([self._y[k:], self._y[:k]]),
        )

    def _seed(self):
        """Cold seed (also the exact re-anchor) of the warm state on the current window."""
        Xa, yr = self._window()
        tw = len(yr)
        kind, a, l1 = self.kind_, self.alpha_, self.l1_
        if kind == "ridge":
            D = np.where(self._locked, 0.0, a)  # sklearn Ridge units, free intercept
            G = Xa.T @ Xa
            G[np.diag_indices_from(G)] += D
            self._K = np.linalg.inv(G)
            self._c = Xa.T @ yr
            self._th = self._K @ self._c
        else:
            mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)  # sklearn ElasticNet units
            self._mu_vec = np.where(self._locked, 0.0, mu)
            Gr = Xa.T @ Xa
            ei = np.where(~self._locked)[0]
            Gr[ei, ei] += lam2
            self._Gr = Gr
            self._c = Xa.T @ yr
            th = _batch_theta(Xa, yr, self._locked, a, l1)
            self._A = list(np.where((np.abs(th) > 1e-9) | self._locked)[0])
            self._s = [0.0 if self._locked[j] else float(np.sign(th[j])) for j in self._A]
            self._th = th

    def roll(self, x_in, y_in, x_out, y_out):
        ua = np.append(np.asarray(x_in, dtype=np.float64), 1.0)
        ur = np.append(np.asarray(x_out, dtype=np.float64), 1.0)
        self._X[self._ptr] = ua
        self._y[self._ptr] = float(y_in)
        self._ptr = (self._ptr + 1) % len(self._y)
        if self.kind_ == "ridge":  # Sherman-Morrison: add entering, drop leaving
            Ku = self._K @ ua
            self._K -= np.outer(Ku, Ku) / (1.0 + ua @ Ku)
            self._c += ua * float(y_in)
            Kv = self._K @ ur
            self._K += np.outer(Kv, Kv) / (1.0 - ur @ Kv)
            self._c -= ur * float(y_out)
        else:  # warm Garrigues homotopy: one rank-1 data change per call
            self._th, self._A, self._s = enet_online(
                self._Gr, self._c, self._mu_vec, ua, float(y_in), +1.0,
                self._th, self._A, self._s, self._locked,
            )
            self._th, self._A, self._s = enet_online(
                self._Gr, self._c, self._mu_vec, ur, float(y_out), -1.0,
                self._th, self._A, self._s, self._locked,
            )

    def _tune(self):
        Xa, yr = self._window()
        n = len(yr)
        fit_lo, fit_hi, val_lo, val_hi = forward_window_split(n, n, VAL_TAIL, EMBARGO)
        Xf, yf = Xa[fit_lo:fit_hi], yr[fit_lo:fit_hi]
        Xv, yv = Xa[val_lo:val_hi], yr[val_lo:val_hi]
        best = None
        for kind, a, l1 in RollingTunedLinear.grid:
            if kind == "ridge":
                D = np.where(self._locked, 0.0, a)
                G = Xf.T @ Xf
                G[np.diag_indices_from(G)] += D
                th = np.linalg.solve(G, Xf.T @ yf)
            else:
                th = _batch_theta(Xf, yf, self._locked, a, l1)
            mse = float(np.mean((Xv @ th - yv) ** 2))
            if best is None or mse < best[0]:
                best = (mse, kind, a, l1)
        _, self.kind_, self.alpha_, self.l1_ = best
        RollingTunedLinear.trace.append((self.kind_, self.alpha_, self.l1_))
        self._seed()

    def solve(self):
        if self._n_solve > 0 and self._n_solve % TUNE_PER == 0:
            self._tune()
        elif self.kind_ == "ridge":
            self._th = self._K @ self._c
        self._n_solve += 1

    def predict_one(self, x):
        return float(np.append(np.asarray(x, dtype=np.float64), 1.0) @ self._th)


def fit_predict_lin_tuned(X_chunk, y_chunk, train_win_periods, hyperparams):
    """Mirrors src.models.ridge.fit_predict_ridge with the rolling-tuned
    wrapper as the regressor; the engine detects the rank-1 rolling protocol
    the wrapper exposes and takes its incremental fast path."""
    RollingTunedLinear.trace = []
    backtest = MultiStageBacktest(
        residualizer=IdentityResidualizer(),
        regressor_factory=RollingTunedLinear,
        refit_frequency=int(hyperparams.get("_refit_frequency", 1)),
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="lin_tuned")


arm_results: dict = {}
for estimator in ESTIMATORS:
    RollingTunedLinear.grid = ESTIMATOR_GRIDS[estimator]
    for bucket in BUCKETS:
        out_csv = os.path.join(RESULTS_ROOT, estimator, bucket, "results.csv")
        # the run_executor invocation of src.models.ridge.run with ONE
        # user-ruled deviation: impute_indicate=True + dropna_with_exog=False
        # — exog NaN are ffilled with availability-INDICATOR features instead
        # of dropping the leading edge, so every bucket keeps the SAME index
        # space (the bucket x estimator x chunk grid is bucket-invariant).
        # Other invariants unchanged: calendar on, diurnal target, winsor
        # 240, prescale=True (whose scale guards are exactly the
        # indicator-safe machinery).
        run_executor(
            method_name=f"lin_tuned_{estimator}",
            fit_predict=fit_predict_lin_tuned,
            hyperparams={"_refit_frequency": REFIT_FREQUENCY},
            data_path=DATA_PATH,
            output_file=out_csv,
            horizon=HORIZON,
            train_window=TRAIN_WIN,
            start=START,
            end=END,
            halo=HALO,
            exog_cols=get_bucket(bucket),
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
        )
        arm_results[(estimator, bucket)] = dict(
            results=pd.read_csv(out_csv), trace=list(RollingTunedLinear.trace)
        )
        alphas = [a for _, a, _ in arm_results[(estimator, bucket)]["trace"]]
        print(f"{estimator}/{bucket}: "
              f"{len(arm_results[(estimator, bucket)]['results'])} OOS rows; "
              f"{len(alphas)} tunings, alpha path "
              f"min={min(alphas):.2e} max={max(alphas):.2e}")

# %%
# hpc-audit-section: baseline
# [VARIABLE-BY-CONTENT, PINNED-BY-IDENTITY — will diff; human sign-off
# expected. DELIBERATE DEVIATION from the template prose (which cites
# configs/ridge_market_ew_prod.yaml): the comparator for the causal-tuning
# question is OLS ON THE NO-BUCKET ARM (user-ruled) — ONE incumbent shared
# by every bucket x estimator arm: the production fit_predict_ridge at
# alpha=0.0 (RollingLeastSquares rank-1 is EXACT OLS at zero penalty,
# per-bar refit) on the empty-exog `baseline` bucket (HAR + calendar only),
# through the IDENTICAL run_executor invocation, CALLED, never re-derived —
# reproduced LIVE here, never a number quoted from results tables on disk.
# The impute_indicate=True + dropna_with_exog=False invariants keep the
# incumbent on the SAME index space as every tuned arm (bucket-invariant
# grid), so the per-bar loss series align bar-for-bar.
# THE CITABLE TABLE: per bucket x estimator, the FULL beat4-writeup spread
# from src.evaluation.metrics.forecast_metrics (qlike primary, mse / rmse /
# mae, hmse / hmae, oos_r2 vs the incumbent, MZ calibration mz_beta /
# mz_r2, n) PLUS the Diebold-Mariano tuned-vs-incumbent test on per-bar
# QLIKE losses (src.evaluation.diebold_mariano.qlike_per_bar + dm_test,
# HAC lag auto) — units from src, never inline; the table is persisted to
# RESULTS_ROOT/metrics_table.csv (per-bar losses live in the executor
# results tables, kept per the persist-raw-outputs ruling). END STATE:
# pred_raw, true_raw, baseline_pred_raw — three aligned 1-D raw-variance
# arrays read from the executor's own Duan-reconstructed results tables —
# the CONTRACT_ESTIMATOR x CONTRACT_BUCKET arm, per the module header.]
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.evaluation.metrics import forecast_metrics as _fm
from src.models.ridge import fit_predict_ridge

# alpha=0.0 -> RollingLeastSquares(alpha=0.0): exact OLS, zero penalty
BASELINE_HP = dict(alpha=0.0, _refit_frequency=1, _incremental=True)
INCUMBENT_BUCKET = "baseline"  # the empty-exog no-bucket arm (HAR + calendar)

incumbent_csv = os.path.join(RESULTS_ROOT, "incumbent_ols", "results.csv")
run_executor(
    method_name="ols_incumbent",
    fit_predict=fit_predict_ridge,
    hyperparams=dict(BASELINE_HP),
    data_path=DATA_PATH,
    output_file=incumbent_csv,
    horizon=HORIZON,
    train_window=TRAIN_WIN,
    start=START,
    end=END,
    halo=HALO,
    exog_cols=get_bucket(INCUMBENT_BUCKET),
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
)
incumbent = pd.read_csv(incumbent_csv)
p_i = incumbent["pred_raw"].values

rows = []
for (estimator, bucket), r in arm_results.items():
    tuned = r["results"]
    p_t, t_t = tuned["pred_raw"].values, tuned["true_raw"].values
    m = _fm(p_t, t_t, benchmark=p_i)
    dm = dm_test(qlike_per_bar(p_t, t_t), qlike_per_bar(p_i, t_t), h=HORIZON)
    rows.append(dict(estimator=estimator, bucket=bucket, **m,
                     incumbent_qlike=_fm(p_i, t_t)["qlike"],
                     dm=dm["dm"], dm_p=dm["p"], dm_mean_diff=dm["mean_diff"],
                     dm_better=dm.get("better", "")))
metrics_table = pd.DataFrame(rows)
os.makedirs(RESULTS_ROOT, exist_ok=True)
metrics_table.to_csv(os.path.join(RESULTS_ROOT, "metrics_table.csv"), index=False)
print(metrics_table.to_string(index=False))

pred_raw = arm_results[(CONTRACT_ESTIMATOR, CONTRACT_BUCKET)]["results"]["pred_raw"].values
true_raw = arm_results[(CONTRACT_ESTIMATOR, CONTRACT_BUCKET)]["results"]["true_raw"].values
baseline_pred_raw = p_i
print(f"contract arrays = {CONTRACT_ESTIMATOR} x {CONTRACT_BUCKET} arm vs OLS no-bucket incumbent")

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
