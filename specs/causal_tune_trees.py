"""causal_tune_trees -- per-bar TREE forecasts beside the per-bar linear arms.

PURPOSE: the professor's follow-up (2026-09-24): per-bar LightGBM, XGBoost and
random-forest forecasts on exactly the design the per-bar linear campaign fits
(specs/causal_tune_linear.py with SEGMENT = one regular-hours bar, LAG_SCOPE =
global, TRAIN_WIN = 2000 sessions), with TreeSHAP contributions persisted raw so
tree importance can later be compared with the linear models' contributions.

DESIGN (identical to the linear arms, not re-derived): the same
src.backtest.executor.run_executor call as the linear spec -- load_and_transform
(diurnal -> sqrt -> winsorize target, winsor 240), HAR ladder (production
powers of 5 unless HAR_LAGS / HAR_BASE are set) + calendar, impute_indicate=True
with dropna_with_exog=False (one index space for every bucket), prescale=True
(the rolling robust scaling the linear arms see -- the pooled tree bank of
2026-08 also fitted the prescaled linear design), SEGMENT slicing with global
lags, horizon 1, the executor's own results table (results_<seg>.csv, with its
look-ahead Duan pred_raw that nothing downstream uses; the stacker carries
pred_adj and the notebooks apply their own causal recalibration).

HYPERPARAMETERS (reused verbatim, not tuned here): the pooled tree forecasts in
the 15:30 deck (results/spxw_pnl/yhat_tree00.parquet = LightGBM,
yhat_tree16.parquet = XGBoost) are arms tree_expert_00 and tree_expert_16 of the
frozen 20-arm bank (src/unification.py::_walk_tree, experiments/tree_menu_dev.py).
The menu file experiments/tree_menu.json is lost, but every chunk npz of those
arms carries its config in the 'meta' string (tree_config), e.g.
harxhar-clean/results/unification_carc/tree_expert_00/chunk_000.npz -- the
params below are copied from there (lgbm_opt_00 sha abab811bd4b6, xgb_opt_06
sha 195b74503d9d).  Model construction mirrors _walk_tree exactly:
LGBMRegressor(**params, num_threads, random_state=42, verbosity=-1) and
XGBRegressor(**params, n_jobs, random_state=42, verbosity=0).  The random forest
has no precedent: sklearn RandomForestRegressor defaults (100 trees,
max_features=1.0, min_samples_leaf=1, bootstrap) plus random_state=42 and
n_jobs = the task's cores -- nothing else.

REFIT CADENCE: REFIT_EVERY = 10 rows of the one-bar series = 10 sessions.  The
precedent is the pooled bank's TREE_REFIT_EVERY=10 (jobs/sge/unification_tree.sge,
the Hoffman2 half of the bank; instrumented runs found 0 leaks at cadence 1 and
10).  Strictly causal: the model in force at row t was fitted on the 2000 rows
ending strictly before its refit anchor <= t (the MultiStageBacktest convention:
refit when i % REFIT_EVERY == 0 on X[t - W : t]); every row still gets its own
forecast.  A 10-session-stale model is 0.5 % of the 2000-session window.

PERSISTED RAW (trees_<seg>.npz beside results_<seg>.csv): date, pred_adj,
true_adj, baseline, true_raw, e2_adj (fit-space squared error), refit_row,
per-refit native importance (LightGBM gain, XGBoost total_gain, RF impurity
decrease), per-refit fit / SHAP seconds, feature names, params, library
versions, and -- for SEGMENT in SHAP_SEGMENTS (default bar1600) -- the TreeSHAP
matrix of every out-of-sample row under the model in force (LightGBM
pred_contrib=True, XGBoost pred_contribs=True, RF shap.TreeExplainer; last
column = the expected value), float32, with the max additivity gap recorded.

ENV AXES (HPC_KW_<name> or <name>): MODEL (lgbm | xgb | rf), EXOG_BUCKET
(all_features | baseline | live_feasible | ...), SEGMENT (bar1000 .. bar1600),
TRAIN_WIN (days, default 2000), LAG_SCOPE (default global), HAR_LAGS / HAR_BASE
(default production ladder), START / END / HALO (smoke slicing; default whole
series), SHAP_SEGMENTS (comma list; default bar1600).  Output root:
$HPC_RESULT_DIR/causal_tune_trees/<model>/<bucket>/.
"""

# %%
import sys
from pathlib import Path

import os

_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _START
while not (_ROOT / "src").is_dir() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if (_ROOT / "src").is_dir():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)

# %%
import json
import time
import warnings

import numpy as np
import pandas as pd

from src.backtest.executor import run_executor
from src.data.loading import get_bucket
from src.features.extractors.har import resolve_har_lags

# LightGBM names numpy columns Column_i at fit; predicting on a bare array
# then trips sklearn's feature-name check on every block -- harmless, silenced.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

DATA_PATH = "data"
HORIZON = 1
SEED = 42
REFIT_EVERY = 10  # sessions between refits (the pooled bank's TREE_REFIT_EVERY=10)

# tree_expert_00 (LightGBM) and tree_expert_16 (XGBoost): the configs behind
# yhat_tree00 / yhat_tree16, copied from the chunk npz 'meta' of those arms.
LGBM_PARAMS: dict = {
    "num_leaves": 20,
    "learning_rate": 0.01296404565731986,
    "n_estimators": 392,
    "min_child_samples": 98,
    "feature_fraction": 0.7017000121697539,
    "bagging_fraction": 0.5636236714089857,
    "lambda_l1": 1.0413823540826713e-08,
    "lambda_l2": 0.015950303832251992,
    "bagging_freq": 1,
}
XGB_PARAMS: dict = {
    "max_depth": 4,
    "learning_rate": 0.010853165384309853,
    "n_estimators": 339,
    "min_child_weight": 24.86134158090589,
    "subsample": 0.6086526390529718,
    "colsample_bytree": 0.6975888053713192,
    "reg_alpha": 1.087850849987857,
    "reg_lambda": 0.00041206887024929935,
    "gamma": 1.1082753230505144e-05,
    "tree_method": "hist",
}
RF_PARAMS: dict = {"n_estimators": 100}  # the sklearn default, stated
PARAMS = {"lgbm": LGBM_PARAMS, "xgb": XGB_PARAMS, "rf": RF_PARAMS}
PROVENANCE = {
    "lgbm": "tree_expert_00 / lgbm_opt_00 sha abab811bd4b6 (chunk npz meta)",
    "xgb": "tree_expert_16 / xgb_opt_06 sha 195b74503d9d (chunk npz meta)",
    "rf": "no precedent: sklearn RandomForestRegressor defaults",
}


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


MODEL = _env("MODEL", "lgbm")
if MODEL not in PARAMS:
    raise SystemExit(f"MODEL must be one of {sorted(PARAMS)}, got {MODEL!r}")
EXOG_BUCKET = _env("EXOG_BUCKET", "baseline")
SEGMENT = _env("SEGMENT", "bar1600")
if not SEGMENT.startswith("bar"):
    raise SystemExit(f"per-bar trees only: SEGMENT must be a one-bar segment, got {SEGMENT!r}")
LAG_SCOPE = _env("LAG_SCOPE", "global")
TRAIN_WIN = int(_env("TRAIN_WIN", "2000"))
# The leaf minimums are row counts tuned on the pooled bank's window
# (window_bars 24000 in the tree_expert_00 and tree_expert_16 chunk meta); a
# per-bar window holds TRAIN_WIN rows, one per session, so each is scaled to keep
# the same share of the window per leaf. LightGBM's min_child_samples is an
# integer (98 -> 8 at TRAIN_WIN = 2000); XGBoost's min_child_weight is a sum of
# hessians, one per row under squared error, and stays continuous (24.86 -> 2.07).
POOLED_WINDOW_ROWS = 24000
_leaf_scale = TRAIN_WIN / POOLED_WINDOW_ROWS
POOLED_MIN_CHILD_SAMPLES = LGBM_PARAMS["min_child_samples"]
LGBM_PARAMS["min_child_samples"] = max(1, round(POOLED_MIN_CHILD_SAMPLES * _leaf_scale))
PROVENANCE["lgbm"] += (
    f"; min_child_samples scaled {POOLED_MIN_CHILD_SAMPLES} -> "
    f"{LGBM_PARAMS['min_child_samples']} for the {TRAIN_WIN}-row per-bar window"
)
POOLED_MIN_CHILD_WEIGHT = XGB_PARAMS["min_child_weight"]
XGB_PARAMS["min_child_weight"] = POOLED_MIN_CHILD_WEIGHT * _leaf_scale
PROVENANCE["xgb"] += (
    f"; min_child_weight scaled {POOLED_MIN_CHILD_WEIGHT:.4g} -> "
    f"{XGB_PARAMS['min_child_weight']:.4g} for the {TRAIN_WIN}-row per-bar window"
)
START = int(_env("START", "0"))
END = int(_env("END", "-1"))
HALO = int(_env("HALO", "0"))
SHAP_SEGMENTS = {s for s in _env("SHAP_SEGMENTS", "bar1600").split(",") if s}
WANT_SHAP = SEGMENT in SHAP_SEGMENTS
_HAR_LAGS_ENV = _env("HAR_LAGS", "")
_HAR_BASE_ENV = _env("HAR_BASE", "")
if _HAR_LAGS_ENV:
    HAR_LAGS: list[int] | None = sorted(
        {int(v) for v in _HAR_LAGS_ENV.replace(";", ",").split(",") if v.strip()}
    )
elif _HAR_BASE_ENV:
    HAR_LAGS = resolve_har_lags(base=int(_HAR_BASE_ENV))
else:
    HAR_LAGS = None
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

RESULTS_ROOT = os.path.join(os.environ.get("HPC_RESULT_DIR", "results"), "causal_tune_trees")
OUT_DIR = os.path.join(RESULTS_ROOT, MODEL, EXOG_BUCKET)
print(
    f"trees: model={MODEL} bucket={EXOG_BUCKET} segment={SEGMENT} lag_scope={LAG_SCOPE} "
    f"tw={TRAIN_WIN} refit_every={REFIT_EVERY} threads={N_THREADS} shap={WANT_SHAP} "
    f"har={HAR_LAGS if HAR_LAGS is not None else 'production'} slice=({START},{END},{HALO})"
)


# %%
def make_model():
    """The model of MODEL, built exactly as src/unification.py::_walk_tree does."""
    if MODEL == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            **LGBM_PARAMS, num_threads=N_THREADS, random_state=SEED, verbosity=-1
        )
    if MODEL == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(**XGB_PARAMS, n_jobs=N_THREADS, random_state=SEED, verbosity=0)
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**RF_PARAMS, random_state=SEED, n_jobs=N_THREADS)


def native_importance(model, p: int) -> np.ndarray:
    if MODEL == "lgbm":
        return model.booster_.feature_importance(importance_type="gain").astype(np.float64)
    if MODEL == "xgb":
        score = model.get_booster().get_score(importance_type="total_gain")
        out = np.zeros(p)
        for k, v in score.items():
            out[int(k[1:])] = v
        return out
    return np.asarray(model.feature_importances_, dtype=np.float64)


def contributions(model, X: np.ndarray) -> np.ndarray:
    """TreeSHAP (path-dependent, exact) per row; last column = the expected value."""
    if MODEL == "lgbm":
        return np.asarray(model.predict(X, pred_contrib=True), dtype=np.float64)
    if MODEL == "xgb":
        import xgboost as xgb

        return np.asarray(
            model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True), dtype=np.float64
        )
    import shap

    ex = shap.TreeExplainer(model)
    sv = np.asarray(ex.shap_values(X, check_additivity=True), dtype=np.float64)
    ev = float(np.ravel(ex.expected_value)[0])
    return np.column_stack([sv, np.full(len(X), ev)])


SIDE: dict = {}


def fit_predict_tree(X_chunk, y_chunk, train_win_periods, hyperparams):
    """Walk-forward on the one-bar series: refit every REFIT_EVERY rows on the
    trailing train_win_periods rows (strictly before the anchor), predict each
    block of rows with the model in force; TreeSHAP on the same rows."""
    X = np.ascontiguousarray(X_chunk, dtype=np.float64)
    y = np.ascontiguousarray(y_chunk, dtype=np.float64)
    W = int(train_win_periods)
    n_test = len(X) - W
    p = X.shape[1]
    preds = np.empty(n_test)
    shap_mat = np.full((n_test, p + 1), np.nan, dtype=np.float32) if WANT_SHAP else None
    refit_row, imp, fit_sec, shap_sec, add_gap = [], [], [], [], 0.0
    t0 = time.time()
    for i in range(0, n_test, REFIT_EVERY):
        t = W + i
        k = min(REFIT_EVERY, n_test - i)
        a = time.time()
        model = make_model()
        model.fit(X[t - W : t], y[t - W : t])
        fit_sec.append(time.time() - a)
        refit_row.append(i)
        imp.append(native_importance(model, p))
        Xb = X[t : t + k]
        preds[i : i + k] = model.predict(Xb)
        if WANT_SHAP:
            a = time.time()
            c = contributions(model, Xb)
            shap_sec.append(time.time() - a)
            add_gap = max(add_gap, float(np.max(np.abs(c.sum(axis=1) - preds[i : i + k]))))
            shap_mat[i : i + k] = c
        if len(refit_row) % 25 == 0:
            print(
                f"  refit {len(refit_row)}/{-(-n_test // REFIT_EVERY)}  "
                f"fit {np.mean(fit_sec):.2f}s  shap {np.mean(shap_sec) if shap_sec else 0:.2f}s  "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
    SIDE.update(
        feature_names=np.array(
            [str(c) for c in hyperparams.get("_feature_names", [f"f{j}" for j in range(p)])]
        ),
        refit_row=np.array(refit_row, dtype=np.int64),
        importance=np.array(imp, dtype=np.float32),
        fit_sec=np.array(fit_sec),
        shap_sec=np.array(shap_sec),
        shap=shap_mat,
        shap_additivity_gap=add_gap,
        n_features=p,
        train_rows=W,
        wall_sec=time.time() - t0,
    )
    if WANT_SHAP:
        print(f"  TreeSHAP additivity gap (max |sum contrib - pred|) = {add_gap:.2e}")
    return preds


# %%
out_csv = os.path.join(OUT_DIR, "results.csv")
out_read = os.path.join(OUT_DIR, f"results_{SEGMENT}.csv")
t_run = time.time()
run_executor(
    method_name=f"trees_{MODEL}",
    fit_predict=fit_predict_tree,
    hyperparams={},
    data_path=DATA_PATH,
    output_file=out_csv,
    horizon=HORIZON,
    train_window=TRAIN_WIN,
    start=START,
    end=END,
    halo=HALO,
    exog_cols=get_bucket(EXOG_BUCKET),
    segment=SEGMENT,
    lag_scope=LAG_SCOPE,
    har_lags=HAR_LAGS,
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
res = pd.read_csv(out_read, parse_dates=["date"])
n_oos = len(res)
assert len(SIDE["refit_row"]) == -(-n_oos // REFIT_EVERY), (len(SIDE["refit_row"]), n_oos)
assert SIDE["shap"] is None or len(SIDE["shap"]) == n_oos, (len(SIDE["shap"]), n_oos)
ok = (res["true_adj"] > 0) & (res["true_raw"] > 0)
baseline = np.where(ok, res["true_raw"] / res["true_adj"].where(ok) ** 2, np.nan)

versions = {}
for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "shap"):
    try:
        versions[mod] = __import__(mod).__version__
    except Exception:  # noqa: BLE001 -- a library the model does not use may be absent
        versions[mod] = "absent"

npz_path = os.path.join(OUT_DIR, f"trees_{SEGMENT}.npz")
np.savez_compressed(
    npz_path,
    date=res["date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype("U19"),
    pred_adj=res["pred_adj"].to_numpy(float),
    true_adj=res["true_adj"].to_numpy(float),
    true_raw=res["true_raw"].to_numpy(float),
    baseline=baseline,
    e2_adj=((res["true_adj"] - res["pred_adj"]) ** 2).to_numpy(float),
    refit_row=SIDE["refit_row"],
    importance=SIDE["importance"],
    fit_sec=SIDE["fit_sec"],
    shap_sec=SIDE["shap_sec"],
    feature_names=SIDE["feature_names"],
    shap=SIDE["shap"] if SIDE["shap"] is not None else np.zeros((0, 0), np.float32),
    shap_additivity_gap=SIDE["shap_additivity_gap"],
    meta=json.dumps(
        {
            "model": MODEL,
            "params": PARAMS[MODEL],
            "provenance": PROVENANCE[MODEL],
            "bucket": EXOG_BUCKET,
            "segment": SEGMENT,
            "lag_scope": LAG_SCOPE,
            "train_win_days": TRAIN_WIN,
            "train_rows": SIDE["train_rows"],
            "refit_every": REFIT_EVERY,
            "threads": N_THREADS,
            "seed": SEED,
            "har_lags": HAR_LAGS,
            "slice": [START, END, HALO],
            "n_features": SIDE["n_features"],
            "shap": WANT_SHAP,
            "versions": versions,
            "walk_sec": SIDE["wall_sec"],
            "run_sec": time.time() - t_run,
        }
    ),
)
e2 = (res["true_adj"] - res["pred_adj"]) ** 2
print(
    f"wrote {out_read} ({n_oos} OOS rows, {res['date'].min()} .. {res['date'].max()}) and {npz_path}; "
    f"{len(SIDE['refit_row'])} refits, fit {SIDE['fit_sec'].mean():.2f}s/refit, "
    f"SHAP {SIDE['shap_sec'].mean() if len(SIDE['shap_sec']) else 0:.2f}s/refit, "
    f"walk {SIDE['wall_sec']:.0f}s, run {time.time() - t_run:.0f}s; "
    f"fit-space MSE {e2.mean():.5f}, p={SIDE['n_features']}"
)
