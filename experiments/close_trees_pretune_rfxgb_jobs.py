"""Worker-side jobs of experiments/close_trees_pretune_rfxgb.py (importable by spawned pool workers).

fit_fold    one validation fold of one configuration: fit on the fold's fit block (2000 sessions),
            score the fold's validation rows.  Random forest: the spec's RandomForestRegressor
            (100 trees, sklearn defaults on the axes not searched), no early stopping -- exactly
            specs/causal_tune_trees_optuna_jobs.fit_trial's branch for a model without rounds.
            XGBoost: early stopping on the validation rows with the Optuna spec's patience
            ceil(1 / lr) and its shrinkage-budget round cap (fit_trial's XGBoost branch).  The
            column mask is src.models.window_mask.window_keep of the FIT BLOCK (the refit rule).
            Returns the validation MSE on the transformed target (the rule of record), the QLIKE
            of the Duan back-transform (recorded), the rounds and the squared errors of the
            validation rows (the retune's paired test uses them).
refit       specs/causal_tune_trees_optuna_jobs.refit_masked, reused unchanged (one refit on the
            trailing window [t - W, t) with the window mask, the forecasts of its block of rows).

Every fit is single-threaded (specs/causal_tune_trees_tuned_jobs.MODEL_THREADS = 1), so no number
depends on the pool size.  Model construction, the leaf-minimum scaling, the search space and the
round rules are the specs' (executed / imported from specs/), not copied.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import causal_tune_trees_optuna_jobs as OJ  # noqa: E402 -- round rules, refit_masked
import causal_tune_trees_tuned_jobs as J  # noqa: E402 -- tree_setup, build_model, val_losses, arr

from src.models.window_mask import window_keep  # noqa: E402


def fit_fold(job: dict) -> dict:
    """job: X / y = .npy paths (memory-mapped once per process), fit = (a, b) and val = (c, d) row
    ranges of those arrays (c >= b, checked), cfg = the searched axes (the others stay shipped)."""
    X, y = J.arr(job["X"]), J.arr(job["y"])
    a, b = job["fit"]
    c, d = job["val"]
    assert 0 <= a < b <= c < d <= len(y), (job["fit"], job["val"], len(y))
    Xw = np.asarray(X[a:b])
    keep = window_keep(Xw)
    Xf = np.ascontiguousarray(Xw[:, keep])
    Xv = np.ascontiguousarray(np.asarray(X[c:d])[:, keep])
    yf = np.ascontiguousarray(y[a:b])
    yv = np.ascontiguousarray(y[c:d])
    cfg = job["cfg"]
    model = J.tree_setup()["MODEL"]
    t0 = time.time()
    if model == "xgb":
        lr = float(cfg["learning_rate"])
        patience, cap = OJ.patience_of(lr), OJ.rounds_cap_of(lr)
        m = J.build_model(cfg, cap)
        m.set_params(early_stopping_rounds=patience, eval_metric="rmse")
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        rounds = int(m.best_iteration) + 1
        pv = m.predict(Xv, iteration_range=(0, rounds))
    elif model == "rf":
        patience, cap = -1, -1
        m = J.build_model(cfg, None)
        m.fit(Xf, yf)
        rounds = int(m.get_params()["n_estimators"])
        pv = m.predict(Xv)
    else:
        raise ValueError(f"random forest or XGBoost only, got {model!r}")
    pv = np.asarray(pv, dtype=np.float64)
    mse, qlk = J.val_losses(pv, yv)
    return {
        "val_mse": mse,
        "val_qlike": qlk,
        "rounds": rounds,
        "rounds_max": int(cap),
        "patience": int(patience),
        "sec": time.time() - t0,
        "n_kept": int(len(keep)),
        "e2": (pv - yv) ** 2,
    }


def refit(job: dict) -> dict:
    """The Optuna spec's stage-2 masked refit, unchanged (job keys: X, y, t, W, k, cfg, rounds,
    qcfg=None, shap=False); returns the forecasts and the fit seconds only."""
    out = OJ.refit_masked(job)
    return {"preds": out["preds"], "fit_sec": out["fit_sec"], "n_kept": out["n_kept"]}
