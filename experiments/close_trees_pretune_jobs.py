"""Worker-side jobs of experiments/close_trees_pretune.py (importable by spawned pool workers).

fit_fold    one validation fold: fit the configuration on the fold's fit block (2000 sessions),
            early-stopped on the fold's validation rows exactly as the Optuna spec's trial
            (specs/causal_tune_trees_optuna_jobs.fit_trial: patience ceil(1 / lr), round cap =
            the tuned spec's 4 x shipped shrinkage budget), scored by validation MSE on the
            transformed target (the rule of record) and the QLIKE of the Duan back-transform
            (recorded).  The column mask is src.models.window_mask.window_keep of the FIT BLOCK
            (the rows the model is fitted on, the refit rule).  Returns the squared errors of the
            validation rows too (the retune's paired test uses them).
refit       specs/causal_tune_trees_optuna_jobs.refit_masked, reused as is (one refit on the
            full trailing window [t - W, t) with the window mask, the forecasts of its block).

Every fit is single-threaded (specs/causal_tune_trees_tuned_jobs.MODEL_THREADS = 1), so no number
depends on the pool size.  The model constructors, the leaf-minimum scaling, the search space
and the round rules are the specs' (executed / imported from specs/), not copied.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import causal_tune_trees_optuna_jobs as OJ  # noqa: E402 -- spaces, round rules, refit_masked
import causal_tune_trees_tuned_jobs as J  # noqa: E402 -- build_model, val_losses, arr

from src.models.window_mask import window_keep  # noqa: E402


def fit_fold(job: dict) -> dict:
    """job: X / y = .npy paths (memory-mapped once per process), fit = (a, b) and val = (c, d)
    row ranges of those arrays (c >= b + embargo, checked by the caller), cfg = the searched
    axes (the rest of the parameters stay at the shipped values)."""
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
    lr = float(cfg["learning_rate"])
    patience, cap = OJ.patience_of(lr), OJ.rounds_cap_of(lr)
    t0 = time.time()
    m = J.build_model(cfg, cap)
    if model == "lgbm":
        import lightgbm as lgb

        m.fit(
            Xf,
            yf,
            eval_set=[(Xv, yv)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
        )
        rounds = int(m.best_iteration_)
        pv = m.predict(Xv, num_iteration=rounds)
    elif model == "xgb":
        m.set_params(early_stopping_rounds=patience, eval_metric="rmse")
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        rounds = int(m.best_iteration) + 1
        pv = m.predict(Xv, iteration_range=(0, rounds))
    else:
        raise ValueError(f"boosting models only, got {model!r}")
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
    qcfg=None, shap=False)."""
    return OJ.refit_masked(job)
