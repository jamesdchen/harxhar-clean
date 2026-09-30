"""Worker-side jobs of specs/causal_tune_trees_tuned.py (importable by pool workers).

Two jobs, both pure functions of their arguments plus the arrays they name on
disk, so a process pool (spawn) can run them and the result does not depend on
which process ran them or how many there are:

  fit_candidate  one grid candidate on the fit block of a tuning window, scored
                 on its validation tail (MSE on the transformed target, QLIKE of
                 the executor's Duan back-transform), rounds early-stopped there
  refit_block    one refit of the configuration in force on the full trailing
                 window, the forecasts of its block of rows, native importance,
                 TreeSHAP (when asked) and the QLIKE-rule twin (when it differs)

EVERY FIT IS SINGLE-THREADED (MODEL_THREADS = 1): LightGBM's histogram sums
change with the thread count (the untuned canary, fitted at 4 threads, differs
from a 1-thread refit by up to 4 % relative), so parallelism is across processes
only and every number is independent of the core count.  At one thread the
constructor is exactly specs/causal_tune_trees.py's make_model() at
SLURM_CPUS_PER_TASK = 1 (the untuned GB fleet's setting; RF trees do not depend
on n_jobs).

The untuned spec's setup (params after leaf scaling, native_importance,
contributions) is executed once per process from its source (tree_setup), so
the untuned arm's code stays the only copy.

WINDOW MASK (job key "mask", the tree spec's WINDOW_MASK axis): a refit fits and
predicts on keep = src.models.window_mask.window_keep(X[t - W : t]) only, computed
here from the window the refit is trained on, and maps native importance and
TreeSHAP back to all p columns with window_mask.scatter (0 for a dropped column).
A candidate is fitted on the window the spec wrote for it, which carries the
tuning point's kept columns already (the mask of that window).  Without "mask"
(or False) every number is the unmasked one, bit for bit.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.window_mask import scatter, window_keep  # noqa: E402

TREE_SPEC = ROOT / "specs" / "causal_tune_trees.py"
_TREE_RUN_CELL = '\nout_csv = os.path.join(OUT_DIR, "results.csv")\n'
MODEL_THREADS = 1  # see the module note: results independent of the core count

_T: dict | None = None
_ARR: dict[str, np.ndarray] = {}


def tree_setup(refresh: bool = False) -> dict:
    """specs/causal_tune_trees.py up to its run cell, executed once per process.

    It reads the env axes (HPC_KW_MODEL, ...; a spawned worker inherits them),
    scales the leaf minimums to TRAIN_WIN and defines make_model /
    native_importance / contributions / fit_predict_tree; nothing is fitted.
    refresh=True re-reads the env (the spec calls it so; a worker process is
    spawned for one spec run and keeps its first read)."""
    global _T
    if _T is None or refresh:
        src = TREE_SPEC.read_text(encoding="utf-8")
        head, sep, _ = src.partition(_TREE_RUN_CELL)
        assert sep, f"{TREE_SPEC} no longer has its run cell marker {_TREE_RUN_CELL!r}"
        ns: dict = {"__file__": str(TREE_SPEC), "__name__": "causal_tune_trees_setup"}
        cwd = os.getcwd()
        exec(compile(head, str(TREE_SPEC), "exec"), ns)
        os.chdir(cwd)
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        _T = ns
    return _T


def arr(path: str) -> np.ndarray:
    """A float64 array saved by the spec, memory-mapped once per process."""
    if path not in _ARR:
        _ARR[path] = np.load(path, mmap_mode="r")
    return _ARR[path]


def build_model(cfg: dict, n_rounds: int | None):
    """The model of MODEL with the tuned axes overridden; cfg = the shipped
    config and n_rounds = None is make_model() at one thread."""
    T = tree_setup()
    model = T["MODEL"]
    if model == "lgbm":
        import lightgbm as lgb

        p = {**T["LGBM_PARAMS"], **cfg}
        if n_rounds is not None:
            p["n_estimators"] = int(n_rounds)
        return lgb.LGBMRegressor(**p, num_threads=MODEL_THREADS, random_state=T["SEED"], verbosity=-1)
    if model == "xgb":
        import xgboost as xgb

        p = {**T["XGB_PARAMS"], **cfg}
        if n_rounds is not None:
            p["n_estimators"] = int(n_rounds)
        return xgb.XGBRegressor(**p, n_jobs=MODEL_THREADS, random_state=T["SEED"], verbosity=0)
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        **{**T["RF_PARAMS"], **cfg}, random_state=T["SEED"], n_jobs=MODEL_THREADS
    )


def val_losses(pv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
    """(MSE on the transformed target, QLIKE of the Duan back-transform) on the tail."""
    from src.evaluation.metrics import apply_duan_smearing

    mse = float(np.mean((pv - yv) ** 2))
    ok = yv > 0  # the scorer's row filter (true_adj > 0)
    pr, tr = apply_duan_smearing(pv[ok], yv[ok], np.ones(int(ok.sum())))  # B cancels
    ratio = tr / pr
    return mse, float(np.mean(ratio - np.log(ratio) - 1.0))


def fit_candidate(job: dict) -> dict:
    """job: X / y = paths of the tuning WINDOW's arrays (the W rows before the
    forecast row, nothing else), fit / val = row ranges inside it, cfg,
    early_stop, patience, rounds_max."""
    a = time.time()
    Xw, yw = arr(job["X"]), arr(job["y"])
    (f0, f1), (v0, v1) = job["fit"], job["val"]
    Xf, yf = np.array(Xw[f0:f1]), np.array(yw[f0:f1])
    Xv, yv = np.array(Xw[v0:v1]), np.array(yw[v0:v1])
    model = tree_setup()["MODEL"]
    if job["early_stop"] and model == "lgbm":
        import lightgbm as lgb

        m = build_model(job["cfg"], job["rounds_max"])
        m.fit(
            Xf,
            yf,
            eval_set=[(Xv, yv)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(job["patience"], first_metric_only=True, verbose=False)],
        )
        rounds = int(m.best_iteration_)
        pv = m.predict(Xv, num_iteration=rounds)
    elif job["early_stop"] and model == "xgb":
        m = build_model(job["cfg"], job["rounds_max"])
        m.set_params(early_stopping_rounds=job["patience"], eval_metric="rmse")
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        rounds = int(m.best_iteration) + 1
        pv = m.predict(Xv, iteration_range=(0, rounds))
    else:
        m = build_model(job["cfg"], None)
        m.fit(Xf, yf)
        rounds = int(m.get_params()["n_estimators"])
        pv = m.predict(Xv)
    mse, qlk = val_losses(np.asarray(pv, dtype=np.float64), yv)
    return {"rounds": rounds, "val_mse": mse, "val_qlike": qlk, "sec": time.time() - a}


def refit_block(job: dict) -> dict:
    """job: X / y = paths of the chunk's arrays, t = the block's first row, W,
    k = rows in the block, cfg / rounds (the rule of record), qcfg / qrounds
    (the QLIKE rule; None = same as the rule of record), shap, mask (optional:
    True = the window mask; the result then carries keep, the kept columns)."""
    T = tree_setup()
    X, y = arr(job["X"]), arr(job["y"])
    t, W, k = job["t"], job["W"], job["k"]
    p = X.shape[1]
    Xw, yw = np.array(X[t - W : t]), np.array(y[t - W : t])
    Xb = np.array(X[t : t + k])
    keep = window_keep(Xw) if job.get("mask") else None
    if keep is not None:  # the columns this refit may use: a function of its window alone
        Xw, Xb = Xw[:, keep], Xb[:, keep]
    a = time.time()
    m = build_model(job["cfg"], job["rounds"])
    m.fit(Xw, yw)
    out: dict = {"fit_sec": time.time() - a, "keep": keep, "kept_n": Xw.shape[1]}
    out["preds"] = np.asarray(m.predict(Xb), dtype=np.float64)
    v = T["native_importance"](m, Xw.shape[1])
    out["importance"] = v if keep is None else scatter(v, keep, p)
    out["shap"], out["shap_sec"], out["gap"] = None, None, 0.0
    if job["shap"]:
        a = time.time()
        c = T["contributions"](m, Xb)
        out["shap_sec"] = time.time() - a
        out["gap"] = float(np.max(np.abs(c.sum(axis=1) - out["preds"])))
        out["shap"] = (c if keep is None else scatter(c, keep, p)).astype(np.float32)
    out["preds_q"], out["qsel_sec"] = out["preds"], None
    if job["qcfg"] is not None:
        a = time.time()
        mq = build_model(job["qcfg"], job["qrounds"])
        mq.fit(Xw, yw)
        out["preds_q"] = np.asarray(mq.predict(Xb), dtype=np.float64)
        out["qsel_sec"] = time.time() - a
    return out
