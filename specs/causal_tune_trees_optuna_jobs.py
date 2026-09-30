"""Worker-side jobs of specs/causal_tune_trees_optuna.py (importable by pool workers).

One new job, plus the tuned spec's refit job reused unchanged:

  tune_point   ONE tuning point: a fresh Optuna study (TPESampler seeded by
               TPE_SEED_BASE + the point's whole-series OOS row, no warm start
               from any other point), N trials run SEQUENTIALLY in this
               process, trial 0 enqueued = the shipped configuration.  Each
               trial fits the model on the fit block of the window it was
               handed and scores the validation tail: validation MSE on the
               transformed target (the objective -- the linear arms'
               criterion), validation QLIKE of the Duan back-transform
               (recorded), early-stopped rounds (boosting), the round cap,
               seconds and the parameters.  The job carries the window's rows
               ONLY (the W rows before the forecast row, as arrays): the
               worker never sees a row at or after the forecast row.
  refit        specs/causal_tune_trees_tuned_jobs.refit_block, reused as is
               (one refit of a configuration on the full trailing window, the
               forecast of its block, native importance, TreeSHAP when asked);
               refit_masked = the same with the per-window column mask
               (WINDOW_MASK=1, user decision 2026-09-29: src/models/window_mask.py,
               the linear arms' identifiability rule -- drop the columns constant on
               the window and the exact copies of an earlier kept column).

EVERY FIT IS SINGLE-THREADED (causal_tune_trees_tuned_jobs.MODEL_THREADS = 1),
so every number is independent of the pool size and of the core count.  The
untuned spec's setup (the shipped params after the leaf-minimum scaling, the
model constructors) is executed once per process by
causal_tune_trees_tuned_jobs.tree_setup, so the untuned arm's code stays the
only copy; ``build_model`` / ``val_losses`` are the tuned spec's.

SEARCH SPACES (continuous; each covers the random-search grid of
specs/causal_tune_trees_tuned.py -- gated in experiments/gate_trees_optuna.py --
and contains the shipped configuration, which is trial 0):

  LightGBM  num_leaves int-log [2, 128]; min_child_samples int-log [1, 256];
            feature_fraction [0.05, 1]; bagging_fraction [0.3, 1] (freq 1 as
            shipped); lambda_l2 log [1e-3, 1e4]; learning_rate log [0.005, 0.1]
  XGBoost   max_depth int [1, 8]; min_child_weight log [0.1, 256];
            subsample [XGB_SUBSAMPLE_LO, 1]; colsample_bytree [0.05, 1];
            reg_lambda log [XGB_REG_LAMBDA_LO, 1e4]; learning_rate log [0.005, 0.1]
  RF        max_features [0.02, 1]; min_samples_leaf int-log [1, 256];
            max_depth categorical {2, 4, 8, 16, 32, None}; n_estimators 100
            (as shipped; more trees only lower variance)

Every other parameter stays at its shipped value (LightGBM lambda_l1,
bagging_freq; XGBoost reg_alpha, gamma, tree_method; random_state 42).

BOOSTING ROUNDS: early stopping on the validation tail (validation MSE) with
patience = ceil(1 / learning_rate) of the trial (the rounds whose shrunken steps
add up to one full step -- the tuned spec's rule, now per trial because the
learning rate is searched) and a cap that holds the TOTAL SHRINKAGE budget of
the tuned spec: rounds_max = ceil(ROUNDS_CAP_MULT x shipped rounds x shipped
learning rate / learning rate) -- exactly the tuned spec's cap (4 x the shipped
rounds) at the shipped learning rate, and proportionally more rounds at a
smaller one.  A best iteration at the cap is recorded (cap_hit).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import causal_tune_trees_tuned_jobs as J  # noqa: E402 -- tree_setup / build_model / val_losses / refit_block
from src.models.window_mask import scatter, window_keep  # noqa: E402

refit_block = (
    J.refit_block
)  # stage 2's job without the mask: the tuned spec's refit, unchanged

# ------------------------------------------------------------------ named constants
# XGBoost subsample: the brief's [0.3, 1] would cut the random-search grid's quarter
# step (0.25 x the shipped 0.609 = 0.152); the lower bound is set just below it so
# the continuous space covers the old grid (gated).
XGB_SUBSAMPLE_LO = 0.15
# XGBoost reg_lambda: the shipped value is 4.1e-4 (trial 0 must be the shipped
# configuration and the old grid starts there), below the brief's 1e-3; one decade
# lower keeps it inside a log-uniform axis.
XGB_REG_LAMBDA_LO = 1e-4
RF_MAX_DEPTH_CHOICES = (2, 4, 8, 16, 32, None)  # None = unbounded (sklearn's default)
# (kind, low, high, log) for numeric axes; ("cat", choices) for categorical ones.
# Order = the order of the suggest calls (it fixes the sampler's random stream).
SPACES: dict[str, dict[str, tuple]] = {
    "lgbm": {
        "num_leaves": ("int", 2, 128, True),
        "min_child_samples": ("int", 1, 256, True),
        "feature_fraction": ("float", 0.05, 1.0, False),
        "bagging_fraction": ("float", 0.3, 1.0, False),
        "lambda_l2": ("float", 1e-3, 1e4, True),
        "learning_rate": ("float", 0.005, 0.1, True),
    },
    "xgb": {
        "max_depth": ("int", 1, 8, False),
        "min_child_weight": ("float", 0.1, 256.0, True),
        "subsample": ("float", XGB_SUBSAMPLE_LO, 1.0, False),
        "colsample_bytree": ("float", 0.05, 1.0, False),
        "reg_lambda": ("float", XGB_REG_LAMBDA_LO, 1e4, True),
        "learning_rate": ("float", 0.005, 0.1, True),
    },
    "rf": {
        "max_features": ("float", 0.02, 1.0, False),
        "min_samples_leaf": ("int", 1, 256, True),
        "max_depth": ("cat", RF_MAX_DEPTH_CHOICES),
    },
}
# the tuned spec's round cap multiple (4 x the shipped rounds), held as a shrinkage budget
ROUNDS_CAP_MULT = 4
TPE_SEED_BASE = (
    20260929  # the day the Optuna campaign was designed; + the whole-series OOS row
)


def model_name() -> str:
    return J.tree_setup()["MODEL"]


def shipped_config(model: str | None = None) -> dict:
    """The shipped configuration on the searched axes (after the untuned spec's leaf scaling)."""
    T = J.tree_setup()
    model = model or T["MODEL"]
    if model == "rf":  # sklearn defaults, as specs/causal_tune_trees.py ships them
        return {"max_features": 1.0, "min_samples_leaf": 1, "max_depth": None}
    shipped = T["PARAMS"][model]
    return {a: shipped[a] for a in SPACES[model]}


def shipped_rounds(model: str | None = None) -> int | None:
    T = J.tree_setup()
    model = model or T["MODEL"]
    return None if model == "rf" else int(T["PARAMS"][model]["n_estimators"])


def patience_of(lr: float) -> int:
    """Rounds whose shrunken steps add up to one full step."""
    return math.ceil(1.0 / lr)


def rounds_cap_of(lr: float, model: str | None = None) -> int:
    """The tuned spec's cap (ROUNDS_CAP_MULT x shipped rounds) as a total-shrinkage budget."""
    T = J.tree_setup()
    model = model or T["MODEL"]
    p = T["PARAMS"][model]
    # (lr_shipped / lr) is exactly 1.0 at the shipped learning rate: the tuned spec's cap
    return math.ceil(ROUNDS_CAP_MULT * p["n_estimators"] * (p["learning_rate"] / lr))


def suggest(trial, space: dict) -> dict:
    cfg = {}
    for name, spec in space.items():
        if spec[0] == "int":
            cfg[name] = trial.suggest_int(name, spec[1], spec[2], log=spec[3])
        elif spec[0] == "float":
            cfg[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[3])
        else:
            cfg[name] = trial.suggest_categorical(name, list(spec[1]))
    return cfg


def encode(model: str, cfg: dict) -> np.ndarray:
    """cfg -> float64 vector in SPACES order (None -> NaN); exact for ints and floats."""
    return np.array(
        [np.nan if cfg[a] is None else float(cfg[a]) for a in SPACES[model]],
        dtype=np.float64,
    )


def decode(model: str, vec) -> dict:
    """The inverse of encode: the configuration exactly as the trial suggested it."""
    out: dict = {}
    for a, v in zip(SPACES[model], np.asarray(vec, dtype=np.float64)):
        kind = SPACES[model][a][0]
        if kind == "cat":
            out[a] = None if np.isnan(v) else int(v)
        elif kind == "int":
            out[a] = int(v)
        else:
            out[a] = float(v)
    return out


def fit_trial(Xf, yf, Xv, yv, cfg: dict, early_stop: bool) -> dict:
    """One trial: fit on the fit block, score the validation tail.

    The logic of causal_tune_trees_tuned_jobs.fit_candidate (same constructor,
    early-stopping calls and losses) on arrays held in memory, with the
    patience and cap of the trial's own learning rate."""
    a = time.time()
    model = model_name()
    patience: int | None = None
    rounds_max: int | None = None
    if early_stop and model in ("lgbm", "xgb"):
        lr = float(cfg["learning_rate"])
        patience, rounds_max = patience_of(lr), rounds_cap_of(lr)
    if early_stop and model == "lgbm":
        import lightgbm as lgb

        assert patience is not None

        m = J.build_model(cfg, rounds_max)
        m.fit(
            Xf,
            yf,
            eval_set=[(Xv, yv)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(patience, first_metric_only=True, verbose=False)
            ],
        )
        rounds = int(m.best_iteration_)
        pv = m.predict(Xv, num_iteration=rounds)
    elif early_stop and model == "xgb":
        m = J.build_model(cfg, rounds_max)
        m.set_params(early_stopping_rounds=patience, eval_metric="rmse")
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        rounds = int(m.best_iteration) + 1
        pv = m.predict(Xv, iteration_range=(0, rounds))
    else:
        m = J.build_model(cfg, None)
        m.fit(Xf, yf)
        rounds = int(m.get_params()["n_estimators"])
        pv = m.predict(Xv)
    mse, qlk = J.val_losses(np.asarray(pv, dtype=np.float64), yv)
    return {
        "rounds": rounds,
        "rounds_max": -1 if rounds_max is None else int(rounds_max),
        "patience": -1 if patience is None else int(patience),
        "val_mse": mse,
        "val_qlike": qlk,
        "sec": time.time() - a,
    }


def tune_point(job: dict) -> dict:
    """job: X / y = the tuning WINDOW's rows (the W rows before the forecast row,
    nothing else), fit / val = row ranges inside it, row = the whole-series OOS
    row of the forecast, seed, n_trials, early_stop, mask.

    mask (WINDOW_MASK=1): the kept columns are src.models.window_mask.window_keep of
    the FULL window [t - W, t), computed once per point; every trial's fit block and
    validation tail use those columns only (n_kept is recorded)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    a0 = time.time()
    model = model_name()
    space = SPACES[model]
    Xw, yw = job["X"], job["y"]
    (f0, f1), (v0, v1) = job["fit"], job["val"]
    assert len(Xw) == len(yw) and f0 == 0 and v1 == len(Xw), (
        len(Xw),
        job["fit"],
        job["val"],
    )
    if job.get("mask"):
        keep = window_keep(Xw)
        Xf = np.ascontiguousarray(np.asarray(Xw[f0:f1])[:, keep])
        Xv = np.ascontiguousarray(np.asarray(Xw[v0:v1])[:, keep])
        n_kept = len(keep)
    else:
        Xf, Xv, n_kept = (
            np.ascontiguousarray(Xw[f0:f1]),
            np.ascontiguousarray(Xw[v0:v1]),
            Xw.shape[1],
        )
    yf, yv = np.ascontiguousarray(yw[f0:f1]), np.ascontiguousarray(yw[v0:v1])
    sampler = optuna.samplers.TPESampler(seed=int(job["seed"]))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(shipped_config(model))
    recs: list[dict] = []

    def objective(trial) -> float:
        cfg = suggest(trial, space)
        r = fit_trial(Xf, yf, Xv, yv, cfg, bool(job["early_stop"]))
        recs.append(r | {"cfg": cfg})
        return r["val_mse"]

    study.optimize(
        objective, n_trials=int(job["n_trials"]), n_jobs=1, show_progress_bar=False
    )
    n = len(recs)
    assert n == int(job["n_trials"]) and len(study.trials) == n, (n, len(study.trials))
    states = [t.state.name for t in study.trials]
    return {
        "row": int(job["row"]),
        "seed": int(job["seed"]),
        "params": np.vstack([encode(model, r["cfg"]) for r in recs]),
        "val_mse": np.array([r["val_mse"] for r in recs]),
        "val_qlike": np.array([r["val_qlike"] for r in recs]),
        "rounds": np.array([r["rounds"] for r in recs], dtype=np.int64),
        "rounds_max": np.array([r["rounds_max"] for r in recs], dtype=np.int64),
        "patience": np.array([r["patience"] for r in recs], dtype=np.int64),
        "sec": np.array([r["sec"] for r in recs]),
        "complete": np.array([s == "COMPLETE" for s in states]),
        "study_sec": time.time() - a0,
        "pid": os.getpid(),
        "n_kept": int(n_kept),
    }


def refit_masked(job: dict) -> dict:
    """causal_tune_trees_tuned_jobs.refit_block with the per-window mask: the kept columns
    are window_keep of the refit's own window [t - W, t); the model is fitted on them and
    predicts the block's rows on them; native importance and TreeSHAP are mapped back to
    all p columns with window_mask.scatter (0 for a dropped column; TreeSHAP's last column,
    the expected value, carried over).  The QLIKE-rule twin of refit_block is not used here."""
    assert job.get("qcfg") is None, "refit_masked fits one configuration"
    T = J.tree_setup()
    X, y = J.arr(job["X"]), J.arr(job["y"])
    t, W, k = job["t"], job["W"], job["k"]
    p = X.shape[1]
    Xw, yw = np.array(X[t - W : t]), np.array(y[t - W : t])
    keep = window_keep(Xw)
    Xk = np.ascontiguousarray(Xw[:, keep])
    Xb = np.ascontiguousarray(np.array(X[t : t + k])[:, keep])
    a = time.time()
    m = J.build_model(job["cfg"], job["rounds"])
    m.fit(Xk, yw)
    out: dict = {"fit_sec": time.time() - a, "n_kept": int(len(keep))}
    out["preds"] = np.asarray(m.predict(Xb), dtype=np.float64)
    out["importance"] = scatter(T["native_importance"](m, len(keep)), keep, p)
    out["shap"], out["shap_sec"], out["gap"] = None, None, 0.0
    if job["shap"]:
        a = time.time()
        c = T["contributions"](m, Xb)
        out["shap_sec"] = time.time() - a
        out["gap"] = float(np.max(np.abs(c.sum(axis=1) - out["preds"])))
        out["shap"] = scatter(c, keep, p).astype(np.float32)
    out["preds_q"], out["qsel_sec"] = out["preds"], None
    return out
