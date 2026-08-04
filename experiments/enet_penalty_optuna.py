"""Optuna FULL-OOS ORACLE CEILING for ridge / lasso / enet on the exog block (HAR held OLS).

*** DIAGNOSTIC ONLY -- NOT A REPORTABLE OOS NUMBER. ***
The objective IS full-OOS QLIKE, i.e. Optuna tunes AGAINST the test set on purpose. The results are
an UPPER BOUND on what penalty tuning could ever buy per penalty type -- test-set overfitting by
construction (slice-tuned winners already failed to transfer; see enet_penalty_sweep). Read it as:
"if even the oracle can't beat the fixed incumbent, penalty choice is a dead lever." Never quote
these as the model's out-of-sample performance.

Same base as enet_penalty_sweep (raw slim, 529 feats; 6 HAR mains OLS via FWL; exog penalized) so
the ceilings are comparable to that grid. NB the numbers sit BELOW deployed enetreg2 0.12314 because
the regime/cumrv add-ons are not included here -- the penalty *ranking* is the deliverable.

Run: /c/Users/james/miniconda3/envs/285J/python.exe enet_penalty_optuna.py   (resumable via SQLite)
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os
import sys
import time

import numpy as np
import optuna

sys.path.insert(0, os.getcwd())
from resid_amortized import PERIODS_PER_DAY, _cadence_enet_har_unpen  # noqa: E402
from src.evaluation.metrics import apply_duan_smearing  # noqa: E402

BUCKET = "all_buckets"
B = f"results/covid_imp_rank/{BUCKET}"
HAR = ("har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125")
OUT = "results/enet_penalty_optuna.json"
STORAGE = "sqlite:///results/enet_penalty_optuna.db"

# full-OOS eval settings (coarse refit -> tractable; oracle ceiling is a diagnostic, cadence-insensitive)
TW = 1000 * PERIODS_PER_DAY  # 48000
REFIT = 20_000
ITER, TOL = 1500, 1e-3
N_TRIALS = {"ridge": 12, "lasso": 12, "enet": 25}

# module globals (loaded once in main)
_X = _Y = _BASE = _HM = None


def _qlike(preds, y, base):
    pr, tr = apply_duan_smearing(preds, y, base)
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def _full_oos(alpha, l1, penalty):
    s, c, ic = _cadence_enet_har_unpen(
        _X,
        _Y,
        TW,
        REFIT,
        _HM,
        alpha=alpha,
        l1=l1,
        penalty=penalty,
        max_iter=ITER,
        tol=TOL,
    )
    n = len(_X)
    oos = np.empty(n - TW, dtype=np.float64)
    for i, t_r in enumerate(s):
        t_end = int(s[i + 1]) if i + 1 < len(s) else n
        oos[t_r - TW : t_end - TW] = _X[t_r:t_end] @ c[i] + ic[i]
    return _qlike(oos, _Y[TW:], _BASE[TW:])


def _make_obj(penalty):
    def obj(trial):
        if penalty == "ridge":
            a = trial.suggest_float("alpha", 0.05, 1000.0, log=True)
            l1 = None
        elif penalty == "lasso":
            a = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
            l1 = 1.0
        else:  # enet: l1 kept >=0.2 to avoid coordinate-descent's non-converging near-ridge corner
            a = trial.suggest_float("alpha", 3e-4, 1e-1, log=True)
            l1 = trial.suggest_float("l1", 0.2, 0.95)
        t0 = time.time()
        q = _full_oos(a, l1, penalty)
        trial.set_user_attr("secs", round(time.time() - t0, 1))
        print(
            f"  [{penalty}] trial {trial.number:2d} a={a:.2e} l1={l1} -> {q:.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        return q

    return obj


def main():
    global _X, _Y, _BASE, _HM
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    _HM = [f in set(HAR) for f in feats]
    assert sum(_HM) == 6, f"expected 6 HAR mains, got {sum(_HM)}"
    _X = np.ascontiguousarray(np.load(f"{B}/X_imp.npy"))
    _Y = np.load(f"{B}/y.npy").astype(np.float64)
    _BASE = np.load(f"{B}/base.npy").astype(np.float64)
    print(
        f"loaded X{_X.shape}  tw={TW} refit={REFIT} iter={ITER} tol={TOL}\n"
        "*** ORACLE (full-OOS objective) = DIAGNOSTIC CEILING ONLY, not reportable ***",
        flush=True,
    )

    # internally-consistent incumbent reference (same TW/REFIT/ITER/TOL as the studies)
    t0 = time.time()
    incumbent = _full_oos(1e-3, 0.2, "enet")
    print(
        f"incumbent enet a=1e-3 l1=0.2 -> {incumbent:.5f}  (reference, {time.time() - t0:.0f}s)",
        flush=True,
    )

    res = {}
    for penalty in ("ridge", "lasso", "enet"):
        print(f"\n=== {penalty} study ({N_TRIALS[penalty]} trials) ===", flush=True)
        study = optuna.create_study(
            direction="minimize",
            study_name=f"pen_{penalty}",
            storage=STORAGE,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(_make_obj(penalty), n_trials=N_TRIALS[penalty])
        res[penalty] = {
            "best_qlike": study.best_value,
            "best_params": study.best_params,
            "vs_incumbent": round(study.best_value - incumbent, 5),
            "n_trials": len(study.trials),
        }
        print(
            f"  BEST {penalty}: {study.best_value:.5f} params={study.best_params} "
            f"vs incumbent {study.best_value - incumbent:+.5f}",
            flush=True,
        )

    json.dump(
        {
            "objective": "FULL-OOS (oracle ceiling; DIAGNOSTIC ONLY; test-set-overfit by design; not reportable)",
            "base": "raw slim (no regime/cumrv add-ons); comparable to enet_penalty_sweep grid, below deployed enetreg2 0.12314",
            "incumbent_full_ref": incumbent,
            "tw": TW,
            "refit": REFIT,
            "results": res,
        },
        open(OUT, "w"),
        indent=2,
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
