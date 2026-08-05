"""Seed the residual study with the heavily-regularized 'bagging' configs that worked.

Reads the existing all_buckets slice study (best = lowest QLIKE = the bagging-ward configs),
ENQUEUEs the top-N into the new residual study so regular TPE evaluates them FIRST under the
new full-OOS residualized objective. enqueue (not add_trial) — the objective changed, so we
re-evaluate the configs, not trust their old slice values.

usage: python seed_resid.py CID MODEL BUCKET SEED_DB SEED_STUDY N
"""

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)
import os
import sys

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.getcwd())
from src.backtest.tree_space import widened_space

CID, MODEL, BUCKET = sys.argv[1], sys.argv[2], sys.argv[3]
SEED_DB, SEED_STUDY, N = sys.argv[4], sys.argv[5], int(sys.argv[6])
optuna.logging.set_verbosity(optuna.logging.WARNING)

space = widened_space(MODEL)
src = optuna.load_study(study_name=SEED_STUDY, storage=f"sqlite:///{SEED_DB}")
trials = sorted((t for t in src.get_trials(deepcopy=False) if t.value is not None), key=lambda t: t.value)
os.makedirs(f".hpc/campaigns/{CID}", exist_ok=True)
dst = optuna.create_study(
    study_name=f"resid_{MODEL}_{BUCKET}", storage=f"sqlite:///.hpc/campaigns/{CID}/optuna.db",
    sampler=TPESampler(constant_liar=False, n_ei_candidates=96, seed=0), direction="minimize", load_if_exists=True,
)
n = 0
for t in trials[:N]:
    p = {k: t.params[k] for k in space if k in t.params}
    if len(p) == len(space):
        dst.enqueue_trial(p, skip_if_exists=True); n += 1
print(f"SEED enqueued {n} bagging configs into resid_{MODEL}_{BUCKET} (from {SEED_STUDY})", flush=True)
