"""Warm-start a campaign's Optuna study with the bespoke fleet's exported trials.

Imports each fleet seed as a COMPLETE trial (``add_trial`` + ``create_trial``) into
the campaign study ``.hpc/campaigns/<CID>/optuna.db`` (study ``camp_<MODEL>_<BUCKET>``),
attaching the single-source widened distributions so the seeds are in-distribution
and the campaign's TPE ``ask`` is immediately informed. Same objective (fit-once,
250d, COVID slice), so the values are directly usable. Idempotent-ish: skips seeds
whose params don't match the current space.

usage: python seed_campaign.py CAMPAIGN_ID MODEL BUCKET SEEDS.json [N_EST_CAP]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os
import sys

sys.path.insert(0, os.getcwd())
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.backtest.tree_space import widened_space
from src.backtest.tune_tree import _load_or_create_study

CID, MODEL, BUCKET, SEEDS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
N_EST_CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 300

N_SEED = int(os.environ.get("N_SEED", "1000"))
seeds = json.load(open(SEEDS))
# Warm-start only the good region: top-N by QLIKE. Seeding all ~20k trials is both
# slow to import (sqlite-on-Lustre fsync/trial) AND slows every subsequent TPE ask
# (it refits its KDEs on the whole set). Top-N points TPE straight at the optimum.
seeds = sorted(seeds, key=lambda s: s["value"])[:N_SEED]
dist = widened_space(MODEL, N_EST_CAP)
study = _load_or_create_study(MODEL, storage_path=f".hpc/campaigns/{CID}/optuna.db", study_name=f"camp_{MODEL}_{BUCKET}")
pre = len(study.get_trials(deepcopy=False))

added = skipped = 0
for s in seeds:
    p = dict(s["params"])
    if "n_estimators" in p:
        p["n_estimators"] = min(int(p["n_estimators"]), N_EST_CAP)
    if set(p) != set(dist):  # param keys must match the space exactly
        skipped += 1
        continue
    try:
        study.add_trial(
            optuna.trial.create_trial(
                params=p, distributions=dist, value=float(s["value"]),
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
        added += 1
    except Exception:
        skipped += 1

post = len(study.get_trials(deepcopy=False))
best = study.best_value if any(t.value is not None for t in study.get_trials(deepcopy=False)) else float("nan")
print("SEED %s %s/%s: +%d seeds (skipped %d), study %d -> %d trials, best=%.5f" % (
    CID, MODEL, BUCKET, added, skipped, pre, post, best), flush=True)
