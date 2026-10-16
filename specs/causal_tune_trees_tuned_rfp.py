"""causal_tune_trees_tuned_rfp -- the random-search rungs (RS10 / RS1) of the per-bar random forest re-centred on a
new shipped forest (2026-10-07: the dev-span search's trial 163, speculative; user: "max capacity").

Runs specs/causal_tune_trees_tuned.py with two changes, both made here and documented:

1. THE SHIPPED FOREST.  specs/causal_tune_trees_tuned_jobs.py executes the untuned spec's setup from
   TREE_SPEC = specs/causal_tune_trees.py (sklearn-default forest).  This wrapper points that module attribute at
   a generated copy whose only difference is the RF_PARAMS line (HPC_KW_TREE_SPEC, default
   specs/causal_tune_trees_rfp163.py, written by experiments/trees_rfp163_make_specs.py from
   cluster/rfprev163_rf_config.json with the leaf minimum scaled to the per-bar window: n_estimators 355,
   max_depth 14, min_samples_leaf 2, max_features 0.7356, max_samples 0.8811, bootstrap).  The patch is made at
   this file's TOP LEVEL, so the spawned pool workers (multiprocessing re-runs the main script as __mp_main__ in
   every child before it unpickles a job) see the same TREE_SPEC.

2. THE FOREST'S GRID, re-centred on the new shipped values the way the LightGBM / XGBoost grids are centred on
   theirs (the shipped value on the grid, the same number of named choices as the default forest's grid):
     max_features      0.25 / 0.5 / 1 x the shipped fraction and the bound 1.0 (the FRACTION_STEPS rule of
                       LightGBM's feature_fraction / XGBoost's colsample_bytree)  -> 0.1839, 0.3678, 0.7356, 1.0
                       (default forest: 1/27, 1/9, 1/3, 1.0 -- 4 choices)
     max_depth         the shipped depth halved twice, the shipped depth, unbounded (the default forest's axis
                       halved from its shipped unbounded: 4 / 8 / 16 / None)  -> 4, 7, 14, None
     min_samples_leaf  the row ladder 1 .. 64 from the bound 1 (unchanged: it already holds the shipped 2)
   and the forest's shipped point on the grid = the shipped values (as for LightGBM / XGBoost) instead of the
   sklearn defaults.  112 configurations; the 32 candidates = the shipped point + 31 drawn with the spec's
   GRID_SEED, as before.  n_estimators, max_samples and bootstrap stay at the shipped values (not tuned, as
   n_estimators for the default forest).
Everything else is the tuned spec's own code, executed from its source (the substitutions are asserted to hit
exactly once).  TUNE_IDENTITY=1 reduces the candidates to the shipped forest: the run must then reproduce the
fixed-settings forest (specs/causal_tune_trees_rfdev.py with the same configuration) bit for bit
(experiments/trees_rfp163_tuned_gate.py).

ENV AXES: those of specs/causal_tune_trees_tuned.py (MODEL must be rf) + TREE_SPEC.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import causal_tune_trees_tuned_jobs as J  # noqa: E402

TREE_SPEC = HERE / os.environ.get(
    "HPC_KW_TREE_SPEC", os.environ.get("TREE_SPEC", "causal_tune_trees_rfp163.py")
)
if not TREE_SPEC.is_file():
    raise SystemExit(
        f"TREE_SPEC {TREE_SPEC} missing (experiments/trees_rfp163_make_specs.py writes it)"
    )
J.TREE_SPEC = TREE_SPEC  # parent and every spawned worker: the shipped forest of the generated spec

TUNED = HERE / "causal_tune_trees_tuned.py"
SUBS = (
    (
        "RF_MAX_FEATURES = (1.0 / 27.0, 1.0 / 9.0, 1.0 / 3.0, 1.0)",
        "RF_MAX_FEATURES = tuple(sorted({SHIPPED['max_features'] * f for f in FRACTION_STEPS} | {FRACTION_BOUND}))"
        "  # rfp: re-centred on the shipped fraction",
    ),
    (
        "RF_MAX_DEPTH = (4, 8, 16, None)",
        "RF_MAX_DEPTH = (*sorted({max(1, round(SHIPPED['max_depth'] * f)) for f in (0.25, 0.5, 1.0)}), None)"
        "  # rfp: the shipped depth halved twice, itself, unbounded",
    ),
    (
        'return {"max_features": 1.0, "min_samples_leaf": 1, "max_depth": None}  # sklearn defaults',
        "return {a: SHIPPED[a] for a in axes}  # rfp: the new shipped forest's values (as LightGBM / XGBoost)",
    ),
)


def main() -> None:
    model = os.environ.get("HPC_KW_MODEL", os.environ.get("MODEL", "lgbm"))
    if model != "rf":
        raise SystemExit(
            f"causal_tune_trees_tuned_rfp runs the random forest only (MODEL={model!r})"
        )
    src = TUNED.read_text(encoding="utf-8")
    for old, new in SUBS:
        if src.count(old) != 1:
            raise SystemExit(f"{TUNED}: expected exactly one {old!r}")
        src = src.replace(old, new)
    print(
        f"tuned_rfp: TREE_SPEC = {TREE_SPEC.name}; forest grid re-centred on the shipped forest",
        flush=True,
    )
    sys.argv = [str(TUNED)]
    g = {"__file__": str(TUNED), "__name__": "__main__"}
    exec(compile(src, str(TUNED), "exec"), g)  # noqa: S102 -- the shipped tuned spec's source, three lines replaced


if __name__ == "__main__":
    main()
