"""causal_tune_trees_tuned_xgbtop -- the random-search rungs (RS10 / RS1) of the per-bar XGBoost re-centred on the top
1998-2002 dev-span entry xgb_opt_00 (user decision 2026-10-07: XGBoost follows LightGBM's and the forest's rule).

Runs specs/causal_tune_trees_tuned.py UNCHANGED (its source executed as __main__) with ONE module attribute replaced:

THE SHIPPED XGBOOST (causal_tune_trees_tuned_jobs.TREE_SPEC).  tree_setup() executes the untuned spec's setup from
TREE_SPEC = specs/causal_tune_trees.py, whose XGB_PARAMS is xgb_opt_06.  It is pointed at HPC_KW_TREE_SPEC (default
specs/causal_tune_trees_xgbtop.py, written by experiments/trees_xgbtop_make_specs.py from cluster/xgbtop_xgb_config.json:
the same file with XGB_PARAMS = xgb_opt_00's pooled-window values, min_child_weight scaled by the spec's own line to the
per-bar window: max_depth 5, learning_rate 0.012337, n_estimators 367, min_child_weight 50.20 -> 4.1834, subsample 0.6551,
colsample_bytree 0.6538, reg_alpha 0.02148, reg_lambda 1.159e-5, gamma 1.5e-6, hist).  The patch is made at this file's
TOP LEVEL, so the spawned pool workers (multiprocessing re-runs the main script as __mp_main__ in every child before it
unpickles a job) see the same TREE_SPEC.  Every fit is single-threaded inside a SPAWNED worker (never a forked one),
as in every stored tuned XGBoost run.

THE GRID needs no substitution: specs/causal_tune_trees_tuned.py builds the XGBoost grid from SHIPPED = the setup's
XGB_PARAMS (after the leaf scaling), so with the patched setup it is centred on xgb_opt_00 by exactly the rule that
centred it on xgb_opt_06 (build_grid):
                      rule                                         xgb_opt_06 (shipped until now)     xgb_opt_00 (this run)
  max_depth           d - 2 .. d + 2 (leaves x1/4 .. x4)            2, 3, 4, 5, 6                       3, 4, 5, 6, 7
  min_child_weight    shipped x 2^k, k = -3 .. 3                    2.0718 x {1/8 .. 8}                 4.1834 x {1/8 .. 8}
                                                                    (0.259 .. 16.57)                    (0.523 .. 33.47)
  subsample           shipped x {1/4, 1/2, 1} and the bound 1.0     0.1522, 0.3043, 0.6087, 1.0         0.1638, 0.3276, 0.6551, 1.0
  colsample_bytree    shipped x {1/4, 1/2, 1} and the bound 1.0     0.1744, 0.3488, 0.6976, 1.0         0.1635, 0.3269, 0.6538, 1.0
  reg_lambda          the shipped value, 1, 10, 100, 1000           4.12e-4, 1 .. 1000                  1.159e-5, 1 .. 1000
  grid size           5 x 7 x 4 x 4 x 5                             2800                                2800
  candidates          shipped + 31 drawn with GRID_SEED 20260929    (same draw indices into the product of the re-centred axes)
  early stopping      patience ceil(1 / lr), cap 4 x n_estimators   93 / 1356                           82 / 1468
learning_rate, reg_alpha, gamma and tree_method stay at the shipped values (not tuned), as before.  The spec writes
the grid it used to grid_<seg>.json (and the npz meta "grid") in every chunk.

TUNE_IDENTITY=1 reduces the candidates to the shipped configuration with early stopping off: the run must then reproduce
the FIXED-settings XGBoost of the same configuration (specs/causal_tune_trees_xgbtop.py; results/
linear_subsection_trees_mask_xgbtop) bit for bit (experiments/trees_xgbtop_tuned_gate.py).

ENV AXES: those of specs/causal_tune_trees_tuned.py (MODEL must be xgb) + TREE_SPEC.
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

TREE_SPEC = (
    HERE
    / Path(
        os.environ.get(
            "HPC_KW_TREE_SPEC",
            os.environ.get("TREE_SPEC", "causal_tune_trees_xgbtop.py"),
        )
    ).name
)
if not TREE_SPEC.is_file():
    raise SystemExit(
        f"TREE_SPEC {TREE_SPEC} missing (experiments/trees_xgbtop_make_specs.py writes it)"
    )
J.TREE_SPEC = TREE_SPEC  # parent and every spawned worker: the shipped XGBoost of the generated spec

TUNED = HERE / "causal_tune_trees_tuned.py"


def main() -> None:
    model = os.environ.get("HPC_KW_MODEL", os.environ.get("MODEL", "lgbm"))
    if model != "xgb":
        raise SystemExit(
            f"causal_tune_trees_tuned_xgbtop runs XGBoost only (MODEL={model!r})"
        )
    T = J.tree_setup(refresh=True)
    print(
        f"tuned_xgbtop: pid {os.getpid()}; TREE_SPEC = {TREE_SPEC.name}; shipped XGB_PARAMS = {T['PARAMS']['xgb']}; "
        f"provenance {T['PROVENANCE']['xgb']}",
        flush=True,
    )
    sys.argv = [str(TUNED)]
    g = {"__file__": str(TUNED), "__name__": "__main__"}
    exec(compile(TUNED.read_text(encoding="utf-8"), str(TUNED), "exec"), g)  # noqa: S102 -- the shipped tuned spec, unchanged


if __name__ == "__main__":
    main()
