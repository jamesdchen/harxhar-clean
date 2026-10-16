"""causal_tune_trees_optuna_xgbtop -- the Optuna rung (OP) of the per-bar XGBoost re-centred on the top 1998-2002
dev-span entry xgb_opt_00 (user decision 2026-10-07): trial 0 of every study = xgb_opt_00 instead of xgb_opt_06.
The XGBoost twin of specs/causal_tune_trees_optuna_rfp.py (the random forest's wrapper).

Runs specs/causal_tune_trees_optuna.py UNCHANGED (its source executed as __main__, __file__ = its real path) with two
module attributes of its worker-side job modules replaced, here and documented:

1. THE SHIPPED XGBOOST (causal_tune_trees_tuned_jobs.TREE_SPEC).  tree_setup() executes the untuned spec's setup from
   TREE_SPEC = specs/causal_tune_trees.py, whose XGB_PARAMS is xgb_opt_06.  It is pointed at HPC_KW_TREE_SPEC (default
   specs/causal_tune_trees_xgbtop.py, written by experiments/trees_xgbtop_make_specs.py from cluster/xgbtop_xgb_config.json:
   XGB_PARAMS = xgb_opt_00, min_child_weight scaled by the spec's own line to the per-bar window: max_depth 5,
   min_child_weight 4.1834, subsample 0.6551, colsample_bytree 0.6538, reg_lambda 1.159e-5, learning_rate 0.012337,
   n_estimators 367; reg_alpha 0.02148, gamma 1.5e-6, hist).  Through it, unchanged code of the job module derives from
   xgb_opt_00: trial 0 (causal_tune_trees_optuna_jobs.shipped_config reads T["PARAMS"]["xgb"] on the searched axes),
   the round cap (rounds_cap_of: 4 x 367 x 0.012337 / lr, = 1468 at the shipped learning rate), and every unsearched
   parameter of every trial and refit (reg_alpha, gamma, tree_method stay xgb_opt_00's: build_model overrides only the
   searched axes on the setup's XGB_PARAMS).

2. THE reg_lambda AXIS (causal_tune_trees_optuna_jobs.SPACES["xgb"]["reg_lambda"], XGB_REG_LAMBDA_LO).  The space is
   log-uniform [1e-4, 1e4]; its lower end was set at the power of ten just below xgb_opt_06's 4.12e-4 so that trial 0
   = the shipped value lies inside a log-uniform axis ("the shipped value is 4.1e-4 ... below the brief's 1e-3; one
   decade lower keeps it inside", the job module's note).  xgb_opt_00's reg_lambda is 1.159e-5, below 1e-4: Optuna
   would run the fixed trial-0 value with an out-of-range warning and TPE would model it outside its support.  Re-
   centred by the same rule: the lower end = the power of ten at or below the shipped value, 10^floor(log10(1.159e-5))
   = 1e-5 (reg_lambda_lo below).  Every other axis already contains xgb_opt_00 and covers the re-centred random-search
   grid (max_depth int [1, 8] holds 3..7; min_child_weight log [0.1, 256] holds 0.523..33.5; subsample [0.15, 1] holds
   0.1638..1, the 0.15 lower end still "just below the grid's quarter step"; colsample_bytree [0.05, 1] holds 0.1635..1;
   learning_rate log [0.005, 0.1] holds 0.012337) and is kept; the axes, their order (it fixes TPE's random stream),
   the sampler, its seeds, the budgets and the paths are unchanged.  The wrapper checks trial 0 lies in the space and
   that the space covers the re-centred grid (stops otherwise).

HOW THE PATCHES REACH THE POOL WORKERS.  The Optuna spec runs its trials (stage 1, OJ.tune_point) and refits (stage 2,
OJ.refit_masked / J.refit_block) in a ProcessPoolExecutor with the SPAWN context (never a forked worker: XGBoost
segfaults in forked workers on CARC epyc-7513); each worker imports the job modules afresh.  The patches are therefore
made at this file's TOP LEVEL, outside `if __name__ == "__main__"`: multiprocessing's spawn bootstrap runs the parent's
main script (this file) as __mp_main__ in every child BEFORE it unpickles any work item, so the child's job modules are
the patched ones.  The Optuna spec is exec'd in its OWN namespace, so this module's __file__ stays this file.

IDENTITY (env IDENTITY=1, the Optuna spec's own switch): one trial = xgb_opt_00 with early stopping off (rounds = 367);
stage 2 must then reproduce the FIXED-settings XGBoost (specs/causal_tune_trees_xgbtop.py, REFIT_EVERY 1,
IMPORTANCE_EVERY 10, WINDOW_MASK 1) bit for bit -- experiments/trees_xgbtop_optuna_gate.py.

ENV AXES: those of specs/causal_tune_trees_optuna.py (MODEL must be xgb) + TREE_SPEC (HPC_KW_TREE_SPEC; absolute,
repo-relative, or a file name inside specs/).  Outputs: the Optuna spec's layout under $HPC_RESULT_DIR (model
directory xgb); the records' meta carry the space ("space", reg_lambda low 1e-05), trial 0 ("shipped_config"), the
shipped parameters ("shipped_params") and the configuration's provenance (xgb_opt_00 sha 88bec9f385c7).
"""

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import causal_tune_trees_optuna_jobs as OJ  # noqa: E402 -- the worker-side jobs, patched below
import causal_tune_trees_tuned_jobs as J  # noqa: E402 -- tree_setup / build_model / refit_block

OPTUNA_SPEC = HERE / "causal_tune_trees_optuna.py"
DEFAULT_TREE_SPEC = "causal_tune_trees_xgbtop.py"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


def resolve_tree_spec(value: str) -> Path:
    """Absolute, repo-relative (specs/x.py) or a bare file name inside specs/."""
    p = Path(value)
    for cand in (p,) if p.is_absolute() else (ROOT / p, HERE / p):
        if cand.is_file():
            return cand.resolve()
    raise SystemExit(
        f"TREE_SPEC {value!r} not found (experiments/trees_xgbtop_make_specs.py writes the default)"
    )


def reg_lambda_lo(shipped: float) -> float:
    """The power of ten at or below the shipped reg_lambda (module note, point 2): 1e-4 for xgb_opt_06's 4.12e-4,
    1e-5 for xgb_opt_00's 1.159e-5; never above the job module's 1e-4."""
    if not shipped > 0:
        raise SystemExit(f"shipped reg_lambda {shipped!r} is not positive (a log axis)")
    return min(OJ.XGB_REG_LAMBDA_LO, 10.0 ** math.floor(math.log10(shipped)))


TREE_SPEC = resolve_tree_spec(_env("TREE_SPEC", DEFAULT_TREE_SPEC))

# ---- the patches (top level: run in the parent AND, as __mp_main__, in every spawned worker) ----
J.TREE_SPEC = TREE_SPEC
_SHIPPED = J.tree_setup()[
    "XGB_PARAMS"
]  # the generated spec's setup (leaf-scaled), read once per process
_LO = reg_lambda_lo(float(_SHIPPED["reg_lambda"]))
_old = OJ.SPACES["xgb"]["reg_lambda"]
assert _old[0] == "float" and _old[3] is True, _old
OJ.XGB_REG_LAMBDA_LO = _LO
OJ.SPACES["xgb"] = {
    a: (("float", _LO, _old[2], True) if a == "reg_lambda" else s)
    for a, s in OJ.SPACES["xgb"].items()
}
assert list(OJ.SPACES["xgb"]) == [
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "learning_rate",
]


def check_space() -> dict:
    """Trial 0 (the job module's shipped_config through the patched setup) inside the space; the space covers the
    random-search grid centred on the same shipped values (specs/causal_tune_trees_tuned.py build_grid's rules)."""
    cfg = OJ.shipped_config("xgb")
    space = OJ.SPACES["xgb"]
    for a, s in space.items():
        v = cfg[a]
        if not (s[1] <= v <= s[2]):
            raise SystemExit(f"trial 0: {a} = {v!r} lies outside the search space {s}")
    s = _SHIPPED
    grid = {
        "max_depth": [max(1, s["max_depth"] + k) for k in range(-2, 3)],
        "min_child_weight": [s["min_child_weight"] * 2.0**k for k in range(-3, 4)],
        "subsample": [s["subsample"] * f for f in (0.25, 0.5, 1.0)] + [1.0],
        "colsample_bytree": [s["colsample_bytree"] * f for f in (0.25, 0.5, 1.0)]
        + [1.0],
        "reg_lambda": [s["reg_lambda"], 1.0, 10.0, 100.0, 1000.0],
    }
    for a, vals in grid.items():
        lo, hi = space[a][1], space[a][2]
        bad = [v for v in vals if not (lo <= v <= hi)]
        if bad:
            raise SystemExit(
                f"the search space {a} [{lo}, {hi}] does not cover the re-centred grid values {bad}"
            )
    return cfg


def main() -> None:
    model = _env("MODEL", "lgbm")
    if model != "xgb":
        raise SystemExit(
            f"causal_tune_trees_optuna_xgbtop runs XGBoost only (MODEL={model!r})"
        )
    T = J.tree_setup(refresh=True)
    trial0 = check_space()
    print(
        f"optuna_xgbtop: pid {os.getpid()}; TREE_SPEC = {TREE_SPEC.name}; shipped XGB_PARAMS = {T['PARAMS']['xgb']}; "
        f"trial 0 = {trial0}; reg_lambda axis {OJ.SPACES['xgb']['reg_lambda']}; round cap at the shipped lr "
        f"{OJ.rounds_cap_of(T['PARAMS']['xgb']['learning_rate'], 'xgb')}",
        flush=True,
    )
    sys.argv = [str(OPTUNA_SPEC)]
    g = {"__file__": str(OPTUNA_SPEC), "__name__": "__main__"}
    exec(compile(OPTUNA_SPEC.read_text(encoding="utf-8"), str(OPTUNA_SPEC), "exec"), g)  # noqa: S102 -- the shipped Optuna spec, unchanged


if __name__ == "__main__":
    main()
