"""causal_tune_trees_optuna_rfp -- the Optuna rung (OP) of the per-bar random forest re-centred on a new SHIPPED
forest: a dev-span configuration instead of sklearn's defaults.  PARAMETERIZED by the generated tree spec
(HPC_KW_TREE_SPEC = specs/causal_tune_trees_<token>.py, written by experiments/trees_rfp163_make_specs.py from a
configuration file); the worked instance is rfp163 (2026-10-07: the search's trial 163, cluster/rfprev163_rf_config.json,
depth 14), the default.  Every number quoted below is that instance's.

Runs specs/causal_tune_trees_optuna.py UNCHANGED (its source executed as __main__, __file__ = its real path)
with three module attributes of its worker-side job modules replaced, here and documented:

1. THE SHIPPED FOREST (causal_tune_trees_tuned_jobs.TREE_SPEC).  tree_setup() executes the untuned spec's setup
   from TREE_SPEC = specs/causal_tune_trees.py, whose RF_PARAMS is {"n_estimators": 100}.  It is pointed at
   HPC_KW_TREE_SPEC (default specs/causal_tune_trees_rfp163.py, written by experiments/trees_rfp163_make_specs.py:
   the same file with RF_PARAMS = the trial-163 configuration of cluster/rfprev163_rf_config.json, leaf minimum
   scaled to the per-bar window: n_estimators 355, max_depth 14, min_samples_leaf 2, max_features 0.7356,
   max_samples 0.8811, bootstrap).  Every forest of both stages is built by tree_setup()'s RF_PARAMS overridden
   on the searched axes, so n_estimators / max_samples / bootstrap stay at the shipped values in every trial and
   refit (not searched, as n_estimators is not searched for the default forest).

2. THE SHIPPED CONFIGURATION ON THE SEARCHED AXES (causal_tune_trees_optuna_jobs.shipped_config, trial 0 of
   every study).  The jobs file hard-codes the forest's as sklearn's defaults (max_features 1.0,
   min_samples_leaf 1, max_depth None); here it is read from the patched setup, T["PARAMS"]["rf"] on the
   axes of SPACES["rf"] -- (0.7356104726223699, 2, 14) -- exactly as the jobs file does for LightGBM / XGBoost.

3. THE max_depth AXIS (causal_tune_trees_optuna_jobs.SPACES["rf"] / RF_MAX_DEPTH_CHOICES).  The categorical
   {2, 4, 8, 16, 32, None} brackets the default forest's shipped depth (None = unbounded) by doubling from 2;
   it does not contain 14, so trial 0 could not be the new shipped forest (Optuna rejects a fixed categorical
   value outside the choices).  Re-centred the same way on the shipped depth d (depth_ladder below): the doubling
   ladder through it, {d/8, d/4, d/2, d, 2d} rounded half up, values below 1 dropped, duplicates removed, + None
   (unbounded) -- for d = 14: {1.75 -> 2, 3.5 -> 4, 7, 14, 28, None} = {2, 4, 7, 14, 28, None}, the same six
   choices as the default forest's, the same two shallow ends (2, 4) and the unbounded end, the middle three
   re-centred (8, 16, 32 -> 7, 14, 28).  (A shallow d gives fewer choices after de-duplication, e.g. d = 4 ->
   {1, 2, 4, 8, None}; a shipped unbounded depth keeps the default forest's ladder.)  max_features float [0.02, 1]
   and min_samples_leaf int-log [1, 256] already contain the shipped 0.7356 and 2 and are kept (a shipped value
   outside them stops the run).  The axes, their order (it fixes TPE's random stream) and the sampler are
   unchanged.

HOW THE PATCHES REACH THE POOL WORKERS.  The Optuna spec runs its trials (stage 1, OJ.tune_point) and refits
(stage 2, OJ.refit_masked / J.refit_block) in a ProcessPoolExecutor with the SPAWN context; each worker
imports the job modules afresh.  The patches are therefore made at this file's TOP LEVEL, outside
`if __name__ == "__main__"`: multiprocessing's spawn bootstrap runs the parent's main script (this file:
sys.modules["__main__"].__file__) as __mp_main__ in every child BEFORE it unpickles any work item, so the
child's causal_tune_trees_tuned_jobs / causal_tune_trees_optuna_jobs are the patched ones when the pickled
reference OJ.tune_point (or OJ.refit_masked) is resolved, and their lazy tree_setup() reads the patched
TREE_SPEC.  The Optuna spec is exec'd in its OWN namespace (never this module's globals), so this module's
__file__ stays this file and the children never run the unwrapped spec.  Evidence: the stage-1 records
(trial 0 = the new shipped configuration, rounds = 355 = the patched n_estimators, max_depth values in the
new set) and the identity gate (stage 2 through the pool = the fixed-settings forest bit for bit) are computed
in spawned workers -- experiments/trees_rfp163_optuna_gate.py.

IDENTITY (env IDENTITY=1, the Optuna spec's own switch): one trial = the shipped configuration; stage 2 must then
reproduce the FIXED-settings forest of the same configuration (specs/causal_tune_trees_rfdev.py with
HPC_KW_RF_CONFIG=<the configuration>, REFIT_EVERY 1, IMPORTANCE_EVERY 10, WINDOW_MASK 1) bit for bit.

ENV AXES: those of specs/causal_tune_trees_optuna.py (MODEL must be rf; TRAIN_WIN must be 2000, the window the
generated spec's leaf minimum is scaled to) + TREE_SPEC (HPC_KW_TREE_SPEC; absolute, repo-relative, or a file
name inside specs/).  Outputs: the Optuna spec's layout under $HPC_RESULT_DIR (model directory rf); the records'
meta carry the new space ("space"), shipped configuration ("shipped_config"), shipped parameters
("shipped_params") and the configuration's provenance.
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
DEFAULT_TREE_SPEC = "causal_tune_trees_rfp163.py"
TRAIN_WIN_RFP = (
    2000  # the generated spec's leaf minimum is scaled to this window (27 -> 2)
)


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


def resolve_tree_spec(value: str) -> Path:
    """Absolute, repo-relative (specs/x.py) or a bare file name inside specs/ (the random-search wrapper's form)."""
    p = Path(value)
    for cand in (p,) if p.is_absolute() else (ROOT / p, HERE / p):
        if cand.is_file():
            return cand.resolve()
    raise SystemExit(
        f"TREE_SPEC {value!r} not found (experiments/trees_rfp163_make_specs.py writes the default)"
    )


def depth_ladder(d: int | None) -> tuple:
    """The max_depth categorical re-centred on the shipped depth d (module note, point 3)."""
    if d is None:
        return tuple(
            OJ.RF_MAX_DEPTH_CHOICES
        )  # the default forest's ladder around its unbounded depth
    d = int(d)
    if d < 1:
        raise SystemExit(f"shipped max_depth {d} < 1")
    half_up = [math.floor(d * f + 0.5) for f in (0.125, 0.25, 0.5, 1.0, 2.0)]
    return (*sorted({v for v in half_up if v >= 1}), None)


TREE_SPEC = resolve_tree_spec(_env("TREE_SPEC", DEFAULT_TREE_SPEC))


def shipped_config(model: str | None = None) -> dict:
    """The shipped configuration on the searched axes: the patched setup's RF_PARAMS (trial 0 of every study)."""
    T = J.tree_setup()
    model = model or T["MODEL"]
    if model != "rf":
        raise SystemExit(
            f"causal_tune_trees_optuna_rfp re-centres the random forest only (model {model!r})"
        )
    shipped = T["PARAMS"]["rf"]
    space = OJ.SPACES["rf"]
    missing = [a for a in space if a not in shipped]
    if missing:
        raise SystemExit(
            f"{J.TREE_SPEC.name}: RF_PARAMS sets no {missing}: no shipped value on those axes"
        )
    cfg = {a: shipped[a] for a in space}
    for a, s in space.items():
        v = cfg[a]
        ok = v in s[1] if s[0] == "cat" else (v is not None and s[1] <= v <= s[2])
        if not ok:
            raise SystemExit(f"shipped {a} = {v!r} lies outside the search space {s}")
    return cfg


# ---- the patches (top level: run in the parent AND, as __mp_main__, in every spawned worker) ----
J.TREE_SPEC = TREE_SPEC
# the shipped forest of the generated spec (tree_setup executes its head once per process; the Optuna spec
# re-reads it with refresh=True in the parent, a worker keeps this first read)
RF_MAX_DEPTH_CHOICES = depth_ladder(J.tree_setup()["RF_PARAMS"].get("max_depth"))
OJ.RF_MAX_DEPTH_CHOICES = RF_MAX_DEPTH_CHOICES
OJ.SPACES["rf"] = {
    "max_features": OJ.SPACES["rf"][
        "max_features"
    ],  # ("float", 0.02, 1.0, False): holds 0.7356
    "min_samples_leaf": OJ.SPACES["rf"][
        "min_samples_leaf"
    ],  # ("int", 1, 256, True): holds 2
    "max_depth": ("cat", RF_MAX_DEPTH_CHOICES),
}
assert list(OJ.SPACES["rf"]) == ["max_features", "min_samples_leaf", "max_depth"], list(
    OJ.SPACES["rf"]
)
OJ.shipped_config = shipped_config


def main() -> None:
    model = _env("MODEL", "lgbm")
    if model != "rf":
        raise SystemExit(
            f"causal_tune_trees_optuna_rfp runs the random forest only (MODEL={model!r})"
        )
    tw = int(_env("TRAIN_WIN", str(TRAIN_WIN_RFP)))
    if tw != TRAIN_WIN_RFP:
        raise SystemExit(
            f"TRAIN_WIN={tw}: {TREE_SPEC.name}'s leaf minimum is scaled to TRAIN_WIN {TRAIN_WIN_RFP}"
        )
    T = J.tree_setup(refresh=True)
    print(
        f"optuna_rfp: pid {os.getpid()}; TREE_SPEC = {TREE_SPEC.name}; shipped RF_PARAMS = {T['PARAMS']['rf']}; "
        f"trial 0 = {shipped_config('rf')}; max_depth choices {list(RF_MAX_DEPTH_CHOICES)}",
        flush=True,
    )
    sys.argv = [str(OPTUNA_SPEC)]
    g = {"__file__": str(OPTUNA_SPEC), "__name__": "__main__"}
    exec(compile(OPTUNA_SPEC.read_text(encoding="utf-8"), str(OPTUNA_SPEC), "exec"), g)  # noqa: S102 -- the shipped Optuna spec, unchanged


if __name__ == "__main__":
    main()
