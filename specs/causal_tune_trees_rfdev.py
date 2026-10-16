"""causal_tune_trees_rfdev -- the per-bar random forest of specs/causal_tune_trees.py with the settings of
the dev-span procedure (user decision 2026-10-06).

The random forest of the per-bar tree campaign had no precedent (sklearn defaults: 100 trees,
max_features 1.0, min_samples_leaf 1, unlimited depth), while LightGBM's and XGBoost's settings came
from the 1998-2002 dev-span menu (experiments/tree_menu_dev.py).  The user decided the random forest
gets its settings from the same dev-span procedure; that search writes its configuration to a JSON
file (results/rf_devmenu_2026-10-06/rf_config.json, shipped as cluster/rfdev_rf_config.json).  This
wrapper runs specs/causal_tune_trees.py UNCHANGED except for one line: RF_PARAMS is replaced by the
configuration read from RF_CONFIG (and the provenance string names it).  Everything else -- design,
window mask, refit and importance cadence, single-threaded fits (n_jobs = SLURM_CPUS_PER_TASK), seed
42, TreeSHAP, outputs and their layout ($HPC_RESULT_DIR/causal_tune_trees/rf/<bucket>/) -- is the
spec's own code, executed from its source.

LEAF MINIMUM (as the spec's LightGBM / XGBoost lines, specs/causal_tune_trees.py "The leaf minimums are
row counts tuned on the pooled bank's window"): the dev-span menu tunes on the pooled window of
POOLED_WINDOW_ROWS = 24000 bars (the configuration's provenance.window); a per-bar window holds
TRAIN_WIN rows, one per session, so the forest's min_samples_leaf (an integer row count, the analogue of
LightGBM's min_child_samples) is scaled to keep the same share of the window per leaf:
max(1, round(min_samples_leaf * TRAIN_WIN / 24000)) -- 24 -> 2 at TRAIN_WIN = 2000.  n_estimators,
max_depth and the fractions max_features / max_samples are scale-free or absolute and are not scaled.
A configuration that declares no tuning window (the shipped forest, {"n_estimators": 100}) is used as is.

Configuration file (env HPC_KW_RF_CONFIG or RF_CONFIG): a JSON object; the RandomForestRegressor
keyword arguments are read from its "rf_params" or "params" member when present, else the object
itself.  random_state and n_jobs are the spec's (SEED, the task's thread count) and may not be set
by the file.  With {"n_estimators": 100} the wrapper is the shipped random forest: the reproduction
gate (experiments/trees_rfdev_gate.py) runs it so against the stored masked chunks, bit for bit.

ENV AXES: those of specs/causal_tune_trees.py (MODEL must be rf; TRAIN_WIN also sets the leaf scaling)
+ RF_CONFIG.
"""

import json
import os
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent / "causal_tune_trees.py"
RF_LINE = 'RF_PARAMS: dict = {"n_estimators": 100}  # the sklearn default, stated'
PROV_LINE = '"rf": "no precedent: sklearn RandomForestRegressor defaults",'
SPEC_OWNED = ("random_state", "n_jobs")
POOLED_WINDOW_ROWS = (
    24000  # the spec's constant: the pooled window the dev-span menus tune on
)


def _train_win() -> int:
    return int(os.environ.get("HPC_KW_TRAIN_WIN", os.environ.get("TRAIN_WIN", "2000")))


def rf_params(path: str) -> dict:
    """The per-bar RandomForestRegressor keyword arguments of a configuration file (leaf minimum scaled)."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    p = cfg.get("rf_params", cfg.get("params", cfg)) if isinstance(cfg, dict) else None
    if not isinstance(p, dict) or not p:
        raise SystemExit(f"{path}: no random-forest parameter object")
    p = dict(p)
    bad = [k for k in SPEC_OWNED if k in p]
    if bad:
        raise SystemExit(f"{path}: {bad} are set by the spec, not the configuration")
    window = (
        (cfg.get("provenance") or {}).get("window") if isinstance(cfg, dict) else None
    )
    if window is not None:
        if int(window) != POOLED_WINDOW_ROWS:
            raise SystemExit(
                f"{path}: tuned on a {window}-row window, the spec scales from {POOLED_WINDOW_ROWS}"
            )
        if "min_samples_leaf" in p:
            p["min_samples_leaf"] = max(
                1, round(int(p["min_samples_leaf"]) * _train_win() / POOLED_WINDOW_ROWS)
            )
    from sklearn.ensemble import RandomForestRegressor

    RandomForestRegressor(
        **p
    )  # unknown keyword -> TypeError here, before any data is loaded
    return p


def main() -> None:
    model = os.environ.get("HPC_KW_MODEL", os.environ.get("MODEL", "lgbm"))
    if model != "rf":
        raise SystemExit(
            f"causal_tune_trees_rfdev runs the random forest only (MODEL={model!r})"
        )
    path = os.environ.get("HPC_KW_RF_CONFIG", os.environ.get("RF_CONFIG", ""))
    if not path:
        raise SystemExit(
            "RF_CONFIG (HPC_KW_RF_CONFIG) must name the configuration JSON"
        )
    params = rf_params(path)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    pooled = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    prov = f"dev-span configuration {path} ({cfg.get('name', '?')} sha {cfg.get('sha', '?')})"
    if "min_samples_leaf" in pooled and pooled.get("min_samples_leaf") != params.get(
        "min_samples_leaf"
    ):
        prov += (
            f"; min_samples_leaf scaled {pooled['min_samples_leaf']} -> {params['min_samples_leaf']} "
            f"for the {_train_win()}-row per-bar window"
        )
    src = SPEC.read_text(encoding="utf-8")
    if src.count(RF_LINE) != 1 or src.count(PROV_LINE) != 1:
        raise SystemExit(
            f"{SPEC}: the RF_PARAMS / provenance lines are not the expected ones"
        )
    src = src.replace(RF_LINE, f"RF_PARAMS: dict = {params!r}  # rfdev: {path}")
    src = src.replace(PROV_LINE, f'"rf": {prov!r},')
    print(f"rfdev: RF_PARAMS = {params} ({prov})", flush=True)
    sys.argv = [str(SPEC)]
    g = {"__file__": str(SPEC), "__name__": "__main__"}
    exec(compile(src, str(SPEC), "exec"), g)  # noqa: S102 -- the shipped spec's own source, one line replaced


if __name__ == "__main__":
    main()
