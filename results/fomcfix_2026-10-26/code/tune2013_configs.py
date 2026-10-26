"""The configuration variants of results/fomcfix_2026-10-26/RULE.txt 7(c) (pre-sample tuning check), written once.

One factor at a time around the transferred (frozen) configuration T of each class; each variant a configuration file
in the frozen files' format (family, name, sha = the first 12 hex digits of sha256(json.dumps({family, **params},
sort_keys=True)), the rule of experiments/tree_menu_dev.py, params, provenance) for experiments/pooled_trees_rerun_13.py
walk --config-file.  The leaf-size variant rescales the frozen value from the 24,000-row window it was tuned at to the
39,000-row window of the final design.  Also writes tasks: one line a (variant, seed) for code/tree_task.sh.

usage: python tune2013_configs.py <code tree> <out dir (relative to the code tree)> <run dir on CARC> <tasks file>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CODE, OUTREL, RUN, TASKS = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
OUT = CODE / OUTREL
FROZEN = {
    "lgbm": CODE / "cluster" / "pooltree_lgbm_config.json",
    "xgb": CODE / "cluster" / "xgbtop_xgb_config.json",
    "rf_skx": CODE / "cluster" / "pooltree_rf400_config.json",
}
SCALE = 39000 / 24000
SEEDS = (43, 44, 45, 46)


def variants(est: str, p: dict) -> list[tuple[str, dict]]:
    v: list[tuple[str, dict]] = []
    if est == "lgbm":
        v += [("num_leaves_10", {"num_leaves": 10}), ("num_leaves_40", {"num_leaves": 40})]
        v += [
            ("lr_half", {"learning_rate": p["learning_rate"] / 2}),
            ("lr_x2", {"learning_rate": p["learning_rate"] * 2}),
        ]
        v += [("n_est_196", {"n_estimators": 196}), ("n_est_784", {"n_estimators": 784})]
        v += [("min_child_samples_159", {"min_child_samples": round(p["min_child_samples"] * SCALE)})]
    elif est == "xgb":
        v += [("max_depth_4", {"max_depth": 4}), ("max_depth_6", {"max_depth": 6})]
        v += [
            ("lr_half", {"learning_rate": p["learning_rate"] / 2}),
            ("lr_x2", {"learning_rate": p["learning_rate"] * 2}),
        ]
        v += [("n_est_184", {"n_estimators": 184}), ("n_est_734", {"n_estimators": 734})]
        v += [("min_child_weight_scaled", {"min_child_weight": p["min_child_weight"] * SCALE})]
    else:
        v += [("max_depth_15", {"max_depth": 15}), ("max_depth_25", {"max_depth": 25})]
        v += [("n_est_243", {"n_estimators": 243}), ("n_est_970", {"n_estimators": 970})]
        v += [
            ("min_leaf_6", {"min_samples_leaf": 6}),
            ("min_leaf_20", {"min_samples_leaf": round(p["min_samples_leaf"] * SCALE)}),
        ]
        v += [("max_features_0.70", {"max_features": 0.70}), ("max_features_1.0", {"max_features": 1.0})]
    return v


def sha(family: str, params: dict) -> str:
    return hashlib.sha256(json.dumps({"family": family, **params}, sort_keys=True).encode()).hexdigest()[:12]


lines = []
OUT.mkdir(parents=True, exist_ok=True)
for est, f in FROZEN.items():
    base = json.loads(f.read_text(encoding="utf-8"))
    p0 = dict(base["params"])
    fam = base["family"]
    assert sha(fam, p0) == base["sha"], (est, sha(fam, p0), base["sha"])
    lines.append(f"{RUN}/{est}_T {est} all_features day 2013-01-02 2013-12-31 1/1 -")
    for s in SEEDS:
        lines.append(f"{RUN}/{est}_T_seed{s} {est} all_features day 2013-01-02 2013-12-31 1/1 - {s}")
    for name, ch in variants(est, p0):
        p = dict(p0, **ch)
        cfg = dict(
            family=fam,
            name=f"{base['name']}__{name}",
            sha=sha(fam, p),
            params=p,
            provenance=dict(
                base=base["name"],
                base_sha=base["sha"],
                change=ch,
                rule="results/fomcfix_2026-10-26/RULE.txt 7(c): one factor at a time around the transferred configuration",
            ),
        )
        fp = OUT / f"{est}__{name}.json"
        fp.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
        lines.append(f"{RUN}/{est}__{name} {est} all_features day 2013-01-02 2013-12-31 1/1 {OUTREL}/{fp.name}")
TASKS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"{len(lines)} tasks -> {TASKS}")
