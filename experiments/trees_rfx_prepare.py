"""Prepare a dev-span forest TOKEN (e.g. rf400 = the 400-trial search's final entry) for cluster/slurm/submit_trees_rfx.sh.

  1. cluster/<TOKEN>_rf_config.json   = the search's configuration file, byte for byte (the wrappers read its "params"
                                        and scale min_samples_leaf from its provenance.window to the per-bar window)
  2. specs/causal_tune_trees_<TOKEN>.py via experiments/trees_rfp163_make_specs.py --config ... --token TOKEN
  3. task files (cluster/trees_<TOKEN>_*.txt): the masked fixed-settings and unmasked T1 / T10 chunks
     (experiments/trees_rfdev_make_tasks.py, 450- / 600-second chunk processes: many cores, little preparation overhead) sized from the trial-163
     preview's measured one-thread seconds per refit (all_features 52.4, lean_vol_vix 8.2, baseline 1.64) scaled by the
     forest's cost ratio n_estimators x max_features x max_samples x max_depth (an estimate, chunk sizes only; the unmasked
     run sees every column: x1.7 / 1.6 / 1.3); the random-search chunk-arms (one 250-session tuning period each) and their
     canary (chunk 0 of each set); the identity-gate lines (rows inside each fixed canary chunk c0).

Run:  python experiments/trees_rfx_prepare.py --token rf400 --config results/rf_devmenu_2026-10-06/budget400/rf_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ("baseline", "all_features", "lean_vol_vix")
PREVIEW = {
    "params": {
        "n_estimators": 355,
        "max_depth": 14,
        "max_features": 0.7356104726223699,
        "max_samples": 0.8810869892182808,
    },
    "fit_sec": {"baseline": 1.64, "all_features": 52.4, "lean_vol_vix": 8.2},
}
NOMASK = {
    "baseline": 22 / 17,
    "all_features": 628 / 369,
    "lean_vol_vix": 94 / 59,
}  # all columns / kept columns
W, SEG, N_OOS, TP = 2000, "bar1600", 1469, 250


def cost_ratio(p: dict) -> float:
    q = PREVIEW["params"]
    # depth barely matters once min_samples_leaf (2) bounds the trees: rf_opt_00 (depth 19) measured 68.2 s against
    # trial 163's (depth 14) 52.4 s on all_features, x1.30, of which the trees / features / samples ratios explain x1.23
    depth = 1.2 if p.get("max_depth") is None else 1.0
    return (
        (p["n_estimators"] / q["n_estimators"])
        * (p.get("max_features", 1.0) / q["max_features"])
        * ((p.get("max_samples") or 1.0) / q["max_samples"])
        * depth
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument(
        "--config",
        required=True,
        help="the search's configuration file (repo-relative)",
    )
    a = ap.parse_args()
    tok = a.token
    cfg_dst = ROOT / "cluster" / f"{tok}_rf_config.json"
    shutil.copyfile(ROOT / a.config, cfg_dst)
    cfg = json.loads(cfg_dst.read_text(encoding="utf-8"))
    print(
        f"config -> {cfg_dst.relative_to(ROOT)}: {cfg.get('name')} params {cfg.get('params')}"
    )
    py = sys.executable
    subprocess.run(
        [
            py,
            "experiments/trees_rfp163_make_specs.py",
            "--config",
            str(cfg_dst.relative_to(ROOT)).replace("\\", "/"),
            "--token",
            tok,
        ],
        cwd=ROOT,
        check=True,
    )
    sys.path.insert(0, str(ROOT / "specs"))
    import causal_tune_trees_rfdev as w

    p = w.rf_params(str(cfg_dst))
    r = cost_ratio(p)
    fs = {b: PREVIEW["fit_sec"][b] * r for b in BUCKETS}
    print(
        f"per-bar params {p}; cost ratio vs trial 163 {r:.2f}; est one-thread seconds per refit {fs}"
    )
    mk = [py, "experiments/trees_rfdev_make_tasks.py", "--shap-sec", "2.0"]
    subprocess.run(
        mk
        + [
            "--target-sec",
            "450",
            "--fit-sec",
            ",".join(f"{b}={fs[b]:.2f}" for b in BUCKETS),
            "--out",
            f"cluster/trees_{tok}_tasks_untuned.txt",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        mk
        + [
            "--target-sec",
            "600",
            "--fit-sec",
            ",".join(f"{b}={fs[b] * NOMASK[b]:.2f}" for b in BUCKETS),
            "--out",
            f"cluster/trees_{tok}_nomask_tasks_untuned.txt",
        ],
        cwd=ROOT,
        check=True,
    )
    n_t = math.ceil(N_OOS / TP)
    rs = [
        f"{b} rf {W} {SEG} {k} {W + TP * k} {-1 if k == n_t - 1 else W + TP * (k + 1)} {W}"
        for b in BUCKETS
        for k in range(n_t)
    ]
    for rung in ("rs10", "rs1"):
        (ROOT / "cluster" / f"trees_{tok}_tasks_{rung}.txt").write_text(
            "\n".join(rs) + "\n", encoding="utf-8", newline="\n"
        )
    (ROOT / "cluster" / f"trees_{tok}_tasks_rs_canary.txt").write_text(
        "".join(f"{b} rf {W} {SEG} 0 {W} {W + TP} {W}\n" for b in BUCKETS),
        encoding="utf-8",
        newline="\n",
    )
    # identity-gate lines: rows inside each fixed canary chunk c0 (T10 needs 20 rows = 2 refits, T1 up to 5)
    ext = {}
    for ln in (
        (ROOT / "cluster" / f"trees_{tok}_tasks_untuned.txt")
        .read_text(encoding="utf-8")
        .split("\n")
    ):
        f = ln.split()
        if f and f[0] == "0":
            ext[(f[1], f[2])] = (int(f[7]), int(f[8]))
    gl = []
    for b in BUCKETS:
        s, e = ext[("t10", b)]
        gl.append(
            f"rs10 {b} 0 {W} {20 if e < 0 or e - s >= 20 else (e - s) // 10 * 10}"
        )
        s, e = ext[("t1", b)]
        gl.append(f"rs1 {b} 0 {W} {5 if e < 0 else min(5, e - s)}")
    (ROOT / "cluster" / f"trees_{tok}_tuned_gate_lines.txt").write_text(
        "\n".join(gl) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"task files cluster/trees_{tok}_*.txt written; gate lines: {gl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
