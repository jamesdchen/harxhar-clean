"""Task files of the single-class re-run of the masked Optuna campaign (orchestrator decision 2026-09-30, option A).

Why: the executor's design matrix differs at rounding level between CPU classes (agent D), and
LightGBM / XGBoost tuning and refits move with it (D on this run's inputs: stage-1 trials
identical xeon-4116 vs epyc-7513 for lgbm 100/300, xgb 211/300, rf 283/300; stage-2 forecasts
lgbm 5/360 (max rel 0.10), xgb 344/360, rf 360/360).  The masked run landed on four node
types (chunk_hosts.csv), so it is kept as the class-MIXED run
(results/linear_subsection_trees_optuna_mask_mixed/) and the canonical run is made
SINGLE-CLASS: every chunk computed on an epyc-7513 node.

Re-run sets (from the mixed run's chunk_hosts.csv and the task files):
  stage 1  every chunk-arm that did not run on epyc-7513 (xeon-4116, xeon-2640v4, epyc-7542)
  stage 2  every chunk-arm that did not run on epyc-7513, and every chunk-arm with a row whose
           configuration in force (on any path: TUNE_PER 1 / 5 / 25 / 250 -> tuning point
           floor(row / TUNE_PER) x TUNE_PER) comes from a re-run tuning point
Writes cluster/optuna_tasks_s1_epyc.txt, cluster/optuna_tasks_s2_epyc.txt (lines of the full
task files) and results/linear_subsection_trees_optuna_mask/single_class_plan.csv.

Run from the repo root:  python cluster/optuna_single_class_tasks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "linear_subsection_trees_optuna_mask"
W, N_OOS = 2000, 1469
CLASS = "epyc-7513"  # the single class (most of CARC main's nodes)
TUNE_PERS = (1, 5, 25, 250)


def tasks(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        b, m, tw, seg, k, s, e, h = line.split()
        rows.append(
            (b, m, int(k), int(s) - W, N_OOS if int(e) < 0 else int(e) - W, line)
        )
    return pd.DataFrame(rows, columns=["bucket", "model", "chunk", "r0", "r1", "line"])


def main() -> None:
    hosts = pd.read_csv(R / "chunk_hosts.csv")
    t1 = tasks(ROOT / "cluster" / "optuna_tasks_s1.txt")
    t2 = tasks(ROOT / "cluster" / "optuna_tasks_s2.txt")
    key = ["bucket", "model", "chunk"]
    s1 = t1.merge(
        hosts[hosts.stage == 1][key + ["cpu_class", "host"]], on=key, how="left"
    )
    s2 = t2.merge(
        hosts[hosts.stage == 2][key + ["cpu_class", "host"]], on=key, how="left"
    )
    assert s1["cpu_class"].notna().all() and s2["cpu_class"].notna().all(), (
        "a chunk has no recorded host"
    )
    re1 = s1[s1.cpu_class != CLASS]
    plan = [
        {
            "stage": 1,
            "bucket": r.bucket,
            "model": r.model,
            "chunk": r.chunk,
            "mixed_class": r.cpu_class,
            "why": "not on " + CLASS,
        }
        for r in re1.itertuples()
    ]
    re2 = []
    for (b, m), g in s2.groupby(["bucket", "model"]):
        pts: set[int] = set()
        for r in re1[(re1.bucket == b) & (re1.model == m)].itertuples():
            pts |= set(range(r.r0, r.r1))
        rows = np.arange(N_OOS)
        aff = np.zeros(N_OOS, dtype=bool)
        for tp in TUNE_PERS:
            aff |= np.isin((rows // tp) * tp, sorted(pts))
        for r in g.itertuples():
            n_aff = int(aff[r.r0 : r.r1].sum())
            if r.cpu_class != CLASS or n_aff:
                why = ("not on " + CLASS if r.cpu_class != CLASS else "") + (
                    f"; {n_aff} rows use a re-run tuning point" if n_aff else ""
                )
                re2.append(r.line)
                plan.append(
                    {
                        "stage": 2,
                        "bucket": b,
                        "model": m,
                        "chunk": r.chunk,
                        "mixed_class": r.cpu_class,
                        "why": why.strip("; "),
                    }
                )
    order2 = [ln for ln in t2.line if ln in set(re2)]
    (ROOT / "cluster" / "optuna_tasks_s1_epyc.txt").write_text(
        "".join(ln + "\n" for ln in re1.line), encoding="utf-8", newline="\n"
    )
    (ROOT / "cluster" / "optuna_tasks_s2_epyc.txt").write_text(
        "".join(ln + "\n" for ln in order2), encoding="utf-8", newline="\n"
    )
    pd.DataFrame(plan).to_csv(R / "single_class_plan.csv", index=False)
    print(
        f"stage 1: {len(re1)} of {len(s1)} chunk-arms re-run; stage 2: {len(order2)} of {len(s2)}"
    )


if __name__ == "__main__":
    main()
