"""Progress of a final tree run (CARC): one row per piece from the pieces' .json metas, CPU-hours by arm.

usage: python collect.py <run dir> [<sacct csv>]
  -> <run dir>/progress.csv (the printed run's columns + n_columns, window_days, versions) and a summary by arm:
     pieces done / tasks, rows, CPU-hours = wall_sec x threads / 3600 (the printed STATUS.md rule); with a sacct CSV
     (JobID,...,AllocCPUS,CPUTimeRAW,TotalCPU) also the allocated CPU-hours by arm from Slurm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

run = Path(sys.argv[1])
tasks = [
    ln.split() for ln in (run / "tasks.txt").read_text().splitlines() if ln.strip()
]
rows = []
for p in sorted((run / "pieces").glob("*.json")):
    if p.name.endswith(".ckpt.json"):
        continue
    m = json.loads(p.read_text())
    rows.append(
        {
            "piece": p.stem,
            "est": m["est"],
            "predictor_set": m["predictor_set"],
            "refit": m["refit"],
            "part": m["part"],
            "block_start": m["block"][0],
            "block_end": m["block"][1],
            "rows": m["rows"],
            "refits": m["refits"],
            "wall_sec": m["wall_sec"],
            "threads": m["threads"],
            "host": m["host"],
            "cpu": m["cpu"],
            "slurm_job": m["slurm_job"],
            "slurm_task": m["slurm_task"],
            "config": m["config"],
            "sha": m["sha"],
            "n_columns": m["n_columns"],
            "window_days": m["window_days"],
            "versions": json.dumps(m["versions"], sort_keys=True),
        }
    )
P = pd.DataFrame(rows)
if len(P):
    P.to_csv(run / "progress.csv", index=False)
want = (
    pd.DataFrame([{"est": t[0], "predictor_set": t[1]} for t in tasks])
    .value_counts()
    .rename("tasks")
    .reset_index()
)
if len(P):
    P["cpu_h"] = P["wall_sec"] * P["threads"] / 3600
    g = (
        P.groupby(["est", "predictor_set"])
        .agg(
            pieces=("piece", "size"),
            rows=("rows", "sum"),
            cpu_h=("cpu_h", "sum"),
            max_wall_h=("wall_sec", lambda x: x.max() / 3600),
        )
        .reset_index()
    )
    want = want.merge(g, on=["est", "predictor_set"], how="left")
if len(sys.argv) > 2:
    S = pd.read_csv(sys.argv[2])
    S = S[
        S["JobID"].astype(str).str.contains("_")
        & ~S["JobID"].astype(str).str.contains(r"\.")
    ]
    S["alloc_cpu_h"] = S["CPUTimeRAW"].astype(float) / 3600
    print(
        f"sacct: {len(S)} array tasks, {S['alloc_cpu_h'].sum():.1f} allocated CPU-h, states "
        f"{S['State'].str.split().str[0].value_counts().to_dict()}"
    )
pd.set_option("display.width", 200)
print(want.to_string(index=False))
if len(P):
    print(
        f"done: {len(P)} of {len(tasks)} pieces; {P['cpu_h'].sum():.1f} CPU-h (wall x threads); CPU models "
        f"{P['cpu'].value_counts().to_dict()}; versions {P['versions'].nunique()} distinct"
    )
