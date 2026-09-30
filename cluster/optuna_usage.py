"""CARC usage of the Optuna per-bar tree campaign, from a sacct dump.

The dump is written on CARC by
  sacct -X -n -P -j <canary>,<s1>,<m1>,<s2>,<m2> -o JobID,JobName,State,Elapsed,AllocCPUS,CPUTimeRAW,Start,End
and pulled next to this campaign's results as results/linear_subsection_trees_optuna_mask/sacct.txt.
Writes results/linear_subsection_trees_optuna_mask/cluster_usage.csv (item, value), which
experiments/optuna_trees_report_1600.py puts in SUMMARY.md: allocated CPU-hours per job
(CPUTimeRAW = allocated cores x elapsed seconds), the peak number of this campaign's CPUs
allocated at once (from the tasks' start / end stamps), wall-clock from the first start to
the last end, and task states.

Run:  python cluster/optuna_usage.py [--sacct results/linear_subsection_trees_optuna_mask/sacct.txt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "results" / "linear_subsection_trees_optuna_mask"
COLS = [
    "JobID",
    "JobName",
    "State",
    "Elapsed",
    "AllocCPUS",
    "CPUTimeRAW",
    "Start",
    "End",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sacct", default=str(OPT / "sacct.txt"))
    ap.add_argument("--out", default=str(OPT / "cluster_usage.csv"))
    a = ap.parse_args()
    d = pd.read_csv(a.sacct, sep="|", header=None, names=COLS, dtype=str)
    d = d[d["Start"].notna() & (d["Start"] != "Unknown") & (d["Start"] != "None")]
    d["cpus"] = d["AllocCPUS"].astype(int)
    d["cpu_sec"] = d["CPUTimeRAW"].astype(float)
    d["start"] = pd.to_datetime(d["Start"], errors="coerce")
    d["end"] = pd.to_datetime(d["End"], errors="coerce")
    d = d[d["start"].notna() & d["end"].notna()]
    ev = pd.concat(
        [
            pd.DataFrame({"t": d["start"], "c": d["cpus"]}),
            pd.DataFrame({"t": d["end"], "c": -d["cpus"]}),
        ]
    )
    ev = ev.sort_values(["t", "c"])  # ends before starts at the same stamp
    peak = int(ev["c"].cumsum().max())
    rows = [
        {
            "item": "allocated CPU-hours, all jobs",
            "value": f"{d['cpu_sec'].sum() / 3600:.1f}",
        },
        {"item": "peak concurrent CPUs (this campaign)", "value": str(peak)},
        {
            "item": "wall-clock, first start to last end",
            "value": str(d["end"].max() - d["start"].min()),
        },
        {
            "item": "first start / last end",
            "value": f"{d['start'].min()} / {d['end'].max()}",
        },
    ]
    for name, g in d.groupby("JobName"):
        states = g["State"].str.split().str[0].value_counts().to_dict()
        rows.append(
            {
                "item": f"{name}: tasks / CPU-hours / states",
                "value": f"{len(g)} / {g['cpu_sec'].sum() / 3600:.1f} / {states}",
            }
        )
    out = Path(a.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
