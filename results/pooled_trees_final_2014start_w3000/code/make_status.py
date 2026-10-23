"""run.json and STATUS.md of a final tree run directory (pulled from CARC): code, environment, design, cadence, pieces,
Slurm jobs and CPU-hours by arm.

usage: python make_status.py <run dir> <window days> <runner commit> <repo commit at submission> [<merge job>]
Reads <run dir>/pieces/*.json, progress.csv, jobs.csv (sacct of the run's arrays), tasks.txt, tasks_*.txt,
forecasts.json, timing/sec_refit.csv, design/MD5SUMS_<set>, design/design_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

run, W, RUNNER, REPO_C = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
MERGE = sys.argv[5] if len(sys.argv) > 5 else ""
P = pd.read_csv(run / "progress.csv")
J = pd.read_csv(run / "jobs.csv", dtype=str)
fj = json.loads((run / "forecasts.json").read_text())
dc = json.loads((run / "design" / "design_check.json").read_text())[f"w{W}"]
md5 = {
    s: (run / "design" / f"MD5SUMS_{s}").read_text().split("\n")
    for s in ("baseline", "all_features")
}
tasks = [
    ln.split() for ln in (run / "tasks.txt").read_text().splitlines() if ln.strip()
]
metas = [
    json.loads(p.read_text())
    for p in sorted((run / "pieces").glob("*.json"))
    if not p.name.endswith(".ckpt.json")
]
P["cpu_h"] = P["wall_sec"] * P["threads"] / 3600
# sacct: array tasks (JobID a_b, no step), allocated CPU-h = CPUTimeRAW / 3600, used = TotalCPU
J = J[J["JobID"].str.contains("_") & ~J["JobID"].str.contains(r"\.")].copy()
J["alloc_cpu_h"] = J["CPUTimeRAW"].astype(float) / 3600


def totalcpu_h(s: str) -> float:
    """sacct TotalCPU [D-]HH:MM:SS(.ms) or MM:SS.ms in hours."""
    days = "0"
    if "-" in s:
        days, s = s.split("-")
    parts = [float(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return int(days) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


J["used_cpu_h"] = J["TotalCPU"].map(totalcpu_h)
J["array"] = J["JobID"].str.split("_").str[0]
by_arm = []
for (e, s), g in P.groupby(["est", "predictor_set"]):
    want = sum(1 for t in tasks if t[0] == e and t[1] == s)
    by_arm.append(
        {
            "est": e,
            "predictor_set": s,
            "pieces": len(g),
            "tasks": want,
            "rows": int(g["rows"].sum()),
            "refits": int(g["refits"].sum()),
            "threads": sorted(g["threads"].unique().tolist()),
            "cpu_h": round(float(g["cpu_h"].sum()), 1),
            "max_piece_wall_h": round(float(g["wall_sec"].max() / 3600), 2),
            "cpu_models": g["cpu"].value_counts().to_dict(),
        }
    )
A = pd.DataFrame(by_arm)
arrays = sorted(set(J["array"]))
m0 = metas[0]
info = {
    "run": run.name,
    "branch": "insurance-selling-2026-09-19-e34-trees-final",
    "runner": "experiments/pooled_trees_rerun_13.py walk, through cluster/slurm/pooltree_rerun13.sbatch (MODE=walk)",
    "runner_commit": RUNNER,
    "repo_commit_at_submission": REPO_C,
    "runner_note": "the printed 2012-start w1000 run's runner (db59af2) with check() accepting the 564-column final design; "
    "gate: results/pooled_trees_final_2014start_w3000/gate/",
    "cluster": "CARC Discovery (USC), Slurm account pollok_1603, partition main, --constraint epyc-7513|epyc-7542",
    "environment": {
        "python_bin": "/scratch1/jc_905/envs/harxhar-r13/bin/python",
        **m0["versions"],
    },
    "design": {
        "source": f"/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w{W}/_work/design_{{baseline,all_features}}",
        "copy": f"/scratch1/jc_905/trees_final/designs/w{W}/_work/design_{{baseline,all_features}} (md5 of the copy = md5 of the source)",
        "md5": {s: [ln for ln in v if ln.strip()] for s, v in md5.items()},
        "check": dc,
    },
    "window_days": W,
    "window_rows": W * 13,
    "span": fj["span"],
    "refit": fj["refit"],
    "first_row": dc["tree_first_date"],
    "bars_per_arm": dc["tree_rows"],
    "days": dc["tree_days"],
    "configs": {
        m["est"]: {
            "name": m["config"],
            "sha": m["sha"],
            "file": m["config_file"],
            "file_sha256": m["config_file_sha256"],
        }
        for m in metas
    },
    "threads": {f"{r.est}/{r.predictor_set}": r.threads for r in A.itertuples()},
    "pieces": {f"{r.est}/{r.predictor_set}": int(r.pieces) for r in A.itertuples()},
    "task_rule": "experiments/pooled_trees_rerun_13.py tasks --refits bar --target-hours 3 on the CARC time probe "
    "(timing/sec_refit.csv, 3 refits at the first 2014-01-02 rows, 8 threads), arms cheapest first (the printed run's rule)",
    "slurm_arrays": arrays,
    "merge_job": MERGE,
    "sacct": {
        "array_tasks": len(J),
        "states": J["State"].str.split().str[0].value_counts().to_dict(),
        "alloc_cpu_h": round(float(J["alloc_cpu_h"].sum()), 1),
        "used_cpu_h": round(float(J["used_cpu_h"].sum()), 1),
    },
    "cpu_h_by_arm": {f"{r.est}/{r.predictor_set}": r.cpu_h for r in A.itertuples()},
    "cpu_h_total": round(float(P["cpu_h"].sum()), 1),
    "cpu_models": P["cpu"].value_counts().to_dict(),
    "versions_distinct": int(P["versions"].nunique()),
}
(run / "run.json").write_text(json.dumps(info, indent=1, default=str))
L = [
    f"# Pooled trees, final design (2014 start), w{W}, bar refits, {fj['span'][0]}..{fj['span'][1]} (CARC Discovery)",
    "",
    (
        f"Runner commit {RUNNER[:7]} (repo clone at {REPO_C[:7]} when submitted), env harxhar-r13 "
        f"(python {m0['versions']['python']}, numpy {m0['versions']['numpy']}, lightgbm {m0['versions']['lightgbm']}, "
        f"xgboost {m0['versions']['xgboost']}, sklearn {m0['versions']['sklearn']}). WINDOW_DAYS={W}, REFIT=bar. First row "
        f"{dc['tree_first_date']}, {dc['tree_rows']:,} bars an arm on {dc['tree_days']:,} days."
    ),
    "",
    "## Jobs",
]
for a in arrays:
    g = J[J["array"] == a]
    L.append(
        f"- {a} {g['JobName'].iloc[0]} ({len(g)} tasks): {g['State'].str.split().str[0].value_counts().to_dict()}, "
        f"{g['alloc_cpu_h'].sum():.1f} allocated CPU-h, {g['used_cpu_h'].sum():.1f} used."
    )
if MERGE:
    L.append(f"- {MERGE} merge.")
L += [
    "",
    "## Merge",
    f"forecasts.parquet: {sum(len(v) for v in fj['pieces'].values())} pieces, arms {sorted(fj['pieces'])}.",
    "",
    "## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)",
]
for r in A.itertuples():
    L.append(
        f"- {r.est} {r.predictor_set}: {r.pieces} / {r.tasks} pieces, {r.rows} rows, {r.cpu_h} CPU-h, threads {r.threads}, "
        f"longest piece {r.max_piece_wall_h} h"
    )
L.append(f"- total: {P['cpu_h'].sum():.1f} CPU-h")
L += ["", "## CPU model by piece"] + [
    f"- {k}: {v}" for k, v in P["cpu"].value_counts().items()
]
L += [
    "",
    "## Files",
    (
        "forecasts.parquet + forecasts.json, pieces/*.json (piece parquets stay on CARC: "
        f"/scratch1/jc_905/trees_final/runs/w{W}/pieces), tasks*.txt, timing/, jobs.csv (sacct), progress.csv, run.json, "
        "design/ (md5, design_check.json), scores/, code/."
    ),
]
(run / "STATUS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
