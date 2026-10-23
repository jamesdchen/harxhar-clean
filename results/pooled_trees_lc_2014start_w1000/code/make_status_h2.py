"""run.json and STATUS.md of a tree learning-curve run (Hoffman2): code, environment, design, cadence, pieces, UGE jobs,
CPU models and CPU-hours by arm.

usage: python make_status_h2.py <run dir> <window days> <runner commit> <code-tree commit> <jobs commit> <merge job>
Reads <run dir>/pieces/*.json (the piece metas), progress.csv (collect.py), jobs.csv (qacct_parse.py of the run's
arrays), forecasts.json, tasks.txt, tasks_gb.txt, timing/sec_refit_*_t1.csv, design/design_check.json,
design/MD5SUMS, environment/freeze.txt and ../pooled_trees_lc_2014start_w1000/gate/cpu_gate.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

run, W = Path(sys.argv[1]), int(sys.argv[2])
RUNNER, CODE, JOBS, MERGE = sys.argv[3:7]
NOTES = (
    (run / "NOTES.md").read_text(encoding="utf-8").strip()
    if (run / "NOTES.md").is_file()
    else ""
)
P = pd.read_csv(run / "progress.csv")
J = pd.read_csv(run / "jobs.csv", dtype={"jobnumber": str, "taskid": str})
fj = json.loads((run / "forecasts.json").read_text())
dc = json.loads((run / "design" / "design_check.json").read_text())[f"w{W}"]
md5 = [ln for ln in (run / "design" / "MD5SUMS").read_text().splitlines() if ln.strip()]
freeze = [
    ln
    for ln in (run / "environment" / "freeze.txt").read_text().splitlines()
    if ln.strip()
]
tasks = [
    ln.split() for ln in (run / "tasks_gb.txt").read_text().splitlines() if ln.strip()
]
metas = [
    json.loads(p.read_text())
    for p in sorted((run / "pieces").glob("*.json"))
    if not p.name.endswith(".ckpt.json")
]
gate = pd.read_csv(
    run.parent / "pooled_trees_lc_2014start_w1000" / "gate" / "cpu_gate.csv"
)
P["cpu_h"] = P["wall_sec"] * P["threads"] / 3600
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
m0 = metas[0]
jobs = sorted(J["jobnumber"].unique())
info = {
    "run": run.name,
    "branch": "insurance-selling-2026-09-19-e34-tree-lc",
    "runner": "experiments/pooled_trees_rerun_13.py walk (code/h2/walk.sh, one UGE array task a piece)",
    "runner_commit": RUNNER,
    "runner_note": "the final tree walks' runner (654c476: the printed 2012-start w1000 run's runner db59af2 with check() "
    "accepting the 564-column final design)",
    "code_tree_commit": CODE,
    "job_scripts_commit": JOBS,
    "cluster": "Hoffman2 (UCLA IDRE, UGE 8.6.4), /u/scratch/j/jamesdc1/tlc; every (window, tree, set) of this run on Hoffman2",
    "environment": {
        "container": "Debian bookworm-slim (copy of /u/scratch/j/jamesdc1/wsweep/deb.sif), apptainer 1.2.2",
        "python": m0["versions"]["python"]
        + " (uv-managed CPython, venv /u/scratch/j/jamesdc1/tlc/venv)",
        "libgomp": "Debian bookworm libgomp1 12.2.0-14+deb12u1 (LightGBM); XGBoost and scikit-learn their wheels' own",
        "env_vars": "OMP/MKL/OpenBLAS/numba threads = 1, OPENBLAS_CORETYPE=Haswell, numpy AVX-512 dispatch off",
        "versions": m0["versions"],
        "freeze": freeze,
    },
    "design": {
        "source": f"CARC /scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w{W}/_work/"
        "design_{baseline,all_features} (the 2014-start window sweep, built on CARC)",
        "copy": f"/u/scratch/j/jamesdc1/tlc/designs/w{W}/_work/ (md5 of the copy = md5 at the source, checked on arrival)",
        "md5": md5,
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
    "task_rule": "experiments/pooled_trees_rerun_13.py tasks --refits bar --target-hours 3 on the Hoffman2 1-thread time "
    "probe (timing/sec_refit_gb_t1.csv: 3 refits at the first 2014-01-02 rows, Gold 6240), arms cheapest first "
    "(the printed run's rule)",
    "uge_arrays": jobs,
    "merge_job": MERGE,
    "accounting": {
        "records": len(J),
        "exit_status": J["exit_status"].value_counts().to_dict(),
        "slot_h": round(float(J["slot_h"].sum()), 1),
        "cpu_h": round(float(J["cpu_h"].sum()), 1),
        "hosts": int(J["hostname"].nunique()),
    },
    "cpu_h_by_arm": {f"{r.est}/{r.predictor_set}": r.cpu_h for r in A.itertuples()},
    "cpu_h_total": round(float(P["cpu_h"].sum()), 1),
    "cpu_models": P["cpu"].value_counts().to_dict(),
    "cpu_gate": gate[
        ["cpu", "threads", "gate_pieces", "gate_bit_equal", "gated"]
    ].to_dict("records"),
    "versions_distinct": int(P["versions"].nunique()),
}
(run / "run.json").write_text(json.dumps(info, indent=1, default=str))
L = [
    f"# Pooled trees, learning curve, final design (2014 start), w{W}, bar refits, {fj['span'][0]}..{fj['span'][1]} (Hoffman2)",
    "",
    (
        f"LightGBM and XGBoost, HAR benchmark and full set.  Runner {RUNNER[:7]} (code tree {CODE[:7]}, job scripts "
        f"{JOBS[:7]}), python {m0['versions']['python']}, numpy {m0['versions']['numpy']}, lightgbm "
        f"{m0['versions']['lightgbm']}, xgboost {m0['versions']['xgboost']}, sklearn {m0['versions']['sklearn']}, one thread "
        f"a task.  WINDOW_DAYS={W}, REFIT=bar.  First row {dc['tree_first_date']}, {dc['tree_rows']:,} bars an arm on "
        f"{dc['tree_days']:,} days."
    ),
    "",
    "## UGE jobs",
]
for a in jobs:
    g = J[J["jobnumber"] == a]
    L.append(
        f"- {a} {g['jobname'].iloc[0]} ({len(g)} records): exit status {g['exit_status'].value_counts().to_dict()}, "
        f"{g['slot_h'].sum():.1f} slot-hours, {g['cpu_h'].sum():.1f} CPU-hours, {g['hostname'].nunique()} hosts."
    )
if MERGE not in jobs:
    L.append(f"- {MERGE} merge (on Hoffman2).")
L += [
    "",
    "## Merge",
    f"forecasts.parquet: {sum(len(v) for v in fj['pieces'].values())} pieces, arms {sorted(fj['pieces'])}.",
    "",
    "## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)",
]
for r in A.itertuples():
    L.append(
        f"- {r.est} {r.predictor_set}: {r.pieces} / {r.tasks} pieces, {r.rows} rows, {r.cpu_h} CPU-h, threads "
        f"{r.threads}, longest piece {r.max_piece_wall_h} h"
    )
L.append(f"- total: {P['cpu_h'].sum():.1f} CPU-h")
L += [
    "",
    "## CPU model by piece (each model passed the 1-thread gate: ../pooled_trees_lc_2014start_w1000/gate/)",
]
L += [f"- {k}: {v}" for k, v in P["cpu"].value_counts().items()]
L += [
    "",
    "## Files",
    (
        "forecasts.parquet + forecasts.json, pieces/*.json (the piece parquets stay on Hoffman2: "
        f"/u/scratch/j/jamesdc1/tlc/runs/w{W}/pieces), tasks*.txt, timing/, jobs.csv (qacct), progress.csv, run.json, "
        "design/ (md5, design_check.json), environment/ (pip freeze), scores/, code/."
    ),
]
if NOTES:
    L += ["", NOTES]
(run / "STATUS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
