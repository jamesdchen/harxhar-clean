"""run_info.json of one CARC run directory (downloaded): code commit, cluster, nodes / CPU models / Slurm job ids by
stage, stage wall times, environment, runner configuration, design checks.

usage: python run_info.py <run dir> <window days> <first fold> <code commit> [first day]
"""

import json
import re
import sys
from pathlib import Path

R, WD, FF, COMMIT = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
W = R / "_work"
hosts = []
for ln in (W / "hosts.txt").read_text().splitlines():
    m = re.match(r"(\S+) host (\S+) \|\s*(.*?) \| SLURM_JOB_ID (\d*) \| (.*)$", ln)
    if m:
        hosts.append(dict(stage=m.group(1), host=m.group(2), cpu=m.group(3), slurm_job=m.group(4), start=m.group(5)))
times = {}
for ln in (W / "stage_times.txt").read_text().splitlines():
    st, sec, host = ln.split()
    times[st] = int(sec)
fr = json.loads((R / "frozen_penalties.json").read_text())
logs = {}
for s in ("baseline", "all_features"):
    t = (W / "logs" / f"design_{s}.log").read_text()
    logs[s] = [ln.strip() for ln in t.splitlines() if "[scale]" in ln]
walk_wall = max(times[k] for k in times if k.startswith("walk1."))
info = dict(
    code_commit=COMMIT,
    branch="insurance-selling-2026-09-19-e34-wsweep",
    code_note={
        "9d3fc057a96237dd1dfefc216de16874c1ce2224": "e34 e0f3a7a (LATE_START_WARMUP, EXCLUDE_FOMC_RELEASE, UNOBSERVED_NEUTRAL, "
        "WALK_KKT_RESEED on by default) + the --first-fold option (818b153)",
        "6c264b71b38cf6daf72496ccc2348ee787a99033": "e34 8898522 (LATE_START_WARMUP, EXCLUDE_FOMC_RELEASE, UNOBSERVED_NEUTRAL, "
        "WALK_KKT_RESEED, STANDARDIZE_PENALTY final -- scaled coordinates + binomial availability rule, grid rule -- on "
        "by default) + the --first-fold option (818b153)",
    }.get(COMMIT, ""),
    first_day=sys.argv[5] if len(sys.argv) > 5 else None,
    cluster="CARC Discovery (USC), Slurm account pollok_1603, partition main, --constraint epyc-7513|epyc-7542",
    window_days=WD,
    first_fold=FF,
    environment=dict(
        python="3.12.3 (uv)", numpy="1.26.4 (OpenBLAS 0.3.23.dev)", numba="0.68.0", pandas="2.3.3", scipy="1.17.1",
        OPENBLAS_CORETYPE="Prescott (run B's kernel on the laptop)",
        NPY_DISABLE_CPU_FEATURES="AVX512 targets where the node has them (none on EPYC)",
        threads="1 per runner process (the orchestrator's run())",
    ),
    runner_cfg=fr["info"]["runner_cfg"],
    folds=fr["info"]["folds"],
    rule=fr["info"]["rule"],
    stages=dict(
        design=dict(wall_sec=times.get("design"), nodes=[h for h in hosts if h["stage"] == "design"]),
        pretune_select=dict(wall_sec=times.get("pretune_select"), nodes=[h for h in hosts if h["stage"] == "pretune_select"]),
        walk=dict(tasks=14, max_task_wall_sec=walk_wall, sum_task_sec=sum(v for k, v in times.items() if k.startswith("walk1.")),
                  nodes=sorted({(h["host"], h["cpu"]) for h in hosts if h["stage"].startswith("walk1.")})),
        score=dict(wall_sec=times.get("score"), nodes=[h for h in hosts if h["stage"] == "score"]),
    ),
    design_log_scale_lines=logs,
)
(R / "run_info.json").write_text(json.dumps(info, indent=1, default=list))
print(json.dumps({k: info[k] for k in ("code_commit", "window_days", "first_fold")}), times)
