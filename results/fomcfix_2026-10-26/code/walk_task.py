"""One linear walk job of results/fomcfix_2026-10-26 (RULE.txt), for a Slurm array task (code/lin_stage.sh walk1).

The final run's orchestrator (experiments/pooled_linear_rerun_2026_10_19_13.py: its argument parsing, walk_jobs and
pool) with its st_walk replaced, as results/window_sweep_2014start_2026-10-22/code/one_walk.py does: line <k> of the
task file names the cell's output directory, its flags and the job.

Task line (whitespace-separated key=value):
  out=<dir> cell=<NAME=0|1,...|-> only=<set/model> chunks=<K> job=<j>
      job j (0-based) of walk_jobs({set/model: K}, [set/model]): the final run's walk of that model, chunk j of K
  out=<dir> cell=<...> only=<set/model> start=<row> end=<row|-1> sub=<name>
      the model's walk (walk_jobs with one chunk) with --start / --end replaced and the output in
      <walk dir of the model>/<name> (the annual re-tuning's year walks, RULE.txt 7(b))
  first_day=<d> (optional, default 2014-01-02): the orchestrator's --first-day (walk start = the 250-solve boundary at
  or before its first row; the penalties are those of <out>/frozen_penalties.json)

usage: python walk_task.py <task file> <k (1-based)>
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / "experiments"))
import pooled_linear_rerun_2026_10_19_13 as O  # noqa: E402

LINE = Path(sys.argv[1]).read_text().splitlines()[int(sys.argv[2]) - 1]
KV = dict(x.split("=", 1) for x in LINE.split())
ARGV = ["walk", "--out", KV["out"], "--window-days", "3000", "--first-day", KV.get("first_day", "2014-01-02")]
ARGV += ["--first-fold", "3", "--ex-flag", "LATE_START_WARMUP=1"]
for f in [] if KV["cell"] == "-" else KV["cell"].split(","):
    ARGV += ["--ex-flag", f]
ARGV += ["--only", KV["only"]]
if "chunks" in KV:
    ARGV += ["--chunks", f"{KV['only']}={KV['chunks']}"]


def st_walk_task(a) -> None:
    for s in O.SETS:
        O.check_design(s)
    chunks = {}
    for spec in a.chunks:  # set/model=K, as st_walk
        sm, k = spec.split("=")
        s, m = sm.split("/")
        chunks[(s, m)] = int(k)
    jobs = O.walk_jobs(chunks, a.only)
    if "job" in KV:
        args, log, key = jobs[int(KV["job"])]
    else:
        assert len(jobs) == 1, [j[2] for j in jobs]
        args, _, key = jobs[0]
        args = list(args)
        for opt in ("start", "end"):
            args[args.index(f"--{opt}") + 1] = str(int(KV[opt]))
        o = O.walk_dir(key[0], key[1]) / KV["sub"]
        args[args.index("--out") + 1] = str(o)
        log = o / "run.log"
    O.say(f"task {sys.argv[2]}: {key}, walk start row {O.walk_start()}: {' '.join(args)}")
    O.pool([(args, log)], 1)


O.st_walk = st_walk_task
sys.argv = [sys.argv[0], *ARGV]
O.main()
