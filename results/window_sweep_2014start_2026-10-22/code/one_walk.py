"""One walk job of the orchestrator (experiments/pooled_linear_rerun_2026_10_19_13.py walk), for a Slurm array task.

The orchestrator's own argument parsing and walk_jobs(); only st_walk is replaced, to run job SLURM_ARRAY_TASK_ID
(1-based, in walk_jobs order) instead of the whole pool.  usage: python one_walk.py walk <the orchestrator's options>
"""

import os
import sys

sys.path.insert(0, "/scratch1/jc_905/wsweep/code/experiments")
import pooled_linear_rerun_2026_10_19_13 as O  # noqa: E402

IDX = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1


def st_walk_one(a) -> None:
    for s in O.SETS:
        O.check_design(s)
    chunks = {}
    for spec in a.chunks:  # set/model=K, as st_walk
        sm, k = spec.split("=")
        s, m = sm.split("/")
        chunks[(s, m)] = int(k)
    jobs = O.walk_jobs(chunks, a.only)
    args, log, key = jobs[IDX]
    O.say(f"walk task {IDX + 1} of {len(jobs)}: {key}, start row {O.walk_start()}: {' '.join(args)}")
    O.pool([(args, log)], 1)


O.st_walk = st_walk_one
O.main()
