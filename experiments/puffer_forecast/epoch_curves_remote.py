"""Stdlib-only (cluster python3 3.6): reduce a task's steps.csv(.gz) to per-epoch curves, so the large step logs
need not leave the cluster.  For every job: the row-weighted epoch mean of the minibatch loss (weights
min(B, N - k B) from jobs_out.csv's N and batch), the mean and max grad norm, the clipped-step count above a
max_grad_norm read from the task file.  Writes <task>/epoch_curves.csv.

    python3 epoch_curves_remote.py RUN_DIR [RUN_DIR ...]
"""

import csv
import glob
import gzip
import io
import os
import sys


def one(task):
    jo = os.path.join(task, "jobs_out.csv")
    if not os.path.isfile(jo):
        return 0
    nb = {}
    with open(jo) as fh:
        for r in csv.DictReader(fh):
            nb[r["id"]] = (int(r["N"]), int(r["batch"]))
    sp = os.path.join(task, "steps.csv.gz")
    fh = (
        gzip.open(sp, "rt")
        if os.path.isfile(sp)
        else open(os.path.join(task, "steps.csv"))
    )
    acc = {}
    for r in csv.DictReader(io.TextIOWrapper(fh.buffer) if False else fh):
        jid = r["id"]
        N, B = nb[jid]
        k = int(r["step"])
        w = min(B, N - k * B)
        key = (jid, int(r["epoch"]))
        a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
        a[0] += float(r["loss"]) * w
        a[1] += w
        g = float(r["grad_norm"])
        a[2] += g
        a[3] = max(a[3], g)
        a[4] += 1
    fh.close()
    with open(os.path.join(task, "epoch_curves.csv"), "w") as out:
        out.write("id,epoch,loss_w,gn_mean,gn_max,steps\n")
        for (jid, e), a in sorted(acc.items()):
            out.write(
                "%s,%d,%.9g,%.9g,%.9g,%d\n"
                % (jid, e, a[0] / a[1], a[2] / a[4], a[3], a[4])
            )
    return len(acc)


for run in sys.argv[1:]:
    for task in sorted(glob.glob(os.path.join(run, "task_*"))):
        print(task, one(task))
