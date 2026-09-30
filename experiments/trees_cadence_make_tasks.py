"""Write the task files of the 16:00 tree cadence campaign (checklist I2: T10, T1, RS10, RS1).

Untuned rungs (specs/causal_tune_trees.py; one line = one single-threaded chunk PROCESS):
  cluster/trees_cadence_tasks_untuned.txt
    "<pack> <rung> <bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
  pack 0 = the canary (chunk c0 of every T10 and T1 arm), packs 1..K = the fleet; one
  Slurm array task hosts one pack (<= PACK_PROCS processes at once on PACK_PROCS cores).
Tuned rungs (specs/causal_tune_trees_tuned.py; one line = one chunk-arm, run with a pool
of POOL_CPUS single-threaded workers), in the tuned campaign's 8-field format that
experiments/reduce_trees_tuned_chunks.py reads:
  cluster/trees_cadence_tasks_rs10.txt, cluster/trees_cadence_tasks_rs1.txt
    "<bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
  cluster/trees_cadence_tasks_rs_canary.txt (chunk 0 of live_feasible, three models)

Chunk geometry: W = TRAIN_WIN = 2000 rows (one row per session at a one-bar segment);
chunk k of an arm covers whole-series OOS rows [o_k, o_k + rows) with START = W + o_k,
END = START + rows (-1 for the last chunk: to the end of the series) and HALO = W,
so a chunk's arrays are exactly the whole series' rows (run_executor slices the full
design).  Untuned T10 chunks start on multiples of REFIT_EVERY = 10 (refit anchors and
the importance cadence are then the whole series'); tuned chunks start on multiples of
TUNE_PER = 250 (the tuned spec asserts it), one tuning period per chunk.

Untuned chunk sizes: rows = TARGET_SEC / (seconds per OOS row), from the one-thread
seconds per refit the earlier 16:00 arms recorded on the same cluster (npz meta
fit_sec of results/linear_subsection_trees and _tuned; the 4-thread random-forest refits
scaled x4, sklearn's forest being parallel over its 100 trees); T10 pays one refit and one
10-row TreeSHAP block per 10 rows, T1 one refit per row and a 1-row TreeSHAP block at
every 10th.  Packs: longest-processing-time-first over the estimated seconds.

Run from the repo root: python experiments/trees_cadence_make_tasks.py
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = 2000
SEG = "bar1600"
N_OOS = 1469  # 16:00 forecast rows of the 2000-session per-bar series (every bucket and model)
BUCKETS = ("live_feasible", "all_features", "baseline")  # needed-first order
MODELS = ("lgbm", "xgb", "rf")
RUNGS = {"t10": (10, 1), "t1": (1, 10)}  # REFIT_EVERY, IMPORTANCE_EVERY
TUNE_PER = 250
TARGET_SEC = 600  # wall seconds per untuned chunk process (one core)
PACK_PROCS = 20  # processes (= cores) per untuned array task
# one-thread seconds per refit on the cluster (see the module note)
FIT_SEC = {
    ("lgbm", "all_features"): 6.07,
    (
        "lgbm",
        "live_feasible",
    ): 1.6,  # 0.72 s at 4 threads; the tuned arms' 1-thread refits 1.27 s
    ("lgbm", "baseline"): 0.27,
    ("xgb", "all_features"): 3.85,
    (
        "xgb",
        "live_feasible",
    ): 1.3,  # 0.57 s at 4 threads; tuned 1-thread 1.64 s (other configs)
    ("xgb", "baseline"): 0.19,
    ("rf", "all_features"): 4 * 8.19,
    ("rf", "live_feasible"): 4 * 2.36,
    ("rf", "baseline"): 4 * 0.47,
}
SHAP_SEC = {"lgbm": 0.02, "xgb": 0.02, "rf": 1.0}  # per TreeSHAP call (<= 10 rows)
PREP_SEC = {
    "all_features": 45,
    "live_feasible": 25,
    "baseline": 10,
}  # data prep per process


def sec_per_row(model: str, bucket: str, rung: str) -> float:
    re_, ie = RUNGS[rung]
    return (FIT_SEC[(model, bucket)] + SHAP_SEC[model] / ie) / re_


def chunks(rows_per: int, step: int) -> list[tuple[int, int, int]]:
    """[(k, start, end)] with chunk offsets on multiples of step."""
    if rows_per >= N_OOS:
        return [(0, W, -1)]  # the whole arm in one process
    rows_per = max(step, rows_per // step * step)
    n = math.ceil(N_OOS / rows_per)
    out = []
    for k in range(n):
        start = W + rows_per * k
        out.append((k, start, -1 if k == n - 1 else start + rows_per))
    return out


def main() -> None:
    procs = []  # (est_sec, rung, bucket, model, chunk, start, end)
    for rung, (re_, _ie) in RUNGS.items():
        for b in BUCKETS:
            for m in MODELS:
                spr = sec_per_row(m, b, rung)
                rows = int((TARGET_SEC - PREP_SEC[b]) / spr)
                for k, s, e in chunks(min(rows, N_OOS), re_):
                    n = (e if e >= 0 else W + N_OOS) - s
                    procs.append((PREP_SEC[b] + n * spr, rung, b, m, k, s, e))
    canary = [p for p in procs if p[4] == 0]
    fleet = sorted((p for p in procs if p[4] != 0), key=lambda p: -p[0])
    n_packs = math.ceil(len(fleet) / PACK_PROCS)
    load = [0.0] * n_packs
    size = [0] * n_packs
    pack_of = []
    for p in fleet:  # LPT: the least-loaded pack with room
        j = min(
            (i for i in range(n_packs) if size[i] < PACK_PROCS), key=lambda i: load[i]
        )
        load[j] += p[0]
        size[j] += 1
        pack_of.append(j + 1)
    lines = [
        f"0 {r} {b} {m} {W} {SEG} {k} {s} {e} {W}" for (_t, r, b, m, k, s, e) in canary
    ] + [
        f"{j} {r} {b} {m} {W} {SEG} {k} {s} {e} {W}"
        for j, (_t, r, b, m, k, s, e) in sorted(
            zip(pack_of, fleet), key=lambda x: (x[0], x[1][1:])
        )
    ]
    f = ROOT / "cluster" / "trees_cadence_tasks_untuned.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(
        f"{f.name}: {len(canary)} canary processes (pack 0), {len(fleet)} fleet processes in "
        f"{n_packs} packs; estimated max process {max(p[0] for p in procs):.0f}s, "
        f"pack sums {min(load):.0f}..{max(load):.0f}s, total {sum(p[0] for p in procs) / 3600:.1f} core-h"
    )
    for rung in RUNGS:
        for b in BUCKETS:
            for m in MODELS:
                ps = [p for p in procs if p[1:4] == (rung, b, m)]
                print(
                    f"  {rung:3s} {b:13s} {m:4s} {len(ps):3d} chunks, est {sum(p[0] for p in ps) / 3600:5.2f} h"
                )

    tuned = []
    n_t = math.ceil(N_OOS / TUNE_PER)
    for b in BUCKETS:
        for m in MODELS:
            for k in range(n_t):
                s = W + TUNE_PER * k
                tuned.append(
                    f"{b} {m} {W} {SEG} {k} {s} {-1 if k == n_t - 1 else s + TUNE_PER} {W}"
                )
    for rung in ("rs10", "rs1"):
        f = ROOT / "cluster" / f"trees_cadence_tasks_{rung}.txt"
        f.write_text("\n".join(tuned) + "\n", encoding="utf-8", newline="\n")
        print(f"{f.name}: {len(tuned)} chunk-arms")
    f = ROOT / "cluster" / "trees_cadence_tasks_rs_canary.txt"
    f.write_text(
        f"live_feasible lgbm,xgb,rf {W} {SEG} 0 {W} {W + TUNE_PER} {W}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{f.name}: 1 line (chunk 0 of live_feasible, three models)")


if __name__ == "__main__":
    main()
