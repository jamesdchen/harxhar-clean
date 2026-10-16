"""Task files of the pooled tree walks (2026-10-07): cluster/pooltree_tasks_<est>_<set>.txt, one line per array task,
"<est> <set> <chunk>[,<chunk>...]" -- the bank's 100 chunks of [24000, 300317) (2,763-2,764 rows each, one refit per row),
packed so a task runs about --target-h hours at the measured seconds per refit (8 threads, the canary's), never more than
--max-chunks chunks.  Prints the projection (core-hours = tasks x hours x 8 CPUs).

Run:  python experiments/pooled_trees_make_tasks.py --est lgbm --set baseline --sec-per-fit 0.9 [--target-h 6]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_CHUNKS = 100
ROWS = (300317 - 24000) / N_CHUNKS  # 2763.17 refits a chunk
CPUS = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument(
        "--sec-per-fit",
        type=float,
        required=True,
        help="wall seconds per refit at 8 threads (canary)",
    )
    ap.add_argument("--target-h", type=float, default=6.0)
    ap.add_argument("--max-chunks", type=int, default=10)
    a = ap.parse_args()
    chunk_h = ROWS * a.sec_per_fit / 3600
    k = max(
        1,
        min(
            a.max_chunks,
            math.floor(a.target_h / chunk_h) if chunk_h < a.target_h else 1,
        ),
    )
    # longest chunks first is irrelevant (equal sizes); keep chunk order so a resubmission resumes in order
    groups = [list(range(i, min(i + k, N_CHUNKS))) for i in range(0, N_CHUNKS, k)]
    lines = [f"{a.est} {a.set} {','.join(map(str, g))}" for g in groups]
    out = ROOT / "cluster" / f"pooltree_tasks_{a.est}_{a.set}.txt"
    out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(
        f"{out.name}: {len(lines)} tasks of {k} chunk(s); {chunk_h:.2f} h a chunk at {a.sec_per_fit:.2f} s/refit; "
        f"wall per task {k * chunk_h:.1f} h; total {N_CHUNKS * chunk_h * CPUS:.0f} core-h ({N_CHUNKS * chunk_h:.0f} task-hours)"
    )


if __name__ == "__main__":
    main()
