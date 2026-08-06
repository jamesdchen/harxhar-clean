"""Standalone runner for the unification campaign — no hpc-agent machinery.

One invocation = one (arm, chunk) task:

    python experiments/run_unification.py --arm a0_ols_har --chunk-index 17 \
        --output-dir results/unification

Chunk geometry mirrors the har-base sweep: the emit space [WARMUP, n_rows)
is split evenly into N_CHUNKS contiguous chunks; --chunk-index selects one.
Each task writes <output-dir>/<arm>/chunk_<index>.npz (the per-bar
persistence contract: row_id, timestamp, y_fit, yhat, y_raw, B_t,
valid_mask) plus a chunk_<index>.meta.json with sufficient statistics.
"""

import _bootstrap  # noqa: F401  (sys.path: repo root + experiments/)

import argparse
import json
import os
import time

from src.unification import ARMS, compute, panel_length

WARMUP = 24_000
N_CHUNKS = 100


def chunk_bounds(index: int, n_rows: int) -> tuple[int, int]:
    """Even split of [WARMUP, n_rows) into N_CHUNKS contiguous chunks."""
    if not 0 <= index < N_CHUNKS:
        raise SystemExit(f"chunk-index {index} outside [0, {N_CHUNKS})")
    span = n_rows - WARMUP
    start = WARMUP + (span * index) // N_CHUNKS
    end = WARMUP + (span * (index + 1)) // N_CHUNKS
    return start, end


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--chunk-index", type=int, required=True)
    p.add_argument("--output-dir", default="results/unification")
    p.add_argument("--halo", type=int, default=WARMUP)
    args = p.parse_args()

    n_rows = panel_length()
    start, end = chunk_bounds(args.chunk_index, n_rows)

    arm_dir = os.path.join(args.output_dir, args.arm)
    os.makedirs(arm_dir, exist_ok=True)
    out = os.path.join(arm_dir, f"chunk_{args.chunk_index:03d}.npz")

    t0 = time.time()
    compute(
        argparse.Namespace(
            arm=args.arm,
            chunk_start=start,
            chunk_end=end,
            halo=args.halo,
            window=ARMS[args.arm].window,
            output_file=out,
        )
    )
    meta = {
        "arm": args.arm,
        "chunk_index": args.chunk_index,
        "chunk_start": start,
        "chunk_end": end,
        "n_rows_panel": n_rows,
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(out.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(
        f"DONE {args.arm} chunk {args.chunk_index} [{start},{end}) {meta['wall_sec']}s"
    )


if __name__ == "__main__":
    main()
