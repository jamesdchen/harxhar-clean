"""Batched unification runner: several chunks in one worker process.

One invocation = one arm x chunk-list task. It reuses the process-global panel
and caches the arm's tuned block design, avoiding a full panel/design rebuild
for every chunk. Output files remain the per-chunk contract, so ordinary
one-chunk tasks and batched tasks are idempotently interchangeable.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import gc
import json
import os
import time
from types import SimpleNamespace

import src.unification as u

WARMUP = 24_000
N_CHUNKS = 100


def chunk_bounds(index: int, n_rows: int) -> tuple[int, int]:
    if not 0 <= index < N_CHUNKS:
        raise SystemExit(f"chunk-index {index} outside [0, {N_CHUNKS})")
    span = n_rows - WARMUP
    start = WARMUP + (span * index) // N_CHUNKS
    end = WARMUP + (span * (index + 1)) // N_CHUNKS
    return start, end


def parse_chunks(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(u.ARMS))
    ap.add_argument("--chunks", required=True, help="e.g. 9-27 or 9,10,14")
    ap.add_argument("--output-dir", default="results/unification")
    ap.add_argument("--halo", type=int, default=WARMUP)
    ap.add_argument("--solver", choices=("exact", "rank2"), default="exact")
    args = ap.parse_args()

    original_design = u._tuned_blocks_design
    design_cache: dict[tuple[str, int], tuple[object, object]] = {}

    def cached_design(p, spec, window: int, arm: str = ""):
        key = (arm, int(window))
        if key not in design_cache:
            design_cache[key] = original_design(p, spec, window, arm=arm)
        return design_cache[key]

    u._tuned_blocks_design = cached_design
    if args.solver == "rank2":
        u._walk_blocks_tuned = u._walk_blocks_tuned_rank2
    n_rows = u.panel_length()
    arm_dir = os.path.join(args.output_dir, args.arm)
    os.makedirs(arm_dir, exist_ok=True)

    for index in parse_chunks(args.chunks):
        start, end = chunk_bounds(index, n_rows)
        out = os.path.join(arm_dir, f"chunk_{index:03d}.npz")
        if os.path.exists(out):
            print(f"SKIP existing {out}", flush=True)
            continue
        t0 = time.time()
        u.compute(
            SimpleNamespace(
                arm=args.arm,
                chunk_start=start,
                chunk_end=end,
                halo=args.halo,
                window=u.ARMS[args.arm].window,
                output_file=out,
            )
        )
        meta = {
            "arm": args.arm,
            "chunk_index": index,
            "chunk_start": start,
            "chunk_end": end,
            "n_rows_panel": n_rows,
            "wall_sec": round(time.time() - t0, 1),
            "batched_design_cache": True,
            "solver": args.solver,
        }
        with open(out.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        print(
            f"DONE {args.arm} chunk {index} [{start},{end}) {meta['wall_sec']}s",
            flush=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()
