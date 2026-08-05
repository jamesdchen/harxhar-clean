"""Thin Slurm/SGE array-job entry point for the ``eval_sim`` Monte Carlo sweep.

Orchestration ONLY -- no simulation logic lives here. One array task maps to one grid coordinate
(optionally one path-chunk of it): resolve array-id -> coordinate, build the (sampler, eval, sim)
triple via :mod:`eval_sim.grid`, run :func:`eval_sim.sim_core.run`, and write a per-path parquet plus a
summary json. Seeding is ``seed*1e6 + array_id*1e3 + chunk`` so every (coordinate, chunk) is independent
and reproducible; path-chunking splits a large ``n_paths`` across tasks that recombine downstream.
"""

from __future__ import annotations

import os
import sys

# Make ``eval_sim`` importable no matter where the scheduler launches this script from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval_sim import grid  # noqa: E402
from eval_sim import sim_core  # noqa: E402
from eval_sim.config import EvalConfig, SimConfig  # noqa: E402
from eval_sim.metrics import summarize  # noqa: E402

# Every coordinate axis the sweep can vary; missing keys are backfilled per-path for a rectangular table.
COORD_KEYS: tuple[str, ...] = ("size_mult", "regime", "max_drawdown", "profit_target")


def _coord_defaults() -> dict[str, float]:
    """Fallback value for each coord key when a coordinate omits it (config default, else NaN)."""
    ev, sm = EvalConfig(), SimConfig()
    return {
        "size_mult": sm.size_mult,
        "regime": float("nan"),  # a pnl-sampler knob; no config field to source
        "max_drawdown": ev.max_drawdown,
        "profit_target": ev.profit_target,
    }


def _resolve_array_id(cli_value: int | None) -> int:
    """CLI value wins; else Slurm (0-based) then SGE (1-based -> 0-based); error if none present."""
    if cli_value is not None:
        return cli_value
    slurm = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm is not None:
        return int(slurm)
    sge = os.environ.get("SGE_TASK_ID")
    if sge is not None:
        return int(sge) - 1
    raise SystemExit(
        "no --array-id and neither SLURM_ARRAY_TASK_ID nor SGE_TASK_ID is set"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the array-job CLI (all knobs have HPC-sane defaults except the required out-dir)."""
    p = argparse.ArgumentParser(
        description="eval_sim HPC array worker (one coordinate per task)"
    )
    p.add_argument(
        "--array-id", type=int, default=None, help="coordinate index (else from env)"
    )
    p.add_argument(
        "--out-dir", required=True, help="output directory (created if absent)"
    )
    p.add_argument(
        "--n-paths",
        type=int,
        default=100_000,
        help="paths per coordinate (pre-chunking)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed for the deterministic seed formula",
    )
    p.add_argument(
        "--max-steps", type=int, default=250, help="hard cap on PnL periods per path"
    )
    p.add_argument(
        "--chunk", type=int, default=0, help="this chunk's index within the coordinate"
    )
    p.add_argument(
        "--n-chunks", type=int, default=1, help="path-chunks per coordinate (>=1)"
    )
    return p.parse_args(argv)


def _build_frame(
    res: sim_core.PathResults, coord: dict, array_id: int, offset: int
) -> pd.DataFrame:
    """One row per path: identifiers + raw outcomes + a backfilled column for every coord key."""
    n = res.n
    df = pd.DataFrame(
        {
            "path_id": np.arange(n, dtype=np.int64) + offset,
            "coord_id": np.full(n, array_id, dtype=np.int64),
            "outcome": res.outcome,
            "stop_step": res.stop_step,
            "terminal_equity": res.terminal_equity,
            "peak_equity": res.peak_equity,
            "max_dd": res.max_dd,
            "days": res.days,
        }
    )
    defaults = _coord_defaults()
    for key in COORD_KEYS:
        df[key] = coord.get(key, defaults[key])
    return df


def _write_paths(df: pd.DataFrame, out_dir: str, array_id: int, chunk: int) -> str:
    """Write the per-path table as parquet, falling back to CSV if no parquet engine is available."""
    stem = os.path.join(out_dir, f"cell_{array_id:03d}_chunk{chunk:02d}")
    try:
        path = f"{stem}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
    except ImportError as exc:
        path = f"{stem}.csv"
        df.to_csv(path, index=False)
        print(f"[warn] pyarrow unavailable ({exc}); wrote CSV fallback: {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    """Resolve one array task to a coordinate, simulate its path-chunk, and persist the outputs."""
    args = _parse_args(argv)
    array_id = _resolve_array_id(args.array_id)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.n_chunks < 1:
        raise SystemExit("--n-chunks must be >= 1")
    chunk_n_paths = args.n_paths // args.n_chunks
    if chunk_n_paths < 1:
        raise SystemExit(
            f"empty chunk: n_paths={args.n_paths} // n_chunks={args.n_chunks} == 0"
        )

    coord = grid.coord_for(array_id)
    seed = args.seed * 1_000_000 + array_id * 1000 + args.chunk
    offset = args.chunk * chunk_n_paths

    t0 = time.perf_counter()
    sampler, eval_cfg, sim_cfg = grid.build(
        coord, n_paths=chunk_n_paths, seed=seed, max_steps=args.max_steps
    )
    res = sim_core.run(sampler, eval_cfg, sim_cfg)
    elapsed = time.perf_counter() - t0

    df = _build_frame(res, coord, array_id, offset)
    paths_file = _write_paths(df, args.out_dir, array_id, args.chunk)

    summary = {**coord, "coord_id": array_id, "chunk": args.chunk, **summarize(res)}
    summary_file = os.path.join(
        args.out_dir, f"cell_{array_id:03d}_chunk{args.chunk:02d}_summary.json"
    )
    with open(summary_file, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"coord_id={array_id} coord={coord} n_paths={res.n} "
        f"pass_rate={summary['pass_rate']:.4f} "
        f"wilson=[{summary['wilson_lo']:.4f}, {summary['wilson_hi']:.4f}] "
        f"elapsed={elapsed:.2f}s -> {os.path.basename(paths_file)}"
    )


if __name__ == "__main__":
    main()
