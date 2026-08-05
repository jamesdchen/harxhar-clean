"""Parameter grid + array-id <-> coordinate map for a chunked HPC sweep.

Owns the single source of truth for the sweep: ``GRID`` is a flat list of coordinate dicts, and a job's
SLURM/SGE array index maps 1:1 to ``GRID[array_id]``. Both the worker (which builds and runs a coord)
and the aggregator (which reconciles chunk outputs back to coords) import from here, so the two can never
disagree on what index means what. Edit the axis tuples to sweep.

PnL source is the cumrv x close strategy trade log (build_cumrv_pnl.py) block-bootstrapped -- so the grid
sweeps the real eval levers: position sizing, trailing drawdown allowance, and profit target.
"""

from __future__ import annotations

import itertools
import os

from eval_sim import pnl
from eval_sim.config import EvalConfig, SimConfig
from eval_sim.pnl import Sampler

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "cumrv_close_pnl.npy")
BLOCK = (
    5  # bootstrap block ~ a trading week of PnL dependence (preserve autocorrelation)
)

# Sweep axes -- edit here. Cartesian product below -> GRID.
_SIZE_MULT = (
    0.5,
    1.0,
    1.5,
    2.0,
)  # position-sizing multiplier (also the common-random-numbers lever)
_MAX_DRAWDOWN = (2000.0, 2500.0, 3000.0)  # trailing drawdown allowance
_PROFIT_TARGET = (2000.0, 3000.0, 4000.0)  # upper profit target

GRID: list[dict] = [
    {"size_mult": s, "max_drawdown": d, "profit_target": u}
    for s, d, u in itertools.product(_SIZE_MULT, _MAX_DRAWDOWN, _PROFIT_TARGET)
]

# Coordinate keys shared with aggregate.py / hpc_worker.py so the parquet schema can't drift.
COORD_KEYS: tuple[str, ...] = ("size_mult", "max_drawdown", "profit_target")


def n_coords() -> int:
    """Number of coordinates in the grid (== the array size to submit)."""
    return len(GRID)


def coord_for(array_id: int) -> dict:
    """Coordinate for a 0-based job array index; raises IndexError if out of range."""
    if not 0 <= array_id < len(GRID):
        raise IndexError(f"array_id {array_id} out of range [0, {len(GRID)})")
    return GRID[array_id]


def build(
    coord: dict,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
) -> tuple[Sampler, EvalConfig, SimConfig]:
    """Map a coordinate to ``(sampler, eval_cfg, sim_cfg)``.

    Sampler = block bootstrap of the cumrv x close daily-PnL log. ``EvalConfig`` overrides
    ``max_drawdown`` / ``profit_target`` when present; ``SimConfig`` takes the coord's ``size_mult``.
    ``n_paths`` / ``seed`` / ``max_steps`` override the SimConfig defaults when given.
    """
    sampler = pnl.from_log(LOG_PATH, block=BLOCK)

    eval_kw: dict = {}
    if "max_drawdown" in coord:
        eval_kw["max_drawdown"] = coord["max_drawdown"]
    if "profit_target" in coord:
        eval_kw["profit_target"] = coord["profit_target"]
    eval_cfg = EvalConfig(**eval_kw)

    sim_kw: dict = {"size_mult": coord.get("size_mult", 1.0)}
    if n_paths is not None:
        sim_kw["n_paths"] = n_paths
    if seed is not None:
        sim_kw["seed"] = seed
    if max_steps is not None:
        sim_kw["max_steps"] = max_steps
    sim_cfg = SimConfig(**sim_kw)

    return sampler, eval_cfg, sim_cfg


def main() -> None:
    """Print the grid (coord_id -> coord) for inspection."""
    print(f"{n_coords()} coordinates:")
    for i, coord in enumerate(GRID):
        print(f"{i:4d} -> {coord}")


if __name__ == "__main__":
    main()
