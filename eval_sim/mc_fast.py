"""Speedups for the first-passage MC. No GPU here (CuPy absent), but the sim is embarrassingly parallel
across paths, so process-parallelism over the CPU cores gives near-linear wall-clock scaling, and
antithetic sampling cuts the paths needed for a target CI. Both wrap the pure sim_core.simulate -- the
model is unchanged. (float32 is intentionally omitted: sim_core upcasts to float64 internally, so it
would not actually speed the step.)"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from eval_sim.config import EvalConfig
from eval_sim.sim_core import simulate


def antithetic_normal(
    n_paths: int, T: int, m: float, s: float, rng: np.random.Generator
) -> np.ndarray:
    """Antithetic Gaussian PnL: draw ceil(n/2) shocks and mirror them (Z, -Z). Halves the RNG cost and
    variance-reduces sign-symmetric functionals (the pass indicator's dependence on the shock mean)."""
    half = (n_paths + 1) // 2
    out = np.empty(
        (n_paths, T)
    )  # fill in place -- avoid the concatenate temporary (2x memory)
    z = rng.standard_normal((half, T))
    out[:half] = z
    out[half:] = -z[: n_paths - half]
    out *= s
    out += m
    return out


def _chunk(args):
    seed, n, T, m, s, cfg, size, anti = args
    rng = np.random.default_rng(seed)
    z = (
        antithetic_normal(n, T, m, s, rng)
        if anti
        else (m + s * rng.standard_normal((n, T)))
    )
    r = simulate(z * size, cfg)
    return int((r.outcome == 1).sum()), n


def parallel_passrate(
    cfg: EvalConfig,
    m: float,
    s: float,
    n_paths: int,
    T: int,
    *,
    size: float = 1.0,
    n_proc: int = 8,
    seed: int = 0,
    antithetic: bool = True,
) -> float:
    """P(pass) for a constant ``size`` multiplier, computed over ``n_proc`` worker processes (each an
    independent path chunk with its own RNG stream) with optional antithetic sampling."""
    per = n_paths // n_proc
    args = [
        (seed + 1000 * i, per, T, m, s, cfg, size, antithetic) for i in range(n_proc)
    ]
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        res = list(ex.map(_chunk, args))
    passes = sum(p for p, _ in res)
    tot = sum(n for _, n in res)
    return passes / tot
