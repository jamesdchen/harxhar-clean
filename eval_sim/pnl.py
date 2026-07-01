"""PnL samplers -- pluggable sources of the (n_paths, n_steps) per-step PnL matrix the simulator steps.

Sampler protocol: ``sample(n_paths, n_steps, rng) -> ndarray[n_paths, n_steps]`` in account-currency PnL
per period (BEFORE the sizing multiplier, which the simulator/worker applies -> common random numbers).

- EmpiricalIID   : classic sample-with-replacement from a historical trade log.
- BlockBootstrap : moving-block resample (preserves autocorrelation -- vol clustering / intraday regime;
                   the i.i.d. bootstrap destroys the loss-run structure that drives drawdowns).

The intended trade log is the cumrv x close strategy PnL (build_cumrv_pnl.py -> eval_sim/data/
cumrv_close_pnl.npy) fed through BlockBootstrap -- a real research edge (writeup FWL attribution:
cumrv_x_close mean-reverts close/AH vol below HAR), not a synthetic shape.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Sampler(Protocol):
    def sample(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray: ...


class EmpiricalIID:
    """Sample historical per-step PnL with replacement (i.i.d.). Fast, but discards serial dependence."""

    def __init__(self, pnl_log: np.ndarray):
        self.log = np.asarray(pnl_log, dtype=np.float64).ravel()
        if self.log.size == 0:
            raise ValueError("empty pnl_log")

    def sample(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        idx = rng.integers(0, self.log.size, size=(n_paths, n_steps))
        return self.log[idx]


class BlockBootstrap:
    """Moving-block bootstrap: preserves autocorrelation up to ~block length. Circular blocks so every
    start is valid. ``block`` should span the PnL's dependence horizon (e.g. an intraday session)."""

    def __init__(self, pnl_log: np.ndarray, block: int = 20):
        self.log = np.asarray(pnl_log, dtype=np.float64).ravel()
        self.block = int(max(1, block))
        if self.log.size == 0:
            raise ValueError("empty pnl_log")

    def sample(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = self.log.size
        nblk = int(np.ceil(n_steps / self.block))
        starts = rng.integers(0, n, size=(n_paths, nblk))  # random block starts
        offs = np.arange(self.block)
        idx = (starts[:, :, None] + offs[None, None, :]).reshape(
            n_paths, nblk * self.block
        ) % n
        return self.log[idx[:, :n_steps]]


def from_log(path: str, block: int = 5) -> BlockBootstrap:
    """Convenience: a BlockBootstrap over a saved daily-PnL trade log (e.g. the cumrv x close strategy
    log at eval_sim/data/cumrv_close_pnl.npy). block~5 keeps a trading-week of PnL dependence."""
    return BlockBootstrap(np.load(path), block=block)
