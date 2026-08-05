"""Vectorized discrete-time first-passage simulator for a prop-eval with a trailing drawdown boundary.

The time loop is inherently sequential (the high-water mark is path-dependent), but ALL paths step
together, so each iteration is O(n_paths) numpy work -> O(T * P). Pure: ``simulate(pnl, cfg)`` takes an
already-sizing-scaled (P, T) PnL matrix and the eval rules and returns per-path outcomes. ``run`` wires
a Sampler + SimConfig on top (draws the PnL, applies the sizing multiplier). No I/O, no globals.

Outcomes: PASS (E >= U after >= min_days), FAIL (E <= trailing floor, or a daily-loss breach), TIMEOUT
(neither within the horizon -- the censored case a first-passage-with-horizon must track).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval_sim.config import DrawdownType, EvalConfig, SimConfig
from eval_sim.pnl import Sampler

PASS = 1
FAIL = 2
TIMEOUT = 3


@dataclass(frozen=True)
class PathResults:
    outcome: np.ndarray  # int8[P]: PASS / FAIL / TIMEOUT
    stop_step: np.ndarray  # int32[P]: 0-based period index at termination
    terminal_equity: np.ndarray  # float64[P]
    peak_equity: np.ndarray  # float64[P]: max equity reached
    max_dd: np.ndarray  # float64[P]: max (peak - equity) reached
    days: np.ndarray  # int32[P]: completed trading days at stop

    @property
    def n(self) -> int:
        return int(self.outcome.shape[0])


def simulate(pnl: np.ndarray, cfg: EvalConfig) -> PathResults:
    """First-passage over (P, T) per-step PnL (already sizing-scaled). Pure, vectorized across paths."""
    pnl = np.ascontiguousarray(pnl, dtype=np.float64)
    P, T = pnl.shape
    U = cfg.upper
    D = cfg.max_drawdown
    spd = max(1, cfg.steps_per_day)
    trailing = cfg.dd_type != DrawdownType.STATIC
    eod_basis = cfg.dd_type == DrawdownType.TRAILING_EOD

    E = np.full(P, cfg.initial, dtype=np.float64)
    M = np.full(P, cfg.initial, dtype=np.float64)  # running intraday equity HWM
    M_eod = np.full(P, cfg.initial, dtype=np.float64)  # running end-of-day balance HWM
    day_pnl = np.zeros(P, dtype=np.float64)
    days = np.zeros(P, dtype=np.int32)
    max_dd = np.zeros(P, dtype=np.float64)
    alive = np.ones(P, dtype=bool)
    outcome = np.zeros(P, dtype=np.int8)
    stop_step = np.full(P, T - 1, dtype=np.int32)

    for t in range(T):
        a = alive
        E[a] += pnl[a, t]
        day_pnl[a] += pnl[a, t]
        np.maximum(M, E, out=M, where=a)
        # drawdown so far (relative to the intraday HWM -- reported regardless of dd_type)
        dd_now = M - E
        np.maximum(max_dd, dd_now, out=max_dd, where=a)

        end_of_day = (t + 1) % spd == 0
        if end_of_day:
            np.maximum(M_eod, E, out=M_eod, where=a)
            days[a] += 1

        # trailing floor
        floor: np.ndarray
        if not trailing:
            floor = np.full(P, cfg.static_floor, dtype=np.float64)
        else:
            floor = (M_eod if eod_basis else M) - D
            if cfg.lock_level is not None:
                np.minimum(floor, cfg.lock_level, out=floor)

        fail = alive & (E <= floor)
        if cfg.daily_loss_limit is not None:
            fail |= alive & (day_pnl <= -cfg.daily_loss_limit)
        passed = alive & (E >= U) & (days >= cfg.min_days)
        # E >= U dominates (equity is high) -> a step can't be both; FAIL only when not passing
        fail &= ~passed

        if passed.any():
            outcome[passed] = PASS
            stop_step[passed] = t
            alive[passed] = False
        if fail.any():
            outcome[fail] = FAIL
            stop_step[fail] = t
            alive[fail] = False

        if end_of_day:
            day_pnl[:] = 0.0  # reset the daily accumulator (dead paths unused)
        if not alive.any():
            break

    outcome[alive] = TIMEOUT  # survived the horizon
    return PathResults(
        outcome=outcome,
        stop_step=stop_step,
        terminal_equity=E,
        peak_equity=M,
        max_dd=max_dd,
        days=days,
    )


def run(sampler: Sampler, eval_cfg: EvalConfig, sim_cfg: SimConfig) -> PathResults:
    """Draw PnL from a Sampler, apply the sizing multiplier, and run the first-passage simulation.

    RNG is seeded from ``sim_cfg.seed`` (deterministic). The sizing multiplier scales the *same* drawn
    PnL, so sweeping size_mult over a grid at a fixed seed gives common random numbers (low-variance
    comparison across sizing)."""
    rng = np.random.default_rng(sim_cfg.seed)
    T = eval_cfg.horizon_steps(sim_cfg.max_steps)
    pnl = sampler.sample(sim_cfg.n_paths, T, rng) * sim_cfg.size_mult
    return simulate(pnl, eval_cfg)
