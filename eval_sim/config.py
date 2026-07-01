"""Frozen configuration for the prop-eval first-passage simulator.

Kept dependency-free (dataclasses + enum only) so the package is trivially movable. The boundary rule
is fully parameterized -- the classic ``L = M - D`` trailing drawdown is one point in this space; real
evaluations differ by trailing basis, a trailing-lock, daily loss limits, and min/max trading days,
each of which materially moves the pass probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DrawdownType(str, Enum):
    TRAILING_INTRADAY = (
        "trailing_intraday"  # floor trails the running intraday equity high-water mark
    )
    TRAILING_EOD = "trailing_eod"  # floor trails the end-of-day balance high-water mark
    STATIC = "static"  # fixed floor at initial - max_drawdown


@dataclass(frozen=True)
class EvalConfig:
    """Rules of one evaluation account. All money in account currency units.

    upper barrier   U = initial + profit_target        (PASS on E >= U, once >= min_days)
    lower barrier   L_t depends on ``dd_type``:
        STATIC             L = initial - max_drawdown
        TRAILING_INTRADAY  L = M_t - max_drawdown            (M_t = running max equity)
        TRAILING_EOD       L = M_eod - max_drawdown          (M_eod = running max end-of-day balance)
    ``lock_level`` (Apex-style): once the trailing floor reaches ``lock_level`` it stops rising, i.e.
        L = min(trailing_floor, lock_level).  None -> pure trailing (never locks).
    """

    initial: float = 50_000.0
    profit_target: float = 3_000.0
    max_drawdown: float = 2_500.0
    dd_type: DrawdownType = DrawdownType.TRAILING_INTRADAY
    lock_level: float | None = None  # e.g. `initial` for a breakeven lock
    daily_loss_limit: float | None = (
        None  # FAIL if intraday (within-day) loss <= -daily_loss_limit
    )
    min_days: int = 0  # a PASS only counts after this many completed trading days
    max_days: int | None = (
        None  # eval time limit in days; None -> run to SimConfig.max_steps
    )
    steps_per_day: int = (
        1  # PnL periods per trading day (1 = one trade/day; >1 = intraday bars)
    )

    @property
    def upper(self) -> float:
        return self.initial + self.profit_target

    @property
    def static_floor(self) -> float:
        return self.initial - self.max_drawdown

    def horizon_steps(self, hard_cap: int) -> int:
        """Number of steps to simulate: min(hard_cap, max_days*steps_per_day) if a time limit is set."""
        if self.max_days is None:
            return hard_cap
        return min(hard_cap, self.max_days * self.steps_per_day)


@dataclass(frozen=True)
class SimConfig:
    """Monte Carlo controls (decoupled from the eval rules and the PnL source)."""

    n_paths: int = 100_000
    max_steps: int = 250  # hard cap on PnL periods per path (also the array width)
    seed: int = 0
    size_mult: float = (
        1.0  # position-sizing multiplier applied to PnL (common-random-numbers lever)
    )
