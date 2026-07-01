"""Summary statistics with Monte Carlo uncertainty for a ``PathResults``.

A prop-eval pass-rate is a binomial proportion estimated from ``n_paths`` draws, so it carries
sampling error: report the Wilson score interval (the correct small-sample CI for a proportion), not a
bare point estimate. ``paths_for_precision`` inverts the Wald half-width to plan ``n_paths`` up front.
Pure numpy, no I/O.
"""

from __future__ import annotations

import numpy as np

from eval_sim.sim_core import FAIL, PASS, TIMEOUT, PathResults


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (nan-pair if ``n == 0``)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (float(center - half), float(center + half))


def standard_error(p: float, n: int) -> float:
    """Standard error of a proportion ``p`` from ``n`` samples: ``sqrt(p(1-p)/n)``."""
    if n <= 0:
        return float("nan")
    return float(np.sqrt(p * (1.0 - p) / n))


def paths_for_precision(p: float, half_width: float, z: float = 1.96) -> int:
    """Min ``n`` so the Wald CI half-width ``z*sqrt(p(1-p)/n) <= half_width`` at proportion ``p``."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    n = (z * z * p * (1.0 - p)) / (half_width * half_width)
    return int(np.ceil(n))


def _steps_stats(res: PathResults, code: int) -> tuple[float, float]:
    """(mean, median) of 1-based step count over paths with the given outcome (nan if none)."""
    mask = res.outcome == code
    if not mask.any():
        return (float("nan"), float("nan"))
    steps = res.stop_step[mask].astype(np.float64) + 1.0
    return (float(steps.mean()), float(np.median(steps)))


def summarize(res: PathResults) -> dict:
    """Point estimates plus a Wilson 95% CI on the pass-rate for a batch of simulated paths."""
    n = res.n
    if n == 0:
        raise ValueError("PathResults has no paths")
    n_pass = int(np.count_nonzero(res.outcome == PASS))
    n_fail = int(np.count_nonzero(res.outcome == FAIL))
    n_timeout = int(np.count_nonzero(res.outcome == TIMEOUT))

    pass_rate = n_pass / n
    lo, hi = wilson_ci(n_pass, n)
    mean_steps_pass, median_steps_pass = _steps_stats(res, PASS)
    mean_steps_fail, median_steps_fail = _steps_stats(res, FAIL)

    return {
        "n": n,
        "pass_rate": pass_rate,
        "fail_rate": n_fail / n,
        "timeout_rate": n_timeout / n,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_steps_pass": mean_steps_pass,
        "median_steps_pass": median_steps_pass,
        "mean_steps_fail": mean_steps_fail,
        "median_steps_fail": median_steps_fail,
        "mean_max_dd": float(res.max_dd.mean()),
        "p95_max_dd": float(np.percentile(res.max_dd, 95)),
        "mean_terminal_equity": float(res.terminal_equity.mean()),
    }
