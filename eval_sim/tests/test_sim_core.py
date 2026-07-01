"""Correctness checks for the prop-eval first-passage simulator (``eval_sim.sim_core``).

Runs standalone (``python -m eval_sim.tests.test_sim_core`` -> prints PASS/FAIL per check, exits
non-zero on any failure) AND under pytest (each check has a ``test_*`` wrapper). Plain-python asserts,
numpy only -- no test framework required to run it.

Checks
------
(a) Gambler's ruin (the correctness anchor). A symmetric +/-1 walk with STATIC symmetric barriers has a
    closed-form pass probability P(PASS) = a / (a + b): the empirical pass-rate must match it to MC noise.
(b) Determinism. Same ``SimConfig.seed`` -> byte-identical outcome/stop_step arrays.
(c) Trailing vs static ordering. The trailing floor is weakly tighter than the static one (floor rises
    with the high-water mark), so under common random numbers the trailing pass-rate <= the static one.
(d) Block bootstrap preserves autocorrelation. On an AR(1) log the moving-block resample keeps a strongly
    positive lag-1 autocorrelation while the i.i.d. resample destroys it.

Sizing note: the closed-form check ideally wants many paths and a horizon >> a*b. ``run`` materializes the
full ``(n_paths, max_steps)`` PnL matrix up front (~16 bytes/element at peak), so the spec's literal
(200_000, 8000) would need ~25 GB and swap on a 16 GB machine. We use (100_000, 2000), which keeps the
same properties -- max_steps >> a*b (worst case a*b = 200) so TIMEOUT is negligible (rate < 1e-3), and the
3-sigma tolerance is ~0.0045 (~ the spec's ~0.004). Seeds are fixed, so the printed numbers are exact.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from eval_sim.config import DrawdownType, EvalConfig, SimConfig
from eval_sim.pnl import BlockBootstrap, EmpiricalIID
from eval_sim.sim_core import FAIL, PASS, TIMEOUT, PathResults, run

# --- MC controls for the gambler's-ruin anchor (see "Sizing note" in the module docstring) ---
_N_PATHS = 100_000
_MAX_STEPS = 2000  # >> a*b (worst case 200) -> censored TIMEOUT branch is negligible


def _decided_pass_rate(res: PathResults) -> tuple[float, int, int, int]:
    """Empirical pass-rate over *decided* paths: PASS / (PASS + FAIL). Returns (rate, n_pass, n_fail, n_to)."""
    n_pass = int(np.count_nonzero(res.outcome == PASS))
    n_fail = int(np.count_nonzero(res.outcome == FAIL))
    n_to = int(np.count_nonzero(res.outcome == TIMEOUT))
    decided = n_pass + n_fail
    rate = n_pass / decided if decided else float("nan")
    return rate, n_pass, n_fail, n_to


# ----------------------------------------------------------------------------------------------
# (a) Gambler's ruin: STATIC symmetric barriers, +/-1 i.i.d. PnL, P(PASS) = a / (a + b).
# ----------------------------------------------------------------------------------------------
def _gamblers_ruin_case(a: float, b: float, *, seed: int = 1) -> None:
    """Assert the empirical pass-rate matches the closed form a/(a+b) within 3 standard errors."""
    sampler = EmpiricalIID(np.array([+1.0, -1.0]))  # each step +/-1 w.p. 0.5
    eval_cfg = EvalConfig(
        initial=100.0,
        profit_target=float(b),  # upper barrier U = 100 + b  (b below start)
        max_drawdown=float(a),  # static floor L = 100 - a    (a above start)
        dd_type=DrawdownType.STATIC,
        lock_level=None,
        daily_loss_limit=None,
        min_days=0,
        steps_per_day=1,
        max_days=None,
    )
    sim_cfg = SimConfig(n_paths=_N_PATHS, max_steps=_MAX_STEPS, seed=seed)
    res = run(sampler, eval_cfg, sim_cfg)

    emp, n_pass, n_fail, n_to = _decided_pass_rate(res)
    analytic = a / (a + b)
    decided = n_pass + n_fail
    se = math.sqrt(analytic * (1.0 - analytic) / decided)
    tol = 3.0 * se
    to_rate = n_to / res.n

    print(
        f"    (a={a:g}, b={b:g}): empirical={emp:.5f}  analytic={analytic:.5f}  "
        f"|diff|={abs(emp - analytic):.5f}  3*se={tol:.5f}  timeout_rate={to_rate:.2e}"
    )
    assert to_rate < 1e-3, f"timeout_rate {to_rate:.2e} not < 1e-3 (raise max_steps)"
    assert abs(emp - analytic) <= tol, (
        f"pass-rate {emp:.5f} off closed form {analytic:.5f} by more than 3 se ({tol:.5f})"
    )


def check_gamblers_ruin() -> None:
    print("  gambler's-ruin closed form  P(PASS) = a/(a+b):")
    _gamblers_ruin_case(8.0, 8.0)  # -> 0.5
    _gamblers_ruin_case(10.0, 20.0)  # -> 1/3


def _demo_sampler() -> BlockBootstrap:
    """Self-contained fat-tailed daily-PnL log (positive drift + occasional large losses) for the
    determinism / trailing-vs-static checks -- no dependency on the cumrv data file."""
    rng = np.random.default_rng(123)
    log = rng.normal(150.0, 400.0, 4000)
    hit = rng.random(4000) < 0.04
    log[hit] -= rng.exponential(2000.0, size=int(hit.sum()))
    return BlockBootstrap(log, block=5)


# ----------------------------------------------------------------------------------------------
# (b) Determinism: same seed -> identical outcome and stop_step arrays.
# ----------------------------------------------------------------------------------------------
def check_determinism() -> None:
    eval_cfg = EvalConfig(initial=50_000.0, profit_target=3_000.0, max_drawdown=2_500.0)
    sim_cfg = SimConfig(n_paths=20_000, max_steps=250, seed=42)
    r1 = run(_demo_sampler(), eval_cfg, sim_cfg)
    r2 = run(_demo_sampler(), eval_cfg, sim_cfg)
    same_outcome = np.array_equal(r1.outcome, r2.outcome)
    same_stop = np.array_equal(r1.stop_step, r2.stop_step)
    print(f"  determinism: outcome match={same_outcome}  stop_step match={same_stop}")
    assert same_outcome, "outcome arrays differ across identically-seeded runs"
    assert same_stop, "stop_step arrays differ across identically-seeded runs"


# ----------------------------------------------------------------------------------------------
# (c) Trailing vs static ordering: trailing floor weakly tighter -> pass_trailing <= pass_static.
# ----------------------------------------------------------------------------------------------
def check_trailing_le_static() -> None:
    sampler = (
        _demo_sampler()
    )  # same sampler + seed -> common random numbers across dd_type

    def rate(dd: DrawdownType) -> float:
        eval_cfg = EvalConfig(
            initial=50_000.0,
            profit_target=3_000.0,
            max_drawdown=2_500.0,
            dd_type=dd,
            lock_level=None,
            daily_loss_limit=None,
            min_days=0,
            steps_per_day=1,
            max_days=None,
        )
        sim_cfg = SimConfig(n_paths=50_000, max_steps=250, seed=7)
        return _decided_pass_rate(run(sampler, eval_cfg, sim_cfg))[0]

    pass_trailing = rate(DrawdownType.TRAILING_INTRADAY)
    pass_static = rate(DrawdownType.STATIC)
    tol = 0.01  # structural inequality (CRN); tol only absorbs MC noise
    print(
        f"  trailing vs static: pass_trailing={pass_trailing:.5f}  "
        f"pass_static={pass_static:.5f}  (trailing floor weakly tighter)"
    )
    assert pass_trailing <= pass_static + tol, (
        f"trailing pass-rate {pass_trailing:.5f} exceeds static {pass_static:.5f} + tol"
    )


# ----------------------------------------------------------------------------------------------
# (d) Block bootstrap preserves autocorrelation; i.i.d. bootstrap destroys it.
# ----------------------------------------------------------------------------------------------
def _ar1(n: int, rho: float, seed: int) -> np.ndarray:
    """A length-n AR(1) series x_t = rho*x_{t-1} + eps_t (unit-variance innovations)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=n)
    x = np.empty(n, dtype=np.float64)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def _acf1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D series."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else 0.0


def check_block_bootstrap_autocorr() -> None:
    log = _ar1(5_000, rho=0.8, seed=123)
    n_steps = 20_000
    block_sample = (
        BlockBootstrap(log, block=50)
        .sample(1, n_steps, np.random.default_rng(9))
        .ravel()
    )
    iid_sample = EmpiricalIID(log).sample(1, n_steps, np.random.default_rng(9)).ravel()
    acf_block = _acf1(block_sample)
    acf_iid = _acf1(iid_sample)
    print(
        f"  block bootstrap: acf1(AR1 log)={_acf1(log):.3f}  "
        f"acf1(block)={acf_block:.3f} (>0.3)  acf1(iid)={acf_iid:.3f} (|.|<0.1)"
    )
    assert acf_block > 0.3, (
        f"block-bootstrap lag-1 acf {acf_block:.3f} not clearly positive (>0.3)"
    )
    assert abs(acf_iid) < 0.1, f"i.i.d. lag-1 acf {acf_iid:.3f} not ~0 (|.|<0.1)"


# ----------------------------------------------------------------------------------------------
# pytest entry points
# ----------------------------------------------------------------------------------------------
def test_gamblers_ruin() -> None:
    check_gamblers_ruin()


def test_determinism() -> None:
    check_determinism()


def test_trailing_le_static() -> None:
    check_trailing_le_static()


def test_block_bootstrap_autocorr() -> None:
    check_block_bootstrap_autocorr()


# ----------------------------------------------------------------------------------------------
# standalone runner
# ----------------------------------------------------------------------------------------------
def main() -> int:
    checks = [
        ("gamblers_ruin_closed_form", check_gamblers_ruin),
        ("determinism", check_determinism),
        ("trailing_le_static", check_trailing_le_static),
        ("block_bootstrap_autocorr", check_block_bootstrap_autocorr),
    ]
    failures = 0
    print("=" * 78)
    print("eval_sim.sim_core correctness checks")
    print("=" * 78)
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}\n")
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}\n")
    print("=" * 78)
    print(f"SUMMARY: {len(checks) - failures}/{len(checks)} checks PASSED")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
