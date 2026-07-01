"""Optimal control of the prop-eval barrier via an explicit backward-induction DP (REFERENCE solver).

SUPERSEDED FOR ACCURACY by :func:`eval_sim.hjb_pde.solve_pde` (implicit backward-Euler + Howard policy
iteration). This explicit semi-Lagrangian DP is monotone -> converges to the right viscosity solution,
but it is LOW ORDER: at practical grids it carries a ~0.085 discretization bias (the per-step Gaussian
convolution under-resolves the barrier tail via Gauss-Hermite quadrature + bilinear interpolation). It is
kept because it is the cleanest statement of the derivation (the sizing law + the bang-bang/interior
dichotomy fall straight out of its one-step Bellman max) and because it matches the DISCRETE-monitoring
sim_core exactly. Use it for intuition and as a cross-check; use ``hjb_pde.solve_pde`` for numbers. The
gap is discretization, not a bug -- fully closing it (exact analytic absorption of the tail) was not done
because the implicit PDE already computes the accurate value.

Replaces guess-and-check ``size_mult`` sweeping. A constant multiplier cannot represent the optimal
control, which is a *dynamic, state-dependent* exposure ``w*(t, E, M)``. This module solves for it and
returns the value function ``V(t, E, M) = sup_w P(pass)`` directly.

Model (matches :mod:`eval_sim.sim_core` exactly -- arithmetic dollar PnL, ``E += w * unit_pnl``, trailing
floor ``L = M - D``). State ``(t, E, M)``: equity, running high-water mark, time. One step:

    E_{t+1} = E_t + w_t * xi_t ,   M_{t+1} = max(M_t, E_{t+1})

with ``xi_t`` the *unit-size* per-step PnL (mean ``m``, std ``s_t`` = the HAR sigma-hat forecast).

The Bellman recursion is the discrete image of the HJB

    V_t + sup_{0<=w<=w_max} [ w m V_E + 1/2 w^2 s_t^2 V_EE ] = 0 ,

whose pointwise maximizer is the sizing law
    interior (V_EE<0):  w* = clip( -(m / s_t^2) * V_E / V_EE , 0, w_max )      ("timid")
    convex   (V_EE>=0): w* in {0, w_max}                                       ("bold", bang-bang)
Reading off the vol exponent: a fixed *return* edge (m const) => w* ~ 1/s^2; a *Sharpe* edge (m = lam*s)
=> w* ~ 1/s. Both are scaled by the value-function risk tolerance (-V_E/V_EE), which -> 0 at the floor.

The DP evaluates the true one-step expectation over ``w`` (no linear-quadratic approximation), so it also
captures the bang-bang region the FOC linearization misses. :func:`simulate_policy` then runs the
:mod:`eval_sim.sim_core` first-passage loop with ``size_mult`` replaced by ``w*(t, E, M)`` -- realized
P(pass) under common random numbers must match ``V_start`` and dominate the best constant multiplier.

min_days / daily_loss_limit are not yet modeled (v1 assumes min_days=0, no intraday daily cap); the
trailing basis, static floor, lock_level, and finite deadline (max_days) are all honored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval_sim.config import DrawdownType, EvalConfig


@dataclass(frozen=True)
class PolicySolution:
    """Optimal-control solution on the (E, M) grid over the eval horizon."""

    E: np.ndarray  # (nE,) equity grid nodes
    M: np.ndarray  # (nM,) high-water-mark grid nodes
    value0: np.ndarray  # (nE, nM) P(pass) at t=0 under the optimal policy
    policy: np.ndarray  # (T, nE, nM) float32 optimal exposure w*(t, E, M)
    V_start: float  # value0 interpolated at the account start (E=M=initial)
    m: float  # per-step unit-PnL mean used
    s_sched: np.ndarray  # (T,) per-step unit-PnL std schedule used
    cfg: EvalConfig

    @property
    def T(self) -> int:
        return int(self.policy.shape[0])


def _interp2(V: np.ndarray, E: np.ndarray, M: np.ndarray, Eq, Mq):
    """Bilinear interpolation of V on the uniform (E, M) grid at query points (Eq, Mq), edge-clamped.

    Eq, Mq broadcast to a common shape; the result has that shape. Works for grid-shaped (nE, nM) queries
    (the backward sweep) and 1-D per-path queries (the verifier)."""
    nE, nM = E.size, M.size
    dE, dM = E[1] - E[0], M[1] - M[0]
    fe = np.clip((Eq - E[0]) / dE, 0.0, nE - 1)
    fm = np.clip((Mq - M[0]) / dM, 0.0, nM - 1)
    i0 = np.minimum(np.floor(fe).astype(np.intp), nE - 2)
    j0 = np.minimum(np.floor(fm).astype(np.intp), nM - 2)
    i1, j1 = i0 + 1, j0 + 1
    te, tm = fe - i0, fm - j0
    return (
        V[i0, j0] * (1 - te) * (1 - tm)
        + V[i1, j0] * te * (1 - tm)
        + V[i0, j1] * (1 - te) * tm
        + V[i1, j1] * te * tm
    )


def _reflect_upper(V: np.ndarray, E: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Fill the infeasible upper triangle {E > M} by the running-max reflection: a state with E>M is
    really '(E, M=E) -- just set a new high', so its value equals the diagonal value V(E, M=E). This
    realizes the Neumann boundary V_M = 0 at E=M for the interpolation near the diagonal."""
    Vr = V.copy()
    upper = E[:, None] > M[None, :] + 1e-9
    if not upper.any():
        return Vr
    diag = np.array(
        [
            np.interp(min(e, M[-1]), M, V[i, :]) if e > M[0] else np.nan
            for i, e in enumerate(E)
        ]
    )
    Dg = np.broadcast_to(diag[:, None], V.shape)
    return np.where(upper & ~np.isnan(Dg), Dg, Vr)


def _gauss_hermite(nZ: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for E[f(Z)], Z~N(0,1): E[f] ~ sum(w_k f(x_k)) with sum(w)=1."""
    x, w = np.polynomial.hermite_e.hermegauss(nZ)
    return x, w / np.sqrt(2.0 * np.pi)


def solve_gaussian(
    cfg: EvalConfig,
    m: float,
    s,
    *,
    w_max: float = 5.0,
    nW: int = 41,
    nE: int = 161,
    nM: int = 161,
    nZ: int = 9,
    T: int | None = None,
    hard_cap: int = 250,
) -> PolicySolution:
    """Solve for the optimal exposure policy under a Gaussian one-step unit PnL xi ~ N(m, s_t^2).

    ``s`` is a scalar (constant vol) or an array of length ``T`` (the HAR sigma-hat forecast path). ``w``
    ranges over ``[0, w_max]`` on ``nW`` nodes; the (E, M) state grid is nE x nM. Returns a
    :class:`PolicySolution`.
    """
    init, U, D = cfg.initial, cfg.upper, cfg.max_drawdown
    if T is None:
        T = cfg.horizon_steps(hard_cap)
    s_sched = np.asarray(s, dtype=float)
    if s_sched.ndim == 0:
        s_sched = np.full(T, s_sched)
    if s_sched.size != T:
        raise ValueError(f"s schedule length {s_sched.size} != horizon {T}")

    E = np.linspace(init - D, U, nE)
    M = np.linspace(init, U, nM)
    w_grid = np.linspace(0.0, w_max, nW)
    zx, zw = _gauss_hermite(nZ)

    EE, MM = np.meshgrid(E, M, indexing="ij")  # (nE, nM)
    if cfg.dd_type == DrawdownType.STATIC:
        floor = np.full_like(MM, cfg.static_floor)
    else:
        floor = MM - D
        if cfg.lock_level is not None:
            floor = np.minimum(floor, cfg.lock_level)
    pass_mask = EE >= U - 1e-9
    fail_mask = EE <= floor + 1e-9
    absorbing = pass_mask | fail_mask

    # Terminal (t = T): timeout = not yet passed -> 0; passed states carry 1.
    V = np.where(pass_mask, 1.0, 0.0)
    policy = np.zeros((T, nE, nM), dtype=np.float32)

    for t in range(T - 1, -1, -1):
        Vn = _reflect_upper(V, E, M)
        st = s_sched[t]
        best = np.full((nE, nM), -np.inf)
        best_w = np.zeros((nE, nM), dtype=np.float32)
        for w in w_grid:
            Q = np.zeros((nE, nM))
            for zk, pk in zip(zx, zw):
                Ep = EE + w * (m + st * zk)
                Mp = np.maximum(MM, Ep)
                Q += pk * _interp2(Vn, E, M, Ep, Mp)
            take = Q > best
            best[take] = Q[take]
            best_w[take] = np.float32(w)
        V = np.where(pass_mask, 1.0, np.where(fail_mask, 0.0, best))
        best_w[absorbing] = 0.0
        policy[t] = best_w

    V_start = float(_interp2(V, E, M, np.array(init), np.array(init)))
    return PolicySolution(
        E=E,
        M=M,
        value0=V,
        policy=policy,
        V_start=V_start,
        m=float(m),
        s_sched=s_sched,
        cfg=cfg,
    )


def simulate_policy(unit_pnl: np.ndarray, cfg: EvalConfig, sol: PolicySolution):
    """Run the first-passage sim with the optimal dynamic policy: at each step size the *unit* PnL by
    ``w*(t, E, M)`` interpolated from ``sol``. Mirrors :func:`eval_sim.sim_core.simulate` semantics and
    returns its :class:`~eval_sim.sim_core.PathResults`. Feed the SAME unit PnL used to fit / to a
    constant-size baseline for a common-random-numbers comparison."""
    from eval_sim.sim_core import (
        PathResults,
    )  # local import: keep hjb importable without the loop

    PASS, FAIL, TIMEOUT = 1, 2, 3
    unit_pnl = np.ascontiguousarray(unit_pnl, dtype=np.float64)
    P, T = unit_pnl.shape
    U, D = cfg.upper, cfg.max_drawdown
    spd = max(1, cfg.steps_per_day)
    trailing = cfg.dd_type != DrawdownType.STATIC
    eod_basis = cfg.dd_type == DrawdownType.TRAILING_EOD

    E = np.full(P, cfg.initial)
    M = np.full(P, cfg.initial)
    M_eod = np.full(P, cfg.initial)
    days = np.zeros(P, dtype=np.int32)
    max_dd = np.zeros(P)
    alive = np.ones(P, dtype=bool)
    outcome = np.zeros(P, dtype=np.int8)
    stop_step = np.full(P, T - 1, dtype=np.int32)
    Tp = sol.policy.shape[0]

    for t in range(T):
        a = alive
        w = _interp2(
            sol.policy[min(t, Tp - 1)], sol.E, sol.M, E, M
        )  # (P,) per-path exposure
        E[a] += w[a] * unit_pnl[a, t]
        np.maximum(M, E, out=M, where=a)
        np.maximum(max_dd, M - E, out=max_dd, where=a)

        if (t + 1) % spd == 0:
            np.maximum(M_eod, E, out=M_eod, where=a)
            days[a] += 1

        floor: np.ndarray
        if not trailing:
            floor = np.full(P, cfg.static_floor)
        else:
            floor = (M_eod if eod_basis else M) - D
            if cfg.lock_level is not None:
                np.minimum(floor, cfg.lock_level, out=floor)

        passed = alive & (E >= U) & (days >= cfg.min_days)
        fail = alive & (E <= floor) & ~passed
        if passed.any():
            outcome[passed] = PASS
            stop_step[passed] = t
            alive[passed] = False
        if fail.any():
            outcome[fail] = FAIL
            stop_step[fail] = t
            alive[fail] = False
        if not alive.any():
            break

    outcome[alive] = TIMEOUT
    return PathResults(
        outcome=outcome,
        stop_step=stop_step,
        terminal_equity=E,
        peak_equity=M,
        max_dd=max_dd,
        days=days,
    )


def pass_rate(res) -> float:
    return float((res.outcome == 1).mean())
