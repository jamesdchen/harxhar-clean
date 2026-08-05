"""Implicit finite-difference solver for the prop-eval HJB -- the proper parabolic-PDE integrator.

Supersedes the explicit semi-Lagrangian DP in :mod:`eval_sim.hjb` (which is monotone -> viscosity-correct
but low order, so it needs a fine mesh). The Bellman equation

    V_t + sup_{0<=w<=w_max} [ w m V_E + 1/2 w^2 s_t^2 V_EE ] = 0                       (backward in t)

is a degenerate fully-nonlinear *parabolic* PDE. This integrates it with:

  * an implicit theta-scheme (Crank-Nicolson, theta=1/2) with **Rannacher startup** (a few fully-implicit
    steps) to damp the oscillations CN produces at the non-smooth barrier / terminal payoff;
  * **Howard policy iteration** for the sup_w nonlinearity (freeze w -> linear tridiagonal solve ->
    update w by the pointwise Hamiltonian max -> repeat);
  * a **positive-coefficient** spatial stencil (central where it preserves monotonicity, upwind
    otherwise) so the scheme converges to the viscosity solution (Barles-Souganidis);
  * the **running-max reduction**: M ratchets with no diffusion, so the 2-D problem is a stack of 1-D
    parabolic PDEs in E coupled only through the reflecting Neumann dV/dM = 0 at E = M. Slices are solved
    top-down in M (Gauss-Seidel in the ratchet direction), each an O(nE) Thomas solve.

Grid is price-aligned: E and M share one lattice of spacing h = D/q, so floor E=M-D, diagonal E=M and
target E=U all fall on nodes and the reflecting coupling is an exact node lookup. Returns the same
:class:`eval_sim.hjb.PolicySolution` as the explicit solver, so :func:`eval_sim.hjb.simulate_policy`
verifies it unchanged. v1: TRAILING_INTRADAY, min_days=0, no daily cap.
"""

from __future__ import annotations

import numpy as np

from eval_sim.config import DrawdownType, EvalConfig
from eval_sim.hjb import PolicySolution


def _thomas(
    sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Tridiagonal solve (sub-diagonal, diagonal, super-diagonal, rhs); sub[0], sup[-1] ignored. O(n)."""
    n = diag.size
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        den = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / den
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / den
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _thomas_batched(
    sub: np.ndarray, dia: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve nB tridiagonal systems at once. All args (nB, n); sub[:,0], sup[:,-1] ignored. The sweep
    loops over the n spatial nodes but vectorizes across the nB systems -- the whole point of batching."""
    n = dia.shape[1]
    cp = np.empty_like(dia)
    dp = np.empty_like(dia)
    cp[:, 0] = sup[:, 0] / dia[:, 0]
    dp[:, 0] = rhs[:, 0] / dia[:, 0]
    for i in range(1, n):
        den = dia[:, i] - sub[:, i] * cp[:, i - 1]
        cp[:, i] = sup[:, i] / den
        dp[:, i] = (rhs[:, i] - sub[:, i] * dp[:, i - 1]) / den
    x = np.empty_like(dia)
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def _hamiltonian_w(
    VE: np.ndarray, VEE: np.ndarray, m: float, s: float, w_max: float
) -> np.ndarray:
    """Pointwise maximizer of  w m VE + 1/2 w^2 s^2 VEE  over [0, w_max]. Concave (VEE<0) -> Merton/Kelly
    FOC clipped; convex -> bang-bang endpoint."""
    s2 = s * s
    concave = VEE < -1e-18
    safe_VEE = np.where(concave, VEE, -1.0)
    w_int = np.clip(-m * VE / (s2 * safe_VEE), 0.0, w_max)
    val_max = (
        m * VE * w_max + 0.5 * s2 * VEE * w_max * w_max
    )  # convex-branch endpoint value
    return np.where(concave, w_int, np.where(val_max > 0.0, w_max, 0.0))


def _pos_coeff(a_conv: np.ndarray, d_diff: np.ndarray, h: float):
    """Central-difference lo/up coefficients, upwinding the drift only where central breaks positivity."""
    lo = d_diff / (h * h) - a_conv / (2 * h)
    up = d_diff / (h * h) + a_conv / (2 * h)
    bad = (lo < 0) | (up < 0)
    if bad.any():
        loU = d_diff / (h * h) + np.maximum(-a_conv, 0.0) / h
        upU = d_diff / (h * h) + np.maximum(a_conv, 0.0) / h
        lo = np.where(bad, loU, lo)
        up = np.where(bad, upU, up)
    return lo, up, -(lo + up)


def solve_pde(
    cfg: EvalConfig,
    m: float,
    s,
    *,
    w_max: float = 8.0,
    q: int = 100,
    T: int | None = None,
    hard_cap: int = 250,
    theta: float = 1.0,
    rannacher: int = 2,
    n_policy: int = 6,
) -> PolicySolution:
    """Solve the eval HJB by implicit CN + Rannacher + Howard policy iteration on the running-max
    reduction. ``q`` = price nodes per drawdown D (mesh h = D/q). ``s`` scalar or length-T schedule."""
    if cfg.dd_type != DrawdownType.TRAILING_INTRADAY:
        raise NotImplementedError("v1 PDE solver: TRAILING_INTRADAY only")
    init, U, D = cfg.initial, cfg.upper, cfg.max_drawdown
    if T is None:
        T = cfg.horizon_steps(hard_cap)
    s_sched = np.asarray(s, dtype=float)
    if s_sched.ndim == 0:
        s_sched = np.full(T, s_sched)
    if s_sched.size != T:
        raise ValueError(f"s schedule length {s_sched.size} != horizon {T}")

    h = D / q
    nD = q
    K = int(round((U - (init - D)) / h))  # E-lattice: P[0]=init-D ... P[K]=U
    P = (init - D) + h * np.arange(K + 1)
    j0, jU = nD, K  # lattice indices of E=init and E=U
    Mslc = np.arange(
        j0, jU + 1
    )  # M-slice lattice indices (M = init..U); also E-lattice indices
    nM = Mslc.size
    dt = 1.0

    V = np.zeros((K + 1, nM))  # V[i, jj]: E=P[i], M=P[Mslc[jj]]
    policy = np.zeros((T, K + 1, nM), dtype=np.float32)

    for t in range(T - 1, -1, -1):
        st = s_sched[t]
        th = (
            1.0 if (T - 1 - t) < rannacher else theta
        )  # Rannacher: first steps fully implicit
        Vnew = V.copy()
        for jj in range(nM - 1, -1, -1):  # top-down in M
            Mj = Mslc[jj]
            i_lo, i_hi = Mj - nD, Mj  # floor node .. diagonal/target node
            ni = i_hi - i_lo + 1
            V_bot = 0.0
            V_top = (
                1.0 if jj == nM - 1 else Vnew[Mj, jj + 1]
            )  # target (M=U) or reflecting dV/dM=0
            if ni < 3:
                Vnew[i_lo, jj], Vnew[i_hi, jj] = V_bot, V_top
                continue
            idx = np.arange(i_lo, i_hi + 1)
            Vslice = V[idx, jj].copy()
            Vslice[0], Vslice[-1] = V_bot, V_top
            k = np.arange(1, ni - 1)  # interior local nodes (Dirichlet ends)
            # Seed a feasible, PROPAGATING control. w=0 is a spurious fixed point: it zeroes the
            # diffusion, the value never develops, and the Hamiltonian then keeps w=0. Start Howard
            # from w_max so the first implicit solve carries the boundary value inward, then refine.
            w = np.zeros(ni)
            w[1:-1] = w_max
            cur = Vslice.copy()
            w_used = w.copy()
            for _ in range(n_policy):
                w_used = (
                    w.copy()
                )  # the control this solve uses (kept consistent with `cur`)
                a_conv = w * m
                d_diff = 0.5 * w * w * st * st
                lo, up, dg = _pos_coeff(a_conv, d_diff, h)
                sub = -th * dt * lo[k]
                dia = 1.0 - th * dt * dg[k]
                sup = -th * dt * up[k]
                rhs = Vslice[k] + (1 - th) * dt * (
                    lo[k] * Vslice[k - 1] + dg[k] * Vslice[k] + up[k] * Vslice[k + 1]
                )
                rhs[0] += th * dt * lo[1] * V_bot
                rhs[-1] += th * dt * up[ni - 2] * V_top
                sub[0], sup[-1] = 0.0, 0.0
                inner = _thomas(sub, dia, sup, rhs)
                cur = np.concatenate(([V_bot], inner, [V_top]))
                # Howard: update the control from the freshly solved value
                VE = (cur[2:] - cur[:-2]) / (2 * h)
                VEE = (cur[2:] - 2 * cur[1:-1] + cur[:-2]) / (h * h)
                w_next = w.copy()
                w_next[1:-1] = _hamiltonian_w(VE, VEE, m, st, w_max)
                if np.max(np.abs(w_next - w)) < 1e-6:
                    break
                w = w_next
            Vnew[idx, jj] = cur
            policy[t, idx, jj] = w_used.astype(np.float32)
        V = Vnew

    # Fill boundary / infeasible policy nodes so the verifier can size at E=M (a new high -- the account
    # start is exactly here) and just above: the diagonal is a reflecting boundary with no optimized
    # control, so inherit the adjacent interior w; the upper triangle E>M inherits the diagonal.
    for jj in range(nM):
        Mj = Mslc[jj]
        if Mj - 1 >= Mj - nD + 1:
            policy[:, Mj, jj] = policy[:, Mj - 1, jj]  # diagonal <- interior neighbor
        if Mj + 1 <= K:
            policy[:, Mj + 1 :, jj] = policy[:, Mj : Mj + 1, jj]  # E>M <- diagonal

    return PolicySolution(
        E=P.copy(),
        M=P[Mslc].copy(),
        value0=V,
        policy=policy,
        V_start=float(V[j0, 0]),
        m=float(m),
        s_sched=s_sched,
        cfg=cfg,
    )


def solve_pde_fast(
    cfg: EvalConfig,
    m: float,
    s,
    *,
    w_max: float = 15.0,
    q: int = 100,
    T: int | None = None,
    hard_cap: int = 250,
    theta: float = 1.0,
    n_policy: int = 6,
) -> PolicySolution:
    """Batched pure-numpy solver: ~18x faster than :func:`solve_pde`, at the cost of a small, KNOWN bias.

    Key structural fact: every M-slice spans exactly ``q+1`` nodes (E in [M-D, M]), so the whole stack of
    nM slices is solved as ONE batched tridiagonal system per Howard iteration -- the 121-slice Python
    loop collapses into vectorized numpy over the slice axis. This forces the reflecting M-coupling to be
    **Jacobi** (top BC lagged one time step) instead of the exact Gauss-Seidel top-down sweep, because the
    exact sweep propagates the target down the whole M-chain WITHIN a step and is inherently sequential.

    Consequence (measured): ``V_start`` is biased LOW by ~0.05 vs :func:`solve_pde` (the lag delays the
    target's descent down the M-chain), but the realized policy is only ~0.01 worse in the MC. So this is
    a fast **exploration / warm-start** solver; use :func:`solve_pde` for final values. (An exact-and-fast
    path would Numba-jit the sequential sweep -- Numba is absent in this env.)"""
    if cfg.dd_type != DrawdownType.TRAILING_INTRADAY:
        raise NotImplementedError("v1 PDE solver: TRAILING_INTRADAY only")
    init, U, D = cfg.initial, cfg.upper, cfg.max_drawdown
    if T is None:
        T = cfg.horizon_steps(hard_cap)
    s_sched = np.asarray(s, dtype=float)
    if s_sched.ndim == 0:
        s_sched = np.full(T, s_sched)
    if s_sched.size != T:
        raise ValueError(f"s schedule length {s_sched.size} != horizon {T}")

    h = D / q
    nD = q
    K = int(round((U - (init - D)) / h))
    P = (init - D) + h * np.arange(K + 1)
    j0, jU = nD, K
    Mslc = np.arange(j0, jU + 1)
    nM = Mslc.size
    width = nD + 1  # nodes per slice: local 0=floor(E=M-D) .. nD=diagonal/target(E=M)
    dt = 1.0
    k = np.arange(1, width - 1)  # interior local indices

    Vsl = np.zeros(
        (nM, width)
    )  # slice-local value; global E-index of (jj,kk) = Mslc[jj]-nD+kk
    policy_loc = np.zeros((T, nM, width), dtype=np.float32)

    for t in range(T - 1, -1, -1):
        st = s_sched[t]
        th = theta
        V_top = np.empty(nM)
        V_top[:-1] = Vsl[
            1:, nD - 1
        ]  # slice above at E=M_jj -> its local node nD-1 (Jacobi: lagged)
        V_top[-1] = 1.0  # M=U slice: diagonal is E=U, the target
        V_bot = np.zeros(nM)
        Vprev = Vsl.copy()
        Vprev[:, 0], Vprev[:, -1] = V_bot, V_top
        w = np.zeros((nM, width))
        w[:, 1:-1] = (
            w_max  # seed feasible propagating control (avoid the w=0 fixed point)
        )
        cur = Vprev.copy()
        w_used = w.copy()
        for _ in range(n_policy):
            w_used = w.copy()
            a_conv = w * m
            d_diff = 0.5 * w * w * st * st
            lo, up, dg = _pos_coeff(a_conv, d_diff, h)  # (nM, width)
            sub = -th * dt * lo[:, k]
            dia = 1.0 - th * dt * dg[:, k]
            sup = -th * dt * up[:, k]
            rhs = Vprev[:, k] + (1 - th) * dt * (
                lo[:, k] * Vprev[:, k - 1]
                + dg[:, k] * Vprev[:, k]
                + up[:, k] * Vprev[:, k + 1]
            )
            rhs[:, 0] += th * dt * lo[:, 1] * V_bot
            rhs[:, -1] += th * dt * up[:, width - 2] * V_top
            sub[:, 0], sup[:, -1] = 0.0, 0.0
            inner = _thomas_batched(sub, dia, sup, rhs)
            cur = np.concatenate([V_bot[:, None], inner, V_top[:, None]], axis=1)
            VE = (cur[:, 2:] - cur[:, :-2]) / (2 * h)
            VEE = (cur[:, 2:] - 2 * cur[:, 1:-1] + cur[:, :-2]) / (h * h)
            w_next = w.copy()
            w_next[:, 1:-1] = _hamiltonian_w(VE, VEE, m, st, w_max)
            if np.max(np.abs(w_next - w)) < 1e-6:
                break
            w = w_next
        Vsl = cur
        policy_loc[t] = w_used

    # reconstruct global (E, M) grids and apply the diagonal / upper-triangle policy fill
    value0 = np.zeros((K + 1, nM))
    policy = np.zeros((T, K + 1, nM), dtype=np.float32)
    for jj in range(nM):
        lo_i = Mslc[jj] - nD
        value0[lo_i : lo_i + width, jj] = Vsl[jj]
        policy[:, lo_i : lo_i + width, jj] = policy_loc[:, jj, :]
        Mj = Mslc[jj]
        if Mj - 1 >= lo_i + 1:
            policy[:, Mj, jj] = policy[:, Mj - 1, jj]  # diagonal <- interior neighbor
        if Mj + 1 <= K:
            policy[:, Mj + 1 :, jj] = policy[:, Mj : Mj + 1, jj]  # E>M <- diagonal

    return PolicySolution(
        E=P.copy(),
        M=P[Mslc].copy(),
        value0=value0,
        policy=policy,
        V_start=float(Vsl[0, nD]),
        m=float(m),
        s_sched=s_sched,
        cfg=cfg,
    )
