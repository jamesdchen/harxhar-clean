"""Simulation-based approximate dynamic programming for the prop-eval control -- the DEPLOYABLE solver.

Why this exists, vs the grid PDE in :mod:`eval_sim.hjb_pde`: the grid needs a trusted, low-dimensional,
*Markov* model. The real edge (cumrv x close) has autocorrelation (vol clustering), fat tails and
non-stationarity, so (a) we don't trust the one-step Gaussian model, and (b) conditioning on the real
state (vol, recent path, the HAR lags) blows past the grid's curse of dimensionality. This module is the
escape -- MC error is dimension-independent:

  * GEOMETRIC wealth on REAL block-bootstrap returns (serial dependence preserved -- the i.i.d. bootstrap
    would destroy the loss-runs that drive drawdowns);
  * a HOMOGENEOUS state (drawdown / cushion in DD-units, log-wealth, time-to-go) -- the E/M scale
    reduction, so the policy generalizes across account sizes;
  * a risk-constrained-Kelly (Busseti-Ryu-Boyd) constant-fraction ANCHOR: max growth s.t. P(DD>D) <= beta;
  * the LITERATURE-optimal :class:`StructuredPolicy` -- the actual solver. Maximizing goal-reaching
    probability before drawdown under a leverage cap has a KNOWN closed form (Yuan & Li 2021; Browne 1999;
    Pestien-Sudderth): Merton fraction far from the barrier, BANG-BANG at the leverage cap (bold) near it.
    It beats the anchor decisively (+0.043, paired DM p~1e-253). Press when behind.
  * :func:`fit_adp` -- LASSO fitted-Q iteration (regression MC for control) as a data-driven refinement.
    NOTE it does NOT beat the closed form here: the optimum is a sharp bang-bang the smooth Qhat can't
    represent, so the FQI froze near the barrier (inverted from optimal). Kept as the vehicle for the
    non-Markov / HAR-state extension the closed-form GBM model can't see -- to be warm-started from the
    StructuredPolicy, not cold.

The grid PDE remains the exact low-dim ANCHOR to validate against on synthetic data; this is the object
you deploy on the real, dependent, non-stationary tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from eval_sim.config import EvalConfig
from reclasso_har import GramState

PASS, FAIL, TIMEOUT = 1, 2, 3
_NS = 11  # width of the homogeneous state basis (see _features)


# --------------------------------------------------------------------------------------------------- #
# Sparse basis regression -- the REPO's elastic-net homotopy (reclasso_har), not a hand-rolled CD.
# --------------------------------------------------------------------------------------------------- #
def _enet_fit(
    X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float
) -> tuple[np.ndarray, float]:
    """Elastic-net (coef, intercept) via ``reclasso_har.GramState`` -- the Garrigues online-homotopy enet
    built this session -- with features standardized around its tested fit_intercept=True path. The L2
    term (l1_ratio<1) RETAINS correlated curvature terms that pure L1 arbitrarily drops; that dropout is
    what flattened the value function to linear -> risk-neutral -> ruinous max-leverage. (The online
    homotopy also warm-updates cheaply across the near-identical consecutive-t regressions -- next step.)"""
    if (
        y.std() < 1e-9
    ):  # constant target -> intercept only (a valid fit, not a bug-dodge)
        return np.zeros(X.shape[1]), float(y.mean())
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-12] = (
        1.0  # numerical guard for a constant column, NOT a signal-altering clip
    )
    Xs = np.ascontiguousarray((X - mu) / sd)
    coef_s, b0_s = GramState.from_block(
        Xs, np.ascontiguousarray(y, float)
    ).coef_intercept(alpha, l1_ratio)
    coef = coef_s / sd
    return coef, float(b0_s - coef @ mu)


def _select_alpha(
    X, y, l1_ratio, *, alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2), val_frac=0.3, seed=0
):
    """Pick the enet penalty by held-out MSE -- so the regularization is CV'd, not hand-set (this is a
    penalty-study repo; hand-picking the penalty would be the wrong look). Returns the best alpha."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    nv = max(1, int(val_frac * X.shape[0]))
    va, tr = idx[:nv], idx[nv:]
    best_a, best_mse = alpha_grid[0], np.inf
    for a in alpha_grid:
        coef, b0 = _enet_fit(X[tr], y[tr], a, l1_ratio)
        mse = float(np.mean((X[va] @ coef + b0 - y[va]) ** 2))
        if mse < best_mse:
            best_mse, best_a = mse, a
    return best_a


def _features(
    W: np.ndarray, M: np.ndarray, t: int, cfg: EvalConfig, T: int
) -> np.ndarray:
    """Homogeneous state basis (scale-free): drawdown & cushion in DD-units, log-wealth, time-to-go, and
    low-order interactions. Rich enough for LASSO to pick a sparse support."""
    D = cfg.max_drawdown
    dd = (M - W) / D  # drawdown as fraction of the allowance, >=0
    cush = (cfg.upper - W) / D  # distance to target, in DD units
    lw = np.log(W / cfg.initial)
    tau = np.full_like(W, (T - t) / T)  # time-to-go fraction (as array for stacking)
    one = np.ones_like(W)
    return np.stack(
        [
            one,
            dd,
            dd * dd,
            cush,
            cush * cush,
            tau,
            tau * tau,
            dd * tau,
            cush * tau,
            dd * cush,
            lw,
        ],
        axis=1,
    )


def _qfeatures(W, M, t, cfg, T, w) -> np.ndarray:
    """Q-basis = state features (x) [1, w, w^2] so Qhat is QUADRATIC in the action -> an INTERIOR
    optimum. (Linear-in-action Q would be risk-neutral -> bang-bang max leverage -> ruin.) (P, 3*_NS)."""
    s = _features(W, M, t, cfg, T)
    w = np.asarray(w, dtype=float)
    if w.ndim == 0:
        w = np.full(W.shape, w)
    return np.concatenate([s, s * w[:, None], s * (w * w)[:, None]], axis=1)


def _maxq(coef, b0, W, M, t, cfg, T, w_grid) -> np.ndarray:
    """max_w Qhat(t, state, w) over the action grid, vectorized across states."""
    return np.maximum.reduce(
        [_qfeatures(W, M, t, cfg, T, w) @ coef + b0 for w in w_grid]
    )


# --------------------------------------------------------------------------------------------------- #
# Geometric first-passage environment (dollar trailing DD, matches eval_sim rules; wealth COMPOUNDS).
# --------------------------------------------------------------------------------------------------- #
def simulate_geo(
    returns: np.ndarray,
    cfg: EvalConfig,
    policy: float | np.ndarray | Callable[..., np.ndarray],
    T: int | None = None,
):
    """Geometric first-passage: W_{t+1} = W_t (1 + w_t R_t), trailing dollar DD floor M-D, target U.
    ``policy`` is a float (constant fraction) or a callable (t, W, M) -> per-path w array. Returns
    (outcome[int8], W_path or None, M_path or None, vol_path or None) -- paths recorded only if track."""
    returns = np.asarray(returns, dtype=np.float64)
    P, Tr = returns.shape
    T = Tr if T is None else min(T, Tr)
    U, D = cfg.upper, cfg.max_drawdown
    W: np.ndarray = np.full(P, cfg.initial)
    M = np.full(P, cfg.initial)
    alive = np.ones(P, bool)
    outcome = np.zeros(P, np.int8)
    Wp = np.empty((P, T + 1))
    Mp = np.empty((P, T + 1))
    Wp[:, 0] = W
    Mp[:, 0] = M
    for t in range(T):
        if callable(policy):
            w = np.asarray(policy(t, W, M), dtype=float)
        elif isinstance(policy, np.ndarray):  # (P, T) per-step exploration control
            w = policy[:, t].astype(float)
        else:
            w = np.full(P, float(policy))  # constant fraction
        step = np.where(alive, w * returns[:, t], 0.0)
        W = W * (1.0 + step)
        np.maximum(M, W, out=M)
        passed = alive & (W >= U) & (t + 1 >= cfg.min_days * max(1, cfg.steps_per_day))
        fail = alive & (W <= M - D) & ~passed
        outcome[passed] = PASS
        outcome[fail] = FAIL
        alive &= ~(passed | fail)
        Wp[:, t + 1] = W
        Mp[:, t + 1] = M
    outcome[alive] = TIMEOUT
    return outcome, Wp, Mp


def pass_rate(outcome: np.ndarray) -> float:
    return float((outcome == PASS).mean())


@dataclass
class StructuredPolicy:
    """The LITERATURE-optimal two-region feedback for maximizing goal-reaching probability before drawdown
    under a leverage cap (Yuan & Li 2021, AIMS Math 6(8):8868; Browne 1999 for the finite-deadline form;
    lineage Dubins-Savage -> Pestien-Sudderth). The optimal control is NOT smooth and NOT a learned Q:

        w*(state) = merton          far from the drawdown barrier   (cushion >= thr; moderate ~Kelly)
                  = cap  (1+k)      near it                         (cushion <  thr; BANG-BANG, bold)

    Goal-reaching before drawdown REWARDS pressing when behind (bold play near the barrier), the opposite
    of the growth-optimal de-levering -- which is exactly why a plain fitted-Q that froze near the barrier
    was inverted. The constant fractional-Kelly anchor is the thr=0 special case, so a tuned structured
    family dominates it. ``cushion = (W - floor)/D`` in [0,1] (1 at the high-water mark, 0 at the floor)."""

    merton: float  # moderate leverage far from the barrier (~ fractional Kelly / Merton fraction)
    cap: float  # max leverage (the borrowing cap 1+k) applied BOLDLY near the barrier
    thr: float  # cushion fraction below which the constraint binds -> switch to bold
    cfg: EvalConfig

    def __call__(self, t, W, M):
        cushion = (W - (M - self.cfg.max_drawdown)) / self.cfg.max_drawdown
        return np.where(cushion < self.thr, self.cap, self.merton)


# --------------------------------------------------------------------------------------------------- #
# Risk-constrained-Kelly anchor (Busseti-Ryu-Boyd): max growth s.t. P(max drawdown > D) <= beta.
# --------------------------------------------------------------------------------------------------- #
def risk_constrained_kelly(
    sampler, cfg, T, *, f_grid=None, beta=0.30, n_paths=40_000, seed=0
):
    """Constant-fraction anchor. For each candidate fraction f: simulate, estimate the per-path growth
    rate and the probability the trailing drawdown ever exceeds D; pick the largest-growth f whose
    drawdown-breach probability stays <= beta. Returns (f_star, diagnostics)."""
    if f_grid is None:
        f_grid = np.round(np.arange(0.25, 6.01, 0.25), 2)
    rng = np.random.default_rng(seed)
    R = sampler.sample(n_paths, T, rng)
    rows = []
    best_f, best_g = None, -np.inf
    for f in f_grid:
        # growth path WITHOUT the knockout (pure compounding) to score growth; DD breach from the peak.
        # Ruin handled EXPLICITLY (no arbitrary log clip): a step with gross return <= 0 is bankruptcy,
        # which is always a >100% drawdown -> a breach, and is excluded from the growth mean (log undefined).
        gross = 1.0 + f * R
        ruined = (gross <= 0.0).any(axis=1)
        lg = np.zeros_like(gross)
        pos = gross > 0.0
        lg[pos] = np.log(gross[pos])
        logW = np.cumsum(lg, axis=1)
        W = cfg.initial * np.exp(logW)
        M = np.maximum.accumulate(
            np.concatenate([np.full((n_paths, 1), cfg.initial), W], axis=1), axis=1
        )
        breach = float((((M[:, 1:] - W) >= cfg.max_drawdown).any(1) | ruined).mean())
        good = ~ruined
        growth = float(logW[good, -1].mean() / T) if good.any() else float("-inf")
        rows.append((float(f), growth, breach))
        if breach <= beta and growth > best_g:
            best_g, best_f = growth, float(f)
    if (
        best_f is None
    ):  # constraint infeasible even at the smallest f -> take the smallest
        best_f = float(f_grid[0])
    return best_f, rows


# --------------------------------------------------------------------------------------------------- #
# LASSO fitted value iteration + one-step policy improvement over the anchor.
# --------------------------------------------------------------------------------------------------- #
@dataclass
class ADPPolicy:
    qcoef: list  # per-t (coef[3*_NS], b0) for Qhat(t, state, w)
    w_grid: np.ndarray
    cfg: EvalConfig
    T: int

    def __call__(self, t, W, M):
        """Greedy control from the fitted action-value: w*(state) = argmax_w Qhat(t, state, w)."""
        if t >= self.T:
            return np.full(W.shape, self.w_grid[0])
        coef, b0 = self.qcoef[t]
        best_v = None
        best_w = None
        for w in self.w_grid:
            v = _qfeatures(W, M, t, self.cfg, self.T, w) @ coef + b0
            if best_v is None:
                best_v, best_w = v, np.full(W.shape, w)
            else:
                take = v > best_v
                best_v = np.where(take, v, best_v)
                best_w = np.where(take, w, best_w)
        return best_w


def _simulate_record(R, cfg, base_action, jitter, eps, rng, w_max, T):
    """On-policy rollout recording (outcome, Wp, Mp, w_used). ``base_action`` is the current policy's
    action -- a scalar (the constant anchor) or a callable (t, W, M) -> per-path greedy w. Exploration is
    EPS-GREEDY: with prob eps a uniform action over [0, w_max] (full-range coverage EVERY round, so a
    frozen policy still generates progressed-state data and Qhat never goes OOD), else the base action
    plus ADDITIVE Gaussian jitter (which -- unlike multiplicative -- can escape w=0)."""
    P = R.shape[0]
    U, D = cfg.upper, cfg.max_drawdown
    W = np.full(P, cfg.initial)
    M = np.full(P, cfg.initial)
    alive = np.ones(P, bool)
    outcome = np.zeros(P, np.int8)
    Wp = np.empty((P, T + 1))
    Mp = np.empty((P, T + 1))
    wused = np.zeros((P, T))
    Wp[:, 0], Mp[:, 0] = W, M
    scalar_base = np.isscalar(base_action)
    for t in range(T):
        wb = np.broadcast_to(
            np.asarray(
                base_action if scalar_base else base_action(t, W, M), dtype=float
            ),
            (P,),
        )
        local = np.clip(
            wb + jitter * w_max * rng.standard_normal(P), 0.0, w_max
        )  # additive: escapes 0
        glob = rng.uniform(
            0.0, w_max, P
        )  # eps-greedy uniform: coverage regardless of the policy
        w = np.where(rng.random(P) < eps, glob, local)
        wused[:, t] = w
        step = np.where(alive, w * R[:, t], 0.0)
        W = W * (1.0 + step)
        np.maximum(M, W, out=M)
        passed = alive & (W >= U) & (t + 1 >= cfg.min_days * max(1, cfg.steps_per_day))
        fail = alive & (W <= M - D) & ~passed
        outcome[passed] = PASS
        outcome[fail] = FAIL
        alive &= ~(passed | fail)
        Wp[:, t + 1], Mp[:, t + 1] = W, M
    outcome[alive] = TIMEOUT
    return outcome, Wp, Mp, wused


def _backward_fqi(Wp, Mp, wused, outcome, cfg, T, alpha, l1_ratio, w_grid):
    """One backward fitted-Q pass over recorded transitions: regress the landed continuation
    V(t+1, s') = max_w Qhat(t+1, s', w) on Q-features(s_t, w_t). Returns per-t (coef, b0)."""
    U, D = cfg.upper, cfg.max_drawdown
    qcoef: list = [None] * T
    Vnext = (outcome == PASS).astype(
        np.float64
    )  # V(T, s_T) terminal (absorbed states carry 0/1)
    for t in range(T - 1, -1, -1):
        alive = (Wp[:, t] < U) & (Wp[:, t] > Mp[:, t] - D)
        if alive.sum() >= 300:
            X = _qfeatures(Wp[alive, t], Mp[alive, t], t, cfg, T, wused[alive, t])
            coef, b0 = _enet_fit(X, Vnext[alive], alpha, l1_ratio)
        else:
            coef, b0 = (
                np.zeros(3 * _NS),
                float(Vnext[alive].mean() if alive.any() else 0.0),
            )
        qcoef[t] = (coef, b0)
        Vt = _maxq(coef, b0, Wp[:, t], Mp[:, t], t, cfg, T, w_grid)
        Vnext = np.where(
            Wp[:, t] >= U, 1.0, np.where(Wp[:, t] <= Mp[:, t] - D, 0.0, Vt)
        )
    return qcoef


def fit_adp(
    sampler,
    cfg,
    T,
    *,
    anchor_f,
    n_iters=5,
    n_train=15_000,
    w_max=6.0,
    n_w=16,
    alpha=None,
    l1_ratio=0.3,
    jitter=0.15,
    eps=0.3,
    seed=1,
) -> ADPPolicy:
    """Approximate Policy Iteration for the prop-eval control (on-policy fitted-Q).

    The one-shot FQI failed the classic offline-RL way: the greedy exploits Qhat's errors on actions the
    fixed exploration barely covered -- it over-levered where Qhat extrapolated, then froze (w=0) when a
    wide exploration flooded it with dying paths (a frozen state is off the exploration's distribution, so
    its fitted continuation is over-optimistic). The fix is to make the data ON-POLICY and re-gather it
    each round: simulate under the CURRENT policy (jittered locally), fit Q on THAT data, go greedy,
    repeat. If a policy freezes, its OWN rollouts time out -> Q learns freezing is worthless -> the next
    round un-freezes. No hand-picked action cap; ``w_max`` is just the leverage limit."""
    rng = np.random.default_rng(seed)
    w_grid = np.linspace(0.0, w_max, n_w)
    qcoef = None
    for _ in range(n_iters):
        R = sampler.sample(n_train, T, rng)
        base = anchor_f if qcoef is None else ADPPolicy(qcoef, w_grid, cfg, T).__call__
        outcome, Wp, Mp, wused = _simulate_record(
            R, cfg, base, jitter, eps, rng, w_max, T
        )
        if (
            alpha is None
        ):  # CV-select the enet penalty once, on a representative mid-horizon slice
            tm = T // 2
            am = (Wp[:, tm] < cfg.upper) & (Wp[:, tm] > Mp[:, tm] - cfg.max_drawdown)
            alpha = (
                _select_alpha(
                    _qfeatures(Wp[am, tm], Mp[am, tm], tm, cfg, T, wused[am, tm]),
                    (outcome[am] == PASS).astype(float),
                    l1_ratio,
                )
                if am.sum() >= 500
                else 1e-3
            )
        qcoef = _backward_fqi(Wp, Mp, wused, outcome, cfg, T, alpha, l1_ratio, w_grid)
    assert qcoef is not None  # n_iters >= 1, so the loop set it
    return ADPPolicy(qcoef=qcoef, w_grid=w_grid, cfg=cfg, T=T)
