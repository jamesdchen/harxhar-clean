"""reclasso_har.py -- recursive (incremental) elastic net over a sliding window.

Re-implementation of the RecLasso online-Lasso homotopy (Garrigues & El Ghaoui,
"An Homotopy Algorithm for the Lasso with Online Observations"). The original lived
only as a flattened chat transcript (indentation lost on export), so this is a clean
re-author of the same algorithm, validated against sklearn.

Elastic net <-> Lasso-on-Gram
-----------------------------
    min_b  0.5 ||y - X b||^2 + mu ||b||_1 + 0.5 lam2 ||b||^2
is a LASSO in "Gram form" with  G = X^T X + lam2 I  and  c = X^T y:
    min_b  0.5 b^T G b - c^T b + mu ||b||_1.
The L2 term is just a fixed ridge on the Gram diagonal -- it keeps G strictly positive
definite, so the active-set solves never hit the collinearity blow-up that the explicit
inverse (K) form of RecLasso suffered (the transcript diagnoses exactly this).

sklearn.linear_model.ElasticNet(alpha, l1_ratio, fit_intercept=False) over N rows maps to
    mu   = N * alpha * l1_ratio          # L1, total penalty
    lam2 = N * alpha * (1 - l1_ratio)    # L2 ridge on the Gram diagonal

The mu-solution path is piecewise linear; we follow it (LARS / homotopy) from mu_max down
to the target, tracking the active set and its signs. Active-set systems are solved
directly (active sets are small relative to the window); rank-1 Cholesky maintenance and
the sliding add/remove-observation updates are layered on top (see GramState).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def _solve(g_aa: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve g_aa x = rhs for SPD g_aa; lstsq fallback if numerically singular."""
    try:
        return np.linalg.solve(g_aa, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(g_aa, rhs, rcond=None)[0]


def _homotopy_walk(
    g: np.ndarray, c: np.ndarray, mu_target: float, max_iter: int = 100000
):
    """Walk the LASSO path  min_b 0.5 b^T g b - c^T b + mu||b||_1  from mu_max = max|c| down to
    mu_target. Returns (segments, mu_max). Each segment is (mu_top, mu_bot, active_idx, w0, w1),
    valid for mu in [mu_bot, mu_top] with theta_active(mu) = w0 - mu * w1 (piecewise linear)."""
    m = c.shape[0]
    cabs = np.abs(c)
    mu_max = float(cabs.max())
    segs: list = []
    if mu_max <= mu_target + _EPS:
        return segs, mu_max  # whole path above target -> zero
    mu = mu_max
    a: list[int] = [int(np.argmax(cabs))]
    s: list[float] = [float(np.sign(c[a[0]]))]
    just_dropped = -1
    for _ in range(max_iter):
        a_arr = np.asarray(
            a, dtype=int
        )  # dtype=int so an EMPTY active set stays integer-indexable
        s_arr = np.asarray(
            s, dtype=float
        )  # (else np.asarray([]) -> float64 -> np.ix_ IndexError)
        g_aa = g[np.ix_(a_arr, a_arr)]
        w0 = _solve(g_aa, c[a_arr])
        w1 = _solve(g_aa, s_arr)
        inactive = np.asarray([j for j in range(m) if j not in a], dtype=int)
        best_mu = mu_target
        best: tuple | None = None
        # drop events: an active coef returns to zero, theta_j(mu) = w0_j - mu w1_j = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_drop = w0 / w1
        for i, mj in enumerate(mu_drop):
            if np.isfinite(mj) and best_mu + _EPS < mj < mu - _EPS:
                best_mu, best = float(mj), ("drop", i)
        # entry events: an inactive feature's correlation hits |rho_j| = mu, rho_j = p_j + mu q_j
        if inactive.size:
            g_ia = g[np.ix_(inactive, a_arr)]
            p = c[inactive] - g_ia @ w0
            q = g_ia @ w1
            with np.errstate(divide="ignore", invalid="ignore"):
                mu_pos = p / (1.0 - q)  # rho_j = +mu -> sign +1
                mu_neg = -p / (1.0 + q)  # rho_j = -mu -> sign -1
            for arr, sign in ((mu_pos, +1.0), (mu_neg, -1.0)):
                for i, me in enumerate(arr):
                    j = int(inactive[i])
                    if j == just_dropped:
                        continue  # guard against immediate re-entry (cycling)
                    if np.isfinite(me) and best_mu + _EPS < me < mu - _EPS:
                        best_mu, best = float(me), ("enter", j, sign)
        segs.append((mu, best_mu, a_arr.copy(), w0.copy(), w1.copy()))
        if best is None:
            return segs, mu_max
        mu = best_mu
        if best[0] == "drop":
            i = best[1]
            just_dropped = a[i]
            del a[i]
            del s[i]
        else:
            _, j, sign = best
            a.append(j)
            s.append(sign)
            just_dropped = -1
    raise RuntimeError("lasso homotopy did not converge within max_iter")


def lasso_homotopy(
    g: np.ndarray, c: np.ndarray, mu_target: float, max_iter: int = 100000
):
    """Coefs at a single mu_target. Returns (theta[m], active_list, signs_list)."""
    segs, _ = _homotopy_walk(g, c, mu_target, max_iter)
    theta = np.zeros(c.shape[0])
    if not segs:
        return theta, [], []
    _, _, a_arr, w0, w1 = segs[-1]
    theta[a_arr] = w0 - mu_target * w1
    return theta, a_arr.tolist(), np.sign(theta[a_arr]).tolist()


def lasso_path_coefs(g: np.ndarray, c: np.ndarray, mu_grid) -> np.ndarray:
    """Coefs at every mu in mu_grid via ONE homotopy walk (the whole regularization path for the
    price of one solve). Returns (len(mu_grid), m). mu_grid order is preserved."""
    grid = np.atleast_1d(np.asarray(mu_grid, dtype=float))
    segs, mu_max = _homotopy_walk(g, c, float(grid.min()))
    out = np.zeros((grid.size, c.shape[0]))
    for k, mg in enumerate(grid):
        if mg >= mu_max - _EPS:
            continue  # above the first entry -> zero coefficients
        for mtop, mbot, a_arr, w0, w1 in segs:
            if mbot - 1e-9 <= mg <= mtop + 1e-9:
                out[k, a_arr] = w0 - mg * w1
                break
    return out


def enet_coef(
    g: np.ndarray, c: np.ndarray, n: int, alpha: float, l1_ratio: float
) -> np.ndarray:
    """Elastic-net coefficients from a precomputed Gram g = X^T X and c = X^T y over n rows,
    matching sklearn ElasticNet(alpha, l1_ratio, fit_intercept=False). g is NOT mutated."""
    lam2 = n * alpha * (1.0 - l1_ratio)
    mu = n * alpha * l1_ratio
    g_reg = g.copy()
    if lam2 > 0.0:
        g_reg[np.diag_indices_from(g_reg)] += lam2
    theta, _, _ = lasso_homotopy(g_reg, c, mu)
    return theta


def enet_fit(x: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float) -> np.ndarray:
    """Batch elastic net on raw (x, y), matching sklearn ElasticNet(fit_intercept=False)."""
    n = x.shape[0]
    return enet_coef(x.T @ x, x.T @ y, n, alpha, l1_ratio)


def enet_online(
    Gr, c, mu_vec, u, w, sigma, theta, A, s, locked, max_events=100000, tol=1e-9
):
    """Garrigues-El Ghaoui ONLINE-observation homotopy for ONE rank-1 data change.

    Transforms the elastic-net solution as the data Gram/correlation move
        Gr -> Gr + sigma * outer(u, u),   c -> c + sigma * u * w
    along t in [0, 1], following the piecewise path and pivoting the active set at each breakpoint.
    Finite breakpoints -> terminates exactly (no fallback); ridge (baked into Gr's diagonal) makes the
    problem strongly convex so the active solves stay well posed.

    Per-coordinate L1 penalty ``mu_vec`` (0 on unpenalized coords). ``locked`` (bool[m]) marks coords
    that never leave the active set -- the unpenalized intercept + HAR block, i.e. OLS-with-free-HAR
    inside the elastic net (matches _cadence_enet_har_unpen, fit_intercept=True).

    (theta, A, s) is optimal for the PRE-update (Gr, c); MUTATES Gr, c to the post-update state and
    returns the post-update (theta, A, s). Warm across bars: pass A/s back in for the next update.
    KKT:  r = c - Gr theta;  active j: r_j = mu_j s_j;  inactive j: |r_j| <= mu_j.
    """
    m = c.shape[0]
    Aset = set(A)
    dcur = 0.0
    n_ev = 0
    while dcur < 1.0 - tol:
        Aa = np.asarray(A, dtype=int)
        sA = np.asarray(s, dtype=float)
        M = Gr[np.ix_(Aa, Aa)]
        tA = _solve(M, c[Aa] - mu_vec[Aa] * sA)  # active theta at this segment base
        p = _solve(M, u[Aa])
        a = float(u[Aa] @ p)
        e0 = float(u[Aa] @ tA)
        dir_ = w - e0
        drem = 1.0 - dcur
        theta_full = np.zeros(m)
        theta_full[Aa] = tA
        r = c - Gr @ theta_full  # correlations at segment base
        inact = np.array([j for j in range(m) if j not in Aset], dtype=int)
        dvec = Gr[np.ix_(inact, Aa)] @ p  # d_j for inactive

        def d_of_g(gstar):
            den = sigma * (dir_ - gstar * a)
            return gstar / den if abs(den) > tol else np.inf

        best_d, best = drem, ("end",)
        # active-exit events (skip locked coords)
        for li, j in enumerate(A):
            if locked[j] or abs(p[li]) <= tol:
                continue
            d = d_of_g(-tA[li] / p[li])
            if tol < d < best_d - tol:
                best_d, best = d, ("exit", li)
        # inactive-entry events
        denom = u[inact] - dvec
        for k in range(inact.size):
            dj = denom[k]
            if abs(dj) <= tol:
                continue
            j = int(inact[k])
            for target in (mu_vec[j], -mu_vec[j]):
                d = d_of_g((target - r[j]) / dj)
                if tol < d < best_d - tol:
                    best_d, best = d, ("enter", j, 1.0 if target > 0 else -1.0)
        # advance by best_d
        gstep = best_d * sigma * dir_ / (1.0 + best_d * sigma * a)
        theta = np.zeros(m)
        theta[Aa] = tA + gstep * p
        Gr += (best_d * sigma) * np.outer(u, u)
        c += (best_d * sigma) * u * w
        dcur += best_d
        if best[0] == "exit":
            li = best[1]
            Aset.discard(A[li])
            del A[li]
            del s[li]
        elif best[0] == "enter":
            _, j, sgn = best
            A.append(j)
            s.append(sgn)
            Aset.add(j)
        n_ev += 1
        if n_ev > max_events:
            raise RuntimeError("online homotopy exceeded max_events")
    return theta, A, s


# --- online regularization: leakage-clean forward validation (Stage 2) ------------------------


def forward_window_split(t_end: int, window: int, val_len: int, embargo: int):
    """Leakage-clean forward split of the training window ending (exclusive) at t_end:
        fit = [t_end-window, t_end-val_len-embargo)   fit candidate coefs here
        gap = [..,           t_end-val_len)           embargo -- kills HAR rolling-window bleed
        val = [t_end-val_len, t_end)                  forward block -- picks the penalty
    The OOS region (>= t_end) is never touched. Returns (fit_lo, fit_hi, val_lo, val_hi)."""
    val_hi = t_end
    val_lo = t_end - val_len
    fit_hi = val_lo - embargo
    fit_lo = t_end - window
    return fit_lo, fit_hi, val_lo, val_hi


def select_l1_forward(g_fit, c_fit, x_val, y_val, mu_grid, xbar, ybar):
    """Pick the L1 penalty mu (ridge already baked into g_fit) that minimizes one-step-ahead MSE on
    a FORWARD validation block -- the principled, deployment-matched, leakage-free selector (vs
    LOOCV, which assumes the within-window exchangeability that vol regimes break). g_fit / c_fit
    are the fit-block sufficient stats CENTERED by the fit-block means xbar, ybar (so the val block
    is scored fit_intercept-style with no leak); the whole mu_grid is scored from one homotopy walk.
    Returns (best_mu, mse_grid)."""
    grid = np.atleast_1d(np.asarray(mu_grid, dtype=float))
    coefs = lasso_path_coefs(g_fit, c_fit, grid)  # (G, m)
    resid = (x_val - xbar) @ coefs.T - (y_val - ybar)[:, None]  # (n_val, G)
    mse = np.mean(resid * resid, axis=0)
    k = int(np.argmin(mse))
    return float(grid[k]), mse


class GramState:
    """Incremental Gram / correlation maintenance for a fixed-width sliding window.

    Holds G = X_win^T X_win and c = X_win^T y over the current window, updated rank-1 as
    rows enter (add) and leave (remove) -- so a window slide costs O(m^2), not the O(W m^2)
    of recomputing the Gram. The homotopy solve then runs on (G, c). Floating-point error
    accumulates over many rank-1 updates (the transcript's documented drift), so callers on
    long streams should periodically refresh() from the exact window block.
    """

    def __init__(self, m: int):
        self.m = m
        self.g = np.zeros((m, m))  # X^T X
        self.c = np.zeros(m)  # X^T y
        self.sx = np.zeros(m)  # column sums of X (for window-centering / intercept)
        self.sy = 0.0  # sum of y
        self.n = 0

    @classmethod
    def from_block(cls, xb: np.ndarray, yb: np.ndarray) -> "GramState":
        st = cls(xb.shape[1])
        st.refresh(xb, yb)
        return st

    def refresh(self, xb: np.ndarray, yb: np.ndarray) -> None:
        """Recompute the sufficient statistics exactly from the window block (resets drift)."""
        self.g = xb.T @ xb
        self.c = xb.T @ yb
        self.sx = xb.sum(axis=0)
        self.sy = float(yb.sum())
        self.n = xb.shape[0]

    def add(self, x: np.ndarray, yv: float) -> None:
        self.g += np.outer(x, x)
        self.c += x * yv
        self.sx += x
        self.sy += yv
        self.n += 1

    def remove(self, x: np.ndarray, yv: float) -> None:
        self.g -= np.outer(x, x)
        self.c -= x * yv
        self.sx -= x
        self.sy -= yv
        self.n -= 1

    def slide(
        self, x_new: np.ndarray, y_new: float, x_old: np.ndarray, y_old: float
    ) -> None:
        """Advance the window one bar: drop the oldest row, append the newest."""
        self.add(x_new, y_new)
        self.remove(x_old, y_old)

    def coef(self, alpha: float, l1_ratio: float) -> np.ndarray:
        """Elastic-net coefficients on the current window, fit_intercept=False."""
        return enet_coef(self.g, self.c, self.n, alpha, l1_ratio)

    def coef_intercept(self, alpha: float, l1_ratio: float) -> tuple[np.ndarray, float]:
        """(coef, intercept) matching sklearn ElasticNet(..., fit_intercept=True): fit on the
        window-CENTERED data (penalty excludes the intercept), then intercept = ybar - xbar.coef.
        Centering is itself incremental -- centered Gram = G - sx sx^T / n, c = c - sx sy / n."""
        inv_n = 1.0 / self.n
        g_c = self.g - np.outer(self.sx, self.sx) * inv_n
        c_c = self.c - self.sx * (self.sy * inv_n)
        coef = enet_coef(g_c, c_c, self.n, alpha, l1_ratio)
        intercept = self.sy * inv_n - (self.sx * inv_n) @ coef
        return coef, float(intercept)

    def centered_ridged(self, lam2: float):
        """(g, c, xbar, ybar) for the current window: window-centered Gram with ridge lam2 added,
        centered correlations, and the window means -- the inputs select_l1_forward expects."""
        inv_n = 1.0 / self.n
        xbar = self.sx * inv_n
        ybar = self.sy * inv_n
        g_c = self.g - np.outer(self.sx, self.sx) * inv_n
        if lam2 > 0.0:
            g_c = g_c + lam2 * np.eye(self.m)
        c_c = self.c - self.sx * (self.sy * inv_n)
        return g_c, c_c, xbar, ybar

    def coef_intercept_at_mu(self, lam2: float, mu: float) -> tuple[np.ndarray, float]:
        """(coef, intercept) at a fixed ridge lam2 and L1 penalty mu (the online-lambda refit, once
        mu has been selected on a forward block). Centering -> fit_intercept=True semantics."""
        g_c, c_c, xbar, ybar = self.centered_ridged(lam2)
        coef, _, _ = lasso_homotopy(g_c, c_c, mu)
        return coef, float(ybar - xbar @ coef)
