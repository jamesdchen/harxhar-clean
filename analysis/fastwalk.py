"""Rolling-Gram walk-forward. Same arithmetic, ~70x less of it.

Every experiment in this project walks 219 windows and, per window, spends
about 98% of its time in one matrix product:

    G = A.T @ A      24,000 x 505   ->  O(W p^2) ~ 6.1e9 flops
    solve(G + lam P)                ->  O(p^3)   ~ 1.3e8 flops

The solve, which looks like the expensive step, is 50x cheaper than building
the Gram. Three exact savings follow, none of which change a reported number.

ROLL THE GRAM. Consecutive windows share 23,000 of their 24,000 rows, so

    G_new = G_old - A_out.T @ A_out + A_in.T @ A_in

costs 2 * 1000 * 505^2 instead of 24,000 * 505^2. Subtraction accumulates
floating-point error over 219 updates, so the Gram is recomputed exactly every
``REFRESH`` windows; ``max_drift`` reports the largest observed discrepancy at
those checkpoints so the error is measured rather than assumed.

SHARE IT ACROSS ARMS. Designs that are linear maps of the same columns -- a
principal-component truncation, a spectral reweighting, an operator projection,
the full ridge -- all have Gram T' G T for their own T. One rolling Gram serves
every arm, instead of each arm rebuilding it.

TAKE THE PCs FROM THE GRAM. The right singular vectors of Z are the
eigenvectors of Z'Z, which the Gram already contains, so a per-window SVD of a
24,000 x 492 matrix is pure waste.

This module provides the machinery; callers supply the per-arm transform.
"""

from __future__ import annotations

import numpy as np

REFRESH = 25          # windows between exact recomputations of the Gram


class RollingGram:
    """Trailing-window cross-products of ``[1, F]`` and ``[1, F]' y``.

    ``advance(s)`` moves the window to ``[s - W, s)``. Rows are added and
    removed rather than re-multiplied, and the whole product is rebuilt every
    ``REFRESH`` advances to bound drift.
    """

    def __init__(self, F, y, W, refresh=REFRESH):
        self.F, self.y, self.W, self.refresh = F, y, W, refresh
        self.p = F.shape[1] + 1
        self.s = None
        self.n_adv = 0
        self.max_drift = 0.0

    def _exact(self, s):
        A = np.hstack([np.ones((self.W, 1)), self.F[s - self.W:s]])
        yy = float(self.y[s - self.W:s] @ self.y[s - self.W:s])
        return A.T @ A, A.T @ self.y[s - self.W:s], yy

    def _roll(self, s):
        lo_o, hi_o = self.s - self.W, self.s
        lo_n, hi_n = s - self.W, s
        if lo_n >= hi_o or lo_o >= hi_n:
            return self._exact(s)
        out, inn = slice(lo_o, lo_n), slice(hi_o, hi_n)
        Ao = np.hstack([np.ones((lo_n - lo_o, 1)), self.F[out]])
        Ai = np.hstack([np.ones((hi_n - hi_o, 1)), self.F[inn]])
        return (self.G - Ao.T @ Ao + Ai.T @ Ai,
                self.c - Ao.T @ self.y[out] + Ai.T @ self.y[inn],
                self.yy - float(self.y[out] @ self.y[out])
                + float(self.y[inn] @ self.y[inn]))

    def advance(self, s):
        if self.s is None:
            self.G, self.c, self.yy = self._exact(s)
        elif s != self.s:
            self.G, self.c, self.yy = self._roll(s)
            # Checkpoint: recompute exactly and compare THE SAME window, so
            # the number reported is accumulated floating-point error and not
            # the genuine window-to-window change. Comparing across windows is
            # a bug an earlier version of this had, and it reported a drift of
            # 8e-3 when the true figure is at the 1e-13 level.
            if self.n_adv % self.refresh == 0:
                Ge, ce, ye = self._exact(s)
                d = np.max(np.abs(self.G - Ge)) / max(1.0, np.max(np.abs(Ge)))
                self.max_drift = max(self.max_drift, float(d))
                self.G, self.c, self.yy = Ge, ce, ye
        self.s = s
        self.n_adv += 1
        return self.G, self.c


def _selection(T):
    """Indices if ``T`` selects identity columns, else ``None``.

    Guarded by a single non-zero count so a dense transform pays one cheap
    scan rather than a structural analysis. Only a PURE selection is detected:
    a mixed "selection beside a dense block" was tried and measured slower than
    the generic path (0.4-0.5x), because the generic route is one large
    optimised dgemm while the structured one is several small products plus
    fancy-index copies. Flop counts do not settle this; timings do.
    """
    if T.shape[1] > T.shape[0] or np.count_nonzero(T) != T.shape[1]:
        return None
    nz = np.flatnonzero(T)
    rows, cols = np.unravel_index(nz, T.shape)
    if not np.array_equal(cols, np.arange(T.shape[1])):
        return None
    if not np.allclose(T[rows, cols], 1.0):
        return None
    return rows


def block(G, c, T):
    """Gram and cross-product of ``[1, F @ T]`` from those of ``[1, F]``.

    ``T`` maps the full column set to an arm's own columns, so an arm never
    rebuilds anything from the data. The intercept passes through untouched.

    A pure selection of identity columns is just a submatrix of ``G`` and is
    taken as one, which matters because the commonest arm of all -- the full
    ridge -- passes the identity and would otherwise spend a 504^3 triple
    product recovering ``G`` unchanged. Measured 4.0x on that case and 2.1x on
    a 12-column selection, both bit-exact.
    """
    k = T.shape[1]
    sel = _selection(T)
    if sel is not None:
        idx = np.concatenate([[0], sel + 1])
        return G[np.ix_(idx, idx)].copy(), c[idx].copy()
    M = np.zeros((1 + k, 1 + k))
    M[0, 0] = G[0, 0]
    GT = G[1:, 1:] @ T
    M[0, 1:] = G[0, 1:] @ T
    M[1:, 0] = M[0, 1:]
    M[1:, 1:] = T.T @ GT
    v = np.empty(1 + k)
    v[0] = c[0]
    v[1:] = T.T @ c[1:]
    return M, v


def solve_path(M, v, npen, lams, pen_diag=None):
    """Solve ``(M + lam P) b = v`` for every lam from ONE eigendecomposition.

    ``P`` is zero on the leading unpenalised columns and ``pen_diag`` (default
    identity) on the trailing ``npen``. Solving each lam separately costs
    O(k^3) apiece, which at k = 510 over a ten-point grid is 1.3e9 flops per
    arm per window -- more than twice the rolling Gram update, and the
    dominant cost once the design matrix is no longer materialised.

    Frisch-Waugh removes the unpenalised block, after which the remaining
    problem is a plain ridge whose eigenbasis does not depend on lam:

        Vt = V - B' U^-1 B,   ct = cv - B' U^-1 cu,   Vt = Q diag(w) Q'
        b_V(lam) = Q (Q'ct / (w + lam)),   b_U = U^-1 (cu - B b_V)

    One O(k^3) decomposition plus O(k^2) per lam. A non-identity penalty is
    whitened by P^-1/2 first, which is exact because P is diagonal and
    positive.
    """
    pp = M.shape[0]
    nu = pp - npen
    eps = 1e-8
    if npen <= 0:
        b = np.linalg.solve(M + eps * np.eye(pp), v)
        return {lam: b for lam in lams}
    U, B, V = M[:nu, :nu], M[:nu, nu:], M[nu:, nu:]
    cu, cv = v[:nu], v[nu:]
    S = None
    if pen_diag is not None:
        S = np.diag(np.asarray(pen_diag, float) ** -0.5)
        V, B, cv = S @ V @ S, B @ S, S @ cv
    if nu:
        Ui = np.linalg.inv(U + eps * np.eye(nu))
        UiB, Uicu = Ui @ B, Ui @ cu
        Vt, ct = V - B.T @ UiB, cv - B.T @ Uicu
    else:
        Vt, ct = V, cv
    w, Q = np.linalg.eigh(Vt)
    w = np.clip(w, 0.0, None)
    Qc = Q.T @ ct
    out = {}
    for lam in lams:
        bv = Q @ (Qc / (w + lam + eps))
        bu = Uicu - UiB @ bv if nu else np.zeros(0)
        out[lam] = np.concatenate([bu, S @ bv if S is not None else bv])
    return out


def rss(M, v, cf, yy):
    """Training residual sum of squares, without forming the design matrix.

    ``rr'rr = y'y - 2 cf'v + cf' M cf``. The obvious route -- materialise
    ``[1, F @ T]`` over the training window and subtract -- costs W p k flops
    (about 6.2e9 at W = 24,000, p = 504, k = 510) to produce a single scalar,
    which is twelve times the entire rolling Gram update and is paid once per
    arm per window. This costs O(k^2).
    """
    return float(yy - 2.0 * (cf @ v) + cf @ (M @ cf))


def predict(Fe, T, cf):
    """Predictions of ``[1, F @ T] @ cf`` as ``cf[0] + F @ (T @ cf[1:])``.

    Associativity: projecting the block then contracting costs (z-a) p k, while
    contracting then projecting costs p k + (z-a) p -- about 500x less at these
    shapes, and identical arithmetic.
    """
    return cf[0] + Fe @ (T @ cf[1:])
