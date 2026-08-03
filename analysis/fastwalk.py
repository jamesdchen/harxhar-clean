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
        return A.T @ A, A.T @ self.y[s - self.W:s]

    def _roll(self, s):
        lo_o, hi_o = self.s - self.W, self.s
        lo_n, hi_n = s - self.W, s
        if lo_n >= hi_o or lo_o >= hi_n:
            return self._exact(s)
        out, inn = slice(lo_o, lo_n), slice(hi_o, hi_n)
        Ao = np.hstack([np.ones((lo_n - lo_o, 1)), self.F[out]])
        Ai = np.hstack([np.ones((hi_n - hi_o, 1)), self.F[inn]])
        return (self.G - Ao.T @ Ao + Ai.T @ Ai,
                self.c - Ao.T @ self.y[out] + Ai.T @ self.y[inn])

    def advance(self, s):
        if self.s is None:
            self.G, self.c = self._exact(s)
        elif s != self.s:
            self.G, self.c = self._roll(s)
            # Checkpoint: recompute exactly and compare THE SAME window, so
            # the number reported is accumulated floating-point error and not
            # the genuine window-to-window change. Comparing across windows is
            # a bug an earlier version of this had, and it reported a drift of
            # 8e-3 when the true figure is at the 1e-13 level.
            if self.n_adv % self.refresh == 0:
                Ge, ce = self._exact(s)
                d = np.max(np.abs(self.G - Ge)) / max(1.0, np.max(np.abs(Ge)))
                self.max_drift = max(self.max_drift, float(d))
                self.G, self.c = Ge, ce
        self.s = s
        self.n_adv += 1
        return self.G, self.c


def block(G, c, T):
    """Gram and cross-product of the design ``[1, F @ T]`` from those of ``[1, F]``.

    ``T`` maps the full column set to an arm's own columns, so an arm never
    rebuilds anything. The intercept passes through untouched.
    """
    k = T.shape[1]
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
    """Solve ``(M + lam P) b = v`` for every lam in ``lams``.

    ``P`` is zero on the unpenalised leading columns and ``pen_diag`` (default
    identity) on the trailing ``npen``. Kept as plain solves: at p ~ 505 each
    costs O(p^3) ~ 1.3e8, which is a quarter of the rolling Gram update, so the
    penalty grid is not the bottleneck once the Gram stops being rebuilt.
    Residualising the unpenalised block and eigendecomposing once would make
    the grid nearly free, and is the next thing to do if this ever dominates.
    """
    pp = M.shape[0]
    d = np.zeros(pp)
    d[pp - npen:] = 1.0 if pen_diag is None else pen_diag
    I = 1e-8 * np.eye(pp)
    return {lam: np.linalg.solve(M + lam * np.diag(d) + I, v) for lam in lams}
