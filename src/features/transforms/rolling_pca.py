"""Causal rolling PCA: compress a dense-weak feature block into K factors, no look-ahead.

Motivation. The exog cross-section is "dense but weak" (Kozak-Nagel-Santosh / Shrinking the
Cross-Section): ~hundreds of features each carrying a tiny, redundant slice of signal. Ridge
with strong shrinkage and the residual tree's ``colsample_bytree -> 0.1`` bagging BOTH grope
toward a LOW-RANK structure — they're implicitly estimating a few factors of the panel.
Make it explicit: project the panel onto its top-K principal components.

Causality (the load-bearing property). At each refit point ``t_r`` the PCA basis is fit on the
trailing window ``[t_r - train_win, t_r)`` ONLY; rows in ``[t_r, t_{r+1})`` are projected onto
THAT basis. So an out-of-sample row's factors never see contemporaneous or future data. The
basis is refit on a cadence (``refit``) and held within the block — this amortizes the SVD,
mirroring the cadence-Ridge in the residualizer. Rows ``[0, train_win)`` use the initial
window's basis (same convention as :func:`rolling_robust_scale`; those rows are never scored
OOS, and the basis is within-window so it is causal w.r.t. the first OOS point).

Sign alignment. A PC's sign is arbitrary and can flip between adjacent windows; left unaligned
the factor time series would have spurious jumps at every refit. Each refit's PCs are sign-
flipped to agree (max dot product) with the previous block's. (Order swaps when eigenvalues
cross are NOT corrected here — a documented v1 limitation; coarse ``refit`` keeps them rare.)
"""
from __future__ import annotations

import numpy as np


def _fit_basis(window: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, Vt[:k]) — the centering mean and top-k principal directions of `window`."""
    mu = window.mean(axis=0)
    _u, _s, vt = np.linalg.svd(window - mu, full_matrices=False)
    return mu, vt[:k]


def rolling_pca(
    X: np.ndarray, train_win: int, n_components: int, refit: int, sign_align: bool = True
) -> np.ndarray:
    """Causal rolling projection of `X` (n_samples, n_features) onto its top-K rolling PCs.

    Returns (n_samples, K) factor scores, K = min(n_components, n_features). Row ``t >= train_win``
    is projected onto the basis fit on ``[t' - train_win, t')`` where ``t'`` is the last refit
    point ``<= t`` — strictly trailing, no look-ahead.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if train_win > n:
        raise ValueError(f"train_win ({train_win}) exceeds series length ({n})")
    k = min(n_components, p)
    out = np.empty((n, k), dtype=np.float64)

    mu, vt = _fit_basis(X[:train_win], k)  # initial-window basis for rows [0, train_win)
    out[:train_win] = (X[:train_win] - mu) @ vt.T
    prev_vt = vt
    for t_r in range(train_win, n, refit):
        mu, vt = _fit_basis(X[t_r - train_win:t_r], k)
        if sign_align:
            flip = np.sign(np.sum(vt * prev_vt, axis=1))
            flip[flip == 0] = 1.0
            vt = vt * flip[:, None]
            prev_vt = vt
        t_end = min(t_r + refit, n)
        out[t_r:t_end] = (X[t_r:t_end] - mu) @ vt.T
    return out
