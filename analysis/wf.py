"""Walk-forward helpers shared by the alpha-manifestation study.

Thin wrappers over :class:`src.models.rolling_least_squares.RollingLeastSquares` — the
same rank-1 sliding-window ridge the production Ridge path uses — so every number in the
study comes from a strictly causal, refit-as-you-go fit rather than a full-sample one.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rolling_least_squares import RollingLeastSquares  # noqa: E402


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    alpha: float = 1.0,
    refit_every: int = 1,
) -> np.ndarray:
    """Rolling-window ridge OOS predictions for rows ``[train_win, n)``.

    Returns an array of length ``n - train_win``: prediction for row
    ``train_win + i`` fit on the ``train_win`` rows strictly before it.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = len(X)
    solver = RollingLeastSquares(alpha=alpha, fit_intercept=True)
    solver.init_window(X[:train_win], y[:train_win])
    solver.solve()
    out = np.empty(n - train_win, dtype=np.float64)
    for i in range(n - train_win):
        t = train_win + i
        if i and i % refit_every == 0:
            solver.solve()
        out[i] = solver.predict_one(X[t])
        solver.roll(X[t], y[t], X[t - train_win], y[t - train_win])
    return out


def walk_forward_coefs(
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    alpha: float = 1.0,
    every: int = 480,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficient path: ``(row_index, coefs)`` sampled every ``every`` rows."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = len(X)
    solver = RollingLeastSquares(alpha=alpha, fit_intercept=True)
    solver.init_window(X[:train_win], y[:train_win])
    rows, coefs = [], []
    for i in range(n - train_win):
        t = train_win + i
        if i % every == 0:
            solver.solve()
            rows.append(t)
            coefs.append(solver._coef.copy())
        solver.roll(X[t], y[t], X[t - train_win], y[t - train_win])
    return np.asarray(rows), np.asarray(coefs)


def nw_tstat(x: np.ndarray, y: np.ndarray, lags: int = 48) -> tuple[float, float]:
    """(slope, Newey-West t) of ``y ~ a + b x`` with ``lags`` Bartlett lags.

    Both series are bar-level and heavily autocorrelated; an OLS t on 200k bars is
    meaningless, so every reported t in the study is HAC.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return 0.0, 0.0
    b = float(xc @ (y - y.mean())) / sxx
    u = (y - y.mean()) - b * xc
    g = xc * u
    s = float(g @ g)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * float(g[L:] @ g[:-L])
    se = np.sqrt(max(s, 0.0)) / sxx
    return b, (b / se if se > 0 else 0.0)


def r2_oos(y: np.ndarray, pred: np.ndarray) -> float:
    """OOS R^2 against the *mean-zero* benchmark (the residual's own null)."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return 1.0 - float(((y - pred) ** 2).sum()) / float((y**2).sum())
