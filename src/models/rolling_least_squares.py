"""Incremental sliding-window ridge least squares (rank-1 updates).

Maintains the training window's sufficient statistics — ``XᵀX``, ``Xᵀy`` and
the column/target sums needed for the (unpenalized) intercept — and rolls them
in ``O(p²)`` per step as the window slides one row, instead of rebuilding the
``O(W·p²)`` normal equations from scratch every bar. Predictions are identical
(to numerical precision) to ``sklearn.linear_model.Ridge(alpha,
fit_intercept=True)`` refit on each window.

This is the *incremental* completion of the already-incremental rolling robust
scaler: same sliding-window spirit, now for the solve rather than the scaling.

Duck-typed for :class:`src.backtest.multi_stage.MultiStageBacktest`: exposing
``init_window`` + ``roll`` is what flips the backtest onto its incremental fast
path. Valid only for a linear model of a *static* target over a sliding window
(plain Ridge under ``IdentityResidualizer``, no feature stage); the backtest
gates on exactly that, so kNN/trees and the residual-window paths fall back to
the normal fresh-refit loop untouched.
"""

from __future__ import annotations

import numpy as np


class RollingLeastSquares:
    """Sliding-window ridge regression by rank-1 sufficient-statistic updates.

    Parameters
    ----------
    alpha
        Ridge penalty (matches ``sklearn`` ``Ridge(alpha=...)``).
    fit_intercept
        If True, an unpenalized intercept is fit by centering the window
        (exactly as ``sklearn`` does), so coefficients solve
        ``(XcᵀXc + αI) b = Xcᵀyc`` with ``Xc = X − X̄``.
    reinit_every
        Rebuild ``XᵀX``/``Xᵀy`` exactly from the current window every this many
        steps, to bound float drift from the running add/subtract. ``0`` =
        never (pure rolling). The backtest reads this and supplies the window
        slice (the solver itself stores no window buffer).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        reinit_every: int = 5000,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.reinit_every = int(reinit_every)
        self.n = 0
        self.p = 0
        self._Sxx: np.ndarray = np.empty((0, 0))
        self._Sxy: np.ndarray = np.empty(0)
        self._sx: np.ndarray = np.empty(0)
        self._sy = 0.0
        self._coef: np.ndarray = np.empty(0)
        self._intercept = 0.0

    def init_window(self, X: np.ndarray, y: np.ndarray) -> "RollingLeastSquares":
        """(Re)build sufficient statistics exactly from a full window. No solve."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        self.n, self.p = X.shape
        self._Sxx = X.T @ X
        self._Sxy = X.T @ y
        self._sx = X.sum(axis=0)
        self._sy = float(y.sum())
        return self

    def roll(
        self,
        x_in: np.ndarray,
        y_in: float,
        x_out: np.ndarray,
        y_out: float,
    ) -> None:
        """Slide the window by one: add row ``x_in`` / drop row ``x_out``. ``O(p²)``."""
        x_in = np.asarray(x_in, dtype=np.float64)
        x_out = np.asarray(x_out, dtype=np.float64)
        self._Sxx += np.outer(x_in, x_in) - np.outer(x_out, x_out)
        self._Sxy += x_in * y_in - x_out * y_out
        self._sx += x_in - x_out
        self._sy += float(y_in) - float(y_out)

    def grow(self, x_in: np.ndarray, y_in: float) -> None:
        """Add one row WITHOUT dropping — the window grows by one. ``O(p²)``. Use while the
        training window is still filling to full size (a rank-1 add), instead of an ``O(W·p²)``
        ``init_window`` rebuild every bar; ``roll`` handles steady state once the window is full."""
        x_in = np.asarray(x_in, dtype=np.float64)
        self._Sxx += np.outer(x_in, x_in)
        self._Sxy += x_in * y_in
        self._sx += x_in
        self._sy += float(y_in)
        self.n += 1

    def solve(self) -> None:
        """Refresh coefficients from the current sufficient statistics. ``O(p³)``."""
        n = self.n
        if self.fit_intercept:
            gram = self._Sxx - np.outer(self._sx, self._sx) / n
            rhs = self._Sxy - self._sx * (self._sy / n)
        else:
            gram = self._Sxx
            rhs = self._Sxy
        a = gram + self.alpha * np.eye(self.p)
        self._coef = np.linalg.solve(a, rhs)
        if self.fit_intercept:
            self._intercept = self._sy / n - float(self._sx @ self._coef) / n
        else:
            self._intercept = 0.0

    def predict_one(self, x: np.ndarray) -> float:
        """Predict a single row (the backtest's per-step call)."""
        return float(np.asarray(x, dtype=np.float64) @ self._coef + self._intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """sklearn-style batch predict."""
        return np.asarray(X, dtype=np.float64) @ self._coef + self._intercept
