"""Residualizer — a baseline regressor wrapped as a feature transform.

A residualizer treats ``baseline.predict(X)`` as a feature transformation of
the target: ``residuals = y - baseline.predict(X)``. The baseline is any
sklearn-style regressor (``.fit(X, y)`` / ``.predict(X)`` — Ridge, OLS, a
tree, etc.); the residuals are the part of ``y`` the baseline didn't explain
and are typically the input to a downstream model.

The factory pattern (``baseline_factory: () -> baseline``) matches the
convention used by :func:`src.features.scaling.run_backtest` and
:class:`src.backtest.multi_stage.MultiStageBacktest` — each call to
:meth:`Residualizer.fit` instantiates a fresh baseline so walk-forward
refits are clean and the previous fit's state doesn't leak in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class Residualizer:
    """Fit a baseline regressor, then expose ``y - baseline.predict(X)``.

    Parameters
    ----------
    baseline_factory
        Zero-argument callable returning a fresh sklearn-style regressor with
        ``.fit(X, y)`` and ``.predict(X)``. Called once per :meth:`fit`.

    Example
    -------
    ::

        from sklearn.linear_model import Ridge
        from src.features.residualizer import Residualizer

        res = Residualizer(lambda: Ridge(alpha=1.0)).fit(X_train, y_train)
        residuals_train = res.residuals(X_train, y_train)
        # At test time t:
        baseline_hat = res.predict(X[t:t + 1])
        v_test = res.residuals(X[t - W : t], y[t - W : t])
    """

    def __init__(self, baseline_factory: Callable[[], Any]) -> None:
        self.baseline_factory = baseline_factory
        self.baseline: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Residualizer":
        """Instantiate a fresh baseline and fit it on ``(X, y)``. Returns ``self``."""
        self.baseline = self.baseline_factory()
        self.baseline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward the baseline's prediction. Requires a prior :meth:`fit`."""
        if self.baseline is None:
            raise RuntimeError("Residualizer.predict() called before fit()")
        return self.baseline.predict(X)

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return ``y - baseline.predict(X)``. Requires a prior :meth:`fit`."""
        return y - self.predict(X)
