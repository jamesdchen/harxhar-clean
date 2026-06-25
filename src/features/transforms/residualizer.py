"""Residualizer — a baseline regressor wrapped as a feature transform.

A residualizer treats ``baseline.predict(X)`` as a feature transformation of
the target: ``residuals = y - baseline.predict(X)``. The baseline is any
sklearn-style regressor (``.fit(X, y)`` / ``.predict(X)`` — Ridge, OLS, a
tree, etc.); the residuals are the part of ``y`` the baseline didn't explain
and are typically the input to a downstream model.

The factory pattern (``baseline_factory: () -> baseline``) matches the
convention used by :class:`src.backtest.multi_stage.MultiStageBacktest` —
each call to :meth:`Residualizer.fit` instantiates a fresh baseline so
walk-forward refits are clean and the previous fit's state doesn't leak in.

:class:`IdentityResidualizer` is the no-op degenerate case used by simple
models (Ridge, XGBoost, kNN, etc.) that route through MultiStageBacktest
without subtracting a baseline.
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
    Single fit (test residuals from one model trained on the first window)::

        from sklearn.linear_model import Ridge
        from src.features.transforms.residualizer import Residualizer

        res = Residualizer(lambda: Ridge(alpha=1.0)).fit(X_train, y_train)
        residuals_train = res.residuals(X_train, y_train)
        v_test = res.residuals(X[t - W : t], y[t - W : t])

    Walk-forward (residual series over the whole OOS region, refitting the
    baseline every step or on a cadence)::

        res = Residualizer(lambda: Ridge(alpha=1.0))
        residuals_oos = res.walk_forward_residuals(X, y, train_win)
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

    def walk_forward_residuals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_win: int,
        refit_frequency: int = 1,
        progress: bool = False,
    ) -> np.ndarray:
        """Walk-forward residuals over ``X[train_win:]``.

        At each test step ``t`` in ``[train_win, len(X))``, refit the baseline
        on the trailing ``train_win`` rows ``[t - train_win, t)`` and emit
        ``y[t] - baseline.predict(X[t])``. The baseline refits every
        ``refit_frequency`` steps (1 = every step). Same causal structure as
        :class:`src.backtest.multi_stage.MultiStageBacktest`.

        Parameters
        ----------
        X, y
            Feature matrix and target. Must be aligned.
        train_win
            Rolling training-window size in samples.
        refit_frequency
            Cadence (in steps) at which the baseline is refit. 1 = every step
            (true walk-forward); larger values amortize fit cost at the price
            of slightly stale baselines between refits.
        progress
            If True, show a tqdm progress bar.

        Returns
        -------
        np.ndarray
            Shape ``(len(X) - train_win,)``. Entry ``k`` is the residual for
            sample ``t = train_win + k``.
        """
        n = len(X)
        if train_win >= n:
            raise ValueError(f"train_win ({train_win}) >= series length ({n})")
        if len(X) != len(y):
            raise ValueError(f"len(X)={len(X)} != len(y)={len(y)}")

        n_test = n - train_win
        out = np.empty(n_test, dtype=np.float64)

        iterator: Any = range(n_test)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="walk_forward_residuals")

        for i in iterator:
            t = train_win + i
            if i % refit_frequency == 0:
                self.fit(X[t - train_win : t], y[t - train_win : t])
            out[i] = y[t] - self.predict(X[t : t + 1])[0]
        return out


class IdentityResidualizer:
    """No-op residualizer: ``predict(X) == 0``, ``residuals(X, y) == y``.

    The degenerate case used by :class:`~src.backtest.multi_stage.MultiStageBacktest`
    when a model has no baseline to subtract — i.e., the regressor is meant to
    model ``y`` directly from ``X``. With this residualizer, every "simple"
    walk-forward backtest (Ridge, XGBoost, kNN, …) routes through the same
    harness as the spectral_knn composition, just with the residualizer + feature
    stages turned off.
    """

    # Duck-typed marker read by MultiStageBacktest: a passthrough residualizer
    # (residuals == y, predict == 0) means the regressor models a *static*
    # target, so a linear regressor can be driven by rank-1 sliding-window
    # updates instead of refit-from-scratch (see RollingLeastSquares).
    is_passthrough = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IdentityResidualizer":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.float64)

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y

    def walk_forward_residuals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_win: int,
        refit_frequency: int = 1,
        progress: bool = False,
    ) -> np.ndarray:
        """Walk-forward (degenerate): residuals = y[train_win:]."""
        del refit_frequency, progress  # unused; the residual stream is just y
        return y[train_win:].astype(np.float64).copy()
