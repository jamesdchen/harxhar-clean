"""Walk-forward backtest for multi-stage residual pipelines.

A pipeline of (baseline -> residual feature transform -> residual regressor)
doesn't fit :func:`src.features.scaling.run_backtest`'s single-``model_fn``
shape: the feature transform consumes sliding windows of **residuals**
(which depend on past ``y``, not just past ``X``), and the baseline and
feature basis can have different refit cadences.

:class:`MultiStageBacktest` is the canonical scaffold for that pattern.
It owns the walk-forward loop and the refit bookkeeping; callers supply
the three pieces via factories.

The motivating composition (see ``run.py::_spectral_knn_pipeline``)::

    MultiStageBacktest(
        baseline_factory   = lambda: Ridge(alpha=...),
        feature_fit        = lambda views: build_embedding(views, d=..., k_graph=...),
        regressor_factory  = lambda: KNeighborsRegressor(n_neighbors=..., weights=gaussian_weights),
        view_window        = 960,
        refit_frequency    = 240,
    )

reads as exactly "Ridge baseline + spectral embedding of residuals + kNN
regression."
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from tqdm import tqdm


class _Basis(Protocol):
    """Structural interface for the feature-fit return value.

    ``phi_train`` is the training-side coordinates passed to the regressor's
    ``fit``. ``embed(v)`` produces the coordinates for a single test view.

    :class:`src.features.spectral_embedding.SpectralBasis` satisfies this.
    """

    phi_train: np.ndarray

    def embed(self, v: np.ndarray) -> np.ndarray: ...


class MultiStageBacktest:
    """Walk-forward backtest of baseline + residual-feature + residual-regressor.

    Each step at index ``t`` (in ``[train_win, n_samples)``):

    1. Refit the baseline on the trailing ``train_win`` rows. (Every step —
       baselines are assumed cheap, like closed-form Ridge.)
    2. On ``refit_frequency`` cadence: compute residuals over the training
       window, form sliding windows of length ``view_window``, fit the
       feature basis on those views, fit the residual regressor on
       ``(basis.phi_train, view_targets)``.
    3. Predict the test row:

       * ``baseline_hat = baseline.predict(X[t])``
       * ``v_test       = y[t-W:t] - baseline.predict(X[t-W:t])``
       * ``phi_test     = basis.embed(v_test)``
       * ``residual     = regressor.predict(phi_test)``
       * ``y_hat        = baseline_hat + residual``

    When the training window is too short for ``W + 1`` residuals, the basis
    and regressor are skipped and ``baseline_hat`` alone is the prediction.

    Strict causality: at step ``t`` only ``X[:t]`` and ``y[:t]`` are used.
    """

    def __init__(
        self,
        baseline_factory: Callable[[], Any],
        feature_fit: Callable[[np.ndarray], _Basis],
        regressor_factory: Callable[[], Any],
        view_window: int,
        refit_frequency: int,
    ) -> None:
        self.baseline_factory = baseline_factory
        self.feature_fit = feature_fit
        self.regressor_factory = regressor_factory
        self.view_window = int(view_window)
        self.refit_frequency = int(refit_frequency)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_win: int,
        *,
        desc: str = "multi_stage",
    ) -> np.ndarray:
        """Run walk-forward.

        Returns predictions of shape ``(n_samples - train_win,)``; entry ``k``
        is the prediction for sample ``t = train_win + k``.
        """
        n_samples = X.shape[0]
        n_test = n_samples - train_win
        if n_test <= 0:
            raise ValueError(f"train_win ({train_win}) >= series length ({n_samples})")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X.shape[0]={X.shape[0]} != y.shape[0]={y.shape[0]}")

        W = self.view_window
        predictions = np.empty(n_test, dtype=np.float64)
        basis: _Basis | None = None
        regressor: Any = None

        for i in tqdm(range(n_test), desc=desc):
            t = train_win + i
            X_train = X[t - train_win : t]
            y_train = y[t - train_win : t]

            # 1. Refit baseline every step (assumed cheap).
            baseline = self.baseline_factory()
            baseline.fit(X_train, y_train)

            # 2. Refit feature basis + regressor on cadence.
            if i % self.refit_frequency == 0:
                residuals_train = y_train - baseline.predict(X_train)
                if len(residuals_train) >= W + 1:
                    view_idx = np.arange(W, len(residuals_train))
                    views = np.stack([residuals_train[j - W : j] for j in view_idx])
                    view_targets = residuals_train[view_idx]
                    basis = self.feature_fit(views)
                    regressor = self.regressor_factory()
                    regressor.fit(basis.phi_train, view_targets)
                else:
                    basis = None
                    regressor = None

            # 3. Predict.
            baseline_hat = float(baseline.predict(X[t : t + 1])[0])
            if basis is None or regressor is None:
                predictions[i] = baseline_hat
                continue

            v_test = y[t - W : t] - baseline.predict(X[t - W : t])
            phi_test = basis.embed(v_test)
            residual_correction = float(regressor.predict(phi_test[None, :])[0])
            predictions[i] = baseline_hat + residual_correction

        return predictions
