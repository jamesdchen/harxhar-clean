"""Walk-forward backtest for residualizer + residual-window feature + regressor.

The pipeline this harness runs is:

* ``residualizer.fit(X_train, y_train)``   — refit every step (closed-form
  baselines are cheap).
* ``feature_fit(views)``                  — refit every ``refit_frequency``
  steps on sliding-window views of training residuals.
* ``regressor.fit(basis.phi_train, view_targets)`` — refit alongside the
  feature basis.

At each step the final prediction is::

    base_hat            = residualizer.predict(X[t])
    v_test              = residualizer.residuals(X[t-W:t], y[t-W:t])
    phi_test            = basis.embed(v_test)
    residual_correction = regressor.predict(phi_test)
    y_hat               = base_hat + residual_correction

Residualization is owned by :class:`src.features.residualizer.Residualizer`
(see that module). This harness is explicit about the pattern: a baseline
residualizes ``y``, a feature transforms residual windows, and a downstream
regressor predicts the leftover. Strict causality holds — at step ``t`` only
``X[:t]`` and ``y[:t]`` are ever read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from tqdm import tqdm

from src.features.transforms.residualizer import Residualizer


class _Basis(Protocol):
    """Structural interface for the feature-fit return value.

    ``phi_train`` is the training-side coordinates passed to the regressor's
    ``.fit``. ``embed(v)`` produces the coordinates for a single test view.

    :class:`src.features.spectral_embedding.SpectralBasis` satisfies this.
    """

    phi_train: np.ndarray

    def embed(self, v: np.ndarray) -> np.ndarray: ...


class MultiStageBacktest:
    """Walk-forward harness for the residualizer + feature + regressor pattern.

    Parameters
    ----------
    residualizer
        A :class:`~src.features.residualizer.Residualizer` instance. Refit
        every step. Owns the baseline regressor and exposes
        ``residuals(X, y) = y - baseline.predict(X)``.
    feature_fit
        Called every ``refit_frequency`` steps with the residuals' sliding
        windows; returns a "basis" object with ``.embed(v) -> np.ndarray``
        and ``.phi_train`` (training-side coordinates).
    regressor_factory
        Zero-arg factory returning a sklearn-style regressor used to predict
        residual corrections from embedded views.
    view_window
        Window length ``W`` for sliding views of residuals.
    refit_frequency
        How often (in steps) to rebuild the feature basis + regressor.
    """

    def __init__(
        self,
        residualizer: Residualizer,
        feature_fit: Callable[[np.ndarray], _Basis],
        regressor_factory: Callable[[], Any],
        view_window: int,
        refit_frequency: int,
    ) -> None:
        self.residualizer = residualizer
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

            # 1. Refit residualizer every step (assumed cheap).
            self.residualizer.fit(X_train, y_train)

            # 2. Refit feature basis + regressor on cadence.
            if i % self.refit_frequency == 0:
                residuals_train = self.residualizer.residuals(X_train, y_train)
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
            base_hat = float(self.residualizer.predict(X[t : t + 1])[0])
            if basis is None or regressor is None:
                predictions[i] = base_hat
                continue

            v_test = self.residualizer.residuals(X[t - W : t], y[t - W : t])
            phi_test = basis.embed(v_test)
            residual_correction = float(regressor.predict(phi_test[None, :])[0])
            predictions[i] = base_hat + residual_correction

        return predictions
