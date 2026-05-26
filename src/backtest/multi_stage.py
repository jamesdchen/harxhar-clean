"""Walk-forward backtest with three optional stages.

``MultiStageBacktest`` is the default backtest in this repo. It runs the
walk-forward loop and lets callers plug in up to three stages:

1. **residualizer** — fits a baseline on ``(X_train, y_train)`` every step
   and subtracts its predictions from ``y``. Use
   :class:`~src.features.transforms.residualizer.Residualizer` for an
   actual baseline (Ridge, OLS, …), or
   :class:`~src.features.transforms.residualizer.IdentityResidualizer` for
   the no-op degenerate case (model ``y`` directly).
2. **feature** — optional. When provided, called every ``refit_frequency``
   steps with sliding views of residuals; returns a "basis" with
   ``.embed(v) → np.ndarray`` and ``.phi_train``. The regressor then trains
   on ``(basis.phi_train, view_targets)`` and predicts from
   ``basis.embed(v_test)``. When ``None``, the regressor trains on
   ``(X_train, residuals_train)`` directly.
3. **regressor** — required. Sklearn-style ``.fit(X, y) / .predict(X)``.

Three useful configurations:

* **Plain model** — Identity + None + Ridge / XGBoost / kNN. Equivalent
  to the legacy single-stage walk-forward; the regressor models ``y``
  from ``X``.
* **Stacking** — Residualizer(Ridge) + None + kNN. kNN learns the residual
  pattern of Ridge.
* **Residual-window feature** (spectral_knn) — Residualizer(Ridge) +
  build_embedding + kNN. The residual trajectory's spectral coordinates
  feed the kNN.

Strict causality holds in every configuration: at step ``t`` only
``X[:t]`` and ``y[:t]`` are read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from tqdm import tqdm


class _Residualizer(Protocol):
    """Structural interface satisfied by Residualizer and IdentityResidualizer."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Residualizer": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class _Basis(Protocol):
    """Structural interface for the feature-fit return value.

    :class:`src.features.extractors.spectral_embedding.SpectralBasis` satisfies this.
    """

    phi_train: np.ndarray

    def embed(self, v: np.ndarray) -> np.ndarray: ...


class MultiStageBacktest:
    """Walk-forward harness for the residualizer + optional-feature + regressor pattern.

    Parameters
    ----------
    residualizer
        A residualizer instance (see protocol above). Use
        :class:`IdentityResidualizer` for the no-baseline-subtraction case.
        Refit every step (baselines are expected to be cheap, e.g. closed-form
        Ridge; Identity is a no-op).
    regressor_factory
        Zero-arg factory returning a fresh sklearn-style regressor.
    refit_frequency
        Cadence (in steps) at which the regressor (and feature basis, if any)
        is rebuilt on the latest training window.
    feature_fit
        Optional. If provided, called with the sliding views of training
        residuals to produce a basis (with ``.embed`` and ``.phi_train``).
        If ``None``, the regressor trains directly on
        ``(X_train, residuals_train)`` and predicts from ``X_test``.
    view_window
        Sliding window length ``W`` for residual views; only consulted when
        ``feature_fit`` is provided.
    """

    def __init__(
        self,
        residualizer: _Residualizer,
        regressor_factory: Callable[[], Any],
        refit_frequency: int,
        feature_fit: Callable[[np.ndarray], _Basis] | None = None,
        view_window: int = 0,
    ) -> None:
        self.residualizer = residualizer
        self.regressor_factory = regressor_factory
        self.refit_frequency = int(refit_frequency)
        self.feature_fit = feature_fit
        self.view_window = int(view_window)

        if feature_fit is not None and view_window <= 0:
            raise ValueError("feature_fit requires view_window > 0")

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

            # 1. Refit residualizer every step (cheap — closed-form or identity).
            self.residualizer.fit(X_train, y_train)

            # 2. Refit regressor (and feature basis) on cadence.
            if i % self.refit_frequency == 0:
                residuals_train = self.residualizer.residuals(X_train, y_train)
                regressor = self.regressor_factory()

                if self.feature_fit is None:
                    regressor.fit(X_train, residuals_train)
                    basis = None
                elif len(residuals_train) >= W + 1:
                    view_idx = np.arange(W, len(residuals_train))
                    views = np.stack([residuals_train[j - W : j] for j in view_idx])
                    view_targets = residuals_train[view_idx]
                    basis = self.feature_fit(views)
                    regressor.fit(basis.phi_train, view_targets)
                else:
                    # Training window too short to form views — skip the regressor;
                    # predictions fall back to baseline alone this refit.
                    basis = None
                    regressor = None

            # 3. Predict.
            base_hat = float(self.residualizer.predict(X[t : t + 1])[0])
            if regressor is None:
                predictions[i] = base_hat
                continue

            if self.feature_fit is None:
                residual_correction = float(regressor.predict(X[t : t + 1])[0])
            else:
                assert basis is not None  # paired with regressor; both set together
                v_test = self.residualizer.residuals(X[t - W : t], y[t - W : t])
                phi_test = basis.embed(v_test)
                residual_correction = float(regressor.predict(phi_test[None, :])[0])

            predictions[i] = base_hat + residual_correction

        return predictions
