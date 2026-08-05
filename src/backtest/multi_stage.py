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
    feature_exog
        Optional, default False. When True the feature stage is *exog-aware*:
        ``feature_fit(views, exog_rows)`` also receives the feature rows
        aligned with the views (``X_train[view_idx]``), and the returned
        basis is called as ``basis.embed(v_test, x_test)``. Off by default,
        so every existing configuration takes the byte-identical 1-D path.
    """

    def __init__(
        self,
        residualizer: _Residualizer,
        regressor_factory: Callable[[], Any],
        refit_frequency: int,
        feature_fit: Callable[..., _Basis] | None = None,
        view_window: int = 0,
        feature_exog: bool = False,
        feature_exog_window: bool = False,
        correction_weight: float = 1.0,
        target_smoother: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        stack_window: int = 0,
    ) -> None:
        self.residualizer = residualizer
        self.regressor_factory = regressor_factory
        self.refit_frequency = int(refit_frequency)
        self.feature_fit = feature_fit
        self.view_window = int(view_window)
        self.feature_exog = bool(feature_exog)
        # multi-channel windows imply the exog-aware stage
        self.feature_exog_window = bool(feature_exog_window)
        # omega: shrink the regressor's CORRECTION toward the residualizer's
        # own forecast. 1.0 = take the correction whole (byte-identical to
        # every prior run). Only meaningful with a real residualizer -- under
        # IdentityResidualizer base_hat is 0, so omega would just rescale the
        # whole forecast and bias it toward zero.
        self.correction_weight = float(correction_weight)
        # target_smoother: optional spectral filtering of the LABEL FIELD.
        # Called as smoother(phi_train, view_targets) -> smoothed targets at
        # each feature refit, before the regressor fits. This is the dual of
        # projecting the coordinates: the graph low-pass lands on the noisy
        # targets (variance reduction of the correction) and by construction
        # cannot discard predictive directions of the view space — the
        # measured failure mode of the coordinate-side spectral embedding.
        # None (default) is byte-identical to every prior configuration.
        self.target_smoother = target_smoother
        # stack_window: LEARNED combination of base and correction instead of
        # the fixed omega sum. Keeps the anchor whole (w on the base is pinned
        # at 1 — exact nesting at w2=0) and regresses the REALIZED residual on
        # the correction over the trailing stack_window scored bars:
        #     pred = base + w0 + w2 * correction,  (w0, w2) by trailing OLS.
        # Orthogonalizes the local/global split at the combination level and
        # restores the can't-lose-to-the-anchor guarantee that omega=1
        # forfeits. 0 (default) = off, byte-identical to every prior run;
        # until the window fills, the omega path is used unchanged. Causal:
        # the pair for bar t enters the history only after t is predicted.
        self.stack_window = int(stack_window)
        if self.feature_exog_window:
            self.feature_exog = True

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

        # Incremental fast path: a linear regressor exposing the rank-1 rolling
        # protocol (init_window + roll), a passthrough residualizer (residuals
        # == y, base == 0) and no feature stage. Gated tightly so every other
        # configuration — kNN/trees, stacking, spectral — falls through to the
        # fresh-refit loop below unchanged.
        if self.feature_fit is None and getattr(
            self.residualizer, "is_passthrough", False
        ):
            probe = self.regressor_factory()
            if hasattr(probe, "roll") and hasattr(probe, "init_window"):
                return self._run_incremental(X, y, train_win, probe, desc=desc)

        W = self.view_window
        predictions = np.empty(n_test, dtype=np.float64)
        basis: _Basis | None = None
        regressor: Any = None
        stk_corr: list[float] = []   # trailing (correction, realized residual)
        stk_resid: list[float] = []  # pairs for the learned combination

        # A rolling residualizer (duck-typed marker) slides its sufficient
        # statistics per step instead of paying a fresh O(W p^2) fit per bar;
        # exact rebuilds happen on its own reinit cadence to bound float
        # drift. Every non-rolling residualizer keeps the byte-identical
        # fresh-refit path.
        rolling = bool(getattr(self.residualizer, "is_rolling", False))
        reinit = int(getattr(self.residualizer, "reinit_every", 0)) if rolling else 0

        for i in tqdm(range(n_test), desc=desc):
            t = train_win + i
            X_train = X[t - train_win : t]
            y_train = y[t - train_win : t]

            # 1. Refit residualizer every step (cheap — closed-form or identity).
            if not rolling or i == 0 or (reinit and i % reinit == 0):
                self.residualizer.fit(X_train, y_train)
            else:
                self.residualizer.roll(
                    X[t - 1], y[t - 1], X[t - 1 - train_win], y[t - 1 - train_win]
                )

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
                    # Exog-aware feature stage: the aligned feature rows ride
                    # alongside the residual windows (X_train[j] is the row of
                    # the bar the view predicts). Default path is unchanged.
                    # feature_exog_window instead hands the WHOLE training
                    # window plus the view index, so the feature stage can cut
                    # multi-CHANNEL windows (exog trajectories, not just a
                    # point-in-time snapshot). Only viable at small W — the
                    # channel tensor is O(n * W * channels).
                    if self.feature_exog_window:
                        basis = self.feature_fit(views, X_train, view_idx)
                    elif self.feature_exog:
                        basis = self.feature_fit(views, X_train[view_idx])
                    else:
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
                if self.feature_exog_window:
                    phi_test = basis.embed(v_test, X[t - W : t])
                elif self.feature_exog:
                    phi_test = basis.embed(v_test, X[t])
                else:
                    phi_test = basis.embed(v_test)
                residual_correction = float(regressor.predict(phi_test[None, :])[0])

            if self.stack_window > 0 and len(stk_corr) >= self.stack_window:
                A = np.stack([np.ones(self.stack_window),
                              np.array(stk_corr[-self.stack_window:])], axis=1)
                wfit, *_ = np.linalg.lstsq(
                    A, np.array(stk_resid[-self.stack_window:]), rcond=None
                )
                predictions[i] = base_hat + wfit[0] + wfit[1] * residual_correction
            else:
                predictions[i] = base_hat + self.correction_weight * residual_correction
            if self.stack_window > 0:
                # append AFTER predicting: bar t's realized pair serves later bars
                stk_corr.append(residual_correction)
                stk_resid.append(float(y[t]) - base_hat)
                if len(stk_corr) > 4 * self.stack_window:  # bound memory
                    del stk_corr[: 2 * self.stack_window]
                    del stk_resid[: 2 * self.stack_window]

        return predictions

    def _run_incremental(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_win: int,
        regressor: Any,
        *,
        desc: str,
    ) -> np.ndarray:
        """Rank-1 sliding-window fast path (see :class:`RollingLeastSquares`).

        Predictions are identical (to numerical precision) to refitting the
        regressor on each trailing window, at ``O(p²)``/bar instead of
        ``O(W·p²)``/bar. Only reached when ``run`` has confirmed a passthrough
        residualizer (so residuals ``== y`` and the base prediction is ``0``),
        no feature stage, and a regressor exposing ``init_window`` + ``roll``.

        The window slides exactly as the fresh-refit loop's
        ``X[t - train_win : t]``: from step ``i-1`` to ``i`` row ``t-1`` enters
        and row ``t-1-train_win`` leaves. ``refit_frequency`` controls how often
        the coefficients are re-solved (stats still roll every step, so a solve
        always reflects the current window — matching the fresh-refit path for
        any cadence). ``reinit_every`` (read off the regressor) periodically
        rebuilds the stats exactly to bound float drift.
        """
        n_test = X.shape[0] - train_win
        predictions = np.empty(n_test, dtype=np.float64)
        reinit_every = int(getattr(regressor, "reinit_every", 0))

        regressor.init_window(X[0:train_win], y[0:train_win])
        regressor.solve()
        predictions[0] = regressor.predict_one(X[train_win])
        for i in tqdm(range(1, n_test), desc=desc):
            t = train_win + i
            if reinit_every and i % reinit_every == 0:
                regressor.init_window(X[t - train_win : t], y[t - train_win : t])
            else:
                regressor.roll(
                    X[t - 1], y[t - 1], X[t - 1 - train_win], y[t - 1 - train_win]
                )
            if i % self.refit_frequency == 0:
                regressor.solve()
            predictions[i] = regressor.predict_one(X[t])
        return predictions
