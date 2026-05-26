"""Walk-forward backtest primitive — single-model variant.

The simple per-step walk-forward loop: keep a rolling buffer of
``(X, y)`` training rows, refit a model on cadence, predict each test row.
The compound residualizer / feature / regressor pattern lives in
:mod:`src.backtest.multi_stage`; this module is just the
single-stage-model primitive (and lets you opt-in to per-step
``RollingRobustScaler`` if you don't want the whole-series prescale path).

Used by simple models (Ridge, XGBoost, LightGBM, RandomForest, kNN) via
their ``fit_predict_<model>`` callables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm import tqdm

from src.features.transforms.scaling import RollingRobustScaler


class RollingBuffer:
    """Ring buffer for (X, y) pairs used in walk-forward backtests."""

    def __init__(self, window_size: int, n_features: int, n_targets: int = 1) -> None:
        self.window_size = window_size
        self.X = np.zeros((window_size, n_features), dtype=np.float64)
        self.y = np.zeros((window_size, n_targets), dtype=np.float64)
        self.pos = 0
        self.count = 0

    def add(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        self.X[self.pos] = x_new
        self.y[self.pos] = y_new
        self.pos = (self.pos + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)

    def get_view(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < self.window_size:
            return self.X[: self.count], self.y[: self.count]
        idx = np.roll(np.arange(self.window_size), -self.pos)
        return self.X[idx], self.y[idx]


def run_backtest(
    model_fn: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    refit_frequency: int = 1,
    use_scaling: bool = True,
) -> np.ndarray:
    """Walk-forward backtest with periodic refit.

    Parameters
    ----------
    model_fn : Callable[[], Any]
        Zero-argument factory returning a fresh model exposing ``.fit(X, y)``
        and ``.predict(X)``. Called once for the initial fit and again at
        every refit step.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Target vector of shape ``(n_samples,)``.
    train_win : int
        Rolling training window size, in samples.
    refit_frequency : int, default=1
        Cadence (in steps) at which the model is refit on the latest window.
        ``1`` refits every step; larger values amortize fit cost.
    use_scaling : bool, default=True
        If True, apply :class:`RollingRobustScaler` (median / IQR) to
        features using statistics computed only over the trailing
        ``train_win`` window. When the feature matrix has been pre-scaled
        whole-series by
        :func:`src.features.transforms.scaling.rolling_robust_scale` (the
        chunking-safe path), pass False.

    Returns
    -------
    np.ndarray
        Predictions of shape ``(n_samples - train_win,)``. Entry ``k``
        corresponds to the prediction for sample ``t = train_win + k``.

    Notes
    -----
    **Strict causality.** At step ``t`` the model is trained on the closed-
    open window ``[t - train_win : t]`` and predicts ``y[t]``. The feature
    row ``X[t]`` is scaled using statistics from ``[t - train_win : t]``
    (the scaler is updated *after* the prediction at step ``t`` is made),
    and ``y[t]`` is appended to the training buffer *after* prediction.
    No look-ahead is possible.

    **Chunking note.** With ``use_scaling=True`` the buffer's effective
    look-back is ``2 * train_win`` (the buffer stores already-scaled rows
    and the scaler looks back ``train_win``). Pre-scaling the whole series
    with :func:`rolling_robust_scale` and passing ``use_scaling=False``
    collapses that to a ``train_win`` halo.
    """
    n_samples, n_features = X.shape
    predictions = np.full(n_samples - train_win, np.nan)

    scaler_obj = RollingRobustScaler(train_win, n_features) if use_scaling else None
    buf = RollingBuffer(train_win, n_features, n_targets=1)

    X_init = X[:train_win].copy()
    y_init = y[:train_win].copy()

    if use_scaling:
        assert scaler_obj is not None
        scaler_obj.initialize(X_init)
        med, iqr = scaler_obj.get_scaler()
        X_scaled_init = (X_init - med) / iqr
    else:
        X_scaled_init = X_init

    for i in range(train_win):
        buf.add(X_scaled_init[i], y_init[i : i + 1])

    X_buf, y_buf = buf.get_view()
    model = model_fn()
    model.fit(X_buf, y_buf.ravel())

    for t in tqdm(range(train_win, n_samples), desc="backtest"):
        x_t_raw = X[t]

        if use_scaling:
            assert scaler_obj is not None
            med, iqr = scaler_obj.get_scaler()
            x_t_scaled = (x_t_raw - med) / iqr
        else:
            x_t_scaled = x_t_raw

        predictions[t - train_win] = model.predict(x_t_scaled.reshape(1, -1))[0]

        if use_scaling:
            assert scaler_obj is not None
            scaler_obj.update(x_t_raw)

        buf.add(x_t_scaled, y[t : t + 1])

        if (t - train_win + 1) % refit_frequency == 0:
            X_buf, y_buf = buf.get_view()
            model = model_fn()
            model.fit(X_buf, y_buf.ravel())

    return predictions
