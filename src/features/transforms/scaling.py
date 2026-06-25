"""Rolling robust scaling — modifies feature-matrix values (X-side).

Numba-accelerated sorted-matrix quantile tracking for O(W) median/IQR
scaling. Two entry points:

* :class:`RollingRobustScaler` — online API; an in-loop scaler that gets
  initialized + updated step by step.
* :func:`rolling_robust_scale` — convenience wrapper that scales a whole
  feature matrix at once (replays the scaler across the series). Used by
  the executor's ``prescale=True`` path; the resulting whole-series
  scaled matrix is what :class:`~src.backtest.multi_stage.MultiStageBacktest`
  consumes.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _update_sorted_matrix(
    sorted_mat: np.ndarray, x_old: np.ndarray, x_new: np.ndarray
) -> None:
    """Replace *x_old* with *x_new* in each feature's sorted window."""
    n_features, w = sorted_mat.shape
    for i in range(n_features):
        v_old = x_old[i]
        v_new = x_new[i]
        idx_old = np.searchsorted(sorted_mat[i], v_old)
        idx_new = np.searchsorted(sorted_mat[i], v_new)
        if idx_old < idx_new:
            idx_new -= 1
            for j in range(idx_old, idx_new):
                sorted_mat[i, j] = sorted_mat[i, j + 1]
        elif idx_old > idx_new:
            for j in range(idx_old, idx_new, -1):
                sorted_mat[i, j] = sorted_mat[i, j - 1]
        sorted_mat[i, idx_new] = v_new


@njit(cache=True)
def _get_robust_stats(
    sorted_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute median and IQR from pre-sorted rolling window."""
    n_features, w = sorted_mat.shape
    median = np.empty(n_features, dtype=np.float64)
    iqr = np.empty(n_features, dtype=np.float64)
    idx_25 = (w - 1) * 0.25
    idx_50 = (w - 1) * 0.50
    idx_75 = (w - 1) * 0.75
    i25_floor, rem_25 = int(idx_25), idx_25 - int(idx_25)
    i50_floor, rem_50 = int(idx_50), idx_50 - int(idx_50)
    i75_floor, rem_75 = int(idx_75), idx_75 - int(idx_75)
    for i in range(n_features):
        q25 = (
            sorted_mat[i, i25_floor] * (1.0 - rem_25)
            + sorted_mat[i, min(i25_floor + 1, w - 1)] * rem_25
        )
        med = (
            sorted_mat[i, i50_floor] * (1.0 - rem_50)
            + sorted_mat[i, min(i50_floor + 1, w - 1)] * rem_50
        )
        q75 = (
            sorted_mat[i, i75_floor] * (1.0 - rem_75)
            + sorted_mat[i, min(i75_floor + 1, w - 1)] * rem_75
        )
        median[i] = med
        iq = q75 - q25
        iqr[i] = iq if iq >= 1e-12 else 1.0
    return median, iqr


class RollingRobustScaler:
    """Online robust scaler backed by sorted-matrix quantile tracking.

    Maintains a rolling window of observations and a parallel sorted matrix
    per feature for O(W) median/IQR computation. Supports two usage styles:

    1. **get_scaler** -- return (median, iqr) for manual scaling.
    2. **transform_single / transform_buffer** -- scale directly.
    """

    buffer: np.ndarray | None
    sorted_mat: np.ndarray | None

    def __init__(self, window_size: int, n_features: int | None = None) -> None:
        self.window_size = window_size
        if n_features is not None:
            self.buffer = np.zeros((window_size, n_features), dtype=np.float64)
            self.sorted_mat = np.zeros((n_features, window_size), dtype=np.float64)
        else:
            self.buffer = None
            self.sorted_mat = None
        self.pos: int = 0

    def initialize(self, data_block: np.ndarray) -> None:
        """Fill buffers from *data_block* ``(window_size, n_features)``."""
        w = self.window_size
        n_features = data_block.shape[1]
        if self.buffer is None:
            self.buffer = np.empty((w, n_features), dtype=np.float64)
            self.sorted_mat = np.empty((n_features, w), dtype=np.float64)
        self.buffer[:] = data_block[:w]
        for i in range(n_features):
            self.sorted_mat[i] = np.sort(data_block[:w, i])  # type: ignore[index]
        self.pos = 0

    def update(self, x_new: np.ndarray) -> None:
        """Slide window: replace oldest row with *x_new*."""
        assert self.buffer is not None and self.sorted_mat is not None
        x_old = self.buffer[self.pos].copy()
        self.buffer[self.pos] = x_new
        _update_sorted_matrix(self.sorted_mat, x_old, x_new)
        self.pos = (self.pos + 1) % self.window_size

    def get_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(median, iqr)`` arrays from the current sorted buffer."""
        assert self.sorted_mat is not None
        return _get_robust_stats(self.sorted_mat)

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """Scale a single observation using current median/IQR."""
        median, iqr = self.get_scaler()
        return (x - median) / iqr

    def transform_buffer(self) -> np.ndarray:
        """Scale the entire current buffer using current median/IQR."""
        assert self.buffer is not None
        median, iqr = self.get_scaler()
        return (self.buffer - median) / iqr


def rolling_robust_scale(
    X: np.ndarray,
    train_win: int,
    ref_iqr: np.ndarray | None = None,
    fixed_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Per-row rolling robust scaling, computed whole-series.

    Rows ``[0, train_win)`` are rescaled by the initial window's median /
    IQR; each row ``t >= train_win`` by the trailing ``[t - train_win : t)``
    window. Replays :class:`RollingRobustScaler` over the whole series.

    Doing this whole-series — rather than refitting the scaler step-by-step
    inside the backtest loop — keeps the scaler's look-back out of the
    per-chunk backtest. A chunked walk-forward over the pre-scaled matrix is
    then fungible with just a ``train_win`` halo (rather than ``2 * train_win``
    if the scaler refit happened inside the loop).

    Parameters
    ----------
    ref_iqr : np.ndarray, optional
        ``(n_samples, n_features)`` per-row lower bound on the scaling IQR:
        ``iqr_eff = max(local_iqr, ref_iqr[t])``. The trailing-window IQR is a
        *local* scale; for a lagged/averaged feature whose window can go
        regime-degenerate (an imputed exog whose history is a flat fill), that
        local IQR collapses toward zero and divides the incoming — different
        regime — point by ~0, detonating the forecast. Flooring at a causal,
        real-values reference scale (see ``executor._build_scale_guards``)
        normalizes the point against the distribution it actually belongs to.
        Zero columns impose no floor; ``None`` restores legacy behaviour, so
        non-imputed runs are bit-for-bit unchanged.
    fixed_cols : np.ndarray, optional
        ``(n_features,)`` bool mask of columns to pass through *unscaled*
        (their scale is known a priori — e.g. the availability indicators
        ``*_avail_ma_*`` ∈ [0, 1]; estimating a data-driven IQR for a
        deterministic structural ramp is a category error). ``None`` scales
        every column.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = len(X)
    if train_win > n_samples:
        raise ValueError(f"train_win ({train_win}) exceeds series length ({n_samples})")
    scaler = RollingRobustScaler(train_win, X.shape[1])
    scaler.initialize(X[:train_win])
    out = np.empty_like(X)
    med, iqr = scaler.get_scaler()
    if ref_iqr is None:
        out[:train_win] = (X[:train_win] - med) / iqr
    else:
        out[:train_win] = (X[:train_win] - med) / np.maximum(iqr, ref_iqr[:train_win])
    for t in range(train_win, n_samples):
        med, iqr = scaler.get_scaler()
        if ref_iqr is not None:
            iqr = np.maximum(iqr, ref_iqr[t])
        out[t] = (X[t] - med) / iqr
        scaler.update(X[t])
    if fixed_cols is not None:
        out[:, fixed_cols] = X[:, fixed_cols]
    return out
