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
from numba import njit, prange

# Relative floor on the rolling IQR, as a fraction of that column's own typical
# IQR so far (a causal running geometric mean of the positive IQRs observed up
# to and including the current window). The reference is a running MAXIMUM,
# not a mean: a mean or geometric mean is dragged down by the very degenerate
# windows the floor exists to catch, which collapses the floor alongside the
# IQR and -- measured on a synthetic degenerate column -- lifts the result just
# above the 1e-12 fallback that had been catching it, making the output worse
# (|max| 293 -> 117,114). A running maximum cannot be dragged down. Its failure
# mode is the safe one: an unusually volatile early window sets a high floor
# and over-attenuates later rows, where under-flooring detonates them.
#
# The historical guard was ``iqr = iq if iq >= 1e-12 else 1.0``: an ABSOLUTE
# threshold, which catches an exactly-degenerate window and misses the case
# that actually bites. An IQR of 1e-11 on a column whose values are of order
# 1e-6 sails past it and divides the incoming point by ~nothing. That is the
# same defect ``target.DIURNAL_STD_FLOOR_FRAC`` was introduced to fix on the
# diurnal std, where the note reads: "The old replace(0, 1.0) only caught an
# *exactly* zero std and assumed an O(1) series, so a tiny-but-nonzero std on a
# large-scale signed feature blew the adjusted value up to ~1e13."
#
# The measured consequence here: 29 of 41 exogenous channels reach past 20
# rolling IQRs somewhere in the panel and the worst reached 2,260, which is not
# a market move but a division by a collapsed scale. Set to 0.0 to restore the
# legacy absolute-only guard bit-for-bit.
IQR_FLOOR_FRAC: float = 0.01


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
    iqr_floor_frac: float = IQR_FLOOR_FRAC,
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
    out = np.empty_like(X)
    use_ref = ref_iqr is not None
    ref = (
        np.asarray(ref_iqr, dtype=np.float64) if use_ref else np.zeros((1, X.shape[1]))
    )
    _rolling_scale_cols(X, train_win, ref, use_ref, out, iqr_floor_frac)
    if fixed_cols is not None:
        out[:, fixed_cols] = X[:, fixed_cols]
    return out


@njit(cache=True, parallel=True)
def _rolling_scale_cols(
    X: np.ndarray, W: int, ref_iqr: np.ndarray, use_ref: bool, out: np.ndarray,
    floor_frac: float = 0.0
) -> None:
    """Whole-series replay of the sorted-window scaler, one column per thread.

    Bit-identical to replaying :class:`RollingRobustScaler` row by row (same
    quantile interpolation, same ``1e-12 -> 1.0`` IQR guard, same ``ref_iqr``
    floor) but fused into a single kernel: columns are independent, so the
    replay parallelizes over features (``prange``) and the per-row Python
    dispatch of the old loop disappears (~13x on a 500-feature matrix).
    """
    n, p = X.shape
    idx_25 = (W - 1) * 0.25
    idx_50 = (W - 1) * 0.50
    idx_75 = (W - 1) * 0.75
    i25, r25 = int(idx_25), idx_25 - int(idx_25)
    i50, r50 = int(idx_50), idx_50 - int(idx_50)
    i75, r75 = int(idx_75), idx_75 - int(idx_75)
    for j in prange(p):
        s = np.sort(X[:W, j].copy())  # sorted window
        b = X[:W, j].copy()  # ring buffer of raw values
        pos = 0
        q25 = s[i25] * (1.0 - r25) + s[min(i25 + 1, W - 1)] * r25
        med = s[i50] * (1.0 - r50) + s[min(i50 + 1, W - 1)] * r50
        q75 = s[i75] * (1.0 - r75) + s[min(i75 + 1, W - 1)] * r75
        iq = q75 - q25
        # causal running MAXIMUM of the IQRs seen so far, for the relative floor
        iq_max = iq if iq > 0.0 else 0.0
        iqr = iq
        if floor_frac > 0.0 and iq_max > 0.0:
            fl = floor_frac * iq_max
            if iqr < fl:
                iqr = fl
        if iqr < 1e-12:
            iqr = 1.0
        for t in range(W):  # rows [0, W): the initial window's stats
            eff = iqr
            if use_ref and ref_iqr[t, j] > eff:
                eff = ref_iqr[t, j]
            out[t, j] = (X[t, j] - med) / eff
        for t in range(W, n):
            q25 = s[i25] * (1.0 - r25) + s[min(i25 + 1, W - 1)] * r25
            med = s[i50] * (1.0 - r50) + s[min(i50 + 1, W - 1)] * r50
            q75 = s[i75] * (1.0 - r75) + s[min(i75 + 1, W - 1)] * r75
            iq = q75 - q25
            if iq > iq_max:
                iq_max = iq
            iqr = iq
            if floor_frac > 0.0 and iq_max > 0.0:
                fl = floor_frac * iq_max
                if iqr < fl:
                    iqr = fl
            if iqr < 1e-12:
                iqr = 1.0
            eff = iqr
            if use_ref and ref_iqr[t, j] > eff:
                eff = ref_iqr[t, j]
            out[t, j] = (X[t, j] - med) / eff
            v_old = b[pos]
            v_new = X[t, j]
            b[pos] = v_new
            pos = pos + 1
            if pos == W:
                pos = 0
            io = np.searchsorted(s, v_old)
            inw = np.searchsorted(s, v_new)
            if io < inw:
                inw -= 1
                for k in range(io, inw):
                    s[k] = s[k + 1]
            elif io > inw:
                for k in range(io, inw, -1):
                    s[k] = s[k - 1]
            s[inw] = v_new
