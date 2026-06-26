"""Causal rolling rank-Gauss (van der Waerden) feature normalization.

For each feature and row ``t``, map ``x[t]`` to its plotting-position percentile
within the trailing ``window`` rows ``[t-window+1, t]`` (causal — no future data),
then push it through the inverse normal CDF so the feature's marginal is ~``N(0,1)``
in *every* window. Unlike affine ``rolling_robust_scale`` (which pins only location
and scale, leaving skew/kurtosis to drift), this pins the *whole* marginal — it is
invariant to location, scale **and shape**, so it neutralizes the non-stationary
*shape* drift (regime-dependent skew/fat-tails) that an affine scaler cannot, and it
makes the linear model's coefficients transfer across regimes.

Two structural properties worth noting:

* **Division-free.** No IQR / std / mean ever sits in a denominator, so the whole
  divide-by-a-transiently-degenerate-scale failure class (the Part-3 bugs) simply
  cannot occur here.
* **Orthogonal to, not a replacement for, diurnal adjustment.** Diurnal adjustment
  is *conditional* on time-of-slot (removes structured seasonality); rank-Gauss is a
  *marginal*, pooled-over-slots reshape. Apply diurnal adjustment FIRST, then rank
  the de-seasonalised residual — ranking the raw feature would bake the diurnal
  pattern into the ranks. Rank-Gauss subsumes the *marginal* layers (semantic
  root/log, winsorize, robust-scale) but NOT the conditional diurnal step.

For trees this is a no-op (monotone per-feature → identical splits); it is meant for
the linear / Ridge path. Binary availability / occurrence indicators are passed
through raw via ``skip_cols`` (ranking a two-value column is meaningless).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri  # inverse standard-normal CDF (probit)


def rolling_rank_gauss(
    X: np.ndarray,
    window: int,
    skip_cols: list[int] | set[int] | None = None,
    min_periods: int | None = None,
) -> np.ndarray:
    """Causal rolling rank-Gauss normalize each column of ``X``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix (already diurnally adjusted upstream).
    window : int
        Trailing window length for the empirical CDF (ties to the model's
        ``train_win`` by convention, so the scaler look-back matches the fit
        look-back).
    skip_cols : indices of columns to pass through unchanged (indicators).
    min_periods : minimum valid obs before emitting a finite value; rows with
        fewer get 0.0 (neutral, like a centered feature). Defaults to
        ``max(20, window // 20)``.

    Returns
    -------
    np.ndarray, shape (n, p), float64.

    Causality: the window ``[t-window+1, t]`` contains no future rows. ``x[t]`` is
    included in its own ranking set (standard for rank-normalization); this uses no
    information from after ``t``. The plotting position ``(rank - 0.5)/count`` keeps
    the percentile in the open interval ``(0, 1)`` so the probit never returns ±inf.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    mp = min_periods if min_periods is not None else max(20, window // 20)
    skip = set(skip_cols or [])
    out = np.empty_like(X)
    for j in range(p):
        if j in skip:
            out[:, j] = X[:, j]
            continue
        s = pd.Series(X[:, j])
        rank = s.rolling(window, min_periods=mp).rank(method="average")
        count = s.rolling(window, min_periods=mp).count()
        u = ((rank - 0.5) / count).clip(1e-6, 1.0 - 1e-6)
        z = ndtri(u.to_numpy())
        z[~np.isfinite(z)] = 0.0  # warm-up rows (count < mp) → neutral
        out[:, j] = z
    return out
