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
from scipy.linalg import qr as scipy_qr


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


def identifiability_mask(X: np.ndarray) -> np.ndarray:
    """Columns that make an UNPENALIZED design singular, per training window.

    Three passes, cheapest first. A column is masked when it is

    1. CONSTANT within the window (no dispersion to identify a coefficient),
    2. an exact byte-duplicate of an earlier kept column, or
    3. outside a MAXIMAL INDEPENDENT SUBSET of what survives 1-2 — i.e. it is
       a linear combination of other kept columns, which pairwise duplicate
       detection cannot see.

    Passes 1-2 are threshold-free. Pass 3 needs a rank, and takes the rank cut
    :func:`numpy.linalg.matrix_rank` itself uses — ``max(m, n) * eps * s_max``,
    machine precision relative to the design's own largest singular value — on
    the CENTRED, UNIT-NORM-COLUMN design, so neither the cut nor the pivot
    order is a units artefact. The retained subset is the column-pivoted-QR
    pivot order truncated at that rank: greedy most-independent-first, so the
    columns dropped are the redundant ones.

    Why pass 3 is needed (MEASURED, not anticipated): availability-indicator
    columns of a late-starting exog source collapse to ONE IDENTICAL step
    function in every window that straddles that source's go-live bar — pass 2
    catches those. But at the go-live bar itself the newly-admitted block is
    only RANK-deficient, not duplicated: on ``implied_vol`` at local window
    t=58800 the mask kept 45 columns of rank 41, and the unpenalized fit
    predicted 2.2e12 (versus 1.3 either side). Two of 100 chunks blew up that
    way — ``chunk_54646`` (2009-07-16..09-17) and ``chunk_87481``
    (2012-02-14..04-17) — and, because the residual series feeds the view
    path block, each poisoned every remaining bar of its chunk: 2% of bars
    carried pooled QLIKE from 0.13 to 1.12. Same mechanism as the audited
    ``lam2=0`` pure-lasso pathology (specs/causal_tune_linear.py,
    ``RollingTunedLinear._recompute_mask``), one rung further down.

    Masked columns are DROPPED by :class:`MaskedResidualizer`, never zeroed:
    a zero column leaves ``X.T @ X`` singular, and sklearn's ``Ridge(alpha=0)``
    cholesky path then returns coefficients of order 1e13 that only cancel
    because the column is zero — a live trip hazard even on FULL-RANK windows.
    """
    Xa = np.asarray(X, dtype=np.float64)
    maskout = np.zeros(Xa.shape[1], dtype=bool)
    seen: dict = {}
    for j in range(Xa.shape[1]):
        col = Xa[:, j]
        if col.max() == col.min():          # no dispersion in this window
            maskout[j] = True
            continue
        key = col.tobytes()
        if key in seen:                     # exact duplicate of a kept column
            maskout[j] = True
        else:
            seen[key] = j

    kept = np.where(~maskout)[0]
    if kept.size == 0:
        return maskout

    # pass 3 — maximal independent subset of the survivors. Centred (an
    # intercept-carrying fit sees the centred design) and unit-norm columns.
    A = Xa[:, kept]
    A = A - A.mean(axis=0, keepdims=True)
    A = A / np.linalg.norm(A, axis=0)       # nonzero by pass 1
    svals = np.linalg.svd(A, compute_uv=False)
    rank = int((svals > max(A.shape) * np.finfo(np.float64).eps * svals[0]).sum())
    if rank < kept.size:
        _, _, piv = scipy_qr(A, mode="economic", pivoting=True)
        maskout[kept[piv[rank:]]] = True
    return maskout


class MaskedResidualizer:
    """:class:`Residualizer` with the per-window identifiability mask applied.

    The mask is recomputed from the RAW training window at every ``fit`` (the
    engine refits each step), then applied to both the fit design and any
    subsequent ``predict`` rows, so a column masked at fit time cannot leak
    back in at predict time.

    Masked columns are DROPPED, not zeroed. Zeroing leaves the column in the
    design, ``X.T @ X`` stays singular, and the baseline's solver has to cope
    with it — ``Ridge(alpha=0)`` returns |coef| ~ 1e13 on those columns, which
    is harmless only for as long as they are exactly zero. Dropping hands the
    baseline a design that is full rank by construction, so the fit is the
    plain OLS/ridge fit on the identifiable subset.
    """

    def __init__(self, baseline_factory: Callable[[], Any]) -> None:
        self.baseline_factory = baseline_factory
        self.baseline: Any = None
        self._maskout: np.ndarray | None = None

    def _apply(self, X: np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        if self._maskout is not None:
            Xa = Xa[:, ~self._maskout]
        return Xa

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MaskedResidualizer":
        self._maskout = identifiability_mask(X)
        self.baseline = self.baseline_factory()
        self.baseline.fit(self._apply(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.baseline is None:
            raise RuntimeError("MaskedResidualizer.predict() called before fit()")
        return self.baseline.predict(self._apply(X))

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y - self.predict(X)


class RollingRidgeResidualizer:
    """Residualizer(Ridge) at rank-1 rolling cost — exact to numerical precision.

    The fresh-refit residualizer pays ``O(W p^2)`` sklearn Ridge per BAR; on a
    24000 x 543 window that is ~0.3 s/bar and dominates every anchored arm.
    This maintains :class:`~src.models.rolling_least_squares.RollingLeastSquares`
    sufficient statistics instead: the engine slides the window with
    :meth:`roll` (``O(p^2)`` + an ``O(p^3)`` solve) and only rebuilds exactly
    on the solver's own ``reinit_every`` cadence to bound float drift.
    Predictions match ``Ridge(alpha, fit_intercept=True)`` refit per window to
    numerical precision — same contract the incumbent's incremental fast path
    has relied on since the rank-1 solver shipped.

    Duck-typed marker ``is_rolling``: :class:`MultiStageBacktest` detects it
    and calls ``roll`` per step instead of ``fit``.
    """

    is_rolling = True

    def __init__(self, alpha: float, reinit_every: int = 5000,
                 fast_inverse: bool = False) -> None:
        from src.models.rolling_least_squares import RollingLeastSquares

        # fast_inverse: maintain (gram + alpha I)^-1 by Sherman-Morrison
        # instead of an O(p^3) solve per bar. The window slide changes the
        # CENTERED gram by exactly four symmetric rank-1 terms —
        # +x_in x_in', -x_out x_out', -(1/n) sx'sx'' (new mean), +(1/n) sx sx'
        # (old mean) — so four O(p^2) inverse updates replace the solve
        # (~200x at p~1800). Exact re-inversion every reinit_every steps
        # bounds float drift; a near-singular downdate denominator triggers
        # an immediate exact rebuild rather than a silent divergence.
        self._rls = RollingLeastSquares(
            alpha=float(alpha),
            reinit_every=(1000 if fast_inverse else reinit_every),
        )
        self._fast = bool(fast_inverse)
        self._Ainv: np.ndarray | None = None

    @property
    def reinit_every(self) -> int:
        return self._rls.reinit_every

    def _gram(self) -> np.ndarray:
        r = self._rls
        g = r._Sxx - np.outer(r._sx, r._sx) / r.n
        return g + r.alpha * np.eye(r.p)

    def _coef_from_inverse(self) -> None:
        r = self._rls
        rhs = r._Sxy - r._sx * (r._sy / r.n)
        r._coef = self._Ainv @ rhs
        r._intercept = r._sy / r.n - float(r._sx @ r._coef) / r.n

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RollingRidgeResidualizer":
        """Exact rebuild from a full window (step 0 and the reinit cadence)."""
        self._rls.init_window(X, y)
        if self._fast:
            self._Ainv = np.linalg.inv(self._gram())
            self._coef_from_inverse()
        else:
            self._rls.solve()
        return self

    def roll(self, x_in, y_in, x_out, y_out) -> None:
        """Slide the window one bar and re-solve."""
        if not self._fast:
            self._rls.roll(x_in, y_in, x_out, y_out)
            self._rls.solve()
            return
        r = self._rls
        sx_old = r._sx.copy()
        r.roll(x_in, y_in, x_out, y_out)
        # rank-4 block Woodbury: A' = A + U C U' with the four update vectors
        # as columns; (A')^-1 = Ainv - V (C^-1 + U'V)^-1 V' with V = Ainv U.
        # One gemm each way — ~3 O(p^2) passes instead of four rank-1 sweeps.
        U = np.stack(
            [np.asarray(x_in, dtype=np.float64),
             np.asarray(x_out, dtype=np.float64),
             r._sx.astype(np.float64),
             sx_old],
            axis=1,
        )
        c = np.array([1.0, -1.0, -1.0 / r.n, 1.0 / r.n])
        V = self._Ainv @ U
        S = np.diag(1.0 / c) + U.T @ V
        try:
            M = np.linalg.solve(S, V.T)
            self._Ainv -= V @ M
        except np.linalg.LinAlgError:  # numerical guard: rebuild exactly
            self._Ainv = np.linalg.inv(self._gram())
        self._coef_from_inverse()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._rls.predict(np.atleast_2d(np.asarray(X, dtype=np.float64)))

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y - self.predict(X)


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
