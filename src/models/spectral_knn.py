"""Spectral embedding of trailing target views + Gaussian-weighted kNN regression.

Default composition (``residualizer_kind="ridge"``): Ridge residualizer +
spectral embedding of residual views + kNN in embedding space. With
``residualizer_kind="identity"`` the first stage is the no-op
:class:`IdentityResidualizer`: views are windows of ``y`` itself — a pure
spectral-ANALOG kNN on the target. NOTE: in identity mode exog features do
not enter the model anywhere (MultiStageBacktest reads ``X`` only through
the residualizer), so every exog-bucket arm is bit-identical.

A non-trivial :class:`MultiStageBacktest` configuration — same shape as every
other ``src/models/<x>.py`` (the simple models just plug ``IdentityResidualizer``
+ ``None`` feature), only with a non-degenerate feature stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.evaluation.metrics import calculate_metrics
from src.features.extractors.spectral_embedding import (
    allocated_cpus,
    build_embedding,
    build_embedding_channels,
    build_embedding_exog,
    build_geometry_exog,
)
from src.features.transforms.target import PERIODS_PER_DAY
from src.features.transforms.residualizer import (
    IdentityResidualizer,
    MaskedResidualizer,
    RollingRidgeResidualizer,
)
from src.models.knn import gaussian_weights

DEFAULT_SPECTRAL_KNN_PARAMS: dict = dict(
    ridge_alpha=1.0,
    view_window=960,
    embedding_dim=8,
    graph_k=10,
    neighbor_k=25,
    refit_frequency=240,
)


class PLSKNN:
    """kNN in the top-``d`` PLS directions of the view — SUPERVISED basis compression.

    The HAR-of-exog state block is present in the raw composed view almost
    exactly (readout R^2 ~ 1, measured 2026-07-30), but unsupervised
    compressions rank directions by VARIANCE, and variance-rank is not
    relevance-rank (PCA tied raw at the same d). PLS ranks directions of the
    view by covariance with the TARGET, so the compressed basis keeps what
    predicts and sheds what merely varies. Fit causally on each training
    window's (views, targets) — the same information the anchor itself uses,
    so no new leakage surface.

    Lives at the regressor seam (fit/predict) so the engine and feature stage
    stay untouched: ``fit`` learns the PLS basis then indexes neighbors in
    score space; ``predict`` projects and queries.

    ``weight_mode`` selects how the kept components enter the DISTANCE:

    * ``"none"`` — hard truncation at ``n_components``, coordinates unweighted
      (the original form; measured U-shape in d says the tail past d=3 holds
      information at falling signal-to-noise — exactly what truncation wastes).
    * ``"exp"`` — SOFT filter: keep all ``n_components``, scale coordinate i
      by ``decay**i``. One knob; ``decay -> 0`` recovers d=1, ``decay -> 1``
      the unweighted rotation.
    * ``"cov"`` — SELF-weighted soft filter: scale each component by its own
      |covariance with the target| measured on the fit window, normalized to
      max 1. The decomposition weights itself — no invented constant.
    * ``"slp"`` — GRADIENT-metric weighting: component i scaled by its
      regression SLOPE ``|cov(t_i, y)| / var(t_i)``, normalized to max 1.
      The locally-optimal kNN metric weights directions by df/dz, so one
      unit of distance in any component equals one unit of conditional-mean
      difference. Distinct from "cov" by the var(t_i) denominator, which
      tempers the high-variance collinear HAR level factor that cov
      double-counts (measured: the slope spectrum tracks the empirically
      optimal exp(0.7) filter; the cov spectrum does not). Knob-free.
    * ``"rdg"`` — RIDGE-shaped filter on the measured relevance spectrum:
      ``w_i = s_i^2 / (s_i^2 + alpha)`` with ``alpha = decay * s_1^2`` — the
      canonical smooth shrinkage (filter factors lam/(lam+alpha)) applied
      along the PLS rotation instead of the covariance eigenbasis; ``decay``
      is consumed as alpha's fraction of the top component's s^2.

    ``ridge_alpha > 0`` extracts the PLS directions from the RIDGE-AUGMENTED
    design (sqrt(alpha)*I rows stacked under the standardized X, zeros under
    y) — exact ridge-regularized PLS via the augmentation identity. alpha
    smooths WITHIN each loading estimate (the shrinkage counterpart of
    SPLSKNN's sparsity), the component filter acts ACROSS directions.
    """

    def __init__(
        self,
        n_components: int,
        n_neighbors: int,
        n_jobs: int = 1,
        weight_mode: str = "none",
        decay: float = 1.0,
        ridge_alpha: float = 0.0,
        local_alpha: float | None = None,
        local_degree: int = 1,
    ) -> None:
        from sklearn.cross_decomposition import PLSRegression

        if weight_mode not in ("none", "exp", "cov", "rdg", "slp"):
            raise ValueError(f"unknown weight_mode: {weight_mode!r}")
        self._pls = PLSRegression(n_components=int(n_components), scale=True)
        # local_alpha selects the LOCAL-LINEAR readout (per-query ridge on the
        # neighbors) instead of the local-constant Gaussian target average;
        # local_degree=2 upgrades it to the local paraboloid (dense
        # interactions as per-neighborhood curvature).
        self._knn = (
            LocalRidgeKNN(int(n_neighbors), float(local_alpha), n_jobs=n_jobs,
                          degree=local_degree)
            if local_alpha is not None
            else KNeighborsRegressor(
                n_neighbors=int(n_neighbors), weights=gaussian_weights, n_jobs=n_jobs
            )
        )
        self._mode = weight_mode
        self._decay = float(decay)
        self._ridge_alpha = float(ridge_alpha)
        self._w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSKNN":
        if self._ridge_alpha > 0.0:
            Xa = np.asarray(X, dtype=np.float64)
            mu, sd = Xa.mean(axis=0), Xa.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            Xs = (Xa - mu) / sd
            aug_X = np.vstack([Xs, np.sqrt(self._ridge_alpha) * np.eye(Xa.shape[1])])
            aug_y = np.concatenate([np.asarray(y, dtype=np.float64),
                                    np.zeros(Xa.shape[1])])
            # scale=False: the augmentation must see the standardized design
            # itself, and PLSRegression would re-standardize the stacked rows.
            from sklearn.cross_decomposition import PLSRegression

            self._pls = PLSRegression(
                n_components=self._pls.n_components, scale=False
            )
            self._pls.fit(aug_X, aug_y)
            self._pre = (mu, sd)
            Z = self._pls.transform(Xs)
        else:
            self._pls.fit(X, y)
            self._pre = None
            Z = self._pls.transform(X)
        if self._mode == "exp":
            self._w = self._decay ** np.arange(Z.shape[1], dtype=np.float64)
        elif self._mode in ("cov", "rdg", "slp"):
            yc = np.asarray(y, dtype=np.float64) - np.mean(y)
            s = np.abs(Z.T @ yc) / len(yc)
            if s.max() <= 0:
                raise ValueError("relevance weighting degenerate: no component covaries")
            if self._mode == "cov":
                self._w = s / s.max()
            elif self._mode == "slp":
                slope = s / Z.var(axis=0)  # var > 0: PLS scores carry variance
                self._w = slope / slope.max()
            else:
                s2 = s**2
                self._w = s2 / (s2 + self._decay * s2.max())
        else:
            self._w = None
        self._knn.fit(Z if self._w is None else Z * self._w, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        if self._pre is not None:  # ridge-augmented fit used scale=False
            mu, sd = self._pre
            Xa = (Xa - mu) / sd
        Z = self._pls.transform(Xa)
        return self._knn.predict(Z if self._w is None else Z * self._w)


class LocalRidgeKNN:
    """LOCAL-LINEAR analog regression: per-query ridge on the k neighbors.

    Nadaraya-Watson (the Gaussian-weighted target average every kNN arm has
    used) is a local-CONSTANT estimator: it cannot nest a linear model, pays
    boundary bias exactly on the extreme bars QLIKE punishes, and inherits
    the refit-cadence staleness. This fits a Gaussian-weighted ridge ON THE
    NEIGHBORS at every query — centered at the query point, so the intercept
    IS the prediction. As k grows toward all views with flat weights it
    RECOVERS the global windowed linear readout by construction; smaller k
    trades toward pure analog locality. The local solve is O(k d + d^3) per
    bar (d = compressed dimension), and is refit per QUERY, so the metric's
    240-bar refit cadence no longer staleness-taxes the regression part.
    """

    def __init__(self, n_neighbors: int, alpha: float, n_jobs: int = 1,
                 degree: int = 1) -> None:
        if degree not in (1, 2):
            raise ValueError(f"degree must be 1 or 2: {degree}")
        self._k = int(n_neighbors)
        self._alpha = float(alpha)
        self._jobs = n_jobs
        self._degree = int(degree)
        self._nn = None

    @staticmethod
    def _expand2(Z: np.ndarray) -> np.ndarray:
        """Append all pairwise products (upper triangle incl. squares).

        Local degree=2: curvature IS interaction structure — the dense-weak
        higher-order terms enter as the local paraboloid's cross-terms,
        estimated fresh per neighborhood under the same scale-free shrinkage
        (dense weak signals are aggregate-estimable under L2, never
        selectable term-by-term).
        """
        d = Z.shape[1]
        iu = np.triu_indices(d)
        return np.hstack([Z, Z[:, iu[0]] * Z[:, iu[1]]])

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LocalRidgeKNN":
        from sklearn.neighbors import NearestNeighbors

        Z = np.asarray(Z, dtype=np.float64)
        k = min(self._k, Z.shape[0])
        self._nn = NearestNeighbors(n_neighbors=k, n_jobs=self._jobs).fit(Z)
        self._Z = Z
        self._y = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, Zq: np.ndarray) -> np.ndarray:
        Zq = np.atleast_2d(np.asarray(Zq, dtype=np.float64))
        dists, idx = self._nn.kneighbors(Zq)
        out = np.empty(len(Zq), dtype=np.float64)
        for r in range(len(Zq)):
            d = dists[r]
            sigma = float(np.median(d))
            w = np.exp(-(d**2) / (2 * sigma**2)) if sigma > 0 else np.ones_like(d)
            w = w / w.sum()
            Zi = self._Z[idx[r]] - Zq[r]          # center at the query
            if self._degree == 2:
                Zi = self._expand2(Zi)            # local paraboloid basis
            yi = self._y[idx[r]]
            zm = w @ Zi
            ym = float(w @ yi)
            Zc = Zi - zm
            yc = yi - ym
            G = (Zc * w[:, None]).T @ Zc
            # scale-free conditioning: penalty proportional to the local
            # weighted covariance's mean eigenvalue — alpha is dimensionless,
            # so re-weighted coordinates are not over-penalized (a fixed
            # absolute penalty underfits shrunken directions; measured on the
            # toy as k=1500 R2 0.75 vs 0.95).
            lam = self._alpha * float(np.trace(G)) / G.shape[0]
            b = np.linalg.solve(G + lam * np.eye(G.shape[0]), (Zc * w[:, None]).T @ yc)
            out[r] = ym - float(zm @ b)           # local plane evaluated at z=0
        return out


class PLSQREG:
    """DENSE-INTERACTION readout: ridge on the poly-2 expansion of PLS scores.

    All pairwise products of the top-d relevance coordinates (d(d+1)/2 + d
    terms), L2-shrunk — the shrink-over-everything estimator the dense-weak
    regime demands (each interaction is individually sub-significance; only
    the AGGREGATE is estimable — the d8-dissection's 36% diffuse finding).
    Interactions of the compressed basis, so every term stays legible as
    summary_i x summary_j.
    """

    def __init__(self, n_components: int, alpha: float = 1.0) -> None:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge

        self._pls = PLSRegression(n_components=int(n_components), scale=True)
        self._ridge = Ridge(alpha=float(alpha))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSQREG":
        self._pls.fit(X, y)
        P = LocalRidgeKNN._expand2(self._pls.transform(X))
        self._mu = P.mean(axis=0)
        sd = P.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        self._ridge.fit((P - self._mu) / self._sd, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = LocalRidgeKNN._expand2(self._pls.transform(X))
        return self._ridge.predict((P - self._mu) / self._sd)


class TICASLPKNN:
    """SHAPED spectral filter: dynamics eigenbasis x slope weights x local-linear.

    The brick-wall failures (kinetic-map TICA = keep-slow; anti-TICA =
    kill-slow) both assumed relevance is a function of spectral POSITION.
    This weights each dynamics mode by its measured predictive SLOPE
    (|cov(z_i, y)| / var(z_i) — the gradient-metric functional validated on
    the PLS basis), so the filter SHAPE over the timescale axis is learned:
    slow-but-predictive modes (the level) survive, slow drift and fast noise
    are damped, and the weight-vs-timescale profile is itself the legible
    'where does predictability live' spectrum. ``n_components`` caps the
    modes kept (by slope rank); 0 = all numerically-supported modes.
    """

    def __init__(self, n_components: int, n_neighbors: int, tau: int = 48,
                 local_alpha: float = 0.01, n_jobs: int = 1) -> None:
        self._d = int(n_components)
        self._tau = int(tau)
        self._knn = LocalRidgeKNN(int(n_neighbors), float(local_alpha),
                                  n_jobs=n_jobs)
        self._logged = False

    def fit(self, V: np.ndarray, y: np.ndarray) -> "TICASLPKNN":
        Va = np.asarray(V, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        self._mu = Va.mean(axis=0)
        sd = Va.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        Z = (Va - self._mu) / self._sd
        n, tau = len(Z), self._tau
        C0 = Z.T @ Z / n
        Ct = (Z[:-tau].T @ Z[tau:] + Z[tau:].T @ Z[:-tau]) / (2 * (n - tau))
        lam0, U0 = np.linalg.eigh(C0)
        keep = lam0 > lam0.max() * (max(n, C0.shape[0]) * np.finfo(float).eps) ** 2
        T0 = U0[:, keep] / np.sqrt(lam0[keep])[None, :]
        M = T0.T @ Ct @ T0
        M = (M + M.T) / 2
        lam, Wm = np.linalg.eigh(M)
        lam, Wm = lam[::-1], Wm[:, ::-1]
        R = T0 @ Wm                                # dynamics modes, whitened
        Zm = Z @ R
        yc = ya - ya.mean()
        s = np.abs(Zm.T @ yc) / len(yc)
        slope = s / Zm.var(axis=0)
        w = slope / slope.max()
        if self._d > 0:                            # keep top-d by slope
            cut = np.argsort(w)[::-1][: self._d]
            R, w, lam = R[:, cut], w[cut], lam[cut]
            Zm = Zm[:, cut]
        self._R = R
        self._w = w
        if not self._logged:                       # the legible spectrum, once
            ts = -tau / np.log(np.clip(np.abs(lam), 1e-12, 0.9999))
            order = np.argsort(w)[::-1][:8]
            prof = ", ".join(f"ts={ts[i]:.0f}b w={w[i]:.2f}" for i in order)
            print(f"[ticaslp] top modes by slope: {prof}")
            self._logged = True
        self._knn.fit(Zm * w, ya)
        return self

    def predict(self, V: np.ndarray) -> np.ndarray:
        Z = (np.asarray(V, dtype=np.float64) - self._mu) / self._sd
        return self._knn.predict((Z @ self._R) * self._w)


class PLSREG:
    """The no-kNN CONTROL: the windowed PLS regression readout itself.

    Identical fit surface to :class:`PLSKNN` (same views, same window, same
    refit cadence) but predicts with the LINEAR readout instead of the
    neighbor average. Whatever the kNN arms beat this by is the measured
    value of the analog search; if they don't beat it, the geometry program
    reduces to windowed regularized linear regression.

    ``ridge_alpha > 0`` fits on the ridge-augmented design (the PLSKNN
    identity): k components of augmented PLS = k conjugate-gradient
    iterations toward the exact ridge solution on this basis, so at moderate
    k this readout approximates ridge-on-the-view-basis by construction.
    """

    def __init__(self, n_components: int, ridge_alpha: float = 0.0) -> None:
        from sklearn.cross_decomposition import PLSRegression

        self._K = int(n_components)
        self._alpha = float(ridge_alpha)
        self._pls = PLSRegression(n_components=self._K, scale=True)
        self._pre: tuple | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSREG":
        from sklearn.cross_decomposition import PLSRegression

        if self._alpha > 0.0:
            Xa = np.asarray(X, dtype=np.float64)
            mu, sd = Xa.mean(axis=0), Xa.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            Xs = (Xa - mu) / sd
            aug_X = np.vstack([Xs, np.sqrt(self._alpha) * np.eye(Xa.shape[1])])
            aug_y = np.concatenate([np.asarray(y, dtype=np.float64),
                                    np.zeros(Xa.shape[1])])
            self._pls = PLSRegression(n_components=self._K, scale=False)
            self._pls.fit(aug_X, aug_y)
            self._pre = (mu, sd)
        else:
            self._pls = PLSRegression(n_components=self._K, scale=True)
            self._pls.fit(X, y)
            self._pre = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        if self._pre is not None:
            mu, sd = self._pre
            Xa = (Xa - mu) / sd
        return np.asarray(self._pls.predict(Xa)).ravel()


class PCOVRKNN:
    """kNN in CONTINUUM-INTERIOR directions between PLS and PCA orderings.

    Per component, the direction maximizes ``beta * (w'c)^2 + (1-beta) *
    w'Sw`` with ``c`` the (unit-normalized) covariance of the deflated design
    with the target and ``S`` its covariance matrix (normalized by its top
    eigenvalue, so beta is scale-free): the top eigenvector of
    ``beta * cc' + (1-beta) * S``. ``beta=1`` recovers the PLS direction,
    ``beta=0`` the PCA direction — the PCovR-style interior of the
    relevance-vs-variance 2x2 this campaign raced at its corners. Components
    are extracted with X-deflation, exactly as PLS1.
    """

    def __init__(
        self, n_components: int, n_neighbors: int, beta: float, n_jobs: int = 1,
        local_alpha: float | None = None,
    ) -> None:
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError(f"beta must be in [0, 1]: {beta}")
        self._K = int(n_components)
        self._beta = float(beta)
        self._knn = (
            LocalRidgeKNN(int(n_neighbors), float(local_alpha), n_jobs=n_jobs)
            if local_alpha is not None
            else KNeighborsRegressor(
                n_neighbors=int(n_neighbors), weights=gaussian_weights, n_jobs=n_jobs
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCOVRKNN":
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        self._mean = Xa.mean(axis=0)
        std = Xa.std(axis=0)
        self._std = np.where(std > 0, std, 1.0)
        Xc = (Xa - self._mean) / self._std
        yd = ya - ya.mean()
        Xd = Xc.copy()
        n = len(ya)
        # S is computed ONCE and rank-1 downdated after each deflation:
        # (X - t p')'(X - t p') = X'X - tt * p p' (using X't = tt * p), so the
        # O(n p^2) Gram rebuild per component collapses to an O(p^2) update.
        S = Xd.T @ Xd / n
        Ws, Ps = [], []
        for _ in range(self._K):
            c = Xd.T @ yd / n
            nc = np.linalg.norm(c)
            lam_top = float(np.linalg.eigvalsh(S)[-1])
            if nc <= 0 or lam_top <= 0:
                break
            M = self._beta * np.outer(c / nc, c / nc) + (1 - self._beta) * S / lam_top
            _, vecs = np.linalg.eigh(M)
            w = vecs[:, -1]
            t = Xd @ w
            tt = float(t @ t)
            if tt <= 0:
                break
            p = Xd.T @ t / tt
            Xd = Xd - np.outer(t, p)
            yd = yd - (float(yd @ t) / tt) * t
            S = S - (tt / n) * np.outer(p, p)
            Ws.append(w)
            Ps.append(p)
        if not Ws:
            raise ValueError("pcovr: no component extracted")
        W = np.stack(Ws, axis=1)
        P = np.stack(Ps, axis=1)
        self._R = W @ np.linalg.inv(P.T @ W)
        self._knn.fit(Xc @ self._R, ya)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xc = (np.asarray(X, dtype=np.float64) - self._mean) / self._std
        return self._knn.predict(Xc @ self._R)


class SPLSKNN:
    """kNN in SPARSE-PLS scores — soft thresholding on the FEATURE axis.

    Chun-Keles-style PLS1: each component's weight vector is the
    soft-thresholded covariance direction ``S(X'y, eta * max|X'y|)`` —
    loadings below the ``eta`` fraction of the strongest are zeroed, the
    rest shrunk — then the design is deflated and the next direction
    extracted. This sparsifies the ROTATION (which of the ~537 columns each
    direction reads), the complementary softness to PLSKNN's component
    filters, and the targeted cure for noise loaded from dead-indicator
    columns. ``eta = 0`` recovers dense PLS1.

    ``slope_weight=True`` composes the two soft axes: after the sparse
    rotation, components are scaled by their regression slope (the
    gradient-metric weighting of :class:`PLSKNN` mode ``"slp"``).
    """

    def __init__(
        self, n_components: int, n_neighbors: int, eta: float, n_jobs: int = 1,
        slope_weight: bool = False, local_alpha: float | None = None,
    ) -> None:
        if not 0.0 <= float(eta) < 1.0:
            raise ValueError(f"eta must be in [0, 1): {eta}")
        self._K = int(n_components)
        self._eta = float(eta)
        self._slope_weight = bool(slope_weight)
        self._w: np.ndarray | None = None
        self._knn = (
            LocalRidgeKNN(int(n_neighbors), float(local_alpha), n_jobs=n_jobs)
            if local_alpha is not None
            else KNeighborsRegressor(
                n_neighbors=int(n_neighbors), weights=gaussian_weights, n_jobs=n_jobs
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SPLSKNN":
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        self._mean = Xa.mean(axis=0)
        std = Xa.std(axis=0)
        self._std = np.where(std > 0, std, 1.0)  # constant cols center to 0
        Xc = (Xa - self._mean) / self._std
        yd = ya - ya.mean()
        Xd = Xc.copy()
        Ws, Ps = [], []
        for _ in range(self._K):
            c = Xd.T @ yd
            thr = self._eta * np.abs(c).max()
            w = np.sign(c) * np.maximum(np.abs(c) - thr, 0.0)
            nw = np.linalg.norm(w)
            if nw <= 0:
                break  # threshold ate every loading — keep components found
            w /= nw
            t = Xd @ w
            tt = float(t @ t)
            if tt <= 0:
                break
            p = Xd.T @ t / tt
            Xd = Xd - np.outer(t, p)
            yd = yd - (float(yd @ t) / tt) * t
            Ws.append(w)
            Ps.append(p)
        if not Ws:
            raise ValueError("spls: no component survived the threshold")
        W = np.stack(Ws, axis=1)
        P = np.stack(Ps, axis=1)
        self._R = W @ np.linalg.inv(P.T @ W)
        Z = Xc @ self._R
        if self._slope_weight:
            yc = ya - ya.mean()
            s = np.abs(Z.T @ yc) / len(yc)
            if s.max() <= 0:
                raise ValueError("slope weighting degenerate: no component covaries")
            slope = s / Z.var(axis=0)
            self._w = slope / slope.max()
            Z = Z * self._w
        self._knn.fit(Z, ya)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xc = (np.asarray(X, dtype=np.float64) - self._mean) / self._std
        Z = Xc @ self._R
        return self._knn.predict(Z if self._w is None else Z * self._w)


def _run(residualizer, feature_fit, hp, X_chunk, y_chunk, train_win_periods,
         *, exog_window=False, exog=True):
    """Build the MultiStageBacktest and run it (shared by every view mode)."""
    backtest = MultiStageBacktest(
        residualizer=residualizer,
        regressor_factory=lambda: KNeighborsRegressor(
            n_neighbors=int(hp["neighbor_k"]),
            weights=gaussian_weights,
            n_jobs=allocated_cpus(),
        ),
        refit_frequency=int(hp["refit_frequency"]),
        feature_fit=feature_fit,
        view_window=int(hp["view_window"]),
        feature_exog=exog,
        feature_exog_window=exog_window,
        correction_weight=float(hp.get('omega', 1.0)),
        stack_window=int(hp.get('stack_window', 0)),
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="spectral_knn")


def fit_predict_spectral_knn(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Spectral-kNN backtest as a MultiStageBacktest composition.

    Reads as the sentence "residualizer + spectral_embedding feature +
    Gaussian-weighted kNN regressor":

        residualizer  = Residualizer(Ridge)        [residualizer_kind="ridge"]
                        IdentityResidualizer()     [residualizer_kind="identity"]
        feature_fit   = build_embedding(views, d, k_graph)
        regressor     = KNeighborsRegressor(n_neighbors, weights=gaussian)

    ``residualizer_kind="identity"`` (user-ruled for the spectral_knn_backtest
    spec) drops the Ridge first stage: views are windows of ``y`` itself (a
    pure spectral-analog kNN), and ``ridge_alpha`` is never read.

    Internal control keys (``_*``) are stripped before forwarding.
    """
    hp = {k: v for k, v in hyperparams.items() if not k.startswith("_")}
    seed = int(hp.get("seed", 42))
    residualizer_kind = str(hp.get("residualizer_kind", "ridge"))
    if residualizer_kind == "identity":
        residualizer = IdentityResidualizer()
    elif residualizer_kind == "ridge":
        # alpha=0 is UNPENALIZED OLS, which goes singular on windows where a
        # late-start source's availability indicators collapse to one
        # identical step function — measured blowups: mz_beta ~1e-27,
        # oos_r2 ~-1e50 on sentiment / implied_vol / vol_demand /
        # all_features. The audited per-window identifiability mask is the
        # fix, so route the unpenalized case through MaskedResidualizer.
        alpha = float(hp["ridge_alpha"])
        if alpha == 0.0:
            residualizer = MaskedResidualizer(lambda: Ridge(alpha=alpha))
        else:
            # rank-1 rolling anchor: exact-to-precision vs Ridge refit per
            # bar, ~30x cheaper (the fresh fit dominated every anchored arm).
            # fast_anchor adds the block-Woodbury inverse (O(p^2)/bar, ~5x
            # again at p~1100; gated exact to ~1e-11).
            residualizer = RollingRidgeResidualizer(
                alpha, fast_inverse=bool(hp.get("fast_anchor", False))
            )
    else:
        raise ValueError(f"unknown residualizer_kind: {residualizer_kind!r}")
    view_exog = bool(hp.get("view_exog", False))
    if view_exog:
        # STATE BLOCK = the NON-target-derived columns. The path block already
        # carries the target's own history at full resolution over the view
        # window, so the ``har_ma_*`` block (the target's moving averages, and
        # the ``har_ma_*_x_{open,close}`` interactions built from them) would
        # re-encode that same information and silently re-weight it in the
        # distance — heavily so for narrow buckets, where target-derived
        # columns would otherwise dominate the state block and confound the
        # bucket comparison. Names arrive from the production scaffold
        # (``_feature_names``), never re-derived here.
        names = hyperparams.get("_feature_names")
        if not names:
            raise ValueError(
                "view_exog=True needs _feature_names (the X column names) to "
                "select the state block; run through src.backtest.executor."
            )
        state_cols = [i for i, nm in enumerate(names) if not nm.startswith("har_ma_")]

        # state_rung_cap: MINIMALITY knob — keep only state columns whose HAR
        # rung is <= cap (columns without a _ma_<r> suffix — calendar, plain
        # indicators — always stay). The reach-redundancy result says the long
        # rungs are collinear passengers; this measures the smallest
        # sufficient ladder instead of assuming the full production one.
        rung_cap = int(hp.get("state_rung_cap", 0))
        if rung_cap > 0:
            import re

            def _keep(nm: str) -> bool:
                m = re.search(r"_ma_(\d+)$", nm)
                return m is None or int(m.group(1)) <= rung_cap

            state_cols = [i for i in state_cols if _keep(names[i])]

        # drop_avail_ma: availability indicators are STEP functions (a source
        # goes live once), so their moving averages are ramps-then-constants —
        # the constant-in-window degenerate family of the identifiability-mask
        # history, and prime noise-loading fodder. Dropping the MAs keeps the
        # plain *_avail flag and the *_active_ma_* occurrence RATES (known
        # signal: the VRP attribution's activity channel).
        if bool(hp.get("drop_avail_ma", False)):
            state_cols = [i for i in state_cols if "_avail_ma_" not in names[i]]

        # view_channels: give each selected exog series its OWN W-bar window
        # instead of a single point-in-time snapshot, so the analog search
        # matches on JOINT trajectories. Channels are the contemporaneous
        # (*_ma_1) columns of the state block -- one per underlying adjusted
        # series -- which keeps the composed matrix at n x (W * (1+channels))
        # instead of exploding over every HAR lag of every series.
        view_channels = bool(hp.get("view_channels", False))
        channel_cols = [
            i for i in state_cols
            if names[i].endswith("_ma_1") or not names[i].startswith("adj_")
        ] if view_channels else None

        if view_channels:
            def feature_fit(views, X_train, view_idx):
                return build_embedding_channels(
                    views, X_train, view_idx,
                    d=int(hp["embedding_dim"]),
                    k_graph=int(hp["graph_k"]),
                    seed=seed,
                    channel_cols=channel_cols,
                    eigen_solver=hp.get("eigen_solver"),
                )
            return _run(residualizer, feature_fit, hp, X_chunk, y_chunk,
                        train_win_periods, exog_window=True)

        # geometry: the METRIC the analog search runs in. "euclid" (default)
        # is the raw composed view — byte-identical to every prior arm. The
        # X-only alternatives (har / whiten / tica / pca) live in
        # spectral_embedding.GeometryExogBasis. "pls" is the SUPERVISED basis
        # compression: raw composed views (euclid passthrough) with the
        # PLS-kNN regressor doing the top-d relevance-ranked projection at
        # the regressor seam — embedding_dim is consumed as the PLS dim.
        geometry = str(hp.get("geometry", "euclid"))
        if geometry in ("pls", "plsexp", "plscov", "plsrdg", "plsslp",
                        "spls", "splsslp", "plsreg", "plsrg", "pcovr",
                        "lplsreg", "lplsrg", "plsloc", "lplsloc",
                        "plsq", "plsqloc", "ticaslp"):
            if int(hp["embedding_dim"]) <= 0 and geometry != "ticaslp":
                # ticaslp: d=0 = keep ALL numerically-supported modes
                raise ValueError(f"geometry={geometry!r} consumes embedding_dim; need > 0")
            weight_mode = {"pls": "none", "plsexp": "exp", "plscov": "cov",
                           "plsrdg": "rdg", "plsslp": "slp"}.get(geometry)

            if geometry in ("lplsreg", "lplsrg", "lplsloc"):
                # FULL-REACH LADDER views: the path block enters as its base-2
                # trailing-mean ladder (path_ladder), so view_window can carry
                # the target's long memory (W = 8192, 16384, ...) at ~log2(W)
                # coordinates — removing the view basis's reach handicap
                # against the anchor's har_ma_* columns.
                def feature_fit(views, exog):
                    return build_geometry_exog(
                        views, exog, kind="har", d=0,
                        k_graph=int(hp["graph_k"]), seed=seed,
                        state_cols=state_cols,
                        ladder_base=int(hp.get("ladder_base", 2)),
                    )
            else:
                def feature_fit(views, exog):
                    return build_embedding_exog(
                        views, exog, d=0, k_graph=int(hp["graph_k"]), seed=seed,
                        state_cols=state_cols,
                    )

            # local_alpha (hp, optional): swap the local-CONSTANT neighbor
            # average for the LOCAL-LINEAR per-query ridge in ANY family.
            loc = hp.get("local_alpha")
            loc = None if loc in (None, 0, 0.0, "") else float(loc)

            def _make_regressor():
                if geometry == "ticaslp":
                    return TICASLPKNN(
                        n_components=int(hp["embedding_dim"]),
                        n_neighbors=int(hp["neighbor_k"]),
                        tau=int(hp.get("geometry_tau", PERIODS_PER_DAY)),
                        local_alpha=float(hp.get("pls_decay", 0.01)) or 0.01,
                        n_jobs=allocated_cpus(),
                    )
                if geometry in ("plsreg", "lplsreg"):
                    return PLSREG(
                        n_components=int(hp["embedding_dim"]),
                        ridge_alpha=float(hp.get("pls_decay", 0.0))
                        if geometry == "lplsreg" else 0.0,
                    )
                if geometry == "pcovr":
                    return PCOVRKNN(
                        n_components=int(hp["embedding_dim"]),
                        n_neighbors=int(hp["neighbor_k"]),
                        beta=float(hp.get("pls_decay", 0.5)),
                        n_jobs=allocated_cpus(),
                        local_alpha=loc,
                    )
                if geometry in ("plsrg", "lplsrg"):
                    # decay slot = the augmentation ridge alpha (absolute)
                    return PLSKNN(
                        n_components=int(hp["embedding_dim"]),
                        n_neighbors=int(hp["neighbor_k"]),
                        n_jobs=allocated_cpus(),
                        weight_mode="none",
                        ridge_alpha=float(hp.get("pls_decay", 0.0)),
                        local_alpha=loc,
                    )
                if geometry in ("plsloc", "lplsloc", "plsqloc"):
                    # slope-weighted metric + LOCAL-LINEAR (plsloc) or
                    # LOCAL-PARABOLOID (plsqloc) readout; decay slot = the
                    # scale-free local ridge alpha
                    return PLSKNN(
                        n_components=int(hp["embedding_dim"]),
                        n_neighbors=int(hp["neighbor_k"]),
                        n_jobs=allocated_cpus(),
                        weight_mode="slp",
                        local_alpha=float(hp.get("pls_decay", 1.0)) or 1.0,
                        local_degree=2 if geometry == "plsqloc" else 1,
                    )
                if geometry == "plsq":
                    return PLSQREG(n_components=int(hp["embedding_dim"]))
                if geometry in ("spls", "splsslp"):
                    return SPLSKNN(
                        n_components=int(hp["embedding_dim"]),
                        n_neighbors=int(hp["neighbor_k"]),
                        eta=float(hp.get("pls_decay", 0.0)),
                        n_jobs=allocated_cpus(),
                        slope_weight=(geometry == "splsslp"),
                        local_alpha=loc,
                    )
                return PLSKNN(
                    n_components=int(hp["embedding_dim"]),
                    n_neighbors=int(hp["neighbor_k"]),
                    n_jobs=allocated_cpus(),
                    weight_mode=weight_mode,
                    decay=float(hp.get("pls_decay", 1.0)),
                    local_alpha=loc,
                )

            backtest = MultiStageBacktest(
                residualizer=residualizer,
                regressor_factory=_make_regressor,
                refit_frequency=int(hp["refit_frequency"]),
                feature_fit=feature_fit,
                view_window=int(hp["view_window"]),
                feature_exog=True,
                correction_weight=float(hp.get("omega", 1.0)),
                stack_window=int(hp.get("stack_window", 0)),
            )
            return backtest.run(X_chunk, y_chunk, train_win_periods,
                                desc="spectral_knn")
        if geometry != "euclid":
            state_names = [names[i] for i in state_cols]

            def feature_fit(views, exog):
                return build_geometry_exog(
                    views,
                    exog,
                    kind=geometry,
                    d=int(hp["embedding_dim"]),
                    k_graph=int(hp["graph_k"]),
                    seed=seed,
                    state_cols=state_cols,
                    eigen_solver=hp.get("eigen_solver"),
                    tau=int(hp.get("geometry_tau", PERIODS_PER_DAY)),
                    ladder_base=int(hp.get("ladder_base", 2)),
                    state_names=state_names,
                )
        else:
            def feature_fit(views, exog):
                return build_embedding_exog(
                    views,
                    exog,
                    d=int(hp["embedding_dim"]),
                    k_graph=int(hp["graph_k"]),
                    seed=seed,
                    state_cols=state_cols,
                    eigen_solver=hp.get("eigen_solver"),
                )
    else:
        def feature_fit(views):
            return build_embedding(
                views,
                d=int(hp["embedding_dim"]),
                k_graph=int(hp["graph_k"]),
                seed=seed,
                eigen_solver=hp.get("eigen_solver"),
            )
    _loc = hp.get("local_alpha")
    _loc = None if _loc in (None, 0, 0.0, "") else float(_loc)

    def _final_regressor():
        if _loc is not None:  # local-linear readout in any geometry's coords
            return LocalRidgeKNN(int(hp["neighbor_k"]), _loc, n_jobs=allocated_cpus())
        return KNeighborsRegressor(
            n_neighbors=int(hp["neighbor_k"]),
            weights=gaussian_weights,
            # scheduler-stated entitlement, never os.cpu_count() (-1 would
            # spawn one worker per NODE core against a 4-slot allocation)
            n_jobs=allocated_cpus(),
        )

    backtest = MultiStageBacktest(
        residualizer=residualizer,
        regressor_factory=_final_regressor,
        refit_frequency=int(hp["refit_frequency"]),
        feature_fit=feature_fit,
        view_window=int(hp["view_window"]),
        feature_exog=view_exog,
        correction_weight=float(hp.get('omega', 1.0)),
        stack_window=int(hp.get('stack_window', 0)),
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="spectral_knn")


def run(
    horizon: int = 1,
    train_window: int = 500,
    ridge_alpha: float = 1.0,
    view_window: int = 960,
    embedding_dim: int = 8,
    graph_k: int = 10,
    neighbor_k: int = 25,
    refit_frequency: int = 240,
    exog_cols: str = "",
    seed: int = 42,
    data_path: str = "data",
    output_file: str = "results/spectral_knn/run.json",
    params_file: str = "",
) -> dict:
    """Spectral-kNN walk-forward volatility backtest.

    Composition: Ridge residualizer + spectral embedding of residual windows
    + Gaussian-weighted kNN regression in embedding space. Returns a metrics
    dict. The per-row prediction table is written next to ``output_file`` as
    ``results.csv`` by the shared backtest scaffold.

    Data-prep invariants match the linear-method path (calendar features on,
    diurnal-adjusted RV target winsorized at a 240-period window, leading-edge
    NaN drop, ``prescale=True`` so view residuals are on a sane scale).
    """
    from src.data.loading import parse_exog_cols

    hp: dict = dict(
        DEFAULT_SPECTRAL_KNN_PARAMS,
        ridge_alpha=ridge_alpha,
        view_window=view_window,
        embedding_dim=embedding_dim,
        graph_k=graph_k,
        neighbor_k=neighbor_k,
        refit_frequency=refit_frequency,
        seed=seed,
    )
    if params_file:
        with open(params_file) as fh:
            hp.update(json.load(fh))

    results_csv = str(Path(output_file).with_name("results.csv"))
    run_executor(
        method_name="spectral_knn",
        fit_predict=fit_predict_spectral_knn,
        hyperparams=hp,
        data_path=data_path,
        output_file=results_csv,
        horizon=horizon,
        train_window=train_window,
        start=0,
        end=-1,
        halo=0,
        exog_cols=parse_exog_cols(exog_cols or None),
        segment=None,
        lag_scope="global",
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=True,
        prescale=True,
        seed=seed,
    )
    metrics = calculate_metrics(pd.read_csv(results_csv))
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
