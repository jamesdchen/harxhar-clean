"""Temporal spectral embedding (sklearn-backed).

A thin wrapper over :class:`sklearn.manifold.SpectralEmbedding` that gives
:class:`~src.backtest.multi_stage.MultiStageBacktest` the
``.phi_train + .embed(v_test)`` interface it expects. Out-of-sample
extension is a weighted-kNN Nyström over the training views.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors


def allocated_cpus() -> int:
    """Cores this process is ENTITLED to, not the cores the node has.

    ``n_jobs=-1`` expands to ``os.cpu_count()`` — the whole machine. On a
    shared scheduler node that is a 32-core box answering a 4-slot request,
    so every task would spawn 8x the thread pools it is entitled to: bad
    citizenship, and the direct cause of OpenBLAS allocation failures. The
    scheduler states the real entitlement (SGE ``NSLOTS``, SLURM
    ``SLURM_CPUS_PER_TASK``); fall back to 1 rather than to the node size,
    since over-subscribing is the failure mode and under-subscribing is not.
    """
    for var in ("NSLOTS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        raw = os.environ.get(var)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 1


@dataclass
class SpectralBasis:
    """Frozen training-side spectral embedding state.

    Holds the training-side embedding ``phi_train`` plus a fitted
    ``NearestNeighbors`` index over the training views (the NN index
    already retains the views internally in ``_fit_X``).
    ``sklearn.manifold.SpectralEmbedding`` doesn't implement
    ``transform()``, so the NN index drives weighted-kNN Nyström
    extension for new test points.
    """

    phi_train: np.ndarray
    k_graph: int
    nn_index: NearestNeighbors

    def embed(self, v_test: np.ndarray) -> np.ndarray:
        """Embed a single test view via weighted-kNN Nyström. Returns shape (d,)."""
        dists, idx = self.nn_index.kneighbors(v_test[None, :], n_neighbors=self.k_graph)
        dists = dists.ravel()
        idx = idx.ravel()
        sigma = float(np.median(dists)) + 1e-12
        w = np.exp(-(dists**2) / (2 * sigma**2))
        s = w.sum()
        if s <= 0:
            return self.phi_train[idx].mean(axis=0)
        w = w / s
        return (w[:, None] * self.phi_train[idx]).sum(axis=0)

    def embed_batch(self, V_test: np.ndarray) -> np.ndarray:
        """Embed a batch of test views. Returns shape (M, d). Vectorized."""
        dists, idx = self.nn_index.kneighbors(V_test, n_neighbors=self.k_graph)
        sigma = np.median(dists, axis=1, keepdims=True) + 1e-12
        w = np.exp(-(dists**2) / (2 * sigma**2))
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        return np.einsum("mk,mkd->md", w, self.phi_train[idx])


@dataclass
class ExogSpectralBasis:
    """Spectral basis over PATH+STATE views (the exog-aware composition).

    A view is two blocks — the trailing residual/target ``path`` window and
    the ``state`` feature row of the bar the view predicts — reconciled with
    NO free weight:

    * the path is divided by the training window's own IQR (a single causal
      scalar per refit), matching the state block, which ``prescale=True``
      has already rolling-robust-scaled to unit IQR by construction;
    * BOTH blocks are then divided by ONE global ``sqrt(W + d)``, so equal
      weight falls on each COORDINATE and the state block's share of the
      distance is ``d / (W + d)`` — it scales with how much information the
      bucket carries, with no tuning constant. Normalizing each block by its
      OWN dimension instead (the earlier form) equalized the two BLOCKS, which
      gave a 9-dim one-hot calendar state the same total weight as a 960-dim
      path and cut the k-NN graph along day-of-week; see
      :func:`build_embedding_exog` for the measured component counts.

    The path is deliberately UNCENTERED: a common level cancels in the
    difference, so genuine level gaps between views still register (the
    regime-analog property). A degenerate window (``iqr <= 0``) zeroes the
    path block rather than dividing by a floor — no epsilon.
    """

    basis: SpectralBasis
    path_scale: float   # iqr_w * sqrt(W); 0.0 marks a degenerate window
    state_scale: float  # sqrt(d)
    state_cols: np.ndarray | None = None  # column subset the state block keeps

    @property
    def phi_train(self) -> np.ndarray:
        return self.basis.phi_train

    def embed(self, v_test: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """Embed one PATH+STATE test view via the same weighted-kNN Nystrom."""
        x = np.asarray(x_test, dtype=np.float64)
        if self.state_cols is not None:
            x = x[self.state_cols]
        return self.basis.embed(
            compose_views(
                np.asarray(v_test, dtype=np.float64)[None, :],
                x[None, :],
                self.path_scale,
                self.state_scale,
            )[0]
        )


@dataclass
class PassthroughBasis:
    """No spectral projection: the composed PATH+STATE view IS the coordinate.

    Selected by ``d <= 0``. Measured motivation (first refit, bucket=baseline,
    23040 views): neighbours found in the RAW composed space have targets
    15.0% closer than random, versus only 10.3% after projecting to 8
    Laplacian eigenvectors — the projection discards about a third of the
    analog signal. The embedding's distance ratio LOOKS far better (0.026 vs
    0.817) but that is a low-dimensional artefact, not predictive value.
    """

    phi_train: np.ndarray
    path_scale: float
    state_scale: float
    state_cols: np.ndarray | None = None

    def embed(self, v_test: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        x = np.asarray(x_test, dtype=np.float64)
        if self.state_cols is not None:
            x = x[self.state_cols]
        return compose_views(
            np.asarray(v_test, dtype=np.float64)[None, :],
            x[None, :],
            self.path_scale,
            self.state_scale,
        )[0]


def window_iqr(views: np.ndarray) -> float:
    """IQR of the training window's own target values (causal, scale-free).

    ``views[:, -1]`` is each window's final bar, i.e. the window's distinct
    target values — its dispersion, read off only rows the engine handed in.
    """
    last = np.asarray(views, dtype=np.float64)[:, -1]
    q25, q75 = np.percentile(last, (25.0, 75.0))
    return float(q75 - q25)


def compose_views(
    paths: np.ndarray, states: np.ndarray, path_scale: float, state_scale: float
) -> np.ndarray:
    """Concatenate the scaled PATH and STATE blocks into view vectors.

    ``path_scale <= 0`` (degenerate window) zeroes the path block instead of
    dividing by a floor.
    """
    p = np.asarray(paths, dtype=np.float64)
    s = np.asarray(states, dtype=np.float64)
    p = np.zeros_like(p) if path_scale <= 0.0 else p / path_scale
    return np.hstack([p, s / state_scale])


def build_embedding_channels(
    views: np.ndarray,
    X_train: np.ndarray,
    view_idx: np.ndarray,
    d: int,
    k_graph: int,
    seed: int = 42,
    channel_cols: np.ndarray | None = None,
    eigen_solver: str | None = None,
    n_jobs: int | None = None,
) -> "ChannelBasis":
    """MULTI-CHANNEL views: the target's window PLUS exog TRAJECTORIES.

    The exog-row form gives each view a single point-in-time snapshot against
    W points of target history — structurally outgunned. Here each selected
    channel contributes its own W-bar window, so the analog search matches on
    JOINT trajectories (what "analog" should mean).

    Only viable at small W: the composed matrix is
    ``n x (W * (1 + n_channels))`` — at W=48 with 82 channels that is ~725 MB,
    at W=960 it would be ~14.5 GB. Scaling is ONE global sqrt of the total
    composed dimension, exactly as the row form, so no block gets weight by
    accident.
    """
    Xt = np.asarray(X_train, dtype=np.float64)
    cols = (
        np.arange(Xt.shape[1])
        if channel_cols is None
        else np.asarray(channel_cols, dtype=int)
    )
    W = int(np.asarray(views).shape[1])
    idx = np.asarray(view_idx, dtype=int)
    # window each channel: (n_views, W) per channel, hstacked
    offs = np.arange(-W, 0)
    win_rows = idx[:, None] + offs[None, :]          # (n_views, W)
    chans = [Xt[win_rows, c] for c in cols]          # each (n_views, W)
    total_dim = W * (1 + len(cols))
    scale = float(np.sqrt(total_dim))
    iqr = window_iqr(views)
    path_scale = iqr * scale
    p = np.zeros_like(views) if path_scale <= 0 else np.asarray(views) / path_scale
    composed = np.hstack([p] + [c / scale for c in chans])
    basis = (
        None
        if d <= 0
        else build_embedding(
            composed, d=d, k_graph=k_graph, seed=seed,
            eigen_solver=eigen_solver, n_jobs=n_jobs,
        )
    )
    return ChannelBasis(
        basis=basis,
        phi_train=composed if basis is None else basis.phi_train,
        path_scale=float(path_scale),
        scale=scale,
        channel_cols=cols,
        W=W,
    )


@dataclass
class ChannelBasis:
    """Basis over multi-channel PATH views (see :func:`build_embedding_channels`)."""

    basis: SpectralBasis | None
    phi_train: np.ndarray
    path_scale: float
    scale: float
    channel_cols: np.ndarray
    W: int

    def embed(self, v_test: np.ndarray, x_window: np.ndarray) -> np.ndarray:
        v = np.asarray(v_test, dtype=np.float64)
        Xw = np.asarray(x_window, dtype=np.float64)[:, self.channel_cols]
        p = np.zeros_like(v) if self.path_scale <= 0 else v / self.path_scale
        composed = np.concatenate([p] + [Xw[:, j] / self.scale
                                         for j in range(Xw.shape[1])])
        return composed if self.basis is None else self.basis.embed(composed)


def path_ladder(views: np.ndarray, base: int = 2) -> tuple[np.ndarray, list[int]]:
    """Trailing-mean ladder over each view's path window.

    Compresses the raw ``(n, W)`` path block to its HAR-ladder coordinates:
    one trailing mean per rung of ``resolve_har_lags(max_lag=W, base=base)``,
    plus the full-window mean when W itself is not a rung (the window IS the
    view's stated reach, so its mean belongs on the ladder). Causal by
    construction — every rung reads only bars inside the view.

    This is the HAR-basis geometry: the exact multi-scale averaging basis the
    linear incumbents read, applied to the path block, so distance in the
    composed view is distance in HAR space rather than in raw-lag space. It
    buys reach cheaply — W=960 costs 11 coordinates instead of 960 — which
    dissolves the "reach belongs to a W sweep" limitation of the raw path.
    """
    from src.features.extractors.har import resolve_har_lags

    V = np.asarray(views, dtype=np.float64)
    W = V.shape[1]
    rungs = resolve_har_lags(max_lag=W, base=base)
    if rungs[-1] < W:
        rungs.append(W)
    ladder = np.stack([V[:, -s:].mean(axis=1) for s in rungs], axis=1)
    return ladder, rungs


def _rank_cut_eigh(C: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of a covariance with the matrix_rank retention cut.

    Returns ``(eigvals, eigvecs)`` restricted to the numerically-supported
    subspace: eigenvalues above ``(max(n_rows, D) * eps)**2 * lam_max`` — the
    square of the cut :func:`numpy.linalg.matrix_rank` applies to singular
    values, since these are singular values squared. No tunable threshold.
    """
    lam, U = np.linalg.eigh(np.asarray(C, dtype=np.float64))
    lam = lam[::-1]
    U = U[:, ::-1]
    if lam[0] <= 0:
        raise ValueError("degenerate covariance: no positive eigenvalue")
    cut = (max(n_rows, C.shape[0]) * np.finfo(np.float64).eps) ** 2 * lam[0]
    keep = lam > cut
    return lam[keep], U[:, keep]


@dataclass
class GeometryExogBasis:
    """PATH+STATE views under a y-free GEOMETRY, with optional spectral stage.

    The July verdict on this pipeline was a verdict on Euclidean-over-raw
    coordinates, not on analog kNN: the Laplacian (and the kNN metric) is
    entirely determined by the geometry it is fed. This basis swaps that
    geometry for one derived from X alone — never from the target — so the
    unsupervised channel stays uncontaminated:

    * ``kind="har"`` — the path block enters as its trailing-mean ladder
      (:func:`path_ladder`) instead of raw lags: distance in the HAR basis
      the linear incumbents read. Prior content, zero target contact.
    * ``kind="whiten"`` — Mahalanobis by the covariance of one-bar INCREMENTS
      of the composed view: directions the dynamics move slowly are magnified,
      fast/noisy directions shrink. Invariant to invertible linear
      reparameterization of the composed coordinates (the empirical-intrinsic-
      geometry property), so the raw-coordinate arbitrariness this geometry
      answers is removed by construction, not by tuning.
    * ``kind="tica"`` — slow modes: the lag-``tau`` autocorrelation
      eigenproblem in the C0-whitened space. Coordinates are scaled by their
      autocorrelation ``lambda`` (the kinetic map), and only ``lambda > 0``
      modes are kept — zero is the canonical cut, not a threshold. Distance
      is then dominated by the slowest intrinsic variables of the joint
      (path, state) dynamics. MEASURED CAVEAT (2026-07-30, first-chunk
      backtest): the slow modes are near-unit-root, so the coordinate frame
      DRIFTS out of the training cloud in the walk-forward — the in-window
      neighbor gain (9.5% vs 8.1% euclid) reversed into a QLIKE collapse
      (0.452 vs 0.233, mz_beta 0.34) under the identity anchor.
    * ``kind="gcluster"`` — graph-structured compression of the HAR basis
      (the Cucuringu-style signed-correlation-clustering idea, v1): spectral
      clustering of the exog SERIES on the correlation of their bar-level
      innovations over the training window, then the state block is
      compressed to cluster-mean HAR features (``d`` clusters x rungs);
      non-series columns (calendar, availability) pass through untouched.
      Structure-ranked rather than variance-ranked (pca) or relevance-ranked
      (PLS at the regressor seam) — and Y-FREE, so it keeps the
      independent-evidence property the supervised compressions spend.
      Requires ``state_names`` (the selected state columns' names) to parse
      the ``adj_<series>_ma_<rung>`` grid.
    * ``kind="pca"`` — variance-optimal linear compression: the top-``d``
      principal components of the composed views (``d`` is consumed as the
      PCA dimension; no inner spectral stage). The one compression with a
      measured POSITIVE result in this codebase: PCA-d8 at W=960 beat the
      raw view on the kNN readout (0.219 vs 0.171, 2026-07-29) while the
      Laplacian projection lost everywhere. Unlike ``whiten`` it never
      amplifies degenerate directions (no inverse scaling), and unlike the
      Laplacian it cannot discard high-variance predictive structure.

    ``basis`` optionally chains the unchanged spectral stage on TOP of the
    geometry coordinates (``d > 0``), so geometry and projection stay
    separately attributable.
    """

    kind: str
    path_scale: float
    state_scale: float
    state_cols: np.ndarray | None
    phi_train: np.ndarray
    ladder_base: int = 2
    mean: np.ndarray | None = None       # whiten/tica/pca: composed-view mean
    transform: np.ndarray | None = None  # whiten/tica/pca: (D, r) linear map
    state_agg: np.ndarray | None = None  # gcluster: (n_state, m) aggregation
    basis: SpectralBasis | None = None   # optional spectral stage on top

    def _compose(self, v_test: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        x = np.asarray(x_test, dtype=np.float64)
        if self.state_cols is not None:
            x = x[self.state_cols]
        if self.state_agg is not None:
            x = x @ self.state_agg
        v = np.asarray(v_test, dtype=np.float64)[None, :]
        if self.kind == "har":
            v, _ = path_ladder(v, base=self.ladder_base)
        z = compose_views(v, x[None, :], self.path_scale, self.state_scale)[0]
        if self.transform is not None:
            z = (z - self.mean) @ self.transform
        return z

    def embed(self, v_test: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        z = self._compose(v_test, x_test)
        return z if self.basis is None else self.basis.embed(z)


def _cluster_aggregation(
    E: np.ndarray, state_names: list[str], m: int, seed: int
) -> np.ndarray:
    """Aggregation matrix compressing the state block by exog-series clusters.

    Parses the state grid — ``adj_<series>_ma_<rung>`` HAR columns and
    ``<series>_avail`` / ``<series>_active`` indicators — clusters the SERIES
    by spectral clustering on the correlation of their bar-level innovations
    (first differences of each series' ``_ma_1`` column over the training
    window; affinity ``(1+rho)/2``), and returns ``(n_state, out)`` mapping
    each cluster x rung (and cluster x indicator-kind) to the MEAN of its
    members. Non-grid columns (calendar) pass through as themselves. Causal:
    everything is estimated on the rows the engine handed in.
    """
    import re

    from sklearn.cluster import SpectralClustering

    grid: dict[str, dict] = {}
    passthrough: list[int] = []
    for j, nm in enumerate(state_names):
        m_har = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        m_ind = re.match(r"^(.+)_(avail|active)$", nm)
        if m_har:
            grid.setdefault(m_har.group(1), {})[f"r{m_har.group(2)}"] = j
        elif m_ind:
            grid.setdefault(m_ind.group(1), {})[m_ind.group(2)] = j
        else:
            passthrough.append(j)
    series = sorted(s for s in grid if "r1" in grid[s])
    inn = np.diff(np.stack([E[:, grid[s]["r1"]] for s in series], axis=1), axis=0)
    # A series whose innovations have EXACTLY zero variance in this window is
    # structurally absent (median-imputed constant — the late-start case):
    # it carries no dependence information to cluster on, and its correlation
    # is undefined. Such series form one degenerate 'inactive' cluster; their
    # compressed features are in-window constants, which contribute nothing
    # to relative distances. Zero is the canonical cut, not a threshold.
    live_mask = inn.std(axis=0) > 0.0
    live = [s for s, ok in zip(series, live_mask) if ok]
    dead = [s for s, ok in zip(series, live_mask) if not ok]
    if len(live) <= m:
        raise ValueError(
            f"gcluster: {len(live)} live series <= {m} clusters — nothing to compress"
        )
    rho = np.corrcoef(inn[:, live_mask], rowvar=False)
    labels = SpectralClustering(
        n_clusters=m, affinity="precomputed", random_state=seed
    ).fit_predict((1.0 + rho) / 2.0)

    cols: list[np.ndarray] = []
    keys = sorted({k for s in series for k in grid[s]})
    groups = [[s for s, lab in zip(live, labels) if lab == c] for c in range(m)]
    if dead:
        groups.append(dead)
    for members in groups:
        for key in keys:
            idx = [grid[s][key] for s in members if key in grid[s]]
            if not idx:
                continue
            w = np.zeros(len(state_names))
            w[idx] = 1.0 / len(idx)
            cols.append(w)
    for s in grid:  # series with indicators but no r1 rung (non-HAR grids)
        if s not in series:
            passthrough.extend(grid[s].values())
    for j in sorted(passthrough):
        w = np.zeros(len(state_names))
        w[j] = 1.0
        cols.append(w)
    return np.stack(cols, axis=1)


def build_geometry_exog(
    views: np.ndarray,
    exog: np.ndarray,
    kind: str,
    d: int,
    k_graph: int,
    seed: int = 42,
    state_cols: np.ndarray | None = None,
    eigen_solver: str | None = None,
    n_jobs: int | None = None,
    tau: int = 48,
    ladder_base: int = 2,
    state_names: list[str] | None = None,
) -> GeometryExogBasis:
    """PATH+STATE basis under a y-free geometry (see :class:`GeometryExogBasis`).

    Interface-compatible with :func:`build_embedding_exog`; the composed-view
    conventions (state-column selection, window-IQR path scale, ONE global
    sqrt of the total dimension) are reused unchanged, so ``kind`` isolates
    exactly one thing — the geometry. ``tau`` (tica only) is the slow-mode
    lag in bars; the caller's default is one trading day
    (``PERIODS_PER_DAY``), a stated prior, not a tuned constant. The views
    the engine hands in are consecutive bars, which is what licenses the
    one-bar increments in ``whiten`` and the lag-``tau`` pairing in ``tica``.
    """
    E = np.asarray(exog, dtype=np.float64)
    if state_cols is not None:
        state_cols = np.asarray(state_cols, dtype=int)
        E = E[:, state_cols]
    if E.shape[1] == 0:
        raise ValueError(
            "state block is empty — the exog-aware view degenerates to the "
            "1-D path; use view_exog=False instead of an empty state block"
        )
    V = np.asarray(views, dtype=np.float64)
    state_agg = None
    if kind == "gcluster":
        if d <= 0:
            raise ValueError("geometry='gcluster' consumes d as the cluster count")
        if state_names is None or len(state_names) != E.shape[1]:
            raise ValueError(
                "geometry='gcluster' needs state_names matching the selected "
                "state columns to parse the series x rung grid"
            )
        state_agg = _cluster_aggregation(E, list(state_names), int(d), seed)
        E = E @ state_agg
    if kind == "har":
        P, rungs = path_ladder(V, base=ladder_base)
        total = len(rungs) + E.shape[1]
    elif kind in ("whiten", "tica", "pca", "gcluster"):
        P = V
        total = V.shape[1] + E.shape[1]
    else:
        raise ValueError(f"unknown geometry kind: {kind!r}")
    if kind == "pca" and d <= 0:
        raise ValueError("geometry='pca' consumes d as the PCA dimension; need d > 0")
    scale = float(np.sqrt(total))
    path_scale = window_iqr(V) * scale
    composed = compose_views(P, E, path_scale, scale)

    mean = None
    transform = None
    if kind == "whiten":
        inc = np.diff(composed, axis=0)
        inc = inc - inc.mean(axis=0, keepdims=True)
        C = (inc.T @ inc) / max(inc.shape[0] - 1, 1)
        lam, U = _rank_cut_eigh(C, inc.shape[0])
        mean = composed.mean(axis=0)
        transform = U / np.sqrt(lam)[None, :]
    elif kind == "tica":
        n = composed.shape[0]
        if tau >= n:
            raise ValueError(f"tica lag tau={tau} >= n_views={n}")
        mean = composed.mean(axis=0)
        Z = composed - mean
        C0 = (Z.T @ Z) / n
        Ct = (Z[:-tau].T @ Z[tau:] + Z[tau:].T @ Z[:-tau]) / (2 * (n - tau))
        lam0, U0 = _rank_cut_eigh(C0, n)
        T0 = U0 / np.sqrt(lam0)[None, :]
        M = T0.T @ Ct @ T0
        M = (M + M.T) / 2
        lam, Wt = np.linalg.eigh(M)
        lam = lam[::-1]
        Wt = Wt[:, ::-1]
        pos = lam > 0  # canonical cut: keep positively-autocorrelated modes
        if not pos.any():
            raise ValueError("tica found no positively-autocorrelated mode")
        # kinetic map: each slow coordinate weighted by its autocorrelation
        transform = (T0 @ Wt[:, pos]) * lam[pos][None, :]
    elif kind == "pca":
        from sklearn.utils.extmath import randomized_svd

        mean = composed.mean(axis=0)
        _, _, Vt = randomized_svd(
            composed - mean, n_components=int(d), random_state=seed
        )
        transform = Vt.T  # (D, d) — plain projection, no whitening

    if kind == "pca":
        # d was the PCA dimension; the inner spectral stage does not apply.
        phi = (composed - mean) @ transform
        return GeometryExogBasis(
            kind=kind,
            path_scale=float(path_scale),
            state_scale=scale,
            state_cols=state_cols,
            phi_train=phi,
            ladder_base=int(ladder_base),
            mean=mean,
            transform=transform,
            basis=None,
        )
    if kind == "gcluster":
        # d was the cluster count; the inner spectral stage does not apply.
        return GeometryExogBasis(
            kind=kind,
            path_scale=float(path_scale),
            state_scale=scale,
            state_cols=state_cols,
            phi_train=composed,
            ladder_base=int(ladder_base),
            state_agg=state_agg,
            basis=None,
        )

    if transform is not None:
        phi = (composed - mean) @ transform
    else:
        phi = composed

    inner = (
        None
        if d <= 0
        else build_embedding(
            phi, d=d, k_graph=k_graph, seed=seed,
            eigen_solver=eigen_solver, n_jobs=n_jobs,
        )
    )
    return GeometryExogBasis(
        kind=kind,
        path_scale=float(path_scale),
        state_scale=scale,
        state_cols=state_cols,
        phi_train=phi if inner is None else inner.phi_train,
        ladder_base=int(ladder_base),
        mean=mean,
        transform=transform,
        basis=inner,
    )


def build_embedding_exog(
    views: np.ndarray,
    exog: np.ndarray,
    d: int,
    k_graph: int,
    seed: int = 42,
    state_cols: np.ndarray | None = None,
    eigen_solver: str | None = None,
    n_jobs: int | None = None,
) -> ExogSpectralBasis:
    """Spectral embedding of PATH+STATE views (see :class:`ExogSpectralBasis`).

    ``views`` are the trailing residual windows (shape ``(n, W)``); ``exog``
    are the aligned feature rows (shape ``(n, d_full)``) the engine passes
    when ``feature_exog=True``. ``state_cols`` selects which of those columns
    the state block keeps — the caller's business, since only it knows what
    the columns MEAN (spectral_knn drops the target-derived ``har_ma_*``
    block, whose information the path already carries at full resolution).
    Delegates to :func:`build_embedding` on the composed matrix, so the graph
    / Laplacian / eigensolver / Nystrom path is byte-identical to the 1-D
    case.
    """
    E = np.asarray(exog, dtype=np.float64)
    if state_cols is not None:
        state_cols = np.asarray(state_cols, dtype=int)
        E = E[:, state_cols]
    W = int(np.asarray(views).shape[1])
    n_state = int(E.shape[1])
    if n_state == 0:
        raise ValueError(
            "state block is empty — the exog-aware view degenerates to the "
            "1-D path; use view_exog=False instead of an empty state block"
        )
    # ONE global sqrt(W + d), not per-block sqrt(W) / sqrt(d). Per-block
    # normalization equalized the two BLOCKS, which handed the tiny state
    # block the same total distance weight as the whole 960-dim path — and
    # since that block is one-hot calendar for a narrow bucket, day-of-week
    # identity outweighed the volatility path and CUT the k-NN graph. Measured
    # at the first refit of task 0 (bucket=baseline, 23040 views, k=10):
    # per-block gave 3 connected components sized [14016, 4512, 4512] whose
    # two smaller components were 100% single-day-of-week (DOW purity 1.000);
    # one global sqrt collapses that to a SINGLE component with DOW purity
    # 0.197 = chance (1/5). A disconnected graph makes the Laplacian's bottom
    # eigenvectors encode component indicators instead of temporal geometry —
    # sklearn says so itself ("Graph is not fully connected, spectral
    # embedding may not work as expected"), and it fired at every refit.
    # Equal weight now falls on each COORDINATE, so the state block's share is
    # d/(W+d) — 0.9% for `baseline` (9 dims), 35% for `all_features` (525) —
    # i.e. it scales with how much information the bucket actually carries,
    # with no tuning constant.
    scale = float(np.sqrt(W + n_state))
    path_scale = window_iqr(views) * scale
    state_scale = scale
    composed = compose_views(views, E, path_scale, state_scale)
    if d <= 0:  # bypass the spectral projection entirely (see PassthroughBasis)
        return PassthroughBasis(
            phi_train=composed,
            path_scale=float(path_scale),
            state_scale=state_scale,
            state_cols=state_cols,
        )
    return ExogSpectralBasis(
        basis=build_embedding(
            composed,
            d=d,
            k_graph=k_graph,
            seed=seed,
            eigen_solver=eigen_solver,
            n_jobs=n_jobs,
        ),
        path_scale=float(path_scale),
        state_scale=state_scale,
        state_cols=state_cols,
    )


def build_embedding(
    views: np.ndarray,
    d: int,
    k_graph: int,
    seed: int = 42,
    eigen_solver: str | None = None,
    n_jobs: int | None = None,
) -> SpectralBasis:
    """Spectral embedding of temporal views.

    Delegates to :class:`sklearn.manifold.SpectralEmbedding` with binary
    ``affinity='nearest_neighbors'`` (k = ``k_graph``) and ARPACK
    eigensolver. The fitted ``NearestNeighbors`` index over ``views`` is
    cached on the returned :class:`SpectralBasis` so out-of-sample test
    points can be embedded via Nyström without re-fitting anything.
    """
    # eigen_solver=None keeps sklearn's default (ARPACK) — byte-identical to
    # every prior call. "amg" is lobpcg WITH an algebraic-multigrid
    # preconditioner (requires pyamg) and is the one that actually wins on a
    # CONNECTED Laplacian: measured on the real first-refit matrix
    # (23040 views x 969 dims, k=10) amg took 19.8s LOCALLY versus ARPACK
    # >22min on a faster cluster node — a 60x+ speedup, output finite and
    # sanely scaled. Bare "lobpcg" is unpreconditioned and on a small
    # spectral gap converges slowly or poorly, so it is not the default win.
    # NOTE a different solver returns the same invariant subspace but not
    # bit-identical eigenvectors (sign / rotation / tolerance differ), so
    # downstream kNN neighbours can shift slightly.
    # n_jobs parallelizes the k-NN GRAPH build (and the Nystrom index below).
    # The eigensolve itself does NOT parallelize — AMG setup and V-cycles are
    # single-threaded scipy.sparse matvecs, and lobpcg's dense Rayleigh-Ritz
    # runs on a tiny (n x d) block — so the requested cores would otherwise
    # sit idle (measured 1.13 cores used against a 4-slot ask). Neighbour
    # search is deterministic, so n_jobs changes speed only, never results.
    # None resolves to the SCHEDULER-STATED entitlement, never os.cpu_count().
    jobs = allocated_cpus() if n_jobs is None else n_jobs
    spectral = SpectralEmbedding(
        n_components=d,
        affinity="nearest_neighbors",
        n_neighbors=k_graph,
        random_state=seed,
        eigen_solver=eigen_solver,
        n_jobs=jobs,
    )
    phi_train = spectral.fit_transform(views)
    nn_index = NearestNeighbors(n_neighbors=k_graph, n_jobs=jobs).fit(views)
    return SpectralBasis(
        phi_train=phi_train,
        k_graph=k_graph,
        nn_index=nn_index,
    )
