"""Temporal spectral embedding (sklearn-backed).

A thin wrapper over :class:`sklearn.manifold.SpectralEmbedding` that gives
:class:`~src.backtest.multi_stage.MultiStageBacktest` the
``.phi_train + .embed(v_test)`` interface it expects. Out-of-sample
extension is a weighted-kNN Nyström over the training views.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors


@dataclass
class SpectralBasis:
    """Frozen training-side spectral embedding state.

    ``sklearn.manifold.SpectralEmbedding`` doesn't implement ``transform()``,
    so we cache a ``NearestNeighbors`` index over the training views to do
    weighted-kNN Nyström extension for new test points.
    """

    views_train: np.ndarray
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


def build_embedding(views: np.ndarray, d: int, k_graph: int, seed: int = 42) -> SpectralBasis:
    """Spectral embedding of temporal views.

    Delegates to :class:`sklearn.manifold.SpectralEmbedding` with binary
    ``affinity='nearest_neighbors'`` (k = ``k_graph``) and ARPACK
    eigensolver. The fitted ``NearestNeighbors`` index over ``views`` is
    cached on the returned :class:`SpectralBasis` so out-of-sample test
    points can be embedded via Nyström without re-fitting anything.
    """
    spectral = SpectralEmbedding(
        n_components=d,
        affinity="nearest_neighbors",
        n_neighbors=k_graph,
        random_state=seed,
    )
    phi_train = spectral.fit_transform(views)
    nn_index = NearestNeighbors(n_neighbors=k_graph).fit(views)
    return SpectralBasis(
        views_train=views,
        phi_train=phi_train,
        k_graph=k_graph,
        nn_index=nn_index,
    )
