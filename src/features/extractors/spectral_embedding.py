"""Temporal spectral embedding via graph Laplacian eigenmaps.

Pure feature transform: takes a stack of temporal views and returns a
low-dimensional coordinate per view. "Spectral" here refers to the
eigenspectrum of the graph Laplacian, not an FFT of the time series.

The public surface is:

* :func:`build_embedding` — fit the embedding on training views, returns a
  :class:`SpectralBasis`.
* :class:`SpectralBasis` — frozen training-side state; supports
  :meth:`SpectralBasis.embed` (single test view) and
  :meth:`SpectralBasis.embed_batch` (batch).

See the prototype notebook ``notebooks/features/spectral_embedding.ipynb``
for derivation and sanity-checks on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    views: np.ndarray, k_graph: int
) -> tuple[csr_matrix, float, NearestNeighbors]:
    """Build a sparse symmetric k-NN affinity graph with self-tuning bandwidth.

    Each row of ``views`` is one point in :math:`\\mathbb R^W`. The graph
    connects every point to its ``k_graph`` nearest neighbours under
    Euclidean distance, with Gaussian edge weights
    ``w_ij = exp(-||v_i - v_j||^2 / (2 sigma^2))``. The bandwidth ``sigma``
    is the median ``k_graph``-th nearest-neighbour distance — Zelnik-Manor
    & Perona's self-tuning heuristic in its global-scale form.

    Parameters
    ----------
    views
        Array of shape ``(N, W)``.
    k_graph
        Number of nearest neighbours per node.

    Returns
    -------
    W
        Symmetric Gaussian-weighted affinity matrix of shape ``(N, N)``.
    sigma
        The bandwidth used (median k-th NN distance).
    nn_index
        Fitted :class:`~sklearn.neighbors.NearestNeighbors` over ``views``,
        reusable for the Nyström out-of-sample extension.
    """
    n = views.shape[0]
    nn_index = NearestNeighbors(n_neighbors=k_graph + 1).fit(views)
    dists, idxs = nn_index.kneighbors(views)  # (N, k_graph+1); col 0 is self

    sigma = float(np.median(dists[:, k_graph]))

    rows = np.repeat(np.arange(n), k_graph)
    cols = idxs[:, 1:].ravel()
    vals = np.exp(-dists[:, 1:].ravel() ** 2 / (2 * sigma**2))
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize
    return graph, sigma, nn_index


def laplacian_eigenmaps(
    affinity: csr_matrix, d: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-d nontrivial eigenvectors of the symmetric normalized Laplacian.

    Computed as the **top**-d eigenvectors of ``M = D^{-1/2} W D^{-1/2} = I - L``,
    because ``eigsh(M, which='LA')`` (largest-algebraic via Lanczos) is much
    faster than asking for the smallest eigenvalues of ``L`` directly.
    Same eigenvectors, eigenvalues map as ``lambda_L = 1 - lambda_M``.

    The leading eigenvalue is trivial (corresponds to the constant eigenvector
    proportional to ``sqrt(degree)``) and is dropped.

    Parameters
    ----------
    affinity
        Symmetric sparse affinity matrix from :func:`build_knn_graph`.
    d
        Number of embedding dimensions.
    seed
        Seed for the Lanczos starting vector — Lanczos is deterministic given
        a fixed start, so this controls reproducibility across rebuilds.

    Returns
    -------
    phi
        Embedding coordinates, shape ``(N, d)``.
    eigvals_L
        Corresponding Laplacian eigenvalues (ascending). Useful for sanity
        checks: a near-zero value here usually means the graph has more than
        one connected component.
    """
    n = affinity.shape[0]
    deg = np.asarray(affinity.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1e-12)
    d_inv_sqrt = diags(1.0 / np.sqrt(deg_safe))

    m = d_inv_sqrt @ affinity @ d_inv_sqrt  # m = I - L_sym

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    eigvals_m, eigvecs = eigsh(m, k=d + 1, which="LA", v0=v0)

    order = np.argsort(-eigvals_m)  # descending in M <-> ascending in L
    eigvals_m = eigvals_m[order]
    eigvecs = eigvecs[:, order]

    phi = eigvecs[:, 1:]  # drop trivial top eigvec of M
    eigvals_l = 1.0 - eigvals_m[1:]
    return phi, eigvals_l


def nystrom_extend(
    v_test: np.ndarray,
    views_train: np.ndarray,
    phi_train: np.ndarray,
    sigma: float,
    k_graph: int,
    nn_index: NearestNeighbors | None = None,
) -> np.ndarray:
    """Extend a single test view to the embedding space.

    Practical out-of-sample extension (Bengio et al. 2003): find the
    ``k_graph`` nearest training views, apply the same Gaussian kernel to
    get edge weights, and return the kernel-weighted average of their
    training embeddings. Falls back to a uniform average if every neighbour
    has zero weight (extreme out-of-distribution test view).

    Parameters
    ----------
    v_test
        Single test view, shape ``(W,)``.
    views_train
        Training views used to build the embedding, shape ``(N, W)``.
    phi_train
        Training embeddings, shape ``(N, d)``.
    sigma
        Gaussian bandwidth from training.
    k_graph
        Number of training neighbours to use (same value that built the
        graph).
    nn_index
        Optional pre-fitted :class:`NearestNeighbors`; passing this avoids
        re-fitting the index on every call when extending many test views.

    Returns
    -------
    phi_test
        Embedding of ``v_test``, shape ``(d,)``.
    """
    if nn_index is None:
        nn_index = NearestNeighbors(n_neighbors=k_graph).fit(views_train)
    dists, idx = nn_index.kneighbors(v_test[None, :], n_neighbors=k_graph)
    dists = dists.ravel()
    idx = idx.ravel()

    weights = np.exp(-(dists**2) / (2 * sigma**2))
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return phi_train[idx].mean(axis=0)
    weights = weights / weight_sum
    return (weights[:, None] * phi_train[idx]).sum(axis=0)


@dataclass
class SpectralBasis:
    """Frozen training-side embedding state — cheap to extend new views against."""

    views_train: np.ndarray  # (N, W)
    phi_train: np.ndarray  # (N, d)
    eigvals_L: np.ndarray  # (d,)
    sigma: float
    k_graph: int
    nn_index: NearestNeighbors

    def embed(self, v_test: np.ndarray) -> np.ndarray:
        """Embed a single test view via Nyström. Returns shape ``(d,)``."""
        return nystrom_extend(
            v_test,
            self.views_train,
            self.phi_train,
            self.sigma,
            self.k_graph,
            nn_index=self.nn_index,
        )

    def embed_batch(self, V_test: np.ndarray) -> np.ndarray:
        """Embed a batch of test views. Returns shape ``(M, d)``.

        Vectorized: avoids the Python loop over rows that calling
        :meth:`embed` repeatedly would incur.
        """
        dists, idx = self.nn_index.kneighbors(V_test, n_neighbors=self.k_graph)
        weights = np.exp(-(dists**2) / (2 * self.sigma**2))
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
        return np.einsum("mk,mkd->md", weights, self.phi_train[idx])


def build_embedding(
    views: np.ndarray, d: int, k_graph: int, seed: int = 42
) -> SpectralBasis:
    """End-to-end pipeline: views → sparse k-NN graph → Laplacian eigenmaps.

    Parameters
    ----------
    views
        Training views, shape ``(N, W)``.
    d
        Embedding dimensionality.
    k_graph
        Number of neighbours for the k-NN graph.
    seed
        Seed for the Lanczos starting vector (see
        :func:`laplacian_eigenmaps`).
    """
    affinity, sigma, nn_index = build_knn_graph(views, k_graph=k_graph)
    phi, eigvals = laplacian_eigenmaps(affinity, d=d, seed=seed)
    return SpectralBasis(
        views_train=views,
        phi_train=phi,
        eigvals_L=eigvals,
        sigma=sigma,
        k_graph=k_graph,
        nn_index=nn_index,
    )
