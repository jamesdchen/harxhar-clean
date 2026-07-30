"""Offline graph-construction diagnostics for the spectral-kNN pipeline.

Answers, without running a walk-forward backtest, the two questions the
spectral_knn ablations raised:

1. **Is the forecast target smooth on the graph?** (spectral energy profile)
   For each candidate graph construction, project the view-target vector onto
   the bottom eigenvectors of the normalized graph Laplacian and record the
   cumulative fraction of target variance captured by the first ``d``
   eigenvectors, against a permutation null. If the profile hugs the null,
   no embedding dimension can retain the signal — the graph (not the
   eigenmaps) is the problem. The Rayleigh quotient f'Lf/f'f (vs the same
   null) is the single-number version.

2. **Do diffusion neighborhoods beat base-metric neighborhoods?**
   (neighbor-swap) On a chronological 80/20 split of the window's views,
   forecast the view target with (a) gaussian-weighted 25-NN in the raw
   representation space and (b) the pipeline's actual mechanism — spectral
   embedding of the train views + heat-kernel Nystrom extension
   (reusing :class:`SpectralBasis`) + gaussian-weighted 25-NN in embedding
   space. If (b) never beats (a), the value is all in the representation and
   the spectral stage is pure loss.

The grid crosses:

* view/target type: ``resid`` (ridge residuals, as the pipeline builds) and
  ``raw`` (identity residualizer — views of adj_RV itself);
* representation:  ``raw960`` (plain delay vector), ``rawW240`` / ``rawW48``
  (shorter delay windows, same targets), ``recw`` (recency-weighted delay
  vector, exp half-life in bars), ``multiscale`` (HAR-style means over the
  last 1/8/48/240/960 bars, per-coordinate standardized), ``har`` (the
  prescaled HAR+calendar feature rows themselves — the plain-kNN model's
  space, so its spectral cells test "embed the HAR features" directly);
* edge weighting:  ``binary`` (current pipeline) vs ``heat`` (self-tuning
  Zelnik-Manor kernel, sigma_i = distance to 7th neighbour).

Data prep is bit-identical to ``spectral_knn.run``: same ``load_and_transform``
flags, HAR burn-in drop, and whole-series rolling robust prescale.

Usage::

    python diag_spectral_graph.py                # full run, artifacts under
                                                 # results/spectral_graph_diagnostics/
    python diag_spectral_graph.py --stride 2     # 4x cheaper kNN (subsampled views)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from src.backtest.executor import (
    _backtest_and_save,  # noqa: F401  (documents provenance of the prep below)
    _build_har_and_calendar,
    _build_scale_guards,
    load_and_transform,
)
from src.features.extractors.har import resolve_har_lags
from src.features.extractors.spectral_embedding import SpectralBasis
from src.features.transforms.scaling import rolling_robust_scale
from src.models.knn import gaussian_weights

PERIODS_PER_DAY = 48
TRAIN_WIN = 500 * PERIODS_PER_DAY  # pipeline default: 24,000 bars
W = 960                            # view window (20 days), pipeline default
K_GRAPH = 10                       # pipeline default
NEIGHBOR_K = 25                    # pipeline default
D_EMBED = (8, 16)                  # pipeline default d=8, plus one richer cell
N_EIG = 64                         # eigenvectors for the energy profile
MULTISCALE_WINDOWS = (1, 8, 48, 240, 960)
DYADIC_WINDOWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 480, 960)
DETAIL_SCALES = DYADIC_WINDOWS[:-1]  # trends need 2*s <= W
RECW_HALFLIFE = 240                # bars (5 days)
SELF_TUNE_M = 7                    # Zelnik-Manor: sigma_i = dist to 7th neighbour


# ---------------------------------------------------------------------------
# Data prep (mirrors spectral_knn.run -> run_executor exactly)
# ---------------------------------------------------------------------------


def prepare_series(
    data_path: str, return_df: bool = False
) -> tuple[np.ndarray, np.ndarray, pd.Series] | tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """Return (X_prescaled, y, dates) exactly as the spectral_knn executor sees them.

    With ``return_df=True`` also returns the aligned post-burn-in DataFrame
    (raw merged columns included) for building alt-data representations.
    """
    df, _ = load_and_transform(
        data_path,
        [],
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=True,
    )
    df, feature_names = _build_har_and_calendar(df, [], add_calendar=True)
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]

    ref_iqr, fixed_cols = _build_scale_guards(X, df, feature_names)
    X = rolling_robust_scale(X, TRAIN_WIN, ref_iqr=ref_iqr, fixed_cols=fixed_cols)
    if return_df:
        return X, y, dates, df
    return X, y, dates


def window_residuals(X_win: np.ndarray, y_win: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge residuals over one training window — the pipeline's view source
    at a refit step (one ridge fit on the window, residuals in-window)."""
    ridge = Ridge(alpha=alpha).fit(X_win, y_win)
    return y_win - ridge.predict(X_win)


def view_index(n: int, stride: int) -> np.ndarray:
    """Shared target positions (anchored at W so every cell — any window
    length, or HAR features — predicts the identical target set)."""
    return np.arange(W, n, stride)


def delay_views(series: np.ndarray, idx: np.ndarray, w: int) -> np.ndarray:
    """Sliding delay views exactly as MultiStageBacktest builds them:
    view_j = series[j-w:j] for the shared target positions ``idx``."""
    return np.stack([series[j - w : j] for j in idx]).astype(np.float32)


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


def represent(views: np.ndarray, kind: str) -> np.ndarray:
    """Map raw delay views to the representation the graph is built on.

    Every representation is a fixed linear map, so Euclidean distance in the
    output equals a (weighted / projected) Euclidean distance on the views.
    """
    if kind == "raw960":
        return views
    if kind == "recw":
        # coordinate c holds the value (W - c) bars before the target
        age = W - np.arange(W, dtype=np.float32)
        wts = np.exp(-np.log(2.0) * age / RECW_HALFLIFE)
        return views * np.sqrt(wts)[None, :]
    if kind == "multiscale":
        return _standardize(
            np.stack([views[:, -w:].mean(axis=1) for w in MULTISCALE_WINDOWS], axis=1)
        )
    if kind == "dyadic":
        # log-uniform ladder of trailing means: one coordinate per octave
        return _standardize(
            np.stack([views[:, -w:].mean(axis=1) for w in DYADIC_WINDOWS], axis=1)
        )
    if kind == "msdetail":
        # Haar-like edge pyramid: coarse level + scale-local trends
        # d_s = mean(last s) - mean(prior s): "is vol rising at scale s"
        cols = [views.mean(axis=1)]
        for s in DETAIL_SCALES:
            cols.append(
                views[:, -s:].mean(axis=1) - views[:, -2 * s : -s].mean(axis=1)
            )
        return _standardize(np.stack(cols, axis=1))
    if kind == "ms_plus_w48":
        # multiscale block + recent 48-bar path block at equal total weight
        ms = represent(views, "multiscale")
        w48 = _standardize(views[:, -48:]) * np.sqrt(ms.shape[1] / 48.0)
        return np.concatenate([ms, w48], axis=1).astype(np.float32)
    raise ValueError(kind)


def _standardize(Z: np.ndarray) -> np.ndarray:
    sd = Z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (Z / sd[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Graphs and Laplacians
# ---------------------------------------------------------------------------


def knn_distances(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # brute force: tree indexes degrade badly in high dimension
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(Z)
    dists, idx = nn.kneighbors(Z)
    return dists[:, 1:], idx[:, 1:]  # drop self


def build_adjacency(
    dists: np.ndarray, idx: np.ndarray, n: int, weighting: str
) -> sp.csr_matrix:
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    if weighting == "binary":
        vals = np.ones(rows.shape[0], dtype=np.float64)
    elif weighting == "heat":
        # Zelnik-Manor & Perona self-tuning: w_ij = exp(-d_ij^2 / (sigma_i * sigma_j))
        sigma = dists[:, SELF_TUNE_M - 1].astype(np.float64)
        sigma[sigma < 1e-12] = 1e-12
        d2 = dists.astype(np.float64) ** 2
        vals = np.exp(-d2 / (sigma[:, None] * sigma[idx])).ravel()
    else:
        raise ValueError(weighting)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def normalized_laplacian_basis(A: sp.csr_matrix, n_eig: int):
    """Bottom eigenpairs of L_sym = I - D^-1/2 A D^-1/2 (orthonormal basis)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg <= 0] = 1e-12
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    S = sp.diags(d_inv_sqrt) @ A @ sp.diags(d_inv_sqrt)
    S = (S + S.T) * 0.5
    k = min(n_eig + 1, A.shape[0] - 2)
    vals, vecs = eigsh(S, k=k, which="LA")
    order = np.argsort(-vals)  # largest of S == smallest of L_sym
    lam = 1.0 - vals[order]
    U = vecs[:, order]
    return lam, U


def laplacian_eigenmap(A: sp.csr_matrix, d: int) -> np.ndarray:
    """Laplacian-eigenmaps coordinates (sklearn-equivalent: generalized
    eigenvectors D^-1/2 u of the normalized Laplacian, first mode dropped) —
    computed with plain eigsh, which is far faster than sklearn's
    shift-invert path on graphs this size."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg <= 0] = 1e-12
    _, U = normalized_laplacian_basis(A, d + 1)
    return (U[:, 1 : d + 1] / np.sqrt(deg)[:, None]).astype(np.float64)


# ---------------------------------------------------------------------------
# Experiment 1+2: spectral energy profile + Rayleigh quotient (with nulls)
# ---------------------------------------------------------------------------


def energy_profile(U: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Cumulative fraction of variance of f captured by eigenvectors 2..d+1
    (the first, quasi-constant mode is skipped; f is centered)."""
    fc = f - f.mean()
    denom = float(fc @ fc)
    coefs = U.T @ fc
    frac = coefs**2 / denom
    return np.cumsum(frac[1:])  # skip trivial mode


def rayleigh(A: sp.csr_matrix, f: np.ndarray) -> float:
    """f' L_sym f / f' f with f centered (null uses the same convention)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg <= 0] = 1e-12
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    g = f - f.mean()
    Sg = d_inv_sqrt * (A @ (d_inv_sqrt * g))
    return float((g @ g - g @ Sg) / (g @ g))


def run_energy_experiment(A, U, f, n_perm: int, rng) -> dict:
    prof = energy_profile(U, f)
    null_profs = np.empty((n_perm, prof.shape[0]))
    null_rq = np.empty(n_perm)
    for p in range(n_perm):
        fp = rng.permutation(f)
        null_profs[p] = energy_profile(U, fp)
        null_rq[p] = rayleigh(A, fp)
    return {
        "profile": prof,
        "null_mean": null_profs.mean(axis=0),
        "null_std": null_profs.std(axis=0),
        "rayleigh": rayleigh(A, f),
        "rayleigh_null_mean": float(null_rq.mean()),
        "rayleigh_null_std": float(null_rq.std()),
    }


# ---------------------------------------------------------------------------
# Experiment 3: neighbor-swap (base-metric kNN vs embed+Nystrom+kNN)
# ---------------------------------------------------------------------------


def knn_forecast(Z_tr, t_tr, Z_te, k: int) -> np.ndarray:
    model = KNeighborsRegressor(
        n_neighbors=k, weights=gaussian_weights, algorithm="brute"
    )
    model.fit(Z_tr, t_tr)
    return model.predict(Z_te)


def neighbor_swap(Z: np.ndarray, targets: np.ndarray, weighting: str, seed: int) -> dict:
    """Chronological 80/20 split; compare base-metric kNN vs the pipeline's
    embed-and-Nystrom kNN, on the identical train pool."""
    n = Z.shape[0]
    split = int(0.8 * n)
    Z_tr, Z_te = Z[:split], Z[split:]
    t_tr, t_te = targets[:split], targets[split:]

    out: dict = {}
    mse_mean = float(np.mean((t_te - t_tr.mean()) ** 2))
    out["mse_const"] = mse_mean

    pred_base = knn_forecast(Z_tr, t_tr, Z_te, NEIGHBOR_K)
    out["mse_base_metric"] = float(np.mean((t_te - pred_base) ** 2))

    dists, idx = knn_distances(Z_tr, K_GRAPH)
    A = build_adjacency(dists, idx, split, weighting)
    n_comp, _ = connected_components(A, directed=False)
    out["n_components"] = int(n_comp)

    nn_index = NearestNeighbors(n_neighbors=K_GRAPH, algorithm="brute").fit(Z_tr)
    for d in D_EMBED:
        phi_train = laplacian_eigenmap(A, d)
        basis = SpectralBasis(phi_train=phi_train, k_graph=K_GRAPH, nn_index=nn_index)
        phi_test = basis.embed_batch(Z_te)
        pred_emb = knn_forecast(phi_train, t_tr, phi_test, NEIGHBOR_K)
        out[f"mse_embed_d{d}"] = float(np.mean((t_te - pred_emb) ** 2))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--out", default="results/spectral_graph_diagnostics")
    ap.add_argument("--stride", type=int, default=2,
                    help="view subsampling stride (1 = every bar)")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--anchors", default="0.5,0.95",
                    help="window end positions as fractions of the usable range")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Preparing series (pipeline-identical prep)...")
    X, y, dates = prepare_series(args.data_path)
    n = len(y)
    print(f"{n} bars, {dates.iloc[0]} .. {dates.iloc[-1]}")

    anchors = [float(a) for a in args.anchors.split(",")]
    profile_rows: list[dict] = []
    scalar_rows: list[dict] = []
    swap_rows: list[dict] = []

    for a_frac in anchors:
        t_end = int(TRAIN_WIN + a_frac * (n - TRAIN_WIN))
        t0 = t_end - TRAIN_WIN
        X_win, y_win = X[t0:t_end], y[t0:t_end]
        win_label = f"{pd.Timestamp(dates.iloc[t0]).date()}..{pd.Timestamp(dates.iloc[t_end - 1]).date()}"
        print(f"\n=== window {win_label} (anchor {a_frac}) ===")

        series_by_type = {
            "resid": window_residuals(X_win, y_win),
            "raw": y_win,
        }

        for vtype, series in series_by_type.items():
            idx = view_index(len(series), args.stride)
            targets = series[idx].astype(np.float64)
            views960 = delay_views(series, idx, W)
            cell_matrices = {
                "raw960": views960,
                "recw": represent(views960, "recw"),
                "multiscale": represent(views960, "multiscale"),
                "rawW240": delay_views(series, idx, 240),
                "rawW48": delay_views(series, idx, 48),
                "dyadic": represent(views960, "dyadic"),
                "msdetail": represent(views960, "msdetail"),
                "ms_plus_w48": represent(views960, "ms_plus_w48"),
                # HAR + calendar feature rows (already prescaled upstream) at
                # the same target positions — the graph the plain-kNN model
                # implicitly uses; its spectral cells test "embed the HAR
                # features" directly.
                "har": X_win[idx].astype(np.float32),
            }
            for repr_kind, Z in cell_matrices.items():
                tic = time.time()
                dists, idx = knn_distances(Z, K_GRAPH)
                print(f"  kNN [{vtype}/{repr_kind}] {Z.shape} "
                      f"({time.time() - tic:.0f}s)")
                for weighting in ("binary", "heat"):
                    cell = dict(window=win_label, views=vtype,
                                representation=repr_kind, weighting=weighting)
                    A = build_adjacency(dists, idx, Z.shape[0], weighting)
                    n_comp, _ = connected_components(A, directed=False)
                    lam, U = normalized_laplacian_basis(A, N_EIG)
                    res = run_energy_experiment(A, U, targets, args.n_perm, rng)
                    for d_i, (c, nm, ns) in enumerate(
                        zip(res["profile"], res["null_mean"], res["null_std"]), start=1
                    ):
                        profile_rows.append({**cell, "d": d_i, "captured": c,
                                             "null_mean": nm, "null_std": ns})
                    z = ((res["rayleigh_null_mean"] - res["rayleigh"])
                         / max(res["rayleigh_null_std"], 1e-12))
                    scalar_rows.append({
                        **cell,
                        "n_views": Z.shape[0],
                        "n_components": n_comp,
                        "rayleigh": res["rayleigh"],
                        "rayleigh_null_mean": res["rayleigh_null_mean"],
                        "rayleigh_null_std": res["rayleigh_null_std"],
                        "smoothness_z": z,
                        "captured_d8": float(res["profile"][7]),
                        "null_d8": float(res["null_mean"][7]),
                        "captured_d16": float(res["profile"][15]),
                        "null_d16": float(res["null_mean"][15]),
                    })
                    print(f"    [{weighting}] comps={n_comp} "
                          f"RQ={res['rayleigh']:.4f} (null {res['rayleigh_null_mean']:.4f}"
                          f"±{res['rayleigh_null_std']:.4f}, z={z:+.1f}) "
                          f"d8={res['profile'][7]:.3%} (null {res['null_mean'][7]:.3%})")

                    swap = neighbor_swap(Z, targets, weighting, args.seed)
                    swap_rows.append({**cell, **swap})
                    msg = " ".join(
                        f"{k.replace('mse_', '')}={v:.5f}"
                        for k, v in swap.items() if k.startswith("mse")
                    )
                    print(f"    [swap/{weighting}] {msg}")

    prof_df = pd.DataFrame(profile_rows)
    scal_df = pd.DataFrame(scalar_rows)
    swap_df = pd.DataFrame(swap_rows)
    prof_df.to_csv(os.path.join(args.out, "energy_profiles.csv"), index=False)
    scal_df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    swap_df.to_csv(os.path.join(args.out, "neighbor_swap.csv"), index=False)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump({**vars(args), "train_win": TRAIN_WIN, "W": W,
                   "k_graph": K_GRAPH, "neighbor_k": NEIGHBOR_K,
                   "d_embed": list(D_EMBED), "recw_halflife": RECW_HALFLIFE,
                   "multiscale_windows": list(MULTISCALE_WINDOWS)}, fh, indent=2)

    _plot(prof_df, scal_df, args.out)
    print(f"\nArtifacts -> {args.out}/")


def _plot(prof_df: pd.DataFrame, scal_df: pd.DataFrame, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = prof_df["window"].unique()
    fig, axes = plt.subplots(
        len(windows), 2, figsize=(12, 4.5 * len(windows)), squeeze=False
    )
    for r, win in enumerate(windows):
        for c, vtype in enumerate(("resid", "raw")):
            ax = axes[r][c]
            sub = prof_df[(prof_df.window == win) & (prof_df.views == vtype)]
            for (repr_kind, weighting), g in sub.groupby(["representation", "weighting"]):
                g = g.sort_values("d")
                ax.plot(g.d, g.captured, label=f"{repr_kind}/{weighting}")
            if len(sub):
                gnull = sub[(sub.representation == "raw960") & (sub.weighting == "binary")].sort_values("d")
                ax.fill_between(gnull.d, gnull.null_mean - 2 * gnull.null_std,
                                gnull.null_mean + 2 * gnull.null_std,
                                color="gray", alpha=0.4, label="perm. null ±2σ")
            ax.set_title(f"{vtype} views — {win}")
            ax.set_xlabel("d (eigenvectors kept)")
            ax.set_ylabel("target variance captured")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "energy_profiles.png"), dpi=130)


if __name__ == "__main__":
    main()
