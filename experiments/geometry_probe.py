"""Geometry probe — which Y-FREE geometry keeps predictive analog structure?

Single-window diagnostic (first refit window, bucket=all_features, identity
anchor) comparing the kNN geometries wired into
src.features.extractors.spectral_embedding on two measures, before any
cluster arm is spent:

* neighbor quality — how much closer (%) are the targets of a query's k=25
  nearest neighbors than random draws (the July why-embed-loses metric);
* probe R^2 — Ridge(alpha=1) readout of the target from the training-side
  coordinates, scored on the held-out probe views.

Geometry is fit on the FIT part only (first n-2000 views); the last 2000
views are the probe queries. All geometries are functions of X alone —
targets are read only by the scoring, never by the geometry.

Data prep is the executor's own (load_and_transform + _build_har_and_calendar
+ prescale + horizon shift), CALLED, never re-derived — the byte-identical
invariants of specs/spectral_knn_var.py. The prepared window is cached to the
local scratch npz so re-probes skip the ~minutes prep.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import os
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from src.backtest.executor import (
    _build_har_and_calendar,
    _build_scale_guards,
    load_and_transform,
)
from src.data.loading import get_bucket
from src.features.extractors.har import resolve_har_lags
from src.features.extractors.spectral_embedding import (
    build_embedding_exog,
    build_geometry_exog,
)
from src.features.transforms.scaling import rolling_robust_scale
from src.features.transforms.target import PERIODS_PER_DAY, apply_horizon_shift

BUCKET = "all_features"
TRAIN_WIN = 500 * PERIODS_PER_DAY  # 24000 bars — the spec's first refit window
HORIZON = 1
N_PROBE = 2000
NEIGH_K = 25
CACHE = os.path.join("results", "geometry_probe_window.npz")


def prepare_window() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """First training window (X, y, feature_names) under the spec invariants."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["X"], z["y"], list(z["names"])
    df, adj_exog = load_and_transform(
        "data",
        get_bucket(BUCKET),
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    df, names = _build_har_and_calendar(df, adj_exog, add_calendar=True)
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)
    X = df[names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)
    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, HORIZON)
    ref_iqr, fixed_cols = _build_scale_guards(X, df, names)
    X = rolling_robust_scale(X, TRAIN_WIN, ref_iqr=ref_iqr, fixed_cols=fixed_cols)
    X, y = X[:TRAIN_WIN], y[:TRAIN_WIN]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, names=np.array(names, dtype=object))
    return X, y, names


def make_views(y: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(W, len(y))
    return np.stack([y[j - W : j] for j in idx]), idx


def probe(tag: str, basis, y_fit: np.ndarray, X_probe: np.ndarray,
          v_probe: np.ndarray, y_probe: np.ndarray, rng: np.random.Generator):
    phi_fit = basis.phi_train
    t0 = time.time()
    phi_probe = np.stack(
        [basis.embed(v_probe[i], X_probe[i]) for i in range(len(v_probe))]
    )
    nn = NearestNeighbors(n_neighbors=NEIGH_K).fit(phi_fit)
    _, nbr = nn.kneighbors(phi_probe)
    d_nbr = np.abs(y_fit[nbr] - y_probe[:, None]).mean()
    rand = rng.integers(0, len(y_fit), size=nbr.shape)
    d_rand = np.abs(y_fit[rand] - y_probe[:, None]).mean()
    gain = 1.0 - d_nbr / d_rand

    ridge = Ridge(alpha=1.0).fit(phi_fit, y_fit)
    ss_res = np.sum((y_probe - ridge.predict(phi_probe)) ** 2)
    ss_tot = np.sum((y_probe - y_probe.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(
        f"{tag:<16} dim={phi_fit.shape[1]:>5}  nbr_gain={gain:7.3%}  "
        f"probe_R2={r2:7.4f}  ({time.time() - t0:.0f}s probe)"
    )
    return dict(tag=tag, dim=int(phi_fit.shape[1]), nbr_gain=float(gain),
                probe_r2=float(r2))


def main() -> None:
    X, y, names = prepare_window()
    state_cols = np.array(
        [i for i, nm in enumerate(names) if not nm.startswith("har_ma_")], dtype=int
    )
    print(f"window: {len(y)} bars, {len(names)} features, "
          f"state block {len(state_cols)} cols")
    rng = np.random.default_rng(42)
    rows = []
    for tag, kind, W in [
        ("euclid_w12", None, 12),
        ("har_w12", "har", 12),
        ("har_w960", "har", 960),
        ("whiten_w12", "whiten", 12),
        ("tica_w12", "tica", 12),
        ("tica_w960", "tica", 960),
    ]:
        views, idx = make_views(y, W)
        exog = X[idx]
        n_fit = len(views) - N_PROBE
        t0 = time.time()
        if kind is None:
            basis = build_embedding_exog(
                views[:n_fit], exog[:n_fit], d=0, k_graph=10,
                state_cols=state_cols,
            )
        else:
            basis = build_geometry_exog(
                views[:n_fit], exog[:n_fit], kind=kind, d=0, k_graph=10,
                state_cols=state_cols, tau=PERIODS_PER_DAY,
            )
        build_s = time.time() - t0
        print(f"{tag}: basis built in {build_s:.0f}s")
        rows.append(
            probe(tag, basis, y[idx[:n_fit]], exog[n_fit:], views[n_fit:],
                  y[idx[n_fit:]], rng)
        )
    import pandas as pd

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join("results", "geometry_probe.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
