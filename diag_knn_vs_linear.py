"""Follow-up diagnostics: what is kNN actually adding (or missing) vs linear?

The graph diagnostics (diag_spectral_graph.py) established that all the
neighbor signal lives in low-dim multi-horizon level coordinates and that
spectral embeddings only subtract. This script asks the questions those runs
left open, on the identical window / targets / chronological 80/20 split:

1. **Linear reference.** Ridge on the HAR feature rows and on the multiscale
   coordinates — is kNN even ahead of the baseline it is meant to challenge?
2. **Supervised metrics.** All representations so far were unsupervised with
   per-coordinate standardization. Here: prognostic-score kNN (1-d metric on
   ridge's fitted value x'beta) and |beta|-weighted coordinates.
3. **Locally-constant vs locally-linear.** kNN averaging cannot extrapolate
   (neighbors bound the prediction); local-linear fits a small ridge on the
   neighbors instead. Includes a k sweep for the smoothing dial.
4. **Failure modes.** MSE by true-target decile (edge/extrapolation bias
   shows up as a top-decile blowout) and the temporal distance distribution
   of selected neighbors (is "analogue matching" just persistence?).

Same prep as the pipeline; reuses builders from diag_spectral_graph.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from diag_spectral_graph import (
    TRAIN_WIN,
    W,
    delay_views,
    prepare_series,
    represent,
    view_index,
)
from src.models.knn import gaussian_weights


def knn_predict(Z_tr, t_tr, Z_te, k: int) -> np.ndarray:
    m = KNeighborsRegressor(n_neighbors=k, weights=gaussian_weights, algorithm="brute")
    return m.fit(Z_tr, t_tr).predict(Z_te)


def local_linear_predict(Z_tr, t_tr, Z_te, k: int, alpha: float = 1.0) -> np.ndarray:
    """Per-query ridge on the k nearest neighbors (LOESS-style, degree 1)."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(Z_tr)
    _, idx = nn.kneighbors(Z_te)
    out = np.empty(len(Z_te))
    for i, nbrs in enumerate(idx):
        m = Ridge(alpha=alpha).fit(Z_tr[nbrs], t_tr[nbrs])
        out[i] = m.predict(Z_te[i : i + 1])[0]
    return out


def mse(t, p) -> float:
    return float(np.mean((t - p) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--out", default="results/spectral_graph_diagnostics")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--anchor", type=float, default=0.5)
    args = ap.parse_args()

    print("Preparing series (pipeline-identical prep)...")
    X, y, dates = prepare_series(args.data_path)
    n = len(y)
    t_end = int(TRAIN_WIN + args.anchor * (n - TRAIN_WIN))
    t0 = t_end - TRAIN_WIN
    X_win, y_win = X[t0:t_end], y[t0:t_end]
    print(f"window {pd.Timestamp(dates.iloc[t0]).date()}"
          f"..{pd.Timestamp(dates.iloc[t_end - 1]).date()}")

    idx = view_index(len(y_win), args.stride)
    targets = y_win[idx].astype(np.float64)
    views960 = delay_views(y_win, idx, W)
    Z_ms = represent(views960, "multiscale")
    Z_har = X_win[idx].astype(np.float64)

    split = int(0.8 * len(idx))
    tr, te = slice(None, split), slice(split, None)
    t_tr, t_te = targets[tr], targets[te]

    rows: list[dict] = []

    def record(name, pred):
        rows.append({"model": name, "mse": mse(t_te, pred)})
        print(f"  {name:34s} {rows[-1]['mse']:.5f}")
        return pred

    print(f"\nconst = {mse(t_te, np.full_like(t_te, t_tr.mean())):.5f}")

    # --- 1. linear references -------------------------------------------
    ridge_har = Ridge(alpha=1.0).fit(Z_har[tr], t_tr)
    record("ridge_har", ridge_har.predict(Z_har[te]))
    ridge_ms = Ridge(alpha=1.0).fit(Z_ms[tr], t_tr)
    record("ridge_multiscale", ridge_ms.predict(Z_ms[te]))

    # --- 2. plain kNN references + k sweep ------------------------------
    for k in (5, 25, 100):
        record(f"knn_multiscale_k{k}", knn_predict(Z_ms[tr], t_tr, Z_ms[te], k))
    record("knn_har_k25", knn_predict(Z_har[tr], t_tr, Z_har[te], 25))

    # --- 3. supervised metrics ------------------------------------------
    s_tr = ridge_har.predict(Z_har[tr])[:, None]
    s_te = ridge_har.predict(Z_har[te])[:, None]
    for k in (25, 100):
        record(f"knn_prognostic_k{k}", knn_predict(s_tr, t_tr, s_te, k))
    wts = np.abs(ridge_har.coef_)
    record("knn_har_betaweighted_k25",
           knn_predict(Z_har[tr] * wts, t_tr, Z_har[te] * wts, 25))

    # --- 4. local-linear -------------------------------------------------
    for k in (25, 100):
        record(f"loclin_multiscale_k{k}",
               local_linear_predict(Z_ms[tr], t_tr, Z_ms[te], k))
    record("loclin_har_k100", local_linear_predict(Z_har[tr], t_tr, Z_har[te], 100))

    # --- 5. failure modes ------------------------------------------------
    pred_knn = knn_predict(Z_ms[tr], t_tr, Z_ms[te], 25)
    pred_lin = ridge_har.predict(Z_har[te])
    dec = np.clip((np.argsort(np.argsort(t_te)) * 10) // len(t_te), 0, 9)
    dec_rows = []
    for d in range(10):
        m = dec == d
        dec_rows.append({"decile": d, "n": int(m.sum()),
                         "mse_knn_ms": mse(t_te[m], pred_knn[m]),
                         "mse_ridge_har": mse(t_te[m], pred_lin[m]),
                         "bias_knn": float(np.mean(pred_knn[m] - t_te[m])),
                         "bias_ridge": float(np.mean(pred_lin[m] - t_te[m]))})
    dec_df = pd.DataFrame(dec_rows)
    print("\nMSE by true-target decile (kNN-multiscale vs ridge-har):")
    print(dec_df.to_string(index=False))

    nn = NearestNeighbors(n_neighbors=25, algorithm="brute").fit(Z_ms[tr])
    _, nidx = nn.kneighbors(Z_ms[te])
    tdist = np.abs(nidx - (split + np.arange(len(t_te)))[:, None]) * args.stride
    q = np.percentile(tdist, [10, 25, 50, 75, 90])
    print(f"\nneighbor temporal distance (bars): "
          f"p10={q[0]:.0f} p25={q[1]:.0f} p50={q[2]:.0f} p75={q[3]:.0f} p90={q[4]:.0f} "
          f"(1 day = 48; window = {TRAIN_WIN})")

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "knn_vs_linear.csv"), index=False)
    dec_df.to_csv(os.path.join(args.out, "knn_vs_linear_deciles.csv"), index=False)
    print(f"\nArtifacts -> {args.out}/knn_vs_linear*.csv")


if __name__ == "__main__":
    main()
