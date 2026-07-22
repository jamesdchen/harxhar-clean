"""How much QLIKE survives when the reservoir STATE is shrunk — ideas 1-3 (+ amortized) head-to-head.

Isolates the reservoir representation: each arm reads out (rank-1 rolling ridge) on its OWN states
only (p = state dim), so the arms are directly comparable at a matched small state budget K. A
features-only linear ridge is the reference floor. All arms share W / alpha / refit / OOS; alpha is
FIXED (no per-arm OOS selection) so the comparison is fair, not tuned.

Arms:
  base_features   529 causal features                    (the linear ridge floor)
  full_random     D_FULL random reservoir states         (the full reservoir)
  random_K        K   random reservoir states            (just fewer random channels)
  structured_K    K   HAR-timescale-decay states  (idea 1: oriented gates)
  pca_K           K   PCA factors of full_random  (idea 2: compress the big reservoir)
  select_K        K   corr-top-K channels of full_random (idea 3: select + prune)
  trained_K       K   SGD-once (amortized) gates         (oriented by one-time SGD)

Run:  python reservoir_state_reduction.py [cache_root]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from src.evaluation.metrics import apply_duan_smearing  # noqa: E402
from src.backtest.rolling_ridge import roll_predict as _roll_predict  # noqa: E402
from src.models.rg_lru_reservoir import Reservoir, train_reservoir  # noqa: E402

CACHE_ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/covid_imp_rank"
BUCKET = "all_buckets"
W = 12000  # rolling readout window
OOS_LEN = 30000  # OOS bars scored: [W, W+OOS_LEN)
ALPHA = 1.0  # FIXED ridge penalty (fair, untuned)
REFIT = 480
D_FULL = 256  # full reservoir width
K = 64  # reduced state budget
SEED = 0
DEV = torch.device("cpu")


def _qlike(preds, y_oos, base_oos):
    pr, tr = apply_duan_smearing(preds, y_oos, base_oos)
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1.0))


def _std(D):  # causal column standardization from the train window
    mu, sd = D[:W].mean(0), D[:W].std(0)
    return (D - mu) / np.where(sd > 0, sd, 1.0)


def _readout(D, y, oos_hi):
    _, preds = _roll_predict(
        np.ascontiguousarray(_std(D)), y, 0, W, W, oos_hi, ALPHA, REFIT, True, 0
    )
    return preds


def main():
    b = f"{CACHE_ROOT}/{BUCKET}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.asarray(np.load(f"{b}/y.npy"), dtype=np.float64)
    base = np.asarray(np.load(f"{b}/base.npy"), dtype=np.float64)
    N, d_in = X.shape
    oos_hi = min(W + OOS_LEN, N)
    y_oos, base_oos = y[W:oos_hi], base[W:oos_hi]
    n_up = oos_hi  # rows needed

    mu, sd = X[:W].mean(0), X[:W].std(0)
    Xs = ((X[:n_up] - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
    y_up = y[:n_up]
    print(
        f"cache={CACHE_ROOT} d_in={d_in} W={W} OOS=[{W},{oos_hi}) alpha={ALPHA} D_FULL={D_FULL} K={K}",
        flush=True,
    )

    def states(gates, d):
        return Reservoir(d_in, d, 1, 1.0, SEED, gates=gates).states(Xs, DEV)

    reps: dict[str, np.ndarray] = {}
    reps["base_features"] = Xs.astype(np.float64)
    t = time.time()
    Hfull = states("random", D_FULL)
    print(f"  full states {time.time() - t:.0f}s", flush=True)
    reps["full_random"] = Hfull.astype(np.float64)
    reps["random_K"] = states("random", K).astype(np.float64)
    reps["structured_K"] = states("structured", K).astype(np.float64)

    # idea 2 — PCA of the full reservoir (train-window fit, causal), top-K
    Hc = Hfull[:W] - Hfull[:W].mean(0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    reps["pca_K"] = ((Hfull - Hfull[:W].mean(0)) @ Vt[:K].T).astype(np.float64)

    # idea 3 — select top-K reservoir channels by |corr with y| on the train window
    hs = (Hfull[:W] - Hfull[:W].mean(0)) / (Hfull[:W].std(0) + 1e-12)
    ys = (y_up[:W] - y_up[:W].mean()) / (y_up[:W].std() + 1e-12)
    corr = np.abs(hs.T @ ys) / W
    sel = np.argsort(corr)[-K:]
    reps["select_K"] = Hfull[:, sel].astype(np.float64)

    # amortized — SGD-once trained gates, K channels
    t = time.time()
    ymu, ysd = float(y_up[:W].mean()), float(y_up[:W].std() or 1.0)
    tr_res = train_reservoir(
        Xs[:W], ((y_up[:W] - ymu) / ysd).astype(np.float32), d_in, K, 1, DEV, SEED
    )
    reps["trained_K"] = tr_res.states(Xs, DEV).astype(np.float64)
    print(f"  trained (SGD-once) {time.time() - t:.0f}s", flush=True)

    print(f"\n{'arm':<16}{'state':>7}{'QLIKE':>12}{'d_vs_base':>12}", flush=True)
    q_base = None
    rows = []
    for name, D in reps.items():
        t = time.time()
        q = _qlike(_readout(D, y_up, oos_hi), y_oos, base_oos)
        if name == "base_features":
            q_base = q
        rows.append(
            (
                name,
                D.shape[1],
                q,
                q - q_base if q_base is not None else 0.0,
                time.time() - t,
            )
        )
    for name, p, q, d, dt in rows:
        print(f"{name:<16}{p:>7}{q:>12.5f}{d:>+12.5f}   ({dt:.0f}s)", flush=True)
    print(
        "\ndelta<0 => beats the features-only linear base; compare structured/pca/select/trained at state=K.",
        flush=True,
    )


if __name__ == "__main__":
    main()
