"""Brute-force check of direct_multihorizon.py on a finished shard.

At several evaluation rows r, refit the k-horizon ridge FROM SCRATCH on
exactly the window the rolling code claims to use -- rows
[r-K-WINDOW, r-K) with targets y[row+k] -- and compare the prediction
for row r against the shard's stored yhat[r]. Also asserts causality
numerically: the fitted window's last target index must be < r+1
(no target at or after the forecast row's own next bar leaks in).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from direct_multihorizon import ALPHA_EXOG, ALPHA_HAR, K, WINDOW  # noqa: E402


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "a0"
    shard = sys.argv[2] if len(sys.argv) > 2 else "0of40"
    import src.unification as U

    p = U._load_panel()
    z = np.load(os.path.join(ROOT, "results", "direct_mh", model, f"shard_{shard}.npz"))
    rows = z["rows"]
    yhat = z["yhat"].astype(np.float64)
    refit = int(z["refit"])
    names = p.names
    bb = U._backbone_cols(names)
    ex = U._exog_all_cols(names)
    if bb.dtype != bool:
        m = np.zeros(len(names), bool)
        m[bb] = True
        bb = m
    if ex.dtype != bool:
        m = np.zeros(len(names), bool)
        m[ex] = True
        ex = m
    if model == "a0":
        cols = np.where(bb)[0]
        scale = np.full(cols.size, 1.0 / np.sqrt(ALPHA_HAR))
    else:
        cols = np.concatenate([np.where(bb)[0], np.where(ex)[0]])
        scale = np.concatenate(
            [
                np.full(int(bb.sum()), 1.0 / np.sqrt(ALPHA_HAR)),
                np.full(int(ex.sum()), 1.0 / np.sqrt(ALPHA_EXOG)),
            ]
        )
    X = p.X[:, cols] * scale[None, :]
    y = p.y
    n = len(y)
    Y = np.full((n, K), np.nan)
    for k in range(1, K + 1):
        Y[: n - k, k - 1] = y[k:]

    # pick rows that are exactly at a refit boundary (coef freshly solved there)
    lo = int(rows[0])
    picks = [lo + refit * j for j in (0, 3, 17, 60, 120) if lo + refit * j < rows[-1]]
    worst = 0.0
    for r in picks:
        w0, w1 = r - K - WINDOW, r - K  # window rows used at r
        assert w0 >= 0
        Xw = X[w0:w1]
        Yw = np.nan_to_num(Y[w0:w1])
        # causality: the latest target used is Y[w1-1, K-1] = y[w1-1+K] = y[r-1] < y[r+1]  OK
        latest_target_row = (w1 - 1) + K
        assert latest_target_row < r + 1, (latest_target_row, r)
        mu = Xw.mean(0)
        my = Yw.mean(0)
        Xc = Xw - mu
        G = Xc.T @ Xc + np.eye(Xw.shape[1])
        rhs = Xc.T @ (Yw - my)
        coef = np.linalg.solve(G, rhs)
        icpt = my - mu @ coef
        pred = X[r] @ coef + icpt
        got = yhat[r - lo]
        err = float(np.max(np.abs(pred - got)))
        worst = max(worst, err)
        print(
            f"row {r}: max|bruteforce - rolling| over {K} horizons = {err:.2e}   "
            f"(pred[0]={pred[0]:.5f} got[0]={got[0]:.5f}; pred[{K - 1}]={pred[K - 1]:.5f} got[{K - 1}]={got[K - 1]:.5f})",
            flush=True,
        )
    print(f"WORST {worst:.2e}  ({'PASS' if worst < 1e-4 else 'FAIL'})", flush=True)
    # sanity on stored truth alignment
    rv_k = z["rv_raw_k"]
    B_k = z["B_k"]
    r0 = int(rows[0])
    assert np.allclose(rv_k[0], [p.rv_raw[r0 + k] for k in range(1, K + 1)])
    assert np.allclose(B_k[0], [p.baseline[r0 + k] for k in range(1, K + 1)])
    print("stored rv_raw_k / B_k alignment: PASS", flush=True)


if __name__ == "__main__":
    main()
