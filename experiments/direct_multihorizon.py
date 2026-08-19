"""Direct multi-horizon forecasts: t+1, ..., t+K from the features at t.

At every bar t, regress the k-bars-ahead per-bar target on the SAME
design the paper's one-step models use -- the HAR backbone (52 cols)
and the wide exogenous block (1,092 cols), as they stand at t -- with
the paper's exact estimator: block-diagonal ridge, alpha=(1,100) via
column scaling, unpenalized intercept, 24,000-bar rolling window. One
left-hand side per horizon k = 1..K, one design, one solver.

The a0 arm uses the backbone only (OLS-HAR incumbent: alpha_1 -> 0 is
the paper's OLS; here ridge at the same alpha_1=1 as the two-block
design's block 1 -- the paper's "differs from OLS-HAR only by that
nominal penalty" -- so the two arms differ ONLY by the exogenous block).

Throughput design:
  * the rolling gram X'X is horizon-independent -- one rank-1 roll per
    bar serves all K horizons; only X'y_k differs. Per solve: one
    Cholesky, K back-solves.
  * refit every REFIT bars (default 48 = daily); coefficients held in
    between. The paper's per-bar path is REFIT=1; the daily refit is a
    documented throughput compromise for a 24,000-bar window.
  * time-sharded: --shard i/N splits the evaluation range into N
    contiguous blocks; each process warms its own window. Shards are
    stitched by direct_multihorizon_reduce.py.

Targets: the paper's fit target is y = winsorised sqrt(RV/B) with
baseline B (per-bar diurnal). Horizon-k target at row t is y[t+k]; the
raw evaluation target is rv_raw[t+k]; forecast on the raw scale is
(yhat_k^2) * B[t+k] with the paper's causal calibrated second-moment
stack applied by the reducer (here we persist yhat, y, B, rv_raw per
horizon so the reducer can score under the paper's contract).

Usage:
  python experiments/direct_multihorizon.py --model blk2 --shard 0/6
  python experiments/direct_multihorizon.py --model a0   --shard 0/6
Outputs results/direct_mh/{model}/shard_{i}of{N}.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

K = 11
WINDOW = 24_000
ALPHA_HAR = 1.0
ALPHA_EXOG = 100.0
BURN = 3125  # panel burn-in already applied by the loader; kept for reference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["a0", "blk2"], required=True)
    ap.add_argument("--shard", default="0/1", help="i/N contiguous time shards")
    ap.add_argument("--refit", type=int, default=48, help="re-solve every N bars")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "direct_mh"))
    ap.add_argument(
        "--ols",
        action="store_true",
        help="a0 only: unpenalized OLS on the backbone (paper's incumbent)",
    )
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))

    t0 = time.time()
    import src.unification as U

    p = U._load_panel()
    X_all = p.X
    y_all = p.y
    B_all = p.baseline
    rv_all = p.rv_raw
    t_all = p.t
    names = p.names
    n, _ = X_all.shape
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
    if a.model == "a0":
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
    X = np.ascontiguousarray(X_all[:, cols] * scale[None, :], dtype=np.float64)
    pdim = X.shape[1]
    print(
        f"[{a.model} {a.shard}] panel {n}x{pdim} loaded {time.time() - t0:.0f}s",
        flush=True,
    )

    # horizon targets: Y[:, k-1] = y[t+k]; valid rows need t+K < n
    Y = np.full((n, K), np.nan)
    for k in range(1, K + 1):
        Y[: n - k, k - 1] = y_all[k:]

    # evaluation range: rows [WINDOW, n-K) split into sn contiguous shards
    lo_all, hi_all = WINDOW + K, n - K  # first row whose causal window fits
    edges = np.linspace(lo_all, hi_all, sn + 1).astype(int)
    lo, hi = int(edges[si]), int(edges[si + 1])
    print(
        f"[{a.model} {a.shard}] eval rows [{lo},{hi}) of [{lo_all},{hi_all})",
        flush=True,
    )

    # sufficient statistics over the CAUSAL window for row lo: rows
    # [lo-K-WINDOW, lo-K). A row's horizon-k target is y[row+k]; the latest
    # target in this window is y[lo-K-1+K] = y[lo-1] < y[lo+1], so nothing at
    # or after the forecast row's own first target is used.
    Xw = X[lo - K - WINDOW : lo - K]
    Yw = Y[lo - K - WINDOW : lo - K]
    Sxx = Xw.T @ Xw
    Sxy = Xw.T @ np.nan_to_num(Yw)  # (p, K)
    sx = Xw.sum(0)
    sy = np.nan_to_num(Yw).sum(0)  # (K,)
    nw = float(WINDOW)
    # (rows in the window with any NaN horizon target are rare at the very end only)

    yhat = np.full((hi - lo, K), np.nan)
    coef = None
    icpt = None
    ridge_I = np.eye(pdim) * (1e-8 if (a.ols and a.model == "a0") else 1.0)
    t1 = time.time()
    for i, r in enumerate(range(lo, hi)):
        if i % a.refit == 0:
            gram = Sxx - np.outer(sx, sx) / nw
            rhs = Sxy - np.outer(sx, sy) / nw  # (p, K)
            L = np.linalg.cholesky(gram + ridge_I)
            coef = np.linalg.solve(L.T, np.linalg.solve(L, rhs))  # (p, K)
            icpt = sy / nw - (sx @ coef) / nw  # (K,)
        yhat[i] = X[r] @ coef + icpt
        # roll: add row r, drop row r-WINDOW  (row r's targets are y[r+1..r+K],
        # which are known once we move to r+K; we roll with a K-bar lag to
        # stay causal: the window used at row r contains rows <= r-K)
        rin, rout = r - K, r - K - WINDOW
        if rin >= 0 and rout >= 0:
            xi, xo = X[rin], X[rout]
            yi, yo = np.nan_to_num(Y[rin]), np.nan_to_num(Y[rout])
            Sxx += np.outer(xi, xi) - np.outer(xo, xo)
            Sxy += np.outer(xi, yi) - np.outer(xo, yo)
            sx += xi - xo
            sy += yi - yo
        if i % 5000 == 0 and i:
            el = time.time() - t1
            print(
                f"[{a.model} {a.shard}] {i}/{hi - lo}  {el:.0f}s  eta {el / i * (hi - lo - i):.0f}s",
                flush=True,
            )

    os.makedirs(os.path.join(a.out, a.model), exist_ok=True)
    tag = (
        a.model
        + ("_ols" if (a.ols and a.model == "a0") else "")
        + (f"_refit{a.refit}" if a.refit != 48 else "")
    )
    os.makedirs(os.path.join(a.out, tag), exist_ok=True)
    path = os.path.join(a.out, tag, f"shard_{si}of{sn}.npz")
    rows = np.arange(lo, hi)
    np.savez_compressed(
        path,
        rows=rows,
        t=t_all[lo:hi],
        yhat=yhat.astype(np.float32),
        y_true=Y[lo:hi].astype(np.float32),  # sqrt-scale fit targets at t+k
        rv_raw_k=np.stack([rv_all[lo + k : hi + k] for k in range(1, K + 1)], 1).astype(
            np.float64
        ),
        B_k=np.stack([B_all[lo + k : hi + k] for k in range(1, K + 1)], 1).astype(
            np.float64
        ),
        model=a.model,
        K=K,
        window=WINDOW,
        refit=a.refit,
    )
    print(
        f"[{a.model} {a.shard}] wrote {path}  total {time.time() - t0:.0f}s", flush=True
    )


if __name__ == "__main__":
    main()
