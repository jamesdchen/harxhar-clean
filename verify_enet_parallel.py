"""Verification of the chunk-parallel rolling-ENet fast path (rolling_enet / bucket_enet).

Synthetic-only: the real all_features bucket lives on CARC (local cache is a dangling symlink), so the
machinery is checked here and the real-bucket QLIKE run happens on-cluster. Checks:

  1. ``parallel_enet`` serial (n_jobs=1) == the incumbent ``_cadence_enet_har_unpen`` BIT-FOR-BIT
     (same warm-started CD chain over the same refit grid).
  2. The empty-HAR path == sklearn ElasticNet over all columns (plain centered enet).
  3. ``parallel_enet`` n_jobs>1 agrees with serial to CD tolerance (chunk cold seeds converge to the
     same optima as the incumbent's warm continuation).
  4. (``--scale``) real-scale timing, N=75k / p=529 / W=24000 / refit=480: serial vs 8 procs.

Run:  python verify_enet_parallel.py [--scale]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from src.backtest.rolling_enet import parallel_enet, refit_chain


def _gen(
    n: int, p: int, k_har: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic design: ``k_har`` HAR-ish cols (cumsum'd, collinear) + exog, y = HAR + sparse exog + noise."""
    rng = np.random.default_rng(seed)
    H = np.cumsum(rng.normal(0, 0.1, (n, k_har)), axis=0)
    E = rng.normal(0, 1, (n, p - k_har))
    X = np.concatenate([H, E], axis=1)
    bh = rng.normal(0.5, 0.2, k_har)
    be = np.zeros(p - k_har)
    nz = rng.choice(p - k_har, size=max(1, (p - k_har) // 10), replace=False)
    be[nz] = rng.normal(0, 0.3, nz.size)
    y = H @ bh + E @ be + rng.normal(0, 0.5, n)
    return X, y, np.arange(k_har)


def test_fwls_vs_incumbent() -> None:
    """Test 1: serial rolling_enet IS the incumbent chain — bit-for-bit identical predictions."""
    from resid_amortized import _cadence_enet_har_unpen

    n, p, k = 3000, 20, 6
    W, refit, alpha, l1 = 1000, 500, 0.001, 0.2
    X, y, har_idx = _gen(n, p, k, seed=1)
    har_mask = np.zeros(p, dtype=bool)
    har_mask[har_idx] = True  # incumbent takes a full-length bool mask
    starts_r, coefs, intercepts = _cadence_enet_har_unpen(
        X, y, W, refit, har_mask=har_mask, alpha=alpha, l1=l1
    )
    ref = np.empty(n - W, dtype=np.float64)
    for t_r, beta, ic in zip(starts_r, coefs, intercepts):
        hi = min(t_r + refit, n)
        ref[t_r - W : hi - W] = X[t_r:hi] @ beta + ic
    got = parallel_enet(X, y, W, alpha, l1, refit, har_idx, W, n, n_jobs=1)
    assert np.array_equal(got, ref), (
        f"serial chain diverges from incumbent: max|d|={np.abs(got - ref).max():.3e}"
    )
    print("PASS test 1  serial rolling_enet == _cadence_enet_har_unpen  (bit-for-bit)")


def test_plain_enet_path() -> None:
    """Test 2: empty HAR mask == sklearn ElasticNet over all columns (per-window centered)."""
    from sklearn.linear_model import ElasticNet

    n, p, W, refit, alpha, l1 = 1600, 15, 800, 400, 0.005, 0.5
    X, y, _ = _gen(n, p, 0, seed=2)
    har_idx = np.zeros(0, dtype=int)
    exog_idx = np.arange(p)
    starts = list(range(W, n, refit))
    for t_r in starts:
        Xw, yw = X[t_r - W : t_r], y[t_r - W : t_r]
        xbar, ybar = Xw.mean(0), float(yw.mean())
        en = ElasticNet(
            alpha=alpha, l1_ratio=l1, fit_intercept=False, max_iter=20000, tol=1e-8
        )
        en.fit(Xw - xbar, yw - ybar)
        te = min(t_r + refit, n)
        ref = X[t_r:te] @ en.coef_ + (ybar - xbar @ en.coef_)
        got = refit_chain(X, y, [t_r], [te], W, alpha, l1, har_idx, exog_idx)[0]
        rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
        assert rel < 1e-5, f"plain-enet path off at t_r={t_r}: rel={rel:.3e}"
    print(f"PASS test 2  empty-HAR path == sklearn ElasticNet  ({len(starts)} refits)")


def test_njobs_equivalence() -> None:
    """Test 3: parallel chunks (cold-seeded chains) agree with serial to CD tolerance."""
    n, p, k = 5000, 30, 6
    W, refit, alpha, l1 = 1500, 480, 0.001, 0.2
    X, y, har_idx = _gen(n, p, k, seed=3)
    base = parallel_enet(X, y, W, alpha, l1, refit, har_idx, W, n, n_jobs=1)
    for nj in (2, 4):
        got = parallel_enet(X, y, W, alpha, l1, refit, har_idx, W, n, n_jobs=nj)
        d = np.abs(got - base).max()
        rel = d / (np.abs(base).max() + 1e-12)
        assert rel < 1e-3, f"n_jobs={nj} exceeds CD tolerance vs serial: rel={rel:.3e}"
        print(
            f"PASS test 3  n_jobs={nj} == serial to CD tol  (max rel {rel:.1e}, {len(base)} preds)"
        )


def test_scale() -> None:
    """Test 4: real-scale (N=75k, p=529, W=24000, refit=480) — serial vs 8 procs + timing."""
    N, p, k = 75000, 529, 6
    W, refit, alpha, l1 = 24000, 480, 0.001, 0.2
    X, y, har_idx = _gen(N, p, k, seed=4)
    # causal standardization, as bucket_enet does
    mu, sd = X[:W].mean(0), X[:W].std(0)
    X = (X - mu) / np.where(sd > 0, sd, 1.0)
    n_refits = len(range(W, N, refit))
    t0 = time.time()
    serial = parallel_enet(X, y, W, alpha, l1, refit, har_idx, W, N, n_jobs=1)
    t_ser = time.time() - t0
    t0 = time.time()
    par = parallel_enet(X, y, W, alpha, l1, refit, har_idx, W, N, n_jobs=8)
    t_par = time.time() - t0
    rel = np.abs(par - serial).max() / (np.abs(serial).max() + 1e-12)
    assert rel < 1e-3, f"real-scale parallel exceeds CD tolerance: rel={rel:.3e}"
    print(
        f"PASS test 4  real-scale {N}x{p}, {n_refits} refits: "
        f"serial {t_ser:.1f}s ({t_ser / n_refits:.2f}s/refit) -> 8-proc {t_par:.1f}s "
        f"(x{t_ser / t_par:.1f}), CD-tol agreement (rel {rel:.1e})"
    )


if __name__ == "__main__":
    test_fwls_vs_incumbent()
    test_plain_enet_path()
    test_njobs_equivalence()
    if "--scale" in sys.argv:
        test_scale()
    print("ALL CHECKS PASSED")
