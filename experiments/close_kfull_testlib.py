"""Test helpers for experiments/close_kfull_tests.py (forecast comparison of the 16:00-bar pools).

numpy only; every resampling here is its own (the repo's circular block bootstrap helper,
notebooks/atm_straddle_lib.circular_block_bootstrap_idx, returns the original order since commit
b761b28 and is never called).  Conventions:

* HAC = Newey-West with the Bartlett kernel, weights 1 - j/(L+1), autocovariances about the mean
  (or the residuals) divided by n, no small-sample correction (the convention of
  atm_straddle_lib.newey_west_t and src/evaluation/diebold_mariano.dm_test).
* d = loss of the linear baseline minus loss of the tree forecast: positive = the tree has lower loss.

References: Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997, 1998); Kiefer & Vogelsang
(2005); Giacomini & White (2006); White (2000); Hansen (2005); Politis & Romano (1994); Hansen,
Lunde & Nason (2011); Mincer & Zarnowitz (1969); Patton & Sheppard (2009); Fair & Shiller (1990);
Giacomini & Rossi (2010); Kish (1965).
"""

from __future__ import annotations

import math

import numpy as np


# ============================================================================ HAC
def hac_cov(U: np.ndarray, lag: int) -> np.ndarray:
    """Long-run covariance of the rows of U (n x q), already centred: Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')."""
    U = np.asarray(U, float)
    if U.ndim == 1:
        U = U[:, None]
    n = U.shape[0]
    S = U.T @ U / n
    for j in range(1, int(lag) + 1):
        G = U[j:].T @ U[:-j] / n
        S += (1.0 - j / (lag + 1.0)) * (G + G.T)
    return S


def hac_t(x: np.ndarray, lag: int) -> float:
    """HAC t of the mean of x against zero."""
    x = np.asarray(x, float)
    v = float(hac_cov(x - x.mean(), lag)[0, 0])
    return float(x.mean() / math.sqrt(v / x.size)) if v > 0 else float("nan")


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chi2_sf(x: float, q: int) -> float:
    """Upper tail of the chi-square with q degrees of freedom (regularized gamma, series / continued fraction)."""
    if not np.isfinite(x):
        return float("nan")
    if x <= 0:
        return 1.0
    a, z = q / 2.0, x / 2.0
    lg = math.lgamma(a)
    if z < a + 1.0:  # series for the lower tail
        s, term, k = 1.0 / a, 1.0 / a, a
        for _ in range(10000):
            k += 1.0
            term *= z / k
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return float(max(0.0, 1.0 - s * math.exp(-z + a * math.log(z) - lg)))
    # Lentz continued fraction for the upper tail
    b, c, d = z + 1.0 - a, 1e300, 1.0 / (z + 1.0 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return float(math.exp(-z + a * math.log(z) - lg) * h)


# ============================================================================ fixed-b (Kiefer-Vogelsang)
def fixedb_draws(bs: list[float], n_grid: int = 2000, reps: int = 50000, seed: int = 1, chunk: int = 2500) -> dict:
    """Draws of the fixed-b limit of the Bartlett-HAC t statistic, by simulation: i.i.d. N(0,1) series of
    length n_grid (the Brownian motion on a grid), the HAC t with bandwidth M = round(b n_grid) (weights
    1 - j/M, j < M; M = lag + 1 in the dm_test convention).  Returns {b: (reps,) draws of t}."""
    rng = np.random.default_rng(seed)
    Ms = {b: max(1, int(round(b * n_grid))) for b in bs}
    Mmax = max(Ms.values())
    nfft = 1 << int(math.ceil(math.log2(2 * n_grid)))
    out = {b: [] for b in bs}
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        e = rng.standard_normal((r, n_grid))
        m = e.mean(axis=1)
        ec = e - m[:, None]
        F = np.fft.rfft(ec, nfft, axis=1)
        ac = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, :Mmax] / n_grid  # gamma_0 .. gamma_{Mmax-1}
        for b, M in Ms.items():
            w = 1.0 - np.arange(1, M) / M
            lrv = ac[:, 0] + 2.0 * (ac[:, 1:M] @ w if M > 1 else 0.0)
            out[b].append(math.sqrt(n_grid) * m / np.sqrt(lrv))
        done += r
    return {b: np.concatenate(v) for b, v in out.items()}


# ============================================================================ Giacomini-White
def gw_test(d: np.ndarray, H: np.ndarray, lag: int) -> dict:
    """Giacomini-White: Z_t = h_{t-1} d_t (H row t already holds the lagged instruments), Wald
    n Zbar' Omega^-1 Zbar ~ chi2(q), Omega HAC of the centred Z."""
    Z = H * d[:, None]
    n, q = Z.shape
    zb = Z.mean(axis=0)
    Om = hac_cov(Z - zb, lag)
    stat = float(n * zb @ np.linalg.solve(Om, zb))
    return dict(stat=stat, q=q, p=chi2_sf(stat, q), n=n)


# ============================================================================ stationary bootstrap
def stationary_idx(n: int, B: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(B, n) indices of the stationary bootstrap (Politis-Romano 1994): blocks of geometric length
    with mean mean_block, wrapping at n."""
    p = 1.0 / float(mean_block)
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    new = rng.random((B, n)) < p
    starts = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def boot_means(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, m) column means of X (n x m) over each resampled index row, via counts."""
    B, n = idx.shape
    W = np.zeros((B, n))
    flat = (idx + np.arange(B)[:, None] * n).ravel()
    W.ravel()[:] = np.bincount(flat, minlength=B * n)
    return W @ X / n


def spa_rc(d: np.ndarray, Dstar: np.ndarray) -> dict:
    """Hansen (2005) SPA and White (2000) Reality Check for H0: max_k E[d_k] <= 0.
    d (n x m) differentials, Dstar (B x m) bootstrap means of d.  omega_k^2 = n x bootstrap variance."""
    n = d.shape[0]
    dbar = d.mean(axis=0)
    om = np.sqrt(n * ((Dstar - dbar) ** 2).mean(axis=0))
    t_spa = max(0.0, float(np.max(np.sqrt(n) * dbar / om)))
    A = np.sqrt(om**2 / n * 2.0 * np.log(np.log(n)))
    g = {
        "lower": np.maximum(dbar, 0.0),
        "consistent": dbar * (dbar >= -A),
        "upper": dbar,
    }
    out = dict(stat_spa=t_spa, best=int(np.argmax(np.sqrt(n) * dbar / om)))
    for k, gk in g.items():
        Ts = np.maximum(0.0, np.max(np.sqrt(n) * (Dstar - gk) / om, axis=1))
        out[f"p_{k}"] = float(np.mean(Ts > t_spa))
    v = float(np.max(np.sqrt(n) * dbar))
    Vs = np.max(np.sqrt(n) * (Dstar - dbar), axis=1)
    out.update(stat_rc=v, p_rc=float(np.mean(Vs > v)), omega=om)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    m = p.size
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ============================================================================ regressions
def ols_hac(y: np.ndarray, X: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients, HAC covariance (Bartlett, no small-sample correction), residuals."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    S = hac_cov(X * u[:, None], lag) * len(y)
    return beta, XtX_inv @ S @ XtX_inv, u


def wald(beta: np.ndarray, cov: np.ndarray, R: np.ndarray, r: np.ndarray) -> tuple[float, int, float]:
    R = np.atleast_2d(R)
    e = R @ beta - r
    stat = float(e @ np.linalg.solve(R @ cov @ R.T, e))
    q = R.shape[0]
    return stat, q, chi2_sf(stat, q)


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def realtime_lambda(y: np.ndarray, fL: np.ndarray, fT: np.ndarray, start: int, grid: np.ndarray) -> np.ndarray:
    """lambda_t (t >= start) minimizing the QLIKE of fL + lambda (fT - fL) over days 0 .. t-1 (expanding);
    NaN before start."""
    L = qlike(y[:, None], fL[:, None] + grid[None, :] * (fT - fL)[:, None])
    C = np.cumsum(L, axis=0)
    lam = np.full(len(y), np.nan)
    lam[start:] = grid[np.argmin(C[start - 1 : len(y) - 1], axis=1)]
    return lam


# ============================================================================ fluctuation (Giacomini-Rossi 2010)
def fluctuation_draws(mu: float, n_grid: int = 2000, reps: int = 50000, seed: int = 2, chunk: int = 2500) -> tuple[np.ndarray, np.ndarray]:
    """Draws of sup_r |B(r) - B(r - mu)| / sqrt(mu) and sup_r (B(r) - B(r - mu)) / sqrt(mu), r in [mu, 1],
    B a standard Brownian motion simulated on n_grid steps."""
    rng = np.random.default_rng(seed)
    m = int(round(mu * n_grid))
    a, s = [], []
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        S = np.concatenate([np.zeros((r, 1)), np.cumsum(rng.standard_normal((r, n_grid)), axis=1)], axis=1)
        F = (S[:, m:] - S[:, :-m]) / math.sqrt(m)
        a.append(np.abs(F).max(axis=1))
        s.append(F.max(axis=1))
        done += r
    return np.concatenate(a), np.concatenate(s)


def fluctuation_path(d: np.ndarray, m: int, sigma: float) -> np.ndarray:
    """Rolling statistic sigma^-1 m^-1/2 sum_{j=t-m+1}^{t} d_j for t = m-1 .. n-1."""
    c = np.concatenate([[0.0], np.cumsum(d)])
    return (c[m:] - c[:-m]) / (sigma * math.sqrt(m))
