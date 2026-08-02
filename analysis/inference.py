"""Forecast-comparison inference: QLIKE loss, Diebold-Mariano, and the
Hansen-Lunde-Nason Model Confidence Set.

Self-contained (numpy only) so it runs anywhere the prediction dumps do.
Every routine takes per-observation losses, never summary statistics.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "qlike_loss",
    "newey_west_lrv",
    "auto_lag",
    "dm_test",
    "stationary_bootstrap_indices",
    "model_confidence_set",
    "holm",
    "benjamini_hochberg",
]


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def qlike_loss(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE on the raw-variance scale.

    L_t = y_t / yhat_t - log(y_t / yhat_t) - 1,  minimised at yhat_t = y_t.
    Both arguments must be strictly positive.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if np.any(y <= 0) or np.any(yhat <= 0):
        raise ValueError("QLIKE requires strictly positive targets and forecasts")
    r = y / yhat
    return r - np.log(r) - 1.0


# --------------------------------------------------------------------------
# HAC variance
# --------------------------------------------------------------------------

def auto_lag(n: int) -> int:
    """Newey-West rule-of-thumb bandwidth, floor(4 (n/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_lrv(d: np.ndarray, lag: int | None = None) -> float:
    """Newey-West long-run variance of the mean of `d` (Bartlett kernel).

    Returns an estimate of n * Var(mean(d)), i.e. the long-run variance of the
    series itself; divide by n to get the variance of the sample mean.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if lag is None:
        lag = auto_lag(n)
    lag = max(0, min(int(lag), n - 1))
    e = d - d.mean()
    gamma0 = float(e @ e) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        gk = float(e[k:] @ e[:-k]) / n
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    # A Bartlett-kernel estimate is psd in population but can be driven to zero
    # by short samples; guard so downstream division stays finite.
    return max(lrv, 1e-300)


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------

def _student_t_sf(t: float, df: int) -> float:
    """Two-sided Student-t tail probability, via the regularised incomplete beta."""
    from math import lgamma, log, exp, isfinite

    if not isfinite(t):
        return 0.0
    if t == 0.0:
        return 1.0
    x = df / (df + t * t)
    if x >= 1.0:
        return 1.0
    if x <= 0.0:
        return 0.0

    def betacf(a, b, x, itmax=300, eps=3e-14):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < 1e-300:
            d = 1e-300
        d = 1.0 / d
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    a, b = df / 2.0, 0.5
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = exp(a * log(x) + b * log(1.0 - x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - exp(b * log(1.0 - x) + a * log(x) - lbeta) * betacf(b, a, 1.0 - x) / b
    return float(min(1.0, max(0.0, ib)))


def dm_test(loss_a, loss_b, h: int = 1, lag: int | None = None, hln: bool = True):
    """Diebold-Mariano test of equal predictive accuracy, arm A vs arm B.

    Negative t favours A (A has the lower loss). With `hln`, applies the
    Harvey-Leybourne-Newbold small-sample correction and refers the statistic to
    a t distribution with n-1 degrees of freedom rather than the normal.

    Returns a dict with the mean loss differential, HAC standard error,
    statistic, p-value, and the bandwidth actually used.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"loss series must align: {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    if lag is None:
        lag = max(h - 1, auto_lag(n))
    lrv = newey_west_lrv(d, lag=lag)
    se = np.sqrt(lrv / n)
    mean = float(d.mean())
    t = mean / se
    if hln:
        corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        t = t * corr
    p = _student_t_sf(abs(t), df=n - 1) if hln else 2 * (1 - _norm_cdf(abs(t)))
    return {
        "mean_diff": mean,
        "se": float(se),
        "t": float(t),
        "p": float(p),
        "lag": int(lag),
        "n": int(n),
    }


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def stationary_bootstrap_indices(n: int, B: int, block_len: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix, shape (B, n).

    Geometric block lengths with mean `block_len`, wrapping at the sample end,
    which preserves stationarity of the resampled series.
    """
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    new_block = rng.random((B, n)) < p
    starts = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(new_block[:, t], starts[:, t], cont)
    return idx


def politis_white_block_length(d: np.ndarray, max_lag: int | None = None) -> float:
    """Simple automatic block length from the first-order autocorrelation.

    Uses the standard b_opt ~ (2 * rho / (1 - rho^2))^(2/3) * n^(1/3) heuristic,
    floored at 1 and capped at n^(1/2). Reported so the choice is auditable;
    `run_*` scripts also report sensitivity at half and double this value.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    e = d - d.mean()
    denom = float(e @ e)
    rho = float(e[1:] @ e[:-1]) / denom if denom > 0 else 0.0
    rho = min(max(rho, 0.0), 0.95)
    if rho <= 0:
        return 1.0
    b = (2.0 * rho / (1.0 - rho ** 2)) ** (2.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), np.sqrt(n)))


# --------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason 2011)
# --------------------------------------------------------------------------

def model_confidence_set(
    L: np.ndarray,
    names: list[str],
    alpha: float = 0.10,
    B: int = 5000,
    block_len: float | None = None,
    seed: int = 0,
    statistic: str = "Tmax",
):
    """Hansen-Lunde-Nason MCS by iterative elimination.

    Parameters
    ----------
    L : (n, m) array of per-observation losses, one column per model.
    names : length-m model labels.
    alpha : test level; the surviving set is the (1-alpha) MCS.
    statistic : "Tmax" (max standardised deviation from the set mean) or
        "TR" (max standardised pairwise range).

    Returns a dict with the surviving set, the elimination order, and the
    monotone MCS p-value for every model.
    """
    L = np.asarray(L, dtype=np.float64)
    n, m = L.shape
    if len(names) != m:
        raise ValueError("names must have one entry per column of L")
    rng = np.random.default_rng(seed)

    if block_len is None:
        # Base the block length on the most autocorrelated pairwise differential
        # among a sample of pairs, so the resample is conservative.
        cand = []
        for _ in range(min(50, m * (m - 1) // 2)):
            i, j = rng.integers(0, m, size=2)
            if i != j:
                cand.append(politis_white_block_length(L[:, i] - L[:, j]))
        block_len = float(np.median(cand)) if cand else 2.0

    idx = stationary_bootstrap_indices(n, B, block_len, rng)
    # Bootstrap means for every model at once: (B, m)
    boot_means = L[idx, :].mean(axis=1)
    obs_means = L.mean(axis=0)

    alive = list(range(m))
    elim_order: list[tuple[str, float, float]] = []
    pvals = {}
    running_max_p = 0.0

    while len(alive) > 1:
        a = np.array(alive)
        bm = boot_means[:, a]          # (B, k)
        om = obs_means[a]              # (k,)
        centred = bm - om              # bootstrap deviations

        if statistic == "Tmax":
            dev = om - om.mean()
            boot_dev = centred - centred.mean(axis=1, keepdims=True)
            var = boot_dev.var(axis=0, ddof=1)
            var = np.maximum(var, 1e-300)
            t = dev / np.sqrt(var)
            boot_t = boot_dev / np.sqrt(var)
            stat = float(t.max())
            boot_stat = boot_t.max(axis=1)
            worst_local = int(np.argmax(t))
        elif statistic == "TR":
            diff = om[:, None] - om[None, :]
            bdiff = centred[:, :, None] - centred[:, None, :]
            var = bdiff.var(axis=0, ddof=1)
            np.fill_diagonal(var, 1.0)
            var = np.maximum(var, 1e-300)
            tij = diff / np.sqrt(var)
            stat = float(np.abs(tij).max())
            boot_tij = np.abs(bdiff / np.sqrt(var))
            boot_stat = boot_tij.reshape(B, -1).max(axis=1)
            worst_local = int(np.argmax(tij.max(axis=1)))
        else:
            raise ValueError("statistic must be 'Tmax' or 'TR'")

        p = float((boot_stat >= stat).mean())
        running_max_p = max(running_max_p, p)

        if p >= alpha:
            for i in alive:
                pvals.setdefault(names[i], running_max_p)
            break

        worst = alive[worst_local]
        pvals[names[worst]] = running_max_p
        elim_order.append((names[worst], running_max_p, float(obs_means[worst])))
        alive.remove(worst)
    else:
        for i in alive:
            pvals.setdefault(names[i], running_max_p)

    return {
        "surviving": [names[i] for i in alive],
        "eliminated": elim_order,
        "pvalues": pvals,
        "alpha": alpha,
        "B": B,
        "block_len": float(block_len),
        "n": int(n),
        "m": int(m),
        "statistic": statistic,
    }


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------

def holm(pvals: dict[str, float], alpha: float = 0.05):
    """Holm-Bonferroni step-down. Returns {name: (adjusted_p, reject)}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)     # enforce monotonicity
        out[name] = (running, running <= alpha)
    return out


def benjamini_hochberg(pvals: dict[str, float], alpha: float = 0.05):
    """Benjamini-Hochberg step-up FDR control. Returns {name: (q, reject)}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 1.0
    for rank in range(m - 1, -1, -1):
        name, p = items[rank]
        q = min(running, m * p / (rank + 1))
        running = q
        out[name] = (q, q <= alpha)
    return out
