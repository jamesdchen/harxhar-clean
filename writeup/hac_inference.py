"""Inference on strategy returns for the rehauled paper (2026-10-21, decision #2 on the Newey-West lag rule).

One module, used by every generator that tests strategy returns (make_rehaul_economics.py, make_rehaul_results.py,
make_fig_qlike_sharpe.py), so every test in the paper has one implementation.

MEANS (mean daily return of a strategy, mean daily return differential of two strategies)
  Lazarus, Lewis, Stock and Watson (2018, JBES 36(4), 541-559; LLSW), rule (3) and Section 2.1: Newey-West long-run
  variance with the Bartlett kernel k(v) = (1 - |v|) 1{|v| <= 1},
      Omega = sum_{|j| < S} (1 - |j|/S) gamma_j,   gamma_j = (1/T) sum_t (x_t - xbar)(x_{t-j} - xbar),
  bandwidth (truncation parameter) S = ceil(1.3 T^(1/2)) (LLSW round S up), t = xbar / sqrt(Omega / T). The
  autocovariances divide by T, with no small-sample factor. The weights 1 - j/S put zero weight on lag S, so this is
  atm_straddle_lib.newey_west_t at lag L = S - 1 (its weights are 1 - k/(L+1)); tested in tests/test_hac_inference.py.
  Critical values and p-values are fixed-b (Kiefer and Vogelsang 2005, Econometric Theory 21(6), 1130-1164) at
  b = S/T: the null distribution of t is simulated from its limit, the same statistic on i.i.d. N(0,1) data on a
  grid of FIXEDB_GRID points with bandwidth b * FIXEDB_GRID (FIXEDB_DRAWS draws, seed FIXEDB_SEED), cached in memory.
  At T = 614: S = 33, b = 0.0537, two-sided 5 % critical value 2.11 (seed FIXEDB_SEED; the Brownian-functional form
  of the limit, KV 2005, gave 2.108 with 200,000 draws on 2,000 points). Cross-check: the Kiefer-Vogelsang (2005)
  cubic for the Bartlett kernel, 97.5 % quantile, cv(b) = 1.9600 + 2.9694 b + 0.4160 b^2 - 0.5324 b^3 (as quoted by
  Shin and Schor 2026, arXiv:2603.07458, eq. (10); the Econometric Theory text was not reachable from the cloud
  session that wrote this module, so the cubic is used only as a check, never for inference).

SHARPE RATIOS (difference of two Sharpe ratios)
  Ledoit and Wolf (2008, J. Empirical Finance 15(5), 850-859), Section 3.2: studentized circular block bootstrap test of
  H0: mu_a / sigma_a = mu_b / sigma_b on the daily returns (not annualized), with u = (mu_a, mu_b, E r_a^2, E r_b^2)
  and the delta method. Standard error in the original sample: Bartlett kernel estimator of Psi with bandwidth equal
  to the block length (whose expectation is that of the block-based estimator below; Ledoit and Wolf use a prewhitened
  quadratic-spectral kernel here instead). In each resample: the block-based estimator Psi* = (1/T) sum_j B_j B_j',
  B_j the sum over block j of the resample centered at its mean (Ledoit and Wolf 2008, Section 3.2.2). Statistic
  d = |Delta| / se, resampled d* = |Delta* - Delta| / se*, p = (#{d* >= d} + 1) / (M + 1). Fixed block length
  LW_BLOCK with the sensitivity lengths LW_BLOCK_ALT, M = LW_BOOT resamples, seed LW_SEED.

NORMAL-THEORY COLUMNS (appendix robustness table only): nw_t_lag(x, L) with the weights 1 - k/(L+1), at
  L = floor(4 (T/100)^(2/9)) (Newey and West 1994, the rule the paper's DM tests use) and L = floor(1.5 T^(1/3)) (the
  paper's previous rule), two-sided normal p.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

LLSW_C = 1.3  # S = ceil(LLSW_C * T^(1/2)), LLSW (2018) rule (3)
FIXEDB_GRID = 1000  # grid points of the simulated limit
FIXEDB_DRAWS = 1_000_000  # draws of the simulated limit (about a minute, once a process)
FIXEDB_SEED = 20181002  # fixed seed of the fixed-b simulation
LW_BLOCK = 10  # block length of the Ledoit-Wolf bootstrap (fixed; sensitivity at LW_BLOCK_ALT)
LW_BLOCK_ALT = (5, 20)
LW_BOOT = 4999  # resamples
LW_SEED = 20080901  # fixed seed of the bootstrap
ALPHA = 0.05


# ----------------------------------------------------------------------------- the LLSW / fixed-b t test of a mean
def llsw_bandwidth(T: int) -> int:
    """S = ceil(1.3 T^(1/2)) (LLSW 2018, rule (3), rounded up)."""
    return int(math.ceil(LLSW_C * math.sqrt(T) - 1e-12))


def bartlett_lrv(x, S: float) -> float:
    """sum_{|j| < S} (1 - |j|/S) gamma_j, gamma_j about the sample mean divided by T (S may be non-integer)."""
    e = np.asarray(x, float)
    e = e - e.mean()
    T = len(e)
    lrv = float(e @ e) / T
    for j in range(1, min(int(math.ceil(S)), T)):
        w = 1.0 - j / S
        if w <= 0:
            break
        lrv += 2.0 * w * float(e[j:] @ e[:-j]) / T
    return lrv


def nw_t_bandwidth(x, S: float) -> float:
    """xbar / sqrt(Omega / T) with the Bartlett kernel at bandwidth S."""
    v = _clean(x)
    lrv = bartlett_lrv(v, S)
    return float(v.mean() / math.sqrt(lrv / len(v))) if lrv > 0 else float("nan")


def nw_t_lag(x, L: int) -> float:
    """The textbook Newey-West t at lag truncation L (weights 1 - k/(L+1)) = nw_t_bandwidth at S = L + 1."""
    return nw_t_bandwidth(x, L + 1)


def nw94_lag(T: int) -> int:
    """floor(4 (T/100)^(2/9)) (Newey and West 1994; 5 at T = 614)."""
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def old_lag(T: int) -> int:
    """floor(1.5 T^(1/3)), the paper's previous lag rule (12 at T = 614); appendix robustness columns only."""
    return int(math.floor(1.5 * T ** (1.0 / 3.0) + 1e-12))


def normal_p(t: float) -> float:
    """Two-sided standard normal p-value."""
    return math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else float("nan")


def _clean(x) -> np.ndarray:
    v = np.asarray(x, float).ravel()
    assert np.isfinite(v).all(), "non-finite return"
    assert len(v) >= 10, len(v)
    return v


def simulate_fixedb_abs_t(b: float, grid: int = FIXEDB_GRID, draws: int = FIXEDB_DRAWS,
                          seed: int = FIXEDB_SEED, chunk: int = 2000) -> np.ndarray:
    """Sorted |t| under the fixed-b limit at b: the Bartlett-kernel t of the mean of i.i.d. N(0,1) data on `grid`
    points with bandwidth b * grid. Autocovariances through FFT, chunk draws at a time."""
    assert 0.0 < b <= 1.0, b
    S = b * grid
    J = min(int(math.ceil(S)), grid)
    w = np.clip(1.0 - np.arange(1, J) / S, 0.0, None)
    nfft = 1 << int(math.ceil(math.log2(2 * grid)))
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        z = rng.standard_normal((m, grid))
        mean = z.mean(axis=1)
        e = z - mean[:, None]
        f = np.fft.rfft(e, n=nfft, axis=1)
        ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=1)[:, :J] / grid
        lrv = ac[:, 0] + 2.0 * ac[:, 1:J] @ w
        out[done:done + m] = np.abs(mean / np.sqrt(lrv / grid))
        done += m
    out.sort()
    return out


@lru_cache(maxsize=8)
def _fixedb_table(b_key: float) -> np.ndarray:
    return simulate_fixedb_abs_t(b_key)


def fixedb_abs_t(b: float) -> np.ndarray:
    """Cached sorted simulated |t| at b (b rounded to 1e-6 for the cache key)."""
    return _fixedb_table(round(float(b), 6))


def fixedb_p(t: float, b: float) -> float:
    """Two-sided fixed-b p-value of t: the share of simulated |t*| >= |t|."""
    if not np.isfinite(t):
        return float("nan")
    a = fixedb_abs_t(b)
    return float((len(a) - np.searchsorted(a, abs(t), side="left")) / len(a))


def fixedb_cv(b: float, alpha: float = ALPHA) -> float:
    """Two-sided fixed-b critical value: the 1 - alpha quantile of |t*|."""
    return float(np.quantile(fixedb_abs_t(b), 1.0 - alpha))


def kv_cubic_cv975(b: float) -> float:
    """Kiefer-Vogelsang (2005) cubic for the Bartlett kernel, 97.5 % quantile (cross-check only)."""
    return 1.9600 + 2.9694 * b + 0.4160 * b**2 - 0.5324 * b**3


def llsw_test(x) -> dict:
    """The paper's test of a mean: LLSW bandwidth, fixed-b p and 5 % critical value. Returns
    {t, p, S, T, b, cv}; p and cv are two-sided."""
    v = _clean(x)
    T = len(v)
    S = llsw_bandwidth(T)
    b = S / T
    t = nw_t_bandwidth(v, S)
    return {"t": t, "p": fixedb_p(t, b), "S": S, "T": T, "b": b, "cv": fixedb_cv(b)}


# ----------------------------------------------------------------------------- Ledoit-Wolf (2008) Sharpe test
def _sr_gap_and_grad(m: np.ndarray):
    """m[..., 4] = (mu_a, mu_b, E r_a^2, E r_b^2) -> (Delta, grad[..., 4])."""
    mu_a, mu_b, g_a, g_b = m[..., 0], m[..., 1], m[..., 2], m[..., 3]
    va, vb = g_a - mu_a**2, g_b - mu_b**2
    delta = mu_a / np.sqrt(va) - mu_b / np.sqrt(vb)
    grad = np.stack(
        [g_a / va**1.5, -g_b / vb**1.5, -0.5 * mu_a / va**1.5, 0.5 * mu_b / vb**1.5], axis=-1
    )
    return delta, grad


def _bartlett_psi(y: np.ndarray, S: float) -> np.ndarray:
    e = y - y.mean(axis=0)
    T = len(e)
    psi = e.T @ e / T
    for j in range(1, min(int(math.ceil(S)), T)):
        w = 1.0 - j / S
        if w <= 0:
            break
        g = e[j:].T @ e[:-j] / T
        psi += w * (g + g.T)
    return psi


def lw_sharpe_test(ra, rb, block: int = LW_BLOCK, n_boot: int = LW_BOOT, seed: int = LW_SEED,
                   chunk: int = 500) -> dict:
    """Ledoit-Wolf (2008) test of equal Sharpe ratios of two return series observed on the same days.
    Returns {delta (daily Sharpe ratio of a minus that of b), se, d, p, block, n_boot, T}."""
    a, b_ = _clean(ra), _clean(rb)
    assert len(a) == len(b_), (len(a), len(b_))
    T = len(a)
    y = np.column_stack([a, b_, a * a, b_ * b_])
    delta, grad = _sr_gap_and_grad(y.mean(axis=0))
    se = math.sqrt(float(grad @ _bartlett_psi(y, block) @ grad) / T)
    d = abs(float(delta)) / se
    nblk = int(math.ceil(T / block))
    L = nblk * block
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        starts = rng.integers(0, T, size=(m, nblk))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
        idx = idx.reshape(m, L)[:, :T]
        ys = y[idx]  # (m, T, 4)
        mean_s = ys.mean(axis=1)
        e = np.zeros((m, L, 4))
        e[:, :T] = ys - mean_s[:, None, :]
        B = e.reshape(m, nblk, block, 4).sum(axis=2)  # block sums (last block truncated at T)
        psi = np.einsum("mji,mjk->mik", B, B) / T
        dlt, g = _sr_gap_and_grad(mean_s)
        se_s = np.sqrt(np.einsum("mi,mik,mk->m", g, psi, g) / T)
        exceed += int(np.sum(np.abs(dlt - delta) / se_s >= d))
        done += m
    return {"delta": float(delta), "se": se, "d": d, "p": (exceed + 1) / (n_boot + 1), "block": block,
            "n_boot": n_boot, "T": T}
