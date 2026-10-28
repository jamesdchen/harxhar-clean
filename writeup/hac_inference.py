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

THE PAPER'S SHARPE RATIO (2026-10-26, author decision D19; supersedes D6 below as the primary ratio)
  paper_sharpe: the standard pooled ratio sqrt(q) mean / sd (ddof 1) of all the returns (each day weighs 1/T), with
  q = T / sum_y c_y (paper_q; c_y as in d6_years): on the 614 post-warmup days sum_y c_y = 2.722, q = 225.5.
  paper_sharpe_test: a difference of two such ratios on the same days, tested by Ledoit and Wolf (2008)
  (lw_sharpe_test; its daily statistic uses the population sd, so the annualized difference is sqrt(q (T - 1) / T)
  times it), with the paired circular block bootstrap of the pooled difference (pooled_diff_boot: d6_diff_test's
  resamples, q fixed) as a check for the CSV files. D6 (below) stays: the robustness column, and its SR_y terms the
  descriptive by-year view.

THE PAPER'S SHARPE RATIO UNTIL 2026-10-26 (2026-10-22, user, D6: "switch to d6")
  d6_sharpe: for each calendar year y of the sample, SR_y = mean_y / sd_y (ddof 1) * sqrt(A_y), with A_y the year's
  expirations per calendar year, A_y = N_y / c_y: N_y the returns in year y and c_y the fraction of calendar year y the
  sample covers (from the later of 1 January and the sample's first day to the earlier of 31 December and its last
  day, inclusive, over 365 or 366 days). The ratio is the simple (equal-weight) mean of the SR_y: each year counts once
  (Mon/Wed/Fri expirations until May 2022, daily after; a possible decline of the strategy), each annualized at its
  own trading frequency. On the 614 post-warmup days (2021-08-11 .. 2024-04-30): N_y = 64 / 219 / 248 / 83,
  c_y = 0.392 / 1 / 1 / 0.331, A_y = 163.4 / 219 / 248 / 251.1. It replaces atm_straddle_lib.sharpe_ann_by_year
  (combine="weighted": SR_y = mean_y / sd_y * sqrt(N_y), N_y-weighted; new-notes 15, 2026-10-18) in the paper
  generators; that function stays, for the robustness rows ("D1").
  d6_diff_test: the difference of two D6 ratios on the same days, tested by a paired circular block bootstrap (block
  LW_BLOCK, LW_BOOT resamples, seed D6_SEED): each resample joins blocks of consecutive days drawn with replacement
  from the days arranged on a circle, the same days for both series; every drawn day keeps its own calendar year, and
  A_y stays at the sample's values. Two-sided p = the share of resamples with |Delta* - Delta| >= |Delta|. Not
  studentized (the Ledoit-Wolf test above tests the ratio of the pooled daily returns, not D6; it is kept, as a
  robustness column). Unit-checked in writeup/tests/test_hac_inference.py against a verbatim copy of stat / boot_p of
  the scratch script sharpe_equalyears.py (2026-10-22), whose six pooled-class values (sharpe_equalyears.csv, 1,999
  resamples) d6_sharpe / d6_diff_test reproduce exactly.
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
D6_SEED = 20261022  # 2026-10-22 (user, D6): fixed seed of the bootstrap of a D6 Sharpe-ratio difference
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


def simulate_fixedb_abs_t(
    b: float, grid: int = FIXEDB_GRID, draws: int = FIXEDB_DRAWS, seed: int = FIXEDB_SEED, chunk: int = 2000
) -> np.ndarray:
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
        out[done : done + m] = np.abs(mean / np.sqrt(lrv / grid))
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
    grad = np.stack([g_a / va**1.5, -g_b / vb**1.5, -0.5 * mu_a / va**1.5, 0.5 * mu_b / vb**1.5], axis=-1)
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


def lw_sharpe_test(ra, rb, block: int = LW_BLOCK, n_boot: int = LW_BOOT, seed: int = LW_SEED, chunk: int = 500) -> dict:
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
    return {
        "delta": float(delta),
        "se": se,
        "d": d,
        "p": (exceed + 1) / (n_boot + 1),
        "block": block,
        "n_boot": n_boot,
        "T": T,
    }


# ----------------------------------------------------------------------------- 2026-10-22 (user, D6): the Sharpe ratio
def d6_years(dates, span=None) -> dict:
    """Calendar years of a sample and their annualization: {"years": the year of each date (int array), "N": {y: N_y},
    "coverage": {y: c_y}, "A": {y: A_y = N_y / c_y}}. c_y is the fraction of calendar year y between the later of
    1 January and the sample's first day and the earlier of 31 December and its last day (inclusive; 365 or 366 days).
    span = (first, last) fixes the sample's first and last day (default: those of dates)."""
    import pandas as pd

    d = pd.DatetimeIndex(dates)
    assert len(d) and not d.hasnans, "d6_years: no dates"
    first, last = (d.min(), d.max()) if span is None else (pd.Timestamp(span[0]), pd.Timestamp(span[1]))
    assert first <= d.min() and d.max() <= last, (first, last, d.min(), d.max())
    yrs = np.asarray(d.year, dtype=int)
    N, cov, A = {}, {}, {}
    for y in sorted(set(yrs.tolist())):
        lo = max(pd.Timestamp(year=y, month=1, day=1), first.normalize())
        hi_ = min(pd.Timestamp(year=y, month=12, day=31), last.normalize())
        n_days = pd.Timestamp(year=y, month=12, day=31).dayofyear  # 365 or 366
        N[y] = int((yrs == y).sum())
        cov[y] = ((hi_ - lo).days + 1) / n_days
        A[y] = N[y] / cov[y]
    return {"years": yrs, "N": N, "coverage": cov, "A": A}


def _d6_stat(x: np.ndarray, yc: np.ndarray, sqrt_a: np.ndarray) -> np.ndarray:
    """D6 on the rows of x (m, T), whose days fall in the years yc (m, T; codes 0..K-1, annualization sqrt_a[k]): the
    mean over years of mean_k / sd_k (ddof 1) * sqrt_a[k], skipping a year with fewer than 2 days or a zero sd."""
    srs = np.full((x.shape[0], len(sqrt_a)), np.nan)
    for k, sa in enumerate(sqrt_a):
        m_ = yc == k
        n = m_.sum(axis=1)
        mu = np.where(m_, x, 0.0).sum(axis=1) / np.maximum(n, 1)
        ss = np.where(m_, (x - mu[:, None]) ** 2, 0.0).sum(axis=1)
        sd = np.sqrt(ss / np.maximum(n - 1, 1))
        ok = (n >= 2) & (sd > 0)
        srs[ok, k] = mu[ok] / sd[ok] * sa
    out = np.full(x.shape[0], np.nan)
    has = np.isfinite(srs).any(axis=1)
    out[has] = np.nanmean(srs[has], axis=1)
    return out


def d6_sharpe(r, dates=None, span=None) -> dict:
    """The paper's Sharpe ratio (D6, module docstring): the simple mean over calendar years of
    SR_y = mean_y / sd_y * sqrt(A_y), A_y = N_y / c_y (d6_years). r: returns (a Series indexed by date, or an array with
    dates); non-finite returns are dropped first. Returns {"sharpe", "by_year": {y: (N_y, A_y, SR_y)}, "coverage"}."""
    import pandas as pd

    idx = pd.DatetimeIndex(r.index if dates is None else dates)
    v = np.asarray(r, dtype=float).ravel()
    assert len(v) == len(idx), (len(v), len(idx))
    keep = np.isfinite(v)
    v, idx = v[keep], idx[keep]
    yy = d6_years(idx, span=span)
    by_year: dict[int, tuple[int, float, float]] = {}
    for y, a_y in yy["A"].items():
        x = v[yy["years"] == y]
        if len(x) < 2:
            continue
        sd = float(x.std(ddof=1))
        if not sd > 0:
            continue
        by_year[int(y)] = (
            int(len(x)),
            float(a_y),
            float(x.mean()) / sd * math.sqrt(a_y),
        )
    sr = float(np.mean([v_[2] for v_ in by_year.values()])) if by_year else float("nan")
    return {"sharpe": sr, "by_year": by_year, "coverage": yy["coverage"]}


@lru_cache(maxsize=4)
def _boot_index(T: int, block: int, n_boot: int, seed: int, chunk: int = 500) -> np.ndarray:
    """(n_boot, T) day indices of the circular block resamples (rng.integers draws in chunks: the same stream as one
    draw of nb block starts per resample)."""
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / block))
    out = np.empty((n_boot, T), dtype=np.int32)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        st = rng.integers(0, T, size=(m, nb))
        out[done : done + m] = ((st[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(m, nb * block)[:, :T]
        done += m
    out.setflags(write=False)
    return out


def d6_diff_test(
    ra,
    rb,
    dates,
    block: int = LW_BLOCK,
    n_boot: int = LW_BOOT,
    seed: int = D6_SEED,
    span=None,
    chunk: int = 1000,
) -> dict:
    """Paired circular block bootstrap of the difference of two D6 Sharpe ratios (a minus b) observed on the same days
    (module docstring): each drawn day keeps its calendar year, A_y fixed at the sample's values; two-sided
    p = share of |Delta* - Delta| >= |Delta|. Returns {delta, sr_a, sr_b, p, block, n_boot, T, seed}."""
    a, b_ = _clean(ra), _clean(rb)
    assert len(a) == len(b_), (len(a), len(b_))
    T = len(a)
    yy = d6_years(dates, span=span)
    assert len(yy["years"]) == T, (len(yy["years"]), T)
    ys = sorted(yy["A"])
    code = np.searchsorted(np.array(ys), yy["years"])
    sqrt_a = np.sqrt(np.array([yy["A"][y] for y in ys], dtype=float))
    one = np.arange(T)[None, :]
    sr_a = float(_d6_stat(a[one], code[one], sqrt_a)[0])
    sr_b = float(_d6_stat(b_[one], code[one], sqrt_a)[0])
    assert math.isclose(sr_a, d6_sharpe(a, dates, span=span)["sharpe"], rel_tol=1e-10, abs_tol=1e-12), sr_a
    delta = sr_a - sr_b
    idx = _boot_index(T, block, n_boot, seed)
    exceed = 0
    for s0 in range(0, n_boot, chunk):
        ix = idx[s0 : s0 + chunk]
        yc = code[ix]
        dstar = _d6_stat(a[ix], yc, sqrt_a) - _d6_stat(b_[ix], yc, sqrt_a)
        exceed += int(np.sum(np.abs(dstar - delta) >= abs(delta)))
    return {
        "delta": float(delta),
        "sr_a": sr_a,
        "sr_b": sr_b,
        "p": exceed / n_boot,
        "block": block,
        "n_boot": n_boot,
        "T": T,
        "seed": seed,
    }


# ----------------------------------------------------------------------------- 2026-10-26 (author decision D19)
# THE PAPER'S SHARPE RATIO (D19, supersedes D6 as the primary ratio): the standard pooled ratio
#     SR = sqrt(q) * mean(r) / sd(r)   (sd with divisor T - 1),  each day weighing 1/T, no per-year averaging,
# on all the days of the sample (the 614 post-warmup option days of T_opt^+ in the paper), annualized by
#     q = T / sum_y c_y,   the average number of option days per calendar year,
# with c_y the fraction of calendar year y the sample covers, exactly as d6_years defines it (0.392, 1, 1, 0.331 on the
# 614 days: q = 225.5).  The equal-year mean of the SR_y (d6_sharpe, D6) stays as a robustness column, and the SR_y
# terms stay the descriptive by-year view.  A difference of two ratios is tested by Ledoit and Wolf (2008)
# (lw_sharpe_test: the studentized circular block bootstrap of equal pooled ratios; the p of the daily ratio is the p of
# the annualized one, q being common); the paired circular block bootstrap of the difference (pooled_diff_boot, the
# resampling of d6_diff_test with the pooled statistic) is kept as a check in the CSV files only.


def paper_q(dates, span=None) -> dict:
    """D19: the annualization of the paper's Sharpe ratio on the days given: {"q": T / sum_c, "sum_c": sum_y c_y,
    "T", "coverage": {y: c_y}} (c_y of d6_years, span as there)."""
    yy = d6_years(dates, span=span)
    T = len(yy["years"])
    sum_c = float(sum(yy["coverage"].values()))
    return {"q": T / sum_c, "sum_c": sum_c, "T": T, "coverage": yy["coverage"]}


def paper_sharpe(r, dates=None, span=None) -> dict:
    """D19: the paper's Sharpe ratio, sqrt(q) * mean / sd (ddof 1) of all the returns, q = T / sum_y c_y (paper_q).
    r: returns (a Series indexed by date, or an array with dates); non-finite returns are dropped first. Returns
    {"sharpe", "q", "sum_c", "T", "by_year" (d6_sharpe's {y: (N_y, A_y, SR_y)}: the descriptive by-year view),
    "coverage", "sharpe_d6" (the equal-year mean, D6: robustness)}."""
    import pandas as pd

    idx = pd.DatetimeIndex(r.index if dates is None else dates)
    v = np.asarray(r, dtype=float).ravel()
    assert len(v) == len(idx), (len(v), len(idx))
    keep = np.isfinite(v)
    v, idx = v[keep], idx[keep]
    pq = paper_q(idx, span=span)
    sd = float(v.std(ddof=1)) if len(v) >= 2 else float("nan")
    sr = math.sqrt(pq["q"]) * float(v.mean()) / sd if sd > 0 else float("nan")
    d6 = d6_sharpe(v, idx, span=span)
    return {
        "sharpe": sr,
        "q": pq["q"],
        "sum_c": pq["sum_c"],
        "T": pq["T"],
        "by_year": d6["by_year"],
        "coverage": d6["coverage"],
        "sharpe_d6": d6["sharpe"],
    }


def _pooled_stat(x: np.ndarray, sqrt_q: float) -> np.ndarray:
    """sqrt(q) * mean / sd (ddof 1) of each row of x (m, T)."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1)
    out = np.full(x.shape[0], np.nan)
    ok = sd > 0
    out[ok] = sqrt_q * mu[ok] / sd[ok]
    return out


def pooled_diff_boot(
    ra,
    rb,
    dates,
    block: int = LW_BLOCK,
    n_boot: int = LW_BOOT,
    seed: int = D6_SEED,
    span=None,
    chunk: int = 1000,
) -> dict:
    """D19 check (CSV only): the paired circular block bootstrap of d6_diff_test (the same resample indices) applied to
    the difference of the paper's pooled ratios, q fixed at the sample's value; two-sided p = share of
    |Delta* - Delta| >= |Delta|. Returns {delta, sr_a, sr_b, p, block, n_boot, T, seed}."""
    a, b_ = _clean(ra), _clean(rb)
    assert len(a) == len(b_), (len(a), len(b_))
    T = len(a)
    pq = paper_q(dates, span=span)
    assert pq["T"] == T, (pq["T"], T)
    sq = math.sqrt(pq["q"])
    one = np.arange(T)[None, :]
    sr_a, sr_b = float(_pooled_stat(a[one], sq)[0]), float(_pooled_stat(b_[one], sq)[0])
    delta = sr_a - sr_b
    idx = _boot_index(T, block, n_boot, seed)
    exceed = 0
    for s0 in range(0, n_boot, chunk):
        ix = idx[s0 : s0 + chunk]
        dstar = _pooled_stat(a[ix], sq) - _pooled_stat(b_[ix], sq)
        exceed += int(np.sum(np.abs(dstar - delta) >= abs(delta)))
    return {
        "delta": float(delta),
        "sr_a": sr_a,
        "sr_b": sr_b,
        "p": exceed / n_boot,
        "block": block,
        "n_boot": n_boot,
        "T": T,
        "seed": seed,
    }


def paper_sharpe_test(
    ra,
    rb,
    dates,
    block: int = LW_BLOCK,
    n_boot: int = LW_BOOT,
    seed: int = LW_SEED,
    span=None,
    check: bool = True,
) -> dict:
    """D19: the paper's test of a difference of two Sharpe ratios (a minus b, the same days): Ledoit and Wolf (2008),
    lw_sharpe_test, on the daily returns; delta = the difference of the paper's annualized ratios (paper_sharpe),
    = sqrt(q) times the daily difference. check: also the paired block bootstrap of pooled_diff_boot (p_boot, CSV only).
    Returns {delta, sr_a, sr_b, p (= p_lw), p_lw, se, d, q, block, n_boot, T, seed, p_boot}."""
    a, b_ = _clean(ra), _clean(rb)
    lw = lw_sharpe_test(a, b_, block=block, n_boot=n_boot, seed=seed)
    pq = paper_q(dates, span=span)
    assert pq["T"] == len(a), (pq["T"], len(a))
    sq = math.sqrt(pq["q"])
    sr_a = sq * float(a.mean()) / float(a.std(ddof=1))
    sr_b = sq * float(b_.mean()) / float(b_.std(ddof=1))
    # the LW statistic is the difference of the daily ratios with the population sd (the moments m = E r, E r^2); the
    # paper's ratio uses ddof 1: the two differ by the factor sqrt((T - 1) / T) only
    assert math.isclose(sr_a - sr_b, sq * lw["delta"] * math.sqrt((len(a) - 1) / len(a)), rel_tol=1e-9, abs_tol=1e-12)
    out = {
        "delta": sr_a - sr_b,
        "sr_a": sr_a,
        "sr_b": sr_b,
        "p": lw["p"],
        "p_lw": lw["p"],
        "se": lw["se"],
        "d": lw["d"],
        "q": pq["q"],
        "block": block,
        "n_boot": n_boot,
        "T": len(a),
        "seed": seed,
        "p_boot": float("nan"),
    }
    if check:
        out["p_boot"] = pooled_diff_boot(a, b_, dates, block=block, n_boot=n_boot, span=span)["p"]
    return out
