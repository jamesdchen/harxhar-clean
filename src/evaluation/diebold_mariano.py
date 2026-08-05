"""Diebold-Mariano test of equal predictive accuracy -- model-level significance.

Complements the feature-level circular-shift gate (`feature_cv.py`): given two models' PER-BAR QLIKE
losses on the SAME out-of-sample rows, it tests H0: E[L_a - L_b] = 0 on the loss differential, using a
Newey-West HAC long-run variance (loss differentials are autocorrelated) and the Harvey-Leybourne-Newbold
(1997) small-sample correction. Two-sided p from the normal tail (erfc) -- no scipy dependency; for the
sample sizes here the t_{T-1} and normal references agree to <1e-3.

Use: turn a QLIKE delta between two configs (e.g. train-window 1000d vs 3000d, or all_buckets vs HAR)
into "significant or noise". Reference: Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
"""

from __future__ import annotations

import math

import numpy as np


def qlike_per_bar(pred_raw: np.ndarray, true_raw: np.ndarray) -> np.ndarray:
    """Per-bar QLIKE loss  r - log r - 1  (r = true/pred), raw-variance scale; NaN where non-positive.
    Averaging this over bars reproduces the scalar QLIKE the ablations report."""
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    true_raw = np.asarray(true_raw, dtype=np.float64)
    out = np.full(pred_raw.shape, np.nan)
    m = (pred_raw > 0) & (true_raw > 0)
    r = true_raw[m] / pred_raw[m]
    out[m] = r - np.log(r) - 1.0
    return out


def _nw_lag(T: int) -> int:
    """Newey-West automatic truncation bandwidth floor(4 (T/100)^(2/9))."""
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def dm_test(loss_a, loss_b, h: int = 1, hac_lag: int | None = None) -> dict:
    """Diebold-Mariano test on two per-bar loss series (same OOS rows, lower = better).

    Parameters
    ----------
    loss_a, loss_b : per-bar losses; non-finite entries are dropped pairwise.
    h : forecast horizon (1 for one-step). Sets the classic minimum HAC lag h-1.
    hac_lag : Newey-West truncation; default max(h-1, auto-bandwidth).

    Returns dict: dm (HLN-corrected stat), p (two-sided), mean_diff (E[L_a-L_b]), T, hac_lag, better.
    dm<0 / mean_diff<0 -> model A has lower loss (better). Small p -> the difference is significant.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    d = a - b
    d = d[np.isfinite(d)]
    T = d.size
    if T < 8:
        return {
            "dm": float("nan"),
            "p": float("nan"),
            "mean_diff": float("nan"),
            "T": T,
        }
    dbar = float(d.mean())
    dc = d - dbar
    if hac_lag is None:
        hac_lag = max(h - 1, _nw_lag(T))
    lrv = float(np.mean(dc * dc))  # gamma_0
    for k in range(1, hac_lag + 1):
        gk = float(np.mean(dc[k:] * dc[:-k]))
        lrv += 2.0 * (1.0 - k / (hac_lag + 1.0)) * gk  # Bartlett-weighted
    if lrv <= 1e-300:  # identical (or constant-offset) series
        dm_hln = 0.0 if abs(dbar) < 1e-300 else math.copysign(math.inf, dbar)
        p = 1.0 if abs(dbar) < 1e-300 else 0.0
        return {
            "dm": dm_hln,
            "p": p,
            "mean_diff": dbar,
            "T": T,
            "hac_lag": hac_lag,
            "better": "tie" if abs(dbar) < 1e-300 else ("a" if dbar < 0 else "b"),
        }
    dm = dbar / math.sqrt(lrv / T)
    corr = math.sqrt(
        max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12)
    )  # HLN correction
    dm_hln = dm * corr
    p = math.erfc(abs(dm_hln) / math.sqrt(2.0))  # two-sided normal tail
    return {
        "dm": dm_hln,
        "p": p,
        "mean_diff": dbar,
        "T": T,
        "hac_lag": hac_lag,
        "better": "a" if dbar < 0 else "b",
    }
