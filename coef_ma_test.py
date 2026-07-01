"""Coefficient moving-average (temporal beta-smoothing) -- the one conceptually-new shrinkage.

Fit the HAR-OLS base ONCE (incumbent enet a=1e-3, l1=0.2), then smooth the per-refit-block coefficients
CAUSALLY before predicting each block:
  * MA(K)    -- trailing simple mean over the last K blocks (K=1 = no smoothing = baseline);
  * EWMA(lam)-- beta~_i = (1-lam)*beta_i + lam*beta~_{i-1}  (lam=0 = no smoothing).
Block i's coefs were fit on the window ENDING at starts[i], strictly before block i's OOS rows, so a
trailing average of blocks <= i is causal.

Tests whether variance-reducing the coef estimates helps -- against the prior that it HURTS: the coef
trajectory carries a REAL time-varying signal (the explainability audit's 0DTE/OPEX gamma-dilution --
fast HAR x close decays, slow strengthens), so smoothing adds bias by blurring that drift.

Full-OOS QLIKE, DIAGNOSTIC: if NO smoothing strength beats K=1, coefficient-MA is dead (this reports the
ceiling shape; it does not claim a test-set-selected winner). One base fit -> the K/lam sweep is ~free.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from resid_amortized import PERIODS_PER_DAY, _cadence_enet_har_unpen  # noqa: E402
from src.evaluation.metrics import apply_duan_smearing  # noqa: E402

B = "results/covid_imp_rank/all_buckets"
HAR = ("har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125")
OUT = "results/coef_ma_test.json"
TW = 1000 * PERIODS_PER_DAY  # 48000
REFIT = 8_000  # finer cadence -> more blocks to average
ITER, TOL = 1500, 1e-3
INC_A, INC_L1 = 1e-3, 0.2  # incumbent penalty (enet)
K_GRID = [1, 2, 3, 5, 8, 13]  # trailing simple MA (1 = no smoothing)
LAM_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]  # EWMA decay (0 = no smoothing)


def _qlike(preds, y, base):
    pr, tr = apply_duan_smearing(preds, y, base)
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def _oos(X, starts, coefs, intercepts, tw):
    n = len(X)
    oos = np.empty(n - tw, dtype=np.float64)
    for i, t_r in enumerate(starts):
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        oos[t_r - tw : t_end - tw] = X[t_r:t_end] @ coefs[i] + intercepts[i]
    return oos


def _ma(coefs, intercepts, K):
    """Causal trailing simple MA over the last K blocks (all coefs + intercept)."""
    nb = len(coefs)
    sc = np.empty_like(coefs)
    si = np.empty_like(intercepts)
    for i in range(nb):
        lo = max(0, i - K + 1)
        sc[i] = coefs[lo : i + 1].mean(axis=0)
        si[i] = intercepts[lo : i + 1].mean()
    return sc, si


def _ewma(coefs, intercepts, lam):
    """Causal EWMA: b~_i = (1-lam)*b_i + lam*b~_{i-1}, b~_0 = b_0."""
    nb = len(coefs)
    sc = np.empty_like(coefs)
    si = np.empty_like(intercepts)
    bc = coefs[0].copy()
    bi = float(intercepts[0])
    for i in range(nb):
        if i == 0:
            bc = coefs[0].copy()
            bi = float(intercepts[0])
        else:
            bc = (1.0 - lam) * coefs[i] + lam * bc
            bi = (1.0 - lam) * intercepts[i] + lam * bi
        sc[i] = bc
        si[i] = bi
    return sc, si


def main():
    t_all = time.time()
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    hm = [f in set(HAR) for f in feats]
    assert sum(hm) == 6
    X = np.ascontiguousarray(np.load(f"{B}/X_imp.npy"))
    y = np.load(f"{B}/y.npy").astype(np.float64)
    base = np.load(f"{B}/base.npy").astype(np.float64)
    print(f"loaded X{X.shape}  tw={TW} refit={REFIT}", flush=True)

    # ---- ONE base fit (incumbent enet, HAR OLS) --------------------------
    t0 = time.time()
    starts, coefs, intercepts = _cadence_enet_har_unpen(
        X,
        y,
        TW,
        REFIT,
        hm,
        alpha=INC_A,
        l1=INC_L1,
        penalty="enet",
        max_iter=ITER,
        tol=TOL,
    )
    print(
        f"base fit: {len(starts)} blocks, coefs{coefs.shape}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    y_oos, base_oos = y[TW:], base[TW:]

    q_base = _qlike(_oos(X, starts, coefs, intercepts, TW), y_oos, base_oos)
    print(f"\nK=1 / lam=0 (no smoothing, baseline): {q_base:.5f}", flush=True)

    res = {"baseline_no_smoothing": q_base, "ma": {}, "ewma": {}}

    print("\n--- trailing simple MA(K) ---", flush=True)
    for K in K_GRID:
        sc, si = _ma(coefs, intercepts, K)
        q = _qlike(_oos(X, starts, sc, si, TW), y_oos, base_oos)
        res["ma"][K] = q
        print(f"  MA K={K:<2d} -> {q:.5f}  ({q - q_base:+.5f})", flush=True)

    print("\n--- EWMA(lambda) ---", flush=True)
    for lam in LAM_GRID:
        sc, si = _ewma(coefs, intercepts, lam)
        q = _qlike(_oos(X, starts, sc, si, TW), y_oos, base_oos)
        res["ewma"][lam] = q
        print(f"  EWMA lam={lam:<4g} -> {q:.5f}  ({q - q_base:+.5f})", flush=True)

    best = min(
        [("K=1", q_base)]
        + [(f"MA{K}", v) for K, v in res["ma"].items()]
        + [(f"EWMA{lam}", v) for lam, v in res["ewma"].items()],
        key=lambda kv: kv[1],
    )
    verdict = (
        "DEAD (no smoothing beats K=1)"
        if best[0] in ("K=1", "MA1") or best[1] >= q_base - 1e-5
        else f"{best[0]} beats baseline by {q_base - best[1]:.5f} (needs CV to claim)"
    )
    res["verdict"] = verdict
    print(f"\nbest: {best[0]} {best[1]:.5f}   VERDICT: {verdict}", flush=True)

    json.dump(
        {"tw": TW, "refit": REFIT, "penalty": "enet a=1e-3 l1=0.2", **res},
        open(OUT, "w"),
        indent=2,
    )
    print(f"wrote {OUT}  total {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
