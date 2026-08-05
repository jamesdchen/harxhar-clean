"""Ridge versus roughness, in the regime where a penalty can matter.

The first pass compared penalties on a 12-parameter basis against 145,000
training rows. Every optimal lambda came back 0: with p/n ~ 1e-4 there is
nothing to shrink, so the comparison was vacuous. That is a flaw in the test,
not a finding about penalties.

A roughness penalty earns its keep by letting you use a RICH trial space
without overfitting -- shrinking toward smooth rather than toward zero. So the
test has to be run where the trial space is rich:

    K in {12, 24, 48, 96, 192} cubic B-spline knots in log-lag
    x  penalty in {none, ridge ||w||^2, roughness ||D2 phi_w||^2}

and, separately, in the short-window regime the production system actually uses
(24,000-bar training window rather than 145,000), where p/n is two orders of
magnitude larger.

If the paper's central finding is "shrink, never select", the direction of
shrinkage is the untested lever. This measures it.

ORACLE WARNING -- READ BEFORE QUOTING ANY NUMBER FROM THIS FILE. Every lambda*
below is chosen by minimising the SAME out-of-sample QLIKE that is then
reported, over a 16-point grid, for each of 15 (K, penalty) cells. These are
oracle figures, not out-of-sample ones. That is defensible for the question the
script was written to ask -- ridge versus roughness at matched oracle tuning,
where both arms get the same unfair advantage -- and it is NOT defensible for
any claim about levels or about which K is best. In particular the "best spline
arm beats the b2 ladder" line at the end is withdrawn: its margin is 0.00013,
selected as the minimum over 5 values of K and 16 of lambda on the evaluation
panel itself, which is far inside the selection noise of that search. See
analysis/greedy_vs_midas_diag.py for the same failure mode measured on a
two-parameter continuous search, where it was worth +0.00235.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basis_horserace import (  # noqa: E402
    w_bspline, w_boxcar, featurize, implied_kernel_map, d2_operator, U, BURN,
)

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
TRAIN_FRAC = 0.60
LAMS = [0.0] + [10.0 ** e for e in range(-6, 9)]


def fit_pred(F, y, tr, te, lam, P):
    Ftr = F[tr]
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd[sd < 1e-12] = 1.0
    Z = np.hstack([np.ones((len(tr), 1)), (Ftr - mu) / sd])
    k = Z.shape[1]
    R = np.zeros((k, k))
    R[1:, 1:] = np.eye(k - 1) if P is None else P
    try:
        beta = np.linalg.solve(Z.T @ Z + lam * R + 1e-10 * np.eye(k), Z.T @ y[tr])
    except np.linalg.LinAlgError:
        return None, None
    resid = y[tr] - Z @ beta
    sig2 = float(resid @ resid) / len(resid)
    Zte = np.hstack([np.ones((len(te), 1)), (F[te] - mu) / sd])
    return (Zte @ beta) ** 2 + sig2, sig2


def qlike(t, p):
    m = (t > 0) & (p > 0) & np.isfinite(t) & np.isfinite(p)
    r = t[m] / p[m]
    return float(np.mean(r - np.log(r) - 1.0))


def rough_penalty(B):
    L = implied_kernel_map(B)
    sel = np.unique(np.round(np.exp(np.linspace(0, np.log(U), 96))).astype(int)) - 1
    A = d2_operator(len(sel)) @ L[sel]
    P = A.T @ A
    return P / max(np.trace(P) / P.shape[0], 1e-12)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from src.data.loading import load_raw_data
    from src.features.transforms import target as T
    t0 = time.time()

    df = load_raw_data("data", allow_missing=True)
    T.WINSOR_LOWER_Q, T.WINSOR_UPPER_Q = 0.05, 0.95
    tgt, base = T.robust_transform(df, "RV", use_diurnal=True,
                                   allow_missing=True, is_target=True)
    y = tgt.to_numpy(np.float64)
    bl = base.to_numpy(np.float64)
    rv = df["RV"].to_numpy(np.float64)
    ok = np.isfinite(y) & np.isfinite(bl) & np.isfinite(rv) & (rv > 0)
    ok[:BURN] = False
    idx = np.flatnonzero(ok)
    ys = np.where(np.isfinite(y), y, 0.0)
    yv, blv, rvv = y[idx], bl[idx], rv[idx]
    cut = int(len(idx) * TRAIN_FRAC)
    print(f"panel {len(idx):,}  ({time.time()-t0:.0f}s)")
    out = {}

    # ---------- long window: the full 60/40 split ----------
    print("\n" + "=" * 78)
    print("A. LONG WINDOW (145k training rows) -- rich basis, three penalties")
    print("=" * 78)
    print(f"{'K':>5}{'none':>12}{'ridge':>12}{'lam*':>10}{'roughness':>12}"
          f"{'lam*':>10}   winner")
    tr, te = np.arange(cut), np.arange(cut, len(idx))
    longres = {}
    for K in (12, 24, 48, 96, 192):
        B, _ = w_bspline(K)
        F = featurize(ys, B)[idx]
        P = rough_penalty(B)
        row = {}
        for tag, pen in (("none", "N"), ("ridge", None), ("rough", P)):
            if tag == "none":
                pr, _ = fit_pred(F, yv, tr, te, 0.0, None)
                row[tag] = (qlike(rvv[te], pr * blv[te]) if pr is not None
                            else np.nan, 0.0)
                continue
            best = (np.inf, None)
            for lam in LAMS:
                pr, _ = fit_pred(F, yv, tr, te, lam, pen)
                if pr is None:
                    continue
                q = qlike(rvv[te], pr * blv[te])
                if np.isfinite(q) and q < best[0]:
                    best = (q, lam)
            row[tag] = best
        w = "roughness" if row["rough"][0] < row["ridge"][0] else "ridge"
        print(f"{B.shape[0]:>5}{row['none'][0]:>12.5f}{row['ridge'][0]:>12.5f}"
              f"{row['ridge'][1]:>10g}{row['rough'][0]:>12.5f}"
              f"{row['rough'][1]:>10g}   {w}")
        longres[str(B.shape[0])] = {k: {"qlike": v[0], "lambda": v[1]}
                                    for k, v in row.items()}
        del F
    out["long_window"] = longres

    # ---------- short window: the production regime ----------
    print("\n" + "=" * 78)
    print("B. SHORT WINDOW (24,000-bar training, the production convention)")
    print("=" * 78)
    W = 24000
    starts = np.arange(cut, len(idx) - 2000, 20000)
    print(f"   {len(starts)} non-overlapping evaluation blocks of 2,000 bars")
    print(f"{'K':>5}{'none':>12}{'ridge':>12}{'lam*':>10}{'roughness':>12}"
          f"{'lam*':>10}   winner")
    shortres = {}
    for K in (12, 24, 48, 96, 192):
        B, _ = w_bspline(K)
        F = featurize(ys, B)[idx]
        P = rough_penalty(B)
        row = {}
        for tag, pen in (("none", "N"), ("ridge", None), ("rough", P)):
            cands = [0.0] if tag == "none" else LAMS
            best = (np.inf, None)
            for lam in cands:
                num, den = 0.0, 0
                bad = False
                for s0 in starts:
                    tr_ = np.arange(s0 - W, s0)
                    te_ = np.arange(s0, s0 + 2000)
                    pr, _ = fit_pred(F, yv, tr_, te_, lam,
                                     None if tag != "rough" else pen)
                    if pr is None:
                        bad = True
                        break
                    p = pr * blv[te_]
                    t = rvv[te_]
                    m = (t > 0) & (p > 0) & np.isfinite(t) & np.isfinite(p)
                    r = t[m] / p[m]
                    num += float(np.sum(r - np.log(r) - 1.0))
                    den += int(m.sum())
                if bad or den == 0:
                    continue
                q = num / den
                if np.isfinite(q) and q < best[0]:
                    best = (q, lam)
            row[tag] = best
        w = "roughness" if row["rough"][0] < row["ridge"][0] else "ridge"
        print(f"{B.shape[0]:>5}{row['none'][0]:>12.5f}{row['ridge'][0]:>12.5f}"
              f"{row['ridge'][1]:>10g}{row['rough'][0]:>12.5f}"
              f"{row['rough'][1]:>10g}   {w}")
        shortres[str(B.shape[0])] = {k: {"qlike": v[0], "lambda": v[1]}
                                     for k, v in row.items()}
        del F
    out["short_window"] = shortres

    # boxcar b2 reference in the same short-window protocol
    B, _ = w_boxcar()
    F = featurize(ys, B)[idx]
    best = (np.inf, None)
    for lam in LAMS:
        num, den = 0.0, 0
        for s0 in starts:
            tr_ = np.arange(s0 - W, s0)
            te_ = np.arange(s0, s0 + 2000)
            pr, _ = fit_pred(F, yv, tr_, te_, lam, None)
            p, t = pr * blv[te_], rvv[te_]
            m = (t > 0) & (p > 0) & np.isfinite(t) & np.isfinite(p)
            r = t[m] / p[m]
            num += float(np.sum(r - np.log(r) - 1.0)); den += int(m.sum())
        q = num / den
        if q < best[0]:
            best = (q, lam)
    print(f"\n  reference: boxcar b2 (K=12, ridge) short-window QLIKE "
          f"{best[0]:.5f} at lambda={best[1]:g}")
    out["short_window_boxcar_b2"] = {"qlike": best[0], "lambda": best[1]}

    bs = min(shortres, key=lambda k: min(shortres[k]["ridge"]["qlike"],
                                         shortres[k]["rough"]["qlike"]))
    bq = min(shortres[bs]["ridge"]["qlike"], shortres[bs]["rough"]["qlike"])
    print(f"  best spline arm: K={bs} at {bq:.5f}  (delta vs b2 ladder "
          f"{bq - best[0]:+.5f})")
    print("  NOT A RESULT: both sides are oracle-tuned and K was selected on "
          "the\n  evaluation panel too, so a margin this size is inside the "
          "selection noise.")
    out["verdict"] = {"best_spline_K": bs, "best_spline_qlike": bq,
                      "boxcar_qlike": best[0], "delta": bq - best[0],
                      "oracle_tuned": True,
                      "withdrawn": "lambda and K both selected on the "
                                   "evaluation panel; margin is within "
                                   "selection noise"}

    with open(os.path.join(OUT_DIR, "penalty_regime.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote penalty_regime.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
