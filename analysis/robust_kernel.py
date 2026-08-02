"""Is the lag kernel limited by the trial space, or by the estimator?

analysis/kernel_ceiling.py claimed that a rich B-spline space with a roughness
penalty bounds what any parametric manifold could achieve, because every such
manifold is a subset of its span. That argument is wrong, and the way it is
wrong matters.

The span argument bounds APPROXIMATION error -- can the space represent the
kernel. What was actually measured is approximation PLUS ESTIMATION error for
one estimator (penalised least squares, roughness penalty, lambda on a
validation tail). In a low signal-to-noise regime the second term dominates,
and a low-dimensional parametric family is not a weaker hypothesis space but a
STRONGER REGULARISER. That is why a 1-parameter Beta kernel (0.26337) beats a
192-dimensional spline space (0.26355), and why K=384 (0.26372) is worse than
K=192: variance, not expressiveness.

So the open question is not "is there a better manifold" but "is there a better
ESTIMATOR", and the headroom the noise is hiding has to be measured rather than
inferred from a test score. Three parts:

  1. BIAS VERSUS VARIANCE. Training and test QLIKE side by side for every
     kernel. If the rich spaces fit training substantially better while testing
     worse, the approximation headroom is real and estimation error is eating
     it -- which is exactly the claim that better tools should recover.

  2. THE MANIFOLD AS A PRIOR, NOT A CONSTRAINT. Rather than choosing between
     the rich space and the Beta family, shrink the rich space TOWARD the Beta
     kernel: penalise ||phi_w - phi_midas||^2 in kernel space instead of
     ||D2 phi_w||^2, strength chosen on validation. A hard manifold is the
     infinite-penalty limit and the unconstrained spline is the zero limit, so
     an interior optimum that beats both endpoints is direct evidence that the
     manifold is the right prior AND that the rich space has more to give.

  3. SHAPE CONSTRAINTS, WHICH COST NO TUNING PARAMETER. A self-exciting kernel
     should be non-negative and decreasing. Write phi as a sum of trailing
     boxcars with NON-NEGATIVE weights -- phi(u) = sum_{j: u_j >= u} d_j, d_j >=
     0 -- and that shape is imposed exactly, with the predictor linear in d
     (the feature for d_j is the trailing SUM over the first u_j lags). This is
     the HAR ladder generalised to a fine grid with a sign constraint, solved
     by non-negative least squares. It is the cheapest robust estimator
     available here: it buys variance reduction from prior knowledge rather
     than from a penalty that has to be tuned.
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
    BURN, TRAIN_FRAC, U, d2_operator, featurize, implied_kernel_map,
    w_boxcar, w_bspline, w_midas,
)
from inference import dm_test  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
LAMS = [0.0] + [10.0 ** e for e in range(-6, 11)]
MIDAS_TH = (1e-6, 116.956)          # validation-selected shape
NGRID = 64


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def solve(F, y, rows, lam, M=None, w0=None):
    """Least squares with an optional quadratic penalty centred at w0.

    Minimises ||y - a - F w||^2 + lam (w - w0)' M (w - w0). The intercept is
    never penalised.
    """
    A = np.hstack([np.ones((len(rows), 1)), F[rows]])
    k = A.shape[1]
    G = A.T @ A
    b = A.T @ y[rows]
    if lam > 0 and M is not None:
        R = np.zeros((k, k))
        R[1:, 1:] = M
        G = G + lam * R
        if w0 is not None:
            b = b + lam * np.concatenate([[0.0], M @ w0])
    cf = np.linalg.solve(G + 1e-12 * np.eye(k), b)
    r = y[rows] - A @ cf
    return cf, float(r @ r) / len(rows)


def pred(F, bl, rows, cf, sig2):
    A = np.hstack([np.ones((len(rows), 1)), F[rows]])
    return ((A @ cf) ** 2 + sig2) * bl[rows]


def trailing_sum_basis(ngrid=NGRID):
    """Basis whose non-negative coefficients give a decreasing, non-negative
    effective kernel: feature_j is the trailing SUM over the first u_j lags."""
    u = np.unique(np.round(np.exp(np.linspace(0, np.log(U), ngrid))).astype(int))
    B = np.zeros((len(u), U + 1))
    for j, uu in enumerate(u):
        B[j, 1:uu + 1] = 1.0
    return B, u


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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
    ysafe = np.where(np.isfinite(y), y, 0.0)
    yv, blv, rvv = y[idx], bl[idx], rv[idx]
    cut = int(len(idx) * TRAIN_FRAC)
    vcut = int(cut * 0.75)
    FIT, VAL, TRN, TST = (np.arange(vcut), np.arange(vcut, cut),
                          np.arange(cut), np.arange(cut, len(idx)))
    rv_v, rv_t, rv_tr = rvv[VAL], rvv[TST], rvv[TRN]
    print(f"panel {len(idx):,}  fit {len(FIT):,}  valid {len(VAL):,}  "
          f"test {len(TST):,}   ({time.time()-t0:.0f}s)")
    out = {}

    def sel_lambda(F, M, w0=None):
        best = (np.inf, 0.0)
        for lam in LAMS:
            try:
                cf, s2 = solve(F, yv, FIT, lam, M, w0)
            except np.linalg.LinAlgError:
                continue
            q = float(np.mean(qlike_bar(rv_v, pred(F, blv, VAL, cf, s2))))
            if np.isfinite(q) and q < best[0]:
                best = (q, lam)
        return best[1]

    def score(F, lam, M=None, w0=None):
        cf, s2 = solve(F, yv, TRN, lam, M, w0)
        ptr = pred(F, blv, TRN, cf, s2)
        pte = pred(F, blv, TST, cf, s2)
        return (float(np.mean(qlike_bar(rv_tr, ptr))),
                float(np.mean(qlike_bar(rv_t, pte))), pte)

    # ---------------------------------------------------- 1. bias vs variance
    print("\n" + "=" * 78)
    print("1. BIAS VERSUS VARIANCE: TRAINING FIT AGAINST TEST FIT")
    print("=" * 78)
    print(f"  {'kernel':>20}{'K':>6}{'lambda*':>11}{'TRAIN QLIKE':>14}"
          f"{'TEST QLIKE':>13}{'gap':>10}")
    arms, preds = {}, {}
    specs = [("boxcar b2", w_boxcar()[0]),
             ("MIDAS beta", w_midas(*MIDAS_TH)[0]),
             ("bspline 48", w_bspline(48)[0]),
             ("bspline 192", w_bspline(192)[0]),
             ("bspline 384", w_bspline(384)[0])]
    for nm, B in specs:
        F = featurize(ysafe, B)[idx]
        L = implied_kernel_map(B)
        sel = np.unique(np.round(np.exp(np.linspace(0, np.log(U), 96)))
                        .astype(int)) - 1
        A2 = d2_operator(len(sel)) @ L[sel]
        M = A2.T @ A2
        M = M / max(np.trace(M) / M.shape[0], 1e-12)
        lam = sel_lambda(F, M) if B.shape[0] > 1 else 0.0
        qtr, qte, p = score(F, lam, M)
        arms[nm] = {"K": int(B.shape[0]), "lambda": lam, "train_qlike": qtr,
                    "test_qlike": qte, "gap": qte - qtr}
        preds[nm] = p
        print(f"  {nm:>20}{B.shape[0]:>6}{lam:>11g}{qtr:>14.5f}{qte:>13.5f}"
              f"{qte - qtr:>10.5f}")
        del F
    out["bias_variance"] = arms

    bl_tr = arms["boxcar b2"]["train_qlike"]
    rich_tr = min(arms[k]["train_qlike"] for k in
                  ("bspline 48", "bspline 192", "bspline 384"))
    print(f"\n  best rich space fits TRAINING better than the ladder by "
          f"{bl_tr - rich_tr:.5f}")
    print(f"  and its TEST advantage is only "
          f"{arms['boxcar b2']['test_qlike'] - min(arms[k]['test_qlike'] for k in ('bspline 48', 'bspline 192', 'bspline 384')):.5f}")
    print("  -> " + ("approximation headroom EXISTS and estimation error is "
                     "eating most of it"
                     if (bl_tr - rich_tr) > 2 * (
                         arms["boxcar b2"]["test_qlike"]
                         - min(arms[k]["test_qlike"] for k in
                               ("bspline 48", "bspline 192", "bspline 384")))
                     else "the rich space does not fit training much better "
                          "either: little hidden headroom"))

    # ---------------------------------------------------- 2. manifold as prior
    print("\n" + "=" * 78)
    print("2. SHRINK THE RICH SPACE TOWARD THE BETA KERNEL")
    print("=" * 78)
    Bsp, _ = w_bspline(192)
    Fsp = featurize(ysafe, Bsp)[idx]
    Lsp = implied_kernel_map(Bsp)                      # (U, K)
    Msp = Lsp.T @ Lsp
    Msp = Msp / max(np.trace(Msp) / Msp.shape[0], 1e-12)

    # the target kernel: the fitted MIDAS kernel, projected into the spline space
    Bm = w_midas(*MIDAS_TH)[0]
    Fm = featurize(ysafe, Bm)[idx]
    cfm, s2m = solve(Fm, yv, TRN, 0.0)
    phi_target = cfm[1] * Bm[0, 1:]
    del Fm
    w0 = np.linalg.lstsq(Lsp, phi_target, rcond=None)[0]
    print(f"  Beta kernel projected onto the 192-dim spline space: "
          f"relative residual "
          f"{np.linalg.norm(Lsp @ w0 - phi_target) / np.linalg.norm(phi_target):.2e}")
    print(f"\n  {'lambda':>12}{'valid QLIKE':>14}{'TEST QLIKE':>13}")
    rows = []
    for lam in LAMS:
        cf, s2 = solve(Fsp, yv, FIT, lam, Msp, w0)
        qv = float(np.mean(qlike_bar(rv_v, pred(Fsp, blv, VAL, cf, s2))))
        _, qt, _ = score(Fsp, lam, Msp, w0)
        rows.append({"lambda": lam, "valid_qlike": qv, "test_qlike": qt})
        print(f"  {lam:>12g}{qv:>14.5f}{qt:>13.5f}")
    best = rows[int(np.argmin([r["valid_qlike"] for r in rows]))]
    _, q_shrunk, p_shrunk = score(Fsp, best["lambda"], Msp, w0)
    interior = 0.0 < best["lambda"] < LAMS[-1]
    print(f"\n  validation picks lambda = {best['lambda']:g}  ->  TEST "
          f"{q_shrunk:.5f}")
    print(f"  endpoints: spline alone {rows[0]['test_qlike']:.5f}, "
          f"Beta family {arms['MIDAS beta']['test_qlike']:.5f}")
    print("  -> " + ("INTERIOR optimum: the manifold is the right PRIOR and "
                     "the rich space\n     still has something to add"
                     if interior else
                     "the optimum sits at an endpoint: shrinking toward the "
                     "manifold adds nothing\n     beyond the manifold itself"))
    out["manifold_prior"] = {"path": rows, "lambda_star": best["lambda"],
                             "test_qlike": q_shrunk, "interior": bool(interior),
                             "spline_alone": rows[0]["test_qlike"],
                             "beta_family": arms["MIDAS beta"]["test_qlike"]}
    preds["spline shrunk to Beta"] = p_shrunk
    del Fsp

    # ---------------------------------------------------- 3. shape constraints
    print("\n" + "=" * 78)
    print("3. NON-NEGATIVE, DECREASING KERNEL (no tuning parameter)")
    print("=" * 78)
    Bt, ugrid = trailing_sum_basis()
    Ft = featurize(ysafe, Bt)[idx]
    print(f"  {len(ugrid)} trailing-sum features, lags 1..{ugrid[-1]}")

    def fit_nnls(rows):
        from scipy.optimize import nnls
        A = Ft[rows]
        mu, my = A.mean(0), yv[rows].mean()
        Ac, yc = A - mu, yv[rows] - my
        G = Ac.T @ Ac
        b = Ac.T @ yc
        # NNLS on the normal equations via the Cholesky-factor form
        Lc = np.linalg.cholesky(G + 1e-10 * np.eye(G.shape[0]))
        d, _ = nnls(Lc.T, np.linalg.solve(Lc, b))
        r = yc - Ac @ d
        return d, my - mu @ d, float(r @ r) / len(rows)

    d_hat, a_hat, s2n = fit_nnls(TRN)
    nz = int((d_hat > 1e-14).sum())
    p_tr = ((a_hat + Ft[TRN] @ d_hat) ** 2 + s2n) * blv[TRN]
    p_te = ((a_hat + Ft[TST] @ d_hat) ** 2 + s2n) * blv[TST]
    q_tr_n = float(np.mean(qlike_bar(rv_tr, p_tr)))
    q_te_n = float(np.mean(qlike_bar(rv_t, p_te)))
    cf_u, s2u = solve(Ft, yv, TRN, 0.0)
    _, q_te_u, p_te_u = score(Ft, 0.0)
    print(f"  constrained (d >= 0)    train {q_tr_n:.5f}   test {q_te_n:.5f}"
          f"   {nz}/{len(ugrid)} active")
    print(f"  unconstrained, same basis            test {q_te_u:.5f}")
    print(f"  the b2 ladder                        test "
          f"{arms['boxcar b2']['test_qlike']:.5f}")
    print(f"  the Beta family                      test "
          f"{arms['MIDAS beta']['test_qlike']:.5f}")
    out["shape_constrained"] = {
        "n_features": int(len(ugrid)), "n_active": nz,
        "train_qlike": q_tr_n, "test_qlike": q_te_n,
        "unconstrained_test_qlike": q_te_u}
    preds["nonneg decreasing"] = p_te
    preds["unconstrained grid"] = p_te_u
    del Ft

    # ---------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("DM AGAINST THE b2 LADDER (test rows, QLIKE)")
    print("=" * 78)
    base_l = qlike_bar(rv_t, preds["boxcar b2"])
    dms = {}
    for nm, p in preds.items():
        if nm == "boxcar b2":
            continue
        r = dm_test(qlike_bar(rv_t, p), base_l, h=1)
        print(f"  {nm:<26}{r['mean_diff']:+.5f}   t {r['t']:+6.2f}   "
              f"p {r['p']:.3g}")
        dms[nm] = {"mean_diff": r["mean_diff"], "t": r["t"], "p": r["p"]}
    out["dm_vs_ladder"] = dms

    bestnm = min(preds, key=lambda k: float(np.mean(qlike_bar(rv_t, preds[k]))))
    bestq = float(np.mean(qlike_bar(rv_t, preds[bestnm])))
    print(f"\n  best estimator overall: {bestnm} at {bestq:.5f}")
    print(f"  previous claimed ceiling (kernel_ceiling.py): 0.26337")
    print("  -> " + (f"the ceiling was an ESTIMATOR ceiling: a better one "
                     f"reaches {bestq:.5f}"
                     if bestq < 0.26337 else
                     "no estimator here beats the Beta family, so the earlier "
                     "number\n     survives as a ceiling for this class of "
                     "tools too"))
    out["verdict"] = {"best_estimator": bestnm, "best_test_qlike": bestq,
                      "previous_ceiling": 0.26337,
                      "beat_previous_ceiling": bool(bestq < 0.26337)}

    with open(os.path.join(OUT_DIR, "robust_kernel.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote robust_kernel.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
