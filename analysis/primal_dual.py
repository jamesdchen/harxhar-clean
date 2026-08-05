"""The payoff not yet collected: a penalty in the primal IS a kernel in the dual.

This project has worked the primal side hard -- ridge, rank-1, and
shape-projection penalties on the 41 x 12 exogenous coefficient matrix -- and
has touched the data side once descriptively (frequency_and_gram.py: the Gram
spectrum against explained y). It has never written down the identity that
links them:

    penalty  beta' P beta      <==>      kernel  k(x, x') = x' P^-1 x'

so that choosing how to shrink a coefficient matrix is the same act as choosing
a similarity measure between observations. Part 1 verifies that numerically;
if it does not hold to machine precision the correspondence below is wrong.

Part 2 collects the payoff. The penalty that wins under the rolling protocol is
    Q = M - M g (g' M g)^-1 g' M
which is singular with null space span{g} in each channel: shrink every
channel's kernel toward the fixed shape g, leave its amplitude free. Under
P = lambda Q + eps I the inverse has eigenvalue 1/eps along the g directions
and 1/lambda everywhere else, so as lambda -> infinity

    k(x, x')  ->  (1/eps) * sum_c <Z_c, g> <Z'_c, g>

THE DUAL OF THAT PENALTY IS THE 41-DIMENSIONAL FEATURE MAP OF CHANNEL
AMPLITUDES. A rank-1-with-fixed-v constraint on the coefficient matrix is a
41-dim sufficient statistic on the data side -- the rank-K-constraint-is-a-
kernel statement, concrete for this operator.

That predicts something testable against numbers already in hand. The retrieval
experiment found 492 raw exogenous coordinates HURT a distance (dropping them
was worth -0.00976, t = -11.3), while the estimated-SVD coordinate merely TIED
the ambient view. The duality says the right exogenous retrieval metric is
neither: it is the 41 fixed-shape amplitudes. And the shrinkage side has since
shown fixed g beating estimated v-hat by a wide margin, so the amplitude
coordinate should beat both.

    primal penalty              dual retrieval coordinate       status
    ridge ||K||^2               ambient 492 dims                loses
    rank-1, v estimated         sigma_k u_k' Z v_k              ties
    rank-1, v = g fixed         <Z_c, g>, 41 dims               THIS

Anchor, candidate pool, evaluation rows and neighbourhood grid are matched to
analysis/supervised_metric_knn.py so the new arms are directly comparable to
the numbers already reported there.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import dm_test, holm  # noqa: E402
from supervised_metric_knn import (  # noqa: E402
    KS, POOL_STRIDE, TEST_STRIDE, TRAIN_FRAC, knn_predict, load_panel,
    qlike_per_bar,
)

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
W = 24


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from src.evaluation.metrics import apply_duan_smearing
    t0 = time.time()
    out = {}

    X, y, b, names, labels, rungs, ladder_cols, har_cols = load_panel()
    n = X.shape[0]
    cut = int(n * TRAIN_FRAC)
    C, R = len(labels), len(rungs)
    print(f"panel {n:,}   {C} channels x {R} rungs   train {cut:,}")

    # ---------------------------------------------------- 1. the identity
    print("\n" + "=" * 78)
    print("1. PENALTY IN THE PRIMAL == KERNEL IN THE DUAL (numerical check)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    sub = np.arange(2000)
    Xs = np.asarray(X[sub][:, ladder_cols], np.float64)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    ys = y[sub] - y[sub].mean()
    Pq = rng.standard_normal((C * R, C * R))
    Pq = Pq @ Pq.T + C * R * np.eye(C * R)          # a random SPD penalty
    lam = 3.7
    beta = np.linalg.solve(Xs.T @ Xs + lam * Pq, Xs.T @ ys)
    pred_primal = Xs @ beta
    Pinv = np.linalg.inv(Pq)
    Kg = Xs @ Pinv @ Xs.T                            # the induced kernel
    alpha = np.linalg.solve(Kg + lam * np.eye(len(sub)), ys)
    pred_dual = Kg @ alpha
    err = float(np.max(np.abs(pred_primal - pred_dual)))
    rel = err / float(np.max(np.abs(pred_primal)))
    print(f"  primal ridge with penalty P versus kernel ridge with x'P^-1x'")
    print(f"  max |difference| = {err:.3e}   relative {rel:.3e}")
    print("  -> " + ("identity CONFIRMED" if rel < 1e-8 else
                     "identity FAILS -- the correspondence below is wrong"))
    out["identity_check"] = {"max_abs_diff": err, "relative": rel,
                             "confirmed": bool(rel < 1e-8)}
    del Xs, Kg, Pinv

    # ---------------------------------------------------- 2. dual coordinates
    print("\n" + "=" * 78)
    print("2. THE DUAL COORDINATE OF EACH PRIMAL PENALTY, AS A METRIC")
    print("=" * 78)

    Xh = np.asarray(X[:, har_cols], np.float64)
    Atr = np.hstack([np.ones((cut, 1)), Xh[:cut]])
    cf = np.linalg.solve(Atr.T @ Atr, Atr.T @ y[:cut])
    anchor = np.hstack([np.ones((n, 1)), Xh]) @ cf
    e = y - anchor
    print(f"  anchor: HAR-{len(har_cols)} OLS, train R2 = "
          f"{1 - e[:cut].var() / y[:cut].var():.4f}")

    Z = np.asarray(X[:, ladder_cols], np.float64)
    mu_z, sd_z = Z[:cut].mean(0), Z[:cut].std(0)
    sd_z[sd_z < 1e-12] = 1.0
    Zs = (Z - mu_z) / sd_z
    del Z
    Zt = Zs.reshape(n, C, R)

    # the dual coordinate of the WINNING penalty: fixed-shape amplitudes
    amps = {}
    for beta_ in (0.5, 1.0, 1.5):
        g = rungs ** (-beta_)
        g = g / np.linalg.norm(g)
        amps[beta_] = np.einsum("ncr,r->nc", Zt, g)      # (n, 41)
    print(f"  fixed-shape amplitude coordinate: {amps[1.0].shape[1]} dims "
          f"(one per channel)")

    # the dual coordinate of the ESTIMATED-rank penalty, for contrast
    G = Zs[:cut].T @ Zs[:cut] + np.eye(Zs.shape[1])
    w_ex = np.linalg.solve(G, Zs[:cut].T @ e[:cut])
    Km = w_ex.reshape(C, R)
    U_, sv, Vt = np.linalg.svd(Km, full_matrices=False)
    S = np.stack([sv[k] * np.einsum("c,ncr,r->n", U_[:, k], Zt, Vt[k])
                  for k in range(5)], axis=1)
    print(f"  estimated-SVD coordinate: 5 dims, sv share "
          f"{np.round(sv[:3] ** 2 / (sv ** 2).sum(), 3).tolist()}")
    del Zt

    lo = W
    pool_idx = np.arange(lo, cut, POOL_STRIDE)
    test_idx = np.arange(cut, n, TEST_STRIDE)
    test_idx = test_idx[y[test_idx] != 0.0]
    offs = np.arange(-W, 0)
    Vp_pool = e[pool_idx[:, None] + offs[None, :]]
    Vp_test = e[test_idx[:, None] + offs[None, :]]
    ps = e[lo:cut].std()
    print(f"  pool {len(pool_idx):,}   evaluation rows {len(test_idx):,}")

    def std_pair(Mfull):
        m, s = Mfull[pool_idx].mean(0), Mfull[pool_idx].std(0)
        s[s < 1e-12] = 1.0
        d = Mfull.shape[1]
        return ((Mfull[pool_idx] - m) / s / np.sqrt(d),
                (Mfull[test_idx] - m) / s / np.sqrt(d))

    amb_p, amb_t = std_pair(Zs)
    a1_p, a1_t = std_pair(amps[1.0])
    coords = {
        "path only": (Vp_pool / ps, Vp_test / ps),
        "ambient exog (492)": (amb_p, amb_t),
        "SVD coord (5)": std_pair(S),
        "amplitudes b=0.5 (41)": std_pair(amps[0.5]),
        "amplitudes b=1.0 (41)": (a1_p, a1_t),
        "amplitudes b=1.5 (41)": std_pair(amps[1.5]),
        "path + amplitudes": (np.hstack([Vp_pool / ps, a1_p]),
                              np.hstack([Vp_test / ps, a1_t])),
    }

    ytrue, btrue = y[test_idx], b[test_idx]
    anc_te, ep = anchor[test_idx], e[pool_idx]
    pr, tr = apply_duan_smearing(anc_te, ytrue, btrue)
    l_anchor = qlike_per_bar(pr, tr)
    print(f"\n  {'coordinate':<24}{'d':>5}" +
          "".join(f"{'k=' + str(k):>11}" for k in KS))
    print(f"  {'anchor only':<24}{'-':>5}" +
          "".join(f"{l_anchor.mean():>11.5f}" for _ in KS))
    losses, rows = {}, {}
    for nm, (Cp, Ct) in coords.items():
        preds = knn_predict(Cp, Ct, ep, KS)
        row = {}
        for k in KS:
            f = anc_te + preds[k]
            pr, tr = apply_duan_smearing(f, ytrue, btrue)
            lb = qlike_per_bar(pr, tr)
            losses[(nm, k)] = lb
            row[str(k)] = float(lb.mean())
        print(f"  {nm:<24}{Cp.shape[1]:>5}" +
              "".join(f"{row[str(k)]:>11.5f}" for k in KS))
        rows[nm] = {"d": int(Cp.shape[1]), "qlike": row}
    out["arms"] = rows

    print("\n" + "=" * 78)
    print("DM VERSUS THE AMBIENT EXOGENOUS BLOCK (same rows, same k)")
    print("=" * 78)
    pvals, dmres = {}, {}
    print(f"  {'coordinate':<24}" + "".join(f"{'k=' + str(k):>18}" for k in KS))
    for nm in coords:
        if nm == "ambient exog (492)":
            continue
        cells = []
        for k in KS:
            r = dm_test(losses[(nm, k)], losses[("ambient exog (492)", k)], h=1)
            cells.append(f"{r['mean_diff']:+.5f} (t{r['t']:+.1f})")
            pvals[f"{nm}@{k}"] = r["p"]
            dmres[f"{nm}@{k}"] = {"diff": r["mean_diff"], "t": r["t"],
                                  "p": r["p"]}
        print(f"  {nm:<24}" + "".join(f"{c:>18}" for c in cells))
    hm = holm(pvals)
    out["dm_vs_ambient"] = dmres
    out["holm"] = {k: {"adjusted_p": v[0], "reject": bool(v[1])}
                   for k, v in hm.items()}

    kb = min(KS, key=lambda k: losses[("ambient exog (492)", k)].mean())
    amb = losses[("ambient exog (492)", kb)].mean()
    svd = losses[("SVD coord (5)", kb)].mean()
    amp = losses[("amplitudes b=1.0 (41)", kb)].mean()
    print(f"\n  at the ambient arm's best k = {kb}:")
    print(f"    ambient 492      {amb:.5f}")
    print(f"    SVD coord (5)    {svd:.5f}  ({svd - amb:+.5f})")
    print(f"    amplitudes (41)  {amp:.5f}  ({amp - amb:+.5f})")
    print("  -> " + ("DUALITY PREDICTION HELD: the dual coordinate of the "
                     "winning penalty beats\n     both the ambient block and "
                     "the estimated-SVD coordinate"
                     if amp < amb and amp < svd else
                     "prediction FAILED: the fixed-shape amplitude coordinate "
                     "does not beat both"))
    out["verdict"] = {"best_k": kb, "ambient": float(amb), "svd": float(svd),
                      "amplitudes": float(amp),
                      "beats_ambient": bool(amp < amb),
                      "beats_svd": bool(amp < svd)}

    with open(os.path.join(OUT_DIR, "primal_dual.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote primal_dual.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
