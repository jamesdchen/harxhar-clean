"""Directed shrinkage where the information actually is: the exogenous operator.

Everything measured so far about prior-directed shrinkage concerns the SELF
kernel, which this paper's own attribution says carries little of the signal --
a Frisch-Waugh-Lovell decomposition assigns 96% of explainable variance to the
backbone, and the exogenous buckets are where the increments come from. The
self-kernel result (-0.00311 at t = -7.53 from shrinking a rich spline toward a
tapered power law) is therefore a proof of concept in the wrong place.

The exogenous block is 41 channels x 12 base-2 rungs = 492 columns, fitted
jointly. It is the textbook setting for partial pooling: 41 kernels that are
each weakly identified, plausibly sharing a shape, differing mainly in
amplitude and sign. That is exactly what the CP decomposition supported --
near-equal fit to six independent per-session SVDs at a fraction of the
parameters, i.e. "the geometry is static and only amplitudes move".

Ridge shrinks all 492 coefficients toward ZERO, which throws away that
structure: it treats a channel whose kernel disagrees with every other channel
exactly like one that agrees. The alternatives here all keep the amplitude free
and shrink only the SHAPE:

  ridge            ||K||^2                        shrink to nothing
  pooled           sum_c ||L(K_c - Kbar)||^2      shrink to the cross-channel
                                                  mean shape, mean unpenalised
  shared shape     sum_c dist(L K_c, span{g})^2   shrink each channel's kernel
                                                  toward a common FIXED shape g,
                                                  per-channel amplitude free
  rank 1           ||K - rank1(K_train)||^2       shrink toward a training-only
                                                  rank-1 operator (the CP
                                                  hypothesis as a point prior)
  smooth           sum_c ||D2 (L K_c)||^2         shrink toward smoothness in
                                                  lag, no shape assumed

"Shared shape" is the direct analogue of what worked on the self kernel, and
its penalty has a closed form: minimising over the free amplitude a_c leaves
K_c' Q K_c with Q = M - M k (k' M k)^-1 k' M, where M = L'L and k is the shape
in coefficient space. Convex, no alternating minimisation, rank 11 per channel.

Every penalty strength is chosen on a validation tail carved out of the
training portion, and the arms are compared per-bar against the same backbone.
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

from inference import dm_test  # noqa: E402

D = os.path.join(ROOT, "results", "b2_mmap")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
TRAIN_FRAC = 0.60
LAMS = [0.0] + [10.0 ** e for e in range(-3, 10)]


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def d2_op(m):
    Dm = np.zeros((m - 2, m))
    for i in range(m - 2):
        Dm[i, i], Dm[i, i + 1], Dm[i, i + 2] = 1.0, -2.0, 1.0
    return Dm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    chan: dict[str, list] = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan:
        chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    rungs = np.array([r for r, _ in chan[labels[0]]], float)
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names)
                    if re.match(r"^har_ma_\d+$", nm)], int)
    C, R = len(labels), len(rungs)
    print(f"panel {X.shape[0]:,}   {C} channels x {R} rungs = {C*R} exog "
          f"columns, {len(har)} backbone")

    cols = np.concatenate([har, lad])
    F = np.asarray(X[:, cols], np.float64)
    n = len(y)
    cut = int(n * TRAIN_FRAC)
    vcut = int(cut * 0.75)
    FIT, VAL, TRN, TST = (np.arange(vcut), np.arange(vcut, cut),
                          np.arange(cut), np.arange(cut, n))
    rvv = y ** 2 * b                      # true_raw, the Duan convention
    good = rvv > 0
    VAL = VAL[good[VAL]]
    TST = TST[good[TST]]
    print(f"  fit {len(FIT):,}  valid {len(VAL):,}  test {len(TST):,}  "
          f"({time.time()-t0:.0f}s)")

    nh, p = len(har), F.shape[1]
    mu, sd = F[FIT].mean(0), F[FIT].std(0)
    sd[sd < 1e-12] = 1.0
    Z = (F - mu) / sd
    del F

    def build(rows):
        A = np.hstack([np.ones((len(rows), 1)), Z[rows]])
        return A.T @ A, A.T @ y[rows], A

    G_f, c_f, A_f = build(FIT)
    G_r, c_r, A_r = build(TRN)
    A_v = np.hstack([np.ones((len(VAL), 1)), Z[VAL]])
    A_t = np.hstack([np.ones((len(TST), 1)), Z[TST]])
    print(f"  grams built ({time.time()-t0:.0f}s)")

    # kernel map on the base-2 ladder: phi(u) = sum_{r >= u} w_r / r
    Umax = int(rungs[-1])
    L = np.zeros((Umax, R))
    for k, r in enumerate(rungs):
        L[:int(r), k] = 1.0 / r
    M = L.T @ L
    M = M / max(np.trace(M) / R, 1e-12)

    def block(Q):
        """Lift a per-channel R x R penalty to the full 1 + nh + C*R space."""
        P = np.zeros((1 + p, 1 + p))
        for c in range(C):
            s = 1 + nh + c * R
            P[s:s + R, s:s + R] = Q
        return P

    # ---- the five penalties
    pen = {}
    pen["ridge"] = block(np.eye(R))

    # pooled: sum_c ||L(K_c - Kbar)||^2  ->  (I_C - J/C) kron M
    P_pool = np.zeros((1 + p, 1 + p))
    for a in range(C):
        for bb in range(C):
            co = (1.0 if a == bb else 0.0) - 1.0 / C
            sa, sb = 1 + nh + a * R, 1 + nh + bb * R
            P_pool[sa:sa + R, sb:sb + R] += co * M
    pen["pooled to mean shape"] = P_pool

    # shared shape: distance to span{g}, amplitude free
    g = rungs ** (-1.0)                      # power-law-ish shape on the rungs
    g = g / np.linalg.norm(g)
    Mg = M @ g
    Q_shape = M - np.outer(Mg, Mg) / float(g @ M @ g)
    pen["shared shape (amp free)"] = block(Q_shape)

    # smoothness of each channel's implied kernel in log-lag
    sel = np.unique(np.round(np.exp(np.linspace(0, np.log(Umax), 48)))
                    .astype(int)) - 1
    A2 = d2_op(len(sel)) @ L[sel]
    Q_sm = A2.T @ A2
    Q_sm = Q_sm / max(np.trace(Q_sm) / R, 1e-12)
    pen["smooth in lag"] = block(Q_sm)

    def solve(G, c, A, rows, lam, P, w0=None):
        k = G.shape[0]
        bb = c.copy()
        if lam > 0 and w0 is not None:
            bb += lam * (P @ w0)
        cf = np.linalg.solve(G + lam * P + 1e-8 * np.eye(k), bb)
        r = y[rows] - A @ cf
        return cf, float(r @ r) / len(rows)

    # rank-1 prior needs a training-only operator first
    cf0, _ = solve(G_f, c_f, A_f, FIT, 1.0, pen["ridge"])
    K0 = cf0[1 + nh:].reshape(C, R)
    Uu, Ss, Vt = np.linalg.svd(K0, full_matrices=False)
    K1 = np.outer(Uu[:, 0] * Ss[0], Vt[0])
    w0_rank1 = np.zeros(1 + p)
    w0_rank1[1 + nh:] = K1.ravel()
    print(f"  rank-1 share of the training operator: "
          f"{Ss[0]**2 / (Ss**2).sum():.3f}")

    # ---------------------------------------------------- run the arms
    print("\n" + "=" * 78)
    print("DIRECTED SHRINKAGE ON THE 41 x 12 EXOGENOUS OPERATOR")
    print("=" * 78)
    print(f"  {'penalty':>26}{'lambda*':>11}{'TEST QLIKE':>13}"
          f"{'TEST MSE':>13}")
    rv_v, rv_t = rvv[VAL], rvv[TST]
    res, preds = {}, {}

    def evaluate(P, w0=None):
        best = (np.inf, 0.0)
        for lam in LAMS:
            try:
                cf, s2 = solve(G_f, c_f, A_f, FIT, lam, P, w0)
            except np.linalg.LinAlgError:
                continue
            pv = ((A_v @ cf) ** 2 + s2) * b[VAL]
            q = float(np.mean(qlike_bar(rv_v, pv)))
            if np.isfinite(q) and q < best[0]:
                best = (q, lam)
        cf, s2 = solve(G_r, c_r, A_r, TRN, best[1], P, w0)
        pt = ((A_t @ cf) ** 2 + s2) * b[TST]
        return best[1], pt, cf

    # backbone-only reference
    Gb = G_r[:1 + nh, :1 + nh].copy()
    cb = c_r[:1 + nh].copy()
    cfb = np.linalg.solve(Gb + 1e-8 * np.eye(1 + nh), cb)
    rb = y[TRN] - A_r[:, :1 + nh] @ cfb
    s2b = float(rb @ rb) / len(TRN)
    p_bb = ((A_t[:, :1 + nh] @ cfb) ** 2 + s2b) * b[TST]
    q_bb = float(np.mean(qlike_bar(rv_t, p_bb)))
    print(f"  {'backbone only (no exog)':>26}{'--':>11}{q_bb:>13.5f}"
          f"{np.mean((rv_t - p_bb) ** 2):>13.4e}")

    order = ["ridge", "pooled to mean shape", "shared shape (amp free)",
             "smooth in lag"]
    for tag in order:
        lam, pt, _ = evaluate(pen[tag])
        q = float(np.mean(qlike_bar(rv_t, pt)))
        preds[tag] = pt
        res[tag] = {"lambda": lam, "test_qlike": q,
                    "test_mse": float(np.mean((rv_t - pt) ** 2))}
        print(f"  {tag:>26}{lam:>11g}{q:>13.5f}"
              f"{res[tag]['test_mse']:>13.4e}")

    lam, pt, _ = evaluate(pen["ridge"], w0_rank1)
    q = float(np.mean(qlike_bar(rv_t, pt)))
    preds["rank-1 prior"] = pt
    res["rank-1 prior"] = {"lambda": lam, "test_qlike": q,
                           "test_mse": float(np.mean((rv_t - pt) ** 2))}
    print(f"  {'rank-1 prior':>26}{lam:>11g}{q:>13.5f}"
          f"{res['rank-1 prior']['test_mse']:>13.4e}")
    out = {"n_channels": C, "n_rungs": R, "backbone_only_qlike": q_bb,
           "rank1_share": float(Ss[0] ** 2 / (Ss ** 2).sum()), "arms": res}

    # ---------------------------------------------------- inference
    print("\n" + "=" * 78)
    print("DM AGAINST PLAIN RIDGE ON THE SAME COLUMNS")
    print("=" * 78)
    base_l = qlike_bar(rv_t, preds["ridge"])
    dms = {}
    for tag, pt in preds.items():
        if tag == "ridge":
            continue
        a = dm_test(qlike_bar(rv_t, pt), base_l, h=1)
        c2 = dm_test((rv_t - pt) ** 2, (rv_t - preds["ridge"]) ** 2, h=1)
        print(f"  {tag:>26}  QLIKE {a['mean_diff']:+.5f} t {a['t']:+6.2f} "
              f"p {a['p']:.3g} | MSE t {c2['t']:+6.2f}")
        dms[tag] = {"qlike_diff": a["mean_diff"], "qlike_t": a["t"],
                    "qlike_p": a["p"], "mse_t": c2["t"],
                    "mse_diff": c2["mean_diff"]}
    out["dm_vs_ridge"] = dms

    best_tag = min(res, key=lambda k: res[k]["test_qlike"])
    gain = res["ridge"]["test_qlike"] - res[best_tag]["test_qlike"]
    print(f"\n  best: {best_tag} at {res[best_tag]['test_qlike']:.5f}, "
          f"{gain:+.5f} versus plain ridge")
    print(f"  for scale, the self-kernel version of this idea was worth "
          f"0.00311")
    out["verdict"] = {"best": best_tag, "gain_vs_ridge": float(gain),
                      "self_kernel_reference": 0.00311,
                      "directed_beats_ridge": bool(gain > 0)}

    with open(os.path.join(OUT_DIR, "pooled_operator.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote pooled_operator.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
