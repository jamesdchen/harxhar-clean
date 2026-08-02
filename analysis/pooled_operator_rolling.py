"""Shape-directed pooling on the exogenous operator, under the PRODUCTION protocol.

analysis/pooled_operator.py found the hypothesised ordering with large margins
-- shared-shape pooling beat plain ridge by 0.048 on the 41 x 12 exogenous
block -- but on a protocol that invalidates the magnitude: one fixed 109k-row
training window spanning roughly 2005-2016, tested on 2016-2024. Under that fit
the backbone ALONE (0.16833) beat every arm carrying the exogenous block
(0.242 to 0.301), which contradicts this paper's own attribution result that
every exogenous bucket improves on the backbone. Exogenous relationships that
drift across twenty years survive a rolling refit and are destroyed by a fixed
one, so the fixed-window number measures protocol damage, not pooling.

This reruns the same five penalties under the convention every production
driver uses: a rolling 24,000-bar training window, refit at a fixed cadence,
walking forward across the evaluation region. Penalty strength is chosen on a
validation region that precedes the test region under the identical rolling
protocol, so nothing about the test rows enters the choice.

The penalties are unchanged:

  ridge            ||K||^2                     shrink all 492 toward zero
  pooled           sum_c ||L(K_c - Kbar)||^2   toward the cross-channel mean
                                               shape, the mean unpenalised
  shared shape     sum_c dist(L K_c, span g)^2 toward a common fixed shape,
                                               per-channel amplitude free
  rank 1           ||K - rank1(K_win)||^2      toward a rank-1 operator
                                               recomputed in each window
  smooth           sum_c ||D2 (L K_c)||^2      toward smoothness in lag

The question the fixed-window run could not answer: does shape-directed
shrinkage still beat plain ridge once the protocol is no longer doing the
damage, and does the exogenous block earn its place at all?
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
W = 24000                 # production training window
CAD = 1000                # refit cadence
LAMS = [0.0] + [10.0 ** e for e in range(-2, 9)]


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
    C, R, nh = len(labels), len(rungs), len(har)
    cols = np.concatenate([har, lad])
    F = np.asarray(X[:, cols], np.float64)
    n = len(y)
    p = F.shape[1]
    rvv = y ** 2 * b
    print(f"panel {n:,}   {C} channels x {R} rungs, {nh} backbone   "
          f"({time.time()-t0:.0f}s)")

    # evaluation regions: validation then test, both walked forward
    v0, v1 = int(n * 0.55), int(n * 0.65)
    t1_, t2_ = int(n * 0.65), n
    print(f"  window {W:,}, cadence {CAD}, validation [{v0:,},{v1:,}), "
          f"test [{t1_:,},{t2_:,})")

    Umax = int(rungs[-1])
    L = np.zeros((Umax, R))
    for k, r in enumerate(rungs):
        L[:int(r), k] = 1.0 / r
    M = L.T @ L
    M = M / max(np.trace(M) / R, 1e-12)

    def block(Q):
        P = np.zeros((1 + p, 1 + p))
        for c in range(C):
            s = 1 + nh + c * R
            P[s:s + R, s:s + R] = Q
        return P

    P_ridge = block(np.eye(R))
    P_pool = np.zeros((1 + p, 1 + p))
    for a in range(C):
        for bb in range(C):
            co = (1.0 if a == bb else 0.0) - 1.0 / C
            sa, sb = 1 + nh + a * R, 1 + nh + bb * R
            P_pool[sa:sa + R, sb:sb + R] += co * M
    g = rungs ** (-1.0)
    g = g / np.linalg.norm(g)
    Mg = M @ g
    P_shape = block(M - np.outer(Mg, Mg) / float(g @ M @ g))
    sel = np.unique(np.round(np.exp(np.linspace(0, np.log(Umax), 48)))
                    .astype(int)) - 1
    A2 = d2_op(len(sel)) @ L[sel]
    Q_sm = A2.T @ A2
    P_smooth = block(Q_sm / max(np.trace(Q_sm) / R, 1e-12))

    PENS = {"ridge": P_ridge, "pooled to mean shape": P_pool,
            "shared shape (amp free)": P_shape, "smooth in lag": P_smooth}
    ARMS = list(PENS) + ["rank-1 prior", "backbone only"]

    def roll(lo, hi, plan):
        """Walk forward over [lo,hi). plan maps arm -> lambda. Returns preds."""
        outp = {a: np.full(hi - lo, np.nan) for a in plan}
        for s in range(lo, hi, CAD):
            e = min(s + CAD, hi)
            tr = slice(s - W, s)
            A = np.hstack([np.ones((W, 1)), F[tr]])
            G = A.T @ A
            c = A.T @ y[tr]
            yt = y[tr]
            Ae = np.hstack([np.ones((e - s, 1)), F[s:e]])
            # rank-1 prior recomputed in this window
            cf0 = np.linalg.solve(G + 1.0 * P_ridge + 1e-8 * np.eye(1 + p), c)
            K0 = cf0[1 + nh:].reshape(C, R)
            Uu, Ss, Vt = np.linalg.svd(K0, full_matrices=False)
            w0 = np.zeros(1 + p)
            w0[1 + nh:] = np.outer(Uu[:, 0] * Ss[0], Vt[0]).ravel()
            for arm, lam in plan.items():
                if arm == "backbone only":
                    Gb = G[:1 + nh, :1 + nh]
                    cfb = np.linalg.solve(Gb + 1e-8 * np.eye(1 + nh),
                                          c[:1 + nh])
                    rr = yt - A[:, :1 + nh] @ cfb
                    s2 = float(rr @ rr) / W
                    outp[arm][s - lo:e - lo] = (
                        (Ae[:, :1 + nh] @ cfb) ** 2 + s2) * b[s:e]
                    continue
                P = P_ridge if arm == "rank-1 prior" else PENS[arm]
                bb_ = c.copy()
                if arm == "rank-1 prior" and lam > 0:
                    bb_ = bb_ + lam * (P @ w0)
                cf = np.linalg.solve(G + lam * P + 1e-8 * np.eye(1 + p), bb_)
                rr = yt - A @ cf
                s2 = float(rr @ rr) / W
                outp[arm][s - lo:e - lo] = ((Ae @ cf) ** 2 + s2) * b[s:e]
            del A, Ae, G
        return outp

    # ---- select lambda per arm on the validation region
    print("\n" + "=" * 78)
    print("LAMBDA SELECTION ON THE VALIDATION REGION (rolling, pre-test)")
    print("=" * 78)
    rv_v = rvv[v0:v1]
    okv = rv_v > 0
    chosen = {}
    print(f"  {'arm':>26}{'lambda*':>11}{'valid QLIKE':>14}")
    for arm in ARMS:
        if arm == "backbone only":
            chosen[arm] = 0.0
            continue
        best = (np.inf, 0.0)
        for lam in LAMS:
            pv = roll(v0, v1, {arm: lam})[arm]
            m = okv & (pv > 0) & np.isfinite(pv)
            if m.sum() == 0:
                continue
            q = float(np.mean(qlike_bar(rv_v[m], pv[m])))
            if np.isfinite(q) and q < best[0]:
                best = (q, lam)
        chosen[arm] = best[1]
        print(f"  {arm:>26}{best[1]:>11g}{best[0]:>14.5f}")
    print(f"  ({time.time()-t0:.0f}s)")

    # ---- walk the test region once, all arms
    print("\n" + "=" * 78)
    print("TEST REGION (rolling 24,000-bar window, cadence 1,000)")
    print("=" * 78)
    preds = roll(t1_, t2_, chosen)
    rv_t = rvv[t1_:t2_]
    good = rv_t > 0
    for a in preds:
        good &= np.isfinite(preds[a]) & (preds[a] > 0)
    print(f"  scored on {int(good.sum()):,} bars common to every arm")
    rv_g = rv_t[good]
    print(f"\n  {'arm':>26}{'lambda*':>11}{'TEST QLIKE':>13}{'TEST MSE':>13}")
    res = {}
    for arm in ARMS:
        pg = preds[arm][good]
        q = float(np.mean(qlike_bar(rv_g, pg)))
        res[arm] = {"lambda": chosen[arm], "test_qlike": q,
                    "test_mse": float(np.mean((rv_g - pg) ** 2))}
        print(f"  {arm:>26}{chosen[arm]:>11g}{q:>13.5f}"
              f"{res[arm]['test_mse']:>13.4e}")

    print("\n" + "=" * 78)
    print("DM AGAINST PLAIN RIDGE, AND AGAINST THE BACKBONE")
    print("=" * 78)
    l_ridge = qlike_bar(rv_g, preds["ridge"][good])
    l_bb = qlike_bar(rv_g, preds["backbone only"][good])
    dms = {}
    for arm in ARMS:
        if arm == "ridge":
            continue
        pg = preds[arm][good]
        a = dm_test(qlike_bar(rv_g, pg), l_ridge, h=1)
        c2 = dm_test(qlike_bar(rv_g, pg), l_bb, h=1)
        print(f"  {arm:>26}  vs ridge {a['mean_diff']:+.5f} t {a['t']:+6.2f}"
              f" | vs backbone {c2['mean_diff']:+.5f} t {c2['t']:+6.2f}")
        dms[arm] = {"vs_ridge": {"diff": a["mean_diff"], "t": a["t"],
                                 "p": a["p"]},
                    "vs_backbone": {"diff": c2["mean_diff"], "t": c2["t"],
                                    "p": c2["p"]}}
    out = {"window": W, "cadence": CAD, "n_scored": int(good.sum()),
           "arms": res, "dm": dms}

    exog_helps = res["shared shape (amp free)"]["test_qlike"] < \
        res["backbone only"]["test_qlike"]
    best_arm = min(res, key=lambda k: res[k]["test_qlike"])
    print(f"\n  best arm: {best_arm} at {res[best_arm]['test_qlike']:.5f}")
    print("  -> " + ("the exogenous block now EARNS its place under the "
                     "rolling protocol,\n     so the fixed-window result was "
                     "protocol damage as suspected"
                     if exog_helps else
                     "the exogenous block still does not beat the backbone "
                     "even rolling:\n     the fixed window was not the whole "
                     "problem"))
    gain = res["ridge"]["test_qlike"] - res["shared shape (amp free)"]["test_qlike"]
    print(f"  shape-directed pooling versus plain ridge: {gain:+.5f} "
          f"(fixed-window run gave +0.04813)")
    out["verdict"] = {"best_arm": best_arm,
                      "exog_beats_backbone": bool(exog_helps),
                      "shape_vs_ridge": float(gain),
                      "fixed_window_shape_vs_ridge": 0.04813}

    with open(os.path.join(OUT_DIR, "pooled_operator_rolling.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote pooled_operator_rolling.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
