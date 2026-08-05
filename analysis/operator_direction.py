"""Where does the operator's lag direction come from: the SVD, or a prior?

Two threads in this project turn out to be the same object from opposite ends.

The SVD thread treats the exogenous operator K (41 channels x 12 rungs) as
low-rank and ESTIMATES its factors: pooled and per-session SVDs, amplitudes
a_k = u_k' K v_k on a common frame, CP/Tucker decompositions, and the retrieval
coordinate sigma_k u_k' Z v_k.

The shrinkage thread IMPOSES one: the "shared shape" penalty asks each channel's
kernel to lie near span{g} with its amplitude free, which is
    K ~ alpha g'
-- rank one, with the RIGHT factor pinned rather than estimated.

Same rank, same structure. Under the rolling protocol they land at opposite
ends of the battery: rank-1 with both factors estimated tests at 0.30711, the
worst arm; rank-1 with g fixed tests at 0.14878, the best. So the payoff is not
"low rank helps" -- the SVD work established that -- it is NOT ESTIMATING v.

The full-sample operator says why. Its leading lag direction is only a rough
match to g = rung^-1 (|cos| = 0.729), and the per-session v1 wanders against the
pooled frame with |cos| from 0.449 to 0.866. A direction that unstable over six
large session bins cannot be estimated reliably inside a 24,000-bar window,
so estimating it imports that instability while a fixed shape imports none.

This isolates the axis. The rolling protocol, the penalty family, the panel and
the evaluation rows are all held fixed; the ONLY thing that varies is where the
lag direction comes from:

    estimated in-window      the SVD of each window's own ridge fit
    frozen from an early     one SVD on an early block disjoint from validation
      block                  and test, then held fixed forever
    power law rung^-beta     imposed, for several exponents
    rank 2, frozen           two directions from the early block
    rank 2, imposed          two fixed power laws

If the frozen-SVD arm performs like the imposed-shape arms, the operator's
estimated content is fine and only its INSTABILITY was the problem. If it
performs like the in-window arm, the leading direction is genuinely
non-stationary and no amount of freezing rescues it. Those are different
diagnoses with different fixes, and the current evidence cannot tell them apart.
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
W = 24000
CAD = 1000
LAMS = [0.0] + [10.0 ** e for e in range(-1, 10)]
FREEZE_END = 120000        # early block for the frozen SVD, before validation


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


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
    F = np.asarray(X[:, np.concatenate([har, lad])], np.float64)
    n, p = len(y), F.shape[1]
    rvv = y ** 2 * b
    v0, v1 = int(n * 0.55), int(n * 0.65)
    t1_, t2_ = int(n * 0.65), n
    print(f"panel {n:,}  {C}x{R} operator   window {W:,} cadence {CAD}")
    print(f"  freeze block [0,{FREEZE_END:,})  validation [{v0:,},{v1:,})  "
          f"test [{t1_:,},{t2_:,})   ({time.time()-t0:.0f}s)")

    Umax = int(rungs[-1])
    L = np.zeros((Umax, R))
    for k, r in enumerate(rungs):
        L[:int(r), k] = 1.0 / r
    M = L.T @ L
    M = M / max(np.trace(M) / R, 1e-12)
    P_ridge = np.zeros((1 + p, 1 + p))
    for c in range(C):
        s = 1 + nh + c * R
        P_ridge[s:s + R, s:s + R] = np.eye(R)

    def pen_from_V(V):
        """Penalty pushing each channel's kernel into span{rows of V}."""
        V = np.atleast_2d(V)
        MV = M @ V.T
        Q = M - MV @ np.linalg.solve(V @ M @ V.T + 1e-12 * np.eye(len(V)), MV.T)
        P = np.zeros((1 + p, 1 + p))
        for c in range(C):
            s = 1 + nh + c * R
            P[s:s + R, s:s + R] = Q
        return P

    def window_operator(s):
        A = np.hstack([np.ones((W, 1)), F[s - W:s]])
        G = A.T @ A
        c = A.T @ y[s - W:s]
        cf = np.linalg.solve(G + 1.0 * P_ridge + 1e-8 * np.eye(1 + p), c)
        return cf[1 + nh:].reshape(C, R)

    # ---- frozen directions from the early block
    Kf = window_operator(FREEZE_END)
    Uf, Sf, Vtf = np.linalg.svd(Kf, full_matrices=False)
    print(f"  frozen-block operator sv share: "
          f"{np.round(Sf**2 / (Sf**2).sum(), 3)[:4].tolist()}")

    def pl(beta):
        g = rungs ** (-beta)
        return g / np.linalg.norm(g)

    ARMS = {
        "v estimated in-window": None,                 # special-cased
        "v frozen from early SVD": Vtf[:1],
        "v = rung^-0.5": pl(0.5)[None, :],
        "v = rung^-1.0": pl(1.0)[None, :],
        "v = rung^-1.5": pl(1.5)[None, :],
        "rank2 frozen SVD": Vtf[:2],
        "rank2 imposed": np.vstack([pl(0.5), pl(1.5)]),
    }
    for tag, V in ARMS.items():
        if V is not None:
            print(f"  {tag:>26}  |cos(row1, rung^-1)| = "
                  f"{abs(V[0] @ pl(1.0)):.3f}")

    pens = {t: (None if V is None else pen_from_V(V)) for t, V in ARMS.items()}

    def roll(lo, hi, plan, track=False):
        outp = {a: np.full(hi - lo, np.nan) for a in plan}
        vprev, coss = None, []
        for s in range(lo, hi, CAD):
            e = min(s + CAD, hi)
            A = np.hstack([np.ones((W, 1)), F[s - W:s]])
            G = A.T @ A
            c = A.T @ y[s - W:s]
            yt = y[s - W:s]
            Ae = np.hstack([np.ones((e - s, 1)), F[s:e]])
            Kw = None
            for arm, lam in plan.items():
                if ARMS[arm] is None:
                    if Kw is None:
                        cf0 = np.linalg.solve(
                            G + 1.0 * P_ridge + 1e-8 * np.eye(1 + p), c)
                        Kw = cf0[1 + nh:].reshape(C, R)
                        vw = np.linalg.svd(Kw, full_matrices=False)[2][:1]
                        if track:
                            if vprev is not None:
                                coss.append(abs(float(vw[0] @ vprev[0])))
                            vprev = vw
                    P = pen_from_V(vw)
                else:
                    P = pens[arm]
                cf = np.linalg.solve(G + lam * P + 1e-8 * np.eye(1 + p), c)
                rr = yt - A @ cf
                s2 = float(rr @ rr) / W
                outp[arm][s - lo:e - lo] = ((Ae @ cf) ** 2 + s2) * b[s:e]
            del A, Ae, G
        return (outp, coss) if track else (outp, None)

    # ---- lambda on validation
    print("\n" + "=" * 78)
    print("LAMBDA SELECTION (validation region, rolling, pre-test)")
    print("=" * 78)
    rv_v = rvv[v0:v1]
    okv = rv_v > 0
    chosen = {}
    print(f"  {'arm':>26}{'lambda*':>11}{'valid QLIKE':>14}")
    for arm in ARMS:
        best = (np.inf, 0.0)
        for lam in LAMS:
            pv = roll(v0, v1, {arm: lam})[0][arm]
            m = okv & (pv > 0) & np.isfinite(pv)
            if m.sum() == 0:
                continue
            q = float(np.mean(qlike_bar(rv_v[m], pv[m])))
            if np.isfinite(q) and q < best[0]:
                best = (q, lam)
        chosen[arm] = best[1]
        print(f"  {arm:>26}{best[1]:>11g}{best[0]:>14.5f}")
    print(f"  ({time.time()-t0:.0f}s)")

    # ---- test walk
    print("\n" + "=" * 78)
    print("TEST REGION")
    print("=" * 78)
    preds, coss = roll(t1_, t2_, chosen, track=True)
    rv_t = rvv[t1_:t2_]
    good = rv_t > 0
    for a in preds:
        good &= np.isfinite(preds[a]) & (preds[a] > 0)
    rv_g = rv_t[good]
    print(f"  scored on {int(good.sum()):,} bars common to every arm")
    if coss:
        print(f"  stability of the IN-WINDOW v1: mean |cos| between "
              f"consecutive windows = {np.mean(coss):.3f} "
              f"(min {np.min(coss):.3f})")
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
    print("DM AGAINST THE IN-WINDOW ESTIMATED DIRECTION")
    print("=" * 78)
    base = qlike_bar(rv_g, preds["v estimated in-window"][good])
    dms = {}
    for arm in ARMS:
        if arm == "v estimated in-window":
            continue
        r = dm_test(qlike_bar(rv_g, preds[arm][good]), base, h=1)
        print(f"  {arm:>26}{r['mean_diff']:>+11.5f}  t {r['t']:>+7.2f}  "
              f"p {r['p']:.3g}")
        dms[arm] = {"diff": r["mean_diff"], "t": r["t"], "p": r["p"]}

    frozen = res["v frozen from early SVD"]["test_qlike"]
    imposed = min(res[a]["test_qlike"] for a in
                  ("v = rung^-0.5", "v = rung^-1.0", "v = rung^-1.5"))
    inwin = res["v estimated in-window"]["test_qlike"]
    print(f"\n  in-window {inwin:.5f} | frozen {frozen:.5f} | "
          f"best imposed {imposed:.5f}")
    verdict = ("FREEZING IS ENOUGH: the operator's estimated content is fine "
               "and only its\n  instability hurt, so the fix is to stop "
               "re-estimating it, not to replace it."
               if abs(frozen - imposed) < abs(frozen - inwin) else
               "FREEZING IS NOT ENOUGH: the early-block direction performs "
               "like the\n  re-estimated one, so the leading direction is "
               "genuinely non-stationary and\n  an imposed shape is doing "
               "something the data's own direction cannot.")
    print("  -> " + verdict)
    out = {"n_scored": int(good.sum()), "arms": res, "dm_vs_in_window": dms,
           "v1_consecutive_cos_mean": float(np.mean(coss)) if coss else None,
           "v1_consecutive_cos_min": float(np.min(coss)) if coss else None,
           "frozen_sv_share": (Sf ** 2 / (Sf ** 2).sum())[:4].tolist(),
           "verdict": verdict.replace("\n", " ")}
    with open(os.path.join(OUT_DIR, "operator_direction.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote operator_direction.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
