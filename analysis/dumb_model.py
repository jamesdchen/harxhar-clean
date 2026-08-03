"""The stripped-down model this whole investigation actually supports.

Everything measured in this session points the same way, so the question is
what survives if you keep only what earned its place.

  * The trial space is a non-lever: every lag basis tried -- boxcar, spline,
    Chebyshev, collocation, MIDAS, greedy reduced basis -- lands within 0.002
    QLIKE. The base-2 ladder is fine, and it beats daily/weekly/monthly by
    0.00082 at t = -10.6 for no added complexity.
  * Estimated structure is a liability: Laplacian eigenmaps, PCA, the operator
    SVD (in-window and frozen), and rank-1 CP factors all lost to a FIXED
    alternative.
  * The direction of shrinkage beat its amount, and the effective-df
    measurement says the winning matrix penalty spends only ~54 degrees of
    freedom on a 505-column design -- almost exactly 1 + 12 backbone + 41
    channel amplitudes, the dimension of its own null space.

That last fact is the interesting one. If the elaborate penalty is really only
buying one amplitude per channel, the same thing should be obtainable by
PRECOMPUTING those amplitudes and running plain ridge -- no penalty geometry,
no SVD, no tuning beyond a single lambda.

    dumb        1 + 12 HAR rungs + 41 channel amplitudes <Z_c, g>, g = rung^-1
    backbone    1 + 12 HAR rungs
    ridge       1 + 12 + 492 raw ladder columns, plain ridge
    shaped      the same 505 columns with the shape-projection penalty

If dumb matches shaped, the paper's most defensible modelling result can be
stated as a feature construction rather than an estimator, which is a much
cheaper thing to adopt and a much easier thing to check.

Run on the pre-fix cache so all four arms share the same data; the pipeline
fix changes every arm together and is measured separately.
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
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 9)]


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
    C, R, nh = len(labels), 12, len(har)
    n = len(y)
    rvv = y ** 2 * b

    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64)
    g = rungs ** (-1.0)
    g = g / np.linalg.norm(g)
    Fa = Fl.reshape(n, C, R) @ g                       # (n, 41) amplitudes
    print(f"panel {n:,}   backbone {nh}   ladder {C*R}   amplitudes {C}")

    Umax = int(rungs[-1])
    L = np.zeros((Umax, R))
    for k, r in enumerate(rungs):
        L[:int(r), k] = 1.0 / r
    M = L.T @ L
    M = M / max(np.trace(M) / R, 1e-12)
    Mg = M @ g
    Q = M - np.outer(Mg, Mg) / float(g @ M @ g)

    designs = {
        "backbone (12)": Fh,
        "dumb: backbone + 41 amplitudes": np.hstack([Fh, Fa]),
        "ridge on 492 raw": np.hstack([Fh, Fl]),
        "shaped penalty on 492": np.hstack([Fh, Fl]),
    }
    pen = {}
    for tag, Fd in designs.items():
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        if tag == "shaped penalty on 492":
            for c in range(C):
                s = 1 + nh + c * R
                P[s:s + R, s:s + R] = Q
        elif tag == "ridge on 492 raw":
            for c in range(C):
                s = 1 + nh + c * R
                P[s:s + R, s:s + R] = np.eye(R)
        elif tag.startswith("dumb"):
            P[1 + nh:, 1 + nh:] = np.eye(C)
        pen[tag] = P

    v0, v1 = int(n * 0.55), int(n * 0.65)
    t1_, t2_ = int(n * 0.65), n
    print(f"  validation [{v0:,},{v1:,})  test [{t1_:,},{t2_:,})")

    def roll(tag, lo, hi, lam):
        Fd, P = designs[tag], pen[tag]
        out = np.full(hi - lo, np.nan)
        for s in range(lo, hi, CAD):
            e = min(s + CAD, hi)
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            G = A.T @ A
            c = A.T @ y[s - W:s]
            cf = np.linalg.solve(G + lam * P + 1e-8 * np.eye(len(P)), c)
            rr = y[s - W:s] - A @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fd[s:e]])
            out[s - lo:e - lo] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del A, G, Ae
        return out

    print("\n" + "=" * 78)
    print("LAMBDA ON VALIDATION, THEN ONE WALK OF THE TEST REGION")
    print("=" * 78)
    rv_v = rvv[v0:v1]
    chosen, preds = {}, {}
    print(f"  {'design':>34}{'cols':>7}{'lambda*':>11}{'valid':>10}")
    for tag in designs:
        best = (np.inf, 0.0)
        cands = [0.0] if tag == "backbone (12)" else LAMS
        for lam in cands:
            pv = roll(tag, v0, v1, lam)
            m = (rv_v > 0) & np.isfinite(pv) & (pv > 0)
            q = float(np.mean(qlike_bar(rv_v[m], pv[m])))
            if np.isfinite(q) and q < best[0]:
                best = (q, lam)
        chosen[tag] = best[1]
        print(f"  {tag:>34}{designs[tag].shape[1]:>7}{best[1]:>11g}"
              f"{best[0]:>10.5f}")
    for tag in designs:
        preds[tag] = roll(tag, t1_, t2_, chosen[tag])
    rv_t = rvv[t1_:t2_]
    good = rv_t > 0
    for tag in designs:
        good &= np.isfinite(preds[tag]) & (preds[tag] > 0)
    rv_g = rv_t[good]
    print(f"\n  scored {int(good.sum()):,} bars common to every arm")
    print(f"\n  {'design':>34}{'cols':>7}{'TEST QLIKE':>13}{'TEST MSE':>13}")
    res = {}
    for tag in designs:
        pg = preds[tag][good]
        q = float(np.mean(qlike_bar(rv_g, pg)))
        res[tag] = {"cols": int(designs[tag].shape[1]),
                    "lambda": chosen[tag], "test_qlike": q,
                    "test_mse": float(np.mean((rv_g - pg) ** 2))}
        print(f"  {tag:>34}{designs[tag].shape[1]:>7}{q:>13.5f}"
              f"{res[tag]['test_mse']:>13.4e}")

    print("\n" + "=" * 78)
    print("DM: THE DUMB MODEL AGAINST EVERYTHING ELSE")
    print("=" * 78)
    base = qlike_bar(rv_g, preds["dumb: backbone + 41 amplitudes"][good])
    dms = {}
    for tag in designs:
        if tag.startswith("dumb"):
            continue
        r = dm_test(base, qlike_bar(rv_g, preds[tag][good]), h=1)
        verdict = ("dumb WINS" if r["mean_diff"] < 0 and r["p"] < 0.05 else
                   "dumb loses" if r["mean_diff"] > 0 and r["p"] < 0.05 else
                   "indistinguishable")
        print(f"  vs {tag:>32}: {r['mean_diff']:+.5f}  t {r['t']:+6.2f}  "
              f"p {r['p']:.3g}   {verdict}")
        dms[tag] = {"diff": r["mean_diff"], "t": r["t"], "p": r["p"],
                    "verdict": verdict}

    d = res["dumb: backbone + 41 amplitudes"]["test_qlike"]
    sh = res["shaped penalty on 492"]["test_qlike"]
    print(f"\n  dumb {d:.5f} with 53 columns and one ridge penalty")
    print(f"  shaped {sh:.5f} with 504 columns and a bespoke matrix penalty")
    print("  -> " + ("the elaborate version buys nothing the precomputed "
                     "amplitudes do not:\n     state it as a FEATURE, not an "
                     "estimator"
                     if d <= sh + 0.001 else
                     "the penalty geometry does something the precomputed "
                     "amplitudes miss"))
    out = {"arms": res, "dm_vs_dumb": dms,
           "dumb_qlike": d, "shaped_qlike": sh,
           "dumb_matches_shaped": bool(d <= sh + 0.001)}
    with open(os.path.join(OUT_DIR, "dumb_model.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote dumb_model.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
