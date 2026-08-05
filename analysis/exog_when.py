"""When does the exogenous block help, and when does it hurt?

Two of this session's results sit against the paper's attribution finding that
every one of eight exogenous buckets improves on the backbone.

  RETRIEVAL. Adding the 492 exogenous ladder coordinates to a distance is a
  cost: matching on the residual path alone beats the composed view by 0.00976
  (t = -11.3), and the best compression the theory can construct -- the 41
  fixed-shape amplitudes that are the dual of the winning penalty -- still does
  not rescue it (path + amplitudes 0.16667 against path alone 0.16347).

  This one is NOT a contradiction. A fitted map learns a weight per column and
  can drive a useless one to zero; a Euclidean metric weights every coordinate
  by its scale and has no such mechanism. Fitting and measuring similarity are
  different operations and only the first tolerates breadth.

  REGRESSION. This one IS a contradiction, and it is the interesting question.
  Under the rolling 24,000-bar protocol, plain ridge on the exogenous block
  tests at 0.17906 against 0.15242 for the backbone alone -- the block is a net
  cost inside a fitted model too, which the paper's battery says it should not
  be. Only shape-directed shrinkage turns it positive (0.14878).

Three candidate explanations, and they are separable:

  1. FEATURES. This uses the raw 41 x 12 ladder. The paper's buckets are
     engineered, mechanism-targeted features, and the paper itself reports them
     as "far more efficient per column than raw exogenous series". Raw ladders
     may simply be the wrong parameterisation.
  2. TUNING. One lambda is chosen for all 492 columns on a validation region
     that -- as measured -- ranks the arms differently from the test region.
     The battery tunes each arm causally.
  3. PERIOD. The rolling run scores bars 157,907 onward, the last 35% of the
     panel. The battery walks the whole thing. If the exogenous block helps
     early and hurts late, both results are true and neither is wrong.

Period is the cheapest to settle and the most consequential if it holds, so it
goes first. This walks the same rolling protocol across the WHOLE post-warmup
panel rather than the last third, and reports the backbone-versus-exogenous
contrast by calendar block. If the sign flips across periods, the paper's
attribution claim needs a stability qualification and this session's negative
result needs a date range -- and the disagreement dissolves into a fact about
when exogenous information is worth carrying.
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
LAM_RIDGE = 1e4          # as selected on validation in pooled_operator_rolling
LAM_SHAPE = 1e8
NBLOCK = 8


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

    Umax = int(rungs[-1])
    L = np.zeros((Umax, R))
    for k, r in enumerate(rungs):
        L[:int(r), k] = 1.0 / r
    M = L.T @ L
    M = M / max(np.trace(M) / R, 1e-12)
    g = rungs ** (-1.0)
    g = g / np.linalg.norm(g)
    Mg = M @ g
    Q = M - np.outer(Mg, Mg) / float(g @ M @ g)

    P_ridge = np.zeros((1 + p, 1 + p))
    P_shape = np.zeros((1 + p, 1 + p))
    for c in range(C):
        s = 1 + nh + c * R
        P_ridge[s:s + R, s:s + R] = np.eye(R)
        P_shape[s:s + R, s:s + R] = Q

    lo = W
    print(f"panel {n:,}   walking [{lo:,},{n:,}) with a {W:,}-bar window, "
          f"cadence {CAD}")
    arms = ["backbone only", "ridge + exog", "shape-directed + exog"]
    preds = {a: np.full(n - lo, np.nan) for a in arms}
    for s in range(lo, n, CAD):
        e = min(s + CAD, n)
        A = np.hstack([np.ones((W, 1)), F[s - W:s]])
        G = A.T @ A
        c = A.T @ y[s - W:s]
        yt = y[s - W:s]
        Ae = np.hstack([np.ones((e - s, 1)), F[s:e]])
        Gb = G[:1 + nh, :1 + nh]
        cfb = np.linalg.solve(Gb + 1e-8 * np.eye(1 + nh), c[:1 + nh])
        rr = yt - A[:, :1 + nh] @ cfb
        preds["backbone only"][s - lo:e - lo] = (
            (Ae[:, :1 + nh] @ cfb) ** 2 + float(rr @ rr) / W) * b[s:e]
        for tag, P, lam in (("ridge + exog", P_ridge, LAM_RIDGE),
                            ("shape-directed + exog", P_shape, LAM_SHAPE)):
            cf = np.linalg.solve(G + lam * P + 1e-8 * np.eye(1 + p), c)
            rr = yt - A @ cf
            preds[tag][s - lo:e - lo] = (
                (Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
        del A, Ae, G
    print(f"  walked ({time.time()-t0:.0f}s)")

    rv_w = rvv[lo:]
    good = rv_w > 0
    for a in arms:
        good &= np.isfinite(preds[a]) & (preds[a] > 0)
    idx = np.flatnonzero(good)
    print(f"  scored {len(idx):,} bars")

    print("\n" + "=" * 78)
    print("WHOLE-PANEL WALK (not just the last third)")
    print("=" * 78)
    tot = {}
    for a in arms:
        q = float(np.mean(qlike_bar(rv_w[idx], preds[a][idx])))
        tot[a] = q
        print(f"  {a:>26}{q:>13.5f}")
    for a in arms[1:]:
        r = dm_test(qlike_bar(rv_w[idx], preds[a][idx]),
                    qlike_bar(rv_w[idx], preds["backbone only"][idx]), h=1)
        print(f"  {a:>26} vs backbone: {r['mean_diff']:+.5f}  t {r['t']:+.2f}"
              f"  p {r['p']:.3g}")

    print("\n" + "=" * 78)
    print(f"BY PERIOD ({NBLOCK} equal blocks of the walk)")
    print("=" * 78)
    print(f"  {'block':>6}{'bars':>9}{'backbone':>11}{'ridge+exog':>12}"
          f"{'shape+exog':>12}{'ridge-bb':>11}{'shape-bb':>11}")
    blocks = np.array_split(idx, NBLOCK)
    rows = []
    for bi, blk in enumerate(blocks):
        qb = float(np.mean(qlike_bar(rv_w[blk], preds["backbone only"][blk])))
        qr = float(np.mean(qlike_bar(rv_w[blk], preds["ridge + exog"][blk])))
        qs = float(np.mean(qlike_bar(rv_w[blk],
                                     preds["shape-directed + exog"][blk])))
        print(f"  {bi+1:>6}{len(blk):>9,}{qb:>11.5f}{qr:>12.5f}{qs:>12.5f}"
              f"{qr - qb:>+11.5f}{qs - qb:>+11.5f}")
        rows.append({"block": bi + 1, "n": len(blk), "backbone": qb,
                     "ridge_exog": qr, "shape_exog": qs,
                     "ridge_minus_bb": qr - qb, "shape_minus_bb": qs - qb})
    nr = sum(1 for r_ in rows if r_["ridge_minus_bb"] < 0)
    ns = sum(1 for r_ in rows if r_["shape_minus_bb"] < 0)
    print(f"\n  plain ridge beats the backbone in {nr}/{NBLOCK} blocks")
    print(f"  shape-directed beats the backbone in {ns}/{NBLOCK} blocks")
    flips = nr not in (0, NBLOCK)
    print("  -> " + ("THE SIGN FLIPS ACROSS PERIODS: the exogenous block helps "
                     "in some eras and\n     hurts in others, so the battery's "
                     "positive result and this session's\n     negative one "
                     "can both be right on their own windows"
                     if flips else
                     "the sign is STABLE across periods: the disagreement with "
                     "the battery is\n     about features or tuning, not about "
                     "when the panel was measured"))
    out = {"n_scored": int(len(idx)), "whole_panel": tot, "blocks": rows,
           "ridge_wins_blocks": nr, "shape_wins_blocks": ns,
           "sign_flips": bool(flips)}
    with open(os.path.join(OUT_DIR, "exog_when.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote exog_when.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
