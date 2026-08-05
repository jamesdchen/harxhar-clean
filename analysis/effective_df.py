"""What the dual says about fitting: effective degrees of freedom, per era.

The primal-dual identity (verified to 3.1e-15 in analysis/primal_dual.py) says a
quadratic penalty IS a kernel, and the kernel view supplies the honest measure
of how much freedom a fit is using:

    df(lambda) = tr( A (A'A + lambda P)^-1 A' ) = tr( (G + lambda P)^-1 G )

This is not the column count. It is how the penalty interacts with the DATA
SPECTRUM, so two arms with the same 505 columns can be spending wildly
different amounts. Three things follow that bear directly on results already in
hand.

  1. WHAT "492 COLUMNS" ACTUALLY COSTS. Plain ridge on the exogenous block and
     shape-directed shrinkage on the same block have identical dimension and
     very different priors: the shape penalty is singular with a 41-dimensional
     null space (one amplitude per channel), so it should spend roughly
     backbone + 41, while ridge spends whatever the spectrum allows.

  2. WHETHER df EXPLAINS THE 2022-2024 COLLAPSE. Across eight eras, plain ridge
     beats the backbone in seven and fails in the last by +0.03321, five times
     worse than shape-directed shrinkage contains it. If that failure is an
     over-fitting failure, ridge's df should be both higher and less stable in
     that era. If df is flat while the loss explodes, the failure is a
     distribution shift and no amount of regularisation accounting predicts it
     -- a different diagnosis with a different fix.

  3. WHERE RIDGE SPENDS IT. The Gram spectrum carries 96.2% of X-variance in
     PC1 and 0.02% of the explained target, with the signal in PCs 6-20. Ridge
     shrinks direction i by d_i / (d_i + lambda), so it preserves exactly the
     high-variance directions that carry no signal. Reporting df split by
     eigenvalue block makes that concrete rather than rhetorical: how much of
     each arm's spent freedom sits in the top few directions versus in the
     ones that actually predict.

Rolling 24,000-bar windows at the production cadence, so the numbers line up
bar-for-bar with analysis/exog_when.py.
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

D = os.path.join(ROOT, "results", "b2_mmap")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
W = 24000
CAD = 4000               # coarser than the forecast walk; df moves slowly
LAM_RIDGE = 1e4
LAM_SHAPE = 1e8
NBLOCK = 8


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
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
    print(f"panel {n:,}   design {1+p} columns "
          f"(1 + {nh} backbone + {C*R} exogenous)")
    print(f"  the shape penalty's null space is {C} dimensional "
          f"(one amplitude per channel)")
    print(f"  so its df should sit near {1 + nh + C} if the null space is "
          f"what it spends")

    starts = np.arange(W, n, CAD)
    rows = []
    for s in starts:
        A = np.hstack([np.ones((W, 1)), F[s - W:s]])
        G = A.T @ A
        rec = {"start": int(s)}
        rec["backbone"] = float(np.trace(np.linalg.solve(
            G[:1 + nh, :1 + nh] + 1e-8 * np.eye(1 + nh), G[:1 + nh, :1 + nh])))
        for tag, P, lam in (("ridge", P_ridge, LAM_RIDGE),
                            ("shape", P_shape, LAM_SHAPE)):
            rec[tag] = float(np.trace(np.linalg.solve(
                G + lam * P + 1e-8 * np.eye(1 + p), G)))
        # where the freedom sits, by Gram eigen-block
        ev, V = np.linalg.eigh(G[1:, 1:])
        order = np.argsort(ev)[::-1]
        ev, V = ev[order], V[:, order]
        for tag, P, lam in (("ridge", P_ridge, LAM_RIDGE),
                            ("shape", P_shape, LAM_SHAPE)):
            Pv = V.T @ P[1:, 1:] @ V
            shrink = ev / (ev + lam * np.diag(Pv) + 1e-12)
            rec[f"{tag}_df_top5"] = float(shrink[:5].sum())
            rec[f"{tag}_df_6_20"] = float(shrink[5:20].sum())
            rec[f"{tag}_df_rest"] = float(shrink[20:].sum())
        rows.append(rec)
        del A, G
    print(f"  {len(rows)} windows ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 78)
    print("EFFECTIVE DEGREES OF FREEDOM BY ERA")
    print("=" * 78)
    print(f"  {'block':>6}{'backbone':>11}{'ridge':>10}{'shape':>10}"
          f"{'ridge sd':>11}{'shape sd':>11}")
    blocks = np.array_split(np.arange(len(rows)), NBLOCK)
    summ = []
    for bi, blk in enumerate(blocks):
        rr = [rows[i] for i in blk]
        bb = np.mean([r["backbone"] for r in rr])
        dr = np.array([r["ridge"] for r in rr])
        ds = np.array([r["shape"] for r in rr])
        print(f"  {bi+1:>6}{bb:>11.1f}{dr.mean():>10.1f}{ds.mean():>10.1f}"
              f"{dr.std():>11.2f}{ds.std():>11.2f}")
        summ.append({"block": bi + 1, "backbone": float(bb),
                     "ridge": float(dr.mean()), "shape": float(ds.mean()),
                     "ridge_sd": float(dr.std()), "shape_sd": float(ds.std())})

    print("\n" + "=" * 78)
    print("WHERE THE FREEDOM SITS (Gram eigen-blocks, whole-walk mean)")
    print("=" * 78)
    print(f"  {'arm':>8}{'PC 1-5':>11}{'PC 6-20':>11}{'PC 21+':>11}")
    spend = {}
    for tag in ("ridge", "shape"):
        a = np.mean([r[f"{tag}_df_top5"] for r in rows])
        b_ = np.mean([r[f"{tag}_df_6_20"] for r in rows])
        c_ = np.mean([r[f"{tag}_df_rest"] for r in rows])
        print(f"  {tag:>8}{a:>11.2f}{b_:>11.2f}{c_:>11.2f}")
        spend[tag] = {"pc1_5": float(a), "pc6_20": float(b_),
                      "pc21plus": float(c_)}
    print("  (the Gram's PC1 carries 96.2% of the design variance and 0.02% "
          "of the\n   explained target; the signal sits in PCs 6-20)")

    last, rest = summ[-1], summ[:-1]
    dr_last, dr_rest = last["ridge"], np.mean([s["ridge"] for s in rest])
    print("\n" + "=" * 78)
    print("DOES df EXPLAIN THE 2022-2024 COLLAPSE?")
    print("=" * 78)
    print(f"  ridge df, blocks 1-7 mean {dr_rest:.1f}   block 8 {dr_last:.1f}"
          f"   ({dr_last - dr_rest:+.1f})")
    print(f"  shape df, blocks 1-7 mean "
          f"{np.mean([s['shape'] for s in rest]):.1f}   block 8 "
          f"{last['shape']:.1f}")
    moved = abs(dr_last - dr_rest) > 0.1 * dr_rest
    verdict = ("df RISES in the failing era, so the collapse is consistent "
               "with an\n  over-fitting failure that the effective-freedom "
               "accounting predicts."
               if moved and dr_last > dr_rest else
               "df is FLAT while the loss explodes, so the collapse is NOT an "
               "over-fitting\n  failure -- it is a distribution shift, and no "
               "regularisation accounting\n  anticipates it. Directed "
               "shrinkage contains the damage without having\n  predicted it.")
    print("  -> " + verdict)

    out = {"n_windows": len(rows), "by_block": summ, "spend": spend,
           "ridge_df_blocks_1_7": float(dr_rest),
           "ridge_df_block_8": float(dr_last),
           "verdict": verdict.replace("\n", " ")}
    with open(os.path.join(OUT_DIR, "effective_df.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote effective_df.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
