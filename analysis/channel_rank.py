"""Is the CHANNEL axis low rank, once the lag axis is already compressed?

The lag axis is: rank 3 of 12 reaches 0.13117 in 135 columns against 0.13146
for all 504, a tie. That is unsurprising -- twelve rungs are twelve EWMA
smoothings of one series, redundant by construction.

The channel axis has resisted every attempt so far, but every attempt has been
either unsupervised (principal components, which improve monotonically toward
plain ridge as directions are added: +0.00416, +0.00272, +0.00115 at k = 5, 20,
50) or joint (rank-1 on the whole 41 x 12 operator, the worst arm ever fitted
here at 0.30711). Compressing the joint space mixes the axes, so it destroys
channel identity while re-compressing an axis that was already compressible.

This tries the form nobody has: SUPERVISED compression of the channel axis
alone, with the lag axis held at its known-good rank 3. Per window, the
residual covariance M = A'res reshapes to 41 x 3; its top-k left singular
vectors are channel directions; features are k x 3. At k = 41 the design is
exactly the 135-column arm, so the sweep nests and k = 41 is a correctness
check rather than a separate result.

Everything is derived from the window Gram -- A'res = A'y - (A_har'A)' b_har --
so no data row is touched twice.

The honest prior: 41 heterogeneous data sources (an implied-volatility index,
order-flow imbalance, effective spread, social-media sentiment, share volume)
have no reason to lie on a low-dimensional manifold, and the principal-
component sweep already says signal keeps arriving as directions are added. A
supervised objective is the one thing that could change that, because it
targets covariance with the target rather than variance of the design.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                       # noqa: E402
from fastwalk import RollingGram, block, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("CR_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
KS = [1, 2, 3, 5, 10, 20, 41]
NSHAPE = 3


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def gram_schmidt(vs):
    out = []
    for v in vs:
        v = np.asarray(v, float).copy()
        for u in out:
            v -= (v @ u) * u
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            out.append(v / nv)
    return out


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy")); b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m: chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan: chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    rungs = np.array([r for r, _ in chan[labels[0]]], float)
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)], int)
    n = len(y); rvv = y ** 2 * b
    C, nh = len(labels), len(har)
    SH = gram_schmidt([rungs ** -1.0, rungs ** -0.5, rungs ** -2.0])[:NSHAPE]
    Fl = np.asarray(X[:, lad], np.float64).reshape(n, C, 12)
    # A[:, c*NSHAPE + j]  -- channel-major so the reshape to (C, NSHAPE) is free
    A = np.concatenate([(Fl @ SH[j])[:, :, None] for j in range(NSHAPE)],
                       axis=2).reshape(n, C * NSHAPE)
    del Fl
    F = np.hstack([np.asarray(X[:, har], np.float64), A])
    del X, A
    P = F.shape[1]
    print(f"cache {CACHE}: {n:,} rows, {C} channels x {NSHAPE} shapes, "
          f"{nh} backbone, {P} cols ({time.time()-t0:.0f}s)", flush=True)

    def transforms(G, c):
        Gh = G[:1 + nh, :1 + nh]
        bh = np.linalg.solve(Gh + 1e-8 * np.eye(1 + nh), c[:1 + nh])
        Ares = c[1 + nh:] - G[:1 + nh, 1 + nh:].T @ bh      # A'res
        M = (Ares / W).reshape(C, NSHAPE)
        U, _, _ = np.linalg.svd(M, full_matrices=False)      # channel directions
        out = {}
        I_h = np.eye(nh)
        for k in KS:
            kk = min(k, U.shape[1]) if k < C else C
            Uk = U[:, :kk] if kk < C else np.eye(C)
            # channel projection acting blockwise on the NSHAPE lag components
            B = np.zeros((C * NSHAPE, Uk.shape[1] * NSHAPE))
            for j in range(NSHAPE):
                B[j::NSHAPE, j::NSHAPE] = Uk
            T = np.zeros((P, nh + B.shape[1])); T[:nh, :nh] = I_h
            T[nh:, nh:] = B
            out[f"k={k}"] = T
        return out

    def go(lo, hi, lam_by=None):
        val = lam_by is None
        rg = RollingGram(F, y, W)
        arms = [f"k={k}" for k in KS]
        acc = {a: ({L: 0.0 for L in LAMS}, {L: 0 for L in LAMS}) for a in arms}
        preds = {a: np.full(hi - lo, np.nan) for a in arms}
        s0 = lo - ((lo - W) % CAD)
        for s in range(s0, hi, CAD):
            e = min(s + CAD, hi)
            if e <= lo: continue
            G, c = rg.advance(s)
            TT = transforms(G, c)
            a, z = max(s, lo), e
            for arm in arms:
                T = TT[arm]
                M, v = block(G, c, T)
                npen = T.shape[1] - nh
                lams = LAMS if val else [lam_by[arm]]
                Dt = np.hstack([np.ones((W, 1)), F[s - W:s] @ T])
                De = np.hstack([np.ones((z - a, 1)), F[a:z] @ T])
                for L, cf in solve_path(M, v, npen, lams).items():
                    rr = y[s - W:s] - Dt @ cf
                    pv = ((De @ cf) ** 2 + float(rr @ rr) / W) * b[a:z]
                    if val:
                        rt = rvv[a:z]; m = (rt > 0) & np.isfinite(pv) & (pv > 0)
                        if m.sum():
                            q = rt[m] / pv[m]
                            acc[arm][0][L] += float(np.sum(q - np.log(q) - 1.0))
                            acc[arm][1][L] += int(m.sum())
                    else:
                        preds[arm][a - lo:z - lo] = pv
        if val:
            return {a: min((acc[a][0][L] / acc[a][1][L], L)
                           for L in LAMS if acc[a][1][L])[1] for a in arms}
        return preds

    lam = go(int(n * 0.55), int(n * 0.65))
    print(f"  selection done ({time.time()-t0:.0f}s)", flush=True)
    preds = go(W, n, lam_by=lam)
    print(f"  walk done ({time.time()-t0:.0f}s)", flush=True)

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    ref = f"k={C}"
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'channels':>10}{'cols':>7}{'QLIKE':>11}{'vs full':>12}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "ks": KS,
           "lambda": {k: float(v) for k, v in lam.items()}, "dm": {}}
    for k in KS:
        a = f"k={k}"
        cols = nh + min(k, C) * NSHAPE
        if a == ref:
            print(f"  {k:>10}{cols:>7}{Q[a]:>11.5f}{'--':>12}{'--':>8}"); continue
        r = dm_test(L[a], L[ref], h=1)
        out["dm"][a] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {k:>10}{cols:>7}{Q[a]:>11.5f}{r['mean_diff']:>+12.5f}{r['t']:>8.2f}")
    with open(os.path.join(OUT_DIR, "channel_rank.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote channel_rank.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
