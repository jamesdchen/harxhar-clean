"""dual_features.py on the rolling-Gram path. Same arms, same arithmetic.

Every arm here turns out to be derivable from the window's Gram alone, which
is what makes the port worth doing rather than merely faster:

    ridge-504   the Gram itself
    pc-k        the principal directions of the exogenous block are the
                eigenvectors of its CENTRED cross-product, and centring is
                Z'Z - s's/W with s the column sums -- both already in the Gram.
                No per-window SVD of a 24,000 x 492 matrix. Centring the
                columns only shifts the intercept, so the transform may use the
                uncentred columns and still predict identically.
    spectral-a  the same eigenbasis, penalty (s_j/s_1)^-2a
    op-d5       the supervised operator needs the HAR residual cross-product
                Z'res = Z'y - (A_har'Z)' b_har, and Z'y, A_har'Z and the HAR
                normal equations are all blocks of the Gram. So the operator is
                available without ever touching a data row.

The slow path spent 287s on pc-5 against 52s for ridge-504 -- a strictly
smaller design costing five times more -- entirely on an SVD the Gram already
answers.

Validated against analysis/dual_features.py rather than assumed equivalent.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test              # noqa: E402
from fastwalk import RollingGram, block, solve_path   # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("DF_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
ARMS = [("ridge-504", "ridge", None), ("pc-5", "pc", 5), ("pc-20", "pc", 20),
        ("pc-50", "pc", 50), ("spectral-0.5", "spectral", 0.5),
        ("spectral-1.0", "spectral", 1.0), ("op-d5", "op", 5)]


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


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
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)], int)
    n = len(y); rvv = y ** 2 * b
    C, R, nh = len(labels), 12, len(har)
    F = np.hstack([np.asarray(X[:, har], np.float64),
                   np.asarray(X[:, lad], np.float64)])
    del X
    P = F.shape[1]                      # 504
    print(f"cache {CACHE}: {n:,} rows, {C}x{R} exog, {nh} backbone, "
          f"{P} cols ({time.time()-t0:.0f}s)", flush=True)

    def transforms(G, c):
        """Per-window {arm: (T, pen_diag)} -- all from the Gram, no data rows."""
        Ge = G[1 + nh:, 1 + nh:]                    # exog'exog
        se = G[0, 1 + nh:]                          # exog column sums
        Cc = Ge - np.outer(se, se) / W              # centred cross-product
        ev, V = np.linalg.eigh(Cc)
        o = np.argsort(ev)[::-1]
        ev, V = np.clip(ev[o], 0.0, None), V[:, o]
        sv = np.sqrt(ev)
        out = {}
        I_h = np.eye(nh)
        for nm, kind, prm in ARMS:
            if kind == "ridge":
                T = np.eye(P); pen = None
            elif kind == "pc":
                Vk = V[:, :prm]
                T = np.zeros((P, nh + prm)); T[:nh, :nh] = I_h
                T[nh:, nh:] = Vk
                pen = None
            elif kind == "spectral":
                T = np.zeros((P, nh + P - nh)); T[:nh, :nh] = I_h
                T[nh:, nh:] = V
                # The centred exogenous cross-product is rank-deficient, so
                # sv holds exact zeros and the ratio must be floored BEFORE
                # the power, not clipped after it -- (0/sv0)^-2a is inf, and
                # inf reaching a penalty matrix is a silent nan risk rather
                # than a loud one. A floored ratio gives those directions the
                # maximum penalty, which is the intended behaviour: a null
                # direction should be shrunk away, not fitted.
                ratio = np.clip(sv / max(sv[0], 1e-300), 1e-12, None)
                pen = np.clip(ratio ** (-2.0 * prm), 0.0, 1e12)
            else:                                    # supervised operator
                Gh = G[:1 + nh, :1 + nh]
                bh = np.linalg.solve(Gh + 1e-8 * np.eye(1 + nh), c[:1 + nh])
                # Z'res = Z'y - (A_har' Z)' b_har
                Zres = c[1 + nh:] - G[:1 + nh, 1 + nh:].T @ bh
                M = (Zres / W).reshape(C, R)
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                B = np.column_stack([np.outer(U[:, k], Vt[k]).ravel()
                                     for k in range(prm)])
                T = np.zeros((P, nh + prm)); T[:nh, :nh] = I_h
                T[nh:, nh:] = B
                pen = None
            out[nm] = (T, pen)
        return out

    def go(lo, hi, lams_by_arm=None):
        val = lams_by_arm is None
        rg = RollingGram(F, y, W)
        acc = {nm: ({L: 0.0 for L in LAMS}, {L: 0 for L in LAMS}) for nm, _, _ in ARMS}
        preds = {nm: np.full(hi - lo, np.nan) for nm, _, _ in ARMS}
        s0 = lo - ((lo - W) % CAD)
        for s in range(s0, hi, CAD):
            e = min(s + CAD, hi)
            if e <= lo: continue
            G, c = rg.advance(s)
            TT = transforms(G, c)
            a, z = max(s, lo), e
            Fe = F[a:z]
            for nm, _, _ in ARMS:
                T, pen = TT[nm]
                M, v = block(G, c, T)
                npen = T.shape[1] - nh
                lams = LAMS if val else [lams_by_arm[nm]]
                sols = solve_path(M, v, npen, lams, pen_diag=pen)
                De = np.hstack([np.ones((z - a, 1)), Fe @ T])
                Dt = np.hstack([np.ones((W, 1)), F[s - W:s] @ T])
                for L, cf in sols.items():
                    rr = y[s - W:s] - Dt @ cf
                    pv = ((De @ cf) ** 2 + float(rr @ rr) / W) * b[a:z]
                    if val:
                        rt = rvv[a:z]; m = (rt > 0) & np.isfinite(pv) & (pv > 0)
                        if m.sum():
                            q = rt[m] / pv[m]
                            acc[nm][0][L] += float(np.sum(q - np.log(q) - 1.0))
                            acc[nm][1][L] += int(m.sum())
                    else:
                        preds[nm][a - lo:z - lo] = pv
        if val:
            return {nm: min((acc[nm][0][L] / acc[nm][1][L], L)
                            for L in LAMS if acc[nm][1][L])[1] for nm, _, _ in ARMS}
        return preds

    v0, v1 = int(n * 0.55), int(n * 0.65)
    lam = go(v0, v1)
    print("  lambda*: " + "  ".join(f"{k}={v:g}" for k, v in lam.items()), flush=True)
    print(f"  selection done ({time.time()-t0:.0f}s)", flush=True)
    preds = go(W, n, lams_by_arm=lam)
    print(f"  walk done ({time.time()-t0:.0f}s)", flush=True)

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'arm':>14}{'QLIKE':>11}{'vs ridge-504':>14}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q,
           "lambda": {k: float(v) for k, v in lam.items()}, "dm": {}}
    for nm, _, _ in ARMS:
        if nm == "ridge-504":
            print(f"  {nm:>14}{Q[nm]:>11.5f}{'--':>14}{'--':>8}"); continue
        r = dm_test(L[nm], L["ridge-504"], h=1)
        out["dm"][nm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {nm:>14}{Q[nm]:>11.5f}{r['mean_diff']:>+14.5f}{r['t']:>8.2f}")
    nb = len(idx) // NBLOCK
    best = min((v, k) for k, v in Q.items())[1]
    print(f"\n  best arm: {best}")
    if best != "ridge-504":
        d = L[best] - L["ridge-504"]
        em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean()) for i in range(NBLOCK)]
        print(f"  era means vs ridge-504: {['%+.5f' % v for v in em]}")
        print(f"  better in {sum(1 for v in em if v < 0)}/{NBLOCK} eras")
        out["best_era_means"] = em
    with open(os.path.join(OUT_DIR, "dual_features_fast.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote dual_features_fast.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
