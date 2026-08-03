"""Does directed flow -- timing -- carry anything the covariance does not?

Every method tried on this panel so far is blind to direction. Ridge, the
amplitude designs, principal components, the operator SVD and the supervised
channel compression all read a CONTEMPORANEOUS second moment: they know which
series move together, never which moves first. The lead-lag machinery in
run_geometry_local.py was written for exactly this and has never been invoked
by an experiment; it also freezes its estimate on the first 24,000 rows rather
than refitting, so it could not have been used in a walk-forward as written.

Estimate the lagged cross-covariance on each window's TRAINING rows only,

    C = B(t)' B(t+tau) / n,      S = (C + C')/2,      A = (C - C')/2,

and split it. A common trend is symmetric -- it leads nothing -- so the
antisymmetric part A is the directed component. Features are projections of the
current cross-section onto the leading singular axes of A.

THE CONTROL IS THE POINT. Projections onto S are included at a MATCHED feature
count. Without that, any gain from A would be indistinguishable from "six more
linear combinations of the same 42 series", which is what four earlier
compressions in this project turned out to be. If A helps and S does not, the
information is in the timing. If both help equally, it is not.

Arms, all with lambda selected on a validation region strictly preceding
evaluation, and every axis refit per window on training rows:

    backbone              12
    backbone + A(3)       12 + 6     directed
    backbone + S(6)       12 + 6     matched control
    ridge-504            504        the incumbent
    ridge-504 + A(3)     504 + 6    does direction add to the full design

Run at tau = 1 and tau = 8 -- one bar and roughly half a session -- because a
lead that exists at one horizon need not exist at the other.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                       # noqa: E402
from fastwalk import RollingGram, block, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("LL_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
TAUS = [int(v) for v in os.environ.get("LL_TAUS", "1,8").split(",")]
NPAIR = 3          # 3 singular pairs of A -> 6 features; S matched at 6


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
    C_, nh = len(labels), len(har)
    F = np.hstack([np.asarray(X[:, har], np.float64),
                   np.asarray(X[:, lad], np.float64)])
    del X
    P = F.shape[1]
    # bar-level series: each channel's ma_1 (rung 1), plus har_ma_1
    ma1 = [nh + i * 12 for i in range(C_)] + [0]
    ma1 = np.array(ma1, int)
    print(f"cache {CACHE}: {n:,} rows, {P} cols, {len(ma1)} bar-level series "
          f"({time.time()-t0:.0f}s)", flush=True)

    def axes(s, tau):
        """Lead/lag axes from the TRAINING window only. Returns (L_A, L_S)."""
        B = F[s - W:s, ma1]
        mu, sd = B.mean(0), B.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Bs = (B - mu) / sd
        m = W - tau
        Cm = Bs[:-tau].T @ Bs[tau:] / m
        Aa = (Cm - Cm.T) / 2.0
        Ss = (Cm + Cm.T) / 2.0
        U, _, Vt = np.linalg.svd(Aa)
        ev, Q = np.linalg.eigh(Ss)
        Q = Q[:, np.argsort(np.abs(ev))[::-1]]
        # columns of the 504-space that realise these projections
        LA = np.zeros((P, 2 * NPAIR)); LS = np.zeros((P, 2 * NPAIR))
        for p in range(NPAIR):
            LA[ma1, 2 * p] = U[:, p] / sd
            LA[ma1, 2 * p + 1] = Vt[p] / sd
        for p in range(2 * NPAIR):
            LS[ma1, p] = Q[:, p] / sd
        return LA, LS

    def transforms(s, tau):
        I_h = np.zeros((P, nh)); I_h[:nh, :nh] = np.eye(nh)
        LA, LS = axes(s, tau)
        out = {"backbone": (I_h, 0)}
        out["bb + A(3)"] = (np.hstack([I_h, LA]), 2 * NPAIR)
        out["bb + S(6)"] = (np.hstack([I_h, LS]), 2 * NPAIR)
        I_f = np.eye(P)
        out["ridge-504"] = (I_f, P - nh)
        out["ridge-504 + A(3)"] = (np.hstack([I_f, LA]), P - nh + 2 * NPAIR)
        return out

    ARMS = ["backbone", "bb + A(3)", "bb + S(6)", "ridge-504", "ridge-504 + A(3)"]

    def go(tau, lo, hi, lam_by=None):
        val = lam_by is None
        rg = RollingGram(F, y, W)
        acc = {a: ({L: 0.0 for L in LAMS}, {L: 0 for L in LAMS}) for a in ARMS}
        preds = {a: np.full(hi - lo, np.nan) for a in ARMS}
        s0 = lo - ((lo - W) % CAD)
        for s in range(s0, hi, CAD):
            e = min(s + CAD, hi)
            if e <= lo: continue
            G, c = rg.advance(s)
            TT = transforms(s, tau)
            a, z = max(s, lo), e
            for arm in ARMS:
                T, npen = TT[arm]
                M, v = block(G, c, T)
                lams = LAMS if (val and npen) else ([0.0] if not npen else [lam_by[arm]])
                Dt = np.hstack([np.ones((W, 1)), F[s - W:s] @ T])
                De = np.hstack([np.ones((z - a, 1)), F[a:z] @ T])
                for L, cf in solve_path(M, v, max(npen, 1), lams).items():
                    rr = y[s - W:s] - Dt @ cf
                    pv = ((De @ cf) ** 2 + float(rr @ rr) / W) * b[a:z]
                    if val:
                        rt = rvv[a:z]; mk = (rt > 0) & np.isfinite(pv) & (pv > 0)
                        if mk.sum():
                            q = rt[mk] / pv[mk]
                            acc[arm][0][L] += float(np.sum(q - np.log(q) - 1.0))
                            acc[arm][1][L] += int(mk.sum())
                    else:
                        preds[arm][a - lo:z - lo] = pv
        if val:
            return {a: min((acc[a][0][L] / acc[a][1][L], L)
                           for L in LAMS if acc[a][1][L])[1] for a in ARMS}
        return preds

    out = {"cache": CACHE, "taus": TAUS, "npair": NPAIR, "by_tau": {}}
    for tau in TAUS:
        print(f"\n{'=' * 70}\ntau = {tau}\n{'=' * 70}", flush=True)
        lam = go(tau, int(n * 0.55), int(n * 0.65))
        preds = go(tau, W, n, lam_by=lam)
        rv_w = rvv[W:]; good = rv_w > 0
        for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
        idx = np.flatnonzero(good)
        L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
        Q = {k: float(L[k].mean()) for k in L}
        print(f"  scored {len(idx):,} bars\n")
        print(f"  {'arm':>20}{'QLIKE':>11}{'vs its base':>14}{'t':>8}")
        rec = {"qlike": Q, "dm": {}}
        for arm in ARMS:
            base = "backbone" if arm.startswith("bb") else (
                "ridge-504" if arm.startswith("ridge-504 +") else None)
            if base is None:
                print(f"  {arm:>20}{Q[arm]:>11.5f}{'--':>14}{'--':>8}"); continue
            r = dm_test(L[arm], L[base], h=1)
            rec["dm"][arm] = {"vs": base, "diff": float(r["mean_diff"]),
                              "t": float(r["t"]), "p": float(r["p"])}
            print(f"  {arm:>20}{Q[arm]:>11.5f}{r['mean_diff']:>+14.5f}{r['t']:>8.2f}")
        dA = Q["bb + A(3)"] - Q["backbone"]; dS = Q["bb + S(6)"] - Q["backbone"]
        print(f"\n  directed A: {dA:+.5f}   symmetric S (matched): {dS:+.5f}")
        print("  -> " + ("timing carries information the covariance does not"
                         if dA < dS - 0.0002 else
                         "no evidence that direction adds over co-movement"))
        rec["delta_A"] = float(dA); rec["delta_S"] = float(dS)
        out["by_tau"][str(tau)] = rec

    with open(os.path.join(OUT_DIR, "leadlag.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote leadlag.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
