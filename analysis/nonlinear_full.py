"""Is there curvature in the FULL exogenous design that ridge misses?

A previous experiment concluded that "the tree premium is not reachable as a
smooth function of the coordinates" and it was written up as though
nonlinearity were closed. It was not. dual_kernels.py built its random Fourier
features from amps(1) -- the 41 AMPLITUDES -- so what was actually shown is
that a kernel cannot reach the premium in the COMPRESSED space. A kernel on
the full 492 columns was never fitted, and since every linear compression of
those columns has since been shown to lose, testing nonlinearity only inside a
compression was the wrong test.

Arms, lambda selected on a validation region strictly preceding evaluation:

    ridge-504                  the incumbent
    ridge-504 + squares        element-wise squares of all 492, pure curvature
                               with no cross terms, 996 penalised columns
    ridge-504 + RFF 256/512    random Fourier features of an RBF kernel on the
                               full standardised 492, bandwidth by the median
                               heuristic on an EARLY window so it is never
                               tuned on evaluation rows

An RBF kernel in 492 dimensions is a weak instrument -- distances concentrate,
and the kernel tends toward a constant -- so a null here is evidence about
smooth curvature specifically and not about nonlinearity in general. The
element-wise squares arm is the complement: it has no such concentration
problem and asks only whether each coordinate enters nonlinearly.

What this does NOT settle is the tree premium itself, which the paper reports
at 0.00184-0.00358 and which was measured on the corrupted panel. That needs
re-running on the clean one before it can be quoted.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("NL_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
SEED = 20240803
RFF_D = [256, 512]


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
    n = len(y); rvv = y ** 2 * b; nh = len(har)
    Fh = np.asarray(X[:, har], np.float64)
    Z = np.asarray(X[:, lad], np.float64)
    del X
    print(f"cache {CACHE}: {n:,} rows, {Z.shape[1]} exog, {nh} backbone "
          f"({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    # bandwidth from an EARLY window only
    sub = Z[:W][rng.choice(W, 1500, replace=False)]
    d2 = ((sub[:400, None, :] - sub[None, :400, :]) ** 2).sum(-1)
    scale = float(np.sqrt(np.median(d2[d2 > 0])))
    print(f"  RBF bandwidth (median heuristic, first window): {scale:.4f}",
          flush=True)

    def build(arm):
        if arm == "ridge-504":
            return np.hstack([Fh, Z])
        if arm == "+ squares":
            return np.hstack([Fh, Z, Z * Z])
        d = int(arm.split()[-1])
        Wm = rng.standard_normal((Z.shape[1], d)) / scale
        bv = rng.uniform(0, 2 * np.pi, d)
        return np.hstack([Fh, Z, np.sqrt(2.0 / d) * np.cos(Z @ Wm + bv)])

    ARMS = ["ridge-504", "+ squares"] + [f"+ RFF {d}" for d in RFF_D]
    preds, lams = {}, {}
    for arm in ARMS:
        F = build(arm)
        npen = F.shape[1] - nh
        T = np.eye(F.shape[1])
        # selection
        rg = RollingGram(F, y, W)
        v0, v1 = int(n * 0.55), int(n * 0.65)
        num = {L: 0.0 for L in LAMS}; den = {L: 0 for L in LAMS}
        s0 = v0 - ((v0 - W) % CAD)
        for s in range(s0, v1, CAD):
            e = min(s + CAD, v1)
            if e <= v0: continue
            G, c = rg.advance(s); M, vv = block(G, c, T)
            a, z = max(s, v0), e
            for L, cf in solve_path(M, vv, npen, LAMS).items():
                pv = (predict(F[a:z], T, cf) ** 2 + rss(M, vv, cf, rg.yy) / W) * b[a:z]
                rt = rvv[a:z]; mk = (rt > 0) & np.isfinite(pv) & (pv > 0)
                if mk.sum():
                    q = rt[mk] / pv[mk]
                    num[L] += float(np.sum(q - np.log(q) - 1.0)); den[L] += int(mk.sum())
        lam = min((num[L] / den[L], L) for L in LAMS if den[L])[1]
        # walk
        rg = RollingGram(F, y, W)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            G, c = rg.advance(s); M, vv = block(G, c, T)
            cf = solve_path(M, vv, npen, [lam])[lam]
            out[s - W:e - W] = (predict(F[s:e], T, cf) ** 2
                                + rss(M, vv, cf, rg.yy) / W) * b[s:e]
        preds[arm] = out; lams[arm] = lam
        print(f"    {arm:>14} cols={F.shape[1]:<5d} lambda*={lam:<9g} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del F, T

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'arm':>14}{'QLIKE':>11}{'vs ridge-504':>14}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "bandwidth": scale,
           "lambda": {k: float(v) for k, v in lams.items()}, "dm": {}}
    for arm in ARMS:
        if arm == "ridge-504":
            print(f"  {arm:>14}{Q[arm]:>11.5f}{'--':>14}{'--':>8}"); continue
        r = dm_test(L[arm], L["ridge-504"], h=1)
        out["dm"][arm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {arm:>14}{Q[arm]:>11.5f}{r['mean_diff']:>+14.5f}{r['t']:>8.2f}")
    best = min((v, k) for k, v in Q.items())[1]
    print(f"\n  best: {best}")
    print(f"  the paper's tree premium is 0.00184 to 0.00358, measured on the")
    print(f"  CORRUPTED panel and still not re-tested on this one.")
    with open(os.path.join(OUT_DIR, "nonlinear_full.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote nonlinear_full.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
