"""Does refit cadence explain the residual to the paper's battery?

The battery refits ridge AT EVERY BAR -- methodology.tex, section on Ridge:
"Ridge is refit at every time step via the closed-form solution", and the trees
likewise "refit at every bar". Every arm in this session refits every 1,000
bars. That is not a footing subtlety, it is a different estimator: coefficients
held for a thousand bars track a regime shift three orders of magnitude more
slowly than coefficients re-solved each bar.

The reconciliation gate showed the 573 omitted columns were worth 0.00045 while
the residual to the paper's 0.12788 is 0.00313, so the columns are not the
explanation. Cadence is the obvious remaining candidate and has never been
varied here.

This sweeps it on the full 1,077-column design, holding everything else fixed:
same cache, same window, same penalty grid, same rows, same selection region.
If loss falls monotonically as the cadence tightens, the residual is a protocol
difference and the session's numbers are internally consistent but not
comparable to the battery. If it is flat, the residual is something else --
the base-5 ladder, or the frozen panel itself -- and that needs naming instead.

Per-bar refitting is not attempted directly: it needs 218,909 solves of a
1,078-square system. The codebase's rolling_ridge does this properly by
Sherman-Morrison rank-1 inverse updates; the sweep here brackets the answer at
a cost that fits the session, and the trend is what carries the inference.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("CD_CACHE", "b2_mmap_warm")
W = 24000
CADS = [int(v) for v in os.environ.get("CD_LIST", "1000,250,100,25").split(",")]
LAMS = [0.0] + [10.0 ** e for e in range(-2, 8)]
NBLOCK = 8


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy")); b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    n = len(y); rvv = y ** 2 * b
    har = np.array([j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)], int)
    rest = np.array([j for j in range(len(names)) if j not in set(har)], int)
    nh = len(har)
    F = np.hstack([np.asarray(X[:, har], np.float64),
                   np.asarray(X[:, rest], np.float64)])
    del X
    pp = F.shape[1]; npen = pp - nh
    T = np.eye(pp)
    print(f"cache {CACHE}: {n:,} rows, {pp} columns (full design)", flush=True)
    v0, v1 = int(n * 0.55), int(n * 0.65)

    preds, lams = {}, {}
    for CAD in CADS:
        rg = RollingGram(F, y, W)
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
        rg = RollingGram(F, y, W)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            G, c = rg.advance(s); M, vv = block(G, c, T)
            cf = solve_path(M, vv, npen, [lam])[lam]
            out[s - W:e - W] = (predict(F[s:e], T, cf) ** 2
                                + rss(M, vv, cf, rg.yy) / W) * b[s:e]
        preds[CAD] = out; lams[CAD] = lam
        print(f"    cadence {CAD:>5}  lambda*={lam:<9g} "
              f"({(n - W) // CAD:,} refits, {time.time()-t0:.0f}s)", flush=True)

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    ref = CADS[0]
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'cadence':>9}{'refits':>9}{'QLIKE':>11}{'vs 1000':>12}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": {str(k): v for k, v in Q.items()},
           "lambda": {str(k): float(v) for k, v in lams.items()},
           "paper_ridge_all_features": 0.12788, "dm": {}}
    for CAD in CADS:
        if CAD == ref:
            print(f"  {CAD:>9}{(n-W)//CAD:>9,}{Q[CAD]:>11.5f}{'--':>12}{'--':>8}")
            continue
        r = dm_test(L[CAD], L[ref], h=1)
        out["dm"][str(CAD)] = {"diff": float(r["mean_diff"]), "t": float(r["t"]),
                               "p": float(r["p"])}
        print(f"  {CAD:>9}{(n-W)//CAD:>9,}{Q[CAD]:>11.5f}{r['mean_diff']:>+12.5f}"
              f"{r['t']:>8.2f}")
    best = Q[CADS[-1]]
    print(f"\n  paper ridge/all_features 0.12788   tightest cadence here "
          f"{best:.5f}   residual {best - 0.12788:+.5f}")
    slope = Q[CADS[0]] - Q[CADS[-1]]
    print(f"  cadence {CADS[0]} -> {CADS[-1]} moves QLIKE by {slope:+.5f}")
    if slope > 0.001:
        print("  -> cadence is a large lever and the session's numbers, all at")
        print("     1000-bar cadence, are internally consistent but NOT")
        print("     comparable to a battery refit every bar")
    else:
        print("  -> cadence does not explain the residual; the remainder is the")
        print("     base-5 ladder or the frozen panel, and must be named as such")
    out["slope"] = float(slope)
    out["residual_at_tightest"] = float(best - 0.12788)
    with open(os.path.join(OUT_DIR, "cadence.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote cadence.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
