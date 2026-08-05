"""Shock BREADTH: how many channels moved, not how much each one moved.

Every design fitted on this panel asks the same question of the exogenous
block -- how large is each channel -- and answers it with a linear map. Ridge
contains all such maps, which is why eight structural restrictions of it have
now tied or lost. The question never asked is how MANY channels moved at once.

That is not pedantry about framing; it is outside the span. A count of
channels exceeding a threshold, a cross-sectional maximum, a rank
concentration -- none is a linear function of the channel levels, and none is
reachable by per-coordinate squares either.

WHY THE SQUARES ARM FAILING IS NOT EVIDENCE AGAINST THIS. Cross-sectional
variance is mean_c(z^2) - mean_c(z)^2, so the sum-of-squares part WAS available
to that arm. But it entered as 492 separate columns and lost 0.00573, which is
a variance cost rather than an absent signal. Aggregated across channels the
same quantity is 12 columns, one per rung -- a 41-fold reduction in dimension
carrying the same information, which is precisely the restriction that can win
where the unrestricted version loses.

THE CONTROL. A cross-sectional MEAN is linear and therefore already inside
ridge's span, so the `tilt` arm must show no gain. If it does, the harness is
wrong and nothing else here can be believed. It is included for that reason
rather than because anyone expects it to help. (The extreme-value machinery in
run_geometry_local.py reached for this idea and then used exactly that mean,
so the trap is not hypothetical.)

Statistics, computed per rung across the 41 channels, on columns that are
already in rolling-IQR units and hence comparable:

    tilt      mean_c(z)              LINEAR -- the control
    disp      std_c(z)               dispersion
    breadth   mean_c(|z| > 2)        how many are shocked
    maxabs    max_c |z|              the largest single move
    skew      standardised third moment across channels
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("BR_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
THRESH = 2.0


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
    n = len(y); rvv = y ** 2 * b; nh = len(har); C = len(labels)
    Fh = np.asarray(X[:, har], np.float64)
    Z = np.asarray(X[:, lad], np.float64)
    del X
    Zc = Z.reshape(n, C, 12)
    print(f"cache {CACHE}: {n:,} rows, {C} channels x 12 rungs ({time.time()-t0:.0f}s)",
          flush=True)

    mu = Zc.mean(1)                                   # (n, 12)  LINEAR control
    dev = Zc - mu[:, None, :]
    disp = np.sqrt((dev ** 2).mean(1))
    breadth = (np.abs(Zc) > THRESH).mean(1)
    maxabs = np.abs(Zc).max(1)
    sd = np.where(disp > 1e-12, disp, 1.0)
    skew = ((dev / sd[:, None, :]) ** 3).mean(1)
    del dev, Zc
    STAT = {"tilt": mu, "disp": disp, "breadth": breadth,
            "maxabs": maxabs, "skew": skew}
    for k, v in STAT.items():
        print(f"    {k:>8}: mean {v.mean():8.4f}  sd {v.std():8.4f}")
    print(f"    breadth: mean fraction of channels beyond {THRESH} IQRs "
          f"= {breadth.mean():.4f}", flush=True)

    ARMS = {"ridge-504": [],
            "+ tilt (LINEAR ctrl)": ["tilt"],
            "+ disp": ["disp"],
            "+ breadth": ["breadth"],
            "+ maxabs": ["maxabs"],
            "+ all nonlinear": ["disp", "breadth", "maxabs", "skew"]}
    preds, lams = {}, {}
    for arm, keys in ARMS.items():
        F = np.hstack([Fh, Z] + [STAT[k] for k in keys])
        npen = F.shape[1] - nh
        T = np.eye(F.shape[1])
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
        rg = RollingGram(F, y, W)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            G, c = rg.advance(s); M, vv = block(G, c, T)
            cf = solve_path(M, vv, npen, [lam])[lam]
            out[s - W:e - W] = (predict(F[s:e], T, cf) ** 2
                                + rss(M, vv, cf, rg.yy) / W) * b[s:e]
        preds[arm] = out; lams[arm] = lam
        print(f"    {arm:>22} cols={F.shape[1]:<5d} lambda*={lam:<9g} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del F, T

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'arm':>22}{'QLIKE':>11}{'vs ridge-504':>14}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "threshold": THRESH,
           "lambda": {k: float(v) for k, v in lams.items()}, "dm": {}}
    nb = len(idx) // NBLOCK
    for arm in ARMS:
        if arm == "ridge-504":
            print(f"  {arm:>22}{Q[arm]:>11.5f}{'--':>14}{'--':>8}"); continue
        r = dm_test(L[arm], L["ridge-504"], h=1)
        out["dm"][arm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {arm:>22}{Q[arm]:>11.5f}{r['mean_diff']:>+14.5f}{r['t']:>8.2f}")
    ctrl = Q["+ tilt (LINEAR ctrl)"] - Q["ridge-504"]
    print(f"\n  linear control moved {ctrl:+.5f} -- it must be ~0, since a "
          f"cross-sectional\n  mean is already inside ridge's span")
    best = min((v, k) for k, v in Q.items())[1]
    print(f"  best: {best}")
    if best != "ridge-504":
        d = L[best] - L["ridge-504"]
        em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean())
              for i in range(NBLOCK)]
        print(f"  {best} by era: {['%+.5f' % v for v in em]}")
        print(f"  better in {sum(1 for v in em if v < 0)}/{NBLOCK} eras")
        out["best_era_means"] = em
    with open(os.path.join(OUT_DIR, "breadth.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote breadth.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
