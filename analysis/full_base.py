"""Gate: reconcile the design to the cache, and find where 0.12788 lives.

Every arm in this session used 504 of the cache's 1,077 columns, dropping all
492 availability indicators, 48 occurrence indicators and 9 calendar columns.
The paper's ridge on all_features is 0.12788; this session's ridge-504 is
0.13146. Until that 0.00358 is accounted for, no comparison here means
anything, so this runs before anything else is re-fitted.

The arms add the omitted blocks back one at a time, so the gap is attributed
rather than merely closed:

    ridge-504        adj(492) + har(12)                this session's baseline
    + calendar       + 9                               the clock
    + active         + 48                              occurrence indicators
    + avail          + 492                             availability indicators
    full-1077        everything the cache holds        the reconciliation

If the full arm lands near 0.128 the gap was the missing columns and the
re-runs can proceed on that base. If it does not, something else differs --
the paper's battery is on a base-5 HAR ladder and a frozen panel, and this
cache is base-2 with a repaired fill, so a residual gap would be a footing
difference rather than a design one, and would need naming before any number
here is set beside a published one.

Penalty structure: the 12 HAR columns are unpenalised, everything else
penalised, matching the convention used throughout this session so the only
thing varying is which columns are present.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("FB_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 8)]
NBLOCK = 8
PAPER = {"ridge/all_features": 0.12788, "lgbm/all_features": 0.12604,
         "dense enet": 0.12530, "deployed enetreg2": 0.12314}


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
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m: chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan: chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)], int)
    avail = np.array([j for j, nm in enumerate(names) if "_avail_ma_" in nm], int)
    active = np.array([j for j, nm in enumerate(names) if "_active_ma_" in nm], int)
    used = set(lad) | set(har) | set(avail) | set(active)
    other = np.array([j for j in range(len(names)) if j not in used], int)
    nh = len(har)
    print(f"cache {CACHE}: {n:,} rows, {len(names)} columns")
    print(f"  har {len(har)}  adj {len(lad)}  avail {len(avail)}  "
          f"active {len(active)}  other {len(other)}")
    print(f"  other columns: {[names[j] for j in other]}")
    tot = len(har) + len(lad) + len(avail) + len(active) + len(other)
    print(f"  reconciliation: {tot} accounted of {len(names)} "
          f"{'OK' if tot == len(names) else 'MISMATCH'}", flush=True)

    H = np.asarray(X[:, har], np.float64)
    BLK = {"adj": np.asarray(X[:, lad], np.float64),
           "other": np.asarray(X[:, other], np.float64),
           "active": np.asarray(X[:, active], np.float64),
           "avail": np.asarray(X[:, avail], np.float64)}
    del X

    ARMS = [("ridge-504", ["adj"]),
            ("+ calendar", ["adj", "other"]),
            ("+ active", ["adj", "other", "active"]),
            ("+ avail (full-1077)", ["adj", "other", "active", "avail"])]

    v0, v1 = int(n * 0.55), int(n * 0.65)
    preds, lams, cols = {}, {}, {}
    for arm, keys in ARMS:
        F = np.hstack([H] + [BLK[k] for k in keys])
        npen = F.shape[1] - nh
        T = np.eye(F.shape[1])
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
        preds[arm] = out; lams[arm] = lam; cols[arm] = F.shape[1]
        print(f"    {arm:>22} cols={F.shape[1]:<5d} lambda*={lam:<9g} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del F, T

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'arm':>22}{'cols':>7}{'QLIKE':>11}{'vs ridge-504':>14}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "cols": cols,
           "lambda": {k: float(v) for k, v in lams.items()},
           "paper": PAPER, "dm": {}}
    for arm, _ in ARMS:
        if arm == "ridge-504":
            print(f"  {arm:>22}{cols[arm]:>7}{Q[arm]:>11.5f}{'--':>14}{'--':>8}")
            continue
        r = dm_test(L[arm], L["ridge-504"], h=1)
        out["dm"][arm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]),
                          "p": float(r["p"])}
        print(f"  {arm:>22}{cols[arm]:>7}{Q[arm]:>11.5f}{r['mean_diff']:>+14.5f}"
              f"{r['t']:>8.2f}")
    full = Q["+ avail (full-1077)"]
    print(f"\n  paper ridge/all_features 0.12788   full-1077 here {full:.5f}   "
          f"residual {full - 0.12788:+.5f}")
    if abs(full - 0.12788) < 0.001:
        print("  -> the gap was the missing columns; re-runs proceed on this base")
    else:
        print("  -> a residual remains. The paper's battery is base-5 HAR on a")
        print("     frozen panel; this cache is base-2 with a repaired fill, so")
        print("     the remainder is a FOOTING difference and must be named")
        print("     before any number here is set beside a published one.")
    nb = len(idx) // NBLOCK
    d = L["+ avail (full-1077)"] - L["ridge-504"]
    em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean())
          for i in range(NBLOCK)]
    print(f"  full-1077 by era: {['%+.5f' % v for v in em]}")
    print(f"  better in {sum(1 for v in em if v < 0)}/{NBLOCK} eras")
    out["full_era_means"] = em
    out["residual_vs_paper"] = float(full - 0.12788)
    with open(os.path.join(OUT_DIR, "full_base.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote full_base.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
