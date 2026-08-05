"""Re-test the tree premium on the corrected panel.

The paper reports gradient-boosted trees beating linear arms by 0.00184 to
0.00358. That is larger than every structural effect measured in this project
-- larger than the whole exogenous block's advantage over the backbone
(0.00425 is the only comparable number) and an order of magnitude larger than
any basis, penalty or rank choice. It was measured on the CORRUPTED panel and
has never been re-run, which makes it the most load-bearing unvalidated number
left, and the cache has already reversed the sign of one headline result.

Two nonlinear routes have now been closed on the clean panel: an RBF kernel on
the full 492 columns gains 0.00001, and element-wise squares LOSE 0.00573. So
if the premium survives, whatever the trees exploit is neither smooth kernel
curvature nor per-coordinate curvature, which points at interactions and
thresholds -- consistent with the paper's own finding that trees saturate at
additive-plus-pairwise structure anchored to the session clock.

Protocol notes, since a tree comparison has more ways to go wrong than a
linear one:

  * both arms share the walk, the refit cadence, the row set and the Duan
    smearing, so the premium is a like-for-like differential
  * the cadence is coarser than the linear experiments (a GBM refit is orders
    of magnitude dearer than a Gram update); the RIDGE arm is refit on the
    SAME cadence rather than its usual one, so the comparison is matched even
    though neither level is comparable to the 1000-bar tables
  * hyperparameters are fixed a priori and modest, never tuned on evaluation
    rows. An untuned tree is a LOWER bound on the premium, which is the safe
    direction for a number this load-bearing

THE CONTROL THIS NEEDS. Running once on the corrected panel confounds three
changes at once -- the data fix, the absence of tuning, and a coarser cadence
than the paper used -- and a result that reverses could be any of them. Run
the IDENTICAL configuration on the corrupted panel (TP_CACHE=b2_mmap) and the
comparison becomes within-configuration: if this tree wins there and loses
here, the data fix is doing the work; if it loses on both, the configuration
is, and nothing has been learned about the premium.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("TP_CACHE", "b2_mmap_warm")
W = 24000
CAD = int(os.environ.get("TP_CAD", "4000"))
LAM = 1e4                       # the ridge arm's selected penalty on this cache
NBLOCK = 8
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=4)


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def main():
    import lightgbm as lgb
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
    F = np.hstack([np.asarray(X[:, har], np.float64),
                   np.asarray(X[:, lad], np.float64)])
    del X
    pp = F.shape[1]
    steps = list(range(W, n, CAD))
    print(f"cache {CACHE}: {n:,} rows, {pp} cols, cadence {CAD} "
          f"-> {len(steps)} refits ({time.time()-t0:.0f}s)", flush=True)

    P = np.zeros((1 + pp, 1 + pp)); P[1 + nh:, 1 + nh:] = np.eye(pp - nh)
    pr_r = np.full(n - W, np.nan); pr_t = np.full(n - W, np.nan)
    for i, s in enumerate(steps):
        e = min(s + CAD, n)
        Xtr, ytr = F[s - W:s], y[s - W:s]
        A = np.hstack([np.ones((W, 1)), Xtr])
        cf = np.linalg.solve(A.T @ A + LAM * P + 1e-8 * np.eye(1 + pp), A.T @ ytr)
        rr = ytr - A @ cf
        Ae = np.hstack([np.ones((e - s, 1)), F[s:e]])
        pr_r[s - W:e - W] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
        m = lgb.LGBMRegressor(**PARAMS).fit(Xtr, ytr)
        rt = ytr - m.predict(Xtr)
        pr_t[s - W:e - W] = (m.predict(F[s:e]) ** 2 + float(rt @ rt) / W) * b[s:e]
        if i % 5 == 0:
            print(f"    refit {i+1}/{len(steps)} ({time.time()-t0:.0f}s)", flush=True)
        del A, Ae

    rv_w = rvv[W:]
    good = (rv_w > 0) & np.isfinite(pr_r) & (pr_r > 0) & np.isfinite(pr_t) & (pr_t > 0)
    idx = np.flatnonzero(good)
    Lr = qlike_bar(rv_w[idx], pr_r[idx]); Lt = qlike_bar(rv_w[idx], pr_t[idx])
    r = dm_test(Lt, Lr, h=1)
    print(f"\n  scored {len(idx):,} bars, cadence {CAD}\n")
    print(f"  {'arm':>18}{'QLIKE':>11}")
    print(f"  {'ridge-504':>18}{Lr.mean():>11.5f}")
    print(f"  {'LightGBM-504':>18}{Lt.mean():>11.5f}")
    print(f"\n  premium (tree - ridge): {r['mean_diff']:+.5f}  t {r['t']:+.2f}  "
          f"p {r['p']:.3g}")
    print(f"  paper reports 0.00184 to 0.00358 on the CORRUPTED panel")
    nb = len(idx) // NBLOCK
    d = Lt - Lr
    em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean())
          for i in range(NBLOCK)]
    print(f"  by era: {['%+.5f' % v for v in em]}")
    print(f"  trees better in {sum(1 for v in em if v < 0)}/{NBLOCK} eras")
    out = {"cache": CACHE, "cadence": CAD, "n": int(len(idx)),
           "qlike_ridge": float(Lr.mean()), "qlike_tree": float(Lt.mean()),
           "premium": float(r["mean_diff"]), "t": float(r["t"]),
           "p": float(r["p"]), "era_means": em,
           "paper_premium_corrupted": [0.00184, 0.00358], "params": PARAMS}
    with open(os.path.join(OUT_DIR, "tree_premium.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote tree_premium.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
