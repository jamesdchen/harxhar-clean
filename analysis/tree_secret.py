"""What is a tree's secret? Test the monotone-invariance hypothesis directly.

The corrupted/clean control already established ONE mechanism: on the
uncorrected panel the tree beat ridge by 0.01746, and on the corrected panel it
LOSES by 0.00341, with the tree itself barely moving (0.13495 to 0.13571) while
ridge improved by 0.02011. The trees were winning by ignoring a feature
standing at 2,260 rolling interquartile ranges, not by modelling anything.

That leaves a sharper question. A tree's only view of a feature is its ORDER:
every split is x < c, so any monotone transformation of a feature is invisible
to it. A tree is fitting in rank space. Ridge fits in value space, where a
heavy-tailed feature carries leverage proportional to its magnitude.

If monotone-invariance is the whole story, then giving ridge RANK-TRANSFORMED
features should absorb whatever advantage remains, and the answer is that there
is no secret -- only robustness, available to a linear model for the price of
one preprocessing step. If it does not, what trees have is genuinely
interaction structure, and the paper's "additive-plus-pairwise anchored to the
session clock" becomes the finding rather than a footnote.

Arms, all on the clean panel, all sharing the walk, cadence, rows and smearing:

    ridge-504            value space, the incumbent
    ridge-504 rank       rank-Gauss per column, estimated on TRAINING rows only
                         and applied to evaluation rows by interpolation, so no
                         evaluation row influences its own transform
    tree                 LightGBM, fixed hyperparameters
    tree rank            LightGBM on the same rank features -- a CONTROL that
                         must show no change, since splits are already
                         rank-based. If it moves, the rank transform is leaking
                         or doing something other than what is claimed, and the
                         ridge-rank number cannot be read.

The tree-rank control is the reason this is a test rather than a demonstration.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("TS_CACHE", "b2_mmap_warm")
W = 24000
CAD = int(os.environ.get("TS_CAD", "4000"))
LAM = 1e4
NBLOCK = 8
NQ = 256          # quantile grid for the rank map
# tree-on-ranks is a CONTROL asserting an invariance, not an estimate.
# Re-confirming it on all 55 windows doubles the dominant cost to learn the
# same thing repeatedly, so it runs as a spot check.
CTRL_EVERY = 7
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, max_bin=127,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=4)


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def rank_gauss(train, test, nq=NQ):
    """Per-column rank-Gauss, knots from TRAINING rows only.

    A column is mapped to its training empirical CDF and then through the
    normal quantile function. Evaluation rows are placed by interpolating the
    training knots, so no evaluation row contributes to its own transform --
    which is the whole point, since a rank computed over train and test jointly
    would leak the test distribution.
    """
    from scipy.special import ndtri
    qs = np.linspace(0.0, 1.0, nq)
    # All columns at once: the per-column Python loop this replaces spent its
    # time in interpreter overhead rather than in the sort.
    K = np.quantile(train, qs, axis=0)
    np.maximum.accumulate(K, axis=0, out=K)
    lo, hi = 0.5 / len(train), 1.0 - 0.5 / len(train)
    dead = (K[-1] - K[0]) < 1e-12
    out_tr = np.empty_like(train); out_te = np.empty_like(test)
    for j in range(train.shape[1]):
        if dead[j]:
            out_tr[:, j] = 0.0; out_te[:, j] = 0.0
            continue
        kj = K[:, j]
        out_tr[:, j] = np.clip(np.interp(train[:, j], kj, qs), lo, hi)
        out_te[:, j] = np.clip(np.interp(test[:, j], kj, qs), lo, hi)
    return ndtri(out_tr), ndtri(out_te)


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
    print(f"cache {CACHE}: {n:,} rows, {pp} cols, cadence {CAD} -> "
          f"{len(steps)} refits ({time.time()-t0:.0f}s)", flush=True)

    P = np.zeros((1 + pp, 1 + pp)); P[1 + nh:, 1 + nh:] = np.eye(pp - nh)
    ARMS = ["ridge", "ridge rank", "tree", "tree rank"]
    pr = {a: np.full(n - W, np.nan) for a in ARMS}
    ctrl_rows = []

    def fit_ridge(Xtr, ytr, Xte, s, e):
        A = np.hstack([np.ones((W, 1)), Xtr])
        cf = np.linalg.solve(A.T @ A + LAM * P + 1e-8 * np.eye(1 + pp), A.T @ ytr)
        rr = ytr - A @ cf
        Ae = np.hstack([np.ones((e - s, 1)), Xte])
        return ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]

    def fit_tree(Xtr, ytr, Xte, s, e):
        m = lgb.LGBMRegressor(**PARAMS).fit(Xtr, ytr)
        rt = ytr - m.predict(Xtr)
        return (m.predict(Xte) ** 2 + float(rt @ rt) / W) * b[s:e]

    for i, s in enumerate(steps):
        e = min(s + CAD, n)
        Xtr, ytr, Xte = F[s - W:s], y[s - W:s], F[s:e]
        Rtr, Rte = rank_gauss(Xtr, Xte)
        pr["ridge"][s - W:e - W] = fit_ridge(Xtr, ytr, Xte, s, e)
        pr["ridge rank"][s - W:e - W] = fit_ridge(Rtr, ytr, Rte, s, e)
        pr["tree"][s - W:e - W] = fit_tree(Xtr, ytr, Xte, s, e)
        if i % CTRL_EVERY == 0:
            pr["tree rank"][s - W:e - W] = fit_tree(Rtr, ytr, Rte, s, e)
            ctrl_rows.append((s - W, e - W))
        if i % 5 == 0:
            print(f"    refit {i+1}/{len(steps)} ({time.time()-t0:.0f}s)", flush=True)

    rv_w = rvv[W:]; good = rv_w > 0
    for a in ("ridge", "ridge rank", "tree"):
        good &= np.isfinite(pr[a]) & (pr[a] > 0)
    idx = np.flatnonzero(good)
    FULL = ["ridge", "ridge rank", "tree"]
    L = {a: qlike_bar(rv_w[idx], pr[a][idx]) for a in FULL}
    Q = {a: float(L[a].mean()) for a in FULL}
    cmask = np.zeros(n - W, bool)
    for a_, z_ in ctrl_rows: cmask[a_:z_] = True
    cidx = np.flatnonzero(good & cmask & np.isfinite(pr["tree rank"])
                          & (pr["tree rank"] > 0))
    print(f"\n  scored {len(idx):,} bars, cadence {CAD}\n")
    print(f"  {'arm':>14}{'QLIKE':>11}{'vs ridge':>12}{'t':>8}")
    out = {"cache": CACHE, "cadence": CAD, "n": int(len(idx)), "qlike": Q, "dm": {}}
    for a in FULL:
        if a == "ridge":
            print(f"  {a:>14}{Q[a]:>11.5f}{'--':>12}{'--':>8}"); continue
        r = dm_test(L[a], L["ridge"], h=1)
        out["dm"][a] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {a:>14}{Q[a]:>11.5f}{r['mean_diff']:>+12.5f}{r['t']:>8.2f}")

    ctrl = float(qlike_bar(rv_w[cidx], pr["tree rank"][cidx]).mean()
                 - qlike_bar(rv_w[cidx], pr["tree"][cidx]).mean())
    print(f"\n  [control] tree on ranks minus tree on values, on "
          f"{len(cidx):,} spot-check bars: {ctrl:+.5f}")
    print("  splits are already rank-based, so this must be ~0; a large value "
          "means\n  the rank map is doing something other than reordering and "
          "nothing below reads")
    gap = Q["tree"] - Q["ridge"]
    closed = Q["ridge"] - Q["ridge rank"]
    print(f"\n  tree minus ridge      : {gap:+.5f}")
    print(f"  rank transform buys   : {closed:+.5f} for ridge")
    if gap > 0:
        print("  -> the tree does not lead on this panel, so there is no gap "
              "to close;\n     the rank number stands on its own as a "
              "preprocessing result")
    out["control_tree_rank_minus_tree"] = float(ctrl)
    out["ridge_gain_from_rank"] = float(closed)
    with open(os.path.join(OUT_DIR, "tree_secret.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote tree_secret.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
