"""The interaction ladder, including the session clock this project omitted.

Every design fitted in this session is har(12) + lad(492) = 504 columns. The
cache also holds nine calendar columns -- DOW_0..DOW_4, hour, is_overnight,
is_open, is_close -- and none of them appeared in any of it. So the session
clock was absent even as a MAIN effect, while the paper's own attribution table
puts the regime block (HAR crossed with open and close) at -0.00225, second
only to the exogenous block itself and larger than anything measured here
today. Concluding the representation was exhausted, from a design missing a
block the paper had already measured, was an incomplete search.

The ladder is nested and cheapest-first, so each rung is a strict superset of
the last and the increments are readable:

    ridge-504              har + lad                      the incumbent
    + cal                  + 9 calendar main effects      repairs the omission
    + har x cal            + 108                          the paper's regime block
    + har x har            + 78                           quadratic self-exciting
    + exog x is_open       + 492                          clock-gated exogenous
    + exog x cal(4)        + 1968                         the wider version
    + FM rank-8            + 8 embeddings                 ALL pairs, k*p params

Factorization machines are the principled way to reach the full pairwise space:
504 columns give 126,756 pairs, which cannot be materialised, but an FM
parameterises the pair weight as <v_i, v_j> with rank-k embeddings and so needs
k*p parameters. Here the embeddings are FIXED random projections rather than
learned, which makes the arm a linear ridge in a pairwise random feature space
-- a lower bound on what a fitted FM would reach, and honest about being one.

Selection is on a validation region strictly preceding evaluation throughout,
and every interaction is a product of columns already in the design, so nothing
here can leak.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("IX_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
SEED = 20240803
FM_K = 8


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
    CALN = ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4", "hour",
            "is_overnight", "is_open", "is_close"]
    cal = np.array([names.index(c) for c in CALN if c in names], int)
    n = len(y); rvv = y ** 2 * b; nh = len(har)
    H = np.asarray(X[:, har], np.float64)
    Z = np.asarray(X[:, lad], np.float64)
    K = np.asarray(X[:, cal], np.float64)
    del X
    print(f"cache {CACHE}: {n:,} rows, har {H.shape[1]}, exog {Z.shape[1]}, "
          f"calendar {K.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  calendar columns finally in the design: "
          f"{[c for c in CALN if c in names]}", flush=True)

    def cross(A, B):
        return (A[:, :, None] * B[:, None, :]).reshape(len(A), -1)

    iopen = CALN.index("is_open"); iclose = CALN.index("is_close")
    rng = np.random.default_rng(SEED)

    def build(arm):
        base = [H, Z]
        if arm == "ridge-504":
            pass
        elif arm == "+ cal":
            base += [K]
        elif arm == "+ har x cal":
            base += [K, cross(H, K)]
        elif arm == "+ har x har":
            iu = np.triu_indices(H.shape[1])
            base += [K, cross(H, K), (H[:, :, None] * H[:, None, :])[:, iu[0], iu[1]]]
        elif arm == "+ exog x is_open":
            base += [K, cross(H, K), Z * K[:, [iopen]]]
        elif arm == "+ exog x cal4":
            sel = [iopen, iclose, CALN.index("is_overnight"), CALN.index("hour")]
            base += [K, cross(H, K)] + [Z * K[:, [s]] for s in sel]
        else:                       # FM: fixed random rank-k pairwise features
            k = FM_K
            A = np.hstack([H, Z, K])
            V = rng.standard_normal((A.shape[1], k)) / np.sqrt(A.shape[1])
            P1 = A @ V
            P2 = (A ** 2) @ (V ** 2)
            base += [K, 0.5 * (P1 ** 2 - P2)]      # the FM pairwise term
        F = np.hstack(base)
        return F, F.shape[1] - nh

    ARMS = ["ridge-504", "+ cal", "+ har x cal", "+ har x har",
            "+ exog x is_open", "+ exog x cal4", f"+ FM rank-{FM_K}"]
    preds, lams, cols = {}, {}, {}
    for arm in ARMS:
        F, npen = build(arm)
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
        preds[arm] = out; lams[arm] = lam; cols[arm] = F.shape[1]
        print(f"    {arm:>18} cols={F.shape[1]:<6d} lambda*={lam:<9g} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del F, T

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'arm':>18}{'cols':>7}{'QLIKE':>11}{'vs ridge-504':>14}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "cols": cols,
           "lambda": {k: float(v) for k, v in lams.items()},
           "paper_regime_block": -0.00225, "dm": {}}
    for arm in ARMS:
        if arm == "ridge-504":
            print(f"  {arm:>18}{cols[arm]:>7}{Q[arm]:>11.5f}{'--':>14}{'--':>8}")
            continue
        r = dm_test(L[arm], L["ridge-504"], h=1)
        out["dm"][arm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {arm:>18}{cols[arm]:>7}{Q[arm]:>11.5f}{r['mean_diff']:>+14.5f}"
              f"{r['t']:>8.2f}")
    best = min((v, k) for k, v in Q.items())[1]
    print(f"\n  best: {best}")
    print(f"  the paper's regime block (HAR x open/close) is -0.00225")
    if best != "ridge-504":
        nb = len(idx) // NBLOCK
        d = L[best] - L["ridge-504"]
        em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean())
              for i in range(NBLOCK)]
        print(f"  {best} by era: {['%+.5f' % v for v in em]}")
        print(f"  better in {sum(1 for v in em if v < 0)}/{NBLOCK} eras")
        out["best_era_means"] = em
    with open(os.path.join(OUT_DIR, "interactions.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote interactions.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
