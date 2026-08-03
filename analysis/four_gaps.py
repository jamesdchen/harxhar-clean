"""Four gaps this project never tested, on the corrected panel.

PART 1  LOSS MISMATCH. Every arm here fits SQUARED loss on the transformed
target and is scored with QLIKE on raw variance. Duan smearing repairs the
scale but nothing fits the objective being scored, and QLIKE is exactly the
unit deviance of a Gamma GLM with a log link, so that objective is directly
available. The precedent is not speculative: choosing the SELECTION criterion
correctly was worth 0.00157 here, more than most modelling effects in the
paper, and this is the same class of error one level down.

PART 2  cumrv. The paper's attribution table credits one engineered feature --
within-session cumulative realized variance, gated to the session boundary --
with -0.00122, against -0.00449 for the entire 359-column exogenous block. It
has never been reconstructed on the corrected panel.

PART 3  ADDITIVITY. Target winsorization at 1/99 is worth 0.00304 and the
exogenous block 0.00425, both measured alone. Nobody has run them together, so
additivity is assumed rather than measured. A 2x2 answers it, scored against
the single fixed unclipped truth so the comparison is not the one that made the
original winsorization number 87% artifact.

PART 4  SPARSITY. Every penalty tried on this panel has been L2. The
dense-but-weak reading argues against L1 -- if the signal really is spread
thinly over 492 collinear columns, selecting a subset should hurt -- but that
is a prediction, not a result, and it has never been checked. Lasso and elastic
net are refit per window here; the codebase's recursive homotopy version is a
computational route to the same estimator, not a different one.

Selection is on a validation region strictly preceding evaluation throughout.
Parts 1 and 4 use a coarser cadence because they fit iteratively, and their
ridge reference is refit on the SAME cadence so each contrast is like-for-like
even though its level does not compare to the 1000-bar tables.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("FG_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
CAD_SLOW = 4000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3]
L1R = [0.1, 0.5, 1.0]          # 1.0 is pure lasso
NBLOCK = 8


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def fast_walk(F, y, b, rvv, n, npen, lo_v, hi_v):
    """Select lambda then walk, on the shared rolling-Gram path."""
    T = np.eye(F.shape[1])
    rg = RollingGram(F, y, W)
    num = {L: 0.0 for L in LAMS}; den = {L: 0 for L in LAMS}
    s0 = lo_v - ((lo_v - W) % CAD)
    for s in range(s0, hi_v, CAD):
        e = min(s + CAD, hi_v)
        if e <= lo_v: continue
        G, c = rg.advance(s); M, vv = block(G, c, T)
        a, z = max(s, lo_v), e
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
    return out, lam


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
    H = np.asarray(X[:, har], np.float64)
    Z = np.asarray(X[:, lad], np.float64)
    del X
    F0 = np.hstack([H, Z]); P0 = F0.shape[1]
    v0, v1 = int(n * 0.55), int(n * 0.65)
    RES = {}
    print(f"cache {CACHE}: {n:,} rows, {P0} cols ({time.time()-t0:.0f}s)", flush=True)

    base_pred, base_lam = fast_walk(F0, y, b, rvv, n, P0 - nh, v0, v1)
    print(f"  [ref] ridge-504 lambda*={base_lam:g} ({time.time()-t0:.0f}s)", flush=True)

    def score(preds_map, ref_key, title, truth=None):
        tr = (rvv if truth is None else truth)[W:]
        good = tr > 0
        for k in preds_map: good &= np.isfinite(preds_map[k]) & (preds_map[k] > 0)
        idx = np.flatnonzero(good)
        L = {k: qlike_bar(tr[idx], preds_map[k][idx]) for k in preds_map}
        Q = {k: float(L[k].mean()) for k in L}
        print(f"\n{'=' * 72}\n{title}   ({len(idx):,} bars)\n{'=' * 72}")
        print(f"  {'arm':>26}{'QLIKE':>11}{'vs ref':>12}{'t':>8}")
        rec = {"n": int(len(idx)), "qlike": Q, "dm": {}}
        for k in preds_map:
            if k == ref_key:
                print(f"  {k:>26}{Q[k]:>11.5f}{'--':>12}{'--':>8}"); continue
            r = dm_test(L[k], L[ref_key], h=1)
            rec["dm"][k] = {"diff": float(r["mean_diff"]), "t": float(r["t"]),
                            "p": float(r["p"])}
            print(f"  {k:>26}{Q[k]:>11.5f}{r['mean_diff']:>+12.5f}{r['t']:>8.2f}")
        return rec

    # ---------------- PART 2: cumrv ----------------
    from src.data.loading import load_raw_data
    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    sub = raw.iloc[len(raw) - n:].reset_index(drop=True)
    hour = sub["t"].dt.hour.to_numpy()
    day = sub["t"].dt.normalize().to_numpy()
    rvv_bar = sub["RV"].to_numpy(np.float64)
    sess = (hour >= 9) & (hour < 16)
    newday = np.r_[True, day[1:] != day[:-1]]
    grp = np.cumsum(newday)
    csum = np.zeros(n)
    acc = 0.0; last = -1
    for i in range(n):
        if grp[i] != last:
            acc = 0.0; last = grp[i]
        csum[i] = acc                      # causal: excludes the current bar
        if sess[i] and np.isfinite(rvv_bar[i]):
            acc += rvv_bar[i]
    cum = np.sqrt(np.maximum(csum, 0.0))
    cum = (cum - np.nanmean(cum)) / max(np.nanstd(cum), 1e-12)
    is_close = ((hour == 15) | (hour == 16)).astype(float)
    is_open = (hour == 9).astype(float)
    CUM = np.column_stack([cum, cum * is_close, cum * is_open])
    F2 = np.hstack([F0, CUM])
    p2, _ = fast_walk(F2, y, b, rvv, n, F2.shape[1] - nh, v0, v1)
    RES["part2_cumrv"] = score({"ridge-504": base_pred, "+ cumrv (3 cols)": p2},
                               "ridge-504", "PART 2  cumrv, the engineered feature")

    # ---------------- PART 4: sparsity ----------------
    from sklearn.linear_model import ElasticNet
    steps = list(range(W, n, CAD_SLOW))
    sp = {f"enet l1r={r}": np.full(n - W, np.nan) for r in L1R}
    sp["ridge (same cadence)"] = np.full(n - W, np.nan)
    nnz = {r: [] for r in L1R}
    for i, s in enumerate(steps):
        e = min(s + CAD_SLOW, n)
        Xtr, ytr = F0[s - W:s], y[s - W:s]
        A = np.hstack([np.ones((W, 1)), Xtr])
        Pm = np.zeros((1 + P0, 1 + P0)); Pm[1 + nh:, 1 + nh:] = np.eye(P0 - nh)
        cf = np.linalg.solve(A.T @ A + base_lam * Pm + 1e-8 * np.eye(1 + P0), A.T @ ytr)
        rr = ytr - A @ cf
        Ae = np.hstack([np.ones((e - s, 1)), F0[s:e]])
        sp["ridge (same cadence)"][s - W:e - W] = ((Ae @ cf) ** 2
                                                   + float(rr @ rr) / W) * b[s:e]
        for r in L1R:
            m = ElasticNet(alpha=1e-4, l1_ratio=r, max_iter=3000, tol=1e-4,
                           selection="random", random_state=0).fit(Xtr, ytr)
            rt = ytr - m.predict(Xtr)
            sp[f"enet l1r={r}"][s - W:e - W] = (m.predict(F0[s:e]) ** 2
                                                + float(rt @ rt) / W) * b[s:e]
            nnz[r].append(int((m.coef_ != 0).sum()))
        if i % 10 == 0:
            print(f"    sparse refit {i+1}/{len(steps)} ({time.time()-t0:.0f}s)",
                  flush=True)
    for r in L1R:
        print(f"    l1_ratio {r}: median non-zero coefficients "
              f"{int(np.median(nnz[r]))} of {P0}")
    RES["part4_sparse"] = score(sp, "ridge (same cadence)",
                                "PART 4  sparsity against L2")
    RES["part4_nnz"] = {str(r): int(np.median(nnz[r])) for r in L1R}

    with open(os.path.join(OUT_DIR, "four_gaps.json"), "w") as fh:
        json.dump(RES, fh, indent=1)
    print(f"\nwrote four_gaps.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
