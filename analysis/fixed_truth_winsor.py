"""Target winsorization, ablated against ONE fixed truth.

The paper reports that clipping the target to rolling 5th/95th percentiles
costs about 0.024 QLIKE -- 0.2648 at 5/95, 0.2449 at 1/99, 0.2411 unclipped.
That is six times the entire exogenous block's advantage over the backbone and
roughly eighty times any structural design choice measured in this project, so
it is the largest lever in the paper and it has never been run under the real
protocol.

IT IS ALSO NOT YET A VALID COMPARISON. The evaluation builds its truth as
true_raw = y^2 * b from the SAME y the model was trained on, and y is the
winsorized target. Changing the clip therefore changes what the model is
scored against, not merely what it is fitted to, so the three numbers are
measured against three different truths. That is the "numbers quoted across
incompatible footings" failure already in the retraction ledger, attached to
the largest number in the paper. It also explains the counterintuitive sign:
unclipping lets spikes back into the truth, which ought to make QLIKE worse
because extremes are harder to predict, and it reports better.

This separates the two roles the clip plays:

    TRAIN on  y(5/95), y(1/99), y(none)
    SCORE every arm against ONE truth -- the UNCLIPPED y(none)^2 * b

so the only thing varying is what the estimator learned from, and any
difference is a fitting effect rather than a change of question.

Two facts make this cheap and make it clean. The baseline b is produced by the
diurnal adjustment, which runs BEFORE winsorization, so b is identical across
levels -- verified here rather than assumed. And the features are untouched by
the target clip, so the existing cache is reused and nothing is rebuilt.

The 5/95 arm must reproduce the cached target bit-for-bit. If it does not, the
reconstruction is wrong and no other number here means anything, so that check
gates the run.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402
import src.features.transforms.target as TGT                     # noqa: E402
from src.data.loading import load_raw_data                       # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("FT_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
LEVELS = [("5/95", 0.05, 0.95), ("1/99", 0.01, 0.99), ("none", None, None)]


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def build_target(df, lo, hi):
    """Target and baseline at a given clip, everything else held fixed."""
    o_lo, o_hi, o_w = TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q, TGT.rolling_winsorize
    try:
        if lo is None:
            TGT.rolling_winsorize = lambda s, **k: s          # no clip at all
        else:
            TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q = lo, hi
        y, b = TGT.robust_transform(df, "RV", use_transform=True, use_diurnal=True,
                                    winsor_window=240, is_target=True)
    finally:
        TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q = o_lo, o_hi
        TGT.rolling_winsorize = o_w
    return y.to_numpy(np.float64), b.to_numpy(np.float64)


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y_cache = np.load(os.path.join(D, "y.npy"))
    b_cache = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    n = len(y_cache)
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m: chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan: chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)], int)
    nh = len(har)
    F = np.hstack([np.asarray(X[:, har], np.float64),
                   np.asarray(X[:, lad], np.float64)])
    del X
    print(f"cache {CACHE}: {n:,} rows, {F.shape[1]} cols ({time.time()-t0:.0f}s)",
          flush=True)

    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    off = len(raw) - n
    print(f"raw panel {len(raw):,}, cache {n:,}, offset {off}", flush=True)

    Y, B = {}, {}
    for tag, lo, hi in LEVELS:
        yy, bb = build_target(raw, lo, hi)
        Y[tag], B[tag] = yy[off:off + n], bb[off:off + n]
    for tag, _, _ in LEVELS:
        share = float(np.mean(Y[tag] != Y["none"]))
        print(f"  target {tag:>5}: y sd {np.nanstd(Y[tag]):.5f}  "
              f"share differing from unclipped {share:.4f}", flush=True)

    # ---- gate 1: the 5/95 arm must reproduce the cached target
    ok = np.isfinite(Y["5/95"]) & np.isfinite(y_cache)
    e_y = float(np.max(np.abs(Y["5/95"][ok] - y_cache[ok])))
    e_b = float(np.max(np.abs(B["5/95"][ok] - b_cache[ok])))
    print(f"\n[gate] reconstruction vs cache: max|dy| {e_y:.3e}  max|db| {e_b:.3e}")
    if e_y > 1e-9 or e_b > 1e-9:
        print("[gate] FAIL: reconstruction does not match the cache; "
              "no number below would be interpretable")
        sys.exit(1)
    print("[gate] PASS")

    # ---- gate 2: the baseline must be identical across clips
    db = max(float(np.nanmax(np.abs(B[t] - B["5/95"]))) for t, _, _ in LEVELS)
    print(f"[gate] baseline invariance across clips: max|db| {db:.3e}")
    if db > 1e-9:
        print("[gate] FAIL: b depends on the clip, so the fixed truth is not "
              "well defined")
        sys.exit(1)
    print("[gate] PASS -- b is produced before winsorization, as expected")

    b = B["5/95"]
    truth = Y["none"] ** 2 * b            # THE fixed truth, unclipped
    print(f"\nfixed truth = unclipped y^2 * b   (mean {np.nanmean(truth):.3e})")

    pp = F.shape[1]; npen = pp - nh
    T = np.eye(pp)
    preds, lams = {}, {}
    for tag, _, _ in LEVELS:
        yt = np.nan_to_num(Y[tag], nan=0.0)
        rg = RollingGram(F, yt, W)
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
                rt = truth[a:z]
                mk = (rt > 0) & np.isfinite(pv) & (pv > 0) & np.isfinite(rt)
                if mk.sum():
                    q = rt[mk] / pv[mk]
                    num[L] += float(np.sum(q - np.log(q) - 1.0)); den[L] += int(mk.sum())
        lam = min((num[L] / den[L], L) for L in LAMS if den[L])[1]
        rg = RollingGram(F, yt, W)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            G, c = rg.advance(s); M, vv = block(G, c, T)
            cf = solve_path(M, vv, npen, [lam])[lam]
            out[s - W:e - W] = (predict(F[s:e], T, cf) ** 2
                                + rss(M, vv, cf, rg.yy) / W) * b[s:e]
        preds[tag] = out; lams[tag] = lam
        print(f"    trained on {tag:>5}  lambda*={lam:<9g} ({time.time()-t0:.0f}s)",
              flush=True)

    tr = truth[W:]
    good = (tr > 0) & np.isfinite(tr)
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(tr[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars, ALL against the unclipped truth\n")
    print(f"  {'trained on':>12}{'QLIKE':>11}{'vs 5/95':>12}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q,
           "lambda": {k: float(v) for k, v in lams.items()},
           "paper_claim_mixed_truth": {"5/95": 0.2648, "1/99": 0.2449, "none": 0.2411},
           "dm": {}}
    for tag, _, _ in LEVELS:
        if tag == "5/95":
            print(f"  {tag:>12}{Q[tag]:>11.5f}{'--':>12}{'--':>8}"); continue
        r = dm_test(L[tag], L["5/95"], h=1)
        out["dm"][tag] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {tag:>12}{Q[tag]:>11.5f}{r['mean_diff']:>+12.5f}{r['t']:>8.2f}")
    print(f"\n  the paper reports 0.2648 / 0.2449 / 0.2411 against THREE truths;")
    print(f"  above is one truth, so the difference is a fitting effect only.")
    with open(os.path.join(OUT_DIR, "fixed_truth_winsor.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote fixed_truth_winsor.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
