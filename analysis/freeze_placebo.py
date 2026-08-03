"""Was the frozen rank-3 win a property of FREEZING, or of THAT freeze?

The published arm estimates the basis once on rows [0, 24000) -- 2005-2007 --
and applies it through 2024, beating the twelve-rung ladder by 0.00080. Three
prior tests say it is not a better estimate of a stable subspace:

  - basis_tracking: every principled rank-3 estimator LOSES to r=12 by +0.00015
    to +0.00018, flat across a 32x range of memory halflife. Rank-3 as a
    dimension costs; only that one draw pays.
  - basis_tracking: the expanding-window arm (h=inf) sees strictly MORE data
    than the freeze at every boundary past the first, and forecasts 0.00098
    WORSE. Under no drift, more data cannot lose to less.
  - manifold_drift_test: at zero training overlap the freeze's directions 2-3
    sit 61.9 degrees apart with no lag structure (p = 0.84 for direction 1),
    so there is no rotation for the extra data to be stale about.

All three are indirect. This is the direct test, and it is a placebo in the
strict sense: repeat the EXACT published procedure on windows the researcher
did not pick. Freeze on rows [k*W, (k+1)*W) for every k the sample allows, and
score each arm causally on the rows after its own window.

  If only k = 0 wins, the win is that draw. The other freezes are the null
  distribution and the published number is a draw from it.

  If most k win, freezing per se buys something -- staleness beats
  re-estimation -- and the result survives in a weaker but real form.

Two readouts. MATCHED: each arm against r=12 on that arm's own valid rows, which
uses all available data but compares differentials over different regions.
COMMON: every arm on the rows after the LAST freeze window, which is a smaller
sample but strictly like-for-like. They should agree; if they do not, the
matched one is confounded by era and the common one is authoritative.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import dm_test                                    # noqa: E402
from ladder_refit import (CACHE, NBLOCK, RANK, RUNGS, W,          # noqa: E402
                          build_target, load_raw_data, qlike_bar, walk)
from basis_tracking import seed_basis                             # noqa: E402

KIND = os.environ.get("FP_KIND", "sup")
OUT = os.path.join(ROOT, "writeup", "stats", "freeze_placebo.json")


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    Xc = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    n = len(np.load(os.path.join(D, "y.npy")))
    CALN = ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4", "hour",
            "is_overnight", "is_open", "is_close"]
    cal = np.asarray(Xc[:, [names.index(c) for c in CALN if c in names]],
                     np.float64)
    del Xc
    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    off = len(raw) - n
    y99, b99 = build_target(raw, 0.01, 0.99)
    y595, _ = build_target(raw, 0.05, 0.95)
    y, b = y99[off:off + n], b99[off:off + n]
    truth = y595[off:off + n] ** 2 * b
    s = pd.Series(y ** 2)
    H = np.column_stack([
        np.sqrt(np.maximum(
            s.rolling(L, min_periods=L).mean().shift(1).to_numpy(), 0.0))
        for L in RUNGS])

    # every freeze window the sample supports, leaving room to score
    starts = [k * W for k in range(n // W) if (k + 1) * W + W <= n]
    lastend = starts[-1] + W
    print(f"cache {CACHE}: {n:,} rows, kind={KIND}")
    print(f"  {len(starts)} freeze windows of {W:,}; "
          f"common region = rows {lastend:,}..{n:,} "
          f"({n - lastend:,} bars)", flush=True)

    # reference: the full twelve-rung ladder, scored everywhere
    F0 = np.column_stack([H, cal])
    ref, _ = walk(np.where(np.isfinite(F0), F0, 0.0), y, b, W, W, n)
    ref[~np.isfinite(F0).all(1)[W:]] = np.nan
    print(f"    {'base-2 (r=12)':>22} ({time.time()-t0:.0f}s)", flush=True)

    preds = {}
    for k, a0 in enumerate(starts):
        a1 = a0 + W
        ok0 = np.isfinite(H[a0:a1]).all(1) & np.isfinite(y[a0:a1])
        Ht, yt = H[a0:a1][ok0], y[a0:a1][ok0]
        C = np.cov(Ht, rowvar=False)
        xy = (Ht - Ht.mean(0)).T @ (yt - yt.mean()) / len(yt)
        Gm = seed_basis(C, xy, RANK, KIND)
        lo = max(W, a1)
        F = np.column_stack([H @ Gm, cal])
        ok = np.isfinite(F).all(1)
        seg, _ = walk(np.where(ok[:, None], F, 0.0), y, b, W, lo, n)
        seg[~ok[lo:n]] = np.nan
        p = np.full(n - W, np.nan)
        p[lo - W:] = seg
        preds[k] = p
        print(f"    freeze k={k} rows [{a0:,},{a1:,})  "
              f"scored from {lo:,} ({time.time()-t0:.0f}s)", flush=True)

    tw = truth[W:]
    out = {"cache": CACHE, "kind": KIND, "W": W, "rank": RANK,
           "starts": starts, "matched": {}, "common": {}}

    print(f"\n  MATCHED -- each freeze vs r=12 on its own valid rows\n")
    print(f"  {'k':>3}{'window':>20}{'bars':>10}{'QLIKE':>10}"
          f"{'d vs r=12':>12}{'t':>8}{'eras':>7}")
    wins = 0
    for k, a0 in enumerate(starts):
        m = (tw > 0) & np.isfinite(preds[k]) & (preds[k] > 0) \
            & np.isfinite(ref) & (ref > 0)
        i = np.flatnonzero(m)
        lk = qlike_bar(tw[i], preds[k][i])
        lr = qlike_bar(tw[i], ref[i])
        d = dm_test(lk, lr, h=1)
        e = np.array_split(lk - lr, NBLOCK)
        nb = int(sum(1 for x in e if x.mean() < 0))
        wins += int(d["mean_diff"] < 0)
        print(f"  {k:>3}{f'[{a0//1000}k,{(a0+W)//1000}k)':>20}{len(i):>10,}"
              f"{lk.mean():>10.5f}{d['mean_diff']:>+12.5f}{d['t']:>+8.2f}"
              f"{nb:>4}/{NBLOCK}")
        out["matched"][str(k)] = {
            "start": a0, "n": int(len(i)), "qlike": float(lk.mean()),
            "diff": d["mean_diff"], "t": d["t"], "eras_better": nb}
    print(f"\n  {wins}/{len(starts)} freeze windows beat r=12")
    out["matched_wins"] = wins

    print(f"\n  COMMON -- all freezes on rows {lastend:,}..{n:,}\n")
    m = (tw > 0) & np.isfinite(ref) & (ref > 0)
    m[:lastend - W] = False
    for k in preds:
        m &= np.isfinite(preds[k]) & (preds[k] > 0)
    i = np.flatnonzero(m)
    lr = qlike_bar(tw[i], ref[i])
    print(f"  {'k':>3}{'window':>20}{'QLIKE':>10}{'d vs r=12':>12}{'t':>8}")
    rows = []
    for k, a0 in enumerate(starts):
        lk = qlike_bar(tw[i], preds[k][i])
        d = dm_test(lk, lr, h=1)
        rows.append((k, a0, float(lk.mean()), d["mean_diff"], d["t"]))
        out["common"][str(k)] = {"start": a0, "qlike": float(lk.mean()),
                                 "diff": d["mean_diff"], "t": d["t"]}
    for k, a0, q, dd, tt in rows:
        print(f"  {k:>3}{f'[{a0//1000}k,{(a0+W)//1000}k)':>20}{q:>10.5f}"
              f"{dd:>+12.5f}{tt:>+8.2f}")
    print(f"  {'--':>3}{'base-2 (r=12)':>20}{float(lr.mean()):>10.5f}")
    out["common_n"] = int(len(i))
    out["common_ref_qlike"] = float(lr.mean())

    ds = np.array([r[3] for r in rows])
    k0 = ds[0]
    rank0 = int(np.sum(ds <= k0))
    print(f"\n  published freeze (k=0) d = {k0:+.5f}; rank {rank0} of "
          f"{len(ds)} (1 = best)")
    print(f"  placebo freezes k>=1: mean {ds[1:].mean():+.5f}, "
          f"sd {ds[1:].std():.5f}, {int(np.sum(ds[1:] < 0))}/{len(ds)-1} < 0")
    out["common_k0_rank"] = rank0
    out["common_placebo_mean"] = float(ds[1:].mean())
    out["common_placebo_sd"] = float(ds[1:].std())

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
