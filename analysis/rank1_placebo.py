"""Is the rank-ONE result reproducible, or is it k=0-specific like the triple?

imposed_longmem left exactly one cell untested, and it is the one that decides
whether this study has a compression result at all.

  k=0's three estimated directions beat the twelve-rung ladder by -0.00080, and
  nothing reproduces that: eight placebo freezes all lose, every imposed shape
  loses by +0.00086 to +0.00119, a shape hand-copied from k=0's own loadings
  loses by +0.00108, and neither half of a direction-swap survives. That triple
  is unreproducible.

  But k=0's FIRST direction alone -- one column against twelve -- beats the
  ladder by -0.00047 (t = -3.15) in six of eight eras, MORE eras than the full
  triple manages. And direction 1 is the stable one: 1-2 degrees across refits,
  flat in lag (permutation p = 0.84), and the direction you get without
  inverting anything.

So: does rank-1 work because it is direction 1, or because it is k=0's
direction 1? Two blocks answer it.

  FULL  (218,909 bars). Arms causal on every row: k=0's d1, and imposed rank-1
        shapes that estimate nothing at all. If an imposed single column beats
        twelve rungs, that is the cleanest result this project could report.

  NULL  (rows 216,000+, common support). The same rank-1 placebo design as
        freeze_placebo: each of the nine freeze windows contributes its own d1,
        all scored on identical rows with identical trailing estimation. If the
        eight windows nobody picked also beat r=12, rank-1 compression is real
        and estimable. If only k=0 does, then even the one-column result is a
        property of 2005-2007 and the compression story is finished.

The rank-3 placebo already gave its answer; this asks the same question of the
only arm still standing.
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
from ladder_refit import (CACHE, NBLOCK, RUNGS, W, build_target,  # noqa: E402
                          load_raw_data, qlike_bar, walk)
from basis_tracking import seed_basis                             # noqa: E402
from imposed_longmem import consensus                             # noqa: E402

OUT = os.path.join(ROOT, "writeup", "stats", "rank1_placebo.json")
RG = np.array(RUNGS, float)


def unit(v):
    v = np.asarray(v, float)
    return (v / np.linalg.norm(v)).reshape(-1, 1)


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
    tw = truth[W:]

    Gs = []
    for k in range(9):
        a0, a1 = k * W, (k + 1) * W
        ok = np.isfinite(H[a0:a1]).all(1) & np.isfinite(y[a0:a1])
        Ht, yt = H[a0:a1][ok], y[a0:a1][ok]
        C = np.cov(Ht, rowvar=False)
        xy = (Ht - Ht.mean(0)).T @ (yt - yt.mean()) / len(yt)
        Gs.append(seed_basis(C, xy, 3, "sup"))
    GC = consensus(Gs[1:])
    print(f"cache {CACHE}: {n:,} rows", flush=True)

    def run(Gm, lo):
        F = np.column_stack([H @ Gm, cal])
        ok = np.isfinite(F).all(1)
        seg, _ = walk(np.where(ok[:, None], F, 0.0), y, b, W, lo, n)
        seg[~ok[lo:n]] = np.nan
        p = np.full(n - W, np.nan)
        p[lo - W:] = seg
        return p

    def table(preds, ref, mask0, title):
        good = mask0 & (tw > 0)
        for k in preds:
            good &= np.isfinite(preds[k]) & (preds[k] > 0)
        good &= np.isfinite(ref) & (ref > 0)
        i = np.flatnonzero(good)
        lr = qlike_bar(tw[i], ref[i])
        print(f"\n  {title} -- {len(i):,} bars\n")
        print(f"  {'arm':>30}{'QLIKE':>10}{'d vs r=12':>12}{'t':>8}{'eras':>7}")
        res = {}
        for k in sorted(preds, key=lambda k: float(
                qlike_bar(tw[i], preds[k][i]).mean())):
            lk = qlike_bar(tw[i], preds[k][i])
            d = dm_test(lk, lr, h=1)
            e = np.array_split(lk - lr, NBLOCK)
            nb = int(sum(1 for x in e if x.mean() < 0))
            print(f"  {k:>30}{lk.mean():>10.5f}{d['mean_diff']:>+12.5f}"
                  f"{d['t']:>+8.2f}{nb:>4}/{NBLOCK}")
            res[k] = {"qlike": float(lk.mean()), "diff": d["mean_diff"],
                      "t": d["t"], "eras_better": nb}
        print(f"  {'base-2 (r=12)':>30}{float(lr.mean()):>10.5f}")
        res["base-2 (r=12)"] = {"qlike": float(lr.mean())}
        return res, len(i)

    out = {"cache": CACHE}
    ref = run(np.eye(12), W)
    print(f"    {'base-2 (r=12)':>30} ({time.time()-t0:.0f}s)", flush=True)

    # ---- block FULL: everything causal on every row
    full = {"k=0 d1 (estimated)": unit(Gs[0][:, 0])}
    for nm, v in [("imposed r^-1", RG ** -1.0),
                  ("imposed r^-0.5", RG ** -0.5),
                  ("imposed r^-0.25", RG ** -0.25),
                  ("imposed log-decay", 1.0 / (1.0 + np.log2(RG))),
                  ("imposed HAR 3-rung", (RG <= 16).astype(float)),
                  ("imposed equal (mean)", np.ones(12))]:
        full[nm] = unit(v)
    P = {}
    for nm, Gm in full.items():
        P[nm] = run(Gm, W)
        print(f"    {nm:>30} ({time.time()-t0:.0f}s)", flush=True)
    r, nfull = table(P, ref, np.ones(n - W, bool),
                     "FULL -- rank 1, causal on every row")
    out["full"] = r
    out["full_n"] = nfull

    # ---- block NULL: rank-1 placebo across the nine freeze windows
    Pn = {"consensus d1 (k>=1)": run(unit(GC[:, 0]), 9 * W)}
    print(f"    {'consensus d1':>30} ({time.time()-t0:.0f}s)", flush=True)
    for k in range(9):
        lo = max(W, (k + 1) * W)
        Pn[f"freeze k={k} d1"] = run(unit(Gs[k][:, 0]), lo)
        print(f"    {f'freeze k={k} d1':>30} ({time.time()-t0:.0f}s)",
              flush=True)
    m0 = np.zeros(n - W, bool)
    m0[9 * W - W:] = True
    r, nn = table(Pn, ref, m0, "NULL -- rank 1, common late region")
    out["null"] = r
    out["null_n"] = nn
    ds = np.array([r[f"freeze k={k} d1"]["diff"] for k in range(9)])
    nb = int(np.sum(ds[1:] < 0))
    print(f"\n  k=0 d1 d = {ds[0]:+.5f}; rank {int(np.sum(ds <= ds[0]))} of 9")
    print(f"  placebo d1 (k>=1): mean {ds[1:].mean():+.5f}, "
          f"sd {ds[1:].std():.5f}, {nb}/8 beat r=12")
    out["null_k0_rank"] = int(np.sum(ds <= ds[0]))
    out["null_placebo_mean"] = float(ds[1:].mean())
    out["null_placebo_sd"] = float(ds[1:].std())
    out["null_placebo_wins"] = nb

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
