"""Is the k=0 win a long-memory FEATURE, or a fortunate reweighting?

freeze_placebo settled that it is not a chance draw: on identical rows with
identical estimation, the eight freeze windows nobody picked land in
[0.13490, 0.13537] while the published one lands at 0.13291 -- outside the pack
by thirteen placebo standard deviations. And the bases say why. Every placebo
is a decaying HAR shape dominated by rung 1; k=0 alone carries long-rung
structure (256 at +0.30, 1024 at -0.27, 2048 at +0.22).

The mechanism is legible. Rows [0, 24000) are 2005-2007, the flattest
long-horizon stretch in the sample -- smallest minimum eigenvalue of any
window (0.000778 against 0.00115-0.00423) and the flattest long rungs
(sd(2048)/sd(1) = 0.124 against 0.17-0.36). C^-1 amplifies the low-variance
tail hardest exactly there, so only that window manufactures a long-memory
contrast.

That leaves two readings, and they differ on whether the paper has a result:

  REAL FEATURE. The ladder genuinely needs a long-memory term, supervised
  estimation reliably fails to find it, and one pathological window stumbled
  into it. Then an IMPOSED long-rung shape -- chosen a priori, estimated from
  nothing -- reproduces the win, and the finding is derivable rather than lucky.

  FORTUNATE REWEIGHTING. Nothing imposed comes close, the win lives only in
  that exact estimated triple, and it is unreproducible by construction.

Imposed bases need no training window, so unlike every estimated arm they are
causal on every row and score on all 218,909 bars.

Two diagnostics decide WHERE the win lives, by swapping directions between the
published basis and the placebo consensus (the dominant subspace of the eight
bases nobody picked):

  consensus d1 + k0 d2,d3   -- if this wins, the long-rung directions carry it
  k0 d1 + consensus d2,d3   -- if this wins, the win is in direction 1

The pair is exhaustive: whichever holds, the other must not.
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
                          gram_schmidt, load_raw_data, qlike_bar, walk)
from basis_tracking import seed_basis                             # noqa: E402

OUT = os.path.join(ROOT, "writeup", "stats", "imposed_longmem.json")
RG = np.array(RUNGS, float)
LG = (np.log2(RG) - np.log2(RG).mean()) / np.log2(RG).std()


def orth(vs):
    return np.column_stack(gram_schmidt([np.asarray(v, float) for v in vs]))


def consensus(Gs):
    """Dominant rank-3 subspace of a set of bases: SVD of the stacked frames."""
    M = np.hstack(Gs)
    u, _, _ = np.linalg.svd(M, full_matrices=False)
    return u[:, :3]


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

    # the nine freeze bases
    Gs = []
    for k in range(9):
        a0, a1 = k * W, (k + 1) * W
        ok = np.isfinite(H[a0:a1]).all(1) & np.isfinite(y[a0:a1])
        Ht, yt = H[a0:a1][ok], y[a0:a1][ok]
        C = np.cov(Ht, rowvar=False)
        xy = (Ht - Ht.mean(0)).T @ (yt - yt.mean()) / len(yt)
        Gs.append(seed_basis(C, xy, 3, "sup"))
    G0, GC = Gs[0], consensus(Gs[1:])
    print(f"cache {CACHE}: {n:,} rows", flush=True)

    long_short = np.where(RG >= 256, 1.0, -1.0)
    arms = {
        # --- imposed a priori: no data touched at all
        "imposed power (r^-1,r^-.5,1)": orth([RG ** -1.0, RG ** -0.5,
                                              np.ones(12)]),
        "imposed logpoly (1,lg,lg^2)": orth([np.ones(12), LG, LG ** 2]),
        "imposed r^-1 + long-short":   orth([RG ** -1.0, RG ** -0.5,
                                             long_short]),
        "imposed r^-1 + lg^2":         orth([RG ** -1.0, RG ** -0.5, LG ** 2]),
        # --- reverse-engineered from k=0: labelled, NOT a priori
        "reverse-eng 256/1024/2048":   orth([RG ** -1.0, RG ** -0.5,
                                             np.where(RG == 256, 1.0, 0.0)
                                             - np.where(RG == 1024, 0.9, 0.0)
                                             + np.where(RG == 2048, 0.7, 0.0)]),
        # --- estimated references and the swap diagnostics
        "k=0 (published)":             G0,
        "placebo consensus (k>=1)":    GC,
        "consensus d1 + k0 d2,d3":     orth([GC[:, 0], G0[:, 1], G0[:, 2]]),
        "k0 d1 + consensus d2,d3":     orth([G0[:, 0], GC[:, 1], GC[:, 2]]),
        "k0 d1 only (rank 1)":         G0[:, :1],
    }

    preds = {}
    F0 = np.column_stack([H, cal])
    pr, _ = walk(np.where(np.isfinite(F0), F0, 0.0), y, b, W, W, n)
    pr[~np.isfinite(F0).all(1)[W:]] = np.nan
    preds["base-2 (r=12)"] = pr
    print(f"    {'base-2 (r=12)':>30} ({time.time()-t0:.0f}s)", flush=True)
    for nm, Gm in arms.items():
        F = np.column_stack([H @ Gm, cal])
        ok = np.isfinite(F).all(1)
        seg, _ = walk(np.where(ok[:, None], F, 0.0), y, b, W, W, n)
        seg[~ok[W:]] = np.nan
        preds[nm] = seg
        print(f"    {nm:>30} ({time.time()-t0:.0f}s)", flush=True)

    tw = truth[W:]
    good = tw > 0
    for k in preds:
        good &= np.isfinite(preds[k]) & (preds[k] > 0)
    i = np.flatnonzero(good)
    L = {k: qlike_bar(tw[i], preds[k][i]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    ref = "base-2 (r=12)"
    print(f"\n  scored {len(i):,} bars (clipped 5/95 truth)\n")
    print(f"  {'arm':>30}{'cols':>6}{'QLIKE':>10}{'d vs r=12':>12}{'t':>8}"
          f"{'eras':>7}")
    out = {"cache": CACHE, "n_scored": int(len(i)), "qlike": Q, "dm": {}}
    for k in sorted(Q, key=lambda k: Q[k]):
        nc = 12 if k == ref else arms[k].shape[1]
        if k == ref:
            print(f"  {k:>30}{nc:>6}{Q[k]:>10.5f}{'--':>12}")
            continue
        d = dm_test(L[k], L[ref], h=1)
        e = np.array_split(L[k] - L[ref], NBLOCK)
        nb = int(sum(1 for x in e if x.mean() < 0))
        print(f"  {k:>30}{nc:>6}{Q[k]:>10.5f}{d['mean_diff']:>+12.5f}"
              f"{d['t']:>+8.2f}{nb:>4}/{NBLOCK}")
        out["dm"][k] = {"diff": d["mean_diff"], "t": d["t"], "p": d["p"],
                        "eras_better": nb, "cols": int(nc)}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
