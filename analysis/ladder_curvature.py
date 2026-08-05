"""Do the twelve rungs lie near a CURVED three-dimensional surface?

manifold_drift_test.py asked whether the rank-3 subspace rotates, and found it
mostly does not -- direction 1 is flat in lag, sup's direction 3 is noise, only
pls's direction 3 drifts. But "manifold not subspace" has a second, sharper
reading that no arm on this footing has tested: not that the subspace MOVES, but
that the true map from lag-ladder to next-bar variance is a NONLINEAR function
of a low-dimensional coordinate.

Those are different objects and they make different predictions:

  ROTATING SUBSPACE. y is linear in a 3-dim projection whose directions change
  with time. Tracking should help. (basis_tracking.py tests that.)

  CURVED SURFACE. y is a nonlinear function of 3 fixed coordinates. Then a
  nonlinear model on THREE scores should beat a linear model on TWELVE rungs,
  because the twelve linear columns cannot represent curvature at all while the
  three scores plus their products can.

The second is what this measures, and it is a clean test because the comparison
is rank-deficient in the manifold's favour: three coordinates against twelve.
If a curved 3-surface is the right description, 3 scores + 6 quadratic terms
(9 columns) should beat 12 rungs. If it is not, the quadratic terms buy nothing
and the ladder is simply a linear object of low effective rank.

Arms, all per-bar OLS on the same rows, calendar block included throughout:

  base-2 (r=12)          the twelve rungs, linear. The thing to beat.
  lin r=3                three frozen scores, linear. The published compression.
  quad r=3               three scores + their 6 squares/cross-products.
  cube r=3               quad plus the 10 pure and mixed cubics.
  quad r=4 / lin r=4     the same at rank 4, since the two-basis agreement in
                         rank_sweep only starts at r = 4.
  quad-on-rungs          all 78 squares/cross-products of the TWELVE rungs.
                         The capacity control: if curvature exists but is not
                         confined to a 3-dim coordinate, this picks it up and
                         the r=3 quadratic does not.

The basis is frozen on rows [0, W) exactly as the published arm freezes it, so
any gain is attributable to the nonlinearity and not to re-estimation. Scored on
the clipped 5/95 truth, the battery footing, on common support across all arms.
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

KIND = os.environ.get("LC_KIND", "sup")
OUT = os.path.join(ROOT, "writeup", "stats", "ladder_curvature.json")


def poly(S, deg):
    """S (n, r) -> [S, all order-2 terms, ... up to deg], no intercept."""
    cols = [S]
    prev = [S[:, [k]] for k in range(S.shape[1])]
    idx = [(k,) for k in range(S.shape[1])]
    for _ in range(deg - 1):
        nxt, nidx = [], []
        for (c, ix) in zip(prev, idx):
            for k in range(ix[-1], S.shape[1]):
                nxt.append(c * S[:, [k]])
                nidx.append(ix + (k,))
        cols.append(np.hstack(nxt))
        prev, idx = nxt, nidx
    return np.hstack(cols)


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
    print(f"cache {CACHE}: {n:,} rows, kind={KIND}, frozen basis on [0,{W})",
          flush=True)

    ok0 = np.isfinite(H[:W]).all(1) & np.isfinite(y[:W])
    H0, y0 = H[:W][ok0], y[:W][ok0]
    C0 = np.cov(H0, rowvar=False)
    xy0 = (H0 - H0.mean(0)).T @ (y0 - y0.mean()) / len(y0)
    G = {r: seed_basis(C0, xy0, r, KIND) for r in (3, 4)}
    # standardise the scores on the SAME frozen window, so the polynomial
    # terms are not dominated by scale differences between coordinates
    SC = {}
    for r, Gm in G.items():
        Sfull = H @ Gm
        mu, sd = Sfull[:W][ok0].mean(0), Sfull[:W][ok0].std(0)
        SC[r] = (Sfull - mu) / np.maximum(sd, 1e-12)
    Hs = (H - H[:W][ok0].mean(0)) / np.maximum(H[:W][ok0].std(0), 1e-12)

    arms = {"base-2 (r=12)": H}
    for r in (3, 4):
        arms[f"lin r={r}"] = SC[r]
        arms[f"quad r={r}"] = poly(SC[r], 2)
    arms["cube r=3"] = poly(SC[3], 3)
    arms["quad-on-rungs"] = poly(Hs, 2)

    preds, ncol = {}, {}
    for nm, F in arms.items():
        Fd = np.column_stack([F, cal])
        okm = np.isfinite(Fd).all(1)
        pr, dr = walk(np.where(okm[:, None], Fd, 0.0), y, b, W, W, n)
        pr[~okm[W:]] = np.nan
        preds[nm] = pr
        ncol[nm] = int(F.shape[1])
        print(f"    {nm:>16} {F.shape[1]:>4} cols  drift {dr:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    tw = truth[W:]
    good = tw > 0
    for k in preds:
        good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(tw[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars (clipped 5/95 truth)\n")
    ref = "base-2 (r=12)"
    print(f"  {'arm':>16}{'cols':>6}{'QLIKE':>10}{'d vs r=12':>12}"
          f"{'t':>8}{'eras':>7}")
    out = {"cache": CACHE, "kind": KIND, "n_scored": int(len(idx)),
           "qlike": Q, "cols": ncol, "dm": {}}
    for k in sorted(Q, key=lambda k: Q[k]):
        if k == ref:
            print(f"  {k:>16}{ncol[k]:>6}{Q[k]:>10.5f}{'--':>12}")
            continue
        d = dm_test(L[k], L[ref], h=1)
        e = np.array_split(L[k] - L[ref], NBLOCK)
        nb = int(sum(1 for x in e if x.mean() < 0))
        out["dm"][k] = {"diff": d["mean_diff"], "t": d["t"], "p": d["p"],
                        "eras_better": nb}
        print(f"  {k:>16}{ncol[k]:>6}{Q[k]:>10.5f}{d['mean_diff']:>+12.5f}"
              f"{d['t']:>+8.2f}{nb:>4}/{NBLOCK}")

    # the curvature contrast that actually answers the question
    print()
    for r in (3, 4):
        d = dm_test(L[f"quad r={r}"], L[f"lin r={r}"], h=1)
        print(f"  curvature at r={r}: quad vs lin  {d['mean_diff']:+.5f}  "
              f"t {d['t']:+.2f}  p {d['p']:.3g}")
        out[f"curvature_r{r}"] = {"diff": d["mean_diff"], "t": d["t"],
                                  "p": d["p"]}
    d = dm_test(L["quad-on-rungs"], L[ref], h=1)
    print(f"  curvature on all 12 rungs vs linear 12: {d['mean_diff']:+.5f}  "
          f"t {d['t']:+.2f}  p {d['p']:.3g}")
    out["curvature_rungs"] = {"diff": d["mean_diff"], "t": d["t"], "p": d["p"]}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
