"""Does the refit subspace DRIFT, or is it a constant subspace seen through
estimation error? Exact test, fixing two flaws in manifold_vs_noise.py.

FLAW 1 -- overlap. Training windows are W = 24,000 wide and refit every
CAD = 6,000, so consecutive windows share 75% of their rows. Overlap fraction
at lag L is max(0, 1 - L*CAD/W), i.e. 75/50/25/0% at L = 1/2/3/4. Any angle
growth over L = 1..3 is mechanical. Only L >= 4 (disjoint training data)
carries information about drift.

FLAW 2 -- the null. Permuting y inside a window drives xy to ~0, so the seed is
essentially random and K_r(C, seed) is a near-random subspace. Beating THAT is
uninformative. The null we actually need is "the true subspace is constant and
only sampling error varies", which predicts a lag profile that is FLAT over the
disjoint region.

The exact test for a flat profile is a permutation of WINDOW ORDER. Under the
constant-subspace null the 37 estimated bases are exchangeable, so any ordering
is equally likely and the observed lag slope is a draw from the permutation
distribution. Under drift the observed slope sits in its right tail. No
distributional assumption is used anywhere.

Reported per direction, because the whole question is whether directions 2-3
behave differently from direction 1.

Also run: OVERLAP-MATCHED SPACING. Refit at CAD = W = 24,000 gives strictly
disjoint training windows at every lag, removing the confound by construction
rather than by conditioning. If drift is real the profile should still rise.
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

from ladder_refit import (RUNGS, W, build_target,  # noqa: E402
                          load_raw_data)
from manifold_vs_noise import angles, basis  # noqa: E402

CACHE = os.environ.get("RANK_CACHE", "b2_mmap_warm")
RANK = 3
NPERM = 5000
SEED = 20260803
OUT = os.path.join(ROOT, "writeup", "stats", "manifold_drift_test.json")


def profile(Gs, lag_min):
    """mean angle per direction at each lag >= lag_min, plus the lags."""
    m = len(Gs)
    acc = {}
    for i in range(m):
        for j in range(i + 1, m):
            if j - i >= lag_min:
                acc.setdefault(j - i, []).append(angles(Gs[i], Gs[j]))
    Ls = sorted(acc)
    A = np.array([np.mean(acc[L], axis=0) for L in Ls])   # (nlag, RANK)
    return np.array(Ls, float), A


def slope(Ls, A):
    """OLS slope of mean angle on ln(lag), per direction."""
    X = np.column_stack([np.ones(len(Ls)), np.log(Ls)])
    return np.linalg.lstsq(X, A, rcond=None)[0][1]


def run(Gs, tag, lag_min, rng, out):
    Ls, A = profile(Gs, lag_min)
    obs = slope(Ls, A)
    null = np.empty((NPERM, A.shape[1]))
    idx = np.arange(len(Gs))
    for p in range(NPERM):
        perm = rng.permutation(idx)
        Lp, Ap = profile([Gs[i] for i in perm], lag_min)
        null[p] = slope(Lp, Ap)
    pv = (1.0 + np.sum(null >= obs[None, :], axis=0)) / (NPERM + 1.0)
    z = (obs - null.mean(0)) / null.std(0)
    print(f"\n  {tag}  (lags {int(Ls[0])}..{int(Ls[-1])}, "
          f"{len(Gs)} windows, {NPERM} permutations)")
    print(f"    {'lag':>5}" + "".join(f"{'dir'+str(k+1):>9}"
                                      for k in range(A.shape[1])))
    for i, L in enumerate(Ls):
        if int(L) in (int(Ls[0]), 6, 8, 12, 16, 24, int(Ls[-1])):
            print(f"    {int(L):>5}" + "".join(f"{v:>9.2f}" for v in A[i]))
    print(f"    {'slope':>5}" + "".join(f"{v:>+9.2f}" for v in obs)
          + "   deg per ln(lag)")
    print(f"    {'z':>5}" + "".join(f"{v:>+9.2f}" for v in z))
    print(f"    {'p':>5}" + "".join(f"{v:>9.4f}" for v in pv)
          + "   <- permutation, one-sided (drift = small p)")
    out[tag] = {"lags": Ls.tolist(), "angles": A.tolist(),
                "slope": obs.tolist(), "z": z.tolist(), "p": pv.tolist(),
                "n_windows": len(Gs), "lag_min": lag_min}
    return obs, pv


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    n = len(np.load(os.path.join(D, "y.npy")))
    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    off = len(raw) - n
    y99, _ = build_target(raw, 0.01, 0.99)
    y = y99[off:off + n]
    s = pd.Series(y ** 2)
    H = np.column_stack([
        np.sqrt(np.maximum(
            s.rolling(L, min_periods=L).mean().shift(1).to_numpy(), 0.0))
        for L in RUNGS])
    rng = np.random.default_rng(SEED)
    out = {"cache": CACHE, "n": int(n), "rank": RANK, "nperm": NPERM,
           "W": W}

    # ---- A. published cadence, conditioned on zero training overlap
    bnds = list(range(W, n, 6000)) + [n]
    wins = [(bnds[i] - W, bnds[i]) for i in range(len(bnds) - 1)]
    print(f"cache {CACHE}: {n:,} rows")
    print(f"A. CAD=6000, W={W}: {len(wins)} windows, overlap 0 from lag 4")
    for kind in ("sup", "pls"):
        Gs = [basis(H[a:b], y[a:b], RANK, kind) for a, b in wins]
        run(Gs, f"{kind} CAD=6000 lag>=4", 4, rng, out)

    # ---- B. disjoint by construction
    bnds = list(range(W, n, W)) + [n]
    wins = [(bnds[i] - W, bnds[i]) for i in range(len(bnds) - 1)]
    print(f"\nB. CAD=W={W}: {len(wins)} windows, DISJOINT at every lag "
          f"({time.time()-t0:.0f}s)")
    for kind in ("sup", "pls"):
        Gs = [basis(H[a:b], y[a:b], RANK, kind) for a, b in wins]
        run(Gs, f"{kind} disjoint lag>=1", 1, rng, out)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
