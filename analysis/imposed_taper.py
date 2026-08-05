"""One fixed, estimation-free column against the twelve-rung ladder.

rank1_anatomy found that a single parameter -- the half-weight horizon -- orders
both the estimated and the imposed rank-1 arms (corr -0.835 over nine windows;
complete separation at half-weight 8/4 winning and 2 losing). r^-0.25 sits at
half-weight 8 and ties the ladder at +0.00034, but it is only an 11.3-degree
approximation to the pooled direction. A shape built to match should do better.

This sweeps a priori shapes and reports, for each, the induced kernel on the
RAW lags -- which is what a power law over rungs actually means.

THE RUNG-TO-LAG MAP. Rung j is a block mean of the last L_j = 2^j squared
returns, so a weighted sum of rungs is a quadrature rule on the lag axis with
piecewise-constant weights. In the variance domain the map is exact:

    sum_j w_j M_j  =  sum_i kappa(i) y^2_{t-i},
    kappa(i) = sum_{j : 2^j >= i} w_j / 2^j

With w_j proportional to L_j^-alpha the inner sum is geometric with ratio
2^-(alpha+1), so kappa(i) is proportional to i^-(alpha+1). A power law of
exponent alpha over rungs IS a power law of exponent alpha+1 over raw lags.
The design uses sqrt(M_j) rather than M_j, so this holds exactly in the
variance domain and to first order in the volatility domain (the sqrt rescales
contrasts by 1/(2 sqrt(Mbar)) without changing their shape).

That places the arms on the long-memory scale, where an ARFIMA(0,d,0) has
AR(inf) weights decaying as i^-(d+1):

    alpha = 0.00  ->  kappa ~ i^-1.00  ->  d = 0    (HAR's implicit target;
                                                     Corsi's mimicry of long
                                                     memory sits here)
    alpha = 0.25  ->  kappa ~ i^-1.25  ->  d = 0.25
    alpha = 1.00  ->  kappa ~ i^-2.00  ->  d = 1

The algebra is checked numerically below rather than trusted.

Every arm here is a priori: fixed before any data is seen, causal on every row,
no estimation step and therefore no direction variance at all. The pooled
consensus direction is carried as a reference but is NOT a priori -- it is built
from windows spanning rows 24,000-216,000, so it is scored only on rows after
that, and labelled.
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

RG = np.array(RUNGS, float)
OUT = os.path.join(ROOT, "writeup", "stats", "imposed_taper.json")


def unit(v):
    v = np.asarray(v, float)
    return (v / np.linalg.norm(v)).reshape(-1, 1)


def lag_kernel(w):
    """kappa(i) = sum_{j: 2^j >= i} w_j / 2^j, for i = 1..2048."""
    i = np.arange(1, 2049)
    return np.array([np.sum(w[RG >= ii] / RG[RG >= ii]) for ii in i]), i


def kernel_exponent(w):
    """LS slope of log kappa on log i -- the induced raw-lag power."""
    k, i = lag_kernel(w)
    m = k > 0
    A = np.column_stack([np.ones(m.sum()), np.log(i[m])])
    return float(-np.linalg.lstsq(A, np.log(k[m]), rcond=None)[0][1])


def half(v):
    w = np.maximum(np.asarray(v, float), 0.0)
    return int(RUNGS[int(np.searchsorted(np.cumsum(w) / w.sum(), 0.5))])


def main():
    t0 = time.time()

    # ---- check the rung-to-lag algebra before relying on it
    print("=" * 70)
    print("CHECK: does w ~ L^-alpha induce kappa(i) ~ i^-(alpha+1)?")
    print("=" * 70)
    print(f"  {'alpha':>8}{'predicted':>12}{'measured':>12}{'half rung':>11}")
    chk = {}
    for a in (0.0, 0.25, 0.5, 1.0, 2.0):
        e = kernel_exponent(RG ** -a)
        chk[str(a)] = e
        print(f"  {a:>8.2f}{a+1:>12.2f}{e:>12.3f}{half(RG**-a):>11d}")

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
    print(f"\ncache {CACHE}: {n:,} rows ({time.time()-t0:.0f}s)", flush=True)

    j = np.arange(12.0)
    arms = {}
    for a in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        arms[f"power a={a:.2f}"] = RG ** -a
    arms["lintaper (1-j/12)"] = 1.0 - j / 12.0
    arms["lintaper^2"] = (1.0 - j / 12.0) ** 2
    arms["lintaper^1.5"] = (1.0 - j / 12.0) ** 1.5
    arms["expo j (half=8)"] = 0.5 ** (j / 4.0)
    arms["lintaper x r^-0.1"] = (1.0 - j / 12.0) * RG ** -0.10

    preds = {}
    F0 = np.column_stack([H, cal])
    pr, _ = walk(np.where(np.isfinite(F0), F0, 0.0), y, b, W, W, n)
    pr[~np.isfinite(F0).all(1)[W:]] = np.nan
    preds["base-2 (r=12)"] = pr
    print(f"    {'base-2 (r=12)':>22} ({time.time()-t0:.0f}s)", flush=True)

    def run(Gm, lo):
        F = np.column_stack([H @ Gm, cal])
        ok = np.isfinite(F).all(1)
        seg, _ = walk(np.where(ok[:, None], F, 0.0), y, b, W, lo, n)
        seg[~ok[lo:n]] = np.nan
        p = np.full(n - W, np.nan)
        p[lo - W:] = seg
        return p

    for nm, v in arms.items():
        preds[nm] = run(unit(v), W)
        print(f"    {nm:>22} ({time.time()-t0:.0f}s)", flush=True)

    good = tw > 0
    for k in preds:
        good &= np.isfinite(preds[k]) & (preds[k] > 0)
    i = np.flatnonzero(good)
    ref = "base-2 (r=12)"
    lr = qlike_bar(tw[i], preds[ref][i])
    print(f"\n  A PRIORI, 1 column, {len(i):,} bars "
          f"(no estimation, no direction variance)\n")
    print(f"  {'arm':>22}{'half':>6}{'kappa exp':>11}{'QLIKE':>10}"
          f"{'d vs r=12':>12}{'t':>8}{'eras':>7}")
    out = {"cache": CACHE, "n_scored": int(len(i)), "algebra_check": chk,
           "arms": {}}
    rows = []
    for nm in arms:
        lk = qlike_bar(tw[i], preds[nm][i])
        d = dm_test(lk, lr, h=1)
        e = np.array_split(lk - lr, NBLOCK)
        nb = int(sum(1 for x in e if x.mean() < 0))
        rows.append((float(lk.mean()), nm, half(arms[nm]),
                     kernel_exponent(arms[nm]), d, nb))
    for q, nm, hw, ke, d, nb in sorted(rows):
        print(f"  {nm:>22}{hw:>6}{ke:>11.3f}{q:>10.5f}"
              f"{d['mean_diff']:>+12.5f}{d['t']:>+8.2f}{nb:>4}/{NBLOCK}")
        out["arms"][nm] = {"half": hw, "kappa_exp": ke, "qlike": q,
                           "diff": d["mean_diff"], "t": d["t"], "p": d["p"],
                           "eras_better": nb}
    print(f"  {ref:>22}{12:>6}{'--':>11}{float(lr.mean()):>10.5f}")
    out["ref_qlike"] = float(lr.mean())

    # ---- reference only: the pooled direction, NOT a priori
    Gs = []
    for k in range(9):
        a0, a1 = k * W, (k + 1) * W
        ok = np.isfinite(H[a0:a1]).all(1) & np.isfinite(y[a0:a1])
        Ht, yt = H[a0:a1][ok], y[a0:a1][ok]
        C = np.cov(Ht, rowvar=False)
        xy = (Ht - Ht.mean(0)).T @ (yt - yt.mean()) / len(yt)
        Gs.append(seed_basis(C, xy, 3, "sup"))
    GC = consensus(Gs[1:])
    dc = GC[:, 0] * np.sign(GC[np.argmax(np.abs(GC[:, 0])), 0])
    best = sorted(rows)[0][1]
    pc = run(unit(dc), 9 * W)
    m = np.zeros(n - W, bool)
    m[9 * W - W:] = True
    m &= (tw > 0) & np.isfinite(pc) & (pc > 0)
    for k in (ref, best):
        m &= np.isfinite(preds[k]) & (preds[k] > 0)
    i2 = np.flatnonzero(m)
    l2 = qlike_bar(tw[i2], preds[ref][i2])
    print(f"\n  LATE REGION ONLY, {len(i2):,} bars -- consensus is NOT a "
          f"priori, shown for scale\n")
    for nm, p in [(f"{best} (a priori)", preds[best]), ("consensus d1", pc)]:
        lk = qlike_bar(tw[i2], p[i2])
        d = dm_test(lk, l2, h=1)
        print(f"  {nm:>34}{lk.mean():>10.5f}{d['mean_diff']:>+12.5f}"
              f"{d['t']:>+8.2f}")
        out.setdefault("late", {})[nm] = {"qlike": float(lk.mean()),
                                          "diff": d["mean_diff"], "t": d["t"]}
    print(f"  {'base-2 (r=12)':>34}{float(l2.mean()):>10.5f}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
