"""One shape, chosen on rows nobody scores, tested on rows nobody chose.

imposed_taper found a fixed single column beating the twelve-rung ladder by
-0.00032 (t = -2.15, 6/8 eras) with no estimation at all. That number is not
reportable: it was the argmin over eleven shapes, and Bonferroni takes p from
0.032 to 0.35. Selection touching the reported loss is the fourth cause in this
study's own ledger, and the fix is not a correction factor -- it is a split.

  SELECT   rows [W, SPLIT).   Every candidate shape is scored here, the best is
           taken, and nothing from this region is reported as a result.
  TEST     rows [SPLIT, n).   The ONE selected shape is scored here against the
           twelve-rung ladder. One arm, one test, no multiplicity to correct.

The two regions are disjoint, so no row that chose the shape is a row that
judges it. Rolling coefficients still refit on trailing windows throughout,
which is the estimation scheme both arms share; only the projection differs.

The candidate grid is deliberately WIDER than the eleven shapes swept before.
Once selection is honest, a larger grid costs nothing but compute -- it cannot
inflate the reported statistic, because the reported statistic never sees it.
That is the whole point of the split, and it is worth stating plainly: the
earlier eleven-shape sweep was not wrong to explore, it was wrong to report its
minimum.

Inference on the test region follows this project's convention for a
concentrated differential rather than a bare DM t: HAC across a bandwidth
ladder, a stationary bootstrap at three block lengths, per-era means, and
leave-one-era-out. A result that only survives at one bandwidth is not a result.
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
from imposed_taper import half, kernel_exponent, unit             # noqa: E402

RG = np.array(RUNGS, float)
SPLIT = 130000
SEED = 20260804
NBOOT = 2000
OUT = os.path.join(ROOT, "writeup", "stats", "preregistered_taper.json")


def candidates():
    j = np.arange(12.0)
    c = {}
    for a in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        c[f"power a={a:.2f}"] = RG ** -a
    for p in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        c[f"lintaper^{p}"] = (1.0 - j / 12.0) ** p
    for hl in (2.0, 3.0, 4.0, 6.0):
        c[f"expo hl={hl:g}"] = 0.5 ** (j / hl)
    for a in (0.10, 0.20):
        c[f"lintaper x r^-{a:.2f}"] = (1.0 - j / 12.0) * RG ** -a
    return c


def stationary_bootstrap(d, block, nboot, rng):
    """Politis-Romano; returns the fraction of resamples with mean >= 0."""
    n = len(d)
    p = 1.0 / block
    out = np.empty(nboot)
    for bi in range(nboot):
        idx = np.empty(n, np.int64)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(n)
            else:
                i = (i + 1) % n
        out[bi] = d[idx].mean()
    return out


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
    CAND = candidates()
    print(f"cache {CACHE}: {n:,} rows")
    print(f"  SELECT rows [{W:,}, {SPLIT:,})   "
          f"TEST rows [{SPLIT:,}, {n:,})")
    print(f"  {len(CAND)} candidate shapes ({time.time()-t0:.0f}s)", flush=True)

    def run(Gm):
        F = np.column_stack([H @ Gm, cal])
        ok = np.isfinite(F).all(1)
        seg, _ = walk(np.where(ok[:, None], F, 0.0), y, b, W, W, n)
        seg[~ok[W:]] = np.nan
        return seg

    preds = {"base-2 (r=12)": run(np.eye(12))}
    print(f"    {'base-2 (r=12)':>22} ({time.time()-t0:.0f}s)", flush=True)
    for nm, v in CAND.items():
        preds[nm] = run(unit(v))
        print(f"    {nm:>22} ({time.time()-t0:.0f}s)", flush=True)

    good = tw > 0
    for k in preds:
        good &= np.isfinite(preds[k]) & (preds[k] > 0)
    sel = good.copy(); sel[SPLIT - W:] = False
    tst = good.copy(); tst[:SPLIT - W] = False
    isel, itst = np.flatnonzero(sel), np.flatnonzero(tst)
    ref = "base-2 (r=12)"
    print(f"\n  SELECTION -- {len(isel):,} bars, rows [{W:,}, {SPLIT:,})")
    print(f"  results here are NOT reported as findings\n")
    print(f"  {'shape':>22}{'half':>6}{'kappa':>8}{'QLIKE':>10}{'d vs r=12':>12}")
    qs = qlike_bar(tw[isel], preds[ref][isel]).mean()
    rows = []
    for nm in CAND:
        q = float(qlike_bar(tw[isel], preds[nm][isel]).mean())
        rows.append((q, nm))
    for q, nm in sorted(rows):
        print(f"  {nm:>22}{half(CAND[nm]):>6}{kernel_exponent(CAND[nm]):>8.3f}"
              f"{q:>10.5f}{q-qs:>+12.5f}")
    print(f"  {ref:>22}{12:>6}{'--':>8}{qs:>10.5f}")
    pick = sorted(rows)[0][1]
    print(f"\n  SELECTED: {pick}   (half {half(CAND[pick])}, "
          f"kappa {kernel_exponent(CAND[pick]):.3f})")

    out = {"cache": CACHE, "split": SPLIT, "n_candidates": len(CAND),
           "selected": pick, "selection": {nm: q for q, nm in rows},
           "selection_ref": float(qs), "n_select": int(len(isel)),
           "n_test": int(len(itst)),
           "selected_half": half(CAND[pick]),
           "selected_kappa": kernel_exponent(CAND[pick])}

    lk = qlike_bar(tw[itst], preds[pick][itst])
    lr = qlike_bar(tw[itst], preds[ref][itst])
    d = lk - lr
    print(f"\n  TEST -- {len(itst):,} bars, rows [{SPLIT:,}, {n:,})")
    print(f"  ONE arm, ONE test, no multiplicity\n")
    print(f"    {pick:>26}  QLIKE {lk.mean():.5f}")
    print(f"    {ref:>26}  QLIKE {lr.mean():.5f}")
    print(f"    {'difference':>26}  {d.mean():+.5f}")
    out["test"] = {"arm_qlike": float(lk.mean()), "ref_qlike": float(lr.mean()),
                   "diff": float(d.mean())}

    print(f"\n    HAC bandwidth ladder")
    out["test"]["dm_by_bandwidth"] = {}
    for bw in (0, 1, 10, 22, 100, 1000, 5000):
        r = dm_test(lk, lr, h=1, lag=bw)
        print(f"      bw {bw:>5}   t {r['t']:+.2f}   p {r['p']:.4g}")
        out["test"]["dm_by_bandwidth"][str(bw)] = {"t": r["t"], "p": r["p"]}

    print(f"\n    stationary bootstrap ({NBOOT} resamples)")
    rng = np.random.default_rng(SEED)
    out["test"]["bootstrap"] = {}
    for blk in (100, 1000, 5000):
        bs = stationary_bootstrap(d, blk, NBOOT, rng)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pv = float(np.mean(bs >= 0.0))
        print(f"      block {blk:>5}   95% CI [{lo:+.5f}, {hi:+.5f}]   "
              f"p {pv:.4f}")
        out["test"]["bootstrap"][str(blk)] = {"lo": float(lo), "hi": float(hi),
                                              "p": pv}

    e = np.array_split(d, NBLOCK)
    em = [float(x.mean()) for x in e]
    nb = int(sum(1 for x in em if x < 0))
    print(f"\n    eras better: {nb}/{NBLOCK}")
    print(f"    era means: " + " ".join(f"{x:+.5f}" for x in em))
    out["test"]["era_means"] = em
    out["test"]["eras_better"] = nb
    print(f"\n    leave-one-era-out")
    out["test"]["leave_one_era_out"] = {}
    for k in range(NBLOCK):
        m = np.ones(len(d), bool)
        a = sum(len(x) for x in e[:k])
        m[a:a + len(e[k])] = False
        r = dm_test(lk[m], lr[m], h=1)
        print(f"      drop era {k+1}   d {r['mean_diff']:+.5f}   "
              f"t {r['t']:+.2f}")
        out["test"]["leave_one_era_out"][str(k + 1)] = {
            "diff": r["mean_diff"], "t": r["t"]}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
