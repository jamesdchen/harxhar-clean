"""Did the pipeline fix work? The paired comparison, old cache against new.

src/backtest/executor.load_and_transform now records exogenous availability
BEFORE the forward fill, bounds staleness at 26 bars, and applies the neutral
median to unobserved rows. results/b2_mmap_fix is the cache rebuilt through
that path. It has the same shape and column names as results/b2_mmap, and its
target and baseline arrays are bit-identical, so every difference below is
attributable to the exogenous features alone.

Four questions, in order of how directly they test the fix:

  1. ARE THE FEATURES TAMED? The cache is expressed in rolling-IQR units, so
     the scaled magnitudes are their own diagnostic. Before the fix, 29 of 41
     channels exceeded 20 IQRs and the volatility-demand family reached 2,260.
     If the availability ordering was the cause, those should collapse.

  2. DOES 13-15 OCTOBER 2023 SURVIVE? The direct test. That window carried 27
     bars holding 47% of the only catastrophic exogenous failure in twenty
     years, 97.2% of it attributable to one channel that is absent from the raw
     data throughout. Under the fix those bars are unobserved and take the
     neutral median instead of a stale value scaled by a collapsed IQR.

  3. WHAT HAPPENS TO THE OTHER NINETEEN YEARS? The fix touches every bar where
     any exogenous series was stale, not only the failing ones, so the era
     table is needed to see whether the block's contribution improves,
     degrades, or merely stops being volatile.

  4. DOES THE STRIPPED-DOWN MODEL STILL WIN? Its 41 amplitudes average over
     each channel's whole rung ladder, which is part of why it contained the
     2023 damage tenfold. If the fix removes the damage at source, that
     robustness argument weakens even as the design's other arguments stand.

Every arm is run on both caches over identical rows, so the comparison is
paired bar-for-bar rather than merely similar.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import dm_test  # noqa: E402

OLD = os.path.join(ROOT, "results", "b2_mmap")
NEW = os.path.join(ROOT, "results", "b2_mmap_fix")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
W, CAD = 24000, 1000
LAM = {"backbone": 0.0, "dumb": 1e2, "ridge492": 1e4, "shaped492": 1e8}
NBLOCK = 8


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def layout(names):
    chan: dict[str, list] = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan:
        chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    rungs = np.array([r for r, _ in chan[labels[0]]], float)
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names)
                    if re.match(r"^har_ma_\d+$", nm)], int)
    return labels, rungs, lad, har, chan


def walk(X, y, b, lad, har, C, R, rungs, arms):
    n = len(y)
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64)
    g = rungs ** (-1.0)
    g = g / np.linalg.norm(g)
    Fa = Fl.reshape(n, C, R) @ g
    nh = len(har)
    Umax = int(rungs[-1])
    L = np.zeros((Umax, R))
    for k, r in enumerate(rungs):
        L[:int(r), k] = 1.0 / r
    M = L.T @ L
    M = M / max(np.trace(M) / R, 1e-12)
    Mg = M @ g
    Q = M - np.outer(Mg, Mg) / float(g @ M @ g)
    design = {"backbone": Fh, "dumb": np.hstack([Fh, Fa]),
              "ridge492": np.hstack([Fh, Fl]),
              "shaped492": np.hstack([Fh, Fl])}
    pen = {}
    for tag, Fd in design.items():
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        if tag == "dumb":
            P[1 + nh:, 1 + nh:] = np.eye(C)
        elif tag in ("ridge492", "shaped492"):
            blk = np.eye(R) if tag == "ridge492" else Q
            for c in range(C):
                s = 1 + nh + c * R
                P[s:s + R, s:s + R] = blk
        pen[tag] = P
    preds = {t: np.full(n - W, np.nan) for t in arms}
    for s in range(W, n, CAD):
        e = min(s + CAD, n)
        yt = y[s - W:s]
        for tag in arms:
            Fd = design[tag]
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            G = A.T @ A
            c = A.T @ yt
            cf = np.linalg.solve(G + LAM[tag] * pen[tag]
                                 + 1e-8 * np.eye(len(pen[tag])), c)
            rr = yt - A @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fd[s:e]])
            preds[tag][s - W:e - W] = (
                (Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del A, G, Ae
    return preds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from src.data.loading import load_raw_data
    t0 = time.time()

    names = [str(s) for s in np.load(os.path.join(OLD, "names.npy"),
                                     allow_pickle=True)]
    labels, rungs, lad, har, chan = layout(names)
    C, R = len(labels), 12
    y = np.load(os.path.join(OLD, "y.npy"))
    b = np.load(os.path.join(OLD, "b.npy"))
    n = len(y)
    rvv = y ** 2 * b
    Xo = np.load(os.path.join(OLD, "X.npy"), mmap_mode="r")
    Xn = np.load(os.path.join(NEW, "X.npy"), mmap_mode="r")
    out = {}

    # ---------------------------------------------- 1. are the features tamed?
    print("=" * 84)
    print("1. SCALED MAGNITUDE, OLD CACHE AGAINST FIXED (rolling-IQR units)")
    print("=" * 84)
    idx = np.arange(0, n, 7)
    rows = []
    for c in labels:
        cols = [j for _, j in chan[c]]
        a = float(np.abs(np.asarray(Xo[idx][:, cols], np.float64)).max())
        bb = float(np.abs(np.asarray(Xn[idx][:, cols], np.float64)).max())
        rows.append({"channel": c, "old": a, "new": bb})
    rows.sort(key=lambda r: -r["old"])
    print(f"  {'channel':>34}{'old |max|':>12}{'new |max|':>12}{'ratio':>9}")
    for r in rows[:12]:
        print(f"  {r['channel']:>34}{r['old']:>12.1f}{r['new']:>12.1f}"
              f"{r['new'] / max(r['old'], 1e-12):>9.3f}")
    o20 = sum(1 for r in rows if r["old"] > 20)
    n20 = sum(1 for r in rows if r["new"] > 20)
    print(f"\n  channels above 20 IQRs: old {o20}/{C}, fixed {n20}/{C}")
    print(f"  worst channel: old {rows[0]['old']:.1f} -> fixed "
          f"{rows[0]['new']:.1f}")
    out["magnitudes"] = {"channels": rows, "old_over_20": o20,
                         "new_over_20": n20}

    av = [j for j, nm in enumerate(names) if "voldemand" in nm
          and "_avail_ma_1" in nm]
    if av:
        ao = float(np.asarray(Xo[idx][:, av[0]], np.float64).mean())
        an = float(np.asarray(Xn[idx][:, av[0]], np.float64).mean())
        print(f"  voldemand availability indicator: old {ao:.4f} -> "
              f"fixed {an:.4f}  (raw coverage 0.223)")
        out["avail_mean"] = {"old": ao, "new": an}

    # ---------------------------------------------- the walks
    arms = ["backbone", "dumb", "ridge492", "shaped492"]
    print(f"\n  walking both caches ({time.time()-t0:.0f}s) ...", flush=True)
    po = walk(Xo, y, b, lad, har, C, R, rungs, arms)
    pn = walk(Xn, y, b, lad, har, C, R, rungs, arms)
    print(f"  done ({time.time()-t0:.0f}s)")

    rv_w = rvv[W:]
    good = rv_w > 0
    for t in arms:
        good &= np.isfinite(po[t]) & (po[t] > 0)
        good &= np.isfinite(pn[t]) & (pn[t] > 0)
    gi = np.flatnonzero(good)
    lo_ = {t: qlike_bar(rv_w[gi], po[t][gi]) for t in arms}
    ln_ = {t: qlike_bar(rv_w[gi], pn[t][gi]) for t in arms}

    raw = load_raw_data("data", allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    tt = pd.to_datetime(raw["t"]).iloc[3125:3125 + n].reset_index(drop=True)
    del raw
    dt = tt.iloc[W:].reset_index(drop=True)

    # ---------------------------------------------- 2. October 2023
    print("\n" + "=" * 84)
    print("2. THE FAILING WINDOW, 13-15 OCTOBER 2023")
    print("=" * 84)
    dd = dt.iloc[gi].reset_index(drop=True)
    oct_ = np.flatnonzero((dd >= "2023-10-12") & (dd < "2023-10-16"))
    print(f"  {len(oct_)} bars in window")
    print(f"  {'arm':>12}{'old QLIKE':>13}{'fixed QLIKE':>14}{'change':>12}")
    octres = {}
    for t in arms:
        a, bq = float(np.mean(lo_[t][oct_])), float(np.mean(ln_[t][oct_]))
        print(f"  {t:>12}{a:>13.5f}{bq:>14.5f}{bq - a:>+12.5f}")
        octres[t] = {"old": a, "new": bq, "change": bq - a}
    out["october_2023"] = octres

    # ---------------------------------------------- 3. whole panel + eras
    print("\n" + "=" * 84)
    print("3. WHOLE PANEL AND ERAS")
    print("=" * 84)
    print(f"  scored {len(gi):,} bars")
    print(f"  {'arm':>12}{'old':>11}{'fixed':>11}{'change':>11}{'DM t':>9}")
    wres = {}
    for t in arms:
        a, bq = float(np.mean(lo_[t])), float(np.mean(ln_[t]))
        r = dm_test(ln_[t], lo_[t], h=1)
        print(f"  {t:>12}{a:>11.5f}{bq:>11.5f}{bq - a:>+11.5f}{r['t']:>9.2f}")
        wres[t] = {"old": a, "new": bq, "change": bq - a, "dm_t": r["t"],
                   "dm_p": r["p"]}
    out["whole_panel"] = wres

    blocks = np.array_split(np.arange(len(gi)), NBLOCK)
    print(f"\n  exogenous minus backbone by era (negative = helps)")
    print(f"  {'blk':>4}{'period':>22}"
          f"{'ridge old':>11}{'ridge fix':>11}{'dumb old':>11}{'dumb fix':>11}")
    eras = []
    for bi, blk in enumerate(blocks):
        ro = float(np.mean(lo_["ridge492"][blk] - lo_["backbone"][blk]))
        rn = float(np.mean(ln_["ridge492"][blk] - ln_["backbone"][blk]))
        do = float(np.mean(lo_["dumb"][blk] - lo_["backbone"][blk]))
        dn = float(np.mean(ln_["dumb"][blk] - ln_["backbone"][blk]))
        per = f"{dd.iloc[blk[0]].date()}..{dd.iloc[blk[-1]].date()}"
        print(f"  {bi+1:>4}{per:>22}{ro:>+11.5f}{rn:>+11.5f}{do:>+11.5f}"
              f"{dn:>+11.5f}")
        eras.append({"block": bi + 1, "period": per, "ridge_old": ro,
                     "ridge_new": rn, "dumb_old": do, "dumb_new": dn})
    out["eras"] = eras

    # ---------------------------------------------- 4. verdict
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    e8 = eras[-1]
    print(f"  failing era ridge-vs-backbone: {e8['ridge_old']:+.5f} -> "
          f"{e8['ridge_new']:+.5f}")
    fixed_era = e8["ridge_new"] < 0.25 * e8["ridge_old"]
    print("  -> " + ("the fix REMOVES most of the failing era's damage"
                     if fixed_era else
                     "the failing era survives the fix: the imputation defect "
                     "was not its only cause"))
    d, s = wres["dumb"]["new"], wres["shaped492"]["new"]
    print(f"\n  on the fixed cache: stripped-down {d:.5f}, shaped {s:.5f}, "
          f"backbone {wres['backbone']['new']:.5f}, "
          f"ridge492 {wres['ridge492']['new']:.5f}")
    r = dm_test(ln_["dumb"], ln_["shaped492"], h=1)
    print(f"  stripped-down vs shaped on the fixed cache: "
          f"{r['mean_diff']:+.5f}  t {r['t']:+.2f}  p {r['p']:.3g}")
    still = wres["dumb"]["new"] < min(wres[t]["new"] for t in arms
                                      if t != "dumb")
    print("  -> " + ("the stripped-down design still wins on the corrected "
                     "data" if still else
                     "the stripped-down design no longer wins once the data is "
                     "corrected"))
    out["verdict"] = {"failing_era_fixed": bool(fixed_era),
                      "stripped_still_wins": bool(still),
                      "dumb_vs_shaped_fixed": {"diff": r["mean_diff"],
                                               "t": r["t"], "p": r["p"]}}
    with open(os.path.join(OUT_DIR, "fixed_cache_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote fixed_cache_comparison.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
