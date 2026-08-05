"""Is the exogenous block dead, or did something break on particular dates?

analysis/exog_when.py found the exogenous block beats the backbone in seven of
eight eras and fails in the last one, 2022-03 to 2024-04, by +0.03321 -- an
order of magnitude larger than any era's gain. Two readings survive that table
and they have opposite consequences:

  REGIME. The relationships have decayed and stayed decayed. Then the paper's
  attribution result describes a period that has ended, the exogenous block
  should not be carried forward, and "every bucket improves on the backbone"
  needs a date range attached.

  EPISODE. Something specific broke over some specific bars and the block is
  fine elsewhere, including at the end of the sample. Then the block is still
  worth carrying and the finding is about detecting and excluding an event.

The distinction is not visible in a two-year average, and it is decisive, so it
gets measured three ways:

  1. WHEN. The cumulative loss differential against the backbone, bar by bar.
     A regime change is a persistent slope; an episode is a step. Reported by
     calendar year, and by half within the failing era -- if the second half is
     clean the block recovered before the sample ended.

  2. HOW CONCENTRATED. What share of the total damage comes from the worst 1%
     and 5% of bars. QLIKE punishes under-forecasting without bound, so a
     handful of bars can carry an era's entire differential. If the worst 1%
     carry most of it, "exogenous stopped working" is the wrong description of
     a tail event.

  3. WHERE. The dates of the worst bars, and whether the arms that use the
     exogenous block are over- or under-forecasting on them.

Both exogenous arms lose to the backbone in that era -- ridge by +0.03321 and
shape-directed by +0.00666 -- so this is not a question about which penalty to
use. It is a question about whether the information is still there.
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

D = os.path.join(ROOT, "results", "b2_mmap")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
W, CAD = 24000, 1000
LAM_RIDGE, LAM_SHAPE = 1e4, 1e8


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from src.data.loading import load_raw_data
    t0 = time.time()

    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
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
    C, R, nh = len(labels), len(rungs), len(har)
    F = np.asarray(X[:, np.concatenate([har, lad])], np.float64)
    n, p = len(y), F.shape[1]
    rvv = y ** 2 * b

    raw = load_raw_data("data", allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    tt = pd.to_datetime(raw["t"]).iloc[3125:3125 + n].reset_index(drop=True)
    del raw

    Umax = int(rungs[-1])
    L = np.zeros((Umax, R))
    for k, r in enumerate(rungs):
        L[:int(r), k] = 1.0 / r
    M = L.T @ L
    M = M / max(np.trace(M) / R, 1e-12)
    g = rungs ** (-1.0)
    g = g / np.linalg.norm(g)
    Mg = M @ g
    Q = M - np.outer(Mg, Mg) / float(g @ M @ g)
    P_ridge = np.zeros((1 + p, 1 + p))
    P_shape = np.zeros((1 + p, 1 + p))
    for c in range(C):
        s = 1 + nh + c * R
        P_ridge[s:s + R, s:s + R] = np.eye(R)
        P_shape[s:s + R, s:s + R] = Q

    lo = W
    arms = ["backbone", "ridge", "shape"]
    preds = {a: np.full(n - lo, np.nan) for a in arms}
    for s in range(lo, n, CAD):
        e = min(s + CAD, n)
        A = np.hstack([np.ones((W, 1)), F[s - W:s]])
        G = A.T @ A
        c = A.T @ y[s - W:s]
        yt = y[s - W:s]
        Ae = np.hstack([np.ones((e - s, 1)), F[s:e]])
        cfb = np.linalg.solve(G[:1 + nh, :1 + nh] + 1e-8 * np.eye(1 + nh),
                              c[:1 + nh])
        rr = yt - A[:, :1 + nh] @ cfb
        preds["backbone"][s - lo:e - lo] = (
            (Ae[:, :1 + nh] @ cfb) ** 2 + float(rr @ rr) / W) * b[s:e]
        for tag, P, lam in (("ridge", P_ridge, LAM_RIDGE),
                            ("shape", P_shape, LAM_SHAPE)):
            cf = np.linalg.solve(G + lam * P + 1e-8 * np.eye(1 + p), c)
            rr = yt - A @ cf
            preds[tag][s - lo:e - lo] = (
                (Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
        del A, Ae, G
    print(f"walked {n - lo:,} bars ({time.time()-t0:.0f}s)")

    rv_w, dt = rvv[lo:], tt.iloc[lo:].reset_index(drop=True)
    good = rv_w > 0
    for a in arms:
        good &= np.isfinite(preds[a]) & (preds[a] > 0)
    gi = np.flatnonzero(good)
    lb = qlike_bar(rv_w[gi], preds["backbone"][gi])
    lr = qlike_bar(rv_w[gi], preds["ridge"][gi])
    ls = qlike_bar(rv_w[gi], preds["shape"][gi])
    yr = dt.iloc[gi].dt.year.to_numpy()
    out = {}

    # ---------------------------------------------------- 1. when
    print("\n" + "=" * 78)
    print("1. EXOGENOUS MINUS BACKBONE, BY CALENDAR YEAR (negative = helps)")
    print("=" * 78)
    print(f"  {'year':>6}{'bars':>8}{'backbone':>11}{'ridge-bb':>11}"
          f"{'shape-bb':>11}")
    rows = []
    for Y in sorted(set(yr.tolist())):
        m = yr == Y
        r_ = float(np.mean(lr[m] - lb[m]))
        s_ = float(np.mean(ls[m] - lb[m]))
        print(f"  {Y:>6}{int(m.sum()):>8}{float(np.mean(lb[m])):>11.5f}"
              f"{r_:>+11.5f}{s_:>+11.5f}")
        rows.append({"year": int(Y), "n": int(m.sum()),
                     "backbone": float(np.mean(lb[m])),
                     "ridge_minus_bb": r_, "shape_minus_bb": s_})
    out["by_year"] = rows

    # ---------------------------------------------------- 2. concentration
    print("\n" + "=" * 78)
    print("2. IS THE DAMAGE CONCENTRATED IN A FEW BARS?")
    print("=" * 78)
    era = dt.iloc[gi].to_numpy() >= np.datetime64("2022-03-08")
    d_r = lr[era] - lb[era]
    tot = float(d_r.sum())
    order = np.argsort(d_r)[::-1]
    conc = {}
    print(f"  failing era: {int(era.sum()):,} bars, total differential "
          f"{tot:+.1f} QLIKE-bars")
    for frac in (0.001, 0.01, 0.05, 0.10):
        k = max(1, int(len(d_r) * frac))
        share = float(d_r[order[:k]].sum()) / tot if tot != 0 else np.nan
        print(f"    worst {frac:>6.1%} of bars ({k:>5,}) carry "
              f"{share:>7.1%} of the damage")
        conc[str(frac)] = {"k": k, "share": share}
    med = float(np.median(d_r))
    print(f"    median per-bar differential {med:+.5f}  "
          f"(mean {float(d_r.mean()):+.5f})")
    print("  -> " + ("the era's damage is a TAIL EVENT: the median bar is "
                     "barely affected"
                     if abs(med) < 0.2 * abs(float(d_r.mean())) else
                     "the damage is BROAD: the typical bar is worse, not just "
                     "the tail"))
    out["concentration"] = {"n_era": int(era.sum()), "total": tot,
                            "levels": conc, "median": med,
                            "mean": float(d_r.mean())}

    # ---------------------------------------------------- 3. where, and after
    print("\n" + "=" * 78)
    print("3. THE WORST BARS, AND WHETHER IT RECOVERED")
    print("=" * 78)
    ei = gi[era]
    dd = dt.iloc[ei].reset_index(drop=True)
    worst = np.argsort(d_r)[::-1][:10]
    print(f"  {'date':>20}{'true RV':>12}{'backbone':>12}{'ridge':>12}"
          f"{'dQLIKE':>10}")
    wl = []
    for w_ in worst:
        i = ei[w_]
        print(f"  {str(dd.iloc[w_]):>20}{rv_w[i]:>12.3e}"
              f"{preds['backbone'][i]:>12.3e}{preds['ridge'][i]:>12.3e}"
              f"{d_r[w_]:>10.2f}")
        wl.append({"date": str(dd.iloc[w_]), "true": float(rv_w[i]),
                   "backbone": float(preds["backbone"][i]),
                   "ridge": float(preds["ridge"][i]), "d": float(d_r[w_])})
    out["worst_bars"] = wl

    half = len(d_r) // 2
    h1, h2 = float(d_r[:half].mean()), float(d_r[half:].mean())
    d1 = str(dd.iloc[0].date())
    dm_ = str(dd.iloc[half].date())
    d2 = str(dd.iloc[-1].date())
    print(f"\n  first half  {d1} .. {dm_}   ridge-backbone {h1:+.5f}")
    print(f"  second half {dm_} .. {d2}   ridge-backbone {h2:+.5f}")
    rec = h2 < 0
    print("  -> " + ("RECOVERED: the exogenous block is helping again by the "
                     "end of the sample,\n     so the era average describes an "
                     "episode that has passed"
                     if rec else
                     "NOT RECOVERED: the block is still a net cost at the end "
                     "of the sample,\n     so this looks like a persistent "
                     "state rather than an episode"))
    r_last = dm_test(lr[era][half:], lb[era][half:], h=1)
    s_last = dm_test(ls[era][half:], lb[era][half:], h=1)
    print(f"     second half DM vs backbone: ridge {r_last['mean_diff']:+.5f} "
          f"t {r_last['t']:+.2f} | shape {s_last['mean_diff']:+.5f} "
          f"t {s_last['t']:+.2f}")
    out["halves"] = {"first": {"range": [d1, dm_], "ridge_minus_bb": h1},
                     "second": {"range": [dm_, d2], "ridge_minus_bb": h2,
                                "ridge_dm_t": r_last["t"],
                                "shape_dm": s_last["mean_diff"],
                                "shape_dm_t": s_last["t"]},
                     "recovered": bool(rec)}

    with open(os.path.join(OUT_DIR, "exog_breakdown.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote exog_breakdown.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
