"""The stripped-down model on the whole panel, so its number is quotable.

analysis/dumb_model.py established the result on the last 35% of the panel:
53 columns and one ridge penalty (0.14830) beat 504 columns with a bespoke
matrix penalty (0.14878), the raw 492-column ridge (0.17906) and the backbone
alone (0.15242).

That evaluation window is not the paper's. This paper commits to one panel and
one benchmark, and refuses to quote results across incompatible evaluation
regimes, so before the stripped-down model can be written into it the same
comparison has to run over the full walk-forward the rest of the paper uses.
This walks all 218,909 post-warmup bars at the production cadence and reports
the whole-panel contrast, the by-era breakdown, and matched-row Diebold-Mariano
against both the backbone and the elaborate arm.

It also re-reports the era table with the stripped-down arm included, because
the exogenous block's single catastrophic era (2023, a data-pipeline artifact
traced in analysis/october_2023.py) is exactly where a 41-number summary should
behave differently from 492 raw columns: the amplitudes average over a
channel's whole rung ladder, so a single detonating rung is diluted rather than
carried at full weight.
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
LAM = {"backbone": 0.0, "dumb": 1e2, "ridge492": 1e4, "shaped492": 1e8}
NBLOCK = 8


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
    C, R, nh = len(labels), 12, len(har)
    n = len(y)
    rvv = y ** 2 * b
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64)
    g = rungs ** (-1.0)
    g = g / np.linalg.norm(g)
    Fa = Fl.reshape(n, C, R) @ g

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
    print(f"panel {n:,}  arms: " +
          ", ".join(f"{t}({design[t].shape[1]})" for t in design))

    lo = W
    preds = {t: np.full(n - lo, np.nan) for t in design}
    for s in range(lo, n, CAD):
        e = min(s + CAD, n)
        yt = y[s - W:s]
        for tag, Fd in design.items():
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            G = A.T @ A
            c = A.T @ yt
            cf = np.linalg.solve(G + LAM[tag] * pen[tag]
                                 + 1e-8 * np.eye(len(pen[tag])), c)
            rr = yt - A @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fd[s:e]])
            preds[tag][s - lo:e - lo] = (
                (Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del A, G, Ae
    print(f"  walked ({time.time()-t0:.0f}s)")

    rv_w, dt = rvv[lo:], tt.iloc[lo:].reset_index(drop=True)
    good = rv_w > 0
    for t in design:
        good &= np.isfinite(preds[t]) & (preds[t] > 0)
    gi = np.flatnonzero(good)
    losses = {t: qlike_bar(rv_w[gi], preds[t][gi]) for t in design}
    print(f"  scored {len(gi):,} bars")

    print("\n" + "=" * 78)
    print("WHOLE PANEL")
    print("=" * 78)
    print(f"  {'arm':>12}{'cols':>7}{'QLIKE':>11}{'MSE':>13}")
    res = {}
    for t in design:
        q = float(np.mean(losses[t]))
        res[t] = {"cols": int(design[t].shape[1]), "qlike": q,
                  "mse": float(np.mean((rv_w[gi] - preds[t][gi]) ** 2))}
        print(f"  {t:>12}{design[t].shape[1]:>7}{q:>11.5f}{res[t]['mse']:>13.4e}")

    print(f"\n  matched-row DM, dumb against each:")
    dms = {}
    for t in design:
        if t == "dumb":
            continue
        r = dm_test(losses["dumb"], losses[t], h=1)
        print(f"    vs {t:>10}: {r['mean_diff']:+.5f}  t {r['t']:+6.2f}  "
              f"p {r['p']:.3g}")
        dms[t] = {"diff": r["mean_diff"], "t": r["t"], "p": r["p"]}

    print("\n" + "=" * 78)
    print("BY ERA (delta versus the backbone; negative = the arm helps)")
    print("=" * 78)
    yr = dt.iloc[gi].dt.year.to_numpy()
    blocks = np.array_split(np.arange(len(gi)), NBLOCK)
    print(f"  {'block':>6}{'period':>22}{'backbone':>11}{'dumb':>11}"
          f"{'ridge492':>11}{'shaped492':>11}")
    rows = []
    for bi, blk in enumerate(blocks):
        qb = float(np.mean(losses["backbone"][blk]))
        d_ = {t: float(np.mean(losses[t][blk] - losses["backbone"][blk]))
              for t in design if t != "backbone"}
        per = f"{dt.iloc[gi[blk[0]]].date()}..{dt.iloc[gi[blk[-1]]].date()}"
        print(f"  {bi+1:>6}{per:>22}{qb:>11.5f}{d_['dumb']:>+11.5f}"
              f"{d_['ridge492']:>+11.5f}{d_['shaped492']:>+11.5f}")
        rows.append({"block": bi + 1, "period": per, "backbone": qb, **d_})
    nd = sum(1 for r_ in rows if r_["dumb"] < 0)
    nr = sum(1 for r_ in rows if r_["ridge492"] < 0)
    print(f"\n  beats the backbone in: dumb {nd}/{NBLOCK}, "
          f"ridge492 {nr}/{NBLOCK}")
    worst = max(rows, key=lambda r_: r_["ridge492"])
    print(f"  worst era for ridge492 ({worst['period']}): "
          f"ridge {worst['ridge492']:+.5f}, dumb {worst['dumb']:+.5f}, "
          f"shaped {worst['shaped492']:+.5f}")

    out = {"n_scored": int(len(gi)), "whole_panel": res, "dm_vs_dumb": dms,
           "eras": rows, "dumb_wins_eras": nd, "ridge_wins_eras": nr}
    with open(os.path.join(OUT_DIR, "dumb_full_panel.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote dumb_full_panel.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
