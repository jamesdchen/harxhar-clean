"""Which exogenous channel collapsed the forecast on 13-15 October 2023?

analysis/exog_breakdown.py localised the exogenous block's only catastrophic
failure in twenty years: 27 bars carry 47% of the damage, all ten worst fall on
13-15 October 2023, and the failure mode is a forecast one to two orders of
magnitude too LOW while the backbone stays within a factor of two on the same
bars. Effective degrees of freedom are flat through the period, so this is not
over-fitting -- some input moved somewhere it had never been, and the fitted
coefficients turned that into a near-zero forecast.

The prediction is ((A cf)^2 + sigma^2) * baseline, and the adjusted-scale fit
A cf normally sits near 1. For the forecast to fall 30-fold, A cf must be
driven toward zero, which requires a large NEGATIVE contribution cancelling the
backbone. That contribution is attributable: for each channel c,

    contribution_c(t) = sum over rungs of  cf[c, r] * X[t, c, r]

so the channel responsible can be named rather than guessed.

Three passes:

  1. ATTRIBUTION. Per-channel contributions on the failing bars against the
     same channels' typical contributions in the preceding month. The channel
     whose contribution swings furthest negative is the culprit.
  2. THE INPUT ITSELF. Columns in results/b2_mmap are already rolling-robust
     scaled -- centred on a rolling median and divided by a rolling IQR -- so a
     value of 50 there means fifty IQRs from recent normal. That makes the
     cache its own anomaly detector; reading the offending channel's scaled
     values on those dates says whether the input spiked, collapsed, or went
     stale.
  3. THE RAW SERIES. Whatever the scaled view shows, the underlying raw column
     around those dates distinguishes a genuine market event from a data
     problem -- a stale feed, a units change, a vendor revision. A real event
     should show up in several related channels at once; a data problem should
     not.
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

D = os.path.join(ROOT, "results", "b2_mmap")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
W, CAD = 24000, 1000
LAM = 1e4
BAD_LO, BAD_HI = "2023-10-12", "2023-10-16"


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
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names)
                    if re.match(r"^har_ma_\d+$", nm)], int)
    C, R, nh = len(labels), 12, len(har)
    n = len(y)

    raw = load_raw_data("data", allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    tt = pd.to_datetime(raw["t"]).iloc[3125:3125 + n].reset_index(drop=True)

    bad = np.flatnonzero((tt >= BAD_LO) & (tt < BAD_HI))
    print(f"failing bars {BAD_LO}..{BAD_HI}: {len(bad)} rows "
          f"({tt.iloc[bad[0]]} .. {tt.iloc[bad[-1]]})")

    # the rolling refit that produced those forecasts
    s = int((bad[0] - W) // CAD) * CAD + W
    while s + CAD <= bad[0]:
        s += CAD
    print(f"  governing refit: train [{s-W:,},{s:,}) = "
          f"{tt.iloc[s-W].date()} .. {tt.iloc[s-1].date()}")

    cols = np.concatenate([har, lad])
    Ftr = np.asarray(X[s - W:s][:, cols], np.float64)
    A = np.hstack([np.ones((W, 1)), Ftr])
    G = A.T @ A
    c_ = A.T @ y[s - W:s]
    P = np.zeros((1 + len(cols), 1 + len(cols)))
    for k in range(C):
        st = 1 + nh + k * R
        P[st:st + R, st:st + R] = np.eye(R)
    cf = np.linalg.solve(G + LAM * P + 1e-8 * np.eye(len(P)), c_)
    del Ftr, A, G
    print(f"  fitted ({time.time()-t0:.0f}s)")

    # ---- 1. attribution
    ref = np.arange(bad[0] - 26 * 20, bad[0])          # preceding ~month
    Xb = np.asarray(X[bad][:, lad], np.float64).reshape(len(bad), C, R)
    Xr = np.asarray(X[ref][:, lad], np.float64).reshape(len(ref), C, R)
    W_ex = cf[1 + nh:].reshape(C, R)
    cb = np.einsum("ncr,cr->nc", Xb, W_ex)
    cr = np.einsum("ncr,cr->nc", Xr, W_ex)
    tot_b = cb.sum(1)
    Xh_b = np.asarray(X[bad][:, har], np.float64)
    back_b = cf[0] + Xh_b @ cf[1:1 + nh]
    print("\n" + "=" * 78)
    print("1. WHAT DROVE THE ADJUSTED-SCALE FIT TO ZERO")
    print("=" * 78)
    print(f"  backbone part on failing bars: mean {back_b.mean():+.3f}")
    print(f"  exogenous part on failing bars: mean {tot_b.mean():+.3f}")
    print(f"  exogenous part in prior month : mean {cr.sum(1).mean():+.3f}")
    print(f"  total A cf on failing bars    : mean "
          f"{(back_b + tot_b).mean():+.3f}   (normally near 1)")

    swing = cb.mean(0) - cr.mean(0)
    order = np.argsort(swing)
    print(f"\n  {'channel':>34}{'prior mo':>11}{'failing':>11}{'swing':>11}"
          f"{'share':>9}")
    neg = swing[swing < 0].sum()
    rows = []
    for i in order[:10]:
        sh = swing[i] / neg if neg != 0 else np.nan
        print(f"  {labels[i]:>34}{cr[:, i].mean():>+11.3f}"
              f"{cb[:, i].mean():>+11.3f}{swing[i]:>+11.3f}{sh:>9.1%}")
        rows.append({"channel": labels[i], "prior": float(cr[:, i].mean()),
                     "failing": float(cb[:, i].mean()),
                     "swing": float(swing[i]), "share": float(sh)})
    out = {"n_bad": int(len(bad)), "refit_start": int(s),
           "backbone_part": float(back_b.mean()),
           "exog_part": float(tot_b.mean()),
           "exog_part_prior": float(cr.sum(1).mean()),
           "attribution": rows}
    top = labels[order[0]]
    print(f"\n  -> the largest negative swing is {top}")

    # ---- 2. the scaled input
    print("\n" + "=" * 78)
    print("2. THE INPUT, IN ROLLING-IQR UNITS (the cache is prescaled)")
    print("=" * 78)
    print(f"  {'channel':>34}{'prior |max|':>13}{'failing |max|':>15}"
          f"{'ratio':>9}")
    inp = []
    for i in order[:6]:
        pm = float(np.abs(Xr[:, i, :]).max())
        fm = float(np.abs(Xb[:, i, :]).max())
        print(f"  {labels[i]:>34}{pm:>13.2f}{fm:>15.2f}"
              f"{fm / max(pm, 1e-12):>9.1f}x")
        inp.append({"channel": labels[i], "prior_absmax": pm,
                    "failing_absmax": fm, "ratio": fm / max(pm, 1e-12)})
    out["input_scale"] = inp

    # ---- 3. the raw series
    print("\n" + "=" * 78)
    print("3. THE RAW SERIES AROUND THE FAILURE")
    print("=" * 78)
    cand = [labels[i] for i in order[:6]]
    have = [c for c in cand if c in raw.columns]
    print(f"  raw columns available for: {have if have else 'none of the top 6'}")
    rawrows = []
    for cnm in have:
        v = raw[cnm].to_numpy(np.float64)[3125:3125 + n]
        pv, bv = v[ref], v[bad]
        fin_p = np.isfinite(pv).mean()
        fin_b = np.isfinite(bv).mean()
        print(f"\n  {cnm}")
        print(f"    prior month : finite {fin_p:.1%}  median "
              f"{np.nanmedian(pv):.4g}  min {np.nanmin(pv):.4g}  "
              f"max {np.nanmax(pv):.4g}")
        print(f"    failing days: finite {fin_b:.1%}  median "
              f"{np.nanmedian(bv):.4g}  min {np.nanmin(bv):.4g}  "
              f"max {np.nanmax(bv):.4g}")
        nuniq = int(len(np.unique(bv[np.isfinite(bv)])))
        print(f"    distinct finite values on the failing days: {nuniq}"
              + ("   <- STALE" if nuniq <= 2 else ""))
        rawrows.append({"channel": cnm, "finite_prior": float(fin_p),
                        "finite_failing": float(fin_b),
                        "median_prior": float(np.nanmedian(pv)),
                        "median_failing": float(np.nanmedian(bv)),
                        "distinct_failing": nuniq})
    out["raw"] = rawrows

    # how many channels moved at once?
    z = np.abs(Xb[:, :, 0]).max(0) / np.maximum(np.abs(Xr[:, :, 0]).max(0), 1e-12)
    nbig = int((z > 3).sum())
    print(f"\n  channels whose bar-level input exceeded 3x its prior-month "
          f"range: {nbig}/{C}")
    print("  -> " + ("a BROAD move: many channels at once, which looks like a "
                     "market event"
                     if nbig >= 5 else
                     "a NARROW move: few channels, which looks like a data "
                     "problem in those series"))
    out["n_channels_moved"] = nbig
    out["broad"] = bool(nbig >= 5)

    with open(os.path.join(OUT_DIR, "october_2023.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote october_2023.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
