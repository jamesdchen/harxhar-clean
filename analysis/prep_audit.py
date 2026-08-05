"""Sweep the whole feature prep for the pathology found in voldemand.

The October 2023 collapse had a signature: a channel whose raw series is mostly
absent, forward-filled into near-constancy, then divided by a rolling IQR that
the constancy drove toward zero, producing scaled coordinates in the hundreds
of IQRs that the model fits coefficients to. The question this answers is
whether voldemand was the only one.

Reading the code first turns up three more instances of the same FAMILY, at
different stages of being fixed:

  * FIXED. src/features/transforms/target.py floors the diurnal std at a
    fraction of its own median, explicitly because "the old replace(0, 1.0)
    only caught an exactly zero std ... so a tiny-but-nonzero std on a
    large-scale signed feature (voldemand ~ +-1e6) blew the adjusted value up
    to ~+-1e13".
  * NOT FIXED. src/features/transforms/scaling.py still guards the rolling IQR
    with ``iqr = iq if iq >= 1e-12 else 1.0`` -- an ABSOLUTE threshold, which
    is the exact-zero check target.py already learned is insufficient. An IQR
    of 1e-11 on a series of scale 1e-6 passes the guard and divides by
    ~nothing.
  * NOT FIXED. The non-negative branch of robust_transform still uses
    ``baseline.replace(0, 1.0)``, the same exact-zero check on a divisor.
  * DEAD. apply_overnight_fills would fill NaN with 1.0 by substring match,
    but RATIO_FILL_COLS is empty so it is a no-op.

Whether the two live ones bite is an empirical question, so this measures every
channel on the signature rather than reasoning about which should be affected:

  raw coverage       fraction of bars where the underlying series is observed
  longest flat run   bars of exact constancy, which is what collapses an IQR
  zero fraction      a point mass at zero collapses it the same way, and the
                     hurdle encoding only triggers above ZERO_INFLATED_FRAC
  scaled |max|       the cache is prescaled in rolling-IQR units, so this is
                     directly readable as "how many IQRs from normal", and
                     anything past ~20 is not a market move
  tail mass          fraction of bars beyond 10 IQRs

Channels ranked by scaled |max|. Any channel pairing a large one with low
coverage or a long flat run is the voldemand pathology under another name.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
D = os.path.join(ROOT, "results", "b2_mmap")
STRIDE = 7          # the cache is 2.1 GB; every 7th row is plenty for tails


def longest_flat_run(v: np.ndarray) -> int:
    f = np.isfinite(v)
    if f.sum() < 2:
        return int(len(v))
    d = np.diff(v[f])
    best = cur = 0
    for x in d:
        cur = cur + 1 if x == 0.0 else 0
        best = max(best, cur)
    return int(best)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from src.data.loading import load_raw_data

    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    n = X.shape[0]
    raw = load_raw_data("data", allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    sub = raw.iloc[3125:3125 + n].reset_index(drop=True)

    chan: dict[str, list[int]] = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append(j)
    print(f"cache {X.shape}, {len(chan)} adjusted channels, "
          f"sampling every {STRIDE} rows")

    rows_idx = np.arange(0, n, STRIDE)
    rows = []
    for c, cols in sorted(chan.items()):
        V = np.asarray(X[rows_idx][:, cols], np.float64)
        amax = float(np.abs(V).max())
        tail = float((np.abs(V) > 10).mean())
        if c in sub.columns:
            rv_ = sub[c].to_numpy(np.float64)
            cov = float(np.isfinite(rv_).mean())
            flat = longest_flat_run(rv_)
            fin = rv_[np.isfinite(rv_)]
            zf = float(np.mean(fin == 0.0)) if fin.size else float("nan")
        else:
            cov, flat, zf = float("nan"), -1, float("nan")
        rows.append({"channel": c, "raw_coverage": cov, "longest_flat": flat,
                     "zero_frac": zf, "scaled_absmax": amax,
                     "tail_frac_gt10": tail,
                     "has_avail": f"{c}_avail_ma_1" in names})
    rows.sort(key=lambda r: -r["scaled_absmax"])

    print("\n" + "=" * 92)
    print("CHANNELS RANKED BY SCALED MAGNITUDE (units = rolling IQRs)")
    print("=" * 92)
    print(f"  {'channel':>34}{'|max|':>11}{'>10 IQR':>10}{'raw cov':>10}"
          f"{'flat run':>10}{'zero%':>8}{'avail':>7}")
    for r in rows:
        flag = ""
        if r["scaled_absmax"] > 20 and (r["raw_coverage"] < 0.9
                                        or r["longest_flat"] > 500):
            flag = "  <-- voldemand pathology"
        elif r["scaled_absmax"] > 20:
            flag = "  <-- extreme, coverage ok"
        print(f"  {r['channel']:>34}{r['scaled_absmax']:>11.1f}"
              f"{r['tail_frac_gt10']:>10.4f}{r['raw_coverage']:>10.3f}"
              f"{r['longest_flat']:>10}{r['zero_frac']:>8.3f}"
              f"{str(r['has_avail']):>7}{flag}")

    bad = [r for r in rows if r["scaled_absmax"] > 20
           and (r["raw_coverage"] < 0.9 or r["longest_flat"] > 500)]
    hot = [r for r in rows if r["scaled_absmax"] > 20]
    print("\n" + "=" * 92)
    print(f"  {len(hot)}/{len(rows)} channels exceed 20 rolling IQRs somewhere")
    print(f"  {len(bad)}/{len(rows)} pair that with sparse coverage or a long "
          f"flat run -- the voldemand signature")
    if bad:
        print("  affected: " + ", ".join(r["channel"] for r in bad))
    lowcov = [r for r in rows if r["raw_coverage"] < 0.5]
    print(f"\n  {len(lowcov)}/{len(rows)} channels are observed on under half "
          f"of bars:")
    for r in lowcov:
        print(f"    {r['channel']:>34}  coverage {r['raw_coverage']:.3f}  "
              f"scaled |max| {r['scaled_absmax']:.1f}")

    out = {"n_channels": len(rows), "stride": STRIDE, "channels": rows,
           "n_over_20_iqr": len(hot), "n_pathology": len(bad),
           "pathology": [r["channel"] for r in bad],
           "low_coverage": [r["channel"] for r in lowcov],
           "code_findings": {
               "scaling.rolling_robust_scale":
                   "absolute 1e-12 IQR guard; target.py fixed the analogous "
                   "bug with a relative floor. LIVE.",
               "target.robust_transform_nonnegative":
                   "baseline.replace(0, 1.0) is an exact-zero check on a "
                   "divisor. LIVE.",
               "target.robust_transform_signed":
                   "floors the std at a fraction of its median. FIXED.",
               "loading.apply_overnight_fills":
                   "fills NaN with 1.0 by substring match, but "
                   "RATIO_FILL_COLS is empty. DEAD."}}
    with open(os.path.join(OUT_DIR, "prep_audit.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote prep_audit.json")


if __name__ == "__main__":
    main()
