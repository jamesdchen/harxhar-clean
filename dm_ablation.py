"""Diebold-Mariano significance over the every-bar window ablation.

Loads the per-bar QLIKE losses saved by winablate_r1.py (cell_<id>_lossbar.npz) and runs the key DM
comparisons -- turning the QLIKE table's deltas into 'significant vs noise':
  (1) exog value:   all_buckets(best enet) vs HAR-only, per window;
  (2) more-data:    all_buckets(best enet) at each window vs the 1000d window (same eval rows);
  (3) penalty:      lasso vs ridge and enet vs ridge, per (window, all_buckets).
All comparisons are on the SAME fixed eval rows, so DM is valid. p<0.05 => significant.
"""

from __future__ import annotations

import os

import numpy as np

from src.evaluation.diebold_mariano import dm_test
from src.evaluation.model_confidence_set import model_confidence_set

D = "results/winablate_full"
WINDOWS = [48_000, 72_000, 96_000, 120_000, 144_000, 168_000]
BUCKETS = [
    "har_only",
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "vol_demand",
    "sentiment",
    "implied_vol",
    "all_buckets",
]
CELLS = [(w, b) for w in WINDOWS for b in BUCKETS]


def load(tid):
    p = f"{D}/cell_{tid:02d}_lossbar.npz"
    return dict(np.load(p)) if os.path.exists(p) else None


def cid(w, b):
    return CELLS.index((w, b))


def fmt(res):
    return f"dm={res['dm']:+.2f} p={res['p']:.4g} mean_diff={res['mean_diff']:+.2e} n={res['T']}"


def main():
    lb = {c: load(c) for c in range(len(CELLS))}

    print("=== (1) exog value: all_buckets(enet) vs HAR-only, per window ===")
    for w in WINDOWS:
        ab, ho = lb[cid(w, "all_buckets")], lb[cid(w, "har_only")]
        if not ab or not ho:
            print(f"  {w // 48}d: missing")
            continue
        r = dm_test(ab["enet"], ho["har_only"])
        sig = "SIG" if r["p"] < 0.05 else "ns"
        print(
            f"  {w // 48:4d}d: all_buckets vs HAR  {fmt(r)}  [{sig}]  ({'exog better' if r['mean_diff'] < 0 else 'HAR better'})"
        )

    print("\n=== (2) more-data: all_buckets(enet) window-w vs 1000d (same eval) ===")
    base = lb[cid(48_000, "all_buckets")]
    for w in WINDOWS[1:]:
        cur = lb[cid(w, "all_buckets")]
        if not cur or not base:
            print(f"  {w // 48}d: missing")
            continue
        r = dm_test(cur["enet"], base["enet"])
        sig = "SIG" if r["p"] < 0.05 else "ns"
        print(
            f"  {w // 48:4d}d vs 1000d  {fmt(r)}  [{sig}]  ({'larger better' if r['mean_diff'] < 0 else 'larger worse'})"
        )

    print(
        "\n=== (3) penalty: all_buckets lasso vs ridge, enet vs ridge, per window ==="
    )
    for w in WINDOWS:
        ab = lb[cid(w, "all_buckets")]
        if not ab:
            print(f"  {w // 48}d: missing")
            continue
        rlr = dm_test(ab["lasso"], ab["ridge"])
        rer = dm_test(ab["enet"], ab["ridge"])
        print(f"  {w // 48:4d}d: lasso-vs-ridge {fmt(rlr)} | enet-vs-ridge {fmt(rer)}")

    print(
        "\n=== (4) MCS: which TRAIN WINDOWS are in the confidence set (all_buckets, enet) ==="
    )
    wl = {
        f"{w // 48}d": lb[cid(w, "all_buckets")]["enet"]
        for w in WINDOWS
        if lb[cid(w, "all_buckets")]
    }
    if len(wl) >= 2:
        r = model_confidence_set(wl, alpha=0.10)
        print(
            f"  MCS_90% windows: {r['mcs']}  (eliminated worst-first: {r['eliminated']}, block={r['block']})"
        )
        print(
            "  MCS p-values:", {k: round(v, 3) for k, v in sorted(r["pvals"].items())}
        )

    print(
        "\n=== (5) MCS: which BUCKETS are in the confidence set (best config/bucket, per window) ==="
    )
    for w in WINDOWS:
        bl = {}
        for b in BUCKETS:
            d = lb[cid(w, b)]
            if not d:
                continue
            key = (
                "har_only"
                if b == "har_only"
                else min(
                    ("lasso", "ridge", "enet"), key=lambda k: float(np.nanmean(d[k]))
                )
            )
            bl[b] = d[key]
        if len(bl) >= 2:
            r = model_confidence_set(bl, alpha=0.10)
            print(f"  {w // 48:4d}d: MCS_90% = {r['mcs']}")


if __name__ == "__main__":
    main()
