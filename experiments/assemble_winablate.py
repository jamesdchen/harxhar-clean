"""Assemble the 54 per-cell JSONs (winablate_cells/) into the full window x bucket x penalty tables.

Emits markdown for writeup section 7:
  A) all_buckets: rows=windows, cols=HAR-only/lasso/ridge/enet best-tuned + best (the window thesis);
  B) best cfg per window (does optimal reg go heavier with more data?);
  C) full grid: best-vs-HAR delta per (bucket, window).
All on the FIXED eval [EVAL_START:n] (covid-inclusive, absolutes ~0.14).
"""

from __future__ import annotations

import glob
import json

CELLDIR = "results/winablate_full"
OUT = "results/winablate_assembled.json"


def main():
    cells = [json.load(open(p)) for p in sorted(glob.glob(f"{CELLDIR}/cell_*.json"))]
    if not cells:
        print("no cells found")
        return
    windows = sorted({c["window"] for c in cells})
    buckets = ["moments", "liquidity", "market_ew", "market_vw", "vol_demand", "sentiment", "implied_vol", "all_buckets"]
    by = {(c["window"], c["bucket"]): c for c in cells}
    har = {w: by.get((w, "har_only"), {}).get("har_only") for w in windows}
    print(f"assembled {len(cells)}/54 cells; windows={[w//48 for w in windows]}d\n")

    # A) all_buckets window x penalty
    print("### A. all_buckets: window x penalty (best OOS-tuned QLIKE, HAR OLS, fixed eval)\n")
    print("| train window | HAR-only | lasso | ridge | enet | best |")
    print("|---|---|---|---|---|---|")
    for w in windows:
        c = by.get((w, "all_buckets"))
        h = har.get(w)
        if not c or h is None:
            print(f"| {w//48}d | (missing) | | | | |")
            continue
        best = min(c.get("lasso", 9), c.get("ridge", 9), c.get("enet", 9))
        print(f"| {w//48}d ({w}) | {h:.5f} | {c.get('lasso',float('nan')):.5f} | {c.get('ridge',float('nan')):.5f} | {c.get('enet',float('nan')):.5f} | {best:.5f} |")

    # B) best cfg per window
    print("\n### B. best cfg by window (heavier vs lighter reg with more data?)\n")
    print("| window | enet cfg (a,l1) | lasso alpha | ridge alpha |")
    print("|---|---|---|---|")
    for w in windows:
        c = by.get((w, "all_buckets"), {})
        print(f"| {w//48}d | {c.get('enet_cfg')} | {c.get('lasso_cfg')} | {c.get('ridge_cfg')} |")

    # C) full grid: best-vs-HAR delta per (bucket, window)
    print("\n### C. full grid -- best all_buckets/bucket QLIKE improvement vs HAR-only, per window\n")
    hdr = "| bucket | " + " | ".join(f"{w//48}d" for w in windows) + " |"
    print(hdr)
    print("|" + "---|" * (len(windows) + 1))
    for bk in buckets:
        cells_row = []
        for w in windows:
            c = by.get((w, bk))
            h = har.get(w)
            if not c or h is None:
                cells_row.append("·")
                continue
            best = min(c.get("lasso", 9), c.get("ridge", 9), c.get("enet", 9))
            cells_row.append(f"{best-h:+.5f}")
        print(f"| {bk} | " + " | ".join(cells_row) + " |")

    print("\n(HAR-only reference QLIKE by window: " + ", ".join(f"{w//48}d={har[w]:.5f}" for w in windows if har.get(w)) + ")")

    json.dump({"windows": windows, "har": {str(w): har[w] for w in windows}, "cells": {f"{c['window']}_{c['bucket']}": c for c in cells}}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
