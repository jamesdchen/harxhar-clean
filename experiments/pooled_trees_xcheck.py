"""Cross-class / thread-count check of the pooled forest (2026-10-07): the first rows of bank chunk 0 computed on other
CPU classes or thread counts (pieces chunk_000_<a>-<b>.npz under <cand root>/chunks/pooltree_rf_<set>/pieces/) must equal
the reference rows -- the CARC epyc-7513 8-thread canary (<ref root>/pooltree_rf_<set>/chunk_000.npz, a rows-limited
chunk-0 run) -- byte for byte on row_id, t, y_fit, yhat, rv_raw and baseline.

Run:  python experiments/pooled_trees_xcheck.py --ref results/pooled_trees_2026-10-07/canary2
          --cand results/pooltree_xcheck/epyc-7542 --cand results/pooltree_xcheck/h2_intel-E5-2670 ... [--out <csv>]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

KEYS = ("row_id", "t", "y_fit", "yhat", "rv_raw", "baseline")
SETS = ("baseline", "lean_vol_vix", "all_features")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows, bad = [], 0
    for s in SETS:
        ref = Path(a.ref) / f"pooltree_rf_{s}" / "chunk_000.npz"
        if not ref.is_file():
            rows.append(
                {
                    "set": s,
                    "cand": "-",
                    "rows": "",
                    "ok": False,
                    "detail": f"no reference {ref}",
                }
            )
            bad += 1
            continue
        zr = np.load(ref, allow_pickle=True)
        for c in a.cand:
            for f in sorted(
                (Path(c) / "chunks" / f"pooltree_rf_{s}" / "pieces").glob(
                    "chunk_000_*.npz"
                )
            ):
                m = re.match(r"chunk_000_(\d{4})-(\d{4})\.npz$", f.name)
                if not m or ".part" in f.name:
                    continue
                r0, r1 = int(m.group(1)), int(m.group(2))
                zc = np.load(f, allow_pickle=True)
                if r1 > len(zr["yhat"]):
                    rows.append(
                        {
                            "set": s,
                            "cand": c,
                            "rows": f"{r0}-{r1}",
                            "ok": False,
                            "detail": "reference too short",
                        }
                    )
                    bad += 1
                    continue
                diffs = [
                    k
                    for k in KEYS
                    if np.asarray(zc[k]).tobytes() != np.asarray(zr[k])[r0:r1].tobytes()
                ]
                node = f.with_name(f.stem + ".node")
                cpu = (
                    next(
                        (
                            ln[4:].strip()
                            for ln in node.read_text().splitlines()
                            if ln.startswith("cpu ")
                        ),
                        "?",
                    )
                    if node.is_file()
                    else "?"
                )
                ok = not diffs
                bad += not ok
                rows.append(
                    {
                        "set": s,
                        "cand": c,
                        "rows": f"{r0}-{r1}",
                        "ok": ok,
                        "detail": ("identical" if ok else f"differs in {diffs}")
                        + f" | {cpu}",
                    }
                )
                print(
                    f"{s:13s} {c}: rows {r0}-{r1} {'IDENTICAL' if ok else 'DIFFERS ' + str(diffs)} ({cpu})"
                )
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["set", "cand", "rows", "ok", "detail"],
                lineterminator="\n",
            )
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows) - bad} of {len(rows)} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
