"""Chunk-level cross-CPU check of the dev forest on CARC (2026-10-07): every finished chunk of a re-run of the six rfdev
cells on another CPU class (results/trees_rfdev_xcpu/<class>/{t1,t10}/..., the stored run's code, chunking, configuration,
threads and seed) compared BYTE FOR BYTE with the same chunk of the stored epyc-7513 run
(results/linear_subsection_trees_mask_rfdev/...): date, pred_adj, true_adj, true_raw, refit_row, importance, shap, kept_n,
kept; plus the max |pred_adj difference| and the number of rows that differ.  The node of every chunk (NODE) is recorded.

Writes <root>/compare_chunks.csv and prints one summary line per cell (rung x bucket).
Run (on CARC, in the rfdev deployment root):
    python experiments/trees_rfdev_xcpu_compare.py --root results/trees_rfdev_xcpu/epyc-7542
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

KEYS = (
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "refit_row",
    "importance",
    "shap",
    "kept_n",
    "kept",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref", default="results/linear_subsection_trees_mask_rfdev")
    a = ap.parse_args()
    root, ref = Path(a.root), Path(a.ref)
    rows = []
    for done in sorted(root.glob("t*/*/bar1600/rf/tw2000/chunks/c*/DONE")):
        d = done.parent
        if not d.name[1:].isdigit():
            continue  # c0_alone and the like
        rel = d.relative_to(root)
        rung, bucket = rel.parts[0], rel.parts[1]
        npz = d / "causal_tune_trees" / "rf" / bucket / "trees_bar1600.npz"
        rnpz = ref / rel / "causal_tune_trees" / "rf" / bucket / "trees_bar1600.npz"
        node = (
            (d / "NODE").read_text(encoding="utf-8").replace("\n", " ").strip()
            if (d / "NODE").is_file()
            else ""
        )
        if not rnpz.is_file():
            rows.append(
                {
                    "rung": rung,
                    "bucket": bucket,
                    "chunk": d.name,
                    "identical": False,
                    "why": "no stored chunk",
                    "node": node,
                }
            )
            continue
        za, zb = np.load(npz), np.load(rnpz)
        same = {
            k: (
                k in za.files
                and k in zb.files
                and za[k].dtype == zb[k].dtype
                and za[k].shape == zb[k].shape
                and np.ascontiguousarray(za[k]).tobytes()
                == np.ascontiguousarray(zb[k]).tobytes()
            )
            for k in KEYS
        }
        dp = (
            np.abs(za["pred_adj"] - zb["pred_adj"])
            if za["pred_adj"].shape == zb["pred_adj"].shape
            else np.array([np.inf])
        )
        rows.append(
            {
                "rung": rung,
                "bucket": bucket,
                "chunk": d.name,
                "rows": len(za["pred_adj"]),
                "identical": all(same.values()),
                "keys_differing": " ".join(k for k, v in same.items() if not v),
                "rows_pred_differ": int((dp > 0).sum()),
                "max_abs_pred_adj_diff": float(dp.max()),
                "node": node,
            }
        )
    out = root / "compare_chunks.csv"
    cols = [
        "rung",
        "bucket",
        "chunk",
        "rows",
        "identical",
        "keys_differing",
        "rows_pred_differ",
        "max_abs_pred_adj_diff",
        "node",
        "why",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    cells: dict = {}
    for r in rows:
        c = cells.setdefault((r["rung"], r["bucket"]), [0, 0, 0, 0.0])
        c[0] += 1
        c[1] += int(bool(r["identical"]))
        c[2] += int(r.get("rows_pred_differ", 0) or 0)
        c[3] = max(c[3], float(r.get("max_abs_pred_adj_diff", 0.0) or 0.0))
    for (rung, b), (n, ok, nd, mx) in sorted(cells.items()):
        print(
            f"{root.name} {rung} {b}: {ok} of {n} chunks byte-identical; {nd} forecast rows differ; max |pred_adj diff| {mx:.3g}"
        )
    print(f"wrote {out} ({len(rows)} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
