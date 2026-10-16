"""Cross-CPU check of the xgbtop fixed-settings XGBoost (2026-10-07): every chunk of a cell re-run on another CARC CPU
class (epyc-7542, xeon-4116) against the same chunk of the epyc-7513 fleet, byte for byte.

Both roots hold the layout of cluster/slurm/trees_xgbtop_pack.sbatch (<root>/<rung>/<bucket>/bar1600/xgb/tw2000/chunks/
c<k>/causal_tune_trees/xgb/<bucket>/trees_bar1600.npz, DONE, NODE).  Every chunk of --other (DONE) is compared with
--ref's chunk by experiments/trees_momvix_gate.py's compare() (unedited: date, pred_adj, true_adj, true_raw, baseline,
e2_adj, TreeSHAP, refit rows, importance, kept columns, feature names, meta params / cadence / threads / window_mask /
versions), and the results CSVs byte for byte.  Writes <out>/xcpu_<label>.csv (one row per chunk and key) and
<out>/XCPU_MATCH_<label> when every chunk matched (XCPU_DIFFERS_<label> otherwise).  Exit 0 iff matched.

Run:  python experiments/trees_xgbtop_xcpu_compare.py --ref results/linear_subsection_trees_mask_xgbtop
          --other results/linear_subsection_trees_mask_xgbtop_xcpu/epyc-7542 --label epyc-7542 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import trees_momvix_gate as g  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref, other, out = Path(a.ref), Path(a.other), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (out / f"XCPU_MATCH_{a.label}", out / f"XCPU_DIFFERS_{a.label}"):
        f.unlink(missing_ok=True)
    rows, n_chunks, bad_chunks = [], 0, []
    for done in sorted(other.glob("*/*/bar1600/xgb/tw2000/chunks/c*/DONE")):
        d = done.parent
        rel = d.relative_to(other)
        rung, bucket = rel.parts[0], rel.parts[1]  # noqa: F841
        r = ref / rel
        node = (
            (d / "NODE").read_text(encoding="utf-8").replace("\n", " | ")
            if (d / "NODE").is_file()
            else ""
        )
        n_chunks += 1
        if not (r / "DONE").is_file():
            rows.append(
                {
                    "chunk": str(rel),
                    "key": "reference chunk",
                    "identical": False,
                    "detail": f"{r} not DONE",
                    "node": node,
                }
            )
            bad_chunks.append(str(rel))
            continue
        inner = Path("causal_tune_trees") / "xgb" / bucket
        got = g.compare(
            d / inner / "trees_bar1600.npz", r / inner / "trees_bar1600.npz", None
        )
        same_csv = (d / inner / "results_bar1600.csv").read_bytes() == (
            r / inner / "results_bar1600.csv"
        ).read_bytes()
        got.append(
            {"key": "results_bar1600.csv bytes", "identical": same_csv, "detail": ""}
        )
        for x in got:
            rows.append({"chunk": str(rel), **x, "node": node})
        if not all(x["identical"] for x in got):
            bad_chunks.append(str(rel))
        print(
            f"  {rel}: {'IDENTICAL' if all(x['identical'] for x in got) else 'DIFFERS'} ({node[:80]})",
            flush=True,
        )
    p = out / f"xcpu_{a.label}.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["chunk", "key", "identical", "detail", "node"]
        )
        w.writeheader()
        w.writerows(rows)
    ok = n_chunks > 0 and not bad_chunks
    flag = out / f"XCPU_{'MATCH' if ok else 'DIFFERS'}_{a.label}"
    flag.write_text(
        f"{n_chunks} chunks compared, {len(bad_chunks)} differ: {bad_chunks}\n",
        encoding="utf-8",
    )
    print(
        f"{a.label}: {n_chunks} chunks compared with {ref}, {len(bad_chunks)} differ; wrote {p} and {flag.name}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
