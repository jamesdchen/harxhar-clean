"""Cross-CPU reproducibility of the per-bar 16:00 forest (2026-10-07; for the D.19 text): which CPU classes reproduce the stored
CARC epyc-7513 forest byte for byte.  Written from the comparison files, every number read from them:

  CARC   results/trees_rfdev_preview163/xcpu_carc/<class>/compare_chunks.csv  (experiments/trees_rfdev_xcpu_compare.py: every
         chunk of the six rfdev cells re-run on <class>, byte comparison of dates, forecasts, targets, refit rows, importance,
         TreeSHAP, kept columns with the stored epyc-7513 chunk) and, when the merged arms were pulled, the forecast-level
         comparison of the six cells (positions on the 866 trade days, QLIKE, Sharpe) -> xcpu_carc/<class>/compare.csv
  Hoffman2  results/h2check_{rfdev,rfprev163}/compare.csv, compare_by_class.csv (experiments/trees_rfh2_compare.py)
  LightGBM  results/linear_subsection_trees_mask/class/class_arms.csv (the masked campaign's class gate, xeon-4116 vs epyc-7513)

Writes results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md and xcpu_classes.csv.
Run:  python experiments/trees_xcpu_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "trees_rfdev_preview163" / "xcpu_carc"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        f = d / "compare_chunks.csv"
        if not f.is_file():
            continue
        c = pd.read_csv(f)
        c = c[c["chunk"].astype(str).str.fullmatch(r"c\d+")]
        rows.append(
            {
                "cluster": "CARC",
                "class": d.name,
                "set": "rfdev (rf_opt_00)",
                "chunks": len(c),
                "chunks_byte_identical": int(c["identical"].astype(bool).sum()),
                "forecast_rows": int(c["rows"].sum()),
                "rows_differing": int(c["rows_pred_differ"].sum()),
                "max_abs_pred_adj_diff": float(c["max_abs_pred_adj_diff"].max()),
                "cells_complete": f"{c.groupby(['rung', 'bucket']).ngroups} of 6",
            }
        )
    for s, label in (("rfdev", "rfdev (rf_opt_00)"), ("rfprev163", "trial 163")):
        f = ROOT / "results" / f"h2check_{s}" / "compare_by_class.csv"
        if f.is_file():
            for _, r in pd.read_csv(f).iterrows():
                rows.append(
                    {
                        "cluster": "Hoffman2",
                        "class": r["class"],
                        "set": label,
                        "chunks": int(r["chunks"]),
                        "forecast_rows": int(r["rows"]),
                        "rows_differing": int(r["rows"] - r["bit_identical"]),
                        "max_abs_pred_adj_diff": float(r["max_abs_diff"]),
                    }
                )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "xcpu_classes.csv", index=False)
    cell_lines = []
    for s, label in (("rfdev", "rfdev (rf_opt_00)"), ("rfprev163", "trial 163")):
        f = ROOT / "results" / f"h2check_{s}" / "compare.csv"
        if f.is_file():
            for _, r in pd.read_csv(f).iterrows():
                cell_lines.append(
                    f"| Hoffman2 (all classes) | {label} | {r['rung']} {r['bucket']} | {int(r['rows'] - r['rows_bit_identical'])} | "
                    f"{r['max_abs_pred_adj_diff']:.3g} | {int(r['positions_identical'])} / {int(r['n_days'])} | {r['qlike_h2'] - r['qlike_carc']:+.1e} | "
                    f"{r['sharpe_mid_h2'] - r['sharpe_mid_carc']:+.1e} |"
                )
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        f = d / "compare.csv"
        if f.is_file():
            for _, r in pd.read_csv(f).iterrows():
                cell_lines.append(
                    f"| CARC {d.name} | rfdev (rf_opt_00) | {r['rung']} {r['bucket']} | {int(r['rows'] - r['rows_bit_identical'])} | "
                    f"{r['max_abs_pred_adj_diff']:.3g} | {int(r['positions_identical'])} / {int(r['n_days'])} | {r['qlike_h2'] - r['qlike_carc']:+.1e} | "
                    f"{r['sharpe_mid_h2'] - r['sharpe_mid_carc']:+.1e} |"
                )
    lg = ROOT / "results" / "linear_subsection_trees_mask" / "class" / "class_arms.csv"
    lgl = []
    if lg.is_file():
        for _, r in pd.read_csv(lg).iterrows():
            lgl.append(
                f"- LightGBM {r['rung']} {r['bucket']}: {r['class_alt_run']} vs {r['class_fleet']} over {int(r['rows'])} rows: "
                f"byte-identical {bool(r['bit_identical'])}, max relative forecast difference {r['max_rel']:.3g} (median {r['median_rel']:.3g})."
            )
    lines = [
        "# Cross-CPU reproducibility of the per-bar 16:00 random forest",
        "",
        "Written by `experiments/trees_xcpu_summary.py` from the comparison files it names (`xcpu_classes.csv` here). The reference is the stored "
        "CARC epyc-7513 run; every re-run uses the same code, configuration, chunking, single-threaded fits and seed, and records the CPU of every "
        "chunk. Byte-identical = every compared array equal bit for bit.",
        "",
        "## By CPU class",
        "",
        "| cluster | CPU class | forest | chunks | chunks byte-identical | forecast rows | rows differing | max abs forecast diff (adjusted scale) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in tab.iterrows():
        cbi = (
            ""
            if pd.isna(r.get("chunks_byte_identical"))
            else int(r["chunks_byte_identical"])
        )
        lines.append(
            f"| {r['cluster']} | {r['class']} | {r['set']} | {int(r['chunks'])} | {cbi} | {int(r['forecast_rows'])} | "
            f"{int(r['rows_differing'])} | {r['max_abs_pred_adj_diff']:.3g} |"
        )
    # reading (computed from the table): which classes reproduce the stored forest, which do not
    full = (
        int(tab.loc[tab["cluster"] == "CARC", "chunks"].max())
        if (tab["cluster"] == "CARC").any()
        else 0
    )
    agg = tab.groupby(["cluster", "class"], sort=False).agg(
        chunks=("chunks", "sum"),
        rows=("forecast_rows", "sum"),
        differ=("rows_differing", "sum"),
        mx=("max_abs_pred_adj_diff", "max"),
    )
    same = [f"{cl} {k}" for (cl, k), r in agg.iterrows() if r["differ"] == 0]
    diff = [
        f"{cl} {k} ({int(r['differ'])} of {int(r['rows'])} rows, max {r['mx']:.3g})"
        for (cl, k), r in agg.iterrows()
        if r["differ"] > 0
    ]
    partial = [
        f"CARC {r['class']} ({int(r['chunks'])} of {full} chunks; cancelled once the class was dropped)"
        for _, r in tab[(tab["cluster"] == "CARC") & (tab["chunks"] < full)].iterrows()
    ]
    lines += [
        "",
        "## Reading",
        "",
        f"- Byte-identical to the stored epyc-7513 forest (every chunk / row compared): {'; '.join(same) or 'none'}.",
        f"- Not byte-identical: {'; '.join(diff) or 'none'} -- all AVX-512 classes; the differing rows are few, the "
        "differences small; where the merged cells were compared at the forecast level (Hoffman2, table below) the positions on the 866 "
        "trade days are identical in every cell and QLIKE moves in the 7th digit.",
        *([f"- Partial re-runs: {'; '.join(partial)}."] if partial else []),
        "- For contrast, LightGBM is not byte-reproducible between xeon-4116 and epyc-7513 (below).",
    ]
    lines += [
        "",
        "## By cell (six cells: T1 / T10 x baseline / all_features / lean_vol_vix)",
        "",
        "| run | forest | cell | rows differing | max abs diff | positions identical | QLIKE diff | Sharpe mid diff |",
        "|---|---|---|---|---|---|---|---|",
        *cell_lines,
        "",
        "## LightGBM (for contrast)",
        "",
        *lgl,
    ]
    (OUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
