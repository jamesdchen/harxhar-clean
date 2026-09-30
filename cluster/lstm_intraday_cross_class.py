"""Record, don't judge: the mixed-CPU-class first attempt of the intraday LSTM vs the pinned (epyc-7513) run.

The first stage 1 / stage 2 ran on whatever main nodes were free: the canary's tuning point (live_feasible
row 0) on an Intel Xeon Silver 4116, the other 17 on AMD EPYC 7513; the refit chunks on EPYC 7513 / 7542
and Intel nodes.  The spec's tuning-cache fingerprint (sha256 of the window's bar inputs) refused the 5
chunks whose tuning point had been computed on the other vendor: the executor's bar-level inputs differ in
their last bits between Intel and AMD.  Everything was re-run pinned to epyc-7513 (the campaign's result);
the mixed attempt was moved aside with the suffix _mixedclass_20260930.  This script compares the two, on
the cluster, and writes gates/cross_class_mixed_vs_pinned.csv:

  tuning points  per (bucket, row): the CPU of each run, both rules' picks equal, every candidate x seed's
                 early-stopped epochs equal, max |validation MSE difference|
  refit chunks   per mixed chunk that finished: its CPU, its tuning point's CPU, rows, and against the
                 pinned merged arm on the same stamps: pred_adj bit-identical, max |difference|

Usage (CARC, deployment root): python cluster/lstm_intraday_cross_class.py [--root DIR] [--tag TAG]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LINE = re.compile(
    r"\(([^\n]*?)\), pool of \d+: (\w+) (tune|all) bar1600 chunk (\d+) rows \[(\d+), (\d+)\)"
)


def cpu_by_task(pattern: str) -> dict[tuple[str, str, int], str]:
    out = {}
    for f in glob.glob(pattern):
        text = Path(f).read_text(encoding="utf-8", errors="replace")
        if "already DONE" in text:  # a task that found its line done ran nothing
            continue
        m = LINE.search(text)
        if m:
            cpu, bucket, stage, chunk = (
                m.group(1),
                m.group(2),
                m.group(3),
                int(m.group(4)),
            )
            out[(bucket, stage, chunk)] = cpu
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_lstm_intraday")
    ap.add_argument("--tag", default="mixedclass_20260930")
    ap.add_argument("--mixed-jobs", default="12481893,12482098,12483604")
    ap.add_argument("--pinned-jobs", default="12484206,12484285")
    a = ap.parse_args()
    root = Path(a.root)
    mixed_cpu: dict[tuple[str, str, int], str] = {}
    pinned_cpu: dict[tuple[str, str, int], str] = {}
    for j in a.mixed_jobs.split(","):
        mixed_cpu |= cpu_by_task(f"logs/lstmi_*.{j}.*out")
    for j in a.pinned_jobs.split(","):
        pinned_cpu |= cpu_by_task(f"logs/lstmi_*.{j}.*out")
    rows = []
    for f in sorted((root / f"tune_cache_{a.tag}").glob("*/tune_R*.npz")):
        bucket, R = f.parent.name, int(f.stem.removeprefix("tune_R"))
        g = root / "tune_cache" / bucket / f.name
        with np.load(f) as zm, np.load(g) as zp:
            sm, sp = json.loads(str(zm["scalars"])), json.loads(str(zp["scalars"]))
            fm, fp = (
                json.loads(str(zm["fingerprint"])),
                json.loads(str(zp["fingerprint"])),
            )
            rows.append(
                {
                    "what": "tuning point",
                    "bucket": bucket,
                    "row": R,
                    "cpu_mixed": mixed_cpu.get((bucket, "tune", R), ""),
                    "cpu_pinned": pinned_cpu.get((bucket, "tune", R), ""),
                    "same_bar_inputs_sha": fm["sha_P"] == fp["sha_P"],
                    "same_target_sha": fm["sha_y"] == fp["sha_y"],
                    "same_picks": (sm["pick_mse"], sm["pick_qlike"])
                    == (sp["pick_mse"], sp["pick_qlike"]),
                    "same_epochs": bool(np.array_equal(zm["epochs"], zp["epochs"])),
                    "max_abs_diff": float(
                        np.max(np.abs(zm["val_mse"] - zp["val_mse"]))
                    ),
                    "bit_identical": bool(
                        np.array_equal(zm["val_mse"], zp["val_mse"])
                        and np.array_equal(zm["val_qlike"], zp["val_qlike"])
                    ),
                }
            )
    for b in ("live_feasible", "all_features", "baseline"):
        arm = root / b / "bar1600" / "lstm_intraday" / "tw2000"
        merged = (
            arm
            / "causal_tune_lstm_intraday"
            / "lstm_intraday"
            / b
            / "results_bar1600.csv"
        )
        if not merged.is_file():
            continue
        pin = pd.read_csv(merged, float_precision="round_trip").set_index("date")
        for d in sorted((arm / f"chunks_{a.tag}").glob("c*")):
            res = (
                d
                / "causal_tune_lstm_intraday"
                / "lstm_intraday"
                / b
                / "results_bar1600.csv"
            )
            if not (d / "DONE").is_file() or not res.is_file():
                continue
            k = int(d.name[1:])
            r = pd.read_csv(res, float_precision="round_trip").set_index("date")
            j = r[["pred_adj"]].join(pin[["pred_adj"]], how="inner", rsuffix="_pinned")
            diff = (j["pred_adj"] - j["pred_adj_pinned"]).abs()
            rows.append(
                {
                    "what": "refit chunk",
                    "bucket": b,
                    "row": k,
                    "cpu_mixed": mixed_cpu.get((b, "all", k), ""),
                    "cpu_pinned": "epyc-7513 class",
                    "n_rows": len(j),
                    "first": j.index.min(),
                    "last": j.index.max(),
                    "max_abs_diff": float(diff.max()),
                    "median_abs_diff": float(diff.median()),
                    "bit_identical": bool((diff == 0).all()),
                }
            )
    out = pd.DataFrame(rows)
    (root / "gates").mkdir(exist_ok=True)
    out.to_csv(root / "gates" / "cross_class_mixed_vs_pinned.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
