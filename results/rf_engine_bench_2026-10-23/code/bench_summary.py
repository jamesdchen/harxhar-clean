"""Summary of the RF engine benchmark: timing by engine / set / window / node, determinism, pre-test fold-3 accuracy,
and the validation forecasts of B / B2048 / C against A.

usage: python bench_summary.py <bench out dir> <summary dir>
-> timing.csv, accuracy.csv, agreement.csv, summary.txt (markdown tables)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
recs, preds = [], {}
for p in sorted(src.glob("*.json")):
    r = json.loads(p.read_text())
    node = "xeon" if "Intel" in r["cpu"] else "epyc"
    f = pd.read_parquet(p.with_suffix(".parquet"))
    preds[(r["engine"], r["set"], r["window_days"], node)] = f
    for run in r["runs"]:
        recs.append(
            {
                "engine": r["engine"],
                "set": r["set"],
                "W": r["window_days"],
                "node": node,
                "cpu": r["cpu"],
                "host": r["host"],
                **{k: v for k, v in run.items() if k != "pred_md5"},
                "pred_md5": run["pred_md5"][:12],
                "same_twice": r["same_twice"],
                "same_1_vs_T": r.get("same_1_vs_T"),
                "max_abs_diff_1_vs_T": r.get("max_abs_diff_1_vs_T"),
                "sklearnex_log": " | ".join(r.get("sklearnex_log", []))[:300],
                "params": json.dumps(
                    r["config"]["params"], sort_keys=True, default=str
                ),
            }
        )
T = pd.DataFrame(recs)
T.to_csv(dst / "runs.csv", index=False)
t8 = T[T["run"] == "T"]
tim = t8.pivot_table(
    index=["engine", "set"], columns=["node", "W"], values="fit_sec"
).round(2)
tim.to_csv(dst / "timing.csv")
acc = t8[
    [
        "engine",
        "set",
        "W",
        "node",
        "mse_adj",
        "qlike",
        "mean_leaves",
        "same_twice",
        "same_1_vs_T",
    ]
].sort_values(["set", "W", "engine", "node"])
acc.to_csv(dst / "accuracy.csv", index=False)
ag = []
for (e, s, W, node), f in preds.items():
    ref = preds.get(("A", s, W, node))
    if ref is None:
        continue
    a, b = ref["pred_T"].to_numpy(), f["pred_T"].to_numpy()
    rel = np.abs(b - a) / np.abs(a)
    ag.append(
        {
            "engine": e,
            "set": s,
            "W": W,
            "node": node,
            "corr_vs_A": float(np.corrcoef(a, b)[0, 1]),
            "median_rel_diff_vs_A": float(np.median(rel)),
            "p90_rel_diff_vs_A": float(np.quantile(rel, 0.9)),
            "max_rel_diff_vs_A": float(rel.max()),
            "bit_equal_vs_A": bool(np.array_equal(a, b)),
        }
    )
for (e, s, W, node), f in preds.items():  # the same engine across the two CPU types
    if node == "epyc" and (e, s, W, "xeon") in preds:
        g = preds[(e, s, W, "xeon")]
        ag.append(
            {
                "engine": e,
                "set": s,
                "W": W,
                "node": "epyc vs xeon",
                "bit_equal_vs_A": bool(
                    np.array_equal(f["pred_T"].to_numpy(), g["pred_T"].to_numpy())
                ),
                "max_rel_diff_vs_A": float(
                    np.max(np.abs(f["pred_T"] - g["pred_T"]) / np.abs(f["pred_T"]))
                ),
            }
        )
G = pd.DataFrame(ag).sort_values(["set", "W", "engine", "node"])
G.to_csv(dst / "agreement.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
txt = [
    "## Fit seconds per refit, 8 threads (fold-3 fit: W x 13 rows)",
    tim.to_string(),
    "",
    "## Pre-test fold 3 (validation 2013, 3,250 bars), 8 threads",
    acc.to_string(index=False),
    "",
    "## Validation forecasts against engine A (same node), and each engine EPYC vs Xeon",
    G.to_string(index=False),
    "",
    "## Thread runs",
    T[
        [
            "engine",
            "set",
            "W",
            "node",
            "run",
            "threads",
            "fit_sec",
            "fit_cpu_sec",
            "pred_md5",
        ]
    ].to_string(index=False),
    "",
    "## sklearnex dispatch log",
    "\n".join(sorted(set(T.loc[T["engine"].str.startswith("B"), "sklearnex_log"]))),
]
(dst / "summary.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
print("\n".join(txt))
