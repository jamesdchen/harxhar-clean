"""Tables of the trees' pre-sample window check from the per-task json files (fold_fit.py).

usage: python aggregate.py <out_dir_with_json> <dest_prefix>
-> <dest_prefix>_per_fold.csv (one row a fit) and <dest_prefix>_tables.csv (est, set, criterion, folds, window,
value, pct_change_vs_previous_window), and a printout.  Folds pooled: MSE = mean of the fold means (equal fold
sizes, the linear criterion); QLIKE = mean over the pooled rows with RV > 0 (n-weighted fold means).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src, dest = Path(sys.argv[1]), sys.argv[2]
recs = [json.loads(p.read_text()) for p in sorted(src.glob("w*_f*_*.json"))]
pf = pd.DataFrame(recs)[["window_days", "fold", "est", "predictor_set", "n_columns", "fit", "val", "val_first",
                         "val_last", "n_val", "mse_adj", "qlike", "n_qlike", "n_rv_nonpos", "fit_wall_sec",
                         "fit_cpu_sec", "threads", "host", "cpu", "config", "config_sha", "slurm_job", "slurm_task"]]
pf = pf.sort_values(["est", "predictor_set", "window_days", "fold"], ignore_index=True)
pf.to_csv(f"{dest}_per_fold.csv", index=False)
print(f"{len(pf)} fits")
AGG = {"fold3": [3], "folds1-3": [1, 2, 3], "folds0-3": [0, 1, 2, 3]}
rows = []
for (est, s), g in pf.groupby(["est", "predictor_set"], sort=False):
    for agg, folds in AGG.items():
        prev = {}
        for W in (1000, 1500, 2000, 2500, 3000):
            h = g[(g["window_days"] == W) & g["fold"].isin(folds)]
            if len(h) != len(folds):
                continue
            assert (h["n_val"] == 3250).all()
            vals = dict(mse_adj=float(h["mse_adj"].mean()),
                        qlike=float((h["qlike"] * h["n_qlike"]).sum() / h["n_qlike"].sum()))
            for crit, v in vals.items():
                p = prev.get(crit)
                rows.append(dict(est=est, predictor_set=s, criterion=crit, folds=agg, window_days=W, value=v,
                                 pct_change_vs_previous_window=np.nan if p is None else 100 * (v / p - 1)))
                prev[crit] = v
tab = pd.DataFrame(rows)
tab.to_csv(f"{dest}_tables.csv", index=False)
pd.set_option("display.width", 220)
for crit in ("mse_adj", "qlike"):
    for agg in AGG:
        t = tab[(tab["criterion"] == crit) & (tab["folds"] == agg)]
        v = t.pivot_table(index=["est", "predictor_set"], columns="window_days", values="value", sort=False)
        c = t.pivot_table(index=["est", "predictor_set"], columns="window_days",
                          values="pct_change_vs_previous_window", sort=False)
        print(f"\n== {crit}, {agg}")
        print(v.to_string(float_format=lambda x: f"{x:.6f}"))
        print("  % change vs the previous window")
        print(c.to_string(float_format=lambda x: f"{x:+.3f}"))
print("\nfit+predict wall seconds (8 threads), by est / set / window (mean over folds)")
print(pf.pivot_table(index=["est", "predictor_set"], columns="window_days", values="fit_wall_sec",
                     aggfunc="mean", sort=False).to_string(float_format=lambda x: f"{x:.1f}"))
