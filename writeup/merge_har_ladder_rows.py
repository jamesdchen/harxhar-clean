"""Append the har_base_sweep 21-arm battery to the master extended-battery CSV.

Rows come VERBATIM from the run-level reducer output
(_aggregated/har_base_sweep-53c27e42_metrics_table.csv, computed cluster-side
by specs/reduce_har_base_sweep.py from the pooled per-bar series); this script
only maps columns into the master schema — no metric is recomputed here.
Idempotent: existing har-ladder rows are replaced, never duplicated.
"""
import pathlib

import pandas as pd

MASTER = pathlib.Path("writeup/metrics_table_causal_tune_plus_spectral.csv")
NEW = pathlib.Path("_aggregated/har_base_sweep-53c27e42_metrics_table.csv")

master = pd.read_csv(MASTER)
new = pd.read_csv(NEW)

rows = pd.DataFrame(
    {
        "estimator": "ols_har",
        "bucket": "baseline",
        **{c: new[c] for c in [
            "qlike", "mse", "rmse", "mae", "hmse", "hmae", "oos_r2",
            "mz_beta", "mz_r2", "n", "incumbent_qlike", "dm", "dm_p",
            "dm_mean_diff", "dm_better",
        ]},
        "family": "har-ladder",
        "arm_tag": new["arm"],
    }
)

master = master[master.get("family") != "har-ladder"]
merged = pd.concat([master, rows], ignore_index=True)
merged.to_csv(MASTER, index=False)
print(f"master now {len(merged)} arms (+{len(rows)} har-ladder)")
