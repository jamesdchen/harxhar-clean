"""Stack the POOLED moments_vix arms into yhat_pool_<est>_moments_vix tables, as build_subsection_yhat does.

2026-10-06, user request: every model also on the return moments + the VIX level alone
(src.data.loading.SUBGROUPS["moments_vix"]: the 7 ES return moments + vix, on the HAR + calendar
backbone).  This is the POOLED twin of that request: one coefficient vector over all 48 half-hour
bars (segment ``none``; the backbone keeps its session-edge interactions), 2000-session window
(96,000 bars), coefficients re-solved every bar, penalty re-chosen every 250 solves -- the stored
pooled arms' spec and settings, run on Hoffman2 whole-series on the CPU architectures of the stored
baseline / all_features pooled arms (cluster/submit_pool_momvix.sh; gate and run record in
results/linear_subsection_pool_momvix/hoffman2/).  A CARC replicate (cluster/slurm/submit_pool_momvix.sh,
epyc-7513, chunked) agrees for ridge to 1e-10 but not for lasso / elastic net (different penalty
paths): results/linear_subsection_pool_momvix/cross_cluster.csv.  To build from the CARC copy instead,
copy its results_none_rth.csv files (results/linear_subsection_pool_momvix/carc/...) into the flat
layout below and re-run.

Input (flat layout of the stored pooled arms, a new bucket subdirectory):
  results/linear_subsection/arms_hoffman2/moments_vix/<est>/tw2000/results_none_rth.csv
  (the 13 regular-hours bars 10:00 .. 16:00 of the whole-series results.csv, the filter the cluster
  gate verified against the stored baseline / all_features files)
Output:
  results/spxw_pnl/yhat_pool_<ridge|lasso|enet>_moments_vix.parquet

Built with build_subsection_yhat's own read_arm and with_production_rv (its gate: on every stamp the
baseline profile B agrees with the production table to 1e-9; the production rv_raw is carried).

GATES  (1) same builder: the same two functions applied to the stored baseline and all_features
       pooled files reproduce the stored yhat_pool_<est>_<bucket>.parquet tables (every column, bit
       for bit) -- 6 tables;  (2) each new table's stamps equal the stored pooled tables' stamps;
       (3) each new table's yhat equals its arm file's pred_adj bit for bit.
Refuses to write unless results/close_master_table/PLAIN_DONE exists (the plain master table is
final), unless --check-only (gates 1-3, nothing written).

Run:  python experiments/build_pool_moments_vix_yhat.py [--check-only] [--bucket moments_vol_vix]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_subsection_yhat import (  # noqa: E402
    ARMS,
    ESTIMATORS,
    OUT,
    TW,
    read_arm,
    with_production_rv,
)

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "moments_vix"  # --bucket moments_vol_vix (2026-10-06: + sumvolume) runs the same path
STORED_BUCKETS = ("baseline", "all_features")
PLAIN_DONE = ROOT / "results" / "close_master_table" / "PLAIN_DONE"


def rth_file(bucket: str, est: str) -> Path:
    return ARMS / bucket / est / f"tw{TW}" / "results_none_rth.csv"


def build(bucket: str, est: str) -> pd.DataFrame:
    """build_subsection_yhat.main's pooled branch, line for line."""
    tab = read_arm(rth_file(bucket, est)).sort_values("t").reset_index(drop=True)
    assert not tab["t"].duplicated().any()
    return with_production_rv(tab, f"yhat_pool_{ESTIMATORS[est]}_{bucket}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument(
        "--bucket",
        default=BUCKET,
        choices=("moments_vix", "moments_vol_vix", "lean_vol_vix"),
    )
    a = ap.parse_args()
    bucket_new = a.bucket
    if not a.check_only and not PLAIN_DONE.is_file():
        sys.exit(
            f"{PLAIN_DONE.relative_to(ROOT)} not found: the plain master table is not final"
        )

    stamps = None
    for (
        bucket
    ) in STORED_BUCKETS:  # GATE 1: the builder reproduces the stored pooled tables
        for est, short in ESTIMATORS.items():
            new = build(bucket, est)
            old = pd.read_parquet(OUT / f"yhat_pool_{short}_{bucket}.parquet")
            assert list(new.columns) == list(old.columns), (bucket, est)
            same = len(new) == len(old) and all(
                np.array_equal(new[c].to_numpy(), old[c].to_numpy())
                for c in new.columns
            )
            assert same, f"GATE 1 failed: yhat_pool_{short}_{bucket}"
            print(
                f"GATE 1  yhat_pool_{short}_{bucket}: rebuilt = stored, {len(new):,} rows, bit for bit"
            )
            stamps = old["t"] if stamps is None else stamps
            assert old["t"].equals(stamps), "stored pooled tables differ in stamps"

    for est, short in ESTIMATORS.items():
        name = f"yhat_pool_{short}_{bucket_new}.parquet"
        tab = build(bucket_new, est)
        assert tab["t"].equals(stamps), f"GATE 2 failed: {name} stamps"
        arm = (
            read_arm(rth_file(bucket_new, est)).sort_values("t").reset_index(drop=True)
        )
        d = float(np.max(np.abs(tab["yhat"].to_numpy() - arm["yhat"].to_numpy())))
        assert d == 0.0, f"GATE 3 failed: {name} {d}"
        print(
            f"GATE 2/3  {name}: stamps = the stored pooled tables' ({len(tab):,}), yhat = arm pred_adj "
            f"bit for bit; {tab['t'].dt.normalize().nunique():,} sessions, "
            f"{tab['t'].min().date()} .. {tab['t'].max().date()}"
        )
        if not a.check_only:
            tab.to_parquet(OUT / name, index=False)
            print(f"wrote {name}")


if __name__ == "__main__":
    main()
