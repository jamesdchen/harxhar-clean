"""Stack the moments_vix per-bar arms into yhat tables, exactly as build_subsection_yhat does.

2026-10-06, user request: every per-bar linear model also on the return moments + the VIX level
alone (src.data.loading.SUBGROUPS["moments_vix"]: HAR + calendar backbone + the 7 ES return
moments + vix), beside baseline and all_features.  The arms ran locally with the de-dup fleet's
command (cluster/linear_dedup_pack.sh: specs/causal_tune_linear.py, 2000 sessions, global lags,
production HAR ladder, one thread) into the de-dup campaign's nested layout:
  results/linear_subsection_dedup/moments_vix/<bar>/<est>/tw2000/
      causal_tune_linear/<est>/moments_vix/results_<bar>.csv

Each estimator's available bars are stacked with build_subsection_yhat's own read_arm and
with_production_rv (the production rv_raw, the baseline gate).  Only the 16:00 bar was run: the
closing-strategy master table reads a table's 16:00 rows only (master_table_close.read_table_1600),
so the tables hold 16:00 rows only, unlike the 13-bar canonical yhat_sub_* tables.

GATE  each table's rows equal its arm CSVs (pred_adj -> yhat, true_raw / true_adj^2 -> baseline)
      bit for bit on the stamps read_arm keeps.

Also stacks moments_vol_vix (moments_vix + sumvolume) and lean_vol_vix (2026-10-06, user) the same way.

Writes results/spxw_pnl/yhat_sub_<ridge|lasso|enet>_<bucket>.parquet.
Run:  python experiments/build_moments_vix_yhat.py [bucket ...]   (default: moments_vix)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_subsection_yhat import BARS, OUT, TW, read_arm, with_production_rv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "results" / "linear_subsection_dedup"
BUCKETS = ("moments_vix", "moments_vol_vix", "lean_vol_vix")
ESTIMATORS = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}


def arm_path(bucket: str, bar: str, est: str) -> Path:
    return (
        LS
        / bucket
        / bar
        / est
        / f"tw{TW}"
        / "causal_tune_linear"
        / est
        / bucket
        / f"results_{bar}.csv"
    )


def main() -> None:
    buckets = sys.argv[1:] or ["moments_vix"]
    assert all(b in BUCKETS for b in buckets), buckets
    for bucket in buckets:
        build(bucket)


def build(bucket: str) -> None:
    for est, short in ESTIMATORS.items():
        paths = [
            arm_path(bucket, b, est) for b in BARS if arm_path(bucket, b, est).is_file()
        ]
        if not paths:
            print(f"skip {bucket} {short}: no bar arm on disk")
            continue
        name = f"yhat_sub_{short}_{bucket}.parquet"
        raw = (
            pd.concat([read_arm(p) for p in paths])
            .sort_values("t")
            .reset_index(drop=True)
        )
        assert not raw["t"].duplicated().any(), name
        tab = with_production_rv(raw, name)
        # GATE: the stacked rows are the arm CSVs' rows
        assert tab["t"].equals(raw["t"]), name
        d_y = float(np.max(np.abs(tab["yhat"].to_numpy() - raw["yhat"].to_numpy())))
        d_b = float(
            np.max(np.abs(tab["baseline"].to_numpy() - raw["baseline"].to_numpy()))
        )
        assert d_y == 0.0 and d_b == 0.0, (name, d_y, d_b)
        tab.to_parquet(OUT / name, index=False)
        bars = ",".join(
            p.name.removeprefix("results_").removesuffix(".csv") for p in paths
        )
        print(
            f"wrote {name}: {len(tab):,} rows ({bars}), {tab['t'].min().date()} .. "
            f"{tab['t'].max().date()}; GATE table = arm CSVs (max |dyhat| {d_y:.1e}, "
            f"max |dB| {d_b:.1e})"
        )


if __name__ == "__main__":
    main()
