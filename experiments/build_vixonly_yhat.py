"""Stack the VIX-bucket per-bar arms into yhat tables, exactly as build_subsection_yhat does.

The 2026-09-23 bucket runs (does the VIX bucket need VVIX and VIX3M; the free
feed's bucket) wrote their thirteen one-bar arms in the spec's nested layout:
  results/linear_subsection/<pulled>/<root>/<bucket>/<bar>/<est>/tw2000/
      causal_tune_linear/<est>/<bucket>/results_<bar>.csv
with <root> = linear_subsection (ridge, elastic net) or linear_subsection_lassofix
(the fixed recursive lasso).  Each is stacked with build_subsection_yhat's own
read_arm and with_production_rv (the production rv_raw, the baseline gate), so
the 15:30 notebook reads them through the same recalibration as the
live-feasible per-bar forecasts.

Buckets (2000-session rolling window, per-bar, 13 regular-hours bars):
  vix_only       HAR + calendar + the VIX level
  live_vix_only  the live-feasible set minus VVIX and VIX3M (14 columns)
  free_vix_only  the free feed's set: ES minute-bar moments incl. volume, the
                 VIX, the FOMC calendar (13 columns; no tick count, no VVIX/VIX3M)

GATE  the incumbent live-feasible arms of the same comparison
      (results/linear_subsection/arms_carc/live_feasible, flat layout) stacked
      the same way equal the tables the notebook already reads
      (yhat_sub_<est>_live_feasible.parquet) row for row -- so the new tables
      and the notebook's live-feasible rows are like with like.

Writes results/spxw_pnl/yhat_sub_<est>_<bucket>.parquet.
Run:  python experiments/build_vixonly_yhat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_subsection_yhat import BARS, OUT, TW, read_arm, with_production_rv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "results" / "linear_subsection"
INCUMBENT = LS / "arms_carc" / "live_feasible"
ESTIMATORS = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
BUCKETS = {
    "vix_only": "vixonly_carc",
    "live_vix_only": "vixonly_carc",
    "free_vix_only": "free_carc",
}


def arm_path(pulled: str, bucket: str, bar: str, est: str) -> Path:
    root = "linear_subsection_lassofix" if est == "reclasso" else "linear_subsection"
    return (
        LS
        / pulled
        / root
        / bucket
        / bar
        / est
        / f"tw{TW}"
        / "causal_tune_linear"
        / est
        / bucket
        / f"results_{bar}.csv"
    )


def stack(paths: list[Path], name: str) -> pd.DataFrame:
    tab = (
        pd.concat([read_arm(p) for p in paths]).sort_values("t").reset_index(drop=True)
    )
    assert not tab["t"].duplicated().any(), name
    return with_production_rv(tab, name)


def main() -> None:
    # GATE: the incumbent of the comparison is the table the notebook already reads
    for est, short in ESTIMATORS.items():
        inc = stack(
            [INCUMBENT / est / f"tw{TW}" / f"results_{b}.csv" for b in BARS],
            f"incumbent live_feasible {short}",
        )
        deck = pd.read_parquet(OUT / f"yhat_sub_{short}_live_feasible.parquet")
        assert len(inc) == len(deck) and inc["t"].equals(deck["t"]), short
        d = float(np.max(np.abs(inc["yhat"].to_numpy() - deck["yhat"].to_numpy())))
        assert d == 0.0, (short, d)
        print(
            f"GATE  live_feasible {short}: the comparison's incumbent arms equal "
            f"yhat_sub_{short}_live_feasible.parquet on {len(inc):,} rows (max |dyhat| {d:.1e})"
        )
    for bucket, pulled in BUCKETS.items():
        for est, short in ESTIMATORS.items():
            paths = [arm_path(pulled, bucket, b, est) for b in BARS]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"skip {bucket} {short}: {len(missing)} of 13 bar arms missing")
                continue
            name = f"yhat_sub_{short}_{bucket}.parquet"
            tab = stack(paths, name)
            tab.to_parquet(OUT / name, index=False)
            print(
                f"wrote {name}: {len(tab):,} rows, {tab['t'].dt.normalize().nunique():,} sessions, "
                f"{tab['t'].min().date()} .. {tab['t'].max().date()}"
            )


if __name__ == "__main__":
    main()
