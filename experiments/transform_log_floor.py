"""The floor of the floored-log target variant V3f (results/transform_ablation_2026-10-30/RULE.txt addendum D): the
pre-sample 1st percentile of the positive diurnally adjusted RV, recorded once as the constant
src/backtest/executor.py TARGET_LOG_FLOOR_VALUE.

Rows: the target's own rows (the 48-interval grid the HAR averages run over; load_and_transform on no predictor, every
data flag at its default, B as the incumbent's: B_REGULAR_BARS_ONLY), dated before 2013-01-01 -- before the
validation year of pre-sample fold 3, so before every row that is validated or evaluated -- with a defined diurnal
baseline (B = 1.0 is diurnal_adjust_with_floor's "too little history" sentinel: 240 rows of 1998-2003, raw RV there)
and RV > 0.  Statistic: numpy's default (linear) quantile at 0.01 of RV / B on those rows, the level of the
incumbent's lower winsorization quantile (target.WINSOR_LOWER_Q).

usage: python experiments/transform_log_floor.py [--data data] [--out results/transform_ablation_2026-10-30/log_floor.json]
       exit 3 if the value differs from executor.TARGET_LOG_FLOOR_VALUE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CUT = pd.Timestamp("2013-01-01")
Q = 0.01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "transform_ablation_2026-10-30" / "log_floor.json"))
    a = ap.parse_args()
    import src.backtest.executor as EX
    import src.features.transforms.target as TG

    assert Q == TG.WINSOR_LOWER_Q, (Q, TG.WINSOR_LOWER_Q)
    EX.TARGET_LEVEL, EX.TARGET_WINSOR = True, False  # y = RV / B, no winsorization: the series the log is taken of
    df, _ = EX.load_and_transform(
        a.data,
        [],
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    t = pd.DatetimeIndex(pd.to_datetime(df["t"]))
    x, B, rv = (df[c].to_numpy(float) for c in ("adj_RV", "baseline", "RV"))
    pre = (t < CUT) & np.isfinite(x)
    use = pre & (B != 1.0) & (rv > 0)
    v = float(np.quantile(x[use], Q))
    rec = dict(
        rule="RULE.txt addendum D: the 1st percentile of positive RV / B, 48-interval grid, rows before 2013-01-01 with a "
        "defined diurnal baseline (B != 1.0)",
        value=v,
        value_repr=repr(v),
        log_value=float(np.log(v)),
        executor_constant=EX.TARGET_LOG_FLOOR_VALUE,
        equal=bool(v == EX.TARGET_LOG_FLOOR_VALUE),
        rows_pre=int(pre.sum()),
        rows_used=int(use.sum()),
        rows_b_undefined=int((pre & (B == 1.0)).sum()),
        zeros_pre=int((pre & (rv == 0)).sum()),
        zeros_all=int((np.isfinite(x) & (rv == 0)).sum()),
        share_pre_below=float(np.mean(x[pre] < v)),
        share_all_below=float(np.mean(x[np.isfinite(x)] < v)),
        min_positive_pre_defined_b=float(x[use].min()),
        min_positive_pre_any=float(x[pre & (rv > 0)].min()),
        first_row=str(t[0]),
        median_pre=float(np.median(x[use])),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0 if rec["equal"] else 3


if __name__ == "__main__":
    sys.exit(main())
