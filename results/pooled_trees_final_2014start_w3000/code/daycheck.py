"""Day against bar refits on one month (owner's question, 2026-10-23): the --refit day pieces of a month against the bar
forecasts of the main W = 3,000 run on the same rows.

usage: python daycheck.py <bar run dir> <day check dir> <first day> <last day> [est ...]
For each (est, all_features): the day piece <day dir>/pieces/<est>_all_features_day_<first>_<last>_1of1.parquet and the
bar rows of the same dates from <bar run>/pieces/<est>_all_features_bar_2014-01-02_2024-04-30_*.parquet.  Checks: the
first row of each day is the same fit under both cadences (bit-equal); then the relative difference |day - bar| / |bar|
of pred_adj (max, median, on the other rows) and the QLIKE of plain against rv_actual (rows with RV > 0) of each.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

bar, day = Path(sys.argv[1]), Path(sys.argv[2])
first, last = sys.argv[3], sys.argv[4]
ests = sys.argv[5:] or ["lgbm", "xgb"]
recs = []


def ql(f: pd.DataFrame, ok: np.ndarray) -> float:
    """QLIKE of plain against rv_actual on the rows ok."""
    r = f["rv_actual"].to_numpy()[ok] / f["plain"].to_numpy()[ok]
    return float(np.mean(r - np.log(r) - 1.0))


for e in ests:
    s = "all_features"
    fd = pd.read_parquet(
        day / "pieces" / f"{e}_{s}_day_{first}_{last}_1of1.parquet"
    ).set_index("date")
    fb = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted(
                (bar / "pieces").glob(f"{e}_{s}_bar_2014-01-02_2024-04-30_*of*.parquet")
            )
        ]
    )
    fb = fb.set_index("date").reindex(fd.index)
    assert fb["pred_adj"].notna().all(), (
        f"{e}: bar pieces do not cover {first} .. {last} yet"
    )
    d = pd.DatetimeIndex(fd.index).normalize()
    is_first = ~pd.Series(d).duplicated().to_numpy()
    assert np.array_equal(
        fd["pred_adj"].to_numpy()[is_first], fb["pred_adj"].to_numpy()[is_first]
    ), e
    assert np.array_equal(
        fd["rv_actual"].to_numpy(), fb["rv_actual"].to_numpy(), equal_nan=True
    )
    rel = (np.abs(fd["pred_adj"] - fb["pred_adj"]) / np.abs(fb["pred_adj"])).to_numpy()[
        ~is_first
    ]
    ok = (fd["rv_actual"] > 0).to_numpy()

    t16 = (pd.DatetimeIndex(fd.index).hour == 16) & ok
    r16d = fd["rv_actual"].to_numpy()[t16] / fd["plain"].to_numpy()[t16]
    r16b = fb["rv_actual"].to_numpy()[t16] / fb["plain"].to_numpy()[t16]
    recs.append(
        {
            "est": e,
            "set": s,
            "first": first,
            "last": last,
            "rows": len(fd),
            "days": int(is_first.sum()),
            "first_rows_bit_equal": True,
            "rel_max": float(rel.max()),
            "rel_median": float(np.median(rel)),
            "rel_p90": float(np.quantile(rel, 0.9)),
            "qlike_day": ql(fd, ok),
            "qlike_bar": ql(fb, ok),
            "qlike16_day": float(np.mean(r16d - np.log(r16d) - 1)),
            "qlike16_bar": float(np.mean(r16b - np.log(r16b) - 1)),
            "n_qlike": int(ok.sum()),
        }
    )
R = pd.DataFrame(recs)
R.to_csv(day / "daycheck.csv", index=False)
pd.set_option("display.width", 200)
print(R.to_string(index=False))
