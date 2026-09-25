"""Gate: the persisted rest-of-day forecasts reproduce the QLIKE table byte for byte.

experiments/multihorizon_ttc.py --save-forecasts writes, next to its QLIKE
tables, the per-day, per-hour forecasts they score and the realized target:
results/multihorizon/ttc<tag>_forecasts.parquet (columns day, hour, y and one
column per forecast key). This script re-scores that file with the script's
own loss and table code and asserts that the result, written as CSV, is
byte-identical to results/multihorizon/ttc<tag>_byhour.csv. A pass means the
parquet holds exactly the forecasts behind the published table -- the same
days, hours, targets and values -- so downstream users (the hold-to-close
notebook) read the forecasts that were scored, not a re-derivation.

Usage: python experiments/multihorizon_frem_check.py [--tag TAG]
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multihorizon_ttc as ttc  # noqa: E402


def check(tag: str = "") -> pd.DataFrame:
    fc = pd.read_parquet(os.path.join(ttc.OUT, f"ttc{tag}_forecasts.parquet"))
    hours = list(dict.fromkeys(fc["hour"]))
    days = fc["day"].drop_duplicates().to_numpy()
    n_days, n_h = len(days), len(hours)
    assert len(fc) == n_days * n_h, "the file is not a full day x hour grid"
    assert (fc["hour"].to_numpy() == np.tile(np.asarray(hours), n_days)).all()
    assert (fc["day"].to_numpy() == np.repeat(days, n_h)).all()
    ttc.ENTRY = hours  # byhour_table labels rows by the module's entry hours
    y = fc["y"].to_numpy(float).reshape(n_days, n_h)
    L = {
        k: ttc._ql(fc[k].to_numpy(float).reshape(n_days, n_h), y)
        for k in ttc.FORECAST_KEYS
    }
    byh = ttc.byhour_table(L, days)
    buf = io.StringIO()
    byh.to_csv(buf, index=False, lineterminator=os.linesep)
    ref_path = os.path.join(ttc.OUT, f"ttc{tag}_byhour.csv")
    with open(ref_path, "rb") as fh:
        ref = fh.read()
    got = buf.getvalue().encode()
    if got != ref:
        a = pd.read_csv(io.BytesIO(got))
        b = pd.read_csv(ref_path)
        num = a.select_dtypes("number").columns
        worst = float(np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy())))
        raise SystemExit(
            f"GATE FAILED: re-scored table differs from {ref_path} (max |diff| {worst:.3e})"
        )
    print(
        f"GATE ttc{tag}_forecasts.parquet re-scored reproduces ttc{tag}_byhour.csv "
        f"byte for byte ({len(byh)} rows; {n_days} days x {n_h} entry hours "
        f"{hours[0]}..{hours[-1]}; {len(ttc.FORECAST_KEYS)} forecasts)"
    )
    return byh


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="re-score persisted ttc forecasts")
    ap.add_argument("--tag", default="")
    check(ap.parse_args().tag)
