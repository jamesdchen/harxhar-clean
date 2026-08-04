"""One-time clean-data prep: parse the WRDS OptionMetrics gz export ONCE into a compact, filtered,
downcast parquet so gex.py and its sign/DTE iterations read a fast local parquet instead of re-parsing
the ~1GB gz every run. Filters to the standing-gamma horizon (OI>0, 0<DTE<=MAX_DTE) -- the bulk of the
rows (zero-OI lines, far-dated LEAPS) contribute ~no gamma and are dropped here, not every run.

Usage:
  python prep_om_chain.py --gz-options <chain.csv.gz> --gz-spot <spot.csv.gz>
  # -> data/optionm_spx_chain.parquet + data/optionm_spx_spot.parquet
  python gex.py --options data/optionm_spx_chain.parquet --spot data/optionm_spx_spot.parquet
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import argparse
import os

import pandas as pd

from gex import MAX_DTE, _OPT_KEEP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz-options", required=True)
    ap.add_argument("--gz-spot", required=True)
    ap.add_argument("--out-options", default="data/optionm_spx_chain.parquet")
    ap.add_argument("--out-spot", default="data/optionm_spx_spot.parquet")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_options) or ".", exist_ok=True)

    o = pd.read_csv(a.gz_options, usecols=lambda c: c.strip().lower() in _OPT_KEEP)
    o.columns = [c.lower() for c in o.columns]
    if "strike" not in o.columns and "strike_price" in o.columns:
        o["strike"] = o["strike_price"].astype("float64") / 1000.0
    o["cp_flag"] = o["cp_flag"].astype(str).str.upper().str[0]
    o["date"] = pd.to_datetime(o["date"])
    o["exdate"] = pd.to_datetime(o["exdate"])
    tau = (o["exdate"] - o["date"]).dt.days / 365.25
    n0 = len(o)
    keep = (o["open_interest"].fillna(0) > 0) & (tau > 0) & (tau <= MAX_DTE / 365.25)
    cols = ["date", "exdate", "cp_flag", "strike", "impl_volatility", "open_interest"]
    if "gamma" in o.columns:
        cols.append("gamma")
    o = o.loc[keep, cols].reset_index(drop=True)
    for c in ("strike", "impl_volatility", "open_interest", "gamma"):
        if c in o.columns:
            o[c] = o[c].astype(
                "float32"
            )  # halve the footprint; gamma/IV precision is ample at f32
    o.to_parquet(a.out_options, index=False)
    sz = os.path.getsize(a.out_options) / 1e6
    print(
        f"chain: {n0:,} -> {len(o):,} rows (OI>0 & 0<DTE<={MAX_DTE}); {sz:.0f} MB -> {a.out_options}"
    )

    s = pd.read_csv(a.gz_spot, usecols=lambda c: c.strip().lower() in {"date", "close"})
    s.columns = [c.lower() for c in s.columns]
    s["date"] = pd.to_datetime(s["date"])
    s.to_parquet(a.out_spot, index=False)
    print(f"spot: {len(s):,} rows -> {a.out_spot}")


if __name__ == "__main__":
    main()
