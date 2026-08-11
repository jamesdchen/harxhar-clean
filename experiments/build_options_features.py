"""End-to-end: date-keyed options features -> endbartime-keyed panel parquet.

Usage:
  python experiments/build_options_features.py \
      --features results/gex/gex_daily.parquet \
      --panel    data/core_stats.parquet \
      --out      data/options_features.parquet

The input is any date-keyed frame (one row per trade date). Known renames
(the gex.py output schema) are applied to produce the registered opt_*
channel names; any extra columns pass through with the opt_ prefix added.
The attachment enforces the publication-lag rule of
src/data/options_features.py and runs the causality gate + coverage report.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.options_features import (
    CHANNEL_PREFIX,
    GO_LIVE_TIME,
    attach_daily_to_bars,
    coverage_report,
    write_options_parquet,
)

GEX_RENAMES = {
    "gex_level": "opt_gex_level",
    "zero_gamma": "opt_gex_zero",
    "dist_to_flip": "opt_gex_flip_dist",
    "pin_strike": "opt_pin_strike",
    "net_vanna": "opt_vanna_net",
    "net_charm": "opt_charm_net",
    "regime_long_gamma": "opt_regime_long",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/gex/gex_daily.parquet")
    ap.add_argument("--panel", default="data/core_stats.parquet",
                    help="RV-source parquet; its endbartime column IS the panel's bar grid")
    ap.add_argument("--out", default="data/options_features.parquet")
    ap.add_argument("--go-live", default=GO_LIVE_TIME,
                    help="HH:MM publication time on the day AFTER the trade date")
    a = ap.parse_args()

    feat = pd.read_parquet(a.features)
    if "date" not in feat.columns:
        feat = feat.reset_index().rename(columns={feat.index.name or "index": "date"})
    feat["date"] = pd.to_datetime(feat["date"])
    renames = {k: v for k, v in GEX_RENAMES.items() if k in feat.columns}
    feat = feat.rename(columns=renames)
    extra = [c for c in feat.columns if c != "date" and not c.startswith(CHANNEL_PREFIX)]
    feat = feat.rename(columns={c: CHANNEL_PREFIX + c for c in extra})
    channels = [c for c in feat.columns if c != "date"]
    print(f"channels: {channels}")

    bars = pd.to_datetime(pd.read_parquet(a.panel, columns=["endbartime"])["endbartime"])
    bars = pd.DatetimeIndex(bars.dropna().unique()).sort_values()
    print(f"panel bars: {len(bars):,}  {bars.min()} .. {bars.max()}")

    merged = attach_daily_to_bars(feat, bars, go_live_time=a.go_live)
    rep = coverage_report(merged, channels)
    print(rep.to_string())
    out = write_options_parquet(merged, channels, a.out)
    print(f"wrote {out}: {len(merged):,} bars x {len(channels)} channels, go-live {a.go_live} next-day")


if __name__ == "__main__":
    main()
