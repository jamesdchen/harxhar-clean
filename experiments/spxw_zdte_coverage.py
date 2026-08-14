"""Two-column 0DTE coverage. Do not load mids."""

from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")


def main() -> None:
    path = os.path.join(ROOT, "data", "spxw_chain.parquet")
    print("read timestamp+expiration", flush=True)
    c = pd.read_parquet(path, columns=["timestamp", "expiration"])
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
    c["expiration"] = pd.to_datetime(c["expiration"])
    et = c["timestamp"].dt.tz_convert("America/New_York")
    tdate = et.dt.normalize()
    edate = (
        c["expiration"]
        .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        .dt.normalize()
    )
    zdte = tdate == edate
    print(
        f"rows={len(c):,} stamps={c['timestamp'].nunique():,} exp={c['expiration'].nunique():,}",
        flush=True,
    )
    print(
        f"0DTE rows={int(zdte.sum()):,} stamps={c.loc[zdte, 'timestamp'].nunique():,} days={tdate[zdte].nunique():,}",
        flush=True,
    )
    hours = et[zdte].dt.hour.value_counts().sort_index()
    print("0DTE hour counts ET", flush=True)
    print(hours.to_string(), flush=True)
    # unique 0DTE stamps for later joins
    stamps = pd.DataFrame(
        {"t": c.loc[zdte, "timestamp"].drop_duplicates().sort_values()}
    ).reset_index(drop=True)
    out = os.path.join(OUT, "zdte_stamps.parquet")
    os.makedirs(OUT, exist_ok=True)
    stamps.to_parquet(out, index=False)
    hours.to_csv(os.path.join(OUT, "zdte_hours.csv"), header=["n"])
    print(f"wrote {out} n={len(stamps):,}", flush=True)


if __name__ == "__main__":
    main()
