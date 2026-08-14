"""SPXW.csv -> compact parquet. Parallel byte-range convert.

Keeps every chain row, including 0-delta / 0-IV / missing-greek prints.
Those neighboring bars are the exit marks for PnL: a contract that went
deep ITM/OTM often prints delta=0 (or 1) with a live mid. Filtering them
at ingest makes the exit unjoinable. Entry filters (need a mid) live in
the PnL picker, not here.

Raw vendor implied_vol / delta columns are dropped (junk: 2.47 or 0.0 on
the same print as a usable new_* surface). The *rows* stay.

  data/spxw_chain.parquet   one row per (timestamp, expiration, strike, cp)
  data/spxw_spot.parquet    one row per timestamp (last underlying_price)
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

KEEP = [
    "expiration",
    "strike",
    "CP",
    "timestamp",
    "bid",
    "ask",
    "underlying_price",
    "early_close",
    "mid",
    "hours_to_expiration",
    "new_implied_vol",
    "new_delta",
]
RENAME = {
    "CP": "cp",
    "new_implied_vol": "impl_volatility",
    "new_delta": "delta",
}
F32 = [
    "strike",
    "bid",
    "ask",
    "underlying_price",
    "mid",
    "hours_to_expiration",
    "impl_volatility",
    "delta",
]


def _line_offsets(path: str, n_parts: int) -> list[int]:
    """Byte starts of n_parts slices; each start is a line boundary after the header."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header_end = len(f.readline())
        cuts = [header_end]
        for i in range(1, n_parts):
            f.seek(i * size // n_parts)
            f.readline()
            pos = f.tell()
            if pos < size and pos > cuts[-1]:
                cuts.append(pos)
        cuts.append(size)
    return cuts


def _cast_table(table: pa.Table) -> pa.Table:
    """Cast/rename only. Never drop rows — 0-delta and 0-IV stay."""
    import pandas as pd

    df = table.to_pandas()
    df = df.rename(columns=RENAME)
    # A split can leave the header string in the first row; drop that
    # row only. Never drop 0-delta / 0-IV prints.
    if "timestamp" in df.columns:
        df = df[df["timestamp"].astype(str) != "timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["cp"] = df["cp"].astype(str).str.upper().str[0]
    df["early_close"] = df["early_close"].astype(bool)
    for c in F32:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return pa.Table.from_pandas(df, preserve_index=False)


def _convert_slice(path: str, start: int, end: int, header: bytes, out_part: str) -> int:
    if end <= start:
        return 0
    with open(path, "rb") as f:
        # Body must not include the file header; we prepend it ourselves.
        if start < len(header):
            start = len(header)
        if end <= start:
            return 0
        f.seek(start)
        body = f.read(end - start)
    table = pacsv.read_csv(
        pa.py_buffer(header + body),
        read_options=pacsv.ReadOptions(use_threads=False),
        convert_options=pacsv.ConvertOptions(
            include_columns=KEEP,
            strings_can_be_null=True,
            null_values=["", "NA", "NaN"],
        ),
        parse_options=pacsv.ParseOptions(newlines_in_values=False),
    )
    table = _cast_table(table)
    pq.write_table(table, out_part, compression="zstd")
    return table.num_rows


def _combine(parts: list[str], out_path: str) -> int:
    writer = None
    n = 0
    try:
        for p in parts:
            if not os.path.exists(p):
                continue
            t = pq.read_table(p)
            n += t.num_rows
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema, compression="zstd")
            writer.write_table(t)
    finally:
        if writer is not None:
            writer.close()
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    return n


def _spot_from_chain(chain_path: str, out_spot: str) -> int:

    t = pq.read_table(chain_path, columns=["timestamp", "underlying_price"])
    df = t.to_pandas()
    df = df.dropna(subset=["underlying_price"])
    spot = (
        df.groupby("timestamp", sort=True)["underlying_price"]
        .last()
        .rename("spot")
        .reset_index()
    )
    spot.to_parquet(out_spot, index=False)
    return len(spot)


def _run_polars(csv_path: str, out_chain: str, out_spot: str) -> tuple[int, int]:
    import polars as pl

    q = (
        pl.scan_csv(csv_path, try_parse_dates=False)
        .select(KEEP)
        .rename(RENAME)
        .with_columns(
            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%z").dt.convert_time_zone("UTC"),
            pl.col("expiration").str.to_date("%Y-%m-%d"),
            pl.col("cp").str.to_uppercase().str.slice(0, 1),
            pl.col("early_close").cast(pl.Boolean),
            *[pl.col(c).cast(pl.Float32) for c in F32],
        )
    )
    q.sink_parquet(out_chain, compression="zstd")
    n_chain = pl.scan_parquet(out_chain).select(pl.len()).collect().item()
    spot = (
        pl.scan_parquet(out_chain)
        .filter(pl.col("underlying_price").is_not_null())
        .group_by("timestamp")
        .agg(pl.col("underlying_price").last().alias("spot"))
        .sort("timestamp")
    )
    spot.sink_parquet(out_spot, compression="zstd")
    n_spot = pl.scan_parquet(out_spot).select(pl.len()).collect().item()
    return int(n_chain), int(n_spot)


def _run_parallel(csv_path: str, out_chain: str, out_spot: str, workers: int) -> tuple[int, int]:
    with open(csv_path, "rb") as f:
        header = f.readline()
    cuts = _line_offsets(csv_path, workers)
    part_dir = Path(out_chain).with_suffix("")
    part_dir = Path(str(part_dir) + "_parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = [str(part_dir / f"part-{i:03d}.parquet") for i in range(len(cuts) - 1)]
    args = [
        (csv_path, cuts[i], cuts[i + 1], header, parts[i])
        for i in range(len(cuts) - 1)
        if cuts[i + 1] > cuts[i]
    ]
    print(f"pyarrow parallel: {len(args)} slices, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_convert_slice_star, args, chunksize=1))
    n_chain = _combine(parts, out_chain)
    try:
        part_dir.rmdir()
    except OSError:
        pass
    n_spot = _spot_from_chain(out_chain, out_spot)
    print(f"slice rows {sum(rows):,}", flush=True)
    return n_chain, n_spot


def _convert_slice_star(a: tuple) -> int:
    return _convert_slice(*a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/SPXW.csv")
    ap.add_argument("--out-chain", default="data/spxw_chain.parquet")
    ap.add_argument("--out-spot", default="data/spxw_spot.parquet")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--engine", choices=("auto", "polars", "pyarrow"), default="auto")
    a = ap.parse_args()

    csv_path = os.path.abspath(a.csv)
    out_chain = os.path.abspath(a.out_chain)
    out_spot = os.path.abspath(a.out_spot)
    os.makedirs(os.path.dirname(out_chain) or ".", exist_ok=True)

    have_polars = False
    try:
        import polars  # noqa: F401

        have_polars = True
    except ImportError:
        pass

    engine = a.engine
    if engine == "auto":
        engine = "polars" if have_polars else "pyarrow"
    print(f"engine={engine}  csv={os.path.getsize(csv_path)/1e6:.0f} MB", flush=True)

    if engine == "polars":
        n_chain, n_spot = _run_polars(csv_path, out_chain, out_spot)
    else:
        n_chain, n_spot = _run_parallel(csv_path, out_chain, out_spot, max(2, a.workers))

    # spot min/max without loading the chain
    spot_t = pq.read_table(out_spot, columns=["timestamp"])
    tmin, tmax = pc.min(spot_t["timestamp"]).as_py(), pc.max(spot_t["timestamp"]).as_py()
    print(
        f"chain: {n_chain:,} rows -> {out_chain} ({os.path.getsize(out_chain)/1e6:.0f} MB); "
        f"spot: {n_spot:,} stamps {tmin} .. {tmax} -> {out_spot} "
        f"({os.path.getsize(out_spot)/1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
