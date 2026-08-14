"""Dump (t, yhat) from a unification arm's chunk npz into one parquet."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--root", default="results/unification")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, a.arm, "chunk_*.npz")))
    if not files:
        raise SystemExit(f"no chunks for {a.arm}")
    chunks = []
    z0 = np.load(files[0], allow_pickle=True)
    print("keys", z0.files, flush=True)
    need = ["t", "yhat", "baseline", "rv_raw"]
    miss = [k for k in need if k not in z0.files]
    if miss:
        raise SystemExit(f"missing {miss} in {z0.files}")
    for f in files:
        z = np.load(f, allow_pickle=True)
        chunks.append(
            pd.DataFrame(
                {
                    "t": pd.to_datetime(np.asarray(z["t"]), utc=True),
                    "yhat": np.asarray(z["yhat"], dtype=np.float64),
                    "baseline": np.asarray(z["baseline"], dtype=np.float64),
                    "rv_raw": np.asarray(z["rv_raw"], dtype=np.float64),
                }
            )
        )
    df = pd.concat(chunks, ignore_index=True).sort_values("t")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(
        f"{a.arm}: {len(df):,} rows {df['t'].min()} .. {df['t'].max()} -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
