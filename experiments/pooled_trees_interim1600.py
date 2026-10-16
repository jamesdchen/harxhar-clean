"""INTERIM 16:00 table of a pooled forest arm (2026-10-13, coordinator): only the rows the 16:00 scorer reads.

experiments/master_table_close.py (read_table_1600 / load_one) reads the 16:00 ET rows of a forecast table, and its gate
wants every target 16:00 stamp of the table's span present.  For 2020-01-03 .. 2024-04-30 that is 1,087 rows (866 of
them scored trade days).  This collects exactly those rows -- listed with their t in --rows (CSV row_id,t) -- from the
arm's pieces (<root>/<arm>/pieces/chunk_<kkk>_<a>-<b>.npz) or merged chunks (<root>/<arm>/chunk_<kkk>.npz), checks each
row's t against the list, and writes t / yhat / baseline / rv_raw (dump_unif_yhat's schema) to --out.  Refuses unless
every listed row is found exactly once in agreeing copies.  The full-table dump is untouched (separate file name).

Run (CARC, inside a batch job):  python experiments/pooled_trees_interim1600.py --root results/pooled_trees_2026-10-07/chunks
    [--root results/pooled_trees_2026-10-07_race/chunks]  (a row found in two places must agree byte for byte)
    --arm pooltree_rf_all_features --rows cluster/pooltree_interim1600_rows.csv
    --out results/pooled_trees_2026-10-07/yhat_pooltree_rf_all_features_INTERIM1600.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_unif_yhat import _et_to_utc  # noqa: E402  (the bank dumper's own t conversion: npz t is naive ET)
from pooled_trees_walk import N_CHUNKS, chunk_bounds  # noqa: E402

N_ROWS_PANEL = 300317
KEYS = ("t", "yhat", "baseline", "rv_raw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        action="append",
        help="chunks root; repeat to add e.g. a race root",
    )
    ap.add_argument("--arm", required=True)
    ap.add_argument(
        "--rows", required=True, help="CSV with row_id,t (ISO) of the rows wanted"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = pd.read_csv(a.rows)
    rows = want["row_id"].to_numpy(np.int64)
    dirs = [Path(r) / a.arm for r in a.root]
    bounds = [chunk_bounds(k, N_ROWS_PANEL) for k in range(N_CHUNKS)]
    chunk_of = np.searchsorted([b[0] for b in bounds], rows, side="right") - 1
    found: dict[int, dict] = {}
    files = {}
    for k in sorted(set(chunk_of.tolist())):
        c0 = bounds[k][0]
        need = set((rows[chunk_of == k] - c0).tolist())
        cands = []
        for d in dirs:
            merged = d / f"chunk_{k:03d}.npz"
            if merged.is_file():
                cands.append((0, bounds[k][1] - c0, merged))
            for f in (d / "pieces").glob(f"chunk_{k:03d}_*.npz"):
                m = re.match(rf"chunk_{k:03d}_(\d{{4}})-(\d{{4}})\.npz$", f.name)
                if m and any(int(m.group(1)) <= i < int(m.group(2)) for i in need):
                    cands.append((int(m.group(1)), int(m.group(2)), f))
        for a0, b0, f in cands:
            if (
                f not in files
            ):  # read the few arrays needed and close the file (an open NpzFile per piece ran out of handles)
                with np.load(f, allow_pickle=True) as zz:
                    files[f] = {key: np.asarray(zz[key]) for key in ("row_id", *KEYS)}
            z = files[f]
            rid = np.asarray(z["row_id"])
            for i in need:
                if a0 <= i < b0:
                    j = np.flatnonzero(rid == c0 + i)
                    if len(j) != 1:
                        continue
                    rec = {key: np.asarray(z[key])[j[0]] for key in KEYS}
                    prev = found.get(c0 + i)
                    if prev is not None and any(
                        np.asarray(prev[key]).tobytes()
                        != np.asarray(rec[key]).tobytes()
                        for key in KEYS
                    ):
                        sys.exit(f"row {c0 + i}: two copies disagree ({f})")
                    found[c0 + i] = rec
    missing = [int(r) for r in rows if int(r) not in found]
    if missing:
        sys.exit(
            f"{len(missing)} of {len(rows)} rows missing, e.g. {missing[:10]}; nothing written"
        )
    out = pd.DataFrame({key: [found[int(r)][key] for r in rows] for key in KEYS})
    out["t"] = pd.Series(_et_to_utc(np.asarray(out["t"].to_numpy()))).dt.as_unit(
        "us"
    )  # the dumped tables' unit
    t_want = pd.to_datetime(want["t"], utc=True)
    if not (
        pd.Series(out["t"]).reset_index(drop=True) == t_want.reset_index(drop=True)
    ).all():
        sys.exit("t of a collected row differs from the list; nothing written")
    for key in ("yhat", "baseline", "rv_raw"):
        out[key] = out[key].astype(np.float64)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(
        f"{a.out}: {len(out)} rows {out['t'].min()} .. {out['t'].max()} (INTERIM, 16:00 rows only)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
