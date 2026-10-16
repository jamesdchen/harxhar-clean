"""Cross-CPU test of the forest's TUNED rungs (2026-10-07): the trial-163 re-tuning canary chunks re-run on Hoffman2 AVX
(cluster/submit_trees_rfp163_h2test.sh; E5-2670 / E5-2650 v2, root /u/scratch/j/jamesdc1/harxhar-rfp163-h2test) compared
with the same chunks of the CARC epyc-7513 run (/scratch1/jc_905/harxhar-rfp163-tuned), both pulled locally (--h2, --carc:
the two results/ trees).

Per chunk (rs10 / rs1 random search, and the unmasked fixed-settings pack-0 chunks): every array of trees_bar1600.npz (dates,
forecasts, targets, refit rows, importance, TreeSHAP, kept columns, the tuning record) byte for byte (the run metadata
equal except its timings and the worker-pool size), and every CSV of the
chunk's output folder (results, QLIKE-rule results, the candidates' validation MSE / QLIKE, the chosen configurations, the
feature health) cell for cell as text, with the timing columns (*sec*, *time*) left out; grid JSON as text.

Writes <out>/h2_tuned_compare.csv and prints one line per chunk.
Run:  python experiments/trees_h2_tuned_compare.py --h2 <dir> --carc <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def lp(p: Path) -> Path:
    """Windows: the long-path form (the chunk files lie deeper than MAX_PATH under the scratch copies)."""
    return Path(r"\\?" + "\\" + str(p.absolute())) if os.name == "nt" else p


def timing(col: str) -> bool:
    c = col.lower()
    return "sec" in c or "time" in c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2", required=True)
    ap.add_argument("--carc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h2, carc, out = Path(a.h2), Path(a.carc), Path(a.out)
    rows = []
    for done in sorted(h2.glob("*/*/*/bar1600/rf/tw2000/chunks/c*/DONE")):
        d = done.parent
        if not d.name[1:].isdigit():
            continue
        rel = d.relative_to(h2)
        root, rung, bucket = rel.parts[0], rel.parts[1], rel.parts[2]
        c = carc / rel
        node = (
            (d / "H2_HOST").read_text(encoding="utf-8").strip().replace("\n", " ")
            if (d / "H2_HOST").is_file()
            else ""
        )
        r: dict = {
            "root": root,
            "rung": rung,
            "bucket": bucket,
            "chunk": d.name,
            "h2_host": node,
        }
        if not (c / "DONE").is_file():
            rows.append(r | {"identical": False, "why": "no CARC chunk"})
            continue
        sub = Path("causal_tune_trees") / "rf" / bucket
        diffs: list[str] = []
        za, zb = (
            np.load(lp(d / sub / "trees_bar1600.npz"), allow_pickle=True),
            np.load(lp(c / sub / "trees_bar1600.npz"), allow_pickle=True),
        )
        keys = sorted(set(za.files) | set(zb.files))
        for k in keys:
            if timing(k):
                continue
            if k not in za.files or k not in zb.files:
                diffs.append(f"npz:{k}(missing)")
                continue
            if (
                k == "meta"
            ):  # run metadata: equal except the timings and the pool size (8 slots here, 20 on CARC)
                ma, mb = (json.loads(str(z[k])) for z in (za, zb))
                drop = {m for m in set(ma) | set(mb) if timing(m) or m == "workers"}
                if {m: ma.get(m) for m in set(ma) - drop} != {
                    m: mb.get(m) for m in set(mb) - drop
                }:
                    diffs.append("npz:meta")
                continue
            x, y = za[k], zb[k]
            if x.dtype == object or y.dtype == object:
                same = repr(x.tolist()) == repr(y.tolist())
            else:
                same = (
                    x.dtype == y.dtype
                    and x.shape == y.shape
                    and np.ascontiguousarray(x).tobytes()
                    == np.ascontiguousarray(y).tobytes()
                )
            if not same:
                diffs.append(f"npz:{k}")
        if (
            "pred_adj" in za.files
            and "pred_adj" in zb.files
            and za["pred_adj"].shape == zb["pred_adj"].shape
        ):
            dp = np.abs(za["pred_adj"] - zb["pred_adj"])
            r |= {
                "rows": len(dp),
                "rows_pred_differ": int((dp > 0).sum()),
                "max_abs_pred_adj_diff": float(dp.max()),
            }
        n_csv = 0
        for f in sorted((d / sub).glob("*.csv")):
            g = c / sub / f.name
            if not lp(g).is_file():
                diffs.append(f"{f.name}(missing)")
                continue
            x, y = (
                pd.read_csv(lp(f), dtype=str, keep_default_na=False),
                pd.read_csv(lp(g), dtype=str, keep_default_na=False),
            )
            cols = [k for k in x.columns if not timing(k)]
            n_csv += 1
            if (
                list(x.columns) != list(y.columns)
                or x.shape != y.shape
                or not x[cols].equals(y[cols])
            ):
                diffs.append(f.name)
        for f in sorted((d / sub).glob("*.json")):
            g = c / sub / f.name
            if f.name.endswith("_reduce.json"):
                continue
            if not g.is_file() or f.read_text(encoding="utf-8") != g.read_text(
                encoding="utf-8"
            ):
                diffs.append(f.name)
        rows.append(
            r
            | {
                "npz_keys": len(keys),
                "csvs": n_csv,
                "identical": not diffs,
                "differing": " ".join(diffs),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "h2_tuned_compare.csv", index=False)
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60
    ):
        print(
            df.drop(columns=[k for k in ("h2_host",) if k in df]).to_string(index=False)
        )
    print(f"{int(df['identical'].sum())} of {len(df)} chunks identical")
    return 0 if bool(df["identical"].all()) else 1


if __name__ == "__main__":
    sys.exit(main())
