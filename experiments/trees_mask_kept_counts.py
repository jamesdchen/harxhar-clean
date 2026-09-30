"""Kept-column counts of the masked 16:00 tree and LSTM arms (checklist I9, WINDOW_MASK=1).

The chunk reducers (experiments/trees_cadence_reduce_chunks.py, reduce_trees_tuned_chunks.py,
reduce_lstm_chunks.py) merge a fixed list of npz keys and drop the mask keys the specs write
under WINDOW_MASK=1 (kept_n / kept per refit; tune_kept_n / tune_kept per tuning point).
This script reads them from the CHUNK npz files, in chunk order (the task files), and writes

  <out>/kept_counts.csv      one row per (rung, model, bucket): p, refits, kept_n median /
                             min / max over refits, the same over tuning points (tuned and
                             LSTM), chunks read, and whether every chunk ran with the mask
  <out>/kept_refits.csv      one row per (rung, model, bucket, refit): whole-series OOS row,
                             kept_n
  <out>/kept_columns.csv     one row per (bucket, column): the share of the T1 refits (a
                             refit every session, so every window) that kept the column

The mask is a function of the design window alone, so every model of a bucket has the same
kept set at the same refit row; the script checks that (kept arrays equal across models at
common refit rows) and records it in kept_counts.csv (column same_as_first_model).

Usage: python experiments/trees_mask_kept_counts.py
           [--untuned-root results/linear_subsection_trees_mask]
           [--tuned-root results/linear_subsection_trees_tuned_mask]
           [--lstm-root results/linear_subsection_lstm_mask]  (REFIT_EVERY 1: rung LSTM)
           [--lstm10-root results/linear_subsection_lstm_mask_re10]  (REFIT_EVERY 10: LSTM10)
           [--out results/linear_subsection_trees_mask/kept]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEG, TW = "bar1600", 2000


def untuned_chunks(task_file: Path, rung: str) -> dict[tuple, list[int]]:
    arms: dict[tuple, list[int]] = {}
    for line in task_file.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if f and f[1] == rung:
            arms.setdefault((f[2], f[3]), []).append(int(f[6]))
    return {k: sorted(set(v)) for k, v in arms.items()}


def tuned_chunks(task_file: Path) -> dict[tuple, list[int]]:
    arms: dict[tuple, list[int]] = {}
    for line in task_file.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if f:
            for m in f[1].split(","):
                arms.setdefault((f[0], m), []).append(int(f[4]))
    return {k: sorted(set(v)) for k, v in arms.items()}


def read_arm(top: Path, chunks: list[int], sub: str, stem: str) -> dict | None:
    """Concatenate the mask keys of an arm's chunks; None if a chunk is missing."""
    parts: list[dict] = []
    for c in chunks:
        p = top / "chunks" / f"c{c}" / sub / f"{stem}_{SEG}.npz"
        if not p.is_file() or not (top / "chunks" / f"c{c}" / "DONE").is_file():
            return None
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            off = (
                int(z["oos_offset"])
                if "oos_offset" in z.files
                else max(0, meta["slice"][0] - meta["slice"][2])
            )
            parts.append(
                {
                    "masked": meta.get("window_mask") == 1 and "kept_n" in z.files,
                    "p": int(meta["n_features"]),
                    "rows": off + np.asarray(z["refit_row"], dtype=np.int64),
                    "kept_n": np.asarray(z["kept_n"])
                    if "kept_n" in z.files
                    else np.zeros(0, np.int64),
                    "kept": np.asarray(z["kept"])
                    if "kept" in z.files
                    else np.zeros((0, 0), np.uint8),
                    "tune_kept_n": np.asarray(z["tune_kept_n"])
                    if "tune_kept_n" in z.files
                    else None,
                }
            )
    tk = [q["tune_kept_n"] for q in parts if q["tune_kept_n"] is not None]
    return {
        "masked": all(q["masked"] for q in parts),
        "p": parts[0]["p"],
        "rows": np.concatenate([q["rows"] for q in parts]),
        "kept_n": np.concatenate([q["kept_n"] for q in parts]),
        "kept": np.vstack([q["kept"] for q in parts]),
        "tune_kept_n": np.concatenate(tk) if tk else None,
        "chunks": len(parts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--untuned-root", default="results/linear_subsection_trees_mask")
    ap.add_argument(
        "--tuned-root", default="results/linear_subsection_trees_tuned_mask"
    )
    ap.add_argument("--lstm-root", default="results/linear_subsection_lstm_mask")
    ap.add_argument("--lstm10-root", default="results/linear_subsection_lstm_mask_re10")
    ap.add_argument("--untuned-tasks", default="cluster/trees_mask_tasks_untuned.txt")
    ap.add_argument("--rs10-tasks", default="cluster/trees_mask_tasks_rs10.txt")
    ap.add_argument("--rs1-tasks", default="cluster/trees_mask_tasks_rs1.txt")
    ap.add_argument("--lstm-tasks", default="cluster/lstm_mask_tasks_fleet.txt")
    ap.add_argument("--out", default="results/linear_subsection_trees_mask/kept")
    a = ap.parse_args()
    R = lambda s: ROOT / s  # noqa: E731
    out = R(a.out)
    out.mkdir(parents=True, exist_ok=True)

    arms: list[tuple[str, str, str, dict | None]] = []  # (rung, model, bucket, data)
    for rung in ("t10", "t1"):
        for (b, m), ch in sorted(untuned_chunks(R(a.untuned_tasks), rung).items()):
            top = R(a.untuned_root) / rung / b / SEG / m / f"tw{TW}"
            arms.append(
                (
                    rung.upper(),
                    m,
                    b,
                    read_arm(top, ch, f"causal_tune_trees/{m}/{b}", "trees"),
                )
            )
    for rung, tf in (("rs10", a.rs10_tasks), ("rs1", a.rs1_tasks)):
        for (b, m), ch in sorted(tuned_chunks(R(tf)).items()):
            top = R(a.tuned_root) / rung / b / SEG / m / f"tw{TW}"
            arms.append(
                (
                    rung.upper(),
                    m,
                    b,
                    read_arm(top, ch, f"causal_tune_trees/{m}/{b}", "trees"),
                )
            )
    for rung, lroot in (("LSTM", a.lstm_root), ("LSTM10", a.lstm10_root)):
        for (b, m), ch in sorted(tuned_chunks(R(a.lstm_tasks)).items()):
            top = R(lroot) / b / SEG / m / f"tw{TW}"
            arms.append(
                (rung, m, b, read_arm(top, ch, f"causal_tune_lstm/lstm/{b}", "lstm"))
            )

    rows, per_refit = [], []
    first: dict[tuple, dict] = {}  # (rung, bucket) -> the first model's arm
    for rung, m, b, d in arms:
        if d is None:
            rows.append({"rung": rung, "model": m, "bucket": b, "complete": False})
            print(f"incomplete: {rung} {m} {b}")
            continue
        kn = d["kept_n"]
        row = {
            "rung": rung,
            "model": m,
            "bucket": b,
            "complete": True,
            "masked_every_chunk": d["masked"],
            "chunks": d["chunks"],
            "p": d["p"],
            "refits": len(d["rows"]),
            "kept_n_median": float(np.median(kn)) if len(kn) else np.nan,
            "kept_n_min": int(kn.min()) if len(kn) else np.nan,
            "kept_n_max": int(kn.max()) if len(kn) else np.nan,
            "kept_share_median": float(np.median(kn) / d["p"]) if len(kn) else np.nan,
        }
        tk = d["tune_kept_n"]
        if tk is not None and len(tk):
            row |= {
                "tunings": len(tk),
                "tune_kept_n_median": float(np.median(tk)),
                "tune_kept_n_min": int(tk.min()),
                "tune_kept_n_max": int(tk.max()),
            }
        ref = first.setdefault((rung, b), {"model": m, **d})
        if ref["model"] != m:
            common, ia, ib = np.intersect1d(ref["rows"], d["rows"], return_indices=True)
            row["same_as_first_model"] = bool(
                len(common) and np.array_equal(ref["kept"][ia], d["kept"][ib])
            )
        rows.append(row)
        per_refit.append(
            pd.DataFrame(
                {
                    "rung": rung,
                    "model": m,
                    "bucket": b,
                    "oos_row": d["rows"],
                    "kept_n": kn,
                }
            )
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "kept_counts.csv", index=False)
    if per_refit:
        pd.concat(per_refit, ignore_index=True).to_csv(
            out / "kept_refits.csv", index=False
        )
    # per-column kept share over every window (T1: a refit every session), per bucket
    col_rows, done = [], set()
    for rung, m, b, d in arms:
        if rung != "T1" or d is None or b in done or not d["kept"].size:
            continue
        done.add(b)
        npz = sorted(
            (R(a.untuned_root) / "t1" / b / SEG / m / f"tw{TW}" / "chunks").glob(
                f"c*/causal_tune_trees/{m}/{b}/trees_{SEG}.npz"
            )
        )
        with np.load(npz[0], allow_pickle=False) as z:
            names = [str(v) for v in z["feature_names"]]
        share = d["kept"].mean(axis=0)
        col_rows += [
            {
                "bucket": b,
                "model_read": m,
                "col": j,
                "name": names[j],
                "kept_share_T1_refits": float(share[j]),
            }
            for j in range(d["p"])
        ]
    pd.DataFrame(col_rows).to_csv(out / "kept_columns.csv", index=False)
    print(tab.to_string(index=False))
    print(f"wrote {out / 'kept_counts.csv'}, kept_refits.csv, kept_columns.csv")


if __name__ == "__main__":
    main()
