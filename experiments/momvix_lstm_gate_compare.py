"""Reproduction gate of the moments_vix LSTM runs: fresh refits of stored chunks, bit for bit.

2026-10-06 (user request: every model also on the moments_vix input set, beside baseline and
all_features).  Before the new bucket runs, the deployment that will run it (code, data, torch /
numpy / pandas, CPU class) must reproduce forecasts the canonical runs stored:

  perbar    specs/causal_tune_lstm.py, WINDOW_MASK=1, REFIT_EVERY=1 (the canonical masked daily
            per-bar LSTM, results/linear_subsection_lstm_mask): a gate task line re-runs the
            first rows of a stored chunk (same START, a short END, HALO = W), so it recomputes
            that chunk's tuning point (every candidate x seed at the full budget) and refits the
            first rows every session.
  intraday  specs/causal_tune_lstm_intraday.py (results/linear_subsection_lstm_intraday): the
            same, STAGE=all with an empty tuning cache (REQUIRE_CACHE=0), so the tuning point is
            recomputed and written to the gate's own cache; that record is also compared with
            the stored stage-1 record (its fingerprint holds sha256 of the window's targets and
            bar inputs, the grid and the torch version).

Compared (gate run vs the stored CHUNK of the same rows, never the merged arm, whose CSV went
through a text re-parse): the results CSVs' text of date / horizon / true_adj / pred_adj /
true_raw on the gate's rows (pred_raw is the executor's look-ahead Duan back-transform over the
whole chunk, so it differs for a shorter chunk by construction: reported, not gated); from the
npz every per-row array on those rows (per-seed forecasts, both rules, kept columns) and every
tuning-record array (candidate validation MSE / QLIKE per seed, early-stopped epochs, picks,
split, kept columns); timings are not compared.  Floats are compared as their bit patterns.

Usage: python experiments/momvix_lstm_gate_compare.py --kind perbar|intraday --tasks FILE
           --gate-root DIR --stored-root DIR [--out CSV]
Exit 0 iff every compared item is bit-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CSV_COLS = ("date", "horizon", "true_adj", "pred_adj", "true_raw")
KIND: dict[str, dict[str, Any]] = {
    "perbar": {
        "model": "lstm",
        "spec_dir": "causal_tune_lstm",
        "npz": "lstm_bar1600.npz",
        "rows": (
            "date",
            "pred_adj",
            "true_adj",
            "true_raw",
            "baseline",
            "e2_adj",
            "pred_adj_seeds",
            "pred_adj_qsel",
            "refit_row",
            "kept_n",
            "kept",
        ),
        "tune": (
            "tune_row",
            "tune_split",
            "cand_val_mse",
            "cand_val_qlike",
            "cand_seed_val_mse",
            "cand_epochs",
            "chosen_mse",
            "chosen_qlike",
            "tune_kept_n",
            "tune_kept",
            "feature_names",
        ),
    },
    "intraday": {
        "model": "lstm_intraday",
        "spec_dir": "causal_tune_lstm_intraday",
        "npz": "lstm_intraday_bar1600.npz",
        "rows": (
            "date",
            "pred_adj",
            "true_adj",
            "true_raw",
            "baseline",
            "e2_adj",
            "pred_adj_seeds",
            "pred_adj_qsel",
            "pred_adj_qsel_seeds",
            "kept_n",
            "kept",
            "kept_n_qsel",
            "kept_qsel",
            "qsel_refit",
            "in_force",
        ),
        "tune": (
            "tune_row",
            "tune_split",
            "tune_kept",
            "cand_val_mse",
            "cand_val_qlike",
            "cand_seed_val_mse",
            "cand_epochs",
            "chosen_mse",
            "chosen_qlike",
            "step_names",
        ),
    },
}
CACHE_ARRAYS = ("val_mse", "val_qlike", "seed_val_mse", "epochs", "keep_mask", "split")


def same(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    """(bit-identical, max |a - b| for numeric arrays else nan)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, float("nan")
    if a.dtype.kind == "f":
        eq = bool(
            np.array_equal(
                a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}")
            )
        )
        diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        return eq, diff
    if a.dtype.kind in "iub":
        return bool(np.array_equal(a, b)), float(
            np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))
        ) if a.size else 0.0
    return bool(np.array_equal(a, b)), float("nan")


def chunk_dir(root: Path, kind: str, bucket: str, chunk: str) -> Path:
    k = KIND[kind]
    return (
        root
        / bucket
        / "bar1600"
        / k["model"]
        / "tw2000"
        / "chunks"
        / f"c{chunk}"
        / k["spec_dir"]
        / k["model"]
        / bucket
    )


def node_of(chunk_out: Path) -> str:
    """The chunk dir's NODE file (CPU class), one line."""
    p = chunk_out.parents[2] / "NODE"
    return (
        " | ".join(p.read_text(encoding="utf-8").split()) if p.is_file() else "no NODE"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(KIND), required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--gate-root", required=True)
    ap.add_argument("--stored-root", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    k = KIND[a.kind]
    groot, sroot = Path(a.gate_root), Path(a.stored_root)
    rows: list[dict] = []
    for line in Path(a.tasks).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        bucket, chunk, start, end, halo = (
            parts[0],
            parts[4],
            int(parts[5]),
            int(parts[6]),
            int(parts[7]),
        )
        n = end - start
        G, S = (
            chunk_dir(groot, a.kind, bucket, chunk),
            chunk_dir(sroot, a.kind, bucket, chunk),
        )
        key = {
            "kind": a.kind,
            "bucket": bucket,
            "chunk": int(chunk),
            "rows": f"[{start - halo}, {end - halo})",
        }

        def emit(
            item: str, ok: bool, diff: float = float("nan"), note: str = ""
        ) -> None:
            rows.append(
                key
                | {
                    "item": item,
                    "bit_identical": bool(ok),
                    "max_abs_diff": diff,
                    "note": note,
                }
            )

        if not (G / "results_bar1600.csv").is_file():
            emit("gate run output", False, note=f"missing {G}")
            continue
        for csv in ("results_bar1600.csv", "results_qsel_bar1600.csv"):
            g = pd.read_csv(G / csv, dtype=str, keep_default_na=False)
            s = (
                pd.read_csv(S / csv, dtype=str, keep_default_na=False)
                .iloc[:n]
                .reset_index(drop=True)
            )
            assert len(g) == n, (csv, len(g), n)
            for c in CSV_COLS:
                ok = bool((g[c].to_numpy() == s[c].to_numpy()).all())
                diff = (
                    float(
                        np.max(
                            np.abs(
                                g[c].astype(float).to_numpy()
                                - s[c].astype(float).to_numpy()
                            )
                        )
                    )
                    if c != "date"
                    else float("nan")
                )
                emit(f"{csv}:{c} (text)", ok, diff)
            emit(
                f"{csv}:pred_raw (look-ahead Duan, not gated)",
                True,
                float(
                    np.max(
                        np.abs(
                            g["pred_raw"].astype(float).to_numpy()
                            - s["pred_raw"].astype(float).to_numpy()
                        )
                    )
                ),
                note="reported only",
            )
        with (
            np.load(G / k["npz"], allow_pickle=False) as zg,
            np.load(S / k["npz"], allow_pickle=False) as zs,
        ):
            for name in k["rows"]:
                ok, diff = same(zg[name][:n], zs[name][:n])
                emit(f"npz:{name}[:{n}]", ok, diff)
            for name in k["tune"]:
                ok, diff = same(zg[name], zs[name])
                emit(f"npz:{name}", ok, diff)
            mg, ms = json.loads(str(zg["meta"])), json.loads(str(zs["meta"]))
            for field in ("versions", "grid", "window_mask", "qsel"):
                emit(
                    f"meta:{field}",
                    mg.get(field) == ms.get(field),
                    note=json.dumps(mg.get(field), default=str)[:160],
                )
            tune_rows = [int(r) for r in zg["tune_row"]]
        if (
            a.kind == "intraday"
        ):  # the stage-1 record the stored chunk read vs the one the gate wrote
            for R in tune_rows:
                cg = groot / "tune_cache" / bucket / f"tune_R{R:05d}.npz"
                cs = sroot / "tune_cache" / bucket / f"tune_R{R:05d}.npz"
                with (
                    np.load(cg, allow_pickle=False) as zg,
                    np.load(cs, allow_pickle=False) as zs,
                ):
                    for name in CACHE_ARRAYS:
                        ok, diff = same(zg[name], zs[name])
                        emit(f"tune_cache R{R}:{name}", ok, diff)
                    sg, ss = (
                        json.loads(str(zg["scalars"])),
                        json.loads(str(zs["scalars"])),
                    )
                    for f in ("R", "pick_mse", "pick_qlike"):
                        emit(
                            f"tune_cache R{R}:{f}",
                            sg[f] == ss[f],
                            note=f"{sg[f]} vs {ss[f]}",
                        )
                    fg, fs = (
                        json.loads(str(zg["fingerprint"])),
                        json.loads(str(zs["fingerprint"])),
                    )
                    bad = sorted(x for x in set(fg) | set(fs) if fg.get(x) != fs.get(x))
                    emit(
                        f"tune_cache R{R}:fingerprint",
                        not bad,
                        note=f"differs: {bad}"
                        if bad
                        else "sha_y, sha_P, sha_ends, grid, torch equal",
                    )
        emit("cpu", True, note=f"gate: {node_of(G)} || stored: {node_of(S)}")
    tab = pd.DataFrame(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(a.out, index=False)
    bad = tab[~tab["bit_identical"]]
    for _, r in tab[tab["item"] == "cpu"].iterrows():
        print(f"{r['bucket']} c{r['chunk']} {r['rows']}: {r['note']}")
    print(
        tab.groupby(["bucket", "chunk"])["bit_identical"]
        .agg(["size", "sum"])
        .to_string()
    )
    if len(bad):
        print(bad.to_string(max_colwidth=120))
        sys.exit(f"REPRODUCTION GATE FAILED: {len(bad)} of {len(tab)} items differ")
    print(
        f"REPRODUCTION GATE PASS: {len(tab)} items bit-identical ({a.kind}, {tab['bucket'].nunique()} buckets)"
    )


if __name__ == "__main__":
    main()
