"""Merge the time chunks of the tuned per-bar tree arms into the unchunked layout.

The tuned campaign (specs/causal_tune_trees_tuned.py) runs every arm as time
chunks through run_executor's START / END / HALO seam (one tuning period per
chunk, HALO = the training window), each chunk in

  <root>/<bucket>/<seg>/<model>/tw<TW>/chunks/c<k>/causal_tune_trees/<model>/<bucket>/

This script writes, for every arm whose expected chunks (from the task files)
all carry DONE, the arm in the UNCHUNKED layout the scorers read,

  <root>/<bucket>/<seg>/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/
    results_<seg>.csv, results_qsel_<seg>.csv, trees_<seg>.npz,
    tune_trace_<seg>.csv, tune_candidates_<seg>.csv, grid_<seg>.json, MERGED

Everything the fit produced is concatenated in chunk order (forecasts, SHAP,
importance, tuning records; refit rows are made whole-series rows).  The one
quantity that is NOT chunk-local is the executor's look-ahead Duan pred_raw
(smear = mean squared adjusted error over the rows of the run): it is recomputed
over the whole arm, with each row's baseline B backed out of the chunk's own
pred_raw (B = pred_raw / (pred_adj^2 + chunk smear)), so it equals the
unchunked executor's to float round-off.  Nothing downstream scores on it except
the scorer's "as scored" column; the causal clock back-transform uses pred_adj.

Gates (reduce_gates.csv; a failing arm is not merged and the exit status is
non-zero): the chunks tile the arm's whole-series OOS rows contiguously
(oos_offset of chunk k = the sum of the previous chunks' rows = its task line's
START - HALO); stamps strictly increase across chunks; every chunk has the same
grid (candidate list), feature names and model; every tuning row is a multiple
of TUNE_PER; the npz and CSV of each chunk carry the same stamps and pred_adj.

Usage: python experiments/reduce_trees_tuned_chunks.py
           [--root results/linear_subsection_trees_tuned]
           --tasks cluster/treestuned_tasks_gb.txt --tasks cluster/treestuned_tasks_rf.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = "causal_tune_trees"
CONCAT_1D = (
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "baseline",
    "e2_adj",
    "fit_sec",
    "shap_sec",
    "qsel_sec",
    "pred_adj_qsel",
    "tune_row",
    "tune_sec",
    "chosen_mse",
    "chosen_qlike",
)
STACK_2D = ("importance", "tune_split", "cand_val_mse", "cand_val_qlike", "cand_rounds", "cand_fit_sec")


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def expected_arms(task_files: list[Path]) -> dict[tuple, list[tuple[int, int, int, int]]]:
    """{(bucket, seg, model, tw): [(chunk, start, end, halo), ...]} from the task lines."""
    arms: dict[tuple, list] = {}
    for tf in task_files:
        for line in tf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            bucket, models, tw, seg, chunk, start, end, halo = parts
            for model in models.split(","):
                arms.setdefault((bucket, seg, model, int(tw)), []).append(
                    (int(chunk), int(start), int(end), int(halo))
                )
    for k in arms:
        arms[k] = sorted(set(arms[k]))
    return arms


def arm_dir(root: Path, bucket: str, seg: str, model: str, tw: int) -> Path:
    return root / bucket / seg / model / f"tw{tw}"


def inner(d: Path, model: str, bucket: str) -> Path:
    return d / SPEC_DIR / model / bucket


def backed_out_baseline(r: pd.DataFrame) -> np.ndarray:
    smear = float(np.mean((r["true_adj"] - r["pred_adj"]) ** 2))
    return (r["pred_raw"] / (r["pred_adj"] ** 2 + smear)).to_numpy(float)


def merge_arm(root: Path, key: tuple, chunks: list[tuple[int, int, int, int]]) -> dict:
    bucket, seg, model, tw = key
    top = arm_dir(root, bucket, seg, model, tw)
    gate = {"bucket": bucket, "segment": seg, "model": model, "train_win": tw, "chunks": len(chunks)}
    dirs = [top / "chunks" / f"c{c}" for c, *_ in chunks]
    missing = [f"c{c}" for (c, *_), d in zip(chunks, dirs) if not (d / "DONE").is_file()]
    if missing:
        print(f"NOT MERGED {key}: chunks without DONE: {', '.join(missing)}")
        return gate | {"ok": False, "why": f"missing {len(missing)} chunks"}
    csvs, qsels, npzs, traces, cands, grids = [], [], [], [], [], []
    for d in dirs:
        s = inner(d, model, bucket)
        csvs.append(pd.read_csv(s / f"results_{seg}.csv"))
        qsels.append(pd.read_csv(s / f"results_qsel_{seg}.csv"))
        with np.load(s / f"trees_{seg}.npz", allow_pickle=False) as z:
            npzs.append({k: z[k] for k in z.files})
        traces.append(pd.read_csv(s / f"tune_trace_{seg}.csv"))
        cands.append(pd.read_csv(s / f"tune_candidates_{seg}.csv"))
        grids.append(json.loads((s / f"grid_{seg}.json").read_text(encoding="utf-8")))
    why: list[str] = []
    # tiling: each chunk starts where the previous one ended, at its task line's START - HALO
    first_off = chunks[0][1] - chunks[0][3]
    off = first_off
    for (c, start, end, halo), r, z in zip(chunks, csvs, npzs):
        o = int(z["oos_offset"])
        if o != off:
            why.append(f"c{c} oos_offset {o} != {off}")
        if start - halo != o:
            why.append(f"c{c} oos_offset {o} != START - HALO {start - halo}")
        if end >= 0 and len(r) != end - start:
            why.append(f"c{c} has {len(r)} rows, task says {end - start}")
        if not (np.asarray(z["date"]).astype(str) == r["date"].astype(str).to_numpy()).all() or not np.array_equal(
            np.asarray(z["pred_adj"], float), r["pred_adj"].to_numpy(float)
        ):
            why.append(f"c{c} npz and CSV disagree")
        off += len(r)
    res = pd.concat(csvs, ignore_index=True)
    stamps = pd.to_datetime(res["date"])
    if not stamps.is_monotonic_increasing or stamps.duplicated().any():
        why.append("stamps not strictly increasing across chunks")
    if any(g["candidates"] != grids[0]["candidates"] for g in grids):
        why.append("candidate lists differ across chunks")
    if any(not np.array_equal(z["feature_names"], npzs[0]["feature_names"]) for z in npzs):
        why.append("feature names differ across chunks")
    tune_per = int(grids[0]["tune_per"])
    for t, z in zip(traces, npzs):  # run-relative rows -> rows of the merged table
        t["tune_row"] = t["tune_row"] + int(z["oos_offset"]) - first_off
        t["chunk_offset"] = first_off
    tr = pd.concat(traces, ignore_index=True)
    if ((tr["tune_row"] + first_off) % tune_per).any():
        why.append("a tuning row is not a whole-series multiple of TUNE_PER")
    if why:
        print(f"GATE FAIL {key}: " + "; ".join(why))
        return gate | {"ok": False, "why": "; ".join(why), "rows": len(res)}

    dst = inner(top, model, bucket)
    dst.mkdir(parents=True, exist_ok=True)
    B = np.concatenate([backed_out_baseline(r) for r in csvs])
    smear = float(np.mean((res["true_adj"] - res["pred_adj"]) ** 2))
    res["pred_raw"] = (res["pred_adj"] ** 2 + smear) * B
    res.to_csv(dst / f"results_{seg}.csv", index=False)
    q = pd.concat(qsels, ignore_index=True)
    if not (q["date"].astype(str).to_numpy() == res["date"].astype(str).to_numpy()).all():
        print(f"GATE FAIL {key}: qsel stamps differ from the rule of record's")
        return gate | {"ok": False, "why": "qsel stamps", "rows": len(res)}
    q_smear = float(np.mean((q["true_adj"] - q["pred_adj"]) ** 2))
    q["pred_raw"] = (q["pred_adj"] ** 2 + q_smear) * B
    q.to_csv(dst / f"results_qsel_{seg}.csv", index=False)

    tr["tune_idx"] = np.arange(len(tr)) // 2  # two rules per tuning point
    tr.to_csv(dst / f"tune_trace_{seg}.csv", index=False)
    for c, z in zip(cands, npzs):
        c["tune_row"] = c["tune_row"] + int(z["oos_offset"]) - first_off
        c["chunk_offset"] = first_off
    cd = pd.concat(cands, ignore_index=True)
    cd["tune_idx"] = cd.groupby("tune_row", sort=False).ngroup()
    cd.to_csv(dst / f"tune_candidates_{seg}.csv", index=False)
    (dst / f"grid_{seg}.json").write_text(json.dumps(grids[0], indent=1), encoding="utf-8")
    fh = inner(dirs[0], model, bucket) / f"results_{seg}_feature_health.csv"
    if fh.is_file():
        shutil.copyfile(fh, dst / f"results_{seg}_feature_health.csv")

    out: dict = {}
    for k in CONCAT_1D:
        out[k] = np.concatenate([np.asarray(z[k]) for z in npzs])
    out["tune_row"] = np.concatenate(
        [np.asarray(z["tune_row"]) + int(z["oos_offset"]) - first_off for z in npzs]
    )
    for k in STACK_2D:
        parts = [np.asarray(z[k]) for z in npzs if np.asarray(z[k]).size]
        out[k] = np.vstack(parts) if parts else np.asarray(npzs[0][k])
    out["refit_row"] = np.concatenate(
        [np.asarray(z["refit_row"]) + int(z["oos_offset"]) - first_off for z in npzs]
    )
    shaps = [np.asarray(z["shap"]) for z in npzs]
    out["shap"] = np.vstack(shaps) if all(s.size for s in shaps) else np.zeros((0, 0), np.float32)
    out["shap_additivity_gap"] = max(float(z["shap_additivity_gap"]) for z in npzs)
    out["feature_names"] = npzs[0]["feature_names"]
    out["oos_offset"] = first_off  # 0 for a whole arm
    metas = [json.loads(str(z["meta"])) for z in npzs]
    meta = dict(metas[0])
    meta["oos_offset"] = first_off
    meta["walk_sec"] = sum(m["walk_sec"] for m in metas)
    meta["run_sec"] = sum(m["run_sec"] for m in metas)
    meta["chunks"] = [
        {"chunk": c, "start": s, "end": e, "halo": h, "rows": len(r), "slice": m["slice"]}
        for (c, s, e, h), r, m in zip(chunks, csvs, metas)
    ]
    out["meta"] = json.dumps(meta)
    np.savez_compressed(dst / f"trees_{seg}.npz", **out)
    assert np.array_equal(out["pred_adj"], res["pred_adj"].to_numpy(float))
    (dst / "MERGED").write_text(json.dumps(meta["chunks"]), encoding="utf-8")
    print(f"merged {key}: {len(chunks)} chunks, {len(res)} rows, {len(tr) // 2} tunings")
    return gate | {
        "ok": True,
        "why": "",
        "rows": len(res),
        "tunings": len(tr) // 2,
        "starts_at_series_start": first_off == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_trees_tuned")
    ap.add_argument("--tasks", action="append", required=True, help="task file(s), repeatable")
    a = ap.parse_args()
    root = resolve(a.root)
    arms = expected_arms([resolve(t) for t in a.tasks])
    print(f"{len(arms)} arms expected under {root}")
    gates = [merge_arm(root, k, v) for k, v in sorted(arms.items())]
    tab = pd.DataFrame(gates)
    tab.to_csv(root / "reduce_gates.csv", index=False)
    n_ok = int(tab["ok"].sum())
    print(f"merged {n_ok} of {len(tab)} arms; gates in {root / 'reduce_gates.csv'}")
    failed = tab[~tab["ok"] & ~tab["why"].str.startswith("missing")]
    if len(failed):
        sys.exit(f"{len(failed)} arm(s) failed a reduce gate")


if __name__ == "__main__":
    main()
