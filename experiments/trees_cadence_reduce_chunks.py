"""Merge the time chunks of the UNTUNED per-bar tree arms (checklist I2, rungs T10 and T1).

The 16:00 cadence campaign (writeup/CAMPAIGN_16H_2026-09-29.md) runs
``specs/causal_tune_trees.py`` as time chunks through run_executor's START / END /
HALO seam (HALO = the training window, so a chunk's X and y are exactly rows of the
whole series' arrays), each chunk in

  <root>/<bucket>/<seg>/<model>/tw<TW>/chunks/c<k>/causal_tune_trees/<model>/<bucket>/

This script writes, for every arm whose expected chunks (from the task file) all
carry DONE, the arm in the UNCHUNKED layout the stackers and scorers read,

  <root>/<bucket>/<seg>/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/
    results_<seg>.csv, trees_<seg>.npz, MERGED

Everything the fit produced is concatenated in chunk order (forecasts, TreeSHAP,
importance; refit rows are made whole-series rows).  The executor's look-ahead Duan
pred_raw (one smear over the rows of a run) is the one quantity that is not
chunk-local: it is recomputed over the whole arm with
``reduce_trees_tuned_chunks.backed_out_baseline`` (the tuned campaign's reducer), so
it equals the unchunked executor's to float round-off.  Nothing downstream scores on
it except the scorers' "as scored" column.

Rungs (the cadence axes of specs/causal_tune_trees.py, set per task line):
  t10  REFIT_EVERY 10, IMPORTANCE_EVERY 1   (the shipped cadence)
  t1   REFIT_EVERY 1,  IMPORTANCE_EVERY 10  (a refit every session; importance and
                                             TreeSHAP at every 10th refit)

Gates (reduce_gates.csv under --root; a failing arm is not merged, exit status 1):
  tiling       chunk k's OOS offset (its task line's START - HALO, and the npz meta
               slice) = the rows of the chunks before it; rows = END - START
  stamps       strictly increasing across chunks; npz and CSV of each chunk carry the
               same stamps and pred_adj
  run keys     every chunk ran with the rung's REFIT_EVERY / IMPORTANCE_EVERY, ONE
               thread per fit (meta threads = 1), the same model, bucket, segment,
               training window and feature names
  refit rows   the merged refit rows are the whole series' refit anchors
               (0, RE, 2 RE, ...: every chunk starts on one)
  importance   a refit's importance row is finite exactly when its whole-series refit
               index is a multiple of IMPORTANCE_EVERY (NaN otherwise)
  TreeSHAP     (when recorded) a forecast row's SHAP row is finite exactly when the
               refit in force recorded importance

Usage: python experiments/trees_cadence_reduce_chunks.py --rung t1
           --root results/linear_subsection_trees_dedup/t1
           --tasks cluster/trees_cadence_tasks_untuned.txt
Task line: "<pack> <rung> <bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>".
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
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from reduce_trees_tuned_chunks import backed_out_baseline  # noqa: E402

SPEC_DIR = "causal_tune_trees"
# rung -> (REFIT_EVERY, IMPORTANCE_EVERY); the campaign plan's T10 / T1
RUNGS = {"t10": (10, 1), "t1": (1, 10)}
CONCAT_1D = (
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "baseline",
    "e2_adj",
    "fit_sec",
    "shap_sec",
)


def resolve(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def expected_arms(
    task_file: Path, rung: str
) -> dict[tuple, list[tuple[int, int, int, int]]]:
    """{(bucket, seg, model, tw): [(chunk, start, end, halo), ...]} for one rung."""
    arms: dict[tuple, list] = {}
    for line in task_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        _pack, r, bucket, model, tw, seg, chunk, start, end, halo = parts
        if r != rung:
            continue
        arms.setdefault((bucket, seg, model, int(tw)), []).append(
            (int(chunk), int(start), int(end), int(halo))
        )
    return {k: sorted(set(v)) for k, v in arms.items()}


def arm_top(root: Path, bucket: str, seg: str, model: str, tw: int) -> Path:
    return root / bucket / seg / model / f"tw{tw}"


def inner(d: Path, model: str, bucket: str) -> Path:
    return d / SPEC_DIR / model / bucket


def load_chunk(d: Path, model: str, bucket: str, seg: str) -> tuple[pd.DataFrame, dict]:
    s = inner(d, model, bucket)
    r = pd.read_csv(s / f"results_{seg}.csv")
    with np.load(s / f"trees_{seg}.npz", allow_pickle=False) as z:
        npz = {k: z[k] for k in z.files}
    npz["meta"] = json.loads(str(npz["meta"]))
    return r, npz


def merge_arm(
    root: Path,
    key: tuple,
    chunks: list[tuple[int, int, int, int]],
    refit_every: int,
    importance_every: int,
    write: bool = True,
) -> dict:
    """Gate and merge one arm; returns its gate row (and, in memory, the merged arrays)."""
    bucket, seg, model, tw = key
    top = arm_top(root, bucket, seg, model, tw)
    gate = {
        "bucket": bucket,
        "segment": seg,
        "model": model,
        "train_win": tw,
        "refit_every": refit_every,
        "importance_every": importance_every,
        "chunks": len(chunks),
    }
    dirs = [top / "chunks" / f"c{c}" for c, *_ in chunks]
    missing = [
        f"c{c}" for (c, *_), d in zip(chunks, dirs) if not (d / "DONE").is_file()
    ]
    if missing:
        print(f"NOT MERGED {key}: chunks without DONE: {', '.join(missing)}")
        return gate | {"ok": False, "why": f"missing {len(missing)} chunks"}
    got = [load_chunk(d, model, bucket, seg) for d in dirs]
    csvs = [r for r, _ in got]
    npzs = [z for _, z in got]
    why: list[str] = []

    # tiling and the run keys
    first_off = chunks[0][1] - chunks[0][3]
    off = first_off
    offsets = []
    for (c, start, end, halo), r, z in zip(chunks, csvs, npzs):
        m = z["meta"]
        o = max(0, start - halo)  # the spec's own chunk offset (START - HALO)
        offsets.append(o)
        if o != off:
            why.append(f"c{c} offset {o} != rows before it {off}")
        if list(m["slice"]) != [start, end, halo]:
            why.append(
                f"c{c} meta slice {m['slice']} != task line {[start, end, halo]}"
            )
        if end >= 0 and len(r) != end - start:
            why.append(f"c{c} has {len(r)} rows, task says {end - start}")
        want = {
            "model": model,
            "bucket": bucket,
            "segment": seg,
            "train_win_days": tw,
            "refit_every": refit_every,
            "importance_every": importance_every,
            "threads": 1,
        }
        bad = {k: m.get(k) for k, v in want.items() if m.get(k) != v}
        if bad:
            why.append(f"c{c} meta {bad} (want {want})")
        same_dates = len(z["date"]) == len(r) and bool(
            (
                np.asarray(z["date"]).astype(str) == r["date"].astype(str).to_numpy()
            ).all()
        )
        if not same_dates or not np.array_equal(
            np.asarray(z["pred_adj"], float), r["pred_adj"].to_numpy(float)
        ):
            why.append(f"c{c} npz and CSV disagree")
        off += len(r)
    res = pd.concat(csvs, ignore_index=True)
    stamps = pd.to_datetime(res["date"])
    if not stamps.is_monotonic_increasing or stamps.duplicated().any():
        why.append("stamps not strictly increasing across chunks")
    if any(
        not np.array_equal(z["feature_names"], npzs[0]["feature_names"]) for z in npzs
    ):
        why.append("feature names differ across chunks")

    # refit rows, importance and SHAP cadence, on whole-series rows
    refit = np.concatenate(
        [
            np.asarray(z["refit_row"], np.int64) + o - first_off
            for z, o in zip(npzs, offsets)
        ]
    )
    n = len(res)
    if first_off % refit_every:
        why.append(f"first chunk offset {first_off} is not a refit row")
    elif not np.array_equal(refit, np.arange(0, n, refit_every)):
        why.append("merged refit rows are not the whole series' refit anchors")
    imp = np.vstack([np.asarray(z["importance"]) for z in npzs])
    recorded = ((first_off + refit) // refit_every) % importance_every == 0
    finite = np.isfinite(imp).all(axis=1)
    all_nan = np.isnan(imp).all(axis=1)
    if not (np.array_equal(finite, recorded) and np.array_equal(all_nan, ~recorded)):
        why.append(
            f"importance cadence: {int(finite.sum())} finite rows, {int(recorded.sum())} expected"
        )
    shaps = [np.asarray(z["shap"]) for z in npzs]
    has_shap = [s.size > 0 for s in shaps]
    if any(has_shap) and not all(has_shap):
        why.append("TreeSHAP recorded in some chunks only")
    shap = np.vstack(shaps) if all(has_shap) else np.zeros((0, 0), np.float32)
    if shap.size:
        in_force = np.searchsorted(refit, np.arange(n), side="right") - 1
        want_row = recorded[in_force]
        row_ok = np.isfinite(shap).all(axis=1)
        if not np.array_equal(row_ok, want_row):
            why.append(
                f"TreeSHAP cadence: {int(row_ok.sum())} finite rows, {int(want_row.sum())} expected"
            )
    if why:
        print(f"GATE FAIL {key}: " + "; ".join(why))
        return gate | {"ok": False, "why": "; ".join(why), "rows": n}

    B = np.concatenate([backed_out_baseline(r) for r in csvs])
    smear = float(np.mean((res["true_adj"] - res["pred_adj"]) ** 2))
    res["pred_raw"] = (res["pred_adj"] ** 2 + smear) * B
    out: dict = {k: np.concatenate([np.asarray(z[k]) for z in npzs]) for k in CONCAT_1D}
    out["refit_row"] = refit
    out["importance"] = imp.astype(np.float32)
    out["shap"] = shap
    out["shap_additivity_gap"] = max(float(z["shap_additivity_gap"]) for z in npzs)
    out["feature_names"] = npzs[0]["feature_names"]
    metas = [z["meta"] for z in npzs]
    meta = dict(metas[0])
    meta["slice"] = [chunks[0][1], chunks[-1][2], chunks[0][3]]
    meta["walk_sec"] = sum(m["walk_sec"] for m in metas)
    meta["run_sec"] = sum(m["run_sec"] for m in metas)
    meta["chunks"] = [
        {
            "chunk": c,
            "start": s,
            "end": e,
            "halo": h,
            "rows": len(r),
            "run_sec": m["run_sec"],
        }
        for (c, s, e, h), r, m in zip(chunks, csvs, metas)
    ]
    out["meta"] = json.dumps(meta)
    assert np.array_equal(out["pred_adj"], res["pred_adj"].to_numpy(float))
    stats = {
        "rows": n,
        "refits": len(refit),
        "importance_rows": int(recorded.sum()),
        "shap_rows": int(np.isfinite(shap).all(axis=1).sum()) if shap.size else 0,
        "fit_sec_sum": float(np.sum(out["fit_sec"])),
        "fit_sec_mean": float(np.mean(out["fit_sec"])),
        "shap_sec_sum": float(np.sum(out["shap_sec"])),
        "run_sec_sum": meta["run_sec"],
        "run_sec_max_chunk": max(m["run_sec"] for m in metas),
        "n_features": meta["n_features"],
        "starts_at_series_start": first_off == 0,
    }
    if write:
        dst = inner(top, model, bucket)
        dst.mkdir(parents=True, exist_ok=True)
        res.to_csv(dst / f"results_{seg}.csv", index=False)
        np.savez_compressed(dst / f"trees_{seg}.npz", **out)
        fh = inner(dirs[0], model, bucket) / f"results_{seg}_feature_health.csv"
        if fh.is_file():
            shutil.copyfile(fh, dst / f"results_{seg}_feature_health.csv")
        (dst / "MERGED").write_text(json.dumps(meta["chunks"]), encoding="utf-8")
        print(f"merged {key}: {len(chunks)} chunks, {n} rows, {len(refit)} refits")
    return gate | {"ok": True, "why": ""} | stats | {"_merged": (res, out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument(
        "--root",
        required=True,
        help="the rung's root, e.g. results/linear_subsection_trees_dedup/t1",
    )
    ap.add_argument("--tasks", required=True, help="the untuned task file")
    a = ap.parse_args()
    root = resolve(a.root)
    re_, ie = RUNGS[a.rung]
    arms = expected_arms(resolve(a.tasks), a.rung)
    print(
        f"{len(arms)} arms of rung {a.rung} (REFIT_EVERY {re_}, IMPORTANCE_EVERY {ie}) under {root}"
    )
    rows = []
    for k, v in sorted(arms.items()):
        g = merge_arm(root, k, v, re_, ie)
        g.pop("_merged", None)
        rows.append(g)
    tab = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    tab.to_csv(root / "reduce_gates.csv", index=False)
    n_ok = int(tab["ok"].sum())
    print(f"merged {n_ok} of {len(tab)} arms; gates in {root / 'reduce_gates.csv'}")
    failed = tab[~tab["ok"] & ~tab["why"].str.startswith("missing")]
    if len(failed) or n_ok < len(tab):
        sys.exit(
            f"{len(failed)} arm(s) failed a reduce gate, {len(tab) - n_ok - len(failed)} incomplete"
        )


if __name__ == "__main__":
    main()
