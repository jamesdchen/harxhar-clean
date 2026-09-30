"""Merge the time chunks of the intraday-sequence LSTM arms into the unchunked layout.

The twin of experiments/reduce_lstm_chunks.py for specs/causal_tune_lstm_intraday.py.
Each chunk (one task line "<bucket> lstm_intraday <tw> <seg> <chunk> <start> <end> <halo>")
ran into

  <root>/<bucket>/<seg>/lstm_intraday/tw<TW>/chunks/c<k>/causal_tune_lstm_intraday/lstm_intraday/<bucket>/

and this script writes, for every arm whose chunks all carry DONE,

  <root>/<bucket>/<seg>/lstm_intraday/tw<TW>/causal_tune_lstm_intraday/lstm_intraday/<bucket>/
    results_<seg>.csv, results_qsel_<seg>.csv, lstm_intraday_<seg>.npz, tune_trace_<seg>.csv,
    tune_candidates_<seg>.csv, refit_trace_<seg>.csv, grid_<seg>.json, MERGED

Everything the fit produced is concatenated in chunk order (the spec writes whole-series
OOS rows everywhere, so nothing is re-indexed); a chunk owns the tuning points inside its
rows, so the tuning records concatenate without repeats.  The one quantity that is not
chunk-local, the executor's look-ahead Duan pred_raw, is recomputed over the whole arm
with each row's baseline B backed out of the chunk's own pred_raw (the per-bar reducer's
rule, imported).

Gates (reduce_gates.csv; a failing arm is not merged, the exit status is non-zero): the
chunks tile the arm's OOS rows contiguously from 0 (each chunk's oos_offset = its task
line's START - HALO = the previous chunks' rows); stamps strictly increase; every chunk
has the same grid (torch version included) and step names; the owned tuning rows are
exactly 0, TUNE_PER, 2 TUNE_PER, ... up to the last row, each once; each chunk's npz and
CSV carry the same stamps and pred_adj; every refit row's tuning point in force is
the last tuning row at or before it.

Usage: python experiments/reduce_lstm_intraday_chunks.py --root DIR --tasks FILE [--tasks FILE]
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

from reduce_lstm_chunks import backed_out_baseline, expected_arms  # noqa: E402

SPEC_DIR = "causal_tune_lstm_intraday"
MODEL = "lstm_intraday"
RT = "round_trip"
CONCAT = (  # row-wise concatenation in chunk order
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "baseline",
    "e2_adj",
    "pred_adj_seeds",
    "pred_adj_qsel",
    "pred_adj_qsel_seeds",
    "fit_sec",
    "qsel_sec",
    "kept_n",
    "kept",
    "kept_n_qsel",
    "kept_qsel",
    "qsel_refit",
    "in_force",
    "tune_row",
    "tune_split",
    "tune_sec",
    "tune_kept",
    "cand_val_mse",
    "cand_val_qlike",
    "cand_seed_val_mse",
    "cand_epochs",
    "cand_fit_sec",
    "chosen_mse",
    "chosen_qlike",
)


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def inner(d: Path, bucket: str) -> Path:
    return d / SPEC_DIR / MODEL / bucket


def merge_arm(root: Path, key: tuple, chunks: list[tuple[int, int, int, int]]) -> dict:
    bucket, seg, model, tw = key
    top = root / bucket / seg / model / f"tw{tw}"
    gate = {
        "bucket": bucket,
        "segment": seg,
        "model": model,
        "train_win": tw,
        "chunks": len(chunks),
    }
    dirs = [top / "chunks" / f"c{c}" for c, *_ in chunks]
    missing = [
        f"c{c}" for (c, *_), d in zip(chunks, dirs) if not (d / "DONE").is_file()
    ]
    if missing:
        print(f"NOT MERGED {key}: chunks without DONE: {', '.join(missing)}")
        return gate | {"ok": False, "why": f"missing {len(missing)} chunks"}
    csvs: list[pd.DataFrame] = []
    qsels: list[pd.DataFrame] = []
    npzs: list[dict] = []
    traces: list[pd.DataFrame] = []
    cands: list[pd.DataFrame] = []
    refits: list[pd.DataFrame] = []
    grids: list[dict] = []
    pa_default: list[np.ndarray] = []
    for d in dirs:
        s = inner(d, bucket)
        csvs.append(pd.read_csv(s / f"results_{seg}.csv", float_precision=RT))
        pa_default.append(
            pd.read_csv(s / f"results_{seg}.csv", usecols=["pred_adj"])[
                "pred_adj"
            ].to_numpy(float)
        )
        qsels.append(pd.read_csv(s / f"results_qsel_{seg}.csv", float_precision=RT))
        with np.load(s / f"lstm_intraday_{seg}.npz", allow_pickle=False) as z:
            npzs.append({k: z[k] for k in z.files})
        for lst, name in (
            (traces, "tune_trace"),
            (cands, "tune_candidates"),
            (refits, "refit_trace"),
        ):
            try:
                lst.append(pd.read_csv(s / f"{name}_{seg}.csv", float_precision=RT))
            except pd.errors.EmptyDataError:  # a chunk that owns no tuning point
                lst.append(pd.DataFrame())
        grids.append(json.loads((s / f"grid_{seg}.json").read_text(encoding="utf-8")))
    why: list[str] = []
    off = 0
    for (c, start, end, halo), r, z, pd0 in zip(chunks, csvs, npzs, pa_default):
        o = int(z["oos_offset"])
        if o != off:
            why.append(f"c{c} oos_offset {o} != {off}")
        if max(0, start - halo) != o:
            why.append(f"c{c} oos_offset {o} != START - HALO {start - halo}")
        if end >= 0 and len(r) != end - start:
            why.append(f"c{c} has {len(r)} rows, task says {end - start}")
        same = (
            np.asarray(z["date"]).astype(str) == r["date"].astype(str).to_numpy()
        ).all()
        if not same or not np.array_equal(np.asarray(z["pred_adj"], float), pd0):
            why.append(f"c{c} npz and CSV disagree")
        off += len(r)
    res = pd.concat(csvs, ignore_index=True)
    stamps = pd.to_datetime(res["date"])
    if not stamps.is_monotonic_increasing or stamps.duplicated().any():
        why.append("stamps not strictly increasing across chunks")
    if any(g != grids[0] for g in grids):
        why.append("grids (or torch versions) differ across chunks")
    if any(not np.array_equal(z["step_names"], npzs[0]["step_names"]) for z in npzs):
        why.append("step names differ across chunks")
    tune_per = int(grids[0]["tune_per"])
    tr = pd.concat([t for t in traces if len(t)], ignore_index=True)
    owned = np.concatenate([np.asarray(z["tune_row"], dtype=np.int64) for z in npzs])
    want = np.arange(0, len(res), tune_per)
    if not np.array_equal(owned, want):
        why.append(f"owned tuning rows {owned.tolist()} != {want.tolist()}")
    if len(tr) != 2 * len(want) or not np.array_equal(
        tr["tune_row"].to_numpy()[::2], want
    ):
        why.append("tune_trace rows do not match the tuning points")
    in_force = np.concatenate([np.asarray(z["in_force"]) for z in npzs])
    if not np.array_equal(in_force, (np.arange(len(res)) // tune_per) * tune_per):
        why.append(
            "a row's tuning point in force is not the last tuning row at or before it"
        )
    if why:
        print(f"GATE FAIL {key}: " + "; ".join(why))
        return gate | {"ok": False, "why": "; ".join(why), "rows": len(res)}

    dst = inner(top, bucket)
    dst.mkdir(parents=True, exist_ok=True)
    B = np.concatenate([backed_out_baseline(r) for r in csvs])
    smear = float(np.mean((res["true_adj"] - res["pred_adj"]) ** 2))
    res["pred_raw"] = (res["pred_adj"] ** 2 + smear) * B
    res.to_csv(dst / f"results_{seg}.csv", index=False)
    q = pd.concat(qsels, ignore_index=True)
    if not (
        q["date"].astype(str).to_numpy() == res["date"].astype(str).to_numpy()
    ).all():
        print(f"GATE FAIL {key}: qsel stamps differ from the rule of record's")
        return gate | {"ok": False, "why": "qsel stamps", "rows": len(res)}
    q_smear = float(np.mean((q["true_adj"] - q["pred_adj"]) ** 2))
    q["pred_raw"] = (q["pred_adj"] ** 2 + q_smear) * B
    q.to_csv(dst / f"results_qsel_{seg}.csv", index=False)
    tr.to_csv(dst / f"tune_trace_{seg}.csv", index=False)
    pd.concat([c for c in cands if len(c)], ignore_index=True).to_csv(
        dst / f"tune_candidates_{seg}.csv", index=False
    )
    pd.concat(refits, ignore_index=True).to_csv(
        dst / f"refit_trace_{seg}.csv", index=False
    )
    (dst / f"grid_{seg}.json").write_text(
        json.dumps(grids[0], indent=1), encoding="utf-8"
    )
    fh = inner(dirs[0], bucket) / f"results_{seg}_feature_health.csv"
    if fh.is_file():
        shutil.copyfile(fh, dst / f"results_{seg}_feature_health.csv")
    out: dict = {}
    for k in CONCAT:
        parts = [np.asarray(z[k]) for z in npzs if np.asarray(z[k]).size]
        out[k] = np.concatenate(parts, axis=0) if parts else np.asarray(npzs[0][k])
    out["step_names"] = npzs[0]["step_names"]
    out["oos_offset"] = 0
    metas = [json.loads(str(z["meta"])) for z in npzs]
    meta = dict(metas[0])
    meta["oos_offset"] = 0
    meta["walk_sec"] = sum(m["walk_sec"] for m in metas)
    meta["run_sec"] = sum(m["run_sec"] for m in metas)
    meta["chunks"] = [
        {
            "chunk": c,
            "start": s,
            "end": e,
            "halo": h,
            "rows": len(r),
            "slice": m["slice"],
            "workers": m["workers"],
        }
        for (c, s, e, h), r, m in zip(chunks, csvs, metas)
    ]
    out["meta"] = json.dumps(meta)
    np.savez_compressed(dst / f"lstm_intraday_{seg}.npz", **out)
    assert np.array_equal(out["pred_adj"], np.concatenate(pa_default))
    (dst / "MERGED").write_text(json.dumps(meta["chunks"]), encoding="utf-8")
    print(f"merged {key}: {len(chunks)} chunks, {len(res)} rows, {len(want)} tunings")
    return gate | {"ok": True, "why": "", "rows": len(res), "tunings": len(want)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_lstm_intraday")
    ap.add_argument(
        "--tasks", action="append", required=True, help="task file(s), repeatable"
    )
    a = ap.parse_args()
    root = resolve(a.root)
    arms = expected_arms([resolve(t) for t in a.tasks])
    print(f"{len(arms)} arms expected under {root}")
    tab = pd.DataFrame([merge_arm(root, k, v) for k, v in sorted(arms.items())])
    tab.to_csv(root / "reduce_gates.csv", index=False)
    print(
        f"merged {int(tab['ok'].sum())} of {len(tab)} arms; gates in {root / 'reduce_gates.csv'}"
    )
    failed = tab[~tab["ok"] & ~tab["why"].str.startswith("missing")]
    if len(failed):
        sys.exit(f"{len(failed)} arm(s) failed a reduce gate")


if __name__ == "__main__":
    main()
