"""Merge the time chunks of the Optuna per-bar tree campaign (specs/causal_tune_trees_optuna.py).

Layout written by cluster/slurm/optuna_pack.sbatch under the campaign root R:

  stage 1  R/stage1/<bucket>/<seg>/<model>/tw<TW>/chunks/c<k>/causal_tune_trees/<model>/<bucket>/
             trials_<seg>.npz, trials_<seg>.csv, stage1_<seg>.csv        (+ chunks/c<k>/DONE)
  stage 2  R/stage2/<bucket>/<seg>/<model>/tw<TW>/chunks/c<k>/<path>/causal_tune_trees/<model>/<bucket>/
             results_<seg>.csv, trees_<seg>.npz                          (+ chunks/c<k>/DONE)

--stage 1 writes, for every arm whose chunks (from the task file) all carry DONE and tile the
arm's whole-series OOS rows 0 .. n_oos - 1 (the last chunk runs to the end of the series):

  R/stage1/<bucket>/<seg>/<model>/tw<TW>/trials_<seg>.npz   every tuning point, row order (the
                                                            input of stage 2: STAGE1_TRIALS)
  R/stage1/<bucket>/<seg>/<model>/tw<TW>/STAGE1_COMPLETE    the completeness flag stage 2 needs

--stage 2 writes, per arm and path, the UNTUNED layout the stackers and scorers read:

  R/paths/<path>/<bucket>/<seg>/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/
      results_<seg>.csv, trees_<seg>.npz, MERGED

Everything is concatenated in chunk order.  The one quantity that is not chunk-local is the
executor's look-ahead Duan pred_raw (smear = mean squared adjusted error over the run): it is
recomputed over the whole arm for every path, with each row's baseline B backed out of the
SHAP path's chunk table (the executor's own: B = pred_raw / (pred_adj^2 + chunk smear)).
Nothing downstream scores on it (the stackers carry pred_adj and B = true_raw / true_adj^2).

Gates (R/reduce_gates_stage<k>.csv; a failing arm is not merged, exit status non-zero):
  both     chunk k's OOS offset = the previous chunks' rows = its task line's START - HALO;
           rows per chunk = END - START; stamps strictly increase across chunks; the same
           model / bucket / axes / trial count / early stopping in every chunk
  stage 1  rows = 0 .. n_oos - 1 exactly; best-of-k picks = the recomputed argmins
  stage 2  the stage-1 arm is COMPLETE; every path's configuration rows (tuning point in
           force, trial) equal the rule recomputed here from the merged stage-1 records;
           the npz and CSV of each chunk carry the same stamps and pred_adj; every path has
           the SHAP path's stamps and targets

Usage: python experiments/reduce_trees_optuna.py --stage 1|2 [--root results/linear_subsection_trees_optuna_mask]
           --tasks cluster/optuna_tasks_s1.txt [--tasks ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = "causal_tune_trees"
DEFAULT_ROOT = "results/linear_subsection_trees_optuna_mask"
# path -> (TUNE_PER, rule, k): the spec's PATHS, restated and checked against every chunk's meta
PATHS: dict[str, tuple[int, str, int]] = {
    "tp1": (1, "mse", 50),
    "tp5": (5, "mse", 50),
    "tp25": (25, "mse", 50),
    "tp250": (250, "mse", 50),
    "tp1_k10": (1, "mse", 10),
    "tp1_k25": (1, "mse", 25),
    "tp25_k10": (25, "mse", 10),
    "tp25_k25": (25, "mse", 25),
    "tp25_q": (25, "qlike", 50),
}
SHAP_PATH = "tp1"
BUDGETS = (10, 25, 50)
RT = "round_trip"  # exact float parsing of the chunk CSVs (the default parser is off by an ULP at times)
S1_CONCAT = (
    "row",
    "seed",
    "split",
    "params",
    "val_mse",
    "val_qlike",
    "sec",
    "rounds",
    "rounds_max",
    "patience",
    "complete",
    "study_sec",
    "date",
    "n_kept",  # kept columns per tuning point (= p without the window mask)
)
S2_CONCAT_1D = (
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "baseline",
    "e2_adj",
    "fit_sec",
    "shap_sec",
    "cfg_point",
    "cfg_trial",
    "cfg_rounds",
    "n_kept",  # kept columns per refit (= p without the window mask)
)
S2_STACK_2D = ("importance", "cfg_params")


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def best_of(losses: np.ndarray, k: int) -> np.ndarray:
    """The spec's best_of: argmin of the first k trials, NaN as +inf, ties -> lower index."""
    v = np.asarray(losses, dtype=np.float64)[:, : min(k, np.shape(losses)[1])]
    return np.argmin(np.where(np.isnan(v), np.inf, v), axis=1).astype(np.int64)


def expected_arms(
    task_files: list[Path],
) -> dict[tuple, list[tuple[int, int, int, int]]]:
    """{(bucket, seg, model, tw): [(chunk, start, end, halo), ...]} from the task lines
    ("<bucket> <model[,model]> <tw> <seg> <chunk> <start> <end> <halo>")."""
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
        arms[k] = sorted(set(arms[k]), key=lambda c: c[1])
    return arms


def arm_top(root: Path, stage: int, bucket: str, seg: str, model: str, tw: int) -> Path:
    return root / f"stage{stage}" / bucket / seg / model / f"tw{tw}"


def savez_atomic(path: Path, **arrays) -> None:
    """np.savez_compressed to a temporary file, then os.replace: a reader never sees a
    partial file.  (2026-09-30: a re-merge rewrote an arm's trials npz in place while a
    stage-2 task of the first submission read it -- EOFError -- and the flag was deleted
    for the length of the re-merge, so another task refused; both chunks were re-run.)"""
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def tiling(chunks, rows_per_chunk, offsets) -> list[str]:
    why = []
    off = chunks[0][1] - chunks[0][3]
    for (c, start, end, halo), n, o in zip(chunks, rows_per_chunk, offsets):
        if o != off:
            why.append(f"c{c} oos_offset {o} != {off}")
        if start - halo != o:
            why.append(f"c{c} oos_offset {o} != START - HALO {start - halo}")
        if end >= 0 and n != end - start:
            why.append(f"c{c} has {n} rows, task says {end - start}")
        off += n
    return why


# --------------------------------------------------------------------------- stage 1
def merge_stage1(root: Path, key: tuple, chunks: list) -> dict:
    bucket, seg, model, tw = key
    top = arm_top(root, 1, bucket, seg, model, tw)
    gate = {
        "stage": 1,
        "bucket": bucket,
        "segment": seg,
        "model": model,
        "train_win": tw,
        "chunks": len(chunks),
    }
    flag = (
        top / "STAGE1_COMPLETE"
    )  # removed only when this merge fails (below), never mid-merge
    dirs = [top / "chunks" / f"c{c}" for c, *_ in chunks]
    missing = [
        f"c{c}" for (c, *_), d in zip(chunks, dirs) if not (d / "DONE").is_file()
    ]
    if missing:
        print(f"NOT MERGED stage 1 {key}: chunks without DONE: {', '.join(missing)}")
        flag.unlink(missing_ok=True)
        return gate | {
            "ok": False,
            "why": f"missing {len(missing)} chunks: {' '.join(missing[:20])}",
        }
    parts = [
        load_npz(d / SPEC_DIR / model / bucket / f"trials_{seg}.npz") for d in dirs
    ]
    metas = [json.loads(str(z["meta"])) for z in parts]
    why = tiling(
        chunks, [len(z["row"]) for z in parts], [int(z["oos_offset"]) for z in parts]
    )
    for f in (
        "model",
        "bucket",
        "segment",
        "train_win",
        "axes",
        "n_trials",
        "early_stop",
        "space",
        "identity",
        "window_mask",
    ):
        if any(m.get(f) != metas[0].get(f) for m in metas):
            why.append(f"meta {f} differs across chunks")
    if chunks[-1][2] >= 0:
        why.append("the last chunk does not run to the end of the series (END != -1)")
    out = {k: np.concatenate([np.asarray(z[k]) for z in parts]) for k in S1_CONCAT}
    n = len(out["row"])
    if not np.array_equal(out["row"], np.arange(n)):
        why.append("tuning rows are not 0 .. n_oos - 1")
    stamps = pd.to_datetime(pd.Series(out["date"]))
    if not stamps.is_monotonic_increasing or stamps.duplicated().any():
        why.append("stamps not strictly increasing across chunks")
    for k in BUDGETS:
        out[f"best_mse_k{k}"] = best_of(out["val_mse"], k)
        got = np.concatenate([z[f"best_mse_k{k}"] for z in parts])
        if not np.array_equal(got, out[f"best_mse_k{k}"]):
            why.append(f"best_mse_k{k} differs from the recomputed argmin")
    out["best_qlike_k50"] = best_of(out["val_qlike"], 50)
    if why:
        print(f"GATE FAIL stage 1 {key}: " + "; ".join(why))
        flag.unlink(missing_ok=True)
        return gate | {"ok": False, "why": "; ".join(why), "rows": n}
    meta = dict(metas[0])
    meta["oos_offset"] = 0
    meta["walk_sec"] = sum(m["walk_sec"] for m in metas)
    meta["run_sec"] = sum(m["run_sec"] for m in metas)
    meta["chunks"] = [
        {"chunk": c, "start": s, "end": e, "halo": h, "rows": len(z["row"])}
        for (c, s, e, h), z in zip(chunks, parts)
    ]
    savez_atomic(top / f"trials_{seg}.npz", oos_offset=0, meta=json.dumps(meta), **out)
    flag.write_text(json.dumps({"rows": n, "chunks": len(chunks)}), encoding="utf-8")
    print(
        f"merged stage 1 {key}: {len(chunks)} chunks, {n} tuning points x {out['val_mse'].shape[1]} trials"
    )
    return gate | {
        "ok": True,
        "why": "",
        "rows": n,
        "trials": int(out["val_mse"].shape[1]),
        "trial_core_h": float(out["sec"].sum() / 3600),
    }


# --------------------------------------------------------------------------- stage 2
def merge_stage2(root: Path, key: tuple, chunks: list) -> list[dict]:
    bucket, seg, model, tw = key
    top = arm_top(root, 2, bucket, seg, model, tw)
    base = {
        "stage": 2,
        "bucket": bucket,
        "segment": seg,
        "model": model,
        "train_win": tw,
        "chunks": len(chunks),
    }
    s1 = arm_top(root, 1, bucket, seg, model, tw)
    if not (s1 / "STAGE1_COMPLETE").is_file():
        print(f"NOT MERGED stage 2 {key}: stage 1 not complete")
        return [base | {"path": "*", "ok": False, "why": "stage 1 not complete"}]
    st = load_npz(s1 / f"trials_{seg}.npz")
    dirs = [top / "chunks" / f"c{c}" for c, *_ in chunks]
    missing = [
        f"c{c}" for (c, *_), d in zip(chunks, dirs) if not (d / "DONE").is_file()
    ]
    if missing:
        print(f"NOT MERGED stage 2 {key}: chunks without DONE: {', '.join(missing)}")
        return [
            base
            | {
                "path": "*",
                "ok": False,
                "why": f"missing {len(missing)} chunks: {' '.join(missing[:20])}",
            }
        ]
    inner = [lambda path, d=d: d / path / SPEC_DIR / model / bucket for d in dirs]
    # the SHAP path's chunk tables are the executor's own: B and the stamps / targets of every path
    ref_csv = [
        pd.read_csv(f(SHAP_PATH) / f"results_{seg}.csv", float_precision=RT)
        for f in inner
    ]
    B = np.concatenate(
        [
            (
                r["pred_raw"]
                / (
                    r["pred_adj"] ** 2
                    + float(np.mean((r["true_adj"] - r["pred_adj"]) ** 2))
                )
            ).to_numpy(float)
            for r in ref_csv
        ]
    )
    ref = pd.concat(ref_csv, ignore_index=True)
    out_rows = []
    for path, (tp, rule, k) in PATHS.items():
        gate = base | {"path": path}
        csvs = [
            pd.read_csv(f(path) / f"results_{seg}.csv", float_precision=RT)
            for f in inner
        ]
        npzs = [load_npz(f(path) / f"trees_{seg}.npz") for f in inner]
        metas = [json.loads(str(z["meta"])) for z in npzs]
        why = tiling(
            chunks, [len(r) for r in csvs], [int(z["oos_offset"]) for z in npzs]
        )
        for m in metas:
            if m["path"] != path or list(m["path_spec"]) != [tp, rule, k]:
                why.append(
                    f"chunk meta path {m['path']} {m['path_spec']} != {path} {[tp, rule, k]}"
                )
                break
        for c, r, z in zip(chunks, csvs, npzs):
            if not (
                np.asarray(z["date"]).astype(str) == r["date"].astype(str).to_numpy()
            ).all() or not np.array_equal(
                np.asarray(z["pred_adj"], float), r["pred_adj"].to_numpy(float)
            ):
                why.append(f"c{c[0]} npz and CSV disagree")
        res = pd.concat(csvs, ignore_index=True)
        if not (
            res["date"].astype(str).to_numpy() == ref["date"].astype(str).to_numpy()
        ).all() or not np.array_equal(
            res["true_adj"].to_numpy(float), ref["true_adj"].to_numpy(float)
        ):
            why.append("stamps / targets differ from the SHAP path's")
        stamps = pd.to_datetime(res["date"])
        if not stamps.is_monotonic_increasing or stamps.duplicated().any():
            why.append("stamps not strictly increasing across chunks")
        first_off = chunks[0][1] - chunks[0][3]
        rows = first_off + np.arange(len(res))
        cfg_point = np.concatenate([z["cfg_point"] for z in npzs])
        cfg_trial = np.concatenate([z["cfg_trial"] for z in npzs])
        want_point = (rows // tp) * tp
        picks = best_of(st["val_mse"] if rule == "mse" else st["val_qlike"], k)
        idx = {int(r): j for j, r in enumerate(st["row"])}
        want_trial = np.array([picks[idx[int(p)]] for p in want_point], dtype=np.int64)
        if not (
            np.array_equal(cfg_point, want_point)
            and np.array_equal(cfg_trial, want_trial)
        ):
            why.append(
                "configuration rows differ from the rule recomputed from stage 1"
            )
        if why:
            print(f"GATE FAIL stage 2 {key} {path}: " + "; ".join(why))
            out_rows.append(
                gate | {"ok": False, "why": "; ".join(why), "rows": len(res)}
            )
            continue
        dst = (
            root
            / "paths"
            / path
            / bucket
            / seg
            / model
            / f"tw{tw}"
            / SPEC_DIR
            / model
            / bucket
        )
        dst.mkdir(parents=True, exist_ok=True)
        smear = float(np.mean((res["true_adj"] - res["pred_adj"]) ** 2))
        res["pred_raw"] = (res["pred_adj"] ** 2 + smear) * B
        res.to_csv(dst / f"results_{seg}.csv", index=False)
        out: dict = {
            k2: np.concatenate([np.asarray(z[k2]) for z in npzs]) for k2 in S2_CONCAT_1D
        }
        for k2 in S2_STACK_2D:
            out[k2] = np.vstack([np.asarray(z[k2]) for z in npzs])
        shaps = [np.asarray(z["shap"]) for z in npzs]
        out["shap"] = (
            np.vstack(shaps)
            if all(s.size for s in shaps)
            else np.zeros((0, 0), np.float32)
        )
        out["shap_additivity_gap"] = max(float(z["shap_additivity_gap"]) for z in npzs)
        out["refit_row"] = rows - first_off
        out["oos_offset"] = first_off
        out["feature_names"] = npzs[0]["feature_names"]
        meta = dict(metas[0])
        meta["oos_offset"] = first_off
        meta["n_fits"] = sum(m["n_fits"] for m in metas)
        meta["walk_sec"] = sum(m["walk_sec"] for m in metas)
        meta["run_sec"] = sum(m["run_sec"] for m in metas)
        meta["chunks"] = [
            {"chunk": c, "start": s, "end": e, "halo": h, "rows": len(r)}
            for (c, s, e, h), r in zip(chunks, csvs)
        ]
        out["meta"] = json.dumps(meta)
        savez_atomic(dst / f"trees_{seg}.npz", **out)
        assert np.array_equal(out["pred_adj"], res["pred_adj"].to_numpy(float))
        (dst / "MERGED").write_text(json.dumps(meta["chunks"]), encoding="utf-8")
        print(f"merged stage 2 {key} {path}: {len(chunks)} chunks, {len(res)} rows")
        out_rows.append(
            gate
            | {
                "ok": True,
                "why": "",
                "rows": len(res),
                "fit_core_h": float(out["fit_sec"].sum() / 3600),
            }
        )
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument(
        "--tasks",
        action="append",
        required=True,
        help="task file(s) of the stage, repeatable",
    )
    a = ap.parse_args()
    root = resolve(a.root)
    arms = expected_arms([resolve(t) for t in a.tasks])
    print(f"{len(arms)} arms expected under {root} (stage {a.stage})")
    gates: list[dict] = []
    for k, v in sorted(arms.items()):
        if a.stage == 1:
            gates.append(merge_stage1(root, k, v))
        else:
            gates.extend(merge_stage2(root, k, v))
    tab = pd.DataFrame(gates)
    tab.to_csv(root / f"reduce_gates_stage{a.stage}.csv", index=False)
    print(
        f"stage {a.stage}: {int(tab['ok'].sum())} of {len(tab)} merged; gates in {root / f'reduce_gates_stage{a.stage}.csv'}"
    )
    failed = tab[
        ~tab["ok"] & ~tab["why"].str.startswith(("missing", "stage 1 not complete"))
    ]
    if len(failed):
        sys.exit(f"{len(failed)} arm(s) failed a reduce gate")


if __name__ == "__main__":
    main()
