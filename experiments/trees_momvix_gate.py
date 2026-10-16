"""Reproduction gate of the moments_vix tree workstream (2026-10-06): re-run a few refits of the
STORED masked 16:00 tree cells and require the stored outputs bit for bit, on the CPU class the
new bucket will run on.

The moments_vix bucket (src/data/loading.py SUBGROUPS["moments_vix"]: the 7 ES return moments +
the VIX level, on the HAR + calendar backbone) is fitted with exactly the machinery of the masked
de-duplicated 16:00 runs (writeup/CAMPAIGN_16H_2026-09-29.md, "Restart with a per-window mask"):
rungs T1 / T10 (specs/causal_tune_trees.py) and RS1 / RS10 (specs/causal_tune_trees_tuned.py),
WINDOW_MASK=1, bar1600, TRAIN_WIN 2000.  Those runs were computed on CARC epyc-7513 nodes; a
forecast is bit-reproducible only within a CPU vector class.  Before the new bucket runs on a
class, this script re-runs stored cells there with the code that will run the new bucket and
compares every recorded array with the stored chunk npz.

Gate line (--lines file, one per line):  "<rung> <bucket> <model> <chunk> <start> <nrows>"
  rung   t1 | t10 | rs1 | rs10 (the cadences of cluster/slurm/trees_mask_pack.sbatch and
         cluster/slurm/treestuned_mask_pack.sbatch: t10 REFIT_EVERY 10 / IMPORTANCE_EVERY 1,
         t1 1 / 10, rs10 REFIT_EVERY 10 + TreeSHAP at bar1600, rs1 REFIT_EVERY 1, TreeSHAP off)
  chunk  the stored chunk whose first rows are re-run (start = that chunk's START)
  nrows  how many rows from start (END = start + nrows, HALO = 2000)
The spec call is the pack scripts' own (same HPC_KW_* axes; untuned: SLURM_CPUS_PER_TASK=1, one
thread per fit; tuned: a pool of --pool spawned single-threaded workers, which the campaign's
gate showed changes no number).

Reference: <ref-root>/results/linear_subsection_trees_mask/<rung>/... (untuned) or
<ref-root>/results/linear_subsection_trees_tuned_mask/<rung>/... (tuned), chunk c<chunk>'s
causal_tune_trees/<model>/<bucket>/trees_bar1600.npz.  Compared BYTE FOR BYTE (dtype, shape,
bytes) on the re-run rows / refits / tuning point:
  per row     date, pred_adj, true_adj, true_raw, baseline, e2_adj, shap rows, pred_adj_qsel
  per refit   refit_row, importance, kept_n, kept
  per tuning  tune_row, tune_split, cand_val_mse, cand_val_qlike, cand_rounds, chosen_mse,
  point       chosen_qlike, tune_kept_n, tune_kept, oos_offset
  whole       feature_names; meta params / refit_every / importance_every / threads /
              window_mask / versions
Timing arrays (fit_sec, shap_sec, qsel_sec, tune_sec, cand_fit_sec) and the per-run maxima
(shap_additivity_gap) are not compared.

Mode --pair A B compares two chunk npz files of the SAME rows (e.g. the new bucket's canary on
CARC against the same chunk on Hoffman2) with the same key rules.

Output (--out): one run dir per line, gate_rows.csv (one row per line and key), gate_lines.csv
(one row per line), GATE_OK when every line matched, and PASS_<rung>_<model> per rung and
model whose lines all matched.  Exit status 0 iff every line matched.

Run:  python experiments/trees_momvix_gate.py --lines cluster/trees_momvix_gate_lines.txt
          --ref-root /scratch1/jc_905/harxhar-mask --out results/trees_momvix_gates/carc_epyc-7513
          [--pool 20] [--procs 20]
      python experiments/trees_momvix_gate.py --pair <a.npz> <b.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEG = "bar1600"
TW = 2000
RUNGS = {  # rung -> (spec, REFIT_EVERY, IMPORTANCE_EVERY or None, SHAP_SEGMENTS, stored root)
    "t10": ("specs/causal_tune_trees.py", 10, 1, SEG, "linear_subsection_trees_mask"),
    "t1": ("specs/causal_tune_trees.py", 1, 10, SEG, "linear_subsection_trees_mask"),
    "rs10": (
        "specs/causal_tune_trees_tuned.py",
        10,
        None,
        SEG,
        "linear_subsection_trees_tuned_mask",
    ),
    "rs1": (
        "specs/causal_tune_trees_tuned.py",
        1,
        None,
        "none",
        "linear_subsection_trees_tuned_mask",
    ),
}
ROW_KEYS = (
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "baseline",
    "e2_adj",
    "pred_adj_qsel",
    "shap",
)
REFIT_KEYS = ("refit_row", "importance", "kept_n", "kept")
TUNE_KEYS = (
    "tune_row",
    "tune_split",
    "cand_val_mse",
    "cand_val_qlike",
    "cand_rounds",
    "chosen_mse",
    "chosen_qlike",
    "tune_kept_n",
    "tune_kept",
)
WHOLE_KEYS = ("feature_names", "oos_offset")
META_KEYS = (
    "params",
    "refit_every",
    "importance_every",
    "threads",
    "window_mask",
    "versions",
)
SKIP = {
    "fit_sec",
    "shap_sec",
    "qsel_sec",
    "tune_sec",
    "cand_fit_sec",
    "shap_additivity_gap",
    "meta",
}


def spec_env(
    rung: str, bucket: str, model: str, start: int, end: int, out: Path, pool: int
) -> dict:
    """The environment of the pack scripts' spec call (cluster/slurm/trees_mask_pack.sbatch
    run_line, cluster/slurm/treestuned_mask_pack.sbatch) for one chunk."""
    _spec, re_, ie, shapseg, _root = RUNGS[rung]
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        TQDM_DISABLE="1",
        PYTHONUNBUFFERED="1",
        HPC_KW_MODEL=model,
        HPC_KW_EXOG_BUCKET=bucket,
        HPC_KW_SEGMENT=SEG,
        HPC_KW_TRAIN_WIN=str(TW),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_HAR_BASE="",
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(TW),
        HPC_KW_REFIT_EVERY=str(re_),
        HPC_KW_SHAP_SEGMENTS=shapseg,
        HPC_KW_WINDOW_MASK="1",
        HPC_RESULT_DIR=str(out),
    )
    if (
        ie is not None
    ):  # untuned: one thread per fit (the pack overrides SLURM_CPUS_PER_TASK to 1)
        env.update(SLURM_CPUS_PER_TASK="1", HPC_KW_IMPORTANCE_EVERY=str(ie))
    else:  # tuned: a pool of single-threaded workers
        env.update(
            SLURM_CPUS_PER_TASK=str(pool), HPC_KW_TUNE_IDENTITY="0", HPC_KW_QSEL="1"
        )
    # numba's JIT cache (no effect on any number): a short local path, like the packs' $TMPDIR one
    env["NUMBA_CACHE_DIR"] = os.path.join(
        tempfile.gettempdir(), f"numba_gate_{os.getpid()}_{rung}_{model}_{start}"
    )
    return env


def npz_of(chunk_dir: Path, model: str, bucket: str) -> Path:
    return chunk_dir / "causal_tune_trees" / model / bucket / f"trees_{SEG}.npz"


def same(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes():
        return True, ""
    if a.dtype.kind == "f":
        fa, fb = a.astype(np.float64), b.astype(np.float64)
        both = np.isfinite(fa) & np.isfinite(fb)
        nan_mismatch = int((np.isnan(fa) != np.isnan(fb)).sum())
        d = np.abs(fa[both] - fb[both])
        s = np.maximum(np.abs(fa[both]), np.abs(fb[both]))
        rel = float(np.max(np.where(s > 0, d / s, 0.0))) if d.size else 0.0
        return (
            False,
            f"{int((d > 0).sum())} values differ, max rel {rel:.3g}, nan mismatch {nan_mismatch}",
        )
    return False, f"{int((a != b).sum())} entries differ"


def compare(new: Path, ref: Path, nrows: int | None) -> list[dict]:
    """Every compared key of new against ref (ref sliced to new's rows / refits / tuning points)."""
    zn, zr = np.load(new, allow_pickle=False), np.load(ref, allow_pickle=False)
    n = len(zn["pred_adj"]) if nrows is None else nrows
    rows: list[dict] = []

    def put(key: str, ok: bool, why: str) -> None:
        rows.append({"key": key, "identical": ok, "detail": why})

    if len(zn["pred_adj"]) != n:
        put("rows", False, f"new run has {len(zn['pred_adj'])} rows, expected {n}")
    for k in ROW_KEYS:
        if k not in zn.files and k not in zr.files:
            continue
        if (k in zn.files) != (k in zr.files):
            put(k, False, "present on one side only")
            continue
        a, b = zn[k], zr[k]
        if a.size == 0 and b.size == 0:
            continue
        put(k, *same(a, b[: len(a)]))
    m = len(zn["refit_row"])
    for k in REFIT_KEYS:
        if (k in zn.files) != (k in zr.files):
            put(k, False, "present on one side only")
            continue
        if k in zn.files:
            put(k, *same(zn[k], zr[k][:m]))
    if "tune_row" in zn.files:
        t = len(zn["tune_row"])
        for k in TUNE_KEYS:
            if (k in zn.files) != (k in zr.files):
                put(k, False, "present on one side only")
                continue
            if k in zn.files:
                put(k, *same(zn[k], zr[k][:t]))
    for k in WHOLE_KEYS:
        if (k in zn.files) != (k in zr.files):
            put(k, False, "present on one side only")
            continue
        if k in zn.files:
            put(k, *same(zn[k], zr[k]))
    mn, mr = json.loads(str(zn["meta"])), json.loads(str(zr["meta"]))
    for k in META_KEYS:
        if k in mn or k in mr:
            put(
                f"meta.{k}",
                mn.get(k) == mr.get(k),
                "" if mn.get(k) == mr.get(k) else f"{mn.get(k)!r} vs {mr.get(k)!r}",
            )
    other = sorted(
        (set(zn.files) | set(zr.files))
        - set(ROW_KEYS)
        - set(REFIT_KEYS)
        - set(TUNE_KEYS)
        - set(WHOLE_KEYS)
        - SKIP
    )
    for k in other:
        put(k, False, "key not covered by the gate (unknown)")
    return rows


def run_line(line: str, ref_root: Path, out: Path, pool: int) -> dict:
    rung, bucket, model, chunk, start, nrows = line.split()
    chunk_i, start_i, n = int(chunk), int(start), int(nrows)
    spec, *_rest, stored = RUNGS[rung]
    tag = f"{rung}_{bucket}_{model}_c{chunk_i}_n{n}"
    run_dir = out / tag
    ref = npz_of(
        ref_root
        / "results"
        / stored
        / rung
        / bucket
        / SEG
        / model
        / f"tw{TW}"
        / "chunks"
        / f"c{chunk_i}",
        model,
        bucket,
    )
    rec = {
        "line": tag,
        "rung": rung,
        "bucket": bucket,
        "model": model,
        "chunk": chunk_i,
        "start": start_i,
        "nrows": n,
    }
    if not ref.is_file():
        return rec | {"ok": False, "why": f"no stored chunk npz {ref}", "seconds": 0.0}
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(run_dir / "run.log", "w", encoding="utf-8") as log:
        p = subprocess.run(
            [sys.executable, "-u", spec],
            cwd=ROOT,
            env=spec_env(rung, bucket, model, start_i, start_i + n, run_dir, pool),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    sec = time.time() - t0
    new = npz_of(run_dir, model, bucket)
    if p.returncode != 0 or not new.is_file():
        return rec | {"ok": False, "why": f"spec exit {p.returncode}", "seconds": sec}
    rows = compare(new, ref, n)
    for r in rows:
        r.update(line=tag)
    bad = [r for r in rows if not r["identical"]]
    return rec | {
        "ok": not bad,
        "why": "; ".join(f"{r['key']}: {r['detail']}" for r in bad)[:500],
        "seconds": sec,
        "keys": len(rows),
        "_rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", help="gate lines file")
    ap.add_argument(
        "--ref-root", help="deployment root holding the stored masked results"
    )
    ap.add_argument(
        "--pair", nargs=2, help="compare two chunk npz files of the same rows"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--pool",
        type=int,
        default=int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("NSLOTS", "1"))
        ),
    )
    ap.add_argument(
        "--procs",
        type=int,
        default=int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("NSLOTS", "1"))
        ),
    )
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    if a.pair:
        rows = compare(Path(a.pair[0]), Path(a.pair[1]), None)
        with open(out / "pair_rows.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["key", "identical", "detail"])
            w.writeheader()
            w.writerows(rows)
        bad = [r for r in rows if not r["identical"]]
        print(
            f"pair {a.pair[0]} vs {a.pair[1]}: {len(rows)} keys, {len(bad)} differ"
            + (f": {bad}" if bad else "")
        )
        if not bad:
            (out / "PAIR_OK").write_text(
                f"{host} {time.strftime('%F %T')}\n", encoding="utf-8"
            )
        return 1 if bad else 0
    if not (a.lines and a.ref_root):
        ap.error("--lines and --ref-root are required (or --pair)")
    lines = [
        ln.strip()
        for ln in Path(a.lines).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for f in out.glob("PASS_*"):
        f.unlink()
    (out / "GATE_OK").unlink(missing_ok=True)
    untuned = [ln for ln in lines if RUNGS[ln.split()[0]][2] is not None]
    tuned = [ln for ln in lines if RUNGS[ln.split()[0]][2] is None]
    print(
        f"gate on {host}: {len(untuned)} untuned lines ({a.procs} at once, 1 thread each), {len(tuned)} tuned lines (pool {a.pool}, one after another)",
        flush=True,
    )
    results: list[dict] = []
    with ThreadPoolExecutor(max(1, a.procs)) as ex:
        for r in ex.map(
            lambda ln: run_line(ln, Path(a.ref_root).resolve(), out, a.pool), untuned
        ):
            print(
                f"  {r['line']}: {'IDENTICAL' if r['ok'] else 'DIFFERS'} ({r['seconds']:.0f}s) {r['why']}",
                flush=True,
            )
            results.append(r)
    for ln in tuned:
        r = run_line(ln, Path(a.ref_root).resolve(), out, a.pool)
        print(
            f"  {r['line']}: {'IDENTICAL' if r['ok'] else 'DIFFERS'} ({r['seconds']:.0f}s) {r['why']}",
            flush=True,
        )
        results.append(r)
    with open(out / "gate_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line", "key", "identical", "detail"])
        w.writeheader()
        for r in results:
            for x in r.get("_rows", []):
                w.writerow({k: x[k] for k in ("line", "key", "identical", "detail")})
    cols = [
        "line",
        "rung",
        "bucket",
        "model",
        "chunk",
        "start",
        "nrows",
        "ok",
        "keys",
        "seconds",
        "why",
    ]
    with open(out / "gate_lines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["host"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in cols} | {"host": host})
    by: dict[tuple[str, str], bool] = {}
    for r in results:
        by[(r["rung"], r["model"])] = by.get((r["rung"], r["model"]), True) and bool(
            r["ok"]
        )
    for (rung, model), ok in sorted(by.items()):
        if ok:
            (out / f"PASS_{rung}_{model}").write_text(
                f"{host} {time.strftime('%F %T')}\n", encoding="utf-8"
            )
    n_ok = sum(bool(r["ok"]) for r in results)
    print(
        f"{n_ok} of {len(results)} lines identical; passed: {sorted(f'{k[0]}:{k[1]}' for k, v in by.items() if v)}"
    )
    if results and n_ok == len(results):
        (out / "GATE_OK").write_text(
            f"{host} {time.strftime('%F %T')}\n", encoding="utf-8"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
