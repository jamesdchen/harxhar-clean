"""Identity gate of the re-centred random-search rungs (2026-10-07, rfp163): with TUNE_IDENTITY=1 the tuned spec
through specs/causal_tune_trees_tuned_rfp.py (one candidate = the new shipped forest, early stopping off) must
reproduce the FIXED-settings forest of the same configuration -- the trial-163 preview run
(specs/causal_tune_trees_rfdev.py with cluster/rfprev163_rf_config.json; results/linear_subsection_trees_mask_rfprev163)
-- byte for byte on the rows it re-runs.

Gate line (--lines): "<rung> <bucket> <chunk> <start> <nrows>" -- rung rs10 (REFIT_EVERY 10, TreeSHAP at bar1600)
compared with the preview's t10 chunk c<chunk>, rung rs1 (REFIT_EVERY 1, TreeSHAP off) with its t1 chunk; start a
multiple of TUNE_PER from 2000 (the tuned spec asserts it).  The spec call is cluster/slurm/treestuned_rfp_pack.sbatch's
(the masked tuned pack's axes) with TUNE_IDENTITY=1.  Compared byte for byte: date, pred_adj, true_adj, true_raw,
refit_row, kept_n, kept; native importance on the refits the reference recorded (T1 records every 10th); TreeSHAP
(rs10 vs t10).  Writes gate_lines.csv / gate_rows.csv and GATE_OK when every line matched; exit 0 iff GATE_OK.

Run:  python experiments/trees_rfp163_tuned_gate.py --lines cluster/trees_rfp163_tuned_gate_lines.txt
          --ref-root /scratch1/jc_905/harxhar-rfprev-trees --out results/trees_rfp163_gates/tuned_identity [--pool 20]
          [--stored linear_subsection_trees_mask_<token>]  (default: the trial-163 preview root)
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = "specs/causal_tune_trees_tuned_rfp.py"
STORED = "linear_subsection_trees_mask_rfprev163"
RUNGS = {"rs10": (10, "bar1600", "t10"), "rs1": (1, "none", "t1")}


def bytes_equal(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False, f"dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
    if np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes():
        return True, ""
    if a.dtype.kind == "f":
        d = np.abs(a.astype(float) - b.astype(float))
        s = np.maximum(np.abs(a.astype(float)), np.abs(b.astype(float)))
        ok = np.isfinite(d)
        rel = (
            float(np.max(np.where(s[ok] > 0, d[ok] / s[ok], 0))) if ok.any() else np.nan
        )
        return (
            False,
            f"{int((d[ok] > 0).sum())} differ, max rel {rel:.3g}, nan mismatch {int((np.isnan(a) != np.isnan(b)).sum())}",
        )
    return False, f"{int((a != b).sum())} entries differ"


def run_line(line: str, ref_root: Path, out: Path, pool: int) -> dict:
    rung, bucket, chunk, start, nrows = line.split()
    re_, shapseg, fixed = RUNGS[rung]
    s0, n = int(start), int(nrows)
    tag = f"{rung}_{bucket}_c{chunk}_n{n}"
    run_dir = out / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        TQDM_DISABLE="1",
        PYTHONUNBUFFERED="1",
        SLURM_CPUS_PER_TASK=str(pool),
        HPC_KW_MODEL="rf",
        HPC_KW_EXOG_BUCKET=bucket,
        HPC_KW_SEGMENT="bar1600",
        HPC_KW_TRAIN_WIN="2000",
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_HAR_BASE="",
        HPC_KW_START=str(s0),
        HPC_KW_END=str(s0 + n),
        HPC_KW_HALO="2000",
        HPC_KW_REFIT_EVERY=str(re_),
        HPC_KW_SHAP_SEGMENTS=shapseg,
        HPC_KW_TUNE_IDENTITY="1",
        HPC_KW_QSEL="1",
        HPC_KW_WINDOW_MASK="1",
        HPC_RESULT_DIR=str(run_dir),
        NUMBA_CACHE_DIR=os.path.join(
            tempfile.gettempdir(), f"numba_rfpgate_{os.getpid()}_{tag}"
        ),
    )
    t0 = time.time()
    with open(run_dir / "run.log", "w", encoding="utf-8") as log:
        p = subprocess.run(
            [sys.executable, "-u", WRAPPER],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    sec = time.time() - t0
    new = run_dir / "causal_tune_trees" / "rf" / bucket / "trees_bar1600.npz"
    ref = (
        ref_root
        / "results"
        / STORED
        / fixed
        / bucket
        / "bar1600"
        / "rf"
        / "tw2000"
        / "chunks"
        / f"c{chunk}"
        / "causal_tune_trees"
        / "rf"
        / bucket
        / "trees_bar1600.npz"
    )
    rec = {
        "line": tag,
        "rung": rung,
        "bucket": bucket,
        "seconds": round(sec, 1),
        "host": socket.gethostname(),
    }
    if p.returncode != 0 or not new.is_file() or not ref.is_file():
        return rec | {
            "ok": False,
            "why": f"spec exit {p.returncode}; new {new.is_file()}; ref {ref.is_file()}",
            "_rows": [],
        }
    zn, zr = np.load(new), np.load(ref)
    m = len(zn["pred_adj"])
    k = len(zn["refit_row"])
    rows = []
    for key in ("date", "pred_adj", "true_adj", "true_raw"):
        rows.append((key, *bytes_equal(zn[key], zr[key][:m])))
    for key in ("refit_row", "kept_n", "kept"):
        rows.append((key, *bytes_equal(zn[key], zr[key][:k])))
    ir = zr["importance"][:k]
    rec_rows = np.isfinite(ir).all(axis=1)
    rows.append(
        (
            "importance (reference-recorded refits)",
            *bytes_equal(zn["importance"][:k][rec_rows], ir[rec_rows]),
        )
    )
    if rung == "rs10":
        rows.append(("shap", *bytes_equal(zn["shap"], zr["shap"][:m])))
    bad = [r for r in rows if not r[1]]
    return rec | {
        "ok": not bad,
        "why": "; ".join(f"{a}: {c}" for a, _b, c in bad)[:400],
        "_rows": [
            {"line": tag, "key": a, "identical": b, "detail": c} for a, b, c in rows
        ],
    }


def main() -> int:  # noqa: C901
    global STORED
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--pool", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    ap.add_argument(
        "--stored",
        default=STORED,
        help="results root (under REF_ROOT/results) of the fixed-settings forest to reproduce",
    )
    a = ap.parse_args()
    STORED = a.stored  # run_line reads the module constant
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "GATE_OK").unlink(missing_ok=True)
    res = []
    for ln in (
        x.strip() for x in Path(a.lines).read_text(encoding="utf-8").splitlines()
    ):
        if ln:
            r = run_line(ln, Path(a.ref_root).resolve(), out, a.pool)
            print(
                f"  {r['line']}: {'IDENTICAL' if r['ok'] else 'DIFFERS'} ({r['seconds']}s) {r['why']}",
                flush=True,
            )
            res.append(r)
    with open(out / "gate_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line", "key", "identical", "detail"])
        w.writeheader()
        for r in res:
            w.writerows(r["_rows"])
    with open(out / "gate_lines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["line", "rung", "bucket", "ok", "seconds", "host", "why"]
        )
        w.writeheader()
        for r in res:
            w.writerow(
                {
                    k: r[k]
                    for k in ("line", "rung", "bucket", "ok", "seconds", "host", "why")
                }
            )
    ok = bool(res) and all(r["ok"] for r in res)
    if ok:
        (out / "GATE_OK").write_text(time.strftime("%F %T") + "\n", encoding="utf-8")
    print(f"{sum(r['ok'] for r in res)} of {len(res)} lines identical")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
