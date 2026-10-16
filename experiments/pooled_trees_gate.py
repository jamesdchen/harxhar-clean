"""Reproduction gate of the pooled tree walks (2026-10-07): experiments/pooled_trees_walk.py with the bank's own arms'
configurations must reproduce the stored bank chunks (harxhar-clean/results/unification_carc/tree_expert_00 = lgbm_opt_00,
tree_expert_10 = xgb_opt_00; all_features = the bank's design) byte for byte.

Gate line (--lines, one per line):  "<est> <config> <chunk> <rows> <threads> <ref chunk npz>"
  the walk runs the first <rows> rows of bank chunk <chunk> with SLURM_CPUS_PER_TASK=<threads> on all_features
  (the bank used 8 threads), and its yhat / y_fit / t / rv_raw / baseline / row_id are compared with the reference
  chunk's first <rows> rows, dtype, shape and bytes.  A line with threads != 8 tests thread-count invariance (XGBoost)
  against the same 8-thread reference.
Output (--out): one run dir per line, gate_lines.csv, NODE (host, CPU model), GATE_OK when every line matched.
Exit 0 iff GATE_OK.

Run:  UNIFY_CACHE_DIR=cache python experiments/pooled_trees_gate.py --lines cluster/pooltree_gate_lines.txt --out <dir>
          [--procs 1]
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("row_id", "t", "y_fit", "yhat", "rv_raw", "baseline")


def run_line(line: str, out: Path) -> dict:
    est, config, chunk, rows, threads, ref = line.split()
    tag = f"{est}_c{chunk}_n{rows}_th{threads}"
    d = out / tag
    d.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        SLURM_CPUS_PER_TASK=threads,
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
        PYTHONUNBUFFERED="1",
    )
    env.pop("TREE_REFIT_EVERY", None)
    t0 = time.time()
    with open(d / "run.log", "w", encoding="utf-8") as log:
        p = subprocess.run(
            [
                sys.executable,
                "-u",
                "experiments/pooled_trees_walk.py",
                "--est",
                est,
                "--set",
                "all_features",
                "--config",
                config,
                "--chunk-index",
                chunk,
                "--rows",
                rows,
                "--out",
                str(d),
                "--arm",
                f"gate_{tag}",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    sec = time.time() - t0
    new = d / f"gate_{tag}" / f"chunk_{int(chunk):03d}.npz"
    rec = {
        "line": tag,
        "est": est,
        "chunk": chunk,
        "rows": rows,
        "threads": threads,
        "seconds": round(sec, 1),
    }
    if p.returncode != 0 or not new.is_file() or not Path(ref).is_file():
        return rec | {
            "ok": False,
            "why": f"walk exit {p.returncode}; new {new.is_file()}; ref {Path(ref).is_file()}",
        }
    zn, zr = np.load(new, allow_pickle=True), np.load(ref, allow_pickle=True)
    n = int(rows)
    bad = []
    for k in KEYS:
        a, b = np.asarray(zn[k]), np.asarray(zr[k])[:n]
        if (
            a.dtype != b.dtype
            or a.shape != b.shape
            or np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes()
        ):
            if a.dtype.kind == "f" and a.shape == b.shape:
                rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
                bad.append(
                    f"{k}: {int((a != b).sum())} of {n} differ, max rel {rel:.3g}"
                )
            else:
                bad.append(f"{k}: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
    return rec | {"ok": not bad, "why": "; ".join(bad)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1, help="lines at once")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "GATE_OK").unlink(missing_ok=True)
    cpu = (
        next(
            (
                ln.split(":", 1)[1].strip()
                for ln in open("/proc/cpuinfo")
                if ln.startswith("model name")
            ),
            "?",
        )
        if os.path.exists("/proc/cpuinfo")
        else "?"
    )
    (out / "NODE").write_text(
        f"host {socket.gethostname()}\ncpu {cpu}\n", encoding="utf-8"
    )
    lines = [
        ln.strip()
        for ln in Path(a.lines).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    print(
        f"pooled tree gate on {socket.gethostname()} [{cpu}]: {len(lines)} lines",
        flush=True,
    )
    res = []
    with ThreadPoolExecutor(max(1, a.procs)) as ex:
        for r in ex.map(lambda ln: run_line(ln, out), lines):
            print(
                f"  {r['line']}: {'IDENTICAL' if r['ok'] else 'DIFFERS'} ({r['seconds']}s) {r['why']}",
                flush=True,
            )
            res.append(r)
    with open(out / "gate_lines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "line",
                "est",
                "chunk",
                "rows",
                "threads",
                "ok",
                "seconds",
                "why",
                "host",
                "cpu",
            ],
        )
        w.writeheader()
        for r in res:
            w.writerow(r | {"host": socket.gethostname(), "cpu": cpu})
    ok = bool(res) and all(r["ok"] for r in res)
    if ok:
        (out / "GATE_OK").write_text(
            f"{socket.gethostname()} {cpu} {time.strftime('%F %T')}\n", encoding="utf-8"
        )
    print(f"{sum(r['ok'] for r in res)} of {len(res)} lines identical")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
