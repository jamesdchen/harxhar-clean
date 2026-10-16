"""Gate of the moments_vix Optuna run: does this node class reproduce the STORED canonical run bit for bit?

Before any moments_vix fleet chunk runs on a node class (CARC epyc-7513, Hoffman2 intel-E5-2650v4),
stored tuning points and refits of an existing bucket are re-computed there with the shipped spec
(cluster/slurm/momvix_gate.sbatch on CARC, cluster/momvix_optuna_h2_gate.sh on Hoffman2: the
fleet's own pack / task script, task line chunk 900 of cluster/momvix_optuna_tasks_gate.txt) and
compared here with the canonical masked run (results/linear_subsection_trees_optuna_mask, every
chunk on CARC epyc-7513):

  stage 1  the gate chunk's tuning records (rows ROW0 .. ROW0 + ROWS - 1, 50 TPE trials each)
           vs the stored merged records of those rows: row, seed, split, params, val_mse,
           val_qlike, rounds, rounds_max, patience, complete, n_kept and the best-of-k picks,
           np.array_equal (NaN == NaN), i.e. bit for bit
  stage 2  the gate chunk's refits of the same rows FROM THE STORED stage-1 records (all 9
           paths) vs the stored merged paths: pred_adj, true_adj, true_raw, baseline, the
           configuration in force (cfg_point / cfg_trial / cfg_rounds / cfg_params), n_kept,
           native importance and (tp1) TreeSHAP on the recorded rows, bit for bit

Writes <gate-root>/gate_<model>.csv (one row per compared array) and, only when every array is
bit-identical, touches --flag (the GATE flag the fleet tasks of that model require).  Exit 1 on
any difference or missing file.

Usage: python experiments/momvix_optuna_gate.py --model lgbm --gate-root <R>/gate \
           --ref-root <canonical root> --flag <R>/GATE_OK_lgbm [--bucket live_feasible --row0 700 --rows 20]
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
from pathlib import Path

import numpy as np

SEG, TW, CHUNK = "bar1600", 2000, 900
PATHS = (
    "tp1",
    "tp5",
    "tp25",
    "tp250",
    "tp1_k10",
    "tp1_k25",
    "tp25_k10",
    "tp25_k25",
    "tp25_q",
)
S1_KEYS = (
    "row",
    "seed",
    "split",
    "params",
    "val_mse",
    "val_qlike",
    "rounds",
    "rounds_max",
    "patience",
    "complete",
    "n_kept",
    "best_mse_k10",
    "best_mse_k25",
    "best_mse_k50",
    "best_qlike_k50",
)
S2_KEYS = (
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "baseline",
    "cfg_point",
    "cfg_trial",
    "cfg_rounds",
    "cfg_params",
    "n_kept",
    "importance",
)


def load(p: Path) -> dict | None:
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def same(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def max_rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype.kind not in "fiu":
        return float("nan")
    a, b = a.astype(float), b.astype(float)
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return float(np.max(np.abs(a[m] / b[m] - 1.0))) if m.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate-root", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--flag", required=True)
    ap.add_argument("--bucket", default="live_feasible")
    ap.add_argument("--row0", type=int, default=700)
    ap.add_argument("--rows", type=int, default=20)
    a = ap.parse_args()
    m, b = a.model, a.bucket
    gr, ref = Path(a.gate_root), Path(a.ref_root)
    arm = Path(b) / SEG / m / f"tw{TW}"
    inner = Path("causal_tune_trees") / m / b
    rows = np.arange(a.row0, a.row0 + a.rows)
    out: list[dict] = []
    ok = True

    def rec(
        stage: str, what: str, key: str, good: bool, rel: float, note: str = ""
    ) -> None:
        nonlocal ok
        ok = ok and good
        out.append(
            {
                "model": m,
                "bucket": b,
                "rows": f"{rows[0]}-{rows[-1]}",
                "stage": stage,
                "what": what,
                "key": key,
                "bit_identical": good,
                "max_rel_diff": rel,
                "note": note,
                "host": socket.gethostname(),
            }
        )

    # ---- stage 1
    g1 = load(
        gr / "stage1" / arm / "chunks" / f"c{CHUNK}" / inner / f"trials_{SEG}.npz"
    )
    r1 = load(ref / "stage1" / arm / f"trials_{SEG}.npz")
    if g1 is None or r1 is None:
        rec(
            "1",
            "trials",
            "-",
            False,
            float("nan"),
            f"missing: gate {g1 is None}, ref {r1 is None}",
        )
    else:
        idx = {int(r): j for j, r in enumerate(r1["row"])}
        sel = np.array([idx[int(r)] for r in g1["row"]], dtype=np.int64)
        rec("1", "trials", "rows", bool(np.array_equal(g1["row"], rows)), 0.0)
        for k in S1_KEYS:
            if k not in g1 or k not in r1:
                rec("1", "trials", k, False, float("nan"), "key missing")
                continue
            rec("1", "trials", k, same(g1[k], r1[k][sel]), max_rel(g1[k], r1[k][sel]))
    # ---- stage 2 (refits from the stored stage-1 records)
    for path in PATHS:
        g2 = load(
            gr
            / "stage2"
            / arm
            / "chunks"
            / f"c{CHUNK}"
            / path
            / inner
            / f"trees_{SEG}.npz"
        )
        r2 = load(ref / "paths" / path / arm / inner / f"trees_{SEG}.npz")
        if g2 is None or r2 is None:
            rec(
                "2",
                path,
                "-",
                False,
                float("nan"),
                f"missing: gate {g2 is None}, ref {r2 is None}",
            )
            continue
        off = int(g2["oos_offset"])
        sl = slice(off, off + len(g2["pred_adj"]))
        rec(
            "2",
            path,
            "oos_offset",
            off == a.row0 and len(g2["pred_adj"]) == a.rows,
            0.0,
        )
        for k in S2_KEYS:
            rec("2", path, k, same(g2[k], r2[k][sl]), max_rel(g2[k], r2[k][sl]))
        if path == "tp1":
            gs, rs = g2["shap"], r2["shap"][sl]
            good = gs.ndim == 2 and same(gs, rs)
            rec("2", path, "shap", good, max_rel(gs, rs))
    gr.mkdir(parents=True, exist_ok=True)
    p = gr / f"gate_{m}.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    bad = [
        f"{r['stage']}/{r['what']}/{r['key']}" for r in out if not r["bit_identical"]
    ]
    print(
        f"gate {m} [{b} rows {rows[0]}-{rows[-1]}] on {socket.gethostname()}: "
        f"{len(out) - len(bad)} of {len(out)} arrays bit-identical; wrote {p}"
    )
    if ok:
        Path(a.flag).parent.mkdir(parents=True, exist_ok=True)
        Path(a.flag).touch()
        print(f"GATE OK: wrote {a.flag}")
        sys.exit(0)
    print("GATE FAIL: " + ", ".join(bad[:30]))
    sys.exit(1)


if __name__ == "__main__":
    main()
