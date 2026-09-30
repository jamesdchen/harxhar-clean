"""Cross-CPU-class gate of the masked Optuna campaign (run on CARC after cluster/slurm/submit_optuna_xclass.sh).

The same 20 tuning points (live_feasible, OOS rows 0-19, lgbm / xgb / rf) computed twice:
  fleet   the first 20 points of the fleet's chunk 0 of the arm (stage1/.../chunks/c0), on
          the node chunk_hosts.csv lists (host.txt / the Slurm log)
  xclass  the same rows run on a node of the other CPU class (--constraint=xeon-4116), with
          the canary's repeat (bit-identical on its node) and stage 2 on its own records
and, once the stage-2 merge has run, the xclass stage-2 forecasts of rows 0-19 against the
merged fleet paths (paths/<path>/... rows 0-19).

Reports per model: whether every trial record is bit-identical, the largest relative
difference of a trial's validation MSE, the number of points whose best-of-50 (and best-of-10
/ 25, QLIKE rule) trial differs, the kept-column counts, and per path the largest relative
forecast difference.  Writes <root>/xclass/xclass_gate.csv.

Usage (CARC, repo root): python experiments/optuna_xclass_compare.py [--root results/linear_subsection_trees_optuna_mask]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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
BUCKET, SEG, TW, ROWS = "live_feasible", "bar1600", 2000, 20


def trials(p: Path) -> dict | None:
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return float(np.max(np.abs(a[m] / b[m] - 1.0))) if m.any() else float("nan")


def host_of(root: Path, model: str) -> str:
    p = root / "chunk_hosts.csv"
    if not p.is_file():
        return ""
    for r in csv.DictReader(p.open()):
        if (r["stage"], r["bucket"], r["model"], r["chunk"]) == (
            "1",
            BUCKET,
            model,
            "0",
        ):
            return f"{r['host']} ({r['cpu_class']})"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_trees_optuna_mask")
    a = ap.parse_args()
    root = ROOT / a.root
    x = root / "xclass"
    rows = []
    for model in ("lgbm", "xgb", "rf"):
        inner = Path("causal_tune_trees") / model / BUCKET / f"trials_{SEG}.npz"
        arm = Path(BUCKET) / SEG / model / f"tw{TW}"
        f = trials(root / "stage1" / arm / "chunks" / "c0" / inner)
        g = trials(x / "stage1" / arm / "chunks" / "c0" / inner)
        r: dict = {"model": model, "fleet_node": host_of(root, model)}
        ht = x / "stage1" / arm / "chunks" / "c0" / "host.txt"
        r["xclass_node"] = ht.read_text().replace("\n", "; ") if ht.is_file() else ""
        if f is None or g is None:
            r["note"] = "fleet or xclass records missing"
            rows.append(r)
            continue
        n = min(ROWS, len(g["row"]))
        assert np.array_equal(f["row"][:n], g["row"][:n]), "rows differ"
        keys = ("params", "val_mse", "val_qlike", "rounds", "rounds_max", "n_kept")
        r |= {
            "points": n,
            "records_bit_identical": all(
                np.array_equal(f[k][:n], g[k][:n], equal_nan=True) for k in keys
            ),
            "kept_identical": bool(np.array_equal(f["n_kept"][:n], g["n_kept"][:n])),
            "val_mse_max_rel_diff": rel(g["val_mse"][:n], f["val_mse"][:n]),
            "points_trial1_identical": int(
                np.sum(g["val_mse"][:n, 0] == f["val_mse"][:n, 0])
            ),
        }
        for k in ("best_mse_k10", "best_mse_k25", "best_mse_k50", "best_qlike_k50"):
            r[f"{k}_differs"] = int(np.sum(f[k][:n] != g[k][:n]))
        for path in PATHS:
            pin = Path(path) / "causal_tune_trees" / model / BUCKET / f"trees_{SEG}.npz"
            xs = trials(x / "canary_stage2" / arm / "c0" / pin)
            fm = trials(
                root
                / "paths"
                / path
                / arm
                / "causal_tune_trees"
                / model
                / BUCKET
                / f"trees_{SEG}.npz"
            )
            if xs is not None and fm is not None:
                m = min(ROWS, len(xs["pred_adj"]))
                r[f"{path}_forecast_max_rel_diff"] = rel(
                    xs["pred_adj"][:m], fm["pred_adj"][:m]
                )
        rows.append(r)
    fields = sorted(
        {k for r in rows for k in r},
        key=lambda k: (k not in ("model", "fleet_node", "xclass_node"), k),
    )
    out = x / "xclass_gate.csv"
    x.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: r[k] for k in r if not k.startswith("tp")})
    print(f"wrote {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
