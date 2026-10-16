"""Gate of the pooled pre-tune study on CARC (2026-10-06; runner experiments/pool_linear_pretune.py).

gate  The canary re-ran arm a (the spec through the runner's penalty-factor class at r = 1, causal) on
      the stored-arm slice (solves 199,250 .. 204,316) of all_features / baseline / lean_vol_vix x ridge /
      recursive lasso / recursive elastic net, plus the smoke tasks (r = 0.01 walks, one pre-tune).
      Required: every canary task finished, finite output, every penalty-factor batch solution certified;
      the ridge slices' regular-hours forecasts within 1e-6 relative of the stored Hoffman2 forecasts (another
      CPU and BLAS: 3.8e-7 on all_features, ~640 columns; 3e-13 on baseline)
      (cluster/pool_pretune_ref_<bucket>_<est>.csv: the stored arms' 10:00-16:00 rows of the slice, from
      results/linear_subsection/arms_hoffman2).  Recorded, not required: the lasso / elastic net gaps (the
      warm homotopy branches on the floating-point path; another machine takes other branches,
      results/linear_subsection_pool_momvix/cross_cluster.csv).
      Writes results/pool_pretune/gate.csv and, on a pass, results/pool_pretune/GATE_OK.

Run (CARC, deployment root):  python cluster/slurm/pool_pretune_check_carc.py gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/pool_pretune")
RT = "round_trip"
RIDGE_TOL = 1e-6  # cross-machine ridge: all_features (~640 columns) differs from Hoffman2 by 3.8e-7 relative
CANARY = Path("cluster/pool_pretune_tasks_canary_carc.txt")


def gate() -> None:
    rows, ok_all = [], True
    for ln in CANARY.read_text().splitlines():
        if not ln.strip():
            continue
        tag, mode, bucket, est, r, sched, start, end, halo = ln.split()
        d = OUT / tag / bucket / est / f"r{r}" / f"s{start}"
        info = (
            json.loads((d / "run.json").read_text())
            if (d / "run.json").is_file()
            else {}
        )
        row = dict(
            tag=tag,
            mode=mode,
            bucket=bucket,
            est=est,
            r=r,
            start=start,
            done=(d / "DONE").is_file(),
            host=info.get("host", ""),
            repair=json.dumps(info.get("repair", {})),
        )
        good = row["done"] and info.get("repair", {}).get("fail", 0) == 0
        if row["done"] and mode == "walk":
            new = pd.read_csv(d / "results_rth.csv", float_precision=RT)
            good = good and bool(np.isfinite(new["pred_adj"]).all())
            ref_p = Path(f"cluster/pool_pretune_ref_{bucket}_{est}.csv")
            if tag == "gate" and ref_p.is_file():
                ref = pd.read_csv(ref_p, float_precision=RT)
                same = len(ref) == len(new) and bool(
                    (ref["date"].to_numpy() == new["date"].to_numpy()).all()
                )
                rel = (
                    np.abs(
                        new["pred_adj"].to_numpy(float)
                        / ref["pred_adj"].to_numpy(float)
                        - 1
                    )
                    if same
                    else np.inf
                )
                m16 = new["date"].str.endswith("16:00:00").to_numpy()
                row.update(
                    rth_rows=len(new),
                    same_stamps=same,
                    max_rel_diff=float(np.max(rel)),
                    max_rel_diff_1600=float(np.max(rel[m16])) if same else np.inf,
                    bitwise_rows=int((rel == 0).sum()) if same else 0,
                )
                if est == "ridge":
                    good = good and same and row["max_rel_diff"] <= RIDGE_TOL
        elif row["done"]:
            good = good and bool(np.isfinite(np.load(d / "fold.npz")["resid"]).all())
        row["verdict"] = "ok" if good else "FAILED"
        ok_all = ok_all and good
        rows.append(row)
        print(row)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gate.csv", index=False)
    flag = OUT / "GATE_OK"
    if ok_all:
        flag.write_text("gate passed\n")
        print("GATE PASSED ->", flag)
    else:
        flag.unlink(missing_ok=True)
        print("GATE FAILED (no flag)")
        sys.exit(1)


if __name__ == "__main__":
    {"gate": gate}[sys.argv[1]]()
