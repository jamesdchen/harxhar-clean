"""Plan stage 2 of the intraday-sequence LSTM: refit chunks sized from stage 1's measured fit times.

Stage 1 (STAGE=tune) wrote one tuning record per (bucket, tuning row R) to
<root>/tune_cache/<bucket>/tune_R<R>.npz: the configuration chosen by each rule, every
candidate x seed's early-stopped epoch count and its fit seconds.  Every refit of the period
[R, R + TUNE_PER) trains the chosen configuration(s) for exactly those epoch counts on the
2000-session window, so its cost is predictable:

  seconds per epoch (candidate g, seed j) = fit_sec[g, j] / epochs run at the tuning point
      (the early-stopped count + PATIENCE, capped at MAX_EPOCHS: the loop stops PATIENCE epochs
      after the best one), scaled by WINDOW_RATIO = W / fit-block rows (a refit trains on the
      whole window, a tuning fit on the fit block);
  one row = sum over seeds of seconds per epoch x epochs + OVERHEAD_SEC (the window's bars,
  standardisation, sequences), for the MSE pick, plus the QLIKE pick when it differs.

Chunks never cross a tuning period and hold ROWS = the most rows whose estimated wall-clock on
NT workers, times SAFETY (nodes differ in speed), plus PREP_SEC (the task's own data load)
stays within TARGET_SEC -- at least MIN_ROWS (every worker busy) unless the period has fewer.
The last row of a bucket is OOS row n_oos - 1, read from the identity gate's CSV.

Writes cluster/lstm_intraday_tasks_fleet.txt (stage-2 task lines) and
<root>/stage2_plan.csv (one row per chunk: rows, picks, estimated seconds).  Usage (CARC, in
the deployment root): python cluster/lstm_intraday_make_stage2.py [--root DIR] [--nt 20]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

W = 2000
VAL_TAIL, EMBARGO = 125, 25
WINDOW_RATIO = W / (W - VAL_TAIL - EMBARGO)  # refit rows / tuning-fit rows
OVERHEAD_SEC = (
    3.0  # per refit fit: slicing the window's bars, standardising, building sequences
)
PREP_SEC = 180.0  # per task: the executor's data load and transforms (all_features the largest)
TARGET_SEC = 30 * 60  # estimated wall-clock per task (the time limit is 1 h)
SAFETY = 1.5  # node-speed spread between the tuning task's node and a refit task's node
MIN_ROWS = 4  # 4 rows x 5 seeds = 20 fits: every worker of a 20-CPU task busy
BUCKETS = ("all_features", "baseline")  # live_feasible commented out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_lstm_intraday")
    ap.add_argument("--nt", type=int, default=20)
    ap.add_argument("--out", default="cluster/lstm_intraday_tasks_fleet.txt")
    a = ap.parse_args()
    root = Path(a.root)
    gates = pd.concat(
        [
            pd.read_csv(f)
            for f in glob.glob(str(root / "gates" / "gate_lstm_intraday_*.csv"))
        ]
    )
    ident = (
        gates[gates["gate"] == "identity"].drop_duplicates("bucket").set_index("bucket")
    )
    lines, plan = [], []
    for b in BUCKETS:
        n_oos = int(ident.loc[b, "n_oos"])
        chunk = 0
        for R in range(0, n_oos, 250):
            with np.load(
                root / "tune_cache" / b / f"tune_R{R:05d}.npz", allow_pickle=False
            ) as z:
                sc = json.loads(str(z["scalars"]))
                fp = json.loads(str(z["fingerprint"]))
                ep, fs = z["epochs"], z["fit_sec"]
            tune_per, pat, cap = (
                fp["grid"]["tune_per"],
                fp["grid"]["patience"],
                fp["grid"]["max_epochs"],
            )
            assert tune_per == 250 and fp["W"] == W
            run = np.minimum(ep + pat, cap).clip(min=1)
            spe = fs / run * WINDOW_RATIO

            def cost(g: int) -> float:
                return float((spe[g] * ep[g] + OVERHEAD_SEC).sum())

            gm, gq = int(sc["pick_mse"]), int(sc["pick_qlike"])
            row_sec = cost(gm) + (cost(gq) if gq != gm else 0.0)
            end = min(R + tune_per, n_oos)
            per_task = max(
                MIN_ROWS, int((TARGET_SEC - PREP_SEC) * a.nt / (row_sec * SAFETY))
            )
            s = R
            while s < end:
                e = min(s + per_task, end)
                if end - e < MIN_ROWS:  # no sliver at the period's end
                    e = end
                lines.append(
                    f"{b} lstm_intraday {W} bar1600 {chunk} {W + s} {W + e} {W} all"
                )
                plan.append(
                    {
                        "bucket": b,
                        "chunk": chunk,
                        "rows": e - s,
                        "oos_lo": s,
                        "oos_hi": e,
                        "tune_row": R,
                        "pick_mse": gm,
                        "pick_qlike": gq,
                        "row_cpu_sec_est": round(row_sec, 1),
                        "task_wall_sec_est": round(
                            PREP_SEC + (e - s) * row_sec / a.nt, 1
                        ),
                    }
                )
                chunk += 1
                s = e
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    p = pd.DataFrame(plan)
    p.to_csv(root / "stage2_plan.csv", index=False)
    p["cpu_h_est"] = p["rows"] * p["row_cpu_sec_est"] / 3600
    print(
        p.groupby("bucket")
        .agg(
            chunks=("chunk", "size"),
            rows=("rows", "sum"),
            cpu_h=("cpu_h_est", "sum"),
            wall_max_sec=("task_wall_sec_est", "max"),
        )
        .round(1)
        .to_string()
    )
    tot = float((p["rows"] * p["row_cpu_sec_est"]).sum()) / 3600
    print(
        f"{len(lines)} stage-2 tasks -> {a.out}; estimated fit CPU-hours {tot:.0f}; "
        f"largest task wall estimate {p['task_wall_sec_est'].max() / 60:.0f} min"
    )


if __name__ == "__main__":
    main()
