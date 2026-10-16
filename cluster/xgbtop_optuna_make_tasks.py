"""Task files of the re-centred XGBoost Optuna rung (xgbtop, 2026-10-07): the per-bar XGBoost with xgb_opt_00 as trial 0.

The XGBoost twin of cluster/rfp163_optuna_make_tasks.py.  The masked Optuna campaign (specs/causal_tune_trees_optuna.py,
root results/linear_subsection_trees_optuna_mask, every chunk on CARC epyc-7513) re-run for XGBoost only on baseline,
lean_vol_vix and all_features through specs/causal_tune_trees_optuna_xgbtop.py (cluster/slurm/submit_trees_xgbtop_optuna.sh).
Line format of cluster/slurm/optuna_xgbtop_pack.sbatch: "<bucket> xgb <tw> <seg> <chunk> <start> <end> <halo>", W = 2000,
start = W + first OOS row, end = W + end row or -1 (the last chunk runs to the end of the series), halo = W.

COST MODEL (CARC epyc-7513 core-seconds, one thread):
  stage 1  per tuning point = the stored xgb_opt_06 run's mean study_sec (50 trials) x the cost factor F.  The stored
           CARC records: baseline 11.0 s, all_features 417.5 s (results/linear_subsection_trees_optuna_mask), lean_vol_vix
           52.5 s (results/linear_subsection_trees_optuna_mask_leanvolvix, chunk records).  The search space is the
           same but for reg_lambda's lower end (1e-4 -> 1e-5) and the round cap (4 x 367 x 0.012337 / lr instead of
           4 x 339 x 0.010853 / lr: 1.23x more rounds allowed at a given lr); trial 0 is xgb_opt_00 (depth 5, 367
           rounds) instead of xgb_opt_06 (depth 4, 339 rounds).  F = --factor (default 1.25: the cap ratio as the upper
           end for the early-stopped trials; measured per-refit ratio xgb_opt_00 / xgb_opt_06 ~1.2 locally).
  stage 2  per arm = the stored run's distinct-fit core-hours (about 7 distinct refits per row across the 9 paths:
           baseline 0.49, all_features 22.73 core-h, measured from the stored paths' npz; lean_vol_vix 2.9, scaled from
           all_features by the stage-1 ratio) x F; TreeSHAP is negligible.
Chunk sizes: stage 1 chunks of about S1_TASK_H hours on a 20-process pool; chunk 0 = OOS rows [0, 20) = the canary's
chunk (one wave of the pool); stage 2 chunks of about S2_TASK_H hours.  Gate lines (chunk id 900, never a fleet chunk):
OOS rows [0, GATE_ROWS) per bucket.

Writes cluster/<token>_optuna_tasks_{gate,canary,s1,s2}.txt (LF) and prints the projection.
Run from the repo root:  python cluster/xgbtop_optuna_make_tasks.py [--token xgbtop] [--factor 1.25]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = 2000  # TRAIN_WIN rows = the halo every chunk carries
N_OOS = 1469  # bar1600 forecast rows (2018-06-25 .. 2024-04-30), as every 16:00 arm
SEG, MODEL = "bar1600", "xgb"
BUCKETS = ("baseline", "lean_vol_vix", "all_features")
CANARY_ROWS = 20  # one wave of the 20-process pool
GATE_ROWS = {
    "baseline": 11,
    "lean_vol_vix": 11,
    "all_features": 5,
}  # 11: two importance rows (0, 10)
POOL = 20
S1_TASK_H = 1.25
S2_TASK_H = 0.5
# stored xgb_opt_06 Optuna run, CARC epyc-7513: mean study_sec per tuning point (50 trials)
STORED_STUDY_SEC = {"baseline": 11.0, "lean_vol_vix": 52.5, "all_features": 417.5}
# stored stage 2: distinct-fit core-hours per arm (the 9 paths' shared refits counted once)
STORED_S2_CORE_H = {"baseline": 0.49, "lean_vol_vix": 2.9, "all_features": 22.73}


def line(bucket: str, k: int, r0: int, r1: int) -> str:
    return (
        f"{bucket} {MODEL} {W} {SEG} {k} {W + r0} {-1 if r1 >= N_OOS else W + r1} {W}\n"
    )


def split(r0: int, r1: int, rows: float) -> list[tuple[int, int]]:
    """[r0, r1) into chunks of about `rows` rows (equal sizes, at least one chunk)."""
    n = max(1, round((r1 - r0) / rows))
    edges = [r0 + (r1 - r0) * i // n for i in range(n + 1)]
    return list(zip(edges[:-1], edges[1:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default="xgbtop")
    ap.add_argument("--factor", type=float, default=1.25)
    a = ap.parse_args()
    fac = a.factor
    out: dict[str, list[str]] = {k: [] for k in ("gate", "canary", "s1", "s2")}
    tot1 = tot2 = 0.0
    for b in BUCKETS:
        sec = STORED_STUDY_SEC[b] * fac  # CARC core-seconds per tuning point
        s1_rows = POOL * S1_TASK_H * 3600 / sec
        out["gate"].append(line(b, 900, 0, GATE_ROWS[b]))
        out["canary"].append(line(b, 0, 0, CANARY_ROWS))
        out["s1"].append(line(b, 0, 0, CANARY_ROWS))
        chunks = split(CANARY_ROWS, N_OOS, s1_rows)
        for k, (r0, r1) in enumerate(chunks, 1):
            out["s1"].append(line(b, k, r0, r1))
        s2_core_h = STORED_S2_CORE_H[b] * fac
        n2 = max(1, math.ceil(s2_core_h / (POOL * S2_TASK_H)))
        for j, (r0, r1) in enumerate(split(0, N_OOS, N_OOS / n2)):
            out["s2"].append(line(b, j, r0, r1))
        core1 = sec * N_OOS / 3600
        tot1, tot2 = tot1 + core1, tot2 + s2_core_h
        big = max(r1 - r0 for r0, r1 in chunks)
        print(
            f"{b}: stage 1 {sec:.0f} core-s/point -> {core1:.1f} core-h; {len(chunks)} fleet chunks of <= {big} rows "
            f"(~{big * sec / POOL / 3600:.2f} h on {POOL}) + the canary chunk; stage 2 {s2_core_h:.1f} core-h in {n2} "
            f"chunk(s) (~{s2_core_h / n2 / POOL:.2f} h each)"
        )
    print(
        f"projection (fits only, CARC core-h, factor {fac}): stage 1 {tot1:.0f}, stage 2 {tot2:.0f}, total {tot1 + tot2:.0f}"
    )
    for name, ls in out.items():
        path = ROOT / "cluster" / f"{a.token}_optuna_tasks_{name}.txt"
        path.write_bytes("".join(ls).encode("utf-8"))
        print(f"wrote {path.relative_to(ROOT)}: {len(ls)} lines")


if __name__ == "__main__":
    main()
