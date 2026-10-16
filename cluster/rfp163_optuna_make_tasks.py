"""Task files of the shipped-forest Optuna rung: the per-bar random forest with a dev-span forest as its shipped forest.

PARAMETERIZED by the configuration file and the key token (--config, --token; defaults = the worked instance rfp163,
cluster/rfprev163_rf_config.json, the dev-span search's trial 163).  The masked Optuna campaign
(specs/causal_tune_trees_optuna.py, root results/linear_subsection_trees_optuna_mask, every chunk on CARC epyc-7513)
re-run for the random forest only on baseline, lean_vol_vix and all_features through specs/causal_tune_trees_optuna_rfp.py
(cluster/slurm/submit_trees_rfp163_optuna.sh).  Line format of cluster/slurm/optuna_rfp_pack.sbatch:
"<bucket> rf <tw> <seg> <chunk> <start> <end> <halo>", W = 2000, start = W + first OOS row, end = W + end row or -1
(the last chunk runs to the end of the series), halo = W.

COST MODEL (CARC epyc-7513 core-seconds, one thread):
  stage 1  per tuning point = the stored default-forest run's mean study_sec (50 trials) x the cost factor F.  The
           stored CARC records: baseline 13.7 s, all_features 352.7 s (results/linear_subsection_trees_optuna_mask),
           lean_vol_vix 58.2 s (results/linear_subsection_trees_optuna_mask_leanvolvix).  F = the per-trial cost of the
           new space over the old for the SAME (max_features, min_samples_leaf, max_depth slot): the shipped
           n_estimators / max_samples against 100 trees on every row.  MEASURED for rfp163 (355 trees, max_samples
           0.881): timed locally on the real masked window of OOS row 700 for a random sample of the stored trials,
           old and new interleaved in one process -- baseline 3.27 (40 trials), lean_vol_vix 3.22 (30), all_features
           3.21 (14).  For another configuration F = n_estimators / 100 x max_samples ** 0.75 (1 without bootstrap or
           max_samples), the form that gives 3.23 for rfp163.  This assumes the TPE studies visit the same mix of
           configurations as the default forest's did; plan for up to ~1.5x (the time limits allow 2.5x; a
           timed-out chunk resumes from its saved points).
  stage 2  per arm = the stored run's distinct-fit core-hours (about 7 distinct refits per row across the 9 paths:
           baseline 0.89, lean_vol_vix 4.11, all_features 24.67 core-h) x F; TreeSHAP is negligible.
Chunk sizes: stage 1 chunks of about S1_TASK_H hours on a 20-process pool (maximum throughput at ~400 idle CPUs, one
wave of 20-CPU tasks); chunk 0 = OOS rows [0, 20) = the canary's chunk (one wave of the pool); stage 2 chunks of
about S2_TASK_H hours.  Gate lines (chunk id 900, never a fleet chunk): OOS rows [0, GATE_ROWS) per bucket.

Writes cluster/<token>_optuna_tasks_{gate,canary,s1,s2}.txt (LF) and prints the projection.
Run from the repo root:  python cluster/rfp163_optuna_make_tasks.py [--config <json>] [--token <token>]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import trees_rfp163_make_specs as MK  # noqa: E402 -- the default CONFIG / TOKEN, rf_params

W = 2000  # TRAIN_WIN rows = the halo every chunk carries
N_OOS = 1469  # bar1600 forecast rows (2018-06-25 .. 2024-04-30), as every 16:00 arm
SEG, MODEL = "bar1600", "rf"
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
# stored default-forest Optuna run, CARC epyc-7513: mean study_sec per tuning point (50 trials)
STORED_STUDY_SEC = {"baseline": 13.7, "lean_vol_vix": 58.2, "all_features": 352.7}
# stored stage 2: distinct-fit core-hours per arm (the 9 paths' shared refits counted once)
STORED_S2_CORE_H = {"baseline": 0.89, "lean_vol_vix": 4.11, "all_features": 24.67}
MEASURED_FACTOR = {
    "rfp163": {"baseline": 3.27, "lean_vol_vix": 3.22, "all_features": 3.21}
}


def cost_factor(p: dict) -> float:
    ms = float(p.get("max_samples") or 1.0) if p.get("bootstrap", True) else 1.0
    return float(p.get("n_estimators", 100)) / 100.0 * ms**0.75


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
    ap.add_argument("--config", default=MK.CONFIG)
    ap.add_argument("--token", default=MK.TOKEN)
    a = ap.parse_args()
    p = MK.rf_params(MK.repo_relative(a.config))
    fac = MEASURED_FACTOR.get(a.token) or {b: cost_factor(p) for b in BUCKETS}
    print(
        f"{a.token}: {a.config} -> {p}; cost factor {fac} ({'measured' if a.token in MEASURED_FACTOR else 'formula'})"
    )
    out: dict[str, list[str]] = {k: [] for k in ("gate", "canary", "s1", "s2")}
    tot1 = tot2 = 0.0
    for b in BUCKETS:
        sec = STORED_STUDY_SEC[b] * fac[b]  # CARC core-seconds per tuning point
        s1_rows = POOL * S1_TASK_H * 3600 / sec
        out["gate"].append(line(b, 900, 0, GATE_ROWS[b]))
        out["canary"].append(line(b, 0, 0, CANARY_ROWS))
        out["s1"].append(line(b, 0, 0, CANARY_ROWS))
        chunks = split(CANARY_ROWS, N_OOS, s1_rows)
        for k, (r0, r1) in enumerate(chunks, 1):
            out["s1"].append(line(b, k, r0, r1))
        s2_core_h = STORED_S2_CORE_H[b] * fac[b]
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
        f"projection (fits only, CARC core-h): stage 1 {tot1:.0f}, stage 2 {tot2:.0f}, total {tot1 + tot2:.0f}"
    )
    for name, ls in out.items():
        path = ROOT / "cluster" / f"{a.token}_optuna_tasks_{name}.txt"
        path.write_text("".join(ls), encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}: {len(ls)} lines")


if __name__ == "__main__":
    main()
