"""Task files of the Optuna per-bar trees on the `moments_vix` bucket (user request 2026-10-06).

The masked Optuna campaign of 2026-09-29/30 (specs/causal_tune_trees_optuna.py, results root
results/linear_subsection_trees_optuna_mask, every chunk on CARC epyc-7513) re-run on ONE new
input set, `moments_vix` = HAR + calendar backbone + SUBGROUPS["moments"] (sumret, sumabsret,
sumret3, sumret4, sumpret2, sumbipow, sumautocov) + vix (src/data/loading.py).  Same spec, same
pack / merge scripts, same task-line format ("<bucket> <model> <tw> <seg> <chunk> <start> <end>
<halo>", W = 2000, start = W + first OOS row, end = W + end row or -1), same 9 stage-2 paths.

Placement (the cross-cluster evidence, results/linear_subsection_trees_optuna/crosscluster/):
  stage 1 on CARC epyc-7513 (the canonical class).  OOS rows [H2_FROM, N_OOS) of every arm go to
  Hoffman2 intel-E5-2650v4 (AVX2, the vector class of epyc-7513) ONLY IF the Hoffman2 gate
  (stored live_feasible tuning points and refits reproduced bit for bit on that class) passes;
  the Hoffman2 tasks refuse to run without their model's GATE flag.  Rows a failed gate leaves
  are re-run on CARC (task file *_s1_h2.txt submitted there).

Chunk sizes (expected work only; stage 1 resumes from saved points): the stored masked run's
per-point cost of live_feasible (CARC core-seconds: lgbm 85, xgb 127, rf 122, the merged
trials' study_sec) scaled by MOMVIX_FACTOR = 0.5 (about half the kept columns), CARC tasks of
~12 min on 20 CPUs, Hoffman2 tasks of ~30 min on 8 slots at H2_SLOWDOWN = 2 per core.

Writes cluster/<prefix>_optuna_tasks_{canary,s1,s1_carc,s1_h2,s2,gate,h2audit}.txt (LF).
Run from the repo root:  python cluster/momvix_optuna_make_tasks.py   (the moments_vix run's files)

Second bucket (user, 2026-10-06): moments_vol_vix = moments_vix + sumvolume, CARC only, sized for
throughput from the moments_vix run's measured CARC costs (merged study_sec: lgbm 48, xgb 67,
rf 77 core-s per point; stage 2 walls lgbm 6.5, xgb 10, rf 18.7 min on one 20-CPU chunk):
  python cluster/momvix_optuna_make_tasks.py --bucket moments_vol_vix --prefix momvolvix       --h2-from 1469 --carc-task-min 6 --measured --s2-chunks lgbm=2,xgb=3,rf=4
"""

from __future__ import annotations

import argparse

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = 2000  # TRAIN_WIN rows = the halo every chunk carries
N_OOS = 1469  # bar1600 forecast rows (2018-06-25 .. 2024-04-30), as every 16:00 arm
SEG = "bar1600"
BUCKET = "moments_vix"
MODELS = ("lgbm", "xgb", "rf")
CANARY_ROWS = 20  # one wave of the 20-process pool (the stored run's canary)
H2_FROM = 1100  # rows [H2_FROM, N_OOS) of each arm: Hoffman2 if its gate passes
# stored masked run, live_feasible, mean CARC core-seconds per tuning point (study_sec)
LIVE_SEC_PER_POINT = {"lgbm": 85.0, "xgb": 127.0, "rf": 122.0}
MOMVIX_FACTOR = 0.5
CARC_POOL, CARC_TASK_SEC = 20, 12 * 60
H2_SLOTS, H2_TASK_SEC, H2_SLOWDOWN = 8, 30 * 60, 2.0
# the gate: stored live_feasible tuning points / refits of the canonical (epyc-7513) run
GATE_BUCKET, GATE_ROW0, GATE_ROWS = "live_feasible", 700, 20


def line(bucket: str, model: str, k: int, r0: int, r1: int) -> str:
    return (
        f"{bucket} {model} {W} {SEG} {k} {W + r0} {-1 if r1 >= N_OOS else W + r1} {W}\n"
    )


def split(r0: int, r1: int, rows: int) -> list[tuple[int, int]]:
    """[r0, r1) into chunks of about `rows` rows (equal sizes, at least one chunk)."""
    n = max(1, round((r1 - r0) / rows))
    edges = [r0 + (r1 - r0) * i // n for i in range(n + 1)]
    return list(zip(edges[:-1], edges[1:]))


# measured on CARC in the moments_vix run (merged trials' study_sec / 1469 points)
MEASURED_SEC_PER_POINT = {"lgbm": 48.0, "xgb": 67.0, "rf": 77.0}


def main() -> None:
    global BUCKET, H2_FROM, CARC_TASK_SEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default="momvix")
    ap.add_argument("--h2-from", type=int, default=H2_FROM)
    ap.add_argument("--carc-task-min", type=float, default=CARC_TASK_SEC / 60)
    ap.add_argument(
        "--measured",
        action="store_true",
        help="size by the moments_vix run's CARC costs",
    )
    ap.add_argument(
        "--s2-chunks",
        default="",
        help="e.g. lgbm=2,xgb=3,rf=4 (default one chunk per arm)",
    )
    a = ap.parse_args()
    BUCKET, H2_FROM, CARC_TASK_SEC = a.bucket, a.h2_from, a.carc_task_min * 60
    s2n = {k: int(v) for k, v in (kv.split("=") for kv in a.s2_chunks.split(",") if kv)}
    out: dict[str, list[str]] = {
        k: [] for k in ("canary", "s1", "s1_carc", "s1_h2", "s2", "gate", "h2audit")
    }
    for m in MODELS:
        sec = (
            MEASURED_SEC_PER_POINT[m]
            if a.measured
            else LIVE_SEC_PER_POINT[m] * MOMVIX_FACTOR
        )
        carc_rows = max(CARC_POOL, int(CARC_POOL * CARC_TASK_SEC / sec))
        h2_rows = max(H2_SLOTS, int(H2_SLOTS * H2_TASK_SEC / (sec * H2_SLOWDOWN)))
        k = 0
        out["canary"].append(line(BUCKET, m, k, 0, CANARY_ROWS))
        out["s1"].append(line(BUCKET, m, k, 0, CANARY_ROWS))
        for r0, r1 in split(CANARY_ROWS, H2_FROM, carc_rows):
            k += 1
            out["s1"].append(line(BUCKET, m, k, r0, r1))
            out["s1_carc"].append(line(BUCKET, m, k, r0, r1))
        if H2_FROM < N_OOS:
            for r0, r1 in split(H2_FROM, N_OOS, h2_rows):
                k += 1
                out["s1"].append(line(BUCKET, m, k, r0, r1))
                out["s1_h2"].append(line(BUCKET, m, k, r0, r1))
        for j, (r0, r1) in enumerate(split(0, N_OOS, -(-N_OOS // s2n.get(m, 1)))):
            out["s2"].append(line(BUCKET, m, j, r0, r1))
        # chunk ids 900 (gate) / 950 (audit): never a fleet chunk
        out["gate"].append(line(GATE_BUCKET, m, 900, GATE_ROW0, GATE_ROW0 + GATE_ROWS))
        out["h2audit"].append(line(BUCKET, m, 950, 0, CANARY_ROWS))
    for name, ls in out.items():
        p = ROOT / "cluster" / f"{a.prefix}_optuna_tasks_{name}.txt"
        p.write_text("".join(ls), encoding="utf-8", newline="\n")
        print(f"wrote {p.relative_to(ROOT)}: {len(ls)} lines")
        for ln in ls:
            print("   ", ln.rstrip())


if __name__ == "__main__":
    main()
