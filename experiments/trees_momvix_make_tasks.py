"""Task files of the moments_vix 16:00 tree workstream (2026-10-06, user request).

The user: "on top of no exog features and all_features, add a version of every model that only
uses the moment bucket + the VIX exog var by itself".  Bucket moments_vix = HAR + calendar
backbone + SUBGROUPS["moments"] (sumret, sumabsret, sumret3, sumret4, sumpret2, sumbipow,
sumautocov) + vix (src/data/loading.py).  The trees run EXACTLY as the masked de-duplicated
16:00 runs (writeup/CAMPAIGN_16H_2026-09-29.md): rungs T1 / T10 (specs/causal_tune_trees.py)
and RS1 / RS10 (specs/causal_tune_trees_tuned.py), WINDOW_MASK=1, bar1600, TRAIN_WIN 2000,
LightGBM / XGBoost / random forest, through the campaign's unedited pack scripts
(cluster/slurm/trees_mask_pack.sbatch, cluster/slurm/treestuned_mask_pack.sbatch) and reducers.

Chunk geometry and packing are the campaign generator's (experiments/trees_cadence_make_tasks.py:
its ``chunks`` and ``sec_per_row`` are called, with this bucket's per-refit seconds added to its
FIT_SEC / PREP_SEC tables); chunking changes no number (the campaign's gates: chunked =
unchunked; one thread per fit).  Per-refit seconds of moments_vix (118 columns, 71 kept by the
window mask on the first window): the live_feasible values (232 columns, 138 kept) scaled by the
kept-column ratio 71 / 138 -- histogram and split costs grow with the column count -- an estimate
used for chunk sizes only.

Files written (cluster/):
  trees_momvix_tasks_untuned.txt   "<pack> <rung> <bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
      pack 0 = the canary (chunk c0 of every T1 and T10 arm, as the campaign);
      packs 1..A  = group A (CARC): every other T10 chunk and the T1 LightGBM / XGBoost chunks;
      packs A+1.. = group B: the T1 random-forest chunks, the share Hoffman2 may host if its
                    gate passes (else CARC runs these packs too)
  trees_momvix_tasks_rs10.txt, trees_momvix_tasks_rs1.txt   "<bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
      one tuning period (250 rows) per chunk-arm, the campaign's format
  trees_momvix_tasks_rs_canary.txt   chunk 0 of the three models (one line)
  trees_momvix_gate_lines.txt   "<rung> <bucket> <model> <chunk> <start> <nrows>": the stored
      all_features cells re-run by experiments/trees_momvix_gate.py (two chunks per untuned arm,
      chunk 0 and a later one; RS1 chunks 0 and 4; RS10 chunk 0)

Run from the repo root: python experiments/trees_momvix_make_tasks.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import trees_cadence_make_tasks as tcm  # noqa: E402

BUCKET = "moments_vix"
KEPT_RATIO = (
    71 / 138
)  # moments_vix kept columns / live_feasible kept columns (first window)
for _m in ("lgbm", "xgb", "rf"):
    tcm.FIT_SEC[(_m, BUCKET)] = tcm.FIT_SEC[(_m, "live_feasible")] * KEPT_RATIO
tcm.PREP_SEC[BUCKET] = tcm.PREP_SEC["live_feasible"]
W, SEG, N_OOS, TUNE_PER = tcm.W, tcm.SEG, tcm.N_OOS, tcm.TUNE_PER
MODELS = tcm.MODELS
GROUP_B = {("t1", "rf")}  # the share Hoffman2 may host (rung, model)

# the gate's stored cells (all_features, the masked campaign's chunks; starts from
# cluster/trees_mask_tasks_untuned.txt and the 250-row tuning periods)
GATE_LINES = [
    "t1 all_features lgbm 0 2000 5",
    "t1 all_features lgbm 12 3092 5",
    "t1 all_features xgb 0 2000 5",
    "t1 all_features xgb 8 3152 5",
    "t1 all_features rf 0 2000 5",
    "t1 all_features rf 69 3104 5",
    "t10 all_features lgbm 0 2000 20",
    "t10 all_features lgbm 1 2910 20",
    "t10 all_features xgb 0 2000 20",
    "t10 all_features xgb 1 3430 20",
    "t10 all_features rf 0 2000 20",
    "t10 all_features rf 6 2960 20",
    "rs1 all_features lgbm 0 2000 3",
    "rs1 all_features lgbm 4 3000 3",
    "rs1 all_features xgb 0 2000 3",
    "rs1 all_features xgb 4 3000 3",
    "rs1 all_features rf 0 2000 3",
    "rs1 all_features rf 4 3000 3",
    "rs10 all_features lgbm 0 2000 20",
    "rs10 all_features xgb 0 2000 20",
    "rs10 all_features rf 0 2000 20",
]


def lpt(procs: list[tuple], first_pack: int) -> list[tuple[int, tuple]]:
    """Longest-processing-time-first packing into packs of <= PACK_PROCS processes (the campaign's)."""
    if not procs:
        return []
    procs = sorted(procs, key=lambda p: -p[0])
    n_packs = math.ceil(len(procs) / tcm.PACK_PROCS)
    load = [0.0] * n_packs
    size = [0] * n_packs
    out = []
    for p in procs:
        j = min(
            (i for i in range(n_packs) if size[i] < tcm.PACK_PROCS),
            key=lambda i: load[i],
        )
        load[j] += p[0]
        size[j] += 1
        out.append((first_pack + j, p))
    return out


def main() -> None:
    procs = []  # (est_sec, rung, bucket, model, chunk, start, end)
    for rung, (re_, _ie) in tcm.RUNGS.items():
        for m in MODELS:
            spr = tcm.sec_per_row(m, BUCKET, rung)
            rows = int((tcm.TARGET_SEC - tcm.PREP_SEC[BUCKET]) / spr)
            for k, s, e in tcm.chunks(min(rows, N_OOS), re_):
                n = (e if e >= 0 else W + N_OOS) - s
                procs.append((tcm.PREP_SEC[BUCKET] + n * spr, rung, BUCKET, m, k, s, e))
    canary = sorted((p for p in procs if p[4] == 0), key=lambda p: (p[1] != "t1", p[3]))
    fleet = [p for p in procs if p[4] != 0]
    group_a = [p for p in fleet if (p[1], p[3]) not in GROUP_B]
    group_b = [p for p in fleet if (p[1], p[3]) in GROUP_B]
    packed_a = lpt(group_a, 1)
    n_a = max((j for j, _ in packed_a), default=0)
    packed_b = lpt(group_b, n_a + 1)
    lines = [
        f"0 {r} {b} {m} {W} {SEG} {k} {s} {e} {W}" for (_t, r, b, m, k, s, e) in canary
    ] + [
        f"{j} {r} {b} {m} {W} {SEG} {k} {s} {e} {W}"
        for j, (_t, r, b, m, k, s, e) in sorted(
            packed_a + packed_b, key=lambda x: (x[0], x[1][1:])
        )
    ]
    f = ROOT / "cluster" / "trees_momvix_tasks_untuned.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    n_b = max((j for j, _ in packed_b), default=n_a)
    print(
        f"{f.name}: {len(canary)} canary processes (pack 0); group A {len(group_a)} processes in packs 1-{n_a}; "
        f"group B {len(group_b)} processes in packs {n_a + 1}-{n_b}; est max process "
        f"{max(p[0] for p in procs):.0f}s, total {sum(p[0] for p in procs) / 3600:.2f} core-h"
    )
    for rung in tcm.RUNGS:
        for m in MODELS:
            ps = [p for p in procs if p[1] == rung and p[3] == m]
            print(
                f"  {rung:3s} {m:4s} {len(ps):3d} chunks, est {sum(p[0] for p in ps) / 3600:5.2f} h"
            )
    n_t = math.ceil(N_OOS / TUNE_PER)
    tuned = []
    for m in MODELS:
        for k in range(n_t):
            s = W + TUNE_PER * k
            tuned.append(
                f"{BUCKET} {m} {W} {SEG} {k} {s} {-1 if k == n_t - 1 else s + TUNE_PER} {W}"
            )
    for rung in ("rs10", "rs1"):
        f = ROOT / "cluster" / f"trees_momvix_tasks_{rung}.txt"
        f.write_text("\n".join(tuned) + "\n", encoding="utf-8", newline="\n")
        print(f"{f.name}: {len(tuned)} chunk-arms")
    f = ROOT / "cluster" / "trees_momvix_tasks_rs_canary.txt"
    f.write_text(
        f"{BUCKET} lgbm,xgb,rf {W} {SEG} 0 {W} {W + TUNE_PER} {W}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{f.name}: 1 line (chunk 0, three models)")
    f = ROOT / "cluster" / "trees_momvix_gate_lines.txt"
    f.write_text("\n".join(GATE_LINES) + "\n", encoding="utf-8", newline="\n")
    print(f"{f.name}: {len(GATE_LINES)} gate lines")


if __name__ == "__main__":
    main()
