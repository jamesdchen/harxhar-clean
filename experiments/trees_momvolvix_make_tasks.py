"""Task files of the moments_vol_vix 16:00 tree workstream (2026-10-06, user request).

The user (2026-10-06): moments_vix plus ES volume, and a lean version of it.  Buckets (src/data/loading.py),
each on the HAR + calendar backbone: moments_vol_vix = SUBGROUPS["moments"] (sumret, sumabsret, sumret3,
sumret4, sumpret2, sumbipow, sumautocov) + sumvolume + vix; lean_vol_vix = sumret, sumabsret, sumret4,
sumautocov, sumvolume, vix.  Both run in one fleet (every task line carries its bucket).  The trees run EXACTLY as the masked de-duplicated
16:00 runs (writeup/CAMPAIGN_16H_2026-09-29.md): rungs T1 / T10 (specs/causal_tune_trees.py)
and RS1 / RS10 (specs/causal_tune_trees_tuned.py), WINDOW_MASK=1, bar1600, TRAIN_WIN 2000,
LightGBM / XGBoost / random forest, through the campaign's unedited pack scripts
(cluster/slurm/trees_mask_pack.sbatch, cluster/slurm/treestuned_mask_pack.sbatch) and reducers.

Chunk geometry and packing are the campaign generator's (experiments/trees_cadence_make_tasks.py:
its ``chunks`` and ``sec_per_row`` are called, with this bucket's per-refit seconds added to its
FIT_SEC / PREP_SEC tables); chunking changes no number (the campaign's gates: chunked =
unchunked; one thread per fit).  Per-refit seconds of moments_vol_vix (118 columns, 71 kept by the
window mask on the first window): the live_feasible values (232 columns, 138 kept) scaled by the
kept-column ratio 71 / 138 -- histogram and split costs grow with the column count -- an estimate
used for chunk sizes only.

Files written (cluster/):
  trees_momvolvix_tasks_untuned.txt   "<pack> <rung> <bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
      pack 0 = the canary (chunk c0 of every T1 and T10 arm, as the campaign);
      packs 1..A  = group A (CARC): every other T10 chunk and the T1 LightGBM / XGBoost chunks;
      packs A+1.. = group B: the T1 random-forest chunks, the share Hoffman2 may host if its
                    gate passes (else CARC runs these packs too)
  trees_momvolvix_tasks_rs10.txt, trees_momvolvix_tasks_rs1.txt   "<bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
      one tuning period (250 rows) per chunk-arm, the campaign's format
  trees_momvolvix_tasks_rs_canary.txt   chunk 0 of the three models, one line per bucket
  trees_momvolvix_gate_lines.txt   "<rung> <bucket> <model> <chunk> <start> <nrows>": the stored
      all_features cells re-run by experiments/trees_momvolvix_gate.py (two chunks per untuned arm,
      chunk 0 and a later one; RS1 chunks 0 and 4; RS10 chunk 0)

Run from the repo root: python experiments/trees_momvolvix_make_tasks.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import trees_cadence_make_tasks as tcm  # noqa: E402

# the two input sets of this run (user, 2026-10-06), each beside moments_vix: moments_vol_vix (moments_vix +
# sumvolume, 9 columns) and lean_vol_vix (sumret, sumabsret, sumret4, sumautocov, sumvolume, vix: 6 columns)
BUCKETS = {"moments_vol_vix": 9, "lean_vol_vix": 6}  # bucket -> exogenous columns
# one-thread seconds per refit measured on the moments_vix arms (fit_sec_mean of their reduce gates,
# results/linear_subsection_trees_mask_momvix/t1/reduce_gates.csv), scaled by the column ratio 9 / 8
# (sumvolume added): an estimate used for chunk sizes only
MOMVIX_FIT_SEC = {"lgbm": 1.018, "xgb": 0.644, "rf": 5.090}
for _b, _ncol in BUCKETS.items():
    for _m in ("lgbm", "xgb", "rf"):
        tcm.FIT_SEC[(_m, _b)] = (
            MOMVIX_FIT_SEC[_m] * _ncol / 8
        )  # moments_vix: 8 exogenous columns
    tcm.PREP_SEC[_b] = tcm.PREP_SEC["live_feasible"]
# maximum throughput (main session, 2026-10-06): half the campaign's 600-second processes, so the
# fleet spreads over twice the cores (chunking changes no number: the campaign's chunked = unchunked)
tcm.TARGET_SEC = 300
W, SEG, N_OOS, TUNE_PER = tcm.W, tcm.SEG, tcm.N_OOS, tcm.TUNE_PER
MODELS = tcm.MODELS
GROUP_B = {("t1", "rf")}  # the share Hoffman2 may host (rung, model)


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
        for bucket in BUCKETS:
            for m in MODELS:
                spr = tcm.sec_per_row(m, bucket, rung)
                rows = int((tcm.TARGET_SEC - tcm.PREP_SEC[bucket]) / spr)
                for k, s, e in tcm.chunks(min(rows, N_OOS), re_):
                    n = (e if e >= 0 else W + N_OOS) - s
                    procs.append(
                        (tcm.PREP_SEC[bucket] + n * spr, rung, bucket, m, k, s, e)
                    )
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
    f = ROOT / "cluster" / "trees_momvolvix_tasks_untuned.txt"
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
            for bucket in BUCKETS:
                tuned.append(
                    f"{bucket} {m} {W} {SEG} {k} {s} {-1 if k == n_t - 1 else s + TUNE_PER} {W}"
                )
    for rung in ("rs10", "rs1"):
        f = ROOT / "cluster" / f"trees_momvolvix_tasks_{rung}.txt"
        f.write_text("\n".join(tuned) + "\n", encoding="utf-8", newline="\n")
        print(f"{f.name}: {len(tuned)} chunk-arms")
    f = ROOT / "cluster" / "trees_momvolvix_tasks_rs_canary.txt"
    f.write_text(
        "".join(
            f"{b} lgbm,xgb,rf {W} {SEG} 0 {W} {W + TUNE_PER} {W}\n" for b in BUCKETS
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"{f.name}: {len(BUCKETS)} lines (chunk 0 of each bucket, three models)")


if __name__ == "__main__":
    main()
