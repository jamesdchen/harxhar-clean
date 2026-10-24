# Three-way training-sample comparison, pooled trees (status)

Rule: `RULE.txt` (committed before any job).  Code: `experiments/trees_train3way.py` (walk / merge / time / tasks /
gates / design check), `experiments/trees_train3way_score.py` (CARC scoring), `experiments/trees_train3way_econ.py`
(local close economics), `code/` (CARC job scripts: `env.sh`, `stage.sbatch`, `line.sbatch`, `walk.sbatch`,
`submit1.sh`).  Staging: `/scratch1/jc_905/trees3way` on CARC Discovery.

| Time (PDT) | Event |
|---|---|
| 2026-10-24 00:06 | RULE.txt and code committed (d6e90d5) and pushed; no job before it |
| 00:08 | phase 1 submitted (`code/submit1.sh d6e90d5`; code archive `/scratch1/jc_905/trees3way/code_d6e90d5`; import of the module checked on the login node with both class environments) |

Phase 1 jobs (CARC, account pollok_1603, partition main, EPYC 7513 / 7542):

| Job | Name | What | Depends on |
|---|---|---|---|
| 13369745 | t3t_stage | design copies (md5) + design check (gate 0) | - |
| 13369746 | t3t_g1a | gate 1, 8 threads (6 tasks) | stage |
| 13369747 | t3t_g1b | gate 1, the forest at 20 threads (2) | stage |
| 13369748 | t3t_g2a | gate 2, LightGBM / XGBoost and the ALL48_N3k forest (20) | stage |
| 13369749 | t3t_g2b | gate 2, the ALL48 forest at 20 threads (4) | stage |
| 13369750 | t3t_tm8 | time probe, 8 threads (10) | stage |
| 13369751 | t3t_tm20 | time probe, the forest at 20 threads (3) | stage |
| 13369752 | t3t_gate | gate 1 + gate 2 comparisons (2) | g1a, g1b, g2a, g2b |
| 13369753 | t3t_tasks | task files and projections of every arm (6) | tm8, tm20 |
| 13369754 | t3t_cap | projection against 25,000 CPU-hours | tasks |
| 13369755 | t3t_w1g | walks CLOSE / RTH13_N3k / ALL48_N3k, LightGBM + XGBoost (<= 50, %30) | gate, tasks |
| 13369756 | t3t_w1r | the same arms, the forest (<= 50, %30) | gate, tasks |
| 13369757 | t3t_w2g | walks ALL48_N39k, LightGBM + XGBoost (<= 40, %15) | gate, cap |
| 13369758 | t3t_w2r | walks ALL48_N39k, the forest at 20 threads (<= 150, %25) | gate, cap |
| 13369759 | t3t_m1 | merge CLOSE, RTH13_N3k, ALL48_N3k (3) | w1g, w1r (afterany) |
| 13369760 | t3t_s1 | interim scores -> scores_w1 | m1 |
| 13369761 | t3t_m2 | merge ALL48_N39k | w2g, w2r (afterany) |
| 13369762 | t3t_s2 | interim scores -> scores_w2 (last job of phase 1) | m2, s1 |

Phase 2 (ALL48, 3,000 days): `code/submit2.sh` after the gates and the projection are read.
