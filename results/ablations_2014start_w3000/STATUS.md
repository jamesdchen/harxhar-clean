# Augmented-HAR benchmarks and predictor-group ablations (status)

Rule: `RULE.txt` (committed before any job).  Code: `experiments/ablate_arms.py` (arms), `experiments/ablate_design.py`
(gate 0, the new-series design, the linear arms' caches, late-start windows), `experiments/ablate_linear.py` (linear
walks through `pool_linear_pretune.py`, gate L, merge), `experiments/ablate_trees.py` (tree walks, gates T1 / T2,
projection, merge), `cluster/ablate/` (Hoffman2 job scripts: `py.sh`, `prep0.sh`, `prep1.sh`, `lingate.sh`,
`tgate.sh`, `project.sh`, `pool.sh`, `merge.sh`).  Task files: `tasks/trees.txt` (470 lines), `tasks/linear.txt`
(135 lines; 1-14 = gate L).  Staging: `/u/scratch/j/jamesdc1/abl` on Hoffman2.

| Time (PDT) | Event |
|---|---|
| 2026-10-25 13:40 | RULE.txt written; code smoke-tested locally on a synthetic full-size design (tree walk / gate / projection code paths, linear walks on column-subset and classical caches) |
