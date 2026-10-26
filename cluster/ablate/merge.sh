#!/bin/bash
# Final tree merge of the ablation run (RULE.txt section 8, addendum B (b)): experiments/ablate_widen.py in one process
# -- T2 over the four gate dirs (Slurm Gold 6240, Xeon 6736P, Gold 6442Y; UGE Gold 6140) -> runs/gate2_final.json,
# ablate_trees.py merge of every complete (estimator, arm) -> runs/trees/forecasts/, and the compact
# runs/trees/trees_wide.parquet for the laptop.  Arms not complete are listed, not merged.  (The linear merge ran on its
# own: runs/lin/forecasts_linear.parquet.)  Writes only those outputs.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
G=$B/gate/s6240,$B/gate/s6736p,$B/gate/s6442y,$B/gate/u6140
echo "merge job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_widen.py --out $B/runs/trees --gates $G \
  --tasks $TD/trees.txt --to $B/runs/trees/trees_wide.parquet
echo "merge done rc $? $(date)"
