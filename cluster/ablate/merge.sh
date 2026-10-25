#!/bin/bash
# Merge every complete arm of the ablation run (RULE.txt section 8): trees -> runs/trees/forecasts/<est>_<arm>.parquet
# (ablate_trees.py merge: one complete piece set, true_adj / B = the templates', CPU models per piece, T2 for mixed
# models), linear -> runs/lin/forecasts_linear.parquet (ablate_linear.py merge).  Arms not complete are listed, not
# merged.  Read-only on everything but those outputs.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
echo "merge job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
for pair in $(awk '{print $1 ":" $2}' $TD/trees.txt | sort -u); do
  e=${pair%%:*}
  a=${pair#*:}
  NT=1 $P experiments/ablate_trees.py merge --arm $a --est $e --out $B/runs/trees --gate2 $B/runs/gate2.json \
    > $B/logs/merge.$e.$a.log 2>&1 && echo "merged $e $a" || echo "NOT merged $e $a: $(tail -n 1 $B/logs/merge.$e.$a.log)"
done
NT=1 $P experiments/ablate_linear.py merge --root $B/runs/lin --tasks $TD/linear.txt --designs $B/designs/lin \
  --full $B/designs/full --out $B/runs/lin/forecasts_linear.parquet
echo "merge done $(date)"
