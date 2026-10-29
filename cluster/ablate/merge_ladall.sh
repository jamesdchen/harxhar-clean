#!/bin/bash
# 2026-10-28 addendum K merges on Hoffman2.  WHAT=lin: the 15 linear rung walks -> runs/lin_ladall/forecasts_linear.parquet
# and the compact runs/lin_ladall/linear_wide.parquet (ablate_linear.py merge --wide).  WHAT=trees: T2 over the addendum
# K gate dirs gate/ladall/*, the XGBoost rungs merged -> runs/trees_ladall/forecasts/, runs/trees_ladall/trees_wide.parquet
# (ablate_ladall.py merge).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
echo "merge_ladall ${WHAT:?} job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
if [ "$WHAT" = lin ]; then
  NT=1 $P experiments/ablate_linear.py merge --root $B/runs/lin_ladall --tasks $TD/linear_ladall.txt \
    --designs $B/designs_fix/lin --full $B/designs_fix/full --out $B/runs/lin_ladall/forecasts_linear.parquet \
    --wide $B/runs/lin_ladall/linear_wide.parquet
else
  G=$(ls -d $B/gate/ladall/*/ | sed 's:/$::' | paste -sd, -)
  NT=1 $P experiments/ablate_ladall.py merge --out $B/runs/trees_ladall --gates $G --tasks $TD/trees_ladall_h2.txt \
    --to $B/runs/trees_ladall/trees_wide.parquet
fi
echo "merge_ladall done rc $? $(date)"
