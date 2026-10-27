#!/bin/bash
# 2026-10-26 addendum F merges.  WHAT=lin: gate LF again (the 10 patched full-set walks of runs/lin_rmfomc against
# fomcfix lin_B.parquet, the 4 HAR walks against w3000) -> gate/gateLF_rmfomc.json, then the merge ->
# runs/lin_rmfomc/forecasts_linear.parquet and the compact runs/lin_rmfomc/linear_wide.parquet (add_rm_fomc x 4 models).
# WHAT=trees: T2 over the gate dirs, add_rm_fomc merged -> runs/trees_rmfomc/forecasts/, runs/trees_rmfomc/trees_wide.parquet.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
echo "merge_rmfomc ${WHAT:?} job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
if [ "$WHAT" = lin ]; then
  NT=1 $P experiments/ablate_linear.py gate --root $B/runs/lin_rmfomc --out $B/gate/gateLF_rmfomc.json \
    --ref-full $B/code/results/fomcfix_2026-10-26/forecasts/lin_B.parquet
  echo "gate LF (rmfomc) rc $?"
  NT=1 $P experiments/ablate_linear.py merge --root $B/runs/lin_rmfomc --tasks $TD/linear_rmfomc.txt \
    --designs $B/designs_fix/lin --full $B/designs_fix/full --out $B/runs/lin_rmfomc/forecasts_linear.parquet \
    --wide $B/runs/lin_rmfomc/linear_wide.parquet
else
  NT=1 $P experiments/ablate_widen.py --out $B/runs/trees_rmfomc --gates $B/gate/s6240,$B/gate/s6736p,$B/gate/s6442y,$B/gate/u6140 \
    --tasks $TD/trees_rmfomc.txt --to $B/runs/trees_rmfomc/trees_wide.parquet
fi
echo "merge_rmfomc done rc $? $(date)"
