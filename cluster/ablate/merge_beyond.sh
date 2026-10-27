#!/bin/bash
# 2026-10-26 addendum G merges.  WHAT=lin: gate LF again (runs/lin_beyond's 10 patched full-set walks against fomcfix
# lin_B.parquet, the 4 HAR walks against w3000) -> gate/gateLF_beyond.json, then the merge ->
# runs/lin_beyond/forecasts_linear.parquet and the compact runs/lin_beyond/linear_wide.parquet (6 arms x 3 models).
# WHAT=trees: T2 over the gate dirs, the 12 (class, arm) pairs merged -> runs/trees_beyond/forecasts/,
# runs/trees_beyond/trees_wide.parquet.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
echo "merge_beyond ${WHAT:?} job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
if [ "$WHAT" = lin ]; then
  NT=1 $P experiments/ablate_linear.py gate --root $B/runs/lin_beyond --out $B/gate/gateLF_beyond.json \
    --ref-full $B/code/results/fomcfix_2026-10-26/forecasts/lin_B.parquet
  echo "gate LF (beyond) rc $?"
  NT=1 $P experiments/ablate_linear.py merge --root $B/runs/lin_beyond --tasks $TD/linear_beyond.txt \
    --designs $B/designs_fix/lin --full $B/designs_fix/full --out $B/runs/lin_beyond/forecasts_linear.parquet \
    --wide $B/runs/lin_beyond/linear_wide.parquet
else
  NT=1 $P experiments/ablate_widen.py --out $B/runs/trees_beyond --gates $B/gate/s6240,$B/gate/s6736p,$B/gate/s6442y,$B/gate/u6140 \
    --tasks $TD/trees_beyond.txt --to $B/runs/trees_beyond/trees_wide.parquet
fi
echo "merge_beyond done rc $? $(date)"
