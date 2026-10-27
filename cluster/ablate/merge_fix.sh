#!/bin/bash
# Addendum D merges.  WHAT=lin: gate LF (the 10 patched full-set walks against fomcfix lin_B.parquet, the 4 HAR walks
# against w3000) -> gate/gateLF.json, then the merge -> runs/lin_fix/forecasts_linear.parquet.  WHAT=trees: T2 over the
# gate dirs, the add_fomc arms merged -> runs/trees_fix/forecasts/, runs/trees_fix/trees_wide.parquet.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
echo "merge_fix ${WHAT:?} job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
if [ "$WHAT" = lin ]; then
  NT=1 $P experiments/ablate_linear.py gate --root $B/runs/lin_fix --out $B/gate/gateLF.json \
    --ref-full $B/code/results/fomcfix_2026-10-26/forecasts/lin_B.parquet
  echo "gate LF rc $?"
  NT=1 $P experiments/ablate_linear.py merge --root $B/runs/lin_fix --tasks $TD/linear.txt --designs $B/designs_fix/lin \
    --full $B/designs_fix/full --out $B/runs/lin_fix/forecasts_linear.parquet
else
  NT=1 $P experiments/ablate_widen.py --out $B/runs/trees_fix --gates $B/gate/s6240,$B/gate/s6736p,$B/gate/s6442y,$B/gate/u6140 \
    --tasks $TD/trees_fomcfix.txt --to $B/runs/trees_fix/trees_wide.parquet
fi
echo "merge_fix done rc $? $(date)"
