#!/bin/bash
# 2026-10-29 addendum L merges on Hoffman2.  WHAT=lin: gate LF on the macro cache's subsets (runs/lin_mac's 10 full-set
# walks against fomcfix lin_B.parquet, the 4 HAR walks against w3000; ablate_linear.py gate) -> gate/gateLF_mac.json;
# only if it passes, the merge of every walk of tasks/linear_mac.txt -> runs/lin_mac/forecasts_linear.parquet and the
# compact runs/lin_mac/linear_wide.parquet (the macro arms, "<model>|<arm>").  WHAT=trees: T2 over the addendum L gate
# dirs gate/mac/* (each class's gate pieces bit-identical across the CPU models that passed it), the XGBoost arms of
# tasks/trees_mac_h2.txt merged -> runs/trees_mac/forecasts/, runs/trees_mac/trees_wide.parquet (ablate_trees.py
# mergemac; exit 3 if an arm is not merged).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
echo "merge_mac ${WHAT:?} job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
if [ "$WHAT" = lin ]; then
  NT=1 $P experiments/ablate_linear.py gate --root $B/runs/lin_mac --out $B/gate/gateLF_mac.json \
    --ref-full $B/code/results/fomcfix_2026-10-26/forecasts/lin_B.parquet
  rc=$?
  echo "gate LF (mac) rc $rc"
  [ $rc -eq 0 ] || { echo "gate LF failed: no merge"; exit 3; }
  NT=1 $P experiments/ablate_linear.py merge --root $B/runs/lin_mac --tasks $TD/linear_mac.txt \
    --designs $B/designs_mac/lin --full $B/designs_mac/lin/full --out $B/runs/lin_mac/forecasts_linear.parquet \
    --wide $B/runs/lin_mac/linear_wide.parquet
  rc=$?
else
  G=$(ls -d $B/gate/mac/*/ | sed 's:/$::' | paste -sd, -)
  echo "gates: $G"
  NT=1 $P experiments/ablate_trees.py mergemac --out $B/runs/trees_mac --gates $G --tasks $TD/trees_mac_h2.txt \
    --to $B/runs/trees_mac/trees_wide.parquet
  rc=$?
fi
echo "merge_mac done rc $rc $(date)"
exit $rc
