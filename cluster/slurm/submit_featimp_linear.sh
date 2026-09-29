#!/bin/bash
# The slow part of the all_features LASSO drop-column walk (refits 75..124, penalty
# 0.001, ~220 units re-solved per refit with the spec's homotopy): one refit per task.
# Canary = the first line alone; the other lines are held afterok on it and refuse to
# run without its flag.  TIMING: ~1.3 s per re-solve on a contended laptop x ~220 units
# = ~5 min per refit; the 1 h limit carries >= 10x margin.
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_featimp_linear.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results/feature_importance_1530/_work/linear
TASKS=cluster/featimp_linear_tasks.txt
CAN_T=cluster/featimp_linear_tasks_canary.txt
FLAG=results/feature_importance_1530/_work/linear/CANARY_OK
for f in experiments/feature_importance_1530.py experiments/feature_importance_1530_trees.py \
         experiments/model_diagnostics_1530.py specs/causal_tune_linear.py \
         cluster/slurm/featimp_linear.sbatch "$TASKS" "$CAN_T" \
         results/feature_importance_1530/_work/input_all_features.npz \
         results/model_diagnostics_1530/capture_bar1600_all_features_reclasso.npz \
         results/linear_subsection/arms_hoffman2/all_features/reclasso/tw2000/results_bar1600.csv; do
  [ -f "$f" ] || { echo "$f missing: run bash cluster/slurm/ship_featimp_linear_carc.sh locally first"; exit 1; }
done
rm -f "$FLAG"
CAN=$(sbatch --parsable -J fil_canary --array=1-1 --export=ALL,TASKFILE=$CAN_T,WRITE_FLAG=$FLAG cluster/slurm/featimp_linear.sbatch)
FL=$(sbatch --parsable -J fil_fleet --array=1-"$(wc -l < "$TASKS")" --dependency=afterok:"$CAN" \
     --export=ALL,TASKFILE=$TASKS,CANARY_FLAG=$FLAG cluster/slurm/featimp_linear.sbatch)
echo "canary=$CAN fleet=$FL" | tee logs/submitted_featimp_linear_carc.txt
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %R" | head -8
