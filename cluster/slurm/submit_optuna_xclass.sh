#!/bin/bash
# Cross-CPU-class gate of the masked Optuna campaign (agent D's finding, 2026-09-29: the
# executor's X differs at rounding level between AVX-512 (xeon-4116) and AVX2 (epyc) nodes).
# Runs live_feasible x lgbm / xgb / rf stage 1 on OOS rows [0, 20) -- the first 20 tuning
# points of the fleet's chunk 0 of those arms -- on a xeon-4116 node (--constraint), with the
# canary's checks (a repeat at 20 processes on the same node must be bit-identical; stage 2
# on the chunk's rows from its own records), into $R/xclass (not part of the fleet's results).
# experiments/optuna_xclass_compare.py then compares the records and forecasts with the
# fleet's, whose chunks ran on the nodes experiments/optuna_chunk_hosts.py lists.
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_optuna_xclass.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}
CLASS=${CLASS:-xeon-4116}
TF=cluster/optuna_tasks_xclass.txt
[ -f "$TF" ] || { echo "$TF missing: ship first"; exit 1; }
J=$(sbatch --parsable -J op_xclass --constraint="$CLASS" --array=1-"$(wc -l < "$TF")" --cpus-per-task=20 --mem=16G \
      --time=1:00:00 "--export=ALL,TASKFILE=$TF,STAGE=tune,RESULTS_ROOT=$R/xclass,WINDOW_MASK=1,CANARY_CHECKS=1,REPEAT_CPUS=20" \
      cluster/slurm/optuna_pack.sbatch)
echo "xclass=$J class=$CLASS root=$R/xclass" | tee -a logs/submitted_optuna_carc.txt
