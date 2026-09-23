#!/bin/bash
# The live_feasible bucket on CARC Discovery (Slurm).  Canary first (fixed lasso x
# 500 x the 15:30-16:00 bar); the ridge / elastic-net packs (13 bars + rth +
# pooled) and the lasso packs are held on it with afterok and refuse to run
# unless it wrote DONE; the scorer is held on both fleets.
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_subsection_live.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
CANARY=cluster/livefeasible_tasks_canary.txt
RE=cluster/livefeasible_tasks_ridge_enet.txt
LA=cluster/livefeasible_tasks_lasso.txt
LROOT=results/linear_subsection_lassofix
FLAG=$LROOT/live_feasible/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J lf_canary --time=2:00:00 --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
A=$($SUBMIT --parsable -J lf_re --array=1-"$(wc -l < "$RE")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/slurm/subsection_pack.sbatch)
L=$($SUBMIT --parsable -J lf_lasso --array=1-"$(wc -l < "$LA")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
S=$($SUBMIT --parsable -J lf_score --dependency=afterok:"$A":"$L" cluster/slurm/subsection_score_live.sbatch)
echo "canary=$CAN ridge_enet=$A lasso=$L score=$S" | tee logs/submitted_live_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -12
