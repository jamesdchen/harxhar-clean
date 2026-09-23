#!/bin/bash
# The ATM implied slice (the deck's own object, re-inverted from the package
# midpoint at 10:00..15:30) as a regressor: buckets ivslice_only,
# live_feasible_ivslice, live_feasible_plus_ivslice; tw 500 only (the chain
# starts 2020-01).  Canary first (ivslice_only x fixed lasso x 500 x the
# 15:30-16:00 bar); the two fleets are held on it with afterok and refuse to
# run unless it wrote DONE; the scorer is held on both fleets.
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_ivslice.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
CANARY=cluster/ivslice_tasks_canary.txt
RE=cluster/ivslice_tasks_ridge_enet.txt
LA=cluster/ivslice_tasks_lasso.txt
LROOT=results/linear_subsection_lassofix
FLAG=$LROOT/ivslice_only/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
[ -f data/spxw_ivslice.parquet ] || { echo "data/spxw_ivslice.parquet missing: ship it first"; exit 1; }
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J iv_canary --time=2:00:00 --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
A=$($SUBMIT --parsable -J iv_re --array=1-"$(wc -l < "$RE")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/slurm/subsection_pack.sbatch)
L=$($SUBMIT --parsable -J iv_lasso --array=1-"$(wc -l < "$LA")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
S=$($SUBMIT --parsable -J iv_score --dependency=afterok:"$A":"$L" cluster/slurm/subsection_score_ivslice.sbatch)
echo "canary=$CAN ridge_enet=$A lasso=$L score=$S" | tee logs/submitted_ivslice_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -10
