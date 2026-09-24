#!/bin/bash
# The FREE-FEED bucket (free_feasible = live_feasible minus the liquidity
# columns; live/close_signal rebuilds it from free 1-minute bars) on CARC
# Discovery (Slurm): 13 per-bar arms x ridge / enet / lasso at tw 2000.  Canary
# first (free_feasible x fixed lasso x 500 x the 15:30-16:00 bar); both fleets
# are held on it with afterok and refuse to run unless it wrote DONE; the
# scorer is held on both fleets.  No pooled arms are fitted: the bucket's
# `none` directories are symlinks to R0's (live_feasible's) pooled arms, so the
# CARC scorer measures it against R0's pooled model (per-bar arms only, ~35 min).
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_free.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
CANARY=cluster/free_tasks_canary.txt
RE=cluster/free_tasks_ridge_enet.txt
LA=cluster/free_tasks_lasso.txt
ROOT=results/linear_subsection
LROOT=results/linear_subsection_lassofix
B=free_feasible
FLAG=$LROOT/$B/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
mkdir -p "$ROOT/$B/none" "$LROOT/$B/none"
for E in ridge reclasticnet; do
  [ -e "$ROOT/$B/none/$E" ] || ln -s "$PWD/$ROOT/live_feasible/none/$E" "$ROOT/$B/none/$E"
done
[ -e "$LROOT/$B/none/reclasso" ] || ln -s "$PWD/$LROOT/live_feasible/none/reclasso" "$LROOT/$B/none/reclasso"
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J free_canary --time=2:00:00 --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
A=$($SUBMIT --parsable -J free_re --array=1-"$(wc -l < "$RE")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/slurm/subsection_pack.sbatch)
L=$($SUBMIT --parsable -J free_lasso --array=1-"$(wc -l < "$LA")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
S=$($SUBMIT --parsable -J free_score --dependency=afterok:"$A":"$L" cluster/slurm/subsection_score_free.sbatch)
echo "canary=$CAN ridge_enet=$A lasso=$L score=$S" | tee logs/submitted_free_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -10
