#!/bin/bash
# Representations of the VIX family as regressors (buckets ivrep_target_scale,
# ivrep_term, ivrep_innovations, ivrep_vrp, ivrep_all at tw 2000; ivrep_slice_slope
# at tw 500 -- the slice starts 2020-01) on CARC Discovery (Slurm).  Canary first
# (ivrep_target_scale x fixed lasso x 500 x the 15:30-16:00 bar); both fleets are
# held on it with afterok and refuse to run unless it wrote DONE; the scorer is
# held on both fleets.  No pooled arms are fitted: every ivrep bucket's `none`
# directory is a symlink to R0's (live_feasible's) pooled arms, so the CARC
# scorer measures each representation against R0's pooled model -- the right
# reference anyway -- and the campaign is per-bar arms only (~35 min, not
# hours).
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_ivrep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
CANARY=cluster/ivrep_tasks_canary.txt
RE=cluster/ivrep_tasks_ridge_enet.txt
LA=cluster/ivrep_tasks_lasso.txt
ROOT=results/linear_subsection
LROOT=results/linear_subsection_lassofix
FLAG=$LROOT/ivrep_target_scale/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
for f in data/vix_representations.parquet data/spxw_ivslice.parquet; do
  [ -f "$f" ] || { echo "$f missing: ship it first"; exit 1; }
done
# R0's pooled arms as every bucket's scorer reference (no fresh pooled fits).
for B in ivrep_target_scale ivrep_term ivrep_innovations ivrep_vrp ivrep_all ivrep_slice_slope; do
  mkdir -p "$ROOT/$B/none" "$LROOT/$B/none"
  for E in ridge reclasticnet; do
    [ -e "$ROOT/$B/none/$E" ] || ln -s "$PWD/$ROOT/live_feasible/none/$E" "$ROOT/$B/none/$E"
  done
  [ -e "$LROOT/$B/none/reclasso" ] || ln -s "$PWD/$LROOT/live_feasible/none/reclasso" "$LROOT/$B/none/reclasso"
done
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J rep_canary --time=2:00:00 --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
A=$($SUBMIT --parsable -J rep_re --array=1-"$(wc -l < "$RE")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/slurm/subsection_pack.sbatch)
L=$($SUBMIT --parsable -J rep_lasso --array=1-"$(wc -l < "$LA")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
S=$($SUBMIT --parsable -J rep_score --dependency=afterok:"$A":"$L" cluster/slurm/subsection_score_ivrep.sbatch)
echo "canary=$CAN ridge_enet=$A lasso=$L score=$S" | tee logs/submitted_ivrep_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -10
