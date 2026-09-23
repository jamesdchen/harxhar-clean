#!/bin/bash
# The chain's own implied variance as a regressor (mfiv_only, live_feasible_mfiv,
# live_feasible_plus_mfiv; tw 500 only -- the chain starts 2020-01) and the HAR-
# ladder axis (live_feasible at bar1430 / bar1600: base-2, base-3, and the same-
# clock ladder lag_scope=intra; tw 500 and 2000) on CARC Discovery (Slurm).
# Canary first (mfiv_only x fixed lasso x 500 x the 15:30-16:00 bar); every fleet
# is held on it with afterok and refuses to run unless it wrote DONE; the scorer
# is held on all fleets.  The HAR-ladder roots borrow the production pooled arms
# (base-5, lag_scope global) as the causal scorer's reference, so every ladder is
# measured against the same pooled model the incumbent per-bar arms were.
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_mfiv_har.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
CANARY=cluster/mfiv_tasks_canary.txt
RE=cluster/mfiv_tasks_ridge_enet.txt
LA=cluster/mfiv_tasks_lasso.txt
HL=cluster/harlag_tasks.txt
ROOT=results/linear_subsection
LROOT=results/linear_subsection_lassofix
HROOT=results/linear_subsection_har
FLAG=$LROOT/mfiv_only/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
[ -f data/spxw_mfiv.parquet ] || { echo "data/spxw_mfiv.parquet missing: ship it first"; exit 1; }
# HAR-ladder roots: one per variant, the production pooled arms linked in as the reference.
for V in base2 base3 intra; do
  R=$HROOT/$V/live_feasible/none
  mkdir -p "$R"
  for E in ridge reclasticnet; do
    [ -e "$R/$E" ] || ln -s "$PWD/$ROOT/live_feasible/none/$E" "$R/$E"
  done
  [ -e "$R/reclasso" ] || ln -s "$PWD/$LROOT/live_feasible/none/reclasso" "$R/reclasso"
done
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J mf_canary --time=2:00:00 --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
A=$($SUBMIT --parsable -J mf_re --array=1-"$(wc -l < "$RE")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/slurm/subsection_pack.sbatch)
L=$($SUBMIT --parsable -J mf_lasso --array=1-"$(wc -l < "$LA")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
H2=$($SUBMIT --parsable -J har_b2 --array=1-"$(wc -l < "$HL")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$HL,CANARY_FLAG=$FLAG,HAR_BASE=2,RESULTS_ROOT=$HROOT/base2 cluster/slurm/subsection_pack.sbatch)
H3=$($SUBMIT --parsable -J har_b3 --array=1-"$(wc -l < "$HL")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$HL,CANARY_FLAG=$FLAG,HAR_BASE=3,RESULTS_ROOT=$HROOT/base3 cluster/slurm/subsection_pack.sbatch)
HI=$($SUBMIT --parsable -J har_intra --array=1-"$(wc -l < "$HL")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$HL,CANARY_FLAG=$FLAG,LAG_SCOPE=intra,RESULTS_ROOT=$HROOT/intra cluster/slurm/subsection_pack.sbatch)
S=$($SUBMIT --parsable -J mf_score --dependency=afterok:"$A":"$L":"$H2":"$H3":"$HI" cluster/slurm/subsection_score_mfiv.sbatch)
echo "canary=$CAN ridge_enet=$A lasso=$L har_base2=$H2 har_base3=$H3 har_intra=$HI score=$S" | tee logs/submitted_mfiv_har_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -14
