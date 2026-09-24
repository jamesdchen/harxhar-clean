#!/bin/bash
# Does the VIX bucket need VVIX and VIX3M, and can the VIX's own path replace
# them?  Buckets vix_only, vix_rvol, live_vix_only, live_vix_rvol at tw 2000 on
# CARC Discovery (Slurm); references are implied_vol (the trio) and
# live_feasible (the trio in the live base), already on disk.  Canary first
# (live_vix_only x fixed lasso x 500 x the 15:30-16:00 bar); both fleets are
# held on it with afterok and refuse to run unless it wrote DONE; the scorer is
# held on both fleets.  No pooled arms are fitted: every bucket's `none`
# directory is a symlink to live_feasible's pooled arms (this CARC root carries
# no implied_vol pooled arms -- those campaigns ran on Hoffman2 -- so the
# HAR-only buckets are also referenced to live_feasible's pooled model; the
# fair comparison is the local per-bar one, compare_mfiv_harlag.py --family
# vixonly, which does not use the pooled arms).  Per-bar arms only, ~35 min.
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_vixonly.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
CANARY=cluster/vixonly_tasks_canary.txt
RE=cluster/vixonly_tasks_ridge_enet.txt
LA=cluster/vixonly_tasks_lasso.txt
ROOT=results/linear_subsection
LROOT=results/linear_subsection_lassofix
FLAG=$LROOT/live_vix_only/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
[ -f data/vix_rvol.parquet ] || { echo "data/vix_rvol.parquet missing: ship it first"; exit 1; }
# live_feasible's pooled arms as every bucket's scorer reference (no fresh pooled fits).
for B in vix_only vix_rvol live_vix_only live_vix_rvol free_feasible_vol; do
  mkdir -p "$ROOT/$B/none" "$LROOT/$B/none"
  for E in ridge reclasticnet; do
    [ -e "$ROOT/$B/none/$E" ] || ln -s "$PWD/$ROOT/live_feasible/none/$E" "$ROOT/$B/none/$E"
  done
  [ -e "$LROOT/$B/none/reclasso" ] || ln -s "$PWD/$LROOT/live_feasible/none/reclasso" "$LROOT/$B/none/reclasso"
done
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J vo_canary --time=2:00:00 --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
A=$($SUBMIT --parsable -J vo_re --array=1-"$(wc -l < "$RE")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/slurm/subsection_pack.sbatch)
L=$($SUBMIT --parsable -J vo_lasso --array=1-"$(wc -l < "$LA")" --dependency=afterok:"$CAN" \
      --export=ALL,TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=$LROOT cluster/slurm/subsection_pack.sbatch)
S=$($SUBMIT --parsable -J vo_score --dependency=afterok:"$A":"$L" cluster/slurm/subsection_score_vixonly.sbatch)
echo "canary=$CAN ridge_enet=$A lasso=$L score=$S" | tee logs/submitted_vixonly_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -10
