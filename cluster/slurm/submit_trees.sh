#!/bin/bash
# Per-bar TREE forecasts (LightGBM, XGBoost, random forest) beside the per-bar
# linear arms: buckets all_features, baseline, live_feasible x the 13 RTH bars
# (bar1000..bar1600) at tw 2000, global lags, production HAR ladder
# (specs/causal_tune_trees.py; TreeSHAP persisted for bar1600).  Canary first
# (live_feasible x lgbm,xgb,rf x 2000 x the 15:30-16:00 bar: all three model
# paths incl. RF's shap import, on real data, full series); it writes
# $ROOT/CANARY_OK only if all three arms succeed.  The GB fleet (lgbm, xgb;
# 1 cpu per task) and the RF fleet (RF_CPUS per task) are held on it with
# afterok and refuse to run unless CANARY_OK exists; the canary's three arms
# carry DONE, so the fleets skip them.  The scorer is held on both fleets with
# afterany (a failed arm does not block scoring the rest; it lists NOT DONE).
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_trees.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results

# Set from the local smoke (2026-09-24, one core, laptop under load; ~160
# refits per arm at REFIT_EVERY=10, data prep 2-4 min per arm):
#   LightGBM all_features (p=640) 3.5-10 s/refit   -> 0.2-0.5 h per arm
#   XGBoost  all_features          6-11 s/refit    -> 0.3-0.5 h per arm
#   RF       all_features        110 s/refit.core  -> ~5 core-h per arm
#   RF       live_feasible (p=244) 35 s/refit.core -> ~1.6 core-h per arm
#   TreeSHAP (bar1600 only): LightGBM/XGBoost ~0.03 s, RF ~6 s per 10-row block
# Limits carry a 3-4x margin.  The canary runs at 4 cores so its RF arm is
# not the bottleneck of the whole campaign.
CANARY_CPUS=${CANARY_CPUS:-4}
CANARY_TIME=${CANARY_TIME:-4:00:00}
CANARY_MEM=${CANARY_MEM:-16G}
GB_TIME=${GB_TIME:-4:00:00}
GB_MEM=${GB_MEM:-16G}
GB_CONC=${GB_CONC:-40}
RF_CPUS=${RF_CPUS:-4}
RF_TIME=${RF_TIME:-8:00:00}
RF_MEM=${RF_MEM:-16G}
RF_CONC=${RF_CONC:-39}

CANARY=cluster/trees_tasks_canary.txt
GBT=cluster/trees_tasks_gb.txt
RFT=cluster/trees_tasks_rf.txt
ROOT=results/linear_subsection_trees
FLAG=$ROOT/CANARY_OK
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing: run the ship script first (bash cluster/slurm/ship_trees_carc.sh, locally)"; exit 1; }
for f in specs/causal_tune_trees.py experiments/score_trees_subsection.py "$CANARY" "$GBT" "$RFT"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first"; exit 1; }
done
mkdir -p "$ROOT"
rm -f "$FLAG"
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J tr_canary --cpus-per-task="$CANARY_CPUS" --mem="$CANARY_MEM" --time="$CANARY_TIME" \
      --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$ROOT,WRITE_FLAG=$FLAG cluster/slurm/trees_pack.sbatch)
GB=$($SUBMIT --parsable -J tr_gb --array=1-"$(wc -l < "$GBT")"%"$GB_CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task=1 --mem="$GB_MEM" --time="$GB_TIME" \
      --export=ALL,TASKFILE=$GBT,RESULTS_ROOT=$ROOT,CANARY_FLAG=$FLAG cluster/slurm/trees_pack.sbatch)
RF=$($SUBMIT --parsable -J tr_rf --array=1-"$(wc -l < "$RFT")"%"$RF_CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task="$RF_CPUS" --mem="$RF_MEM" --time="$RF_TIME" \
      --export=ALL,TASKFILE=$RFT,RESULTS_ROOT=$ROOT,CANARY_FLAG=$FLAG cluster/slurm/trees_pack.sbatch)
S=$($SUBMIT --parsable -J tr_score --dependency=afterany:"$GB":"$RF" cluster/slurm/trees_score.sbatch)
echo "canary=$CAN gb=$GB rf=$RF score=$S" | tee logs/submitted_trees_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -10
