#!/bin/bash
# 16:00-bar FEATURE IMPORTANCE with the professor's measures (MDI, split count,
# permutation importance on the causal held-out tail, TreeSHAP, plus noise probes)
# for the untuned per-bar trees: experiments/feature_importance_1530_trees.py stage
# `trees` over buckets baseline / live_feasible / all_features x lgbm / xgb / rf,
# 147 refits each (every 10 sessions, 1,469 forecasts 2018-06-25 .. 2024-04-30).
# Canary first (cluster/featimp_tasks_canary.txt: 3 lines = every model path incl.
# RF's shap and the probe fit, on all three buckets' thread settings; CHECK_GATE=1
# reports whether the refits reproduce the stored forecasts).  The fleet
# (cluster/featimp_tasks.txt: 45 lines, 147 refits split into 7 chunks of 21 for
# live_feasible / all_features, one line per model for baseline) is held afterok on
# the canary and also refuses to run without the canary's flags.  The collector,
# held afterany on the fleet, lists missing outputs and writes FLEET_DONE.
#
# TIMINGS (the stored untuned runs on this cluster, seconds per refit fit): all_features
# lgbm 6.1 (1 thread) / xgb 3.9 (1) / rf 8.2 (4); live_feasible 0.7 / 0.6 / 2.4 (4 each);
# baseline <= 0.5.  A refit here adds the probe fit on every third refit (+1/3), TreeSHAP
# on the 10-row tail (RF ~1 s) and the permuted tails (<= 790 units x 2 schemes x 10 draws
# x 10 rows; 7 s on a contended laptop for all_features LightGBM, ~3x faster here).  An
# all_features RF chunk (21 refits) is ~10 min; the limits carry >= 6x margin.
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_featimp.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results/feature_importance_1530/_work/trees

CPUS=${CPUS:-4}                  # the largest stored thread count
MEM=${MEM:-8G}                   # the all_features input (3469 x 640) + <= 20,000-row predict batches
CANARY_TIME=${CANARY_TIME:-1:00:00}
FLEET_TIME=${FLEET_TIME:-2:00:00}
CONC=${CONC:-45}                 # every fleet task at once (45 x 4 = 180 cores)

CANARY=cluster/featimp_tasks_canary.txt
FLEET=cluster/featimp_tasks.txt
FLAG=results/feature_importance_1530/_work/trees/CANARY_OK
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing (the untuned tree campaign installs it)"; exit 1; }
for f in experiments/feature_importance_1530_trees.py cluster/slurm/featimp_pack.sbatch \
         cluster/slurm/featimp_collect.sbatch "$CANARY" "$FLEET" \
         results/feature_importance_1530/_work/input_baseline.npz \
         results/feature_importance_1530/_work/input_live_feasible.npz \
         results/feature_importance_1530/_work/input_all_features.npz; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_featimp_carc.sh, locally)"; exit 1; }
done
rm -f "$FLAG".* results/feature_importance_1530/_work/trees/FLEET_DONE
NC=$(wc -l < "$CANARY")
NF=$(wc -l < "$FLEET")
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J fi_canary --array=1-"$NC" --cpus-per-task="$CPUS" --mem="$MEM" --time="$CANARY_TIME" \
      --export=ALL,TASKFILE=$CANARY,WRITE_FLAG=$FLAG,CHECK_GATE=1 cluster/slurm/featimp_pack.sbatch)
FL=$($SUBMIT --parsable -J fi_fleet --array=1-"$NF"%"$CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task="$CPUS" --mem="$MEM" --time="$FLEET_TIME" \
      --export=ALL,TASKFILE=$FLEET,CANARY_FLAG=$FLAG,CANARY_N=$NC cluster/slurm/featimp_pack.sbatch)
CO=$($SUBMIT --parsable -J fi_collect --dependency=afterany:"$FL" \
      --export=ALL,TASKFILE=$FLEET cluster/slurm/featimp_collect.sbatch)
echo "canary=$CAN fleet=$FL collect=$CO" | tee logs/submitted_featimp_carc.txt
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.6D %R" | head -12
