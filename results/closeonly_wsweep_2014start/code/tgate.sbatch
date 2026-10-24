#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
# RULE.txt section 3, gate T: CLOSE_w3000, the first and last 10 16:00 origins of each class and set ($NT threads), then
# the comparison with the three-way CLOSE forecasts (exit 3 on a failure; every tree walk is afterok on this job).
set -euo pipefail
: "${CODE:?}"
source "$CODE/results/closeonly_wsweep_2014start/code/env.sh"
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
G=$BASE/runs/trees/gate
hostlog "tgate"
for e in lgbm xgb rf_skx; do
  for s in baseline all_features; do
    echo "[$(date +%T)] gate walk $e $s"
    "$(pyof $e)" experiments/closewin_trees.py walk --arm CLOSE_w3000 --est $e --set $s --ends 10 --tag ends10 \
      --threads "$NT" --design-root "$DES" --out "$G"
  done
done
"$(pyof gb)" experiments/closewin_trees.py gate --out "$G" --runs "$BASE/runs/trees" \
  --ref /scratch1/jc_905/trees3way/runs/CLOSE/forecasts.parquet --ref-md5 2f5661cde01750be1f5a44c47f88aad7
