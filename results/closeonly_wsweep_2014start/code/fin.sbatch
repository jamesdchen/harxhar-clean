#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
# RULE.txt section 4: FAM=linear -- the 16:00 scores of the six W; FAM=trees -- merge of the five new W (every piece,
# --require-all), the CPU-model check, then the scores (W = 3,000 = the reused three-way CLOSE forecasts after gate T).
set -euo pipefail
: "${CODE:?}" "${FAM:?}"
source "$CODE/results/closeonly_wsweep_2014start/code/env.sh"
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
hostlog "fin $FAM"
if [ "$FAM" = trees ]; then
  R=$BASE/runs/trees
  A=()
  for W in 1000 1500 2000 2500 3500; do
    "$(pyof gb)" experiments/closewin_trees.py merge --arm "CLOSE_w$W" --design-root "$DES" --out "$R" --require-all
    A+=(--arm "CLOSE_w$W")
  done
  "$(pyof gb)" experiments/closewin_trees.py cpus --runs "$R" "${A[@]}"
  sweep_env
  "$(pyof sweep)" experiments/closewin_score.py trees "$R" /scratch1/jc_905/trees3way/scores_w1 "$BASE/scores/trees"
else
  sweep_env
  "$(pyof sweep)" experiments/closewin_score.py linear "$BASE/runs/linear" \
    "$CODE/results/linear_train3way_2014start_w3000/scores_all" "$BASE/scores/linear"
fi
echo "[$(date +%T)] done"
