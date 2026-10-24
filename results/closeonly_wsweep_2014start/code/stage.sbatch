#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
# RULE.txt section 1: the pooled sweep's six W designs copied off /scratch1 to $DES (md5-checked) with dates_all.csv,
# and their checks -> $BASE/design_check.json (exit 3 on a failure; every later job is afterok on this one).
set -euo pipefail
: "${CODE:?}"
source "$CODE/results/closeonly_wsweep_2014start/code/env.sh"
hostlog stage
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
"$(pyof sweep)" experiments/closewin.py stage \
  --sweep /scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22 \
  --rth13 /scratch1/jc_905/train3way/code/results/linear_train3way_2014start_w3000/RTH13/_work \
  --design-root $DES --out $BASE
