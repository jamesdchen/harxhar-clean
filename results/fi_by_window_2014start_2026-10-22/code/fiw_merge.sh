#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch1/jc_905/fi_window/logs/fiw_merge_%j.log
set -euo pipefail
B=/scratch1/jc_905/fi_window
export FIW_BASE=$B
for W in 1000 1500 2000 2500 3000 3500; do
  /scratch1/jc_905/wsweep/venv/bin/python -u $B/scripts/fiw_replay.py --wdays $W --merge
  cp $B/w$W/design_all_features/dates_oos.csv $B/out/dates_oos_w$W.csv
done
