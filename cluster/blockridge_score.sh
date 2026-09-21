#!/bin/bash
# Reduce the block-ridge chunk trees to yhat tables and score clock arms vs pooled.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=2:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments"

R=results/unification_clock
for ARM in blk2_user blk2_clockhar_a1 blk2_clockhar_a100 blk2_clockhar_a1000; do
  echo "$ARM: $(ls $R/$ARM/chunk_*.npz 2>/dev/null | wc -l) of 100 chunks"
  python -u experiments/dump_unif_yhat.py --arm "$ARM" --root "$R" --out "$R/yhat_$ARM.parquet"
done
python -u experiments/score_blockridge_clock.py --root "$R" > "$R/score.log" 2>&1 \
  || echo "SCORER FAILED: $(tail -3 "$R/score.log")"
tail -6 "$R/score.log"
touch "$R/SCORED"
