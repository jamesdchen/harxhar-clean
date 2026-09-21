#!/bin/bash
# Score every bucket of wave 2 once its packs and pooled arms are done.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=3:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

R=results/linear_subsection
for B in moments liquidity market_ew market_vw sentiment implied_vol vol_demand fomc all_features; do
  echo "$B: arms finished $(find $R/$B -name DONE | wc -l) of 108"
  find $R/$B -name run.log | while read -r f; do
    [ -f "$(dirname "$f")/DONE" ] || echo "NOT DONE: $f: $(tail -1 "$f" | cut -c1-200)"
  done
  python -u experiments/score_linear_subsection.py --bucket "$B" > "$R/$B/score.log" 2>&1 \
    || echo "SCORER FAILED for $B: $(tail -2 "$R/$B/score.log")"
  touch "$R/$B/SCORED"
done
touch "$R/WAVE2_SCORED"
