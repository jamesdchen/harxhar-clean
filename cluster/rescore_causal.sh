#!/bin/bash
# Re-score one feature bucket's subsection arms with a causal back-transform
# (experiments/score_linear_subsection_causal.py).  Read-only on the arm outputs.
#   ROOT_DIR   results root to score (default results/linear_subsection)
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=3:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments"

BUCKETS=(baseline moments liquidity market_ew market_vw sentiment implied_vol vol_demand fomc all_features)
B="${BUCKETS[$(( SGE_TASK_ID - 1 ))]}"
R="${ROOT_DIR:-results/linear_subsection}"
echo "task $SGE_TASK_ID on $(hostname): $B under $R  $(date)"
python -u experiments/score_linear_subsection_causal.py --bucket "$B" --root "$R" > "$R/$B/score_causal.log" 2>&1
tail -3 "$R/$B/score_causal.log"
touch "$R/$B/SCORED_CAUSAL"
