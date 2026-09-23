#!/bin/bash
# Score the live_feasible bucket with both scorers on both roots: the spec's own
# comparison (score_linear_subsection.py) and the fair causal re-score
# (score_linear_subsection_causal.py, which admits the one-bar arms and "rth").
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=3:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments"
B=live_feasible
for R in results/linear_subsection results/linear_subsection_lassofix; do
  echo "$R/$B: arms finished $(find "$R/$B" -name DONE | wc -l)"
  find "$R/$B" -name run.log | while read -r f; do
    [ -f "$(dirname "$f")/DONE" ] || echo "NOT DONE: $f: $(tail -1 "$f" | cut -c1-200)"
  done
  python -u experiments/score_linear_subsection.py --bucket "$B" --root "$R" > "$R/$B/score.log" 2>&1 \
    || echo "SCORER FAILED for $R/$B: $(tail -2 "$R/$B/score.log")"
  python -u experiments/score_linear_subsection_causal.py --bucket "$B" --root "$R" > "$R/$B/score_causal.log" 2>&1 \
    || echo "CAUSAL SCORER FAILED for $R/$B: $(tail -2 "$R/$B/score_causal.log")"
  touch "$R/$B/SCORED" "$R/$B/SCORED_CAUSAL"
done
