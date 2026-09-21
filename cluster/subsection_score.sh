#!/bin/bash
# Score every subsection arm against the pooled arm once the fleet is done.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=1:00:00,h_data=8G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

B=results/linear_subsection/baseline
echo "arms finished: $(find $B -name DONE | wc -l) of 108"
find $B -name run.log | while read -r f; do
  [ -f "$(dirname "$f")/DONE" ] || echo "NOT DONE: $f: $(tail -1 "$f" | cut -c1-200)"
done
python -u experiments/score_linear_subsection.py > "$B/score.log" 2>&1
tail -5 "$B/score.log"
touch "$B/SCORED"
