#!/bin/bash
# Score the re-run lasso arms (results/linear_subsection_lassofix).  BUCKETS is
# a colon-separated list; a bucket whose pooled arms are not both DONE is
# skipped with a note, so the job can run once early and once at the end.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=3:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

R=results/linear_subsection_lassofix
for B in ${BUCKETS//:/ }; do
  N=$(find "$R/$B" -name DONE | wc -l)
  P=$(find "$R/$B/none" -name DONE 2>/dev/null | wc -l)
  echo "$B: arms finished $N of 36, pooled $P of 2"
  if [ "$P" -lt 2 ]; then echo "  pooled arms not finished; skipped"; continue; fi
  python -u experiments/score_linear_subsection.py --bucket "$B" --root "$R" > "$R/$B/score.log" 2>&1 \
    || echo "SCORER FAILED for $B: $(tail -2 "$R/$B/score.log")"
  touch "$R/$B/SCORED"
done
touch "$R/SCORED_${TAG}"
