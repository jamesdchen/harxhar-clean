#!/bin/bash
# Score the VIX-representation buckets on both roots with both scorers: the
# spec's own comparison and the fair causal re-score (one-bar arms admitted).
#SBATCH --job-name=rep_score
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:$PWD/notebooks"
score_one() {  # <root> <bucket>
  local R=$1 B=$2
  [ -d "$R/$B" ] || { echo "no arms at $R/$B"; return 0; }
  echo "$R/$B: arms finished $(find "$R/$B" -name DONE | wc -l)"
  find "$R/$B" -name run.log | while read -r f; do
    [ -f "$(dirname "$f")/DONE" ] || echo "NOT DONE: $f: $(tail -1 "$f" | cut -c1-200)"
  done
  python -u experiments/score_linear_subsection.py --bucket "$B" --root "$R" > "$R/$B/score.log" 2>&1 \
    || echo "SCORER FAILED for $R/$B: $(tail -2 "$R/$B/score.log")"
  python -u experiments/score_linear_subsection_causal.py --bucket "$B" --root "$R" > "$R/$B/score_causal.log" 2>&1 \
    || echo "CAUSAL SCORER FAILED for $R/$B: $(tail -2 "$R/$B/score_causal.log")"
  touch "$R/$B/SCORED" "$R/$B/SCORED_CAUSAL"
}
for B in ivrep_target_scale ivrep_term ivrep_innovations ivrep_vrp ivrep_all ivrep_slice_slope; do
  for R in results/linear_subsection results/linear_subsection_lassofix; do
    score_one "$R" "$B"
  done
done
