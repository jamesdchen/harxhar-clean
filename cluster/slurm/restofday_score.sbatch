#!/bin/bash
# Score the DIRECT rest-of-day arms (results/linear_subsection_restofday)
# against the per-bar reference tables shipped to $R/ref.  Held on the fleet
# with afterany, so a failed arm does not block the rest: every expected arm
# (from the canary + fleet task files: estimators x clocks per line) that did
# not finish is listed here (NOT DONE + the last line of its run.log).
#SBATCH --job-name=rod_score
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments"
R=results/linear_subsection_restofday
TASKS=(cluster/restofday_tasks_canary.txt cluster/restofday_tasks_fleet.txt)
mkdir -p "$R"
EXPECTED=$(cat "${TASKS[@]}" \
  | awk 'NF {m = split($2, a, ","); s = split($4, b, ","); n += m * s} END {print n + 0}')
echo "$R: arms finished $(find "$R" -name DONE | wc -l) of $EXPECTED"
cat "${TASKS[@]}" | while read -r BUCKET ESTS TW CLOCKS; do
  [ -n "${BUCKET:-}" ] || continue
  for EST in ${ESTS//,/ }; do
    for CLOCK in ${CLOCKS//,/ }; do
      OUT="$R/$BUCKET/rod$CLOCK/$EST/tw$TW"
      [ -f "$OUT/DONE" ] && continue
      if [ -f "$OUT/run.log" ]; then
        echo "NOT DONE: $OUT/run.log: $(tail -1 "$OUT/run.log" | cut -c1-200)"
      else
        echo "NOT DONE: $OUT: no run.log (never started)"
      fi
    done
  done
done
python -u experiments/score_rest_of_day.py --root "$R" --ref-dir "$R/ref" \
  > "$R/score.log" 2>&1 \
  || echo "SCORER FAILED for $R: $(tail -2 "$R/score.log")"
touch "$R/SCORED"
