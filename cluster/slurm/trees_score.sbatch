#!/bin/bash
# Score the per-bar TREE arms (results/linear_subsection_trees) beside the
# per-bar linear arms (results/linear_subsection).  Held on both tree fleets
# with afterany, so a failed arm does not block the rest: the arms that did not
# finish are listed here (NOT DONE + the last line of their run.log).
#SBATCH --job-name=tr_score
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
export PYTHONPATH="$PWD:$PWD/experiments:$PWD/notebooks:$PWD/pylib_trees"
R=results/linear_subsection_trees
mkdir -p "$R"
EXPECTED=$(cat cluster/trees_tasks_gb.txt cluster/trees_tasks_rf.txt \
  | awk 'NF {m = split($2, a, ","); s = split($4, b, ","); n += m * s} END {print n + 0}')
echo "$R: arms finished $(find "$R" -name DONE | wc -l) of $EXPECTED"
find "$R" -name run.log | while read -r f; do
  [ -f "$(dirname "$f")/DONE" ] || echo "NOT DONE: $f: $(tail -1 "$f" | cut -c1-200)"
done
python -u experiments/score_trees_subsection.py --root "$R" \
  --linear-root results/linear_subsection --linear-root results/linear_subsection_lassofix \
  > "$R/score.log" 2>&1 \
  || echo "SCORER FAILED for $R: $(tail -2 "$R/score.log")"
touch "$R/SCORED"
