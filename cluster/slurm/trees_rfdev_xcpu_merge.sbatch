#!/bin/bash
# rfdev cross-CPU test (2026-10-07): PART=compare -- the chunk-level byte comparison of RESULTS_ROOT with the stored epyc-7513
# rfdev chunks (experiments/trees_rfdev_xcpu_compare.py); PART=all -- that comparison over every chunk, then the campaign's
# unedited reducer (experiments/trees_cadence_reduce_chunks.py) for t10 and t1, so the merged arms can be pulled and scored.
#SBATCH --job-name=xc_merge
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments:$PWD/pylib_trees"
R="${RESULTS_ROOT:?RESULTS_ROOT}"
P="${PART:?PART must be compare or all}"
python -u experiments/trees_rfdev_xcpu_compare.py --root "$R" | tee "$R/compare_$P.log"
if [ "$P" = all ]; then
  for RUNG in t10 t1; do
    mkdir -p "$R/$RUNG"
    python -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$R/$RUNG" --tasks cluster/trees_rfdev_tasks_untuned.txt \
      > "$R/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$R/$RUNG/reduce.log")"
    echo "$RUNG $(grep -c '^merged (' "$R/$RUNG/reduce.log" || true) merged; $(tail -1 "$R/$RUNG/reduce.log")"
  done
  touch "$R/MERGE_DONE"
fi
