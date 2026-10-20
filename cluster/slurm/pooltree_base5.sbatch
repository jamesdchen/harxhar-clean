#!/bin/bash
# Pooled trees at base 5 (2026-10-19, user: "we need base 5 across the entire paper"): experiments/pooled_trees_base5.py.
#   MODE design  build the base-5 prep cache once (B5_CACHE) and write results/pooled_trees_base5_2026-10-19_<S>/
#                design.json + rows.csv for S = 48 and 13 (the walk tasks read the cache; run this first, alone)
#   MODE walk    line $SLURM_ARRAY_TASK_ID of TASKFILE ("<sample> <est> <set> <i>/<n>"): the 16:00 forecast rows
#                i, i+n, i+2n, ... of that arm -> results/pooled_trees_base5_2026-10-19_<sample>/pieces/
#                <est>_<set>_<i>of<n>.parquet (resumes: rows already in the piece are skipped)
# Submit (CARC, from the repo root, no conda env active):
#   sbatch --export=ALL,MODE=design --cpus-per-task=1 --mem=24G --time=4:00:00 cluster/slurm/pooltree_base5.sbatch
#   sbatch --dependency=afterok:<design job> --array=1-$(wc -l < cluster/pooltree_base5_tasks_rf_48.txt) \
#          --export=ALL,MODE=walk,TASKFILE=cluster/pooltree_base5_tasks_rf_48.txt cluster/slurm/pooltree_base5.sbatch
#   then, when every piece exists: python experiments/pooled_trees_base5.py merge --sample 48 (and 13)
#SBATCH --job-name=ptr_b5
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%A_%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export PYTHONPATH="$PWD:$PWD/experiments" TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK B5_CACHE=${B5_CACHE:-$PWD/results/pooled_trees_base5_cache}
MODE="${MODE:?MODE must be design or walk}"
echo "ptr_b5 $MODE on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)] $(date)"
if [ "$MODE" = design ]; then
  python experiments/pooled_trees_base5.py design --sample 48
  python experiments/pooled_trees_base5.py design --sample 13
  exit 0
fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TASKFILE:?TASKFILE}")
read -r S EST SET PART <<< "$LINE"
python experiments/pooled_trees_base5.py walk --sample "$S" --est "$EST" --set "$SET" --part "$PART" \
  --threads "$SLURM_CPUS_PER_TASK"
