#!/bin/bash
# Pooled tree walks (2026-10-07): the reproduction gate on CARC -- experiments/pooled_trees_gate.py runs
# experiments/pooled_trees_walk.py with the bank's own arms' configurations (lgbm_opt_00 = tree_expert_00, xgb_opt_00 =
# tree_expert_10) on all_features over the first rows of bank chunks (cluster/pooltree_gate_lines.txt) and requires the
# stored bank chunks (ref/, copied read only from harxhar-clean/results/unification_carc) byte for byte.  8 CPUs (the
# bank's thread count); lines run one after another.  Pinned by the submit script (--constraint).  Output: $OUT.
#SBATCH --job-name=ptr_gate
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments" UNIFY_CACHE_DIR="${UNIFY_CACHE_DIR:-cache}"
python -c "import lightgbm, xgboost, sklearn, numpy; print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)"
python -u experiments/pooled_trees_gate.py --lines "${LINES:-cluster/pooltree_gate_lines.txt}" --out "${OUT:?OUT}"
