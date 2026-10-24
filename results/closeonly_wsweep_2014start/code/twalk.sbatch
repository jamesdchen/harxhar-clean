#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
# RULE.txt section 3: one tree walk piece, line $SLURM_ARRAY_TASK_ID of $TASKFILE ("<arm> <est> <set> <i>/<n>") ->
# $BASE/runs/trees/pieces/<arm>_<est>_<set>_<i>of<n>.parquet (+ .json; resumable from its checkpoint on the same CPU
# family).  A line past the end of the file is a no-op.
set -euo pipefail
: "${TASKFILE:?}" "${CODE:?}"
source "$CODE/results/closeonly_wsweep_2014start/code/env.sh"
K=${SLURM_ARRAY_TASK_ID:?}
LINE=$(sed -n "${K}p" "$TASKFILE")
if [ -z "$LINE" ]; then
  echo "line $K of $TASKFILE: none"
  exit 0
fi
read -r ARM EST SET PART <<< "$LINE"
if [ -z "${PART:-}" ]; then
  echo "malformed line $K: '$LINE'"
  exit 1
fi
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
hostlog "walk $(basename "$TASKFILE"):$K $LINE"
echo "[$(date +%T)] $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')] $NT threads: $LINE"
"$(pyof "$EST")" experiments/closewin_trees.py walk --arm "$ARM" --est "$EST" --set "$SET" --part "$PART" \
  --threads "$NT" --design-root "$DES" --out "$BASE/runs/trees"
echo "[$(date +%T)] done"
