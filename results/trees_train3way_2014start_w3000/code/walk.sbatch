#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One walk piece: line $SLURM_ARRAY_TASK_ID of $TASKFILE, "<arm> <est> <set> <i>/<n>" (experiments/trees_train3way.py
# tasks), -> $BASE/runs/pieces/<arm>_<est>_<set>_<i>of<n>.parquet (+ .json; resumable from its checkpoint).  A line past
# the end of the file is a no-op (arrays sized for an upper bound); $ONLY (optional, a regex) skips non-matching lines.
set -euo pipefail
: "${TASKFILE:?}" "${CODE:?}"
source "$CODE/results/trees_train3way_2014start_w3000/code/env.sh"
K=${SLURM_ARRAY_TASK_ID:?}
LINE=$(sed -n "${K}p" "$TASKFILE")
[ -n "$LINE" ] || { echo "line $K of $TASKFILE: none"; exit 0; }
if [ -n "${ONLY:-}" ] && ! echo "$LINE" | grep -qE "$ONLY"; then echo "line $K '$LINE': not in $ONLY"; exit 0; fi
read -r ARM EST SET PART <<< "$LINE"
[ -n "${PART:-}" ] || { echo "malformed line $K: '$LINE'"; exit 1; }
PY=$(pyof "$EST")
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
hostlog "walk $(basename $TASKFILE):$K $LINE"
echo "[$(date +%T)] $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')] $NT threads: $LINE"
"$PY" experiments/trees_train3way.py walk --arm "$ARM" --est "$EST" --set "$SET" --part "$PART" --threads "$NT" \
  --design-root $BASE/designs --out $BASE/runs
echo "[$(date +%T)] done"
