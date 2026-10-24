#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --nodes=1
#SBATCH --ntasks=1
# RULE.txt addendum E: a pool walk task.  It claims the lowest unclaimed line k in $LO..$HI of $TASKFILE with an atomic
# mkdir $CLAIMS/<k> (one task a line, whichever CPU model gets there first), records who claimed it, and runs that line
# (experiments/trees_train3way.py walk) to completion on this node; a task that finds every line claimed exits 0.  A
# failed or timed-out line keeps its claim with a 'failed' / 'term' marker (its checkpointed piece resumes only on the
# same CPU family); such lines are released by hand after a look.  Env: TASKFILE, CODE, LO, HI, CLAIMS, [DESIGNS]
# (2026-10-24 addendum E note: DESIGNS = a design root off /scratch1, for nodes whose /scratch1 mount fails).
set -euo pipefail
: "${TASKFILE:?}" "${CODE:?}" "${LO:?}" "${HI:?}" "${CLAIMS:?}"
source "$CODE/results/trees_train3way_2014start_w3000/code/env.sh"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}_${SLURM_ARRAY_TASK_ID:-}"
mkdir -p "$CLAIMS"
K=""
for ((k = LO; k <= HI; k++)); do
  if mkdir "$CLAIMS/$k" 2> /dev/null; then
    K=$k
    break
  fi
done
if [ -z "$K" ]; then
  echo "pool task $ID: every line of $LO-$HI claimed"
  exit 0
fi
echo "$(date +%F_%T) $ID $(hostname) [$CPU] $NT threads" > "$CLAIMS/$K/claim"
trap 'echo "$(date +%F_%T) $ID TERM" > "$CLAIMS/$K/term"' TERM
LINE=$(sed -n "${K}p" "$TASKFILE")
read -r ARM EST SET PART <<< "$LINE"
[ -n "${PART:-}" ] || { echo "malformed line $K: '$LINE'" > "$CLAIMS/$K/failed"; exit 1; }
PY=$(pyof "$EST")
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
hostlog "pool $(basename "$TASKFILE"):$K $LINE"
echo "[$(date +%T)] $ID $(hostname) [$CPU] $NT threads: line $K: $LINE"
if "$PY" experiments/trees_train3way.py walk --arm "$ARM" --est "$EST" --set "$SET" --part "$PART" --threads "$NT" \
  --design-root "${DESIGNS:-$BASE/designs}" --out $BASE/runs; then
  echo "$(date +%F_%T) $ID" > "$CLAIMS/$K/done"
  echo "[$(date +%T)] line $K done"
else
  rc=$?
  echo "$(date +%F_%T) $ID rc $rc" > "$CLAIMS/$K/failed"
  exit $rc
fi
