#!/bin/bash
# One factor-test walk piece (results/beyond_trees_2026-10-27/RULE.txt, CARC addendum), a CARC job payload: line
# SLURM_ARRAY_TASK_ID of $TASKS ("<est> <arm> <i>/<n> <hours>") through experiments/beyond_factor_trees.py walk.
# No sbatch, copy or move in here.  env: TASKS, PY, CODE, FULL (design dir), MAPS (maps.npz), OUT (run dir)
set -eo pipefail
: "${TASKS:?}" "${PY:?}" "${CODE:?}" "${FULL:?}" "${MAPS:?}" "${OUT:?}"
cd "$CODE"
if [ -f "$(dirname "$PY")/activate" ]; then source "$(dirname "$PY")/activate"; fi
export PATH="$(dirname "$PY"):$PATH"
set -u
NT="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export PYTHONPATH="$PWD:$PWD/experiments"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" "$TASKS")
read -r EST ARM PART HOURS <<< "$LINE"
[ -n "${PART:-}" ] || { echo "malformed line ${SLURM_ARRAY_TASK_ID}: '$LINE'"; exit 1; }
echo "bf_task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')], $NT threads, $("$PY" -V): $LINE  $(date)"
"$PY" experiments/beyond_factor_trees.py walk --est "$EST" --arm "$ARM" --part "$PART" --full "$FULL" --maps "$MAPS" \
  --out "$OUT" --threads "$NT"
echo "bf_task done $(date)"
