#!/bin/bash
# One tree walk piece of results/fomcfix_2026-10-26 (RULE.txt sections 6 and 7(c)), a CARC job payload: line
# SLURM_ARRAY_TASK_ID of $TASKS through experiments/pooled_trees_rerun_13.py walk, in the environment of
# cluster/slurm/pooltree_rerun13.sbatch (threads = --cpus-per-task, PYTHONPATH = the code tree and its experiments/).
# No sbatch, copy or move in here.
#   line: <out dir> <est> <set> <refit> <first day> <last day> <i>/<n> [<config file>|-] [<seed>]
#   env TASKS, PY (python), CODE (code tree), DESIGN (root holding _work/design_<set>), WINDOW_DAYS (default 3000)
set -eo pipefail
: "${TASKS:?}" "${PY:?}" "${CODE:?}" "${DESIGN:?}"
cd "$CODE"
if [ -f "$(dirname "$PY")/activate" ]; then source "$(dirname "$PY")/activate"; fi
export PATH="$(dirname "$PY"):$PATH"
set -u
NT="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export PYTHONPATH="$PWD:$PWD/experiments"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" "$TASKS")
read -r OUT EST SET REFIT FIRST LAST PART CFG SEED <<< "$LINE"
[ -n "${PART:-}" ] || { echo "malformed line ${SLURM_ARRAY_TASK_ID}: '$LINE'"; exit 1; }
X=()
if [ -n "${CFG:-}" ] && [ "$CFG" != "-" ]; then X+=(--config-file "$CFG"); fi
if [ -n "${SEED:-}" ]; then X+=(--seed "$SEED"); fi
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "tree_task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$CPU], $NT threads, $("$PY" -V): $LINE  $(date)"
"$PY" experiments/pooled_trees_rerun_13.py walk --design "$DESIGN" --out "$OUT" --window-days "${WINDOW_DAYS:-3000}" \
  --est "$EST" --set "$SET" --refit "$REFIT" --first-day "$FIRST" --last-day "$LAST" --part "$PART" --threads "$NT" \
  "${X[@]}"
echo "tree_task done $(date)"
