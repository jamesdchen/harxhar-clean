#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One line of $LINES (line $SLURM_ARRAY_TASK_ID, or line 1 outside an array): "<python> <script> <arguments>",
# python = gb | skx | sweep (env.sh), script relative to the code tree $CODE.  A line past the end of the file is a
# no-op (arrays sized for an upper bound).  sweep runs with OPENBLAS_CORETYPE=Prescott (the linear score job's setting).
set -euo pipefail
: "${LINES:?}" "${CODE:?}"
source "$CODE/results/trees_train3way_2014start_w3000/code/env.sh"
K=${SLURM_ARRAY_TASK_ID:-1}
LINE=$(sed -n "${K}p" "$LINES")
[ -n "$LINE" ] || { echo "line $K of $LINES: none"; exit 0; }
set -- $LINE
P=$1
shift
PY=$(pyof "$P")
if [ "$P" = sweep ]; then export OPENBLAS_CORETYPE=Prescott; fi
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
hostlog "line $(basename "$LINES"):$K $*"
echo "[$(date +%T)] $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')] $NT threads: $PY $*"
"$PY" "$@"
echo "[$(date +%T)] done"
