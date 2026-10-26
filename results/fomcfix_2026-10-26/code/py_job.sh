#!/bin/bash
# One python call of results/fomcfix_2026-10-26 (RULE.txt) as a CARC job payload, in the trees' environment
# (cluster/slurm/pooltree_rerun13.sbatch's variables; PYTHONPATH = the code tree and its experiments/).  No sbatch, copy
# or move in here.  usage: py_job.sh <script> <args...>; env PY (python), CODE (code tree)
set -eo pipefail
: "${PY:?}" "${CODE:?}"
cd "$CODE"
if [ -f "$(dirname "$PY")/activate" ]; then source "$(dirname "$PY")/activate"; fi
export PATH="$(dirname "$PY"):$PATH"
NT="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export PYTHONPATH="$PWD:$PWD/experiments"
echo "py_job on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')], $NT CPUs, $("$PY" -V), $(date): $*"
"$PY" "$@"
echo "py_job done $(date)"
