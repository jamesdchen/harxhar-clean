#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=lgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --output=/scratch1/jc_905/ladder_grid/logs/%x.%A_%a.out
# Window x feature-count grid (results/ladder_2026-10-26/RULE.txt ADDENDUM 1) on CARC: line $SLURM_ARRAY_TASK_ID of
# $TASKFILE = "trees rank-trees <W> <est>" | "trees run-trees <W> <est> <i/n>" | "linear rank-linear <W>" |
# "linear run-linear <W> <model>".  Environments as the ladder's ladder_job.sbatch.
set -euo pipefail
B=/scratch1/jc_905/ladder_grid
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export GRID_WORK=$B/work LADDER_CODE=$B/code/ladder
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "lgrid task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$CPU] $(date)"
case "$CPU" in *"EPYC 7513"*|*"EPYC 7542"*) ;; *) echo "not an EPYC 7513/7542 node: $CPU"; exit 4 ;; esac
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKFILE")
[ -n "$LINE" ] || { echo "no line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
echo "line: $LINE"
read -r KIND MODE WD A1 A2 <<< "$LINE"
cd $B/code/grid
if [ "$KIND" = "trees" ]; then
  PY=/home1/jc_905/trees_final/venv_skx/bin/python
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
  export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
  export PYTHONPATH=/home1/jc_905/tree_shap/pylib
  if [ "$MODE" = "rank-trees" ]; then
    "$PY" grid_runs.py rank-trees --wdays "$WD" --est "$A1"
  else
    "$PY" grid_runs.py run-trees --wdays "$WD" --est "$A1" --part "$A2"
  fi
else
  PY=/scratch1/jc_905/wsweep/venv/bin/python
  export OPENBLAS_CORETYPE=Prescott
  export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
  export FIW_CODE=/scratch1/jc_905/fi_window/code
  if [ "$MODE" = "rank-linear" ]; then
    "$PY" -u grid_runs.py rank-linear --wdays "$WD" --workers 3
  else
    "$PY" -u grid_runs.py run-linear --wdays "$WD" --model "$A1" --workers "$SLURM_CPUS_PER_TASK"
  fi
fi
echo "lgrid task ${SLURM_ARRAY_TASK_ID} done $(date)"
