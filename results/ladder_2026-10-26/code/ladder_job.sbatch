#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=ladder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --output=/scratch1/jc_905/ladder/logs/%x.%A_%a.out
# Feature-count ladder (results/ladder_2026-10-26/RULE.txt) on CARC: line $SLURM_ARRAY_TASK_ID of $TASKFILE =
#   "trees rank <est> <procs>" | "trees run <est> <i/n>" | "linear rank <workers>" | "linear run <model> <workers>"
# Trees: the tree SHAP study's python (/home1 venv_skx, read only) + its shap pylib, fit threads 8 and the walk's thread
# environment.  Linear: fi_by_window's fiw_job.sh environment (the sweep's venv, OpenBLAS Prescott, AVX512 targets off).
set -euo pipefail
B=/scratch1/jc_905/ladder
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export LADDER_WORK=$B/work
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "ladder task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$CPU] $(date)"
case "$CPU" in *"EPYC 7513"*|*"EPYC 7542"*) ;; *) echo "not an EPYC 7513/7542 node: $CPU"; exit 4 ;; esac
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKFILE")
[ -n "$LINE" ] || { echo "no line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
echo "line: $LINE"
read -r KIND MODE A1 A2 <<< "$LINE"
cd $B/code
if [ "$KIND" = "trees" ]; then
  PY=/home1/jc_905/trees_final/venv_skx/bin/python
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
  export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
  export PYTHONPATH=/home1/jc_905/tree_shap/pylib
  (cd /home1/jc_905/trees_final/designs/w3000/_work/design_all_features && md5sum -c --quiet MD5SUMS) && echo "design md5 ok"
  if [ "$MODE" = "rank" ]; then
    "$PY" ladder_trees.py rank --est "$A1" --procs "$A2"
  else
    "$PY" ladder_trees.py run --est "$A1" --part "$A2"
  fi
else
  PY=/scratch1/jc_905/wsweep/venv/bin/python
  export OPENBLAS_CORETYPE=Prescott
  export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
  export FIW_CODE=/scratch1/jc_905/fi_window/code
  echo "NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
  if [ "$MODE" = "rank" ]; then
    "$PY" -u ladder_linear.py rank --workers "$A1"
  else
    "$PY" -u ladder_linear.py run --model "$A1" --workers "$A2"
  fi
fi
echo "ladder task ${SLURM_ARRAY_TASK_ID} done $(date)"
