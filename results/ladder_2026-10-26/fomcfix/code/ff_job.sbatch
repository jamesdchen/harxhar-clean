#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=lff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --output=/home1/jc_905/ladder_fomcfix/logs/%x.%A_%a.out
# Ladder / grid on the FOMC-patched design (results/ladder_2026-10-26/RULE.txt ADDENDUM 3): line $SLURM_ARRAY_TASK_ID of
# $TASKFILE = "trees rank-trees <W> <est>" | "trees run-trees <W> <est> <i/n> <secondary 0|1>" | "linear rank-linear <W>"
# | "linear run-linear <W> <model> <secondary 0|1>" | "gate gate-design <W>".  Everything on /home1.
set -euo pipefail
B=/home1/jc_905/ladder_fomcfix
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export FF_WORK=$B/work LADDER_CODE=$B/code/ladder FF_CODE=/home1/jc_905/fomcfix_lin/code
export LADDER_SHAP_WORK=/home1/jc_905/tree_shap_fomcfix/work
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "lff task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$CPU] $(date)"
case "$CPU" in *"EPYC 7513"*|*"EPYC 7542"*) ;; *) echo "not an EPYC 7513/7542 node: $CPU"; exit 4 ;; esac
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKFILE")
[ -n "$LINE" ] || { echo "no line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
echo "line: $LINE"
read -r KIND MODE WD A1 A2 A3 <<< "$LINE"
if [ "${A3:-0}" = "1" ] || { [ "$MODE" = "run-linear" ] && [ "${A2:-0}" = "1" ]; }; then export FF_SECONDARY=$B/code/ff/secondary_importance_fomcfix.json; fi
cd $B/code/ff
if [ "$KIND" = "trees" ] || [ "$KIND" = "gate" ]; then
  PY=/home1/jc_905/trees_final/venv_skx/bin/python
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
  export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
  export PYTHONPATH=/home1/jc_905/tree_shap/pylib
  case "$MODE" in
    gate-design) "$PY" ff_runs.py gate-design --wdays "$WD" ;;
    rank-trees) "$PY" ff_runs.py rank-trees --wdays "$WD" --est "$A1" ;;
    run-trees) "$PY" ff_runs.py run-trees --wdays "$WD" --est "$A1" --part "$A2" --secondary "${A3:-0}" ;;
  esac
else
  PY=/home1/jc_905/fomcfix_lin/venv/bin/python
  export OPENBLAS_CORETYPE=Prescott
  export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
  if [ "$MODE" = "rank-linear" ]; then
    "$PY" -u ff_runs.py rank-linear --wdays "$WD"
  else
    "$PY" -u ff_runs.py run-linear --wdays "$WD" --model "$A1" --secondary "${A2:-0}" --workers "$SLURM_CPUS_PER_TASK"
  fi
fi
echo "lff task ${SLURM_ARRAY_TASK_ID} done $(date)"
