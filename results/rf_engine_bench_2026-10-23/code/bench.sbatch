#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/jc_905/trees_final/logs/%x.%A_%a.out
# One line of $TASKS: "<engine> <set> <window days> <one-thread 0|1>" -> bench_rf.py on this node (the array's
# --constraint picks AMD EPYC or Intel Xeon).  Engines A and C run in harxhar-r13 (untouched), B / B2048 in the separate
# venv /scratch1/jc_905/trees_final/venv_skx (harxhar-r13 + scikit-learn-intelex, --system-site-packages).
set -euo pipefail
BASE=/scratch1/jc_905/trees_final
: "${TASKS:?}" "${OUTD:?}"
read -r E S W ONE <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")"
case $E in B*) PY=$BASE/venv_skx/bin/python ;; *) PY=/scratch1/jc_905/envs/harxhar-r13/bin/python ;; esac
NT=$SLURM_CPUS_PER_TASK
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
cd $BASE/repo
export PYTHONPATH="$PWD:$PWD/experiments"
O=""
[ "$ONE" = 1 ] && O="--one-thread"
echo "bench $E $S w$W on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')] $NT threads, $PY $(date)"
"$PY" $BASE/repo/results/rf_engine_bench_2026-10-23/code/bench_rf.py --repo $BASE/repo --design $BASE/designs/w$W \
  --window-days "$W" --set "$S" --engine "$E" --threads "$NT" --out "$OUTD" $O
echo "bench done $(date)"
