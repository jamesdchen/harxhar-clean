#!/bin/bash
#$ -cwd
#$ -j y
# One line of $LINES on Hoffman2 (UGE array task $SGE_TASK_ID, or line 1): "<py> <script> <args...>", py = gb | skx
# (h2/code/env.sh), run from the code tree $B/code inside the Debian container, as CARC's code/line.sbatch: threads =
# NSLOTS (OMP / MKL / OpenBLAS / numba threads and SLURM_CPUS_PER_TASK, the default of trees_train3way.py --threads),
# numpy's AVX-512 dispatch off and OpenBLAS's Haswell kernel (the tree learning curve's Hoffman2 settings; CARC's EPYC
# nodes have no AVX-512).  WANT (regex, optional): the CPU model the line must run on; another model -> exit 99
# (UGE reschedules), so every piece of a gate tag comes from one CPU model.  The UGE job / task ids go into the piece
# meta's slurm_job / slurm_task fields.
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/t3t
NT=${NSLOTS:-1}
TASK=${SGE_TASK_ID:-1}; [ "$TASK" = undefined ] && TASK=1
LINE=$(sed -n "${TASK}p" "${LINES:?LINES}" | tr -d '\r')
[ -n "$LINE" ] || { echo "line $TASK of $LINES: none"; exit 0; }
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "[$(date +%T)] t3t line $TASK of $LINES, job ${JOB_ID:-} on $(hostname) [$CPU], $NT slots: $LINE"
if [ -n "${WANT:-}" ] && ! echo "$CPU" | grep -qE "$WANT"; then
  echo "CPU '$CPU' is not '$WANT': refusing (exit 99 = reschedule)"
  exit 99
fi
[ -f $B/ENV_OK ] || { echo "no env ($B/ENV_OK)"; exit 1; }
set -- $LINE
case $1 in
  gb) PY=$B/venv_gb/bin/python ;;
  skx) PY=$B/venv_skx/bin/python ;;
  *) echo "unknown python $1"; exit 1 ;;
esac
shift
echo "line $(basename "$LINES"):$TASK $* | $(hostname) | $CPU | job ${JOB_ID:-}.$TASK | $NT slots | $(date +%F_%T)" >> $B/logs/hosts.txt
export T3T_NT=$NT T3T_PY=$PY SLURM_ARRAY_JOB_ID=${JOB_ID:-} SLURM_ARRAY_TASK_ID=$TASK SLURM_CPUS_PER_TASK=$NT
a=$(date +%s)
apptainer exec -B /u/scratch $B/deb.sif bash -c '
set -euo pipefail
B=/u/scratch/j/jamesdc1/t3t
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$B/lib
export OMP_NUM_THREADS=$T3T_NT MKL_NUM_THREADS=$T3T_NT OPENBLAS_NUM_THREADS=$T3T_NT NUMBA_NUM_THREADS=$T3T_NT
export OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES="$($T3T_PY -c "from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(\" \".join(k for k in d if k.startswith(\"AVX512\") and f.get(k)))")"
echo "NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
cd $B/code
export PYTHONPATH=$B/code:$B/code/experiments
"$T3T_PY" "$@"
' bash "$@"
echo "[$(date +%T)] done in $(( $(date +%s) - a )) s"
