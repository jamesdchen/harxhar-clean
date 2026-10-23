#!/bin/bash
#$ -cwd
#$ -j y
# One walk task on Hoffman2 (UGE array task): line $SGE_TASK_ID of $TASKFILE, "<est> <set> <refit> <first> <last> <i>/<n>",
# through experiments/pooled_trees_rerun_13.py walk (the printed run's runner, check() accepting the 564-column final
# design) -> $OUT/pieces/<est>_<set>_<refit>_<first>_<last>_<i>of<n>.parquet + .json; resumable (checkpoints every 300 s,
# rows already in the piece skipped).  As cluster/slurm/pooltree_rerun13.sbatch on CARC, inside the Debian container:
# threads = $THREADS (default NSLOTS), OMP/MKL/OpenBLAS/numba threads = the same; numpy's AVX-512 dispatch off where the
# node has it (CARC's EPYC nodes have none) and OpenBLAS's Haswell kernel (CARC Zen); AVX2 nodes only.  The UGE job and
# task ids go into the piece meta's slurm_job / slurm_task fields.
# Env (qsub -v): TASKFILE, DESIGN (holds _work/design_{baseline,all_features}), OUT, WINDOW_DAYS, [THREADS].
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/tlc
NT=${THREADS:-${NSLOTS:-1}}
TASK=${SGE_TASK_ID:-1}; [ "$TASK" = undefined ] && TASK=1
LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}" | tr -d '\r')  # task files written on Windows: CRLF
read -r EST SET REFIT FIRST LAST PART <<< "$LINE"
[ -n "${PART:-}" ] || { echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1; }
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "tlc walk task $TASK job ${JOB_ID:-} on $(hostname) [$CPU], $NT threads (NSLOTS ${NSLOTS:-}), $(date): $LINE"
grep -qw avx2 /proc/cpuinfo || { echo "no AVX2 on $(hostname): refusing (exit 99 = reschedule)"; exit 99; }
[ -f $B/ENV_OK ] || { echo "no env ($B/ENV_OK)"; exit 1; }
export TLC_NT=$NT TLC_EST=$EST TLC_SET=$SET TLC_REFIT=$REFIT TLC_FIRST=$FIRST TLC_LAST=$LAST TLC_PART=$PART
export TLC_DESIGN=${DESIGN:?DESIGN} TLC_OUT=${OUT:?OUT} TLC_WD=${WINDOW_DAYS:?WINDOW_DAYS}
export SLURM_ARRAY_JOB_ID=${JOB_ID:-} SLURM_ARRAY_TASK_ID=$TASK
a=$(date +%s)
apptainer exec -B /u/scratch $B/deb.sif bash -c '
set -euo pipefail
B=/u/scratch/j/jamesdc1/tlc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$B/lib
export OMP_NUM_THREADS=$TLC_NT MKL_NUM_THREADS=$TLC_NT OPENBLAS_NUM_THREADS=$TLC_NT NUMBA_NUM_THREADS=$TLC_NT
export OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES="$($B/venv/bin/python -c "from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(\" \".join(k for k in d if k.startswith(\"AVX512\") and f.get(k)))")"
echo "NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
cd $B/code
export PYTHONPATH=$B/code:$B/code/experiments
$B/venv/bin/python experiments/pooled_trees_rerun_13.py walk --design "$TLC_DESIGN" --out "$TLC_OUT" \
  --window-days "$TLC_WD" --est "$TLC_EST" --set "$TLC_SET" --refit "$TLC_REFIT" --first-day "$TLC_FIRST" \
  --last-day "$TLC_LAST" --part "$TLC_PART" --threads "$TLC_NT"
'
echo "tlc walk task $TASK done in $(( $(date +%s) - a )) s on $(hostname -s) $(date)"
