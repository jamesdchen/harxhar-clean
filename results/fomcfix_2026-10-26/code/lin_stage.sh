#!/bin/bash
# One stage of the linear runs of results/fomcfix_2026-10-26 (RULE.txt), a CARC job payload: the final run's stage.sh
# (results/window_sweep_2014start_2026-10-22/code/stage.sh: its venv, OpenBLAS kernel and numpy dispatch) with this
# run's code tree and a cell's flags.  No sbatch, copy or move in here.
# usage: lin_stage.sh <out dir, relative to the code tree> <stage> [jobs]
#   env CELL      comma-separated NAME=0|1 flags of the cell (FOMC_SCHEDULE_2020_FIX, FOMC_PROXIMITY_NO_WINSOR,
#                 DROP_HOLIDAY_SESSIONS); "-" = none
#   env FIRST_DAY first evaluation day (default 2014-01-02); FIRST_FOLD (default 3)
#   stage design          the two design caches -> <out>/_work/design_{baseline,all_features}
#         pretune_select  pretune + select (frozen_penalties.json) with --first-fold FIRST_FOLD
#         walk1           line SLURM_ARRAY_TASK_ID of $TASKS through code/walk_task.py
#         py              python code/<script> <args...> (env PYARGS), e.g. the gates and the collection
set -euo pipefail
B=/scratch1/jc_905/fomcfix
PY=/scratch1/jc_905/wsweep/venv/bin/python
cd $B/code
OUT=$1; ST=$2; J=${3:-1}
F="--window-days 3000 --first-day ${FIRST_DAY:-2014-01-02} --ex-flag LATE_START_WARMUP=1"
if [ "${CELL:--}" != "-" ]; then for kv in ${CELL//,/ }; do F="$F --ex-flag $kv"; done; fi
FFO="--first-fold ${FIRST_FOLD:-3}"
# the final linear runs' numerical environment: run B's OpenBLAS kernel (threadpoolctl: Prescott), numpy's run-time
# dispatch without AVX-512 where the node has it (none on EPYC), numba for the node
export OPENBLAS_CORETYPE=Prescott
export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
export PYTHONDONTWRITEBYTECODE=1
echo "NPY_DISABLE_CPU_FEATURES=${NPY_DISABLE_CPU_FEATURES:-} CELL=${CELL:--} F=$F"
mkdir -p "$OUT/_work"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then TAG=$ST; else TAG="$ST.$SLURM_ARRAY_TASK_ID"; fi
echo "$TAG host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | SLURM_JOB_ID ${SLURM_JOB_ID:-} | $(date)" >> "$OUT/_work/hosts.txt"
"$PY" -c "import threadpoolctl, numpy, numba; print('blas', [(d['internal_api'], d.get('architecture'), d.get('version')) for d in threadpoolctl.threadpool_info()], 'numpy', numpy.__version__, 'numba', numba.__version__); print('logspace', repr(numpy.logspace(-6, -2, 5).tolist()), repr(numpy.logspace(-2, 3, 6).tolist()))"
t0=$(date +%s)
echo "[$(date +%H:%M:%S)] stage $TAG start"
case "$ST" in
  design) "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs "$J" $F --out "$OUT" ;;
  pretune_select)
    "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs "$J" $F $FFO --out "$OUT"
    "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py select $F $FFO --out "$OUT" ;;
  walk1) "$PY" results/fomcfix_2026-10-26/code/walk_task.py "${TASKS:?TASKS}" "${SLURM_ARRAY_TASK_ID:?}" ;;
  py) "$PY" ${PYARGS:?PYARGS} ;;
  *) echo "unknown stage $ST"; exit 2 ;;
esac
t1=$(date +%s)
echo "$TAG $((t1 - t0)) $(hostname)" >> "$OUT/_work/stage_times.txt"
echo "[$(date +%H:%M:%S)] stage $TAG done in $((t1 - t0)) s"
