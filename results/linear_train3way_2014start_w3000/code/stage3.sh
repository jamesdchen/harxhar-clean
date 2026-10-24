#!/bin/bash
# One stage of the three-way training-sample comparison (RULE.txt) on a CARC node, in the template's environment
# (results/window_sweep_2014start_2026-10-22/code/stage.sh with WS_NPY=1, WS_NUMBA=0: the wsweep uv venv, Python 3.12.3,
# numpy 1.26.4, OpenBLAS kernel Prescott, numpy's AVX512 dispatch off where the node has it, 1 thread a process).
# usage: stage3.sh <arm> <stage> [options of experiments/linear_train3way.py]
set -euo pipefail
B=/scratch1/jc_905/train3way
PY=/scratch1/jc_905/wsweep/venv/bin/python
cd $B/code
ARM=$1; ST=$2; shift 2
OUT=results/linear_train3way_2014start_w3000/$ARM
export OPENBLAS_CORETYPE=Prescott
export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
echo "NPY_DISABLE_CPU_FEATURES=${NPY_DISABLE_CPU_FEATURES:-}"
mkdir -p "$OUT/_work"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then TAG=$ST; else TAG="$ST.$SLURM_ARRAY_TASK_ID"; fi
echo "$TAG host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | SLURM_JOB_ID ${SLURM_JOB_ID:-} | $(date)" >> "$OUT/_work/hosts.txt"
"$PY" -c "import threadpoolctl, numpy, numba; print('blas', [(d['internal_api'], d.get('architecture'), d.get('version')) for d in threadpoolctl.threadpool_info()], 'numpy', numpy.__version__, 'numba', numba.__version__); from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print('numpy dispatch on', [k for k in d if f.get(k)]); print('logspace', repr(numpy.logspace(-6, -2, 5).tolist()), repr(numpy.logspace(-2, 3, 6).tolist()))"
t0=$(date +%s)
echo "[$(date +%H:%M:%S)] $ARM stage $TAG start"
"$PY" experiments/linear_train3way.py "$ST" --arm "$ARM" --out "$OUT" "$@"
t1=$(date +%s)
echo "$TAG $((t1 - t0)) $(hostname)" >> "$OUT/_work/stage_times.txt"
echo "[$(date +%H:%M:%S)] $ARM stage $TAG done in $((t1 - t0)) s"
