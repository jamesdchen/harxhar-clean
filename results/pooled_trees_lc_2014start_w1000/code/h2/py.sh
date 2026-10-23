#!/bin/bash
#$ -cwd
#$ -j y
# One python call in the walk tasks' environment (h2/walk.sh: the Debian container, the venv, threads = $THREADS or
# NSLOTS, numpy's AVX-512 dispatch off, OpenBLAS Haswell), from the code tree:  qsub ... py.sh <script> <args...>
# (the time probe: experiments/pooled_trees_rerun_13.py time ...).
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/tlc
export TLC_NT=${THREADS:-${NSLOTS:-1}}
echo "tlc py job ${JOB_ID:-} on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')], $TLC_NT threads, $(date): $*"
grep -qw avx2 /proc/cpuinfo || { echo "no AVX2: refusing (exit 99)"; exit 99; }
[ -f $B/ENV_OK ] || { echo "no env"; exit 1; }
apptainer exec -B /u/scratch $B/deb.sif bash -c '
set -euo pipefail
B=/u/scratch/j/jamesdc1/tlc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$B/lib
export OMP_NUM_THREADS=$TLC_NT MKL_NUM_THREADS=$TLC_NT OPENBLAS_NUM_THREADS=$TLC_NT NUMBA_NUM_THREADS=$TLC_NT
export OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES="$($B/venv/bin/python -c "from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(\" \".join(k for k in d if k.startswith(\"AVX512\") and f.get(k)))")"
cd $B/code
export PYTHONPATH=$B/code:$B/code/experiments
$B/venv/bin/python "$@"
' bash "$@"
echo "tlc py done $(date)"
