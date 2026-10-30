#!/bin/bash
# One python command of the ALL48 study (results/all48_2026-10-30/RULE.txt) on Hoffman2: cluster/ablate/py.sh with the
# staging directory and the container root from the environment (py.sh fixes them to the ablation's) -- inside the
# three-way comparison's Debian container, its venvs and libgomp read only: threads = $NT (OMP / MKL / OpenBLAS / numba),
# numpy's AVX-512 dispatch off, OpenBLAS's Prescott kernel, numba's cache under /tmp; cwd = $A48_B/code.
# usage: NT=<threads> py_h2.sh <gb|skx> <script> [args...]
# env: A48_B (staging holding code/; default /u/scratch/j/jamesdc1/all48), A48_T3 (container root; default
#      /u/scratch/j/jamesdc1/t3t: deb.sif, venv_gb, venv_skx, lib)
set -euo pipefail
B=${A48_B:-/u/scratch/j/jamesdc1/all48}
T3=${A48_T3:-/u/scratch/j/jamesdc1/t3t}
NT=${NT:-1}
case ${1:-} in
  gb) PY=$T3/venv_gb/bin/python ;;
  skx) PY=$T3/venv_skx/bin/python ;;
  *) echo "py_h2.sh: unknown env ${1:-}"; exit 2 ;;
esac
shift
if ! command -v apptainer > /dev/null 2>&1; then
  . /u/local/Modules/default/init/modules.sh
  module load apptainer/1.3.4 2> /dev/null || module load apptainer/1.2.2
fi
export ABL_NT=$NT ABL_PY=$PY ABL_B=$B ABL_T3=$T3
apptainer exec -B /u/scratch $T3/deb.sif bash -c '
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$ABL_T3/lib
export OMP_NUM_THREADS=$ABL_NT MKL_NUM_THREADS=$ABL_NT OPENBLAS_NUM_THREADS=$ABL_NT NUMBA_NUM_THREADS=$ABL_NT
export SLURM_CPUS_PER_TASK=$ABL_NT OPENBLAS_CORETYPE=Prescott NUMBA_CACHE_DIR=/tmp/numba_a48_$(id -u)
export NPY_DISABLE_CPU_FEATURES="$($ABL_PY -c "from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(\" \".join(k for k in d if k.startswith(\"AVX512\") and f.get(k)))")"
echo "py_h2.sh $(hostname) [$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2)] NT=$ABL_NT NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
cd $ABL_B/code
export PYTHONPATH=$ABL_B/code:$ABL_B/code/experiments
exec "$ABL_PY" "$@"
' bash "$@"
