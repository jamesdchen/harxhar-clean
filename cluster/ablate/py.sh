#!/bin/bash
# One python command of the ablation run (results/ablations_2014start_w3000/RULE.txt) inside the three-way
# comparison's Debian container, its venvs and libgomp used read only (/u/scratch/j/jamesdc1/t3t; RULE.txt section 6),
# as its line.sh: threads = $NT (OMP / MKL / OpenBLAS / numba), numpy's AVX-512 dispatch off, OpenBLAS's Prescott
# kernel (the final linear run's settings, results/window_sweep_2014start_2026-10-22/code/stage.sh), numba's cache
# under /tmp.  usage: NT=<threads> py.sh <gb|skx> <script> [args...]   (run from anywhere; cwd = the code tree)
set -euo pipefail
B=/u/scratch/j/jamesdc1/abl
T3=/u/scratch/j/jamesdc1/t3t
NT=${NT:-1}
case $1 in
  gb) PY=$T3/venv_gb/bin/python ;;
  skx) PY=$T3/venv_skx/bin/python ;;
  *) echo "py.sh: unknown env $1"; exit 2 ;;
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
export SLURM_CPUS_PER_TASK=$ABL_NT OPENBLAS_CORETYPE=Prescott NUMBA_CACHE_DIR=/tmp/numba_abl_$(id -u)
export NPY_DISABLE_CPU_FEATURES="$($ABL_PY -c "from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(\" \".join(k for k in d if k.startswith(\"AVX512\") and f.get(k)))")"
echo "py.sh $(hostname) [$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2)] NT=$ABL_NT NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
cd $ABL_B/code
export PYTHONPATH=$ABL_B/code:$ABL_B/code/experiments
exec "$ABL_PY" "$@"
' bash "$@"
