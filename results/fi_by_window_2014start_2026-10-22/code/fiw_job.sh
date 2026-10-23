#!/bin/bash
# One window of the FI-by-window replay (fiw_replay.py) on a CARC node, in the sweep's numerical environment (stage.sh:
# the uv venv, OpenBLAS kernel Prescott, numpy's AVX512 targets off where the node has them, numba for the host CPU).
# usage: sbatch --array=<W list> fiw_job.sh [what] [limit]    (SLURM_ARRAY_TASK_ID = W in days)
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch1/jc_905/fi_window/logs/fiw_%a_%A.log
set -euo pipefail
B=/scratch1/jc_905/fi_window
PY=/scratch1/jc_905/wsweep/venv/bin/python
W=${SLURM_ARRAY_TASK_ID}
WHAT=${1:-replay,batch}
LIMIT=${2:-0}
export OPENBLAS_CORETYPE=Prescott
export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
export FIW_BASE=$B
echo "W $W host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | job ${SLURM_JOB_ID:-} | $(date)"
cd $B
"$PY" -u $B/scripts/fiw_replay.py --wdays "$W" --workers $((SLURM_CPUS_PER_TASK - 1)) --what "$WHAT" --limit "$LIMIT"
