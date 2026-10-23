#!/bin/bash
# One shard of the FI-by-window replay (fiw_replay.py --shard k/n) on a CARC node, in the sweep's numerical environment
# (as fiw_job.sh / the sweep's stage.sh).  Array task t = (W index) x NS + k, W = 1,000 .. 3,500 days; small tasks fit
# the partition's fragmented idle cores.  Then fiw_job.sh-style merge: fiw_replay.py --wdays W --merge.
# usage: sbatch --array=0-$((6 * NS - 1)) fiw_shard.sh [what] [limit]      (NS = FIW_NSHARD, default 30)
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch1/jc_905/fi_window/logs/fiws_%a_%A.log
set -euo pipefail
B=/scratch1/jc_905/fi_window
PY=/scratch1/jc_905/wsweep/venv/bin/python
WS=(1000 1500 2000 2500 3000 3500)
NS=${FIW_NSHARD:-30}
T=${SLURM_ARRAY_TASK_ID}
W=${WS[$((T / NS))]}
K=$((T % NS))
WHAT=${1:-replay,batch}
LIMIT=${2:-0}
export OPENBLAS_CORETYPE=Prescott
export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
export FIW_BASE=$B
echo "W $W shard $K/$NS host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | job ${SLURM_JOB_ID:-} | $(date)"
cd $B
"$PY" -u $B/scripts/fiw_replay.py --wdays "$W" --workers "$SLURM_CPUS_PER_TASK" --what "$WHAT" --limit "$LIMIT" \
  --shard "$K/$NS"
