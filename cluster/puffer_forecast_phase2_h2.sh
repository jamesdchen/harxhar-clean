#!/bin/bash
# Phase 2 walk-forward on one Hoffman2 L40S. Phase 1 already checked the loss,
# the gradients and the toy fit (run of record: CARC runs/13037177). This job
# re-tunes every 250 sessions and cold-refits the chosen config, fc_n=1.
#   cd $H2ROOT && qsub puffer_forecast_phase2_h2.sh
#$ -cwd
#$ -j y
#$ -o logs/
#$ -N pf_phase2
#$ -l gpu,cuda=1,L40S,h_rt=2:00:00,h_data=24G
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load gcc/11.3.0 cuda/12.3
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$PWD
PUFFER=$ROOT/PufferLib5
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
RUN=$ROOT/runs/$JOB_ID
mkdir -p "$RUN"
cd "$RUN"
ln -sfn "$ROOT/data" data
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -1; gcc --version | head -1
  (cd "$PUFFER" && git log -1 --format='%H %ci'); } > env.txt 2>&1
cat env.txt
[ "$(cd "$PUFFER" && git rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
(cd "$PUFFER" && git diff --quiet -- src) || { echo "PufferLib src/ modified"; exit 1; }
ARCHS="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
( cd "$ROOT/native" && nvcc -O2 --threads 4 -Wno-deprecated-gpu-targets -std=c++17 $ARCHS -DPRECISION_FLOAT \
    -I. -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$CUDA_HOME/include" \
    -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 --diag-suppress=550 \
    forecast_train.cu -L"$CUDA_HOME/lib64" -lcublas -lcurand -o "$RUN/forecast_train" ) > build.log 2>&1
BIN=$RUN/forecast_train
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  # shellcheck disable=SC1091
  source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
  conda activate hpc-pi
  PY=python
fi
for k in 0 1 2 3 4; do
  mkdir -p "$RUN/block${k}"
  echo "BLOCK $k tune $(date)"
  "$BIN" run --jobs "$ROOT/inputs/phase2/tune_block${k}.txt" --out "$RUN/block${k}/tune" --concurrent 48 --chunk 5
  "$PY" "$ROOT/phase2_select.py" --blocks "$ROOT/inputs/phase2/blocks.json" --k "$k" \
    --tune-out "$RUN/block${k}/tune" --data-root "$ROOT/data" \
    --out "$RUN/block${k}/jobs_refit.txt" --selection "$RUN/block${k}/selection.json"
  echo "BLOCK $k refit $(date)"
  "$BIN" run --jobs "$RUN/block${k}/jobs_refit.txt" --out "$RUN/block${k}/refit" --concurrent 48 --chunk 5
done
date
echo PHASE2_DONE
