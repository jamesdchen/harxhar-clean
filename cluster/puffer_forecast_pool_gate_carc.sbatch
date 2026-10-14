#!/bin/bash
# 2026-10-13 puffer_forecast: trainer-pool gate and timing on one CARC P100 (the GPU class of Phases 2 / 2b).
# OLD = the Phase 2b source (native/forecast_train.cu, md5 b9c595...), NEW = native_pool/forecast_train.cu
# (--pool: per-slot allocation + epoch-graph capture once per shape, per-job reload, slot refill on stop).
# Block 0 of Phase 2 (no mask) and of Phase 2b (mask + clip + wide lr grid): tune (the block's tune jobs)
# and refit (the block's jobs_refit.txt from the Phase 2 / 2b runs), OLD (48-model waves) vs NEW (--pool,
# 48 slots) back to back; the forecasts / step logs are compared byte for byte afterwards (locally).
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_pool_gate_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=pf_poolgate
#SBATCH --output=logs/pf_poolgate_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
RUN=$ROOT/runs/poolgate_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -1; git -C "$PUFFER" log -1 --format='%H %ci'; } > env.txt 2>&1
cat env.txt
md5sum "$ROOT/native/forecast_train.cu" "$ROOT/native_pool/forecast_train.cu" > code_md5.txt
# the trainer, built ONCE per submission by puffer_forecast_build_once_carc.sbatch (gate 4, 2026-10-13), not in
# this job:  B=$(sbatch --parsable --export=NONE,SRC=<dir> puffer_forecast_build_once_carc.sbatch), then submit
# this script with --dependency=afterok:$B
prebuilt() {  # prebuilt SRC_DIR DEST: $ROOT/bin/SRC_DIR/forecast_train, checked against the current source
  local b=$ROOT/bin/$1
  [ -x "$b/forecast_train" ] || { echo "no prebuilt $b/forecast_train (puffer_forecast_build_once_carc.sbatch, SRC=$1)"; exit 1; }
  grep -q "$(md5sum < "$ROOT/$1/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" \
    || { echo "prebuilt $b is stale: $ROOT/$1/forecast_train.cu changed since it was built"; exit 1; }
  cp "$b/forecast_train" "$2"
  { echo "prebuilt $1:"; cat "$b/provenance.txt"; } >> env.txt
}
prebuilt "native" "$RUN/ft_old"
prebuilt "native_pool" "$RUN/ft_new"
timed() {  # $1 tag, rest: command
  local tag=$1; shift
  local a; a=$(date +%s.%N)
  "$@" > "$tag.log" 2>&1
  echo "$tag $(echo "$(date +%s.%N) - $a" | bc)" | tee -a walls.txt
}
for ph in phase2 phase2b; do
  if [ "$ph" = phase2 ]; then RR=$ROOT/runs/13076458/block0/jobs_refit.txt; else RR=$ROOT/runs/phase2b_13080112/block0/jobs_refit.txt; fi
  T=$ROOT/inputs/$ph/tune_block0.txt
  timed "${ph}_old_tune" ./ft_old run --jobs "$T" --out "${ph}_old_tune" --concurrent 48 --chunk 5
  timed "${ph}_new_tune" ./ft_new run --jobs "$T" --out "${ph}_new_tune" --concurrent 48 --chunk 5 --pool
  timed "${ph}_old_refit" ./ft_old run --jobs "$RR" --out "${ph}_old_refit" --concurrent 48 --chunk 5
  timed "${ph}_new_refit" ./ft_new run --jobs "$RR" --out "${ph}_new_refit" --concurrent 48 --chunk 5 --pool
done
# the new binary's wave path (device-side loss parameters) on the Phase 2 tune block: identical to OLD too
timed phase2_newwave_tune ./ft_new run --jobs "$ROOT/inputs/phase2/tune_block0.txt" --out phase2_newwave_tune --concurrent 48 --chunk 5
cat walls.txt
date; echo POOLGATE_DONE
