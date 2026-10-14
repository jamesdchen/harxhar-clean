#!/bin/bash
# 2026-10-13 puffer_forecast: trainer pool (stream-ordered loads, polling refill) gate + timing, and the
# packing experiment (--pack K), on one CARC P100 (the GPU class of Phases 2 / 2b).
# OLD  = native/forecast_train.cu (the Phase 2b source)
# NEW  = native_pool3/forecast_train.cu (--pool; --pack K)
# 1. pool gate: block 0 of Phase 2 and Phase 2b (tune + refit), OLD waves vs NEW --pool; compared byte for byte
# 2. packing: Phase 2b block-0 refits, NEW --pack 4 / 8 / 16 vs NEW --pool (wall; forecasts' QLIKE)
# 3. packing exactness: 16 refits of one (design, seed) group, separate (--pool) vs packed (K 4 / 8 / 16),
#    after 1 epoch and at the end (save_params=1: per-model parameters in the single-model layout)
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_pack_gate_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=pf_packgate
#SBATCH --output=logs/pf_packgate_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
RUN=$ROOT/runs/packgate_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -1; git -C "$PUFFER" log -1 --format='%H %ci'; } > env.txt 2>&1
cat env.txt
md5sum "$ROOT/native/forecast_train.cu" "$ROOT/native_pool3/forecast_train.cu" > code_md5.txt
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
prebuilt "native_pool3" "$RUN/ft_new"
timed() {
  local tag=$1; shift
  local a; a=$(date +%s.%N)
  "$@" > "$tag.log" 2>&1
  echo "$tag $(echo "$(date +%s.%N) - $a" | bc)" | tee -a walls.txt
}
P2R=$ROOT/runs/13076458/block0/jobs_refit.txt
PBR=$ROOT/runs/phase2b_13080112/block0/jobs_refit.txt
# ---- 1. pool gate + timing
for ph in phase2 phase2b; do
  if [ "$ph" = phase2 ]; then RR=$P2R; else RR=$PBR; fi
  T=$ROOT/inputs/$ph/tune_block0.txt
  timed "${ph}_old_tune" ./ft_old run --jobs "$T" --out "${ph}_old_tune" --concurrent 48 --chunk 5
  timed "${ph}_new_tune" ./ft_new run --jobs "$T" --out "${ph}_new_tune" --concurrent 48 --chunk 5 --pool
  timed "${ph}_old_refit" ./ft_old run --jobs "$RR" --out "${ph}_old_refit" --concurrent 48 --chunk 5
  timed "${ph}_new_refit" ./ft_new run --jobs "$RR" --out "${ph}_new_refit" --concurrent 48 --chunk 5 --pool
done
# ---- 2. packing on the Phase 2b block-0 refits
for K in 4 8 16; do
  timed "phase2b_pack${K}_refit" ./ft_new run --jobs "$PBR" --out "phase2b_pack${K}_refit" --concurrent 48 --chunk 5 --pack $K --pack-slots 8
done
# ---- 3. packing exactness on 16-refit groups (all_features and baseline, seed 20261012), epochs 1 and full
for d in all_features baseline; do
  { grep "id=refit_${d}_r" "$PBR" | grep " seed=20261012 " || true; } | awk 'NR <= 16 {print $0 " save_params=1"}' > "sub_${d}.txt"
  sed 's/ epochs=[0-9]*/ epochs=1/' "sub_${d}.txt" > "sub_${d}_e1.txt"
  for v in "" _e1; do
    timed "sep_${d}${v}" ./ft_new run --jobs "sub_${d}${v}.txt" --out "sep_${d}${v}" --concurrent 16 --chunk 5 --pool
    for K in 4 8 16; do
      timed "pack${K}_${d}${v}" ./ft_new run --jobs "sub_${d}${v}.txt" --out "pack${K}_${d}${v}" --concurrent 16 --chunk 5 --pack $K --pack-slots 4
    done
  done
done
cat walls.txt
date; echo PACKGATE_DONE
