#!/bin/bash
# 2026-10-14 puffer_forecast: can MPS take small-model throughput past the 32-hardware-queue ceiling?
# The 2,500 Phase 2c block-0 refits split round-robin into N = 1..4 trainer processes on ONE GPU, each --pool with
# --concurrent 48/N (and 48 each), CUDA_DEVICE_MAX_CONNECTIONS=32; with a job-local MPS daemon and without (plain
# time-sliced processes).  Total wall per configuration; outputs compared with the single-process run locally.
#   cd $HOME/puffer_forecast && sbatch --export=NONE,GPU_TYPE=p100,SRC=native_g5b --constraint=p100 puffer_forecast_mps_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=pf_mps
#SBATCH --output=logs/pf_mps_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
SRC=${SRC:-native_g5b}
GPU_TYPE=${GPU_TYPE:?GPU_TYPE required}
RUN=$ROOT/runs/mps_${GPU_TYPE}_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv,noheader
  echo "GPU_TYPE=$GPU_TYPE constraint=${SLURM_JOB_CONSTRAINTS:-unset}"; } > env.txt 2>&1
cat env.txt
# the trainer, built ONCE per submission by puffer_forecast_build_once_carc.sbatch (gate 4, 2026-10-13), not in
# this job
prebuilt() {  # prebuilt SRC_DIR DEST: $ROOT/bin/SRC_DIR/forecast_train, checked against the current source
  local b=$ROOT/bin/$1
  [ -x "$b/forecast_train" ] || { echo "no prebuilt $b/forecast_train (puffer_forecast_build_once_carc.sbatch, SRC=$1)"; exit 1; }
  grep -q "$(md5sum < "$ROOT/$1/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" \
    || { echo "prebuilt $b is stale: $ROOT/$1/forecast_train.cu changed since it was built"; exit 1; }
  cp "$b/forecast_train" "$2"
  { echo "prebuilt $1:"; cat "$b/provenance.txt"; } >> env.txt
}
prebuilt "$SRC" ft
cp "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" refit_all.txt
for N in 2 3 4; do
  for i in $(seq 0 $((N - 1))); do awk -v n=$N -v i=$i '(NR - 1) % n == i' refit_all.txt > part_N${N}_$i.txt; done
done
echo "name,mps,N,K,wall_s,rcs" > timing.csv
runset() {  # runset NAME MPS N K: N processes at once, total wall
  local name=$1 mps=$2 N=$3 K=$4 a rcs=""
  mkdir -p "$name"
  a=$(date +%s.%N)
  if [ "$N" = 1 ]; then
    ./ft run --jobs refit_all.txt --out "$name/p0" --concurrent "$K" --chunk 5 --pool > "$name/p0.log" 2>&1 &
  else
    for i in $(seq 0 $((N - 1))); do
      ./ft run --jobs part_N${N}_$i.txt --out "$name/p$i" --concurrent "$K" --chunk 5 --pool > "$name/p$i.log" 2>&1 &
    done
  fi
  for p in $(jobs -p); do wait "$p"; rcs="$rcs$?"; done
  echo "$name,$mps,$N,$K,$(awk -v a="$a" -v b="$(date +%s.%N)" 'BEGIN{printf "%.3f", b - a}'),$rcs" >> timing.csv
}
# ---- without MPS (time-sliced processes)
runset plain_N1_K48 0 1 48
runset plain_N2_K24 0 2 24
runset plain_N4_K12 0 4 12
runset plain_N2_K48 0 2 48
# ---- with a job-local MPS daemon
export CUDA_MPS_PIPE_DIRECTORY=$RUN/mps_pipe CUDA_MPS_LOG_DIRECTORY=$RUN/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if nvidia-cuda-mps-control -d > mps_start.log 2>&1; then
  sleep 2
  echo "MPS started: $(echo get_server_list | nvidia-cuda-mps-control 2>&1 | head -2)" | tee -a env.txt
  runset mps_N1_K48 1 1 48
  runset mps_N2_K24 1 2 24
  runset mps_N3_K16 1 3 16
  runset mps_N4_K12 1 4 12
  runset mps_N2_K48 1 2 48
  runset mps_N3_K48 1 3 48
  runset mps_N4_K48 1 4 48
  echo "MPS server list after: $(echo get_server_list | nvidia-cuda-mps-control 2>&1 | head -2)" >> env.txt
  echo quit | nvidia-cuda-mps-control
  sleep 2
  cp "$CUDA_MPS_LOG_DIRECTORY"/*.log . 2>/dev/null
else
  echo "MPS could not start: $(cat mps_start.log)" | tee -a env.txt
fi
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
for d in plain_* mps_*; do
  [ -d "$d" ] && find "$d" -name steps.csv -o -name epochs.csv | xargs -r gzip -f
done
rm -f ft
date; echo MPS_DONE
