#!/bin/bash
# 2026-10-23 MinGRU v2: Hoffman2 GPU feasibility test (results/pooled_mingru_v2_2014start_w3000/hoffman2_gpu_test/).
# One SGE GPU job on one GPU:
#  (1) the G1 HAR file of CARC job 13353680 (inputs/g1_baseline.txt = the Stage-1 tune line L 20 H 64 lr 0.003 E 300
#      seed 20261012, its refit twin with fc_train, the shortcut alone), run TWICE (g1_r1, g1_r2) with CARC's command
#      line (forecast_train run --jobs --out tune --concurrent 3 --chunk 5 --pool), node-local data as on CARC;
#      binary: the CARC one (md5 44ab6fff) if it runs here (Hoffman2 libstdc++ of gcc 11.3 and libcublas / libcurand
#      of cuda 12.3); else a Hoffman2 build of the same source (native_g9, .cu af1964d0) with the CARC build flags,
#      recorded in env.txt / build.log (a DIFFERENT build);
#  (2) a throughput bench of the HAR (64, 20) Stage-1 tune line (60 epochs; 48 or 96 seeds) at --concurrent
#      12 / 24 / 48 (48 jobs) and 96 (96 jobs); GPU memory of the trainer process sampled every 2 s.
# Slurm copy for the Hoffman2 production cluster (hoffman2.oarc.ucla.edu, Rocky 9; same /u/scratch): identical body,
# toolchain cuda/12.8 + system gcc 11.5 (no cuda/12.3 module there).
#   cd /u/scratch/j/jamesdc1/pmf2_h2 && sbatch -C RTXPRO6000Bw h2_g1_bench_slurm.sh
#SBATCH -p campus
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -J v2h2g1s
#SBATCH -o logs/%x.o%j
type module >/dev/null 2>&1 || . /u/local/Modules/default/init/bash
module load cuda/12.8 >/dev/null 2>&1
JOB_ID=$SLURM_JOB_ID
QUEUE=slurm_${SLURM_JOB_PARTITION:-}_${SLURMD_NODENAME:-}
export CUDA_DEVICE_MAX_CONNECTIONS=32
set -uo pipefail
ROOT=$PWD
RUN=$ROOT/runs/$JOB_ID
LOCAL=${TMPDIR:-/tmp}/v2h2_$JOB_ID
PUFFER=/u/scratch/j/jamesdc1/puffer_forecast/PufferLib5
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
mkdir -p "$RUN" "$LOCAL"
SMI=0
trap 'kill $SMI 2>/dev/null; rm -rf "$LOCAL"' EXIT
cp -r "$ROOT/data" "$LOCAL/"
(cd "$ROOT/data" && find . -type f -exec md5sum {} + | sort -k2) > "$RUN/data.md5"
(cd "$LOCAL/data" && find . -type f -exec md5sum {} + | sort -k2) | cmp -s - "$RUN/data.md5" || { echo "stage data"; exit 1; }
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
cp "$ROOT/inputs/g1_baseline.txt" "$ROOT/inputs/bench48.txt" "$ROOT/inputs/bench96.txt" "$LOCAL/"
{ hostname; date; cat /etc/redhat-release
  echo "JOB_ID=$JOB_ID QUEUE=${QUEUE:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SGE_GPU=${SGE_GPU:-unset}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv
  module list 2>&1; echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"; which nvcc; nvcc --version | tail -1; gcc --version | head -1
  md5sum "$ROOT/bin/forecast_train" "$ROOT"/inputs/*.txt "$ROOT"/src/native_g9/*
  echo "--- ldd (CARC binary)"; ldd "$LOCAL/ft"
  echo "--- load"; uptime; nproc; } > "$RUN/env.txt" 2>&1
cat "$RUN/env.txt"

sampler() {  # $1 label: the trainer process' GPU memory (MiB) and the GPUs' memory / utilisation, every 2 s
  while :; do
    t=$(date +%s)
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,$1,/" >> "$RUN/gpu_apps.csv"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,$1,/" >> "$RUN/gpu_dev.csv"
    sleep 2
  done
}
runft() {  # label jobs-file concurrency binary
  local lab=$1 jf=$2 k=$3 bin=$4 t0 rc
  mkdir -p "$LOCAL/$lab"
  cd "$LOCAL/$lab" || return 9
  ln -sfn "$LOCAL/data" data
  sampler "$lab" &
  SMI=$!
  t0=$(date +%s.%N)
  "$bin" run --jobs "$LOCAL/$jf" --out tune --concurrent "$k" --chunk 5 --pool > tune.log 2>&1
  rc=$?
  kill $SMI 2>/dev/null
  wait $SMI 2>/dev/null
  echo "$lab rc=$rc wall_s=$(echo "$(date +%s.%N) - $t0" | bc) conc=$k jobs=$(wc -l < "$LOCAL/$jf") bin=$(basename "$bin") $(md5sum < "$bin" | cut -c1-8)" | tee -a "$RUN/runs.txt"
  rm -f data
  cd "$LOCAL" || return 9
  return $rc
}
nok() {  # number of jobs with status ok
  [ -f "$LOCAL/$1/tune/jobs_out.csv" ] && awk -F, 'NR > 1 && $0 ~ /,ok,/' "$LOCAL/$1/tune/jobs_out.csv" | wc -l || echo 0
}

BIN=$LOCAL/ft
runft g1_r1 g1_baseline.txt 3 "$BIN"
if [ "$(nok g1_r1)" != 3 ]; then
  echo "CARC binary did not complete G1 here: building native_g9 from source with the CARC build flags" | tee -a "$RUN/runs.txt"
  tail -20 "$LOCAL/g1_r1/tune.log" | tee "$RUN/carcbin_fail.log"
  mv "$LOCAL/g1_r1" "$LOCAL/g1_r1_carcbin_failed"
  [ "$(cd "$PUFFER" && git rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
  (cd "$PUFFER" && git diff --quiet -- src) || { echo "PufferLib src/ modified"; exit 1; }
  CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
  mkdir -p "$LOCAL/build" "$LOCAL/nvtmp"
  cp "$ROOT"/src/native_g9/* "$LOCAL/build/"
  ARCHS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
  ( cd "$LOCAL/build" && TMPDIR=$LOCAL/nvtmp nvcc -O2 --threads 4 -Wno-deprecated-gpu-targets -std=c++17 $ARCHS -DPRECISION_FLOAT \
    -I. -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$CUDA_HOME/include" \
    -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 \
    --diag-suppress=550 forecast_train.cu -L"$CUDA_HOME/lib64" -lcublas -lcurand -o "$LOCAL/ft_h2" ) > "$RUN/build.log" 2>&1 \
    || { tail -30 "$RUN/build.log"; exit 1; }
  { echo "H2 build: $(nvcc --version | tail -1); $(gcc --version | head -1); ARCHS $ARCHS"
    md5sum "$LOCAL/build/forecast_train.cu" "$LOCAL/build/puffer_substrate.cuh" "$LOCAL/ft_h2"; ldd "$LOCAL/ft_h2"; } >> "$RUN/build.log"
  cp "$LOCAL/ft_h2" "$RUN/forecast_train_h2"
  BIN=$LOCAL/ft_h2
  runft g1_r1 g1_baseline.txt 3 "$BIN"
fi
runft g1_r2 g1_baseline.txt 3 "$BIN"
{ for f in steps.csv epochs.csv preds.csv; do
    cmp -s "$LOCAL/g1_r1/tune/$f" "$LOCAL/g1_r2/tune/$f" && echo "$f identical" || echo "$f differs"
  done; } | tee "$RUN/r1_vs_r2.txt"
for c in 12 24 48; do runft "b48_c$c" bench48.txt "$c" "$BIN"; done
runft b96_c96 bench96.txt 96 "$BIN"
# copy back: the G1 runs whole (steps / preds gzipped), the bench runs without steps.csv / preds.csv
for r in g1_r1 g1_r2 g1_r1_carcbin_failed; do
  [ -d "$LOCAL/$r" ] || continue
  gzip -f "$LOCAL/$r/tune/steps.csv" "$LOCAL/$r/tune/preds.csv" 2>/dev/null
  cp -r "$LOCAL/$r" "$RUN/"
done
for r in b48_c12 b48_c24 b48_c48 b96_c96; do
  [ -d "$LOCAL/$r" ] || continue
  mkdir -p "$RUN/$r/tune"
  cp "$LOCAL/$r/tune.log" "$RUN/$r/"
  for f in jobs_out.csv pool_summary.csv epochs.csv; do cp "$LOCAL/$r/tune/$f" "$RUN/$r/tune/" 2>/dev/null; done
done
date >> "$RUN/runs.txt"
echo H2G1_DONE
