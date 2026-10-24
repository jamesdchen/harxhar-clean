#!/bin/bash
# 2026-10-24 MinGRU v2 Hoffman2 GPU test, bench 2 (Slurm, hoffman2.oarc.ucla.edu): throughput of the larger Stage-1 cells
# and of HAR (64, 20) beyond 24 concurrent models, by the CARC binary (md5 44ab6fff; PTX JIT on sm_120), tune mode as
# Stage 1 (HAR, L 20, lr 0.003), at the old run's concurrency of each cell:
#   b256_c12   HAR (256, 20), 24 seeds x 20 epochs, --concurrent 12   (P100: 0.250 s a model-epoch)
#   b1024_c7   HAR (1024, 20), 7 seeds x 10 epochs, --concurrent 7   (P100: 5.53 s a model-epoch)
#   b96_c48 / b96_c96   HAR (64, 20), 96 seeds x 60 epochs, --concurrent 48 / 96 (bench 1 ran out of host RAM at 16G)
#   cd /u/scratch/j/jamesdc1/pmf2_h2 && sbatch -C RTXPRO6000Bw h2_bench2_slurm.sh
#SBATCH -p campus
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -J v2h2b2
#SBATCH -o logs/%x.o%j
type module >/dev/null 2>&1 || . /u/local/Modules/default/init/bash
module load cuda/12.8 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
set -uo pipefail
ROOT=$PWD
JOB_ID=$SLURM_JOB_ID
RUN=$ROOT/runs/$JOB_ID
LOCAL=${TMPDIR:-/tmp}/v2h2_$JOB_ID
mkdir -p "$RUN" "$LOCAL"
SMI=0
trap 'kill $SMI 2>/dev/null; rm -rf "$LOCAL"' EXIT
cp -r "$ROOT/data" "$LOCAL/"
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
cp "$ROOT"/inputs/bench96.txt "$ROOT"/inputs/bench_H256_E20_K24.txt "$ROOT"/inputs/bench_H1024_E10_K7.txt "$LOCAL/"
{ hostname; date; cat /etc/redhat-release; echo "JOB_ID=$JOB_ID node=${SLURMD_NODENAME:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv
  md5sum "$ROOT/bin/forecast_train" "$ROOT"/inputs/bench96.txt "$ROOT"/inputs/bench_H*.txt; uptime; nproc; } > "$RUN/env.txt" 2>&1
cat "$RUN/env.txt"
sampler() {
  while :; do
    t=$(date +%s)
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,$1,/" >> "$RUN/gpu_apps.csv"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,$1,/" >> "$RUN/gpu_dev.csv"
    ps -o rss= -C ft 2>/dev/null | awk -v t="$t" -v l="$1" '{s += $1} END {print t "," l "," s}' >> "$RUN/host_rss_kib.csv"
    sleep 2
  done
}
runft() {  # label jobs-file concurrency
  local lab=$1 jf=$2 k=$3 t0 rc
  mkdir -p "$LOCAL/$lab"
  cd "$LOCAL/$lab" || return 9
  ln -sfn "$LOCAL/data" data
  sampler "$lab" &
  SMI=$!
  t0=$(date +%s)
  "$LOCAL/ft" run --jobs "$LOCAL/$jf" --out tune --concurrent "$k" --chunk 5 --pool > tune.log 2>&1
  rc=$?
  kill $SMI 2>/dev/null
  wait $SMI 2>/dev/null
  echo "$lab rc=$rc wall_s=$(( $(date +%s) - t0 )) conc=$k jobs=$(wc -l < "$LOCAL/$jf") bin=ft 44ab6fff" | tee -a "$RUN/runs.txt"
  rm -f data
  cd "$LOCAL" || return 9
  mkdir -p "$RUN/$lab/tune"
  cp "$LOCAL/$lab/tune.log" "$RUN/$lab/"
  for f in jobs_out.csv pool_summary.csv epochs.csv; do cp "$LOCAL/$lab/tune/$f" "$RUN/$lab/tune/" 2>/dev/null; done
  rm -rf "${LOCAL:?}/$lab"
  return $rc
}
runft b256_c12 bench_H256_E20_K24.txt 12
runft b1024_c7 bench_H1024_E10_K7.txt 7
runft b96_c48 bench96.txt 48
runft b96_c96 bench96.txt 96
date >> "$RUN/runs.txt"
echo H2B2_DONE
