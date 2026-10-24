#!/bin/bash
# 2026-10-23 MinGRU v2 Hoffman2 GPU test, diagnostic job (after 15260303: "illegal memory access" on an RTX 2080 Ti,
# sm_75, intermittently, with the CARC binary and with a Hoffman2 build of the CARC flags, whose fat binary has no
# sm_75 SASS and so runs its sm_70 SASS there).
#  (1) build native_g9 twice with Hoffman2's nvcc 12.3 / gcc 11.3: ft_std = the CARC flags; ft_75 = the CARC flags
#      plus -gencode arch=compute_75,code=sm_75 (a DIFFERENT build);
#  (2) compute-sanitizer memcheck of a short G1 file (the three G1 lines at 2 epochs) with each build: does any
#      kernel read / write out of bounds (a latent trainer bug would also matter on CARC)?
#  (3) the G1 file twice with ft_75 and, if both complete, the throughput bench as h2_g1_bench.sh.
#   cd /u/scratch/j/jamesdc1/pmf2_h2 && qsub -l gpu,RTX2080Ti,cuda=1,h_rt=1:00:00,h_data=16G h2_diag.sh
#$ -cwd
#$ -j y
#$ -o logs/
#$ -N v2h2dg
. /u/local/Modules/default/init/modules.sh
module load gcc/11.3.0 cuda/12.3 >/dev/null 2>&1
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
cp "$ROOT/inputs/g1_baseline.txt" "$ROOT/inputs/bench48.txt" "$ROOT/inputs/bench96.txt" "$LOCAL/"
sed -e 's/ epochs=300 max_epochs=300 patience=300 / epochs=2 max_epochs=2 patience=2 /' -e 's/ max_epochs=300 patience=300 / max_epochs=2 patience=2 /' \
  "$LOCAL/g1_baseline.txt" > "$LOCAL/g1_short2.txt"
{ hostname; date; cat /etc/redhat-release
  echo "JOB_ID=$JOB_ID QUEUE=${QUEUE:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv
  module list 2>&1; which nvcc compute-sanitizer; nvcc --version | tail -1; gcc --version | head -1
  md5sum "$ROOT"/inputs/*.txt "$ROOT"/src/native_g9/* "$LOCAL/g1_short2.txt"; cat "$LOCAL/g1_short2.txt"
  echo "--- load"; uptime; nproc; } > "$RUN/env.txt" 2>&1
cat "$RUN/env.txt"
[ "$(cd "$PUFFER" && git rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
(cd "$PUFFER" && git diff --quiet -- src) || { echo "PufferLib src/ modified"; exit 1; }
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
mkdir -p "$LOCAL/build" "$LOCAL/nvtmp"
cp "$ROOT"/src/native_g9/* "$LOCAL/build/"
ARCHS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
build() {  # output-name extra-archs
  ( cd "$LOCAL/build" && TMPDIR=$LOCAL/nvtmp nvcc -O2 --threads 4 -Wno-deprecated-gpu-targets -std=c++17 $ARCHS $2 -DPRECISION_FLOAT \
    -I. -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$CUDA_HOME/include" \
    -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 \
    --diag-suppress=550 forecast_train.cu -L"$CUDA_HOME/lib64" -lcublas -lcurand -o "$LOCAL/$1" ) > "$RUN/build_$1.log" 2>&1 \
    || { tail -30 "$RUN/build_$1.log"; exit 1; }
  echo "$1: $(nvcc --version | tail -1); $(gcc --version | head -1); ARCHS $ARCHS $2; $(md5sum < "$LOCAL/$1" | cut -c1-32)" | tee -a "$RUN/builds.txt"
  cp "$LOCAL/$1" "$RUN/"
}
build ft_std ""
build ft_75 "-gencode arch=compute_75,code=sm_75"

sampler() {
  while :; do
    t=$(date +%s)
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,$1,/" >> "$RUN/gpu_apps.csv"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,$1,/" >> "$RUN/gpu_dev.csv"
    sleep 2
  done
}
runft() {  # label jobs-file concurrency binary [prefix command...]
  local lab=$1 jf=$2 k=$3 bin=$4 t0 rc
  shift 4
  mkdir -p "$LOCAL/$lab"
  cd "$LOCAL/$lab" || return 9
  ln -sfn "$LOCAL/data" data
  sampler "$lab" &
  SMI=$!
  t0=$(date +%s.%N)
  "$@" "$bin" run --jobs "$LOCAL/$jf" --out tune --concurrent "$k" --chunk 5 --pool > tune.log 2>&1
  rc=$?
  kill $SMI 2>/dev/null
  wait $SMI 2>/dev/null
  echo "$lab rc=$rc wall_s=$(echo "$(date +%s.%N) - $t0" | bc) conc=$k jobs=$(wc -l < "$LOCAL/$jf") bin=$(basename "$bin") $(md5sum < "$bin" | cut -c1-8) $*" | tee -a "$RUN/runs.txt"
  rm -f data
  cd "$LOCAL" || return 9
  return $rc
}
nok() {
  [ -f "$LOCAL/$1/tune/jobs_out.csv" ] && awk -F, 'NR > 1 && $0 ~ /,ok,/' "$LOCAL/$1/tune/jobs_out.csv" | wc -l || echo 0
}
SAN="compute-sanitizer --tool memcheck --print-limit 30 --error-exitcode 86"
runft san_std g1_short2.txt 3 "$LOCAL/ft_std" $SAN
runft san_75 g1_short2.txt 3 "$LOCAL/ft_75" $SAN
runft g1_r1 g1_baseline.txt 3 "$LOCAL/ft_75"
runft g1_r2 g1_baseline.txt 3 "$LOCAL/ft_75"
{ for f in steps.csv epochs.csv preds.csv; do
    cmp -s "$LOCAL/g1_r1/tune/$f" "$LOCAL/g1_r2/tune/$f" && echo "$f identical" || echo "$f differs"
  done; } | tee "$RUN/r1_vs_r2.txt"
if [ "$(nok g1_r1)" = 3 ] && [ "$(nok g1_r2)" = 3 ]; then
  for c in 12 24 48; do runft "b48_c$c" bench48.txt "$c" "$LOCAL/ft_75"; done
  runft b96_c96 bench96.txt 96 "$LOCAL/ft_75"
fi
for r in san_std san_75 g1_r1 g1_r2; do
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
echo H2DIAG_DONE
