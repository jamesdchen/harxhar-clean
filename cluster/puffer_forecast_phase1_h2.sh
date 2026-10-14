#!/bin/bash
# 2026-10-12 puffer_forecast Phase 1 on one Hoffman2 GPU (SGE) -- the same steps as
# cluster/puffer_forecast_phase1_carc.sbatch (the CARC job is the primary; this is the fallback when the
# CARC gpu partition is slow to schedule).  Toolchain: CUDA 12.3 + gcc 11.3 (Hoffman2 is CentOS 7).
#   cd $H2ROOT && qsub -l gpu,cuda=1,L40S,h_rt=3:00:00,h_data=24G puffer_forecast_phase1_h2.sh
#$ -cwd
#$ -j y
#$ -o logs/
#$ -N pf_phase1
. /u/local/Modules/default/init/modules.sh
module load gcc/11.3.0 cuda/12.3 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
set -uo pipefail
ROOT=$PWD
PUFFER=$ROOT/PufferLib5
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
RUN=$ROOT/runs/$JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
{ hostname; date; cat /etc/redhat-release; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv;
  nvcc --version | tail -1; gcc --version | head -1; (cd "$PUFFER" && git log -1 --format='%H %ci'); } > env.txt 2>&1
cat env.txt
[ "$(cd "$PUFFER" && git rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
(cd "$PUFFER" && git diff --quiet -- src) || { echo "PufferLib src/ modified"; exit 1; }
sha256sum "$PUFFER/src/algo.cu" "$PUFFER/src/ocean.cu" "$PUFFER/src/pufferl.cu" > upstream_sha256.txt
ARCHS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
( cd "$ROOT/native" && nvcc -O2 --threads 4 -Wno-deprecated-gpu-targets -std=c++17 $ARCHS -DPRECISION_FLOAT \
    -I. -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$CUDA_HOME/include" \
    -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 --diag-suppress=550 \
    forecast_train.cu -L"$CUDA_HOME/lib64" -lcublas -lcurand -o "$RUN/forecast_train" ) > build.log 2>&1 || { cat build.log; exit 1; }
BIN=$RUN/forecast_train
for k in mse huber qlike; do "$BIN" losscheck "$ROOT/inputs/losscheck_$k.bin" "losscheck_$k.out" ; done
"$BIN" gradcheck --jobs "$ROOT/inputs/jobs_grad.txt" --out gradcheck > gradcheck.log 2>&1 || cat gradcheck.log
"$BIN" run --jobs "$ROOT/inputs/jobs_toy.txt" --out toy --concurrent 3 > toy.log 2>&1 || cat toy.log
t0=$(date +%s.%N); "$BIN" run --jobs "$ROOT/inputs/jobs_phase1.txt" --out phase1 --concurrent 48 --chunk 5 > phase1.log 2>&1 || tail -30 phase1.log; echo "phase1 wall $(echo "$(date +%s.%N) - $t0" | bc) s" | tee phase1.time
"$BIN" run --jobs "$ROOT/inputs/jobs_scaling.txt" --out scale_k1 --concurrent 1 > scale_k1.log 2>&1
"$BIN" run --jobs "$ROOT/inputs/jobs_scaling.txt" --out scale_k24 --concurrent 24 > scale_k24.log 2>&1
"$BIN" run --jobs "$ROOT/inputs/jobs_scaling.txt" --out scale_k8 --concurrent 8 > scale_k8.log 2>&1
date; echo PHASE1_DONE
