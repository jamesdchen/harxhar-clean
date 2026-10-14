#!/bin/bash
# 2026-10-13 puffer_forecast speed benchmark on ONE Hoffman2 GPU node: the native PufferLib 5.0 MinGRU trainer
# (experiments/puffer_forecast/native/forecast_train.cu) vs PyTorch eager / torch.compile / batched PyTorch
# (experiments/puffer_forecast/torch_bench.py, which imports the other session's torch_train.py unchanged),
# back to back on the same physical GPU.  fp32 everywhere.  Torch env: $HOME/.conda/envs/pfbench (torch
# 2.5.1+cu124, built by cluster/puffer_forecast_bench_env_h2.sh), called by absolute path, never activated.
#   cd /u/scratch/j/jamesdc1/puffer_forecast && qsub bench/puffer_forecast_bench_h2.sh
#$ -cwd
#$ -j y
#$ -o logs/
#$ -N pf_bench
#$ -l gpu,cuda=1,L40S,h_rt=3:00:00,h_data=32G
. /u/local/Modules/default/init/modules.sh
module load gcc/11.3.0 cuda/12.3 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
set -uo pipefail
ROOT=$PWD
PUFFER=$ROOT/PufferLib5
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
PY=$HOME/.conda/envs/pfbench/bin/python
RUN=$ROOT/bench/runs/$JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_$JOB_ID TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_$JOB_ID  # cold caches
{ hostname; date; cat /etc/redhat-release; nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id,uuid --format=csv;
  nvcc --version | tail -1; gcc --version | head -1; (cd "$PUFFER" && git log -1 --format='%H %ci')
  "$PY" -c "import torch, triton, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__, 'numpy', numpy.__version__, 'device', torch.cuda.get_device_name(0))"
} > env.txt 2>&1
cat env.txt
[ "$(cd "$PUFFER" && git rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
(cd "$PUFFER" && git diff --quiet -- src) || { echo "PufferLib src/ modified"; exit 1; }
md5sum "$ROOT/native/forecast_train.cu" "$ROOT/bench/torch_bench.py" "$ROOT/bench/torch_train.py" > code_md5.txt
# ---- native build (fat binary; upstream algo.cu / ocean.cu unchanged)
ARCHS="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
( cd "$ROOT/native" && nvcc -O2 --threads 4 -Wno-deprecated-gpu-targets -std=c++17 $ARCHS -DPRECISION_FLOAT \
    -I. -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$CUDA_HOME/include" \
    -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 --diag-suppress=550 \
    forecast_train.cu -L"$CUDA_HOME/lib64" -lcublas -lcurand -o "$RUN/forecast_train" ) > build.log 2>&1 || { cat build.log; exit 1; }
BIN=$RUN/forecast_train
I=$ROOT/inputs
# ---- (1) native: suite A (the 24 scaling jobs; K = 48 runs them twice), suite B (K jobs of one shape)
"$BIN" run --jobs "$I/jobs_scaling.txt" --out nA_k1 --concurrent 1 > nA_k1.log 2>&1
"$BIN" run --jobs "$I/jobs_scaling.txt" --out nA_k8 --concurrent 8 > nA_k8.log 2>&1
"$BIN" run --jobs "$I/jobs_scaling.txt" --out nA_k24 --concurrent 24 > nA_k24.log 2>&1
"$BIN" run --jobs "$I/bench_A48.txt" --out nA_k48 --concurrent 48 > nA_k48.log 2>&1
for d in baseline all_features; do for K in 8 24 48; do
  "$BIN" run --jobs "$I/bench_B_${d}_K${K}.txt" --out "nB_${d}_K${K}" --concurrent $K > "nB_${d}_K${K}.log" 2>&1
done; done
tail -n 2 nA_*.log nB_*.log
# ---- (2)-(4) PyTorch on the same GPU (eager, torch.compile reduce-overhead / max-autotune, batched)
"$PY" "$ROOT/bench/torch_bench.py" --root "$ROOT" --native-a "$RUN/nA_k1" --out "$RUN/torch" --ma-budget 1800 > torch.log 2>&1 \
  || { echo "torch_bench failed"; tail -40 torch.log; }
tail -5 torch.log
# ---- native K = 1 again after the PyTorch runs (clock / thermal drift check on the same GPU)
"$BIN" run --jobs "$I/jobs_scaling.txt" --out nA_k1_again --concurrent 1 > nA_k1_again.log 2>&1
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv
date; echo BENCH_DONE
