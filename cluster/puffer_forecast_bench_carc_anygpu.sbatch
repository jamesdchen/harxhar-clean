#!/bin/bash
# 2026-10-13 puffer_forecast speed benchmark on ONE CARC L40S node -- identical to
# cluster/puffer_forecast_bench_h2.sh (same native source, same torch_bench.py / torch_train.py, same job files,
# same torch 2.5.1+cu124 env built by cluster/puffer_forecast_bench_env_carc.sh); submitted on CARC because the
# Hoffman2 L40S queue had not started the benchmark; whichever starts first runs, the other is held.
# Twin with any Triton-capable GPU (sm_70+; P100 cannot run torch.compile) after no L40S scheduled in time.
# Toolchain for the native build: CUDA 12.9.1 / gcc 14.3.0 (Hoffman2: CUDA 12.3 / gcc 11.3.0).
#   cd $HOME/puffer_forecast && sbatch --export=NONE bench/puffer_forecast_bench_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100|a40|a100|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=pf_bencha
#SBATCH --output=logs/pf_bench_%j.out
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
JOB_ID=$SLURM_JOB_ID
cd "$HOME/puffer_forecast"
set -uo pipefail
ROOT=$PWD
PUFFER=$HOME/src/PufferLib5
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
prebuilt "native" "$RUN/forecast_train"
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
