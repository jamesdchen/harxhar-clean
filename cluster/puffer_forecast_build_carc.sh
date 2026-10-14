#!/bin/bash
# 2026-10-12 puffer_forecast: build the supervised native trainer against PufferLib 5.0 (no PyTorch).
#   bash cluster/puffer_forecast_build_carc.sh            (CARC login or GPU node; ~1-2 min)
# Upstream: PufferAI/PufferLib branch 5.0 @ 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2 (= release tag
# 5.0-experiments 89414204 + 2 commits, no src/ change), checked out at $PUFFER; its src/algo.cu and
# src/ocean.cu are compiled unchanged, our native/ files are compiled with them.  fp32 build
# (-DPRECISION_FLOAT: upstream build.sh --float).  Fat binary for every CARC GPU class.
set -euo pipefail
ROOT=${ROOT:-$HOME/puffer_forecast}
PUFFER=${PUFFER:-$HOME/src/PufferLib5}
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
[ "$(git -C "$PUFFER" rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
git -C "$PUFFER" diff --quiet -- src || { echo "PufferLib src/ modified"; exit 1; }
cd "$ROOT/native"
sha256sum "$PUFFER/src/algo.cu" "$PUFFER/src/ocean.cu" "$PUFFER/src/pufferl.cu" > upstream_sha256.txt
ARCHS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
echo "nvcc: $(nvcc --version | tail -1); gcc: $(gcc --version | head -1)"
nvcc -O2 --threads 1 -std=c++17 $ARCHS -DPRECISION_FLOAT \
  -I. -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$CUDA_HOME/include" \
  -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 --diag-suppress=550 -Wno-deprecated-gpu-targets \
  forecast_train.cu -L"$CUDA_HOME/lib64" -lcublas -lcurand -o forecast_train
echo "built $ROOT/native/forecast_train"
