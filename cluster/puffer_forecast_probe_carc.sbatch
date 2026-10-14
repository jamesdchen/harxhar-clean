#!/bin/bash
# 2026-10-12 puffer_forecast Phase 1: GPU probe (driver / GPU model / toolchain) on CARC gpu partition.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=pf_probe
#SBATCH --output=logs/pf_probe_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 nccl/2.27.7-1-cuda-12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
echo "CUDA_HOME=$CUDA_HOME NCCL_HOME=${NCCL_HOME:-?}"; nvcc --version | tail -2; gcc --version | head -1; bash --version | head -1
ls /usr/lib64/libnvidia-ml.so* /usr/lib64/libGL.so* 2>&1 | head
