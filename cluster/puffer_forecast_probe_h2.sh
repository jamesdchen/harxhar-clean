#!/bin/bash
# 2026-10-12 puffer_forecast Phase 1: GPU probe (driver / GPU model / OS / toolchain) on Hoffman2 (SGE).
#   qsub -l gpu,cuda=1,h_rt=0:10:00,h_data=4G cluster/puffer_forecast_probe_h2.sh
#$ -cwd
#$ -j y
#$ -o logs/
#$ -N pf_probe
. /u/local/Modules/default/init/modules.sh
module load gcc/11.3.0 cuda/12.3 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
hostname; date; cat /etc/redhat-release; ldd --version | head -1
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
which nvcc; nvcc --version | tail -2; gcc --version | head -1; bash --version | head -1
