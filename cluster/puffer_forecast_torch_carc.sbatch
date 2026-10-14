#!/bin/bash
# PyTorch MinGRU on one L40S, same jobs and epoch counts as native scale_k1 (runs/13037177).
# The harxhar env ships torch 2.12.1+cpu. This job uses a prebuilt GPU wheel in $ROOT/py-gpu.
# It does not compile PyTorch. It exits if the process is not running on a GPU.
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_torch_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=pollok_1603
#SBATCH --job-name=pf_torch
#SBATCH --output=logs/pf_torch_%j.out
set -euo pipefail
ROOT=$HOME/puffer_forecast
cd "$ROOT"
module purge >/dev/null 2>&1
module load ver/2506 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
PY=$ROOT/py-gpu/bin/python
if ! "$PY" -c 'import torch; assert torch.version.cuda' 2>/dev/null; then
  echo "GPU torch wheel missing in $ROOT/py-gpu" >&2
  exit 1
fi
{
  hostname
  date
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
} | tee "$ROOT/logs/pf_torch_env_${SLURM_JOB_ID}.txt"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "pytorch is not using a GPU"'
"$PY" "$ROOT/torch_train.py" \
  --jobs "$ROOT/inputs/jobs_scaling.txt" \
  --epochs-from "$ROOT/runs/13037177/scale_k1/jobs_out.csv" \
  --data-root "$ROOT" \
  --out "$ROOT/runs/torch_${SLURM_JOB_ID}"
date
echo TORCH_DONE
