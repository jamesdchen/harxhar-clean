#!/bin/bash
# 2026-10-13 puffer_forecast speed benchmark: the PyTorch env on Hoffman2 (login node, ~10 min).
#   nohup bash puffer_forecast_bench_env_h2.sh > bench_env.log 2>&1 &
# Hoffman2 is CentOS 7 (glibc 2.17): torch >= 2.6 wheels need glibc 2.28, so the newest CUDA build that
# installs is torch 2.5.1+cu124 (triton 3.1.0; the GPU nodes' driver 550 runs CUDA 12.4).  Never activated
# in a submitting shell: the job calls $ENV/bin/python by absolute path.
set -euo pipefail
ENV=$HOME/.conda/envs/pfbench
CONDA=/u/local/apps/anaconda3/2023.03/bin/conda
[ -x "$ENV/bin/python" ] || "$CONDA" create -p "$ENV" -y python=3.11 numpy
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
"$ENV/bin/python" -c "import torch, triton, numpy; print('ENV_OK torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__, 'numpy', numpy.__version__)"
