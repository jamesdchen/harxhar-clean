#!/bin/bash
# 2026-10-13 puffer_forecast speed benchmark: the PyTorch env on CARC (login node), identical to the Hoffman2
# one (torch 2.5.1+cu124, python 3.11) so the benchmark script and versions match whichever cluster runs it.
#   nohup bash puffer_forecast_bench_env_carc.sh > bench_env.log 2>&1 &
set -euo pipefail
ENV=$HOME/.conda/envs/pfbench
CONDA=/apps/conda/miniforge3/25.3.0/bin/conda
[ -x "$ENV/bin/python" ] || "$CONDA" create -p "$ENV" -y --solver=libmamba python=3.11 numpy
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
"$ENV/bin/python" -c "import torch, triton, numpy; print('ENV_OK torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__, 'numpy', numpy.__version__)"
