#!/bin/bash
# Login-node prep, then the L40S job, then the next pooled-forest wave.
# Prebuilt GPU wheel only. Does not compile PyTorch.
set -euo pipefail
ROOT=$HOME/puffer_forecast
cd "$ROOT"
module load conda >/dev/null 2>&1
if ! "$ROOT/py-gpu/bin/python" -c 'import torch; assert torch.version.cuda' >/dev/null 2>&1; then
  source activate
  python -m venv "$ROOT/py-gpu"
  "$ROOT/py-gpu/bin/pip" install --upgrade pip
  "$ROOT/py-gpu/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu128
fi
"$ROOT/py-gpu/bin/python" -c 'import torch; print("WHEEL", torch.__version__, "CUDA", torch.version.cuda); assert torch.version.cuda'
test -f "$ROOT/runs/13037177/scale_k1/jobs_out.csv"
test -f "$ROOT/inputs/jobs_scaling.txt"
test -f "$ROOT/torch_train.py"
mkdir -p "$ROOT/logs"
J=$(sbatch --parsable --export=NONE "$ROOT/puffer_forecast_torch_carc.sbatch")
echo "TORCH_JOB $J"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE
cd /scratch1/jc_905/harxhar-pooltrees
WAVE=7 NICE=0 MAXTASKS=4800 bash cluster/slurm/pooltree_af_wave.sh
echo WAVE_DONE
