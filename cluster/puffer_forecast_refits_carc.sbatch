#!/bin/bash
# 2026-10-15 puffer_forecast refit-only runs (the QLIKE round's option 1: the refits of the blocks / designs whose
# QLIKE re-selection changed, from phase2_select --criterion qlike on a finished walk's tune outputs).
#   cd $HOME/puffer_forecast
#   sbatch --export=NONE,GPU_TYPE=p100,SRC=native_g8,JOBS=inputs/qsel/rinit_b0.txt,TAG=qsel_rinit_b0 --constraint=p100 puffer_forecast_refits_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=pf_refits
#SBATCH --output=logs/pf_refits_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=$HOME/puffer_forecast
SRC=${SRC:-native_g8}
JOBS=$ROOT/${JOBS:?JOBS=inputs/...}
TAG=${TAG:?TAG}
RUN=$ROOT/runs/refits_${TAG}_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$SRC"/*.cu* "$JOBS"; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CONS=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'Features=[^ ]*' | head -1 | cut -d= -f2)
[ "$CONS" = "$GPU_TYPE" ] || { echo "--constraint '$CONS' != GPU_TYPE $GPU_TYPE"; exit 1; }
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT|constraint=$CONS" | tee gpu_type.txt >> env.txt
b=$ROOT/bin/$SRC
grep -q "$(md5sum < "$ROOT/$SRC/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" || { echo "stale $b"; exit 1; }
cp "$b/forecast_train" ft; cat "$b/provenance.txt" >> env.txt
./ft run --jobs "$JOBS" --out refit --concurrent 48 --chunk 5 --pool
gzip -f refit/steps.csv
rm -f ft
date; echo REFITS_DONE
