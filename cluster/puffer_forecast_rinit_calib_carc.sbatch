#!/bin/bash
# 2026-10-15 puffer_forecast variant (b) calibration: the shortcut initialised at the per-window closed-form ridge
# (lin_init = data/lin_init/<design>/*.f32, experiments/puffer_forecast/ridge_init_study.py), native_g8 unchanged.
#   check   the init's forecast through the trainer (identity, nothing trained) at the stack's block starts
#   val     the init's validation loss on the block-0 tune window (nothing trained), softplus and identity
#   drift   block-0 refits after 5 / 10 / 20 / 40 epochs (save_params: v) and tunes (best epochs), lr_lin
#           {0.0003, 0.001, 0.01} x wd_lin x wd_enc x two configurations, 5 seeds
#   cd $HOME/puffer_forecast
#   sbatch --export=NONE,GPU_TYPE=p100 --constraint=p100 puffer_forecast_rinit_calib_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=pf_rinit
#SBATCH --output=logs/pf_rinit_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=$HOME/puffer_forecast
SRC=${SRC:-native_g8}
RUN=$ROOT/runs/${CALIB:-rinit_calib}_$SLURM_JOB_ID
I=$ROOT/inputs/${CALIB:-rinit_calib}  # CALIB=qcal: the QLIKE round's drift check
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$SRC"/*.cu* "$I"/*.txt; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT" | tee gpu_type.txt >> env.txt
b=$ROOT/bin/$SRC
grep -q "$(md5sum < "$ROOT/$SRC/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" || { echo "stale $b"; exit 1; }
cp "$b/forecast_train" ft; cat "$b/provenance.txt" >> env.txt
echo "name,rc,t0,t1" > rc.csv
R() { local n=$1 a; shift; a=$(date +%s); "$@" > "$n.log" 2>&1; echo "$n,$?,$a,$(date +%s)" >> rc.csv; }
for n in check val drift_tune drift_refit; do
  [ -f "$I/$n.txt" ] && R $n ./ft run --jobs "$I/$n.txt" --out $n --concurrent 48 --chunk 5 --pool
done
gzip -f */steps.csv 2>/dev/null
rm -f ft
date; echo RINIT_DONE
