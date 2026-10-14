#!/bin/bash
# 2026-10-13 puffer_forecast capacity / memorisation sweep (experiments/puffer_forecast/overfit.py): one Slurm
# array task per task file inputs/<ROUND>/task_XX_cC.txt (C = models trained together, --pool), any CARC GPU.
# The binary is built ONCE per submission by puffer_forecast_build_once_carc.sbatch into $ROOT/bin/$SRC/
# (fc_train: exact in-sample forecasts of every training row):
#   cd $HOME/puffer_forecast && B=$(sbatch --parsable --export=NONE,SRC=native_g4 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE,SRC=native_g4,ROUND=overfit_r1 --dependency=afterok:$B --array=0-10 puffer_forecast_overfit_carc.sbatch
# Round 1 (13083705 / 13083909 / 13083999) ran an earlier version of this script that built native_of3 in every
# task (0e59b80c) without CUDA_DEVICE_MAX_CONNECTIONS.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=pf_overfit
#SBATCH --output=logs/pf_overfit_%A_%a.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
ROUND=${ROUND:-overfit_r1}
SRC=${SRC:-native_g4}
ROOT=$HOME/puffer_forecast
BIN=$ROOT/bin/$SRC/forecast_train
TASK=$(printf %02d "$SLURM_ARRAY_TASK_ID")
F=$(ls "$ROOT/inputs/$ROUND"/task_${TASK}_c*.txt)
C=${F##*_c}; C=${C%.txt}
RUN=$ROOT/runs/${ROUND}_${SLURM_ARRAY_JOB_ID}
OUT=$RUN/task_$TASK
mkdir -p "$OUT"; cd "$RUN"; ln -sfn "$ROOT/data" data
[ -x "$BIN" ] || { echo "no binary $BIN (submit puffer_forecast_build_once_carc.sbatch first)"; exit 1; }
export CUDA_DEVICE_MAX_CONNECTIONS=32  # 2 streams per model on 32 hardware queues instead of 8 (byte-identical)
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  cat "$ROOT/bin/$SRC/provenance.txt"; md5sum "$BIN" "$F" "$ROOT"/data/*/*
  echo "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"; } > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
a=$(date +%s)
"$BIN" run --jobs "$F" --out "$OUT" --concurrent "$C" --chunk 50 --pool > "$OUT/run.log" 2>&1
echo "task $TASK: $(wc -l < "$F") jobs, $C at a time, $(( $(date +%s) - a )) s" | tee -a "$OUT/env.txt"
gzip -f "$OUT/steps.csv"
echo OVERFIT_TASK_DONE
