#!/bin/bash
# 2026-10-13 puffer_forecast: gate of the fc_train source (native_of/) -- with fc_train off it reproduces the
# Phase 2c binary (native_2c/) byte for byte on a block-0 sample (Phase 2c tune block 0 + 500 of its refits,
# --pool both), compared locally (pool_gate_report.same).  Any CARC GPU.
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_overfit_gate_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=pf_ofgate
#SBATCH --output=logs/pf_ofgate_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
RUN=$ROOT/runs/ofgate_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/native_2c/forecast_train.cu" "$ROOT/native_of/forecast_train.cu"; } > env.txt 2>&1
cat env.txt
# the trainer, built ONCE per submission by puffer_forecast_build_once_carc.sbatch (gate 4, 2026-10-13), not in
# this job:  B=$(sbatch --parsable --export=NONE,SRC=<dir> puffer_forecast_build_once_carc.sbatch), then submit
# this script with --dependency=afterok:$B
prebuilt() {  # prebuilt SRC_DIR DEST: $ROOT/bin/SRC_DIR/forecast_train, checked against the current source
  local b=$ROOT/bin/$1
  [ -x "$b/forecast_train" ] || { echo "no prebuilt $b/forecast_train (puffer_forecast_build_once_carc.sbatch, SRC=$1)"; exit 1; }
  grep -q "$(md5sum < "$ROOT/$1/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" \
    || { echo "prebuilt $b is stale: $ROOT/$1/forecast_train.cu changed since it was built"; exit 1; }
  cp "$b/forecast_train" "$2"
  { echo "prebuilt $1:"; cat "$b/provenance.txt"; } >> env.txt
}
prebuilt "native_2c" "$RUN/ft_old"
prebuilt "native_of" "$RUN/ft_new"
awk 'NR <= 500' "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" > refit500.txt
for v in old new; do
  ./ft_$v run --jobs "$ROOT/inputs/phase2c/tune_block0.txt" --out "${v}_tune" --concurrent 48 --chunk 5 --pool > "${v}_tune.log" 2>&1
  ./ft_$v run --jobs refit500.txt --out "${v}_refit" --concurrent 48 --chunk 5 --pool > "${v}_refit.log" 2>&1
done
gzip -f */steps.csv
rm -f ft_old ft_new
date; echo OFGATE_DONE
