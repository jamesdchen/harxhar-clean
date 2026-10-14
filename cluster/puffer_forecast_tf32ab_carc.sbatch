#!/bin/bash
# 2026-10-13 puffer_forecast: TF32 A/B on an A100 or L40S (the only --constraint: TF32 needs Ampere+):
# H1024 Ly2 L20, 30 epochs, both designs, fc_train=1 + save_params: default (fp32) vs --tf32-ns (Newton-Schulz
# GEMMs) vs --tf32 (every GEMM); wall, final in-sample 1-R2, forecast differences (overfit.py tf32ab).
#   cd $HOME/puffer_forecast && sbatch --export=NONE --dependency=afterok:<build of native_g5> puffer_forecast_tf32ab_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=pf_tf32ab
#SBATCH --output=logs/pf_tf32ab_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
NEW=${NEW:-native_g5}
RUN=$ROOT/runs/tf32ab_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
export TMPDIR=$RUN/tmp; mkdir -p "$TMPDIR"
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$NEW"/*.cu* "$ROOT/inputs/gate4/tf32ab.txt" "$ROOT"/data/*/*; } > env.txt 2>&1
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
prebuilt "$NEW" "$RUN/ft"
echo "name,wall_s" > timing.csv
for m in fp32 tf32-ns tf32; do
  opt=""
  [ "$m" != fp32 ] && opt="--$m"
  a=$(date +%s.%N)
  ./ft run --jobs "$ROOT/inputs/gate4/tf32ab.txt" --out ab_$m --concurrent 2 --chunk 30 --pool $opt > ab_$m.log 2>&1
  echo "$m,$(awk -v a="$a" -v b="$(date +%s.%N)" 'BEGIN{printf "%.3f", b - a}')" >> timing.csv
done
gzip -f ab_*/steps.csv
rm -rf ft "$TMPDIR"
date; echo TF32AB_DONE
