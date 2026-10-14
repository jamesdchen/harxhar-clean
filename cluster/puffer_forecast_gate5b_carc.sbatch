#!/bin/bash
# 2026-10-14 puffer_forecast gate 5b: native_g5b/ = native_g5/ with the pool's queue choice back to most-jobs-left
# (gate 5: the largest-first choice doubled the tune pool's re-captures and cost +35 % tune wall).  Against
# native_g4/ (OLD) and native_g5/: identity (tune preds / best_epoch with the past-stop log rows dropped; 500
# refits; the fc_train jobs) and the tune / refit walls, CUDA_DEVICE_MAX_CONNECTIONS=32.
#   cd $HOME/puffer_forecast && sbatch --export=NONE --dependency=afterok:<build of native_g5b> puffer_forecast_gate5b_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=pf_gate5b
#SBATCH --output=logs/pf_gate5b_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
RUN=$ROOT/runs/gate5b_$SLURM_JOB_ID
I=$ROOT/inputs
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT"/native_g4/*.cu* "$ROOT"/native_g5/*.cu* "$ROOT"/native_g5b/*.cu*; } > env.txt 2>&1
cat env.txt
# the trainer, built ONCE per submission by puffer_forecast_build_once_carc.sbatch (gate 4, 2026-10-13), not in
# this job
prebuilt() {  # prebuilt SRC_DIR DEST: $ROOT/bin/SRC_DIR/forecast_train, checked against the current source
  local b=$ROOT/bin/$1
  [ -x "$b/forecast_train" ] || { echo "no prebuilt $b/forecast_train (puffer_forecast_build_once_carc.sbatch, SRC=$1)"; exit 1; }
  grep -q "$(md5sum < "$ROOT/$1/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" \
    || { echo "prebuilt $b is stale: $ROOT/$1/forecast_train.cu changed since it was built"; exit 1; }
  cp "$b/forecast_train" "$2"
  { echo "prebuilt $1:"; cat "$b/provenance.txt"; } >> env.txt
}
prebuilt native_g4 ft_old
prebuilt native_g5 ft_g5
prebuilt native_g5b ft_new
awk 'NR <= 500' "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" > refit500.txt
cp "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" refit_all.txt
TUNE=$I/phase2c/tune_block0.txt
FCT=$I/overfit_gate/fctrain_jobs.txt
T() {  # T name cmd... : wall-clock seconds of one run -> timing.csv
  local n=$1; shift
  local a=$(date +%s.%N)
  "$@" > "$n.log" 2>&1
  local rc=$?
  echo "$n,$rc,$(awk -v a="$a" -v b="$(date +%s.%N)" 'BEGIN{printf "%.3f", b - a}')" >> timing.csv
}
echo "name,rc,wall_s" > timing.csv
for v in old new; do
  T id_${v}_tune ./ft_$v run --jobs "$TUNE" --out id_${v}_tune --concurrent 48 --chunk 5 --pool
  T id_${v}_refit ./ft_$v run --jobs refit500.txt --out id_${v}_refit --concurrent 48 --chunk 5 --pool
  T id_${v}_fct ./ft_$v run --jobs "$FCT" --out id_${v}_fct --concurrent 6 --chunk 5 --pool
done
for rep in 1 2 3; do
  for v in old g5 new; do
    T tT_${v}_r$rep ./ft_$v run --jobs "$TUNE" --out tT_${v}_r$rep --concurrent 48 --chunk 5 --pool
  done
done
for rep in 1 2; do
  for v in old g5 new; do
    T tR_${v}_r$rep ./ft_$v run --jobs refit_all.txt --out tR_${v}_r$rep --concurrent 48 --chunk 5 --pool
  done
done
for d in tT_* tR_*; do
  [ -d "$d" ] && rm -f "$d/steps.csv" "$d/epochs.csv"
done
gzip -f id_*/steps.csv 2>/dev/null
rm -f ft_old ft_g5 ft_new
date; echo GATE5B_DONE
