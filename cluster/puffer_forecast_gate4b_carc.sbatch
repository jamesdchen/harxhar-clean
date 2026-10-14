#!/bin/bash
# 2026-10-13 puffer_forecast gate 4b: native_of5/ = native_of4/ with the largest-first queue choice by each
# shape queue's REMAINING work (gate 4: by the front job's cost, every slot piled onto one queue and the tune
# pool re-captured 120 times instead of 53).  Against the current binary (native_of3/), one GPU:
#   identity  old vs new --pool: Phase 2c block-0 tune (preds, best_epoch; new drops past-stop log rows) and
#             500 block-0 refits
#   timing    tune block 0 (120 jobs) and the 2,500 block-0 refits, --pool K48, CUDA_DEVICE_MAX_CONNECTIONS
#             unset / 32, --chunk 5 / 10 / 20
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_gate4b_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=pf_gate4b
#SBATCH --output=logs/pf_gate4b_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
OLD=${OLD:-native_of3}
NEW=${NEW:-native_of5}
RUN=$ROOT/runs/gate4b_$SLURM_JOB_ID
I=$ROOT/inputs
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
export TMPDIR=$RUN/tmp; mkdir -p "$TMPDIR"
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$ROOT"/data/*/*; } > env.txt 2>&1
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
prebuilt "$OLD" "$RUN/ft_old"
prebuilt "$NEW" "$RUN/ft_new"
awk 'NR <= 500' "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" > refit500.txt
cp "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" refit_all.txt
TUNE=$I/phase2c/tune_block0.txt
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
done
T tT_old_ch5 ./ft_old run --jobs "$TUNE" --out tT_old_ch5 --concurrent 48 --chunk 5 --pool
for ch in 5 10 20; do
  T tT_new_ch$ch ./ft_new run --jobs "$TUNE" --out tT_new_ch$ch --concurrent 48 --chunk $ch --pool
  T tT_c32_ch$ch env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs "$TUNE" --out tT_c32_ch$ch --concurrent 48 --chunk $ch --pool
done
T tR_old_ch5 ./ft_old run --jobs refit_all.txt --out tR_old_ch5 --concurrent 48 --chunk 5 --pool
T tR_new_ch5 ./ft_new run --jobs refit_all.txt --out tR_new_ch5 --concurrent 48 --chunk 5 --pool
for ch in 5 10 20; do
  T tR_c32_ch$ch env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs refit_all.txt --out tR_c32_ch$ch --concurrent 48 --chunk $ch --pool
done
for d in tT_* tR_*; do
  [ -d "$d" ] && rm -f "$d/steps.csv" "$d/epochs.csv"
done
gzip -f id_*/steps.csv 2>/dev/null
rm -rf ft_old ft_new "$TMPDIR"
date; echo GATE4B_DONE
