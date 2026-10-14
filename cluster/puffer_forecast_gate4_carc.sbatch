#!/bin/bash
# 2026-10-13 puffer_forecast gate 4 (user: the four speed items ONLY).  NEW = native_g4/ = native_of3/ (0e59b80c)
# plus exactly (1) no free / realloc between pool jobs (stream-ordered shape buffers, slot-lifetime stream /
# cuBLAS handles + workspaces / Muon scalars, per-slot fc_train buffer, initial weights pre-warmed) and (2) the
# contiguous-copy gather (FT_GATHER_OLD=1 switches it back, for the A/B); (3) CUDA_DEVICE_MAX_CONNECTIONS=32 and
# (4) build once per submission are script items.  Both binaries come prebuilt (item 4):
#   cd $HOME/puffer_forecast
#   B1=$(sbatch --parsable --export=NONE,SRC=native_of3 puffer_forecast_build_once_carc.sbatch)
#   B2=$(sbatch --parsable --export=NONE,SRC=native_g4 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE --dependency=afterok:$B1:$B2 puffer_forecast_gate4_carc.sbatch
# One GPU:
#   identity  OLD vs NEW --pool: Phase 2c block-0 tune (preds, best_epoch, every log row), 500 block-0 refits, the
#             fc_train jobs, a warm start; NEW --pool vs NEW wave; NEW with FT_GATHER_OLD=1
#   timing    bench suite B (wave, K 8 / 24 / 48, both designs, 3 repeats) and the 2,500 block-0 refits and the
#             block-0 tune (--pool K48 chunk 5): OLD; NEW FT_GATHER_OLD=1 (item 1); NEW (1+2); OLD with
#             CUDA_DEVICE_MAX_CONNECTIONS=32 (item 3); NEW with it (1+2+3)
#   nsys      500 refits, OLD and NEW: device-wide-sync-type API calls after the pool's fill
# CUDA_DEVICE_MAX_CONNECTIONS is set per command here (the A/B), not exported for the whole job.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=pf_gate4
#SBATCH --output=logs/pf_gate4_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
ROOT=$HOME/puffer_forecast
OLD=${OLD:-native_of3}
NEW=${NEW:-native_g4}
RUN=$ROOT/runs/gate4s_$SLURM_JOB_ID
I=$ROOT/inputs
G=$I/gate4
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$G/warm.txt" "$I/overfit_gate/fctrain_jobs.txt" "$ROOT"/data/*/*; } > env.txt 2>&1
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
prebuilt "$OLD" ft_old
prebuilt "$NEW" ft_new
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
# ---- identity
for v in old new; do
  T id_${v}_tune ./ft_$v run --jobs "$TUNE" --out id_${v}_tune --concurrent 48 --chunk 5 --pool
  T id_${v}_refit ./ft_$v run --jobs refit500.txt --out id_${v}_refit --concurrent 48 --chunk 5 --pool
  T id_${v}_fct ./ft_$v run --jobs "$FCT" --out id_${v}_fct --concurrent 6 --chunk 5 --pool
  T id_${v}_warm ./ft_$v run --jobs "$G/warm.txt" --out id_${v}_warm --concurrent 2 --chunk 5 --pool
done
T id_new_wave_tune ./ft_new run --jobs "$TUNE" --out id_new_wave_tune --concurrent 48 --chunk 5
T id_new_wave_refit ./ft_new run --jobs refit500.txt --out id_new_wave_refit --concurrent 48 --chunk 5
T id_new_wave_fct ./ft_new run --jobs "$FCT" --out id_new_wave_fct --concurrent 6 --chunk 5
T id_new_gold_refit env FT_GATHER_OLD=1 ./ft_new run --jobs refit500.txt --out id_new_gold_refit --concurrent 48 --chunk 5 --pool
T id_new_c32_refit env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs refit500.txt --out id_new_c32_refit --concurrent 48 --chunk 5 --pool
# ---- timing
for rep in 1 2 3; do
  for d in baseline all_features; do
    for K in 8 24 48; do
      f=$I/bench_B_${d}_K${K}.txt
      x=${d}_K${K}_r$rep
      T tB_old_$x ./ft_old run --jobs "$f" --out tB_old_$x --concurrent $K
      T tB_gold_$x env FT_GATHER_OLD=1 ./ft_new run --jobs "$f" --out tB_gold_$x --concurrent $K
      T tB_new_$x ./ft_new run --jobs "$f" --out tB_new_$x --concurrent $K
      T tB_oldc32_$x env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_old run --jobs "$f" --out tB_oldc32_$x --concurrent $K
      T tB_newc32_$x env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs "$f" --out tB_newc32_$x --concurrent $K
    done
  done
done
for rep in 1 2; do
  T tR_old_r$rep ./ft_old run --jobs refit_all.txt --out tR_old_r$rep --concurrent 48 --chunk 5 --pool
  T tR_gold_r$rep env FT_GATHER_OLD=1 ./ft_new run --jobs refit_all.txt --out tR_gold_r$rep --concurrent 48 --chunk 5 --pool
  T tR_new_r$rep ./ft_new run --jobs refit_all.txt --out tR_new_r$rep --concurrent 48 --chunk 5 --pool
  T tR_oldc32_r$rep env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_old run --jobs refit_all.txt --out tR_oldc32_r$rep --concurrent 48 --chunk 5 --pool
  T tR_newc32_r$rep env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs refit_all.txt --out tR_newc32_r$rep --concurrent 48 --chunk 5 --pool
  T tT_old_r$rep ./ft_old run --jobs "$TUNE" --out tT_old_r$rep --concurrent 48 --chunk 5 --pool
  T tT_new_r$rep ./ft_new run --jobs "$TUNE" --out tT_new_r$rep --concurrent 48 --chunk 5 --pool
  T tT_newc32_r$rep env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs "$TUNE" --out tT_newc32_r$rep --concurrent 48 --chunk 5 --pool
done
# ---- nsys: API trace of 500 refits, old and new
for v in old new; do
  nsys profile -t cuda,cublas,osrt -o nsys_$v --force-overwrite true ./ft_$v run --jobs refit500.txt --out nsys_${v}_out --concurrent 48 --chunk 5 --pool > nsys_$v.log 2>&1
  nsys stats --report cuda_api_trace --format csv -o nsys_$v nsys_$v.nsys-rep > /dev/null 2>&1
  gzip -f nsys_${v}_cuda_api_trace.csv
done
for d in tB_* tR_* tT_* nsys_*_out; do
  [ -d "$d" ] && rm -f "$d/steps.csv" "$d/epochs.csv"
done
gzip -f id_*/steps.csv 2>/dev/null
rm -f ft_old ft_new nsys_*.sqlite
date; echo GATE4_DONE
