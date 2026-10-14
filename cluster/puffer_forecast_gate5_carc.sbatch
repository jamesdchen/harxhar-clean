#!/bin/bash
# 2026-10-13 puffer_forecast gate 5, on top of gate 4: NEW = native_g5/ = native_g4/ plus the correctness guards
# (F2-F7), the tune stop at best_epoch + patience, largest-first take() by each shape queue's remaining work, the
# refit forecast collected at a later poll, opt-in --tf32-ns / --tf32 / stop_loss, and the softplus output link
# (job key link = identity | softplus); sm_90 + PTX in the build.  OLD = native_g4/.  Both prebuilt:
#   cd $HOME/puffer_forecast
#   B=$(sbatch --parsable --export=NONE,SRC=native_g5 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE --dependency=afterok:$B puffer_forecast_gate5_carc.sbatch
# One GPU, CUDA_DEVICE_MAX_CONNECTIONS=32 throughout:
#   identity  OLD vs NEW --pool: Phase 2c block-0 tune (preds, best_epoch; NEW drops the log rows past the stop),
#             500 block-0 refits, the fc_train jobs, a warm start; link=identity written out vs absent; NEW pool
#             vs wave; NEW --tf32-ns / --tf32 (no TF32 hardware on P100 / V100: identical expected)
#   guards    malformed job lines / options exit 2
#   stop_loss a 1,000-epoch overfit job with stop_loss=1e-2 vs without
#   softplus  loss kernel (mse / huber / qlike) vs float64 + finite differences; network gradients (gradcheck);
#             an overfit job (baseline H64 Ly2 L20 lr 0.01, 1,000 epochs, fc_train) vs identity; Phase 2c block 0
#             tune + select + refit, both designs, softplus and identity
#   timing    OLD vs NEW: block-0 tune and the 2,500 block-0 refits; overfit task 0 (72 small models, 100 epochs)
#             at 12 vs 72 concurrent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=pf_gate5
#SBATCH --output=logs/pf_gate5_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
OLD=${OLD:-native_g4}
NEW=${NEW:-native_g5}
RUN=$ROOT/runs/gate5_$SLURM_JOB_ID
I=$ROOT/inputs
G=$I/gate4
G5=$I/gate5
PY=/usr/bin/python3
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$G5"/* "$ROOT/phase2_select.py" "$ROOT"/data/*/*; } > env.txt 2>&1
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
T id_new_linkid_refit ./ft_new run --jobs "$G5/refit500_link_identity.txt" --out id_new_linkid_refit --concurrent 48 --chunk 5 --pool
T id_new_wave_tune ./ft_new run --jobs "$TUNE" --out id_new_wave_tune --concurrent 48 --chunk 5
T id_new_wave_refit ./ft_new run --jobs refit500.txt --out id_new_wave_refit --concurrent 48 --chunk 5
T id_new_wave_fct ./ft_new run --jobs "$FCT" --out id_new_wave_fct --concurrent 6 --chunk 5
T id_new_tf32ns_refit ./ft_new run --jobs refit500.txt --out id_new_tf32ns_refit --concurrent 48 --chunk 5 --pool --tf32-ns
T id_new_tf32_refit ./ft_new run --jobs refit500.txt --out id_new_tf32_refit --concurrent 48 --chunk 5 --pool --tf32
# ---- guards (each must exit 2 with a message)
printf 'abcd' > tiny.f32
echo "case,rc,last_line" > guards.csv
for f in "$G"/bad_*.txt "$G5"/bad_*.txt; do
  b=$(basename "$f" .txt)
  ./ft_new run --jobs "$f" --out g_$b --pool > g_$b.log 2>&1
  echo "$b,$?,$(tail -1 g_$b.log)" >> guards.csv
done
./ft_new run --jobs "$G5/softplus_pack.txt" --out g_softplus_pack --pack 2 > g_softplus_pack.log 2>&1
echo "softplus_pack,$?,$(tail -1 g_softplus_pack.log)" >> guards.csv
./ft_new run --jobs "$G/stop_nopool.txt" --out g_stop_nopool > g_stop_nopool.log 2>&1
echo "stop_nopool,$?,$(tail -1 g_stop_nopool.log)" >> guards.csv
./ft_new run --jobs "$G/ok_small.txt" --out g_chunk0 --pool --chunk 0 > g_chunk0.log 2>&1
echo "chunk0,$?,$(tail -1 g_chunk0.log)" >> guards.csv
./ft_new run --jobs "$G/ok_small.txt" --out g_badopt --pool --fast > g_badopt.log 2>&1
echo "unknown_option,$?,$(tail -1 g_badopt.log)" >> guards.csv
./ft_new run --jobs "$G/ok_small.txt" --out g_ok --pool > g_ok.log 2>&1
echo "ok_small,$?,$(tail -1 g_ok.log)" >> guards.csv
# ---- stop_loss
T stop ./ft_new run --jobs "$G/stop.txt" --out stop --concurrent 2 --chunk 10 --pool
# ---- softplus
for l in mse huber qlike; do
  ./ft_new losscheck "$G5/losscheck_${l}_softplus.bin" "losscheck_${l}_softplus.out" softplus > losscheck_$l.log 2>&1
done
mkdir -p gradcheck
T gradcheck_softplus ./ft_new gradcheck --jobs "$G5/grad_softplus.txt" --out gradcheck
T sp_overfit ./ft_new run --jobs "$G5/overfit_softplus.txt" --out sp_overfit --concurrent 2 --chunk 50 --pool
for link in identity softplus; do
  if [ $link = softplus ]; then TF=$G5/tune_block0_softplus.txt; BJ=$G5/blocks_softplus.json; else TF=$TUNE; BJ=$I/phase2c/blocks.json; fi
  mkdir -p p2c_$link
  T p2c_${link}_tune ./ft_new run --jobs "$TF" --out p2c_$link/tune --concurrent 48 --chunk 5 --pool
  "$PY" "$ROOT/phase2_select.py" --blocks "$BJ" --k 0 --tune-out p2c_$link/tune --data-root "$ROOT/data" \
    --out p2c_$link/jobs_refit.txt --selection p2c_$link/selection.json > p2c_${link}_select.log 2>&1
  T p2c_${link}_refit ./ft_new run --jobs p2c_$link/jobs_refit.txt --out p2c_$link/refit --concurrent 48 --chunk 5 --pool
done
# ---- timing
for rep in 1 2; do
  T tT_old_r$rep ./ft_old run --jobs "$TUNE" --out tT_old_r$rep --concurrent 48 --chunk 5 --pool
  T tT_new_r$rep ./ft_new run --jobs "$TUNE" --out tT_new_r$rep --concurrent 48 --chunk 5 --pool
  T tR_old_r$rep ./ft_old run --jobs refit_all.txt --out tR_old_r$rep --concurrent 48 --chunk 5 --pool
  T tR_new_r$rep ./ft_new run --jobs refit_all.txt --out tR_new_r$rep --concurrent 48 --chunk 5 --pool
done
T t0_c12 ./ft_new run --jobs "$G5/task0_100ep.txt" --out t0_c12 --concurrent 12 --chunk 50 --pool
T t0_c72 ./ft_new run --jobs "$G5/task0_100ep.txt" --out t0_c72 --concurrent 72 --chunk 50 --pool
for d in tT_* tR_* t0_*; do
  [ -d "$d" ] && rm -f "$d/steps.csv" "$d/epochs.csv"
done
gzip -f id_*/steps.csv stop/steps.csv sp_overfit/steps.csv p2c_*/*/steps.csv 2>/dev/null
rm -f ft_old ft_new
date; echo GATE5_DONE
