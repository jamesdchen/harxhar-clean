#!/bin/bash
# 2026-10-13 puffer_forecast gate 4c (final review build): native_of6/ = gate 4's native_of4/ plus the
# largest-first choice by each queue's remaining work (gate 4b) and the refit forecast collected at a later
# poll instead of a host-blocking cudaStreamSynchronize.  Same checks as gate 4 (native_of4/: correctness guards F2/F4/F6/F7/F3/F5 and the
# speed items -- slot-lifetime cuBLAS / stream state with stream-ordered shape buffers, per-slot fc_train
# buffer, pre-warmed initial weights, one-block-per-sequence gather, patience-aware tune chunks, largest-first
# queue; opt-in --tf32-ns / --tf32 / stop_loss) against the current binary (native_of3/, 0e59b80c), one GPU:
#   identity  old vs new --pool: Phase 2c block-0 tune (preds, best_epoch; new drops past-stop log rows),
#             500 block-0 refits, the fc_train jobs, a warm start (init_from); new --pool vs new wave;
#             new --tf32-ns / --tf32 on the 500 refits (no TF32 hardware on P100 / V100: identical expected)
#   guards    malformed job lines / options exit 2 with a message
#   stop_loss a 1,000-epoch overfit job with stop_loss=1e-2 vs without (prefix identical up to the stop)
#   timing    bench suite B (wave, K 8 / 24 / 48, 3 repeats) and the 2,500 block-0 refits (--pool K48):
#             old, new, new FT_GATHER_OLD=1, new CUDA_DEVICE_MAX_CONNECTIONS=32, new conn32 --chunk 200
#   nsys      500 refits, old and new, API summaries (the device-wide syncs)
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_gate4c_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=pf_gate4c
#SBATCH --output=logs/pf_gate4c_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
OLD=${OLD:-native_of3}
NEW=${NEW:-native_of6}
RUN=$ROOT/runs/gate4c_$SLURM_JOB_ID
I=$ROOT/inputs
G=$I/gate4
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
export TMPDIR=$RUN/tmp; mkdir -p "$TMPDIR"
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -1; git -C "$PUFFER" log -1 --format='%H %ci'
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$G"/*.txt "$ROOT"/data/*/*; } > env.txt 2>&1
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
T id_new_tf32ns_refit ./ft_new run --jobs refit500.txt --out id_new_tf32ns_refit --concurrent 48 --chunk 5 --pool --tf32-ns
T id_new_tf32_refit ./ft_new run --jobs refit500.txt --out id_new_tf32_refit --concurrent 48 --chunk 5 --pool --tf32
# ---- guards (each must exit 2 with a message)
printf 'abcd' > tiny.f32
echo "case,rc,last_line" > guards.csv
for f in "$G"/bad_*.txt; do
  b=$(basename "$f" .txt)
  ./ft_new run --jobs "$f" --out g_$b --pool > g_$b.log 2>&1
  echo "$b,$?,$(tail -1 g_$b.log)" >> guards.csv
done
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
# ---- timing (suite B: wave; pools: --pool K48)
for rep in 1 2 3; do
  for d in baseline all_features; do
    for K in 8 24 48; do
      f=$I/bench_B_${d}_K${K}.txt
      T tB_old_${d}_K${K}_r$rep ./ft_old run --jobs "$f" --out tB_old_${d}_K${K}_r$rep --concurrent $K
      T tB_new_${d}_K${K}_r$rep ./ft_new run --jobs "$f" --out tB_new_${d}_K${K}_r$rep --concurrent $K
      T tB_c32_${d}_K${K}_r$rep env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs "$f" --out tB_c32_${d}_K${K}_r$rep --concurrent $K
    done
  done
done
T tR_old_ch5 ./ft_old run --jobs refit_all.txt --out tR_old_ch5 --concurrent 48 --chunk 5 --pool
T tR_new_ch5 ./ft_new run --jobs refit_all.txt --out tR_new_ch5 --concurrent 48 --chunk 5 --pool
T tR_gold_ch5 env FT_GATHER_OLD=1 ./ft_new run --jobs refit_all.txt --out tR_gold_ch5 --concurrent 48 --chunk 5 --pool
for ch in 5 10 20; do
  T tR_c32_ch$ch env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs refit_all.txt --out tR_c32_ch$ch --concurrent 48 --chunk $ch --pool
done
T tT_old_ch5 ./ft_old run --jobs "$TUNE" --out tT_old_ch5 --concurrent 48 --chunk 5 --pool
T tT_new_ch5 ./ft_new run --jobs "$TUNE" --out tT_new_ch5 --concurrent 48 --chunk 5 --pool
for ch in 5 10 20; do
  T tT_c32_ch$ch env CUDA_DEVICE_MAX_CONNECTIONS=32 ./ft_new run --jobs "$TUNE" --out tT_c32_ch$ch --concurrent 48 --chunk $ch --pool
done
# ---- nsys (API summaries)
for v in old new; do
  nsys profile -t cuda,cublas,osrt -o nsys_$v --force-overwrite true ./ft_$v run --jobs refit500.txt --out nsys_${v}_out --concurrent 48 --chunk 5 --pool > nsys_$v.log 2>&1
  nsys stats --report cuda_api_sum --format csv -o nsys_$v nsys_$v.nsys-rep > /dev/null 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o nsys_$v nsys_$v.nsys-rep > /dev/null 2>&1
  nsys stats --report cuda_api_trace --format csv -o nsys_$v nsys_$v.nsys-rep > /dev/null 2>&1
  gzip -f nsys_${v}_cuda_api_trace.csv
done
# keep the identity / stop / guard runs whole; timing and profile runs: jobs_out, preds, pool_summary
for d in tB_* tR_* tT_* nsys_*_out; do
  [ -d "$d" ] && rm -f "$d/steps.csv" "$d/epochs.csv"
done
gzip -f id_*/steps.csv stop/steps.csv 2>/dev/null
rm -rf ft_old ft_new "$TMPDIR" nsys_*.sqlite
date; echo GATE4_DONE
