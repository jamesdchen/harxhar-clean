#!/bin/bash
# 2026-10-15 puffer_forecast gate 7: encoder-input decay (job key wd_enc) -- NEW = native_g7/ = native_g6/ plus
# W_enc[:, c] <- W_enc[:, c] (1 - lr_t wd_enc pf[c]) on the encoder's input columns before each Muon step (pf from
# <data>/enc_pf.f64 through the window's keep set; wd_enc = 0 launches nothing) and a pool slot whose new shape
# does not fit next to the running slots waits for a running job instead of exiting.  OLD = native_g6/.  Both prebuilt
# (P100, the walks' type):
#   cd $HOME/puffer_forecast
#   B=$(sbatch --parsable --export=NONE,SRC=native_g7 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE,GPU_TYPE=p100 --constraint=p100 --dependency=afterok:$B puffer_forecast_gate7_carc.sbatch
# One GPU, CUDA_DEVICE_MAX_CONNECTIONS=32:
#   identity  OLD vs NEW --pool (wd_enc absent): the softplus walk's block-0 tune (Phase 2c, 120 jobs), 500 block-0
#             refits of the wd study's wd = 0 softplus walk, the fc_train jobs; NEW pool vs wave; NEW with wd_enc=0
#   guards    wd_enc < 0, nan, no pf file, pf length != columns, a negative / NaN factor, wd_enc > 0 under --pack
#   one step  batch >= N (one Muon step), wd_enc 0 / > 0 / with wd, + gradcheck dumps (float64 reference locally)
#   zero grad max_grad_norm = 0 (Muon's update 0): 40 epochs of decay alone on the toy (pf 1 / 0.3 / 0.1 / 0.01 / 0 / 2)
#   norms     the toy with its gradient, 40 epochs, wd_enc 0 .. 10
#   pf map    refits (save_params) on 8 windows x 2 designs: the factors used, column for column
#   calib     block-0 window, both designs, L {5, 20} x H {16, 64} at lr 0.03, 10 epochs, wd_enc 0 .. 30, 2 seeds
#   trim      the pooled walk's wd-grid block-0 tune that ran out of memory in 13137036 (one pool per design:
#             baseline 12 slots, all_features 6) with NEW; then OLD once more on its baseline part
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=pf_gate7
#SBATCH --output=logs/pf_gate7_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
OLD=${OLD:-native_g6}
NEW=${NEW:-native_g7}
RUN=$ROOT/runs/gate7_$SLURM_JOB_ID
I=$ROOT/inputs
G7=$I/gate7
WREF=$ROOT/runs/walk_wd0_13135336  # the wd study's wd = 0 softplus walk (native_g6, P100)
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$G7"/* "$I"/wd_enc_calib/calib.txt "$ROOT"/data/*/enc_pf.f64; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; a40) M="A40" ;; a100) M="A100" ;; l40s) M="L40S" ;;
  *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT" | tee gpu_type.txt >> env.txt
cat env.txt
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
awk 'NR <= 500' "$WREF/block0/jobs_refit.txt" > refit500.txt
TUNE=$I/walk_softplus/tune_block0.txt
FCT=$I/overfit_gate/fctrain_jobs.txt
sed 's/$/ wd_enc=0/' "$TUNE" > tune_we0.txt
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 5 > gpu_mem.log 2>/dev/null &
MEMPID=$!
echo "name,rc,t0,t1" > rc.csv
R() { local n=$1 a; shift; a=$(date +%s); "$@" > "$n.log" 2>&1; echo "$n,$?,$a,$(date +%s)" >> rc.csv; }
# ---- identity
for v in old new; do
  R id_${v}_tune ./ft_$v run --jobs "$TUNE" --out id_${v}_tune --concurrent 48 --chunk 5 --pool
  R id_${v}_refit ./ft_$v run --jobs refit500.txt --out id_${v}_refit --concurrent 48 --chunk 5 --pool
  R id_${v}_fct ./ft_$v run --jobs "$FCT" --out id_${v}_fct --concurrent 6 --chunk 5 --pool
done
R id_new_wave_tune ./ft_new run --jobs "$TUNE" --out id_new_wave_tune --concurrent 48 --chunk 5
R id_new_wave_refit ./ft_new run --jobs refit500.txt --out id_new_wave_refit --concurrent 48 --chunk 5
R id_new_we0_tune ./ft_new run --jobs tune_we0.txt --out id_new_we0_tune --concurrent 48 --chunk 5 --pool
# ---- guards
echo "case,rc,last_line" > guards.csv
for f in "$G7"/bad_we_*.txt; do
  b=$(basename "$f" .txt)
  ./ft_new run --jobs "$f" --out g_$b --pool > g_$b.log 2>&1
  echo "$b,$?,$(tail -1 g_$b.log)" >> guards.csv
done
./ft_new run --jobs "$G7/we_pack.txt" --out g_we_pack --pack 2 > g_we_pack.log 2>&1
echo "we_pack,$?,$(tail -1 g_we_pack.log)" >> guards.csv
./ft_new run --jobs "$G7/ok_small.txt" --out g_ok --pool > g_ok.log 2>&1
echo "ok_small,$?,$(tail -1 g_ok.log)" >> guards.csv
# ---- one step, zero gradient, norms, pf mapping, calibration
R step ./ft_new run --jobs "$G7/step.txt" --out step --concurrent 5 --chunk 1 --pool
mkdir -p gradcheck
R gradcheck ./ft_new gradcheck --jobs "$G7/step_grad.txt" --out gradcheck
R zerograd ./ft_new run --jobs "$G7/zerograd.txt" --out zerograd --concurrent 4 --chunk 5 --pool
R norm ./ft_new run --jobs "$G7/norm_toy.txt" --out norm --concurrent 5 --chunk 5 --pool
R pfmap ./ft_new run --jobs "$G7/pfmap.txt" --out pfmap --concurrent 16 --chunk 1 --pool
R calib ./ft_new run --jobs "$I/wd_enc_calib/calib.txt" --out calib --concurrent 48 --chunk 5 --pool
# ---- trim: the pooled wd-grid block-0 tune (13137036's OOM), NEW per design; then OLD on its baseline part
for d in baseline all_features; do
  grep "data=data/pooled_walk_${d} " "$I/pwalk_wdgrid/tune_block0.txt" > trim_$d.txt
done
R trim_new_baseline ./ft_new run --jobs trim_baseline.txt --out trim_new_baseline --concurrent 12 --chunk 5 --pool
R trim_new_all_features ./ft_new run --jobs trim_all_features.txt --out trim_new_all_features --concurrent 6 --chunk 5 --pool
R trim_old_baseline ./ft_old run --jobs trim_baseline.txt --out trim_old_baseline --concurrent 12 --chunk 5 --pool
kill $MEMPID 2>/dev/null || true
gzip -f id_*/steps.csv step/steps.csv zerograd/steps.csv norm/steps.csv pfmap/steps.csv calib/steps.csv \
  trim_*/steps.csv 2>/dev/null
rm -f ft_old ft_new gradcheck/*.obs.f32 gradcheck/*.dec.f32 gradcheck/*.Z.f32
date; echo GATE7_DONE
