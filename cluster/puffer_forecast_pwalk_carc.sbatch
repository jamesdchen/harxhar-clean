#!/bin/bash
# 2026-10-14 puffer_forecast POOLED walk-forward (experiments/puffer_forecast/pooled_walk.py): at each block start a
# tune on the trailing 24,000 half-hour bars (fit / embargo 48 / validation 2,400; early stopping), the
# configuration picked by pooled_select.py (seed-averaged validation MSE), then a refit every 21 sessions on the
# trailing 24,000 bars, each forecasting every bar up to the next refit.  INPUTS = inputs/pwalk_wd<x> (wd fixed)
# or inputs/pwalk_wdgrid (wd in the tune grid).  ONE GPU type for the whole study (GPU_TYPE, = --constraint).
# Each design's jobs run in their own pool (CA / CB slots: the all_features validation pass holds 2,400 x L x P
# inputs per slot).  BLOCKS limits the blocks (the calibration runs block 0 only).
#   cd $HOME/puffer_forecast
#   sbatch --export=NONE,GPU_TYPE=p100,SRC=native_g6,INPUTS=pwalk_wd0.1,TAG=wd0.1 --constraint=p100 puffer_forecast_pwalk_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=pf_pwalk
#SBATCH --output=logs/pf_pwalk_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
SRC=${SRC:-native_g6}
INPUTS=${INPUTS:?INPUTS=pwalk_wd<x> or pwalk_wdgrid}
TAG=${TAG:-$INPUTS}
BLOCKS=${BLOCKS:-0 1 2 3 4}
CA=${CA:-6}
CB=${CB:-12}
RUN=$ROOT/runs/pwalk_${TAG}_$SLURM_JOB_ID
IN=$ROOT/inputs/$INPUTS
PY=/usr/bin/python3
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$SRC"/*.cu* "$ROOT/pooled_select.py" "$IN"/plan.json "$IN"/tune_block*.txt "$ROOT"/data/pooled_walk_*/*; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE (p100 / v100 / a40 / a100 / l40s) is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; a40) M="A40" ;; a100) M="A100" ;; l40s) M="L40S" ;;
  *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CONS=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'Features=[^ ]*' | head -1 | cut -d= -f2)  # the job's --constraint
[ "$CONS" = "$GPU_TYPE" ] || { echo "--constraint '$CONS' != GPU_TYPE $GPU_TYPE"; exit 1; }
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT|constraint=$CONS" | tee gpu_type.txt >> env.txt
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
prebuilt "$SRC" forecast_train
BIN=$RUN/forecast_train
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 10 > gpu_mem.log 2>/dev/null &
MEMPID=$!
run_split() {  # run_split JOBS OUTDIR: each design's jobs in its own pool, outputs merged into OUTDIR
  local jobs=$1 out=$2 d c
  mkdir -p "$out"
  for d in baseline all_features; do
    c=$CB; [ $d = all_features ] && c=$CA
    grep "data=data/pooled_walk_${d} " "$jobs" > "$out.$d.txt" || true
    [ -s "$out.$d.txt" ] || continue
    "$BIN" run --jobs "$out.$d.txt" --out "${out}_$d" --concurrent "$c" --chunk 5 --pool > "${out}_$d.log" 2>&1
  done
  for f in jobs_out.csv preds.csv epochs.csv steps.csv; do
    head -1 "${out}_baseline/$f" > "$out/$f"
    for d in baseline all_features; do tail -n +2 "${out}_$d/$f" >> "$out/$f"; done
  done
  for d in baseline all_features; do
    [ -d "${out}_$d/params" ] && mkdir -p "$out/params" && mv "${out}_$d"/params/* "$out/params/"
    cp "${out}_$d/pool_summary.csv" "$out/pool_summary_$d.csv"
    rm -rf "${out}_$d"
  done
  gzip -f "$out/steps.csv"
}
SEL_EXTRA="--save-first-origin"
for k in $BLOCKS; do
  mkdir -p "$RUN/block${k}"
  echo "BLOCK $k tune $(date)"
  run_split "$IN/tune_block${k}.txt" "$RUN/block${k}/tune"
  "$PY" "$ROOT/pooled_select.py" --plan "$IN/plan.json" --k "$k" --tune-out "$RUN/block${k}/tune" \
    --data-root "$ROOT/data" --out "$RUN/block${k}/jobs_refit.txt" --selection "$RUN/block${k}/selection.json" $SEL_EXTRA
  echo "BLOCK $k refit $(date)"
  run_split "$RUN/block${k}/jobs_refit.txt" "$RUN/block${k}/refit"
done
kill $MEMPID 2>/dev/null || true
rm -f "$BIN"
date
echo PWALK_DONE
