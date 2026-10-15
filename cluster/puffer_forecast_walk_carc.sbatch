#!/bin/bash
# 2026-10-14 puffer_forecast walk-forward (Phase 2c inputs) on ONE required GPU type: 5 blocks, each re-tuned
# (the 12-configuration grid, 5 seeds) and selected with phase2_select.py, then a cold refit at every session.
# INPUTS names the tune files + blocks manifest: inputs/phase2c (link=identity, the 13082282 inputs) or
# inputs/walk_softplus (the same lines with link=softplus), or the weight-decay study's walk_softplus_wd<x> (wd fixed)
# / walk_softplus_wdgrid (wd in the tune grid; experiments/puffer_forecast/wd_study.py).  The trainer comes
# prebuilt (SRC; native_g6 for wd).  SAVE_FIRST=1: each block's first origin's refits save their weights.
#   cd $HOME/puffer_forecast
#   sbatch --export=NONE,GPU_TYPE=p100,SRC=native_g5b,INPUTS=phase2c,TAG=identity --constraint=p100 puffer_forecast_walk_carc.sbatch
#   sbatch --export=NONE,GPU_TYPE=p100,SRC=native_g5b,INPUTS=walk_softplus,TAG=softplus --constraint=p100 puffer_forecast_walk_carc.sbatch
# GPU_TYPE is REQUIRED (user, 2026-10-13: every NN number for the paper from ONE recorded GPU type) and must equal
# the --constraint; the job checks the GPU it got and records it in gpu_type.txt / env.txt (phase2_score.py
# refuses a run whose env files show more than one GPU model).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=pf_walk
#SBATCH --output=logs/pf_walk_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
SRC=${SRC:-native_g5b}
INPUTS=${INPUTS:?INPUTS=phase2c or walk_softplus}
TAG=${TAG:-$INPUTS}
RUN=$ROOT/runs/walk_${TAG}_$SLURM_JOB_ID
IN=$ROOT/inputs/$INPUTS
PY=/usr/bin/python3
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$SRC"/*.cu* "$ROOT/phase2_select.py" "$IN"/blocks.json "$IN"/tune_block*.txt "$ROOT"/data/*/*; } > env.txt 2>&1
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
# SAVE_FIRST=1 (2026-10-14, weight-decay study): each block's first origin's refits save their weights
SEL_EXTRA=""
[ "${SAVE_FIRST:-0}" = 1 ] && SEL_EXTRA="--save-first-origin"
for k in 0 1 2 3 4; do
  mkdir -p "$RUN/block${k}"
  echo "BLOCK $k tune $(date)"
  "$BIN" run --jobs "$IN/tune_block${k}.txt" --out "$RUN/block${k}/tune" --concurrent 48 --chunk 5 --pool
  "$PY" "$ROOT/phase2_select.py" --blocks "$IN/blocks.json" --k "$k" \
    --tune-out "$RUN/block${k}/tune" --data-root "$ROOT/data" \
    --out "$RUN/block${k}/jobs_refit.txt" --selection "$RUN/block${k}/selection.json" $SEL_EXTRA
  echo "BLOCK $k refit $(date)"
  "$BIN" run --jobs "$RUN/block${k}/jobs_refit.txt" --out "$RUN/block${k}/refit" --concurrent 48 --chunk 5 --pool
  gzip -f "$RUN/block${k}/tune/steps.csv" "$RUN/block${k}/refit/steps.csv"
done
rm -f "$BIN"
date
echo WALK_DONE
