#!/bin/bash
# 2026-10-15 puffer_forecast pooled walk, sharded (the pooled ridge init): one block of one design per walk, its tune
# grid split over several GPU jobs.
#   MODE=tune    run one shard of a block's tune jobs (JOBS=inputs/...), one design, the pool at CONC slots
#   MODE=select  merge the shards' tune outputs (SHARDS="jobid jobid ..."), pooled_select (SEL_CRITERION, plan
#                INPUTS/plan.json) and the block's refits
#   cd $HOME/puffer_forecast
#   sbatch --export=NONE,MODE=tune,GPU_TYPE=p100,SRC=native_g9,JOBS=inputs/X/tune_block0.s0.txt,TAG=..,CONC=6 \
#          --constraint=p100 puffer_forecast_pshard_carc.sbatch
#   sbatch --export=NONE,MODE=select,GPU_TYPE=p100,SRC=native_g9,INPUTS=X,K=0,SHARDS="1 2 3",TAG=..,CONC=6 \
#          --constraint=p100 --dependency=afterok:1:2:3 puffer_forecast_pshard_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=pf_pshard
#SBATCH --output=logs/pf_pshard_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=$HOME/puffer_forecast
SRC=${SRC:-native_g9}
MODE=${MODE:?MODE=tune or select}
TAG=${TAG:?TAG}
CONC=${CONC:-6}
RUN=$ROOT/runs/${TAG}_$SLURM_JOB_ID
PY=/usr/bin/python3
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$SRC"/*.cu* "$ROOT/pooled_select.py"; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CONS=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'Features=[^ ]*' | head -1 | cut -d= -f2)
[ "$CONS" = "$GPU_TYPE" ] || { echo "--constraint '$CONS' != GPU_TYPE $GPU_TYPE"; exit 1; }
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT|constraint=$CONS" | tee gpu_type.txt >> env.txt
b=$ROOT/bin/$SRC
grep -q "$(md5sum < "$ROOT/$SRC/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" || { echo "stale $b"; exit 1; }
cp "$b/forecast_train" ft; cat "$b/provenance.txt" >> env.txt
if [ "$MODE" = tune ]; then
  JOBS=$ROOT/${JOBS:?JOBS}
  md5sum "$JOBS" >> env.txt
  ./ft run --jobs "$JOBS" --out tune --concurrent "$CONC" --chunk 5 --pool > tune.log 2>&1
  gzip -f tune/steps.csv
else
  IN=$ROOT/inputs/${INPUTS:?INPUTS}
  K=${K:?K}
  md5sum "$IN/plan.json" >> env.txt
  mkdir -p tune
  first=1
  for s in ${SHARDS:?SHARDS}; do
    d=$(ls -d "$ROOT"/runs/*_"$s")
    echo "shard $s $d" >> env.txt
    for f in jobs_out.csv preds.csv epochs.csv; do
      if [ $first = 1 ]; then head -1 "$d/tune/$f" > "tune/$f"; fi
      tail -n +2 "$d/tune/$f" >> "tune/$f"
    done
    first=0
  done
  SEL="--save-first-origin"
  [ -n "${SEL_CRITERION:-}" ] && SEL="$SEL --criterion $SEL_CRITERION"
  "$PY" "$ROOT/pooled_select.py" --plan "$IN/plan.json" --k "$K" --tune-out tune --data-root "$ROOT/data" \
    --out jobs_refit.txt --selection selection.json $SEL
  ./ft run --jobs jobs_refit.txt --out refit --concurrent "$CONC" --chunk 5 --pool > refit.log 2>&1
  gzip -f refit/steps.csv
fi
rm -f ft
date; echo PSHARD_DONE
