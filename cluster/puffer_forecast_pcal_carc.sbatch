#!/bin/bash
# 2026-10-15 puffer_forecast pooled ridge init: block-0 lr_lin drift calibration under QLIKE for one design
# (inputs/pcal/{tune,refit}_<DESIGN>.txt) and, with MERGE_TEST=1, the shard-merge test: a small tune run whole and
# as two shards, merged, pooled_select on both (selection.json and the refit jobs must be byte-identical).
#   sbatch --export=NONE,GPU_TYPE=p100,SRC=native_g9,DESIGN=baseline,CONC=12,MERGE_TEST=1 --constraint=p100 \
#          puffer_forecast_pcal_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=pf_pcal
#SBATCH --output=logs/pf_pcal_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=$HOME/puffer_forecast
SRC=${SRC:-native_g9}
DESIGN=${DESIGN:?DESIGN}
CONC=${CONC:-6}
RUN=$ROOT/runs/${PCAL:-pcal}_${DESIGN}_$SLURM_JOB_ID
I=$ROOT/inputs/${PCAL:-pcal}
PY=/usr/bin/python3
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$SRC"/*.cu* "$I"/*; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE is required"}
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$GOT" | grep -q "P100" || { echo "got a $GOT, P100 required"; exit 1; }
echo "$GPU_TYPE|$GOT" | tee gpu_type.txt >> env.txt
b=$ROOT/bin/$SRC
grep -q "$(md5sum < "$ROOT/$SRC/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" || { echo "stale $b"; exit 1; }
cp "$b/forecast_train" ft; cat "$b/provenance.txt" >> env.txt
echo "name,rc,t0,t1" > rc.csv
R() { local n=$1 a; shift; a=$(date +%s); "$@" > "$n.log" 2>&1; echo "$n,$?,$a,$(date +%s)" >> rc.csv; }
R tune ./ft run --jobs "$I/tune_$DESIGN.txt" --out tune --concurrent "$CONC" --chunk 5 --pool
R refit ./ft run --jobs "$I/refit_$DESIGN.txt" --out refit --concurrent "$CONC" --chunk 5 --pool
if [ "${MERGE_TEST:-0}" = 1 ]; then
  R m_whole ./ft run --jobs "$I/mtest_all.txt" --out m_whole --concurrent "$CONC" --chunk 5 --pool
  R m_s0 ./ft run --jobs "$I/mtest_s0.txt" --out m_s0 --concurrent "$CONC" --chunk 5 --pool
  R m_s1 ./ft run --jobs "$I/mtest_s1.txt" --out m_s1 --concurrent "$CONC" --chunk 5 --pool
  mkdir -p m_merged
  for f in jobs_out.csv preds.csv epochs.csv; do
    head -1 m_s0/$f > m_merged/$f; tail -n +2 m_s0/$f >> m_merged/$f; tail -n +2 m_s1/$f >> m_merged/$f
  done
  for v in whole merged; do
    "$PY" "$ROOT/pooled_select.py" --plan "$I/mtest_plan.json" --k 0 --tune-out m_$v --data-root "$ROOT/data" \
      --out m_$v.refit.txt --selection m_$v.sel.json --save-first-origin --criterion qlike > m_$v.select.log 2>&1
  done
  { cmp m_whole.sel.json m_merged.sel.json && cmp m_whole.refit.txt m_merged.refit.txt && echo MERGE_TEST_PASS; } \
    > merge_test.txt 2>&1 || echo MERGE_TEST_FAIL >> merge_test.txt
  cat merge_test.txt
fi
gzip -f */steps.csv 2>/dev/null
rm -f ft
date; echo PCAL_DONE
