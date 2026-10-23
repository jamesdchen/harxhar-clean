#!/bin/bash
# 2026-10-23 final pooled MinGRU (coordinator, packing): do TUNE shards give text-identical forecasts at a higher
# concurrency, and run faster?  On one P100: a completed W = 3,000 round-0 shard per set rerun at higher concurrencies
# (HAR tune_baseline_r0.0, stored at CONC 12 -> 24, 48; full tune_all_features_r0.0, stored at CONC 6 -> 12, 24); each
# rerun's preds.csv compared, sorted, with the stored run's; pool-phase times printed.  Outputs in
# /home1/jc_905/pmf/conctest/<tag>_<job>/.
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --job-name=pmfconc
#SBATCH --output=/home1/jc_905/pmf/logs/%x_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
R=/home1/jc_905/pmf
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(hostname) $GOT"; echo "$GOT" | grep -q P100 || exit 1
LOCAL=${TMPDIR:-/tmp}/pmfconc_$SLURM_JOB_ID
mkdir -p "$LOCAL/data/pmf_lin_init"; trap 'rm -rf "$LOCAL"' EXIT
cp /home1/jc_905/pmf/bin/forecast_train "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || exit 1
for s in baseline all_features; do
  cp -r "$R/data/pmf_$s" "$LOCAL/data/"; mkdir -p "$LOCAL/data/pmf_lin_init/$s"
  find "$R/data/pmf_lin_init/$s" -maxdepth 1 -type f -exec cp -t "$LOCAL/data/pmf_lin_init/$s/" {} +
done
O=$R/conctest; mkdir -p "$O"; cd "$LOCAL"
for spec in baseline:24 baseline:48 all_features:12 all_features:24; do
  s=${spec%%:*}; c=${spec##*:}; f=tune_${s}_r0.0.txt
  ref=$(ls -d $R/runs/tune_${s}_r0_0_* | head -1)
  T0=$(date +%s%N)
  "$LOCAL/ft" run --jobs "$R/inputs/$f" --out "$LOCAL/${s}_c$c" --concurrent "$c" --chunk 5 --pool > "$LOCAL/${s}_c$c.log" 2>&1
  MS=$(( ($(date +%s%N) - T0) / 1000000 ))
  D=$O/${s}_c${c}_$SLURM_JOB_ID; mkdir -p "$D"; cp "$LOCAL/${s}_c$c/jobs_out.csv" "$LOCAL/${s}_c$c.log" "$D/"
  if cmp -s <(sort "$LOCAL/${s}_c$c/preds.csv") <(sort "$ref/tune/preds.csv"); then same=IDENTICAL; else same=DIFFERENT; fi
  echo "$s conc $c wall ${MS} ms | $(grep -h "pool phase" "$LOCAL/${s}_c$c.log" | sed -E 's/.*(pool phase.*)/\1/') | vs stored ($(grep -h "pool phase" $ref/tune.log | sed -E 's/.*in ([0-9.]+) s.*/\1 s/')): $same"
  rm -rf "$LOCAL/${s}_c$c"
done
date; echo PMFCONC_DONE
