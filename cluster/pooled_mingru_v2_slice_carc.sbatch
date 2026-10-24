#!/bin/bash
# 2026-10-24 MinGRU v2 walk: the slice check before NOSTEPS=1 is used (coordinator: reduce the per-step logs only
# after a slice shows forecasts text-identical).  One GPU: the slice file (bar refits of both sets' first walk file)
# run twice by the same binary, once with the per-step log written (steps.csv) and once with it pointed at /dev/null;
# PASS iff preds.csv is text-identical (sorted) and every job is ok.  Exit 1 on a mismatch (the walk files wait
# afterok on this job).
#   ROOT (W 3,000 walk root), JOBS (inputs/<slice file>), CONC, GPUTYPE; data staged as the GPU sbatch.
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:40:00
#SBATCH --job-name=v2slice
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=${ROOT:?ROOT}; JOBS=${JOBS:?JOBS}; CONC=${CONC:-12}; GPUTYPE=${GPUTYPE:-P100}
case "$ROOT" in */pmf2_35) XD=/home1/jc_905/pmf35/data ;; */pmf2) XD=/home1/jc_905/pmf/data ;; *) echo "ROOT $ROOT"; exit 1 ;; esac
RUN=$ROOT/runs/v2slice_$SLURM_JOB_ID
LOCAL=${TMPDIR:-/tmp}/v2slice_$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT
mkdir -p "$LOCAL/data/pmf_lin_init" "$RUN"
for s in baseline all_features; do
  mkdir -p "$LOCAL/data/pmf_$s" "$LOCAL/data/pmf_lin_init/$s"
  for f in X.f32 X.f64 y.f32 enc_pf.f64 meta.txt columns.txt; do cp "$XD/pmf_$s/$f" "$LOCAL/data/pmf_$s/"; done
  cp -r "$ROOT/data/pmf_lin_init/$s/w000" "$LOCAL/data/pmf_lin_init/$s/"
done
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(hostname) $GOT" | tee "$RUN/env.txt"
echo "$GOT" | grep -qi "$GPUTYPE" || { echo "got a $GOT, $GPUTYPE required"; exit 1; }
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
cp "$ROOT/inputs/$JOBS" "$LOCAL/jobs.txt"
cd "$LOCAL"
"$LOCAL/ft" run --jobs jobs.txt --out "$LOCAL/a" --concurrent "$CONC" --chunk 5 --pool > a.log 2>&1
mkdir -p "$LOCAL/b"; ln -sfn /dev/null "$LOCAL/b/steps.csv"
"$LOCAL/ft" run --jobs jobs.txt --out "$LOCAL/b" --concurrent "$CONC" --chunk 5 --pool > b.log 2>&1
rm -f "$LOCAL/b/steps.csv"
for k in a b; do cp "$LOCAL/$k/preds.csv" "$RUN/preds_$k.csv"; cp "$LOCAL/$k/jobs_out.csv" "$RUN/jobs_out_$k.csv"; cp "$k.log" "$RUN/"; done
gzip -c "$LOCAL/a/steps.csv" > "$RUN/steps_a.csv.gz"
NJ=$(wc -l < jobs.txt); NOK=$(grep -c ",ok," "$LOCAL/b/jobs_out.csv" || true)
if cmp -s <(sort "$LOCAL/a/preds.csv") <(sort "$LOCAL/b/preds.csv") && [ "$NOK" = "$NJ" ]; then
  echo "SLICE PASS: preds.csv text-identical with and without the step log ($NJ jobs, $(($(wc -l < "$LOCAL/a/preds.csv") - 1)) forecast rows, all ok)" | tee "$RUN/verdict.txt"
else
  echo "SLICE FAIL: preds differ or jobs not ok (ok $NOK of $NJ)" | tee "$RUN/verdict.txt"; exit 1
fi
date; echo V2SLICE_DONE
