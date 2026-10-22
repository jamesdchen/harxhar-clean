#!/bin/bash
# 2026-10-22 pooled ridge-init MinGRU on the 13-bar design (experiments/pooled_mingru_rerun_13.py): one GPU job runs
# one jobs file of the native_g9 trainer (the original pooled study's binary, bin/native_g9/forecast_train md5
# 44ab6fff, .cu af1964d0) in pool mode, as cluster/puffer_forecast_pshard_carc.sbatch MODE=tune did.
#   ROOT (default /home1/jc_905/pm13; it was /scratch1/jc_905/pm13 until 2026-10-22 08:45 PDT, when Lustre client errors on several GPU nodes failed four jobs): data/ (pm13_<set>/, pm13_lin_init/), inputs/ (jobs files), bin/, runs/
#   JOBS   a file under $ROOT/inputs (tune_<set>_f<k>.s<i>.txt or walk_<set>.p<i>.txt)
#   SUB    tune | refit (the output directory of the run; tune jobs -> tune/, refits -> refit/)
#   TAG    run name; the run directory is $ROOT/runs/${TAG}_$SLURM_JOB_ID
#   CONC   concurrent models (the original: 6 on all features, 12 on the HAR benchmark)
# Submit (CARC Discovery, from $ROOT):
#   sbatch --export=NONE,JOBS=tune_all_features_f0.s0.txt,SUB=tune,TAG=tune_all_features_f0_s0,CONC=6 \
#          pooled_mingru13_gpu_carc.sbatch
# GPU type pinned to the original's (P100, --constraint=p100); the job exits if it lands on another model.
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --job-name=pm13
#SBATCH --output=/home1/jc_905/pm13/logs/%x_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=${ROOT:-/home1/jc_905/pm13}
JOBS=${JOBS:?JOBS}
SUB=${SUB:?SUB tune or refit}
TAG=${TAG:?TAG}
CONC=${CONC:-6}
RUN=$ROOT/runs/${TAG}_$SLURM_JOB_ID
# 2026-10-22: the jobs file's design and its ridge inits, and the binary, staged on the node's local disk (two tune
# shards died on /scratch1 reads: 13303119 "bad arrays in data/pm13_all_features", 13303123 a bus error)
case "$JOBS" in *all_features*) SET=all_features ;; *baseline*) SET=baseline ;; *) SET="baseline all_features" ;; esac
LOCAL=${TMPDIR:-/tmp}/pm13_$SLURM_JOB_ID
mkdir -p "$LOCAL/data/pm13_lin_init"
trap 'rm -rf "$LOCAL"' EXIT
for s in $SET; do
  cp -r "$ROOT/data/pm13_$s" "$LOCAL/data/"
  cp -r "$ROOT/data/pm13_lin_init/$s" "$LOCAL/data/pm13_lin_init/"
  for f in X.f32 X.f64 y.f32 enc_pf.f64 meta.txt; do
    [ "$(stat -c %s "$ROOT/data/pm13_$s/$f")" = "$(stat -c %s "$LOCAL/data/pm13_$s/$f")" ] || { echo "stage $s/$f"; exit 1; }
  done
  [ "$(ls "$ROOT/data/pm13_lin_init/$s" | wc -l)" = "$(ls "$LOCAL/data/pm13_lin_init/$s" | wc -l)" ] || { echo "stage inits $s"; exit 1; }
done
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$LOCAL/data" data
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/bin/forecast_train" "$ROOT/inputs/$JOBS"; cat "$ROOT/bin/provenance.txt"; } > env.txt 2>&1
echo "$GOT" | grep -q "P100" || { echo "got a $GOT, P100 required"; exit 1; }
echo "p100|$GOT" > gpu_type.txt
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
cp "$ROOT/inputs/$JOBS" "$LOCAL/jobs.txt"
T0=$(date +%s)
# 2026-10-22: outputs written on the local disk, then copied to $RUN and checked (13303120 finished its 432 jobs but
# its pool_summary.csv was empty and the job failed after the pool phase: a write on /scratch1)
cd "$LOCAL"
"$LOCAL/ft" run --jobs "$LOCAL/jobs.txt" --out "$LOCAL/$SUB" --concurrent "$CONC" --chunk 5 --pool > "$LOCAL/$SUB.log" 2>&1
[ -f "$LOCAL/$SUB/steps.csv" ] && gzip -f "$LOCAL/$SUB/steps.csv"
(cd "$LOCAL" && find "$SUB" -type f -exec md5sum {} + | sort -k2 > "$SUB.md5")
cd "$RUN"
cp -r "$LOCAL/$SUB" "$LOCAL/$SUB.log" "$LOCAL/$SUB.md5" "$RUN/"
md5sum -c --quiet "$SUB.md5" || { echo "copy of $SUB to $RUN does not match"; exit 1; }
echo "wall_s $(( $(date +%s) - T0 ))" >> env.txt
rm -f data
date; echo PM13_DONE
