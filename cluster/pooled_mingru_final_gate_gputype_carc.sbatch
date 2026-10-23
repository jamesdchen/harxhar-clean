#!/bin/bash
# 2026-10-23 final pooled MinGRU: the gate on another GPU type (owner: speed up; the P100 queue is full).  On one GPU
# of type GPUTYPE (submit with --constraint=<feature>): the previous w1000 frozen configurations on the previous design
# (/home1/jc_905/pm13/data), the same jobs files as the P100 gate (inputs/gate_tune_prev.txt: fold-3 validation, 3 seeds
# a set; inputs/gate_walk_prev.txt: the first walk origin, 3 seeds a set), the same binary (md5 44ab6fff, built for
# sm_60/70/80/86/89/90) and concurrency (6); then the tune file a second time (self-consistency on this type).
# Checks (experiments/pooled_mingru_final_gate.py): text-identical to the stored P100 runs; tune vs tune2 identical.
#   sbatch --constraint=v100 --export=NONE,GPUTYPE=V100 pooled_mingru_final_gate_gputype_carc.sbatch
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:40:00
#SBATCH --job-name=pmfgt
#SBATCH --output=/home1/jc_905/pmf/logs/%x_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
GPUTYPE=${GPUTYPE:?GPUTYPE e.g. V100 A40 A100 L40S}
ROOT=/home1/jc_905/pmf
DATA=/home1/jc_905/pm13/data
PY=/home1/jc_905/envs/harxhar-r13/bin/python
export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(hostname) $GOT"
echo "$GOT" | grep -qi "$GPUTYPE" || { echo "got a $GOT, $GPUTYPE required"; exit 1; }
LOCAL=${TMPDIR:-/tmp}/pmfgt_$SLURM_JOB_ID
mkdir -p "$LOCAL/data/pm13_lin_init"
trap 'rm -rf "$LOCAL"' EXIT
for s in baseline all_features; do
  cp -r "$DATA/pm13_$s" "$LOCAL/data/"
  cp -r "$DATA/pm13_lin_init/$s" "$LOCAL/data/pm13_lin_init/"
done
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
T=$(echo "$GPUTYPE" | tr 'A-Z' 'a-z')
cd "$LOCAL"
for k in tune refit tune2; do
  case $k in tune|tune2) J=gate_tune_prev.txt; SUB=tune ;; refit) J=gate_walk_prev.txt; SUB=refit ;; esac
  RUN=$ROOT/runs/gate_${T}_${k}_$SLURM_JOB_ID
  T0=$(date +%s%N)
  "$LOCAL/ft" run --jobs "$ROOT/inputs/$J" --out "$LOCAL/$k/$SUB" --concurrent 6 --chunk 5 --pool > "$LOCAL/$k.log" 2>&1
  MS=$(( ($(date +%s%N) - T0) / 1000000 ))
  [ -f "$LOCAL/$k/$SUB/steps.csv" ] && gzip -f "$LOCAL/$k/$SUB/steps.csv"
  mkdir -p "$RUN"; cp -r "$LOCAL/$k/$SUB" "$RUN/"; cp "$LOCAL/$k.log" "$RUN/$SUB.log"
  { hostname; echo "$GOT"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo "wall_ms $MS"; } > "$RUN/env.txt"
  echo "$k: $MS ms"
done
set +e
for k in tune refit; do
  "$PY" "$ROOT/code/pooled_mingru_final_gate.py" "$ROOT/runs/gate_${T}_${k}_$SLURM_JOB_ID"; echo "gate $k exit $?"
done
A=$ROOT/runs/gate_${T}_tune_$SLURM_JOB_ID/tune/preds.csv; B=$ROOT/runs/gate_${T}_tune2_$SLURM_JOB_ID/tune/preds.csv
if cmp -s <(sort "$A") <(sort "$B"); then echo "SELF-CONSISTENT ($GPUTYPE: tune twice, preds identical)"; else echo "NOT SELF-CONSISTENT ($GPUTYPE)"; fi
date; echo PMFGT_DONE
