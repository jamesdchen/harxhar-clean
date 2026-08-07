#!/bin/bash
#SBATCH --job-name=var_a0_ols_har_backbone
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-99
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/unif_%x_%A_%a.out
#
# LEVEL-CHANNEL CONTROL array (2026-08-07): a0_ols_har under --design backbone
# (HAR ladder + session-edge interactions + calendar ONLY). Paired against the
# exog sidecar it answers "does the exog state predict forecast uncertainty
# BEYOND the volatility level it proxies?".
#
# ONE array, ONE arm, ONE design. It does NOT touch — and cannot resubmit — the
# two exog arrays already running (a0_ols_har, blk4_trail_tuned under the
# default --design exog): those write results/variance_sidecar/<arm>/, this
# writes results/variance_sidecar/<arm>__backbone/, so the output roots are
# disjoint. jobs/slurm/variance_sidecar.sbatch is left untouched.
#
# Resource shape mirrors jobs/slurm/variance_sidecar.sbatch exactly (main,
# 6h, 4 cpus, 16G, array 0-99, same log path).
#
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean, either way
# (the script self-submits when invoked outside Slurm):
#   bash   jobs/submit/variance_sidecar_control.sh
#   sbatch jobs/submit/variance_sidecar_control.sh

set -u
ARM=${ARM:-a0_ols_har}
DESIGN=backbone

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  # ── submitter mode ──────────────────────────────────────────────────────────
  set -e
  cd /scratch1/jc_905/harxhar-clean
  mkdir -p logs
  [ -d "results/unification/$ARM" ] || {
    echo "ABORT: results/unification/$ARM absent"
    exit 1
  }
  sbatch --export=ALL,ARM="$ARM" jobs/submit/variance_sidecar_control.sh
  squeue -u "$USER" -h | wc -l
  exit 0
fi

# ── array-task mode ───────────────────────────────────────────────────────────
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
# Idempotent resume: an existing output for this (arm, design, chunk) is done.
# Burn-in / structurally-absent chunks exit 0 writing nothing and are simply
# recomputed (cheap) on a resubmit — same convention as the exog array.
OUT=$(printf 'results/variance_sidecar/%s__%s/chunk_%03d.npz' "$ARM" "$DESIGN" "$SLURM_ARRAY_TASK_ID")
[ -f "$OUT" ] && { echo "SKIP existing $OUT"; exit 0; }
$PY experiments/variance_sidecar.py --arm "$ARM" --chunk-index "$SLURM_ARRAY_TASK_ID" --design "$DESIGN"
