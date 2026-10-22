#!/bin/bash
# 2026-10-22 pooled ridge-init MinGRU on the 13-bar design: the CPU stages of experiments/pooled_mingru_rerun_13.py.
#   MODE prep    export (the linear rerun's CARC design caches -> the trainer's data dirs), plan, ridge inits, tune jobs
#   MODE select  the frozen configuration from the 4 folds' tune runs, the walk jobs files
#   MODE check   CHECK=<script> runs one check script under $ROOT/code with the env
# ROOT (default /home1/jc_905/pm13; it was /scratch1/jc_905/pm13 until 2026-10-22 08:45 PDT, when Lustre client errors on several GPU nodes failed four jobs): code/ (this script's experiments/ copy, results/<linear run>/frozen_penalties
# .json), data/, inputs/, runs/, out/.  PM13_REPO: the repo checkout whose src/ is imported (window_keep), the trees'
# CARC checkout /scratch1/jc_905/harxhar-trees13 (db59af24).  Python: the CARC env of the trees' run (harxhar-r13).
#   sbatch --export=NONE,MODE=prep pooled_mingru13_cpu_carc.sbatch
#   sbatch --export=NONE,MODE=select --dependency=afterok:<tune jobs> pooled_mingru13_cpu_carc.sbatch
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=pm13cpu
#SBATCH --output=/home1/jc_905/pm13/logs/%x_%j.out
set -euo pipefail
ROOT=${ROOT:-/home1/jc_905/pm13}
PY=${PY:-/scratch1/jc_905/envs/harxhar-r13/bin/python}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PM13_REPO=${PM13_REPO:-/scratch1/jc_905/harxhar-trees13}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
DESIGN=${DESIGN:-/scratch1/jc_905/harxhar-trees13/results/pooled_linear_rerun_2026-10-19_13_w1000/_work}
S=$ROOT/code/experiments/pooled_mingru_rerun_13.py
cd "$ROOT"
echo "pm13 $MODE on $(hostname) $(date) $("$PY" -V)"
md5sum "$S"
case "${MODE:?MODE}" in
  prep)
    "$PY" "$S" export --design "$DESIGN" --data data
    "$PY" "$S" plan --data data --inputs inputs
    "$PY" "$S" inits --data data --inputs inputs
    "$PY" "$S" tune-jobs --inputs inputs ;;
  select)
    "$PY" "$S" select --data data --inputs inputs --runs runs --out out --parts "${PARTS:-6}" ;;
  check)
    "$PY" "$ROOT/code/${CHECK:?CHECK}" ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac
date; echo PM13CPU_DONE
