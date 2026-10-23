#!/bin/bash
# 2026-10-22 final pooled MinGRU: the CPU stages of experiments/pooled_mingru_final_2014_w3000.py.
#   MODE prep       export (the 2014-start W = 3,000 design caches -> data/pmf_<set>/), plan, fold-3 ridge inits,
#                   round-0 tune jobs, canary jobs
#   MODE tunejobs   ROUND=<r>: the tune jobs of round r (after a select that extended a grid)
#   MODE select     ROUND=<r> [GPUH=<allocated tune P100-hours>]: picks, extension or frozen configuration
#   MODE walkinits  array task i of N (--array=0-<N-1>, PARTS=N): the walk origins' ridge inits
#   MODE walkjobs   the walk jobs files
#   MODE check      RUNDIR=<run dir>: check-run on one GPU run
# ROOT (default /home1/jc_905/pmf): code/ (the runner), data/, inputs/, runs/, out/, logs/.  PMF_REPO: the checkout
# whose src/ is imported (window_keep), the previous run's (/scratch1/jc_905/harxhar-trees13).  Python: the previous
# run's CARC env (harxhar-r13).
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=pmfcpu
#SBATCH --output=/home1/jc_905/pmf/logs/%x_%j.out
set -euo pipefail
WD=${WD:-3000}  # window days: 3000 (ROOT /home1/jc_905/pmf) or 3500 (ROOT /home1/jc_905/pmf35; sbatch -o its logs/)
if [ "$WD" = 3000 ]; then ROOT=${ROOT:-/home1/jc_905/pmf}; else ROOT=${ROOT:-/home1/jc_905/pmf$((WD / 100))}; fi
export PMF_WINDOW_DAYS=$WD
PY=${PY:-/scratch1/jc_905/envs/harxhar-r13/bin/python}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PMF_REPO=${PMF_REPO:-/scratch1/jc_905/harxhar-trees13}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
DESIGN=${DESIGN:-/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w$WD/_work}
LINRUN=${LINRUN:-/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w$WD}
S=$ROOT/code/pooled_mingru_final_2014_w3000.py
cd "$ROOT"
echo "pmf W=$WD $MODE on $(hostname) $(date) $("$PY" -V)"
md5sum "$S"
case "${MODE:?MODE}" in
  prep)
    "$PY" "$S" export --design "$DESIGN" --data data --prev-rows /home1/jc_905/pm13/data/pm13_rows.csv
    "$PY" "$S" plan --data data --inputs inputs --lin-run "$LINRUN"
    "$PY" "$S" inits-tune --data data --inputs inputs
    "$PY" "$S" tune-jobs --inputs inputs --round 0
    "$PY" "$S" canary --data data --inputs inputs ;;
  tunejobs)
    "$PY" "$S" tune-jobs --inputs inputs --round "${ROUND:?ROUND}" ;;
  select)
    G=""; [ -n "${GPUH:-}" ] && G="--tune-gpu-hours $GPUH"
    "$PY" "$S" select --data data --inputs inputs --runs runs --out out --round "${ROUND:?ROUND}" $G ;;
  walkinits)
    "$PY" "$S" walk-inits --data data --inputs inputs --out out --part "${SLURM_ARRAY_TASK_ID:?array}" --parts "${PARTS:?PARTS}" ;;
  walkjobs)
    "$PY" "$S" walk-jobs --inputs inputs --out out ;;
  check)
    "$PY" "$S" check-run --inputs inputs --run "${RUNDIR:?RUNDIR}" ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac
date; echo PMFCPU_DONE
