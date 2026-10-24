#!/bin/bash
# 2026-10-23 MinGRU v2: the CPU stages of experiments/pooled_mingru_v2.py on a window's CARC root.
#   MODE prep       G0: folds, Stage-1 inits (the old fold-3 tune inits recomputed and copied), TQ_0, Stage-2 per-fold
#                   ridge inits, kept / late-start columns -> inputs/g0.json, inputs/init_index_v2.csv
#   MODE barinits   L=<L*> SETS=<sets>: the G7 walk probe's 24 bar-origin ridge inits
#   MODE post       RUNDIR=<run dir> JOBS=<jobs file>: the POST step of a GPU run, rerun on a CPU node
# ROOT /home1/jc_905/pmf2 (W 3,000) or /home1/jc_905/pmf2_35 (W 3,500); XDATA the old run's data (read-only).
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --job-name=v2cpu
set -euo pipefail
ROOT=${ROOT:?ROOT}
case "$ROOT" in */pmf2_35) WD=3500; XD=/home1/jc_905/pmf35/data ;; */pmf2) WD=3000; XD=/home1/jc_905/pmf/data ;; *) echo "ROOT $ROOT"; exit 1 ;; esac
XDATA=${XDATA:-$XD}
PY=${PY:-/home1/jc_905/envs/harxhar-r13/bin/python}
EP=$(dirname "$(dirname "$PY")")
export OPENSSL_CONF=$EP/ssl/openssl.cnf OPENSSL_MODULES=$EP/lib/ossl-modules OPENSSL_ENGINES=$EP/lib/engines-3
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PMF_WINDOW_DAYS=$WD PMF_REPO=/home1/jc_905/pmf/repo
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
S=$ROOT/code/pooled_mingru_v2.py  # 2026-10-23: ROOT may be /scratch1/jc_905/pmf2(_35) (the /home1 quota)
echo "v2 W=$WD $MODE on $(hostname) $(date) $("$PY" -V)"; md5sum "$S"
case "${MODE:?MODE}" in
  prep) ALT=""; [ "$WD" = 3500 ] && ALT="--alt-rows /home1/jc_905/pmf/data/pmf_rows.csv"
        "$PY" "$S" prep --root "$ROOT" --xdata "$XDATA" --folds-json $ROOT/code/folds14.json $ALT ;;
  barinits) "$PY" "$S" bar-inits --root "$ROOT" --xdata "$XDATA" --L "${L:?L}" --sets "${SETS:-baseline,all_features}" ;;
  post) "$PY" "$S" post --run "${RUNDIR:?RUNDIR}" --jobs "${JOBS:?JOBS}" ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac
date; echo V2CPU_DONE
