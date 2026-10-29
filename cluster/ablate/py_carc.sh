#!/bin/bash
# 2026-10-28 addendum J (e): one python command of the ablation run on USC CARC Discovery, the counterpart of py.sh
# (Hoffman2's container) for pool5.sh / tgate_carc.sh: the fomcfix trees' T1-gated environment
# /home1/jc_905/trees_final/venv_skx (lightgbm 4.6.0, xgboost 3.2.0; results/fomcfix_2026-10-26/RULE.txt section 6) with
# the variables of results/fomcfix_2026-10-26/code/tree_task.sh, the payload that produced forecasts_gb.parquet: threads
# = $NT (OMP / MKL / OpenBLAS / numba), no user site, cwd = the code tree, PYTHONPATH = the code tree and its
# experiments/.  usage: NT=<threads> py_carc.sh <gb|skx> <script> [args...]   env: ABL_B (staging holding code/),
# CARC_PY (python; default the venv above)
set -euo pipefail
B=${ABL_B:?ABL_B must be the staging directory}
NT=${NT:-1}
PY=${CARC_PY:-/home1/jc_905/trees_final/venv_skx/bin/python}
case ${1:-} in
  gb | skx) ;;
  *) echo "py_carc.sh: unknown env ${1:-}"; exit 2 ;;
esac
shift
cd "$B/code"
if [ -f "$(dirname "$PY")/activate" ]; then set +u; source "$(dirname "$PY")/activate"; set -u; fi
export PATH="$(dirname "$PY"):$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT SLURM_CPUS_PER_TASK=$NT
export PYTHONPATH="$B/code:$B/code/experiments"
echo "py_carc.sh $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')] NT=$NT $("$PY" -V 2>&1)"
exec "$PY" "$@"
