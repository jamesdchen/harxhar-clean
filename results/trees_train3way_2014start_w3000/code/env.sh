#!/bin/bash
# Sourced by the job scripts of the trees' three-way comparison (RULE.txt section 8): BASE, the python of each class
# (the templates' environments, read only) and the thread / site settings of cluster/slurm/pooltree_rerun13.sbatch.
#   gb     /scratch1/jc_905/envs/harxhar-r13 (LightGBM 4.6.0, XGBoost 3.2.0; the LightGBM / XGBoost template's)
#   skx    /home1/jc_905/trees_final/venv_skx (the same env + scikit-learn-intelex 2026.2.0; the rf_skx template's)
#   sweep  /scratch1/jc_905/wsweep/venv (scoring, as the linear comparison's score job)
# 2026-10-24 (addendum E note): T3T_BASE moves runs / logs off /scratch1 for pools on nodes with failing mounts.
BASE=${T3T_BASE:-/scratch1/jc_905/trees3way}
pyof() {
  case $1 in
    gb | lgbm | xgb) echo /scratch1/jc_905/envs/harxhar-r13/bin/python ;;
    skx | rf_skx) echo /home1/jc_905/trees_final/venv_skx/bin/python ;;
    sweep) echo /scratch1/jc_905/wsweep/venv/bin/python ;;
    *) echo "unknown python $1" >&2; return 2 ;;
  esac
}
NT="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
[ -f /home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf ] && export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
hostlog() {  # <tag>
  mkdir -p $BASE/logs
  echo "$1 | $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//') | job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}_${SLURM_ARRAY_TASK_ID:-} | $NT threads | $(date +%F_%T)" >> $BASE/logs/hosts.txt
}
