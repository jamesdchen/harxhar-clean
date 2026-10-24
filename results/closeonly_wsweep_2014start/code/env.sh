#!/bin/bash
# Sourced by the job scripts of the close-only window sweep (RULE.txt section 7).  BASE (staging), DES (the design
# copies, off /scratch1), the python of each environment (the three-way comparisons', read only), thread settings.
#   gb     /scratch1/jc_905/envs/harxhar-r13 (LightGBM 4.6.0, XGBoost 3.2.0; the trees three-way's)
#   skx    /home1/jc_905/trees_final/venv_skx (+ scikit-learn-intelex 2026.2.0; the forest's)
#   sweep  /scratch1/jc_905/wsweep/venv (the linear runner and the scorers, as the linear three-way)
BASE=/scratch1/jc_905/closewin
DES=/home1/jc_905/closewin/designs
pyof() {
  case $1 in
    gb | lgbm | xgb) echo /scratch1/jc_905/envs/harxhar-r13/bin/python ;;
    skx | rf_skx) echo /home1/jc_905/trees_final/venv_skx/bin/python ;;
    sweep) echo /scratch1/jc_905/wsweep/venv/bin/python ;;
    *)
      echo "unknown python $1" >&2
      return 2
      ;;
  esac
}
NT="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
[ -f /home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf ] && export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
# the linear three-way's stage3.sh settings: OpenBLAS kernel Prescott, numpy's AVX512 dispatch off where the node has it
sweep_env() {
  export OPENBLAS_CORETYPE=Prescott
  NPY_DISABLE_CPU_FEATURES="$("$(pyof sweep)" "$CODE/results/closeonly_wsweep_2014start/code/avx512.py")"
  export NPY_DISABLE_CPU_FEATURES
  echo "NPY_DISABLE_CPU_FEATURES=${NPY_DISABLE_CPU_FEATURES:-}"
}
hostlog() { # <tag>
  mkdir -p $BASE/logs
  local cpu
  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
  echo "$1 | $(hostname) | $cpu | job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}_${SLURM_ARRAY_TASK_ID:-} | $NT threads | $(date +%F_%T)" >> $BASE/logs/hosts.txt
}
