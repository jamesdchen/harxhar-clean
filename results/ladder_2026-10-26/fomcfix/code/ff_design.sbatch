#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=lffdes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --output=/home1/jc_905/ladder_fomcfix/logs/%x.%A_%a.out
# Patched design caches at W = SLURM_ARRAY_TASK_ID days (RULE.txt ADDENDUM 3 (1)): the fomcfix code tree and venv, its
# lin_stage.sh "design" command with --window-days W, out /home1/jc_905/ladder_fomcfix/designs/w<W>.  No copy or move.
set -euo pipefail
W=${SLURM_ARRAY_TASK_ID}
PY=/home1/jc_905/fomcfix_lin/venv/bin/python
OUT=/home1/jc_905/ladder_fomcfix/designs/w$W
cd /home1/jc_905/fomcfix_lin/code
export OPENBLAS_CORETYPE=Prescott PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
echo "design W $W on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)] $(date)"
"$PY" experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs 8 --window-days "$W" --first-day 2014-01-02 \
  --ex-flag LATE_START_WARMUP=1 --ex-flag FOMC_SCHEDULE_2020_FIX=1 --ex-flag FOMC_PROXIMITY_NO_WINSOR=1 --out "$OUT"
ls -la "$OUT/_work/design_all_features"
echo "design W $W done $(date)"
