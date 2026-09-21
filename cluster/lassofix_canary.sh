#!/bin/bash
# Canary for the recursive-lasso fix: the worst offender of wave 2 (one-bar
# implied_vol lasso on the bar ending 13:30, 500-day window: 77 absurd
# forecasts on this cluster before the fix) plus the other six blown one-bar
# and block arms.  The HEALTH flag the fleet waits for is written only if the
# worst offender reproduces the reference run made with the fixed code on
# another machine (cluster/lassofix_reference_iv_bar1330_tw500.csv), bar by bar.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=3:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
R=results/linear_subsection_lassofix

run_arm () {  # bucket segment window
  local OUT="$R/$1/$2/reclasso/tw$3"
  [ -f "$OUT/DONE" ] && return 0
  mkdir -p "$OUT"
  HPC_KW_SEGMENT="$2" HPC_KW_LAG_SCOPE=global HPC_KW_ESTIMATOR=reclasso \
  HPC_KW_EXOG_BUCKET="$1" HPC_KW_TRAIN_WIN="$3" HPC_RESULT_DIR="$OUT" \
    python -u specs/causal_tune_linear.py > "$OUT/run.log" 2>&1
  touch "$OUT/DONE"
  echo "done $1 $2 tw$3  $(date)"
}

run_arm implied_vol bar1330 500
python - <<'PY'
import sys
import numpy as np, pandas as pd
got = pd.read_csv("results/linear_subsection_lassofix/implied_vol/bar1330/reclasso/tw500/causal_tune_linear/reclasso/implied_vol/results_bar1330.csv")
ref = pd.read_csv("cluster/lassofix_reference_iv_bar1330_tw500.csv")
assert len(got) == len(ref) and (got["date"] == ref["date"]).all(), "row sets differ"
dev = float(np.max(np.abs(got["pred_adj"] - ref["pred_adj"])))
print(f"worst offender vs the fixed-code reference: {len(got)} bars, max |pred_adj difference| {dev:.3e}; "
      f"pred_adj range [{got['pred_adj'].min():.3f}, {got['pred_adj'].max():.3f}]")
# a float-path tolerance, not a model constant: before the fix the differences were 1e6..1e10
sys.exit(0 if dev < 1e-6 else 1)
PY
touch "$R/CANARY_HEALTHY"
echo "CANARY_HEALTHY written  $(date)"

run_arm implied_vol bar1030 500
run_arm implied_vol bar1300 2000
run_arm implied_vol bar1500 2000
run_arm sentiment bar1500 2000
run_arm sentiment blk_close 500
run_arm vol_demand blk_release 500
grep -h "between-tune" $R/*/*/reclasso/tw*/run.log | cut -c1-200
