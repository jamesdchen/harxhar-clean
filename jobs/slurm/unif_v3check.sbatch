#!/bin/bash
#SBATCH --job-name=unif_v3check
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/unif_%x_%A.out
# Panel v3 verification job (single task; SAFE TO RUN WHILE v2 IS LIVE):
# builds the v3 panel — data-derived grid start (first RV print ~1998-01) +
# dead-session rule + 1/99 winsor — into an ISOLATED cache dir so the live
# v2 prep cache is untouched, prints panel_length, then runs a0_ols_har on
# chunks 0 and 50. Submit with:
#   sbatch jobs/slurm/unif_v3check.sbatch
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Isolation: v3 panel cache lands here; the v2 campaign's results/ cache is
# never read or written by this job.
export UNIFY_CACHE_DIR=/scratch1/jc_905/harxhar-clean/results/v3cache
mkdir -p "$UNIFY_CACHE_DIR" results/unification_v3check
PY=/home1/jc_905/.conda/envs/harxhar/bin/python

$PY - <<'EOF'
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from src.unification import panel_length
n = panel_length()  # builds the v3 panel into $UNIFY_CACHE_DIR on first call
print(f"V3CHECK panel_length = {n}", flush=True)
EOF

for CHUNK in 0 50; do
  $PY experiments/run_unification.py --arm a0_ols_har --chunk-index "$CHUNK" \
      --output-dir results/unification_v3check
  echo "V3CHECK DONE arm=a0_ols_har chunk=$CHUNK"
done
