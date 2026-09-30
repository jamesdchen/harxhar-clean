#!/bin/bash
# CARC (Slurm): the slow part of the all_features LASSO permutation / drop-column walk of the C1
# re-run on the de-duplicated per-bar design (checklist I7): the refits whose causally chosen
# penalty is the grid's smaller value (0.001; 109-123 active columns, each unit re-solved with the
# spec's homotopy, ~5 min per refit), one refit per line of TASKFILE
# ("<bucket> <estimator> <k0> <k1>"), run through a pool of SLURM_CPUS_PER_TASK single-threaded
# workers (experiments/feature_importance_1530_dedup.py stage linear_part = the first pass's
# code on the re-run's root).  Output:
# results/feature_importance_1530_dedup/_work/linear/linear_<bucket>_<est>_part<k0>_<k1>.npz.
# Reads: the shipped input_all_features.npz, the re-run's coefficient capture, the stored
# research forecasts (the capture-vs-stored gap is recorded), experiments/model_diagnostics_1530.py
# (executes the spec's estimator section from specs/causal_tune_linear.py, byte-identical here).
#SBATCH --job-name=featimp_dedup_lin
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
NCPU="${SLURM_CPUS_PER_TASK:-1}"
LOGD="logs/featimp_dedup_lin_${SLURM_JOB_ID}"
mkdir -p "$LOGD"
echo "$(grep -c . "$TASKFILE") refits through a pool of $NCPU workers on $(hostname)  $(date)"
run_line() {  # "<bucket> <est> <k0> <k1>"
  local B E K0 K1 T0
  read -r B E K0 K1 <<< "$1"
  T0=$SECONDS
  if python -u experiments/feature_importance_1530_dedup.py linear_part "$B" "$E" "$K0" "$K1" \
       > "$LOGD/${B}_${E}_${K0}_${K1}.log" 2>&1; then
    echo "  done $B $E [$K0, $K1) in $((SECONDS - T0))s  $(date +%T)"
  else
    echo "  FAILED $B $E [$K0, $K1): $(tail -1 "$LOGD/${B}_${E}_${K0}_${K1}.log" | cut -c1-200)"
    return 1
  fi
}
export -f run_line
export LOGD
set +e
grep . "$TASKFILE" | xargs -P "$NCPU" -I{} bash -c 'run_line "$1"' _ {}
FAIL=$?
set -e
echo "pool finished, xargs exit $FAIL  $(date)"
exit $FAIL
