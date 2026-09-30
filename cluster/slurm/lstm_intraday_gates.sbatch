#!/bin/bash
# I10 (2026-09-29): the gates of the intraday-sequence LSTM on CARC (experiments/gate_lstm_intraday.py),
# on the cluster's own torch / CPUs.  GATE_SET = A: identity + seq_end + mask_flatten for the three
# input sets (the campaign's W = 2000); B: determinism, pool, chunk (incl. a mid-period chunk and the
# tuning cache), causal_target, causal_bar (live_feasible, reduced window W_G), the synthetic mask gate,
# and the end-to-end chunk gate through the run cell and the reducer at W = 2000.  Writes
# $RESULTS_ROOT/GATES_<GATE_SET>_OK only if every gate of the set passes; outputs in
# $RESULTS_ROOT/gates/ (gate_lstm_intraday_*.csv).
#SBATCH --job-name=lstmi_gates
#SBATCH --account=pollok_1603
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
R="${RESULTS_ROOT:-results/linear_subsection_lstm_intraday}"
NT="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$R/gates"
echo "gates $GATE_SET on $(hostname) ($(lscpu | grep -m1 'Model name' | sed 's/.*: *//')), $NT CPUs $(date)"
if [ "$GATE_SET" = "A" ]; then
  ARGS=(--gates identity,seq_end,mask_flatten --buckets live_feasible,all_features,baseline)
else
  ARGS=(--gates determinism,pool,chunk,causal_target,causal_bar,mask --buckets live_feasible --e2e)
fi
if python -u experiments/gate_lstm_intraday.py "${ARGS[@]}" --pool "$NT" \
     --scratch "$R/gates/scratch_$GATE_SET" --out "$R/gates"; then
  touch "$R/GATES_${GATE_SET}_OK"
  echo "gates $GATE_SET PASS $(date)"
else
  echo "gates $GATE_SET FAILED $(date)"
  exit 1
fi
