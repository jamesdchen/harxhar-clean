#!/bin/bash
# 2026-10-06 (moments_vix LSTMs, deployment root /scratch1/jc_905/harxhar-momvix-lstm): the gates the
# new bucket's fleets are held on (afterok, --kill-on-invalid-dep=yes in submit_momvix_lstm.sh).
# MODE (exported at submit time):
#   perbar      the reproduction gate of the per-bar LSTM: experiments/momvix_lstm_gate_compare.py on
#               the gate runs of cluster/lstm_momvix_tasks_gate.txt (stored chunks of the canonical
#               masked daily run, harxhar-mask) + experiments/gate_lstm.py identity for moments_vix
#               (the LSTM spec's design / target / stamps = the per-bar linear spec's own invocation,
#               sha256) -> $GP/GATE_OK
#   intraday    the reproduction gate of the intraday LSTM (cluster/lstmi_momvix_tasks_gate.txt vs
#               harxhar-lstmi's stored chunks and stage-1 tuning records) -> $GI/GATE_OK
#   intraday_A  experiments/gate_lstm_intraday.py identity, seq_end for moments_vix and live_feasible,
#               mask_flatten on live_feasible's windows (the campaign's gate set A; the identity row's
#               n_oos is read by the stage-2 planner) -> $LI/GATES_A_OK
# Exits 1 when a gate fails (the held jobs are then cancelled by Slurm, never started).
#SBATCH --job-name=mvl_check
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
GP=results/momvix_gates/perbar
GI=results/momvix_gates/intraday
LI=results/linear_subsection_lstm_intraday_momvix
MASK_STORED=/scratch1/jc_905/harxhar-mask/results/linear_subsection_lstm_mask
LSTMI_STORED=/scratch1/jc_905/harxhar-lstmi/results/linear_subsection_lstm_intraday
NT="${SLURM_CPUS_PER_TASK:-1}"
echo "check ${MODE:?MODE must be perbar, intraday or intraday_A} on $(hostname) ($(lscpu | grep -m1 'Model name' | sed 's/.*: *//')), $NT CPUs $(date)"
python -c "import torch, numpy, pandas; print('torch', torch.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)"
FAIL=0
case "$MODE" in
  perbar)
    rm -f "$GP/GATE_OK"
    python -u experiments/momvix_lstm_gate_compare.py --kind perbar --tasks cluster/lstm_momvix_tasks_gate.txt \
      --gate-root "$GP" --stored-root "$MASK_STORED" --out "$GP/compare.csv" || FAIL=1
    python -u experiments/gate_lstm.py --gates identity --buckets moments_vix --pool 1 \
      --scratch "$GP/identity_scratch" --out "$GP/identity" || FAIL=1
    [ "$FAIL" -eq 0 ] && touch "$GP/GATE_OK"
    ;;
  intraday)
    rm -f "$GI/GATE_OK"
    python -u experiments/momvix_lstm_gate_compare.py --kind intraday --tasks cluster/lstmi_momvix_tasks_gate.txt \
      --gate-root "$GI" --stored-root "$LSTMI_STORED" --out "$GI/compare.csv" || FAIL=1
    [ "$FAIL" -eq 0 ] && touch "$GI/GATE_OK"
    ;;
  intraday_A)
    rm -f "$LI/GATES_A_OK"
    mkdir -p "$LI/gates"
    # mask_flatten is a code-path gate on live_feasible windows only (the script's C_live): with
    # --buckets moments_vix alone it stops on "assert C_live is not None" (job 12747230)
    python -u experiments/gate_lstm_intraday.py --gates identity,seq_end,mask_flatten --buckets moments_vix,live_feasible \
      --pool "$NT" --scratch "$LI/gates/scratch_A" --out "$LI/gates" || FAIL=1
    [ "$FAIL" -eq 0 ] && touch "$LI/GATES_A_OK"
    ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac
if [ "$FAIL" -eq 0 ]; then echo "check $MODE PASS $(date)"; else echo "check $MODE FAILED $(date)"; exit 1; fi
