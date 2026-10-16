#!/bin/bash
# moments_vix tree workstream (2026-10-06): the reproduction gate on CARC -- re-run the stored masked
# all_features cells of cluster/trees_momvix_gate_lines.txt (rungs T1, T10, RS1, RS10; LightGBM,
# XGBoost, RF) with the code shipped for the new bucket, and compare every recorded array with the
# stored chunk npz of the masked campaign's deployment root (REF_ROOT, read only), byte for byte
# (experiments/trees_momvix_gate.py).  Pinned by the submit script to the fleet's CPU class
# (--constraint, epyc-7513: the class every masked chunk of the reference ran on).
# Untuned lines run at once, one single-threaded process each; tuned lines one after another, each
# with a pool of the task's CPUs (single-threaded fits).  Output: $OUT (GATE_OK, PASS_<rung>_<model>,
# gate_lines.csv, gate_rows.csv) + NODE (host and CPU features of this node).
#SBATCH --job-name=tmv_gates
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
python -c "import shap, lightgbm, xgboost, sklearn; print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"
OUT="${OUT:?OUT}"
mkdir -p "$OUT"
{ echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null | grep -o "AvailableFeatures=[^ ]*"; } > "$OUT/NODE"
cat "$OUT/NODE"
python -u experiments/trees_momvix_gate.py --lines "${LINES:-cluster/trees_momvix_gate_lines.txt}" \
  --ref-root "${REF_ROOT:?REF_ROOT}" --out "$OUT" --pool "$SLURM_CPUS_PER_TASK" --procs "$SLURM_CPUS_PER_TASK"
