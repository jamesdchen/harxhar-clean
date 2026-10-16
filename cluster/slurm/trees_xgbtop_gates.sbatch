#!/bin/bash
# xgbtop (2026-10-07): the reproduction gate on CARC (experiments/trees_xgbtop_gate.py) -- the generator's spec for the
# SHIPPED XGBoost (specs/causal_tune_trees_xgb06.py from cluster/xgbtop_shipped_xgb.json = xgb_opt_06) re-runs the stored
# masked XGBoost cells of cluster/trees_xgbtop_gate_lines.txt (T1 / T10 on baseline, all_features, lean_vol_vix) and
# must equal the stored chunk npz (harxhar-mask, harxhar-momvolvix-trees; read only) and the published table cells
# (ref_tables/ = results/spxw_pnl/'s yhat_subtree*_xgb* tables, shipped) byte for byte.  Pinned by the submit script
# to the fleet's CPU class.  Lines run at once, one single-threaded process each.  Output: $OUT (GATE_OK,
# gate_lines.csv, gate_rows.csv) + NODE.
#SBATCH --job-name=txt_gates
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
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
{ echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null | grep -o "AvailableFeatures=[^ ]*"; echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"; } > "$OUT/NODE"
cat "$OUT/NODE"
python experiments/trees_xgbtop_make_specs.py --config cluster/xgbtop_shipped_xgb.json --token xgb06 --check \
  || { echo "specs/causal_tune_trees_xgb06.py is not the generated file"; exit 1; }
python -u experiments/trees_xgbtop_gate.py --lines "${LINES:-cluster/trees_xgbtop_gate_lines.txt}" --out "$OUT" \
  --spec specs/causal_tune_trees_xgb06.py --tables ref_tables --procs "$SLURM_CPUS_PER_TASK"
