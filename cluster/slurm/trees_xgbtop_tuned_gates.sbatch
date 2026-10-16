#!/bin/bash
# xgbtop (2026-10-07): the identity gate of the re-centred random-search rungs on CARC -- the tuned spec through
# specs/causal_tune_trees_tuned_xgbtop.py with TUNE_IDENTITY=1 must reproduce the FIXED-settings xgb_opt_00 cells of
# results/linear_subsection_trees_mask_xgbtop (this deployment, REF_ROOT) byte for byte
# (experiments/trees_xgbtop_tuned_gate.py; lines cluster/trees_xgbtop_tuned_gate_lines.txt, one after another, each with a
# spawned pool of the task's CPUs, single-threaded fits).  Pinned by the submit script to the fleet's class.
# Output: $OUT (GATE_OK, gate_lines.csv, gate_rows.csv) + NODE.
#SBATCH --job-name=txt_rsgate
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
export HPC_KW_TREE_SPEC=causal_tune_trees_xgbtop.py
OUT="${OUT:?OUT}"
mkdir -p "$OUT"
{ echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null | grep -o "AvailableFeatures=[^ ]*"; echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"; } > "$OUT/NODE"
python experiments/trees_xgbtop_make_specs.py --check || { echo "specs/causal_tune_trees_xgbtop.py is not the generated file"; exit 1; }
python -u experiments/trees_xgbtop_tuned_gate.py --lines "${LINES:-cluster/trees_xgbtop_tuned_gate_lines.txt}" \
  --ref-root "${REF_ROOT:?REF_ROOT}" --out "$OUT" --pool "$SLURM_CPUS_PER_TASK" --stored "${GATE_STORED:-linear_subsection_trees_mask_xgbtop}"
