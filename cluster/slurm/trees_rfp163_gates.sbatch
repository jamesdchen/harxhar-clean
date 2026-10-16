#!/bin/bash
# rfp163 (2026-10-07): the identity gate of the re-centred random-search rungs on CARC -- the tuned spec through
# specs/causal_tune_trees_tuned_rfp.py with TUNE_IDENTITY=1 must reproduce the trial-163 FIXED-settings forest of the
# preview run (REF_ROOT, read only) byte for byte (experiments/trees_rfp163_tuned_gate.py).  Pinned by the submit
# script to epyc-7513.  Output: $OUT (GATE_OK, gate_lines.csv, gate_rows.csv) + NODE.
#SBATCH --job-name=trq_gates
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
OUT="${OUT:?OUT}"
mkdir -p "$OUT"
{ echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null | grep -o "AvailableFeatures=[^ ]*"; } > "$OUT/NODE"
python -u experiments/trees_rfp163_tuned_gate.py --lines "${LINES:-cluster/trees_rfp163_tuned_gate_lines.txt}" \
  --ref-root "${REF_ROOT:?REF_ROOT}" --out "$OUT" --pool "$SLURM_CPUS_PER_TASK" --stored "${GATE_STORED:-linear_subsection_trees_mask_rfprev163}"
