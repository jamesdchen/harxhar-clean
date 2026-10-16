#!/bin/bash
# GATE of the moments_vix Optuna run on CARC (cluster/slurm/submit_momvix_optuna.sh, --constraint
# epyc-7513): one array task per model re-computes STORED work of the canonical masked run
# (results/linear_subsection_trees_optuna_mask on the main CARC root, every chunk epyc-7513) with
# the code shipped to this root, through the fleet's own pack script:
#   stage 1  the task line of cluster/momvix_optuna_tasks_gate.txt (live_feasible, OOS rows
#            700-719, chunk id 900): 20 tuning points x 50 TPE trials
#   stage 2  the same rows refitted along the 9 paths FROM THE STORED merged stage-1 records
#            (copied here as the gate root's arm records, cmp-checked)
# then experiments/momvix_optuna_gate.py compares both with the stored records / merged paths
# bit for bit and, only if every array is identical, writes $RESULTS_ROOT/GATE_OK_<model>, the
# flag the fleet tasks of that model require.  Nothing here is part of the fleet's results
# ($RESULTS_ROOT/gate/).
# Exported: RESULTS_ROOT (this run's root), REF_ROOT (the stored canonical root, absolute),
# GATE_TASKFILE.
#SBATCH --job-name=mv_gate
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
R="${RESULTS_ROOT:?RESULTS_ROOT}"
REF="${REF_ROOT:?REF_ROOT}"
GT="${GATE_TASKFILE:?GATE_TASKFILE}"
I="${SLURM_ARRAY_TASK_ID:-1}"
LINE=$(sed -n "${I}p" "$GT")
read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
[ -n "${HALO:-}" ] || { echo "malformed line $I of $GT: '$LINE'"; exit 1; }
GR="$R/gate"
ARM="$BUCKET/$SEG/$MODEL/tw$TW"
echo "gate $MODEL on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')]: $LINE $(date)"
mkdir -p "$GR/stage1/$ARM" logs
cp "$REF/stage1/$ARM/trials_$SEG.npz" "$GR/stage1/$ARM/trials_$SEG.npz"
cmp "$REF/stage1/$ARM/trials_$SEG.npz" "$GR/stage1/$ARM/trials_$SEG.npz"
touch "$GR/stage1/$ARM/STAGE1_COMPLETE"
TF="$GR/task_$MODEL.txt"
echo "$LINE" > "$TF"
for ST in tune refit; do
  env -u CANARY_FLAG -u HOLD_FLAG -u WRITE_FLAG -u CANARY_CHECKS TASKFILE="$TF" STAGE="$ST" RESULTS_ROOT="$GR" \
    WINDOW_MASK=1 SLURM_ARRAY_TASK_ID=1 bash cluster/slurm/optuna_pack.sbatch \
    || echo "pack STAGE=$ST returned non-zero (the comparison below reports what is missing)"
done
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 PYTHONPATH="$PWD"
python -u experiments/momvix_optuna_gate.py --model "$MODEL" --bucket "$BUCKET" --gate-root "$GR" --ref-root "$REF" \
  --flag "$R/GATE_OK_$MODEL" --row0 $((START - HALO)) --rows $((END - START))
