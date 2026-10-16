#!/bin/bash
# Hoffman2 GATE of the moments_vix Optuna run (the twin of cluster/slurm/momvix_gate.sbatch): one
# array task per model, pinned by the submit script to one CPU class (REQUIRE_CPU checked by the
# task script), re-computes STORED work of the canonical masked run (every chunk on CARC
# epyc-7513) through the fleet's own Hoffman2 task script:
#   stage 1  the line of cluster/momvix_optuna_tasks_gate.txt (live_feasible OOS rows 700-719,
#            chunk 900): 20 tuning points x 50 TPE trials
#   stage 2  the same rows refitted along the 9 paths FROM THE STORED merged stage-1 records
#            ($PWD/refs, shipped from the local copies whose md5 equals the CARC main root's)
# then experiments/momvix_optuna_gate.py compares both with the stored records bit for bit and,
# only if every array is identical, writes $RESULTS_ROOT/GATE_OK_<model>, the flag the Hoffman2
# fleet tasks of that model require.
# Exported: RESULTS_ROOT, GATE_TASKFILE, REQUIRE_CPU (passed on), RUNTIME_ROOT (optional).
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
RT=${RUNTIME_ROOT:-/u/scratch/j/jamesdc1/harxhar-optuna}
R="${RESULTS_ROOT:?RESULTS_ROOT}"
GT="${GATE_TASKFILE:?GATE_TASKFILE}"
REF=$D/refs
I="${SGE_TASK_ID:-1}"; [ "$I" = "undefined" ] && I=1
LINE=$(sed -n "${I}p" "$GT")
read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
[ -n "${HALO:-}" ] || { echo "malformed line $I of $GT: '$LINE'"; exit 1; }
GR="$R/gate"
ARM="$BUCKET/$SEG/$MODEL/tw$TW"
echo "h2 gate $MODEL on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')]: $LINE $(date)"
mkdir -p "$GR/stage1/$ARM" logs
cp "$REF/stage1/$ARM/trials_$SEG.npz" "$GR/stage1/$ARM/trials_$SEG.npz"
cmp "$REF/stage1/$ARM/trials_$SEG.npz" "$GR/stage1/$ARM/trials_$SEG.npz"
touch "$GR/stage1/$ARM/STAGE1_COMPLETE"
TF="$GR/task_$MODEL.txt"
echo "$LINE" > "$TF"
for ST in tune refit; do
  env -u REQUIRE_FLAGS TASKFILE="$TF" STAGE="$ST" RESULTS_ROOT="$GR" WINDOW_MASK=1 SGE_TASK_ID=1 \
    bash cluster/momvix_optuna_h2_task.sh \
    || echo "task STAGE=$ST returned non-zero (the comparison below reports what is missing)"
done
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME
export PATH="$RT/carc_env/harxhar/bin:/usr/bin:/bin" PYTHONUSERBASE="$RT/carc_usersite" PYTHONPATH="$D" OMP_NUM_THREADS=1
set -u
"$RT/carc_env/harxhar/bin/python3.11" -u experiments/momvix_optuna_gate.py --model "$MODEL" --bucket "$BUCKET" \
  --gate-root "$GR" --ref-root "$REF" --flag "$R/GATE_OK_$MODEL" --row0 $((START - HALO)) --rows $((END - START))
