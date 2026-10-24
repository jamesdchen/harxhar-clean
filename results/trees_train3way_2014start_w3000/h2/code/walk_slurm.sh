#!/bin/bash
# One forest walk piece on the Hoffman2 Slurm cluster: line SLURM_ARRAY_TASK_ID of TASKFILE -> runs_slurm/pieces
# (staging apart from the UGE runs/pieces; delivered only after the Slurm Gold 6342 check passes).  As walk.sh; runs
# only if the line is ASSIGNED, the CPU model is in h2/passed_cpus_slurm.txt, no UGE piece of the line exists, and a
# checkpointed piece resumes on its CPU model; else exit 3.
# (Written on Hoffman2 2026-10-24 12:49 PDT from the session heredoc, md5 5c791306a6199b242f8ce96fa9a4cf43; never run:
# the Slurm pi_shared partition refused the account.)
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.3.4
B=/u/scratch/j/jamesdc1/t3t
NT=${SLURM_CPUS_PER_TASK:?}
TASK=${SLURM_ARRAY_TASK_ID:?}
LINE=$(sed -n "${TASK}p" "${TASKFILE:?}" | tr -d '\r')
[ -n "$LINE" ] || { echo "line $TASK: none"; exit 0; }
read -r ARM EST SET PART <<< "$LINE"
K=${PART%/*}
N=${PART#*/}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "[$(date +%T)] t3t slurm walk task $TASK job ${SLURM_ARRAY_JOB_ID:-} on $(hostname) [$CPU], $NT cpus: $LINE"
SRC=$(cat "$TASKFILE.source")
if ! awk -v f="$SRC" -v k="$K" -v a="$ARM" -v e="$EST" -v s="$SET" '$1 == "ASSIGNED" && $2 == f && $4 == a && $5 == e && $6 == s { split($3, r, "-"); if (k + 0 >= r[1] + 0 && k + 0 <= r[2] + 0) ok = 1 } END { exit ok ? 0 : 1 }' $B/runs/tasks/h2_assignment.txt; then
  echo "line $K not ASSIGNED: skipped"
  exit 0
fi
grep -qxF "$CPU" $B/h2/passed_cpus_slurm.txt || { echo "CPU not gate-passed for Slurm: exit 3"; exit 3; }
U=$B/runs/pieces/${ARM}_${EST}_${SET}_${K}of${N}
for x in parquet json ckpt.json; do [ ! -e $U.$x ] || { echo "UGE piece file $U.$x exists: exit 3"; exit 3; }; done
P=$B/runs_slurm/pieces/${ARM}_${EST}_${SET}_${K}of${N}
if [ -f $P.ckpt.json ] && ! grep -qF "\"$CPU\"" $P.ckpt.json; then echo "begun on another CPU model: exit 3"; exit 3; fi
[ -f $B/ENV_OK ] || { echo "no env"; exit 1; }
echo "slurm walk $TASK $LINE | $(hostname) | $CPU | job ${SLURM_ARRAY_JOB_ID:-}.$TASK | $NT cpus | $(date +%F_%T)" >> $B/logs/hosts.txt
export T3T_NT=$NT T3T_ARM=$ARM T3T_EST=$EST T3T_SET=$SET T3T_PART=$PART
a=$(date +%s)
apptainer exec -B /u/scratch $B/deb.sif bash -c '
set -euo pipefail
B=/u/scratch/j/jamesdc1/t3t
PY=$B/venv_skx/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$B/lib
export OMP_NUM_THREADS=$T3T_NT MKL_NUM_THREADS=$T3T_NT OPENBLAS_NUM_THREADS=$T3T_NT NUMBA_NUM_THREADS=$T3T_NT
export OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES="$($PY -c "from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(\" \".join(k for k in d if k.startswith(\"AVX512\") and f.get(k)))")"
echo "NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
cd $B/code
export PYTHONPATH=$B/code:$B/code/experiments
$PY experiments/trees_train3way.py walk --arm "$T3T_ARM" --est "$T3T_EST" --set "$T3T_SET" --part "$T3T_PART" --threads "$T3T_NT" --design-root $B/carc/designs --out $B/runs_slurm
'
echo "[$(date +%T)] t3t slurm walk task $TASK done in $(( $(date +%s) - a )) s"
