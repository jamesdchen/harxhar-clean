#!/bin/bash
#$ -cwd
#$ -j y
# One walk piece on Hoffman2 (UGE array task $SGE_TASK_ID): line $SGE_TASK_ID of $TASKFILE, "<arm> <est> <set> <i>/<n>"
# (a subset of a CARC task file, line i of CARC's file), through experiments/trees_train3way.py walk -> $B/runs/pieces/
# <arm>_<est>_<set>_<i>of<n>.parquet + .json (checkpointed every 300 s; resumable).  As line.sh (container, threads =
# NSLOTS, numpy AVX-512 dispatch off, OpenBLAS Haswell; the forest in venv_skx, LightGBM / XGBoost in venv_gb).  Runs only if
#   - the line is ASSIGNED to Hoffman2 in $B/runs/tasks/h2_assignment.txt (the relayed copy of CARC's; otherwise exit 0),
#   - this node's CPU model is gate-passed ($B/h2/passed_cpus.txt, one model a line; otherwise exit 99 = reschedule),
#   - a checkpointed piece resumes on the CPU model that began it (the forest's one-family rule; otherwise exit 99).
# Env (qsub -v): TASKFILE.
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/t3t
NT=${NSLOTS:-1}
TASK=${SGE_TASK_ID:-1}; [ "$TASK" = undefined ] && TASK=1
LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}" | tr -d '\r')
[ -n "$LINE" ] || { echo "line $TASK of $TASKFILE: none"; exit 0; }
read -r ARM EST SET PART <<< "$LINE"
[ -n "${PART:-}" ] || { echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1; }
K=${PART%/*}
N=${PART#*/}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "[$(date +%T)] t3t walk task $TASK job ${JOB_ID:-} on $(hostname) [$CPU], $NT slots: $LINE"
SRC=$(cat "$TASKFILE.source")  # the CARC task file name the lines come from (e.g. w3rf_full.txt)
if ! awk -v f="$SRC" -v k="$K" -v a="$ARM" -v e="$EST" -v s="$SET" '
  $1 == "ASSIGNED" && $2 == f && $4 == a && $5 == e && $6 == s { split($3, r, "-"); if (k + 0 >= r[1] + 0 && k + 0 <= r[2] + 0) ok = 1 }
  END { exit ok ? 0 : 1 }' $B/runs/tasks/h2_assignment.txt; then
  echo "line $SRC:$K is not ASSIGNED to Hoffman2 in $B/runs/tasks/h2_assignment.txt: skipped (exit 0)"
  exit 0
fi
if ! grep -qxF "$CPU" $B/h2/passed_cpus.txt 2>/dev/null; then
  echo "CPU '$CPU' is not gate-passed ($B/h2/passed_cpus.txt): refusing (exit 99 = reschedule)"
  exit 99
fi
P=$B/runs/pieces/${ARM}_${EST}_${SET}_${K}of${N}
if [ -f $P.ckpt.json ] && ! grep -qF "\"$CPU\"" $P.ckpt.json; then
  echo "piece $P begun on another CPU model ($(cat $P.ckpt.json)): refusing (exit 99 = reschedule)"
  exit 99
fi
[ -f $B/ENV_OK ] || { echo "no env"; exit 1; }
mkdir -p $B/runs/pieces
echo "walk $(basename $TASKFILE):$TASK $LINE | $(hostname) | $CPU | job ${JOB_ID:-}.$TASK | $NT slots | $(date +%F_%T)" >> $B/logs/hosts.txt
PY=$B/venv_gb/bin/python
[ "$EST" = rf_skx ] && PY=$B/venv_skx/bin/python
export T3T_NT=$NT T3T_ARM=$ARM T3T_EST=$EST T3T_SET=$SET T3T_PART=$PART T3T_PY=$PY
export SLURM_ARRAY_JOB_ID=${JOB_ID:-} SLURM_ARRAY_TASK_ID=$TASK SLURM_CPUS_PER_TASK=$NT
a=$(date +%s)
apptainer exec -B /u/scratch $B/deb.sif bash -c '
set -euo pipefail
B=/u/scratch/j/jamesdc1/t3t
PY=$T3T_PY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$B/lib
export OMP_NUM_THREADS=$T3T_NT MKL_NUM_THREADS=$T3T_NT OPENBLAS_NUM_THREADS=$T3T_NT NUMBA_NUM_THREADS=$T3T_NT
export OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES="$($PY -c "from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(\" \".join(k for k in d if k.startswith(\"AVX512\") and f.get(k)))")"
echo "NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES"
cd $B/code
export PYTHONPATH=$B/code:$B/code/experiments
$PY experiments/trees_train3way.py walk --arm "$T3T_ARM" --est "$T3T_EST" --set "$T3T_SET" --part "$T3T_PART" \
  --threads "$T3T_NT" --design-root $B/carc/designs --out $B/runs
'
echo "[$(date +%T)] t3t walk task $TASK done in $(( $(date +%s) - a )) s on $(hostname -s)"
