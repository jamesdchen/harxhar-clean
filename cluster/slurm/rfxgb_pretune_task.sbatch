#!/bin/bash
# Random forest / XGBoost pre-tune study (2026-10-06): one CARC task of experiments/close_trees_pretune_rfxgb.py.
# STAGE (exported by cluster/slurm/submit_rfxgb_pretune.sh) and one line of TASKFILE (the array index; 1-based):
#   canary   no task file: for every cell of CELLS ("<model>_<bucket> ...") the gate (fold fitter ==
#            the Optuna spec's fit_trial; control refits at sample rows == the stored masked tables)
#            and the whole pipeline at toy size under tag canary (2 studies x 2 trials, the 4 real
#            folds; 2 retune trials at row 380; refits of rows 370..381; assemble); writes
#            CANARY_OK_<cell> in the run tag's cell folder when all of it passed
#   pretune  "<model> <bucket> <seeds, comma-separated>": one Optuna study per seed, run at once,
#            each with a pool of RXP_WORKERS (4: the folds of a trial) single-threaded fit
#            processes; PRETUNE_DONE_s<seed> once the study exported its trials
#   plan     "<model> <bucket> <seeds>": refuses unless every study is done; merges them, runs the
#            light retunes (a pool of the task's CPUs), writes plan.json and PLAN_DONE
#   refit    "<model> <bucket> <phase> <slice> <slices>": phase control (no prerequisite) or arms
#            (refuses without PLAN_DONE); a pool of the task's CPUs; REFIT_DONE_<phase>_s<i>of<n>
# Every stage refuses to run unless CANARY_OK_<cell> exists (canary itself excepted).  Every fit is
# single-threaded (SLURM_CPUS_PER_TASK=1 for the spec; the pool is the parallelism), so no number
# depends on the task size.  The submit script pins one CPU class (--constraint).
# Env: RXP_TAG (run tag), RXP_TRIALS, RXP_TIMEOUT, PYBIN (a local python instead of the conda env).
#SBATCH --job-name=rxp
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
if [ -n "${PYBIN:-}" ]; then
  PY=$PYBIN
else
  module load conda >/dev/null 2>&1
  source activate harxhar
  PY=python
fi
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
NCPU=${SLURM_CPUS_PER_TASK:-4}
TAG=${RXP_TAG:?RXP_TAG}
WORK=results/trees_pretune_2026-10-06/_work
S=experiments/close_trees_pretune_rfxgb.py
"$PY" -c "import numpy, pandas, sklearn, xgboost, optuna; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'xgboost', xgboost.__version__, 'optuna', optuna.__version__)" \
  || { echo "library import failed; refusing to run"; exit 1; }
CPU=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//' || true)
echo "stage ${STAGE:?STAGE} on $(hostname) [$CPU] cpus $NCPU tag $TAG job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-} $(date)"

line() {
  local n=${SLURM_ARRAY_TASK_ID:?array index}
  sed -n "${n}p" "${TASKFILE:?TASKFILE}"
}
need_canary() {  # $1 = cell
  [ -f "$WORK/$TAG/$1/CANARY_OK_$1" ] || [ -f "$WORK/canary/$1/CANARY_OK_$1" ] \
    || { echo "CANARY_OK_$1 missing: the canary did not pass for $1; refusing"; exit 1; }
}

case "$STAGE" in
  canary)
    FAIL=0
    for cell in ${CELLS//:/ }; do
      m=${cell%%_*}; b=${cell#*_}
      D="$WORK/canary/$cell"
      mkdir -p "$D"
      rm -f "$D/CANARY_OK_$cell"
      ok=1
      export RXP_MODEL=$m RXP_BUCKET=$b RXP_WORKERS=$NCPU
      RXP_TAG=canary "$PY" -u $S gate > "$D/gate.log" 2>&1 || ok=0
      tail -n 15 "$D/gate.log"
      if [ "$ok" = 1 ]; then
        # errexit does not apply inside a tested command list: every step is chained with &&
        export RXP_TAG=canary RXP_TRIALS=2 RXP_RETUNE_TRIALS=2 RXP_CHECKPOINTS=1 RXP_WALK_ROWS=370:382 RXP_SEEDS=0,1
        RXP_SEED=0 RXP_WORKERS=4 "$PY" -u $S pretune & p0=$!
        RXP_SEED=1 RXP_WORKERS=4 "$PY" -u $S pretune & p1=$!
        { wait "$p0" && wait "$p1" \
          && "$PY" -u $S plan \
          && RXP_PHASE=control RXP_SLICE=0/1 "$PY" -u $S refit \
          && RXP_PHASE=arms RXP_SLICE=0/2 "$PY" -u $S refit \
          && RXP_PHASE=arms RXP_SLICE=1/2 "$PY" -u $S refit \
          && "$PY" -u $S assemble; } || ok=0
        unset RXP_TAG RXP_TRIALS RXP_RETUNE_TRIALS RXP_CHECKPOINTS RXP_WALK_ROWS RXP_SEEDS
        export RXP_TAG=$TAG
      fi
      if [ "$ok" = 1 ]; then
        echo "$(date) $(hostname) [$CPU]" > "$D/CANARY_OK_$cell"
        echo "CANARY $cell: OK"
      else
        echo "CANARY $cell: FAILED"; FAIL=1
      fi
    done
    exit $FAIL
    ;;
  pretune)
    read -r m b seeds <<< "$(line)"
    cell=${m}_$b; need_canary "$cell"
    D="$WORK/$TAG/$cell"; mkdir -p "$D"
    pids=()
    for s in ${seeds//,/ }; do
      if [ -f "$D/PRETUNE_DONE_s$s" ]; then echo "study s$s already done"; continue; fi
      RXP_MODEL=$m RXP_BUCKET=$b RXP_SEED=$s RXP_WORKERS=${RXP_WORKERS:-4} \
        "$PY" -u $S pretune >> "$D/pretune_s$s.log" 2>&1 &
      pids+=("$!:$s")
    done
    FAIL=0
    for ps in "${pids[@]}"; do
      if wait "${ps%%:*}" && [ -f "$D/pretune_s${ps#*:}.npz" ]; then
        echo "$(date) $(hostname) [$CPU]" > "$D/PRETUNE_DONE_s${ps#*:}"
      else
        echo "study s${ps#*:} FAILED (see $D/pretune_s${ps#*:}.log)"; FAIL=1
      fi
    done
    tail -n 2 "$D"/pretune_s*.log
    exit $FAIL
    ;;
  plan)
    read -r m b seeds <<< "$(line)"
    cell=${m}_$b; need_canary "$cell"
    D="$WORK/$TAG/$cell"
    for s in ${seeds//,/ }; do
      [ -f "$D/PRETUNE_DONE_s$s" ] || { echo "study s$s of $cell not done; refusing"; exit 1; }
    done
    RXP_MODEL=$m RXP_BUCKET=$b RXP_SEEDS=$seeds RXP_WORKERS=$NCPU "$PY" -u $S plan
    echo "$(date) $(hostname) [$CPU]" > "$D/PLAN_DONE"
    ;;
  refit)
    read -r m b phase i n <<< "$(line)"
    cell=${m}_$b; need_canary "$cell"
    D="$WORK/$TAG/$cell"
    if [ "$phase" = arms ]; then [ -f "$D/PLAN_DONE" ] || { echo "PLAN_DONE missing for $cell; refusing"; exit 1; }; fi
    RXP_MODEL=$m RXP_BUCKET=$b RXP_PHASE=$phase RXP_SLICE="$i/$n" RXP_WORKERS=$NCPU "$PY" -u $S refit
    echo "$(date) $(hostname) [$CPU]" > "$D/REFIT_DONE_${phase}_s${i}of$n"
    ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
