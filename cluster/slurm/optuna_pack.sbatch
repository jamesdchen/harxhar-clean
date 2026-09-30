#!/bin/bash
# CARC Discovery (Slurm) task of the Optuna per-bar tree campaign
# (specs/causal_tune_trees_optuna.py, checklist I3): one array task runs one
# TIME CHUNK of one arm (bucket x model at the 16:00 bar) for one STAGE.
# TASKFILE holds one
#   "<bucket> <model[,model...]> <train_win_days> <seg> <chunk> <start> <end> <halo>"
# line per task (see submit_optuna.sh); the models of a line run one after
# another, each into
#   $RESULTS_ROOT/stage<1|2>/<bucket>/<seg>/<model>/tw<TW>/chunks/c<chunk>
# (experiments/reduce_trees_optuna.py merges an arm's chunks later).  A chunk-arm
# that already wrote DONE is skipped, and a stage-1 chunk resumes from the tuning
# points it saved (points/pt_<row>.npz, RESUME=1), so a resubmitted task resumes.
# Exported at submit time: TASKFILE, STAGE (tune | refit), RESULTS_ROOT (default
# results/linear_subsection_trees_optuna_mask), CANARY_FLAG (a file the task requires),
# HOLD_FLAG (stage 2: a file the stage-1 merge writes), WRITE_FLAG (a file this task
# touches when every chunk-arm of its line succeeded, %m = the line's model(s) -- the
# canary writes the CANARY_FLAG files this way), CANARY_CHECKS (1 = the canary's extra
# checks below), N_TRIALS (default 50), WINDOW_MASK (0 | 1, default 0; the spec's
# per-window column mask -- submit_optuna.sh passes 1).
# STAGE refit reads the arm's merged stage-1 records
#   $RESULTS_ROOT/stage1/<bucket>/<seg>/<model>/tw<TW>/trials_<seg>.npz
# and refuses to run unless that arm's STAGE1_COMPLETE flag exists (the stage-1
# completeness check of experiments/reduce_trees_optuna.py --stage 1).
# CANARY_CHECKS=1 (the canary): after the stage-1 chunk, (a) the chunk is re-run in
# a second directory with a pool of REPEAT_CPUS processes and its tuning records must
# equal the first run's bit for bit (determinism and pool-size invariance on CARC),
# (b) stage 2 runs on the chunk's rows from the chunk's own records into
# $RESULTS_ROOT/canary_stage2 (every path's tuning points lie inside the chunk when it
# starts at row 0 and spans >= 5 rows); neither is part of the fleet's results.
# Parallelism is across PROCESSES: the spec runs a pool of SLURM_CPUS_PER_TASK
# workers and every fit is single-threaded, so the BLAS/OpenMP/numba thread counts
# are pinned to 1.  optuna is in the harxhar env; shap (RF TreeSHAP) lives in
# ./pylib_trees (the untuned campaign installed it), on PYTHONPATH only here.
# The #SBATCH values are fallbacks for a hand-submitted task; submit_optuna.sh
# passes --cpus-per-task/--mem/--time per job.
#SBATCH --job-name=optuna
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

NT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
R="${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}"
STAGE="${STAGE:?STAGE must be tune or refit}"
case "$STAGE" in tune) SN=1 ;; refit) SN=2 ;; *) echo "bad STAGE $STAGE"; exit 1 ;; esac

for f in ${CANARY_FLAG:-}; do  # space-separated: one flag per canary task
  if [ ! -f "$f" ]; then
    echo "canary did not finish ($f missing); refusing to run"
    exit 1
  fi
done
if [ -n "${HOLD_FLAG:-}" ] && [ ! -f "$HOLD_FLAG" ]; then
  echo "hold flag $HOLD_FLAG missing (the previous stage's merge did not run); refusing to run"
  exit 1
fi
if ! python -c "import optuna, shap, lightgbm, xgboost, sklearn; print('optuna', optuna.__version__, 'shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"; then
  echo "import of optuna/shap/lightgbm/xgboost/sklearn failed (PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET MODELS TW SEG CHUNK START END HALO <<< "$LINE"
if [ -z "$HALO" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE (want 8 fields): '$LINE'"
  exit 1
fi
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname), stage $SN ($STAGE), pool of $NT: $BUCKET [$MODELS] tw$TW $SEG chunk $CHUNK rows [$START, $END) halo $HALO  $(date)"

run_spec() {  # $1 model, $2 result dir, $3 pool size, $4 stage-1 records (stage 2), $5 start, $6 end
  SLURM_CPUS_PER_TASK="$3" HPC_KW_MODEL="$1" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
    HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_START="$5" HPC_KW_END="$6" HPC_KW_HALO="$HALO" \
    HPC_KW_STAGE="$STAGE" HPC_KW_N_TRIALS="${N_TRIALS:-50}" HPC_KW_IDENTITY=0 HPC_KW_RESUME=1 \
    HPC_KW_STAGE1_TRIALS="$4" HPC_RESULT_DIR="$2" HPC_KW_WINDOW_MASK="${WINDOW_MASK:-0}" \
    python -u specs/causal_tune_trees_optuna.py
}

FAIL=0
for MODEL in ${MODELS//,/ }; do
  ARM="$BUCKET/$SEG/$MODEL/tw$TW"
  OUT="$R/stage$SN/$ARM/chunks/c$CHUNK"
  [ -f "$OUT/DONE" ] && { echo "  $MODEL c$CHUNK already DONE"; continue; }
  S1=""
  if [ "$SN" = 2 ]; then
    if [ ! -f "$R/stage1/$ARM/STAGE1_COMPLETE" ]; then
      echo "  $MODEL: stage 1 of $ARM is not complete (no STAGE1_COMPLETE); refusing"
      FAIL=1
      continue
    fi
    S1="$R/stage1/$ARM/trials_$SEG.npz"
  fi
  mkdir -p "$OUT"
  a=$(date +%s)
  if run_spec "$MODEL" "$OUT" "$NT" "$S1" "$START" "$END" > "$OUT/run.log" 2>&1; then
    # the CPU class of the node (the executor's X differs at rounding level between AVX-512
    # and AVX2 nodes -- agent D, 2026-09-29): hostname, CPU model, avx512f flag, job ids
    printf 'host=%s\ncpu=%s\navx512f=%s\njob=%s_%s\n' "$(hostname)" \
      "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')" \
      "$(grep -m1 -c -w avx512f /proc/cpuinfo)" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}" \
      "${SLURM_ARRAY_TASK_ID:-}" > "$OUT/host.txt"
    touch "$OUT/DONE"
    echo "  $MODEL $SEG c$CHUNK done in $(( $(date +%s) - a )) s $(date)"
  else
    FAIL=1
    echo "  $MODEL $SEG c$CHUNK FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
    continue
  fi
  if [ "${CANARY_CHECKS:-0}" = 1 ] && [ "$SN" = 1 ]; then
    REP="$R/canary_repeat/$ARM/c$CHUNK"
    rm -rf "$REP"; mkdir -p "$REP"
    if run_spec "$MODEL" "$REP" "${REPEAT_CPUS:-7}" "" "$START" "$END" > "$REP/run.log" 2>&1 \
       && python - "$OUT" "$REP" "$MODEL" "$BUCKET" "$SEG" <<'EOF'
import sys, numpy as np
a, b, model, bucket, seg = sys.argv[1:]
f = f"causal_tune_trees/{model}/{bucket}/trials_{seg}.npz"
za, zb = np.load(f"{a}/{f}"), np.load(f"{b}/{f}")
keys = ["row", "seed", "split", "params", "val_mse", "val_qlike", "rounds", "rounds_max", "patience", "complete", "best_mse_k10", "best_mse_k25", "best_mse_k50", "best_qlike_k50", "n_kept"]
bad = [k for k in keys if not np.array_equal(za[k], zb[k], equal_nan=True)]
print("canary repeat (pool size differs):", "records bit-identical" if not bad else f"DIFFER in {bad}")
sys.exit(1 if bad else 0)
EOF
    then echo "  $MODEL canary repeat OK"
    else FAIL=1; echo "  $MODEL canary repeat FAILED (see $REP/run.log)"; fi
    C2="$R/canary_stage2/$ARM/c$CHUNK"
    rm -rf "$C2"; mkdir -p "$C2"
    if STAGE=refit run_spec "$MODEL" "$C2" "$NT" "$OUT/causal_tune_trees/$MODEL/$BUCKET/trials_$SEG.npz" "$START" "$END" > "$C2/run.log" 2>&1; then
      echo "  $MODEL canary stage 2 OK: $(tail -1 "$C2/run.log" | cut -c1-220)"
    else
      FAIL=1; echo "  $MODEL canary stage 2 FAILED: $(tail -1 "$C2/run.log" | cut -c1-200)"
    fi
  fi
done
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  WF="${WRITE_FLAG//%m/$MODELS}"  # %m = the line's model(s): one flag per canary task
  mkdir -p "$(dirname "$WF")"
  touch "$WF"
  echo "wrote $WF"
fi
exit $FAIL
