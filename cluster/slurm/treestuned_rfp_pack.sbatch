#!/bin/bash
# rfp163 (2026-10-07): cluster/slurm/treestuned_mask_pack.sbatch with ONE change -- every chunk-arm runs
# specs/causal_tune_trees_tuned_rfp.py (the tuned spec with the shipped forest of HPC_KW_TREE_SPEC and its grid
# re-centred on it) instead of the tuned spec; a finished chunk's npz must also record exactly the configuration's
# parameters (meta params == rf_params(HPC_KW_RF_CONFIG)) before DONE.  The masked tuned pack's text follows.
# I9 (2026-09-29 evening): the MASKED twin of cluster/slurm/treestuned_cadence_pack.sbatch --
# one chunk-arm of the TUNED per-bar tree rungs RS10 / RS1 (specs/causal_tune_trees_tuned.py,
# random search over 32 fixed candidates every TUNE_PER = 250 sessions, de-duplicated per-bar
# design of commit 47f7f9c) with WINDOW_MASK=1: a tuning point's fit block, validation tail
# and candidates see the kept columns of its window, every refit those of its own window
# (a finished chunk's npz must record window_mask 1: checked before DONE is written).
# The rung's cadence is set per job:
#   RUNG rs10: REFIT_EVERY 10, TreeSHAP at 16:00 (as the 2026-09-29 tuned campaign)
#   RUNG rs1 : REFIT_EVERY 1,  TreeSHAP OFF (SHAP_SEGMENTS=none): the tuned workers
#              (specs/causal_tune_trees_tuned_jobs.py) do not apply IMPORTANCE_EVERY, so
#              TreeSHAP would run at every one of ~1469 refits; native importance is
#              still recorded at every refit (cheap)
# The spec reads REFIT_EVERY from specs/causal_tune_trees.py's env axis (it execs that
# spec's setup), which reads HPC_KW_REFIT_EVERY.
# TASKFILE holds one
#   "<bucket> <model[,model...]> <train_win_days> <seg> <chunk> <start> <end> <halo>"
# line per array task (chunk = one tuning period; start - halo a multiple of 250); the
# models of a line run one after another, each into
#   $RESULTS_ROOT/$RUNG/<bucket>/<seg>/<model>/tw<TW>/chunks/c<chunk>
# (experiments/reduce_trees_tuned_chunks.py merges an arm's chunks).  DONE = skip.
# Parallelism is across PROCESSES: the spec runs a pool of SLURM_CPUS_PER_TASK spawned
# workers, every fit single-threaded (MODEL_THREADS = 1), so no number depends on the
# pool size (gate g4b); BLAS/OpenMP/numba threads are pinned to 1.
# Exported at submit time: TASKFILE, RUNG, RESULTS_ROOT (default
# results/linear_subsection_trees_tuned_mask), CANARY_FLAG, WRITE_FLAG.
# CPU CLASS (2026-09-29 late): forecasts are bit-reproducible only within a CPU vector class
# (AVX-512 xeon vs AVX2 epyc move the design at ~1e-11 and LightGBM's bins with it), so the
# submit script pins every task to one class (--constraint, default epyc-7513) and every finished
# chunk records its node and CPU features in NODE beside DONE.
#SBATCH --job-name=treestuned_rfp
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
[ -n "${HPC_KW_RF_CONFIG:-}" ] && [ -f "$HPC_KW_RF_CONFIG" ] || { echo "HPC_KW_RF_CONFIG missing; refusing to run"; exit 1; }
echo "rfp tuned: TREE_SPEC ${HPC_KW_TREE_SPEC:-causal_tune_trees_rfp163.py}, configuration check $HPC_KW_RF_CONFIG"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"

if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi
if ! python -c "import shap, lightgbm, xgboost, sklearn; print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"; then
  echo "tree stack import failed (PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi

case "${RUNG:-}" in
  rs10) RE=10; SHAPSEG=bar1600 ;;
  rs1) RE=1; SHAPSEG=none ;;
  *) echo "RUNG must be rs10 or rs1, got '${RUNG:-}'"; exit 1 ;;
esac
ROOT="${RESULTS_ROOT:-results/linear_subsection_trees_tuned_mask}/$RUNG"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET MODELS TW SEG CHUNK START END HALO <<< "$LINE"
if [ -z "${HALO:-}" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE (want 8 fields): '$LINE'"
  exit 1
fi
NT="${SLURM_CPUS_PER_TASK:-1}"
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname), pool of $NT: $RUNG (refit_every $RE, shap $SHAPSEG) $BUCKET [$MODELS] tw$TW $SEG chunk $CHUNK rows [$START, $END) halo $HALO  $(date)"
node_tag() {  # $1 = a finished chunk dir: the node it ran on and that node's CPU features (the class)
  { echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null \
      | grep -o "AvailableFeatures=[^ ]*"; } > "$1/NODE"
}

FAIL=0
for MODEL in ${MODELS//,/ }; do
  OUT="$ROOT/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
  [ -f "$OUT/DONE" ] && { echo "  skip (DONE) $MODEL c$CHUNK"; continue; }
  mkdir -p "$OUT"
  T0=$SECONDS
  if HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
     HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
     HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
     HPC_KW_REFIT_EVERY="$RE" HPC_KW_SHAP_SEGMENTS="$SHAPSEG" \
     HPC_KW_TUNE_IDENTITY=0 HPC_KW_QSEL=1 HPC_KW_WINDOW_MASK=1 \
     HPC_RESULT_DIR="$OUT" \
     python -u specs/causal_tune_trees_tuned_rfp.py > "$OUT/run.log" 2>&1 \
     && python -c "import json, sys, numpy as np; sys.path.insert(0, 'specs'); import causal_tune_trees_rfdev as w; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); sys.exit(0 if m.get('window_mask') == 1 and 'kept_n' in z.files and m.get('params') == json.loads(json.dumps(w.rf_params(sys.argv[2]))) else 1)" \
          "$OUT/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz" "$HPC_KW_RF_CONFIG"; then
    node_tag "$OUT"
    touch "$OUT/DONE"
    echo "  $MODEL c$CHUNK done in $((SECONDS - T0))s  $(date +%T)"
  else
    FAIL=1
    echo "  $MODEL c$CHUNK FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
  fi
done
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
exit $FAIL
