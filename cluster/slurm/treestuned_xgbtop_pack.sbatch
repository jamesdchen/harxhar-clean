#!/bin/bash
# xgbtop (2026-10-07): cluster/slurm/treestuned_rfp_pack.sbatch for XGBoost -- every chunk-arm runs
# specs/causal_tune_trees_tuned_xgbtop.py (the tuned spec, unchanged, with the shipped XGBoost of
# specs/causal_tune_trees_xgbtop.py = xgb_opt_00; its grid is built from the shipped values, so it is re-centred on
# xgb_opt_00 by the spec's own rule) instead of the tuned spec; before running, the generated spec must equal what
# experiments/trees_xgbtop_make_specs.py writes from $XGB_CONFIG (--check); a finished chunk's npz must record
# window_mask 1, the kept columns and exactly the configuration's per-bar parameters (meta params == expected_params)
# before DONE.  XGBoost lines only.  The masked tuned pack's text follows.
# One chunk-arm of the TUNED per-bar tree rungs RS10 / RS1 (random search over 32 fixed candidates every TUNE_PER = 250
# sessions) with WINDOW_MASK=1.  The rung's cadence is set per job:
#   RUNG rs10: REFIT_EVERY 10, TreeSHAP at 16:00
#   RUNG rs1 : REFIT_EVERY 1,  TreeSHAP OFF (SHAP_SEGMENTS=none); native importance at every refit
# TASKFILE holds one "<bucket> <model> <train_win_days> <seg> <chunk> <start> <end> <halo>" line per array task (chunk =
# one tuning period; start - halo a multiple of 250), run into $RESULTS_ROOT/$RUNG/<bucket>/<seg>/<model>/tw<TW>/chunks/
# c<chunk> (experiments/reduce_trees_tuned_chunks.py merges an arm's chunks).  DONE = skip.
# Parallelism is across PROCESSES: the spec runs a pool of SLURM_CPUS_PER_TASK SPAWNED workers (never forked), every fit
# single-threaded (MODEL_THREADS = 1), so no number depends on the pool size; BLAS/OpenMP/numba threads pinned to 1.
# Exported at submit time: TASKFILE, RUNG, RESULTS_ROOT, CANARY_FLAG, WRITE_FLAG, XGB_CONFIG (default
# cluster/xgbtop_xgb_config.json), TOKEN (default xgbtop).
# CPU CLASS: the submit script pins every task (--constraint) and every finished chunk records its node in NODE.
#SBATCH --job-name=treestuned_xgbtop
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
TOKEN="${TOKEN:-xgbtop}"
if [ "$TOKEN" = xgbtop ]; then XGB_CONFIG="${XGB_CONFIG:-cluster/xgbtop_xgb_config.json}"; else XGB_CONFIG="${XGB_CONFIG:?XGB_CONFIG must name the configuration of token $TOKEN}"; fi
export HPC_KW_TREE_SPEC="causal_tune_trees_$TOKEN.py"
echo "xgbtop tuned: TREE_SPEC specs/$HPC_KW_TREE_SPEC from $XGB_CONFIG"

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
if ! python experiments/trees_xgbtop_make_specs.py --config "$XGB_CONFIG" --token "$TOKEN" --check; then
  echo "specs/causal_tune_trees_$TOKEN.py is not the file generated from $XGB_CONFIG; refusing to run"
  exit 1
fi

case "${RUNG:-}" in
  rs10) RE=10; SHAPSEG=bar1600 ;;
  rs1) RE=1; SHAPSEG=none ;;
  *) echo "RUNG must be rs10 or rs1, got '${RUNG:-}'"; exit 1 ;;
esac
ROOT="${RESULTS_ROOT:?RESULTS_ROOT}/$RUNG"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET MODELS TW SEG CHUNK START END HALO <<< "$LINE"
if [ -z "${HALO:-}" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE (want 8 fields): '$LINE'"
  exit 1
fi
[ "$MODELS" = xgb ] || { echo "line model '$MODELS': this pack runs XGBoost only; refusing"; exit 1; }
NT="${SLURM_CPUS_PER_TASK:-1}"
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')], pool of $NT: $RUNG (refit_every $RE, shap $SHAPSEG) $BUCKET [$MODELS] tw$TW $SEG chunk $CHUNK rows [$START, $END) halo $HALO  $(date)"
node_tag() {  # $1 = a finished chunk dir: the node it ran on and that node's CPU features (the class)
  { echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null \
      | grep -o "AvailableFeatures=[^ ]*"; echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"; } > "$1/NODE"
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
     python -u specs/causal_tune_trees_tuned_xgbtop.py > "$OUT/run.log" 2>&1 \
     && python -c "import json, sys, numpy as np; sys.path.insert(0, 'experiments'); import trees_xgbtop_make_specs as mk; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); sys.exit(0 if m.get('model') == 'xgb' and m.get('window_mask') == 1 and 'kept_n' in z.files and m.get('params') == json.loads(json.dumps(mk.expected_params(sys.argv[2], int(sys.argv[3])))) else 1)" \
          "$OUT/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz" "$XGB_CONFIG" "$TW"; then
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
