#!/bin/bash
# Feature MCS, the tree candidates (2026-10-08): one PACK of chunk processes, every line of pack PACK of TASKFILE at once, one
# single-threaded python process per line (SLURM_CPUS_PER_TASK=1 per process = the models' thread count, as in the stored
# T1 runs, cluster/slurm/trees_mask_pack.sbatch).  A line
#   "<pack> <est> <design> <chunk> <start> <end> <halo> <planned seconds>"
# runs specs/causal_tune_trees_mcs.py (a shipped tree spec, unchanged but for the design's columns and the listed rows) with
# the stored T1 runs' env axes: EXOG_BUCKET all_features (the design's columns are passed on), SEGMENT bar1600, TRAIN_WIN
# 2000, LAG_SCOPE global, HAR_BASE "", REFIT_EVERY 1, WINDOW_MASK 1; native importance and TreeSHAP off (IMPORTANCE_EVERY
# 1e9, SHAP_SEGMENTS none: neither touches a forecast); rows = cluster/feature_mcs_trees_rows.json (the 866 MCS trade days).
#   est lgbm: specs/causal_tune_trees.py (lgbm_opt_00) | xgb: specs/causal_tune_trees_xgbtop.py | rf: specs/causal_tune_trees_rf400.py
# Output: $RESULTS_ROOT/<est>/<design>/c<chunk> (+ DONE, NODE = host and CPU features, CPU = the CPU model, TIME = real user
# sys seconds of the process); a chunk with DONE is skipped, so a resubmitted pack resumes.  Before DONE the npz meta must
# name the design and the model with window_mask 1.  Exported at submit time: TASKFILE, RESULTS_ROOT, PACK (default the
# array index), CANARY_FLAG (files that must exist), REQUIRE_CPU (a substring of the CPU model, e.g. "EPYC 7513").
#SBATCH --job-name=mct_pack
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
TASKFILE="${TASKFILE:?TASKFILE}"
ROOT="${RESULTS_ROOT:-results/feature_mcs_trees_work/chunks}"
PACK="${PACK:-${SLURM_ARRAY_TASK_ID:?PACK or an array index}}"
STAGGER="${STAGGER:-15}"
for f in ${CANARY_FLAG:-}; do
  [ -f "$f" ] || { echo "flag $f missing; refusing to run"; exit 1; }
done
CPUMODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
if [ -n "${REQUIRE_CPU:-}" ]; then  # one substring, or several separated by '|' (e.g. "EPYC 7513|EPYC 7542")
  CPU_OK=0
  IFS='|' read -r -a WANT_CPU <<< "$REQUIRE_CPU"
  for w in "${WANT_CPU[@]}"; do
    case "$CPUMODEL" in *"$w"*) CPU_OK=1 ;; esac
  done
  [ "$CPU_OK" = 1 ] || { echo "CPU '$CPUMODEL' is not one of '$REQUIRE_CPU'; refusing to run"; exit 1; }
fi
python -c "import lightgbm, xgboost, sklearn; print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"
python experiments/trees_xgbtop_make_specs.py --config cluster/xgbtop_xgb_config.json --token xgbtop --check
python experiments/trees_rfp163_make_specs.py --config cluster/rf400_rf_config.json --token rf400 --check
NCPU="${SLURM_CPUS_PER_TASK:-1}"
mapfile -t LINES < <(awk -v p="$PACK" 'NF && $1 == p' "$TASKFILE")
if [ "${#LINES[@]}" -eq 0 ] || [ "${#LINES[@]}" -gt "$NCPU" ]; then
  echo "pack $PACK of $TASKFILE has ${#LINES[@]} lines for $NCPU cores; refusing to run"
  exit 1
fi
echo "pack $PACK of $TASKFILE on $(hostname) [$CPUMODEL]: ${#LINES[@]} processes on $NCPU cores  $(date)"

run_line() {
  local PK EST DESIGN CHUNK START END HALO SEC MODEL BASE OUT T0
  read -r PK EST DESIGN CHUNK START END HALO SEC <<< "$1"
  case "$EST" in
    lgbm) MODEL=lgbm; BASE=specs/causal_tune_trees.py ;;
    xgb) MODEL=xgb; BASE=specs/causal_tune_trees_xgbtop.py ;;
    rf) MODEL=rf; BASE=specs/causal_tune_trees_rf400.py ;;
    *) echo "  unknown estimator '$EST'"; return 1 ;;
  esac
  OUT="$ROOT/$EST/$DESIGN/c$CHUNK"
  if [ -f "$OUT/DONE" ]; then
    echo "  skip (DONE) $EST $DESIGN c$CHUNK"
    return 0
  fi
  mkdir -p "$OUT"
  T0=$SECONDS
  if ( TIMEFORMAT='%R %U %S'; time ( SLURM_CPUS_PER_TASK=1 \
         HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET=all_features HPC_KW_SEGMENT=bar1600 \
         HPC_KW_TRAIN_WIN=2000 HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
         HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
         HPC_KW_REFIT_EVERY=1 HPC_KW_IMPORTANCE_EVERY=1000000000 HPC_KW_SHAP_SEGMENTS=none HPC_KW_WINDOW_MASK=1 \
         HPC_KW_MCS_DESIGN="$DESIGN" HPC_KW_MCS_BASE="$BASE" \
         HPC_KW_MCS_DESIGNS=cluster/feature_mcs_trees_designs.json HPC_KW_MCS_ROWS=cluster/feature_mcs_trees_rows.json \
         HPC_RESULT_DIR="$OUT" \
         python -u specs/causal_tune_trees_mcs.py > "$OUT/run.log" 2>&1 ) ) 2> "$OUT/TIME" \
     && python -c "import json, sys, numpy as np; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); sys.exit(0 if m.get('mcs_design') == sys.argv[2] and m.get('model') == sys.argv[3] and m.get('window_mask') == 1 and m.get('refit_every') == 1 else 1)" \
          "$OUT/causal_tune_trees/$MODEL/all_features/trees_bar1600.npz" "$DESIGN" "$MODEL"; then
    { echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null | grep -o "AvailableFeatures=[^ ]*"; } > "$OUT/NODE"
    echo "$CPUMODEL" > "$OUT/CPU"
    touch "$OUT/DONE"
    echo "  done $EST $DESIGN c$CHUNK [$START, $END) in $((SECONDS - T0)) s (planned $SEC)  $(date +%T)"
  else
    echo "  FAILED $EST $DESIGN c$CHUNK: $(tail -1 "$OUT/run.log" | cut -c1-200)"
    return 1
  fi
}

PIDS=()
n=0
for L in "${LINES[@]}"; do
  run_line "$L" &
  PIDS+=("$!")
  n=$((n + 1))
  if [ "$n" -lt "${#LINES[@]}" ]; then sleep "$STAGGER"; fi
done
FAIL=0
for p in "${PIDS[@]}"; do
  wait "$p" || FAIL=1
done
echo "pack $PACK: processes finished, fail=$FAIL  $(date)"
exit $FAIL
