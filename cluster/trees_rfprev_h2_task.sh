#!/bin/bash
# Dev-forest PREVIEW (2026-10-07): the per-bar 16:00 random forest with the 400-trial search's current
# leader (trial 163, cluster/rfprev163_rf_config.json), T1 / T10 on baseline, all_features, lean_vol_vix,
# on Hoffman2 (SGE) through specs/causal_tune_trees_rfdev.py -- the twin of cluster/slurm/trees_rfdev_pack.sbatch's
# run_line (same HPC_KW_* axes, WINDOW_MASK=1, one thread per fit, seed 42), one array task = one chunk line.
#   MODE gate     experiments/trees_rfprev_gate.py on GATE_LINES with the rf_opt_00 configuration
#                 (HPC_KW_RF_CONFIG=cluster/rfdev_rf_config.json) against the CARC rfdev chunks in ./ref
#                 -> $OUT (gate_lines.csv / gate_rows.csv: identical or the max relative difference)
#   MODE chunk    (WINDOW_MASK=0: the unmasked forest; REQUIRE_CPU: refuse other CPU models) line $SGE_TASK_ID of TASKFILE ("<pack> <rung> <bucket> rf <tw> <seg> <chunk> <start> <end> <halo>")
#                 with HPC_KW_RF_CONFIG (the preview configuration) into $RESULTS_ROOT/<rung>/<bucket>/<seg>/rf/
#                 tw<TW>/chunks/c<chunk> (+ DONE, run.log, H2_HOST); the npz must record window_mask 1, kept_n and
#                 exactly the configuration's parameters.  REPEAT=1 (the canary): the line is run a second time
#                 into <dir>_repeat and the two npz must agree bit for bit; then $CANARY_DIR/ok_<task> is written.
#                 CANARY_DIR + CANARY_N (fleet): refuses to run unless CANARY_N ok_* flags exist.
#   MODE tuned    line $SGE_TASK_ID of TASKFILE (the tuned 8-field format), RUNG rs10 | rs1: the re-centred random-search
#                 chunk-arm through specs/causal_tune_trees_tuned_rfp.py (HPC_KW_TREE_SPEC), a pool of $NSLOTS
#   MODE merge    experiments/trees_cadence_reduce_chunks.py for t10 and t1 (TASKFILE), the CPU census of every chunk
# Runtime: CARC's own Python stack (agent D's build in RUNTIME_ROOT, read only); every task checks in-process glibc
# 2.28 and that no shared object comes from outside the runtime; the node's CPU model is recorded in H2_HOST.
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
RT=${RUNTIME_ROOT:-/u/scratch/j/jamesdc1/harxhar-optuna}
PY=$RT/carc_env/harxhar/bin/python3.11
[ -x "$PY" ] || { echo "no runtime at $PY"; exit 1; }
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONSTARTUP PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV
export PATH="$RT/carc_env/harxhar/bin:/usr/bin:/bin" PYTHONUSERBASE="$RT/carc_usersite"
export PYTHONPATH="$D:$D/experiments:$RT/pylib_trees" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
set -u
NT="${NSLOTS:-1}"
MODE="${MODE:?MODE must be gate, chunk, tuned or merge}"
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx|avx2|fma|avx512f' | tr '\n' ',')
case ",$FLAGS" in *,avx512f,*) CLASS=AVX-512 ;; *,avx2,*) CLASS=AVX2 ;; *) CLASS=AVX ;; esac
echo "task $TASK mode $MODE on $(hostname) [$CPU | $FLAGS | $CLASS], $NT slots $(date)"
# REQUIRE_CPU (optional): colon-separated substrings, one of which the CPU model must contain (spaces as underscores, e.g. "E5-2670:E5-2650_v2",
# the AVX models that reproduced CARC's epyc-7513 forest byte for byte); any other node refuses to run.
WM="${WINDOW_MASK:-1}"   # 1 = the masked forest (default); 0 = the unmasked fixed-settings forest
if [ -n "${REQUIRE_CPU:-}" ]; then
  OKCPU=0
  IFS=: read -r -a _REQ <<< "$REQUIRE_CPU"
  for q in "${_REQ[@]}"; do [[ "${CPU// /_}" == *"$q"* ]] && OKCPU=1; done  # CPU model spaces read as underscores
  [ "$OKCPU" = 1 ] || { echo "node $(hostname) CPU '$CPU' is not one of '$REQUIRE_CPU'; refusing to run"; exit 1; }
fi
"$PY" - "$RT" <<'EOF' || { echo "runtime check FAILED; refusing to run"; exit 1; }
import sys, ctypes
import numpy, scipy, pandas, sklearn, lightgbm, xgboost, shap, numba, sklearn.ensemble
libc = ctypes.CDLL(None); libc.gnu_get_libc_version.restype = ctypes.c_char_p
g = libc.gnu_get_libc_version().decode()
maps = {l.split()[-1] for l in open("/proc/self/maps") if l.split()[-1].startswith("/") and ".so" in l}
out = sorted(p for p in maps if not p.startswith(sys.argv[1] + "/"))
print("runtime: glibc", g, "|", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, scipy, pandas, sklearn, shap, numba)),
      f"| {len(maps)} shared objects, outside the runtime: {out if out else 'none'}")
sys.exit(0 if g == "2.28" and not out else 1)
EOF

npz_ok() {  # $1 npz: window_mask 1, kept_n, meta params == the configuration's (the CARC pack's check)
  "$PY" -c "import json, sys, numpy as np; sys.path.insert(0, 'specs'); import causal_tune_trees_rfdev as w; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); sys.exit(0 if m.get('window_mask') == int(sys.argv[3]) and (('kept_n' in z.files) == (int(sys.argv[3]) == 1)) and m.get('params') == json.loads(json.dumps(w.rf_params(sys.argv[2]))) else 1)" "$1" "$HPC_KW_RF_CONFIG" "$WM"
}
run_chunk() {  # "<line>" <out dir>
  local PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO RE IE OUT a
  read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO <<< "$1"
  OUT="$2"
  case "$RUNG" in t10) RE=10; IE=1 ;; t1) RE=1; IE=10 ;; *) echo "unknown rung '$RUNG'"; return 1 ;; esac
  [ -f "$OUT/DONE" ] && { echo "  skip (DONE) $OUT"; return 0; }
  mkdir -p "$OUT"
  a=$(date +%s)
  if SLURM_CPUS_PER_TASK=1 \
     HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
     HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
     HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
     HPC_KW_REFIT_EVERY="$RE" HPC_KW_IMPORTANCE_EVERY="$IE" HPC_KW_SHAP_SEGMENTS="$SEG" \
     HPC_KW_WINDOW_MASK="$WM" HPC_RESULT_DIR="$OUT" \
     "$PY" -u specs/causal_tune_trees_rfdev.py > "$OUT/run.log" 2>&1 \
     && npz_ok "$OUT/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz"; then
    echo "host=$(hostname -s) class=$CLASS cpu=$CPU seconds=$(( $(date +%s) - a )) job=${JOB_ID:-0}.${TASK}" > "$OUT/H2_HOST"
    touch "$OUT/DONE"
    echo "  done $RUNG $BUCKET c$CHUNK [$START, $END) in $(( $(date +%s) - a ))s"
  else
    echo "  FAILED $RUNG $BUCKET c$CHUNK: $(tail -1 "$OUT/run.log" | cut -c1-200)"
    return 1
  fi
}

case "$MODE" in
  gate)
    OUT="${OUT:?OUT}"
    mkdir -p "$OUT"
    echo "host=$(hostname -s) class=$CLASS cpu=$CPU slots=$NT job=${JOB_ID:-0}" > "$OUT/H2_HOST"
    "$PY" -u experiments/trees_rfprev_gate.py --lines "${GATE_LINES:?GATE_LINES}" --ref-root ref --out "$OUT" --pool "$NT" --procs "$NT" || true
    echo "gate finished (bit-identity not required; see $OUT/gate_lines.csv)"
    ;;
  chunk)
    R="${RESULTS_ROOT:?RESULTS_ROOT}"
    [ -f "${HPC_KW_RF_CONFIG:-}" ] || { echo "HPC_KW_RF_CONFIG missing"; exit 1; }
    if [ -n "${CANARY_DIR:-}" ] && [ -n "${CANARY_N:-}" ]; then
      n=$(ls "$CANARY_DIR"/ok_* 2>/dev/null | wc -l)
      [ "$n" -ge "$CANARY_N" ] || { echo "canary: $n of $CANARY_N chunks ok; refusing to run"; exit 1; }
    fi
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
    [ -n "${HALO:-}" ] || { echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1; }
    DIR="$R/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    run_chunk "$LINE" "$DIR"
    if [ "${REPEAT:-0}" = 1 ]; then
      rm -rf "${DIR}_repeat"
      run_chunk "$LINE" "${DIR}_repeat"
      "$PY" - "$DIR/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz" "${DIR}_repeat/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz" <<'EOF'
import sys
import numpy as np
a, b = (np.load(p, allow_pickle=False) for p in sys.argv[1:3])
keys = ("date", "pred_adj", "true_adj", "true_raw", "refit_row", "importance", "shap", "kept_n", "kept")
same = {k: bool(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")) for k in keys if k in a.files or k in b.files}
print("repeat vs first run (bit for bit):", same)
sys.exit(0 if all(same.values()) else 1)
EOF
      mkdir -p "$CANARY_OUT"
      echo "$(hostname -s) $CPU $(date +%T)" > "$CANARY_OUT/ok_$TASK"
    fi
    ;;
  tuned)
    # one chunk-arm of the re-centred random-search rungs, exactly cluster/slurm/treestuned_rfp_pack.sbatch's call:
    # RUNG rs10 (REFIT_EVERY 10, TreeSHAP at bar1600) / rs1 (REFIT_EVERY 1, TreeSHAP off), a pool of $NSLOTS spawned
    # single-threaded workers, TREE_SPEC from HPC_KW_TREE_SPEC (the generated spec of the forest), WINDOW_MASK=1
    R="${RESULTS_ROOT:?RESULTS_ROOT}"
    [ -f "${HPC_KW_RF_CONFIG:-}" ] || { echo "HPC_KW_RF_CONFIG missing"; exit 1; }
    case "${RUNG:-}" in rs10) RE=10; SHAPSEG=bar1600 ;; rs1) RE=1; SHAPSEG=none ;; *) echo "RUNG must be rs10 or rs1"; exit 1 ;; esac
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r BUCKET MODELS TW SEG CHUNK START END HALO <<< "$LINE"
    [ -n "${HALO:-}" ] || { echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1; }
    FAIL=0
    for MODEL in ${MODELS//,/ }; do
      OUT="$R/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
      [ -f "$OUT/DONE" ] && { echo "  skip (DONE) $OUT"; continue; }
      mkdir -p "$OUT"
      a=$(date +%s)
      if SLURM_CPUS_PER_TASK="$NT" HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
         HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
         HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
         HPC_KW_REFIT_EVERY="$RE" HPC_KW_SHAP_SEGMENTS="$SHAPSEG" \
         HPC_KW_TUNE_IDENTITY=0 HPC_KW_QSEL=1 HPC_KW_WINDOW_MASK=1 \
         HPC_RESULT_DIR="$OUT" \
         "$PY" -u specs/causal_tune_trees_tuned_rfp.py > "$OUT/run.log" 2>&1 \
         && npz_ok "$OUT/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz"; then
        echo "host=$(hostname -s) class=$CLASS cpu=$CPU seconds=$(( $(date +%s) - a )) job=${JOB_ID:-0}.${TASK}" > "$OUT/H2_HOST"
        touch "$OUT/DONE"
        echo "  $RUNG $BUCKET $MODEL c$CHUNK done in $(( $(date +%s) - a ))s"
      else
        FAIL=1
        echo "  $RUNG $BUCKET $MODEL c$CHUNK FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
      fi
    done
    exit $FAIL
    ;;
  merge)
    R="${RESULTS_ROOT:?RESULTS_ROOT}"
    for RUNG in t10 t1; do
      mkdir -p "$R/$RUNG"
      "$PY" -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$R/$RUNG" --tasks "${TASKFILE:?TASKFILE}" \
        > "$R/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$R/$RUNG/reduce.log")"
      echo "$RUNG: $(grep -c '^merged (' "$R/$RUNG/reduce.log" || true) merged; $(tail -1 "$R/$RUNG/reduce.log")"
    done
    { echo "rung,bucket,chunk,host,class,cpu,seconds"
      for X in $(find "$R" -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
        REL=${X#"$R"/}; IFS=/ read -r RUNG BUCKET SEG MODEL TWD CH CK <<< "$REL"
        H=$(grep -o 'host=[^ ]*' "$X/H2_HOST" | cut -d= -f2); C=$(grep -o 'class=[^ ]*' "$X/H2_HOST" | cut -d= -f2)
        P=$(sed 's/.*cpu=\(.*\) seconds=.*/\1/' "$X/H2_HOST"); S=$(grep -o 'seconds=[0-9]*' "$X/H2_HOST" | cut -d= -f2)
        echo "$RUNG,$BUCKET,$CK,$H,$C,\"$P\",$S"
      done; } > "$R/class_census.csv"
    echo "census: $(tail -n +2 "$R/class_census.csv" | cut -d, -f5 | sort | uniq -c | tr '\n' ' ')"
    touch "$R/MERGE_DONE"
    ;;
  *) echo "MODE must be gate, chunk or merge"; exit 1 ;;
esac
