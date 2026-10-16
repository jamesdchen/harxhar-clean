#!/bin/bash
# xgbtop on Hoffman2 (2026-10-07, coordinator: CARC is fair-share limited, Hoffman2 idle): the per-bar XGBoost runs of
# cluster/slurm/submit_trees_xgbtop.sh on Hoffman2 AVX-only nodes (intel-E5-2670, intel-E5-2650v2: the classes on which the
# forest reproduced CARC epyc-7513 byte for byte, results/h2check_rfdev/compare_by_class.csv), with CARC's own Python stack
# (agent D's runtime in RUNTIME_ROOT, read only; glibc 2.28 and no shared object from outside the runtime, checked in
# process) -- the twin of cluster/trees_rfprev_h2_task.sh for XGBoost.  Every task REFUSES to run on a CPU whose model is
# not in ALLOW_CPU (default: the two AVX-only models) and records the node's CPU model.
#   MODE gate    experiments/trees_xgbtop_gate.py on GATE_LINES (cluster/trees_xgbtop_h2_gate_lines.txt: the CARC gate's
#                lines with the reference chunks copied under ./ref) through specs/causal_tune_trees_xgb06.py (xgb_opt_06):
#                every recorded array of the stored CARC epyc-7513 chunks byte for byte, + the published table cells
#                (ref_tables/) -> $OUT (GATE_OK only when every line matched)
#   MODE chunk   line $SGE_TASK_ID of TASKFILE ("<pack> <rung> <bucket> xgb <tw> <seg> <chunk> <start> <end> <halo>") through
#                the generated spec specs/causal_tune_trees_$TOKEN.py (xgb_opt_00), WINDOW_MASK=$MASK, one thread, into
#                $RESULTS_ROOT/<rung>/<bucket>/<seg>/xgb/tw<TW>/chunks/c<chunk> (+ DONE, run.log, NODE: host / CPU model);
#                the npz must record window_mask MASK and exactly the configuration's per-bar parameters
#   MODE optuna  line $SGE_TASK_ID of TASKFILE (an Optuna stage-1 line, "<bucket> xgb <tw> <seg> <chunk> <start> <end>
#                <halo>") through specs/causal_tune_trees_optuna_xgbtop.py with a spawned pool of NSLOTS single-threaded
#                workers into $RESULTS_ROOT/stage1/<bucket>/<seg>/xgb/tw<TW>/chunks/c<chunk>; verified by
#                experiments/trees_xgbtop_optuna_gate.py --verify (the CARC pack's check) before DONE
#   MODE tuned   line $SGE_TASK_ID of TASKFILE (a random-search chunk-arm) with RUNG rs10 | rs1 through
#                specs/causal_tune_trees_tuned_xgbtop.py, a spawned pool of NSLOTS workers, into $RESULTS_ROOT/$RUNG/...
#   MODE merge   experiments/trees_cadence_reduce_chunks.py for t10 / t1 over TASKFILE under RESULTS_ROOT + the CPU census
#                (no CPU pin needed)
# Runtime: CARC's own Python stack; the node's CPU model is recorded (NODE).
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
MODE="${MODE:?MODE must be gate or chunk}"
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx|avx2|fma|avx512f' | tr '\n' ',')
case ",$FLAGS" in *,avx512f,*) CLASS=AVX-512 ;; *,avx2,*) CLASS=AVX2 ;; *) CLASS=AVX ;; esac
echo "task $TASK mode $MODE on $(hostname) [$CPU | $FLAGS | $CLASS], $NT slots $(date)"
ALLOW_CPU=${ALLOW_CPU:-"E5-2670 0 @|E5-2650 v2 @"}
if [ "$MODE" != merge ] && ! printf '%s\n' "$CPU" | grep -qE "$ALLOW_CPU"; then
  echo "CPU '$CPU' is not an allowed class ($ALLOW_CPU); refusing to run"
  exit 1
fi
"$PY" - "$RT" <<'EOF' || { echo "runtime check FAILED; refusing to run"; exit 1; }
import sys, ctypes
import numpy, scipy, pandas, sklearn, lightgbm, xgboost, shap, numba, sklearn.ensemble
libc = ctypes.CDLL(None); libc.gnu_get_libc_version.restype = ctypes.c_char_p
g = libc.gnu_get_libc_version().decode()
maps = {l.split()[-1] for l in open("/proc/self/maps") if l.split()[-1].startswith("/") and ".so" in l}
out = sorted(p for p in maps if not p.startswith(sys.argv[1] + "/"))
print("runtime: glibc", g, "|", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, scipy, pandas, sklearn, lightgbm, xgboost, shap, numba)),
      f"| {len(maps)} shared objects, outside the runtime: {out if out else 'none'}")
sys.exit(0 if g == "2.28" and not out else 1)
EOF
node_tag() {  # $1 dir
  { echo "host $(hostname -s)"; echo "class $CLASS"; echo "cpu $CPU"; echo "job ${JOB_ID:-0}.${TASK}"; } > "$1/NODE"
}

case "$MODE" in
  gate)
    OUT="${OUT:?OUT}"
    mkdir -p "$OUT"
    node_tag "$OUT"
    "$PY" experiments/trees_xgbtop_make_specs.py --config cluster/xgbtop_shipped_xgb.json --token xgb06 --check
    "$PY" -u experiments/trees_xgbtop_gate.py --lines "${GATE_LINES:?GATE_LINES}" --out "$OUT" \
      --spec specs/causal_tune_trees_xgb06.py --tables ref_tables --procs "$NT"
    ;;
  chunk)
    R="${RESULTS_ROOT:?RESULTS_ROOT}"
    TOKEN="${TOKEN:-xgbtop}"
    XGB_CONFIG="${XGB_CONFIG:-cluster/xgbtop_xgb_config.json}"
    MASK="${MASK:-1}"
    "$PY" experiments/trees_xgbtop_make_specs.py --config "$XGB_CONFIG" --token "$TOKEN" --check
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
    [ -n "${HALO:-}" ] && [ "$MODEL" = xgb ] || { echo "malformed or non-xgb line $TASK of $TASKFILE: '$LINE'"; exit 1; }
    case "$RUNG" in t10) RE=10; IE=1 ;; t1) RE=1; IE=10 ;; *) echo "unknown rung '$RUNG'"; exit 1 ;; esac
    OUTD="$R/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    [ -f "$OUTD/DONE" ] && { echo "skip (DONE) $OUTD"; exit 0; }
    mkdir -p "$OUTD"
    a=$(date +%s)
    SLURM_CPUS_PER_TASK=1 \
      HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
      HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
      HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
      HPC_KW_REFIT_EVERY="$RE" HPC_KW_IMPORTANCE_EVERY="$IE" HPC_KW_SHAP_SEGMENTS="$SEG" \
      HPC_KW_WINDOW_MASK="$MASK" HPC_RESULT_DIR="$OUTD" \
      "$PY" -u "specs/causal_tune_trees_$TOKEN.py" > "$OUTD/run.log" 2>&1
    "$PY" -c "import json, sys, numpy as np; sys.path.insert(0, 'experiments'); import trees_xgbtop_make_specs as mk; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); mask = int(sys.argv[3]); sys.exit(0 if m.get('model') == 'xgb' and m.get('window_mask') == mask and ('kept_n' in z.files) == (mask == 1) and m.get('params') == json.loads(json.dumps(mk.expected_params(sys.argv[2], int(sys.argv[4])))) else 1)" \
      "$OUTD/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz" "$XGB_CONFIG" "$MASK" "$TW"
    node_tag "$OUTD"
    echo "seconds $(( $(date +%s) - a ))" >> "$OUTD/NODE"
    touch "$OUTD/DONE"
    echo "done $RUNG $BUCKET c$CHUNK [$START, $END) in $(( $(date +%s) - a ))s"
    ;;
  optuna)  # one stage-1 line ("<bucket> xgb <tw> <seg> <chunk> <start> <end> <halo>") of TASKFILE, a pool of NSLOTS
           # SPAWNED single-threaded workers (cluster/slurm/optuna_xgbtop_pack.sbatch's run_spec), into
           # $RESULTS_ROOT/stage1/<bucket>/<seg>/xgb/tw<TW>/chunks/c<chunk>; verified as the CARC pack verifies
    R="${RESULTS_ROOT:?RESULTS_ROOT}"
    TOKEN="${TOKEN:-xgbtop}"
    XGB_CONFIG="${XGB_CONFIG:-cluster/xgbtop_xgb_config.json}"
    export HPC_KW_TREE_SPEC="specs/causal_tune_trees_$TOKEN.py"
    "$PY" experiments/trees_xgbtop_make_specs.py --config "$XGB_CONFIG" --token "$TOKEN" --check
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
    [ -n "${HALO:-}" ] && [ "$MODEL" = xgb ] || { echo "malformed or non-xgb line $TASK of $TASKFILE: '$LINE'"; exit 1; }
    OUTD="$R/stage1/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    [ -f "$OUTD/DONE" ] && { echo "skip (DONE) $OUTD"; exit 0; }
    mkdir -p "$OUTD"
    a=$(date +%s)
    SLURM_CPUS_PER_TASK="$NT" HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
      HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
      HPC_KW_STAGE=tune HPC_KW_N_TRIALS="${N_TRIALS:-50}" HPC_KW_IDENTITY=0 HPC_KW_RESUME=1 \
      HPC_KW_STAGE1_TRIALS="" HPC_RESULT_DIR="$OUTD" HPC_KW_WINDOW_MASK=1 \
      "$PY" -u specs/causal_tune_trees_optuna_xgbtop.py > "$OUTD/run.log" 2>&1
    "$PY" experiments/trees_xgbtop_optuna_gate.py --verify "$OUTD/causal_tune_trees/$MODEL/$BUCKET/trials_$SEG.npz" \
      --config "$XGB_CONFIG" --token "$TOKEN" > "$OUTD/verify.log" 2>&1
    node_tag "$OUTD"
    echo "seconds $(( $(date +%s) - a ))" >> "$OUTD/NODE"
    touch "$OUTD/DONE"
    echo "done stage 1 $BUCKET c$CHUNK [$START, $END) in $(( $(date +%s) - a ))s"
    ;;
  tuned)  # one random-search chunk-arm ("<bucket> xgb <tw> <seg> <chunk> <start> <end> <halo>") of TASKFILE, RUNG rs10 | rs1,
          # a spawned pool of NSLOTS single-threaded workers (cluster/slurm/treestuned_xgbtop_pack.sbatch's call), into
          # $RESULTS_ROOT/$RUNG/<bucket>/<seg>/xgb/tw<TW>/chunks/c<chunk>; the npz checked as the CARC pack checks it
    R="${RESULTS_ROOT:?RESULTS_ROOT}"
    TOKEN="${TOKEN:-xgbtop}"
    XGB_CONFIG="${XGB_CONFIG:-cluster/xgbtop_xgb_config.json}"
    export HPC_KW_TREE_SPEC="causal_tune_trees_$TOKEN.py"
    "$PY" experiments/trees_xgbtop_make_specs.py --config "$XGB_CONFIG" --token "$TOKEN" --check
    case "${RUNG:-}" in rs10) RE=10; SHAPSEG=bar1600 ;; rs1) RE=1; SHAPSEG=none ;; *) echo "RUNG must be rs10 or rs1"; exit 1 ;; esac
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
    [ -n "${HALO:-}" ] && [ "$MODEL" = xgb ] || { echo "malformed or non-xgb line $TASK of $TASKFILE: '$LINE'"; exit 1; }
    OUTD="$R/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    [ -f "$OUTD/DONE" ] && { echo "skip (DONE) $OUTD"; exit 0; }
    mkdir -p "$OUTD"
    a=$(date +%s)
    SLURM_CPUS_PER_TASK="$NT" HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
      HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
      HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
      HPC_KW_REFIT_EVERY="$RE" HPC_KW_SHAP_SEGMENTS="$SHAPSEG" \
      HPC_KW_TUNE_IDENTITY=0 HPC_KW_QSEL=1 HPC_KW_WINDOW_MASK=1 HPC_RESULT_DIR="$OUTD" \
      "$PY" -u specs/causal_tune_trees_tuned_xgbtop.py > "$OUTD/run.log" 2>&1
    "$PY" -c "import json, sys, numpy as np; sys.path.insert(0, 'experiments'); import trees_xgbtop_make_specs as mk; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); sys.exit(0 if m.get('model') == 'xgb' and m.get('window_mask') == 1 and 'kept_n' in z.files and m.get('params') == json.loads(json.dumps(mk.expected_params(sys.argv[2], int(sys.argv[3])))) else 1)" \
      "$OUTD/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz" "$XGB_CONFIG" "$TW"
    node_tag "$OUTD"
    echo "seconds $(( $(date +%s) - a ))" >> "$OUTD/NODE"
    touch "$OUTD/DONE"
    echo "done $RUNG $BUCKET c$CHUNK [$START, $END) in $(( $(date +%s) - a ))s"
    ;;
  merge)  # the fixed-settings chunks of TASKFILE under RESULTS_ROOT: the campaign's unedited reducer + the CPU census
    R="${RESULTS_ROOT:?RESULTS_ROOT}"
    for RUNG in t10 t1; do
      mkdir -p "$R/$RUNG"
      "$PY" -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$R/$RUNG" --tasks "${TASKFILE:?TASKFILE}" \
        > "$R/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$R/$RUNG/reduce.log")"
      echo "$RUNG: $(grep -c '^merged (' "$R/$RUNG/reduce.log" || true) merged; $(tail -1 "$R/$RUNG/reduce.log")"
    done
    { echo "chunk_dir,host,class,cpu"
      for X in $(find "$R" -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
        echo "${X#"$R"/},$(awk '$1 == "host" {print $2}' "$X/NODE"),$(awk '$1 == "class" {print $2}' "$X/NODE"),\"$(sed -n 's/^cpu //p' "$X/NODE")\""
      done; } > "$R/class_census.csv"
    echo "census: $(tail -n +2 "$R/class_census.csv" | cut -d'"' -f2 | sort | uniq -c | tr '\n' ' ')"
    touch "$R/MERGE_DONE"
    ;;
  *) echo "MODE must be gate, chunk, optuna, tuned or merge"; exit 1 ;;
esac
