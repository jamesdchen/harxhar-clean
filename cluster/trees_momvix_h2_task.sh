#!/bin/bash
# moments_vix tree workstream (2026-10-06): the Hoffman2 (SGE) twin of the CARC pack scripts of the
# masked 16:00 tree rungs.  One array task runs, with the CARC call (the same HPC_KW_* axes), one of
#   MODE gate     experiments/trees_momvix_gate.py on GATE_LINES against the stored masked chunks
#                 relayed into ./ref (REF_ROOT=ref), pool of $NSLOTS  -> $OUT (GATE_OK, PASS_*)
#   MODE untuned  line $SGE_TASK_ID of TASKFILE ("<pack> <rung> <bucket> <model> <tw> <seg> <chunk>
#                 <start> <end> <halo>"): one single-threaded chunk process, exactly
#                 cluster/slurm/trees_mask_pack.sbatch's run_line (t10: REFIT_EVERY 10 /
#                 IMPORTANCE_EVERY 1; t1: 1 / 10; TreeSHAP at the line's segment; WINDOW_MASK=1)
#   MODE tuned    line $SGE_TASK_ID of TASKFILE ("<bucket> <model[,..]> <tw> <seg> <chunk> <start>
#                 <end> <halo>"), RUNG rs10 | rs1: exactly cluster/slurm/treestuned_mask_pack.sbatch
#                 (pool of $NSLOTS spawned single-threaded workers)
# into the CARC layout ($RESULTS_ROOT/<rung>/<bucket>/<seg>/<model>/tw<TW>/chunks/c<chunk> + DONE,
# run.log, H2_HOST), so a chunk run here is relayed into the CARC results tree and merged there.
# A finished chunk's npz must record window_mask 1 and kept_n before DONE is written (as on CARC).
# Runtime: CARC's own Python stack (agent D's cluster/optuna_h2_runtime_setup.sh build in
# RUNTIME_ROOT, used read only); code and data from this root.  Every task checks in-process glibc
# 2.28 and no shared object from outside RUNTIME_ROOT, and refuses a node whose CPU model lacks
# REQUIRE_CPU (the class the gate qualified).  REQUIRE_FLAGS: colon-separated files that must exist (e.g. PASS_*).
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
export PYTHONPATH="$D:$RT/pylib_trees" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
set -u
NT="${NSLOTS:-1}"
MODE="${MODE:?MODE must be gate, untuned or tuned}"
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
RF="${REQUIRE_FLAGS:-}"
for f in ${RF//:/ }; do
  [ -f "$f" ] || { echo "required flag $f missing; refusing to run"; exit 1; }
done
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx|avx2|fma|avx512f|avx512dq|avx512bw|avx512vl' | tr '\n' ',')
case ",$FLAGS" in *,avx512f,*) CLASS=AVX-512 ;; *,avx2,*) CLASS=AVX2 ;; *) CLASS=AVX ;; esac
if [ -n "${REQUIRE_CPU:-}" ] && [[ "${CPU// /_}" != *"$REQUIRE_CPU"* ]]; then
  echo "node $(hostname) CPU '$CPU' is not the qualified class '$REQUIRE_CPU'; refusing to run"
  exit 1
fi
echo "task $TASK mode $MODE on $(hostname) [$CPU | $FLAGS | $CLASS], $NT slots $(date)"
"$PY" - "$RT" <<'EOF' || { echo "runtime check FAILED; refusing to run"; exit 1; }
import sys, ctypes
import numpy, scipy, pandas, sklearn, lightgbm, xgboost, shap, numba, sklearn.ensemble, scipy.special
libc = ctypes.CDLL(None); libc.gnu_get_libc_version.restype = ctypes.c_char_p
g = libc.gnu_get_libc_version().decode()
maps = {l.split()[-1] for l in open("/proc/self/maps") if l.split()[-1].startswith("/") and ".so" in l}
out = sorted(p for p in maps if not p.startswith(sys.argv[1] + "/"))
print("runtime: glibc", g, "|", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, scipy, pandas, sklearn, lightgbm, xgboost, shap, numba)),
      f"| {len(maps)} shared objects, outside the runtime: {out if out else 'none'}")
sys.exit(0 if g == "2.28" and not out else 1)
EOF
host_tag() {  # $1 = a finished chunk dir, $2 = seconds
  echo "host=$(hostname -s) class=$CLASS cpu=$CPU seconds=$2 slots=$NT job=${JOB_ID:-0}.${TASK}" > "$1/H2_HOST"
}
npz_masked() {  # $1 = the chunk's npz: window_mask 1 and kept_n recorded (the CARC packs' check)
  "$PY" -c "import json, sys, numpy as np; z = np.load(sys.argv[1]); sys.exit(0 if json.loads(str(z['meta'])).get('window_mask') == 1 and 'kept_n' in z.files else 1)" "$1"
}

case "$MODE" in
  gate)
    OUT="${OUT:?OUT}"
    mkdir -p "$OUT"
    echo "host=$(hostname -s) class=$CLASS cpu=$CPU slots=$NT job=${JOB_ID:-0}" > "$OUT/H2_HOST"
    "$PY" -u experiments/trees_momvix_gate.py --lines "${GATE_LINES:?GATE_LINES}" --ref-root "${REF_ROOT:-ref}" \
      --out "$OUT" --pool "$NT" --procs "$NT"
    ;;
  untuned)
    ROOT="${RESULTS_ROOT:?RESULTS_ROOT}"
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
    [ -n "${HALO:-}" ] || { echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1; }
    case "$RUNG" in t10) RE=10; IE=1 ;; t1) RE=1; IE=10 ;; *) echo "unknown rung '$RUNG'"; exit 1 ;; esac
    OUT="$ROOT/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    [ -f "$OUT/DONE" ] && { echo "  skip (DONE) $RUNG $BUCKET $MODEL c$CHUNK"; exit 0; }
    mkdir -p "$OUT"
    a=$(date +%s)
    if SLURM_CPUS_PER_TASK=1 \
       HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
       HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
       HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
       HPC_KW_REFIT_EVERY="$RE" HPC_KW_IMPORTANCE_EVERY="$IE" HPC_KW_SHAP_SEGMENTS="$SEG" \
       HPC_KW_WINDOW_MASK=1 \
       HPC_RESULT_DIR="$OUT" \
       "$PY" -u specs/causal_tune_trees.py > "$OUT/run.log" 2>&1 \
       && npz_masked "$OUT/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz"; then
      host_tag "$OUT" $(( $(date +%s) - a ))
      touch "$OUT/DONE"
      echo "  done $RUNG $BUCKET $MODEL c$CHUNK [$START, $END) in $(( $(date +%s) - a ))s  $(date +%T)"
    else
      echo "  FAILED $RUNG $BUCKET $MODEL c$CHUNK: $(tail -1 "$OUT/run.log" | cut -c1-200)"
      exit 1
    fi
    ;;
  tuned)
    case "${RUNG:-}" in
      rs10) RE=10; SHAPSEG=bar1600 ;;
      rs1) RE=1; SHAPSEG=none ;;
      *) echo "RUNG must be rs10 or rs1, got '${RUNG:-}'"; exit 1 ;;
    esac
    ROOT="${RESULTS_ROOT:?RESULTS_ROOT}/$RUNG"
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r BUCKET MODELS TW SEG CHUNK START END HALO <<< "$LINE"
    [ -n "${HALO:-}" ] || { echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1; }
    FAIL=0
    for MODEL in ${MODELS//,/ }; do
      OUT="$ROOT/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
      [ -f "$OUT/DONE" ] && { echo "  skip (DONE) $MODEL c$CHUNK"; continue; }
      mkdir -p "$OUT"
      a=$(date +%s)
      if SLURM_CPUS_PER_TASK="$NT" HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
         HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
         HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
         HPC_KW_REFIT_EVERY="$RE" HPC_KW_SHAP_SEGMENTS="$SHAPSEG" \
         HPC_KW_TUNE_IDENTITY=0 HPC_KW_QSEL=1 HPC_KW_WINDOW_MASK=1 \
         HPC_RESULT_DIR="$OUT" \
         "$PY" -u specs/causal_tune_trees_tuned.py > "$OUT/run.log" 2>&1 \
         && npz_masked "$OUT/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz"; then
        host_tag "$OUT" $(( $(date +%s) - a ))
        touch "$OUT/DONE"
        echo "  $MODEL c$CHUNK done in $(( $(date +%s) - a ))s  $(date +%T)"
      else
        FAIL=1
        echo "  $MODEL c$CHUNK FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
      fi
    done
    exit $FAIL
    ;;
  *) echo "MODE must be gate, untuned or tuned"; exit 1 ;;
esac
