#!/bin/bash
# Feature MCS trees, the 16 bucket designs on Hoffman2 (2026-10-13; CARC /scratch1 storage failure): XGBoost xgb_opt_00 and
# the forest trial 255 through specs/causal_tune_trees_mcs.py, the twin of cluster/slurm/feature_mcs_trees_pack.sbatch's
# run_line (same HPC_KW_* axes: EXOG_BUCKET all_features + the design's columns, REFIT_EVERY 1, WINDOW_MASK 1, importance
# and TreeSHAP off, one thread, seed 42, the 866 MCS rows), ONE array task = one line of TASKFILE
#   "<pack> <est> <design> <chunk> <start> <end> <halo> <planned seconds>"   (est xgb | rf; LightGBM runs on CARC only)
# into $RESULTS_ROOT/<est>/<design>/c<chunk> (+ DONE, NODE = "host <host> AvailableFeatures=<AVX class>", CPU = the CPU
# model, TIME = real user sys seconds).  REQUIRE_CPU: colon-separated substrings of the CPU model (spaces as underscores),
# default E5-2670:E5-2650_v2 -- the AVX models byte-identical to CARC epyc-7513 for both estimators; any other node refuses.
# CANARY_OUT (canary): writes $CANARY_OUT/ok_<task> after its line; CANARY_DIR + CANARY_N (fleet): refuses to run unless
# CANARY_N ok_* flags exist.  Runtime: CARC's Python stack (RUNTIME_ROOT, read only), checked in-process.
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
export PYTHONPATH="$D:$D/experiments" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
set -u
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx|avx2|fma|avx512f' | tr '\n' ',')
case ",$FLAGS" in *,avx512f,*) CLASS=AVX-512 ;; *,avx2,*) CLASS=AVX2 ;; *) CLASS=AVX ;; esac
echo "task $TASK on $(hostname) [$CPU | $FLAGS | $CLASS] $(date)"
REQ="${REQUIRE_CPU:-E5-2670:E5-2650_v2}"
OKCPU=0
IFS=: read -r -a _REQ <<< "$REQ"
for q in "${_REQ[@]}"; do [[ "${CPU// /_}" == *"$q"* ]] && OKCPU=1; done
[ "$OKCPU" = 1 ] || { echo "node $(hostname) CPU '$CPU' is not one of '$REQ'; refusing to run"; exit 1; }
if [ -n "${CANARY_DIR:-}" ] && [ -n "${CANARY_N:-}" ]; then
  n=$(ls "$CANARY_DIR"/ok_* 2>/dev/null | wc -l)
  [ "$n" -ge "$CANARY_N" ] || { echo "canary: $n of $CANARY_N tasks ok; refusing to run"; exit 1; }
fi
"$PY" - "$RT" <<'EOF' || { echo "runtime check FAILED; refusing to run"; exit 1; }
import sys, ctypes
import numpy, scipy, pandas, sklearn, xgboost, sklearn.ensemble
libc = ctypes.CDLL(None); libc.gnu_get_libc_version.restype = ctypes.c_char_p
g = libc.gnu_get_libc_version().decode()
maps = {l.split()[-1] for l in open("/proc/self/maps") if l.split()[-1].startswith("/") and ".so" in l}
out = sorted(p for p in maps if not p.startswith(sys.argv[1] + "/"))
print("runtime: glibc", g, "|", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, scipy, pandas, sklearn, xgboost)),
      f"| outside the runtime: {out if out else 'none'}")
sys.exit(0 if g == "2.28" and not out else 1)
EOF
LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
read -r PK EST DESIGN CHUNK START END HALO SEC <<< "$LINE"
case "$EST" in
  xgb) MODEL=xgb; BASE=specs/causal_tune_trees_xgbtop.py ;;
  rf) MODEL=rf; BASE=specs/causal_tune_trees_rf400.py ;;
  *) echo "estimator '$EST' does not run on Hoffman2 (LightGBM: CARC epyc-7513 only)"; exit 1 ;;
esac
OUT="${RESULTS_ROOT:?RESULTS_ROOT}/$EST/$DESIGN/c$CHUNK"
if [ -f "$OUT/DONE" ]; then echo "skip (DONE) $OUT"; exit 0; fi
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
       "$PY" -u specs/causal_tune_trees_mcs.py > "$OUT/run.log" 2>&1 ) ) 2> "$OUT/TIME" \
   && "$PY" -c "import json, sys, numpy as np; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); sys.exit(0 if m.get('mcs_design') == sys.argv[2] and m.get('model') == sys.argv[3] and m.get('window_mask') == 1 and m.get('refit_every') == 1 else 1)" \
        "$OUT/causal_tune_trees/$MODEL/all_features/trees_bar1600.npz" "$DESIGN" "$MODEL"; then
  echo "host $(hostname) AvailableFeatures=hoffman2,$CLASS" > "$OUT/NODE"
  echo "$CPU" > "$OUT/CPU"
  touch "$OUT/DONE"
  echo "done $EST $DESIGN c$CHUNK [$START, $END) in $((SECONDS - T0)) s (planned $SEC on epyc)  $(date +%T)"
  if [ -n "${CANARY_OUT:-}" ]; then mkdir -p "$CANARY_OUT"; echo "$(hostname -s) $CPU $(date +%T)" > "$CANARY_OUT/ok_$TASK"; fi
else
  echo "FAILED $EST $DESIGN c$CHUNK: $(tail -1 "$OUT/run.log" | cut -c1-200)"
  exit 1
fi
