#!/bin/bash
# I5 (2026-09-29): Hoffman2 (SGE) twin of cluster/slurm/optuna_pack.sbatch -- one array task runs one
# TIME CHUNK of one arm of the Optuna per-bar tree campaign (specs/causal_tune_trees_optuna.py) for
# one STAGE, with the SAME spec call (the same HPC_KW_* axes as the CARC task), into the SAME layout
#   $RESULTS_ROOT/stage<1|2>/<bucket>/<seg>/<model>/tw<TW>/chunks/c<chunk>   (+ DONE, run.log)
# so a chunk run here can be copied into the CARC results tree and merged by
# experiments/reduce_trees_optuna.py like any CARC chunk.  Only the runtime differs: CARC's own
# Python stack (cluster/optuna_h2_runtime_setup.sh: CARC's harxhar env, user site, ./pylib_trees and
# glibc, bytes md5-listed in runtime_manifest.md5), started through the patched python3.11.
#
# Task lines (TASKFILE): "<bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>" (C's format).
# Exported at submit time (cluster/submit_optuna_h2.sh): TASKFILE, STAGE (tune | refit),
# RESULTS_ROOT, WINDOW_MASK (0 | 1, passed to the spec as C's CARC task does), N_TRIALS (default 50).
# STAGE refit reads $RESULTS_ROOT/stage1/<arm>/trials_<seg>.npz and refuses to run without that
# arm's STAGE1_COMPLETE (as the CARC task does).  A chunk-arm with DONE is skipped.
# Parallelism: a pool of $NSLOTS spawned processes, every fit single-threaded (the spec pins
# MODEL_THREADS = 1; BLAS / OpenMP / numba threads pinned to 1 here), so the records do not depend
# on the pool size (C's gate: pool 1 == pooled; the CARC canary: 7 vs 20 processes bit-identical).
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
PY=$D/carc_env/harxhar/bin/python3.11
[ -x "$PY" ] || { echo "no runtime at $PY (run cluster/optuna_h2_runtime_setup.sh)"; exit 1; }
# CARC's stack only: nothing from the host's Python / conda / module environment
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONSTARTUP PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV
export PATH="$D/carc_env/harxhar/bin:/usr/bin:/bin" PYTHONUSERBASE="$D/carc_usersite"
export PYTHONPATH="$D:$D/pylib_trees" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
set -u
NT="${NSLOTS:-1}"
R="${RESULTS_ROOT:?RESULTS_ROOT}"
STAGE="${STAGE:?STAGE must be tune or refit}"
case "$STAGE" in tune) SN=1 ;; refit) SN=2 ;; *) echo "bad STAGE $STAGE"; exit 1 ;; esac
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "$TASKFILE")
read -r BUCKET MODELS TW SEG CHUNK START END HALO <<< "$LINE"
[ -n "${HALO:-}" ] || { echo "malformed line $TASK of $TASKFILE (want 8 fields): '$LINE'"; exit 1; }

CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx|avx2|fma|avx512f|avx512dq|avx512bw|avx512vl' | tr '\n' ',')
# the CPU class the design depends on (numpy's dispatched loops): AVX-512 / AVX2 / AVX
case ",$FLAGS" in *,avx512f,*) CLASS=AVX-512 ;; *,avx2,*) CLASS=AVX2 ;; *) CLASS=AVX ;; esac
echo "task $TASK on $(hostname) [$CPU | $FLAGS | $CLASS] stage $SN ($STAGE), pool of $NT: $BUCKET [$MODELS] tw$TW $SEG chunk $CHUNK rows [$START, $END) halo $HALO mask ${WINDOW_MASK:-0} $(date)"
# the runtime in this process: CARC's glibc and versions, no shared object from outside $D
"$PY" - "$D" <<'EOF' || { echo "runtime check FAILED; refusing to run"; exit 1; }
import sys, ctypes
import numpy, scipy, pandas, sklearn, lightgbm, xgboost, optuna, shap, numba, sklearn.ensemble, scipy.special
libc = ctypes.CDLL(None); libc.gnu_get_libc_version.restype = ctypes.c_char_p
g = libc.gnu_get_libc_version().decode()
maps = {l.split()[-1] for l in open("/proc/self/maps") if l.split()[-1].startswith("/") and ".so" in l}
out = sorted(p for p in maps if not p.startswith(sys.argv[1] + "/"))
print("runtime: glibc", g, "|", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, scipy, pandas, sklearn, lightgbm, xgboost, optuna, shap, numba)),
      f"| {len(maps)} shared objects, outside the runtime: {out if out else 'none'}")
sys.exit(0 if g == "2.28" and not out else 1)
EOF

run_spec() {  # $1 model, $2 result dir, $3 pool size, $4 stage-1 records (stage 2), $5 start, $6 end -- C's call
  SLURM_CPUS_PER_TASK="$3" HPC_KW_MODEL="$1" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
    HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_START="$5" HPC_KW_END="$6" HPC_KW_HALO="$HALO" \
    HPC_KW_STAGE="$STAGE" HPC_KW_N_TRIALS="${N_TRIALS:-50}" HPC_KW_IDENTITY=0 HPC_KW_RESUME=1 \
    HPC_KW_WINDOW_MASK="${WINDOW_MASK:-0}" HPC_KW_STAGE1_TRIALS="$4" HPC_RESULT_DIR="$2" \
    "$PY" -u specs/causal_tune_trees_optuna.py
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
    touch "$OUT/DONE"
    echo "host=$(hostname -s) class=$CLASS cpu=$CPU seconds=$(( $(date +%s) - a )) pool=$NT" > "$OUT/H2_HOST"
    echo "  $MODEL $SEG c$CHUNK done in $(( $(date +%s) - a )) s $(date)"
  else
    FAIL=1
    echo "  $MODEL $SEG c$CHUNK FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
  fi
done
exit $FAIL
