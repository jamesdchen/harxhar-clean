#!/bin/bash
# 2026-10-06 (moments_vix LSTMs): Hoffman2 (SGE) twin of cluster/slurm/lstm_mask_pack.sbatch (per-bar,
# model "lstm": WINDOW_MASK=1, REFIT_EVERY=1) and cluster/slurm/lstm_intraday_pack.sbatch (model
# "lstm_intraday": STAGE tune | all, TUNE_CACHE=$RESULTS_ROOT/tune_cache) -- the same spec calls with the
# same HPC_KW_* axes, into the same layout, under CARC's own runtime (cluster/momvix_lstm_h2_setup.sh:
# CARC's harxhar env incl. torch 2.12.1+cpu, CARC's user site and glibc, the patched python3.11).
# Task lines (TASKFILE) as on CARC: "<bucket> lstm <tw> <seg> <chunk> <start> <end> <halo>" or
# "<bucket> lstm_intraday <tw> <seg> <chunk> <start> <end> <halo> <stage>".  A line with DONE is skipped.
# Every fit is single-threaded (the specs pin torch to one thread; BLAS / OpenMP pinned here), the pool is
# $NSLOTS spawned processes, so the numbers do not depend on the slot count.  NODE records the host, the
# CPU model and its vector flags.  Exported at submit time: TASKFILE, RESULTS_ROOT, REQUIRE_CACHE.
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
PY=$D/carc_env/harxhar/bin/python3.11
[ -x "$PY" ] || { echo "no runtime at $PY (run cluster/momvix_lstm_h2_setup.sh)"; exit 1; }
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONSTARTUP PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV
export PATH="$D/carc_env/harxhar/bin:/usr/bin:/bin" PYTHONUSERBASE="$D/carc_usersite"
export PYTHONPATH="$D" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
set -u
NT="${NSLOTS:-1}"
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "$TASKFILE")
read -r BUCKET MODEL TW SEG CHUNK START END HALO STAGE <<< "$LINE"
R="${RESULTS_ROOT:?RESULTS_ROOT must be set}"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
AVX=$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|fma|avx512f)$' | paste -sd, -)
case "$MODEL" in
  lstm)
    OUT="$R/$BUCKET/$SEG/lstm/tw$TW/chunks/c$CHUNK" ;;
  lstm_intraday)
    if [ "$STAGE" = tune ]; then OUT="$R/$BUCKET/$SEG/lstm_intraday/tw$TW/stage_tune/r$CHUNK"; else OUT="$R/$BUCKET/$SEG/lstm_intraday/tw$TW/chunks/c$CHUNK"; fi ;;
  *) echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1 ;;
esac
echo "task $TASK on $(hostname) [$CPU; $AVX], pool of $NT: $BUCKET $MODEL ${STAGE:-} c$CHUNK [$START, $END) halo $HALO $(date)"
if [ -f "$OUT/DONE" ]; then echo "  already DONE"; exit 0; fi
mkdir -p "$OUT"
if [ "$MODEL" = lstm ]; then
  SLURM_CPUS_PER_TASK="$NT" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" HPC_KW_TRAIN_WIN="$TW" \
  HPC_KW_LAG_SCOPE=global HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" HPC_KW_QSEL=1 \
  HPC_KW_WINDOW_MASK=1 HPC_KW_REFIT_EVERY=1 HPC_RESULT_DIR="$OUT" \
  "$PY" -u specs/causal_tune_lstm.py > "$OUT/run.log" 2>&1 \
  && "$PY" -c "import json, sys, numpy as np; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); sys.exit(0 if m.get('window_mask') == 1 and m['grid']['refit_every'] == 1 and 'kept_n' in z.files else 1)" \
       "$OUT/causal_tune_lstm/lstm/$BUCKET/lstm_$SEG.npz" || { echo "  FAILED: $(tail -3 "$OUT/run.log" | cut -c1-300)"; exit 1; }
else
  SLURM_CPUS_PER_TASK="$NT" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" HPC_KW_TRAIN_WIN="$TW" \
  HPC_KW_LAG_SCOPE=global HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" HPC_KW_QSEL=1 \
  HPC_KW_WINDOW_MASK=1 HPC_KW_STAGE="$STAGE" HPC_KW_TUNE_CACHE="$R/tune_cache" \
  HPC_KW_REQUIRE_CACHE="$([ "$STAGE" = tune ] && echo 0 || echo "${REQUIRE_CACHE:-1}")" \
  HPC_RESULT_DIR="$OUT" "$PY" -u specs/causal_tune_lstm_intraday.py > "$OUT/run.log" 2>&1 \
    || { echo "  FAILED: $(tail -3 "$OUT/run.log" | cut -c1-300)"; exit 1; }
fi
printf 'host %s\ncpu %s\nflags %s\n' "$(hostname)" "$CPU" "$AVX" > "$OUT/NODE"
touch "$OUT/DONE"
echo "  done $(date): $(tail -1 "$OUT/run.log" | cut -c1-300)"
