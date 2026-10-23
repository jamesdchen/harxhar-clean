#!/bin/bash
# One stage of the 13-bar 2012 pooled linear rerun (as results/pooled_linear_rerun_2026-10-22_13_2012_w1000_
# latestart_nofomcrel: --ex-flag LATE_START_WARMUP=1, EXCLUDE_FOMC_RELEASE on) for one window, on a CARC node
# (Rocky 8, the uv venv natively).
# usage: stage.sh <out dir, relative to the code tree> <window days> <first fold> <stage> [jobs]
#   stage: design | pretune_select | walk1 (Slurm array task SLURM_ARRAY_TASK_ID of walk_jobs) | score
set -euo pipefail
B=/scratch1/jc_905/wsweep
PY=$B/venv/bin/python
cd $B/code
OUT=$1; WD=$2; FF=$3; ST=$4; J=${5:-1}
# the first evaluation day: WS_FIRST_DAY in the environment (the 2014-start sweep), default 2012-01-03
F="--window-days $WD --first-day ${WS_FIRST_DAY:-2012-01-03} --ex-flag LATE_START_WARMUP=1"
FFO=""
if [ "$FF" != 0 ]; then FFO="--first-fold $FF"; fi
CH="--chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4"
# run B's OpenBLAS kernel on the laptop (threadpoolctl: Prescott; OpenBLAS 0.3.23.dev has no Raptor Lake entry)
export OPENBLAS_CORETYPE=Prescott
# numpy's run-time dispatch as on the laptop (no AVX-512; on Hoffman2's AVX-512 nodes np.logspace came out one ulp
# low and the full-set design was not run B's): the AVX512 targets off where the node has them (WS_NPY=0: default).
# WS_NUMBA=1: numba compiled for the laptop's CPU (raptorlake, its LLVM host features).
if [ "${WS_NPY:-1}" = 1 ]; then
  export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
fi
if [ "${WS_NUMBA:-0}" = 1 ]; then
  export NUMBA_CPU_NAME=raptorlake NUMBA_CPU_FEATURES="$(cat $B/jobs/laptop_cpu_features.txt)"
fi
echo "NPY_DISABLE_CPU_FEATURES=${NPY_DISABLE_CPU_FEATURES:-} NUMBA_CPU_NAME=${NUMBA_CPU_NAME:-}"
mkdir -p "$OUT/_work"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then TAG=$ST; else TAG="$ST.$SLURM_ARRAY_TASK_ID"; fi
echo "$TAG host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | SLURM_JOB_ID ${SLURM_JOB_ID:-} | $(date)" >> "$OUT/_work/hosts.txt"
"$PY" -c "import threadpoolctl, numpy, numba, llvmlite.binding as ll; print('blas', [(d['internal_api'], d.get('architecture'), d.get('version')) for d in threadpoolctl.threadpool_info()], 'numpy', numpy.__version__, 'numba', numba.__version__, 'llvm cpu', ll.get_host_cpu_name()); from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print('numpy dispatch on', [k for k in d if f.get(k)]); print('logspace', repr(numpy.logspace(-6, -2, 5).tolist()), repr(numpy.logspace(-2, 3, 6).tolist()))"
t0=$(date +%s)
echo "[$(date +%H:%M:%S)] stage $TAG start"
case "$ST" in
  design) "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs "$J" $F --out "$OUT" ;;
  pretune_select)
    "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs "$J" $F $FFO --out "$OUT"
    "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py select $F $FFO --out "$OUT" ;;
  walk1) "$PY" $B/jobs/one_walk.py walk $CH $F $FFO --out "$OUT" ;;
  score)
    "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py score $F $FFO --out "$OUT"
    # LATE_START_WARMUP's zeroed values on the 16:00 rows of the 866 option days (read-only on the design cache).
    # Its prefix-sum check fails under UNOBSERVED_NEUTRAL (2 more zeros at a prefix end at w1000): the counts come from
    # latestart_rule.py off the cluster, checked on each design log; this step's failure leaves the score files as written
    "$PY" $B/jobs/latestart_count.py "$OUT/_work/design_all_features" "$OUT/_work/logs/design_all_features.log" \
      $B/jobs/option_days.csv "$OUT/_work/latestart_counts.csv" "w$WD" || true ;;
  *) echo "unknown stage $ST"; exit 2 ;;
esac
t1=$(date +%s)
echo "$TAG $((t1 - t0)) $(hostname)" >> "$OUT/_work/stage_times.txt"
echo "[$(date +%H:%M:%S)] stage $TAG done in $((t1 - t0)) s"
