#!/bin/bash
# 2026-10-29 addendum L: the macro cache (bucket all_features_mac, 780 columns) on USC CARC, a job payload (no sbatch,
# copy or move in here).  experiments/ablate_design.py buildmac = experiments/pool_linear_pretune.py design with the
# FOMC-patched design's arguments (CFG, --window-days 3000, LATE_START_WARMUP and cell B's flags
# FOMC_SCHEDULE_2020_FIX=1, FOMC_PROXIMITY_NO_WINSOR=1, DROP_HOLIDAY_SESSIONS=0) + MACRO_CALENDARS=1, one thread for
# OpenMP / OpenBLAS / MKL / numba (the orchestrator's design stage), in the numerical environment of
# results/fomcfix_2026-10-26/code/lin_stage.sh, which built the patched cache: the final linear runs' venv
# (/home1/jc_905/fomcfix_lin/venv), OPENBLAS_CORETYPE=Prescott, numpy's AVX-512 dispatch off where the node has it.
# Then the md5 manifest of the five files ($OUT/MD5SUMS, read by the transfer and by prep_mac.sh) and gate GM0
# (ablate_design.py checkmac) against the patched cache read in place.  Exit: the build's, else GM0's (3 = FAIL).
# env: CODE (code tree of this commit), OUT (the macro cache dir; refused if it holds an X.npy), FULL (patched cache;
#      default fomcfix cell B's), FF_PY (python; default the fomcfix_lin venv), GM0 (gate JSON; default
#      <dir of OUT>/gateGM0.json)
set -euo pipefail
: "${CODE:?}" "${OUT:?}"
FULL=${FULL:-/home1/jc_905/fomcfix_lin/runs/B/_work/design_all_features}
PY=${FF_PY:-/home1/jc_905/fomcfix_lin/venv/bin/python}
GM0=${GM0:-$(dirname "$OUT")/gateGM0.json}
cd "$CODE"
export OPENBLAS_CORETYPE=Prescott
export NPY_DISABLE_CPU_FEATURES="$("$PY" -c 'from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))')"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "carc_build_mac job ${SLURM_JOB_ID:-} on $(hostname) [$CPU], $("$PY" -V 2>&1), $(date); code $CODE ($(head -1 COMMIT 2> /dev/null || echo 'no COMMIT')), out $OUT"
echo "NPY_DISABLE_CPU_FEATURES=${NPY_DISABLE_CPU_FEATURES:-}"
"$PY" -c "import threadpoolctl, numpy, numba; print('blas', [(d['internal_api'], d.get('architecture'), d.get('version')) for d in threadpoolctl.threadpool_info()], 'numpy', numpy.__version__, 'numba', numba.__version__); print('logspace', repr(numpy.logspace(-6, -2, 5).tolist()), repr(numpy.logspace(-2, 3, 6).tolist()))"
if [ -e "$OUT/X.npy" ]; then echo "$OUT/X.npy exists: refusing to overwrite"; exit 2; fi
m=$(md5sum "$FULL/X.npy" | cut -d' ' -f1)
[ "$m" = 725eb201044783d3cdc62501307dcc3a ] || { echo "patched cache X md5 $m != 725eb201...: exit 2"; exit 2; }
mkdir -p "$(dirname "$OUT")"
t0=$(date +%s)
"$PY" experiments/ablate_design.py buildmac --out "$OUT"
echo "buildmac done in $(($(date +%s) - t0)) s"
(cd "$OUT" && md5sum X.npy y.npy b_oos.npy dates_oos.csv design.json > MD5SUMS && cat MD5SUMS)
rc=0
"$PY" experiments/ablate_design.py checkmac --mac "$OUT" --full "$FULL" --out "$GM0" || rc=$?
echo "carc_build_mac done: GM0 rc $rc ($GM0) $(date)"
exit $rc
