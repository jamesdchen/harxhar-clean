#!/bin/bash
# Ship the feature Model Confidence Set study (2026-10-08) to a deployment root of its own on CARC and FAIL
# unless every shipped file's remote md5 equals the local one; then, on the login node: the reference
# LAPACK / BLAS with unique SONAMEs, the kernel build, a smoke run with its LD_DEBUG binding check, bash -n.
#
#   NETLIB_SRC=<dir> STAGE=<empty local scratch dir> bash cluster/slurm/ship_feature_mcs_carc.sh
#
# NETLIB_SRC holds, extracted from conda-forge linux-64 packages: libblas.so.3.12.0 (libblas-3.12.1-
# 6_h08cb66b_netlib), liblapack.so.3.12.0 (liblapack-3.12.1-6_h582c038_netlib) -- the reference LAPACK /
# BLAS 3.12 the 2026-10-06 study linked on Windows (the win-64 builds of the same packages) -- and patchelf
# (patchelf-0.19.1-h54a6638_0).  The libraries are renamed (SONAME libnetlib_blas.so / libnetlib_lapack.so)
# so that the OpenBLAS the env's scipy loads as liblapack.so.3 / libblas.so.3 can never stand in for them.
# Code: the working-tree files (LF; checked).  Inputs: results/feature_mcs_2026-10-08/_work/inputs/
# (experiments/feature_mcs_runner.py inputs).
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=${REMOTE:-/scratch1/jc_905/harxhar-feature-mcs}  # 2026-10-13: REMOTE=<fresh root> when /scratch1 files hang
R=results/feature_mcs_2026-10-08
STAGE=${STAGE:?set STAGE to an empty local scratch directory}
NETLIB_SRC=${NETLIB_SRC:?set NETLIB_SRC to the directory with the reference libraries and patchelf}
mkdir -p "$STAGE"
[ -z "$(ls -A "$STAGE")" ] || { echo "STAGE $STAGE is not empty"; exit 1; }

CODE=(
  experiments/feature_mcs_runner.py experiments/close_linear_pretune.py experiments/close_exogpen.py
  experiments/close_exogpen_kernel.c specs/causal_tune_trees_tuned_jobs.py
  src/__init__.py src/models/__init__.py src/models/window_mask.py
  src/evaluation/__init__.py src/evaluation/metrics.py src/evaluation/diebold_mariano.py
  cluster/slurm/feature_mcs.sbatch cluster/slurm/submit_feature_mcs.sh
  cluster/feature_mcs_tasks_canary.txt cluster/feature_mcs_tasks_fleet.txt
  cluster/feature_mcs_tasks2_canary.txt cluster/feature_mcs_tasks2_fleet.txt
  cluster/feature_mcs_tasks3_canary.txt cluster/feature_mcs_tasks3_fleet.txt
)
DATA=(
  $R/_work/inputs/bar_X.npy $R/_work/inputs/bar_y.npy $R/_work/inputs/pre_X.npy $R/_work/inputs/pre_y.npy
  $R/_work/inputs/inputs.json $R/_work/inputs/models.csv $R/_work/inputs/ref_armf.npz
)
LIBS=(netlib_src/libblas.so.3.12.0 netlib_src/liblapack.so.3.12.0 netlib_src/patchelf)
for f in "${CODE[@]}"; do  # a tracked file with no change against HEAD is taken from HEAD (LF)
  mkdir -p "$STAGE/$(dirname "$f")"
  if git ls-files --error-unmatch "$f" > /dev/null 2>&1 && git diff --quiet HEAD -- "$f"; then
    git show "HEAD:$f" > "$STAGE/$f"
  else
    cp "$f" "$STAGE/$f"
  fi
done
for f in "${DATA[@]}"; do mkdir -p "$STAGE/$(dirname "$f")"; cp "$f" "$STAGE/$f"; done
mkdir -p "$STAGE/netlib_src"
for f in "${LIBS[@]}"; do cp "$NETLIB_SRC/$(basename "$f")" "$STAGE/$f"; done
CR=$(cd "$STAGE" && cat "${CODE[@]}" | tr -cd '\r' | wc -c)
[ "$CR" -eq 0 ] || { echo "staged code carries $CR CR bytes"; exit 1; }
ALL=("${CODE[@]}" "${DATA[@]}" "${LIBS[@]}")
"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/logs"
(cd "$STAGE" && tar czf - "${ALL[@]}") | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf -"

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LOCAL_ALL=$( (cd "$STAGE" && md5sum "${ALL[@]}") | sums)
# every shipped file is read end to end by md5sum (a hanging storage target fails here, under a timeout)
REMOTE_ALL=$(printf '%s\n' "${ALL[@]}" | timeout 600 "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs timeout 300 md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs from local:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files"

"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && bash -s" <<'REMOTE_EOF'
set -e
module load conda >/dev/null 2>&1; source activate harxhar
set -u
R=results/feature_mcs_2026-10-08
NL=$PWD/$R/_work/kernel/netlib
mkdir -p "$NL"
cp netlib_src/libblas.so.3.12.0 "$NL/libnetlib_blas.so"
cp netlib_src/liblapack.so.3.12.0 "$NL/libnetlib_lapack.so"
chmod +x netlib_src/patchelf
PE="env LD_LIBRARY_PATH=$CONDA_PREFIX/lib netlib_src/patchelf"
$PE --set-soname libnetlib_blas.so "$NL/libnetlib_blas.so"
$PE --set-soname libnetlib_lapack.so --replace-needed libblas.so.3 libnetlib_blas.so "$NL/libnetlib_lapack.so"
for f in "$NL/libnetlib_blas.so" "$NL/libnetlib_lapack.so"; do
  $PE --set-rpath "\$ORIGIN:$CONDA_PREFIX/lib" "$f"
  echo "$(basename "$f"): soname $($PE --print-soname "$f"); needed $($PE --print-needed "$f" | tr '\n' ' '); rpath $($PE --print-rpath "$f")"
done
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
python experiments/feature_mcs_runner.py build
echo "ldd kernel:"; ldd $R/_work/kernel/close_exogpen_kernel_nofma.so | grep -E "netlib|lapack|blas|gfortran|not found"
md5sum $R/_work/kernel/close_exogpen_kernel_nofma.so $NL/*.so
D=$PWD/logs/lddebug_login
LD_DEBUG=bindings LD_DEBUG_OUTPUT=$D python experiments/feature_mcs_runner.py smoke
python experiments/feature_mcs_runner.py bindings --tag _login --files $D.*
rm -f $D.*
for s in cluster/slurm/feature_mcs.sbatch cluster/slurm/submit_feature_mcs.sh; do bash -n "$s" && echo "bash -n ok: $s"; done
REMOTE_EOF
echo "ship OK -> usc-discovery:$REMOTE"
