#!/bin/bash
# I5 (2026-09-29): build, on the Hoffman2 LOGIN node, a runtime that is CARC's exactly, for the
# Optuna per-bar tree campaign (specs/causal_tune_trees_optuna.py) to run on Hoffman2 compute nodes.
#
# Why not `pip install lightgbm==4.6.0 xgboost==3.2.0 scikit-learn==1.9.0 ...` into hpc-pi: Hoffman2
# runs CentOS 7 (glibc 2.17) and none of CARC's versions (numpy 2.3.5, pandas 3.0.1, scipy 1.17.1,
# scikit-learn 1.9.0, lightgbm 4.6.0, xgboost 3.2.0) has a wheel for glibc < 2.27 (pip: "No
# matching distribution", 2026-09-29 21:30; only shap 0.51.0 has one).  Apptainer cannot run there
# for this user (user namespaces disabled, no setuid starter; mksquashfs absent).  So the runtime
# is CARC's own bytes, relayed by cluster/ship_optuna_h2.sh into $D/runtime_tgz:
#   env.tgz        /home1/jc_905/.conda/envs/harxhar (python 3.11.15, scipy, sklearn, lightgbm,
#                  xgboost, optuna, numba, pyarrow ...), minus torch / nvidia / triton (not imported
#                  by the spec: checked on CARC, 'torch' not in sys.modules after the spec's imports,
#                  a TPE study and the executor import) and __pycache__
#   usersite.tgz   /home1/jc_905/.local/lib/python3.11/site-packages (numpy 2.3.5, pandas 3.0.1 --
#                  on CARC they shadow the env's copies from the user site)
#   pylib_trees.tgz  the project's ./pylib_trees (shap 0.51.0, slicer)
#   carc_sys.tgz   CARC's glibc 2.28-251.el8_10.40 shared objects (+ libcrypt.so.1)
# and CARC's glibc is made the one every Python process of this runtime loads: the copied
# python3.11 gets CARC's loader as its ELF interpreter and an RPATH ($ORIGIN/../lib, then
# carc_sys: DT_RPATH, which also covers dlopen from libc) -- no LD_LIBRARY_PATH, so a host
# program a Python process may start is untouched.  multiprocessing's spawn re-executes the same
# patched binary.  The only bytes that differ from CARC's are python3.11's interpreter path and
# rpath (patchelf); every library, extension module and .py file is CARC's (md5 manifest below).
#
#   cd /u/scratch/j/jamesdc1/harxhar-optuna && bash cluster/optuna_h2_runtime_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=$PWD
T=runtime_tgz
( cd $T && md5sum -c tgz.md5 )
for d in carc_env carc_usersite pylib_trees carc_sys; do  # a fresh root only: nothing is overwritten
  [ ! -e "$d" ] || { echo "$D/$d exists: this script builds the runtime once, into a fresh root"; exit 1; }
done
mkdir -p carc_env carc_usersite/lib/python3.11 carc_sys tools
tar xzf $T/env.tgz -C carc_env
tar xzf $T/usersite.tgz -C carc_usersite/lib/python3.11
tar xzf $T/pylib_trees.tgz
tar xzf $T/carc_sys.tgz -C carc_sys

# patchelf (a static binary from its manylinux wheel; hpc-pi's pip, glibc-2.17-compatible wheel)
if [ ! -x tools/bin/patchelf ]; then
  set +u; source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh; conda activate hpc-pi; set -u
  pip install --quiet --target tools patchelf
  conda deactivate
fi
PE=tools/bin/patchelf
PY=carc_env/harxhar/bin/python3.11
cp -p "$PY" "$PY.carc"            # CARC's bytes, kept for the md5 manifest
$PE --set-interpreter "$D/carc_sys/ld-linux-x86-64.so.2" "$PY"
$PE --force-rpath --set-rpath "\$ORIGIN/../lib:$D/carc_sys" "$PY"
# the glibc sonames an extension may ask for later: loaded at start-up from carc_sys, so a later
# request by soname (from an object whose RUNPATH does not reach carc_sys) reuses them
for so in librt.so.1 libresolv.so.2 libmvec.so.1 libcrypt.so.1; do
  $PE --add-needed "$so" "$PY"
done
echo "patched: $($PE --print-interpreter "$PY"), rpath $($PE --print-rpath "$PY"), needed $($PE --print-needed "$PY" | tr '\n' ' ')"

# md5 manifest of the runtime as extracted (python3.11 = the patched copy; python3.11.carc = CARC's)
( find carc_env carc_usersite pylib_trees carc_sys -type f -print0 | sort -z | xargs -0 md5sum ) > runtime_manifest.md5
echo "runtime files: $(wc -l < runtime_manifest.md5)"

# smoke: versions, the loader and libc actually mapped, and no host library in the process
env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONUSERBASE="$D/carc_usersite" PYTHONPATH="$D:$D/pylib_trees" \
  "$PY" - <<'EOF'
import sys, os, platform, ctypes
import numpy, scipy, pandas, sklearn, lightgbm, xgboost, optuna, shap, numba, pyarrow
import sklearn.ensemble, scipy.special
for m in (numpy, scipy, pandas, sklearn, lightgbm, xgboost, optuna, shap, numba, pyarrow):
    print(f"{m.__name__:10s} {m.__version__:10s} {m.__file__}")
print("python", sys.version.split()[0], sys.executable)
libc = ctypes.CDLL(None)
libc.gnu_get_libc_version.restype = ctypes.c_char_p
print("glibc in process:", libc.gnu_get_libc_version().decode())
maps = {l.split()[-1] for l in open("/proc/self/maps") if "/" in l.split()[-1] and ".so" in l}
host = sorted(p for p in maps if not p.startswith(os.environ.get("PYTHONUSERBASE").rsplit("/", 1)[0]))
print(f"shared objects mapped: {len(maps)}; outside the runtime root: {host if host else 'none'}")
print("sys.path:", sys.path)
EOF
