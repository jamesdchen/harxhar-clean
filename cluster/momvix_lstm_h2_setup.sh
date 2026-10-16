#!/bin/bash
# 2026-10-06 (moments_vix LSTMs): build, on the Hoffman2 LOGIN node, CARC's exact runtime WITH torch in a
# fresh root /u/scratch/j/jamesdc1/harxhar-momvix-lstm, so the LSTM specs can be tried on Hoffman2 nodes
# against the stored CARC refits (the sample check that decides whether Hoffman2 may host any task).
# The I5 runtime (cluster/optuna_h2_runtime_setup.sh, harxhar-optuna/runtime_tgz) is CARC's harxhar env
# WITHOUT torch / nvidia / triton, CARC's user site (numpy 2.3.5, pandas 3.0.1) and CARC's glibc 2.28;
# this script extracts those tarballs (md5-checked) into this root, adds CARC's torch package
# (torch.tgz: site-packages/torch + its dist-info, from /home1/jc_905/.conda/envs/harxhar, relayed by
# cluster/momvix_lstm_h2_ship.sh) and patches python3.11 exactly as the I5 script does (CARC's loader
# as interpreter, rpath to the copied glibc; nothing from the host's libraries).
#   cd /u/scratch/j/jamesdc1/harxhar-momvix-lstm && bash cluster/momvix_lstm_h2_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=$PWD
SRC=/u/scratch/j/jamesdc1/harxhar-optuna
( cd $SRC/runtime_tgz && md5sum -c tgz.md5 )
( md5sum -c torch.tgz.md5 )
for d in carc_env carc_usersite carc_sys tools; do
  [ ! -e "$d" ] || { echo "$D/$d exists: this script builds the runtime once, into a fresh root"; exit 1; }
done
mkdir -p carc_env carc_usersite/lib/python3.11 carc_sys tools
tar xzf $SRC/runtime_tgz/env.tgz -C carc_env
tar xzf $SRC/runtime_tgz/usersite.tgz -C carc_usersite/lib/python3.11
tar xzf $SRC/runtime_tgz/carc_sys.tgz -C carc_sys
tar xzf torch.tgz -C carc_env/harxhar/lib/python3.11/site-packages
cp -rp $SRC/tools/. tools/
PE=tools/bin/patchelf
PY=carc_env/harxhar/bin/python3.11
cp -p "$PY" "$PY.carc"
$PE --set-interpreter "$D/carc_sys/ld-linux-x86-64.so.2" "$PY"
$PE --force-rpath --set-rpath "\$ORIGIN/../lib:$D/carc_sys" "$PY"
for so in librt.so.1 libresolv.so.2 libmvec.so.1 libcrypt.so.1; do
  $PE --add-needed "$so" "$PY"
done
echo "patched: $($PE --print-interpreter "$PY"), rpath $($PE --print-rpath "$PY")"
( find carc_env carc_usersite carc_sys -type f -print0 | sort -z | xargs -0 md5sum ) > runtime_manifest.md5
echo "runtime files: $(wc -l < runtime_manifest.md5)"
env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONUSERBASE="$D/carc_usersite" PYTHONPATH="$D" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" - <<'EOF'
import sys, os, ctypes
import numpy, pandas, torch
from src.data.loading import get_bucket
for m in (numpy, pandas, torch):
    print(f"{m.__name__:8s} {m.__version__:14s} {m.__file__}")
print("python", sys.version.split()[0], "torch cpu capability", torch.backends.cpu.get_cpu_capability(),
      "mkldnn", torch.backends.mkldnn.is_available())
libc = ctypes.CDLL(None)
libc.gnu_get_libc_version.restype = ctypes.c_char_p
print("glibc in process:", libc.gnu_get_libc_version().decode())
maps = {l.split()[-1] for l in open("/proc/self/maps") if "/" in l.split()[-1] and ".so" in l}
root = os.environ["PYTHONUSERBASE"].rsplit("/", 1)[0]
host = sorted(p for p in maps if not p.startswith(root))
print(f"shared objects mapped: {len(maps)}; outside the runtime root: {host if host else 'none'}")
print("moments_vix =", get_bucket("moments_vix"))
EOF
