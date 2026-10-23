#!/bin/bash
# 2026-10-23 diagnosis of the bus error of the gate check's Python (the /scratch1 env harxhar-r13) on the GPU nodes
# (d23-14, e21-09): run on one of those nodes (CPU only), same check as the gate's POST step, under faulthandler and
# gdb; checks the env's binaries against the login node's md5 list and times the imports.
#   usage (CARC): sbatch -A pollok_1603 -p gpu -w <node> -c 2 --mem=16G -t 0:30:00 pooled_mingru_final_diag_buserror.sh
set -uo pipefail
E=${E:-/scratch1/jc_905/envs/harxhar-r13}  # the fix: E=/home1/jc_905/envs/harxhar-r13 MD5LIST=$D/env_bin_home1.md5
PY=$E/bin/python
D=/home1/jc_905/pmf/diag
H=$(hostname -s)
O=$D/${H}_$SLURM_JOB_ID
mkdir -p "$O"; cd "$O"
{ hostname; uname -r; ldd --version | head -1; lscpu | grep -E "^Model name|^Flags" | cut -c1-400; df -h /scratch1 /home1 | cat; } > node.txt 2>&1
md5sum -c --quiet "${MD5LIST:-$D/env_bin.md5}" > md5_check.txt 2>&1; echo "md5 check exit $? ($(wc -l < md5_check.txt) mismatching lines)" >> node.txt
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
T0=$(date +%s%N)
"$PY" -X faulthandler -X importtime -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)" > import.txt 2> importtime.txt
echo "import exit $? in $(( ($(date +%s%N) - T0) / 1000000 )) ms" >> node.txt
for r in gate_tune_prev_13319779 gate_walk_prev_13319780; do
  cp -r /home1/jc_905/pmf/runs/$r "$O/$r"; rm -f "$O/$r/gate_check.json"
  T0=$(date +%s%N)
  "$PY" -X faulthandler /home1/jc_905/pmf/code/pooled_mingru_final_gate.py "$O/$r" > "gate_$r.txt" 2>&1
  echo "gate $r exit $? in $(( ($(date +%s%N) - T0) / 1000000 )) ms" >> node.txt
  gdb -q -batch -ex run -ex bt -ex "info sharedlibrary" --args "$PY" /home1/jc_905/pmf/code/pooled_mingru_final_gate.py "$O/$r" > "gdb_$r.txt" 2>&1
  echo "gdb $r done: $(grep -c "SIGBUS" "gdb_$r.txt") SIGBUS lines" >> node.txt
done
cat node.txt
