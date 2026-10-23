#!/bin/bash
# 2026-10-23 read-only diagnosis (coordinator): why the native_g9 trainer (md5 44ab6fff, built for sm_60 .. sm_90)
# segfaults at start on V100 / A40 (gate jobs 13324299, 13324300) and runs on P100.  On one GPU of type GPUTYPE: one
# refit job of the P100 gate (the previous design, first walk origin, 1 seed) under gdb (backtrace of the crash) and
# under compute-sanitizer (memcheck); the binary is not changed.  Outputs in /home1/jc_905/pmf/diag/segv_<type>_<job>/.
#   sbatch --constraint=v100 --export=NONE,GPUTYPE=V100 pooled_mingru_final_diag_segv.sbatch
# (v2: also the gate's 6 tune jobs of both sets at concurrency 6, CUDA_DEVICE_MAX_CONNECTIONS 32 and 8)
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --job-name=pmfsegv
#SBATCH --output=/home1/jc_905/pmf/logs/%x_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
GPUTYPE=${GPUTYPE:?GPUTYPE}
GOT=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | head -1)
echo "$(hostname) $GOT"; echo "$GOT" | grep -qi "$GPUTYPE" || exit 1
O=/home1/jc_905/pmf/diag/segv_$(echo "$GPUTYPE" | tr A-Z a-z)_$SLURM_JOB_ID; mkdir -p "$O"; cd "$O"
LOCAL=${TMPDIR:-/tmp}/pmfsegv_$SLURM_JOB_ID; mkdir -p "$LOCAL/data/pm13_lin_init"; trap 'rm -rf "$LOCAL"' EXIT
for st in baseline all_features; do cp -r /home1/jc_905/pm13/data/pm13_$st "$LOCAL/data/"; cp -r /home1/jc_905/pm13/data/pm13_lin_init/$st "$LOCAL/data/pm13_lin_init/"; done
cp /home1/jc_905/pmf/bin/forecast_train "$LOCAL/ft"
grep "id=pwalk_baseline_r44250_s20261012" /home1/jc_905/pmf/inputs/gate_walk_prev.txt > "$LOCAL/one.txt"
cp /home1/jc_905/pmf/inputs/gate_tune_prev.txt "$LOCAL/tune6.txt"
cd "$LOCAL"
# v2 (the single refit ran cleanly on V100 and A40): the gate's own setting, its 6 tune jobs of both sets at
# concurrency 6, with and without CUDA_DEVICE_MAX_CONNECTIONS=32 (the gate set 32), under gdb
for mc in 32 8; do
  CUDA_DEVICE_MAX_CONNECTIONS=$mc gdb -q -batch -ex run -ex bt -ex "info threads" --args "$LOCAL/ft" run --jobs "$LOCAL/tune6.txt" --out "$LOCAL/t$mc" --concurrent 6 --chunk 5 --pool > "$O/gdb_tune6_mc$mc.txt" 2>&1
  echo "tune6 MAXCONN $mc: $(grep -cE "SIGSEGV|received signal" "$O/gdb_tune6_mc$mc.txt") signal lines; $(grep -m1 "pool phase" "$O/gdb_tune6_mc$mc.txt" | sed -E 's/.*(pool phase[^(]*).*//')" >> "$O/summary.txt"
  grep -A20 -E "SIGSEGV|received signal" "$O/gdb_tune6_mc$mc.txt" | grep -E "^#|signal" | head -20 >> "$O/summary.txt"
done
{ nvidia-smi; ldd "$LOCAL/ft"; } > "$O/env.txt" 2>&1
gdb -q -batch -ex run -ex bt -ex "info registers rip" -ex "x/6i \$pc" --args "$LOCAL/ft" run --jobs "$LOCAL/one.txt" --out "$LOCAL/g" --concurrent 1 --chunk 5 --pool > "$O/gdb.txt" 2>&1
echo "gdb exit $?" >> "$O/summary.txt"
grep -A25 "SIGSEGV\|Program received" "$O/gdb.txt" | head -40 >> "$O/summary.txt"
timeout 600 compute-sanitizer --tool memcheck --show-backtrace device "$LOCAL/ft" run --jobs "$LOCAL/one.txt" --out "$LOCAL/s" --concurrent 1 --chunk 5 --pool > "$O/memcheck.txt" 2>&1
echo "memcheck exit $?" >> "$O/summary.txt"
grep -m20 -E "Invalid|ERROR SUMMARY|error|Program hit" "$O/memcheck.txt" >> "$O/summary.txt"
cat "$O/summary.txt"
