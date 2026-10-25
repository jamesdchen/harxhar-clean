#!/bin/bash
# 2026-10-25 MinGRU v2 on Hoffman2's UGE L40S (gpu_l40s.q; RULE ADDENDUM 3f): one jobs file of the native_g9 trainer in
# pool mode, as cluster/pooled_mingru_v2_gpu_carc.sbatch (node-local staging of the design and the bar inits, md5-checked
# binary 44ab6fff, NOSTEPS = steps.csv to /dev/null, outputs copied back with an md5 check, POST = experiments/
# pooled_mingru_v2.py post into RUN/summary/), ported to UGE for the full-set W 3,500 arm (ROOT /u/scratch/j/jamesdc1/
# pmf2v_h2: bin/, code/, inputs/, data/ = CARC's bytes, inputs/carc_manifest.md5).
# Runtime = CARC's bytes, as cluster/pooled_mingru_v2_gate_gputype_h2.sh: the unmodified binary started by CARC's glibc 2.28
# loader with carclib/ (CARC's glibc, cuBLAS 12.9.1.4, cuRAND 10.3.10.19, gcc 14.3.0 runtime); only libcuda is Hoffman2's.
# The job refuses any GPU that is not an L40S.  REQUIRE_GATE=<job id>: run only if that Hoffman2 GPU-type gate passed
# (runs/gate_h2l40s_verdict_<id>.txt "VALIDATION PASS"); env.txt's GPUTYPE is the gate's TYPE (L40S if byte-identical to
# CARC's L40S gate, else H2-L40S), so walk-merge --gputype keeps types apart.  No time-limit salvage (UGE sends no USR1).
#   cd /u/scratch/j/jamesdc1/pmf2v_h2 && qsub -N v2h2c -pe shared 2 -l gpu,L40S,cuda=1,h_rt=0:30:00,h_data=8G \
#     -v JOBS=pieces/canary/v2w_all_features.000.c24.txt,TAG=v2wcanary_all_features_H2L40S,CONC=24,NOSTEPS=1,INITSUB=w000,REQUIRE_GATE=<id> \
#     code/pooled_mingru_v2_gpu_h2.sh
#$ -cwd
#$ -j y
#$ -o logs/
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
export CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_ORDER=PCI_BUS_ID
ROOT=/u/scratch/j/jamesdc1/pmf2v_h2
WD=3500
XDATA=$ROOT/data
JOBS=${JOBS:?JOBS}
SUB=${SUB:-refit}
TAG=${TAG:?TAG}
CONC=${CONC:-24}
INITSUB=${INITSUB:?INITSUB}
JOB=${JOB_ID:?JOB_ID}
CL=$ROOT/carclib
LDSO=$CL/ld-linux-x86-64.so.2
RT="$LDSO --library-path $CL"
GPUTYPE=H2-L40S
if [ -n "${REQUIRE_GATE:-}" ]; then
  V=$ROOT/runs/gate_h2l40s_verdict_$REQUIRE_GATE.txt
  grep -qx "VALIDATION PASS" "$V" 2>/dev/null || { echo "gate $REQUIRE_GATE did not pass ($V): not run"; exit 0; }
  if grep -qx "TYPE L40S" "$V"; then GPUTYPE=L40S; fi
fi
CVD=${CUDA_VISIBLE_DEVICES:-}
{ [ -n "$CVD" ] && [ "${CVD#*,}" = "$CVD" ]; } || { echo "CUDA_VISIBLE_DEVICES '$CVD': exactly one GPU expected"; exit 1; }
GOT=$(nvidia-smi -i "$CVD" --query-gpu=name --format=csv,noheader)
echo "$(hostname) GPU $CVD: $GOT"
[ "$GOT" = "NVIDIA L40S" ] || { echo "got a '$GOT', an L40S required: refusing"; exit 1; }
[ -z "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vx 'NVIDIA L40S')" ] || { echo "a GPU of this node is not an L40S: refusing"; exit 1; }
(cd "$ROOT" && grep '  carclib/' inputs/carc_manifest.md5 | md5sum -c --quiet) || { echo "carclib md5"; exit 1; }
if $RT --list "$ROOT/bin/forecast_train" | grep "=>" | grep -v "=> $CL/"; then echo "a library resolves outside carclib"; exit 1; fi
case "$JOBS" in *all_features*) SET=all_features ;; *baseline*) SET=baseline ;; *) echo "JOBS $JOBS: one set a file"; exit 1 ;; esac
RUN=$ROOT/runs/${TAG}_$JOB
LOCAL=${TMPDIR:-/tmp}/v2h2_$JOB
mkdir -p "$LOCAL/data/pmf_$SET" "$LOCAL/data/pmf_lin_init/$SET"
trap 'rm -rf "$LOCAL"' EXIT
for f in X.f32 X.f64 y.f32 enc_pf.f64 meta.txt columns.txt; do
  cp "$XDATA/pmf_$SET/$f" "$LOCAL/data/pmf_$SET/"
  [ "$(stat -c %s "$XDATA/pmf_$SET/$f")" = "$(stat -c %s "$LOCAL/data/pmf_$SET/$f")" ] || { echo "stage $SET/$f"; exit 1; }
done
cp -r "$ROOT/data/pmf_lin_init/$SET/$INITSUB" "$LOCAL/data/pmf_lin_init/$SET/"
[ "$(ls "$ROOT/data/pmf_lin_init/$SET/$INITSUB" | wc -l)" = "$(ls "$LOCAL/data/pmf_lin_init/$SET/$INITSUB" | wc -l)" ] || { echo "stage inits $SET/$INITSUB"; exit 1; }
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$LOCAL/data" data
{ hostname; date; nvidia-smi -i "$CVD" --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/bin/forecast_train" "$ROOT/inputs/$JOBS"; echo "XDATA=$XDATA CONC=$CONC WD=$WD GPUTYPE=$GPUTYPE"
  echo "GPUMODEL=$GOT CLUSTER=hoffman2 QUEUE=${QUEUE:-} JOB_ID=$JOB CUDA_VISIBLE_DEVICES=$CVD NSLOTS=${NSLOTS:-} RUNTIME=carclib"; } > env.txt 2>&1
{ cat /etc/redhat-release; uname -r; grep -m1 'model name' /proc/cpuinfo; nproc
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv; nvidia-smi | head -4 | tail -2
  echo "--- runtime: $RT"; $LDSO --version | head -1; $RT --list "$ROOT/bin/forecast_train"; } > env_full.txt 2>&1
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
cp "$ROOT/inputs/$JOBS" "$LOCAL/jobs.txt"
T0=$(date +%s)
cd "$LOCAL"
if [ "${NOSTEPS:-0}" = 1 ]; then mkdir -p "$LOCAL/$SUB"; ln -sfn /dev/null "$LOCAL/$SUB/steps.csv"; fi
$RT "$LOCAL/ft" run --jobs "$LOCAL/jobs.txt" --out "$LOCAL/$SUB" --concurrent "$CONC" --chunk 5 --pool > "$LOCAL/$SUB.log" 2>&1 &
FTPID=$!
( sleep 15; { echo "--- shared objects mapped by the trainer (pid $FTPID)"; grep -o '/[^ ]*\.so[^ ]*' "/proc/$FTPID/maps" | sort -u; } >> "$RUN/env_full.txt" 2>&1
  while kill -0 "$FTPID" 2>/dev/null; do
    nvidia-smi -i "$CVD" --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> "$RUN/gpu_util.csv" 2>/dev/null
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader >> "$RUN/gpu_apps.csv" 2>/dev/null
    sleep 10
  done ) &
SMP=$!
set +e; wait "$FTPID"; RC=$?; set -e
kill "$SMP" 2>/dev/null || true; wait "$SMP" 2>/dev/null || true
[ -L "$LOCAL/$SUB/steps.csv" ] && rm -f "$LOCAL/$SUB/steps.csv"
if [ "$RC" != 0 ]; then
  S=$ROOT/salvage/${TAG}_$JOB; mkdir -p "$S"
  cp -r "$LOCAL/$SUB" "$LOCAL/$SUB.log" "$S/" 2>/dev/null || true
  echo "trainer exit $RC: log and partial outputs in $S"; tail -n 20 "$LOCAL/$SUB.log" || true; exit "$RC"
fi
[ -f "$LOCAL/$SUB/steps.csv" ] && gzip -f "$LOCAL/$SUB/steps.csv"
(cd "$LOCAL" && find "$SUB" -type f -exec md5sum {} + | sort -k2 > "$SUB.md5")
cd "$RUN"
cp -r "$LOCAL/$SUB" "$LOCAL/$SUB.log" "$LOCAL/$SUB.md5" "$RUN/"
md5sum -c --quiet "$SUB.md5" || { echo "copy of $SUB to $RUN does not match"; exit 1; }
echo "wall_s $(( $(date +%s) - T0 ))" >> env.txt
rm -f data
mkdir -p summary; cp env.txt summary/; grep -h "pool phase" "$SUB.log" > summary/pool.txt || true
module load apptainer/1.2.2
apptainer exec -B /u/scratch /u/scratch/j/jamesdc1/t3t/deb.sif env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  PMF_WINDOW_DAYS=$WD PMF_REPO="$ROOT/code" /u/scratch/j/jamesdc1/t3t/venv_gb/bin/python "$ROOT/code/pooled_mingru_v2.py" post \
  --run "$RUN" --jobs "$ROOT/inputs/$JOBS"
date; echo V2_DONE
