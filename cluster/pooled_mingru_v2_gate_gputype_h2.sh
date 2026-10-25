#!/bin/bash
# 2026-10-25 MinGRU v2, RULE ADDENDUM 1 VALIDATION of Hoffman2's UGE L40S (gpu_l40s.q; RULE ADDENDUM 3f): CARC's GPU-type
# gate (cluster/pooled_mingru_final_gate_gputype_carc.sbatch) ported to UGE, the same in substance: the P100 gate's jobs
# (inputs/gate_tune_prev.txt: fold-3 tune of the previous frozen configuration, 3 seeds a set; inputs/gate_walk_prev.txt:
# the first walk origin, 3 seeds a set) on the previous design (data/pm13_*, node-local, md5-checked), the same binary
# (md5 44ab6fff) and command line (--concurrent 6 --chunk 5 --pool, CUDA_DEVICE_MAX_CONNECTIONS 32), then the tune file a
# second time (self-consistency).  Inputs, reference runs and runtime copied from CARC with an md5 manifest
# (inputs/carc_manifest.md5).
# Runtime = CARC's bytes: Hoffman2's CentOS 7 glibc 2.17 cannot load CARC's cuBLAS / cuRAND (they need GLIBC_2.27), so the
# unmodified binary is started by CARC's own loader, carclib/ld-linux-x86-64.so.2 --library-path carclib/: CARC's glibc
# 2.28-251.el8_10.40 (libc, libm, libpthread, librt, libdl), cuBLAS / cuBLASLt 12.9.1.4 and cuRAND 10.3.10.19 of CARC's
# cuda/12.9.1 module, libstdc++ 6.0.33 / libgcc_s of its gcc/14.3.0 -- what CARC's ldd resolves; only the NVIDIA driver
# (libcuda) is Hoffman2's.  The job refuses any GPU that is not an L40S (the GPU UGE assigned, and every GPU of the node).
# Checks: experiments/pooled_mingru_v2_h2_gate_check.py (in the t3t Debian container's venv) -> runs/gate_h2l40s_verdict_<job>.
#   cd /u/scratch/j/jamesdc1/pmf2v_h2 && qsub -N v2h2gt -pe shared 2 -l gpu,L40S,cuda=1,h_rt=0:40:00,h_data=12G code/pooled_mingru_v2_gate_gputype_h2.sh
#$ -cwd
#$ -j y
#$ -o logs/
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
export CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_ORDER=PCI_BUS_ID
ROOT=/u/scratch/j/jamesdc1/pmf2v_h2
T=h2l40s
JOB=${JOB_ID:?JOB_ID}
CL=$ROOT/carclib
LDSO=$CL/ld-linux-x86-64.so.2
RT="$LDSO --library-path $CL"
mkdir -p "$ROOT/runs"
# the GPU: exactly one, the one UGE assigned, an L40S, on a node whose GPUs are all L40S
CVD=${CUDA_VISIBLE_DEVICES:-}
{ [ -n "$CVD" ] && [ "${CVD#*,}" = "$CVD" ]; } || { echo "CUDA_VISIBLE_DEVICES '$CVD': exactly one GPU expected"; exit 1; }
GOT=$(nvidia-smi -i "$CVD" --query-gpu=name --format=csv,noheader)
echo "$(hostname) GPU $CVD: $GOT"
[ "$GOT" = "NVIDIA L40S" ] || { echo "got a '$GOT', an L40S required: refusing"; exit 1; }
[ -z "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vx 'NVIDIA L40S')" ] || { echo "a GPU of this node is not an L40S: refusing"; exit 1; }
# runtime: CARC's libraries (md5 = CARC's), every library of the binary resolved from carclib
(cd "$ROOT" && grep '  carclib/' inputs/carc_manifest.md5 | md5sum -c --quiet) || { echo "carclib md5"; exit 1; }
if $RT --list "$ROOT/bin/forecast_train" | grep "=>" | grep -v "=> $CL/"; then echo "a library resolves outside carclib"; exit 1; fi
LOCAL=${TMPDIR:-/tmp}/v2h2gt_$JOB
mkdir -p "$LOCAL/data/pm13_lin_init"
trap 'rm -rf "$LOCAL"' EXIT
for s in baseline all_features; do
  cp -r "$ROOT/data/pm13_$s" "$LOCAL/data/"
  cp -r "$ROOT/data/pm13_lin_init/$s" "$LOCAL/data/pm13_lin_init/"
done
(cd "$ROOT" && grep -E '  data/pm13_' inputs/carc_manifest.md5 | sed 's,  data/,  ,') > "$LOCAL/data.md5"
(cd "$LOCAL/data" && md5sum -c --quiet "$LOCAL/data.md5") || { echo "stage data"; exit 1; }
[ "$(find "$LOCAL/data" -type f | wc -l)" = "$(wc -l < "$LOCAL/data.md5")" ] || { echo "stage data: file count"; exit 1; }
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
ENVF=$ROOT/runs/gate_${T}_env_$JOB.txt
{ echo "JOB_ID=$JOB QUEUE=${QUEUE:-} HOST=$(hostname) CUDA_VISIBLE_DEVICES=$CVD SGE_HGR_cuda=${SGE_HGR_cuda:-unset} NSLOTS=${NSLOTS:-}"
  date; cat /etc/redhat-release; uname -r; grep -m1 'model name' /proc/cpuinfo; nproc
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv
  nvidia-smi | head -4 | tail -2
  echo "--- runtime: $RT"; $LDSO --version | head -1; $RT --list "$LOCAL/ft"
  md5sum "$ROOT/bin/forecast_train" "$ROOT"/inputs/gate_tune_prev.txt "$ROOT"/inputs/gate_walk_prev.txt
  echo "--- carclib (md5 = CARC's)"; grep '  carclib/' "$ROOT/inputs/carc_manifest.md5"; } > "$ENVF" 2>&1
cd "$LOCAL"
for k in tune refit tune2; do
  case $k in tune|tune2) J=gate_tune_prev.txt; SUB=tune ;; refit) J=gate_walk_prev.txt; SUB=refit ;; esac
  RUN=$ROOT/runs/gate_${T}_${k}_$JOB
  mkdir -p "$LOCAL/$k"  # the trainer does not create a nested --out
  T0=$(date +%s%N)
  $RT "$LOCAL/ft" run --jobs "$ROOT/inputs/$J" --out "$LOCAL/$k/$SUB" --concurrent 6 --chunk 5 --pool > "$LOCAL/$k.log" 2>&1 &
  FT=$!
  ( sleep 1; echo "--- $k: shared objects mapped by the trainer (pid $FT)"; grep -o '/[^ ]*\.so[^ ]*' "/proc/$FT/maps" | sort -u
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader ) >> "$ENVF" 2>&1 &
  SMP=$!
  set +e; wait "$FT"; RC=$?; wait "$SMP"; set -e
  MS=$(( ($(date +%s%N) - T0) / 1000000 ))
  if [ "$RC" != 0 ]; then
    echo "$k: trainer exit $RC"; tail -20 "$LOCAL/$k.log"
    mkdir -p "$RUN"; cp -r "$LOCAL/$k" "$LOCAL/$k.log" "$RUN/" 2>/dev/null || true; exit "$RC"
  fi
  [ -f "$LOCAL/$k/$SUB/steps.csv" ] && gzip -f "$LOCAL/$k/$SUB/steps.csv"
  mkdir -p "$RUN"; cp -r "$LOCAL/$k/$SUB" "$RUN/"; cp "$LOCAL/$k.log" "$RUN/$SUB.log"
  { hostname; echo "$GOT"; nvidia-smi -i "$CVD" --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo "wall_ms $MS"
    echo "GPUMODEL=$GOT CLUSTER=hoffman2 JOB_ID=$JOB CUDA_VISIBLE_DEVICES=$CVD RUNTIME=carclib"; } > "$RUN/env.txt"
  echo "$k: $MS ms"
done
set +e
A=$ROOT/runs/gate_${T}_tune_$JOB/tune/preds.csv; B=$ROOT/runs/gate_${T}_tune2_$JOB/tune/preds.csv
if cmp -s <(sort "$A") <(sort "$B"); then echo "SELF-CONSISTENT (H2 L40S: tune twice, preds identical)"; else echo "NOT SELF-CONSISTENT (H2 L40S)"; fi
module load apptainer/1.2.2
apptainer exec -B /u/scratch /u/scratch/j/jamesdc1/t3t/deb.sif env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /u/scratch/j/jamesdc1/t3t/venv_gb/bin/python "$ROOT/code/pooled_mingru_v2_h2_gate_check.py" --root "$ROOT" --job "$JOB" --tag "$T" \
  > "$ROOT/runs/gate_${T}_check_$JOB.log" 2>&1
echo "gate check exit $?"; tail -1 "$ROOT/runs/gate_${T}_check_$JOB.log"
date; echo V2H2GT_DONE
