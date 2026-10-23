#!/bin/bash
# 2026-10-22 final pooled ridge-init MinGRU (experiments/pooled_mingru_final_2014_w3000.py): one GPU job runs one jobs
# file of the native_g9 trainer (bin/forecast_train md5 44ab6fff, .cu af1964d0, the previous run's binary) in pool mode,
# exactly as cluster/pooled_mingru13_gpu_carc.sbatch (staging on the node's local disk, outputs copied back with an
# md5 check), generalised to a data root and prefix so that the gate can run the previous design's data unchanged.
#   ROOT  (default /home1/jc_905/pmf): bin/, inputs/ (jobs files), runs/, logs/, code/, data/
#   DATA  the data root staged on the node (default $ROOT/data; the gate: /home1/jc_905/pm13/data)
#   PFX   the data directories' prefix: pmf (this run, data/pmf_<set>/, data/pmf_lin_init/<set>/) or pm13 (the gate)
#   JOBS  a file under $ROOT/inputs; SUB tune | refit; TAG run name (run dir $ROOT/runs/${TAG}_$SLURM_JOB_ID)
#   CONC  concurrent models (the previous run: 6 on all features, 12 on the HAR benchmark)
#   POST  optional: a python script under $ROOT/code, run after the copy-back with the run directory as argument
#   INITSUB  optional: the ridge-init subdirectory of a bar-refit walk block (bar_b<k>), staged with the set's inits
# GPU type: P100 by default (--constraint=p100, the previous run's); GPUTYPE + --constraint select another (ADDENDUM 3);
# the job exits if it lands on another model.
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --job-name=pmf
#SBATCH --output=/home1/jc_905/pmf/logs/%x_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=${ROOT:-/home1/jc_905/pmf}
DATA=${DATA:-$ROOT/data}
PFX=${PFX:-pmf}
JOBS=${JOBS:?JOBS}
SUB=${SUB:?SUB tune or refit}
TAG=${TAG:?TAG}
CONC=${CONC:-6}
# 2026-10-23: the python env runs from /home1 (an md5-verified copy of /scratch1/jc_905/envs/harxhar-r13). On some GPU
# nodes the /scratch1 Lustre client is broken (df: connection timed out; reads: "Input/output error", "Cannot send
# after transport endpoint shutdown"), and paging in the env's memory-mapped .so files there raised SIGBUS (the gate
# checks of 13319779 / 13319780; reproduced on e21-09 by 13322650). Nothing a GPU job reads is on /scratch1 now.
PY=${PY:-/home1/jc_905/envs/harxhar-r13/bin/python}
# the env's libcrypto has the /scratch1 prefix compiled in (OPENSSLDIR, ENGINESDIR, MODULESDIR): point it at the copy
EP=$(dirname "$(dirname "$PY")")
export OPENSSL_CONF=$EP/ssl/openssl.cnf OPENSSL_MODULES=$EP/lib/ossl-modules OPENSSL_ENGINES=$EP/lib/engines-3
RUN=$ROOT/runs/${TAG}_$SLURM_JOB_ID
case "$JOBS" in *all_features*) SET=all_features ;; *baseline*) SET=baseline ;; *) SET="baseline all_features" ;; esac
LOCAL=${TMPDIR:-/tmp}/pmf_$SLURM_JOB_ID
mkdir -p "$LOCAL/data/${PFX}_lin_init"
trap 'rm -rf "$LOCAL"' EXIT
# 2026-10-22 (bar refits): the set's top-level ridge inits (tune, canary) and, for a walk block, only its own
# subdirectory INITSUB (bar_b<k>/), not the other blocks' (34 blocks of 1,000 origins a set)
for s in $SET; do
  cp -r "$DATA/${PFX}_$s" "$LOCAL/data/"
  mkdir -p "$LOCAL/data/${PFX}_lin_init/$s"
  find "$DATA/${PFX}_lin_init/$s" -maxdepth 1 -type f -exec cp -t "$LOCAL/data/${PFX}_lin_init/$s/" {} +
  for f in X.f32 X.f64 y.f32 enc_pf.f64 meta.txt; do
    [ "$(stat -c %s "$DATA/${PFX}_$s/$f")" = "$(stat -c %s "$LOCAL/data/${PFX}_$s/$f")" ] || { echo "stage $s/$f"; exit 1; }
  done
  [ "$(find "$DATA/${PFX}_lin_init/$s" -maxdepth 1 -type f | wc -l)" = "$(find "$LOCAL/data/${PFX}_lin_init/$s" -maxdepth 1 -type f | wc -l)" ] || { echo "stage inits $s"; exit 1; }
  if [ -n "${INITSUB:-}" ]; then
    cp -r "$DATA/${PFX}_lin_init/$s/$INITSUB" "$LOCAL/data/${PFX}_lin_init/$s/"
    [ "$(ls "$DATA/${PFX}_lin_init/$s/$INITSUB" | wc -l)" = "$(ls "$LOCAL/data/${PFX}_lin_init/$s/$INITSUB" | wc -l)" ] || { echo "stage inits $s/$INITSUB"; exit 1; }
  fi
done
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$LOCAL/data" data
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/bin/forecast_train" "$ROOT/inputs/$JOBS"; cat "$ROOT/bin/provenance.txt"; echo "DATA=$DATA PFX=$PFX CONC=$CONC"; } > env.txt 2>&1
# 2026-10-23: the window's GPU type (RULE_gru.txt ADDENDUM 3; default P100); submit with --constraint=<its feature>
GPUTYPE=${GPUTYPE:-P100}
echo "$GOT" | grep -qi "$GPUTYPE" || { echo "got a $GOT, $GPUTYPE required"; exit 1; }
echo "$GPUTYPE|$GOT" > gpu_type.txt
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
cp "$ROOT/inputs/$JOBS" "$LOCAL/jobs.txt"
T0=$(date +%s)
cd "$LOCAL"
"$LOCAL/ft" run --jobs "$LOCAL/jobs.txt" --out "$LOCAL/$SUB" --concurrent "$CONC" --chunk 5 --pool > "$LOCAL/$SUB.log" 2>&1
[ -f "$LOCAL/$SUB/steps.csv" ] && gzip -f "$LOCAL/$SUB/steps.csv"
(cd "$LOCAL" && find "$SUB" -type f -exec md5sum {} + | sort -k2 > "$SUB.md5")
cd "$RUN"
cp -r "$LOCAL/$SUB" "$LOCAL/$SUB.log" "$LOCAL/$SUB.md5" "$RUN/"
md5sum -c --quiet "$SUB.md5" || { echo "copy of $SUB to $RUN does not match"; exit 1; }
echo "wall_s $(( $(date +%s) - T0 ))" >> env.txt
rm -f data
if [ -n "${POST:-}" ]; then
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  "$PY" "$ROOT/code/$POST" "$RUN"
fi
date; echo PMF_DONE
