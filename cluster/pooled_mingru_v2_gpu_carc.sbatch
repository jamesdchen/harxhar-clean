#!/bin/bash
# 2026-10-23 MinGRU v2 (experiments/pooled_mingru_v2.py; RULE results/pooled_mingru_v2_2014start_w3000/RULE_gru_v2.txt):
# one GPU job runs one jobs file of the native_g9 trainer in pool mode, as cluster/pooled_mingru_final_gpu_carc.sbatch
# (node-local staging, outputs copied back with an md5 check, a USR1 salvage of partial outputs), with the old run's
# design copies read from XDATA (read-only) and this run's ridge inits from ROOT.
#   ROOT   /home1/jc_905/pmf2 (W 3,000) or /home1/jc_905/pmf2_35 (W 3,500): bin/, code/, inputs/, data/pmf_lin_init/, runs/
#   XDATA  the old run's data root (default by ROOT: /home1/jc_905/pmf/data or /home1/jc_905/pmf35/data), read-only
#   JOBS   a file under $ROOT/inputs; SUB tune | refit; TAG run name ($ROOT/runs/${TAG}_$SLURM_JOB_ID); CONC models at once
#   INITSUB optional ridge-init subdirectory (bar24 for the G7 walk probe)
#   GPUTYPE the GPU model required (default P100; RULE ADDENDUM 1: a validated type, one type per comparison group);
#           submit with the matching --constraint (the submit script does); the job exits on any other model
# POST: experiments/pooled_mingru_v2.py post (check + epoch curves + compact forecasts into the run's summary/).
# 2026-10-23 23:00 PDT: ROOT may be /scratch1/jc_905/pmf2(_35) (the /home1/jc_905 quota was exhausted at 22:38 PDT);
# a trainer that exits non-zero now leaves its log and partial outputs in ROOT/salvage (they were lost with LOCAL).
# 2026-10-24 walk files: NOSTEPS=1 points the trainer's per-step log (steps.csv) at /dev/null (forecasts unchanged;
# the slice check pooled_mingru_v2_slice_carc.sbatch verifies it); PARTIAL_S=<s> copies jobs_out.csv / preds.csv to
# RUN/partial every s seconds while the trainer runs (a node failure loses at most that much).
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --job-name=v2
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=${ROOT:?ROOT}
case "$ROOT" in */pmf2_35) WD=3500; XD=/home1/jc_905/pmf35/data ;; */pmf2) WD=3000; XD=/home1/jc_905/pmf/data ;; *) echo "ROOT $ROOT"; exit 1 ;; esac
XDATA=${XDATA:-$XD}
JOBS=${JOBS:?JOBS}
SUB=${SUB:?SUB tune or refit}
TAG=${TAG:?TAG}
CONC=${CONC:-6}
GPUTYPE=${GPUTYPE:-P100}
# the python env from /home1 (a broken /scratch1 Lustre client on some GPU nodes raised SIGBUS: old STATUS.md 2026-10-23)
PY=${PY:-/home1/jc_905/envs/harxhar-r13/bin/python}
EP=$(dirname "$(dirname "$PY")")
export OPENSSL_CONF=$EP/ssl/openssl.cnf OPENSSL_MODULES=$EP/lib/ossl-modules OPENSSL_ENGINES=$EP/lib/engines-3
RUN=$ROOT/runs/${TAG}_$SLURM_JOB_ID
case "$JOBS" in *all_features*) SET=all_features ;; *baseline*) SET=baseline ;; *) SET="baseline all_features" ;; esac
LOCAL=${TMPDIR:-/tmp}/v2_$SLURM_JOB_ID
mkdir -p "$LOCAL/data/pmf_lin_init"
trap 'rm -rf "$LOCAL"' EXIT
for s in $SET; do
  mkdir -p "$LOCAL/data/pmf_$s" "$LOCAL/data/pmf_lin_init/$s"
  for f in X.f32 X.f64 y.f32 enc_pf.f64 meta.txt columns.txt; do
    cp "$XDATA/pmf_$s/$f" "$LOCAL/data/pmf_$s/"
    [ "$(stat -c %s "$XDATA/pmf_$s/$f")" = "$(stat -c %s "$LOCAL/data/pmf_$s/$f")" ] || { echo "stage $s/$f"; exit 1; }
  done
  find "$ROOT/data/pmf_lin_init/$s" -maxdepth 1 -type f -exec cp -t "$LOCAL/data/pmf_lin_init/$s/" {} +
  [ "$(find "$ROOT/data/pmf_lin_init/$s" -maxdepth 1 -type f | wc -l)" = "$(find "$LOCAL/data/pmf_lin_init/$s" -maxdepth 1 -type f | wc -l)" ] || { echo "stage inits $s"; exit 1; }
  if [ -n "${INITSUB:-}" ]; then
    cp -r "$ROOT/data/pmf_lin_init/$s/$INITSUB" "$LOCAL/data/pmf_lin_init/$s/"
    [ "$(ls "$ROOT/data/pmf_lin_init/$s/$INITSUB" | wc -l)" = "$(ls "$LOCAL/data/pmf_lin_init/$s/$INITSUB" | wc -l)" ] || { echo "stage inits $s/$INITSUB"; exit 1; }
  fi
done
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$LOCAL/data" data
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/bin/forecast_train" "$ROOT/inputs/$JOBS"; echo "XDATA=$XDATA CONC=$CONC WD=$WD GPUTYPE=$GPUTYPE"; } > env.txt 2>&1
echo "$GOT" | grep -qi "$GPUTYPE" || { echo "got a $GOT, $GPUTYPE required"; exit 1; }
cp "$ROOT/bin/forecast_train" "$LOCAL/ft"
[ "$(md5sum < "$LOCAL/ft" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
cp "$ROOT/inputs/$JOBS" "$LOCAL/jobs.txt"
T0=$(date +%s)
cd "$LOCAL"
if [ "${NOSTEPS:-0}" = 1 ]; then mkdir -p "$LOCAL/$SUB"; ln -sfn /dev/null "$LOCAL/$SUB/steps.csv"; fi
SYNCPID=""
salvage() {
  [ -n "$SYNCPID" ] && kill "$SYNCPID" 2>/dev/null || true
  S=$ROOT/salvage/${TAG}_$SLURM_JOB_ID; mkdir -p "$S/$SUB"
  for f in jobs_out.csv preds.csv epochs.csv; do [ -f "$LOCAL/$SUB/$f" ] && cp "$LOCAL/$SUB/$f" "$S/$SUB/"; done
  cp "$LOCAL/$SUB.log" "$S/" 2>/dev/null
  echo "time limit near: partial outputs salvaged to $S"
  kill "$FTPID" 2>/dev/null; exit 3
}
trap salvage USR1
"$LOCAL/ft" run --jobs "$LOCAL/jobs.txt" --out "$LOCAL/$SUB" --concurrent "$CONC" --chunk 5 --pool > "$LOCAL/$SUB.log" 2>&1 &
FTPID=$!
if [ "${PARTIAL_S:-0}" -gt 0 ]; then
  ( while sleep "$PARTIAL_S"; do mkdir -p "$RUN/partial"
      for f in jobs_out.csv preds.csv; do cp "$LOCAL/$SUB/$f" "$RUN/partial/$f.tmp" 2>/dev/null && mv -f "$RUN/partial/$f.tmp" "$RUN/partial/$f"; done
    done ) &
  SYNCPID=$!
fi
set +e; wait "$FTPID"; RC=$?; set -e
[ -n "$SYNCPID" ] && { kill "$SYNCPID" 2>/dev/null || true; wait "$SYNCPID" 2>/dev/null || true; SYNCPID=""; }  # 2026-10-24 15:35: || true (set -e killed the job here with 143 before the copy back)
[ -L "$LOCAL/$SUB/steps.csv" ] && rm -f "$LOCAL/$SUB/steps.csv"
if [ "$RC" != 0 ]; then
  S=$ROOT/salvage/${TAG}_$SLURM_JOB_ID; mkdir -p "$S"
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
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PMF_WINDOW_DAYS=$WD PMF_REPO=/home1/jc_905/pmf/repo
mkdir -p summary; cp env.txt summary/; grep -h "pool phase" "$SUB.log" > summary/pool.txt || true
"$PY" "$ROOT/code/pooled_mingru_v2.py" post --run "$RUN" --jobs "$ROOT/inputs/$JOBS"
date; echo V2_DONE
