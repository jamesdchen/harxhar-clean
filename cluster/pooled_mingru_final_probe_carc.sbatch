#!/bin/bash
# 2026-10-22 final pooled MinGRU, bar refits (RULE_gru.txt ADDENDUM 2): the timing probe, one P100 allocation for both
# windows and both sets.  Each root's inputs/probe_<set>.txt (the first PROBE origins of walk block 0 x 5 seeds, lines
# identical to the walk's) runs with the trainer at the walk's concurrency C (6 full set, 12 HAR) and at 2C and 4C;
# outputs to <root>/runs/probe_<set>_c<conc>_<job>/refit (+ refit.log with the pool phase's wall time), then
# experiments/pooled_mingru_final_probe.py summarises seconds per refit and checks that the forecasts are identical
# across concurrencies.  ROOTS (default "/home1/jc_905/pmf /home1/jc_905/pmf35"), PSETS (default "baseline all_features").
#SBATCH --account=pollok_1603
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=pmfprobe
#SBATCH --output=/home1/jc_905/pmf/logs/%x_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOTS=${ROOTS:-/home1/jc_905/pmf /home1/jc_905/pmf35}
MULTS=${MULTS:-1 2 4}
# 2026-10-23: the python env runs from /home1 (an md5-verified copy of /scratch1/jc_905/envs/harxhar-r13). On some GPU
# nodes the /scratch1 Lustre client is broken (df: connection timed out; reads: "Input/output error", "Cannot send
# after transport endpoint shutdown"), and paging in the env's memory-mapped .so files there raised SIGBUS (the gate
# checks of 13319779 / 13319780; reproduced on e21-09 by 13322650). Nothing a GPU job reads is on /scratch1 now.
PY=${PY:-/home1/jc_905/envs/harxhar-r13/bin/python}
# the env's libcrypto has the /scratch1 prefix compiled in (OPENSSLDIR, ENGINESDIR, MODULESDIR): point it at the copy
EP=$(dirname "$(dirname "$PY")")
export OPENSSL_CONF=$EP/ssl/openssl.cnf OPENSSL_MODULES=$EP/lib/ossl-modules OPENSSL_ENGINES=$EP/lib/engines-3
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(hostname) $GOT"
GPUTYPE=${GPUTYPE:-P100}  # ADDENDUM 3: the window's GPU type (submit with --constraint=<feature>, ROOTS=<that window>)
echo "$GOT" | grep -qi "$GPUTYPE" || { echo "got a $GOT, $GPUTYPE required"; exit 1; }
LOCAL=${TMPDIR:-/tmp}/pmfprobe_$SLURM_JOB_ID
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
FT=$LOCAL/ft
cp /home1/jc_905/pmf/bin/forecast_train "$FT"
[ "$(md5sum < "$FT" | cut -c1-32)" = 44ab6fff270e15547335c6b36b3ceead ] || { echo "binary md5"; exit 1; }
RUNS=""
for R in $ROOTS; do
  for s in ${PSETS:-baseline all_features}; do
    if [ $s = all_features ]; then C=6; else C=12; fi
    W=$LOCAL/$(basename "$R")_$s
    mkdir -p "$W/data/pmf_lin_init/$s"
    cp -r "$R/data/pmf_$s" "$W/data/"
    find "$R/data/pmf_lin_init/$s" -maxdepth 1 -type f -exec cp -t "$W/data/pmf_lin_init/$s/" {} +
    cp -r "$R/data/pmf_lin_init/$s/bar_b0" "$W/data/pmf_lin_init/$s/"
    cp "$R/inputs/probe_$s.txt" "$W/jobs.txt"
    for m in $MULTS; do
      conc=$((C * m)); tag=probe_${s}_c${conc}
      cd "$W"
      T0=$(date +%s%N)
      "$FT" run --jobs "$W/jobs.txt" --out "$W/$tag" --concurrent "$conc" --chunk 5 --pool > "$W/$tag.log" 2>&1
      T1=$(date +%s%N); MS=$(( (T1 - T0) / 1000000 ))
      [ -f "$W/$tag/steps.csv" ] && gzip -f "$W/$tag/steps.csv"
      RUN=$R/runs/${tag}_$SLURM_JOB_ID
      mkdir -p "$RUN"
      cp -r "$W/$tag" "$RUN/refit"; cp "$W/$tag.log" "$RUN/refit.log"
      { hostname; echo "$GOT"; echo "conc $conc"; echo "wall_ms $MS"; md5sum "$R/inputs/probe_$s.txt"; } > "$RUN/env.txt"
      echo "$R $s conc $conc wall $MS ms"
      RUNS="$RUNS $RUN"
    done
    cd "$LOCAL"; rm -rf "$W"
  done
done
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"$PY" /home1/jc_905/pmf/code/pooled_mingru_final_probe.py $RUNS
date; echo PMFPROBE_DONE
