#!/bin/bash
# Submit one window as a chain of Slurm jobs (account pollok_1603): design (2 CPUs) -> pretune + select (4 CPUs) ->
# 14 walk tasks (array, 1 CPU each) -> score.
# usage: submit_window.sh <tag> <out dir> <window days> <first fold> [first stage] [env list] [constraint]
#   first stage: design (default) | pretune | walk | score -- later stages only, for a resubmission
#   env list: for stage.sh, default WS_NPY=1,WS_NUMBA=0; constraint: Slurm --constraint (default: none)
set -euo pipefail
B=/scratch1/jc_905/wsweep
TAG=$1; OUT=$2; WD=$3; FF=$4; FROM=${5:-design}; V=${6:-WS_NPY=1,WS_NUMBA=0}; C=${7:-}
L=$B/logs
mkdir -p $L
cd $B/jobs
A=(--account=pollok_1603 --partition=main --export=ALL,$V)
[ -n "$C" ] && A+=(--constraint="$C")
dep=()
if [ "$FROM" = designonly ]; then  # a design test: the design stage alone
  d=$(sbatch --parsable "${A[@]}" -J wst_$TAG -o $L/$TAG.design.log -c 2 --mem=24G -t 2:00:00 stage.sh "$OUT" "$WD" "$FF" design 2)
  echo "design test $d"; exit 0
fi
if [ "$FROM" = pretuneonly ]; then  # pretune + select alone (the walks submitted later with first stage "walk")
  p=$(sbatch --parsable "${A[@]}" -J wsp_$TAG -o $L/$TAG.pretune.log -c 4 --mem=16G -t 2:00:00 stage.sh "$OUT" "$WD" "$FF" pretune_select 4)
  echo "pretune $p"; exit 0
fi
if [ "$FROM" = design ]; then
  d=$(sbatch --parsable "${A[@]}" -J wsd_$TAG -o $L/$TAG.design.log -c 2 --mem=24G -t 4:00:00 stage.sh "$OUT" "$WD" "$FF" design 2)
  dep=(--dependency=afterok:$d); echo "design $d"
fi
if [ "$FROM" = design ] || [ "$FROM" = pretune ]; then
  p=$(sbatch --parsable "${A[@]}" "${dep[@]}" -J wsp_$TAG -o $L/$TAG.pretune.log -c 4 --mem=16G -t 2:00:00 stage.sh "$OUT" "$WD" "$FF" pretune_select 4)
  dep=(--dependency=afterok:$p); echo "pretune $p"
fi
if [ "$FROM" != score ]; then
  w=$(sbatch --parsable "${A[@]}" "${dep[@]}" -J wsw_$TAG -o "$L/$TAG.walk.%a.log" --array=1-14 -c 1 --mem=6G -t 4:00:00 stage.sh "$OUT" "$WD" "$FF" walk1)
  dep=(--dependency=afterok:$w); echo "walk $w"
fi
s=$(sbatch --parsable "${A[@]}" "${dep[@]}" -J wss_$TAG -o $L/$TAG.score.log -c 1 --mem=8G -t 1:00:00 stage.sh "$OUT" "$WD" "$FF" score)
echo "score $s"
