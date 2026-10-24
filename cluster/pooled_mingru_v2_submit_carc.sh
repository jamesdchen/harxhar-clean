#!/bin/bash
# 2026-10-23 MinGRU v2: submit the GPU jobs of one files list (inputs/files_<tag>.csv: file,set,conc,n_jobs,est_gpu_s,
# tlim_min) from a window's CARC root.  One GPU job per file; prints "<file> <job id>" and appends it to
# inputs/submitted.txt.
#   usage: bash pooled_mingru_v2_submit_carc.sh <root> <tag> <sub tune|refit> <partition gpu|debug> [nice] [dependency] [initsub]
# The debug partition's limit is 1 h (a file's time limit is capped there).  W 3,500 (root pmf2_35) jobs pass a nice
# value so that the W 3,000 jobs of the same priority go first.  Carriage returns of a list written on Windows are
# dropped.
set -euo pipefail
ROOT=$1; TAG=$2; SUB=$3; PART=$4; NICE=${5:-0}; DEP=${6:-}; ISUB=${7:-}
G=/home1/jc_905/pmf2/code/pooled_mingru_v2_gpu_carc.sbatch
LIST=$ROOT/inputs/files_$TAG.csv
[ -f "$LIST" ] || { echo "no $LIST"; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/runs"
tail -n +2 "$LIST" | sed 's/\r$//' | while IFS=, read -r file set conc njobs est tl; do
  [ "$PART" = debug ] && [ "$tl" -gt 60 ] && tl=60
  rtag=${file%.txt}; rtag=${rtag//./_}
  id=$(sbatch --parsable -p "$PART" --time="$tl" --nice="$NICE" ${DEP:+--dependency=$DEP} --job-name="v2${SUB:0:1}" \
       -o "$ROOT/logs/%x_%j.out" \
       --export=NONE,ROOT=$ROOT,JOBS=$file,SUB=$SUB,TAG=$rtag,CONC=$conc,INITSUB=$ISUB "$G")
  echo "$file $id $PART $tl" | tee -a "$ROOT/inputs/submitted.txt"
done
