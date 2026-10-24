#!/bin/bash
# 2026-10-24 MinGRU v2 every-bar walk (RULE_gru_v2.txt ADDENDUM 2): submit one arm's walk files from a window's root.
#   usage: bash pooled_mingru_v2_walk_submit_carc.sh <root> <set> <partition> <gputype> <nice> [dependency] [hold]
# inputs/files_v2w_<set>.csv (file,set,conc,n_jobs,est_gpu_s,tlim_min,initsub,lane,seq,o0,o1,gpu) from walk-plan;
# inputs/walkinit_<set>.jobid: the array job of the arm's bar inits (walk file i waits afterok on its task i).
# A lane's files run one after another (seq k + 1 waits afterany on seq k: a failed or timed-out file does not
# block its lane; its remainder is rerun from the merge's remainder file). [dependency] is added to every file (the
# slice check). hold = submit held (scontrol release later).  Every walk file runs with NOSTEPS=1 (the per-step log
# to /dev/null) and PARTIAL_S=900 (partial outputs copied to RUN/partial every 15 min).
set -euo pipefail
ROOT=$1; SET=$2; PART=$3; GT=$4; NICE=${5:-0}; DEP=${6:-}; HOLD=${7:-}
FEAT=$(echo "$GT" | tr 'A-Z' 'a-z')
G=$ROOT/code/pooled_mingru_v2_gpu_carc.sbatch
LIST=$ROOT/inputs/files_v2w_$SET.csv
ARR=$(cat "$ROOT/inputs/walkinit_$SET.jobid")
[ -f "$LIST" ] && [ -n "$ARR" ] || { echo "no $LIST or init job"; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/runs"
# a dependency on a finished job fails ("Job dependency problem" once the controller has purged it): an init task
# that already COMPLETED is not a dependency; one still queued or running is (afterok); any other state stops here
dep_task() {
  local st; st=$(sacct -j "$1" -X -n -P -o State 2>/dev/null | head -1)
  case "$st" in COMPLETED) echo "" ;; PENDING|RUNNING|REQUEUED|"") echo "afterok:$1" ;; *) echo "BAD:$1:$st" ;; esac
}
declare -A PREV
OUT=$ROOT/inputs/walk_ids_$SET.txt
for seq in $(tail -n +2 "$LIST" | sed 's/\r$//' | cut -d, -f9 | sort -n | uniq); do
  while IFS=, read -r file set conc njobs est tl initsub lane sq o0 o1 gpu; do
    [ "$sq" = "$seq" ] || continue
    task=$((10#${initsub#w}))
    D=$(dep_task "${ARR}_$task")
    case "$D" in BAD:*) echo "bar-init task $D: not submitting $file"; exit 1 ;; esac
    [ -n "$DEP" ] && D="${D:+$D,}$DEP"
    [ -n "${PREV[$lane]:-}" ] && D="${D:+$D,}afterany:${PREV[$lane]}"
    rtag=${file%.txt}; rtag=${rtag//./_}
    id=$(sbatch --parsable -p "$PART" --time="$tl" --nice="$NICE" ${D:+--dependency=$D} --job-name=v2w ${HOLD:+-H} \
         --constraint="$FEAT" -o "$ROOT/logs/%x_%j.out" \
         --export=NONE,ROOT=$ROOT,JOBS=$file,SUB=refit,TAG=$rtag,CONC=$conc,INITSUB=$initsub,GPUTYPE=$GT,NOSTEPS=1,PARTIAL_S=900 "$G")
    PREV[$lane]=$id
    echo "$file $id $PART $tl $GT lane $lane seq $seq" | tee -a "$ROOT/inputs/submitted.txt" >> "$OUT"
  done < <(tail -n +2 "$LIST" | sed 's/\r$//')
done
echo "$SET: $(wc -l < "$OUT") walk files submitted (ids in $OUT)"
