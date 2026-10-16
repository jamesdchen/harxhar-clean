#!/bin/bash
# Pooled tree walks, row pieces (2026-10-08): collect the UNFINISHED pieces of array tasks that ended without finishing
# (TIMEOUT / FAILED / NODE_FAIL / OUT_OF_MEMORY / PREEMPTED) into a new task file, PER pieces a line, for
# `submit_pooltrees.sh pieces` (a piece with an npz is never listed; a task still pending or running is never touched,
# and a task file line is array index = line number, so ARRAY offsets are honoured: OFFSET = first index - 1 = 0 here).
#   bash cluster/slurm/pooltree_sweep.sh <array job id> <task file> <out task file> [PER (default 10)]
set -euo pipefail
# CANCELLED tasks are NOT swept: a cancel here means the task's pieces were moved to another array (revoke-first);
# sweeping them re-listed 300 moved pieces on 2026-10-13 (caught and revoked before any ran)
cd "$(dirname "$0")/../.."
J=${1:?array job id}; TF=${2:?task file}; OUTF=${3:?out task file}; PER=${4:-10}
R=${RESULTS_ROOT:-results/pooled_trees_2026-10-07}
[ -f "$OUTF" ] && { echo "$OUTF exists; refusing"; exit 1; }
mapfile -t BAD < <(sacct -X -n -P -j "$J" -o JobID,State | awk -F'|' '$2 ~ /TIMEOUT|FAILED|NODE_FAIL|OUT_OF_ME|PREEMPTED/ {split($1, a, "_"); print a[2]}' | sort -un)
# squeue fails on an array with no live task left (pipefail + set -e ended the sweep silently, 2026-10-13)
LIVE=$( (squeue -h -r -j "$J" -o "%i" 2>/dev/null || true) | sed 's/.*_//' | sort -un | tr '\n' ' ')
LEFT=()
for i in "${BAD[@]}"; do
  case " $LIVE " in *" $i "*) continue ;; esac   # requeued / still alive
  read -r EST SET PIECES <<< "$(sed -n "${i}p" "$TF")"
  for P in ${PIECES//,/ }; do
    IFS=: read -r K A B <<< "$P"
    [ -f "$(printf '%s/chunks/pooltree_%s_%s/pieces/chunk_%03d_%04d-%04d.npz' "$R" "$EST" "$SET" "$K" "$A" "$B")" ] || LEFT+=("$EST $SET $P")
  done
done
echo "array $J: ${#BAD[@]} tasks ended unfinished; ${#LEFT[@]} pieces without an npz"
[ "${#LEFT[@]}" -gt 0 ] || exit 0
printf '%s\n' "${LEFT[@]}" | awk -v per="$PER" '{k = $1 " " $2; p[k] = (p[k] == "" ? $3 : p[k] "," $3); n[k]++; if (n[k] == per) {print k, p[k]; p[k] = ""; n[k] = 0}} END {for (k in p) if (p[k] != "") print k, p[k]}' > "$OUTF"
echo "wrote $OUTF: $(wc -l < "$OUTF") lines"
