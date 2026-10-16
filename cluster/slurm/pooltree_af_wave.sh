#!/bin/bash
# Pooled forest, all_features (2026-10-09): submit the next WAVE of backfill-sized tasks -- unstarted 12-row pieces of the
# CARC plan (cluster/pooltree_pieces_rf_all_features_{carc,carc2,carc3}.txt; never the largemem lines 801-840 of carc,
# never Hoffman2's chunks 98-99), each cut into two 6-row pieces, ONE a task (8 CPUs, 7G, 0:45; measured 242 s/row
# median at 8 CPUs -> 24-30 min of work + ~3 min load).  A 12-row piece is a candidate when it has no npz, is not covered
# by npz halves, and is not in an earlier wave file (cluster/pooltree_pieces_rf_all_features_w*.txt).  Run ONLY after
# the superseded arrays that listed these pieces are gone (cancelled), so no task of theirs can still run a piece.
# Budget: the wave is cut so that this user's ptrp_* task count stays <= MAXTASKS (default 3400).
#   WAVE=2 [NICE=400] [MAXTASKS=3400] [LIMIT=<pieces>] [SPLIT=2|1] [CPUS=8] [MEM=7G] [TIME=00:45:00] [DRY=1]
#        [GONE='<array ids that must be gone>'] bash cluster/slurm/pooltree_af_wave.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
WAVE=${WAVE:?WAVE number}; MAXTASKS=${MAXTASKS:-3400}
R=results/pooled_trees_2026-10-07; C=$R/chunks/pooltree_rf_all_features/pieces
OUT=cluster/pooltree_pieces_rf_all_features_w$WAVE.txt
[ -f $OUT ] && { echo "$OUT exists; refusing"; exit 1; }
for j in ${GONE:-12811055 12811893 12816345}; do
  n=$( (squeue -j $j -h -r 2>/dev/null || true) | wc -l); [ "$n" = 0 ] || { echo "array $j still has $n tasks; refusing"; exit 1; }
done
MINE=$(squeue -u "$USER" -h -r -o %j | grep -c '^ptrp' || true)
SPLIT=${SPLIT:-2}   # 2: two 6-row halves a 12-row piece (8 CPUs); 1: the whole piece a task (e.g. CPUS=20 TIME=00:35:00)
ROOM=$(( (MAXTASKS - MINE) / SPLIT ))
[ -n "${LIMIT:-}" ] && [ "$LIMIT" -lt "$ROOM" ] && ROOM=$LIMIT
echo "ptrp tasks now: $MINE; room for $ROOM 12-row pieces (budget $MAXTASKS)"
[ "$ROOM" -gt 0 ] || { echo "no room"; exit 0; }
W=logs/af_wave_$WAVE; mkdir -p $W
{ sed -n '1,800p' cluster/pooltree_pieces_rf_all_features_carc.txt; cat cluster/pooltree_pieces_rf_all_features_carc2.txt cluster/pooltree_pieces_rf_all_features_carc3.txt; } \
  | awk '{print $3}' | tr , '\n' | sort -u > $W/plan.all
# chunks handed to Hoffman2 (cluster/pooltree_af_exclude_chunks.txt, one chunk number a line) are never CARC candidates
if [ -f cluster/pooltree_af_exclude_chunks.txt ]; then
  awk -F: 'NR == FNR {x[$1]; next} !($1 in x)' cluster/pooltree_af_exclude_chunks.txt $W/plan.all > $W/plan
else
  cp $W/plan.all $W/plan
fi
# parents of earlier halves: k:a:b of a half -> the plan piece that contains it
# one piece a line: wave files may list several pieces a line (wscore*, wrest*); unsplit, those were MISSED (2026-10-13)
cat cluster/pooltree_pieces_rf_all_features_w*.txt 2>/dev/null | awk '{print $3}' | tr , '
' | grep . > $W/halves || : > $W/halves
awk -F: 'NR == FNR {h[$1] = h[$1] " " $2 "-" $3; next} {split(h[$1], x, " "); for (i in x) {split(x[i], y, "-"); if (y[1] >= $2 && y[2] <= $3) {print $0; break}}}' \
  $W/halves $W/plan | sort -u > $W/in_waves
: > $W/cand
while read -r p; do
  IFS=: read -r k a b <<< "$p"; m=$(( a + (b - a) / 2 ))
  f=$(printf '%s/chunk_%03d_%04d-%04d.npz' $C $k $a $b)
  h1=$(printf '%s/chunk_%03d_%04d-%04d.npz' $C $k $a $m); h2=$(printf '%s/chunk_%03d_%04d-%04d.npz' $C $k $m $b)
  [ -f $f ] && continue
  [ -f $h1 ] && [ -f $h2 ] && continue
  echo "$p"
done < <(comm -23 $W/plan $W/in_waves) | sort -t: -k1,1n -k2,2n > $W/cand
echo "candidates: $(wc -l < $W/cand) unstarted 12-row pieces"
if [ "$SPLIT" = 1 ]; then
  head -n $ROOM $W/cand | sed 's/^/rf all_features /' > $OUT
else
  head -n $ROOM $W/cand | awk -F: '{m = $2 + int(($3 - $2) / 2); print "rf all_features " $1 ":" $2 ":" m; print "rf all_features " $1 ":" m ":" $3}' > $OUT
fi
echo "$OUT: $(wc -l < $OUT) tasks"; md5sum $OUT
[ "${DRY:-0}" = 1 ] && { echo "DRY: not submitted; removing $OUT"; rm -f $OUT; exit 0; }
# NOSUBMIT=1 keeps the file unsubmitted, for a trickle within a small QOS (oneweek: 50 jobs) via pooltree_af_poll.sh OWF=$OUT
[ "${NOSUBMIT:-0}" = 1 ] && { echo "NOSUBMIT: $OUT written, not submitted"; exit 0; }
# belt and braces while superseded arrays are only HELD (not cancelled): revoke the wave's 12-row parents for them too
# (12811055 / 12811893 read revoked_pieces.txt; 12816345 reads revoked_pooltree_pieces_rf_all_features_carc3.txt)
head -n $ROOM $W/cand | sed 's/^/rf all_features /' | tee -a $R/revoked_pieces.txt >> $R/revoked_pooltree_pieces_rf_all_features_carc3.txt
CLASSES="${CLASSES:-epyc-7513|epyc-7542}" PARTITION="${PARTITION:-main}" CPUS=${CPUS:-8} MEM=${MEM:-7G} TIME=${TIME:-00:45:00} NICE=${NICE:-400} CANARY_FLAG=$R/canary_ok_rf_all_features TAG=w$WAVE \
  bash cluster/slurm/submit_pooltrees.sh pieces rf all_features $OUT > /dev/null 2>&1 || true
tail -1 logs/submitted_pooltrees.txt
