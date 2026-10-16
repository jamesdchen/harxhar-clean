#!/bin/bash
# Pooled forest all_features (2026-10-14): one CARC poll -- status line, top-up of the held small tasks
# (pooltree_af_topup.sh), and two trickles of unsubmitted wave files into small partitions, each up to its QOS:
#   oneweek  (QOS: 50 jobs submitted a user)  OWF, xeon-2640v4 (byte-identity verified 2026-10-14), 8 CPUs, 6G, 1:00
#   largemem (QOS: 3 jobs, 64 CPUs a user)     LMF, epyc-7513 only (verified; its epyc-9354 node is not), 20 CPUs, 6G, 0:40
# Both count toward the 100-running-job cap, so they add placement (nodes), not slots.  The next line of each file is
# kept in logs/ow_next_<name>.     bash cluster/slurm/pooltree_af_poll.sh   (on CARC, no conda)
set -euo pipefail
cd "$(dirname "$0")/../.."
R=results/pooled_trees_2026-10-07
# oneweek files in order: the first with lines left is trickled (w14 done 2026-10-14 16:33)
OWFS=${OWFS:-cluster/pooltree_pieces_rf_all_features_w17.txt cluster/pooltree_pieces_rf_all_features_w25.txt}
OWF=""
for f in $OWFS; do n=$(basename "$f" .txt); n=${n##*_}; [ "$(cat "logs/ow_next_$n" 2>/dev/null || echo 1)" -le "$(wc -l < "$f")" ] && { OWF=$f; break; }; done
[ -n "$OWF" ] || OWF=${f}
LMF=${LMF:-cluster/pooltree_pieces_rf_all_features_w18.txt}
by() { (squeue -u "$USER" -h -t R -p "$1" -o %C || true) | sort -n | uniq -c | awk '{printf "%dx%d ", $1, $2}'; }
echo "$(date +%T) running $( (squeue -u "$USER" -h -t R -o %i || true) | wc -l) (main $(by main); oneweek $(by oneweek); largemem $(by largemem)) cpus $( (squeue -u "$USER" -h -t R -o %C || true) | paste -sd+ - | bc)"
bash cluster/slurm/pooltree_af_topup.sh | tail -n +2
# trickle FILE PARTITION CLASSES CPUS MEM TIME MAXJOBS MINBATCH
trickle() {
  local F=$1 PART=$2 CL=$3 C=$4 MM=$5 T=$6 MX=$7 MINB=$8 N NEXTF NEXT LAST Q ROOM END L
  [ -f "$F" ] || { echo "$PART: $F missing"; return 0; }
  N=$(basename "$F" .txt); N=${N##*_}
  NEXTF=logs/ow_next_$N
  NEXT=$(cat "$NEXTF" 2>/dev/null || echo 1)
  LAST=$(wc -l < "$F")
  Q=$( (squeue -u "$USER" -h -r -p "$PART" -o %i || true) | wc -l)
  ROOM=$((MX - Q))
  if [ "$NEXT" -le "$LAST" ] && [ "$ROOM" -ge "$MINB" ]; then
    END=$((NEXT + ROOM - 1)); [ "$END" -le "$LAST" ] || END=$LAST
    PARTITION=$PART CLASSES=$CL CPUS=$C MEM=$MM TIME=$T NICE=95 ARRAY=$NEXT-$END \
      CANARY_FLAG=$R/canary_ok_rf_all_features TAG=${N}n$NEXT \
      bash cluster/slurm/submit_pooltrees.sh pieces rf all_features "$F" > /dev/null 2>&1 || true
    L=$(tail -1 logs/submitted_pooltrees.txt)
    case "$L" in *"_${N}n$NEXT="*) echo $((END + 1)) > "$NEXTF"; echo "$PART: $L" ;; *) echo "$PART submit of $N $NEXT-$END FAILED" ;; esac
  else
    echo "$PART: $Q of $MX submitted; $N next line $NEXT of $LAST"
  fi
}
trickle "$OWF" oneweek xeon-2640v4 8 6G 01:00:00 50 5
trickle "$LMF" largemem epyc-7513 20 6G 00:40:00 3 1
