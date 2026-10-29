#!/bin/bash
# 2026-10-28 addendum J (e): the CARC share's pieces back to Hoffman2, so that cluster/ablate/merge_lofix.sh merges all
# 171 lines of tasks/trees_lofix.txt in one place.  Laptop relay CARC -> laptop -> Hoffman2, md5-checked at each hop;
# then each relayed line's Hoffman2 claim (the reservation, cpu = the marker) gets done as pool3.sh's run_line writes it.
# Incremental and idempotent: only lines whose CARC claim is done are relayed; a line already relayed (Hoffman2 done,
# the same md5 in place) is skipped; anything else (a Hoffman2 piece of the same name with another md5, a Hoffman2
# claim that is not the reservation) stops the relay before any change.
# usage (laptop, Git Bash): the LF text (the checkout may hold CRLF, which bash and the remote heredocs refuse):
#   git -c core.autocrlf=false show origin/<branch>:cluster/ablate/relay_lofix_carc.sh > <tmp>/relay.sh
#   bash <tmp>/relay.sh [--apply]
#   default = dry run: CARC checks, fetch to the laptop, md5 check, Hoffman2 checks; nothing written on Hoffman2.
#   --apply: also upload to $BH/tmp/relay_lofix_carc/<stamp>/, md5 check there, mv -n into runs/trees_lofix/pieces/,
#   md5 check in place, date +%F_%T > runs/claims_trees_lofix/<n>/done.
# env: SSH (native OpenSSH), CARC / H2 (ssh aliases), BC / BH (staging dirs), FIRST / LAST (the reserved lines).
# Remote arguments go through printf %q (the login shells are bash on both clusters).
set -euo pipefail
SSH=${SSH:-/c/Windows/System32/OpenSSH/ssh.exe}
CARC=${CARC:-usc-discovery}
H2=${H2:-hoffman2}
BC=${BC:-/scratch1/jc_905/abl_lofix}
BH=${BH:-/u/scratch/j/jamesdc1/abl}
FIRST=${FIRST:-134}
LAST=${LAST:-171}
MARK="CARC-RESERVED addendum J (e)"
APPLY=0
[ "${1:-}" = --apply ] && APPLY=1
W=$(mktemp -d "${TMPDIR:-/tmp}/relay_lofix_carc.XXXXXX")
sx() { "$SSH" -o BatchMode=yes -o ConnectTimeout=90 -o ServerAliveInterval=30 "$@"; }
echo "relay_lofix_carc $(date +%F_%T): lines $FIRST-$LAST, apply $APPLY, laptop dir $W"

# 1. CARC: done lines of the share and their pieces (parquet + json, no checkpoint) -> relay/<stamp>/pieces.tar + md5
sx "$CARC" "bash -s -- $(printf '%q ' "$BC" "$FIRST" "$LAST")" > "$W/carc.out" << 'EOF'
set -euo pipefail
BC=$1 a=$2 b=$3
T=$BC/code/results/ablations_2014start_w3000/tasks/trees_lofix_carc.txt
CL=$BC/runs/claims_trees_lofix P=$BC/runs/trees_lofix/pieces
R=$BC/relay/$(date +%Y%m%d_%H%M%S)
mkdir -p $R
: > $R/lines.txt
for n in $(seq $a $b); do
  read -r est arm part h <<< "$(sed -n "${n}p" $T)"
  [ -n "${part:-}" ] || { echo "BAD $n: blank task line"; continue; }
  f=${arm}_${est}_${part/\//of}
  d=""
  for g in $CL/$n $CL/$n.r[1-9]; do [ -e "$g/done" ] && d=$g; done
  if [ -z "$d" ]; then echo "OPEN $n $f"; continue; fi
  if [ -f $P/$f.parquet ] && [ -f $P/$f.json ] && [ ! -e $P/$f.ckpt.json ]; then
    echo "$n $f" >> $R/lines.txt
  else
    echo "BAD $n $f: claim done, piece incomplete"
  fi
done
if [ -s $R/lines.txt ]; then
  (cd $P && md5sum $(awk '{print $2 ".parquet", $2 ".json"}' $R/lines.txt)) > $R/pieces.md5
  tar cf $R/pieces.tar -C $P $(awk '{print $2 ".parquet", $2 ".json"}' $R/lines.txt)
fi
echo "RELAY $R $(wc -l < $R/lines.txt) $( [ -s $R/lines.txt ] && md5sum < $R/pieces.tar | cut -c1-32 || echo none)"
EOF
grep -E "^(OPEN|BAD)" "$W/carc.out" || true
if grep -q "^BAD" "$W/carc.out"; then echo "CARC: incomplete pieces -- stop"; exit 3; fi
read -r _ R NL TMD5 <<< "$(grep "^RELAY" "$W/carc.out")"
echo "CARC: $NL done lines in $R (tar md5 $TMD5)"
[ "$NL" -gt 0 ] || { echo "nothing to relay"; exit 0; }

# 2. laptop: fetch, md5 of the tar, unpack, md5 of every file
sx "$CARC" "cat $R/pieces.tar" > "$W/pieces.tar" < /dev/null
sx "$CARC" "cat $R/pieces.md5" > "$W/pieces.md5" < /dev/null
sx "$CARC" "cat $R/lines.txt" > "$W/lines.txt" < /dev/null
[ "$(md5sum < "$W/pieces.tar" | cut -c1-32)" = "$TMD5" ] || { echo "laptop: tar md5 differs -- stop"; exit 3; }
mkdir -p "$W/x"
tar xf "$W/pieces.tar" -C "$W/x"
(cd "$W/x" && md5sum -c --quiet "$W/pieces.md5") || { echo "laptop: piece md5 differs -- stop"; exit 3; }
echo "laptop: $(wc -l < "$W/pieces.md5") files md5-equal to CARC's"

# 3. Hoffman2: checks (dry run), or upload + md5 + mv -n + md5 + done (apply)
STAMP=$(basename "$R")
if [ $APPLY -eq 1 ]; then
  sx "$H2" "mkdir -p $BH/tmp/relay_lofix_carc/$STAMP && cat > $BH/tmp/relay_lofix_carc/$STAMP/pieces.tar" < "$W/pieces.tar"
  sx "$H2" "cat > $BH/tmp/relay_lofix_carc/$STAMP/pieces.md5" < "$W/pieces.md5"
  sx "$H2" "cat > $BH/tmp/relay_lofix_carc/$STAMP/lines.txt" < "$W/lines.txt"
fi
sx "$H2" "bash -s -- $(printf '%q ' "$BH" "$STAMP" "$TMD5" "$APPLY" "$MARK" "$(tr '\n' ' ' < "$W/lines.txt")" \
  "$(awk '{printf "%s:%s ", $2, $1}' "$W/pieces.md5")")" << 'EOF'
set -euo pipefail
BH=$1 STAMP=$2 TMD5=$3 APPLY=$4 MARK=$5 LINES=$6 MD5S=$7
CL=$BH/runs/claims_trees_lofix P=$BH/runs/trees_lofix/pieces U=$BH/tmp/relay_lofix_carc/$STAMP
declare -A want
for kv in $MD5S; do want[${kv%%:*}]=${kv#*:}; done
set -- $LINES
todo=() skip=() bad=0
while [ $# -ge 2 ]; do
  n=$1 f=$2
  shift 2
  r=$(ls -d $CL/$n.r[1-9] 2> /dev/null | head -1 || true)
  cpu=$(cat $CL/$n/cpu 2> /dev/null || true)
  if [ ! -d $CL/$n ] || [ -n "$r" ] || [ "$cpu" != "$MARK" ]; then
    echo "BAD $n $f: Hoffman2 claim is not the reservation (cpu '$cpu'${r:+, $r})"; bad=1; continue
  fi
  same=1 any=0
  for e in parquet json; do
    if [ -e $P/$f.$e ]; then any=1; [ "$(md5sum < $P/$f.$e | cut -c1-32)" = "${want[$f.$e]}" ] || same=0; else same=0; fi
  done
  [ -e $P/$f.ckpt.json ] && { echo "BAD $n $f: a Hoffman2 checkpoint exists"; bad=1; continue; }
  if [ -e $CL/$n/done ]; then
    [ $same -eq 1 ] && skip+=($n) || { echo "BAD $n $f: done on Hoffman2 but the piece differs"; bad=1; }
  elif [ $any -eq 1 ]; then
    [ $same -eq 1 ] && todo+=("$n:$f") || { echo "BAD $n $f: a Hoffman2 piece of this name differs"; bad=1; }
  else
    todo+=("$n:$f")
  fi
done
echo "Hoffman2: to relay ${#todo[@]}, already relayed ${#skip[@]}"
[ $bad -eq 0 ] || { echo "Hoffman2 checks failed -- nothing changed"; exit 3; }
[ "$APPLY" = 1 ] || { echo "dry run: no change on Hoffman2"; exit 0; }
[ "$(md5sum < $U/pieces.tar | cut -c1-32)" = "$TMD5" ] || { echo "Hoffman2: tar md5 differs -- nothing changed"; exit 3; }
mkdir -p $U/x
tar xf $U/pieces.tar -C $U/x
(cd $U/x && md5sum -c --quiet ../pieces.md5) || { echo "Hoffman2: unpacked md5 differs -- nothing changed"; exit 3; }
for t in "${todo[@]}"; do
  n=${t%%:*} f=${t#*:}
  for e in parquet json; do
    [ -e $P/$f.$e ] || mv -n $U/x/$f.$e $P/$f.$e
    [ "$(md5sum < $P/$f.$e | cut -c1-32)" = "${want[$f.$e]}" ] || { echo "Hoffman2: $f.$e md5 after the move differs -- stop"; exit 4; }
  done
  [ ! -e $CL/$n/done ] && [ "$(cat $CL/$n/cpu)" = "$MARK" ] && date +%F_%T > $CL/$n/done
  echo "relayed $n $f"
done
echo "Hoffman2: $(ls $CL/*/done 2> /dev/null | wc -l) of 171 claims done"
EOF
echo "relay_lofix_carc done $(date +%F_%T)"
