#!/bin/bash
# CARC side of a Hoffman2 delivery (h2/code/deliver.sh), run from the staging directory the stream was untarred into:
#   bash <stage>/h2/out/<K>/carc_recv.sh <stage> <K>
# Checks every file against the sender's MD5SUMS_ALL, then moves it into the run (never overwriting a different file):
#   gates   gate/h2/<tag>/{g1,g2}/pieces/* -> $B/gate/h2/...; h2/* (summary, accept, freezes) -> $B/h2/delivery_<K>/
#   pieces  runs/pieces/* -> $B/runs/pieces_h2/: the md5 lines appended to $B/runs/pieces_h2/MD5SUMS first, then the
#           json, then the parquet (a rename on the same file system), so an intake never sees an unlisted piece
set -euo pipefail
B=/scratch1/jc_905/trees3way
S=$1
K=$2
cd $S
M=h2/out/$K/MD5SUMS_ALL
MODE=$(cat h2/out/$K/MODE)
md5sum -c --quiet $M && echo "staging $S: $(grep -c . $M) files md5-verified ($MODE)"
put() {  # <relative path in staging> <destination path>
  local src=$1 dst=$2
  if [ -e "$dst" ]; then
    if [ "$(md5sum < "$src")" = "$(md5sum < "$dst")" ]; then echo "same: $dst"; else echo "CONFLICT (kept the existing): $dst"; fi
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  mv "$src" "$dst"
}
case $MODE in
  gates)
    while read -r h f; do
      case $f in
        gate/h2/*) put "$f" "$B/$f" ;;
        h2/*) put "$f" "$B/h2/delivery_$K/${f#h2/}" ;;
      esac
    done < $M
    for d in $B/gate/h2/*/g1/pieces $B/gate/h2/*/g2/pieces; do (cd $d && md5sum -c --quiet MD5SUMS && echo "$d: MD5SUMS ok"); done
    ;;
  pieces)
    I=$B/runs/pieces_h2
    mkdir -p $I
    touch $I/MD5SUMS
    n=0
    for p in $(awk '{print $2}' $M | grep '[.]parquet$'); do
      j=${p%.parquet}.json
      bp=$(basename $p)
      bj=$(basename $j)
      if [ -e $I/$bp ] || [ -e $I/$bj ]; then
        put "$p" "$I/$bp"
        put "$j" "$I/$bj"
        continue
      fi
      grep -E "  (runs/pieces/)?($bp|$bj)\$" $M | sed 's|  runs/pieces/|  |' >> $I/MD5SUMS
      mv "$j" "$I/$bj"
      mv "$p" "$I/$bp"
      n=$((n + 1))
    done
    (cd $I && md5sum -c --quiet MD5SUMS) && echo "$I: MD5SUMS ok ($(grep -c '[.]parquet$' $I/MD5SUMS) pieces listed, $n new)"
    ;;
esac
echo "CARC receive done $(date +%T)"
