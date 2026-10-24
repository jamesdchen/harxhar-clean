#!/bin/bash
# Hoffman2 side of a delivery to CARC (run over ssh; stdout = one tar stream, everything else to stderr), relayed through
# the laptop into CARC's h2/carc_recv.sh (inside the stream).  usage: bash deliver.sh gates | pieces
#   gates   gate/h2/<tag>/{g1,g2}/pieces (+ MD5SUMS, written by summary.sh) and h2/ (gate summary, accept, freezes,
#           tags, code md5s) -> CARC gate/h2/<tag>/... and h2/
#   pieces  the finished walk pieces (json present, no checkpoint) not yet sent -> CARC runs/pieces_h2/ (MD5SUMS
#           appended there by carc_recv.sh; a delivered file is never overwritten); recorded in h2/delivered.txt
exec 3>&1 1>&2
set -euo pipefail
B=/u/scratch/j/jamesdc1/t3t
cd $B
K=$(date +%Y%m%d_%H%M%S)
mkdir -p h2/out/$K
cp jobs/carc_recv.sh h2/out/$K/
case ${1:?gates or pieces} in
  gates)
    for d in gate/h2/*/g1/pieces gate/h2/*/g2/pieces; do find $d -maxdepth 1 -type f \( -name '*.parquet' -o -name '*.json' -o -name MD5SUMS \) ! -name '*.ckpt.json'; done > h2/out/$K/FILES
    find h2 -maxdepth 1 -type f \( -name '*.txt' -o -name '*.json' \) >> h2/out/$K/FILES
    find h2/accept -type f >> h2/out/$K/FILES 2>/dev/null || true
    ;;
  pieces)
    touch h2/delivered.txt
    for j in runs/pieces/*.json; do
      case $j in *.ckpt.json) continue ;; esac
      p=${j%.json}.parquet
      [ -f $p ] && [ ! -f ${j%.json}.ckpt.json ] || continue
      grep -qxF "$(basename $p)" h2/delivered.txt && continue
      echo $p
      echo $j
    done > h2/out/$K/FILES
    ;;
  *) echo "usage: deliver.sh gates | pieces"; exit 2 ;;
esac
echo "$1" > h2/out/$K/MODE
(xargs -r md5sum < h2/out/$K/FILES) > h2/out/$K/MD5SUMS_ALL
echo "delivery $K ($1): $(grep -c . h2/out/$K/FILES) files, $(xargs -r du -cb < h2/out/$K/FILES | tail -1 | cut -f1) bytes; manifest md5 $(md5sum < h2/out/$K/MD5SUMS_ALL | cut -c1-32)"
if [ "$1" = pieces ]; then grep '[.]parquet$' h2/out/$K/FILES | xargs -r -n1 basename >> h2/delivered.txt; fi
tar cf - -T h2/out/$K/FILES h2/out/$K >&3
echo "H2 tar done $(date +%T)"
