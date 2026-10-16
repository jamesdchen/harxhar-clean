#!/bin/bash
# Pooled forest all_features (2026-10-13): move the LAST N unstarted lines of a 6-row wave array to 4-CPU tasks of one
# 3-row piece each (they fill backfill holes the 8-CPU tasks cannot; any cut tiles for the merge).  Revoke-first: the
# source array's pending tasks for those lines are HELD, their pieces appended to the revoke list that array reads
# (results/.../revoked_<source task file>.txt), then pieces with an npz or a .claim are dropped (they started).
#   SRC=13074358 SRCF=cluster/pooltree_pieces_rf_all_features_w7.txt N=150 NAME=w9 bash cluster/slurm/pooltree_af_split.sh
#   (on CARC, no conda env; NICE default 100, TIME 00:40:00, MEM 7G -- the panel load peaks at 5.3-5.6 GB)
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=${SRC:?source array}; SRCF=${SRCF:?source task file}; N=${N:?lines}; NAME=${NAME:?new name, e.g. w9}
R=results/pooled_trees_2026-10-07; P=$R/chunks/pooltree_rf_all_features/pieces
OUT=cluster/pooltree_pieces_rf_all_features_$NAME.txt
[ -f $OUT ] && { echo "$OUT exists; refusing"; exit 1; }
W=logs/split_$NAME; mkdir -p $W
(squeue -j $SRC -h -r -t PD -o "%i %r" || true) | grep -v JobHeldUser | awk '{print $1}' | sed 's/.*_//' | sort -n | tail -n $N > $W/idx
[ -s $W/idx ] || { echo "no unstarted, unheld tasks left in $SRC"; exit 0; }
for i in $(cat $W/idx); do scontrol hold ${SRC}_$i 2>/dev/null || true; done
while read -r i; do sed -n "${i}p" $SRCF | awk '{print $3}' | tr , '\n'; done < $W/idx > $W/p6
sed 's/^/rf all_features /' $W/p6 >> $R/revoked_$(basename $SRCF .txt).txt
sleep 20
: > $W/ok
while IFS=: read -r k a b; do f=$(printf '%s/chunk_%03d_%04d-%04d' $P $k $a $b); [ -f $f.npz ] || [ -f $f.claim ] || echo "$k:$a:$b" >> $W/ok; done < $W/p6
echo "held $(wc -l < $W/idx) tasks of $SRC; moved $(wc -l < $W/ok) of $(wc -l < $W/p6) pieces"
awk -F: '{m = $2 + int(($3 - $2) / 2); print "rf all_features " $1 ":" $2 ":" m; print "rf all_features " $1 ":" m ":" $3}' $W/ok > $OUT
md5sum $OUT
CLASSES="epyc-7513|epyc-7542" CPUS=4 MEM=${MEM:-7G} TIME=${TIME:-00:40:00} NICE=${NICE:-100} CANARY_FLAG=$R/canary_ok_rf_all_features TAG=$NAME \
  bash cluster/slurm/submit_pooltrees.sh pieces rf all_features $OUT > /dev/null 2>&1 || true
tail -1 logs/submitted_pooltrees.txt
echo "held source tasks (superseded; for the user to cancel): ${SRC}_[$(paste -sd, $W/idx)]"
