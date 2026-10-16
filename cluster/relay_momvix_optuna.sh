#!/bin/bash
# Relays of the moments_vix Optuna run between Hoffman2, this machine and CARC (no direct
# cluster-to-cluster ssh; tar over native OpenSSH, md5 of every relayed file checked at the far end).
#   bash cluster/relay_momvix_optuna.sh h2s1 [models]   Hoffman2 stage-1 chunks (DONE only; without
#        the per-point points/ files) -> CARC $R/stage1/... (refuses a chunk CARC already has)
#   bash cluster/relay_momvix_optuna.sh audit          Hoffman2 audit chunk (moments_vix rows 0-19) and
#        CARC's canary chunk 0 -> local $LR/audit/{h2,carc}/ (compared by experiments/momvix_optuna_audit.py)
#   bash cluster/relay_momvix_optuna.sh s1toh2 <models> CARC merged stage-1 records (+ STAGE1_COMPLETE)
#        -> Hoffman2 $R/stage1/...
#   bash cluster/relay_momvix_optuna.sh h2s2 <models>  Hoffman2 stage-2 chunks (DONE) -> CARC $R/stage2/...
#   bash cluster/relay_momvix_optuna.sh collect        CARC results (merged stage 1 + paths, gates, logs,
#        host files, run logs; no points/) -> local $LR
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
PREFIX=${PREFIX:-momvix}   # momvolvix: the moments_vol_vix run (CARC only; collect)
CARC=${CARC_ROOT:-/scratch1/jc_905/harxhar-$PREFIX}
H2=/u/scratch/j/jamesdc1/harxhar-momvix
R=results/linear_subsection_trees_optuna_mask_$PREFIX
LR=$R
ST=${STAGE_DIR:-${TMPDIR:-/tmp}/relay_momvix}
mkdir -p "$ST" "$LR"
MODE=${1:?usage: relay_momvix_optuna.sh h2s1|audit|s1toh2|h2s2|collect [models]}
MODELS=${2:-lgbm xgb rf}
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
arm() { echo "moments_vix/bar1600/$1/tw2000"; }

relay_dirs() {  # $1 from-host, $2 from-root, $3 to-host, $4 to-root, rest: dirs relative to the roots
  local fh=$1 fr=$2 th=$3 tr=$4; shift 4
  local list="$ST/list.txt" a b
  "$SSH" -o BatchMode=yes "$fh" "cd $fr && find $* -type f -not -path '*/points/*' | sort" > "$list"
  [ -s "$list" ] || { echo "nothing to relay"; return 1; }
  "$SSH" -o BatchMode=yes "$fh" "cd $fr && tar czf - -T -" < "$list" > "$ST/x.tgz"
  "$SSH" -o BatchMode=yes "$th" "mkdir -p $tr && cd $tr && tar xzf -" < "$ST/x.tgz"
  a=$("$SSH" -o BatchMode=yes "$fh" "cd $fr && xargs -d '\n' md5sum" < "$list" | sums)
  b=$("$SSH" -o BatchMode=yes "$th" "cd $tr && xargs -d '\n' md5sum" < "$list" | sums)
  [ "$a" = "$b" ] || { echo "RELAY FAILED: md5 differ"; diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") | head; return 1; }
  echo "relayed $(wc -l < "$list") files $fh:$fr -> $th:$tr, md5 equal"
}

pick_dirs() {  # DIRS = the candidate chunk dirs DONE on Hoffman2 and absent on CARC (one ssh each)
  local done_h2 on_carc d
  done_h2=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && for d in $*; do [ -f \$d/DONE ] && echo \$d; done; true")
  on_carc=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && for d in $*; do [ -e \$d ] && echo \$d; done; true")
  DIRS=()
  for d in "$@"; do
    if ! grep -qxF "$d" <<< "$done_h2"; then echo "not DONE on Hoffman2: $d"
    elif grep -qxF "$d" <<< "$on_carc"; then echo "CARC has $d already; not relayed"
    else DIRS+=("$d"); fi
  done
}

case "$MODE" in
  h2s1)
    CAND=()
    while read -r B M TW SEG C S E H; do
      [[ " $MODELS " == *" $M "* ]] && CAND+=("$R/stage1/$(arm "$M")/chunks/c$C")
    done < cluster/momvix_optuna_tasks_s1_h2.txt
    pick_dirs "${CAND[@]}"
    [ "${#DIRS[@]}" -gt 0 ] && relay_dirs hoffman2 "$H2" usc-discovery "$CARC" "${DIRS[@]}"
    ;;
  audit)
    mkdir -p "$LR/audit/h2" "$LR/audit/carc"
    HA=(); CA=()
    for M in $MODELS; do HA+=("$R/h2audit/stage1/$(arm "$M")/chunks/c950"); CA+=("$R/stage1/$(arm "$M")/chunks/c0"); done
    "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar czf - --exclude=points ${HA[*]}" | tar xzf - -C "$LR/audit/h2"
    "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && tar czf - --exclude=points ${CA[*]}" | tar xzf - -C "$LR/audit/carc"
    find "$LR/audit" -name 'trials_bar1600.npz' | sort
    ;;
  s1toh2)
    DIRS=()
    for M in $MODELS; do DIRS+=("$R/stage1/$(arm "$M")/trials_bar1600.npz" "$R/stage1/$(arm "$M")/STAGE1_COMPLETE"); done
    relay_dirs usc-discovery "$CARC" hoffman2 "$H2" "${DIRS[@]}"
    ;;
  h2s2)
    CAND=()
    for M in $MODELS; do CAND+=("$R/stage2/$(arm "$M")/chunks/c0"); done
    pick_dirs "${CAND[@]}"
    [ "${#DIRS[@]}" -gt 0 ] && relay_dirs hoffman2 "$H2" usc-discovery "$CARC" "${DIRS[@]}"
    ;;
  collect)
    "$SSH" -o BatchMode=yes usc-discovery "cd $CARC/$R && tar czf - --exclude='*/points/*' --exclude='*/points' --exclude=canary_repeat \
       \$(ls -d paths stage1 stage2 gate canary_stage2 *.csv *.log *.txt STAGE* canary_ok_* GATE_OK_* 2>/dev/null)" > "$ST/collect.tgz"
    tar xzf "$ST/collect.tgz" -C "$LR"
    echo "collected into $LR: $(find "$LR" -type f | wc -l) files"
    ;;
  *) echo "MODE must be h2s1, audit, s1toh2, h2s2 or collect"; exit 1 ;;
esac
