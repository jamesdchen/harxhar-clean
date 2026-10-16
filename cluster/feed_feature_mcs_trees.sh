#!/bin/bash
# Feature MCS trees (2026-10-09): ONE feeding cycle of the backfill-sized packs under the per-user submit cap (5,000 jobs,
# array tasks counted).  Reads the CARC queue size and the forest gate flag, then feeds the next array slices of the
# plan's fleet task files that fit (keeping SPARE slots free) through cluster/slurm/submit_feature_mcs_trees.sh feed, in
# the order LightGBM, XGBoost, forest (the forest only once GATE_OK_rf exists, or held afterok on GATE).
# The next unfed pack of each array is kept in a local state file, advanced only when every slice of the cycle was
# submitted; when every pack is fed and no fleet task of the plan is pending or running, it submits the collect step once.
# SUFFIX (default _v2) picks the plan: _v2 = the 96 MCS designs (collect -> TREES_DONE); _v3 = the 16 bucket designs of
# 2026-10-12 (COLLECT_MODE=collect_buckets -> BUCKETS_DONE).  ROOT = the CARC deployment root (2026-10-13: the fresh root
# /scratch1/jc_905/harxhar-mcstrees-b after the storage failure), JOBNAME_PREFIX = the jobs' name prefix (default
# mct<N> for _v<N>), STATE_TAG = a tag in the state file's name: logs/feed_feature_mcs_trees[<SUFFIX><STATE_TAG>].state
# (the _v2 run's file has no suffix).  ESTS (default "lgbm xgb rf") = the estimators to feed.
# Run from Git Bash locally, repeatedly:
#   [SUFFIX=_v3 COLLECT_MODE=collect_buckets ROOT=... JOBNAME_PREFIX=... STATE_TAG=...] [GATE=<forest gate id>] \
#     bash cluster/feed_feature_mcs_trees.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
ROOT=${ROOT:-/scratch1/jc_905/harxhar-mcstrees}
CAP=${CAP:-5000}
SPARE=${SPARE:-8}
MAXFEED=${MAXFEED:-200}  # at most this many tasks per cycle
SUFFIX=${SUFFIX:-_v2}
ESTS=${ESTS:-lgbm xgb rf}
JP=${JOBNAME_PREFIX:-mct${SUFFIX#_v}}
if [ "$SUFFIX" = _v2 ]; then ST=logs/feed_feature_mcs_trees.state; else ST="logs/feed_feature_mcs_trees$SUFFIX${STATE_TAG:-}.state"; fi
mkdir -p logs
[ -f "$ST" ] || printf 'lgbm 1\nxgb 1\nrf 1\ncollect 0\n' > "$ST"
total() { awk 'NF {print $1}' "cluster/feature_mcs_trees_fleet_$1$SUFFIX.txt" | sort -n | tail -1; }
INFO=$(timeout 300 "$SSH" -o BatchMode=yes -o ConnectTimeout=60 usc-discovery "squeue -u \$USER -h -r | wc -l; ls $ROOT/results/feature_mcs_trees_work/GATE_OK_rf 2>/dev/null | wc -l; squeue -u \$USER -h -r -o %j | grep -c '^${JP}_.*_fleet' || true" 2>/dev/null | grep -E '^[0-9]+$' || true)
N=$(echo "$INFO" | sed -n 1p); GOK=$(echo "$INFO" | sed -n 2p); LEFT=$(echo "$INFO" | sed -n 3p)
[ -n "$N" ] && [ -n "$GOK" ] && [ -n "$LEFT" ] || { echo "$(date +%T) no queue reading (ssh failed); try later"; exit 1; }
HEAD=$((CAP - SPARE - N)); [ "$HEAD" -le "$MAXFEED" ] || HEAD=$MAXFEED
echo "$(date +%T) submitted $N, headroom $HEAD, forest gate passed: $GOK, $JP fleet tasks queued: $LEFT"
CMDS=""
NWANT=0
cp "$ST" "$ST.new"  # the advanced state is applied only after the submission went through
for E in $ESTS; do
  NEXT=$(awk -v e="$E" '$1 == e {print $2}' "$ST.new"); LAST=$(total "$E")
  [ "$NEXT" -le "$LAST" ] || continue
  [ "$HEAD" -gt 0 ] || break
  if [ "$E" = rf ] && [ "$GOK" = 0 ] && [ -z "${GATE:-}" ]; then echo "  forest: gate not passed and no GATE id; not fed"; continue; fi
  END=$((NEXT + HEAD - 1)); [ "$END" -le "$LAST" ] || END=$LAST
  G=""; [ "$E" != rf ] || [ "$GOK" = 1 ] || G="GATE=$GATE "
  CMDS="$CMDS ${G}SUFFIX=$SUFFIX JOBNAME_PREFIX=$JP bash cluster/slurm/submit_feature_mcs_trees.sh feed $E fleet $NEXT-$END | head -1;"
  NWANT=$((NWANT + 1))
  HEAD=$((HEAD - (END - NEXT + 1)))
  awk -v e="$E" -v n=$((END + 1)) '$1 == e {$2 = n} {print}' "$ST.new" > "$ST.tmp" && mv "$ST.tmp" "$ST.new"
done
if [ -n "$CMDS" ]; then
  OUTF=$(timeout 600 "$SSH" -o BatchMode=yes -o ConnectTimeout=60 usc-discovery \
           "cd $ROOT && case \"\${CONDA_PREFIX:-}\" in ?*) echo 'conda env active'; exit 1 ;; esac; $CMDS" 2>&1 \
         | grep -v "reloaded with a version\|python/3\|^$" || true)
  printf '%s\n' "$OUTF"
  NFED=$(printf '%s\n' "$OUTF" | grep -c ' feed .* = [0-9]' || true)
  if [ "$NFED" = "$NWANT" ]; then mv "$ST.new" "$ST"
  elif [ "$NFED" = 0 ]; then rm -f "$ST.new"; echo "FEED NOT SUBMITTED (state unchanged); try later"; exit 1
  else echo "FEED PARTIAL ($NFED of $NWANT slices): reconcile $ST.new with logs/submitted_feature_mcs_trees.txt on CARC by hand"; exit 1; fi
else
  rm -f "$ST.new"
fi
ALLFED=1
for E in $ESTS; do [ "$(awk -v e="$E" '$1 == e {print $2}' "$ST")" -gt "$(total "$E")" ] || ALLFED=0; done
if [ "$ALLFED" = 1 ] && [ -z "$CMDS" ] && [ "$LEFT" = 0 ] && [ "$(awk '$1 == "collect" {print $2}' "$ST")" = 0 ]; then
  OUTC=$(timeout 300 "$SSH" -o BatchMode=yes -o ConnectTimeout=60 usc-discovery \
           "cd $ROOT && COLLECT_MODE=${COLLECT_MODE:-collect} bash cluster/slurm/submit_feature_mcs_trees.sh collect | head -1" 2>&1 \
         | grep -v "reloaded with a version\|python/3\|^$" || true)
  printf '%s\n' "$OUTC"
  if printf '%s\n' "$OUTC" | grep -q 'collect=[0-9]'; then
    awk '$1 == "collect" {$2 = 1} {print}' "$ST" > "$ST.tmp" && mv "$ST.tmp" "$ST"
  fi
fi
tr '\n' ' ' < "$ST"; echo
