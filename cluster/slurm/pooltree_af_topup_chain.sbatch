#!/bin/bash
#SBATCH -J ptr_topup
#SBATCH -p main
#SBATCH -c 1
#SBATCH --mem=512M
#SBATCH -t 00:02:00
#SBATCH --open-mode=append
# Pooled forest all_features (2026-10-14): top-up every 30 MINUTES (hourly until 2026-10-14 19:09) as a SELF-CHAINING 1-CPU job (CARC has scrontab disabled; no
# login-node cron).  On 2026-10-14 CARC sat at 0 running for 8.5 h while the 4/8-CPU reserve was held and the agent
# session was suspended; this keeps the reserve flowing without the agent.
#   RELEASE ONLY: `scontrol release` on HELD tasks of the forest reserve arrays, 8-CPU w7 (13074358) and w24 (13166916) first, then
#   the 4-CPU w9 / w10 -- it never submits forest work, holds, cancels or touches any other job.  It acts only while
#   fewer than TARGET (97) of my jobs run (GPU jobs counted, this job excluded), and releases until BUF (50) unheld small
#   (<= 8 CPU) tasks are pending on main (the buffer stays 50; the 30-min cadence covers the 8-CPU tasks' ~25-min turnover).
#   Guards: STOP file ($HOME/pooltree_topup.STOP) -> exit: no release, no resubmit (ends the chain);
#           expiry -> no resubmit once no held task is left, the full table exists, or after 2026-10-19 00:00 PT;
#           single chain -> resubmits only if no OTHER ptr_topup job is pending or running;
#           fails closed -> if squeue / scontrol errors, log it, release nothing this run, still resubmit.
#   Log: one line a run appended to $HOME/pooltree_topup.log.
#   Installed as $HOME/pooltree_topup.sbatch; first submitted by cluster/slurm/submit_pooltree_topup.sh.
set -u
SELFSCRIPT=$HOME/pooltree_topup.sbatch
LOG=$HOME/pooltree_topup.log
ARRAYS8="13074358 13166916 13174440"; ARRAYS4="13077718 13080577"   # 8-CPU w7, w24 (2026-10-15 13:16), w28 (17:27), then 4-CPU w9 / w10
ARRAYS="$ARRAYS8 $ARRAYS4"
TARGET=97; BUF=50
TABLE=/scratch1/jc_905/harxhar-pooltrees/results/pooled_trees_2026-10-07/yhat_pooltree_rf_all_features.parquet
HARD_END=$(TZ=America/Los_Angeles date -d '2026-10-19 00:00' +%s)
SELF=${SLURM_JOB_ID:-0}
DRY=${TOPUP_DRY:-0}   # TOPUP_DRY=1 bash <this file>: a test -- no release, no resubmit, the line goes to stdout
NOW=$(date '+%F %T')
if [ -e "$HOME/pooltree_topup.STOP" ]; then echo "$NOW job $SELF: STOP file present; nothing done, chain ended" >> "$LOG"; exit 0; fi
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT
ERR=""
# ONE squeue call for everything (a per-array `squeue -j` errors once an array has no task left)
snap() { local t; for t in 1 2 3; do squeue -u "$USER" -h -r -o "%A|%i|%j|%T|%r|%C|%P" > "$TMP" 2>/dev/null && return 0; sleep 15; done; return 1; }
if snap; then
  RUN=$(awk -F'|' -v s="$SELF" '$4 == "RUNNING" && $1 != s' "$TMP" | wc -l)
  SMALL=$(awk -F'|' '$4 == "PENDING" && $7 == "main" && $6 <= 8 && $5 != "JobHeldUser"' "$TMP" | wc -l)
  held() { awk -F'|' -v j="$1" '$4 == "PENDING" && $5 == "JobHeldUser" && index($2, j "_") == 1 {sub(/.*_/, "", $2); print $2}' "$TMP" | sort -n; }
  H8=0; for J in $ARRAYS8; do H8=$((H8 + $(held "$J" | wc -l))); done
  H4=0; for J in $ARRAYS4; do H4=$((H4 + $(held "$J" | wc -l))); done
  NH=$((H8 + H4))
else
  ERR="squeue failed 3 times"; RUN="?"; SMALL="?"; H8="?"; H4="?"; NH="?"
fi

REL=""
if [ -z "$ERR" ] && [ "$RUN" -lt "$TARGET" ] && [ "$SMALL" -lt "$BUF" ]; then
  N=$((BUF - SMALL))
  for J in $ARRAYS; do
    [ "$N" -gt 0 ] || break
    IDX=$(held "$J" | head -n "$N" | paste -sd, -)
    [ -n "$IDX" ] || continue
    if [ "$DRY" = 1 ]; then
      K=$(echo "$IDX" | tr , '\n' | wc -l); N=$((N - K)); REL="$REL ${J}:would-release-${K}(${IDX%%,*}..${IDX##*,})"; continue
    fi
    if scontrol release "${J}_[$IDX]" 2>/dev/null; then
      K=$(echo "$IDX" | tr , '\n' | wc -l); N=$((N - K)); REL="$REL ${J}:${K}(${IDX%%,*}..${IDX##*,})"
      case " $ARRAYS8 " in *" $J "*) H8=$((H8 - K)) ;; *) H4=$((H4 - K)) ;; esac
    else
      ERR="scontrol release failed on $J"; break
    fi
  done
fi
[ -n "$ERR" ] && REL="$REL [error: $ERR; nothing further released this run]"

if [ -f "$TABLE" ]; then NEXT="not resubmitted (the full table exists)"
elif [ "$(date +%s)" -ge "$HARD_END" ]; then NEXT="not resubmitted (hard expiry 2026-10-19 00:00 PT)"
elif [ "$NH" = 0 ]; then NEXT="not resubmitted (no held tasks left)"
else
  # single chain: no OTHER ptr_topup job may be pending or running (fails closed: unknown -> resubmit, logged)
  if snap; then
    OTHERS=$(awk -F'|' -v s="$SELF" '$3 == "ptr_topup" && $1 != s && ($4 == "PENDING" || $4 == "RUNNING") {print $1}' "$TMP" | sort -u | paste -sd, -)
  else
    OTHERS=""; ERR="$ERR; chain check failed (squeue)"
  fi
  if [ -n "$OTHERS" ]; then
    NEXT="not resubmitted (another ptr_topup job is queued: $OTHERS)"
  else
    if [ "$DRY" = 1 ]; then J2="(dry run: not resubmitted)"; else
    J2=$(sbatch --parsable --begin=now+30minutes -o "$HOME/pooltree_topup.slurm.out" "$SELFSCRIPT" 2>&1) || J2="FAILED: $J2"
    fi
    NEXT="resubmitted as $J2${ERR:+ (despite: $ERR)}"
  fi
fi
LINE="$NOW job $SELF: running $RUN (GPU incl.) small-pending $SMALL released${REL:- none} still-held 8cpu $H8 4cpu $H4; $NEXT"
if [ "$DRY" = 1 ]; then echo "$LINE"; else echo "$LINE" >> "$LOG"; fi
