#!/bin/bash
# I5 (2026-09-29): move whole models' Optuna stage-2 tasks from CARC to Hoffman2 and back, for C's
# run (results root CROOT, stage-2 array S2JOB).  Run locally (Git Bash), one step at a time:
#
#   prep      read the stage-2 task file the CARC array reads (cluster/optuna_tasks_s2.txt on CARC;
#             snapshot + md5 kept), write cluster/optuna_h2_offload_s2_<tag>.txt = the lines of MODELS
#             (C's own lines, unchanged) and $O/carc_tasks.txt = their CARC array task ids (= line
#             numbers); refuse unless every one of those tasks is PENDING on CARC.
#   hold      scontrol hold those tasks (all PENDING), then show their state / reason.
#   stage     once $CROOT/STAGE1_MERGED exists on CARC: relay each arm's merged stage-1 records
#             ($CROOT/stage1/<arm>/trials_<seg>.npz + STAGE1_COMPLETE) and the task file to the
#             Hoffman2 root (CARC -> here -> Hoffman2, md5 == CARC's).  Then submit on Hoffman2:
#               OROOT=$CROOT WINDOW_MASK=1 bash cluster/submit_optuna_h2.sh offload <tag>
#   copyback  for every line: the Hoffman2 chunk dir ($CROOT/stage2/<arm>/chunks/c<k>) must carry DONE;
#             the CARC chunk dir must NOT (never overwrite a CARC result) and the CARC task must still
#             be held; relay the whole chunk dir Hoffman2 -> here -> CARC, then FAIL unless every
#             file's CARC md5 equals its Hoffman2 md5.
#   release   scontrol release the held tasks (whatever happened before: CARC then computes any chunk
#             that was not copied back), then show their state; `verify` later greps their Slurm logs
#             for "already DONE".
#   verify    the released tasks' Slurm logs: "already DONE" and the elapsed time from sacct.
#
#   TAG=mask CROOT=results/linear_subsection_trees_optuna_mask S2JOB=12481000 MODELS="lgbm rf" \
#     bash cluster/optuna_h2_offload.sh prep
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
CARC=/scratch1/jc_905/harxhar-subsection
H2=/u/scratch/j/jamesdc1/harxhar-optuna
STEP=${1:?step: prep | hold | stage | copyback | release | verify}
TAG=${TAG:?TAG}
CROOT=${CROOT:?CROOT}
S2JOB=${S2JOB:?S2JOB (the CARC stage-2 array job id)}
O=results/linear_subsection_trees_optuna/crosscluster/offload_$TAG
OT=cluster/optuna_h2_offload_s2_$TAG.txt
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/optuna_h2_offload_$TAG}
mkdir -p "$O" "$STAGE_DIR"
carc() { "$SSH" -o BatchMode=yes usc-discovery "$@" 2>/dev/null | tr -d '\r'; }
h2() { "$SSH" -o BatchMode=yes hoffman2 "$@" | tr -d '\r'; }
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
ids() { awk -v j="$S2JOB" '{printf "%s%s_%s", (NR > 1 ? "," : ""), j, $1}' "$O/carc_tasks.txt"; }

case "$STEP" in
  prep)
    MODELS=${MODELS:?MODELS (e.g. "lgbm rf")}
    carc "cd $CARC && cat cluster/optuna_tasks_s2.txt" > "$O/tasks_s2_snapshot.txt"
    carc "cd $CARC && md5sum cluster/optuna_tasks_s2.txt" | cut -c1-32 > "$O/tasks_s2_snapshot.md5"
    [ "$(md5sum < "$O/tasks_s2_snapshot.txt" | cut -c1-32)" = "$(cat "$O/tasks_s2_snapshot.md5")" ] \
      || { echo "snapshot md5 != CARC task file md5"; exit 1; }
    awk -v ms=" $MODELS " 'index(ms, " " $2 " ") {print NR}' "$O/tasks_s2_snapshot.txt" > "$O/carc_tasks.txt"
    awk -v ms=" $MODELS " 'index(ms, " " $2 " ")' "$O/tasks_s2_snapshot.txt" > "$OT"
    echo "$(wc -l < "$OT") stage-2 lines of [$MODELS] -> $OT; CARC tasks $(ids)"
    ST=$(carc "squeue -h -r -j $S2JOB -o '%K %T %r'" | sort -n)
    NP=$(awk 'NR == FNR {w[$1] = 1; next} ($1 in w) && $2 == "PENDING"' "$O/carc_tasks.txt" <(printf '%s\n' "$ST") | wc -l)
    echo "PENDING on CARC: $NP of $(wc -l < "$O/carc_tasks.txt")"
    [ "$NP" = "$(wc -l < "$O/carc_tasks.txt")" ] || { echo "not every task is pending: refusing"; exit 1; }
    ;;
  hold)
    "$SSH" -o BatchMode=yes usc-discovery "scontrol hold $(ids)" 2>&1 | grep -v 'reloaded\|python/3' | tr -d '\r' || true
    echo "scontrol hold $(ids) $(date)" >> "$O/log.txt"
    carc "squeue -h -r -j $S2JOB -o '%K %T %r'" | sort -n \
      | awk 'NR == FNR {w[$1] = 1; next} ($1 in w)' "$O/carc_tasks.txt" - | tee "$O/held_state.txt"
    echo "held: $(grep -c JobHeldUser "$O/held_state.txt") of $(wc -l < "$O/carc_tasks.txt") $(date)" | tee -a "$O/log.txt"
    ;;
  stage)
    carc "test -f $CARC/$CROOT/STAGE1_MERGED" || { echo "no $CROOT/STAGE1_MERGED on CARC yet"; exit 1; }
    awk '{print $1, $2, $3, $4}' "$OT" | sort -u > "$STAGE_DIR/arms"
    : > "$STAGE_DIR/list"
    while read -r B M TW SEG; do
      echo "$CROOT/stage1/$B/$SEG/$M/tw$TW/trials_$SEG.npz" >> "$STAGE_DIR/list"
      echo "$CROOT/stage1/$B/$SEG/$M/tw$TW/STAGE1_COMPLETE" >> "$STAGE_DIR/list"
    done < "$STAGE_DIR/arms"
    "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && tar czf - -T -" < "$STAGE_DIR/list" 2>/dev/null > "$STAGE_DIR/s1.tgz"
    h2 "cd $H2 && tar xzf -" < "$STAGE_DIR/s1.tgz"
    tar czf - "$OT" | h2 "cd $H2 && tar xzf -"
    echo "$OT" >> "$STAGE_DIR/list"
    C=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && xargs -d '\n' md5sum" < <(grep -v "^$OT\$" "$STAGE_DIR/list") 2>/dev/null | sums)
    C=$(printf '%s\n%s\n' "$C" "$(md5sum "$OT" | sums)" | sort)
    H=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs -d '\n' md5sum" < "$STAGE_DIR/list" | sums)
    [ "$C" = "$H" ] || { echo "STAGE FAILED: md5"; diff <(printf '%s\n' "$C") <(printf '%s\n' "$H"); exit 1; }
    echo "staged on Hoffman2: $(grep -c trials_ "$STAGE_DIR/list") arms' merged stage-1 records + $OT, md5 == CARC $(date)" | tee -a "$O/log.txt"
    ;;
  copyback)
    : > "$STAGE_DIR/dirs"
    while read -r B M TW SEG C S E HALO; do
      echo "$CROOT/stage2/$B/$SEG/$M/tw$TW/chunks/c$C" >> "$STAGE_DIR/dirs"
    done < "$OT"
    NOTDONE=$(h2 "cd $H2 && while read -r d; do [ -f \$d/DONE ] || echo \$d; done" < "$STAGE_DIR/dirs")
    [ -z "$NOTDONE" ] || { echo "not DONE on Hoffman2:"; printf '  %s\n' $NOTDONE; exit 1; }
    ONCARC=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && while read -r d; do [ -f \$d/DONE ] && echo \$d; done; true" < "$STAGE_DIR/dirs" 2>/dev/null | tr -d '\r')
    [ -z "$ONCARC" ] || { echo "already DONE on CARC (not overwritten): $ONCARC"; exit 1; }
    ST=$(carc "squeue -h -r -j $S2JOB -o '%K %T %r'")
    NH=$(awk 'NR == FNR {w[$1] = 1; next} ($1 in w) && $2 == "PENDING" && $3 == "JobHeldUser"' "$O/carc_tasks.txt" <(printf '%s\n' "$ST") | wc -l)
    [ "$NH" = "$(wc -l < "$O/carc_tasks.txt")" ] || { echo "only $NH tasks still held: refusing to copy"; exit 1; }
    h2 "cd $H2 && while read -r d; do find \$d -type f; done | sort" < "$STAGE_DIR/dirs" > "$STAGE_DIR/files"
    "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar czf - -T -" < "$STAGE_DIR/files" > "$STAGE_DIR/s2.tgz"
    "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && tar xzf -" < "$STAGE_DIR/s2.tgz" 2>/dev/null
    H=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs -d '\n' md5sum" < "$STAGE_DIR/files" | sums)
    C=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && xargs -d '\n' md5sum" < "$STAGE_DIR/files" 2>/dev/null | sums)
    [ "$C" = "$H" ] || { echo "COPYBACK FAILED: md5 differs"; diff <(printf '%s\n' "$H") <(printf '%s\n' "$C") | head; exit 1; }
    printf '%s\n' "$H" > "$O/copyback_md5.txt"
    echo "copied back: $(wc -l < "$STAGE_DIR/dirs") chunk dirs, $(wc -l < "$STAGE_DIR/files") files, CARC md5 == Hoffman2 $(date)" | tee -a "$O/log.txt"
    ;;
  release)
    "$SSH" -o BatchMode=yes usc-discovery "scontrol release $(ids)" 2>&1 | grep -v 'reloaded\|python/3' | tr -d '\r' || true
    echo "scontrol release $(ids) $(date)" >> "$O/log.txt"
    carc "squeue -h -r -j $S2JOB -o '%K %T %r'" | sort -n \
      | awk 'NR == FNR {w[$1] = 1; next} ($1 in w)' "$O/carc_tasks.txt" - | tee "$O/released_state.txt"
    echo "released $(date); still held: $(grep -c JobHeldUser "$O/released_state.txt" || true)" | tee -a "$O/log.txt"
    ;;
  verify)
    carc "sacct -j $S2JOB -X -n -o JobID%16,State%12,Elapsed,NodeList%12" \
      | awk -v j="$S2JOB" 'NR == FNR {w[j "_" $1] = 1; next} ($1 in w)' "$O/carc_tasks.txt" - | tee "$O/released_sacct.txt"
    carc "cd $CARC && for t in \$(cat); do f=logs/op_s2.${S2JOB}.\$t.out; [ -f \$f ] || f=\$(ls logs/*.${S2JOB}.\$t.out 2>/dev/null | head -1); echo \"\$t: \$(grep -c 'already DONE' \$f 2>/dev/null) already-DONE lines | \$(tail -1 \$f 2>/dev/null)\"; done" \
      < "$O/carc_tasks.txt" | tee "$O/released_logs.txt"
    ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
