#!/bin/bash
# I5 (2026-09-29): submit the Hoffman2 side of the Optuna campaign's cross-cluster work, from the
# Hoffman2 deployment root that cluster/ship_optuna_h2.sh fills:
#
#   cd /u/scratch/j/jamesdc1/harxhar-optuna && bash cluster/submit_optuna_h2.sh xcheck <tag>
#   cd /u/scratch/j/jamesdc1/harxhar-optuna && bash cluster/submit_optuna_h2.sh offload <tag>
#
# xcheck   the cross-cluster check (results under $XROOT = results/<optuna root>/crosscluster/h2_<tag>):
#            s1     stage 1 on a few tuning points (cluster/optuna_h2_xcheck_s1_<tag>.txt), to be
#                   compared trial by trial with CARC's records of the same rows
#            s2     stage 2 on a refit block (cluster/optuna_h2_xcheck_s2_<tag>.txt) from CARC's own
#                   stage-1 records of that block (staged by the ship script under $XROOT/stage1/),
#                   to be compared with CARC's stage 2 of the same block
#            probe  experiments/optuna_h2_design_probe.py on every s2 line (design / target hashes and
#                   the per-window kept-column set)
# archprobe  the design probe pinned to each CPU architecture in ARCHS (one array per arch)
# offload  whole models' stage-2 tasks of C's run (cluster/optuna_h2_offload_s2_<tag>.txt: C's own
#          task lines, one per held CARC task) into C's layout under $OROOT (C's results root), from
#          the merged stage-1 records the relay staged there.
# Env: WINDOW_MASK (0 | 1, as C's run), OROOT (offload: C's results root), ARCH (pin an SGE arch), SLOTS (pool per task,
# default 8), H_RT, H_DATA (per slot).  Every fit is single-threaded; the pool size changes nothing.
# REFUSES a second submission of the same mode + tag (logs/submitted_optuna_h2_<mode>_<tag>.txt):
# jobs cannot be cancelled from here; DONE chunk-arms are skipped, so a resubmission resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN_ARCH=${ARCH:-}   # captured first: UGE's settings.sh uses (and unsets) ARCH
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
MODE=${1:?usage: submit_optuna_h2.sh xcheck|archprobe|offload <tag>}
TAG=${2:?usage: submit_optuna_h2.sh xcheck|offload <tag>}
SLOTS=${SLOTS:-8}
H_RT=${H_RT:-3:00:00}
H_DATA=${H_DATA:-3G}
WM=${WINDOW_MASK:?WINDOW_MASK must be set (0 or 1, as the CARC run)}
mkdir -p logs
XR=${XROOT:-${OROOT:-}}
STAMP="logs/submitted_optuna_h2_${MODE}_${TAG}${XR:+_${XR##*/}}${PARTS:+_${PARTS// /_}}${PIN_ARCH:+_${PIN_ARCH//|/+}}.txt"
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
[ -x carc_env/harxhar/bin/python3.11 ] || { echo "no CARC runtime (cluster/optuna_h2_runtime_setup.sh)"; exit 1; }

submit () {  # the scheduler's submission verifier times out now and then: retry
  local out n=0
  until out=$(qsub -terse -S /bin/bash "$@" 2>&1); do
    n=$((n + 1))
    echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}
RES="h_rt=$H_RT,h_data=$H_DATA${PIN_ARCH:+,arch=$PIN_ARCH}"   # ARCH: pin one CPU architecture (qsub -l arch=...)

case "$MODE" in
  xcheck)
    XROOT=${XROOT:?XROOT must be set (the check results root)}
    S1T=cluster/optuna_h2_xcheck_s1_$TAG.txt
    S2T=cluster/optuna_h2_xcheck_s2_$TAG.txt
    NEED="$S2T"; [[ " ${PARTS:-s1 s2 probe} " == *" s1 "* ]] && NEED="$S1T $S2T"
    for f in $NEED specs/causal_tune_trees_optuna.py experiments/optuna_h2_design_probe.py; do
      [ -f "$f" ] || { echo "$f missing (run cluster/ship_optuna_h2.sh locally)"; exit 1; }
    done
    PARTS=${PARTS:-s1 s2 probe}   # a subset when the stage-1 records are not staged yet
    J1=-; J2=-; J3=-
    if [[ " $PARTS " == *" s1 "* ]]; then
      J1=$(submit -N op_h2_xs1 -t 1-"$(wc -l < "$S1T")" -pe shared "$SLOTS" -l "$RES"             -v TASKFILE=$S1T,STAGE=tune,RESULTS_ROOT=$XROOT,WINDOW_MASK=$WM cluster/optuna_h2_task.sh)
    fi
    if [[ " $PARTS " == *" s2 "* ]]; then
      while read -r B M TW SEG C S E H; do
        [ -f "$XROOT/stage1/$B/$SEG/$M/tw$TW/STAGE1_COMPLETE" ] || { echo "records of $B $M not staged; refusing s2"; exit 1; }
      done < "$S2T"
      J2=$(submit -N op_h2_xs2 -t 1-"$(wc -l < "$S2T")" -pe shared "$SLOTS" -l "$RES"             -v TASKFILE=$S2T,STAGE=refit,RESULTS_ROOT=$XROOT,WINDOW_MASK=$WM cluster/optuna_h2_task.sh)
    fi
    PT=cluster/optuna_h2_xcheck_probe_$TAG.txt; [ -f "$PT" ] || PT=$S2T
    if [[ " $PARTS " == *" probe "* ]]; then
      J3=$(submit -N op_h2_probe -t 1-"$(wc -l < "$PT")" -l "h_rt=1:00:00,h_data=12G"             -v TASKFILE=$PT,XROOT=$XROOT,WINDOW_MASK=$WM cluster/optuna_h2_probe.sh)
    fi
    echo "xcheck $TAG [$PARTS]: s1=$J1 s2=$J2 probe=$J3 (mask $WM, $SLOTS slots, $RES) $(date)" | tee -a "$STAMP"
    ;;
  archprobe)
    # the design probe pinned to each Hoffman2 CPU architecture (ARCHS), on the lines of
    # cluster/optuna_h2_xcheck_archprobe_<tag>.txt: does the design X depend on the CPU?
    XROOT=${XROOT:?XROOT must be set (the check results root)}
    PT=cluster/optuna_h2_xcheck_archprobe_$TAG.txt
    [ -f "$PT" ] || { echo "$PT missing"; exit 1; }
    IDS=""
    for A in ${ARCHS:?ARCHS (space-separated SGE arch values)}; do
      J=$(submit -N "op_h2_ap" -t 1-"$(wc -l < "$PT")" -l "h_rt=1:00:00,h_data=12G,arch=$A"             -v TASKFILE=$PT,XROOT=$XROOT,WINDOW_MASK=$WM cluster/optuna_h2_probe.sh)
      IDS="$IDS $A:$J"
    done
    echo "archprobe $TAG:$IDS $(date)" | tee "$STAMP"
    ;;
  offload)
    OROOT=${OROOT:?OROOT (C results root) must be set}
    OT=cluster/optuna_h2_offload_s2_$TAG.txt
    [ -f "$OT" ] || { echo "$OT missing"; exit 1; }
    while read -r B M TW SEG C S E H; do
      [ -f "$OROOT/stage1/$B/$SEG/$M/tw$TW/STAGE1_COMPLETE" ] \
        || { echo "stage-1 records of $B $M not staged under $OROOT; refusing"; exit 1; }
    done < "$OT"
    J=$(submit -N op_h2_s2 -t 1-"$(wc -l < "$OT")" -pe shared "$SLOTS" -l "$RES" \
          -v TASKFILE=$OT,STAGE=refit,RESULTS_ROOT=$OROOT,WINDOW_MASK=$WM cluster/optuna_h2_task.sh)
    echo "offload $TAG: s2=$J ($(wc -l < "$OT") tasks, mask $WM, $SLOTS slots, $RES) $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be xcheck, archprobe or offload"; exit 1 ;;
esac
qstat -u "$USER" | tail -8
