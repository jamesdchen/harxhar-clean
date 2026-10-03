#!/bin/bash
# 2026-10-03 close studies: submit the full-size trees pre-tune study on Hoffman2 (SGE), from the
# deployment root that cluster/close_pretune_ship_h2.sh fills:
#
#   cd /u/scratch/j/jamesdc1/harxhar-close-pretune && bash cluster/submit_close_pretune_h2.sh all <tag>
#
# Modes
#   all <tag>      the pre-tune array (cluster/close_pretune_tasks_pretune.txt: one independent
#                  seeded Optuna study per line, "<model> <seed>") and the walk array
#                  (cluster/close_pretune_tasks_walk.txt: "<model> <seeds> <refit_every>") held on
#                  it with -hold_jid; a walk task refuses to run unless every study it merges wrote
#                  its PRETUNE_DONE flag
#   resume <tag>   the same two arrays again (studies resume from their journals; finished tasks exit
#                  at once) -- after tasks hit h_rt before CTP_TRIALS trials
#   smoke <tag>    one pre-tune task (2 trials, 1 fold) and one walk task (2 retune trials, refit
#                  rows 370-399), 2 slots, 1 h: the pipeline end to end on a compute node
# Sizes (env, defaults = the full run): TRIALS (2000 a study), FOLDS (7), FOLD_LEN (250 sessions:
# 7 x 250 = 1750 pre-2020 validation sessions, 2013-01 .. 2019-12), RETUNE_TRIALS (15),
# CHECKPOINTS (frozen arms: best of the first k trials of every study), SLOTS (8: the 7 folds of a
# trial in parallel), H_RT (24:00:00), H_DATA (2G a slot), ARCH (pin one SGE CPU architecture for
# every task: LightGBM's picks change across CPU classes -- recommended).
# REFUSES a second submission of the same mode + tag (logs/submitted_close_pretune_<mode>_<tag>.txt);
# jobs cannot be cancelled from the laptop's allow-list (qdel needs the user).
set -euo pipefail
cd "$(dirname "$0")/.."
D=$PWD
PIN_ARCH=${ARCH:-}   # captured first: UGE's settings.sh uses (and unsets) ARCH
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
MODE=${1:?usage: submit_close_pretune_h2.sh all|resume|smoke <tag>}
TAG=${2:?usage: submit_close_pretune_h2.sh all|resume|smoke <tag>}
SLOTS=${SLOTS:-8}
H_RT=${H_RT:-24:00:00}
H_DATA=${H_DATA:-2G}
TRIALS=${TRIALS:-2000}
FOLDS=${FOLDS:-7}
FOLD_LEN=${FOLD_LEN:-250}
RETUNE_TRIALS=${RETUNE_TRIALS:-15}
CHECKPOINTS=${CHECKPOINTS:-50,200,1000}
WORKROOT=$D/results/close_studies_2026-10-03/trees_pretune/_work
mkdir -p logs "$WORKROOT"
STAMP="logs/submitted_close_pretune_${MODE}_${TAG}.txt"
if [ "$MODE" = resume ]; then STAMP="logs/submitted_close_pretune_resume_${TAG}_$(date +%Y%m%d%H%M%S).txt"; fi
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }

# inputs: scratch is purged -- every shipped file must be present with its shipped md5
[ -f close_pretune_manifest.md5 ] || { echo "close_pretune_manifest.md5 missing: run cluster/close_pretune_ship_h2.sh on the laptop"; exit 1; }
md5sum -c --quiet close_pretune_manifest.md5 || { echo "shipped files missing or changed (scratch purge?): re-ship"; exit 1; }
RT=${RT:-/u/scratch/j/jamesdc1/harxhar-optuna}
if [ -x "$RT/carc_env/harxhar/bin/python3.11" ]; then echo "runtime: CARC stack at $RT"
else echo "runtime: CARC stack not found at $RT -> conda env hpc-pi (older LightGBM / XGBoost)"; fi

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
RES="h_rt=$H_RT,h_data=$H_DATA${PIN_ARCH:+,arch=$PIN_ARCH}"
PT=cluster/close_pretune_tasks_pretune.txt
WT=cluster/close_pretune_tasks_walk.txt
if [ "$MODE" = smoke ]; then
  head -1 "$PT" > "logs/close_pretune_smoke_pretune_$TAG.txt"
  echo "$(head -1 "$PT" | awk '{print $1, $2}') 10" > "logs/close_pretune_smoke_walk_$TAG.txt"
  PT="logs/close_pretune_smoke_pretune_$TAG.txt"; WT="logs/close_pretune_smoke_walk_$TAG.txt"
  TRIALS=2; FOLDS=1; RETUNE_TRIALS=2; CHECKPOINTS=1; SLOTS=2; RES="h_rt=1:00:00,h_data=$H_DATA${PIN_ARCH:+,arch=$PIN_ARCH}"
  EXTRA=",CTP_WALK_ROWS=370:400"
else
  EXTRA=""
fi
# the time guard: a study stops cleanly (and exports) 45 minutes before h_rt
IFS=: read -r hh mm ss <<< "$H_RT"
TIMEOUT=$(( 10#$hh * 3600 + 10#$mm * 60 + 10#$ss - 2700 ))
[ "$TIMEOUT" -gt 0 ] || TIMEOUT=600
COMMON="CTP_TAG=$TAG,CTP_WORK=$WORKROOT,CTP_TRIALS=$TRIALS,CTP_FOLDS=$FOLDS,CTP_FOLD_LEN=$FOLD_LEN,CTP_TIMEOUT=$TIMEOUT,CTP_RETUNE_TRIALS=$RETUNE_TRIALS,CTP_CHECKPOINTS=${CHECKPOINTS//,/:},RT=$RT$EXTRA"
case "$MODE" in
  all|resume|smoke)
    J1=$(submit -N ctp_pre -t 1-"$(wc -l < "$PT")" -pe shared "$SLOTS" -l "$RES" \
          -v "TASKFILE=$PT,STAGE=pretune,$COMMON" cluster/close_pretune_task.sh)
    J2=$(submit -N ctp_walk -hold_jid "$J1" -t 1-"$(wc -l < "$WT")" -pe shared "$SLOTS" -l "$RES" \
          -v "TASKFILE=$WT,STAGE=walk,$COMMON" cluster/close_pretune_task.sh)
    echo "$MODE $TAG: pretune=$J1 ($(wc -l < "$PT") studies x $TRIALS trials, $FOLDS folds x $FOLD_LEN) walk=$J2 ($(wc -l < "$WT") tasks, held) [$SLOTS slots, $RES, timeout ${TIMEOUT}s] $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be all, resume or smoke"; exit 1 ;;
esac
qstat -u "$USER" | tail -12
