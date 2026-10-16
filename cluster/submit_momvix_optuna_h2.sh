#!/bin/bash
# Hoffman2 side of the moments_vix Optuna run (user request 2026-10-06: "use both hoffman and carc
# as much as possible"), from the root cluster/ship_momvix_optuna_h2.sh fills
# (/u/scratch/j/jamesdc1/harxhar-momvix; runtime = CARC's stack in harxhar-optuna, agent D's).
# Every job is pinned to ONE CPU class (ARCH, default intel-E5-2650v4: AVX2 like CARC's canonical
# epyc-7513) and every task refuses a node whose CPU model lacks REQUIRE_CPU.
#
#   bash cluster/submit_momvix_optuna_h2.sh start
#       gate     array of 3 (cluster/momvix_optuna_h2_gate.sh): stored live_feasible tuning points
#                and refits re-computed on this class, compared bit for bit -> GATE_OK_<model>
#       audit    array of 3: moments_vix stage 1 on OOS rows [0, 20) (CARC's canary chunk) into
#                $R/h2audit -- compared afterwards with CARC's canary records (the new bucket itself
#                across clusters); not part of the fleet's results
#       s1_<m>   per model, the Hoffman2 stage-1 lines (cluster/momvix_optuna_tasks_s1_h2.txt),
#                -hold_jid on the gate, each refusing to run without GATE_OK_<model>
#   S2_MODELS="rf" bash cluster/submit_momvix_optuna_h2.sh s2
#       stage-2 lines of those models (cluster/momvix_optuna_tasks_s2.txt) from the merged stage-1
#       records relayed from CARC (STAGE1_COMPLETE checked by the task), GATE_OK_<model> required
# Env: ARCH, REQUIRE_CPU, SLOTS (pool per task, default 8), H_RT, H_DATA (per slot).
# REFUSES a second submission of the same mode (logs/submitted_momvix_h2_<mode>*.txt): jobs cannot
# be cancelled from this session; DONE chunk-arms are skipped, so a resubmission resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN_ARCH=${ARCH:-intel-E5-2650v4}   # captured first: UGE's settings.sh uses (and unsets) ARCH
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
MODE=${1:?usage: submit_momvix_optuna_h2.sh start|s2}
REQUIRE_CPU=${REQUIRE_CPU:-E5-2650_v4}   # spaces of the CPU model read as underscores
SLOTS=${SLOTS:-8}
H_RT=${H_RT:-3:00:00}
H_DATA=${H_DATA:-2G}
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask_momvix}
GT=cluster/momvix_optuna_tasks_gate.txt
AUD=cluster/momvix_optuna_tasks_h2audit.txt
S1H=cluster/momvix_optuna_tasks_s1_h2.txt
S2T=cluster/momvix_optuna_tasks_s2.txt
mkdir -p logs "$R"
for f in "$GT" "$AUD" "$S1H" "$S2T" cluster/momvix_optuna_h2_task.sh cluster/momvix_optuna_h2_gate.sh \
         specs/causal_tune_trees_optuna.py experiments/momvix_optuna_gate.py src/data/loading.py; do
  [ -f "$f" ] || { echo "$f missing (run cluster/ship_momvix_optuna_h2.sh locally)"; exit 1; }
done
grep -q '"moments_vix"' src/data/loading.py || { echo "src/data/loading.py has no moments_vix bucket"; exit 1; }
STAMP="logs/submitted_momvix_h2_${MODE}${S2_MODELS:+_${S2_MODELS// /_}}.txt"
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }

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
RES="h_rt=$H_RT,h_data=$H_DATA,arch=$PIN_ARCH"
ENVV="RESULTS_ROOT=$R,WINDOW_MASK=1,REQUIRE_CPU=$REQUIRE_CPU"

case "$MODE" in
  start)
    rm -f "$R"/GATE_OK_*
    G=$(submit -N mv_h2_gate -t 1-"$(wc -l < "$GT")" -pe shared "$SLOTS" -l "$RES" \
          -v "GATE_TASKFILE=$GT,$ENVV" cluster/momvix_optuna_h2_gate.sh)
    A=$(submit -N mv_h2_aud -t 1-"$(wc -l < "$AUD")" -pe shared "$SLOTS" -l "$RES" \
          -v "TASKFILE=$AUD,STAGE=tune,RESULTS_ROOT=$R/h2audit,WINDOW_MASK=1,REQUIRE_CPU=$REQUIRE_CPU" cluster/momvix_optuna_h2_task.sh)
    IDS=""
    for m in lgbm xgb rf; do
      sub="$R/tasks_s1_h2_$m.txt"
      awk -v m="$m" '$2 == m' "$S1H" > "$sub"
      n=$(wc -l < "$sub")
      [ "$n" -gt 0 ] || continue
      J=$(submit -N "mv_h2_s1_$m" -hold_jid "$G" -t 1-"$n" -pe shared "$SLOTS" -l "$RES" \
            -v "TASKFILE=$sub,STAGE=tune,$ENVV,REQUIRE_FLAGS=$R/GATE_OK_$m" cluster/momvix_optuna_h2_task.sh)
      IDS="$IDS $m:$J($n)"
    done
    echo "start: gate=$G audit=$A s1:$IDS ($SLOTS slots, $RES, cpu '$REQUIRE_CPU') root=$R $(date)" | tee "$STAMP"
    ;;
  s2)
    MODELS=${S2_MODELS:?S2_MODELS must be set}
    sub="$R/tasks_s2_h2_${MODELS// /_}.txt"
    : > "$sub"
    for m in $MODELS; do
      [ -f "$R/GATE_OK_$m" ] || { echo "GATE_OK_$m missing; refusing"; exit 1; }
      [ -f "$R/stage1/moments_vix/bar1600/$m/tw2000/STAGE1_COMPLETE" ] || { echo "stage-1 records of $m not staged; refusing"; exit 1; }
      awk -v m="$m" '$2 == m' "$S2T" >> "$sub"
    done
    J=$(submit -N mv_h2_s2 -t 1-"$(wc -l < "$sub")" -pe shared "$SLOTS" -l "$RES" \
          -v "TASKFILE=$sub,STAGE=refit,$ENVV" cluster/momvix_optuna_h2_task.sh)
    echo "s2 [$MODELS]: $J ($SLOTS slots, $RES) root=$R $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be start or s2"; exit 1 ;;
esac
qstat -u "$USER" | tail -12
