#!/bin/bash
# moments_vix tree workstream (2026-10-06): the Hoffman2 side, from the root cluster/ship_trees_momvix_h2.sh
# fills (/u/scratch/j/jamesdc1/harxhar-momvix-trees; runtime = CARC's stack in agent D's harxhar-optuna,
# read only).  Every job is pinned to ONE CPU class (ARCH, default intel-E5-2650v4: AVX2 without
# AVX-512, like CARC's epyc-7513 on which every stored masked chunk ran) and every task refuses a
# node whose CPU model lacks REQUIRE_CPU.
#
#   bash cluster/submit_trees_momvix_h2.sh gate
#       one job (pool of SLOTS): cluster/trees_momvix_gate_lines.txt -- the stored all_features cells of
#       every rung and model re-run here, compared byte for byte with the CARC chunks relayed into ./ref
#       -> results/trees_momvix_gates/h2_<ARCH>/ (GATE_OK, PASS_<rung>_<model>)
#   bash cluster/submit_trees_momvix_h2.sh untuned <taskfile>
#       one array task per line (1 slot, single-threaded fit), lines in the untuned 10-field format;
#       each task requires the PASS flags of its lines' rung and model (checked here at submission and
#       by the task: REQUIRE_FLAGS)
#   bash cluster/submit_trees_momvix_h2.sh tuned <rs10|rs1> <taskfile>
#       one array task per line (pool of SLOTS), tuned 8-field lines, PASS flags required
# Results: RESULTS_ROOT_U (results/linear_subsection_trees_mask_momvix) / RESULTS_ROOT_T
# (results/linear_subsection_trees_tuned_mask_momvix): the CARC layout, relayed to CARC afterwards.
# Env: ARCH, REQUIRE_CPU, SLOTS (default 8), H_RT, H_DATA (per slot), RESULTS_ROOT_U / RESULTS_ROOT_T,
# HOLD_JID (the gate job: with it, untuned / tuned may be submitted before the PASS flags exist --
# -hold_jid on the gate, and every task refuses to run without its PASS flags, REQUIRE_FLAGS).
# REFUSES a second identical submission (logs/submitted_trees_momvix_h2_<mode>_<file>.txt): jobs cannot
# be cancelled from this session; DONE chunks are skipped, so a resubmission resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN_ARCH=${ARCH:-intel-E5-2650v4}   # captured first: UGE's settings.sh uses (and unsets) ARCH
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
MODE=${1:?usage: submit_trees_momvix_h2.sh gate|untuned <taskfile>|tuned <rung> <taskfile>}
REQUIRE_CPU=${REQUIRE_CPU:-E5-2650_v4}   # spaces of the CPU model read as underscores
SLOTS=${SLOTS:-8}
H_RT=${H_RT:-3:00:00}
H_DATA=${H_DATA:-4G}
U=${RESULTS_ROOT_U:-results/linear_subsection_trees_mask_momvix}
T=${RESULTS_ROOT_T:-results/linear_subsection_trees_tuned_mask_momvix}
G=results/trees_momvix_gates/h2_$PIN_ARCH
mkdir -p logs "$U" "$T"
for f in cluster/trees_momvix_h2_task.sh experiments/trees_momvix_gate.py cluster/trees_momvix_gate_lines.txt \
         specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_linear.py src/data/loading.py; do
  [ -f "$f" ] || { echo "$f missing (run cluster/ship_trees_momvix_h2.sh locally)"; exit 1; }
done
grep -q '"moments_vix"' src/data/loading.py || { echo "src/data/loading.py has no moments_vix bucket"; exit 1; }
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

case "$MODE" in
  gate)
    STAMP=logs/submitted_trees_momvix_h2_gate_$PIN_ARCH.txt
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
    mkdir -p "$G"
    rm -f "$G/GATE_OK" "$G"/PASS_*
    J=$(submit -N tmv_h2_gate -pe shared "$SLOTS" -l "$RES" \
          -v "MODE=gate,GATE_LINES=cluster/trees_momvix_gate_lines.txt,REF_ROOT=ref,OUT=$G,REQUIRE_CPU=$REQUIRE_CPU" \
          cluster/trees_momvix_h2_task.sh)
    echo "gate=$J ($SLOTS slots, $RES, cpu '$REQUIRE_CPU') out=$G $(date)" | tee "$STAMP"
    ;;
  untuned)
    TF=${2:?untuned task file}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    STAMP="logs/submitted_trees_momvix_h2_untuned_$(basename "$TF" .txt).txt"
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
    FL=""
    for k in $(awk '{print $2 "_" $4}' "$TF" | sort -u); do
      [ -f "$G/PASS_$k" ] || [ -n "${HOLD_JID:-}" ] || { echo "$G/PASS_$k missing: the gate did not qualify $k on $PIN_ARCH; refusing"; exit 1; }
      FL="$FL:$G/PASS_$k"
    done
    N=$(grep -c . "$TF")
    J=$(submit -N tmv_h2_untuned ${HOLD_JID:+-hold_jid $HOLD_JID} -t 1-"$N" -l "$RES" \
          -v "MODE=untuned,TASKFILE=$TF,RESULTS_ROOT=$U,REQUIRE_CPU=$REQUIRE_CPU,REQUIRE_FLAGS=${FL#:}" \
          cluster/trees_momvix_h2_task.sh)
    echo "untuned $TF: $J (1-$N, 1 slot, $RES${HOLD_JID:+, hold_jid $HOLD_JID}) root $U $(date)" | tee "$STAMP"
    ;;
  tuned)
    R=${2:?rung rs10 or rs1}
    TF=${3:?tuned task file}
    case "$R" in rs10|rs1) ;; *) echo "rung must be rs10 or rs1"; exit 1 ;; esac
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    STAMP="logs/submitted_trees_momvix_h2_tuned_${R}_$(basename "$TF" .txt).txt"
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
    FL=""
    for m in $(awk '{print $2}' "$TF" | tr ',' '\n' | sort -u); do
      [ -f "$G/PASS_${R}_$m" ] || [ -n "${HOLD_JID:-}" ] || { echo "$G/PASS_${R}_$m missing; refusing"; exit 1; }
      FL="$FL:$G/PASS_${R}_$m"
    done
    N=$(grep -c . "$TF")
    J=$(submit -N "tmv_h2_$R" ${HOLD_JID:+-hold_jid $HOLD_JID} -t 1-"$N" -pe shared "$SLOTS" -l "$RES" \
          -v "MODE=tuned,RUNG=$R,TASKFILE=$TF,RESULTS_ROOT=$T,REQUIRE_CPU=$REQUIRE_CPU,REQUIRE_FLAGS=${FL#:}" \
          cluster/trees_momvix_h2_task.sh)
    echo "tuned $R $TF: $J (1-$N, $SLOTS slots, $RES${HOLD_JID:+, hold_jid $HOLD_JID}) root $T $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be gate, untuned or tuned"; exit 1 ;;
esac
qstat -u "$USER" | tail -8
