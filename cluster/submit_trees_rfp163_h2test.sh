#!/bin/bash
# Cross-CPU test of the forest's TUNED rungs on Hoffman2 AVX (2026-10-07): the trial-163 re-tuning canary chunks that ran on
# CARC epyc-7513 (random search rs10 chunk 0 of baseline / all_features / lean_vol_vix, rs1 chunk 0 of baseline / lean_vol_vix,
# the unmasked fixed-settings pack-0 chunks) re-run on Hoffman2 pinned to the AVX models E5-2670 / E5-2650 v2 (arch request +
# in-task CPU-model check), same code (specs/causal_tune_trees_tuned_rfp.py with specs/causal_tune_trees_rfp163.py,
# specs/causal_tune_trees_rfdev.py), configuration and chunk geometry, to be compared byte for byte with the CARC chunks
# (forecasts, the tuning trace: candidates' validation MSE / QLIKE, chosen configurations).  Task script
# cluster/trees_rfprev_h2_task.sh (MODE tuned / chunk).  Root /u/scratch/j/jamesdc1/harxhar-rfp163-h2test.
#   bash cluster/submit_trees_rfp163_h2test.sh      (once; stamp)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} active; submit from a clean shell"; exit 1 ;; esac
ARCHES=${ARCHES:-intel-E5-2670|intel-E5-2650v2}
REQ=${REQUIRE_CPU:-E5-2670:E5-2650_v2}
CFG=cluster/rfprev163_rf_config.json
T=results/linear_subsection_trees_tuned_mask_rfp163
N=results/linear_subsection_trees_nomask_rfp163
SLOTS=${SLOTS:-8}
mkdir -p logs "$T" "$N"
STAMP=logs/submitted_trees_rfp163_h2test.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
submit () {
  local out n=0
  until out=$(qsub -terse -S /bin/bash "$@" 2>&1); do
    n=$((n + 1)); echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}
COMMON="RESULTS_ROOT=$T,HPC_KW_RF_CONFIG=$CFG,HPC_KW_TREE_SPEC=causal_tune_trees_rfp163.py,REQUIRE_CPU=$REQ"
A=$(submit -N rfpt_rs10 -t 1-3 -pe shared "$SLOTS" -l "h_rt=6:00:00,h_data=4G,arch=$ARCHES" \
      -v "MODE=tuned,RUNG=rs10,TASKFILE=cluster/trees_rfp163_tasks_rs_canary.txt,$COMMON" cluster/trees_rfprev_h2_task.sh)
B=$(submit -N rfpt_rs1 -t 1-2 -pe shared "$SLOTS" -l "h_rt=6:00:00,h_data=4G,arch=$ARCHES" \
      -v "MODE=tuned,RUNG=rs1,TASKFILE=cluster/trees_rfp163_h2test_rs1_lines.txt,$COMMON" cluster/trees_rfprev_h2_task.sh)
C=$(submit -N rfpt_nm -t 1-"$(grep -c . cluster/trees_rfp163_h2test_nm_lines.txt)" -l "h_rt=4:00:00,h_data=8G,arch=$ARCHES" \
      -v "MODE=chunk,WINDOW_MASK=0,TASKFILE=cluster/trees_rfp163_h2test_nm_lines.txt,RESULTS_ROOT=$N,HPC_KW_RF_CONFIG=$CFG,REQUIRE_CPU=$REQ" \
      cluster/trees_rfprev_h2_task.sh)
echo "rs10=$A rs1=$B nm=$C (arch $ARCHES, cpu $REQ) $(date)" | tee "$STAMP"
qstat -u "$USER" | tail -6
