#!/bin/bash
# rfp163 (2026-10-07, SPECULATIVE: the dev-span search's trial 163 as the forest's shipped configuration; user "max
# capacity"): the per-bar 16:00 random forest's random-search rungs re-centred on the new forest and its UNMASKED
# fixed-settings rungs, on baseline / all_features / lean_vol_vix, CARC epyc-7513, single-threaded fits.
#   RS10 / RS1   specs/causal_tune_trees_tuned_rfp.py (cluster/slurm/treestuned_rfp_pack.sbatch: REFIT_EVERY 10 with
#                TreeSHAP / 1 without; random search, 32 candidates, TUNE_PER 250, QSEL=1 -> MSE and QLIKE rules)
#                -> results/linear_subsection_trees_tuned_mask_rfp163/{rs10,rs1}
#   T10 / T1 unmasked  specs/causal_tune_trees_rfdev.py with WINDOW_MASK=0 (cluster/slurm/trees_rfdev_nomask_pack.sbatch)
#                -> results/linear_subsection_trees_nomask_rfp163/{t10,t1}
# Deployment root /scratch1/jc_905/harxhar-rfp163-tuned (cluster/slurm/ship_trees_rfp163_carc.sh).  Submit from a shell
# with NO conda env active (sbatch --export=ALL would hand it to the jobs).
#   bash cluster/slurm/submit_trees_rfp163.sh gate          the identity gate (TUNE_IDENTITY=1 through the wrapper must
#        reproduce the trial-163 fixed-settings preview chunks of REF_ROOT byte for byte)
#   AFTER=<gate job> bash cluster/slurm/submit_trees_rfp163.sh canary_rs    chunk 0 of each set, RS10 and RS1
#   AFTER=<rs canaries, colon-separated> bash cluster/slurm/submit_trees_rfp163.sh fleet_rs
#   bash cluster/slurm/submit_trees_rfp163.sh canary_nm    unmasked pack 0 + one chunk re-run alone (bit for bit)
#   AFTER=<nm canary> bash cluster/slurm/submit_trees_rfp163.sh fleet_nm <packs>
#   bash cluster/slurm/submit_trees_rfp163.sh merge <tuned|nomask|all> [after job ids]
# Stamps (logs/submitted_trees_rfp163_<mode>*.txt) refuse a second identical submission.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} is active in this shell; submit from a clean one"; exit 1 ;; esac
MODE="${1:-}"
CLASS=${CLASS:-epyc-7513}
T=results/linear_subsection_trees_tuned_mask_rfp163
N=results/linear_subsection_trees_nomask_rfp163
G=results/trees_rfp163_gates/tuned_identity_$CLASS
REF_ROOT=${REF_ROOT:-/scratch1/jc_905/harxhar-rfprev-trees}
CFG=cluster/rfprev163_rf_config.json
NT=cluster/trees_rfp163_nomask_tasks_untuned.txt
RSC=cluster/trees_rfp163_tasks_rs_canary.txt
POOL_CPUS=${POOL_CPUS:-20}
TUNED_MEM=${TUNED_MEM:-48G}
TUNED_TIME=${TUNED_TIME:-3:00:00}
PACK_CPUS=${PACK_CPUS:-20}
PACK_MEM=${PACK_MEM:-96G}
PACK_TIME=${PACK_TIME:-2:00:00}
ALONE=${ALONE:-t1:lean_vol_vix:rf:0}
for f in specs/causal_tune_trees.py specs/causal_tune_trees_rfp163.py specs/causal_tune_trees_tuned.py \
         specs/causal_tune_trees_tuned_jobs.py specs/causal_tune_trees_tuned_rfp.py specs/causal_tune_trees_rfdev.py \
         specs/causal_tune_linear.py src/data/loading.py experiments/trees_rfp163_tuned_gate.py \
         experiments/reduce_trees_tuned_chunks.py experiments/trees_cadence_reduce_chunks.py \
         cluster/slurm/treestuned_rfp_pack.sbatch cluster/slurm/trees_rfdev_nomask_pack.sbatch \
         cluster/slurm/trees_rfp163_gates.sbatch cluster/slurm/trees_rfp163_merge.sbatch "$CFG" "$NT" "$RSC" \
         cluster/trees_rfp163_tasks_rs10.txt cluster/trees_rfp163_tasks_rs1.txt cluster/trees_rfp163_tuned_gate_lines.txt; do
  [ -f "$f" ] || { echo "$f missing: run cluster/slurm/ship_trees_rfp163_carc.sh locally"; exit 1; }
done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
LOG=logs/submitted_trees_rfp163_carc.txt
AFTER=${AFTER:-}
HOLD=()
held() {
  local f miss=0
  for f in "$@"; do [ -f "$f" ] || miss=1; done
  if [ "$miss" = 1 ]; then
    [ -n "$AFTER" ] || { echo "$* not all present and no AFTER job ids given; refusing"; exit 1; }
    HOLD=(--dependency="afterok:$AFTER")
    echo "held afterok:$AFTER (flags not yet present: $*)"
  fi
}
stamp() {
  local S="logs/submitted_trees_rfp163_$1.txt"
  [ -f "$S" ] && { echo "already submitted ($(cat "$S")); move $S aside by hand to resubmit"; exit 1; }
  echo "$S"
}
case "$MODE" in
  gate)
    S=$(stamp "gate_$CLASS")
    mkdir -p "$G"
    rm -f "$G/GATE_OK"
    J=$(sbatch --parsable -J trq_gates --constraint="$CLASS" --partition=main --export=ALL,OUT=$G,REF_ROOT=$REF_ROOT \
         cluster/slurm/trees_rfp163_gates.sbatch)
    echo "$(date +%F_%T) gate=$J (ref $REF_ROOT)" | tee -a "$LOG" > "$S"
    ;;
  canary_rs)
    held "$G/GATE_OK"
    S=$(stamp canary_rs)
    mkdir -p "$T/rs10" "$T/rs1"
    rm -f "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    NC=$(grep -c . "$RSC")
    C10=$(sbatch --parsable -J trq_rs10_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --array=1-"$NC" --cpus-per-task="$POOL_CPUS" \
          --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$RSC,RUNG=rs10,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs10/CANARY_OK,HPC_KW_RF_CONFIG=$CFG \
          cluster/slurm/treestuned_rfp_pack.sbatch)
    C1=$(sbatch --parsable -J trq_rs1_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --array=1-"$NC" --cpus-per-task="$POOL_CPUS" \
         --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$RSC,RUNG=rs1,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs1/CANARY_OK,HPC_KW_RF_CONFIG=$CFG \
         cluster/slurm/treestuned_rfp_pack.sbatch)
    echo "$(date +%F_%T) canary_rs10=$C10 canary_rs1=$C1${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet_rs)
    held "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    S=$(stamp fleet_rs)
    F=""
    for R in rs10 rs1; do
      TF=cluster/trees_rfp163_tasks_$R.txt
      J=$(sbatch --parsable -J "trq_$R" "${HOLD[@]}" --constraint="$CLASS" --partition=main --array=1-"$(grep -c . "$TF")" --cpus-per-task="$POOL_CPUS" \
           --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$TF,RUNG=$R,RESULTS_ROOT=$T,HPC_KW_RF_CONFIG=$CFG \
           cluster/slurm/treestuned_rfp_pack.sbatch)
      F="$F fleet_$R=$J"
    done
    echo "$(date +%F_%T)$F${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  canary_nm)
    S=$(stamp canary_nm)
    mkdir -p "$N"
    rm -f "$N/CANARY_OK"
    C=$(sbatch --parsable -J trq_nm_canary --constraint="$CLASS" --partition=main --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$NT,PACK=0,RESULTS_ROOT=$N,WRITE_FLAG=$N/CANARY_OK,ALONE_CHECK=$ALONE,HPC_KW_RF_CONFIG=$CFG \
         cluster/slurm/trees_rfdev_nomask_pack.sbatch)
    echo "$(date +%F_%T) canary_nm=$C (alone $ALONE)" | tee -a "$LOG" > "$S"
    ;;
  fleet_nm)
    PACKS=${2:?packs}
    held "$N/CANARY_OK"
    S=$(stamp "fleet_nm_$PACKS")
    F=$(sbatch --parsable -J trq_nm "${HOLD[@]}" --constraint="$CLASS" --partition=main --array="$PACKS" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" \
         --time="$PACK_TIME" --export=ALL,TASKFILE=$NT,RESULTS_ROOT=$N,CANARY_FLAG=$N/CANARY_OK,HPC_KW_RF_CONFIG=$CFG \
         cluster/slurm/trees_rfdev_nomask_pack.sbatch)
    echo "$(date +%F_%T) fleet_nm=$F (packs $PACKS)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  merge)
    PART=${2:?tuned, nomask or all}
    A=${3:-}
    S=$(stamp "merge_${PART}_${A//:/_}")
    DEP=()
    [ -z "$A" ] || DEP=(--dependency="afterany:$A")
    M=$(sbatch --parsable -J trq_merge "${DEP[@]}" --export=ALL,PART=$PART cluster/slurm/trees_rfp163_merge.sbatch)
    echo "$(date +%F_%T) merge_$PART=$M${A:+ (afterany $A)}" | tee -a "$LOG" > "$S"
    ;;
  *) echo "usage: submit_trees_rfp163.sh gate|canary_rs|fleet_rs|canary_nm|fleet_nm <packs>|merge <part> [after]"; exit 1 ;;
esac
squeue -u "$USER" -o "%.16i %.16j %.3t %.10M %.5C %R" | grep -E "JOBID|trq_" | head -12
