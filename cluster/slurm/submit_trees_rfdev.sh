#!/bin/bash
# rfdev (user decision 2026-10-06): the per-bar 16:00 random forest with the dev-span configuration
# (results/rf_devmenu_2026-10-06/rf_config.json, shipped as cluster/rfdev_rf_config.json) on baseline,
# all_features and lean_vol_vix, rungs T1 (refit every session, IMPORTANCE_EVERY 10) and T10 (refit every 10,
# IMPORTANCE_EVERY 1), everything else exactly the masked campaign's shipped-configuration rungs (WINDOW_MASK=1,
# bar1600, TRAIN_WIN 2000, single-threaded fits, epyc-7513), through specs/causal_tune_trees_rfdev.py (the
# shipped spec with RF_PARAMS read from HPC_KW_RF_CONFIG) and cluster/slurm/trees_rfdev_pack.sbatch.
# Deployment root /scratch1/jc_905/harxhar-rfdev-trees (cluster/slurm/ship_trees_rfdev_carc.sh); result root
# U = results/linear_subsection_trees_mask_rfdev (t1/, t10/), gate G = results/trees_rfdev_gates/carc_<CLASS>.
# Submit from a shell WITHOUT a conda env active (an inherited activation makes the job's python the
# module's, not the env's: the 2026-10-06 18:17 canary failure).
#   bash cluster/slurm/submit_trees_rfdev.sh gates          the stored masked random-forest cells of
#        cluster/trees_rfdev_gate_lines.txt re-run through the wrapper with the SHIPPED configuration
#        (cluster/rfdev_shipped_rf.json), compared byte for byte with REF_ROOT's chunks
#   AFTER=<gate job> bash cluster/slurm/submit_trees_rfdev.sh canary   pack 0 (chunk c0 of the 6 arms) with
#        the dev configuration + one chunk re-run alone (bit for bit); held afterok on the gate job
#   AFTER=<canary job> bash cluster/slurm/submit_trees_rfdev.sh fleet <packs>
#   bash cluster/slurm/submit_trees_rfdev.sh merge [job ids, colon-separated]
# Stamps (logs/submitted_trees_rfdev_<mode>*.txt) refuse a second identical submission.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
MODE="${1:-}"
U=results/linear_subsection_trees_mask_rfdev
CLASS=${CLASS:-epyc-7513}
G=results/trees_rfdev_gates/carc_$CLASS
REF_ROOT=${REF_ROOT:-/scratch1/jc_905/harxhar-mask}
UT=cluster/trees_rfdev_tasks_untuned.txt
CFG=cluster/rfdev_rf_config.json
SHIPPED=cluster/rfdev_shipped_rf.json
ALONE=${ALONE:-t1:lean_vol_vix:rf:0}
PACK_CPUS=${PACK_CPUS:-20}
PACK_MEM=${PACK_MEM:-80G}
PACK_TIME=${PACK_TIME:-2:00:00}
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} is active in this shell; submit from a clean one"; exit 1 ;; esac
for f in specs/causal_tune_trees.py specs/causal_tune_trees_rfdev.py src/models/window_mask.py src/data/loading.py \
         experiments/trees_momvix_gate.py experiments/trees_rfdev_gate.py experiments/trees_cadence_reduce_chunks.py \
         experiments/reduce_trees_tuned_chunks.py experiments/trees_mask_kept_counts.py \
         cluster/slurm/trees_rfdev_pack.sbatch cluster/slurm/trees_rfdev_gates.sbatch cluster/slurm/trees_rfdev_merge.sbatch \
         cluster/trees_rfdev_gate_lines.txt cluster/trees_rfdev_tasks_none.txt "$SHIPPED"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_trees_rfdev_carc.sh, locally)"; exit 1; }
done
grep -q '"lean_vol_vix"' src/data/loading.py || { echo "src/data/loading.py has no lean_vol_vix bucket"; exit 1; }
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
LOG=logs/submitted_trees_rfdev_carc.txt
AFTER=${AFTER:-}
HOLD=()
held() {  # $@ = flags a stage needs: present -> no hold; absent -> held afterok on AFTER, else refuse
  local f miss=0
  for f in "$@"; do [ -f "$f" ] || miss=1; done
  if [ "$miss" = 1 ]; then
    [ -n "$AFTER" ] || { echo "$* not all present and no AFTER job ids given; refusing"; exit 1; }
    HOLD=(--dependency="afterok:$AFTER")
    echo "held afterok:$AFTER (flags not yet present: $*)"
  fi
}
stamp() {
  local S="logs/submitted_trees_rfdev_$1.txt"
  [ -f "$S" ] && { echo "already submitted ($(cat "$S")); move $S aside by hand to resubmit (DONE chunks are skipped)"; exit 1; }
  echo "$S"
}
case "$MODE" in
  gates)
    S=$(stamp "gates_$CLASS")
    mkdir -p "$G"
    rm -f "$G/GATE_OK" "$G"/PASS_*
    J=$(sbatch --parsable -J trf_gates --constraint="$CLASS" --partition=main \
         --export=ALL,OUT=$G,REF_ROOT=$REF_ROOT,HPC_KW_RF_CONFIG=$SHIPPED cluster/slurm/trees_rfdev_gates.sbatch)
    echo "$(date +%F_%T) gates_$CLASS=$J (ref $REF_ROOT, wrapper with $SHIPPED)" | tee -a "$LOG" > "$S"
    ;;
  canary)
    [ -f "$CFG" ] && [ -f "$UT" ] || { echo "$CFG or $UT missing"; exit 1; }
    held "$G/GATE_OK"
    S=$(stamp canary)
    mkdir -p "$U"
    rm -f "$U/CANARY_OK"
    C=$(sbatch --parsable -J trf_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$U,WRITE_FLAG=$U/CANARY_OK,ALONE_CHECK=$ALONE,HPC_KW_RF_CONFIG=$CFG \
         cluster/slurm/trees_rfdev_pack.sbatch)
    echo "$(date +%F_%T) canary=$C (alone $ALONE)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet)
    PACKS=${2:?packs, e.g. 1-6}
    held "$U/CANARY_OK"
    S=$(stamp "fleet_$PACKS")
    F=$(sbatch --parsable -J trf_fleet "${HOLD[@]}" --constraint="$CLASS" --partition=main --array="$PACKS" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$U,CANARY_FLAG=$U/CANARY_OK,HPC_KW_RF_CONFIG=$CFG cluster/slurm/trees_rfdev_pack.sbatch)
    echo "$(date +%F_%T) fleet=$F (packs $PACKS)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  merge)
    A=${2:-}
    S=$(stamp "merge_${A//:/_}")
    DEP=()
    [ -z "$A" ] || DEP=(--dependency="afterany:$A")
    M=$(sbatch --parsable -J trf_merge "${DEP[@]}" cluster/slurm/trees_rfdev_merge.sbatch)
    echo "$(date +%F_%T) merge=$M${A:+ (afterany $A)}" | tee -a "$LOG" > "$S"
    ;;
  *) echo "usage: bash cluster/slurm/submit_trees_rfdev.sh gates|canary|fleet <packs>|merge [after]"; exit 1 ;;
esac
squeue -u "$USER" -o "%.16i %.14j %.3t %.10M %.5C %R" | grep -E "JOBID|trf_" | head -10
