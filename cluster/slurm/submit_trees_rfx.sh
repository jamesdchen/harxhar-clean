#!/bin/bash
# rfx (2026-10-07): the per-bar 16:00 random forest with a dev-span configuration CFG (key token TOKEN, e.g. rf400 =
# the 400-trial search's final entry) on baseline / all_features / lean_vol_vix, CARC epyc-7513, single-threaded fits,
# every rung through the campaign's pack scripts with the shipped forest replaced:
#   FX  (fixed settings, masked)   T1 / T10   specs/causal_tune_trees_rfdev.py, cluster/slurm/trees_rfdev_pack.sbatch
#                                             -> results/linear_subsection_trees_mask_$TOKEN/{t1,t10}
#   RS  (random search, masked)    RS10 / RS1 specs/causal_tune_trees_tuned_rfp.py (TREE_SPEC = specs/causal_tune_trees_$TOKEN.py,
#                                             the grid re-centred on the forest), cluster/slurm/treestuned_rfp_pack.sbatch
#                                             -> results/linear_subsection_trees_tuned_mask_$TOKEN/{rs10,rs1}
#   NM  (fixed settings, UNMASKED) T1 / T10   cluster/slurm/trees_rfdev_nomask_pack.sbatch -> results/linear_subsection_trees_nomask_$TOKEN
# Gates: FX canary = pack 0 + one chunk re-run alone (bit for bit); the RS identity gate (TUNE_IDENTITY=1 through the
# tuned wrapper must reproduce the FX canary's chunks byte for byte, experiments/trees_rfp163_tuned_gate.py --stored);
# RS canaries; NM canary = pack 0 + alone.  Every fleet is held afterok on its canary (AFTER=<job ids>).
# Task files: cluster/trees_${TOKEN}_tasks_untuned.txt, cluster/trees_${TOKEN}_nomask_tasks_untuned.txt,
# cluster/trees_${TOKEN}_tasks_{rs10,rs1,rs_canary}.txt, cluster/trees_${TOKEN}_tuned_gate_lines.txt.
# Submit from a shell with NO conda env active.  Usage:
#   TOKEN=rf400 bash cluster/slurm/submit_trees_rfx.sh fx_canary | fx_fleet <packs> | gate | rs_canary | rs_fleet |
#                nm_canary | nm_fleet <packs> | merge <fx|tuned|nomask|all> [after job ids]
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} is active in this shell; submit from a clean one"; exit 1 ;; esac
TOKEN=${TOKEN:?TOKEN}
MODE="${1:-}"
CLASS=${CLASS:-epyc-7513}
PARTITION=${PARTITION:-main}   # e.g. main,largemem
CONSTRAINT=${CONSTRAINT:-$CLASS}   # e.g. "epyc-7513|epyc-7542" (the CPU classes shown byte-identical for the forest)
CFG=${CFG:-cluster/${TOKEN}_rf_config.json}
TREE_SPEC_NAME=causal_tune_trees_${TOKEN}.py
U=results/linear_subsection_trees_mask_$TOKEN
T=results/linear_subsection_trees_tuned_mask_$TOKEN
N=results/linear_subsection_trees_nomask_$TOKEN
G=results/trees_${TOKEN}_gates/tuned_identity_$CLASS
UT=cluster/trees_${TOKEN}_tasks_untuned.txt
NT=${NT_FILE:-cluster/trees_${TOKEN}_nomask_tasks_untuned.txt}   # NT_FILE: e.g. one chunk per "pack" (1-CPU backfill pieces)
RSC=cluster/trees_${TOKEN}_tasks_rs_canary.txt
GL=cluster/trees_${TOKEN}_tuned_gate_lines.txt
POOL_CPUS=${POOL_CPUS:-20}
TUNED_MEM=${TUNED_MEM:-48G}
TUNED_TIME=${TUNED_TIME:-4:00:00}
PACK_CPUS=${PACK_CPUS:-20}
PACK_MEM=${PACK_MEM:-96G}
PACK_TIME=${PACK_TIME:-2:00:00}
ALONE=${ALONE:-t1:lean_vol_vix:rf:0}
for f in specs/causal_tune_trees.py "specs/$TREE_SPEC_NAME" specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_trees_tuned_rfp.py specs/causal_tune_trees_rfdev.py specs/causal_tune_linear.py src/data/loading.py \
         experiments/trees_rfp163_tuned_gate.py experiments/reduce_trees_tuned_chunks.py experiments/trees_cadence_reduce_chunks.py \
         cluster/slurm/trees_rfdev_pack.sbatch cluster/slurm/treestuned_rfp_pack.sbatch cluster/slurm/trees_rfdev_nomask_pack.sbatch \
         cluster/slurm/trees_rfp163_gates.sbatch cluster/slurm/trees_rfx_merge.sbatch "$CFG" "$UT" "$NT" "$RSC" "$GL" \
         "cluster/trees_${TOKEN}_tasks_rs10.txt" "cluster/trees_${TOKEN}_tasks_rs1.txt"; do
  [ -f "$f" ] || { echo "$f missing: run TOKEN=$TOKEN cluster/slurm/ship_trees_rfx_carc.sh locally"; exit 1; }
done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
LOG=logs/submitted_trees_${TOKEN}_carc.txt
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
  local S="logs/submitted_trees_${TOKEN}_$1.txt"
  [ -f "$S" ] && { echo "already submitted ($(cat "$S")); move $S aside by hand to resubmit"; exit 1; }
  echo "$S"
}
P="${TOKEN:0:4}"   # job-name prefix
case "$MODE" in
  fx_canary)
    S=$(stamp fx_canary); mkdir -p "$U"; rm -f "$U/CANARY_OK"
    C=$(sbatch --parsable -J "${P}_fx_canary" --constraint="$CONSTRAINT" --partition="$PARTITION" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$U,WRITE_FLAG=$U/CANARY_OK,ALONE_CHECK=$ALONE,HPC_KW_RF_CONFIG=$CFG cluster/slurm/trees_rfdev_pack.sbatch)
    echo "$(date +%F_%T) fx_canary=$C (alone $ALONE)" | tee -a "$LOG" > "$S" ;;
  fx_fleet)
    PACKS=${2:?packs}; held "$U/CANARY_OK"; S=$(stamp "fx_fleet_$PACKS")
    F=$(sbatch --parsable -J "${P}_fx" "${HOLD[@]}" --constraint="$CONSTRAINT" --partition="$PARTITION" --array="$PACKS" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" \
         --time="$PACK_TIME" --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$U,CANARY_FLAG=$U/CANARY_OK,HPC_KW_RF_CONFIG=$CFG cluster/slurm/trees_rfdev_pack.sbatch)
    echo "$(date +%F_%T) fx_fleet=$F (packs $PACKS)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S" ;;
  gate)
    held "$U/CANARY_OK"; S=$(stamp "gate_$CLASS"); mkdir -p "$G"; rm -f "$G/GATE_OK"
    J=$(sbatch --parsable -J "${P}_gates" "${HOLD[@]}" --constraint="$CONSTRAINT" --partition="$PARTITION" \
         --export=ALL,OUT=$G,REF_ROOT=$PWD,LINES=$GL,HPC_KW_TREE_SPEC=$TREE_SPEC_NAME,GATE_STORED=linear_subsection_trees_mask_$TOKEN \
         cluster/slurm/trees_rfp163_gates.sbatch)
    echo "$(date +%F_%T) gate=$J (identity vs $U)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S" ;;
  rs_canary)
    held "$G/GATE_OK"; S=$(stamp rs_canary); mkdir -p "$T/rs10" "$T/rs1"; rm -f "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    NC=$(grep -c . "$RSC"); IDS=""
    for R in rs10 rs1; do
      J=$(sbatch --parsable -J "${P}_${R}_canary" "${HOLD[@]}" --constraint="$CONSTRAINT" --partition="$PARTITION" --array=1-"$NC" --cpus-per-task="$POOL_CPUS" \
           --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$RSC,RUNG=$R,RESULTS_ROOT=$T,WRITE_FLAG=$T/$R/CANARY_OK,HPC_KW_RF_CONFIG=$CFG,HPC_KW_TREE_SPEC=$TREE_SPEC_NAME \
           cluster/slurm/treestuned_rfp_pack.sbatch)
      IDS="$IDS canary_$R=$J"
    done
    echo "$(date +%F_%T)$IDS${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S" ;;
  rs_fleet)
    held "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"; S=$(stamp rs_fleet); IDS=""
    for R in rs10 rs1; do
      TF=cluster/trees_${TOKEN}_tasks_$R.txt
      J=$(sbatch --parsable -J "${P}_$R" "${HOLD[@]}" --constraint="$CONSTRAINT" --partition="$PARTITION" --array=1-"$(grep -c . "$TF")" --cpus-per-task="$POOL_CPUS" \
           --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$TF,RUNG=$R,RESULTS_ROOT=$T,HPC_KW_RF_CONFIG=$CFG,HPC_KW_TREE_SPEC=$TREE_SPEC_NAME \
           cluster/slurm/treestuned_rfp_pack.sbatch)
      IDS="$IDS fleet_$R=$J"
    done
    echo "$(date +%F_%T)$IDS${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S" ;;
  nm_canary)
    S=$(stamp nm_canary); mkdir -p "$N"; rm -f "$N/CANARY_OK"
    C=$(sbatch --parsable -J "${P}_nm_canary" --constraint="$CONSTRAINT" --partition="$PARTITION" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$NT,PACK=0,RESULTS_ROOT=$N,WRITE_FLAG=$N/CANARY_OK,ALONE_CHECK=$ALONE,HPC_KW_RF_CONFIG=$CFG cluster/slurm/trees_rfdev_nomask_pack.sbatch)
    echo "$(date +%F_%T) nm_canary=$C (alone $ALONE)" | tee -a "$LOG" > "$S" ;;
  nm_fleet)
    PACKS=${2:?packs}; held "$N/CANARY_OK"; S=$(stamp "nm_fleet_$PACKS")
    F=$(sbatch --parsable -J "${P}_nm" "${HOLD[@]}" --constraint="$CONSTRAINT" --partition="$PARTITION" --array="$PACKS" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" \
         --time="$PACK_TIME" --export=ALL,TASKFILE=$NT,RESULTS_ROOT=$N,CANARY_FLAG=$N/CANARY_OK,HPC_KW_RF_CONFIG=$CFG cluster/slurm/trees_rfdev_nomask_pack.sbatch)
    echo "$(date +%F_%T) nm_fleet=$F (packs $PACKS)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S" ;;
  merge)
    PART=${2:?fx, tuned, nomask or all}; A=${3:-}; S=$(stamp "merge_${PART}_${A//:/_}")
    DEP=(); [ -z "$A" ] || DEP=(--dependency="afterany:$A")
    M=$(sbatch --parsable -J "${P}_merge" "${DEP[@]}" --export=ALL,PART=$PART,TOKEN=$TOKEN cluster/slurm/trees_rfx_merge.sbatch)
    echo "$(date +%F_%T) merge_$PART=$M${A:+ (afterany $A)}" | tee -a "$LOG" > "$S" ;;
  *) echo "usage: TOKEN=<tok> submit_trees_rfx.sh fx_canary|fx_fleet <packs>|gate|rs_canary|rs_fleet|nm_canary|nm_fleet <packs>|merge <part> [after]"; exit 1 ;;
esac
squeue -u "$USER" -o "%.16i %.18j %.3t %.10M %.5C %R" | grep -E "JOBID|${P}_" | head -14
