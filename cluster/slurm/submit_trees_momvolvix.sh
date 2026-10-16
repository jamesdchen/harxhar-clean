#!/bin/bash
# moments_vol_vix tree workstream (2026-10-06, user request: every model on the moments + VIX input set
# beside baseline and all_features).  The masked 16:00 tree rungs on bucket moments_vol_vix with the
# campaign's UNEDITED machinery (cluster/slurm/submit_trees_mask.sh; writeup/CAMPAIGN_16H_2026-09-29.md):
#   T1    specs/causal_tune_trees.py        REFIT_EVERY 1,  IMPORTANCE_EVERY 10   (rung t1)
#   T10   specs/causal_tune_trees.py        REFIT_EVERY 10, IMPORTANCE_EVERY 1    (rung t10)
#   RS10  specs/causal_tune_trees_tuned.py  REFIT_EVERY 10, random search, TUNE_PER 250
#   RS1   specs/causal_tune_trees_tuned.py  REFIT_EVERY 1, TreeSHAP off
# WINDOW_MASK=1, SEGMENT bar1600, TRAIN_WIN 2000, single-threaded fits, every task pinned to the
# masked campaign's CPU class (CLASS, default epyc-7513), through the campaign's own pack scripts
# cluster/slurm/trees_mask_pack.sbatch (untuned) and treestuned_mask_pack.sbatch (tuned).
# Deployment root /scratch1/jc_905/harxhar-momvolvix-trees (cluster/slurm/ship_trees_momvolvix_carc.sh); new
# result roots (no earlier root or flag is touched):
#   U = results/linear_subsection_trees_mask_momvolvix        t1/, t10/
#   T = results/linear_subsection_trees_tuned_mask_momvolvix  rs10/, rs1/
#   G = results/trees_momvix_gates/carc_<CLASS>            the reproduction gate
# Stages, each submitted after the previous one was read:
#   bash cluster/slurm/submit_trees_momvolvix.sh gates          re-run stored all_features cells of every
#        rung and model (cluster/trees_momvix_gate_lines.txt) against the masked campaign's chunk npz
#        (REF_ROOT, default /scratch1/jc_905/harxhar-mask), byte for byte
#   bash cluster/slurm/submit_trees_momvolvix.sh canary         refuses unless $G/GATE_OK; untuned pack 0
#        (chunk c0 of the 6 T1 / T10 arms + the T1 LightGBM c0 re-run alone, compared bit for bit) and
#        the tuned canaries (chunk 0 of the three models, RS10 and RS1)
#   bash cluster/slurm/submit_trees_momvolvix.sh fleet_untuned <packs>   refuses unless $U/CANARY_OK;
#        an array over the given packs of cluster/trees_momvolvix_tasks_untuned.txt (e.g. 1-2)
#   bash cluster/slurm/submit_trees_momvolvix.sh fleet_tuned <rs10|rs1> <lines>   refuses unless the
#        rung's CANARY_OK; an array over the given lines of cluster/trees_momvolvix_tasks_<rung>.txt
#   bash cluster/slurm/submit_trees_momvolvix.sh merge <untuned|tuned|all> [job ids, colon-separated]
#        cluster/slurm/trees_momvolvix_merge.sbatch, held afterany on the given jobs
# HELD AFTEROK (speed-up, main session 2026-10-06): with AFTER=<job ids, colon-separated> a stage may be
# submitted before the flag it needs exists -- canary held afterok on the gate job (the gate exits
# non-zero on any difference, so the canary never runs on a failed gate), a fleet held afterok on its
# canary job(s) (the pack scripts also refuse to run without the canary flag).
# A fleet / merge stamp (logs/submitted_trees_momvolvix_<mode>_<arg>.txt) refuses a second identical
# submission (jobs cannot be cancelled from here); chunks with DONE are skipped by the pack scripts.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
MODE="${1:-}"
U=results/linear_subsection_trees_mask_momvolvix
T=results/linear_subsection_trees_tuned_mask_momvolvix
CLASS=${CLASS:-epyc-7513}
G=results/trees_momvix_gates/carc_$CLASS
REF_ROOT=${REF_ROOT:-/scratch1/jc_905/harxhar-mask}
UT=cluster/trees_momvolvix_tasks_untuned.txt
RS_CANARY=cluster/trees_momvolvix_tasks_rs_canary.txt
PACK_CPUS=${PACK_CPUS:-20}
PACK_MEM=${PACK_MEM:-80G}
PACK_TIME=${PACK_TIME:-1:00:00}
POOL_CPUS=${POOL_CPUS:-20}
TUNED_MEM=${TUNED_MEM:-32G}
TUNED_TIME=${TUNED_TIME:-1:00:00}
for f in specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_linear.py src/models/window_mask.py src/data/loading.py \
         experiments/trees_momvix_gate.py experiments/trees_cadence_reduce_chunks.py \
         experiments/reduce_trees_tuned_chunks.py experiments/trees_mask_kept_counts.py \
         cluster/slurm/trees_mask_pack.sbatch cluster/slurm/treestuned_mask_pack.sbatch \
         cluster/slurm/trees_momvix_gates.sbatch cluster/slurm/trees_momvolvix_merge.sbatch \
         "$UT" "$RS_CANARY" cluster/trees_momvolvix_tasks_rs10.txt cluster/trees_momvolvix_tasks_rs1.txt \
         cluster/trees_momvolvix_tasks_none.txt cluster/trees_momvix_gate_lines.txt; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_trees_momvolvix_carc.sh, locally)"; exit 1; }
done
for B in moments_vol_vix lean_vol_vix; do grep -q "\"$B\"" src/data/loading.py || { echo "src/data/loading.py has no $B bucket"; exit 1; }; done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
SUBMIT=sbatch
LOG=logs/submitted_trees_momvolvix_carc.txt
need() { for f in "$@"; do [ -f "$f" ] || { echo "$f missing: not passed yet"; exit 1; }; done; }
AFTER=${AFTER:-}
HOLD=()
held() {  # $@ = the flags a stage needs: present -> no hold; absent -> held afterok on AFTER, else refuse
  local f miss=0
  for f in "$@"; do [ -f "$f" ] || miss=1; done
  if [ "$miss" = 1 ]; then
    [ -n "$AFTER" ] || { echo "$* not all present and no AFTER job ids given; refusing"; exit 1; }
    HOLD=(--dependency="afterok:$AFTER")
    echo "held afterok:$AFTER (flags not yet present: $*)"
  fi
}
stamp() {  # refuse a second identical submission
  local S="logs/submitted_trees_momvolvix_$1.txt"
  [ -f "$S" ] && { echo "already submitted ($(cat "$S")); delete $S by hand to resubmit (DONE chunks are skipped)"; exit 1; }
  echo "$S"
}

case "$MODE" in
  gates)
    S=$(stamp "gates_$CLASS")
    mkdir -p "$G"
    rm -f "$G/GATE_OK" "$G"/PASS_*
    J=$($SUBMIT --parsable -J tmvv_gates --constraint="$CLASS" --partition=main \
         --export=ALL,OUT=$G,REF_ROOT=$REF_ROOT cluster/slurm/trees_momvix_gates.sbatch)
    echo "$(date +%F_%T) gates_$CLASS=$J (ref $REF_ROOT)" | tee -a "$LOG" > "$S"
    ;;
  canary)
    held "$G/GATE_OK"
    S=$(stamp canary)
    mkdir -p "$U" "$T/rs10" "$T/rs1"
    rm -f "$U/CANARY_OK" "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    CU=$($SUBMIT --parsable -J tmvv_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$U,WRITE_FLAG=$U/CANARY_OK,ALONE_CHECK=t1:moments_vol_vix:lgbm:0 \
         cluster/slurm/trees_mask_pack.sbatch)
    C10=$($SUBMIT --parsable -J tmvv_rs10_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --array=1-"$(grep -c . "$RS_CANARY")" --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
          --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs10,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs10/CANARY_OK \
          cluster/slurm/treestuned_mask_pack.sbatch)
    C1=$($SUBMIT --parsable -J tmvv_rs1_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --array=1-"$(grep -c . "$RS_CANARY")" --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
         --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs1,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs1/CANARY_OK \
         cluster/slurm/treestuned_mask_pack.sbatch)
    echo "$(date +%F_%T) canary_untuned=$CU canary_rs10=$C10 canary_rs1=$C1${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet_untuned)
    PACKS=${2:?packs, e.g. 1-2}
    held "$U/CANARY_OK"
    S=$(stamp "fleet_untuned_$PACKS")
    FU=$($SUBMIT --parsable -J tmvv_untuned "${HOLD[@]}" --constraint="$CLASS" --partition=main --array="$PACKS" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$U,CANARY_FLAG=$U/CANARY_OK cluster/slurm/trees_mask_pack.sbatch)
    echo "$(date +%F_%T) fleet_untuned=$FU (packs $PACKS)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet_tuned)
    R=${2:?rung rs10 or rs1}
    LINES=${3:?lines, e.g. 1-18}
    case "$R" in rs10|rs1) ;; *) echo "rung must be rs10 or rs1"; exit 1 ;; esac
    held "$T/$R/CANARY_OK"
    S=$(stamp "fleet_${R}_$LINES")
    F=$($SUBMIT --parsable -J "tmvv_$R" "${HOLD[@]}" --constraint="$CLASS" --partition=main --array="$LINES" --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
        --export=ALL,TASKFILE=cluster/trees_momvolvix_tasks_$R.txt,RUNG=$R,RESULTS_ROOT=$T,CANARY_FLAG=$T/$R/CANARY_OK \
        cluster/slurm/treestuned_mask_pack.sbatch)
    echo "$(date +%F_%T) fleet_$R=$F (lines $LINES)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  merge)
    PART=${2:?untuned, tuned or all}
    AFTER=${3:-}
    S=$(stamp "merge_${PART}_${AFTER//:/_}")
    DEP=()
    [ -z "$AFTER" ] || DEP=(--dependency="afterany:$AFTER")
    M=$($SUBMIT --parsable -J tmvv_merge "${DEP[@]}" --export=ALL,PART=$PART cluster/slurm/trees_momvolvix_merge.sbatch)
    echo "$(date +%F_%T) merge_$PART=$M${AFTER:+ (afterany $AFTER)}" | tee -a "$LOG" > "$S"
    ;;
  *)
    echo "usage: bash cluster/slurm/submit_trees_momvolvix.sh gates|canary|fleet_untuned <packs>|fleet_tuned <rung> <lines>|merge <part> [after]"
    exit 1
    ;;
esac
cat "$LOG" | tail -3
squeue -u "$USER" -o "%.16i %.18j %.3t %.10M %.5C %R" | grep -E "JOBID|tmvv_" | head -12
