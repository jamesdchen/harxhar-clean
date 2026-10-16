#!/bin/bash
# xgbtop (user decision 2026-10-07): the per-bar 16:00 XGBoost with the TOP 1998-2002 dev-span entry xgb_opt_00
# (cluster/xgbtop_xgb_config.json; tree_expert_10, sha 88bec9f385c7) instead of xgb_opt_06 -- on baseline, all_features
# and lean_vol_vix, CARC, single-threaded fits, deployment root /scratch1/jc_905/harxhar-xgbtop
# (cluster/slurm/ship_trees_xgbtop_carc.sh).  Fixed settings: the GENERATED spec specs/causal_tune_trees_xgbtop.py
# (experiments/trees_xgbtop_make_specs.py: specs/causal_tune_trees.py with the XGB_PARAMS block replaced) through
# cluster/slurm/trees_xgbtop_pack.sbatch, rungs T1 (refit every session, IMPORTANCE_EVERY 10) and T10 (refit every 10,
# IMPORTANCE_EVERY 1), WINDOW_MASK=1 (the masked campaign's rungs) and, as a twin, WINDOW_MASK=0.  Result roots:
#   U = results/linear_subsection_trees_mask_xgbtop        t1/, t10/    (masked; the paper's per-bar cells)
#   N = results/linear_subsection_trees_nomask_xgbtop      t1/, t10/    (unmasked twin)
#   X = results/linear_subsection_trees_mask_xgbtop_xcpu/<class>        (the cross-CPU cell)
#   G = results/trees_xgbtop_gates/carc_<CLASS>                          (the reproduction gate)
# Random-search rungs RS10 / RS1 (re-centred on xgb_opt_00): see the rs modes below.  Optuna: submit_trees_xgbtop_optuna.sh.
# Submit from a shell WITHOUT a conda env active (sbatch --export=ALL would hand it to the jobs).
#   bash cluster/slurm/submit_trees_xgbtop.sh gates
#        the stored masked XGBoost cells of cluster/trees_xgbtop_gate_lines.txt re-run through the generator's spec of the
#        SHIPPED configuration (specs/causal_tune_trees_xgb06.py), compared byte for byte with the stored chunks and the
#        published table cells
#   AFTER=<gate job> bash cluster/slurm/submit_trees_xgbtop.sh canary    pack 0 (chunk c0 of the 6 arms) + one chunk re-run
#        alone (bit for bit); held afterok on the gate job; writes U/CANARY_OK
#   AFTER=<canary job> bash cluster/slurm/submit_trees_xgbtop.sh fleet <packs>
#   CLASS=<epyc-7542|xeon-4116> bash cluster/slurm/submit_trees_xgbtop.sh xcpu   the cross-CPU cell (every chunk of the
#        T1 all_features arm, cluster/trees_xgbtop_tasks_xcpu.txt, one pack) on that class
#   AFTER=<gate job> bash cluster/slurm/submit_trees_xgbtop.sh canary_nm / AFTER=<canary_nm> ... fleet_nm <packs>    the
#        unmasked twin (WINDOW_MASK=0)
#   bash cluster/slurm/submit_trees_xgbtop.sh merge <fixed|nomask|xcpu> [job ids, colon-separated]
#   --- random search (phase 2): specs/causal_tune_trees_tuned_xgbtop.py through cluster/slurm/treestuned_xgbtop_pack.sbatch
#   AFTER=<phase-1 canary> bash cluster/slurm/submit_trees_xgbtop.sh gate_rs   TUNE_IDENTITY=1 must reproduce the fixed
#        cells of U (the phase-1 canary's chunks c0) byte for byte
#   AFTER=<gate_rs job> bash cluster/slurm/submit_trees_xgbtop.sh canary_rs     chunk 0 of each set, RS10 and RS1
#   AFTER=<rs canaries> bash cluster/slurm/submit_trees_xgbtop.sh fleet_rs
#   bash cluster/slurm/submit_trees_xgbtop.sh merge_rs [job ids]
# CLASS (default epyc-7513) pins every task; CLASSES (e.g. "epyc-7513|epyc-7542") may widen a FLEET once the
# cross-CPU cell matched byte for byte.  Stamps (logs/submitted_trees_xgbtop_<mode>*.txt) refuse a second identical
# submission.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} is active in this shell; submit from a clean one"; exit 1 ;; esac
[ -z "${CONDA_DEFAULT_ENV:-}" ] || { echo "conda env ${CONDA_DEFAULT_ENV} is active in this shell; submit from a clean one"; exit 1; }
MODE="${1:-}"
U=results/linear_subsection_trees_mask_xgbtop
N=results/linear_subsection_trees_nomask_xgbtop
X=results/linear_subsection_trees_mask_xgbtop_xcpu
T=results/linear_subsection_trees_tuned_mask_xgbtop
CLASS=${CLASS:-epyc-7513}
CLASSES=${CLASSES:-$CLASS}
G=results/trees_xgbtop_gates/carc_$CLASS
GRS=results/trees_xgbtop_gates/tuned_identity_$CLASS
UT=cluster/trees_xgbtop_tasks_untuned.txt
XT=cluster/trees_xgbtop_tasks_xcpu.txt
RSC=cluster/trees_xgbtop_tasks_rs_canary.txt
CFG=cluster/xgbtop_xgb_config.json
ALONE=${ALONE:-t1:lean_vol_vix:xgb:0}
PACK_CPUS=${PACK_CPUS:-20}
PACK_MEM=${PACK_MEM:-80G}
PACK_TIME=${PACK_TIME:-2:00:00}
POOL_CPUS=${POOL_CPUS:-20}
TUNED_MEM=${TUNED_MEM:-48G}
TUNED_TIME=${TUNED_TIME:-4:00:00}
for f in specs/causal_tune_trees.py specs/causal_tune_trees_xgbtop.py specs/causal_tune_trees_xgb06.py src/models/window_mask.py \
         src/data/loading.py experiments/trees_momvix_gate.py experiments/trees_xgbtop_gate.py experiments/trees_xgbtop_make_specs.py \
         experiments/trees_cadence_reduce_chunks.py experiments/trees_mask_kept_counts.py experiments/trees_xgbtop_xcpu_compare.py \
         cluster/slurm/trees_xgbtop_pack.sbatch cluster/slurm/trees_xgbtop_gates.sbatch cluster/slurm/trees_xgbtop_merge.sbatch \
         cluster/trees_xgbtop_gate_lines.txt cluster/trees_rfdev_tasks_none.txt "$UT" "$XT" "$CFG" cluster/xgbtop_shipped_xgb.json; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_trees_xgbtop_carc.sh, locally)"; exit 1; }
done
grep -q '"lean_vol_vix"' src/data/loading.py || { echo "src/data/loading.py has no lean_vol_vix bucket"; exit 1; }
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
LOG=logs/submitted_trees_xgbtop_carc.txt
AFTER=${AFTER:-}
HOLD=()
held() {  # $@ = flags a stage needs: present -> no hold; absent -> held afterok on AFTER, else refuse
  local f miss=0
  for f in "$@"; do [ -f "$f" ] || miss=1; done
  if [ "$miss" = 1 ]; then
    [ -n "$AFTER" ] || { echo "$* not all present and no AFTER job ids given; refusing"; exit 1; }
    HOLD=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)
    echo "held afterok:$AFTER (flags not yet present: $*)"
  fi
}
stamp() {
  local S="logs/submitted_trees_xgbtop_$1.txt"
  [ -f "$S" ] && { echo "already submitted ($(cat "$S")); move $S aside by hand to resubmit (DONE chunks are skipped)"; exit 1; }
  echo "$S"
}
need_tuned() {
  for f in specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py specs/causal_tune_trees_tuned_xgbtop.py \
           specs/causal_tune_linear.py experiments/reduce_trees_tuned_chunks.py experiments/trees_xgbtop_tuned_gate.py \
           cluster/slurm/treestuned_xgbtop_pack.sbatch cluster/slurm/trees_xgbtop_tuned_gates.sbatch \
           cluster/slurm/trees_xgbtop_tuned_merge.sbatch cluster/trees_xgbtop_tuned_gate_lines.txt "$RSC" \
           cluster/trees_xgbtop_tasks_rs10.txt cluster/trees_xgbtop_tasks_rs1.txt; do
    [ -f "$f" ] || { echo "$f missing: ship the phase-2 files first"; exit 1; }
  done
}
case "$MODE" in
  gates)
    S=$(stamp "gates_$CLASS")
    mkdir -p "$G"
    rm -f "$G/GATE_OK"
    J=$(sbatch --parsable -J txt_gates --constraint="$CLASS" --partition=main --export=ALL,OUT=$G cluster/slurm/trees_xgbtop_gates.sbatch)
    echo "$(date +%F_%T) gates_$CLASS=$J" | tee -a "$LOG" > "$S"
    ;;
  canary)
    held "$G/GATE_OK"
    S=$(stamp canary)
    mkdir -p "$U"
    rm -f "$U/CANARY_OK"
    C=$(sbatch --parsable -J txt_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$U,MASK=1,WRITE_FLAG=$U/CANARY_OK,ALONE_CHECK=$ALONE,TOKEN=xgbtop,XGB_CONFIG=$CFG \
         cluster/slurm/trees_xgbtop_pack.sbatch)
    echo "$(date +%F_%T) canary=$C (alone $ALONE)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet)
    PACKS=${2:?packs, e.g. 1-6}
    held "$U/CANARY_OK"
    S=$(stamp "fleet_$PACKS")
    F=$(sbatch --parsable -J txt_fleet "${HOLD[@]}" --constraint="$CLASSES" --partition=main --array="$PACKS" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$U,MASK=1,CANARY_FLAG=$U/CANARY_OK,TOKEN=xgbtop,XGB_CONFIG=$CFG cluster/slurm/trees_xgbtop_pack.sbatch)
    echo "$(date +%F_%T) fleet=$F (packs $PACKS, $CLASSES)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  xcpu)
    [ "$CLASS" != epyc-7513 ] || { echo "xcpu: CLASS must be another class (epyc-7542, xeon-4116)"; exit 1; }
    S=$(stamp "xcpu_$CLASS")
    mkdir -p "$X/$CLASS"
    J=$(sbatch --parsable -J "txt_xcpu_$CLASS" --constraint="$CLASS" --partition=main --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$XT,PACK=0,RESULTS_ROOT=$X/$CLASS,MASK=1,WRITE_FLAG=$X/$CLASS/XCPU_RUN_OK,TOKEN=xgbtop,XGB_CONFIG=$CFG \
         cluster/slurm/trees_xgbtop_pack.sbatch)
    echo "$(date +%F_%T) xcpu_$CLASS=$J ($XT)" | tee -a "$LOG" > "$S"
    ;;
  canary_nm)
    held "$G/GATE_OK"
    S=$(stamp canary_nm)
    mkdir -p "$N"
    rm -f "$N/CANARY_OK"
    C=$(sbatch --parsable -J txt_nm_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$N,MASK=0,WRITE_FLAG=$N/CANARY_OK,ALONE_CHECK=$ALONE,TOKEN=xgbtop,XGB_CONFIG=$CFG \
         cluster/slurm/trees_xgbtop_pack.sbatch)
    echo "$(date +%F_%T) canary_nm=$C (alone $ALONE)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet_nm)
    PACKS=${2:?packs}
    held "$N/CANARY_OK"
    S=$(stamp "fleet_nm_$PACKS")
    F=$(sbatch --parsable -J txt_nm "${HOLD[@]}" --constraint="$CLASSES" --partition=main --array="$PACKS" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" \
         --time="$PACK_TIME" --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$N,MASK=0,CANARY_FLAG=$N/CANARY_OK,TOKEN=xgbtop,XGB_CONFIG=$CFG \
         cluster/slurm/trees_xgbtop_pack.sbatch)
    echo "$(date +%F_%T) fleet_nm=$F (packs $PACKS, $CLASSES)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  merge)
    PART=${2:?fixed, nomask or xcpu}
    A=${3:-}
    S=$(stamp "merge_${PART}_${A//:/_}")
    DEP=()
    [ -z "$A" ] || DEP=(--dependency="afterany:$A")
    M=$(sbatch --parsable -J txt_merge "${DEP[@]}" --export=ALL,PART=$PART cluster/slurm/trees_xgbtop_merge.sbatch)
    echo "$(date +%F_%T) merge_$PART=$M${A:+ (afterany $A)}" | tee -a "$LOG" > "$S"
    ;;
  gate_rs)
    need_tuned
    held "$U/CANARY_OK"  # the fixed cells it reproduces are the phase-1 canary's chunks c0
    S=$(stamp "gate_rs_$CLASS")
    mkdir -p "$GRS"
    rm -f "$GRS/GATE_OK"
    J=$(sbatch --parsable -J txt_rsgate "${HOLD[@]}" --constraint="$CLASS" --partition=main --export=ALL,OUT=$GRS,REF_ROOT=$PWD \
         cluster/slurm/trees_xgbtop_tuned_gates.sbatch)
    echo "$(date +%F_%T) gate_rs_$CLASS=$J (reference: the fixed cells of $U)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  canary_rs)
    need_tuned
    held "$GRS/GATE_OK"
    S=$(stamp canary_rs)
    mkdir -p "$T/rs10" "$T/rs1"
    rm -f "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    NC=$(grep -c . "$RSC")
    C10=$(sbatch --parsable -J txt_rs10_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --array=1-"$NC" --cpus-per-task="$POOL_CPUS" \
          --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$RSC,RUNG=rs10,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs10/CANARY_OK,XGB_CONFIG=$CFG \
          cluster/slurm/treestuned_xgbtop_pack.sbatch)
    C1=$(sbatch --parsable -J txt_rs1_canary "${HOLD[@]}" --constraint="$CLASS" --partition=main --array=1-"$NC" --cpus-per-task="$POOL_CPUS" \
         --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$RSC,RUNG=rs1,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs1/CANARY_OK,XGB_CONFIG=$CFG \
         cluster/slurm/treestuned_xgbtop_pack.sbatch)
    echo "$(date +%F_%T) canary_rs10=$C10 canary_rs1=$C1${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet_rs)
    need_tuned
    held "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    S=$(stamp fleet_rs)
    F=""
    for R in rs10 rs1; do
      TF=cluster/trees_xgbtop_tasks_$R.txt
      J=$(sbatch --parsable -J "txt_$R" "${HOLD[@]}" --constraint="$CLASSES" --partition=main --array=1-"$(grep -c . "$TF")" --cpus-per-task="$POOL_CPUS" \
           --mem="$TUNED_MEM" --time="$TUNED_TIME" --export=ALL,TASKFILE=$TF,RUNG=$R,RESULTS_ROOT=$T,XGB_CONFIG=$CFG \
           cluster/slurm/treestuned_xgbtop_pack.sbatch)
      F="$F fleet_$R=$J"
    done
    echo "$(date +%F_%T)$F ($CLASSES)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  merge_rs)
    need_tuned
    A=${2:-}
    S=$(stamp "merge_rs_${A//:/_}")
    DEP=()
    [ -z "$A" ] || DEP=(--dependency="afterany:$A")
    M=$(sbatch --parsable -J txt_rsmerge "${DEP[@]}" cluster/slurm/trees_xgbtop_tuned_merge.sbatch)
    echo "$(date +%F_%T) merge_rs=$M${A:+ (afterany $A)}" | tee -a "$LOG" > "$S"
    ;;
  *) echo "usage: submit_trees_xgbtop.sh gates|canary|fleet <packs>|xcpu|canary_nm|fleet_nm <packs>|merge <part> [after]|gate_rs|canary_rs|fleet_rs|merge_rs [after]"; exit 1 ;;
esac
squeue -u "$USER" -o "%.16i %.16j %.3t %.10M %.5C %R" | grep -E "JOBID|txt_" | head -12
