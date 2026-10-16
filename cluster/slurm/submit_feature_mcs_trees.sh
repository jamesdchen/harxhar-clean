#!/bin/bash
# Feature MCS, the tree candidates (2026-10-08): submit on CARC, from the deployment root /scratch1/jc_905/harxhar-mcstrees
# (cluster/slurm/ship_feature_mcs_trees_carc.sh), from a shell with NO conda env active.  Every job is pinned to one CPU
# class, CONSTRAINT (default epyc-7513: the class of the stored LightGBM T1 runs; every estimator on it, so each
# estimator's 96 candidates share one CPU model) and checks the CPU model in every pack (REQUIRE_CPU, default "EPYC 7513").
#   canary   pack 0 of cluster/feature_mcs_trees_canary.txt (12 processes: per estimator the two anchors, one add-one and
#            one leave-one-out design on a slice of the rows) -> results/feature_mcs_trees_work/canary/, then
#            canary_check afterok (anchor rows bit-identical to the stored T1 tables, the measured costs, CANARY_OK)
#   fleet    needs CANARY_OK and the planned task files (experiments/feature_mcs_trees.py plan); per estimator E (an
#            estimator whose GATE_OK_E exists -- a re-shaped resubmission -- gets its fleet only, unheld):
#            anchors array (cluster/feature_mcs_trees_anchors_E.txt) -> gate E afterok (GATE_OK_E) -> fleet array
#            (cluster/feature_mcs_trees_fleet_E.txt) afterok the gate, --kill-on-invalid-dep; then collect afterany every
#            fleet (results/feature_mcs_2026-10-08/trees/, TREES_DONE last)
#   collect  the collect step alone (after a resumed pack)
# Times: the planned per-array limits in cluster/feature_mcs_trees_projection.csv.  A stamp refuses a second identical
# submission (logs/submitted_feature_mcs_trees_<mode>.txt; move it aside by hand to resubmit).
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${CONDA_PREFIX:-}${CONDA_DEFAULT_ENV:-}" in ?*) echo "a conda env is active (${CONDA_PREFIX:-}${CONDA_DEFAULT_ENV:-}); submit from a clean shell"; exit 1 ;; esac
MODE=${1:?usage: submit_feature_mcs_trees.sh canary | fleet | fleet2 | feed <est> <stage> <range> | gate2 <est> <anchors id> | collect}
CONSTRAINT=${CONSTRAINT:-epyc-7513}
REQUIRE_CPU=${REQUIRE_CPU:-EPYC 7513}
MEM_PER_PROC=${MEM_PER_PROC:-3}  # GB per chunk process (the all_features data preparation peaks above 2 GB: canary 12857408)
WK=results/feature_mcs_trees_work
LOG=logs/submitted_feature_mcs_trees.txt
mkdir -p logs "$WK"
STAMP=logs/submitted_feature_mcs_trees_$MODE.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); move $STAMP aside by hand to resubmit"; exit 1; }
sb() { sbatch --parsable --account=pollok_1603 --partition=main --constraint="$CONSTRAINT" "$@"; }
limit() {  # est stage -> minutes from the projection
  awk -F, -v e="$1" -v s="$2" '{sub(/\r$/, "")} NR == 1 {for (i = 1; i <= NF; i++) h[$i] = i; next} $h["est"] == e && $h["stage"] == s {print $h["time_limit_min"]}' \
    cluster/feature_mcs_trees_projection.csv
}
packs() { awk 'NF {print $1}' "$1" | sort -n | tail -1; }
width() { awk 'NF {c[$1]++} END {m = 0; for (k in c) if (c[k] > m) m = c[k]; print m}' "$1"; }  # the largest pack = CPUs per task
case "$MODE" in
  canary)
    DEP=(); [ -z "${AFTER:-}" ] || DEP=(--dependency=afterany:"$AFTER")  # a resumed canary (DONE chunks skipped): after the last one
    C=$(sb -J mct_canary "${DEP[@]}" --cpus-per-task=12 --mem="$((12 * MEM_PER_PROC))G" --time=1:00:00 \
          --export=ALL,TASKFILE=cluster/feature_mcs_trees_canary.txt,PACK=0,RESULTS_ROOT=$WK/canary,REQUIRE_CPU="$REQUIRE_CPU" \
          cluster/slurm/feature_mcs_trees_pack.sbatch)
    K=$(sb -J mct_canary_check --dependency=afterok:"$C" --kill-on-invalid-dep=yes --export=ALL,MODE=canary_check \
          cluster/slurm/feature_mcs_trees_check.sbatch)
    echo "$(date +%F_%T) canary=$C canary_check=$K ($CONSTRAINT)" | tee -a "$LOG" > "$STAMP"
    ;;
  fleet)
    [ -f "$WK/CANARY_OK" ] || { echo "no $WK/CANARY_OK (the canary has not passed)"; exit 1; }
    FLEETS=()
    MSG=""
    for E in lgbm xgb rf; do
      AT=cluster/feature_mcs_trees_anchors_$E.txt
      FT=cluster/feature_mcs_trees_fleet_$E.txt
      if [ -f "$WK/GATE_OK_$E" ]; then  # a resubmission (re-shaped packs): this estimator's gate already passed
        A=passed; G=passed; DEPF=()
      else
        A=$(sb -J "mct_${E}_anchors" --array=1-"$(packs "$AT")" --cpus-per-task="$(width "$AT")" --mem="$((MEM_PER_PROC * $(width "$AT")))G" --time="$(limit "$E" anchors)" \
              --export=ALL,TASKFILE=$AT,RESULTS_ROOT=$WK/chunks,CANARY_FLAG=$WK/CANARY_OK,REQUIRE_CPU="$REQUIRE_CPU" \
              cluster/slurm/feature_mcs_trees_pack.sbatch)
        G=$(sb -J "mct_${E}_gate" --dependency=afterok:"$A" --kill-on-invalid-dep=yes --export=ALL,MODE=gate,EST=$E \
              cluster/slurm/feature_mcs_trees_check.sbatch)
        DEPF=(--dependency=afterok:"$G" --kill-on-invalid-dep=yes)
      fi
      F=$(sb -J "mct_${E}_fleet" --array=1-"$(packs "$FT")" --cpus-per-task="$(width "$FT")" --mem="$((MEM_PER_PROC * $(width "$FT")))G" --time="$(limit "$E" fleet)" \
            "${DEPF[@]}" \
            --export=ALL,TASKFILE=$FT,RESULTS_ROOT=$WK/chunks,CANARY_FLAG="$WK/CANARY_OK $WK/GATE_OK_$E",REQUIRE_CPU="$REQUIRE_CPU" \
            cluster/slurm/feature_mcs_trees_pack.sbatch)
      FLEETS+=("$F")
      MSG="$MSG ${E}: anchors=$A gate=$G fleet=$F;"
    done
    D=$(IFS=:; echo "${FLEETS[*]}")
    K=$(sb -J mct_collect --dependency=afterany:"$D" --export=ALL,MODE=collect cluster/slurm/feature_mcs_trees_check.sbatch)
    echo "$(date +%F_%T)$MSG collect=$K ($CONSTRAINT)" | tee -a "$LOG" > "$STAMP"
    ;;
  fleet2)
    # 2026-10-09 01:30, the backfill-sized re-plan (experiments/feature_mcs_trees.py plan --suffix _v2, MCS_TARGET_SEC 700,
    # MCS_PACK_MAX 4): 4 one-CPU processes / 12 GB / ~25 min per job, chunks in results/feature_mcs_trees_work/chunks2 (a
    # new geometry; the collect step reads both roots).  CPU classes: LightGBM epyc-7513 only; XGBoost and forest fleets
    # epyc-7513|epyc-7542 (byte-identical); the forest's anchors and gate on epyc-7513.  An estimator whose gate passed
    # gets its fleet only.
    [ -f "$WK/CANARY_OK" ] || { echo "no $WK/CANARY_OK (the canary has not passed)"; exit 1; }
    lim2() { awk -F, -v e="$1" -v s="$2" '{sub(/\r$/, "")} NR == 1 {for (i = 1; i <= NF; i++) h[$i] = i; next} $h["est"] == e && $h["stage"] == s {print $h["time_limit_min"]}' cluster/feature_mcs_trees_projection_v2.csv; }
    sb2() { sbatch --parsable --account=pollok_1603 --partition=main "$@"; }
    FLEETS=()
    MSG=""
    for E in lgbm xgb rf; do
      AT=cluster/feature_mcs_trees_anchors_${E}_v2.txt
      FT=cluster/feature_mcs_trees_fleet_${E}_v2.txt
      case "$E" in lgbm) FC=epyc-7513; FR="EPYC 7513" ;; *) FC="epyc-7513|epyc-7542"; FR="EPYC 7513|EPYC 7542" ;; esac
      if [ -f "$WK/GATE_OK_$E" ]; then
        A=passed; G=passed; DEPF=()
      else
        A=$(sb2 -J "mct2_${E}_anchors" --constraint=epyc-7513 --array=1-"$(packs "$AT")" --cpus-per-task="$(width "$AT")" \
              --mem="$((MEM_PER_PROC * $(width "$AT")))G" --time="$(lim2 "$E" anchors)" \
              --export=ALL,TASKFILE=$AT,RESULTS_ROOT=$WK/chunks2,CANARY_FLAG=$WK/CANARY_OK,REQUIRE_CPU="EPYC 7513" \
              cluster/slurm/feature_mcs_trees_pack.sbatch)
        G=$(sb2 -J "mct2_${E}_gate" --constraint=epyc-7513 --dependency=afterok:"$A" --kill-on-invalid-dep=yes \
              --export=ALL,MODE=gate,EST=$E cluster/slurm/feature_mcs_trees_check.sbatch)
        DEPF=(--dependency=afterok:"$G" --kill-on-invalid-dep=yes)
      fi
      F=$(sb2 -J "mct2_${E}_fleet" --constraint="$FC" --array=1-"$(packs "$FT")" --cpus-per-task="$(width "$FT")" \
            --mem="$((MEM_PER_PROC * $(width "$FT")))G" --time="$(lim2 "$E" fleet)" "${DEPF[@]}" \
            --export=ALL,TASKFILE=$FT,RESULTS_ROOT=$WK/chunks2,CANARY_FLAG="$WK/CANARY_OK $WK/GATE_OK_$E",REQUIRE_CPU="$FR" \
            cluster/slurm/feature_mcs_trees_pack.sbatch)
      FLEETS+=("$F")
      MSG="$MSG ${E}: anchors=$A gate=$G fleet=$F ($FC);"
    done
    D=$(IFS=:; echo "${FLEETS[*]}")
    K=$(sb2 -J mct2_collect --dependency=afterany:"$D" --export=ALL,MODE=collect cluster/slurm/feature_mcs_trees_check.sbatch)
    echo "$(date +%F_%T) fleet2:$MSG collect=$K" | tee -a "$LOG" > "$STAMP"
    ;;
  feed)
    # the v2 packs in slices (2026-10-09 01:50): the per-user SUBMIT cap (5,000 jobs, array tasks counted) leaves little
    # headroom, so the v2 arrays go in as ranges that fit, refilled as packs finish.  feed <est> <anchors|fleet> <range>;
    # a forest fleet slice before GATE_OK_rf exists needs GATE=<the gate job id> (afterok, --kill-on-invalid-dep).
    E=${2:?est}; ST=${3:?anchors or fleet}; R=${4:?array range, e.g. 1-40}
    SFX=${SUFFIX:-_v2}  # the plan's task / projection files (2026-10-12 bucket designs: _v3)
    TF=cluster/feature_mcs_trees_${ST}_${E}${SFX}.txt
    lim2() { awk -F, -v e="$1" -v s="$2" '{sub(/\r$/, "")} NR == 1 {for (i = 1; i <= NF; i++) h[$i] = i; next} $h["est"] == e && $h["stage"] == s {print $h["time_limit_min"]}' cluster/feature_mcs_trees_projection${SFX}.csv; }
    if [ "$ST" = anchors ] || [ "$E" = lgbm ]; then FC=epyc-7513; FR="EPYC 7513"; else FC="epyc-7513|epyc-7542"; FR="EPYC 7513|EPYC 7542"; fi
    FLAGS="$WK/CANARY_OK"; DEPF=()
    if [ "$ST" = fleet ]; then
      FLAGS="$WK/CANARY_OK $WK/GATE_OK_$E"
      if [ ! -f "$WK/GATE_OK_$E" ]; then DEPF=(--dependency=afterok:"${GATE:?GATE (the gate job id): $E has not passed its gate}" --kill-on-invalid-dep=yes); fi
    fi
    F=$(sbatch --parsable --account=pollok_1603 --partition=main -J "${JOBNAME_PREFIX:-mct${SFX#_v}}_${E}_${ST}" --constraint="$FC" --array="$R" \
          --cpus-per-task="$(width "$TF")" --mem="$((MEM_PER_PROC * $(width "$TF")))G" --time="$(lim2 "$E" "$ST")" "${DEPF[@]}" \
          --export=ALL,TASKFILE=$TF,RESULTS_ROOT=$WK/chunks2,CANARY_FLAG="$FLAGS",REQUIRE_CPU="$FR" \
          cluster/slurm/feature_mcs_trees_pack.sbatch)
    echo "$(date +%F_%T) feed $E $ST $R = $F ($FC)${GATE:+ after gate $GATE}" | tee -a "$LOG"
    rm -f "$STAMP"
    ;;
  gate2)
    # the gate of an estimator's v2 anchors: gate2 <est> <anchors job id>
    E=${2:?est}; A=${3:?anchors job id}
    G=$(sbatch --parsable --account=pollok_1603 --partition=main --constraint=epyc-7513 -J "mct2_${E}_gate" \
          --dependency=afterok:"$A" --kill-on-invalid-dep=yes --export=ALL,MODE=gate,EST=$E cluster/slurm/feature_mcs_trees_check.sbatch)
    echo "$(date +%F_%T) gate2 $E = $G (afterok $A)" | tee -a "$LOG"
    rm -f "$STAMP"
    ;;
  collect)
    K=$(sb -J mct_collect --export=ALL,MODE=${COLLECT_MODE:-collect} cluster/slurm/feature_mcs_trees_check.sbatch)
    echo "$(date +%F_%T) collect=$K" | tee -a "$LOG"
    rm -f "$STAMP"
    ;;
  *) echo "MODE must be canary, fleet, fleet2, feed, gate2 or collect"; exit 1 ;;
esac
squeue -u "$USER" -o "%.16i %.20j %.3t %.10M %.5C %R" | grep -E "JOBID|mct_" | head -14
