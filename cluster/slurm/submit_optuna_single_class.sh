#!/bin/bash
# Single-class re-run of the masked Optuna campaign (orchestrator decision 2026-09-30, option A):
# every chunk of the canonical run computed on an epyc-7513 node.  The class-mixed run must
# already be preserved as ${R}_mixed (cp -a of the whole root, STAGE2_MERGED present).
#
#   1. the chunk dirs of the lines of cluster/optuna_tasks_s1_epyc.txt (stage 1) and
#      cluster/optuna_tasks_s2_epyc.txt (stage 2) -- cluster/optuna_single_class_tasks.py --
#      are moved to $R/replaced_mixed/stage<k>/... (never resumed from: a stage-1 chunk would
#      otherwise reload its mixed-class saved points); STAGE1_MERGED / STAGE2_MERGED removed
#   2. s1e  those stage-1 lines, --constraint=epyc-7513, 20 CPUs / 16G / 1 h
#      m1e  the stage-1 merge over the FULL stage-1 task file (afterany s1e)
#      s2e  those stage-2 lines, --constraint=epyc-7513 (afterany m1e; HOLD_FLAG STAGE1_MERGED)
#      m2e  the stage-2 merge over the FULL stage-2 task file (afterany s2e)
# Every re-run chunk writes host.txt (host, CPU model, avx512f) beside its DONE.
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_optuna_single_class.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
POOL_CPUS=${POOL_CPUS:-20}
MEM=${MEM:-16G}
CLASS=${CLASS:-epyc-7513}
TL=${TL:-1:00:00}
MERGE_TIME=${MERGE_TIME:-2:00:00}
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}
S1T=cluster/optuna_tasks_s1.txt
S2T=cluster/optuna_tasks_s2.txt
S1E=cluster/optuna_tasks_s1_epyc.txt
S2E=cluster/optuna_tasks_s2_epyc.txt
for f in "$S1T" "$S2T" "$S1E" "$S2E" cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch; do
  [ -f "$f" ] || { echo "$f missing"; exit 1; }
done
[ -f "${R}_mixed/STAGE2_MERGED" ] || { echo "${R}_mixed/STAGE2_MERGED missing: preserve the class-mixed run first (cp -a $R ${R}_mixed)"; exit 1; }
grep -qvxFf "$S1T" "$S1E" && { echo "$S1E has a line that is not in $S1T"; exit 1; }
grep -qvxFf "$S2T" "$S2E" && { echo "$S2E has a line that is not in $S2T"; exit 1; }
moved=0
for spec in "1 $S1E" "2 $S2E"; do
  read -r K TF <<< "$spec"
  while read -r BUCKET MODEL TW SEG CHUNK START END HALO; do
    [ -n "${BUCKET:-}" ] || continue
    SRC="$R/stage$K/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    DST="$R/replaced_mixed/stage$K/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    if [ -d "$SRC" ]; then
      mkdir -p "$(dirname "$DST")"
      [ -e "$DST" ] && { echo "$DST exists already; refusing"; exit 1; }
      mv "$SRC" "$DST"
      moved=$((moved + 1))
    fi
  done < "$TF"
done
rm -f "$R/STAGE1_MERGED" "$R/STAGE2_MERGED"
echo "moved $moved chunk dirs to $R/replaced_mixed"
COMMON="RESULTS_ROOT=$R,WINDOW_MASK=1"
S1=$(sbatch --parsable -J op_s1e --constraint="$CLASS" --array=1-"$(wc -l < "$S1E")" --cpus-per-task="$POOL_CPUS" \
      --mem="$MEM" --time="$TL" "--export=ALL,TASKFILE=$S1E,STAGE=tune,$COMMON" cluster/slurm/optuna_pack.sbatch)
M1=$(sbatch --parsable -J op_m1e --dependency=afterany:"$S1" --time="$MERGE_TIME" \
      "--export=ALL,MERGE_STAGE=1,TASKFILE=$S1T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
S2=$(sbatch --parsable -J op_s2e --constraint="$CLASS" --array=1-"$(wc -l < "$S2E")" --dependency=afterany:"$M1" \
      --cpus-per-task="$POOL_CPUS" --mem="$MEM" --time="$TL" \
      "--export=ALL,TASKFILE=$S2E,STAGE=refit,$COMMON,HOLD_FLAG=$R/STAGE1_MERGED" cluster/slurm/optuna_pack.sbatch)
M2=$(sbatch --parsable -J op_m2e --dependency=afterany:"$S2" --time="$MERGE_TIME" \
      "--export=ALL,MERGE_STAGE=2,TASKFILE=$S2T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
echo "single-class ($CLASS): s1e=$S1 ($(wc -l < "$S1E")) m1e=$M1 s2e=$S2 ($(wc -l < "$S2E")) m2e=$M2 root=$R" | tee -a logs/submitted_optuna_carc.txt
squeue -u "$USER" -o "%.14i %.10j %.4t %.10M %.5C %P %R" | grep -E "op_|JOBID" | head -8
