#!/bin/bash
# xgbtop (user decision 2026-10-07): merge the chunks of the fixed-settings XGBoost runs with the campaign's UNEDITED
# reducer (experiments/trees_cadence_reduce_chunks.py, as cluster/slurm/trees_rfdev_merge.sbatch), per PART (exported):
#   fixed   the masked T10 / T1 arms, root results/linear_subsection_trees_mask_xgbtop (+ kept-column counts)
#   nomask  the unmasked twin, root results/linear_subsection_trees_nomask_xgbtop
#   xcpu    no merge: every chunk of the cross-CPU cells (results/linear_subsection_trees_mask_xgbtop_xcpu/<class>)
#           against the epyc-7513 fleet's chunks, byte for byte (experiments/trees_xgbtop_xcpu_compare.py)
# Chunks without DONE are listed; the CPU class of every chunk (NODE) goes to <root>/class_census.csv; <root>/
# MERGE_SUMMARY.txt gets one line per rung; <root>/MERGE_DONE is touched at the end.
#SBATCH --job-name=txt_merge
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments:$PWD/pylib_trees"
UT=cluster/trees_xgbtop_tasks_untuned.txt
NONE=cluster/trees_rfdev_tasks_none.txt
P="${PART:?PART must be fixed, nomask or xcpu}"
census() {  # $1 root
  { echo "rung,bucket,model,chunk,host,class"
    for D in $(find "$1" -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
      REL=${D#"$1"/}
      IFS=/ read -r RUNG BUCKET SEG MODEL TWD CH CK <<< "$REL"
      H=$(head -1 "$D/NODE" 2>/dev/null | awk '{print $2}'); C=$(grep -o 'AvailableFeatures=[^ ]*' "$D/NODE" 2>/dev/null | cut -d= -f2)
      echo "$RUNG,$BUCKET,$MODEL,$CK,$H,\"$C\""
    done; } > "$1/class_census.csv"
  echo "class census: $(($(wc -l < "$1/class_census.csv") - 1)) chunks; $(tail -n +2 "$1/class_census.csv" | cut -d, -f6 | sort | uniq -c | tr '\n' ';')"
}
case "$P" in
  fixed|nomask)
    if [ "$P" = fixed ]; then U=results/linear_subsection_trees_mask_xgbtop; else U=results/linear_subsection_trees_nomask_xgbtop; fi
    mkdir -p "$U/t10" "$U/t1"
    S="$U/MERGE_SUMMARY.txt"
    : > "$S"
    while read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO; do
      [ -n "${RUNG:-}" ] || continue
      D="$U/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
      [ -f "$D/DONE" ] && continue
      if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
    done < "$UT" | tee -a "$S"
    for RUNG in t10 t1; do
      python -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$U/$RUNG" --tasks "$UT" \
        > "$U/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$U/$RUNG/reduce.log")"
      echo "$RUNG $(grep -c '^merged (' "$U/$RUNG/reduce.log" || true) merged; $(tail -1 "$U/$RUNG/reduce.log")" | tee -a "$S"
    done
    if [ "$P" = fixed ]; then
      python -u experiments/trees_mask_kept_counts.py --untuned-root "$U" --tuned-root "$U/none" \
        --untuned-tasks "$UT" --rs10-tasks "$NONE" --rs1-tasks "$NONE" --lstm-tasks "$NONE" --out "$U/kept" > "$U/kept.log" 2>&1 \
        || echo "KEPT COUNTS FAILED: $(tail -2 "$U/kept.log")"
      tail -2 "$U/kept.log" | tee -a "$S"
    fi
    census "$U" | tee -a "$S"
    touch "$U/MERGE_DONE"
    ;;
  xcpu)
    X=results/linear_subsection_trees_mask_xgbtop_xcpu
    for C in ${XCPU_CLASSES:-epyc-7542 xeon-4116}; do
      [ -d "$X/$C" ] || { echo "no $X/$C"; continue; }
      python -u experiments/trees_xgbtop_xcpu_compare.py --ref results/linear_subsection_trees_mask_xgbtop \
        --other "$X/$C" --label "$C" --out "$X" > "$X/xcpu_$C.log" 2>&1 || true
      tail -1 "$X/xcpu_$C.log"
    done
    touch "$X/XCPU_COMPARED"
    ;;
  *) echo "PART must be fixed, nomask or xcpu"; exit 1 ;;
esac
