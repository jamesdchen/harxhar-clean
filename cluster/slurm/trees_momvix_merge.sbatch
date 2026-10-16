#!/bin/bash
# moments_vix tree workstream (2026-10-06): merge the chunks of the moments_vix 16:00 rungs with the
# campaign's UNEDITED reducers (as cluster/slurm/trees_mask_merge.sbatch, own roots and task files):
#   T10, T1    experiments/trees_cadence_reduce_chunks.py   ($U/t10, $U/t1)
#   RS10, RS1  experiments/reduce_trees_tuned_chunks.py     ($T/rs10, $T/rs1)
# then the kept-column counts (experiments/trees_mask_kept_counts.py, own roots, no LSTM arms) and
# the CPU class of every chunk (NODE on CARC, H2_HOST for a chunk relayed from Hoffman2) into
# $U/class_census.csv.  PART (exported): untuned | tuned | all.  Chunks without DONE are listed.
# $U/MERGE_SUMMARY_<part>.txt gets one line per rung; $U/MERGE_DONE_<part> is touched at the end.
#SBATCH --job-name=tmv_merge
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
U=results/linear_subsection_trees_mask_momvix
T=results/linear_subsection_trees_tuned_mask_momvix
UT=cluster/trees_momvix_tasks_untuned.txt
P="${PART:?PART must be untuned, tuned or all}"
mkdir -p "$U/t10" "$U/t1" "$T/rs10" "$T/rs1"
S="$U/MERGE_SUMMARY_$P.txt"
: > "$S"

if [ "$P" = untuned ] || [ "$P" = all ]; then
  while read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO; do
    [ -n "${RUNG:-}" ] || continue
    D="$U/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    [ -f "$D/DONE" ] && continue
    if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
  done < "$UT"
  for RUNG in t10 t1; do
    python -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$U/$RUNG" --tasks "$UT" \
      > "$U/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$U/$RUNG/reduce.log")"
    echo "$RUNG $(grep -c '^merged (' "$U/$RUNG/reduce.log" || true) merged; $(tail -1 "$U/$RUNG/reduce.log")" | tee -a "$S"
  done
fi
if [ "$P" = tuned ] || [ "$P" = all ]; then
  for RUNG in rs10 rs1; do
    TF=cluster/trees_momvix_tasks_$RUNG.txt
    while read -r BUCKET MODELS TW SEG CHUNK START END HALO; do
      [ -n "${BUCKET:-}" ] || continue
      for MODEL in ${MODELS//,/ }; do
        D="$T/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
        [ -f "$D/DONE" ] && continue
        if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
      done
    done < "$TF"
    python -u experiments/reduce_trees_tuned_chunks.py --root "$T/$RUNG" --tasks "$TF" \
      > "$T/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -2 "$T/$RUNG/reduce.log")"
    echo "$RUNG $(grep -c '^merged (' "$T/$RUNG/reduce.log" || true) merged; $(tail -1 "$T/$RUNG/reduce.log")" | tee -a "$S"
  done
fi

python -u experiments/trees_mask_kept_counts.py --untuned-root "$U" --tuned-root "$T" \
  --untuned-tasks "$UT" --rs10-tasks cluster/trees_momvix_tasks_rs10.txt --rs1-tasks cluster/trees_momvix_tasks_rs1.txt \
  --lstm-tasks cluster/trees_momvix_tasks_none.txt --out "$U/kept" > "$U/kept.log" 2>&1 \
  || echo "KEPT COUNTS FAILED: $(tail -2 "$U/kept.log")"
tail -2 "$U/kept.log" | tee -a "$S"

# the CPU class of every finished chunk: CARC's NODE (host + AvailableFeatures) or the H2_HOST record
{
  echo "root,rung,bucket,model,chunk,cluster,host,class"
  for D in $(find "$U" "$T" -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
    REL=${D#results/}
    IFS=/ read -r ROOTN RUNG BUCKET SEG MODEL TWD CH CK <<< "$REL"
    if [ -f "$D/H2_HOST" ]; then
      H=$(grep -o 'host=[^ ]*' "$D/H2_HOST" | cut -d= -f2); C=$(grep -o 'cpu=.*seconds' "$D/H2_HOST" | sed 's/^cpu=//; s/ seconds$//')
      echo "$ROOTN,$RUNG,$BUCKET,$MODEL,$CK,hoffman2,$H,\"$C\""
    elif [ -f "$D/NODE" ]; then
      H=$(head -1 "$D/NODE" | awk '{print $2}'); C=$(grep -o 'AvailableFeatures=[^ ]*' "$D/NODE" | cut -d= -f2)
      echo "$ROOTN,$RUNG,$BUCKET,$MODEL,$CK,carc,$H,\"$C\""
    else
      echo "$ROOTN,$RUNG,$BUCKET,$MODEL,$CK,unknown,,"
    fi
  done
} > "$U/class_census.csv"
echo "class census: $(($(wc -l < "$U/class_census.csv") - 1)) chunks; $(tail -n +2 "$U/class_census.csv" | cut -d, -f6,8 | sort | uniq -c | tr '\n' ';')" | tee -a "$S"
touch "$U/MERGE_DONE_$P"
