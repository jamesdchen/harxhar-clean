#!/bin/bash
# rfprev163 (2026-10-07): copy of the rfdev merge on the preview root.
# rfdev (user decision 2026-10-06): merge the chunks of the dev-tuned random forest's T10 / T1 arms with the
# campaign's UNEDITED reducer (experiments/trees_cadence_reduce_chunks.py, as cluster/slurm/trees_mask_merge.sbatch),
# then the kept-column counts (experiments/trees_mask_kept_counts.py; no tuned or LSTM arms) and the CPU class
# of every chunk (NODE) into $U/class_census.csv.  Chunks without DONE are listed.  $U/MERGE_SUMMARY.txt gets
# one line per rung; $U/MERGE_DONE is touched at the end.
#SBATCH --job-name=trp_merge
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
U=results/linear_subsection_trees_mask_rfprev163
UT=cluster/trees_rfprev_tasks_untuned.txt
NONE=cluster/trees_rfdev_tasks_none.txt
mkdir -p "$U/t10" "$U/t1"
S="$U/MERGE_SUMMARY.txt"
: > "$S"
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
python -u experiments/trees_mask_kept_counts.py --untuned-root "$U" --tuned-root "$U/none" \
  --untuned-tasks "$UT" --rs10-tasks "$NONE" --rs1-tasks "$NONE" --lstm-tasks "$NONE" --out "$U/kept" > "$U/kept.log" 2>&1 \
  || echo "KEPT COUNTS FAILED: $(tail -2 "$U/kept.log")"
tail -2 "$U/kept.log" | tee -a "$S"
{
  echo "rung,bucket,model,chunk,host,class"
  for D in $(find "$U" -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
    REL=${D#"$U"/}
    IFS=/ read -r RUNG BUCKET SEG MODEL TWD CH CK <<< "$REL"
    H=$(head -1 "$D/NODE" 2>/dev/null | awk '{print $2}'); C=$(grep -o 'AvailableFeatures=[^ ]*' "$D/NODE" 2>/dev/null | cut -d= -f2)
    echo "$RUNG,$BUCKET,$MODEL,$CK,$H,\"$C\""
  done
} > "$U/class_census.csv"
echo "class census: $(($(wc -l < "$U/class_census.csv") - 1)) chunks; $(tail -n +2 "$U/class_census.csv" | cut -d, -f6 | sort | uniq -c | tr '\n' ';')" | tee -a "$S"
touch "$U/MERGE_DONE"
