#!/bin/bash
# I9: merge the chunks of the MASKED 16:00 rungs after the fleets (held afterany: a failed chunk
# does not block merging the rest), with the unedited reducers, then count the kept columns.
#   T10, T1    experiments/trees_cadence_reduce_chunks.py   ($U/t10, $U/t1)
#   RS10, RS1  experiments/reduce_trees_tuned_chunks.py     ($T/rs10, $T/rs1)
#   LSTM       experiments/reduce_lstm_chunks.py            ($L)
#   kept       experiments/trees_mask_kept_counts.py        ($U/kept: the reducers drop the mask
#              keys, so the counts are read from the chunk npz files)
# Chunks without DONE are listed (NOT DONE + the last line of their run.log).  Every reducer
# writes its reduce_gates.csv / reduce.log in its rung root; $U/MERGE_SUMMARY.txt gets one line
# per rung; $U/MERGE_DONE is touched at the end (whatever merged).
#SBATCH --job-name=tm_merge
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments:$PWD/pylib_trees"
U="${UNTUNED_ROOT:-results/linear_subsection_trees_mask}"
T="${TUNED_ROOT:-results/linear_subsection_trees_tuned_mask}"
L="${LSTM_ROOT:-results/linear_subsection_lstm_mask}"
UT=cluster/trees_mask_tasks_untuned.txt
LT=cluster/lstm_mask_tasks_fleet.txt
mkdir -p "$U/t10" "$U/t1" "$T/rs10" "$T/rs1" "$L"
: > "$U/MERGE_SUMMARY.txt"

# 1. untuned
while read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO; do
  [ -n "${RUNG:-}" ] || continue
  D="$U/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
  [ -f "$D/DONE" ] && continue
  if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
done < "$UT"
for RUNG in t10 t1; do
  python -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$U/$RUNG" --tasks "$UT" \
    > "$U/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$U/$RUNG/reduce.log")"
  echo "$RUNG $(grep -c '^merged (' "$U/$RUNG/reduce.log" || true) merged; $(tail -1 "$U/$RUNG/reduce.log")" | tee -a "$U/MERGE_SUMMARY.txt"
done

# 2. tuned
for RUNG in rs10 rs1; do
  TF=cluster/trees_mask_tasks_$RUNG.txt
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
  echo "$RUNG $(grep -c '^merged (' "$T/$RUNG/reduce.log" || true) merged; $(tail -1 "$T/$RUNG/reduce.log")" | tee -a "$U/MERGE_SUMMARY.txt"
done

# 3. LSTM
while read -r BUCKET MODEL TW SEG CHUNK START END HALO; do
  [ -n "${BUCKET:-}" ] || continue
  D="$L/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
  [ -f "$D/DONE" ] && continue
  if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
done < "$LT"
python -u experiments/reduce_lstm_chunks.py --root "$L" --tasks "$LT" > "$L/reduce.log" 2>&1 \
  || echo "REDUCE lstm: $(tail -2 "$L/reduce.log")"
echo "lstm $(grep -c 'merged ' "$L/reduce.log" || true) merged-lines; $(tail -1 "$L/reduce.log")" | tee -a "$U/MERGE_SUMMARY.txt"

# 4. kept-column counts from the chunk npz files
python -u experiments/trees_mask_kept_counts.py --untuned-root "$U" --tuned-root "$T" --lstm-root "$L" \
  --out "$U/kept" > "$U/kept.log" 2>&1 || echo "KEPT COUNTS FAILED: $(tail -2 "$U/kept.log")"
tail -3 "$U/kept.log" | tee -a "$U/MERGE_SUMMARY.txt"
touch "$U/MERGE_DONE"
