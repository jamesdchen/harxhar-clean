#!/bin/bash
# I9: merge the chunks of the MASKED 16:00 rungs after their fleets (held afterany: a failed chunk
# does not block merging the rest), with the unedited reducers, then count the kept columns.
# PART (exported at submit time):
#   trees  T10, T1    experiments/trees_cadence_reduce_chunks.py   ($U/t10, $U/t1)
#          RS10, RS1  experiments/reduce_trees_tuned_chunks.py     ($T/rs10, $T/rs1)
#   lstm   LSTM       experiments/reduce_lstm_chunks.py            ($L1 = REFIT_EVERY 1, $L10 = 10)
#   classgate  the live_feasible LightGBM T10 / T1 arms re-run on the other CPU class
#          (results/linear_subsection_trees_mask_xeon; submit mode classgate), same reducer
#   kept   (every part ends with it) experiments/trees_mask_kept_counts.py -> $U/kept: the reducers
#          drop the mask keys, so the counts are read from the chunk npz files; arms not finished
#          yet are listed as incomplete; then experiments/trees_mask_class_gate.py -> $U/class
#          (which CPU class every chunk ran on, the canaries moved aside vs their re-runs, the
#          class-gate arms vs the fleet's)
# Chunks without DONE are listed (NOT DONE + the last line of their run.log).  Every reducer
# writes its reduce_gates.csv / reduce.log in its root; $U/MERGE_SUMMARY_<part>.txt gets one line
# per rung; $U/MERGE_DONE_<part> is touched at the end (whatever merged).
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
U=results/linear_subsection_trees_mask
T=results/linear_subsection_trees_tuned_mask
L1=results/linear_subsection_lstm_mask
L10=results/linear_subsection_lstm_mask_re10
UT=cluster/trees_mask_tasks_untuned.txt
LT=cluster/lstm_mask_tasks_fleet.txt
P="${PART:?PART must be trees, lstm, classgate or kept}"
mkdir -p "$U/t10" "$U/t1" "$T/rs10" "$T/rs1" "$L1" "$L10"
S="$U/MERGE_SUMMARY_$P.txt"
: > "$S"

if [ "$P" = trees ]; then
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
    echo "$RUNG $(grep -c '^merged (' "$T/$RUNG/reduce.log" || true) merged; $(tail -1 "$T/$RUNG/reduce.log")" | tee -a "$S"
  done
elif [ "$P" = classgate ]; then
  X=results/linear_subsection_trees_mask_xeon
  for RUNG in t10 t1; do
    mkdir -p "$X/$RUNG"
    python -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$X/$RUNG" \
      --tasks cluster/trees_mask_tasks_classgate.txt > "$X/$RUNG/reduce.log" 2>&1 \
      || echo "REDUCE classgate $RUNG: $(tail -1 "$X/$RUNG/reduce.log")"
    echo "classgate $RUNG $(grep -c '^merged (' "$X/$RUNG/reduce.log" || true) merged; $(tail -1 "$X/$RUNG/reduce.log")" | tee -a "$S"
  done
elif [ "$P" = lstm ]; then
  for L in "$L1" "$L10"; do
    while read -r BUCKET MODEL TW SEG CHUNK START END HALO; do
      [ -n "${BUCKET:-}" ] || continue
      D="$L/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
      [ -f "$D/DONE" ] && continue
      if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
    done < "$LT"
    python -u experiments/reduce_lstm_chunks.py --root "$L" --tasks "$LT" > "$L/reduce.log" 2>&1 \
      || echo "REDUCE $L: $(tail -2 "$L/reduce.log")"
    echo "$L $(grep -c '^merged ' "$L/reduce.log" || true) merged; $(tail -1 "$L/reduce.log")" | tee -a "$S"
  done
fi

python -u experiments/trees_mask_kept_counts.py --untuned-root "$U" --tuned-root "$T" --lstm-root "$L1" \
  --lstm10-root "$L10" --out "$U/kept" > "$U/kept.log" 2>&1 || echo "KEPT COUNTS FAILED: $(tail -2 "$U/kept.log")"
tail -2 "$U/kept.log" | tee -a "$S"
python -u experiments/trees_mask_class_gate.py --out "$U/class" > "$U/class.log" 2>&1 \
  || echo "CLASS GATE FAILED: $(tail -2 "$U/class.log")"
tail -1 "$U/class.log" | tee -a "$S"
touch "$U/MERGE_DONE_$P"
