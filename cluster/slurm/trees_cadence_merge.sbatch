#!/bin/bash
# Merge the chunks of the 16:00 tree cadence rungs (checklist I2) into the unchunked
# layout the stackers read, after the fleets (held afterany: a failed chunk does not
# block merging the rest).
#   T10, T1   experiments/trees_cadence_reduce_chunks.py   ($U/t10, $U/t1)
#   RS10, RS1 experiments/reduce_trees_tuned_chunks.py     ($T/rs10, $T/rs1)
# Chunks without DONE are listed (NOT DONE + the last line of their run.log).  Every
# reducer writes reduce_gates.csv in its rung root; MERGE_DONE is touched at the end
# (whatever merged), with one line per rung: rung, arms merged, arms expected.
#SBATCH --job-name=tc_merge
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
U="${UNTUNED_ROOT:-results/linear_subsection_trees_dedup}"
T="${TUNED_ROOT:-results/linear_subsection_trees_tuned_dedup}"
UT=cluster/trees_cadence_tasks_untuned.txt
mkdir -p "$U/t10" "$U/t1"
: > "$U/MERGE_SUMMARY.txt"

# 1. untuned chunk processes not DONE
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

# 2. tuned chunk-arms not DONE, then merge
for RUNG in rs10 rs1; do
  TF=cluster/trees_cadence_tasks_$RUNG.txt
  while read -r BUCKET MODELS TW SEG CHUNK START END HALO; do
    [ -n "${BUCKET:-}" ] || continue
    for MODEL in ${MODELS//,/ }; do
      D="$T/$RUNG/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
      [ -f "$D/DONE" ] && continue
      if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
    done
  done < "$TF"
  mkdir -p "$T/$RUNG"
  python -u experiments/reduce_trees_tuned_chunks.py --root "$T/$RUNG" --tasks "$TF" \
    > "$T/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -2 "$T/$RUNG/reduce.log")"
  echo "$RUNG $(grep -c '^merged (' "$T/$RUNG/reduce.log" || true) merged; $(tail -1 "$T/$RUNG/reduce.log")" | tee -a "$U/MERGE_SUMMARY.txt"
done
touch "$U/MERGE_DONE"
