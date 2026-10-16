#!/bin/bash
# rfx (2026-10-07, TOKEN-driven twin of cluster/slurm/trees_rfp163_merge.sbatch; PART fx adds the masked fixed-settings
# T10 / T1 of results/linear_subsection_trees_mask_$TOKEN): merge the re-centred random-search rungs (RS10 / RS1: experiments/reduce_trees_tuned_chunks.py)
# and the unmasked fixed-settings rungs (T10 / T1, WINDOW_MASK=0: experiments/trees_cadence_reduce_chunks.py) with the
# campaign's UNEDITED reducers, then the CPU class of every chunk (NODE) into class_census.csv per root.  PART (exported):
# tuned | nomask | all.  Chunks without DONE are listed; MERGE_DONE_<part> is touched at the end.
#SBATCH --job-name=rfx_merge
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
TOKEN="${TOKEN:?TOKEN}"
U=results/linear_subsection_trees_mask_$TOKEN
UT=cluster/trees_${TOKEN}_tasks_untuned.txt
T=results/linear_subsection_trees_tuned_mask_$TOKEN
N=results/linear_subsection_trees_nomask_$TOKEN
NT=cluster/trees_${TOKEN}_nomask_tasks_untuned.txt
P="${PART:?PART must be fx, tuned, nomask or all}"
census() {  # $1 root: CARC chunks carry NODE (host + AvailableFeatures), Hoffman2 chunks H2_HOST (host, class, CPU model)
  { echo "chunk_dir,cluster,host,class"
    for X in $(find "$1" -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
      if [ -f "$X/H2_HOST" ]; then
        echo "${X#"$1"/},hoffman2,$(grep -o 'host=[^ ]*' "$X/H2_HOST" | cut -d= -f2),\"$(sed 's/.*class=\([^ ]*\) cpu=\(.*\) seconds=.*/\1 \2/' "$X/H2_HOST")\""
      else
        echo "${X#"$1"/},carc,$(head -1 "$X/NODE" 2>/dev/null | awk '{print $2}'),\"$(grep -o 'AvailableFeatures=[^ ]*' "$X/NODE" 2>/dev/null | cut -d= -f2)\""
      fi
    done; } > "$1/class_census.csv"
  echo "$1 census: $(tail -n +2 "$1/class_census.csv" | cut -d'"' -f2 | sort | uniq -c | tr '\n' ' ')"
}
if [ "$P" = fx ] || [ "$P" = all ]; then
  for RUNG in t10 t1; do
    mkdir -p "$U/$RUNG"
    python -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$U/$RUNG" --tasks "$UT"       > "$U/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$U/$RUNG/reduce.log")"
    echo "$RUNG $(grep -c '^merged (' "$U/$RUNG/reduce.log" || true) merged; $(tail -1 "$U/$RUNG/reduce.log")"
  done
  census "$U"
  touch "$U/MERGE_DONE_fx"
fi
if [ "$P" = tuned ] || [ "$P" = all ]; then
  for RUNG in rs10 rs1; do
    TF=cluster/trees_${TOKEN}_tasks_$RUNG.txt
    mkdir -p "$T/$RUNG"
    while read -r B M TW SEG C S E H; do
      [ -n "${B:-}" ] || continue
      D="$T/$RUNG/$B/$SEG/$M/tw$TW/chunks/c$C"
      [ -f "$D/DONE" ] || echo "NOT DONE: $D: $(tail -1 "$D/run.log" 2>/dev/null | cut -c1-200)"
    done < "$TF"
    python -u experiments/reduce_trees_tuned_chunks.py --root "$T/$RUNG" --tasks "$TF" > "$T/$RUNG/reduce.log" 2>&1 \
      || echo "REDUCE $RUNG: $(tail -2 "$T/$RUNG/reduce.log")"
    echo "$RUNG $(grep -c '^merged (' "$T/$RUNG/reduce.log" || true) merged; $(tail -1 "$T/$RUNG/reduce.log")"
  done
  census "$T"
  touch "$T/MERGE_DONE_tuned"
fi
if [ "$P" = nomask ] || [ "$P" = all ]; then
  for RUNG in t10 t1; do
    mkdir -p "$N/$RUNG"
    python -u experiments/trees_cadence_reduce_chunks.py --rung "$RUNG" --root "$N/$RUNG" --tasks "$NT" \
      > "$N/$RUNG/reduce.log" 2>&1 || echo "REDUCE $RUNG: $(tail -1 "$N/$RUNG/reduce.log")"
    echo "$RUNG $(grep -c '^merged (' "$N/$RUNG/reduce.log" || true) merged; $(tail -1 "$N/$RUNG/reduce.log")"
  done
  census "$N"
  touch "$N/MERGE_DONE_nomask"
fi
