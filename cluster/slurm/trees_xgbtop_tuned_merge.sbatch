#!/bin/bash
# xgbtop (2026-10-07): merge the re-centred random-search rungs (RS10 / RS1) of the per-bar XGBoost with the campaign's
# UNEDITED reducer (experiments/reduce_trees_tuned_chunks.py), then the CPU class of every chunk (NODE) into
# class_census.csv (cluster/slurm/trees_rfp163_merge.sbatch, PART tuned, with this run's root and task files).  Chunks
# without DONE are listed; MERGE_DONE_tuned is touched at the end.
#SBATCH --job-name=txt_rsmerge
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
T=results/linear_subsection_trees_tuned_mask_xgbtop
census() {  # $1 root
  { echo "chunk_dir,host,class"
    for X in $(find "$1" -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
      echo "${X#"$1"/},$(head -1 "$X/NODE" 2>/dev/null | awk '{print $2}'),\"$(grep -o 'AvailableFeatures=[^ ]*' "$X/NODE" 2>/dev/null | cut -d= -f2)\""
    done; } > "$1/class_census.csv"
  echo "$1 census: $(tail -n +2 "$1/class_census.csv" | cut -d'"' -f2 | sort | uniq -c | tr '\n' ' ')"
}
for RUNG in rs10 rs1; do
  TF=cluster/trees_xgbtop_tasks_$RUNG.txt
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
