#!/bin/bash
# Pooled tree walks (2026-10-07): dump every COMPLETE arm (100 of 100 chunks) of $RESULTS_ROOT/chunks into
# $RESULTS_ROOT/yhat_pooltree_<est>_<set>.parquet with the bank's own dumper (experiments/dump_unif_yhat.py, unchanged:
# t / yhat / baseline / rv_raw, as yhat_tree00 / yhat_tree16), and list the chunk CPU models (class_census.csv).
#SBATCH --job-name=ptr_dump
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
export PYTHONPATH="$PWD:$PWD/experiments"
R="${RESULTS_ROOT:-results/pooled_trees_2026-10-07}"
# row pieces first: every chunk whose pieces tile it is merged into the bank's chunk file
python experiments/pooled_trees_merge.py --root "$R/chunks" || echo "MERGE reported a problem"
for D in "$R"/chunks/pooltree_*; do
  [ -d "$D" ] || continue
  # find, not ls: an arm with no chunk file yet (only pieces/) made `ls` fail, and under pipefail + set -e that ended
  # the whole dump after the first incomplete arm (12806435, 2026-10-08)
  A=$(basename "$D"); N=$(find "$D" -maxdepth 1 -name "chunk_*.npz" ! -name "*.part.npz" | wc -l)
  if [ "$N" = 100 ]; then
    if [ -f "$R/yhat_${A}.parquet" ] && [ "${REDUMP:-0}" != 1 ]; then echo "$A: table exists; kept"; continue; fi
    python experiments/dump_unif_yhat.py --arm "$A" --root "$R/chunks" --out "$R/yhat_${A}.parquet" 2>&1 | tail -1
  else
    echo "$A: $N of 100 chunks; not dumped"
  fi
done
{ echo "arm,chunk,host,cpu,threads,seconds"
  for F in "$R"/chunks/pooltree_*/chunk_*.node; do
    [ -f "$F" ] || continue
    A=$(basename "$(dirname "$F")"); C=$(basename "$F" .node)
    echo "$A,$C,$(awk '$1 == "host" {print $2}' "$F"),\"$(sed -n 's/^cpu //p' "$F")\",$(awk '$1 == "threads" {print $2}' "$F"),$(awk '$1 == "seconds" {print $2}' "$F")"
  done; } > "$R/class_census.csv"
echo "census: $(tail -n +2 "$R/class_census.csv" | cut -d'"' -f2 | sort | uniq -c | tr '\n' ' ')"
# row pieces (the forest): one line a piece, from its .node (host, CPU model, threads)
{ echo "arm,piece,host,cpu,threads"
  for F in "$R"/chunks/pooltree_*/pieces/chunk_*.node; do
    [ -f "$F" ] || continue
    A=$(basename "$(dirname "$(dirname "$F")")"); C=$(basename "$F" .node)
    echo "$A,$C,$(awk '$1 == "host" {print $2}' "$F"),\"$(sed -n 's/^cpu //p' "$F")\",$(awk '$1 == "threads" {print $2}' "$F")"
  done; } > "$R/class_census_pieces.csv"
echo "piece census: $(tail -n +2 "$R/class_census_pieces.csv" | cut -d'"' -f2 | sort | uniq -c | tr '\n' ' ')"
touch "$R/DUMP_DONE"
