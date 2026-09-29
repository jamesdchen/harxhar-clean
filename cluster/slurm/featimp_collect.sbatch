#!/bin/bash
# After the feature-importance fleet (held afterany): list every expected output
# of the fleet task file (one per line x model), say which are missing (with the
# last log line of their task), and write FLEET_DONE with the counts -- the file
# the local side polls for before pulling the outputs.
#SBATCH --job-name=featimp_collect
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
set -u
D=results/feature_importance_1530/_work/trees
N=0; OK=0
while read -r BUCKET MODELS K0 K1; do
  [ -z "${K1:-}" ] && continue
  for M in ${MODELS//,/ }; do
    N=$((N + 1))
    F=$(printf '%s/trees_%s_%s_%03d_%03d.npz' "$D" "$BUCKET" "$M" "$K0" "$K1")
    if [ -f "$F" ]; then OK=$((OK + 1)); else echo "MISSING $F"; fi
  done
done < "$TASKFILE"
echo "$OK of $N outputs present $(date)" | tee "$D/FLEET_DONE"
