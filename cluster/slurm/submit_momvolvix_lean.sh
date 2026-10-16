#!/bin/bash
# The Optuna per-bar trees on TWO new buckets in one fleet (user, 2026-10-06), from the isolated root
# /scratch1/jc_905/harxhar-momvolvix (cluster/slurm/ship_momvix_optuna_carc.sh with
# PREFIX="momvolvix leanvolvix"): moments_vol_vix (moments_vix + sumvolume, results root
# results/linear_subsection_trees_optuna_mask_momvolvix) and lean_vol_vix (sumret, sumabsret, sumret4,
# sumautocov, sumvolume, vix; ..._leanvolvix).  For each bucket, cluster/slurm/submit_momvix_optuna.sh
#   start   gate (stored live_feasible points / refits, bit for bit) + canary + stage-1 fleet held on both
#   finish  stage-1 merge (afterany on that bucket's stage-1 arrays) -> stage 2 -> stage-2 merge
# Must run in a clean shell (no conda env active; the submit script refuses otherwise).
#   cd /scratch1/jc_905/harxhar-momvolvix && bash cluster/slurm/submit_momvolvix_lean.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
for spec in "momvolvix moments_vol_vix mvv" "leanvolvix lean_vol_vix lvv"; do
  read -r P B J <<< "$spec"
  PREFIX=$P BUCKET=$B JP=$J bash cluster/slurm/submit_momvix_optuna.sh start
  S1=$(grep -o "s1:.*class" "logs/submitted_${P}_start.txt" | grep -oE ":[0-9]+\(" | tr -d ':(' | paste -sd: -)
  [ -n "$S1" ] || { echo "no stage-1 job ids for $P"; exit 1; }
  PREFIX=$P BUCKET=$B JP=$J DEP="afterany:$S1" bash cluster/slurm/submit_momvix_optuna.sh finish
done
