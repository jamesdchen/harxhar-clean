#!/bin/bash
# C1 FEATURE IMPORTANCE of the 16:00-bar forecast re-run on the de-duplicated per-bar design
# (checklist I7): the untuned per-bar LightGBM / XGBoost / random forest refits of
# experiments/feature_importance_1530_trees.py (MDI, split count, permutation on the causal
# held-out tail, TreeSHAP, noise probes), 147 refits x 3 buckets x 3 models, in two variants:
# de-dup + per-window mask (the user's decision, commit f9a19b6) and de-dup without it.
#   bash cluster/slurm/submit_featimp_dedup.sh canary   pack 0 (cluster/featimp_dedup_tasks_canary.txt:
#        every bucket x model x variant, refits 0..1 (baseline 0..2) incl. the probe fit; CHECK_GATE
#        reports the refit-vs-stored forecast gaps)
#   bash cluster/slurm/submit_featimp_dedup.sh fleet    packs 1 (mask) and 2 (nomask) of
#        cluster/featimp_dedup_tasks.txt (45 lines each), refusing without the canary's flag, and
#        the collector held afterany
#   bash cluster/slurm/submit_featimp_dedup.sh linear   the slow all_features LASSO drop-column refits
#        (cluster/featimp_dedup_linear_tasks.txt; one pack, a pool of single-threaded workers)
# SIZING. Every fit is single-threaded.  One-thread seconds per refit of the stored T10 runs
# (fit + TreeSHAP; walk_sec / 147): all_features RF 31, LightGBM 6.1, XGBoost 3.9; live_feasible
# RF 9.4, LightGBM 1.7, XGBoost 1.2; baseline <= 1.2.  A refit here adds the probe fit on every
# third refit (+1/3) and the permuted tails (<= 790 units x 2 schemes x 10 draws x 10 rows),
# so an all_features RF line (21 refits) is ~1,000 s, every other line <= ~350 s; a pack's 45
# lines are ~14,500 core-s (unmasked; the masked fits see fewer columns) through 20 workers:
# ~20 min, bounded by the longest line.  2 h limit.
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_featimp_dedup.sh canary|fleet|linear
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:-}"
D=results/feature_importance_1530_dedup
DN=results/feature_importance_1530_dedup_nomask
mkdir -p logs "$D/_work/trees" "$DN/_work/trees" "$D/_work/linear"
CPUS=${CPUS:-20}
MEM=${MEM:-48G}
TIME=${TIME:-2:00:00}
PART=${PART:-main}
FLAG=$D/_work/trees/CANARY_OK
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing (the untuned tree campaign installs it)"; exit 1; }
for f in experiments/feature_importance_1530_trees.py src/models/window_mask.py \
         cluster/slurm/featimp_dedup_pack.sbatch cluster/slurm/featimp_dedup_collect.sbatch \
         cluster/featimp_dedup_tasks_canary.txt cluster/featimp_dedup_tasks.txt \
         "$D/_work/input_baseline.npz" "$D/_work/input_live_feasible.npz" "$D/_work/input_all_features.npz" \
         "$DN/_work/input_baseline.npz" "$DN/_work/input_live_feasible.npz" "$DN/_work/input_all_features.npz"; do
  [ -f "$f" ] || { echo "$f missing: run bash cluster/slurm/ship_featimp_dedup_carc.sh locally first"; exit 1; }
done
LOG=logs/submitted_featimp_dedup_carc.txt
case "$MODE" in
  canary)
    rm -f "$FLAG" "$FLAG".*
    NC=$(awk 'NF && $1 == 0' cluster/featimp_dedup_tasks_canary.txt | wc -l)
    CAN=$(sbatch --parsable -J fid_canary --partition="$PART" --cpus-per-task="$NC" --mem="$MEM" --time=1:00:00 \
          --export=ALL,TASKFILE=cluster/featimp_dedup_tasks_canary.txt,PACK=0,WRITE_FLAG=$FLAG,CHECK_GATE=1 \
          cluster/slurm/featimp_dedup_pack.sbatch)
    echo "$(date +%F_%T) canary=$CAN ($NC lines)" | tee -a "$LOG"
    ;;
  fleet)
    [ -f "$FLAG" ] || { echo "$FLAG missing: the canary has not passed"; exit 1; }
    rm -f "$D/_work/trees/FLEET_DONE"
    NP=$(awk 'NF {print $1}' cluster/featimp_dedup_tasks.txt | sort -n | tail -1)
    FL=$(sbatch --parsable -J fid_fleet --array=1-"$NP" --partition="$PART" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
         --export=ALL,TASKFILE=cluster/featimp_dedup_tasks.txt,CANARY_FLAG=$FLAG cluster/slurm/featimp_dedup_pack.sbatch)
    CO=$(sbatch --parsable -J fid_collect --dependency=afterany:"$FL" \
         --export=ALL,TASKFILE=cluster/featimp_dedup_tasks.txt cluster/slurm/featimp_dedup_collect.sbatch)
    echo "$(date +%F_%T) fleet=$FL (packs 1-$NP) collect=$CO" | tee -a "$LOG"
    ;;
  linear)
    for f in experiments/feature_importance_1530.py experiments/feature_importance_1530_dedup.py \
             experiments/model_diagnostics_1530.py cluster/slurm/featimp_dedup_linear.sbatch \
             cluster/featimp_dedup_linear_tasks.txt "$D/_work/linear_captures/capture_bar1600_all_features_reclasso.npz" \
             results/linear_subsection_dedup/arms_hoffman2/all_features/reclasso/tw2000/results_bar1600.csv; do
      [ -f "$f" ] || { echo "$f missing: run bash cluster/slurm/ship_featimp_dedup_carc.sh linear locally first"; exit 1; }
    done
    NL=$(grep -c . cluster/featimp_dedup_linear_tasks.txt)
    NCL=$(( NL < CPUS ? NL : CPUS ))
    LI=$(sbatch --parsable -J fid_linear --partition="$PART" --cpus-per-task="$NCL" --mem="$MEM" --time="$TIME" \
         --export=ALL,TASKFILE=cluster/featimp_dedup_linear_tasks.txt cluster/slurm/featimp_dedup_linear.sbatch)
    echo "$(date +%F_%T) linear=$LI ($NL refits, $NCL workers)" | tee -a "$LOG"
    ;;
  *)
    echo "usage: bash cluster/slurm/submit_featimp_dedup.sh canary|fleet|linear"
    exit 1
    ;;
esac
squeue -u "$USER" -h -o "%.12i %.14j %.3t %.10M %.4C %R" | grep -E "fid_" | head -8 || true
