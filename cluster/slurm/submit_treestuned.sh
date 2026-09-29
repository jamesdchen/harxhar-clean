#!/bin/bash
# Per-bar TREE forecasts (LightGBM, XGBoost, random forest) with CAUSAL
# PERIODIC HYPERPARAMETER TUNING, beside the untuned per-bar trees and the
# per-bar linear arms: buckets live_feasible, all_features, baseline x the 13
# RTH bars (bar1000..bar1600) at tw 2000, global lags, production HAR ladder
# (specs/causal_tune_trees_tuned.py, which reuses the setup of
# specs/causal_tune_trees.py; its pool workers run
# specs/causal_tune_trees_tuned_jobs.py).
# Each arm (bucket x model x bar) is split into TIME CHUNKS of CHUNK_ROWS = 250
# forecast rows (= TUNE_PER, one tuning period per chunk), so one array task
# runs one chunk of one arm; every task runs a process pool of POOL_CPUS
# workers (single-threaded fits; the spec reads SLURM_CPUS_PER_TASK as the pool
# size).  Canary first: ONE line, chunk 0 of live_feasible x lgbm,xgb,rf x the
# 15:30-16:00 bar (all three model paths incl. RF's shap import, the tuning
# loop and the pool, on real data); it writes $ROOT/CANARY_OK only if all three
# chunk-arms succeed.  The GB fleet (lgbm, xgb) and the RF fleet are held on it
# with afterok and refuse to run unless CANARY_OK exists; the canary's three
# chunk-arms carry DONE, so the fleets skip them.  The scorer is held on both
# fleets with afterany: it first merges every arm whose chunks are all DONE
# (experiments/reduce_trees_tuned_chunks.py), lists the rest (NOT DONE / NOT
# MERGED), then scores what merged.
# Task order in the fleet files: live_feasible first, then all_features, then
# baseline (within a bucket: lgbm bars, then xgb bars; within a bar its chunks
# in time order).  live_feasible and all_features are the arms the comparison
# needs first; if the fleet is cut short, baseline is what is lost.
#
# Task line: "<bucket> <models> <tw> <seg> <chunk> <start> <end> <halo>".
# W = TRAIN_WIN = 2000 rows (one row per session at a one-bar segment).  Chunk
# k of a bar: start = W + CHUNK_ROWS*k, end = start + CHUNK_ROWS, except the
# LAST chunk of the bar, whose end = -1 (to the end of the series); halo = W
# always (the training window the chunk's first forecast needs).  Chunks per
# bar = ceil(n_oos / CHUNK_ROWS), n_oos = forecast rows of the bar (identical
# for every bucket and model; from the untuned CARC results):
#   bar1000 1619  bar1030 1619  bar1100 1619  bar1130 1621  bar1200 1560
#   bar1230 1560  bar1300 1560  bar1330 1499  bar1400 1487  bar1430 1479
#   bar1500 1477  bar1530 1475  bar1600 1469
#   -> 7 chunks for bar1000..bar1300, 6 for bar1330..bar1600 = 85 per (model,
#   bucket); 85 x 3 models x 3 buckets = 765 chunk-arms (GB 510 lines, RF 255).
# The three task files were generated once, locally, from the repo root with:
# --- task-file generator ---
#   python - <<'EOF'
#   import math
#   W = 2000  # TRAIN_WIN rows = the halo every chunk carries
#   CHUNK_ROWS = 250  # = TUNE_PER: one tuning period per chunk
#   N_OOS = {"bar1000": 1619, "bar1030": 1619, "bar1100": 1619, "bar1130": 1621,
#            "bar1200": 1560, "bar1230": 1560, "bar1300": 1560, "bar1330": 1499,
#            "bar1400": 1487, "bar1430": 1479, "bar1500": 1477, "bar1530": 1475,
#            "bar1600": 1469}
#   BUCKETS = ["live_feasible", "all_features", "baseline"]  # needed-first order
#   def chunks(seg):
#       n_chunks = math.ceil(N_OOS[seg] / CHUNK_ROWS)
#       for k in range(n_chunks):
#           start = W + CHUNK_ROWS * k
#           yield k, start, (-1 if k == n_chunks - 1 else start + CHUNK_ROWS)
#   def write(path, models):
#       with open(path, "w", newline="\n") as f:
#           for b in BUCKETS:
#               for m in models:
#                   for seg in N_OOS:
#                       for k, s, e in chunks(seg):
#                           f.write(f"{b} {m} {W} {seg} {k} {s} {e} {W}\n")
#   write("cluster/treestuned_tasks_gb.txt", ["lgbm", "xgb"])
#   write("cluster/treestuned_tasks_rf.txt", ["rf"])
#   k, s, e = next(chunks("bar1600"))
#   with open("cluster/treestuned_tasks_canary.txt", "w", newline="\n") as f:
#       f.write(f"live_feasible lgbm,xgb,rf {W} bar1600 {k} {s} {e} {W}\n")
#   EOF
# --- end task-file generator ---
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_treestuned.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results

# TIMINGS: from the local smoke (2026-09-24, experiments/gate_trees_tuned.py,
# live_feasible bar1600, OOS rows 1000..1269 = 2 tuning points + 27 refits, pool
# of 6 on a laptop; mean seconds per single-threaded fit, contended):
#   lgbm  candidate 12.5 (early-stopped, 100..871 rounds)  refit 46.7 (chosen 40 leaves,
#         867 rounds)   walk 431 s
#   xgb   candidate 11.1 (early-stopped, up to 1287 rounds) refit  9.2   walk 214 s
#   rf    candidate  8.5                                     refit  3.1 (+27 QLIKE-rule
#         refits 7.5 s: the rules disagreed)                 walk 188 s
# Laptop / CARC per-core speed (untuned 1-thread refit, same design): lgbm 3.1x,
# xgb 3.8x, rf 3.3x.  Scaled to CARC with the untuned bucket ratios (all_features
# 3.1-3.8x live_feasible, baseline 0.11x): a chunk-task is 1-5 min of wall at 8
# workers (mean), ~15-20 min in the worst case (LightGBM all_features choosing
# 80 leaves x the 1568-round cap: ~22x the shipped refit cost); total 130-210
# core-h for the 765 chunk-tasks (the range = data prep 10 s as the CARC
# untuned logs show, or 60 s as locally).  The limits below carry >= 6x margin
# over that worst case; the canary runs three models one after another.
POOL_CPUS=${POOL_CPUS:-8}            # pool size = cores per task: 32 grid candidates / 8 = 4 waves per tuning point, 25 refits per chunk / 8 = 4 waves
CANARY_TIME=${CANARY_TIME:-3:00:00}  # one chunk of each of the three models, one after another
CANARY_MEM=${CANARY_MEM:-16G}        # POOL_CPUS workers, each with one 2000-row window (<= 10 MB) and its model
GB_TIME=${GB_TIME:-2:00:00}          # one chunk of one lgbm/xgb arm (worst case ~20 min)
GB_MEM=${GB_MEM:-16G}                # as the canary
GB_CONC=${GB_CONC:-60}               # GB array tasks running at once
RF_TIME=${RF_TIME:-2:00:00}          # one chunk of one rf arm (all_features ~3-5 min expected)
RF_MEM=${RF_MEM:-16G}                # as the canary
RF_CONC=${RF_CONC:-60}               # RF array tasks running at once
# Ceiling: (GB_CONC + RF_CONC) x POOL_CPUS = 120 x 8 = 960 cores at once; the
# QOS may cap the running jobs / cores lower (Slurm then just queues the rest).

CANARY=cluster/treestuned_tasks_canary.txt
GBT=cluster/treestuned_tasks_gb.txt
RFT=cluster/treestuned_tasks_rf.txt
ROOT=${RESULTS_ROOT:-results/linear_subsection_trees_tuned}
FLAG=$ROOT/CANARY_OK
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing: the untuned campaign installs it (bash cluster/slurm/ship_trees_carc.sh, locally)"; exit 1; }
for f in specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py specs/causal_tune_trees.py \
         specs/causal_tune_linear.py experiments/reduce_trees_tuned_chunks.py experiments/score_trees_tuned.py \
         cluster/slurm/treestuned_pack.sbatch cluster/slurm/treestuned_score.sbatch \
         "$CANARY" "$GBT" "$RFT"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_treestuned_carc.sh, locally)"; exit 1; }
done
mkdir -p "$ROOT"
rm -f "$FLAG"
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J tt_canary --cpus-per-task="$POOL_CPUS" --mem="$CANARY_MEM" --time="$CANARY_TIME" \
      --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$ROOT,WRITE_FLAG=$FLAG cluster/slurm/treestuned_pack.sbatch)
GB=$($SUBMIT --parsable -J tt_gb --array=1-"$(wc -l < "$GBT")"%"$GB_CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task="$POOL_CPUS" --mem="$GB_MEM" --time="$GB_TIME" \
      --export=ALL,TASKFILE=$GBT,RESULTS_ROOT=$ROOT,CANARY_FLAG=$FLAG cluster/slurm/treestuned_pack.sbatch)
RF=$($SUBMIT --parsable -J tt_rf --array=1-"$(wc -l < "$RFT")"%"$RF_CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task="$POOL_CPUS" --mem="$RF_MEM" --time="$RF_TIME" \
      --export=ALL,TASKFILE=$RFT,RESULTS_ROOT=$ROOT,CANARY_FLAG=$FLAG cluster/slurm/treestuned_pack.sbatch)
S=$($SUBMIT --parsable -J tt_score --dependency=afterany:"$GB":"$RF" \
      --export=ALL,RESULTS_ROOT=$ROOT cluster/slurm/treestuned_score.sbatch)
echo "canary=$CAN gb=$GB rf=$RF score=$S" | tee logs/submitted_treestuned_carc.txt
# the queue head: the four jobs just submitted (pending arrays show as one row each) + header
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -10
