#!/bin/bash
# The close-only window sweep on CARC Discovery (login node; RULE.txt section 7), one chain:
#   code     git archive of <commit> (tracked tree without results/, plus this run's and the linear three-way's results
#            directories) -> $BASE/code_<sha7>
#   stage    the six W designs copied to $DES (md5) + checks -> $BASE/design_check.json
#   lin3k    linear W = 3,000 (derive / pretune / select / walks / export) + gate L      (afterok stage)
#   linW     linear W = 1,000, 1,500, 2,000, 2,500, 3,500                               (afterok lin3k)
#   linS     linear 16:00 scores -> $BASE/scores/linear                                 (afterok linW)
#   tgate    trees gate T (W = 3,000, 20 origins of each class and set) + reuse          (afterok stage)
#   twgb     tree walks LightGBM / XGBoost, 20 lines (W ascending), EPYC 7513 / 7542, %8 (afterok tgate)
#   twrf     tree walks, the forest, 10 lines, EPYC 7513 / 7542 or Xeon 4116, %4         (afterok tgate)
#   tfin     merge + CPU check + tree 16:00 scores -> $BASE/scores/trees                (afterany twgb, twrf)
# Refuses a second submission (jobids.txt).  usage: bash submit.sh <commit>
set -euo pipefail
C=$1
BASE=/scratch1/jc_905/closewin
BR=insurance-selling-2026-09-19-e34-closeonly-wsweep
if [ -f $BASE/jobids.txt ] && grep -q "^chain " $BASE/jobids.txt; then
  echo "ABORT: the chain was already submitted"
  exit 3
fi
mkdir -p $BASE/logs $BASE/runs/linear $BASE/runs/trees/tasks /home1/jc_905/closewin/designs
cd $BASE
if [ ! -d repo/.git ]; then
  git clone -q --reference-if-able /scratch1/jc_905/trees3way/repo --no-checkout \
    https://github.com/jamesdchen/harxhar-clean.git repo
fi
git -C repo fetch -q origin $BR
S=$(git -C repo rev-parse --short=7 "$C^{commit}")
CODE=$BASE/code_$S
if [ ! -d $CODE ]; then
  mkdir -p $CODE
  P=$(git -C repo ls-tree --name-only "$C" | grep -v '^results$')
  # shellcheck disable=SC2086
  git -C repo archive "$C" $P results/closeonly_wsweep_2014start results/linear_train3way_2014start_w3000 | tar -x -C $CODE
  git -C repo rev-parse "$C^{commit}" > $CODE/COMMIT
fi
echo "code $CODE at $(cat $CODE/COMMIT), $(du -sh $CODE | cut -f1)"
J=$CODE/results/closeonly_wsweep_2014start/code
# tree task files, cheapest first: W ascending; LightGBM before XGBoost; the HAR set first
TG=$BASE/runs/trees/tasks/walk_gb.txt
TR=$BASE/runs/trees/tasks/walk_rf.txt
: > $TG
: > $TR
for W in 1000 1500 2000 2500 3500; do
  for e in lgbm xgb; do
    for s in baseline all_features; do echo "CLOSE_w$W $e $s 1/1" >> $TG; done
  done
  for s in baseline all_features; do echo "CLOSE_w$W rf_skx $s 1/1" >> $TR; done
done
EP="epyc-7513|epyc-7542"
BAD=d05-06,d18-17,d18-26,d18-28,d18-29 # the nodes with failing /scratch1 mounts (trees three-way, addendum E note)
A=(--parsable --account=pollok_1603 --partition=main)
L=$BASE/logs
st=$(sbatch "${A[@]}" --constraint="$EP" -J cw_stage -c 2 --mem=8G -t 1:00:00 -o "$L/%x.%j.out" \
  --export=ALL,CODE=$CODE $J/stage.sbatch)
la=$(sbatch "${A[@]}" --constraint="$EP" -J cw_lin3k -c 4 --mem=8G -t 1:00:00 --dependency=afterok:$st \
  -o "$L/%x.%j.out" --export=ALL,CODE=$CODE,WLIST=3000,GATE=1 $J/lin.sbatch)
lb=$(sbatch "${A[@]}" --constraint="$EP" -J cw_linW -c 4 --mem=8G -t 2:00:00 --dependency=afterok:$la \
  -o "$L/%x.%j.out" --export=ALL,CODE=$CODE,WLIST=1000:1500:2000:2500:3500 $J/lin.sbatch)
ls_=$(sbatch "${A[@]}" -J cw_linS -c 1 --mem=16G -t 1:00:00 --dependency=afterok:$lb --exclude=$BAD \
  -o "$L/%x.%j.out" --export=ALL,CODE=$CODE,FAM=linear $J/fin.sbatch)
tg=$(sbatch "${A[@]}" --constraint="$EP" -J cw_tgate -c 8 --mem=16G -t 1:00:00 --dependency=afterok:$st \
  -o "$L/%x.%j.out" --export=ALL,CODE=$CODE $J/tgate.sbatch)
tw=$(sbatch "${A[@]}" --constraint="$EP" -J cw_twgb -c 8 --mem=8G -t 6:00:00 --array="1-$(wc -l < $TG)%8" \
  --dependency=afterok:$tg -o "$L/%x.%A_%a.out" --export=ALL,CODE=$CODE,TASKFILE=$TG $J/twalk.sbatch)
tr=$(sbatch "${A[@]}" --constraint="$EP|xeon-4116" --exclude=$BAD -J cw_twrf -c 8 --mem=8G -t 6:00:00 \
  --array="1-$(wc -l < $TR)%4" --dependency=afterok:$tg -o "$L/%x.%A_%a.out" \
  --export=ALL,CODE=$CODE,TASKFILE=$TR $J/twalk.sbatch)
tf=$(sbatch "${A[@]}" -J cw_tfin -c 1 --mem=16G -t 1:00:00 --dependency=afterany:$tw:$tr --exclude=$BAD \
  -o "$L/%x.%j.out" --export=ALL,CODE=$CODE,FAM=trees $J/fin.sbatch)
echo "chain $(date +%F_%T) code=$S stage=$st lin3k=$la linW=$lb linS=$ls_ tgate=$tg twgb=$tw twrf=$tr tfin=$tf" |
  tee -a $BASE/jobids.txt
sleep 2
squeue -u jc_905 -h -o "%.16i %.10j %.8T %.4C %R" | grep " cw_" | head -20
