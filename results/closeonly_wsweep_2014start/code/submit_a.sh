#!/bin/bash
# RULE.txt addendum A (login node): the code of <commit> archived to /home1, the reruns (linear W = 1,500 .. 3,500; the
# failed forest line), the pending forest tasks restricted to EPYC, the linear score job replaced, the tree finish job
# re-pointed.  Refuses a second submission (jobids.txt).  usage: bash submit_a.sh <commit>
set -euo pipefail
C=$1
BASE=/scratch1/jc_905/closewin
HB=/home1/jc_905/closewin
BR=insurance-selling-2026-09-19-e34-closeonly-wsweep
if grep -q "^addendumA " $BASE/jobids.txt; then
  echo "ABORT: addendum A already submitted"
  exit 3
fi
git -C $BASE/repo fetch -q origin $BR
S=$(git -C $BASE/repo rev-parse --short=7 "$C^{commit}")
CODE=$HB/code_$S
if [ ! -d $CODE ]; then
  mkdir -p $CODE
  P=$(git -C $BASE/repo ls-tree --name-only "$C" | grep -v '^results$')
  # shellcheck disable=SC2086
  git -C $BASE/repo archive "$C" $P results/closeonly_wsweep_2014start results/linear_train3way_2014start_w3000 |
    tar -x -C $CODE
  git -C $BASE/repo rev-parse "$C^{commit}" > $CODE/COMMIT
fi
echo "code $CODE at $(cat $CODE/COMMIT)"
# the walks' code is 62cfc8c's
for f in closewin.py closewin_linear.py closewin_trees.py linear_train3way.py linear_train3way_runner.py \
  pool_linear_pretune.py pooled_linear_rerun_2026_10_19_13.py trees_train3way.py pooled_trees_rerun_13.py; do
  cmp "$CODE/experiments/$f" "$BASE/code_62cfc8c/experiments/$f"
done
diff -rq "$CODE/src" "$BASE/code_62cfc8c/src" && diff -rq "$CODE/specs" "$BASE/code_62cfc8c/specs" &&
  diff -rq "$CODE/data" "$BASE/code_62cfc8c/data" && echo "walk code, src, specs, data = 62cfc8c's"
# the failed forest line left no piece
if ls $BASE/runs/trees/pieces/CLOSE_w1000_rf_skx_baseline_* 2> /dev/null; then
  echo "ABORT: a piece of the failed line exists"
  exit 3
fi
TR=$BASE/runs/trees/tasks/walk_rf_rerunA.txt
sed -n 1p $BASE/runs/trees/tasks/walk_rf.txt > $TR
cat $TR
J=$CODE/results/closeonly_wsweep_2014start/code
EP="epyc-7513|epyc-7542"
BAD=d05-06,d05-39,d18-17,d18-26,d18-28,d18-29
A=(--parsable --account=pollok_1603 --partition=main --constraint=$EP --exclude=$BAD)
L=$BASE/logs
for t in 9 10; do
  scontrol update JobId=13380322_$t Features="$EP" ExcNodeList=$BAD || echo "update 13380322_$t: failed"
done
lb=$(sbatch "${A[@]}" -J cw_linW2 -c 4 --mem=8G -t 2:00:00 -o "$L/%x.%j.out" \
  --export=ALL,CODE=$CODE,WLIST=1500:2000:2500:3500 $J/lin.sbatch)
tr=$(sbatch "${A[@]}" -J cw_twrf2 -c 8 --mem=8G -t 6:00:00 --array=1-1 -o "$L/%x.%A_%a.out" \
  --export=ALL,CODE=$CODE,TASKFILE=$TR $J/twalk.sbatch)
scancel 13380319
ls_=$(sbatch "${A[@]}" -J cw_linS2 -c 1 --mem=16G -t 1:00:00 --dependency=afterok:$lb -o "$L/%x.%j.out" \
  --export=ALL,CODE=$CODE,FAM=linear $J/fin.sbatch)
scontrol update JobId=13380323 Dependency=afterany:13380321:13380322:$tr Features="$EP" ExcNodeList=$BAD
echo "addendumA $(date +%F_%T) code=$S linW2=$lb twrf2=$tr linS2=$ls_ cancelled=13380319 tfin=13380323(re-pointed)" |
  tee -a $BASE/jobids.txt
scontrol show job 13380323 | grep -E "Dependency|Features|ExcNodeList"
scontrol show job 13380322_9 | grep -E "JobState|Features|ExcNodeList" | head -3
sleep 2
squeue -u jc_905 -h -o "%.16i %.10j %.8T %.4C %R" | grep " cw_"
