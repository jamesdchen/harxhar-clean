#!/bin/bash
# Completes submit_e.sh after its Xeon 4116 pool (-p main,oneweek, array of 200) was refused by the oneweek QOS's
# MaxSubmitPU of 50 (QOSMaxSubmitJobPerUserLimit; a multi-partition job counts against it), which aborted the script
# after the gates and the EPYC pool (held) were submitted and before any cancel: the Xeon 4116 pool on main only
# (array of 150), the E5-2640 v4 pool as an array of 20 (oneweek; held), then the pending tasks of
# 13372432 / 13372801 cancelled, the pools released only if none is left pending, the held final merge's dependency
# extended to the pools, and the job ids recorded.
# usage: bash submit_e_fix.sh <code dir> <held merge job id>
set -uo pipefail
CODE=$1
M4=$2
BASE=/scratch1/jc_905/trees3way
T=$BASE/runs/tasks
J=$CODE/results/trees_train3way_2014start_w3000/code
CL=$BASE/runs/claims_w3rf_full
id() { squeue -u jc_905 -h -o "%F %j" | awk -v n="$1" '$2 == n {print $1}' | sort -u | head -1; }
a1=$(id t3t_ge41a); b1=$(id t3t_ge41b); c1=$(id t3t_ge41c); a2=$(id t3t_ge26a); b2=$(id t3t_ge26b); c2=$(id t3t_ge26c)
pE=$(id t3t_pE)
echo "found: ge41a=$a1 ge41b=$b1 ge41c=$c1 ge26a=$a2 ge26b=$b2 ge26c=$c2 pE=$pE"
for v in "$a1" "$b1" "$c1" "$a2" "$b2" "$c2" "$pE"; do
  [ -n "$v" ] || { echo "ABORT: a job of submit_e.sh is missing"; exit 3; }
done
mkdir -p $CL
p41=$(cd $CODE && sbatch --parsable --account=pollok_1603 -H -p main --constraint=xeon-4116 -J t3t_p41 -c 8 \
  --mem=16G -t 24:00:00 --array=1-150%40 --dependency=afterok:$c1 -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,CODE=$CODE,TASKFILE=$T/w3rf_full.txt,LO=5,HI=348,CLAIMS=$CL $J/pool.sbatch) || p41=""
echo "p41=${p41:-REFUSED}"
[ -n "$p41" ] || { echo "ABORT: the Xeon 4116 pool was refused; nothing cancelled"; exit 3; }
p26=$(cd $CODE && sbatch --parsable --account=pollok_1603 -H -p oneweek --constraint=xeon-2640v4 -J t3t_p26 -c 20 \
  --mem=24G -t 24:00:00 --array=1-20%10 --dependency=afterok:$c2 -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,CODE=$CODE,TASKFILE=$T/w3rf_full.txt,LO=5,HI=348,CLAIMS=$CL $J/pool.sbatch) || p26=""
echo "p26=${p26:-REFUSED}"
scancel --state=PENDING 13372432 13372801
sleep 3
left=$(squeue -h -r -j 13372432,13372801 -t PD -o %i | wc -l)
echo "13372432 running: $(squeue -h -r -j 13372432 -t R -o %i | paste -sd' '); pending left: $left"
if [ "$left" -ne 0 ]; then
  echo "ABORT: pending tasks remain; the pools stay held"
  exit 3
fi
scontrol release $pE $p41 ${p26:+$p26}
scontrol update JobId=$M4 Dependency=afterany:13372432:$pE:$p41${p26:+:$p26}:13372427:13371788:13371790 && echo "m4 $M4 waits on the pools too"
scontrol show job $M4 | grep -o -E "JobState=[A-Z]+|Reason=[A-Za-z]+" | paste -sd' '
echo "phase_e $(date +%F_%T) code=$CODE ge41a=$a1 ge41b=$b1 ge41c=$c1 ge26a=$a2 ge26b=$b2 ge26c=$c2 pE=$pE p41=$p41 p26=${p26:-refused} claims=$CL lines 5-348; cancelled pending tasks of 13372432 and 13372801" | tee -a $BASE/jobids.txt
squeue -u jc_905 -h -o "%j %T %r" | grep -E "t3t_(ge|p[E0-9])" | sort | uniq -c
