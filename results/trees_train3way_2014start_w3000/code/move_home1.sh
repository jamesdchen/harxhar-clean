#!/bin/bash
# Note to addendum E (CARC login node): the Xeon 4116 pool's not-yet-started work moves off /scratch1 (failing Lustre
# mounts on many d-nodes): code, task file, designs, claims, pieces and logs on /home1 (NFS).  Running tasks and written
# pieces are not touched.
#   1. the failed tasks of the old Xeon 4116 pool: lines without a written piece released (release_failed.sh logic);
#      every node a task of it failed on is excluded from the new pool;
#   2. the old pool's pending tasks cancelled (never started);
#   3. the unclaimed lines of 5-348 split at their median B: the /scratch1 pools (EPYC, E5-2640 v4) keep the lines
#      below B, the new /home1 pool gets B-348.  Each line >= B is reserved in the /scratch1 claim namespace (mkdir,
#      marker 'reserved_home1') so no /scratch1 pool takes it; a line >= B already claimed there is marked
#      'taken_scratch1' in the /home1 namespace so the /home1 pool skips it;
#   4. the /home1 pool submitted (Xeon 4116, 8 threads, 20 h, main), and the held final merge made to wait on it.
# The /home1 pieces are copied (md5) into /scratch1/jc_905/trees3way/runs/pieces before the final merge is released.
# usage: bash move_home1.sh <commit sha> <old Xeon 4116 pool id> <held merge id> <EPYC pool id> <E5-2640 v4 pool id>
set -euo pipefail
C=$1
OLD=$2
M4=$3
PE=$4
P26=$5
SB=/scratch1/jc_905/trees3way
HB=/home1/jc_905/trees3way
CLS=$SB/runs/claims_w3rf_full
CLH=$HB/claims_w3rf_full
if grep -q "^move_home1 " $SB/jobids.txt; then
  echo "ABORT: already done"
  exit 3
fi
S=$(git -C $SB/repo rev-parse --short=7 "$C^{commit}")
HC=$HB/code_$S
if [ ! -d $HC ]; then
  mkdir -p $HC
  P=$(git -C $SB/repo ls-tree --name-only "$C" | grep -v '^results$')
  git -C $SB/repo archive "$C" $P results/trees_train3way_2014start_w3000 | tar -x -C $HC
  git -C $SB/repo rev-parse "$C^{commit}" > $HC/COMMIT
fi
mkdir -p $HB/runs/pieces $HB/runs/tasks $HB/logs $CLH
cp $SB/runs/tasks/w3rf_full.txt $HB/runs/tasks/w3rf_full.txt
cmp $SB/runs/tasks/w3rf_full.txt $HB/runs/tasks/w3rf_full.txt
(cd $HB/designs && for d in */_work/design_*; do (cd $d && md5sum -c --quiet MD5SUMS); done) && echo "/home1 designs md5 ok"
echo "code $HC ($(cat $HC/COMMIT)), $(du -sh $HC | cut -f1); /home1 use: $(du -sh $HB | cut -f1)"
# 1. failed tasks of the old pool
bad=$(sacct -j $OLD -X -n -P -s F,NF,OOM,TO,BF -o NodeList | sort -u | paste -sd,)
BAD=$(printf "%s\n" d05-06 d06-25 d11-23 d11-24 d18-17 d18-26 d18-28 d18-29 ${bad//,/ } | sort -u | paste -sd,)
while IFS='|' read -r jid st ex node; do
  k=$(grep -l -E "^[^ ]+ ${jid} " $CLS/*/claim 2> /dev/null | head -1 | xargs -r dirname | xargs -r basename)
  if [ -z "$k" ]; then echo "$jid ($st $ex, $node): no claim"; continue; fi
  pc=$SB/runs/pieces/$(sed -n "${k}p" $SB/runs/tasks/w3rf_full.txt | awk '{split($4,a,"/"); print $1 "_" $2 "_" $3 "_" a[1] "of" a[2]}')
  if [ -e $CLS/$k/done ] || [ -e $pc.parquet ] || [ -e $pc.json ] || [ -e $pc.ckpt.json ]; then  # 2026-10-24 fix: was `ls a b c`, false when any one is absent
    echo "$jid line $k: a piece exists or done -> left claimed"
    continue
  fi
  rm -f $pc.part
  echo "$(date +%F_%T) released line $k (claim of $jid on $node, $st $ex; no piece written): $(cat $CLS/$k/claim 2> /dev/null)" >> $SB/runs/claims_released.log
  rm -rf "$CLS/$k"
  echo "$jid line $k on $node: released"
done < <(sacct -j $OLD -X -n -P -s F,NF,OOM,TO,BF -o JobID,State,ExitCode,NodeList)
# 2. the old pool's pending tasks
scancel --state=PENDING $OLD
sleep 2
echo "old pool $OLD: running $(squeue -h -r -j $OLD -t R | wc -l), pending $(squeue -h -r -j $OLD -t PD | wc -l)"
# 3. split the unclaimed lines
U=()
for ((k = 5; k <= 348; k++)); do [ -e $CLS/$k ] || U+=("$k"); done
n=${#U[@]}
B=${U[$((n / 2))]}
echo "unclaimed lines in 5-348: $n; boundary B = $B"
res=0
tak=0
for ((k = B; k <= 348; k++)); do
  if mkdir $CLS/$k 2> /dev/null; then
    echo "$(date +%F_%T) reserved for the /home1 Xeon 4116 pool ($CLH)" > $CLS/$k/reserved_home1
    res=$((res + 1))
  else
    mkdir -p $CLH/$k
    echo "$(date +%F_%T) claimed in $CLS: $(cat $CLS/$k/claim 2> /dev/null)" > $CLH/$k/taken_scratch1
    tak=$((tak + 1))
  fi
done
echo "lines $B-348: $res reserved for /home1, $tak already claimed on /scratch1"
# 4. the /home1 pool
J=$HC/results/trees_train3way_2014start_w3000/code
nt=$((res + 10))
p=$(cd $HC && sbatch --parsable --account=pollok_1603 -p main --constraint=xeon-4116 --exclude=$BAD -J t3t_p41h -c 8 \
  --mem=16G -t 20:00:00 --array=1-$nt%40 -o "$HB/logs/%x.%A_%a.out" \
  --export=ALL,CODE=$HC,T3T_BASE=$HB,TASKFILE=$HB/runs/tasks/w3rf_full.txt,LO=$B,HI=348,CLAIMS=$CLH,DESIGNS=$HB/designs \
  $J/pool.sbatch)
scontrol update JobId=$M4 Dependency=afterany:13372432:$PE:$OLD:$P26:$p:13372427:13371788:13371790 && echo "m4 $M4 waits on $p too"
echo "move_home1 $(date +%F_%T) code=$HC old=$OLD (pending cancelled) bad=$BAD B=$B reserved=$res taken=$tak p41h=$p ($nt tasks %40, /home1)" | tee -a $SB/jobids.txt
