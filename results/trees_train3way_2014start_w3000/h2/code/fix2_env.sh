#!/bin/bash
# Hoffman2 login node (bash -l), once (2026-10-24 ~04:15 PDT): env.sh failed (no ENV_OK: daal4py needs distutils =
# setuptools on Python 3.12), so every gate task that started exited "no env" without a piece.  This
#   1. puts a user hold on every pending t3t gate array (none starts before the env is fixed);
#   2. submits t3t_envfix (h2/code/envfix.sh: setuptools 84.1.0 = CARC's, the checks, ENV_OK);
#   3. resubmits, at 8 threads and held on t3t_envfix, the lines of every finished gate task whose log says "no env"
#      (t3t_gRL_<tag>: the ALL48 full-set every-row line, h_rt 8 h; t3t_gRS_<tag>: the others, h_rt 2 h);
#   4. holds the pending arrays on t3t_envfix too and releases their user hold; re-holds t3t_gsum on every gate job;
#   5. relays CARC's assignment of 04:05 (RETURNED 240-348, ASSIGNED 349-368) into runs/tasks/h2_assignment.txt and
#      rebuilds runs/tasks/h2_w3rf_full.txt from it.
set -uo pipefail
B=/u/scratch/j/jamesdc1/t3t
cd $B
echo "== $(hostname) $(date)"
PEND=$(qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t_g(20|8)/ && $5 ~ /qw/ {print $1}' | sort -u | paste -sd,)
echo "== user hold on pending gate arrays: $PEND"
[ -n "$PEND" ] && qhold -h u $PEND
echo "== env.log (end)"; tail -n 25 logs/env.log
grep -q '^setuptools==' jobs/req_skx.txt || echo setuptools==84.1.0 >> jobs/req_skx.txt
echo "== req_gb.txt / req_skx.txt"; cat jobs/req_gb.txt; echo; cat jobs/req_skx.txt
ef=$(qsub -terse -N t3t_envfix -o $B/logs/envfix.log jobs/envfix.sh | cut -d. -f1)
echo "t3t_envfix $ef"
echo "== gate task logs that exited 'no env' (finished tasks only)"
qstat -u jamesdc1 -g d > h2/.qstat_gd2
NEW=""
while read -r t arch rx; do
  : > lines/gRL_$t.txt
  : > lines/gRS_$t.txt
  for kind in g20L g20 g8; do
    for lg in logs/${kind}_$t.*.log; do
      [ -f "$lg" ] || continue
      grep -q "no env" "$lg" || continue
      k=$(basename "$lg" .log); k=${k##*.}
      ln=$(sed -n "${k}p" lines/${kind}_$t.txt)
      if echo "$ln" | grep -q -- "--arm ALL48 --est rf_skx --set all_features .*--every-row"; then echo "$ln" >> lines/gRL_$t.txt; else echo "$ln" >> lines/gRS_$t.txt; fi
      echo "  $t $kind task $k (no env) -> 8 threads: ${ln:0:130}"
      mv "$lg" "$lg.noenv"
    done
  done
  for kind in L S; do
    f=lines/gR${kind}_$t.txt
    n=$(grep -c . $f)
    [ "$n" -gt 0 ] || continue
    hrt=2:00:00
    [ $kind = L ] && hrt=8:00:00
    nj=$(qsub -terse -N t3t_gR${kind}_$t -hold_jid $ef -t 1-$n -pe shared 8 -l "h_rt=$hrt,h_data=2G,arch=$arch" \
      -o "$B/logs/gR${kind}_$t.\$TASK_ID.log" -v "LINES=$B/$f,WANT=$rx" jobs/line.sh | cut -d. -f1)
    echo "  $t: t3t_gR${kind}_$t $nj ($n lines, h_rt $hrt, held on $ef)"
    NEW="$NEW,$nj"
  done
done < h2/tags.txt
for j in ${PEND//,/ }; do qalter -hold_jid $ef $j > /dev/null && echo "  $j held on $ef"; done
[ -n "$PEND" ] && qrls -h u $PEND
GSUM=$(grep '^gates ' jobids.txt | tail -1 | grep -o -E 'gsum=[0-9]+' | cut -d= -f2)
sleep 2
LIVE=$(qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t_g(20|8|R)/ {print $1}' | sort -u | paste -sd,)
qalter -hold_jid "$ef,$LIVE" $GSUM
echo "fix2 $(date +%F_%T) envfix=$ef rerun=${NEW#,} pending_held_on_envfix=$PEND gsum=$GSUM hold=$ef,$LIVE" | tee -a jobids.txt
# 5. the assignment of 04:05 (CARC runs/tasks/h2_assignment.txt, rows; relayed)
cp runs/tasks/h2_assignment.txt runs/tasks/h2_assignment.txt.bak_$(date +%H%M)
{
  echo "# relayed from CARC /scratch1/jc_905/trees3way/runs/tasks/h2_assignment.txt (mtime 02:13 PDT), read 04:05 PDT; rows only"
  echo "RETURNED w3rf_full.txt 240-348 ALL48 rf_skx all_features 20"
  echo "ASSIGNED w3rf_full.txt 349-368 ALL48 rf_skx all_features 20"
} > runs/tasks/h2_assignment.txt
: > runs/tasks/h2_w3rf_full.txt
awk '$1 == "ASSIGNED" && $2 == "w3rf_full.txt" {print $3}' runs/tasks/h2_assignment.txt | while IFS=- read -r lo hi; do
  awk -v lo="$lo" -v hi="$hi" 'NR >= lo && NR <= hi { split($4, p, "/"); if (p[1] + 0 != NR) exit 1; print }' carc/runs/tasks/w3rf_full.txt >> runs/tasks/h2_w3rf_full.txt
done
echo "assignment: $(grep -v '^#' runs/tasks/h2_assignment.txt | paste -sd';'); task file $(grep -c . runs/tasks/h2_w3rf_full.txt) lines"
qstat -j $GSUM 2>/dev/null | grep -E "jid_predecessor_list:"
echo "== my jobs after"; qstat -u jamesdc1
echo "== done $(date)"
