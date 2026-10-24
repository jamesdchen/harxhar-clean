#!/bin/bash
# Hoffman2 login node (bash -l), once (2026-10-24 ~04:12 PDT): fix3_env.sh read two ids for the env job (its pattern
# also matched "pending_held_on_envfix=" of the fix2 record), so its qsub calls failed (nothing submitted) while its
# qalter calls held every queued gate job on 15261300 (t3t_envfix, still queued: it installs setuptools 84.1.0 and
# packaging 26.3).  This submits the four lines fix3 collected (lines/gR2{L,S}_<tag>.txt: gold6140 2, e5v4 1 + 1),
# 8 threads, held on 15261300; checks every queued gate job's hold; sets t3t_gsum's hold to 15261300 + every gate job.
set -uo pipefail
B=/u/scratch/j/jamesdc1/t3t
EF=15261300
GSUM=15261029
cd $B
echo "== $(hostname) $(date)"
echo "== req files"; paste -sd' ' jobs/req_gb.txt; paste -sd' ' jobs/req_skx.txt
echo "== t3t_envfix $EF: $(qstat -u jamesdc1 | awk -v j=$EF 'NR > 2 && $1 == j {print $5}')"
NEW=""
while read -r t arch rx; do
  for kind in L S; do
    f=lines/gR2${kind}_$t.txt
    [ -f $f ] || continue
    n=$(grep -c . $f)
    [ "$n" -gt 0 ] || continue
    hrt=2:00:00
    [ $kind = L ] && hrt=8:00:00
    nj=$(qsub -terse -N t3t_gR2${kind}_$t -hold_jid $EF -t 1-$n -pe shared 8 -l "h_rt=$hrt,h_data=2G,arch=$arch" \
      -o "$B/logs/gR2${kind}_$t.\$TASK_ID.log" -v "LINES=$B/$f,WANT=$rx" jobs/line.sh | cut -d. -f1)
    echo "  $t: t3t_gR2${kind}_$t $nj ($n lines, h_rt $hrt, held on $EF): $(cut -c1-110 $f | paste -sd'|')"
    NEW="$NEW,$nj"
  done
done < h2/tags.txt
sleep 2
LIVE=$(qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t_g(20|8|R)/ {print $1}' | sort -u | paste -sd,)
qalter -hold_jid "$EF,$LIVE" $GSUM
echo "fix4 $(date +%F_%T) envfix=$EF rerun=${NEW#,} gsum=$GSUM hold=$EF,$LIVE" | tee -a jobids.txt
echo "== holds"
for j in ${LIVE//,/ } $GSUM; do echo "$j $(qstat -j $j 2>/dev/null | awk '/^job_name/ {n = $2} /^jid_predecessor_list \(req\)/ {p = $3} END {print n, "hold:", p}')"; done
echo "== my t3t jobs"; qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t/'
echo "== done $(date)"
