#!/bin/bash
# Hoffman2 login node (bash -l), once (2026-10-24 ~04:10 PDT), right after fix2_env.sh:
#   - daal4py imports packaging first (distutils only as its fallback): CARC's env has packaging 26.3, so it is pinned
#     too (req_gb.txt / req_skx.txt); if t3t_envfix has already started, t3t_envfix2 (the same envfix.sh) installs it;
#   - the 8-thread arrays of fix1 started tasks before the env existed (they exited "no env"): every gate log saying
#     "no env" that fix2 did not already handle is resubmitted (t3t_gR2L_<tag> / t3t_gR2S_<tag>, 8 threads);
#   - every queued gate task (qw, hqw, Rq) is held on the env job(s); t3t_gsum re-held on every gate job.
set -uo pipefail
B=/u/scratch/j/jamesdc1/t3t
cd $B
echo "== $(hostname) $(date)"
for f in jobs/req_gb.txt jobs/req_skx.txt; do grep -q '^packaging==' $f || echo packaging==26.3 >> $f; done
echo "== req_skx.txt"; cat jobs/req_skx.txt | paste -sd' '
EF=$(grep '^fix2 ' jobids.txt | tail -1 | grep -o -E 'envfix=[0-9]+' | cut -d= -f2)
st=$(qstat -u jamesdc1 | awk -v j=$EF 'NR > 2 && $1 == j {print $5}')
echo "t3t_envfix $EF state '${st:-finished}'; ENV_OK $(ls ENV_OK 2>/dev/null || echo absent)"
HOLD=$EF
if [ "$st" != qw ]; then
  rm -f ENV_OK
  e2=$(qsub -terse -N t3t_envfix2 -hold_jid $EF -o $B/logs/envfix2.log jobs/envfix.sh | cut -d. -f1)
  echo "t3t_envfix2 $e2 (after $EF)"
  HOLD="$EF,$e2"
fi
echo "== gate logs with 'no env' not yet resubmitted"
NEW=""
while read -r t arch rx; do
  : > lines/gR2L_$t.txt
  : > lines/gR2S_$t.txt
  for lg in logs/g*_$t.*.log; do
    [ -f "$lg" ] || continue
    grep -q "no env" "$lg" || continue
    b=$(basename "$lg" .log)
    k=${b##*.}
    kind=${b%%_$t.*}
    ln=$(sed -n "${k}p" lines/${kind}_$t.txt)
    if echo "$ln" | grep -q -- "--arm ALL48 --est rf_skx --set all_features .*--every-row"; then echo "$ln" >> lines/gR2L_$t.txt; else echo "$ln" >> lines/gR2S_$t.txt; fi
    echo "  $t $kind task $k (no env) -> 8 threads: ${ln:0:120}"
    mv "$lg" "$lg.noenv"
  done
  for kind in L S; do
    f=lines/gR2${kind}_$t.txt
    n=$(grep -c . $f)
    [ "$n" -gt 0 ] || continue
    hrt=2:00:00
    [ $kind = L ] && hrt=8:00:00
    nj=$(qsub -terse -N t3t_gR2${kind}_$t -hold_jid $HOLD -t 1-$n -pe shared 8 -l "h_rt=$hrt,h_data=2G,arch=$arch" \
      -o "$B/logs/gR2${kind}_$t.\$TASK_ID.log" -v "LINES=$B/$f,WANT=$rx" jobs/line.sh | cut -d. -f1)
    echo "  $t: t3t_gR2${kind}_$t $nj ($n lines, h_rt $hrt, held on $HOLD)"
    NEW="$NEW,$nj"
  done
done < h2/tags.txt
Q=$(qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t_g(20|8|R)/ && $5 ~ /q/ {print $1}' | sort -u)
for j in $Q; do qalter -hold_jid $HOLD $j > /dev/null && echo "  $j held on $HOLD"; done
GSUM=$(grep '^gates ' jobids.txt | tail -1 | grep -o -E 'gsum=[0-9]+' | cut -d= -f2)
sleep 2
LIVE=$(qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t_g(20|8|R)/ {print $1}' | sort -u | paste -sd,)
qalter -hold_jid "$HOLD,$LIVE" $GSUM > /dev/null
echo "fix3 $(date +%F_%T) env_hold=$HOLD rerun=${NEW#,} gsum=$GSUM hold=$HOLD,$LIVE" | tee -a jobids.txt
echo "== my jobs after"; qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t/'
echo "== done $(date)"
