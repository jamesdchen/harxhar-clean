#!/bin/bash
# Hoffman2 login node (bash -l), once (2026-10-24 ~04:10 PDT, coordinator's check-in: the 20-slot gate jobs did not
# schedule in 1 h 45 min).  Diagnoses why (scheduler info of a pending job, the per-user resource quotas, free slots
# per CPU model) and makes the gates schedulable without weakening them: every gate task of the 20-slot arrays that has
# not started (state qw) is deleted (qdel -t, own pending tasks only; started tasks run on) and its line is resubmitted
# unchanged at 8 threads (-pe shared 8, h_data 2G a slot, the same CPU model pin): t3t_g8L_<tag> (the ALL48 full-set
# every-row line, h_rt 8 h) and t3t_g8S_<tag> (the other lines, h_rt 2 h).  The forest's thread count does not change
# its forecasts (CARC's rf_skx gate (d): 8 = 20 threads bit for bit), each piece records its threads, and gate 1 / 2
# compare these 8-thread pieces against the templates and each other as before; the walks then run at the gated 8.
# The check t3t_gsum is re-held on every gate job still queued or running plus the new ones.
set -uo pipefail
B=/u/scratch/j/jamesdc1/t3t
cd $B
echo "== $(hostname) $(date)"
echo "== env: $(ls $B/ENV_OK 2>/dev/null)"; tail -n 3 $B/logs/env.log
echo "== my jobs"; qstat -u jamesdc1
qstat -u jamesdc1 -g d > h2/.qstat_gd
echo "== t3t tasks (-g d)"; awk 'NR > 2 && $3 ~ /^t3t/' h2/.qstat_gd
echo "== scheduler settings"; qconf -ssconf 2>/dev/null | grep -i -E "schedd_job_info|max_reservation|default_duration|job_load_adjust|load_formula|queue_sort"
P20=$(awk 'NR > 2 && $3 ~ /^t3t_g20/ && $5 ~ /qw/ {print $1; exit}' h2/.qstat_gd)
P8=$(awk 'NR > 2 && $3 ~ /^t3t_g8/ && $5 ~ /qw/ {print $1; exit}' h2/.qstat_gd)
for j in $P20 $P8; do
  echo "== qstat -j $j (resources, scheduling info)"
  qstat -j $j 2>&1 | grep -E "job_name|hard resource_list|parallel environment|scheduling info|cannot run|reason|queue instance|dropped|hard_queue|soft" | head -40
done
echo "== resource quota sets touching all users / pe shared"
qconf -srqs 2>/dev/null | awk '/^\{/ {buf = ""; keep = 0} {buf = buf $0 "\n"} /pes single,shared|users +\{?\*\}? +(queues|hosts|pes)/ {keep = 1} /^\}/ {if (keep) printf "%s", buf}' | head -60
echo "== qquota (my applicable limits)"; qquota -u jamesdc1 2>&1 | head -20
echo "== free slots per CPU model: hosts, hosts with >= 8 / >= 20 free slots (NCPU minus slots used or reserved, queue instances not disabled / alarmed / in error), free memory"
qhost -F arch 2>/dev/null | awk '
  NF >= 10 && $1 != "HOSTNAME" && $1 != "global" {h = $1; nc[h] = $3; mt[h] = $8; mu[h] = $9; next}
  /hf:arch=/ {a = $0; sub(/.*hf:arch=/, "", a); ar[h] = a}
  END {for (x in ar) print x, ar[x], nc[x], mt[x], mu[x]}' > h2/.hosts
qstat -f 2>/dev/null | awk '$1 ~ /@/ {split($1, q, "@"); split(q[2], hh, "."); split($3, s, "/"); used[hh[1]] += s[1] + s[2]; if (NF >= 6 && $6 ~ /[duEaAsSCc]/) bad[hh[1]] = 1; seen[hh[1]] = 1}
  END {for (x in seen) print x, used[x] + 0, (x in bad) ? 1 : 0}' > h2/.used
awk 'NR == FNR {u[$1] = $2; b[$1] = $3; next} ($1 in u) {
    f = $3 - u[$1]; m = $4; gsub(/G/, "", m); mm = $5; gsub(/G/, "", mm)
    n[$2]++; if (!b[$1]) {ok[$2]++; if (f >= 8) f8[$2]++; if (f >= 20) f20[$2]++; if (f >= 8 && m - mm >= 16) f8m[$2]++; if (f >= 20 && m - mm >= 30) f20m[$2]++; fs[$2] += (f > 0 ? f : 0)}}
  END {printf "%-20s %6s %6s %8s %9s %9s %10s %9s\n", "arch", "hosts", "usable", "free>=8", "+16G", "free>=20", "+30G", "free_sl"
       for (a in n) printf "%-20s %6d %6d %8d %9d %9d %10d %9d\n", a, n[a], ok[a], f8[a], f8m[a], f20[a], f20m[a], fs[a]}' h2/.used h2/.hosts | sort
for t in gold6140 gold6240 gold6342 e5v4; do
  a=$(awk -v t=$t '$1 == t {print $2}' h2/tags.txt)
  echo "$t ($a): hosts with >= 8 free: $(awk -v a=$a 'NR == FNR {u[$1] = $2; b[$1] = $3; next} $2 == a && ($1 in u) && !b[$1] && $3 - u[$1] >= 8 {printf "%s(%d) ", $1, $3 - u[$1]}' h2/.used h2/.hosts | head -c 400)"
done
echo "== fix: the pending tasks of the 20-slot gate arrays -> 8 threads"
G=$(grep '^gates ' jobids.txt | tail -1)
NEWIDS=""
while read -r t arch rx; do
  ids=$(echo "$G" | grep -o -E "$t=[0-9]+,[0-9]+,[0-9]+" | cut -d= -f2)
  jL=${ids%%,*}
  rest=${ids#*,}
  jS=${rest%%,*}
  : > lines/g8L_$t.txt
  : > lines/g8S_$t.txt
  for pair in "$jL g20L" "$jS g20"; do
    set -- $pair
    j=$1
    kind=$2
    for k in $(awk -v j=$j 'NR > 2 && $1 == j && $5 ~ /qw/ {print $NF}' h2/.qstat_gd); do
      ln=$(sed -n "${k}p" lines/${kind}_$t.txt)
      if qdel $j -t $k > /dev/null 2>&1; then
        if echo "$ln" | grep -q -- "--arm ALL48 --est rf_skx --set all_features .*--every-row"; then echo "$ln" >> lines/g8L_$t.txt; else echo "$ln" >> lines/g8S_$t.txt; fi
        echo "  $t: deleted pending $j.$k ($kind) -> 8 threads: $ln"
      else
        echo "  $t: qdel $j -t $k failed (started meanwhile?): kept"
      fi
    done
  done
  for kind in L S; do
    f=lines/g8${kind}_$t.txt
    n=$(grep -c . $f)
    [ "$n" -gt 0 ] || continue
    hrt=2:00:00
    [ $kind = L ] && hrt=8:00:00
    nj=$(qsub -terse -N t3t_g8${kind}_$t -t 1-$n -pe shared 8 -l "h_rt=$hrt,h_data=2G,arch=$arch" \
      -o "$B/logs/g8${kind}_$t.\$TASK_ID.log" -v "LINES=$B/$f,WANT=$rx" jobs/line.sh | cut -d. -f1)
    echo "  $t: t3t_g8${kind}_$t $nj ($n lines, h_rt $hrt)"
    NEWIDS="$NEWIDS,$nj"
  done
done < h2/tags.txt
GSUM=$(echo "$G" | grep -o -E 'gsum=[0-9]+' | cut -d= -f2)
sleep 2
LIVE=$(qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t_g(20|8)/ {print $1}' | sort -u | paste -sd,)
echo "== t3t_gsum $GSUM re-held on: $LIVE"
qalter -hold_jid "$LIVE" $GSUM
echo "fix1 $(date +%F_%T) 8-thread gate resubmission new=${NEWIDS#,} gsum=$GSUM hold=$LIVE" | tee -a jobids.txt
qstat -j $GSUM 2>/dev/null | grep -E "jid_predecessor_list|hold"
echo "== my jobs after"; qstat -u jamesdc1
echo "== done $(date)"
