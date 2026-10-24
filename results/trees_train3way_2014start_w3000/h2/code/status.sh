#!/bin/bash
# Hoffman2 login node (bash -l), read only: the t3t jobs (qstat by name and state), slots running, gate pieces per tag,
# walk pieces (done / checkpointed), the gate summary, errors in the logs since the last call, and qacct of finished
# gate / walk tasks (host, wall, maxvmem, exit; slow: only with 'acct').  usage: bash status.sh [n log lines] [acct]
B=/u/scratch/j/jamesdc1/t3t
N=${1:-4}
echo "== $(hostname) $(date)"
echo "== jobids"; cat $B/jobids.txt
echo "== qstat t3t (count name state)"
qstat -u jamesdc1 -g d 2>/dev/null | awk 'NR > 2 && $3 ~ /^t3t/ {print $3, $5}' | sort | uniq -c
echo "== t3t slots running: $(qstat -u jamesdc1 -s r 2>/dev/null | awk 'NR > 2 && $3 ~ /^t3t/ {s += $9} END {print s + 0}')"
echo "== all jamesdc1 jobs by name/state"; qstat -u jamesdc1 2>/dev/null | awk 'NR > 2 {print substr($3, 1, 8), $5}' | sort | uniq -c
for d in $B/gate/h2/*; do
  echo "gate $(basename $d): g1 $(ls $d/g1/pieces 2>/dev/null | grep -c '[.]parquet$') pieces ($(ls $d/g1/pieces 2>/dev/null | grep -c 'ckpt')) ckpt, g2 $(ls $d/g2/pieces 2>/dev/null | grep -c '[.]parquet$') ($(ls $d/g2/pieces 2>/dev/null | grep -c 'ckpt') ckpt)"
done
echo "== walk pieces: $(ls $B/runs/pieces 2>/dev/null | grep -E '[.]json$' | grep -vc ckpt) done, $(ls $B/runs/pieces 2>/dev/null | grep -c '[.]ckpt[.]json$') checkpointed; delivered $(cat $B/h2/delivered.txt 2>/dev/null | grep -c .)"
[ -f $B/h2/gate_summary.txt ] && { echo "== gate summary"; head -c 6000 $B/h2/gate_summary.txt; }
echo "== gsum log"; tail -n 12 $B/logs/gsum.log 2>/dev/null
echo "== env log"; tail -n $N $B/logs/env.log 2>/dev/null
M=$B/logs/.status_mark
echo "== log errors since the last status"
NEW=(); [ -f $M ] && NEW=(-newer $M)
find $B/logs -name '*.log' "${NEW[@]}" 2>/dev/null | xargs -r grep -l -E "Traceback|Error|error:|Killed|refusing|exceeded|ABORT" 2>/dev/null | head -20 | while read -r f; do
  echo "-- $f"; grep -E "Traceback|Error|error:|Killed|refusing|exceeded|ABORT" "$f" | tail -n $N
done
touch $M
if [ "${2:-}" = acct ]; then
echo "== qacct of finished t3t tasks (job.task host wall_s maxvmem exit failed)"
for j in $(grep -o -E '(env|gsum|job)=[0-9]+|=[0-9]+,[0-9]+,[0-9]+' $B/jobids.txt | grep -o -E '[0-9]{6,}' | sort -u); do
  qacct -j $j 2>/dev/null | awk '/^jobname/ {n = $2} /^taskid/ {t = $2} /^hostname/ {h = $2} /^ru_wallclock/ {w = $2} /^maxvmem/ {m = $2} /^exit_status/ {e = $2} /^failed/ {f = $2; print n, t, h, w, m, e, f}'
done | sort | tail -n 300
fi
echo "== done $(date)"
