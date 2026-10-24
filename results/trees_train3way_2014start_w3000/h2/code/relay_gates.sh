#!/bin/bash
# Hoffman2 side of the gate delivery (run over ssh; stdout = the tar stream of deliver.sh gates, all else to stderr):
# prints the gate results (gate_summary.txt, accepted_cpus.json, logs with errors), the memory of the long gate
# pieces (qacct), the free 8-slot capacity per CPU model now, writes h2/h2_throughput.txt (mk_throughput.py), then
# streams deliver.sh gates (gate pieces with MD5SUMS, summary, throughput; carc_recv.sh inside).
exec 3>&1 1>&2
set -uo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/t3t
cd $B
echo "== $(hostname) $(date)"
echo "== t3t jobs in qstat: $(qstat -u jamesdc1 | awk 'NR > 2 && $3 ~ /^t3t/' | wc -l)"
echo "== envfix log"; tail -n 4 logs/envfix.log
echo "== gsum log (end)"; tail -n 30 logs/gsum.log | cut -c1-400
echo "== gate summary"; cut -c1-330 h2/gate_summary.txt
echo "== accepted"; cut -c1-2500 h2/accept/pieces_h2/accepted_cpus.json
echo "== gate logs with errors"; grep -l -E "Traceback|Error|refusing|exceeded|Killed" logs/g*.log 2>/dev/null | head; for f in $(grep -l -E "Traceback|Killed" logs/g*.log 2>/dev/null | head -4); do echo "-- $f"; grep -E "Traceback|Error|Killed" $f | tail -3; done
echo "== hosts / CPU of gate lines"; grep -E "^line (g|l)" logs/hosts.txt | awk -F' [|] ' '{print $3, "|", $2}' | sort | uniq -c | sort -rn | head -20
echo "== qacct of the ALL48 full-set every-row gate jobs (wall s, maxvmem)"
for j in 15261301 15261294 15261296 15261307; do timeout 150 qacct -j $j 2>/dev/null | awk '/^jobname/ {n = $2} /^hostname/ {h = $2} /^ru_wallclock/ {w = $2} /^maxvmem/ {m = $2} /^exit_status/ {e = $2; print n, h, w, m, "exit", e}'; done
echo "== free 8-slot capacity per CPU model now"
qhost -F arch 2>/dev/null | awk 'NF >= 10 && $1 != "HOSTNAME" && $1 != "global" {h = $1; nc[h] = $3; next} /hf:arch=/ {a = $0; sub(/.*hf:arch=/, "", a); ar[h] = a} END {for (x in ar) print x, ar[x], nc[x]}' > h2/.hosts
qstat -f 2>/dev/null | awk '$1 ~ /@/ {split($1, q, "@"); split(q[2], hh, "."); split($3, s, "/"); used[hh[1]] += s[1] + s[2]; if (NF >= 6 && $6 ~ /[duEaAsSCc]/) bad[hh[1]] = 1; seen[hh[1]] = 1} END {for (x in seen) print x, used[x] + 0, (x in bad) ? 1 : 0}' > h2/.used
awk 'NR == FNR {u[$1] = $2; b[$1] = $3; next} ($1 in u) && !b[$1] {f = $3 - u[$1]; if (f >= 8) {h8[$2]++; c8[$2] += int(f / 8)}} END {for (a in h8) print a, h8[a], c8[a]}' h2/.used h2/.hosts | sort > h2/.free8
cat h2/.free8
apptainer exec -B /u/scratch $B/deb.sif $B/venv_gb/bin/python $B/jobs/mk_throughput.py --summary h2/gate_summary.json \
  --accepted h2/accept/pieces_h2/accepted_cpus.json --tags h2/tags.txt --free8 h2/.free8 --out h2/h2_throughput.txt
cat h2/h2_throughput.txt
bash jobs/deliver.sh gates >&3
