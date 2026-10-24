#!/bin/bash
# Hoffman2 login node (bash -l), once: the setup and the gates of the trees' three-way comparison on Hoffman2 (RULE.txt
# addendum C).
#   inputs   $B/carc = CARC's files (transfer 1, md5 manifest checked on arrival); $B/code = git archive of the commit
#            in CARC's runs/tasks/h2_code_commit.txt (checked here against CARC's code dir md5s); $B/jobs = h2/code
#   env      t3t_env (h2/code/env.sh): venv_gb, venv_skx (req_skx.txt = req_gb.txt + the packages of CARC's venv_skx
#            own site-packages, from env_carc.txt)
#   gates    per tag (h2/code/tags.txt, one CPU model a tag; line.sh refuses another model), the forest only (the
#            assigned class): t3t_g20L_<tag> ALL48 full set every row (20 threads), t3t_g20_<tag> RTH13 --ends 10 at 20
#            threads, ALL48 scored-only both sets and HAR every row, t3t_g8_<tag> RTH13 --ends 10 at 8 threads and
#            ALL48_N3k every row / scored both sets -> $B/gate/h2/<tag>/{g1,g2}/pieces (CARC's delivery layout)
#   check    t3t_gsum (h2/code/summary.sh): accept + gate_summary.py -> passed CPU models, seconds a refit, h_rt
#   walks    not here: waves of the lines still ASSIGNED, sized to the slots obtained (submit_wave.sh; owner, 02:40)
# usage: bash submit_h2.sh <commit sha>
set -euo pipefail
SHA=$1
B=/u/scratch/j/jamesdc1/t3t
cd $B
if [ -f jobids.txt ] && grep -q "^gates " jobids.txt; then echo "ABORT: already submitted"; cat jobids.txt; exit 3; fi
mkdir -p logs h2 lines runs/tasks runs/pieces
chmod +x jobs/*.sh
echo "$SHA" > code/COMMIT
# code = CARC's code dir of that commit (md5 of the files CARC listed)
awk -v c="code_${SHA:0:7}" '$1 == "==" {f = ($2 == c); next} f' carc/h2/xfer1/INFO.txt > h2/code_md5_carc.txt
[ -s h2/code_md5_carc.txt ] || { echo "ABORT: no md5 of CARC's code_${SHA:0:7} in carc/h2/xfer1/INFO.txt"; exit 3; }
(cd code && md5sum -c ../h2/code_md5_carc.txt) || { echo "ABORT: code differs from CARC's"; exit 3; }
echo "== CARC env (transfer 1)"; cat carc/h2/xfer1/env_carc.txt | grep -v -E '^\s+"(include|lib|pc file) directory"'
# req_skx = req_gb + venv_skx's own packages (dist-info names in CARC's listing), pip / setuptools and req_gb's names
# excluded (CARC's venv_skx reports req_gb's versions: results/pooled_rf_skx_2014start_w3000/run.json)
cp jobs/req_gb.txt jobs/req_skx.txt
awk '/^===== venv_skx own site-packages/ {f = 1; next} /^=====/ {f = 0} f' carc/h2/xfer1/env_carc.txt \
  | sed -n 's/^\(.*\)-\([^-]*\)[.]dist-info$/\1==\2/p' | grep -v -i -E '^(pip|setuptools|wheel)==' \
  | grep -v -i -E '^(numpy|pandas|scipy|scikit[-_]learn|numba|pyarrow|tqdm|lightgbm|xgboost)==' >> jobs/req_skx.txt
echo "== req_skx.txt"; cat jobs/req_skx.txt
# assignment and the Hoffman2 task file (line k of CARC's file = "<arm> <est> <set> k/n", asserted)
cp carc/runs/tasks/h2_assignment.txt runs/tasks/h2_assignment.txt
echo "== assignment"; grep -v '^#' runs/tasks/h2_assignment.txt
: > runs/tasks/h2_w3rf_full.txt
echo w3rf_full.txt > runs/tasks/h2_w3rf_full.txt.source
awk '$1 == "ASSIGNED" && $2 == "w3rf_full.txt" {print $3}' runs/tasks/h2_assignment.txt | while IFS=- read -r lo hi; do
  awk -v lo="$lo" -v hi="$hi" 'NR >= lo && NR <= hi { split($4, p, "/"); if (p[1] + 0 != NR) { print "BAD line " NR > "/dev/stderr"; exit 1 } print }' \
    carc/runs/tasks/w3rf_full.txt >> runs/tasks/h2_w3rf_full.txt
done
NW=$(grep -c . runs/tasks/h2_w3rf_full.txt)
echo "h2 task file: $NW lines, $(head -1 runs/tasks/h2_w3rf_full.txt) .. $(tail -1 runs/tasks/h2_w3rf_full.txt)"
[ "$NW" -gt 0 ] || { echo "ABORT: no assigned lines"; exit 3; }
grep -v '^#' jobs/tags.txt > h2/tags.txt
q() { qsub -terse "$@" | cut -d. -f1; }
e=$(q -N t3t_env -o $B/logs/env.log jobs/env.sh)
X=experiments/trees_train3way.py
D="--design-root $B/carc/designs"
W48="--from 2014-01-02T15:30 --to 2014-01-03T10:00"
W3K="--from 2014-01-02T15:30 --to 2014-01-06T10:30"
G=""
ALLG=""
while read -r t arch rx; do
  O=$B/gate/h2/$t
  mkdir -p $O/g1/pieces $O/g2/pieces
  echo "skx $X walk --arm ALL48 --est rf_skx --set all_features $W48 --every-row --tag every $D --out $O/g2" > lines/g20L_$t.txt
  : > lines/g20_$t.txt
  : > lines/g8_$t.txt
  for s in baseline all_features; do
    echo "skx $X walk --arm RTH13 --est rf_skx --set $s --ends 10 --tag ends10_$t $D --out $O/g1" >> lines/g20_$t.txt
    echo "skx $X walk --arm ALL48 --est rf_skx --set $s $W48 --tag scored $D --out $O/g2" >> lines/g20_$t.txt
    echo "skx $X walk --arm RTH13 --est rf_skx --set $s --ends 10 --tag ends10_${t}_t8 $D --out $O/g1" >> lines/g8_$t.txt
    echo "skx $X walk --arm ALL48_N3k --est rf_skx --set $s $W3K --every-row --tag every $D --out $O/g2" >> lines/g8_$t.txt
    echo "skx $X walk --arm ALL48_N3k --est rf_skx --set $s $W3K --tag scored $D --out $O/g2" >> lines/g8_$t.txt
  done
  echo "skx $X walk --arm ALL48 --est rf_skx --set baseline $W48 --every-row --tag every $D --out $O/g2" >> lines/g20_$t.txt
  a=$(q -N t3t_g20L_$t -hold_jid $e -t 1-1 -pe shared 20 -l "h_rt=10:00:00,h_data=1536M,arch=$arch" \
    -o "$B/logs/g20L_$t.\$TASK_ID.log" -v "LINES=$B/lines/g20L_$t.txt,WANT=$rx" jobs/line.sh)
  b=$(q -N t3t_g20_$t -hold_jid $e -t 1-$(grep -c . lines/g20_$t.txt) -pe shared 20 -l "h_rt=2:00:00,h_data=1536M,arch=$arch" \
    -o "$B/logs/g20_$t.\$TASK_ID.log" -v "LINES=$B/lines/g20_$t.txt,WANT=$rx" jobs/line.sh)
  c=$(q -N t3t_g8_$t -hold_jid $e -t 1-$(grep -c . lines/g8_$t.txt) -pe shared 8 -l "h_rt=2:00:00,h_data=2G,arch=$arch" \
    -o "$B/logs/g8_$t.\$TASK_ID.log" -v "LINES=$B/lines/g8_$t.txt,WANT=$rx" jobs/line.sh)
  G="$G $t=$a,$b,$c"
  ALLG="$ALLG,$a,$b,$c"
done < h2/tags.txt
s=$(q -N t3t_gsum -hold_jid "$e$ALLG" -o $B/logs/gsum.log jobs/summary.sh)
echo "gates $(date +%F_%T) code=$SHA env=$e$G gsum=$s (walks: waves by submit_wave.sh after the gates)" | tee -a jobids.txt
echo "== qquota / rqs naming every user"
qquota -u jamesdc1 2>&1 | grep -v -E "^(econdjs|kappel|gpu|eeskin)/" | head
qconf -srqs 2>/dev/null | grep -n -E "users +(\{?\*|\{\*\})" | head -10
sleep 3
qstat -u jamesdc1 | head -40
