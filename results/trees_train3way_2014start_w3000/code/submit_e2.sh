#!/bin/bash
# Note to addendum E (CARC login node): the Xeon gate tasks that failed on infrastructure (SIGBUS on the mmap of the
# /scratch1 design, ESHUTDOWN reading it: nodes whose /scratch1 Lustre mount fails) are rerun on the same CPU model with
# the designs read from a /home1 copy (md5-checked) and those nodes excluded; their incomplete pieces are removed first
# (a gate walk is never resumed); new check jobs wait on the original arrays (afterany) and the reruns (afterok); the
# never-started Xeon pools and the old checks are cancelled and resubmitted (designs from /home1, the nodes excluded).
# usage: bash submit_e2.sh <code dir> <held merge job id> <EPYC pool id> <comma-separated bad nodes>
set -euo pipefail
CODE=$1
M4=$2
PE=$3
BAD=$4
BASE=/scratch1/jc_905/trees3way
HB=/home1/jc_905/trees3way
HD=$HB/designs
R=$BASE/runs
L=$BASE/lines
T=$R/tasks
G=$BASE/gate
J=$CODE/results/trees_train3way_2014start_w3000/code
CL=$R/claims_w3rf_full
if grep -q "^phase_e2 " $BASE/jobids.txt; then
  echo "ABORT: already submitted"
  exit 3
fi
mkdir -p $HB/logs
# 1. the designs on /home1 (a job: copy + md5 against the /scratch1 copies' MD5SUMS)
cat > $L/stage_home.sh << 'SH'
#!/bin/bash
set -euo pipefail
for A in RTH13 ALL48; do
  for s in baseline all_features; do
    S=/scratch1/jc_905/trees3way/designs/$A/_work/design_$s
    D=/home1/jc_905/trees3way/designs/$A/_work/design_$s
    mkdir -p $D
    cp $S/X.npy $S/y.npy $S/b_oos.npy $S/dates_oos.csv $S/dates_all.csv $S/design.json $S/MD5SUMS $D/
    (cd $D && md5sum -c --quiet MD5SUMS) && echo "$A design_$s: /home1 copy md5 = MD5SUMS"
  done
done
du -sh /home1/jc_905/trees3way/designs
SH
st=$(sbatch --parsable --account=pollok_1603 -p main --constraint="epyc-7513|epyc-7542" -J t3t_sth -c 2 --mem=8G -t 1:00:00 \
  -o "$BASE/logs/%x.%j.out" $L/stage_home.sh)
# 2. the failed gate lines -> rerun files (designs from /home1); incomplete pieces of those lines removed
rerun() {  # <array id> <line file> <out file>
  : > "$3"
  for t in $(sacct -j "$1" -X -n -P -s F,OOM,TO,NF,BF -o JobID | sed 's/.*_//'); do
    ln=$(sed -n "${t}p" "$2")
    echo "${ln/--design-root \/scratch1\/jc_905\/trees3way\/designs/--design-root $HD}" >> "$3"
    arm=$(echo "$ln" | grep -o -- "--arm [^ ]*" | cut -d' ' -f2)
    est=$(echo "$ln" | grep -o -- "--est [^ ]*" | cut -d' ' -f2)
    set_=$(echo "$ln" | grep -o -- "--set [^ ]*" | cut -d' ' -f2)
    tag=$(echo "$ln" | grep -o -- "--tag [^ ]*" | cut -d' ' -f2)
    out=$(echo "$ln" | grep -o -- "--out [^ ]*" | cut -d' ' -f2)
    pc=$out/pieces/${arm}_${est}_${set_}_${tag}
    if [ ! -f $pc.json ]; then rm -f $pc.parquet $pc.ckpt.json $pc.part; echo "removed incomplete $pc"; fi
  done
}
rerun 13377713 $L/ge_carc_x4116_t8.txt $L/ge2_x4116_t8.txt
rerun 13377714 $L/ge_carc_x4116_t20.txt $L/ge2_x4116_t20.txt
rerun 13377716 $L/ge_carc_x2640v4_t8.txt $L/ge2_x2640v4_t8.txt
rerun 13377717 $L/ge_carc_x2640v4_t20.txt $L/ge2_x2640v4_t20.txt
wc -l $L/ge2_*.txt
cd $CODE
sb() { sbatch --parsable --account=pollok_1603 "$@"; }
gl() {  # <name> <partition> <constraint> <cpus> <mem> <time> <lines> <dependency>
  local n
  n=$(grep -c . "$7" || true)
  if [ "$n" -eq 0 ]; then echo ""; return 0; fi
  sb -p "$2" --constraint="$3" --exclude="$BAD" -J "$1" -c "$4" --mem="$5" -t "$6" --array="1-$n" --dependency="$8" \
    -o "$BASE/logs/%x.%A_%a.out" --export=ALL,LINES=$7,CODE=$CODE $J/line.sbatch
}
r41a=$(gl t3t_gr41a main xeon-4116 8 16G 4:00:00 $L/ge2_x4116_t8.txt afterok:$st)
r41b=$(gl t3t_gr41b main,oneweek xeon-4116 20 24G 8:00:00 $L/ge2_x4116_t20.txt afterok:$st)
r26a=$(gl t3t_gr26a oneweek xeon-2640v4 8 16G 4:00:00 $L/ge2_x2640v4_t8.txt afterok:$st)
r26b=$(gl t3t_gr26b oneweek xeon-2640v4 20 24G 8:00:00 $L/ge2_x2640v4_t20.txt afterok:$st)
dep() {  # afterany on the originals, afterok on the reruns that exist
  local d="afterany:$1:$2"
  local ok=""
  for x in "${@:3}"; do if [ -n "$x" ]; then ok="$ok:$x"; fi; done
  if [ -n "$ok" ]; then d="$d,afterok$ok"; fi
  echo "$d"
}
c41=$(sb -p main --constraint=xeon-4116 --exclude="$BAD" -J t3t_gc41 -c 1 --mem=8G -t 0:30:00 \
  --dependency="$(dep 13377713 13377714 $r41a $r41b)" -o "$BASE/logs/%x.%j.out" \
  --export=ALL,LINES=$L/ge_carc_x4116_chk.txt,CODE=$CODE $J/line.sbatch)
c26=$(sb -p oneweek --constraint=xeon-2640v4 --exclude="$BAD" -J t3t_gc26 -c 1 --mem=8G -t 0:30:00 \
  --dependency="$(dep 13377716 13377717 $r26a $r26b)" -o "$BASE/logs/%x.%j.out" \
  --export=ALL,LINES=$L/ge_carc_x2640v4_chk.txt,CODE=$CODE $J/line.sbatch)
# 3. the never-started Xeon pools and the old checks: cancelled, the pools resubmitted
scancel 13377715 13377718 13377761 13377762
PX=(--export=ALL,CODE=$CODE,TASKFILE=$T/w3rf_full.txt,LO=5,HI=348,CLAIMS=$CL,DESIGNS=$HD)
p41=$(sb -p main --constraint=xeon-4116 --exclude="$BAD" -J t3t_p41 -c 8 --mem=16G -t 24:00:00 --array=1-150%40 \
  --dependency=afterok:$c41 -o "$BASE/logs/%x.%A_%a.out" "${PX[@]}" $J/pool.sbatch)
p26=$(sb -p oneweek --constraint=xeon-2640v4 --exclude="$BAD" -J t3t_p26 -c 20 --mem=24G -t 24:00:00 --array=1-20%10 \
  --dependency=afterok:$c26 -o "$BASE/logs/%x.%A_%a.out" "${PX[@]}" $J/pool.sbatch)
scontrol update JobId=$M4 Dependency=afterany:13372432:$PE:$p41:$p26:13372427:13371788:13371790 && echo "m4 $M4 dependency updated"
echo "phase_e2 $(date +%F_%T) code=$CODE bad=$BAD stage_home=$st gr41a=${r41a:-none} gr41b=${r41b:-none} gr26a=${r26a:-none} gr26b=${r26b:-none} gc41=$c41 gc26=$c26 p41=$p41 p26=$p26; cancelled 13377715 13377718 13377761 13377762" | tee -a $BASE/jobids.txt
