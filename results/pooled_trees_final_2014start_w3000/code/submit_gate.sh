#!/bin/bash
# Gate of the final tree walks, on the CARC login node: code trees, gate pieces, and the jobs
#   tf_stage      the final designs copied (md5) + design_check.py (stage.sbatch)
#   tfg_A         gate A: the printed run's runner (db59af24, git archive of its CARC checkout) on the printed design,
#                 EPYC 7513, 8 threads: gate_spec.txt
#   tfg_B         gate B: this branch's runner, same pieces, same nodes and threads
#   tfg_B7542     this branch's runner on EPYC 7542 nodes (gate_spec_7542.txt)
#   tfg_Brf16     the random forest with 16 threads (gate_spec_rf16.txt)
#   tf_time_w<W>  the time probe (3 refits of each arm at the first 2014-01-02 rows, 8 threads) on each final design
# usage: bash submit_gate.sh <commit of insurance-selling-2026-09-19-e34-trees-final>
set -euo pipefail
BASE=/scratch1/jc_905/trees_final
C=$1
BR=insurance-selling-2026-09-19-e34-trees-final
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PR=/scratch1/jc_905/harxhar-trees13/results/pooled_trees_rerun13_2026-10-20/w1000
PD=/scratch1/jc_905/harxhar-trees13/results/pooled_linear_rerun_2026-10-19_13_w1000
if squeue -u jc_905 -h -o "%j" | grep -q '^tf'; then echo "ABORT: tf jobs already queued"; squeue -u jc_905; exit 3; fi
mkdir -p $BASE/logs $BASE/gate $BASE/runs $BASE/designs
cd $BASE
if [ ! -d repo/.git ]; then
  git clone -q --reference-if-able /scratch1/jc_905/harxhar-trees13 --dissociate --branch $BR \
    https://github.com/jamesdchen/harxhar-clean.git repo
fi
git -C repo fetch -q origin $BR
git -C repo checkout -q --detach "$C"
[ "$(git -C repo rev-parse HEAD)" = "$(git -C repo rev-parse "$C^{commit}")" ]
echo "repo at $(git -C repo log --oneline -1)"
if [ ! -d code_db59af2 ]; then
  mkdir code_db59af2
  git -C /scratch1/jc_905/harxhar-trees13 archive db59af24 | tar -x -C code_db59af2
fi
echo "code_db59af2: $(git -C /scratch1/jc_905/harxhar-trees13 rev-parse db59af24)"
CD=$BASE/repo/results/pooled_trees_final_2014start_w3000/code
for g in A B B7542 Brf16; do
  case $g in A | B) S=gate_spec.txt ;; B7542) S=gate_spec_7542.txt ;; Brf16) S=gate_spec_rf16.txt ;; esac
  if [ ! -d gate/$g ]; then echo "-- gate $g"; $PY $CD/gate_prep.py $PR/pieces gate/$g $CD/$S; fi
done
st=$(sbatch --parsable $CD/stage.sbatch)
X="HARXHAR_PY=$PY,DESIGN=$PD,WINDOW_DAYS=1000,MODE=walk"
O=(-o "$BASE/logs/%x.%A_%a.out" -t 3:00:00 --mem=8G)
sub() {  # <name> <code dir> <gate> <cpus> <constraint>
  local n=$(wc -l < $BASE/gate/$3/tasks.txt)
  (cd "$2" && sbatch --parsable -J "$1" --array=1-$n -c "$4" --constraint="$5" "${O[@]}" \
    --export=ALL,$X,OUT=$BASE/gate/$3,TASKFILE=$BASE/gate/$3/tasks.txt cluster/slurm/pooltree_rerun13.sbatch)
}
ga=$(sub tfg_A $BASE/code_db59af2 A 8 epyc-7513)
gb=$(sub tfg_B $BASE/repo B 8 epyc-7513)
gc=$(sub tfg_B7542 $BASE/repo B7542 8 epyc-7542)
gd=$(sub tfg_Brf16 $BASE/repo Brf16 16 epyc-7513)
tt=""
for W in 3000 3500; do
  t=$(cd $BASE/repo && sbatch --parsable -J tf_time_w$W --dependency=afterok:$st -c 8 --mem=16G -t 2:00:00 \
    --constraint=epyc-7513 $CD/tf_py.sbatch experiments/pooled_trees_rerun_13.py time --design $BASE/designs/w$W \
    --out $BASE/runs/w$W --window-days $W --k 3 --threads 8 --span 2014-01-02:2024-04-30)
  tt="$tt $t"
done
echo "stage $st gateA $ga gateB $gb gateB7542 $gc gateBrf16 $gd time$tt" | tee -a $BASE/jobids.txt
sleep 3
squeue -u jc_905 -o "%.18i %.14j %.8T %.10M %.4C %R" | head -40
