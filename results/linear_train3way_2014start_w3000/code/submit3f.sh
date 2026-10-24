#!/bin/bash
# RULE.txt note to addendum B (scored-only refits withdrawn, owner option A): the four ALL48 walk arrays with every row
# refitted (the runner's walk), then export, then the final scoring over both designs (scores_all/).  Replaces the
# cancelled 13366981-13366986.  Every prerequisite job (ALL48 pretune / select 13361324, select16 13361440, the N arms'
# derive / pretune / select 13364084, every RTH13 / CLOSE walk) has completed, so the walks carry no dependency.
# Chunks for full refits: the 250-row cold seeds and the KKT reseeds on the 144,000-row window dominate (gate slice:
# lasso full 126 s / 400 rows, enet 69 s, ridge 26 s), about 9 / 6 / 2 CPU-hours a model at 3,000 days.
# usage: submit3f.sh   -> prints every job id; the score job is the end of the chain
set -euo pipefail
B=/scratch1/jc_905/train3way
L=$B/logs
S=$B/jobs/stage3.sh
R=results/linear_train3way_2014start_w3000
REF=/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w3000   # the template (read only)
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
cd $B/jobs
w1=$(sbatch "${A[@]}" -J t3f_all -o "$L/ALL48.walkf.%a.log" -c 1 --mem=10G -t 10:00:00 --array=1-19 \
     $S ALL48 walk1 --chunks all_features/ridge=2 --chunks all_features/lasso=8 --chunks all_features/enet=5)
w2=$(sbatch "${A[@]}" -J t3f_a16 -o "$L/ALL48.walk16f.%a.log" -c 1 --mem=10G -t 10:00:00 --array=1-18 \
     $S ALL48 walk16 --chunks all_features/ridge=2 --chunks all_features/lasso=8 --chunks all_features/enet=5)
w3=$(sbatch "${A[@]}" -J t3f_a3k -o "$L/ALL48_N3k.walkf.%a.log" -c 1 --mem=8G -t 10:00:00 --array=1-8 \
     $S ALL48_N3k walk1 --chunks all_features/lasso=2)
w4=$(sbatch "${A[@]}" -J t3f_a39 -o "$L/ALL48_N39k.walkf.%a.log" -c 1 --mem=8G -t 10:00:00 --array=1-10 \
     $S ALL48_N39k walk1 --chunks all_features/lasso=3 --chunks all_features/enet=2)
e=$(sbatch "${A[@]}" -J t3x4 -o $L/export4.log -c 1 --mem=16G -t 2:00:00 --dependency=afterok:$w1:$w2:$w3:$w4 \
    --wrap "bash $S ALL48 export --with16 && bash $S RTH13 export --with16 --ref $REF && bash $S CLOSE export && bash $S CLOSE_NAT export && bash $S ALL48_N3k export && bash $S RTH13_N3k export && bash $S ALL48_N39k export")
s=$(sbatch "${A[@]}" -J t3sc4 -o $L/score4.log -c 1 --mem=24G -t 2:00:00 --dependency=afterok:$e \
    --wrap "cd $B/code && OPENBLAS_CORETYPE=Prescott /scratch1/jc_905/wsweep/venv/bin/python experiments/linear_train3way_score.py $R $REF both")
echo "walks ALL48 $w1 ALL48@16 $w2 ALL48_N3k $w3 ALL48_N39k $w4 | export $e | SCORE (last) $s"
