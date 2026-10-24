#!/bin/bash
# RULE.txt addendum B (scored-only refits in the 48-interval arms) on CARC Discovery (account pollok_1603, partition
# main, epyc-7513|7542): the refit gate after the ALL48 pretune / select, every ALL48 walk afterok on it, then ONE
# export + score over both designs (scores_all/).  Replaces the cancelled 13361327, 13361442, 13364086, 13364087,
# 13361444, 13361445, 13364094, 13364095.
# usage: submit3r.sh   -> prints every job id; the score job is the end of the chain
set -euo pipefail
B=/scratch1/jc_905/train3way
L=$B/logs
S=$B/jobs/stage3.sh
R=results/linear_train3way_2014start_w3000
REF=/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w3000   # the template (read only)
PRE48=13361324      # ALL48 pretune / select (3,000-day)
SEL16=13361440      # ALL48 select16 (after PRE48)
PREN=13364084       # ALL48_N3k + ALL48_N39k derive / pretune / select
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
cd $B/jobs
g=$(sbatch "${A[@]}" -J t3g_ref -o $L/ALL48.gate_refit.log -c 4 --mem=64G -t 2:00:00 --dependency=afterok:$PRE48 \
    $S ALL48 gate_refit --jobs 4 --gate-rows 400)
w1=$(sbatch "${A[@]}" -J t3r_all -o "$L/ALL48.walkr.%a.log" -c 1 --mem=12G -t 8:00:00 --array=1-22 \
     --dependency=afterok:$g $S ALL48 walk1 --chunks all_features/ridge=6 --chunks all_features/lasso=6 --chunks all_features/enet=6)
w2=$(sbatch "${A[@]}" -J t3r_a16 -o "$L/ALL48.walk16r.%a.log" -c 1 --mem=12G -t 8:00:00 --array=1-21 \
     --dependency=afterok:$SEL16:$g $S ALL48 walk16 --chunks all_features/ridge=6 --chunks all_features/lasso=6 --chunks all_features/enet=6)
w3=$(sbatch "${A[@]}" -J t3r_a3k -o "$L/ALL48_N3k.walkr.%a.log" -c 1 --mem=8G -t 8:00:00 --array=1-9 \
     --dependency=afterok:$PREN:$g $S ALL48_N3k walk1 --chunks all_features/ridge=1 --chunks all_features/lasso=2 --chunks all_features/enet=2)
w4=$(sbatch "${A[@]}" -J t3r_a39 -o "$L/ALL48_N39k.walkr.%a.log" -c 1 --mem=8G -t 8:00:00 --array=1-13 \
     --dependency=afterok:$PREN:$g $S ALL48_N39k walk1 --chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=3)
e=$(sbatch "${A[@]}" -J t3x3 -o $L/export3.log -c 1 --mem=16G -t 2:00:00 --dependency=afterok:$w1:$w2:$w3:$w4 \
    --wrap "bash $S ALL48 export --with16 && bash $S RTH13 export --with16 --ref $REF && bash $S CLOSE export && bash $S CLOSE_NAT export && bash $S ALL48_N3k export && bash $S RTH13_N3k export && bash $S ALL48_N39k export")
s=$(sbatch "${A[@]}" -J t3sc3 -o $L/score3.log -c 1 --mem=24G -t 2:00:00 --dependency=afterok:$e \
    --wrap "cd $B/code && OPENBLAS_CORETYPE=Prescott /scratch1/jc_905/wsweep/venv/bin/python experiments/linear_train3way_score.py $R $REF both")
echo "gate_refit $g | walks ALL48 $w1 ALL48@16 $w2 ALL48_N3k $w3 ALL48_N39k $w4 | export $e | SCORE (last) $s"
