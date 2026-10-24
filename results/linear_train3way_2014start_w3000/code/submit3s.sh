#!/bin/bash
# Finer splits (coordinator 23:36 PDT, owner: use the CARC resources) for the full-refit walks that had not started:
# ALL48@16 and ALL48_N39k.  ALL48 13367775 and ALL48_N3k 13367777 had started and keep their splits.  Every chunk
# starts on the walk's own cold-seed rows (every 250th solve; the runner asserts it), checked bit for bit by gate_seam
# (ALL48, 3,000-day penalties: full-set ridge / lasso / elastic net and the OLS, rows [S0, S0+500) long vs a chunk from
# S0+250); the re-split walks run afterok on it.  Replaces the cancelled (not started) 13367776, 13367778, 13367779,
# 13367780.  Target about 30-45 min a task.
# AS RUN (23:52 PDT): ALL48_N39k 13367778 had started by then (2 tasks running), so it kept its split too; only
# ALL48@16 was re-split: 13367776, 13367779, 13367780 cancelled; gate_seam 13367866, ALL48@16 13367867 (21 tasks),
# export5 13367868 (afterok 13367775, 13367777, 13367778, 13367867), score5 13367869 -- the w4 line was not run.
# usage: submit3s.sh   -> prints every job id; the score job is the end of the chain
set -euo pipefail
B=/scratch1/jc_905/train3way
L=$B/logs
S=$B/jobs/stage3.sh
R=results/linear_train3way_2014start_w3000
REF=/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w3000   # the template (read only)
W_ALL=13367775   # ALL48 full-refit walks (running, kept)
W_N3K=13367777   # ALL48_N3k full-refit walks (running, kept)
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
cd $B/jobs
c=$(sbatch "${A[@]}" -J t3seam -o $L/ALL48.gate_seam.log -c 4 --mem=48G -t 1:00:00 $S ALL48 gate_seam --jobs 4)
w2=$(sbatch "${A[@]}" -J t3s_a16 -o "$L/ALL48.walk16s.%a.log" -c 1 --mem=10G -t 10:00:00 --array=1-21 \
     --dependency=afterok:$c $S ALL48 walk16 --chunks all_features/ridge=4 --chunks all_features/lasso=15 --chunks all_features/enet=5)
w4=$(sbatch "${A[@]}" -J t3s_a39 -o "$L/ALL48_N39k.walks.%a.log" -c 1 --mem=8G -t 10:00:00 --array=1-14 \
     --dependency=afterok:$c $S ALL48_N39k walk1 --chunks all_features/ridge=2 --chunks all_features/lasso=5 --chunks all_features/enet=3)
e=$(sbatch "${A[@]}" -J t3x5 -o $L/export5.log -c 1 --mem=16G -t 2:00:00 --dependency=afterok:$W_ALL:$W_N3K:$w2:$w4 \
    --wrap "bash $S ALL48 export --with16 && bash $S RTH13 export --with16 --ref $REF && bash $S CLOSE export && bash $S CLOSE_NAT export && bash $S ALL48_N3k export && bash $S RTH13_N3k export && bash $S ALL48_N39k export")
s=$(sbatch "${A[@]}" -J t3sc5 -o $L/score5.log -c 1 --mem=24G -t 2:00:00 --dependency=afterok:$e \
    --wrap "cd $B/code && OPENBLAS_CORETYPE=Prescott /scratch1/jc_905/wsweep/venv/bin/python experiments/linear_train3way_score.py $R $REF both")
echo "gate_seam $c | walks ALL48@16 $w2 ALL48_N39k $w4 (kept: ALL48 $W_ALL, ALL48_N3k $W_N3K) | export $e | SCORE (last) $s"
