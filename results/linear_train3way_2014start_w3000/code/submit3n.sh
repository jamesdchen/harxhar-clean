#!/bin/bash
# RULE.txt addendum A (equal training rows) on CARC Discovery (account pollok_1603, partition main, epyc-7513|7542):
# derive / pretune / select of ALL48_N3k, ALL48_N39k (after the ALL48 design) and RTH13_N3k, their walks, then ONE
# export + score over both designs (after every walk of both designs, and after the first chain's export 13361444 /
# score 13361445, which still run: the scancel of those two was denied by the auto-mode classifier, so this chain
# waits for them (afterany) instead of racing them; its scores go to scores_all/, theirs to scores/).
# usage: submit3n.sh   -> prints every job id; the last line's score job is the end of the chain
set -euo pipefail
B=/scratch1/jc_905/train3way
L=$B/logs
S=$B/jobs/stage3.sh
R=results/linear_train3way_2014start_w3000
REF=/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w3000   # the template (read only)
ALLD=13361323                                                # the ALL48 design (equal-calendar chain)
# As run 2026-10-23 22:20 PDT: jobs 13364083-13364087 from this script; its export line then failed ("Job dependency
# problem": the finished walk arrays 13361328, 13361329, 13361443 had left slurmctld, MinJobAge 300 s; sacct: every
# task COMPLETED 0:0), so export2 13364094 / score2 13364095 were submitted by hand with the list below.
WALKS_CAL=13361327:13361442   # the equal-calendar walks still pending (ALL48, ALL48@16); the others finished OK
OLD_SCORE=13361445                                            # the first chain's score job (after its export 13361444)
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
cd $B/jobs
n1=$(sbatch "${A[@]}" -J t3n_rth -o $L/RTH13_N3k.pre.log -c 4 --mem=24G -t 2:00:00 \
     --wrap "bash $S RTH13_N3k derive_window --from $R/RTH13 && bash $S RTH13_N3k pretune --jobs 4 && bash $S RTH13_N3k select")
n2=$(sbatch "${A[@]}" -J t3n_all -o $L/ALL48_N.pre.log -c 4 --mem=64G -t 4:00:00 --dependency=afterok:$ALLD \
     --wrap "bash $S ALL48_N3k derive_window --from $R/ALL48 && bash $S ALL48_N3k pretune --jobs 4 && bash $S ALL48_N3k select && bash $S ALL48_N39k derive_window --from $R/ALL48 && bash $S ALL48_N39k pretune --jobs 4 && bash $S ALL48_N39k select")
w1=$(sbatch "${A[@]}" -J t3m_rth -o "$L/RTH13_N3k.walk.%a.log" -c 1 --mem=6G -t 4:00:00 --array=1-8 \
     --dependency=afterok:$n1 $S RTH13_N3k walk1 --chunks all_features/lasso=2)
w2=$(sbatch "${A[@]}" -J t3m_a3k -o "$L/ALL48_N3k.walk.%a.log" -c 1 --mem=8G -t 8:00:00 --array=1-14 \
     --dependency=afterok:$n2 $S ALL48_N3k walk1 --chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4)
w3=$(sbatch "${A[@]}" -J t3m_a39 -o "$L/ALL48_N39k.walk.%a.log" -c 1 --mem=8G -t 8:00:00 --array=1-22 \
     --dependency=afterok:$n2 $S ALL48_N39k walk1 --chunks all_features/ridge=4 --chunks all_features/lasso=8 --chunks all_features/enet=6)
e=$(sbatch "${A[@]}" -J t3x2 -o $L/export2.log -c 1 --mem=16G -t 2:00:00 --dependency=afterok:$WALKS_CAL:$w1:$w2:$w3,afterany:$OLD_SCORE \
    --wrap "bash $S ALL48 export --with16 && bash $S RTH13 export --with16 --ref $REF && bash $S CLOSE export && bash $S CLOSE_NAT export && bash $S ALL48_N3k export && bash $S RTH13_N3k export && bash $S ALL48_N39k export")
s=$(sbatch "${A[@]}" -J t3sc2 -o $L/score2.log -c 1 --mem=24G -t 2:00:00 --dependency=afterok:$e \
    --wrap "cd $B/code && OPENBLAS_CORETYPE=Prescott /scratch1/jc_905/wsweep/venv/bin/python experiments/linear_train3way_score.py $R $REF both")
echo "pretune/select RTH13_N3k $n1 ALL48_N3k+N39k $n2 | walks RTH13_N3k $w1 ALL48_N3k $w2 ALL48_N39k $w3 | export $e | SCORE (last) $s"
