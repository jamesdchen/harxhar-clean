#!/bin/bash
# Submit the three-way comparison's primary chain (RULE.txt) on CARC Discovery: account pollok_1603, partition main,
# --constraint epyc-7513|epyc-7542 (the template's nodes).  Every job writes under $B/code/results/linear_train3way_*.
# usage: submit3.sh primary | secondary | score
set -euo pipefail
B=/scratch1/jc_905/train3way
L=$B/logs
REF=/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w3000   # the template (read only)
mkdir -p $L
cd $B/jobs
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
S=$B/jobs/stage3.sh
CH48=(--chunks all_features/ridge=6 --chunks all_features/lasso=12 --chunks all_features/enet=8)
CH13=(--chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4)
sb() { sbatch "${A[@]}" "$@"; }
case $1 in
primary)
  # RTH13 rebuilt (gate + secondary re-tune only), CLOSE derived from it, ALL48 and CLOSE_NAT built natively
  j1=$(sb -J t3d_rth -o $L/RTH13.design.log -c 2 --mem=32G -t 2:00:00 $S RTH13 design --jobs 2)
  j2=$(sb -J t3g_rth -o $L/RTH13.gate.log -c 4 --mem=24G -t 2:00:00 --dependency=afterok:$j1 \
       --wrap "bash $S RTH13 pretune --jobs 4 && bash $S RTH13 select && bash $S RTH13 gate --jobs 4 --ref $REF")
  j3=$(sb -J t3p_cls -o $L/CLOSE.pre.log -c 4 --mem=16G -t 2:00:00 --dependency=afterok:$j1 \
       --wrap "bash $S CLOSE derive_close --from results/linear_train3way_2014start_w3000/RTH13 && bash $S CLOSE pretune --jobs 4 && bash $S CLOSE select")
  j4=$(sb -J t3d_all -o $L/ALL48.design.log -c 2 --mem=96G -t 6:00:00 $S ALL48 design --jobs 2)
  j5=$(sb -J t3p_all -o $L/ALL48.pre.log -c 4 --mem=48G -t 3:00:00 --dependency=afterok:$j4 \
       --wrap "bash $S ALL48 pretune --jobs 4 && bash $S ALL48 select")
  j6=$(sb -J t3d_nat -o $L/CLOSE_NAT.design.log -c 2 --mem=32G -t 2:00:00 $S CLOSE_NAT design --jobs 2)
  j7=$(sb -J t3p_nat -o $L/CLOSE_NAT.pre.log -c 4 --mem=16G -t 2:00:00 --dependency=afterok:$j6 \
       --wrap "bash $S CLOSE_NAT pretune --jobs 4 && bash $S CLOSE_NAT select")
  # walks: after the arm's selection and the gate
  w1=$(sb -J t3w_all -o "$L/ALL48.walk.%a.log" -c 1 --mem=12G -t 10:00:00 --array=1-30 \
       --dependency=afterok:$j5:$j2 $S ALL48 walk1 "${CH48[@]}")
  w2=$(sb -J t3w_cls -o "$L/CLOSE.walk.%a.log" -c 1 --mem=6G -t 2:00:00 --array=1-7 \
       --dependency=afterok:$j3:$j2 $S CLOSE walk1)
  w3=$(sb -J t3w_nat -o "$L/CLOSE_NAT.walk.%a.log" -c 1 --mem=6G -t 2:00:00 --array=1-7 \
       --dependency=afterok:$j7:$j2 $S CLOSE_NAT walk1)
  echo "RTH13 design $j1 gate $j2 | CLOSE $j3 | ALL48 design $j4 pre $j5 | CLOSE_NAT design $j6 pre $j7 | walks ALL48 $w1 CLOSE $w2 CLOSE_NAT $w3"
  ;;
secondary)
  # the penalty re-selected on fold 3's 16:00 validation rows, and the walks of the models whose pick changes
  k1=$(sb -J t3s_all -o $L/ALL48.select16.log -c 4 --mem=48G -t 3:00:00 $S ALL48 select16 --jobs 4)
  k2=$(sb -J t3s_rth -o $L/RTH13.select16.log -c 4 --mem=24G -t 2:00:00 $S RTH13 select16 --jobs 4)
  v1=$(sb -J t3v_all -o "$L/ALL48.walk16.%a.log" -c 1 --mem=12G -t 10:00:00 --array=1-29 \
       --dependency=afterok:$k1 $S ALL48 walk16 "${CH48[@]}")
  v2=$(sb -J t3v_rth -o "$L/RTH13.walk16.%a.log" -c 1 --mem=6G -t 4:00:00 --array=1-13 \
       --dependency=afterok:$k2 $S RTH13 walk16 "${CH13[@]}")
  echo "select16 ALL48 $k1 RTH13 $k2 | walk16 ALL48 $v1 RTH13 $v2"
  ;;
score)
  e=$(sb -J t3x -o $L/export.log -c 1 --mem=16G -t 1:00:00 \
      --wrap "bash $S ALL48 export --with16 && bash $S RTH13 export --with16 --ref $REF && bash $S CLOSE export && bash $S CLOSE_NAT export")
  s=$(sb -J t3sc -o $L/score.log -c 1 --mem=16G -t 1:00:00 --dependency=afterok:$e \
      --wrap "cd $B/code && OPENBLAS_CORETYPE=Prescott /scratch1/jc_905/wsweep/venv/bin/python experiments/linear_train3way_score.py results/linear_train3way_2014start_w3000 $REF")
  echo "export $e score $s"
  ;;
esac
