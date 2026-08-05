#!/bin/bash
# UNDER-TRAINING test (user's idea): the gate trains FULL-BATCH (1 step/epoch), 150 epochs, patience 12 ->
# on a 93%-noise target val bottoms out fast, so the gate's hyperplane DIRECTION W may never train to a
# useful axis (gate_temp sharpens the split steepness but not where it points). Train longer + faster gate
# LR; if a WELL-trained confident gate beats 0.12079, under-training hid it; if it ties, routing is earned.
# On the d8-starved residual (default forward, h16-19) -- isolates the training-length variable.
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
E="EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,GWD=0.0,AUXW=0.0"
sub () {  # epochs patience glr gtemp label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=2,GATEKIND=tree,NREG=4,$E,EPOCHS=$1,PATIENCE=$2,GLR=$3,GTEMP=$4,RDIAG=false,MOE_DIAG=1,LBL=$5 jobs/sge/moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$5 jobs/sge/sig_collect.sge)
  echo "  $5 (ep=$1 pat=$2 glr=$3 gtemp=$4) dig=$J col=$C"
}
echo TRAINLONG_SUBMITTED
sub 800 80  null  4.0  tl_long_soft        # 800 epochs, default gate LR, mid temp
sub 800 80  0.01  4.0  tl_long_glr         # + 10x gate LR (train W's direction faster)
sub 800 80  0.01  8.0  tl_long_glr_sharp   # + sharp: well-trained AND confident routing
