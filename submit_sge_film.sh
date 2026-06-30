#!/bin/bash
# FiLM (continuous context modulation) vs the discrete gate. FiLM modulates each FM latent interaction
# smoothly by raw context (partial pooling) instead of routing a hard partition. Local smoke: it gracefully
# degrades to plain-FM on the d8-starved leftover. Fair test here: (1) deployment number on linbest (vs gate
# 0.12080 / EBM 0.12033); (2) the UN-STARVED enet base where the regime is intact -- does FiLM ENGAGE + pay?
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
E="EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,GWD=0.0,AUXW=0.0"
sub () {  # cid rfull glr label
  local J C
  J=$(qsub -terse -v CID=$1,RDEPTH=2,GATEKIND=film,GCTX=raw,NREG=4,$E,RFULL=$2,GLR=$3,RDIAG=false,MOE_DIAG=1,LBL=$4 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$1,ARM=resid_regime,LBL=$4 sig_collect.sge)
  echo "  $4 (cid=$1 rfull=$2 film_lr=$3) dig=$J col=$C"
}
LIN=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
ENET=ebm_all_buckets_tw1000_enet_rf480_slim
echo FILM_SUBMITTED
sub $LIN  0 null  film_lin            # linbest, forward h16-19 -> deployment number vs 0.12080
sub $LIN  0 0.01  film_lin_hilr       # + 10x FiLM-head LR: force engagement, does it then help or hurt?
sub $ENET 1 null  film_enet_full      # enet (un-starved) all-hours -> does FiLM engage on the intact regime?
