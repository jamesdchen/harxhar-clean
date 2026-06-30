#!/bin/bash
# THE NEW-HERO EXPERIMENT: EBM (+) multi-task-FM ensemble regime stage on linbest, full-OOS, vs EBM-alone 0.12033.
# Pre-check (local, realrank): corr(EBM,MTFM)=0.34 (diverse!), ensemble beats EBM. This is the deployment test
# with the REAL EBM + full data + folds. ENS_W = weight on EBM; MT_AUXW = the un-starving aux-head weight.
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # ens_w mt_auxw label
  local J C
  J=$(qsub -terse -v CID=$CID,RMODEL=ebm_mtfm,RDEPTH=2,EXPERT=fm,BASIS=hinge,NKNOTS=3,ENS_W=$1,MT_AUXW=$2,MT_NBAGS=8,MT_RANK=4,MT_EPOCHS=250,EBM_CFG={},RDIAG=false,LBL=$3 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$3 sig_collect.sge)
  echo "  $3 (ens_w=$1 mt_auxw=$2) dig=$J col=$C"
}
echo ENSEMBLE_SUBMITTED
sub 0.4 0.3  ens_w04_aux03    # robust pre-check winner
sub 0.2 1.0  ens_w02_aux10    # aggressive pre-check winner (bigger local gain)
sub 1.0 0.3  ens_ebmonly      # control = EBM alone (ens_w=1) -> must reproduce ~0.12033 (sanity)
