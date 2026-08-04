#!/bin/bash
# THE PROPOSED NEW HERO: EBM (+) ghat-gated multi-task-FM ensemble regime stage, full-OOS on linbest vs EBM 0.12033.
# Recursive un-starving: r1-aux un-starves the PREDICTION; |ghat|-gated per-row aux un-starves the ANTICIPATION
# (spend aux where d8 took a big bite). Local: ghat-gating ~doubled the gain + removed the reversal.
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # ens_w gate label
  local J C
  J=$(qsub -terse -v CID=$CID,RMODEL=ebm_mtfm,RDEPTH=2,EXPERT=fm,ENS_W=$1,MT_GATE=$2,MT_AUXHI=0.6,MT_AUXW=0.3,MT_NBAGS=8,MT_RANK=4,MT_EPOCHS=250,EBM_CFG={},RDIAG=false,LBL=$3 jobs/sge/moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$3 jobs/sge/sig_collect.sge)
  echo "  $3 (ens_w=$1 gate=$2) dig=$J col=$C"
}
echo GGATED_ENSEMBLE_SUBMITTED
sub 1.0 uniform gg_ebmonly      # sanity = EBM alone -> must reproduce ~0.12033
sub 0.3 uniform gg_uniform_w03  # EBM + uniform-aux MTFM (lower bound)
sub 0.3 ghat    gg_ghat_w03     # EBM + ghat-gated MTFM = THE PROPOSAL
sub 0.2 ghat    gg_ghat_w02     # omega-sweep
sub 0.4 ghat    gg_ghat_w04     # omega-sweep
