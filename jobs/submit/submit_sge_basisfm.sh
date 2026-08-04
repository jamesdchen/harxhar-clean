#!/bin/bash
# Basis-expansion FM sweep (Hoffman2/SGE). HINGE basis -> nonlinear 1D shapes + nonlinear 2D interactions
# = a differentiable GA2M (EBM-style shapes). The contingency for if plain fm_d0 >> EBM (bilinear too weak).
# Run AFTER fm_ collects + this code is deployed (so the deploy doesn't contaminate the fm_ arrays).
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # depth rank wdv fmlin nknots label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=$1,RANK=$2,WDV=$3,FMLIN=$4,BASIS=hinge,NKNOTS=$5,WD=0.1,RDIAG=false,EXPERT=fm,LBL=$6 jobs/sge/moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$6 jobs/sge/sig_collect.sge)
  echo "  $6 (d$1 r$2 wdv$3 lin$4 knots$5 hinge) dig=$J col=$C"
}
echo BASISFM_SUBMITTED
sub 0 4 1.0 false 3 bfm_d0_pure   # hinge pure-pairwise + strong V-decay = EBM-match with nonlinear shapes
sub 0 4 1.0 true  3 bfm_d0_lin    # hinge + linear main effects
sub 0 4 1.0 false 5 bfm_d0_k5     # more knots (flexier shapes)
sub 1 4 1.0 false 3 bfm_d1_pure   # gated hinge pure-pairwise -> the config that could BEAT the EBM
