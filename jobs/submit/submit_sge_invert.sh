#!/bin/bash
# INVERT the cascade (user's idea): MoE regime FIRST on the un-starved r1 = y - base (gate's first crack
# at the routable structure), then d8 soaks up the leftover. vs the default d8-first (gate starved on r2).
# Compare full-OOS qlike to the d8-first bar 0.12080 + read gate Hnorm: does the gate route on un-starved r1?
#   inv_full_*  = inverted, REGIME_FULL (MoE all hours first)            <- the test
#   fwd_full_soft = forward, REGIME_FULL (d8 first, MoE all hours)       <- isolates ordering vs all-hours
# Gate = gfix basis-FM hinge, both suppressors off (gwd0+aux0), diag on.
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
E="EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,GWD=0.0,AUXW=0.0"
sub () {  # rinv rfull gtemp label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=2,GATEKIND=tree,NREG=4,$E,GTEMP=$3,RINV=$1,RFULL=$2,RDIAG=false,MOE_DIAG=1,LBL=$4 jobs/sge/moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$4 jobs/sge/sig_collect.sge)
  echo "  $4 (rinv=$1 rfull=$2 gtemp=$3) dig=$J col=$C"
}
echo INVERT_SUBMITTED
sub 1 1 1.0  inv_full_soft     # MoE-first all-hours, soft gate, d8 mop-up
sub 1 1 8.0  inv_full_sharp    # MoE-first all-hours, sharp gate, d8 mop-up
sub 0 1 1.0  fwd_full_soft     # control: d8-first, MoE all-hours (regime_full forward)
sub 1 0 1.0  inv_h16_soft      # MoE-first on the close window only, d8 mop-up
