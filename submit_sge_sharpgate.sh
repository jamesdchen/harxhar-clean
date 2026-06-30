#!/bin/bash
# CLOSING arm of the liveness test: the soft gate sits ~uniform (Hnorm~0.95) in every free arm, buying 0
# QLIKE. Residual doubt: does the SOFT sigmoid structurally fail to test a HARD partition? Force it sharp
# (gate_temp >> 1 -> near-hard splits), with BOTH suppressors off (gwd0+aux0) so the optimizer can place a
# confident partition if one helps. If a confidently-routing gate STILL ties 0.12079 -> routing earned.
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # gtemp label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=2,GATEKIND=tree,NREG=4,EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,GWD=0.0,AUXW=0.0,GTEMP=$1,RDIAG=false,MOE_DIAG=1,LBL=$2 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$2 sig_collect.sge)
  echo "  $2 (gate_temp=$1 gwd0 aux0) dig=$J col=$C"
}
echo SHARPGATE_SUBMITTED
sub 4   live_d2_sharp4    # sharp, free gate
sub 16  live_d2_sharp16  # near-hard routing
