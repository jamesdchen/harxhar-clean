#!/bin/bash
# FM-specific tuning sweep (Hoffman2/SGE). Run AFTER the fm_ sweep collects + this code is deployed,
# so the deploy doesn't contaminate the running fm_ arrays. Exercises the two FM-specific knobs at
# depth 0 (the EBM-match) + a gated depth-1:
#   rank   : FM capacity / primary regularizer (low = more shrinkage)
#   wd_v   : separate STRONGER L2 on the factors V than the rest (where FM overfitting lives)
#   fm_linear=false : pure-pairwise (drop main effects the base+tree already took)
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # depth rank wdv fmlin label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=$1,RANK=$2,WDV=$3,FMLIN=$4,WD=0.1,RDIAG=false,EXPERT=fm,LBL=$5 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$5 sig_collect.sge)
  echo "  $5 (d$1 rank$2 wdv$3 lin$4) dig=$J col=$C"
}
echo FMTUNE_SUBMITTED
sub 0 2 null true  fmt_d0_r2      # low rank (strong capacity control)
sub 0 4 null true  fmt_d0_r4
sub 0 8 1.0  true  fmt_d0_wdv1    # separate stronger factor decay (the key FM regularizer)
sub 0 8 3.0  true  fmt_d0_wdv3
sub 0 4 1.0  false fmt_d0_pure    # pure-pairwise (no main effects) + strong factor decay
sub 1 4 1.0  false fmt_d1_pure    # gated pure-pairwise -> the config that could BEAT the EBM
