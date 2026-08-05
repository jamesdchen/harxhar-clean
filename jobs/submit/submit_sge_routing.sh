#!/bin/bash
# Routing sweep: #1 FREE-the-gate (REGIME_FULL, fit/predict all hours) + #3 ATTENTION router (learned-
# metric soft-kNN over regime prototypes), on the best expert so far (basis-FM hinge pure-pairwise).
# Run AFTER bfm_ collects + this code is deployed (so the deploy doesn't contaminate bfm_).
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # depth gatekind nreg rfull label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=$1,GATEKIND=$2,NREG=$3,RFULL=$4,EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,RDIAG=false,LBL=$5 jobs/sge/moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$5 jobs/sge/sig_collect.sge)
  echo "  $5 (d$1 gate=$2 K$3 full$4) dig=$J col=$C"
}
echo ROUTING_SUBMITTED
sub 2 tree      4 0 rt_tree_d2        # ref: tree gate, hand h16-19 pre-gate (vs bfm_d1)
sub 1 attention 4 0 rt_attn_k4        # #3: attention routing, K=4, pre-gated
sub 1 attention 6 0 rt_attn_k6        # #3: attention routing, K=6
sub 2 tree      4 1 rt_free_tree_d2   # #1: freed tree gate (full data, clock as routing feature)
sub 1 attention 6 1 rt_free_attn_k6   # #1+#3: freed attention routing over the full data
