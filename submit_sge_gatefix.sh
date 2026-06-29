#!/bin/bash
# Gate-fix re-run: the load-balance aux loss was suppressing per-sample routing (gate went inert -> all
# variants collapsed to no-gate). With the fixed mean-utilization balance + a gentler aux_weight, this
# re-tests whether routing ACTUALLY helps. On the basis-FM hinge pure-pairwise expert.
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # depth gatekind nreg auxw label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=$1,GATEKIND=$2,NREG=$3,AUXW=$4,EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,RDIAG=false,LBL=$5 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$5 sig_collect.sge)
  echo "  $5 (d$1 gate=$2 K$3 auxw$4) dig=$J col=$C"
}
echo GATEFIX_SUBMITTED
sub 0 tree      1 0.1 gfix_d0         # no-gate control (baseline)
sub 2 tree      4 0.1 gfix_tree_d2    # FIXED soft-tree gate
sub 1 attention 4 0.1 gfix_attn_k4    # FIXED attention gate
sub 1 attention 6 0.1 gfix_attn_k6    # FIXED attention, K=6
