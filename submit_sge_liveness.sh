#!/bin/bash
# GATE-LIVENESS probe (H1 test): is the d2 soft-tree gate actually routing, or pinned ~uniform?
# Aggregate QLIKE ties (gfix_* all ~0.12080) cannot tell a SUPPRESSED gate from an UNHELPFUL one.
# MOE_DIAG=1 prints, per chunk, the gate's routing distribution at predict (Hnorm/wstd/util/decisive).
# Two suppressor candidates beyond the (already-fixed) load-balance bug:
#   GWD  = gate_weight_decay decoupled from the experts' heavy WD (shared 0.1 shrinks gate W -> uniform)
#   AUXW = load-balance weight (even the fixed mean-util term pulls toward uniform)
# Arms isolate each + the both-off "best chance to wake" arm. Same expert/config as gfix_tree_d2.
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
sub () {  # gwd auxw label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=2,GATEKIND=tree,NREG=4,EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,GWD=$1,AUXW=$2,RDIAG=false,MOE_DIAG=1,LBL=$3 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$3 sig_collect.sge)
  echo "  $3 (gwd=$1 auxw=$2) dig=$J col=$C"
}
echo LIVENESS_SUBMITTED
sub null 0.1  live_d2_base        # CURRENT config (= gfix_tree_d2) + diag: is today's gate live?
sub 0.0  0.1  live_d2_gwd0        # decouple gate decay only
sub null 0.0  live_d2_aux0        # kill load-balance only
sub 0.0  0.0  live_d2_gwd0_aux0   # both off = max chance the gate wakes
