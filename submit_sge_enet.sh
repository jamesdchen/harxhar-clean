#!/bin/bash
# UN-STARVE BOTH STAGES: run the regime MoE on the PLAIN enet base (base_kind=enet, NO close-gated
# distillation -> the clock regime is intact) instead of linbest (which pre-ate it). Same cell throughout
# so arms are directly comparable. Does the gate now route on CLOCK and match the d8/EBM regime?
#   en_noreg      = enet + d8, no regime            (baseline: hard router as sole regime)
#   en_d8_moe     = enet + d8 + MoE (forward)        (gate STARVED by d8, on plain base)
#   en_inv        = enet + MoE-first + d8 mop-up      (gate un-starved of d8)
#   en_inv_nod8_* = enet + MoE ONLY (no d8)           (PUREST un-starved cascade: additive base + gate-as-regime)
# All MoE arms: REGIME_FULL (gate finds the regime across all hours), basis-FM hinge, suppressors off, diag.
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=ebm_all_buckets_tw1000_enet_rf480_slim
E="EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,GWD=0.0,AUXW=0.0"
moe () {  # rinv rnog gtemp label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=2,GATEKIND=tree,NREG=4,$E,RFULL=1,RINV=$1,RNOG=$2,GTEMP=$3,RDIAG=false,MOE_DIAG=1,LBL=$4 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$4 sig_collect.sge)
  echo "  $4 (rinv=$1 nod8=$2 gtemp=$3) dig=$J col=$C"
}
echo ENET_SUBMITTED
J=$(qsub -terse -v CID=$CID,NOREG=1,LBL=en_noreg moe_dig.sge | cut -d. -f1)
C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=en_noreg sig_collect.sge)
echo "  en_noreg (enet+d8, NO-REGIME) dig=$J col=$C"
moe 0 0 1.0  en_d8_moe          # forward: d8 then MoE (starved)
moe 1 0 1.0  en_inv             # inverted: MoE then d8 mop-up
moe 1 1 1.0  en_inv_nod8_soft   # MoE only, no d8 (purest un-starve), soft gate
moe 1 1 8.0  en_inv_nod8_sharp  # MoE only, no d8, sharp gate
