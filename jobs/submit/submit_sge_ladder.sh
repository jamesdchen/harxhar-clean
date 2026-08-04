#!/bin/bash
# STRUCTURAL-STARVATION depth ladder. The regime target r2 = r1 - d8.predict(X) is orthogonal to the
# upstream tree's partition of X, so a soft gate routing on X finds nothing (liveness null = earned).
# Test the fix: SHRINK the upstream global tree (GDEPTH 2/4/8) so it leaves routable structure, and see
# whether the gate WAKES (Hnorm drops) and the regime stage RECOVERS more at shallower depth.
# Metric: qlike_h16-19 GAIN = noreg - regime, at each depth (full-OOS is confounded: shallow global is
# worse off-window too). Gate config = gfix basis-FM hinge, both suppressors off (gwd0+aux0).
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
E="EXPERT=fm,BASIS=hinge,NKNOTS=3,RANK=4,FMLIN=false,WDV=1.0,WD=0.1,GWD=0.0,AUXW=0.0"
moe () {  # gdepth gtemp label
  local J C
  J=$(qsub -terse -v CID=$CID,GDEPTH=$1,RDEPTH=2,GATEKIND=tree,NREG=4,$E,GTEMP=$2,RDIAG=false,MOE_DIAG=1,LBL=$3 jobs/sge/moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$3 jobs/sge/sig_collect.sge)
  echo "  $3 (gdepth=$1 gtemp=$2) dig=$J col=$C"
}
noreg () {  # gdepth label
  local J C
  J=$(qsub -terse -v CID=$CID,GDEPTH=$1,NOREG=1,LBL=$2 jobs/sge/moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$2 jobs/sge/sig_collect.sge)
  echo "  $2 (gdepth=$1 NO-REGIME) dig=$J col=$C"
}
echo LADDER_SUBMITTED
for d in 2 4 8; do
  noreg $d      ladder_d${d}_noreg
  moe   $d 1.0  ladder_d${d}_soft
  moe   $d 8.0  ladder_d${d}_sharp
done
