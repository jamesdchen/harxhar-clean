#!/bin/bash
# Linear arms of the learning-curve sweeps on the code of 6c264b7 (STANDARDIZE_PENALTY final: scaled coordinates +
# binomial availability rule; defaults on, as the flag's author ran it).
# usage: sweep_lin.sh pretune | walk [sweep ...]     sweeps: gate s16 s14 (default: all)
#   gate: the 2012-start w1000 4-fold pretune (results/gate_w1000_nofomcrel_carc) against the flag author's
#   s16: --first-day 2016-01-04, --first-fold 3, W = 1,000 .. 4,000;  s14: --first-day 2014-01-02, --first-fold 3,
#   W = 1,000 .. 3,500 (fold 3 alone at every W, the single-fold learning curve)
set -euo pipefail
B=/scratch1/jc_905/wsweep
MODE=$1; shift
SW=${*:-gate s16 s14}
C="epyc-7513|epyc-7542"
cd $B/jobs
FROM=pretuneonly; [ "$MODE" = walk ] && FROM=walk
for sw in $SW; do
  case $sw in
    gate) bash submit_window.sh g1000f results/gate_w1000_nofomcrel_carc 1000 0 $FROM WS_NPY=1,WS_NUMBA=0 "$C" ;;
    s16) for W in 1000 1500 2000 2500 3000 3500 4000; do
           bash submit_window.sh 16w$W results/window_sweep_2016start_2026-10-22/w$W $W 3 $FROM \
             WS_NPY=1,WS_NUMBA=0,WS_FIRST_DAY=2016-01-04 "$C"; done ;;
    s14) for W in 1000 1500 2000 2500 3000 3500; do
           bash submit_window.sh 14w$W results/window_sweep_2014start_2026-10-22/w$W $W 3 $FROM \
             WS_NPY=1,WS_NUMBA=0,WS_FIRST_DAY=2014-01-02 "$C"; done ;;
  esac
done
