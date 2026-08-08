#!/bin/bash
# One-chunk canary for the serial-correlation-corrected JS arm, per the
# standing one-chunk-canary rule: a single mid-history task whose meta
# (tau_resid plausible 50-300, factor below the uncorrected 0.77, or the
# documented positive-part floor) is read before any fleet is released.
cd /u/scratch/j/jamesdc1/harxhar-clean
qsub -N unif_canary_jsneff -v ARM=blk3_jsNeff_tuned -t 50-50 jobs/sge/unification_serial.sge
