#!/bin/bash
# Post-canary fleet release: the seven tuned-ridge PCR arms (extended
# rotexog grid — re-canary showed top endpoint at 8%, spread selections)
# and the non-PLS rungs of the capture ladder. The two PLS arms are
# deliberately ABSENT: their canary showed healthy penalties but the
# pls_diag provenance hash is missing from production meta, and
# synthetic-verified-but-production-absent is the exact pattern behind
# the void JS fleet. They fleet after the hash lands and one re-canary.
cd /u/scratch/j/jamesdc1/harxhar-clean
for a in blk2_rotVarFullK_tuned blk2_rotVarThirtyK_tuned blk2_rotVarTwentyK_tuned \
         blk2_rotPredThirtyK_tuned blk2_rotPredTwentyK_tuned blk2_rotPredTenK_tuned \
         blk2_rotPredFiveK_tuned \
         blk2_gfull_tuned blk2_gFortyRungs_tuned blk2_gFortyRungsInd_tuned; do
  qsub -N unif_$a -v ARM=$a -tc 60 jobs/sge/unification_serial.sge
done
