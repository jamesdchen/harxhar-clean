#!/bin/bash
# One-chunk canaries: PCR re-canary on the extended 1e0..1e6 grid (previous
# canary pinned 67% at the old 1e4 top), and the repair-ladder's PLS arm
# (newest machinery: frozen supervised weights; healthy = stable
# weights_sha256 across chunks + no endpoint pileup at the trans_sub top).
cd /u/scratch/j/jamesdc1/harxhar-clean
qsub -N unif_canary_pcr2 -v ARM=blk2_rotPredTwentyK_tuned -t 51-51 jobs/sge/unification_serial.sge
qsub -N unif_canary_pls -v ARM=blk2_plsTwentyRungsInd_tuned -t 51-51 jobs/sge/unification_serial.sge
