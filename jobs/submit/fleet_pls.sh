#!/bin/bash
# PLS fleet, released after the re-canary showed a fully populated
# pls_diag (sha 3c3186ac..., shape [33,20], frame_rows [24000,48000])
# under the write-time provenance gate.
cd /u/scratch/j/jamesdc1/harxhar-clean
for a in blk2_plsTwentyRungsInd_tuned blk2_plsTenRungsInd_tuned; do
  qsub -N unif_$a -v ARM=$a -tc 60 jobs/sge/unification_serial.sge
done
