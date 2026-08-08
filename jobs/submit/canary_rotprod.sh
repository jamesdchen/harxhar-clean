#!/bin/bash
# One-chunk canary for the standardization-corrected full-rank rotation.
# Healthy = |dQLIKE| vs blk2_gated_tuned's same chunk at float noise
# (~1e-12): the offline identity holds at 3.7e-15, so anything material
# here means the standardization explanation is wrong, not the rotation.
cd /u/scratch/j/jamesdc1/harxhar-clean
qsub -N unif_canary_rotprod -v ARM=blk2_rotFullProd_tuned -t 51-51 jobs/sge/unification_serial.sge
