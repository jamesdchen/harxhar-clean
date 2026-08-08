#!/bin/bash
# EW-gram fleet, released after the one-chunk canary passed (chunk_049:
# 12 boundaries, H selections spread across the grid, n_eff sane, 162
# tail evals — matches the builder's healthy-chunk spec exactly).
cd /u/scratch/j/jamesdc1/harxhar-clean
for a in blk3_ew_tuned blk3_ewFixed_H3000 blk3_ewFixed_H12000; do
  qsub -N unif_$a -v ARM=$a -tc 60 jobs/sge/unification_serial.sge
done
