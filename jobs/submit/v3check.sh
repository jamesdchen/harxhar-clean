#!/bin/bash
# Submit the panel-v3 verification job (isolated cache + results; safe
# beside v2). Run ON the CARC login node.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch jobs/slurm/unif_v3check.sbatch
squeue -u "$USER" -h | wc -l
