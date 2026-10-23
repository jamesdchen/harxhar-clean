#!/bin/bash
# 2026-10-22 final pooled MinGRU: submit the GPU jobs of one stage from /home1/jc_905/pmf.
#   tune <round>   one GPU job per file of inputs/tune_files_r<round>.csv (written by the tune-jobs stage)
#   walk           one GPU job per file of inputs/walk_files.csv (written by the walk-jobs stage)
# DEP (optional): a Slurm dependency for every job (e.g. afterok:<gate>:<canary>); TLIM: the time limit.
# Prints "<file> <job id>" lines and appends them to inputs/submitted.txt.
set -euo pipefail
cd /home1/jc_905/pmf
G=pooled_mingru_final_gpu_carc.sbatch
MODE=$1
case $MODE in
  tune) R=$2; LIST=inputs/tune_files_r$R.csv; SUB=tune ;;
  walk) LIST=inputs/walk_files.csv; SUB=refit ;;
  *) echo "usage: $0 tune <round> | walk"; exit 1 ;;
esac
tail -n +2 "$LIST" | while IFS=, read -r file set conc rest; do
  tag=${file%.txt}; tag=${tag//./_}
  id=$(sbatch --parsable --job-name=pmf${SUB:0:1} --mem=24G --time=${TLIM:-4:00:00} ${DEP:+--dependency=$DEP} \
       --export=NONE,JOBS=$file,SUB=$SUB,TAG=$tag,CONC=$conc $G)
  echo "$file $id" | tee -a inputs/submitted.txt
done
