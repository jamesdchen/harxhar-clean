#!/bin/bash
# 2026-10-22 final pooled MinGRU: submit the GPU jobs of one stage from the window's CARC root.
#   tune <round>   one GPU job per file of inputs/tune_files_r<round>.csv (written by the tune-jobs stage)
#   walk           one GPU job per file of inputs/walk_files.csv (written by the walk-jobs stage)
# ROOT: /home1/jc_905/pmf (W = 3,000, default) or /home1/jc_905/pmf35 (W = 3,500, owner 2026-10-22); the job logs go
# to $ROOT/logs.  DEP (optional): a Slurm dependency for every job (e.g. afterok:<gate>:<canary>); TLIM: the time
# limit; NICE (optional): --nice (the W = 3,500 jobs run after the W = 3,000 jobs of the same priority).
# Prints "<file> <job id>" lines and appends them to inputs/submitted.txt.
set -euo pipefail
ROOT=${ROOT:-/home1/jc_905/pmf}
cd "$ROOT"
G=/home1/jc_905/pmf/pooled_mingru_final_gpu_carc.sbatch
MODE=$1
case $MODE in
  tune) R=$2; LIST=inputs/tune_files_r$R.csv; SUB=tune ;;
  walk) LIST=inputs/${WLIST:-walk_files.csv}; SUB=refit ;;
  *) echo "usage: $0 tune <round> | walk     (walk: WLIST=probe_files.csv for the timing probe)"; exit 1 ;;
esac
tail -n +2 "$LIST" | while IFS=, read -r file set conc rest; do
  tag=${file%.txt}; tag=${tag//./_}
  isub=""
  [ "$SUB" = refit ] && isub=$(echo "$rest" | cut -d, -f1)  # walk_files.csv: file,set,conc,initsub,n_jobs
  id=$(sbatch --parsable --job-name=pmf${SUB:0:1} --mem=24G --time=${TLIM:-4:00:00} ${DEP:+--dependency=$DEP} \
       ${NICE:+--nice=$NICE} -o "$ROOT/logs/%x_%j.out" \
       --export=NONE,ROOT=$ROOT,JOBS=$file,SUB=$SUB,TAG=$tag,CONC=$conc,INITSUB=$isub,POST=pooled_mingru_final_post_check.py $G)
  echo "$file $id" | tee -a inputs/submitted.txt
done
