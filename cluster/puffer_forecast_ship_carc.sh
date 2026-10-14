#!/bin/bash
# 2026-10-12 puffer_forecast: ship code + inputs to CARC -- PHASE 1 ONLY (run on the laptop, Git Bash, repo root).
#   bash cluster/puffer_forecast_ship_carc.sh      (then: ssh usc-discovery; cd ~/puffer_forecast;
#                                                    sbatch --export=NONE puffer_forecast_phase1_carc.sbatch)
# It ships native/ and the Phase 1 inputs only.  Later phases were shipped by hand into their own directories
# (native_2c/, native_of3/, native_of4/, inputs/phase2*/, inputs/overfit_*/, inputs/gate4/, phase2_select.py);
# see each sbatch header.  $R/data is copied only when absent (FORCE_DATA=1 replaces it, which breaks any
# running job whose run directory symlinks it -- check squeue first).
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=${SSH:-/c/Windows/System32/OpenSSH/ssh.exe}
W=results/puffer_forecast_2026-10-12/_work
tar cf - experiments/puffer_forecast/native/forecast_train.cu experiments/puffer_forecast/native/puffer_substrate.cuh \
  cluster/puffer_forecast_build_carc.sh cluster/puffer_forecast_phase1_carc.sbatch \
  "$W"/data "$W"/jobs_*.txt "$W"/losscheck_*.bin "$W"/prep.json \
  | "$SSH" -o BatchMode=yes "${CARC_HOST:-usc-discovery1}" "set -e; R=\$HOME/puffer_forecast; FORCE_DATA=${FORCE_DATA:-0};
      mkdir -p \$R/stage \$R/native \$R/inputs \$R/logs; cd \$R/stage; find . -mindepth 1 -delete; tar xf -;
      cp experiments/puffer_forecast/native/* \$R/native/; cp cluster/puffer_forecast_*_carc.* \$R/;
      if [ ! -e \$R/data ]; then cp -r results/puffer_forecast_2026-10-12/_work/data \$R/data;
      elif [ \$FORCE_DATA = 1 ]; then rm -rf \$R/data.old; mv \$R/data \$R/data.old; cp -r results/puffer_forecast_2026-10-12/_work/data \$R/data;
      else echo 'data/ exists: kept (FORCE_DATA=1 to replace)'; fi;
      cp results/puffer_forecast_2026-10-12/_work/jobs_*.txt results/puffer_forecast_2026-10-12/_work/losscheck_*.bin results/puffer_forecast_2026-10-12/_work/prep.json \$R/inputs/;
      cd \$R && md5sum native/forecast_train.cu native/puffer_substrate.cuh data/*/X.f32 inputs/jobs_phase1.txt"
