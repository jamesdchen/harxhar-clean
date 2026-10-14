#!/bin/bash
# 2026-10-13 puffer_forecast: gate of the fixed fc_train source ($NEW/, default native_of2/: weights_init and the allocators'
# zeroing ordered on the legacy stream).  On one GPU:
#   old_{A,B}  the Phase 2c binary (native_2c/), --pool, twice      (its run-to-run behaviour)
#   new_{A,B}  the fixed binary, --pool, twice                       (must be identical to each other)
#   new_wave   the fixed binary, wave path (no --pool)               (must equal new_A)
#   each on the Phase 2c tune block 0 (120 jobs) and 500 of its block-0 refits; compared locally
#   fct_{pool,wave}  fc_train jobs (inputs/overfit_gate/fctrain_jobs.txt, save_params=1): train-row forecasts
#              checked locally against a float64 forward of the saved final weights
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_overfit_gate2_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=pf_ofgate2
#SBATCH --output=logs/pf_ofgate2_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
NEW=${NEW:-native_of2}
RUN=$ROOT/runs/ofgate2_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/native_2c/forecast_train.cu" "$ROOT/$NEW/forecast_train.cu" "$ROOT/inputs/overfit_gate/fctrain_jobs.txt"; } > env.txt 2>&1
cat env.txt
# the trainer, built ONCE per submission by puffer_forecast_build_once_carc.sbatch (gate 4, 2026-10-13), not in
# this job:  B=$(sbatch --parsable --export=NONE,SRC=<dir> puffer_forecast_build_once_carc.sbatch), then submit
# this script with --dependency=afterok:$B
prebuilt() {  # prebuilt SRC_DIR DEST: $ROOT/bin/SRC_DIR/forecast_train, checked against the current source
  local b=$ROOT/bin/$1
  [ -x "$b/forecast_train" ] || { echo "no prebuilt $b/forecast_train (puffer_forecast_build_once_carc.sbatch, SRC=$1)"; exit 1; }
  grep -q "$(md5sum < "$ROOT/$1/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" \
    || { echo "prebuilt $b is stale: $ROOT/$1/forecast_train.cu changed since it was built"; exit 1; }
  cp "$b/forecast_train" "$2"
  { echo "prebuilt $1:"; cat "$b/provenance.txt"; } >> env.txt
}
prebuilt "native_2c" "$RUN/ft_old"
prebuilt "$NEW" "$RUN/ft_new"
export TMPDIR=$RUN/tmp; mkdir -p "$TMPDIR"  # nvcc temporaries (a node /tmp read failure killed 13083704)
awk 'NR <= 500' "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" > refit500.txt
TUNE=$ROOT/inputs/phase2c/tune_block0.txt
for v in old_A old_B new_A new_B; do
  b=./ft_${v%_*}
  $b run --jobs "$TUNE" --out "${v}_tune" --concurrent 48 --chunk 5 --pool > "${v}_tune.log" 2>&1
  $b run --jobs refit500.txt --out "${v}_refit" --concurrent 48 --chunk 5 --pool > "${v}_refit.log" 2>&1
done
./ft_new run --jobs "$TUNE" --out new_wave_tune --concurrent 48 --chunk 5 > new_wave_tune.log 2>&1
./ft_new run --jobs refit500.txt --out new_wave_refit --concurrent 48 --chunk 5 > new_wave_refit.log 2>&1
./ft_new run --jobs "$ROOT/inputs/overfit_gate/fctrain_jobs.txt" --out fct_pool --concurrent 6 --chunk 5 --pool > fct_pool.log 2>&1
./ft_new run --jobs "$ROOT/inputs/overfit_gate/fctrain_jobs.txt" --out fct_wave --concurrent 6 --chunk 5 > fct_wave.log 2>&1
gzip -f */steps.csv
rm -rf ft_old ft_new "$TMPDIR"
date; echo OFGATE2_DONE
