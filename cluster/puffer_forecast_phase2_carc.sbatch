#!/bin/bash
# 2026-10-13 puffer_forecast Phase 2 walk-forward on one CARC L40S -- the same job as
# cluster/puffer_forecast_phase2_h2.sh (same PufferLib commit, same native source, same inputs/phase2
# blocks.json + tune_block*.txt, same phase2_select.py, same flags: sm_89, -O2, fp32), submitted on CARC
# because the Hoffman2 L40S queue had not started it after 1.5 h; whichever starts first runs, the other is
# held.  Toolchain differs by cluster (CARC: CUDA 12.9.1 / gcc 14.3.0; Hoffman2: CUDA 12.3 / gcc 11.3.0).
#   cd $HOME/puffer_forecast && sbatch --export=NONE puffer_forecast_phase2_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=pf_phase2
#SBATCH --output=logs/pf_phase2_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
RUN=$ROOT/runs/$SLURM_JOB_ID
mkdir -p "$RUN"
cd "$RUN"
ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -1; gcc --version | head -1
  git -C "$PUFFER" log -1 --format='%H %ci'; } > env.txt 2>&1
cat env.txt
[ "$(git -C "$PUFFER" rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
git -C "$PUFFER" diff --quiet -- src || { echo "PufferLib src/ modified"; exit 1; }
md5sum "$ROOT/native/forecast_train.cu" "$ROOT/phase2_select.py" "$ROOT/inputs/phase2/blocks.json" "$ROOT"/inputs/phase2/tune_block*.txt > code_md5.txt
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
prebuilt "native" "$RUN/forecast_train"
BIN=$RUN/forecast_train
PY=/usr/bin/python3
for k in 0 1 2 3 4; do
  mkdir -p "$RUN/block${k}"
  echo "BLOCK $k tune $(date)"
  "$BIN" run --jobs "$ROOT/inputs/phase2/tune_block${k}.txt" --out "$RUN/block${k}/tune" --concurrent 48 --chunk 5
  "$PY" "$ROOT/phase2_select.py" --blocks "$ROOT/inputs/phase2/blocks.json" --k "$k" \
    --tune-out "$RUN/block${k}/tune" --data-root "$ROOT/data" \
    --out "$RUN/block${k}/jobs_refit.txt" --selection "$RUN/block${k}/selection.json"
  echo "BLOCK $k refit $(date)"
  "$BIN" run --jobs "$RUN/block${k}/jobs_refit.txt" --out "$RUN/block${k}/refit" --concurrent 48 --chunk 5
done
date
echo PHASE2_DONE
