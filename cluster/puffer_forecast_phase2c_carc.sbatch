#!/bin/bash
# 2026-10-13 puffer_forecast Phase 2c walk-forward on one CARC GPU of ONE required type (GPU_TYPE): Phase 2 (5 blocks, re-tune every
# 250 sessions with phase2_select.py, cold refit at every session, 5 seeds) with the prescaled design fed as
# is (xstd=0, experiments/puffer_forecast/phase2c.py), the regular stack's window mask (mask=1), no clip,
# Phase 2's 12-configuration grid; trainer pool (--pool).  Gates first:
#   A  input check: gradcheck dumps on 18 windows (Z, first minibatch, kept columns), compared locally with
#      the prescaled design's kept columns bit for bit (phase2c.py check)
#   B  xstd=1 reproduces the Phase 2b binary: OLD (native/, Phase 2b source) vs NEW (native_2c/, --pool) on
#      the Phase 2b block-0 tune jobs and 500 of its refits, same GPU, compared byte for byte locally
#   cd $HOME/puffer_forecast && sbatch --export=NONE,GPU_TYPE=p100 --constraint=p100 puffer_forecast_phase2c_carc.sbatch
# GPU_TYPE is REQUIRED (user, 2026-10-13: every NN number for the paper from ONE recorded GPU type) and must
# equal the --constraint; the job checks the GPU it got against it and records it in gpu_type.txt / env.txt
# (phase2_score.py refuses a run whose blocks saw more than one GPU model).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=pf_phase2c
#SBATCH --output=logs/pf_phase2c_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
RUN=$ROOT/runs/phase2c_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -1; gcc --version | head -1
  git -C "$PUFFER" log -1 --format='%H %ci'; } > env.txt 2>&1
cat env.txt
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE (p100 / v100 / a40 / a100 / l40s) is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; a40) M="A40" ;; a100) M="A100" ;; l40s) M="L40S" ;;
  *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[ "${SLURM_JOB_CONSTRAINTS:-$GPU_TYPE}" = "$GPU_TYPE" ] || { echo "--constraint ${SLURM_JOB_CONSTRAINTS:-} != GPU_TYPE $GPU_TYPE"; exit 1; }
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT|constraint=${SLURM_JOB_CONSTRAINTS:-unset}" | tee gpu_type.txt >> env.txt
[ "$(git -C "$PUFFER" rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
git -C "$PUFFER" diff --quiet -- src || { echo "PufferLib src/ modified"; exit 1; }
md5sum "$ROOT/native/forecast_train.cu" "$ROOT/native_2c/forecast_train.cu" "$ROOT/phase2_select.py" \
  "$ROOT/inputs/phase2c/blocks.json" "$ROOT"/inputs/phase2c/tune_block*.txt "$ROOT/inputs/phase2c/maskcheck.txt"   "$ROOT"/native_2c/*.cuh "$ROOT"/data/*/* > code_md5.txt
# (2026-10-13 gate 4, after 13082282 ran: byte-identical) 32 hardware queues instead of 8 for the 2 streams
# per model
export CUDA_DEVICE_MAX_CONNECTIONS=32
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
prebuilt "native" "$RUN/ft_old"
prebuilt "native_2c" "$RUN/forecast_train"
BIN=$RUN/forecast_train
PY=/usr/bin/python3
# ---- gate A: input check dumps
"$BIN" gradcheck --jobs "$ROOT/inputs/phase2c/maskcheck.txt" --out "$RUN/maskcheck" > "$RUN/maskcheck.log" 2>&1 || { cat "$RUN/maskcheck.log"; exit 1; }
rm -f "$RUN"/maskcheck/*.params0.f32 "$RUN"/maskcheck/*.grads.f32
# ---- gate B: xstd=1 (Phase 2b jobs) reproduces the Phase 2b binary
PBR=$ROOT/runs/phase2b_13080112/block0/jobs_refit.txt
awk 'NR <= 500' "$PBR" > "$RUN/gateB_refit500.txt"
./ft_old run --jobs "$ROOT/inputs/phase2b/tune_block0.txt" --out gateB_old_tune --concurrent 48 --chunk 5 > gateB_old_tune.log 2>&1
"$BIN" run --jobs "$ROOT/inputs/phase2b/tune_block0.txt" --out gateB_new_tune --concurrent 48 --chunk 5 --pool > gateB_new_tune.log 2>&1
./ft_old run --jobs "$RUN/gateB_refit500.txt" --out gateB_old_refit --concurrent 48 --chunk 5 > gateB_old_refit.log 2>&1
"$BIN" run --jobs "$RUN/gateB_refit500.txt" --out gateB_new_refit --concurrent 48 --chunk 5 --pool > gateB_new_refit.log 2>&1
# ---- Phase 2c
for k in 0 1 2 3 4; do
  mkdir -p "$RUN/block${k}"
  echo "BLOCK $k tune $(date)"
  "$BIN" run --jobs "$ROOT/inputs/phase2c/tune_block${k}.txt" --out "$RUN/block${k}/tune" --concurrent 48 --chunk 5 --pool
  "$PY" "$ROOT/phase2_select.py" --blocks "$ROOT/inputs/phase2c/blocks.json" --k "$k" \
    --tune-out "$RUN/block${k}/tune" --data-root "$ROOT/data" \
    --out "$RUN/block${k}/jobs_refit.txt" --selection "$RUN/block${k}/selection.json"
  echo "BLOCK $k refit $(date)"
  "$BIN" run --jobs "$RUN/block${k}/jobs_refit.txt" --out "$RUN/block${k}/refit" --concurrent 48 --chunk 5 --pool
done
date
echo PHASE2_DONE
