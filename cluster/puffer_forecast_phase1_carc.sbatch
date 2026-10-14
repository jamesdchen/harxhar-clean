#!/bin/bash
# 2026-10-12 puffer_forecast Phase 1 on one CARC GPU: PufferLib 5.0 install check (upstream's own RL trainer
# built and run briefly on cartpole), then the supervised native MinGRU forecaster: loss-kernel checks,
# gradient dumps, toy trajectories, the window of record (both designs, the grid, 5 seeds, refits,
# QLIKE / Huber variants, determinism duplicates, warm starts) and the concurrency scaling runs.
# Submit from a clean shell (no env activated):  sbatch --export=NONE puffer_forecast_phase1_carc.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=pf_phase1
#SBATCH --output=logs/pf_phase1_%j.out
set -uo pipefail
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 nccl/2.27.7-1-cuda-12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
RUN=$ROOT/runs/$SLURM_JOB_ID
mkdir -p "$RUN"
cd "$RUN"
ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv;
  nvcc --version | tail -1; gcc --version | head -1; git -C "$PUFFER" log -1 --format='%H %cI'; } > env.txt 2>&1
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
# ---- 1. our binary (prebuilt, fat binary; upstream algo.cu / ocean.cu unchanged)
prebuilt native "$RUN/forecast_train"
cp "$ROOT/native/upstream_sha256.txt" . 2>/dev/null || true
BIN=$RUN/forecast_train
# ---- 2. upstream install check: PufferLib 5.0's own native RL trainer (build.sh cartpole), 2 minutes
(
  set +e
  SHIM=$RUN/shim; mkdir -p "$SHIM/bin" "$SHIM/lib"
  printf '#!/bin/bash\nexec "$@"\n' > "$SHIM/bin/ccache"; chmod +x "$SHIM/bin/ccache"   # build.sh hard-codes ccache
  ln -sf /lib64/libgomp.so.1 "$SHIM/lib/libomp5.so"                                       # build.sh links -lomp5
  cd "$PUFFER"
  PATH=$SHIM/bin:$PATH NVCC_ARCH=native NVCC_EXTRA="-I$NCCL_ROOT/include -L$NCCL_ROOT/lib -L$SHIM/lib" \
    timeout 1200 bash build.sh cartpole "$RUN/puffer_cartpole" > "$RUN/upstream_build.log" 2>&1
  echo "upstream build exit $?" >> "$RUN/upstream_build.log"
  if [ -x "$RUN/puffer_cartpole" ]; then
    timeout 240 "$RUN/puffer_cartpole" train --headless --train.total_timesteps=30000000 > "$RUN/upstream_train.log" 2>&1
    echo "upstream train exit $?" >> "$RUN/upstream_train.log"
  fi
) || true
tail -3 upstream_build.log; tail -5 upstream_train.log 2>/dev/null
# ---- 3. loss-kernel check, gradient dumps, toy trajectories
for k in mse huber qlike; do "$BIN" losscheck "$ROOT/inputs/losscheck_$k.bin" "losscheck_$k.out" ; done
"$BIN" gradcheck --jobs "$ROOT/inputs/jobs_grad.txt" --out gradcheck > gradcheck.log 2>&1 || cat gradcheck.log
"$BIN" run --jobs "$ROOT/inputs/jobs_toy.txt" --out toy --concurrent 3 > toy.log 2>&1 || cat toy.log
# ---- 4. the window of record (all Phase-1 jobs, 48 models at a time)
t0=$(date +%s.%N); "$BIN" run --jobs "$ROOT/inputs/jobs_phase1.txt" --out phase1 --concurrent 48 --chunk 5 > phase1.log 2>&1 || tail -30 phase1.log; echo "phase1 wall $(echo "$(date +%s.%N) - $t0" | bc) s" | tee phase1.time
grep -c . phase1/jobs_out.csv
# ---- 5. concurrency scaling + determinism across waves / runs: the 24 seed-1 tune jobs one at a time,
# then all 24 together, then 8 together
"$BIN" run --jobs "$ROOT/inputs/jobs_scaling.txt" --out scale_k1 --concurrent 1 > scale_k1.log 2>&1
"$BIN" run --jobs "$ROOT/inputs/jobs_scaling.txt" --out scale_k24 --concurrent 24 > scale_k24.log 2>&1
"$BIN" run --jobs "$ROOT/inputs/jobs_scaling.txt" --out scale_k8 --concurrent 8 > scale_k8.log 2>&1
nvidia-smi --query-gpu=name,memory.used --format=csv
date; echo PHASE1_DONE
