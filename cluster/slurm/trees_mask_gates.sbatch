#!/bin/bash
# I9: the gates of the masked 16:00 re-run (WINDOW_MASK=1), on the cluster the fleets run on,
# before any canary.  GATESET (exported at submit time) picks one of three suites, submitted as
# three concurrent jobs by cluster/slurm/submit_trees_mask.sh gates:
#   cadence  experiments/trees_cadence_gates.py UNEDITED with HPC_KW_WINDOW_MASK=1 in the
#            environment (every spec run it launches inherits it): g0 design, g1 importance
#            cadence, g2 / g2b chunks t1 / t10, g3 tuned TUNE_IDENTITY=1 = untuned, g4a process
#            count, g4b pool size -- all with the mask on.  OK flag: $OUT/cadence/GATES_OK
#   lstm     experiments/gate_lstm.py UNEDITED with WINDOW_MASK=1 in the environment (its
#            subprocess runs drop HPC_KW_* but keep the plain name, which the spec also reads):
#            determinism, pool, chunk.  OK flag: $OUT/lstm/GATES_OK (written here when it passes)
#   h        experiments/trees_mask_gates.py: h1 WINDOW_MASK=0 = the pre-edit spec (gates_ref/),
#            h2 full-rank design: mask 1 = mask 0, h3 the kept sets = window_keep of each window,
#            importance / TreeSHAP 0 on dropped columns, h4 chunks with the mask (kept arrays,
#            tuned over two tuning periods).  OK flag: $OUT/h/GATES_OK
# Exported: GATESET, OUT (default results/linear_subsection_trees_mask/gates), PROCS, POOL.
#SBATCH --job-name=tm_gates
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:$PWD/pylib_trees"
OUT="${OUT:-results/linear_subsection_trees_mask/gates}"
PROCS="${PROCS:-18}"
POOL="${POOL:-4}"
echo "gates $GATESET on $(hostname), ${SLURM_CPUS_PER_TASK:-1} cpus  $(date)"
case "$GATESET" in
  cadence)
    D="$OUT/cadence"; mkdir -p "$D"; rm -f "$D/GATES_OK"
    HPC_KW_WINDOW_MASK=1 python -u experiments/trees_cadence_gates.py --procs "$PROCS" --pool "$POOL" --out "$D" \
      > "$D/gates.log" 2>&1 || echo "GATES FAILED: $(tail -3 "$D/gates.log")"
    tail -25 "$D/gates.log"
    ;;
  lstm)
    D="$OUT/lstm"; mkdir -p "$D"; rm -f "$D/GATES_OK"
    if WINDOW_MASK=1 python -u experiments/gate_lstm.py --gates determinism,pool,chunk --pool "$POOL" \
         --scratch "$D/scratch" --out "$D" > "$D/gates.log" 2>&1; then
      python - "$D/gate_lstm.csv" <<'EOF' && date +%F_%T > "$D/GATES_OK"
import sys
import pandas as pd
t = pd.read_csv(sys.argv[1])
print(t[["gate", "ok"]].to_string(index=False))
sys.exit(0 if len(t) == 3 and t["ok"].all() else 1)
EOF
    else
      echo "GATES FAILED: $(tail -3 "$D/gates.log")"
    fi
    tail -25 "$D/gates.log"
    ;;
  h)
    D="$OUT/h"; mkdir -p "$D"; rm -f "$D/GATES_OK"
    python -u experiments/trees_mask_gates.py --procs "$PROCS" --pool "$POOL" --ref-root gates_ref --out "$D" \
      > "$D/gates.log" 2>&1 || echo "GATES FAILED: $(tail -3 "$D/gates.log")"
    tail -40 "$D/gates.log"
    ;;
  *)
    echo "GATESET must be cadence, lstm or h, got '$GATESET'"; exit 1 ;;
esac
touch "$OUT/FINISHED_$GATESET"
