#!/bin/bash
# 2026-10-06, user request: "on top of no exog features and all_features, add a version of every model
# that only uses the moment bucket + the VIX exog var by itself" -- the bucket moments_vix
# (src.data.loading.SUBGROUPS["moments_vix"]: the 7 ES return moments + vix, beside the HAR + calendar
# backbone) for the two LSTMs, each in the configuration of its canonical stored run, 16:00 bar:
#   per-bar   specs/causal_tune_lstm.py, WINDOW_MASK=1, REFIT_EVERY=1 (the canonical yhat_lstm_* =
#             results/linear_subsection_lstm_mask, harxhar-mask, cluster/slurm/lstm_mask_pack.sbatch)
#   intraday  specs/causal_tune_lstm_intraday.py (results/linear_subsection_lstm_intraday, harxhar-lstmi,
#             two stages: tuning points, then refit chunks from the tuning cache)
# Deployment root /scratch1/jc_905/harxhar-momvix-lstm (cluster/slurm/ship_momvix_lstm_carc.sh): the
# stored runs' code and data copied from their roots, plus the bucket.  ONE CPU class, the stored runs'
# own: every task --constraint=$CLASS (epyc-7513; the intraday campaign's mixed-class first attempt was
# refused by its tuning-cache fingerprint, Intel vs AMD inputs differ in the last bits).
# Modes:
#   start   (1) the reproduction gates: a few stored refits of existing buckets re-run here, same spec,
#           same class (cluster/lstm[i]_momvix_tasks_gate.txt), compared bit for bit with the stored
#           chunks by cluster/slurm/momvix_lstm_check.sbatch (perbar / intraday), plus the identity
#           gates of the new bucket (per-bar: gate_lstm identity; intraday: gate set A);
#           (2) the canaries of the new bucket: per-bar chunk 0 run twice at once (REPEAT_CHECK,
#           byte-identical tables), intraday tuning point R = 0 (STAGE=tune);
#           (3) held afterok on (1) + (2) with --kill-on-invalid-dep=yes (a failed gate cancels them,
#           nothing untested ever starts): the per-bar fleet (chunks 1..5; the canary's chunk carries
#           DONE and is skipped) + its merge (afterany), and the intraday stage 1 (tuning points
#           250..1250; R = 0 skipped).
#   stage2  after stage 1: plan the refit chunks from the measured fit times (the campaign's planner,
#           cluster/lstmi_momvix_make_stage2.py), submit them (REQUIRE_CACHE=1) + the merge (afterany).
# REFUSES a second submission of a mode while its stamp exists (RESUBMIT=1 to resubmit; DONE skipped).
#   cd /scratch1/jc_905/harxhar-momvix-lstm && bash cluster/slurm/submit_momvix_lstm.sh start|stage2
set -euo pipefail
cd "$(dirname "$0")/../.."
[ "$(basename "$PWD")" = harxhar-momvix-lstm ] || { echo "run in /scratch1/jc_905/harxhar-momvix-lstm, not $PWD"; exit 1; }
mkdir -p logs results
MODE="${1:-}"
CLASS=${CLASS:-epyc-7513}
L=results/linear_subsection_lstm_mask_momvix
LI=results/linear_subsection_lstm_intraday_momvix
GP=results/momvix_gates/perbar
GI=results/momvix_gates/intraday
PB_TIME=${PB_TIME:-3:00:00}      # stored masked daily chunks: 1 tuning + 250 daily refits (x2 rules), minutes on 20 CPUs
LI_TIME=${LI_TIME:-1:30:00}      # the intraday campaign's TUNE_TIME
STAMP=logs/submitted_momvix_lstm_$MODE.txt
for f in specs/causal_tune_lstm.py specs/causal_tune_lstm_jobs.py specs/causal_tune_lstm_intraday.py \
         specs/causal_tune_lstm_intraday_jobs.py specs/causal_tune_linear.py src/data/loading.py \
         experiments/momvix_lstm_gate_compare.py experiments/gate_lstm.py experiments/gate_lstm_intraday.py \
         experiments/reduce_lstm_chunks.py experiments/reduce_lstm_intraday_chunks.py \
         cluster/slurm/lstm_mask_pack.sbatch cluster/slurm/lstm_intraday_pack.sbatch cluster/slurm/lstm_intraday_merge.sbatch \
         cluster/slurm/momvix_lstm_check.sbatch cluster/slurm/momvix_lstm_merge.sbatch \
         cluster/lstm_intraday_make_stage2.py cluster/lstmi_momvix_make_stage2.py \
         cluster/lstm_momvix_tasks_gate.txt cluster/lstm_momvix_tasks_canary.txt cluster/lstm_momvix_tasks_fleet.txt \
         cluster/lstmi_momvix_tasks_gate.txt cluster/lstmi_momvix_tasks_canary.txt cluster/lstmi_momvix_tasks_tune.txt \
         results/linear_subsection/arms_hoffman2/moments_vix/ridge/tw2000/results_bar1600.csv; do
  [ -f "$f" ] || { echo "$f missing: run cluster/slurm/ship_momvix_lstm_carc.sh locally first"; exit 1; }
done
if [ -f "$STAMP" ]; then
  if [ "${RESUBMIT:-0}" = "1" ]; then mv "$STAMP" "$STAMP.$(date +%s)"; else
    echo "already submitted ($(cat "$STAMP")); RESUBMIT=1 to resubmit (DONE lines are skipped)"; exit 1; fi
fi
SB() { sbatch --parsable --account=pollok_1603 --partition=main "$@"; }

case "$MODE" in
  start)
    mkdir -p "$L" "$LI" "$GP" "$GI"
    rm -f "$L/CANARY_OK" "$LI/CANARY_OK" "$GP/GATE_OK" "$GI/GATE_OK" "$LI/GATES_A_OK"
    # (1) reproduction gates: stored refits re-run on the class, then compared
    GPJ=$(SB -J mvl_gate_pb --constraint="$CLASS" --array=1-"$(grep -c . cluster/lstm_momvix_tasks_gate.txt)" \
          --cpus-per-task=20 --mem=32G --time=1:00:00 \
          --export=ALL,TASKFILE=cluster/lstm_momvix_tasks_gate.txt,RESULTS_ROOT=$GP,LSTM_RE=1 cluster/slurm/lstm_mask_pack.sbatch)
    PCJ=$(SB -J mvl_check_pb --constraint="$CLASS" --dependency=afterany:"$GPJ" --cpus-per-task=2 --mem=24G \
          --export=ALL,MODE=perbar cluster/slurm/momvix_lstm_check.sbatch)
    GIJ=$(SB -J mvl_gate_i --constraint="$CLASS" --array=1-"$(grep -c . cluster/lstmi_momvix_tasks_gate.txt)" \
          --cpus-per-task=20 --mem=40G --time="$LI_TIME" \
          --export=ALL,TASKFILE=cluster/lstmi_momvix_tasks_gate.txt,RESULTS_ROOT=$GI,REQUIRE_CACHE=0 cluster/slurm/lstm_intraday_pack.sbatch)
    ICJ=$(SB -J mvl_check_i --constraint="$CLASS" --dependency=afterany:"$GIJ" --cpus-per-task=2 --mem=16G \
          --export=ALL,MODE=intraday cluster/slurm/momvix_lstm_check.sbatch)
    GAJ=$(SB -J mvl_gatesA_i --constraint="$CLASS" --cpus-per-task=20 --mem=48G \
          --export=ALL,MODE=intraday_A cluster/slurm/momvix_lstm_check.sbatch)
    # (2) canaries of the new bucket
    CPJ=$(SB -J mvl_canary_pb --constraint="$CLASS" --cpus-per-task=40 --mem=64G --time="$PB_TIME" \
          --export=ALL,TASKFILE=cluster/lstm_momvix_tasks_canary.txt,RESULTS_ROOT=$L,LSTM_RE=1,WRITE_FLAG=$L/CANARY_OK,REPEAT_CHECK=1 \
          cluster/slurm/lstm_mask_pack.sbatch)
    CIJ=$(SB -J mvl_canary_i --constraint="$CLASS" --cpus-per-task=20 --mem=40G --time="$LI_TIME" \
          --export=ALL,TASKFILE=cluster/lstmi_momvix_tasks_canary.txt,RESULTS_ROOT=$LI,WRITE_FLAG=$LI/CANARY_OK \
          cluster/slurm/lstm_intraday_pack.sbatch)
    # (3) held afterok on the gates and the canary; Slurm cancels them if one fails
    FPJ=$(SB -J mvl_fleet_pb --constraint="$CLASS" --array=1-"$(grep -c . cluster/lstm_momvix_tasks_fleet.txt)" \
          --dependency=afterok:"$PCJ":"$CPJ" --kill-on-invalid-dep=yes --cpus-per-task=20 --mem=32G --time="$PB_TIME" \
          --export=ALL,TASKFILE=cluster/lstm_momvix_tasks_fleet.txt,RESULTS_ROOT=$L,LSTM_RE=1,CANARY_FLAG=$L/CANARY_OK \
          cluster/slurm/lstm_mask_pack.sbatch)
    MPJ=$(SB -J mvl_merge_pb --dependency=afterany:"$FPJ" --cpus-per-task=1 --mem=16G cluster/slurm/momvix_lstm_merge.sbatch)
    S1J=$(SB -J mvl_tune_i --constraint="$CLASS" --array=1-"$(grep -c . cluster/lstmi_momvix_tasks_tune.txt)" \
          --dependency=afterok:"$ICJ":"$GAJ":"$CIJ" --kill-on-invalid-dep=yes --cpus-per-task=20 --mem=40G --time="$LI_TIME" \
          --export=ALL,TASKFILE=cluster/lstmi_momvix_tasks_tune.txt,RESULTS_ROOT=$LI,REQUIRE_FLAGS="$LI/GATES_A_OK $LI/CANARY_OK $GI/GATE_OK" \
          cluster/slurm/lstm_intraday_pack.sbatch)
    echo "$(date +%F_%T) class=$CLASS gate_pb=$GPJ check_pb=$PCJ gate_i=$GIJ check_i=$ICJ gatesA_i=$GAJ canary_pb=$CPJ canary_i=$CIJ fleet_pb=$FPJ merge_pb=$MPJ stage1_i=$S1J" | tee "$STAMP"
    ;;
  restart_i)
    # the first gate set A (12747230) stopped on its live_feasible-only mask_flatten gate; stage 1
    # (12747235) was cancelled by Slurm.  Gate set A again (momvix_lstm_check.sbatch, now with
    # live_feasible beside moments_vix) and stage 1 held afterok on it; the reproduction gate
    # ($GI/GATE_OK) and the canary ($LI/CANARY_OK, tuning point R = 0) already passed.
    for f in "$LI/CANARY_OK" "$GI/GATE_OK"; do [ -f "$f" ] || { echo "$f missing"; exit 1; }; done
    rm -f "$LI/GATES_A_OK"
    GAJ=$(SB -J mvl_gatesA_i --constraint="$CLASS" --cpus-per-task=20 --mem=48G \
          --export=ALL,MODE=intraday_A cluster/slurm/momvix_lstm_check.sbatch)
    S1J=$(SB -J mvl_tune_i --constraint="$CLASS" --array=1-"$(grep -c . cluster/lstmi_momvix_tasks_tune.txt)" \
          --dependency=afterok:"$GAJ" --kill-on-invalid-dep=yes --cpus-per-task=20 --mem=40G --time="$LI_TIME" \
          --export=ALL,TASKFILE=cluster/lstmi_momvix_tasks_tune.txt,RESULTS_ROOT=$LI,REQUIRE_FLAGS="$LI/GATES_A_OK $LI/CANARY_OK $GI/GATE_OK" \
          cluster/slurm/lstm_intraday_pack.sbatch)
    echo "$(date +%F_%T) class=$CLASS gatesA_i=$GAJ stage1_i=$S1J" | tee "$STAMP"
    ;;
  stage2)
    for f in "$LI/GATES_A_OK" "$LI/CANARY_OK" "$GI/GATE_OK"; do [ -f "$f" ] || { echo "$f missing"; exit 1; }; done
    N_CACHE=$(ls "$LI"/tune_cache/moments_vix/tune_R*.npz 2>/dev/null | wc -l)
    N_TUNE=$(grep -c . cluster/lstmi_momvix_tasks_tune.txt)
    [ "$N_CACHE" -eq "$N_TUNE" ] || { echo "stage 1 incomplete: $N_CACHE of $N_TUNE tuning records"; exit 1; }
    FLEET=cluster/lstmi_momvix_tasks_fleet.txt
    ( module load conda >/dev/null 2>&1 && source activate harxhar && python cluster/lstmi_momvix_make_stage2.py --root "$LI" --nt 20 --out "$FLEET" )
    N=$(grep -c . "$FLEET")
    [ "$N" -le 50 ] || { echo "$N stage-2 chunks: more than one array of 50, split it first"; exit 1; }
    S2J=$(SB -J mvl_refit_i --constraint="$CLASS" --array=1-"$N" --cpus-per-task=20 --mem=32G --time=1:00:00 \
          --export=ALL,TASKFILE=$FLEET,RESULTS_ROOT=$LI,REQUIRE_CACHE=1,REQUIRE_FLAGS="$LI/GATES_A_OK $LI/CANARY_OK $GI/GATE_OK" \
          cluster/slurm/lstm_intraday_pack.sbatch)
    MIJ=$(SB -J mvl_merge_i --dependency=afterany:"$S2J" --cpus-per-task=1 --mem=8G --time=0:30:00 \
          --export=ALL,RESULTS_ROOT=$LI,FLEET=$FLEET cluster/slurm/lstm_intraday_merge.sbatch)
    echo "$(date +%F_%T) class=$CLASS stage2_i=$S2J ($N chunks) merge_i=$MIJ" | tee "$STAMP"
    ;;
  *)
    echo "usage: bash cluster/slurm/submit_momvix_lstm.sh start|restart_i|stage2"; exit 1 ;;
esac
squeue -u "$USER" -o "%.14i %.16j %.4t %.10M %.5C %.9P %R" | grep -E "JOBID|mvl_" | head -20
