#!/bin/bash
# Incremental harvest waiter: ONE held ssh connection that blocks (cluster-side, 60s squeue poll) until the
# number of completed results for label-prefix PREFIX exceeds BASE, or the queue fully drains — then dumps
# every result-so-far (incl. the explain coef-trajectory + gate readout) and exits. Agent re-arms with the
# new count. One connection at a time, no polling storm. Usage: wait_next.sh <BASE> <PREFIX>
BASE=${1:-0}
PREFIX=${2:-sat_}
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
nres() {
  grep -hE "resid_regime_${PREFIX}.* qlike=" logs/sig_col_*.out 2>/dev/null | awk '{print $3}' | sort -u | wc -l
}
qempty() { [ -z "$(squeue -u jc_905 -h -o %j 2>/dev/null | grep -E 'moe_dig|sig_col|explain')" ]; }
while true; do
  N=$(nres)
  if [ "$N" -gt "$BASE" ] || qempty; then break; fi
  sleep 60
done
echo "===HARVEST=== prefix=${PREFIX} nres=$(nres) queue_empty=$(qempty && echo yes || echo no) ts=$(date +%Y-%m-%dT%H:%M:%S)"
echo "--- queue ---"
squeue -u jc_905 -o "%.11i %.16j %.9T %.10M" 2>/dev/null | grep -E "JOBID|moe_dig|sig_col|explain" | head -12
echo "--- ${PREFIX} collects (full-OOS + h16-19) ---"
grep -hE "resid_regime_${PREFIX}.*(qlike=|qlike_h16)" logs/sig_col_*.out 2>/dev/null | sort -u
echo "--- reference: prior labels ---"
grep -hE "resid_regime_(moe_d|moe2_).* qlike=" logs/sig_col_*.out 2>/dev/null | sort -u
echo "--- explain (coef trajectory + gate readout) ---"
grep -hE "blocks=|mean=|gate node|w=|hour_rank|hour NOT|GATE_READOUT_DONE|EXPLAIN_ALLDONE" logs/explain_*.out 2>/dev/null | tail -45
echo "--- tracebacks ---"
grep -hE "Traceback|Error" logs/moe_*.out logs/explain_*.out 2>/dev/null | sort -u | head -12
echo "===WAIT_NEXT_DONE==="
