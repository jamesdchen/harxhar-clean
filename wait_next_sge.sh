#!/bin/bash
# Hoffman2/SGE incremental harvest waiter (qstat-based; the SLURM wait_next.sh won't work here).
# Blocks until #completed reg_* collects exceeds BASE or the queue drains, then dumps results + exits.
# Usage: wait_next_sge.sh <BASE> <PREFIX>
BASE=${1:-0}
PREFIX=${2:-reg_}
cd /u/scratch/j/jamesdc1/harxhar-clean
nres() {
  grep -hE "resid_regime_${PREFIX}.* qlike=" logs/sig_col.o* 2>/dev/null | awk '{print $3}' | sort -u | wc -l
}
qbusy() { qstat -u jamesdc1 2>/dev/null | grep -qE "moe_dig|sig_col"; }
while true; do
  N=$(nres)
  if [ "$N" -gt "$BASE" ] || ! qbusy; then break; fi
  sleep 60
done
echo "===HARVEST=== prefix=${PREFIX} nres=$(nres) qbusy=$(qbusy && echo yes || echo no) ts=$(date +%Y-%m-%dT%H:%M:%S)"
echo "--- qstat ---"
qstat -u jamesdc1 2>/dev/null | grep -E "moe_dig|sig_col" | head -14
echo "--- ${PREFIX} collects (full-OOS + h16-19) ---"
grep -hE "resid_regime_${PREFIX}.*(qlike=|qlike_h16)" logs/sig_col.o* 2>/dev/null | sort -u
echo "--- tracebacks/errors ---"
grep -hE "Traceback|Error|No such file" logs/moe_dig.o* logs/sig_col.o* 2>/dev/null | sort -u | head -12
echo "===WAIT_NEXT_SGE_DONE==="
