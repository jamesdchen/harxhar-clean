#!/bin/bash
# The scheduler's submission verifier answered "no response" on submissions it
# had in fact accepted, so the first lasso re-run left DUPLICATE arrays in the
# queue.  Remove every lf_* job of mine except the ids recorded in
# logs/submitted_lassofix.txt (duplicates would run the same arms concurrently
# into the same directories).
set -euo pipefail
cd "$(dirname "$0")/.."
KEEP=$(grep -o '[0-9]\{6,\}' logs/submitted_lassofix.txt | sort -u)
for J in $(qstat -u "$USER" | awk '$3 ~ /^lf_/ {print $1}' | sort -u); do
  if echo "$KEEP" | grep -qx "$J"; then echo "keep $J"; else echo "remove duplicate $J"; qdel "$J" || true; fi
done
qstat -u "$USER" | tail -8
