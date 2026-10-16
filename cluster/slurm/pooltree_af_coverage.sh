#!/bin/bash
# Pooled forest all_features (2026-10-09): which rows of the 100 bank chunks have a piece npz (CARC), chunk by chunk.
# A chunk is complete when its pieces tile [0, L) exactly (what experiments/pooled_trees_merge.py needs; a merged
# chunk_<kkk>.npz also counts).  Prints complete / incomplete counts, rows missing, and the first gaps of each
# incomplete chunk (up to MAXGAPS, default 3).  Hoffman2 pieces count only after they were relayed to CARC
# (cluster/transfer_pooltrees_h2_to_carc.sh).
#   bash cluster/slurm/pooltree_af_coverage.sh [ARM (default pooltree_rf_all_features)]
cd "$(dirname "$0")/../.."
ARM=${1:-pooltree_rf_all_features}
D=results/pooled_trees_2026-10-07/chunks/$ARM
python3 - "$D" "${MAXGAPS:-3}" <<'EOF'
import re
import sys
from pathlib import Path

d, maxgaps = Path(sys.argv[1]), int(sys.argv[2])
N, W = 300317, 24000
def bounds(k):
    # experiments/run_unification.chunk_bounds: 100 near-equal chunks of [24000, 300317)
    span = N - W
    return W + (k * span) // 100, W + ((k + 1) * span) // 100
pieces = {}
for f in (d / "pieces").glob("chunk_*.npz"):
    m = re.match(r"chunk_(\d{3})_(\d{4})-(\d{4})\.npz$", f.name)
    if m:
        pieces.setdefault(int(m.group(1)), []).append((int(m.group(2)), int(m.group(3))))
done, missing_rows, lines = 0, 0, []
for k in range(100):
    c0, c1 = bounds(k)
    L = c1 - c0
    if (d / f"chunk_{k:03d}.npz").is_file():
        done += 1
        continue
    cur, gaps = 0, []
    for a, b in sorted(pieces.get(k, [])):
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < L:
        gaps.append((cur, L))
    if not gaps:
        done += 1
        continue
    miss = sum(b - a for a, b in gaps)
    missing_rows += miss
    lines.append(f"  chunk {k:3d}: {miss:4d} rows missing; first gaps {gaps[:maxgaps]}")
print(f"{d.name}: {done} of 100 chunks complete (pieces tile or merged); {missing_rows} rows missing")
print("\n".join(lines))
EOF
