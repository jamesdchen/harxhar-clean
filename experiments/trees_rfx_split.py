"""Split a fixed-settings forest task file between CARC (packs, cluster/slurm/trees_rfdev_pack.sbatch) and Hoffman2 (one
SGE task per chunk line, cluster/submit_trees_rfx_h2.sh) (2026-10-07).

Pack 0 (the CARC canary: chunk c0 of every arm, re-run alone) always stays on CARC.  Hoffman2 gets the LAST
ceil(share x fleet packs) packs (a contiguous range, so the CARC array is one range) -- whole packs, so a chunk is computed on
exactly one cluster -- written as its own line file (the same 10-field lines); --buckets restricts Hoffman2 to some input
sets (e.g. baseline,lean_vol_vix when its AVX pin cannot be relied on: those were byte-identical even on AVX-512).

Run:  python experiments/trees_rfx_split.py --tasks cluster/trees_rf400_tasks_untuned.txt --share 0.35
          --out cluster/trees_rf400_fx_h2lines.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument(
        "--share",
        type=float,
        required=True,
        help="share of the fleet packs (1..K) Hoffman2 takes",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--buckets",
        default="",
        help="comma list: Hoffman2 takes only these input sets' lines of its packs",
    )
    a = ap.parse_args()
    lines = [
        ln
        for ln in (ROOT / a.tasks).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    packs = sorted({int(ln.split()[0]) for ln in lines})
    fleet = [p for p in packs if p != 0]
    k = math.ceil(a.share * len(fleet)) if a.share > 0 else 0
    h2_packs = set(fleet[len(fleet) - k :]) if k else set()
    keep = set(a.buckets.split(",")) if a.buckets else None
    h2 = [
        ln
        for ln in lines
        if int(ln.split()[0]) in h2_packs and (keep is None or ln.split()[2] in keep)
    ]
    (ROOT / a.out).write_text(
        "\n".join(h2) + ("\n" if h2 else ""), encoding="utf-8", newline="\n"
    )
    carc = [p for p in fleet if p not in h2_packs]
    left = [ln for ln in lines if int(ln.split()[0]) in h2_packs and ln not in h2]
    print(
        f"Hoffman2: {len(h2)} chunk lines of packs {min(h2_packs) if h2_packs else '-'}-{max(h2_packs) if h2_packs else '-'} -> {a.out}"
    )
    print(
        f"CARC: pack 0 (canary) + packs {carc[0] if carc else '-'}-{carc[-1] if carc else '-'} ({len(carc)} packs)"
    )
    if left:
        print(
            f"WARNING: {len(left)} lines of the Hoffman2 packs are outside --buckets and need a CARC run of those packs too"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
