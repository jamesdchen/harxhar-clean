"""Record a wave / sweep task file of the pooled forest in its piece manifest (2026-10-09).

results/pooled_trees_2026-10-07/manifest_rf_<set>.csv lists every piece of the plan (chunk, a, b, rows, assignment,
task_line).  A task file (cluster/pooltree_pieces_rf_<set>_<name>.txt, "rf <set> k:a:b[,k:a:b...]" a line) either takes
plan pieces whole (relabelled to <name>) or cuts a plan piece into sub-pieces that tile it (the piece's row is replaced
by one row per sub-piece).  Anything else (a sub-piece that does not tile its parent, a piece outside the plan) stops
the update with nothing written.  The manifest keeps covering every row of [24000, 300317) exactly once.

Run:  python experiments/pooled_trees_manifest_wave.py --set all_features --name w7
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument(
        "--name",
        required=True,
        action="append",
        help="task file suffix, e.g. w7 -> cluster/pooltree_pieces_rf_<set>_w7.txt; repeat for files that together tile",
    )
    a = ap.parse_args()
    m = ROOT / "results" / "pooled_trees_2026-10-07" / f"manifest_rf_{a.set}.csv"
    rows = list(csv.DictReader(open(m, encoding="utf-8")))
    line_of: dict[tuple[int, int, int], int] = {}
    name_of: dict[tuple[int, int, int], str] = {}
    for name in a.name:
        tf = ROOT / "cluster" / f"pooltree_pieces_rf_{a.set}_{name}.txt"
        for i, ln in enumerate(tf.read_text(encoding="utf-8").splitlines()):
            for p in ln.split()[2].split(","):
                k, x, y = map(int, p.split(":"))
                line_of[(k, x, y)] = i + 1
                name_of[(k, x, y)] = name
    tf_names = "+".join(a.name)
    by_chunk = defaultdict(list)
    for key in line_of:
        by_chunk[key[0]].append(key)
    out, used = [], set()
    for r in rows:
        k, x, y = int(r["chunk"]), int(r["a"]), int(r["b"])
        inside = sorted(p for p in by_chunk.get(k, []) if x <= p[1] and p[2] <= y)
        if not inside:
            out.append(r)
            continue
        edges = [x] + [p[2] for p in inside]
        if [p[1] for p in inside] != edges[:-1] or edges[-1] != y:
            sys.exit(
                f"{tf_names}: pieces {inside} do not tile manifest piece {k}:{x}:{y}; nothing written"
            )
        for p in inside:
            out.append(
                {
                    "chunk": k,
                    "a": p[1],
                    "b": p[2],
                    "rows": p[2] - p[1],
                    "assignment": name_of[p],
                    "task_line": line_of[p],
                }
            )
            used.add(p)
    if used != set(line_of):
        sys.exit(
            f"{tf_names}: {len(set(line_of) - used)} pieces lie across manifest pieces; nothing written"
        )
    assert sum(int(r["rows"]) for r in out) == 300317 - 24000
    with open(m, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["chunk", "a", "b", "rows", "assignment", "task_line"],
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(out)
    print(
        f"{m.name}: {len(used)} pieces of {tf_names} recorded; {dict(Counter(r['assignment'] for r in out))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
