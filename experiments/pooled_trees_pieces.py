"""Piece manifest of the pooled tree walks (2026-10-07): the bank's 100 chunks of an arm cut into row pieces, assigned
to task files (CARC or Hoffman2 arrays) and grouped into task lines.

At refit-every-bar each row is fitted on its own trailing 24,000 bars, so any cut of a chunk into row pieces gives the
unsplit chunk once concatenated (experiments/pooled_trees_merge.py; tested byte for byte on XGBoost baseline chunk 0).
Every chunk belongs to exactly one assignment NAME:CHUNKS:ROWS:PER[:desc] -- chunk k (length L_k = 2,763 or 2,764) is cut
into P_k = ceil(L_k / ROWS) pieces of equal size (edges round(i L_k / P_k)), PER pieces a task line, lines from the last
chunk down with ":desc".  NAME starts with "carc" or "h2" (the cluster).  Task lines, "<est> <set> k:a:b[,k:a:b...]":

  cluster/pooltree_pieces_<est>_<set>_<NAME>.txt              one array task a line
  results/pooled_trees_2026-10-07/manifest_<est>_<set>.csv    one row per piece: chunk, a, b, rows, assignment, task line

A submitted task file is read by its tasks at run time: only the files named in --write are (re)written; every other
assignment's file must already equal this plan byte for byte (else nothing is written).  Unstarted pieces move between
clusters through <results root>/revoked_pieces.txt (see cluster/slurm/pooltree_piece.sbatch, cluster/pooltree_h2_task.sh).

Run:  python experiments/pooled_trees_pieces.py --est rf --set baseline --assign carc:0-49:150:30
          --assign carc2:50-89:150:30 --assign h2:90-99:100:1:desc --write carc2,h2
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from pooled_trees_merge import N_ROWS_PANEL  # noqa: E402
from pooled_trees_walk import N_CHUNKS, chunk_bounds  # noqa: E402


def chunk_set(spec: str) -> list[int]:
    out: set[int] = set()
    for part in filter(None, spec.split("+")):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument(
        "--assign",
        action="append",
        required=True,
        help="NAME:CHUNKS:ROWS:PER[:desc], CHUNKS like 0-49+60-69",
    )
    ap.add_argument(
        "--write",
        default="",
        help="comma list of assignment names whose task files are (re)written",
    )
    a = ap.parse_args()
    owner: dict[int, str] = {}
    plans = []
    for spec in a.assign:
        f = spec.split(":")
        name, chunks, rows, per, desc = (
            f[0],
            chunk_set(f[1]),
            int(f[2]),
            int(f[3]),
            len(f) > 4 and f[4] == "desc",
        )
        if not name.startswith(("carc", "h2")):
            sys.exit(f"assignment {name}: the name must start with carc or h2")
        for k in chunks:
            if k in owner:
                sys.exit(f"chunk {k} is in both {owner[k]} and {name}")
            owner[k] = name
        pieces = []
        for k in sorted(chunks, reverse=desc):
            c0, c1 = chunk_bounds(k, N_ROWS_PANEL)
            L = c1 - c0
            P = math.ceil(L / rows)
            edges = [round(i * L / P) for i in range(P + 1)]
            pieces += [(k, edges[i], edges[i + 1]) for i in range(P)]
        plans.append((name, rows, per, pieces))
    missing = sorted(set(range(N_CHUNKS)) - set(owner))
    if missing:
        sys.exit(f"chunks without an assignment: {missing}")
    write = set(filter(None, a.write.split(",")))
    bodies, rows_out = {}, []
    for name, rows, per, pieces in plans:
        lines = []
        for i in range(0, len(pieces), per):
            grp = pieces[i : i + per]
            lines.append(
                f"{a.est} {a.set} " + ",".join(f"{k}:{x}:{y}" for k, x, y in grp)
            )
            rows_out += [
                {
                    "chunk": k,
                    "a": x,
                    "b": y,
                    "rows": y - x,
                    "assignment": name,
                    "task_line": len(lines),
                }
                for k, x, y in grp
            ]
        f = ROOT / "cluster" / f"pooltree_pieces_{a.est}_{a.set}_{name}.txt"
        body = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        if name not in write and (not f.is_file() or f.read_bytes() != body):
            sys.exit(
                f"{f.name}: not in --write and {'missing' if not f.is_file() else 'differs from this plan'}; nothing written"
            )
        bodies[f] = body
        n_rows = sum(y - x for _, x, y in pieces)
        print(
            f"{f.name}: {len(lines)} tasks, {len(pieces)} pieces of ~{rows} rows, {n_rows} rows"
            + ("" if name in write else " (unchanged)")
        )
    for f, body in bodies.items():
        if (
            f.name.removeprefix(f"pooltree_pieces_{a.est}_{a.set}_").removesuffix(
                ".txt"
            )
            in write
        ):
            f.write_bytes(body)
    m = ROOT / "results" / "pooled_trees_2026-10-07" / f"manifest_{a.est}_{a.set}.csv"
    m.parent.mkdir(parents=True, exist_ok=True)
    with open(m, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["chunk", "a", "b", "rows", "assignment", "task_line"],
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(sorted(rows_out, key=lambda r: (r["chunk"], r["a"])))
    print(
        f"{m.relative_to(ROOT)}: {len(rows_out)} pieces, {sum(r['rows'] for r in rows_out)} rows"
    )


if __name__ == "__main__":
    main()
