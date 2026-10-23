"""Gate prep: the pieces of the printed 2012-start w1000 run (CARC jobs 13262219 / 13265608 / 13265651, runner db59af2)
that one gate re-walks.

usage: python gate_prep.py <printed pieces dir> <gate dir> <gate spec>

Each spec line: "<est> <set> <refit> <first> <last> <i>/<n> <k>" (a line of the printed tasks.txt plus k).
  k = 0  the piece is walked from scratch (every row refitted);
  k > 0  the printed piece minus the rows of its first k and last k trading days goes to <gate dir>/pieces/ (without
         its .json meta), so the walk's resume path (rows already in the piece are skipped) refits exactly those rows.
Either way the finished piece is compared with the printed one bit for bit (gate_cmp.py).  Writes <gate dir>/tasks.txt
(the six fields cluster/slurm/pooltree_rerun13.sbatch reads) and <gate dir>/gate_prep.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

src, gate, spec = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
(gate / "pieces").mkdir(parents=True, exist_ok=True)
rows, lines = [], []
for ln in spec.read_text().splitlines():
    if not ln.strip() or ln.startswith("#"):
        continue
    est, s, refit, first, last, part, ks = ln.split()
    i, n = part.split("/")
    name = f"{est}_{s}_{refit}_{first}_{last}_{i}of{n}"
    dst = gate / "pieces" / f"{name}.parquet"
    assert not dst.exists() and not dst.with_suffix(".json").exists(), f"{dst} exists"
    k = int(ks)
    f = pd.read_parquet(src / f"{name}.parquet")
    day = pd.DatetimeIndex(f["date"]).normalize()
    u = day.unique()
    if k:
        drop = day.isin(u[:k]) | day.isin(u[-k:])
        f[~drop].reset_index(drop=True).to_parquet(dst, index=False)
        n_drop = int(drop.sum())
    else:
        n_drop = len(f)
    rows.append(
        {
            "piece": name,
            "k": k,
            "rows": len(f),
            "days": len(u),
            "refit_rows": n_drop,
            "first": str(f["date"].iloc[0]),
            "last": str(f["date"].iloc[-1]),
        }
    )
    lines.append(f"{est} {s} {refit} {first} {last} {part}")
(gate / "tasks.txt").write_text("\n".join(lines) + "\n")
pd.DataFrame(rows).to_csv(gate / "gate_prep.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
