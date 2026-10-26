"""Task files of the linear walks of results/fomcfix_2026-10-26/RULE.txt (cells A-D, 7(a) neighbours), written once.

Lines for code/walk_task.py (code/lin_stage.sh walk1).  The final run's chunking: full-set ridge / lasso / elastic net
in 2 / 4 / 4 chunks, the HAR benchmark set and the OLS in 1.  Cell B walks the full set only (its HAR benchmark design is
cell A's, gate G3).  The 7(a) neighbours: one directory each, the w3000 frozen_penalties.json with the full-set penalty
of one class replaced (an alpha off the spec's grid recorded in info.off_spec_grid, so the walk gets --grid-alphas).

usage: python lin_tasks.py <code tree> <runs dir on CARC> <out dir for the task files (relative to the code tree)>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE, RUNS, OUTREL = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
OUT = CODE / OUTREL
FLAGS = ("FOMC_SCHEDULE_2020_FIX", "FOMC_PROXIMITY_NO_WINSOR", "DROP_HOLIDAY_SESSIONS")
CELLS = {"A": (0, 0, 0), "B": (1, 1, 0), "C": (0, 0, 1), "D": (1, 1, 1), "P1": (1, 0, 0)}
CHUNKS = [
    ("all_features", "ridge", 2),
    ("all_features", "lasso", 4),
    ("all_features", "enet", 4),
    ("baseline", "ridge", 1),
    ("baseline", "lasso", 1),
    ("baseline", "enet", 1),
    ("baseline", "ols", 1),
]
SPEC_GRID = {"ridge": [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3], "lasso": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}
SPEC_GRID["enet"] = SPEC_GRID["lasso"]
NEIGHBOURS = {"ridge": (100.0, 10000.0), "lasso": (1e-5, 1e-3), "enet": (1e-4, 1e-2)}


def cell(c: str) -> str:
    return ",".join(f"{f}={v}" for f, v in zip(FLAGS, CELLS[c]))


OUT.mkdir(parents=True, exist_ok=True)
for c in ("A", "B", "C", "D"):
    lines = []
    for s, m, k in CHUNKS:
        if c == "B" and s == "baseline":
            continue
        for j in range(k):
            lines.append(f"out={RUNS}/{c} cell={cell(c)} only={s}/{m} chunks={k} job={j}")
    (OUT / f"tasks_{c}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(c, len(lines), cell(c))
fr0 = json.loads(
    (CODE / "results" / "window_sweep_2014start_2026-10-22" / "w3000" / "frozen_penalties.json").read_text()
)
lines = []
for m, alphas in NEIGHBOURS.items():
    k = dict((x[1], x[2]) for x in CHUNKS if x[0] == "all_features")[m]
    for al in alphas:
        name = f"{m}_{al:g}"
        fr = json.loads(json.dumps(fr0))
        fr[m]["all_features"] = al
        if al not in SPEC_GRID[m]:
            fr["info"]["off_spec_grid"].setdefault(m, {})["all_features"] = al
        fr["info"]["fomcfix_neighbour"] = dict(
            model=m, frozen=fr0[m]["all_features"], neighbour=al, rule="results/fomcfix_2026-10-26/RULE.txt 7(a)"
        )
        d = CODE / "results" / "fomcfix_2026-10-26" / "neighbours" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "frozen_penalties.json").write_text(json.dumps(fr, indent=1), encoding="utf-8", newline="\n")
        for j in range(k):
            lines.append(f"out={RUNS}/nbr/{name} cell={cell('B')} only=all_features/{m} chunks={k} job={j}")
(OUT / "tasks_nbr.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("nbr", len(lines))
