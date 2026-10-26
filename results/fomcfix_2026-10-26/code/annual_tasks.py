"""The year walks of results/fomcfix_2026-10-26/RULE.txt 7(b) (annual re-tuning): one task line a (year, set, model).

For each year Y, <root>/y<Y> holds the cell-B design (_work/design_* linked) and the year's frozen_penalties.json (the
final run's pretune + select with --first-day Y-01-01 --first-fold 3).  The year's walk runs from the main walk's
250-solve block boundary at or before the first bar of Y (S0 + 250 floor((r_Y - S0) / 250), S0 = the main walk's start,
the boundary at or before the first 2014-01-02 row) to the first bar of Y + 1 (the end of the data for the last year),
with a W halo; collect_linear.py annual keeps the rows of Y.  A year whose penalty is the frozen one reproduces the main
walk's rows of that year (a chunk at a block boundary with a W halo reproduces the walk).

usage: python annual_tasks.py <root> <cell flags> <years, e.g. 2014-2024> <tasks file>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, CELL, YEARS, TASKS = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
W, TUNE_PER = 39000, 250
y0, y1 = (int(v) for v in YEARS.split("-"))
lines, info = [], {}
for Y in range(y0, y1 + 1):
    root = ROOT / f"y{Y}"
    d = pd.DatetimeIndex(
        pd.read_csv(root / "_work" / "design_baseline" / "dates_oos.csv", parse_dates=["date"])["date"]
    )
    r = {y: W + int(np.flatnonzero(d >= pd.Timestamp(f"{y}-01-01"))[0]) for y in (2014, Y)}
    S0 = (r[2014] - W) // TUNE_PER * TUNE_PER + W
    start = S0 + (r[Y] - S0) // TUNE_PER * TUNE_PER
    nxt = np.flatnonzero(d >= pd.Timestamp(f"{Y + 1}-01-01"))
    end = W + int(nxt[0]) if nxt.size else -1
    fr = json.loads((root / "frozen_penalties.json").read_text())
    info[Y] = dict(
        first_row=r[Y],
        first=str(d[r[Y] - W]),
        start=start,
        end=end,
        S0=S0,
        penalties={m: fr[m] for m in ("ridge", "lasso", "enet")},
    )
    for s in ("baseline", "all_features"):
        for m in ("ridge", "lasso", "enet"):
            lines.append(f"out={root} cell={CELL} only={s}/{m} start={start} end={end} sub=y{Y}")
TASKS.write_text("\n".join(lines) + "\n")
(TASKS.parent / (TASKS.stem + "_info.json")).write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
print(f"{len(lines)} tasks -> {TASKS}")
