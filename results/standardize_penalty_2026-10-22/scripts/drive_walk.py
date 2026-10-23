"""Selected walk chunks through the orchestrator's own job builder (walk_jobs) and pool, with main()'s setup for the
13-bar 2012 w1000 late-start configuration.

usage: python drive_walk.py <out_dir> <jobs> <kkt 0|1|""> <std 0|1|""> <set/model/chunk/of> ... [-- extra runner args]
  e.g. all_features/lasso/0/4 = chunk 0 of the full-set lasso walk split in 4 (the KKT run's split); baseline/ridge/0/1
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

WT = Path(r"C:\Users\james\AppData\Local\Temp\std_wt")
spec = importlib.util.spec_from_file_location(
    "orch", WT / "experiments" / "pooled_linear_rerun_2026_10_19_13.py"
)
O = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(O)

argv = sys.argv[1:]
extra: list[str] = []
if "--" in argv:
    extra = argv[argv.index("--") + 1 :]
    argv = argv[: argv.index("--")]
out, jobs, kkt, std, *want = argv
# main()'s setup for: --window-days 1000 --first-day 2012-01-03 --ex-flag LATE_START_WARMUP=1 --out <out>
O.OUT = Path(out).resolve()
O.WORK = O.OUT / "_work"
O.TW_DAYS = 1000
O.W = O.TW_DAYS * O.PPD
O.PRE_HALO = O.PRE_FOLDS * O.PRE_FOLD_SESS * O.PPD + O.PRE_EMBARGO_SESS * O.PPD + O.W
O.CFG.extend(["--window-days", "1000"])
O.CFG.extend(["--ex-flag", "LATE_START_WARMUP=1"])
O.FIRST_DAY = "2012-01-03"
O.WALK_KKT = kkt
O.STD = std
for s in O.SETS:
    O.check_design(s)
chunks, sel = {}, set()
for w in want:
    s, m, c, k = w.split("/")
    chunks[(s, m)] = int(k)
    sel.add((s, m, int(c)))
js = [j for j in O.walk_jobs(chunks) if j[2] in sel]
assert len(js) == len(sel), (len(js), sel)
for j in js:
    O.say(f"job {j[2]}: {' '.join(j[0] + extra)}")
O.pool([(j[0] + extra, j[1]) for j in js], int(jobs))
O.say("all done")
