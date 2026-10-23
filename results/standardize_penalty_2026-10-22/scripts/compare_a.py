"""Check (a): the flag-off rerun against the KKT run (results/..._nofomcrel_kkt) and its base run B.

usage: python compare_a.py <a_off dir> [walk]
  pretune / select: fold.npz arrays of all six arms vs run B's pretune; frozen_penalties.json and pretune_mse.csv bytes.
  walk: every walk table present under <a_off>/_work/walk against the KKT run's same chunk (CSV bytes; results_rth.csv).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

A = Path(sys.argv[1])
KK = Path(r"C:\Users\james\AppData\Local\Temp\kk_wt\results\pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel_kkt")
KK_COMMITTED = Path(r"C:\Users\james\AppData\Local\Temp\std_wt\results\pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel_kkt")
LSB = Path(r"C:\Users\james\AppData\Local\Temp\ls_wt\results\pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel")
rep: dict = {}
if len(sys.argv) < 3:
    for s in ("baseline", "all_features"):
        for m in ("ridge", "lasso", "enet"):
            a = np.load(A / "_work" / "pretune" / s / m / "fold.npz")
            b = np.load(LSB / "_work" / "pretune" / s / m / "fold.npz")
            rep[f"fold.npz {s}/{m} arrays identical to run B"] = sorted(a.files) == sorted(b.files) and all(
                np.array_equal(a[k], b[k]) for k in b.files
            )
    rep["frozen_penalties.json bytes == KKT run's (committed)"] = (
        (A / "frozen_penalties.json").read_bytes() == (KK_COMMITTED / "frozen_penalties.json").read_bytes()
    )
    rep["pretune_mse.csv bytes == run B's"] = (A / "pretune_mse.csv").read_bytes() == (LSB / "pretune_mse.csv").read_bytes()
    fr = json.loads((A / "frozen_penalties.json").read_text())
    rep["penalties"] = {m: fr[m] for m in ("ridge", "lasso", "enet")}
else:
    for p in sorted((A / "_work" / "walk").glob("**/results.csv")):
        rel = p.relative_to(A / "_work" / "walk")
        q = KK / "_work" / "walk" / rel
        rq = p.parents[3] / "results_rth.csv"
        rk = q.parents[3] / "results_rth.csv"
        run = json.loads((p.parents[3] / "run.json").read_text())
        runk = json.loads((q.parents[3] / "run.json").read_text())
        rep[str(rel.parents[2])] = dict(
            rows=run["rows"],
            start=run["start"],
            end=run["end"],
            csv_bytes_identical=p.read_bytes() == q.read_bytes(),
            rth_bytes_identical=rq.read_bytes() == rk.read_bytes(),
            kkt_reseeds=(run.get("kkt_reseeds"), runk.get("kkt_reseeds")),
            standardize_penalty=run.get("standardize_penalty"),
            wall_sec=(round(run["wall_sec"]), round(runk["wall_sec"])),
        )
print(json.dumps(rep, indent=1))
