"""Canary (2026-10-22): two refit lines a set through the trainer before the walk, to check the plumbing (the trainer's
window mask has the ridge init's column count, the forecast rows are [origin, next)).  Not a selection input: a fixed
configuration of the grid (L 20, H 64, lr 0.003, wd_enc 3, wd_lin 0.3, lr_lin 1e-5), 3 epochs, seed 20261012, at the
walk's first and last origins.  Writes inputs/canary_walk.txt; run on CARC from /scratch1/jc_905/pm13.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pooled_mingru_rerun_13 as M  # noqa: E402

plan = json.loads(Path("inputs/plan.json").read_text())
o_all, n_all = plan["walk"]["origins"], plan["walk"]["next"]
lines = []
for s in M.SETS:
    for i in (0, len(o_all) - 1):
        o, nx = o_all[i], n_all[i]
        lines.append(
            M.kv(
                id=f"pcanary_{s}_r{o}_s20261012",
                data=f"data/pm13_{s}",
                mode="refit",
                w0=o - M.W,
                w1=o,
                L=20,
                H=64,
                lr="0.003",
                seed=20261012,
                epochs=3,
                fc_n=nx - o,
                wd=0,
                **M.COMMON,
                wd_enc="3",
                wd_lin="0.3",
                lr_lin="1e-05",
                lin_init=f"data/pm13_lin_init/{s}/r{o}_L20_wl0.3.f32",
                save_params=1,
            )
        )
Path("inputs/canary_walk.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
