"""Validation predictions of the linear pretune at each arm's frozen penalty, as per-fold parquet files in the trees'
layout (date, pred_adj, true_adj, B, plain, rv_actual), and the panel rows for part1_learning.py (read-only).

pred_adj = y + resid (fold.npz: resid = Xv theta - yv, the spec grid's file and every grid extension's, the penalty =
frozen_penalties.json's); y, B and the dates from the run's design cache (y.npy, b_oos.npy, dates_oos.csv), RV of the
bar from data/core_stats.parquet (sumret2), as the trees' driver.

usage: python linear_panel.py <core_stats.parquet> <out dir> <panel csv> W=<run dir> [...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CORE, OUT, PANEL = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
RUNS = {int(a.split("=", 1)[0]): Path(a.split("=", 1)[1]) for a in sys.argv[4:]}
OUT.mkdir(parents=True, exist_ok=True)
core = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
rvb = pd.Series(core["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(core["endbartime"])))
rvb = rvb[~rvb.index.duplicated()]
rows, picks = [], []
for W, run in sorted(RUNS.items()):
    fr = json.loads((run / "frozen_penalties.json").read_text())
    Wr = int(fr["info"]["window_rows"])
    assert Wr == 13 * W, (W, Wr)
    for s in ("baseline", "all_features"):
        dz = run / "_work" / f"design_{s}"
        y = np.load(dz / "y.npy")
        b = np.load(dz / "b_oos.npy")
        d = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(dz / "dates_oos.csv")["date"]))
        for m in ("ridge", "lasso", "enet"):
            base = run / "_work" / "pretune" / s / m
            zs = [np.load(base / "fold.npz")] + [np.load(p) for p in sorted(base.glob("ext_*/fold.npz"))]
            alpha = float(fr[m][s])
            hit = [(z, j) for z in zs for j, g in enumerate(z["grid"]) if np.isclose(float(g), alpha, rtol=1e-12)]
            assert len(hit) == 1, (W, s, m, alpha, [z["grid"] for z in zs])
            z, j = hit[0]
            val = z["val"]  # (folds, 2) global rows
            folds = fr["info"].get("first_fold", 0) + np.arange(len(val))
            for k, f in enumerate(folds):
                v0, v1 = int(val[k][0]), int(val[k][1])
                yy = y[v0:v1]
                pred = yy + z["resid"][k, j]
                idx = np.arange(v0, v1) - Wr
                dates, B = d[idx], b[idx]
                p = OUT / f"lin_w{W}_f{f}_{m}_{s}.parquet"
                pd.DataFrame(dict(row=np.arange(v0, v1), date=dates, pred_adj=pred, true_adj=yy, B=B, plain=pred**2 * B,
                                  rv_actual=rvb.reindex(dates).to_numpy(float))).to_parquet(p, index=False)
                rows.append(dict(family="linear", est=m, set=s, W=W, fold=int(f), path=str(p)))
            picks.append(dict(W=W, model=m, set=s, alpha=alpha, folds=",".join(map(str, folds))))
pd.DataFrame(rows).to_csv(PANEL, index=False)
print(pd.DataFrame(picks).pivot_table(index=["model", "set"], columns="W", values="alpha", aggfunc="first"))
