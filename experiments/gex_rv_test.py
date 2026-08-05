"""Fork A: does the dealer-gamma regime add INCREMENTAL realized-variance forecast content over HAR+VIX?

The regime is directionally null (gex_bar_test) but predicts the RANGE at corr -0.30 (lagged) -- a real
VARIANCE signal. Useless at a futures prop (can't trade vol -- the spanning wall), but the variance edge
DOES have a home (first-loss capital on CFE vol products). The honest question is INCREMENTAL: HAR already
captures vol persistence and VIX the implied level, and GEX is correlated with both, so most of the -0.30
is likely already priced. Test GEX on top of HAR+VIX through the gate.

Target = today's log daily realized variance (sum of squared session bar returns). Base = HAR (log-RV lags
1/5/22, causal) + log VIX (prior EOD). Candidate = lagged GEX regime (z_gex, z_flip, |z_gex|). Gate =
feature_cv (purged WF + circular-shift placebo + replication), objective = log-RV MSE. Readout = OOS
incremental R^2 (base vs base+GEX). Writes results/gex_rv/summary.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np
import pandas as pd

from flow_dir_pilot import fit_ridge, predict_ridge, score_dir
from gex_regime_test import lag_regime
from src.evaluation.feature_cv import purged_walk_forward, significance_gate
from vrp_screen_worker import build


def oos_r2(Xb, Xf, y, folds) -> tuple[float, float]:
    """Pooled OOS R^2 for base (Xb) and full (Xf) across test folds."""
    pb, pf, yy = [], [], []
    for tr, te in folds:
        pb.append(predict_ridge(fit_ridge(Xb[tr], y[tr], 10.0), Xb[te]))
        pf.append(predict_ridge(fit_ridge(Xf[tr], y[tr], 10.0), Xf[te]))
        yy.append(y[te])
    pb, pf, yy = np.concatenate(pb), np.concatenate(pf), np.concatenate(yy)
    sst = float(((yy - yy.mean()) ** 2).sum())
    return 1 - float(((yy - pb) ** 2).sum()) / sst, 1 - float(
        ((yy - pf) ** 2).sum()
    ) / sst


def main() -> None:
    os.makedirs("results/gex_rv", exist_ok=True)
    df, *_ = build()
    t = pd.to_datetime(df["t"])
    sess = (t.dt.hour >= 9) & (t.dt.hour <= 15)
    d = pd.DataFrame(
        {
            "day": t.dt.normalize().to_numpy(),
            "r": df["sumret"].to_numpy(float),
            "vix": df["vd_vix"].to_numpy(float),
            "keep": (sess.to_numpy() & (df["sumret_ind"].to_numpy(float) <= 0)),
        }
    )
    d = d[d["keep"]]
    rv = d.groupby("day")["r"].apply(lambda x: float((x**2).sum()))
    vix = d.groupby("day")["vix"].last()
    p = pd.DataFrame({"rv": rv, "vix": vix}).sort_index()
    p = p[p["rv"] > 0]
    lrv = np.log(p["rv"])
    p["y"] = lrv
    p["har1"] = lrv.shift(1)
    p["har5"] = lrv.rolling(5).mean().shift(1)
    p["har22"] = lrv.rolling(22).mean().shift(1)
    p["lvix"] = np.log(p["vix"]).shift(1)

    gex = pd.read_parquet("results/gex/gex_daily.parquet")
    reg = lag_regime(gex, pd.DatetimeIndex(p.index))
    p["z_gex"] = reg["z_gex"].to_numpy()
    p["z_flip"] = reg["z_flip"].to_numpy()
    p["abs_gex"] = np.abs(reg["z_gex"].to_numpy())

    base_cols = ["har1", "har5", "har22", "lvix"]
    cand_cols = ["z_gex", "z_flip", "abs_gex"]
    both = p.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["y"] + base_cols + cand_cols
    )
    y = both["y"].to_numpy(float)
    Xb = both[base_cols].to_numpy(float)
    cand = both[cand_cols].to_numpy(float)
    Xf = np.column_stack([Xb, cand])
    folds = purged_walk_forward(len(both), n_folds=5, embargo=0.01)

    sc = score_dir(Xb, cand, y, folds)
    passed, comp = significance_gate(sc, k=2.0)
    r2b, r2f = oos_r2(Xb, Xf, y, folds)
    res = {
        "n_days": int(len(both)),
        "gate_pass": bool(passed),
        **comp,
        "oos_r2_base_HARVIX": r2b,
        "oos_r2_with_GEX": r2f,
        "delta_r2": r2f - r2b,
    }
    with open("results/gex_rv/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(
        f"[GEX->RV | HAR+VIX] GATE={'PASS' if passed else 'fail'} mean_dMSE={comp['mean']:+.2e} "
        f"ci0={comp['ci_excludes_0']} repl={comp['repl_frac']:.2f} beats_plac={comp['beats_placebo']}",
        flush=True,
    )
    print(
        f"  OOS R2: HAR+VIX={r2b:.4f}  +GEX={r2f:.4f}  delta={r2f - r2b:+.5f}  (n={len(both)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
