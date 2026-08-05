"""Keep digging for a DIRECTIONAL edge: do the CALENDAR-TIMED dealer flows (charm/vanna) predict the
close-window ES direction? -- the one options-flow signal that is directional AND prop-legal.

Reactive gamma hedging is contemporaneous with the move (HFT latency, and gex_bar_test showed the bar
autocorrelation doesn't flip with the regime). But CHARM (dDelta/dTime) and VANNA (dDelta/dVol) flows are
driven by TIME DECAY and IV changes, not by needing to see a tick first: as expiry nears (into the close,
into OPEX) dealer deltas drift in a PREDICTABLE direction, forcing dated ES buying/selling ("vanna rally",
EOD MOC drift). That is directional, linear (ES), intraday, flat-by-close => Topstep-legal. We have
net_charm/net_vanna in gex_daily but never tested them directionally. Do it, through the gate.

Design (daily, causal): features = the lagged (prior-EOD) standing charm/vanna/flip, known at today's open;
target = the CLOSE-WINDOW return (session bars with hour==15, the final hour where charm flow concentrates);
base = the day's move UP TO the close window (hour<15 session return) so we test charm's INCREMENTAL
directional content over intraday momentum. Conditioning: OPEX proximity (charm/vanna peak into monthly
expiry) and the day's VIX change (vanna keys off IV moves). Gate = feature_cv (fwd MSE) + net-of-1-tick
sign P&L. Writes results/gex_flow_dir/summary.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np
import pandas as pd

from flow_dir_pilot import oos_readout, score_dir
from gex_regime_test import lag_regime
from src.evaluation.feature_cv import purged_walk_forward, significance_gate
from vrp_screen_worker import build


def is_opex(days: pd.DatetimeIndex) -> np.ndarray:
    """1.0 on the monthly OPEX week (the trading week containing the 3rd Friday) -- charm/vanna flows peak
    into standard expiry. Approximated as day-of-month in [15,21] & weekday (Mon-Fri)."""
    dom = days.day.to_numpy()
    wd = days.weekday.to_numpy()
    return ((dom >= 15) & (dom <= 21) & (wd < 5)).astype(float)


def main() -> None:
    os.makedirs("results/gex_flow_dir", exist_ok=True)
    df, *_ = build()
    t = pd.to_datetime(df["t"])
    r = df["sumret"].to_numpy(float)
    fab = df["sumret_ind"].to_numpy(float) > 0
    hour = t.dt.hour.to_numpy()
    day = t.dt.normalize()
    sess = (hour >= 9) & (hour <= 15)

    d = pd.DataFrame(
        {
            "day": day.to_numpy(),
            "hour": hour,
            "r": r,
            "vix": df["vd_vix"].to_numpy(float),
        }
    )
    d = d[sess & ~fab]
    preclose = (
        d[d["hour"] < 15].groupby("day")["r"].sum()
    )  # move up to the close window (known @ ~15:00)
    closewin = (
        d[d["hour"] == 15].groupby("day")["r"].sum()
    )  # the final-hour return = the target
    vix_eod = d.groupby("day")["vix"].last()
    p = (
        pd.DataFrame({"preclose": preclose, "closewin": closewin, "vix": vix_eod})
        .dropna()
        .sort_index()
    )
    p["dvix"] = np.log(
        p["vix"]
    ).diff()  # today's VIX change (known by ~15:00; vanna flow keys off it)

    reg = lag_regime(
        pd.read_parquet("results/gex/gex_daily.parquet"), pd.DatetimeIndex(p.index)
    )
    for c in ("z_charm", "z_vanna", "z_flip", "z_gex"):
        p[c] = reg[c].to_numpy()
    p["opex"] = is_opex(pd.DatetimeIndex(p.index))

    p = p.replace([np.inf, -np.inf], np.nan).dropna()
    y = p["closewin"].to_numpy(float)
    base = p[["preclose"]].to_numpy(
        float
    )  # intraday momentum into the close = the control
    folds = purged_walk_forward(len(p), n_folds=5, embargo=0.01)

    arms = {
        "charm": p[["z_charm"]].to_numpy(float),
        "charm_x_opex": np.column_stack([p["z_charm"], p["z_charm"] * p["opex"]]),
        "vanna": p[["z_vanna"]].to_numpy(float),
        "vanna_x_dvix": np.column_stack([p["z_vanna"], p["z_vanna"] * p["dvix"]]),
        "flip_gex": p[["z_flip", "z_gex"]].to_numpy(float),
        "all_flow": p[["z_charm", "z_vanna", "z_flip"]].to_numpy(float),
    }
    res: dict = {"n_days": int(len(p)), "target": "close_hour15_return", "arms": {}}
    res["preclose_base"] = oos_readout(base, None, y, folds)
    for name, cand in arms.items():
        sc = score_dir(base, cand, y, folds)
        passed, comp = significance_gate(sc, k=2.0)
        rd = oos_readout(base, cand, y, folds)
        res["arms"][name] = {"gate_pass": bool(passed), **comp, "readout": rd}
        print(
            f"[{name}] GATE={'PASS' if passed else 'fail'} repl={comp['repl_frac']:.2f} "
            f"beats_plac={comp['beats_placebo']} | corr={rd['oos_corr']:+.4f} "
            f"cov_t={rd['cov_t_hac5']:+.2f} net={rd['net_sign_mean_bp']:+.3f}bp",
            flush=True,
        )
    b = res["preclose_base"]
    print(
        f"[base preclose] corr={b['oos_corr']:+.4f} cov_t={b['cov_t_hac5']:+.2f} n={b['n']}",
        flush=True,
    )
    with open("results/gex_flow_dir/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
