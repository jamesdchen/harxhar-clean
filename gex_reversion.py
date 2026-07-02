"""THE pre-registered directional shot: break-conditioned reversion to fair price, gated by gamma.

The user's trade, precisely: consolidation = fair price; a BREAK (deviation from it, variance expansion)
in a LONG-GAMMA regime should REVERT (dealers resist the break) back to the last fair price. This is the
NONLINEAR reversion the linear tests missed -- unconditional pin-reversion was null (gex_regime_test C) and
the linear regime x momentum was null (gex_bar_test: bar autocorrelation identical across regimes), but
neither tested reversion CONDITIONED ON a break event AND the long-gamma regime jointly. A zero
unconditional autocorrelation does not exclude a conditional-on-large-deviation reversion.

Spec (causal, per 30-min session bar): fair price = the recent intraday consolidation level (rolling-6-bar
mean of the within-day cumulative return); dev = current level - that anchor (the break); long-gamma =
relu(lagged z_gex). Reversion signal = -dev (revert toward fair price), tested plain, x long-gamma, and
x long-gamma x |dev| (reversion only after a BIG break). Base = recent-bar momentum (the control). Target =
next same-day session bar return. Gate = feature_cv (fwd MSE + circular-shift placebo + replication) with
the net-of-1-tick sign P&L. PRE-COMMIT: clears the gate => a real edge; fails => directional hunt is done,
no further slicing. Writes results/gex_reversion/summary.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from flow_dir_pilot import oos_readout, score_dir
from gex_regime_test import lag_regime
from src.evaluation.feature_cv import purged_walk_forward, significance_gate
from vrp_screen_worker import build

ANCHOR_WIN = 6  # bars (~3h) defining the "recent consolidation" fair-price level


def main() -> None:
    os.makedirs("results/gex_reversion", exist_ok=True)
    df, *_ = build()
    t = pd.to_datetime(df["t"])
    hour = t.dt.hour.to_numpy()
    day = t.dt.normalize()
    sess = ((hour >= 9) & (hour <= 15)) & (df["sumret_ind"].to_numpy(float) <= 0)

    sd = pd.DataFrame({"day": day.to_numpy(), "r": df["sumret"].to_numpy(float)})[
        sess
    ].copy()
    sd["cumret"] = sd.groupby("day")[
        "r"
    ].cumsum()  # intraday price level (from the day's open)
    sd["anchor"] = sd.groupby("day")["cumret"].transform(
        lambda x: x.rolling(ANCHOR_WIN, min_periods=1).mean()
    )  # the recent consolidation / fair-price level
    sd["dev"] = sd["cumret"] - sd["anchor"]  # deviation from fair price = the break
    sd["m1"] = sd["r"]
    sd["m3"] = sd.groupby("day")["r"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    sd["r_fwd"] = sd.groupby("day")["r"].shift(-1)  # next SAME-DAY session bar

    # lagged long-gamma strength broadcast to bars (causal; pre-1996/missing -> 0)
    reg = lag_regime(
        pd.read_parquet("results/gex/gex_daily.parquet"),
        pd.DatetimeIndex(sorted(sd["day"].unique())),
    )
    zg = np.nan_to_num(reg["z_gex"].reindex(sd["day"]).to_numpy(), nan=0.0)
    pos_gamma = np.maximum(zg, 0.0)  # relu: long-gamma strength (0 in short-gamma)

    dev = sd["dev"].to_numpy()
    rev = -dev  # revert toward fair price
    feats = {
        "rev_plain": rev,
        "rev_x_longgamma": rev * pos_gamma,
        "rev_x_longgamma_x_break": rev * pos_gamma * np.abs(dev),
    }
    ok = np.isfinite(sd["r_fwd"].to_numpy())
    idx = np.where(ok)[0]
    y = sd["r_fwd"].to_numpy()[idx]
    base = np.column_stack([sd["m1"].to_numpy()[idx], sd["m3"].to_numpy()[idx]])
    folds = purged_walk_forward(len(idx), n_folds=5, embargo=0.01)

    res: dict = {"n_bars": int(len(idx)), "anchor_win": ANCHOR_WIN, "arms": {}}
    res["mom_base"] = oos_readout(base, None, y, folds)
    for name, f in feats.items():
        cand = f[idx].reshape(-1, 1)
        sc = score_dir(base, cand, y, folds)
        passed, comp = significance_gate(sc, k=2.0)
        rd = oos_readout(base, cand, y, folds)
        res["arms"][name] = {"gate_pass": bool(passed), **comp, "readout": rd}
        print(
            f"[{name}] GATE={'PASS' if passed else 'fail'} repl={comp['repl_frac']:.2f} "
            f"foldrepl={comp['fold_repl_frac']:.2f} beats_plac={comp['beats_placebo']} | "
            f"corr={rd['oos_corr']:+.4f} cov_t={rd['cov_t_hac5']:+.2f} net={rd['net_sign_mean_bp']:+.3f}bp",
            flush=True,
        )
    b = res["mom_base"]
    print(
        f"[mom base] corr={b['oos_corr']:+.4f} cov_t={b['cov_t_hac5']:+.2f} n={b['n']}",
        flush=True,
    )
    with open("results/gex_reversion/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
