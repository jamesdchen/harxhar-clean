"""Test JJ Simon's 'fair value' core hypothesis with our gate: reversion to the SESSION-OPEN anchor, and
the two-phase (continuation-then-reversion) structure -- the correctly-specified version of the reversion
trade (my earlier gex_reversion used a rolling-mean anchor + all bars, and was null; this uses the FIXED
open anchor + the intraday PHASE, which is his actual claim).

His thesis: deviation from the 9:30 open predicts CONTINUATION early in the session (price pushes away) then
REVERSION later (price returns to the open). We (a) MEASURE the phase crossover per intraday bar-position k
(does corr(dev_from_open, next_bar_return) flip + -> - as k grows?) instead of guessing 10-15 min, and (b)
GATE the reversion-phase signal, plain and conditioned on the GEX long-gamma regime and on a displacement
(only revert after a real break). Caveat: 30-min SPX bars, not his 1-min NQ -- this tests whether the EDGE
exists, not his tick execution. Writes results/fairvalue/summary.json.
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


def main() -> None:
    os.makedirs("results/fairvalue", exist_ok=True)
    df, *_ = build()
    t = pd.to_datetime(df["t"])
    hour = t.dt.hour.to_numpy()
    day = t.dt.normalize()
    sess = ((hour >= 9) & (hour <= 15)) & (df["sumret_ind"].to_numpy(float) <= 0)

    sd = pd.DataFrame({"day": day.to_numpy(), "r": df["sumret"].to_numpy(float)})[
        sess
    ].copy()
    sd["k"] = sd.groupby(
        "day"
    ).cumcount()  # intraday bar position from the session open (the anchor)
    sd["dev"] = sd.groupby("day")[
        "r"
    ].cumsum()  # cumulative return since open = deviation from anchor
    sd["r_fwd"] = sd.groupby("day")["r"].shift(-1)  # next same-day bar

    # (a) PHASE CROSSOVER: per bar-position k, does dev-from-open predict continuation (+) or reversion (-)?
    phase = {}
    for k in range(0, 13):
        m = (sd["k"] == k) & np.isfinite(sd["r_fwd"])
        if m.sum() > 200:
            dv, fw = sd.loc[m, "dev"].to_numpy(), sd.loc[m, "r_fwd"].to_numpy()
            phase[k] = {
                "n": int(m.sum()),
                "corr_dev_fwd": float(np.corrcoef(dv, fw)[0, 1]),
            }
    xline = "  ".join(f"k{k}:{v['corr_dev_fwd']:+.3f}" for k, v in phase.items())
    print(f"[phase corr(dev_from_open, next_ret) by bar]  {xline}", flush=True)
    # crossover = first k where the sign turns negative (continuation -> reversion)
    kc = next((k for k in phase if phase[k]["corr_dev_fwd"] < 0), 3)
    print(
        f"[crossover] continuation->reversion at bar k={kc} (his guess ~10-15min = k1)",
        flush=True,
    )

    # (b) GATED reversion in the measured reversion window (k >= kc)
    reg = lag_regime(
        pd.read_parquet("results/gex/gex_daily.parquet"),
        pd.DatetimeIndex(sorted(sd["day"].unique())),
    )
    zg = np.nan_to_num(reg["z_gex"].reindex(sd["day"]).to_numpy(), nan=0.0)
    pos_gamma = np.maximum(zg, 0.0)
    win = (sd["k"].to_numpy() >= kc) & np.isfinite(sd["r_fwd"].to_numpy())
    idx = np.where(win)[0]
    y = sd["r_fwd"].to_numpy()[idx]
    dev = sd["dev"].to_numpy()[idx]
    base = sd["r"].to_numpy()[idx].reshape(-1, 1)  # last-bar momentum control
    folds = purged_walk_forward(len(idx), n_folds=5, embargo=0.01)

    feats = {
        "revert_to_open": -dev,
        "revert_x_longgamma": -dev * pos_gamma[idx],
        "revert_x_displacement": -dev
        * np.abs(dev),  # revert harder after a bigger break
    }
    res: dict = {"n_bars": int(len(idx)), "crossover_k": kc, "phase": phase, "arms": {}}
    res["base"] = oos_readout(base, None, y, folds)
    for name, f in feats.items():
        cand = f.reshape(-1, 1)
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
    with open("results/fairvalue/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
