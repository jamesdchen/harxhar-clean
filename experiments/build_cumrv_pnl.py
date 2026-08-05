"""Construct a historical trade-log for eval_sim from the cumrv x close edge (research finding).

The edge (writeup: FWL attribution): cumrv_x_close -- within-day cumulative abnormal RV gated to the
16-19 close/AH window -- carries a NEGATIVE coefficient: after a high-intraday-vol day, close/AH realized
vol mean-reverts BELOW the HAR baseline. A simple, CAUSAL strategy that trades this:

  signal_t = cumrv_x_close_t - E_causal[cumrv]        (how far above its expanding mean)
  r_t      = y_t - HAR_forecast_t                     (close-vol surprise vs a HAR forecast)
  pnl_t    = -signal_t * r_t                          (short vol when cumrv high; edge => E[pnl] > 0)

We aggregate close-bar PnL to one trade per calendar day, scale to a target daily $ std for a prop
account, and save the daily PnL series as eval_sim/data/cumrv_close_pnl.npy -- the empirical log the
eval simulator block-bootstraps. Reconstruction of cumrv mirrors resid_amortized._cumrv_close (cumsum of
har_ma_1 within the hour-wrap day, gated close); HAR forecast proxy = har_ma_5.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

B = "results/covid_imp_rank/all_buckets"
OUT = "eval_sim/data/cumrv_close_pnl.npy"
TARGET_DAILY_STD = (
    400.0  # scale the (unit-scale) strategy PnL to ~$/day for a ~$50k account
)
CLOSE_LO, CLOSE_HI = 16, 19


def main():
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    X = np.load(f"{B}/X_imp.npy", mmap_mode="r")
    y = np.load(f"{B}/y.npy").astype(np.float64)
    hour = np.asarray(X[:, feats.index("hour")], dtype=np.float64)
    har1 = np.asarray(X[:, feats.index("har_ma_1")], dtype=np.float64)
    har5 = np.asarray(X[:, feats.index("har_ma_5")], dtype=np.float64)

    # reconstruct cumrv_x_close: cumsum of har_ma_1 within the hour-wrap day, gated to the close window
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    cumrv = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    close = (hour >= CLOSE_LO) & (hour <= CLOSE_HI)

    r = y - har5  # close-vol surprise vs a HAR forecast
    ci = np.where(close)[0]
    g, rr, dids = cumrv[ci], r[ci], day_id[ci]

    # causal expanding mean of the signal (shifted so bar i sees only prior close bars)
    exp_g = np.concatenate([[np.nan], np.cumsum(g)[:-1] / np.arange(1, g.size)])
    signal = g - exp_g
    pnl_bar = -signal * rr  # short vol when cumrv above expectation
    ok = np.isfinite(pnl_bar)
    corr = float(np.corrcoef(g[ok], rr[ok])[0, 1])  # edge check: should be < 0

    # aggregate close-bar PnL to one trade per day
    daily = (
        pd.DataFrame({"d": dids[ok], "p": pnl_bar[ok]})
        .groupby("d")["p"]
        .sum()
        .to_numpy()
    )
    raw_std = daily.std()
    daily_scaled = daily / raw_std * TARGET_DAILY_STD if raw_std > 0 else daily

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, daily_scaled)

    mu, sd = daily_scaled.mean(), daily_scaled.std()
    sharpe = mu / sd * np.sqrt(252) if sd > 0 else float("nan")
    skew = float(((daily_scaled - mu) ** 3).mean() / sd**3) if sd > 0 else float("nan")
    print(f"close bars={ok.sum()}  trading days={daily.size}")
    print(f"edge corr(cumrv, r) on close bars = {corr:+.4f}   (research: negative)")
    print(
        f"daily PnL $: mean={mu:+.2f} std={sd:.2f}  ann.Sharpe={sharpe:+.2f}  skew={skew:+.2f}"
    )
    print(
        f"  win-rate={float((daily_scaled > 0).mean()):.3f}  min={daily_scaled.min():.0f} max={daily_scaled.max():.0f}"
    )
    print(f"  cumulative $ over sample = {daily_scaled.sum():+.0f}")
    print(
        f"saved {OUT}  ({daily.size} daily PnLs, scaled to ${TARGET_DAILY_STD:.0f}/day std)"
    )


if __name__ == "__main__":
    main()
