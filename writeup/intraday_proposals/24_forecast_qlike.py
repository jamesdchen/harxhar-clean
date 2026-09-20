"""Proposal 24 — QLIKE of the eight RV forecasts on the scored 2020-2024 bars.

The book trades sign(s) = sign(rv_hat - slice). If ridge QLIKE on these
bars is close to (or better than) the lagged per-clock mean of RV, the
daytime Sharpe gap is not "the forecast failed in 2020-2024".

QLIKE = y/f - log(y/f) - 1 (Patton). Lower is better.
Naive = expanding per-clock mean of RV on the in-fit panel, shift(1), min 63.

Run:  python writeup/intraday_proposals/24_forecast_qlike.py
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

WORKERS = min(7, os.cpu_count() or 4)
REPO = asl.find_repo(Path(__file__).resolve().parent)
INTRA = REPO / "results" / "atm_straddle_intraday"
CACHE = INTRA / "cache"
OUT = INTRA / "proposals" / "24"
CLOSE = "15:30"
PROFILE_MIN_DAYS = 63


def load_panel_for_tag(tag: str):
    return tag, asl.load_yhat_panel_mz(asl.yhat_paths(REPO)[tag])


def load_panels_parallel():
    n = min(WORKERS, len(asl.MODEL_ORDER))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=n) as pool:
        loaded = dict(pool.map(load_panel_for_tag, asl.MODEL_ORDER))
    print(f"loaded {len(loaded)} panels in {time.time() - t0:.1f}s ({n} workers)")
    return loaded


def qlike_mean(y, f):
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    m = np.isfinite(y) & np.isfinite(f) & (y > 0) & (f > 0)
    if int(m.sum()) < 2:
        return float("nan"), 0, float("nan"), float("nan")
    r = y[m] / f[m]
    corr = float(pd.Series(y[m]).corr(pd.Series(f[m])))
    ratio = float(y[m].mean() / f[m].mean())
    return float(np.mean(r - np.log(r) - 1.0)), int(m.sum()), corr, ratio


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    pkg = pd.read_parquet(sorted(CACHE.glob("trade_*.parquet"))[-1])
    loaded = load_panels_parallel()
    pan = loaded["blk2"]
    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    clocks = sorted(pkg["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    pkg["h_rem"] = pkg["hhmm"].map(n_rem).astype(float) * 0.5
    pkg["V_M"] = pkg["iv_hourly"].astype(float) ** 2 * pkg["h_rem"]

    pm = pan.set_index("t")[["rv_hat", "rv_raw", "in_fit"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()

    pf = pan.loc[pan["in_fit"].to_numpy(dtype=bool)].copy()
    clock = pd.to_datetime(pf["t"], utc=True).dt.tz_convert(
        "America/New_York"
    ) - pd.Timedelta(minutes=30)
    pf["pdate"] = clock.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clock.dt.strftime("%H:%M")
    wide = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    wide = wide.reindex(columns=clocks)
    naive = wide.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    pi = pd.DataFrame(index=wide.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        rem = wide[clocks[i:]].sum(axis=1)
        pi[c] = wide[c] / rem.replace(0.0, np.nan)
    w = pi.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work["rv_naive"] = naive.stack().reindex(mi).to_numpy()
    work["w_slice"] = w.stack().reindex(mi).to_numpy()
    work["slice"] = work["w_slice"] * work["V_M"]

    for tag in asl.MODEL_ORDER:
        d = loaded[tag][["t", "rv_hat"]].copy()
        d["t"] = pd.to_datetime(d["t"], utc=True) - pd.Timedelta(minutes=30)
        work["rv_hat_" + tag] = (
            work[["t"]].merge(d, on="t", how="left")["rv_hat"].to_numpy()
        )

    print(
        f"scored bars {len(work):,}  days {work['date'].nunique()}  "
        f"{work['date'].min().date()} -> {work['date'].max().date()}"
    )

    forecasts = {asl.YHAT_LABEL[t]: "rv_hat_" + t for t in asl.MODEL_ORDER}
    forecasts["naive clock mean"] = "rv_naive"
    forecasts["implied slice"] = "slice"

    rows = []
    chunks = [("pooled", work), ("10:00-15:00", work[work["hhmm"] != CLOSE])]
    chunks += [(str(hh), g) for hh, g in work.groupby("hhmm", sort=True)]
    for hh, g in chunks:
        y = g["rv_raw"]
        for name, col in forecasts.items():
            q, n, corr, ratio = qlike_mean(y, g[col])
            rows.append(
                {
                    "window": hh,
                    "forecast": name,
                    "n": n,
                    "QLIKE": q,
                    "corr": corr,
                    "RV/f": ratio,
                }
            )
    tab = pd.DataFrame(rows)
    print("\nQLIKE pooled / body / close")
    sub = tab[tab["window"].isin(["pooled", "10:00-15:00", "15:30"])]
    print(
        sub.pivot(index="forecast", columns="window", values="QLIKE").to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )
    print("\ncorr(RV, f) pooled")
    print(
        tab[tab["window"] == "pooled"]
        .set_index("forecast")[["QLIKE", "corr", "RV/f", "n"]]
        .to_string(float_format=lambda x: f"{x:.4f}")
    )
    print("\nQLIKE by clock, block-diagonal ridge vs naive vs slice")
    byc = tab[
        tab["forecast"].isin(
            ["block-diagonal ridge", "naive clock mean", "implied slice"]
        )
    ]
    byc = byc[~byc["window"].isin(["pooled", "10:00-15:00", "15:30"])]
    print(
        byc.pivot(index="window", columns="forecast", values="QLIKE").to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )
    tab.to_csv(OUT / "24_qlike.csv", index=False)
    print(f"\nwrote {OUT}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
