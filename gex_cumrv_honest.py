"""Fork-A closure: does GEX move the HONEST cumrv AH edge (the 0.271 sigma/day breakeven number)?

gex_vrp answered the every-bar VRP premium-ranking deployment (wash). This answers the remaining
deployment: the cumrv x close edge whose honest baseline (causal expanding-OLS on the 6 HAR MAs +
VIX/VVIX MAs, refit-480, strictly OOS -- commit bd22177) set daily Sharpe 0.271 = the breakeven
instrument cost. Two readings from one computation, rebuilding the surprise r = y - forecast with
GEX (lagged daily z_gex/z_flip/|z_gex|, broadcast to bars) added to the forecast:
  - Sharpe UP   => GEX widens the cost headroom (the step-3 hope);
  - Sharpe DOWN => part of the "edge" was GEX-priceable income (a gamma-aware counterparty keeps it);
  - flat        => orthogonal to dealer gamma, same as the VIX/VVIX hardening result.

The original honest-builder script was not committed (only the .npy), so this REPLICATES the
baseline first -- the baseline arm's Sharpe vs the recorded 0.271 is the validity check -- and the
GEX delta is read within this one verified machinery (both arms differ only in forecast columns).
Signal (cumrv - expanding mean) is IDENTICAL across arms; only the counterparty forecast changes.
Writes results/gex_cumrv_honest/summary.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from build_cumrv_signal_dated import align_dates
from gex_regime_test import lag_regime

B = "results/covid_imp_rank/all_buckets"
OUT = "results/gex_cumrv_honest"
CLOSE_LO, CLOSE_HI = 16, 19
REFIT = 480
MIN_TRAIN = 2400


def expanding_ols_forecast(
    X: np.ndarray, y: np.ndarray, refit: int, min_train: int
) -> np.ndarray:
    """Strictly-OOS expanding OLS: bars [t, t+refit) predicted by the fit on bars [0, t).
    Incremental Gram accumulation so the all-bars variant stays O(n p^2)."""
    n, p = X.shape
    Xi = np.column_stack([np.ones(n), X])
    G = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)
    pred = np.full(n, np.nan)
    coef = None
    for t in range(0, n, refit):
        if t >= min_train:
            coef = np.linalg.lstsq(G, b, rcond=None)[0]
        chunk = Xi[t : t + refit]
        if coef is not None:
            pred[t : t + refit] = chunk @ coef
        G += chunk.T @ chunk
        b += chunk.T @ y[t : t + refit]
    return pred


def daily_pnl(signal, surprise, dids, dates):
    """-signal * surprise on close bars, one trade per day. Returns (dates, pnl)."""
    pnl = -signal * surprise
    ok = np.isfinite(pnl)
    g = (
        pd.DataFrame({"d": dids[ok], "date": dates[ok], "p": pnl[ok]})
        .groupby("d")
        .agg(date=("date", "first"), p=("p", "sum"))
    )
    return g["date"].to_numpy(), g["p"].to_numpy()


def sharpe(x):
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std()) if len(x) and x.std() > 0 else float("nan")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    X = np.load(f"{B}/X_imp.npy", mmap_mode="r")
    y = np.load(f"{B}/y.npy").astype(np.float64)
    hour = np.asarray(X[:, feats.index("hour")], dtype=np.float64)
    t = align_dates()
    day = pd.to_datetime(t).dt.normalize()

    har_cols = [f"har_ma_{w}" for w in (1, 5, 25, 125, 625, 3125)]
    iv_cols = [
        f"adj_{s}_ma_{w}" for s in ("vix", "vvix") for w in (1, 5, 25, 125, 625, 3125)
    ]
    base_cols = har_cols + iv_cols
    Xb = np.asarray(X[:, [feats.index(c) for c in base_cols]], dtype=np.float64)

    # lagged daily GEX regime broadcast to bars (causal; pre-coverage/missing -> 0, as in gex_vrp)
    reg = lag_regime(
        pd.read_parquet("results/gex/gex_daily.parquet"),
        pd.DatetimeIndex(sorted(day.unique())),
    )
    covered = reg["z_gex"].notna()

    def bcast(c):
        return np.nan_to_num(reg[c].reindex(day).to_numpy(float), nan=0.0)

    z_gex = bcast("z_gex")
    Xg = np.column_stack([Xb, z_gex, bcast("z_flip"), np.abs(z_gex)])

    # signal: identical reconstruction to build_cumrv_pnl.py
    har1 = np.asarray(X[:, feats.index("har_ma_1")], dtype=np.float64)
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    cumrv = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    ci = np.where((hour >= CLOSE_LO) & (hour <= CLOSE_HI))[0]
    g = cumrv[ci]
    exp_g = np.concatenate([[np.nan], np.cumsum(g)[:-1] / np.arange(1, g.size)])
    signal = g - exp_g

    res: dict = {
        "refit": REFIT,
        "min_train": MIN_TRAIN,
        "base_cols": base_cols,
        "arms": {},
    }
    for arm, Xf in (("baseline_HARVIXVVIX", Xb), ("plus_gex", Xg)):
        pred = expanding_ols_forecast(Xf, y, REFIT, MIN_TRAIN)
        surprise = (y - pred)[ci]
        dts, pnl = daily_pnl(signal, surprise, day_id[ci], day.to_numpy()[ci])
        cov_days = set(reg.index[covered].to_numpy())
        cmask = np.isin(dts, list(cov_days))
        res["arms"][arm] = {
            "n_days": int(len(pnl)),
            "sharpe_daily": sharpe(pnl),
            "n_days_gex_covered": int(cmask.sum()),
            "sharpe_daily_gex_covered": sharpe(pnl[cmask]),
        }
        r = res["arms"][arm]
        print(
            f"[{arm}] Sharpe/day full={r['sharpe_daily']:+.4f} (n={r['n_days']})  "
            f"gex-covered={r['sharpe_daily_gex_covered']:+.4f} (n={r['n_days_gex_covered']})",
            flush=True,
        )
    b0 = res["arms"]["baseline_HARVIXVVIX"]["sharpe_daily_gex_covered"]
    g0 = res["arms"]["plus_gex"]["sharpe_daily_gex_covered"]
    res["delta_sharpe_gex_covered"] = g0 - b0
    res["replication_check_vs_0.271"] = res["arms"]["baseline_HARVIXVVIX"][
        "sharpe_daily"
    ]
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(
        f"delta Sharpe (gex-covered) = {g0 - b0:+.4f}   [breakeven ref: baseline 0.271]",
        flush=True,
    )


if __name__ == "__main__":
    main()
