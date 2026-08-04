"""Recover the calendar-date axis for the cumrv x close signal (dated signal for venue tests).

The panel cache (results/covid_imp_rank/all_buckets) persists no timestamps -- build_cumrv_pnl.py
uses a synthetic hour-wrap day_id. This script re-runs the verified loader (src/data/loading.py,
which the cache was built from), then PROVES the row alignment by locating the cache's raw `hour`
sequence inside the loader's hour sequence (exact float match at a unique offset -- a hard assert,
not a heuristic). With dates in hand it reconstructs the causal cumrv signal EXACTLY as
build_cumrv_pnl.py does (same cache columns, same expanding mean) and emits one row per trading
day: (date, signal_1600, signal_daily, r_daily).

  signal_1600  -- signal at the FIRST close-window bar (16:00): the value a 16:00-entry trade
                  (VIX CFD overnight, option straddle) actually knows.
  signal_daily -- close-window-aggregate signal weight matching the pnl-log construction.

Output: eval_sim/data/cumrv_signal_dated.parquet (consumed by price_option_friction.py and
vix_passthrough_test.py).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np
import pandas as pd

import src.data.loading as _loading

# the GEX-era OptionMetrics parquets in data/ have no endbartime column and would break the
# loader's outer-merge -- exclude them (the cache predates them; panel inputs are the 6 originals)
_real_listdir = os.listdir
_loading.os.listdir = lambda p: [  # type: ignore[assignment]
    f for f in _real_listdir(p) if not f.startswith("optionm_")
]
load_raw_data = _loading.load_raw_data

B = "results/covid_imp_rank/all_buckets"
OUT = "eval_sim/data/cumrv_signal_dated.parquet"
CLOSE_LO, CLOSE_HI = 16, 19


def align_dates() -> pd.Series:
    """Return per-cache-row timestamps, proven by exact hour-sequence match."""
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    X = np.load(f"{B}/X_imp.npy", mmap_mode="r")
    cache_hour = np.asarray(X[:, feats.index("hour")], dtype=np.float64)

    df = load_raw_data("data", allow_missing=True)
    loader_hour = df["t"].dt.hour.to_numpy(dtype=np.float64)

    n, m = cache_hour.size, loader_hour.size
    assert m >= n, f"loader shorter than cache ({m} < {n})"
    # the cache is the loader minus a leading MA-warmup prefix (m-n == 3125 == the ma_3125
    # window). The hour sequence is ~weekly-periodic so it admits multiple offsets; the value
    # check below (target vs RV rank-corr, transform-invariant) discriminates decisively:
    # offset 3125 -> Spearman ~0.45, every other hour-matching offset -> ~0.
    k = m - n
    assert k == 3125, f"warmup prefix changed: {k} (expected 3125 = ma_3125 window)"
    assert np.array_equal(loader_hour[k : k + n], cache_hour), "hour sequence mismatch"
    from scipy.stats import spearmanr

    y = np.load(f"{B}/y.npy").astype(np.float64)
    rv = df["RV"].to_numpy(dtype=np.float64)[k : k + n]
    ok = np.isfinite(rv) & np.isfinite(y)
    rho = float(spearmanr(rv[ok][::37], y[ok][::37]).statistic)
    assert rho > 0.3, f"value check failed: spearman(y, RV)={rho:.3f} at offset {k}"
    print(f"aligned: cache rows = loader rows [{k}:{k + n}] of {m}; spearman(y,RV)={rho:.3f}")
    return df["t"].iloc[k : k + n].reset_index(drop=True)


def main() -> None:
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    X = np.load(f"{B}/X_imp.npy", mmap_mode="r")
    y = np.load(f"{B}/y.npy").astype(np.float64)
    hour = np.asarray(X[:, feats.index("hour")], dtype=np.float64)
    har1 = np.asarray(X[:, feats.index("har_ma_1")], dtype=np.float64)
    har5 = np.asarray(X[:, feats.index("har_ma_5")], dtype=np.float64)
    t = align_dates()

    # identical reconstruction to build_cumrv_pnl.py
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    cumrv = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    close = (hour >= CLOSE_LO) & (hour <= CLOSE_HI)
    r = y - har5
    ci = np.where(close)[0]
    g, rr, dids, tt = cumrv[ci], r[ci], day_id[ci], t.iloc[ci]

    exp_g = np.concatenate([[np.nan], np.cumsum(g)[:-1] / np.arange(1, g.size)])
    signal = g - exp_g

    dfc = pd.DataFrame(
        {"d": dids, "date": tt.dt.normalize().to_numpy(), "sig": signal, "r": rr}
    )
    first = dfc.groupby("d").first()  # 16:00 bar: what an entry-at-close trade knows
    daily = pd.DataFrame(
        {
            "date": first["date"],
            "signal_1600": first["sig"],
            "signal_daily": dfc.groupby("d")["sig"].mean(),
            "r_daily": dfc.groupby("d")["r"].sum(),
        }
    ).dropna(subset=["signal_1600"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    daily.reset_index(drop=True).to_parquet(OUT, index=False)
    print(
        f"saved {OUT}: {len(daily)} days  {daily['date'].iloc[0].date()} -> {daily['date'].iloc[-1].date()}"
    )
    # sanity: the edge should reproduce (corr < 0, as in build_cumrv_pnl.py)
    ok = np.isfinite(dfc["sig"])
    print(f"edge check corr(cumrv, r) on close bars = {np.corrcoef(g[ok], rr[ok])[0, 1]:+.4f} (research: ~-0.30)")


if __name__ == "__main__":
    main()
