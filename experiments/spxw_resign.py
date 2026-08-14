"""Re-sign existing ATM-straddle trades with paper a0 / blk2 forecasts.

pred_var_H = H * yhat^2 * baseline   (yhat is sqrt-scale)
impl_var_H = iv^2 * H / (252 * 48)
sign = sign(pred - impl)

IV is the mean of the two legs' impl_volatility at t0 (0-delta rows kept).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

BARS_PER_YEAR = 252.0 * 48.0


def _load_iv(chain_path: str, trades: pd.DataFrame) -> np.ndarray:
    ch = pd.read_parquet(
        chain_path,
        columns=["timestamp", "expiration", "strike", "impl_volatility"],
    )
    ch["timestamp"] = pd.to_datetime(ch["timestamp"], utc=True)
    ch["expiration"] = pd.to_datetime(ch["expiration"]).dt.normalize()
    g = ch.groupby(["timestamp", "expiration", "strike"], sort=False)[
        "impl_volatility"
    ].mean()
    exp = pd.to_datetime(trades["expiration"]).dt.normalize()
    key = pd.MultiIndex.from_arrays([trades["t0"], exp, trades["strike"]])
    return g.reindex(key).to_numpy(float)


def _report(name: str, x: np.ndarray) -> None:
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"  {name}: empty")
        return
    sh = float(x.mean() / x.std()) if float(x.std()) > 0 else float("nan")
    print(
        f"  {name}: n={x.size:,}  mean={x.mean():+.4f}  sharpe/trade={sh:+.3f}  long%={(x > 0).mean():.2%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--yhat", required=True)
    ap.add_argument("--label", default="fcst")
    ap.add_argument("--chain", default="data/spxw_chain.parquet")
    a = ap.parse_args()

    tr = pd.read_parquet(a.trades)
    tr["t0"] = pd.to_datetime(tr["t0"], utc=True).astype("datetime64[ns, UTC]")
    yh = pd.read_parquet(a.yhat)
    yh["t"] = pd.to_datetime(yh["t"], utc=True).astype("datetime64[ns, UTC]")
    yh = yh.sort_values("t")
    m = pd.merge_asof(
        tr.sort_values("t0"), yh, left_on="t0", right_on="t", direction="backward"
    )
    # Vendor new_implied_vol on this tape is ~0.002 at ATM while mids
    # price ~20% vol. Invert from the straddle mid + time to 16:00 ET expiry.
    exp_close = (
        pd.to_datetime(m["expiration"]).dt.tz_localize("America/New_York")
        + pd.Timedelta(hours=16)
    ).dt.tz_convert("UTC")
    tau = (exp_close - m["t0"]).dt.total_seconds() / (365.25 * 24 * 3600)
    tau = tau.to_numpy(float)
    mid = m["entry"].to_numpy(float)
    S = m["spot"].to_numpy(float)
    iv = mid / (S * np.sqrt(np.maximum(tau, 1e-8)) * np.sqrt(2.0 / np.pi))
    iv = np.where((tau > 0) & np.isfinite(iv) & (iv > 0), iv, np.nan)
    m["iv"] = iv
    h = m["h"].to_numpy(int)
    yhat = m["yhat"].to_numpy(float)
    base = m["baseline"].to_numpy(float) if "baseline" in m.columns else np.ones(len(m))
    pred = h * (yhat**2) * base
    impl = (iv**2) * h / BARS_PER_YEAR
    sgn = np.sign(pred - impl)
    sgn[~np.isfinite(pred) | ~np.isfinite(impl)] = 0.0
    d = (
        m["d_mid"].to_numpy(float)
        if "d_mid" in m.columns
        else m["d_long"].to_numpy(float)
    )
    print(f"{a.label}  file={os.path.basename(a.trades)}  n={len(m):,}")
    print(
        f"  yhat~{np.nanmedian(yhat):.3f}  iv~{np.nanmedian(iv):.3f}  sign+={(sgn > 0).mean():.2%}"
    )
    _report("long", d)
    _report("signed", sgn * d)


if __name__ == "__main__":
    main()
