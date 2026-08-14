"""One-bar variance trade that spends the blk2-vs-a0 QLIKE increment.

pred = yhat^2 * baseline   (paper sqrt-scale, one bar)
size = pred_blk2 - pred_a0   (extra long RV when exog raises E[RV])
payoff struck at a0:          size * (rv - pred_a0)
payoff struck at midpoint:    size * (rv - (pred_a0+pred_blk2)/2)
  the second is the MSE identity: (rv-a0)^2 - (rv-blk2)^2
  = 2 * size * (rv - mid)

Uses the dumped unification series (already on disk). No option tape.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _load(path: str, tag: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["t"] = pd.to_datetime(df["t"], utc=True).astype("datetime64[ns, UTC]")
    df["pred"] = df["yhat"].to_numpy(float) ** 2 * df["baseline"].to_numpy(float)
    return df[["t", "pred", "rv_raw"]].rename(
        columns={"pred": f"pred_{tag}", "rv_raw": f"rv_{tag}"}
    )


def _report(name: str, x: np.ndarray) -> None:
    x = x[np.isfinite(x)]
    if x.size < 3:
        print(f"  {name}: empty")
        return
    sh = float(x.mean() / x.std()) if float(x.std()) > 0 else float("nan")
    print(
        f"  {name}: n={x.size:,}  mean={x.mean():+.4e}  "
        f"Sharpe/bar={sh:+.4f}  Sharpe*sqrt(252*48)={sh * np.sqrt(252 * 48):+.2f}"
    )


def _eval(m: pd.DataFrame, label: str) -> None:
    a = m["pred_a0"].to_numpy(float)
    b = m["pred_b2"].to_numpy(float)
    rv = m["rv_a0"].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(rv) & (rv > 0)
    a, b, rv = a[ok], b[ok], rv[ok]
    gap = b - a
    z = gap / (gap.std() + 1e-18)
    print(f"{label}  n={ok.sum():,}  P(blk2>a0)={(gap > 0).mean():.2%}")
    _report("size*(rv - pred_a0)", z * (rv - a))
    _report("size*(rv - mid)  [MSE]", z * (rv - 0.5 * (a + b)))
    _report("size*rv", z * rv)
    # also unnormalized MSE check
    mse_a = float(np.mean((rv - a) ** 2))
    mse_b = float(np.mean((rv - b) ** 2))
    print(f"  MSE a0={mse_a:.6e}  blk2={mse_b:.6e}  dMSE={mse_a - mse_b:+.6e}")

    # Paper score as a book: QLIKE(a0)-QLIKE(blk2); + means blk2 wins that bar.
    def _ql(f, y):
        f = np.maximum(f, 1e-18)
        y = np.maximum(y, 1e-18)
        return y / f - np.log(y / f) - 1.0

    dq = _ql(a, rv) - _ql(b, rv)
    _report("QLIKE(a0)-QLIKE(blk2)", dq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="results/spxw_pnl/yhat_a0.parquet")
    ap.add_argument("--blk2", default="results/spxw_pnl/yhat_blk2.parquet")
    ap.add_argument(
        "--trades", default=None, help="optional SPXW trade file to restrict times"
    )
    a = ap.parse_args()
    a0 = _load(a.a0, "a0")
    b2 = _load(a.blk2, "b2")
    m = pd.merge(a0, b2, on="t", how="inner")
    _eval(m, "full panel (every scored bar)")
    if a.trades and os.path.exists(a.trades):
        tr = pd.read_parquet(a.trades)
        t0 = (
            pd.to_datetime(tr["t0"], utc=True)
            .astype("datetime64[ns, UTC]")
            .drop_duplicates()
        )
        sub = pd.merge_asof(
            pd.DataFrame({"t": np.sort(t0)}),
            m.sort_values("t"),
            on="t",
            direction="backward",
        )
        _eval(sub, "SPXW stamps only")


if __name__ == "__main__":
    main()
