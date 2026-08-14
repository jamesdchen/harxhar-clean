"""Incremental ATM-weekly book that is only the exog forecast gap.

Long-straddle size = pred_blk2 - pred_a0
  pred = H * yhat^2 * baseline   (paper sqrt-scale)

That is s_blk2 - s_a0 if each model sizes s ∝ (pred - IV^2).
IV cancels; the book is exog and nothing else.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _align(tr: pd.DataFrame, yh: pd.DataFrame, suffix: str) -> pd.DataFrame:
    yh = yh.copy()
    yh["t"] = pd.to_datetime(yh["t"], utc=True).astype("datetime64[ns, UTC]")
    yh = yh.sort_values("t").rename(
        columns={
            "yhat": f"yhat_{suffix}",
            "baseline": f"base_{suffix}",
            "t": f"t_{suffix}",
        }
    )
    tr = tr.sort_values("t0")
    return pd.merge_asof(
        tr, yh, left_on="t0", right_on=f"t_{suffix}", direction="backward"
    )


def _one(path: str, a0_path: str, b2_path: str) -> None:
    tr = pd.read_parquet(path)
    tr["t0"] = pd.to_datetime(tr["t0"], utc=True).astype("datetime64[ns, UTC]")
    dcol = "d_long" if "d_long" in tr.columns else "d_mid"
    a0 = pd.read_parquet(a0_path)
    b2 = pd.read_parquet(b2_path)
    m = _align(tr, a0, "a0")
    m = _align(m, b2, "b2")
    h = m["h"].to_numpy(float)
    p_a0 = h * (m["yhat_a0"].to_numpy(float) ** 2) * m["base_a0"].to_numpy(float)
    p_b2 = h * (m["yhat_b2"].to_numpy(float) ** 2) * m["base_b2"].to_numpy(float)
    gap = p_b2 - p_a0
    d = m[dcol].to_numpy(float)
    ok = np.isfinite(gap) & np.isfinite(d)
    gap, d = gap[ok], d[ok]
    # unit-variance size so Sharpe is scale-free
    z = gap / (gap.std() + 1e-18)
    pnl = z * d
    sh = float(pnl.mean() / pnl.std()) if float(pnl.std()) > 0 else float("nan")
    corr = float(np.corrcoef(gap, d)[0, 1]) if gap.size > 2 else float("nan")
    print(
        f"{os.path.basename(path)}  n={gap.size:,}  "
        f"corr(gap, d_long)={corr:+.4f}  "
        f"incr Sharpe/trade={sh:+.4f}  "
        f"mean gap={gap.mean():+.3e}  "
        f"P(blk2>a0)={(gap > 0).mean():.2%}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="results/spxw_pnl/parts")
    ap.add_argument("--a0", default="results/spxw_pnl/yhat_a0.parquet")
    ap.add_argument("--blk2", default="results/spxw_pnl/yhat_blk2.parquet")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.parts, "h*_sweep.parquet")))
    if not files:
        raise SystemExit("no sweep parts")
    print("size = pred_blk2 - pred_a0  (extra long straddle when exog raises E[RV])")
    for f in files:
        _one(f, a.a0, a.blk2)


if __name__ == "__main__":
    main()
