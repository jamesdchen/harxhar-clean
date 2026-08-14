"""ATM closest C+P straddle PnL on the SPXW tape.

Always-long and forecast-signed. Exit is the same contract's mid (0-delta
kept). If the quote is gone and the contract has hit the 16:00 ET PM
settle, payoff is |S-K|.

The first tape runs were always-long on purpose (join/plumbing). Sign uses
a causal HAR on core_stats bar RV (sumret2), not a cluster ridge harvest —
those npz are not on this box. pred_H = H * (rv1+rv8+rv40)/3 vs iv^2 * tau_H.

Usage:
  python experiments/spxw_horizon_pnl.py --h-from 1 --h-to 5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.spxw import (
    exit_or_settle,
    index_chain,
    load_chain,
    pick_atm_straddle,
)

TIERS = {"mid": 0.0, "base": 0.25, "conservative": 0.5}
OUT = "results/spxw_pnl"
BARS_PER_YEAR = 252.0 * 48.0


def _har_forecast(panel_path: str, stamps: np.ndarray) -> pd.Series:
    """Causal equal-weight HAR on bar RV, aligned to SPXW stamps (UTC)."""
    raw = pd.read_parquet(panel_path, columns=["endbartime", "sumret2"])
    t = pd.to_datetime(raw["endbartime"])
    t = t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    raw = raw.assign(
        t=t.dt.tz_convert("UTC"), rv=pd.to_numeric(raw["sumret2"], errors="coerce")
    )
    raw = raw.dropna(subset=["t"]).sort_values("t")
    rv = raw.set_index("t")["rv"]
    har = (
        rv + rv.rolling(8, min_periods=1).mean() + rv.rolling(40, min_periods=8).mean()
    ) / 3.0
    har = har.shift(1)  # known at bar open / before this stamp's RV
    idx = pd.DatetimeIndex(stamps)
    return har.reindex(idx, method="ffill")


def _iv_of_legs(legs: pd.DataFrame) -> float:
    v = pd.to_numeric(legs["impl_volatility"], errors="coerce").to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(v.mean()) if v.size else float("nan")


def _atm_book(chain: pd.DataFrame) -> list[dict]:
    stamps = np.sort(chain["timestamp"].unique())
    by_ts = {t: g for t, g in chain.groupby("timestamp", sort=False)}
    book = []
    for t0 in stamps:
        snap = by_ts[t0]
        spot = snap["underlying_price"].dropna()
        if spot.empty:
            continue
        S = float(spot.iloc[-1])
        legs = pick_atm_straddle(snap, S)
        if legs is None:
            continue
        mids, spreads = {}, {}
        for _, r in legs.iterrows():
            mids[r["cp"]] = float(r["mid"])
            spreads[r["cp"]] = float(r["spread"]) if np.isfinite(r["spread"]) else 0.0
        if "C" not in mids or "P" not in mids:
            continue
        book.append(
            {
                "t0": t0,
                "expiration": legs["expiration"].iloc[0],
                "strike": float(legs["strike"].iloc[0]),
                "spot": S,
                "entry": mids["C"] + mids["P"],
                "spread": spreads["C"] + spreads["P"],
                "iv": _iv_of_legs(legs),
            }
        )
    return book


def _pnl_h(
    book: list[dict], stamps: np.ndarray, chain, idx, har: pd.Series, h: int
) -> pd.DataFrame:
    pos = {t: i for i, t in enumerate(stamps)}
    rows = []
    for b in book:
        i = pos.get(b["t0"])
        if i is None or i + h >= len(stamps):
            t1 = pd.Timestamp(b["t0"]) + pd.Timedelta(minutes=30 * h)
        else:
            t1 = stamps[i + h]
        exit_px, how = exit_or_settle(
            chain, idx, b["expiration"], b["strike"], t1, b["spot"]
        )
        if not np.isfinite(exit_px):
            continue
        d_long = exit_px - b["entry"]
        pred = (
            float(har.loc[b["t0"]])
            if b["t0"] in har.index and np.isfinite(har.loc[b["t0"]])
            else float("nan")
        )
        pred_h = pred * h if np.isfinite(pred) else float("nan")
        # Mid-implied ATM vol (vendor new_implied_vol is ~100x too small).
        exp = pd.Timestamp(b["expiration"])
        close = (
            exp.tz_localize("America/New_York") + pd.Timedelta(hours=16)
        ).tz_convert("UTC")
        tau_cal = (close - pd.Timestamp(b["t0"])).total_seconds() / (365.25 * 24 * 3600)
        if tau_cal > 0 and b["spot"] > 0 and b["entry"] > 0:
            iv_mid = b["entry"] / (b["spot"] * np.sqrt(tau_cal) * np.sqrt(2.0 / np.pi))
        else:
            iv_mid = (
                b["iv"] * 100.0 if np.isfinite(b["iv"]) and b["iv"] < 0.05 else b["iv"]
            )
        tau = h / BARS_PER_YEAR
        impl = (iv_mid**2) * tau if np.isfinite(iv_mid) else float("nan")
        sgn = (
            float(np.sign(pred_h - impl))
            if np.isfinite(pred_h) and np.isfinite(impl)
            else 0.0
        )
        rows.append(
            {
                **b,
                "t1": t1,
                "h": h,
                "exit": exit_px,
                "how": how,
                "d_long": d_long,
                "pred_h": pred_h,
                "impl_h": impl,
                "sign": sgn,
                "d_signed": sgn * d_long,
            }
        )
    return pd.DataFrame(rows)


def _report(pnl: pd.DataFrame, h: int) -> None:
    if pnl.empty:
        print(f"h={h}: 0 trades", flush=True)
        return
    n_set = int((pnl["how"] == "settle").sum()) if "how" in pnl.columns else 0
    for kind, col in (("long", "d_long"), ("sign", "d_signed")):
        x = pnl[col].to_numpy(float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            print(f"h={h} {kind}: empty", flush=True)
            continue
        sh = float(x.mean() / x.std()) if float(x.std()) > 0 else float("nan")
        print(
            f"h={h} {kind}: n={x.size:,}  settle={n_set}  mean={x.mean():+.4f}  "
            f"sharpe/trade={sh:+.3f}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-from", type=int, default=1)
    ap.add_argument("--h-to", type=int, default=13)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--chain", default="data/spxw_chain.parquet")
    ap.add_argument("--panel", default="data/core_stats.parquet")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    hs = [a.horizon] if a.horizon is not None else list(range(a.h_from, a.h_to + 1))

    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    chain = load_chain(a.chain)
    stamps = np.sort(chain["timestamp"].unique())
    print(f"chain stamps={len(stamps):,} rows={len(chain):,}", flush=True)
    har = _har_forecast(a.panel, stamps)
    book = _atm_book(chain)
    print(f"atm entries={len(book):,}", flush=True)
    idx = index_chain(chain)
    for h in hs:
        pnl = _pnl_h(book, stamps, chain, idx, har, h)
        path = os.path.join(a.out, "parts", f"h{h}_sweep.parquet")
        pnl.to_parquet(path, index=False)
        _report(pnl, h)


if __name__ == "__main__":
    main()
