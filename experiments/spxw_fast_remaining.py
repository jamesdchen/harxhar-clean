"""Vectorized remaining-horizon ATM PnL. Avoids per-trade 8M-row settle scans."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.spxw import expiry_close_utc, load_chain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "results", "spxw_pnl", "parts")


def _book() -> pd.DataFrame:
    src = os.path.join(PARTS, "h1_sweep.parquet")
    b = pd.read_parquet(src)
    keep = ["t0", "expiration", "strike", "spot", "entry", "spread", "iv"]
    b = b[keep].copy()
    b["t0"] = pd.to_datetime(b["t0"], utc=True)
    b["expiration"] = pd.to_datetime(b["expiration"])
    return b


def _last_spot(chain: pd.DataFrame) -> pd.DataFrame:
    exps = pd.Series(pd.to_datetime(chain["expiration"].unique()))
    rows = []
    for e in exps:
        close = expiry_close_utc(e)
        rows.append({"expiration": pd.Timestamp(e), "start": close.normalize(), "close": close})
    win = pd.DataFrame(rows)
    spots = (
        chain.dropna(subset=["underlying_price"])[["timestamp", "underlying_price"]]
        .sort_values("timestamp")
        .rename(columns={"timestamp": "close", "underlying_price": "last_spot"})
    )
    spots["close"] = pd.to_datetime(spots["close"], utc=True)
    win = win.sort_values("close")
    win["close"] = pd.to_datetime(win["close"], utc=True)
    m = pd.merge_asof(win, spots, on="close", direction="backward")
    m = m[m["close"] >= m["start"]]
    return m[["expiration", "last_spot"]]


def _t1(book: pd.DataFrame, stamps: np.ndarray, h: int) -> pd.Series:
    pos = pd.Series(np.arange(len(stamps), dtype=np.int64), index=pd.DatetimeIndex(stamps))
    i = book["t0"].map(pos)
    t1 = pd.Series(pd.NaT, index=book.index, dtype="datetime64[ns, UTC]")
    ok = i.notna() & ((i + h) < len(stamps))
    t1.loc[ok] = stamps[(i.loc[ok] + h).astype(int).to_numpy()]
    t1.loc[~ok] = book.loc[~ok, "t0"] + pd.Timedelta(minutes=30 * h)
    return pd.to_datetime(t1, utc=True)


def _leg_exit(book: pd.DataFrame, legs: pd.DataFrame, cp: str) -> pd.Series:
    sub = legs[legs["cp"] == cp][["expiration", "strike", "timestamp", "mid", "underlying_price"]]
    sub = sub.rename(columns={"timestamp": "t1", "mid": f"mid_{cp}", "underlying_price": f"S_{cp}"})
    m = book.merge(sub, on=["expiration", "strike", "t1"], how="left")
    mid = m[f"mid_{cp}"].to_numpy(float)
    S = m[f"S_{cp}"].to_numpy(float)
    K = m["strike"].to_numpy(float)
    out = mid.copy()
    miss = ~np.isfinite(out)
    if cp == "C":
        out[miss] = np.where(np.isfinite(S[miss]), np.maximum(S[miss] - K[miss], 0.0), np.nan)
    else:
        out[miss] = np.where(np.isfinite(S[miss]), np.maximum(K[miss] - S[miss], 0.0), np.nan)
    return pd.Series(out, index=book.index)


def _pnl_h(book: pd.DataFrame, stamps: np.ndarray, legs: pd.DataFrame, last_spot: pd.DataFrame, h: int) -> pd.DataFrame:
    b = book.copy()
    b["t1"] = _t1(b, stamps, h)
    b["expiration"] = pd.to_datetime(b["expiration"])
    xC = _leg_exit(b, legs, "C")
    xP = _leg_exit(b, legs, "P")
    close = b["expiration"].map(lambda e: expiry_close_utc(e))
    close = pd.to_datetime(close, utc=True)
    have_mid = np.isfinite(xC.to_numpy(float)) & np.isfinite(xP.to_numpy(float))
    past = b["t1"] >= close
    b = b.merge(last_spot, on="expiration", how="left")
    exit_px = np.full(len(b), np.nan)
    how = np.array(["missing"] * len(b), dtype=object)
    exit_px[have_mid] = xC.to_numpy(float)[have_mid] + xP.to_numpy(float)[have_mid]
    how[have_mid] = "mid"
    settle = (~have_mid) & past.to_numpy()
    S = b["last_spot"].to_numpy(float)
    fb = b["spot"].to_numpy(float)
    S = np.where(np.isfinite(S), S, fb)
    K = b["strike"].to_numpy(float)
    exit_px[settle] = np.abs(S[settle] - K[settle])
    how[settle] = "settle"
    b["h"] = h
    b["exit"] = exit_px
    b["how"] = how
    b["d_long"] = b["exit"] - b["entry"]
    b["pred_h"] = np.nan
    b["impl_h"] = np.nan
    b["sign"] = np.nan
    b["d_signed"] = np.nan
    keep = np.isfinite(exit_px)
    cols = [
        "t0",
        "expiration",
        "strike",
        "spot",
        "entry",
        "spread",
        "iv",
        "t1",
        "h",
        "exit",
        "how",
        "d_long",
        "pred_h",
        "impl_h",
        "sign",
        "d_signed",
    ]
    return b.loc[keep, cols].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="9,12,13")
    ap.add_argument("--chain", default=os.path.join(ROOT, "data", "spxw_chain.parquet"))
    a = ap.parse_args()
    hs = [int(x) for x in a.horizons.split(",") if x.strip()]
    print("load book + chain", flush=True)
    book = _book()
    chain = load_chain(a.chain)
    stamps = np.sort(chain["timestamp"].unique())
    print(f"book={len(book):,} stamps={len(stamps):,} rows={len(chain):,}", flush=True)
    last_spot = _last_spot(chain)
    print(f"expiries with last_spot={len(last_spot):,}", flush=True)
    legs = chain[["expiration", "strike", "cp", "timestamp", "mid", "underlying_price"]].copy()
    legs["expiration"] = pd.to_datetime(legs["expiration"])
    legs["timestamp"] = pd.to_datetime(legs["timestamp"], utc=True)
    os.makedirs(PARTS, exist_ok=True)
    for h in hs:
        pnl = _pnl_h(book, stamps, legs, last_spot, h)
        path = os.path.join(PARTS, f"h{h}_sweep.parquet")
        pnl.to_parquet(path, index=False)
        nset = int((pnl["how"] == "settle").sum())
        x = pnl["d_long"].to_numpy(float)
        x = x[np.isfinite(x)]
        sh = float(x.mean() / x.std()) if x.size > 2 and float(x.std()) > 0 else float("nan")
        print(
            f"h={h} long: n={len(pnl):,}  settle={nset}  mean={x.mean():+.4f}  sharpe/trade={sh:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
