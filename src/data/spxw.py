"""SPXW weekly chain: load + ATM pick + horizon exit.

Rule (professor): do not drop 0-delta rows. Neighboring bars often print
delta=0 (or 1) with a live mid; those are the exit marks. Greeks may be
missing or zero; the join key is (expiration, strike, cp, timestamp).

Entry still requires a finite mid on both legs. Exit uses the mid if it
exists, otherwise intrinsic from the bar's underlying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR_HOURS = 0.5  # 30-minute grid


def load_chain(
    path: str = "data/spxw_chain.parquet",
    start: str | None = None,
    end: str | None = None,
    halo_days: int = 0,
) -> pd.DataFrame:
    """Chain slice. No greek filter, no mid filter. ``end`` is exclusive.

    ``halo_days`` extends the read past ``end`` so t+H exits still join
    (H=16 bars can land after a weekend). Entry filtering stays with the caller.
    """
    filters = []
    if start is not None:
        filters.append(("timestamp", ">=", pd.Timestamp(start, tz="UTC")))
    if end is not None:
        read_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=halo_days)
        filters.append(("timestamp", "<", read_end))
    df = pd.read_parquet(path, filters=filters or None)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["cp"] = df["cp"].astype(str).str.upper().str[0]
    if "spread" not in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    return df


def pick_atm_straddle(snap: pd.DataFrame, spot: float) -> pd.DataFrame | None:
    """ATM C+P at this timestamp. Entry filter is mid, not delta.

    ``snap`` is one timestamp's chain (0-delta rows included).
    """
    if snap.empty or not np.isfinite(spot):
        return None
    live = snap[np.isfinite(snap["mid"])]
    if live.empty:
        return None
    both = live.groupby("strike")["cp"].nunique()
    strikes = both[both == 2].index.to_numpy()
    if strikes.size == 0:
        return None
    k = float(strikes[np.argmin(np.abs(strikes.astype(float) - spot))])
    legs = live[live["strike"] == k]
    if set(legs["cp"]) != {"C", "P"}:
        return None
    return legs


def expiry_close_utc(expiration) -> pd.Timestamp:
    """SPXW is PM-settled: cash at the official 16:00 ET close on expiration day."""
    exp = pd.Timestamp(expiration)
    if exp.tzinfo is None:
        exp = exp.tz_localize("America/New_York")
    else:
        exp = exp.tz_convert("America/New_York")
    close = exp.normalize() + pd.Timedelta(hours=16)
    return close.tz_convert("UTC")


def settle_straddle(spot: float, strike: float) -> float:
    """Cash-settled ATM straddle payoff: call + put intrinsic = |S-K|."""
    if not (np.isfinite(spot) and np.isfinite(strike)):
        return float("nan")
    return float(abs(spot - strike))


def last_spot_on_expiry(chain: pd.DataFrame, expiration, fallback: float) -> float:
    """Last printed underlying on the expiration session; else fallback."""
    close = expiry_close_utc(expiration)
    start = close.normalize()
    m = (chain["timestamp"] >= start) & (chain["timestamp"] <= close)
    s = chain.loc[m, "underlying_price"].dropna()
    if s.empty:
        return float(fallback)
    return float(s.iloc[-1])


def exit_or_settle(
    chain: pd.DataFrame,
    chain_idx: pd.DataFrame,
    expiration,
    strike: float,
    t1: pd.Timestamp,
    entry_spot: float,
) -> tuple[float, str]:
    """Mid if the same contract still quotes (0-delta ok). Else settle |S-K|
    when t1 is at/after the 16:00 ET expiration close. Else missing."""
    xC = exit_mid(chain_idx, expiration, strike, "C", t1)
    xP = exit_mid(chain_idx, expiration, strike, "P", t1)
    if np.isfinite(xC) and np.isfinite(xP):
        return xC + xP, "mid"
    if pd.Timestamp(t1) >= expiry_close_utc(expiration):
        S = last_spot_on_expiry(chain, expiration, entry_spot)
        return settle_straddle(S, strike), "settle"
    return float("nan"), "missing"


def exit_mid(chain_idx: pd.DataFrame, expiration, strike: float, cp: str, ts) -> float:
    """Look up the same contract at ``ts``. 0-delta is fine. NaN if absent."""
    try:
        row = chain_idx.loc[
            (pd.Timestamp(expiration), float(strike), cp, pd.Timestamp(ts))
        ]
    except KeyError:
        return float("nan")
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    mid = row.get("mid", np.nan)
    if np.isfinite(mid):
        return float(mid)
    S = row.get("underlying_price", np.nan)
    if not np.isfinite(S):
        return float("nan")
    return float(max(S - strike, 0.0) if cp == "C" else max(strike - S, 0.0))


def index_chain(chain: pd.DataFrame) -> pd.DataFrame:
    return chain.set_index(["expiration", "strike", "cp", "timestamp"]).sort_index()
