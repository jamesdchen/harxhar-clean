"""One-time history ingest: the gap between the vendor panel (ends 2024-04-30;
the Cboe feed 2024-02-12) and the first free-feed day.

Two file families, both shifted to the panel's bar-END naive-ET convention
before anything is joined:

* Databento ``GLBX.MDP3`` ``ohlcv-1m`` for ``ES.FUT`` (all months) and the
  volume-ranked continuous ``ES.v.0`` -- CSV (or DBN through the ``databento``
  package if installed).  Timestamps ``ts_event`` are UTC nanoseconds at the
  bar START.  The continuous symbol is what the moments are built from; the
  per-month file is kept for the roll check (the day the ``v.0`` symbol changes
  the returns across the roll are dropped, as a stitched series must).
* FirstRate Data 1-minute CSVs for VIX / VVIX / VIX3M (and SPX as the optional
  fallback): ``datetime,open,high,low,close[,volume]`` in US/Eastern at the
  bar START.

Everything lands in the StateStore with a source tag; nothing overwrites the
vendor parquets in data/.  ``forecast.build_ext_root`` stitches the vendor
panel and these rows when it runs the arm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from live.close_signal.common import CBOE_COLS, ET, session_grid
from live.close_signal.features import cboe_stamp_values, thirty_minute_moments
from live.close_signal.state import StateStore


def read_databento_ohlcv_1m(path: Path, symbol: str | None = None) -> pd.DataFrame:
    """Databento ohlcv-1m -> 1-minute bars indexed by naive-ET bar START.

    CSV columns: ``ts_event`` (UTC ns), ``open``, ``high``, ``low``, ``close``,
    ``volume``, ``symbol`` (plus rtype/publisher_id/instrument_id).  A ``.dbn``
    or ``.dbn.zst`` file is read with the ``databento`` package.  ``symbol``
    keeps one instrument (``ES.v.0`` for the continuous series).
    """
    p = Path(path)
    if p.suffix in (".dbn", ".zst"):
        try:
            import databento as db  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "reading DBN needs the databento package; export CSV instead"
            ) from e
        df = db.DBNStore.from_file(str(p)).to_df().reset_index()
    else:
        df = pd.read_csv(p)
    if "ts_event" not in df.columns:
        raise ValueError(f"{p}: no ts_event column")
    ts = pd.to_datetime(df["ts_event"], utc=True)
    df = df.assign(ts_start=ts.dt.tz_convert(ET).dt.tz_localize(None))
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"].astype(str) == symbol]
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            raise ValueError(f"{p}: column {c} missing")
    out = df.set_index("ts_start")[["open", "high", "low", "close", "volume"]].astype(
        float
    )
    return out[~out.index.duplicated(keep="last")].sort_index()


def read_firstrate_1m(path: Path) -> pd.DataFrame:
    """FirstRate Data 1-minute CSV (US/Eastern bar START) -> bars indexed by naive ET."""
    p = Path(path)
    df = pd.read_csv(p)
    df.columns = [str(c).lower() for c in df.columns]
    tcol = next((c for c in df.columns if c in ("datetime", "timestamp", "date")), None)
    if tcol is None:
        # FirstRate ships headerless files too: datetime,open,high,low,close[,volume]
        df = pd.read_csv(p, header=None)
        names = ["datetime", "open", "high", "low", "close", "volume"][: df.shape[1]]
        df.columns = names
        tcol = "datetime"
    ts = pd.to_datetime(df[tcol])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(ET).dt.tz_localize(None)
    df = df.assign(ts_start=ts)
    if "volume" not in df.columns:
        df["volume"] = np.nan
    out = df.set_index("ts_start")[["open", "high", "low", "close", "volume"]].astype(
        float
    )
    return out[~out.index.duplicated(keep="last")].sort_index()


def drop_roll_days(bars: pd.DataFrame, symbol_series: pd.Series | None) -> pd.DataFrame:
    """Drop the minutes of the days on which the continuous symbol rolled.

    A stitched series has a price jump at the roll that is not a return; the
    panel's own vendor handled its ES series somehow we cannot see, so the
    honest choice is to drop those days' rows (they become NaN in the panel
    and take the availability indicator) and to record how many.
    """
    if symbol_series is None or symbol_series.empty:
        return bars
    sym = symbol_series.reindex(bars.index).ffill()
    roll = sym != sym.shift(1)
    roll.iloc[0] = False
    roll_days = pd.DatetimeIndex(bars.index[roll.to_numpy()]).normalize().unique()
    keep = ~pd.DatetimeIndex(bars.index).normalize().isin(roll_days)
    return bars[keep]


def ingest_es(
    store: StateStore, path: Path, *, continuous_symbol: str = "ES.v.0"
) -> int:
    """Databento ES 1-minute -> the panel's ES moments, appended as ``databento_es``."""
    raw = read_databento_ohlcv_1m(path, symbol=None)
    df = pd.read_csv(path) if Path(path).suffix == ".csv" else None
    sym = None
    if df is not None and "symbol" in df.columns:
        ts = (
            pd.to_datetime(df["ts_event"], utc=True)
            .dt.tz_convert(ET)
            .dt.tz_localize(None)
        )
        s = pd.Series(df["symbol"].astype(str).to_numpy(), index=ts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        want = s[s == continuous_symbol]
        raw = raw.loc[raw.index.isin(want.index)]
        sym = s
    bars = drop_roll_days(raw, sym)
    mom = thirty_minute_moments(bars)
    return store.append_panel(mom, source="databento_es")


def ingest_cboe(store: StateStore, paths: dict[str, Path]) -> int:
    """FirstRate VIX / VVIX / VIX3M 1-minute CSVs -> the prints at every panel stamp."""
    prints = {col: read_firstrate_1m(paths[col]) for col in CBOE_COLS if col in paths}
    if not prints:
        raise ValueError("no Cboe files given")
    lo = min(fr.index.min() for fr in prints.values()).normalize()
    hi = max(fr.index.max() for fr in prints.values()).normalize()
    days = pd.date_range(lo, hi, freq="D")
    stamps = pd.DatetimeIndex(
        np.concatenate([session_grid(d).to_numpy() for d in days])
    )
    vals = cboe_stamp_values(prints, stamps)
    vals = vals.dropna(subset=list(prints), how="all")
    return store.append_panel(vals, source="firstrate")


__all__ = [
    "drop_roll_days",
    "ingest_cboe",
    "ingest_es",
    "read_databento_ohlcv_1m",
    "read_firstrate_1m",
]
