"""One-time history ingest: the gap between the vendor panel (ends 2024-04-30;
the Cboe feed 2024-02-12) and the first free-feed day.

Three file families, all shifted to the panel's bar-END naive-ET convention
before anything is joined:

* Databento ``GLBX.MDP3`` ``ohlcv-1m`` for ``ES.FUT`` (all months) and the
  volume-ranked continuous ``ES.v.0`` -- CSV (or DBN through the ``databento``
  package if installed).  Timestamps ``ts_event`` are UTC nanoseconds at the
  bar START.  The continuous symbol is what the moments are built from; the
  per-month file is kept for the roll check (the day the ``v.0`` symbol changes
  the returns across the roll are dropped, as a stitched series must).
* Yahoo Finance HOURLY bars of ^VIX / ^VVIX / ^VIX3M (``data/free_feed/
  <col>_1h_yahoo.parquet``, snapshotted 2026-09-23; Yahoo keeps 730 days) plus
  the DAILY files as the fallback.  This is the path of record for the Cboe
  gap 2024-02-13 -> today: no purchase is needed (README, "History and
  refits").  Convention, measured on the vendor overlap 2023-12-07 ..
  2024-02-12: the vendor's value at stamp T is the print standing just
  BEFORE T, and the Yahoo hourly bar that ENDS at T closes at exactly that
  print (log error MAD 0.0000, p95 0.0000 on 585 / 270 / 270 stamps); the
  bar's OPEN at T is a different print (MAD 0.3 %, p95 1.6 %).  So a bar
  starting at ``s`` with close ``c`` is the panel's value at ``s + 1 h``
  (VIX bars start on the hour -> the :00 stamps; VVIX / VIX3M bars start
  on the half hour from 09:30 -> the :30 stamps, 15:30 included).  The other
  half-hour stamps CARRY the last print of the same calendar day (error MAD
  0.4-0.7 %, p95 2-3 % on the overlap; the vendor-panel study puts the cost of
  a 30-minute-stale VIX at R^2 0.001).  Stamps before the day's first print,
  and whole days without hourly bars, take the PREVIOUS session's daily
  close -- usable only from the next calendar day, never the same day.
  Every row records its provenance (``<col>_source`` in the standalone file:
  ``hourly`` / ``hourly_carry`` / ``daily_carry``).
* FirstRate Data 1-minute CSVs for VIX / VVIX / VIX3M (and SPX): ``datetime,
  open,high,low,close[,volume]`` in US/Eastern at the bar START.  Kept as the
  optional cross-check; not needed.

Everything lands in the StateStore with a source tag; nothing overwrites the
vendor parquets in data/ (``forecast._extend`` adds only the stamps the
vendor lacks).  ``forecast.build_ext_root`` stitches the vendor panel and
these rows when it runs the arm.

    python -m live.close_signal.ingest cboe-yahoo            # the gap, from the snapshot
    python -m live.close_signal.ingest gate-yahoo            # the overlap gate table
    python -m live.close_signal.ingest es-databento ES.csv   # the ES gap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from live.close_signal.common import CBOE_COLS, ET, session_grid
from live.close_signal.features import cboe_stamp_values, thirty_minute_moments
from live.close_signal.state import StateStore

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = Path(__file__).resolve().parent / "state"
FREE_FEED_DIR = REPO / "data" / "free_feed"
VENDOR_CBOE = REPO / "data" / "vix_and_voldemand.parquet"
#: The first stamp the vendor's Cboe feed does not carry.
GAP_START = pd.Timestamp("2024-02-13")
#: The vendor overlap the hourly convention was measured on.
OVERLAP_START = pd.Timestamp("2023-12-07")
OVERLAP_END = pd.Timestamp("2024-02-12")
YAHOO_HOURLY_SOURCE = "yahoo_cboe_hourly"
HOUR = pd.Timedelta(hours=1)
SOURCE_TAGS: tuple[str, ...] = ("hourly", "hourly_carry", "daily_carry")


def yahoo_hourly_paths(feed_dir: Path = FREE_FEED_DIR) -> dict[str, Path]:
    return {col: Path(feed_dir) / f"{col}_1h_yahoo.parquet" for col in CBOE_COLS}


def yahoo_daily_paths(feed_dir: Path = FREE_FEED_DIR) -> dict[str, Path]:
    return {col: Path(feed_dir) / f"{col}_1d_yahoo.parquet" for col in CBOE_COLS}


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


def read_yahoo_hourly(path: Path) -> pd.DataFrame:
    """A ``data/free_feed/<col>_1h_yahoo.parquet`` snapshot -> hourly bars indexed by naive-ET bar START.

    Columns ``t`` (UTC), ``open``, ``high``, ``low``, ``close`` (``volume`` is
    meaningless for an index and dropped).
    """
    df = pd.read_parquet(path)
    t = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    out = df.assign(ts_start=t).set_index("ts_start")[["open", "high", "low", "close"]]
    out = out.astype(float)
    return out[~out.index.duplicated(keep="last")].sort_index()


def read_yahoo_daily(path: Path) -> pd.Series:
    """A ``data/free_feed/<col>_1d_yahoo.parquet`` snapshot -> daily close by session date."""
    df = pd.read_parquet(path)
    t = pd.to_datetime(df["t"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(ET).dt.tz_localize(None)
    s = pd.Series(df["close"].astype(float).to_numpy(), index=t.dt.normalize())
    return s[~s.index.duplicated(keep="last")].sort_index()


def _previous_daily_close(
    daily: pd.Series | None, days: pd.DatetimeIndex
) -> np.ndarray:
    """The latest daily close DATED BEFORE each day (never the same day: no look-ahead)."""
    out = np.full(len(days), np.nan)
    if daily is None or daily.empty:
        return out
    d = daily.dropna().sort_index()
    pos = d.index.searchsorted(days, side="left") - 1
    ok = pos >= 0
    out[ok] = d.to_numpy()[pos[ok]]
    return out


def cboe_rows_from_yahoo(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Panel rows (``endbartime`` + CBOE_COLS + ``<col>_source``) on the 30-minute grid.

    For each column: the hourly bar starting at ``s`` puts its CLOSE at the
    stamp ``s + 1 h`` (``hourly``); the other half-hour stamps of the same
    calendar day carry the last such print forward (``hourly_carry``); stamps
    before the day's first print, and days with no hourly bar at all, take the
    previous session's daily close (``daily_carry``) -- a daily close is usable
    only from the next calendar day.  Stamps none of these reach stay NaN with
    an empty source.
    """
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    if hi < lo:
        raise ValueError("end before start")
    days = pd.date_range(lo, hi, freq="D")
    stamps = pd.DatetimeIndex(
        np.concatenate([session_grid(d).to_numpy() for d in days])
    )
    day_of = stamps.normalize()
    out = pd.DataFrame({"endbartime": stamps})
    for col in CBOE_COLS:
        exact = pd.Series(np.nan, index=stamps, dtype=float)
        h = hourly.get(col)
        if h is not None and not h.empty:
            ends = pd.DatetimeIndex(h.index) + HOUR
            if not ((ends.minute == 0) | (ends.minute == 30)).all():
                raise ValueError(f"{col}: hourly bars do not end on the half-hour grid")
            closes = pd.Series(h["close"].astype(float).to_numpy(), index=ends)
            closes = closes[~closes.index.duplicated(keep="last")].sort_index()
            exact = closes.reindex(stamps)
        carried = exact.groupby(day_of).ffill()
        prev_daily = pd.Series(
            _previous_daily_close(daily.get(col), day_of), index=stamps
        )
        vals = carried.where(carried.notna(), prev_daily)
        src = np.where(
            exact.notna(),
            "hourly",
            np.where(
                carried.notna(),
                "hourly_carry",
                np.where(prev_daily.notna(), "daily_carry", ""),
            ),
        )
        out[col] = vals.to_numpy(float)
        out[f"{col}_source"] = src
    return out


def yahoo_overlap_gate(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.Series],
    vendor_path: Path = VENDOR_CBOE,
    start: pd.Timestamp = OVERLAP_START,
    end: pd.Timestamp = OVERLAP_END,
) -> pd.DataFrame:
    """Log errors of the Yahoo-built values vs the vendor's prints, per column and source tag.

    The gate the convention rests on: ``hourly`` rows must match the vendor to
    numerical precision; ``hourly_carry`` rows carry the half-hour staleness.
    """
    rows = cboe_rows_from_yahoo(hourly, daily, start, end).set_index("endbartime")
    v = pd.read_parquet(vendor_path, columns=["endbartime", *CBOE_COLS])
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    v = v.set_index("endbartime")
    v = v[
        (v.index >= pd.Timestamp(start))
        & (v.index <= pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=30))
    ]
    recs = []
    for col in CBOE_COLS:
        j = pd.concat(
            [rows[[col, f"{col}_source"]], v[col].rename("vendor")],
            axis=1,
            join="inner",
        )
        j = j[(j[col] > 0) & (j["vendor"] > 0)]
        err = np.log(j[col] / j["vendor"])
        for tag in SOURCE_TAGS:
            e = err[j[f"{col}_source"] == tag]
            if len(e) == 0:
                continue
            recs.append(
                {
                    "column": col,
                    "source": tag,
                    "n": int(len(e)),
                    "log_err_median": float(e.median()),
                    "log_err_mad": float(e.abs().median()),
                    "log_err_p95": float(e.abs().quantile(0.95)),
                    "log_err_max": float(e.abs().max()),
                }
            )
    return pd.DataFrame(recs)


def ingest_cboe_yahoo(
    store: StateStore,
    hourly_paths: dict[str, Path] | None = None,
    daily_paths: dict[str, Path] | None = None,
    start: pd.Timestamp = GAP_START,
    end: pd.Timestamp | None = None,
    out_path: Path | None = None,
) -> dict[str, object]:
    """Yahoo hourly + daily snapshots -> the Cboe columns on the gap, into the store.

    Writes the standalone ``cboe_gap_yahoo.parquet`` (with the per-column source
    tags) beside the store and appends the three columns to the panel under
    source ``yahoo_cboe_hourly``.  Returns the row count and the tag tallies.
    """
    hp = hourly_paths or yahoo_hourly_paths()
    dp = daily_paths or yahoo_daily_paths()
    hourly = {c: read_yahoo_hourly(p) for c, p in hp.items() if Path(p).exists()}
    daily = {c: read_yahoo_daily(p) for c, p in dp.items() if Path(p).exists()}
    if not hourly:
        raise FileNotFoundError("no Yahoo hourly snapshots found")
    last = max(h.index.max() for h in hourly.values()) + HOUR
    stop = pd.Timestamp(end) if end is not None else last
    rows = cboe_rows_from_yahoo(hourly, daily, start, stop)
    rows = rows[rows[list(CBOE_COLS)].notna().any(axis=1)].reset_index(drop=True)
    target = (
        Path(out_path)
        if out_path is not None
        else store.root / "cboe_gap_yahoo.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(target, index=False)
    n = store.append_panel(rows[["endbartime", *CBOE_COLS]], source=YAHOO_HOURLY_SOURCE)
    tallies = {
        f"{col}:{tag}": int((rows[f"{col}_source"] == tag).sum())
        for col in CBOE_COLS
        for tag in SOURCE_TAGS
    }
    return {
        "rows": int(n),
        "first": str(rows["endbartime"].min()),
        "last": str(rows["endbartime"].max()),
        "file": str(target),
        **tallies,
    }


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
    """FirstRate VIX / VVIX / VIX3M 1-minute CSVs -> the prints at every panel stamp (optional cross-check)."""
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("cboe-yahoo", help="the Cboe gap from the Yahoo snapshots")
    p1.add_argument("--start", default=str(GAP_START.date()))
    p1.add_argument("--end", default=None)
    p1.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    p1.add_argument("--feed-dir", default=str(FREE_FEED_DIR))
    p2 = sub.add_parser("gate-yahoo", help="the overlap gate vs the vendor prints")
    p2.add_argument("--start", default=str(OVERLAP_START.date()))
    p2.add_argument("--end", default=str(OVERLAP_END.date()))
    p2.add_argument("--feed-dir", default=str(FREE_FEED_DIR))
    p3 = sub.add_parser(
        "es-databento", help="the ES gap from a Databento ohlcv-1m file"
    )
    p3.add_argument("path")
    p3.add_argument("--symbol", default="ES.v.0")
    p3.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    a = ap.parse_args(argv)
    if a.cmd == "gate-yahoo":
        fd = Path(a.feed_dir)
        hourly = {
            c: read_yahoo_hourly(p)
            for c, p in yahoo_hourly_paths(fd).items()
            if p.exists()
        }
        daily = {
            c: read_yahoo_daily(p)
            for c, p in yahoo_daily_paths(fd).items()
            if p.exists()
        }
        table = yahoo_overlap_gate(
            hourly, daily, start=pd.Timestamp(a.start), end=pd.Timestamp(a.end)
        )
        pd.set_option("display.width", 200)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        return 0
    store = StateStore(Path(a.state_dir))
    if a.cmd == "cboe-yahoo":
        fd = Path(a.feed_dir)
        info = ingest_cboe_yahoo(
            store,
            yahoo_hourly_paths(fd),
            yahoo_daily_paths(fd),
            start=pd.Timestamp(a.start),
            end=pd.Timestamp(a.end) if a.end else None,
        )
        for k, v in info.items():
            print(f"{k}: {v}")
        return 0
    n = ingest_es(store, Path(a.path), continuous_symbol=a.symbol)
    print(f"databento_es rows appended: {n}")
    return 0


__all__ = [
    "GAP_START",
    "YAHOO_HOURLY_SOURCE",
    "cboe_rows_from_yahoo",
    "drop_roll_days",
    "ingest_cboe",
    "ingest_cboe_yahoo",
    "ingest_es",
    "read_databento_ohlcv_1m",
    "read_firstrate_1m",
    "read_yahoo_daily",
    "read_yahoo_hourly",
    "yahoo_daily_paths",
    "yahoo_hourly_paths",
    "yahoo_overlap_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
