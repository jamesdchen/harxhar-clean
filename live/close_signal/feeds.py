"""Free intraday feeds (Yahoo Finance via yfinance), with their delays measured.

Every adapter returns 1-minute bars in naive US/Eastern with the bar START as
the index, plus the wall-clock fetch time and the measured delay, so the caller
can decide whether the bar it needs has actually arrived.  Nothing here fills
a gap silently: an empty or malformed download raises ``FeedError``.

Yahoo's published delays (2026-09-23): Cboe indices 15 minutes, CME futures
10 minutes, the S&P 500 index real-time.  ``stamp_complete`` is the test the
run applies before it uses a bar as the panel's stamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from live.close_signal.common import (
    BAR,
    CBOE_SYMBOLS,
    ES_SYMBOL,
    ET,
    ONE_MINUTE,
    SPX_SYMBOL,
    YAHOO_NOMINAL_DELAY_MIN,
)


class FeedError(RuntimeError):
    """A feed did not deliver what the run needs.  Never swallowed."""


@dataclass(frozen=True)
class Bars1m:
    """One symbol's 1-minute bars, naive ET, indexed by bar START."""

    symbol: str
    frame: pd.DataFrame  # columns open, high, low, close, volume
    fetched_at: pd.Timestamp  # naive ET wall clock at fetch
    nominal_delay_min: int

    @property
    def last_bar_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index[-1])

    @property
    def delay_minutes(self) -> float:
        """Wall clock minus the end of the last bar delivered, in minutes."""
        end = self.last_bar_start + ONE_MINUTE
        return float((self.fetched_at - end) / ONE_MINUTE)

    def stamp_complete(self, stamp_end: pd.Timestamp) -> bool:
        """True when the panel bar ending at ``stamp_end`` is fully delivered."""
        return self.last_bar_start >= pd.Timestamp(stamp_end) - ONE_MINUTE

    def latency_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "fetched_at": self.fetched_at,
            "last_bar_start": self.last_bar_start,
            "delay_minutes": self.delay_minutes,
            "nominal_delay_minutes": self.nominal_delay_min,
            "n_bars": int(len(self.frame)),
        }


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET).tz_localize(None)


def _tidy(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance's frame (possibly MultiIndex columns) -> lower-case OHLCV, naive ET."""
    if raw is None or len(raw) == 0:
        raise FeedError(f"{symbol}: empty download")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # (field, ticker) or (ticker, field) depending on the yfinance version
        lvl = 0 if "Close" in df.columns.get_level_values(0) else 1
        df.columns = df.columns.get_level_values(lvl)
    df.columns = [str(c).lower() for c in df.columns]
    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise FeedError(f"{symbol}: columns missing {missing}")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise FeedError(
            f"{symbol}: naive timestamps from the feed; cannot place them in ET"
        )
    idx = idx.tz_convert(ET).tz_localize(None)
    out = df[need].astype(float)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    if out.empty:
        raise FeedError(f"{symbol}: no finite closes")
    return out


def fetch_1m(
    symbol: str,
    start_et: pd.Timestamp,
    end_et: pd.Timestamp,
    *,
    prepost: bool = True,
    downloader: Any | None = None,
) -> Bars1m:
    """1-minute bars for ``[start_et, end_et)``; ``downloader`` is injectable for tests.

    Yahoo serves at most 8 days of 1-minute bars per request and 30 days back.
    """
    if downloader is None:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise FeedError("yfinance is not installed") from e
        downloader = yf.download
    start = pd.Timestamp(start_et).tz_localize(ET)
    end = pd.Timestamp(end_et).tz_localize(ET)
    try:
        raw = downloader(
            symbol,
            start=start,
            end=end,
            interval="1m",
            prepost=prepost,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:  # noqa: BLE001 -- the feed's failure is the point
        raise FeedError(f"{symbol}: download failed: {type(e).__name__}: {e}") from e
    frame = _tidy(raw, symbol)
    return Bars1m(
        symbol=symbol,
        frame=frame,
        fetched_at=now_et(),
        nominal_delay_min=YAHOO_NOMINAL_DELAY_MIN.get(symbol, 0),
    )


def _window(session: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fetch window covering the session's 48 panel bars (the previous evening on)."""
    d = pd.Timestamp(session).normalize()
    return d - pd.Timedelta(hours=6), d + pd.Timedelta(days=1)


def es_minute_bars(session: pd.Timestamp, downloader: Any | None = None) -> Bars1m:
    """ES front-month continuous (Yahoo ``ES=F``), all sessions of the day."""
    lo, hi = _window(session)
    return fetch_1m(ES_SYMBOL, lo, hi, prepost=True, downloader=downloader)


def spx_minute_bars(session: pd.Timestamp, downloader: Any | None = None) -> Bars1m:
    """The S&P 500 index (Yahoo ``^GSPC``), regular hours; real-time on Yahoo."""
    lo, hi = _window(session)
    return fetch_1m(SPX_SYMBOL, lo, hi, prepost=False, downloader=downloader)


def cboe_prints(
    session: pd.Timestamp, downloader: Any | None = None
) -> dict[str, Bars1m]:
    """VIX, VVIX, VIX3M 1-minute prints keyed by the panel column name."""
    lo, hi = _window(session)
    return {
        col: fetch_1m(sym, lo, hi, prepost=True, downloader=downloader)
        for col, sym in CBOE_SYMBOLS.items()
    }


def history_download(
    symbol: str,
    start: Any = None,
    end: Any = None,
    interval: str = "1m",
    prepost: bool = False,
    auto_adjust: bool = False,
    **_: Any,
) -> pd.DataFrame:
    """``yf.download``'s own per-symbol call, without ``yf.download`` around it.

    ``yf.download`` resets and fills module-level result dicts (and swaps the
    shared session) on every call, so two calls from two threads overwrite
    each other's results.  Underneath it calls ``Ticker(symbol).history`` with
    these arguments -- what this does -- which is what its own worker threads
    run concurrently.  Same bars (checked 2026-09-24 on ES=F, ^GSPC, ^VIX,
    ^VVIX, ^VIX3M: identical frames after ``_tidy``).
    """
    import yfinance as yf

    return yf.Ticker(symbol).history(
        start=start,
        end=end,
        interval=interval,
        prepost=prepost,
        actions=False,
        auto_adjust=auto_adjust,
        back_adjust=False,
        repair=False,
        rounding=False,
        keepna=False,
        timeout=10,
        raise_errors=True,
    )


#: fetch_all's keys: the ES bars, the Cboe prints by panel column, the index.
ALL_FEEDS: tuple[str, ...] = ("es", *CBOE_SYMBOLS, "spx")


def fetch_all(
    session: pd.Timestamp, downloader: Any | None = None
) -> dict[str, Bars1m | FeedError]:
    """ES, the three Cboe prints and ^GSPC fetched at once, one thread each.

    Each value is the symbol's ``Bars1m`` or the ``FeedError`` it raised, so
    the caller decides which failures matter (``take``).  ``downloader``
    defaults to ``history_download`` (thread-safe; see there).
    """
    from concurrent.futures import ThreadPoolExecutor

    dl = downloader or history_download
    lo, hi = _window(session)
    jobs: dict[str, Any] = {
        "es": lambda: es_minute_bars(session, downloader=dl),
        "spx": lambda: spx_minute_bars(session, downloader=dl),
    }
    for col, sym in CBOE_SYMBOLS.items():
        jobs[col] = lambda sym=sym: fetch_1m(sym, lo, hi, prepost=True, downloader=dl)

    def run(k: str) -> Bars1m | FeedError:
        try:
            return jobs[k]()
        except FeedError as e:
            return e
        except Exception as e:  # noqa: BLE001 -- every feed failure is a FeedError
            return FeedError(f"{k}: {type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=len(ALL_FEEDS)) as ex:
        futs = {k: ex.submit(run, k) for k in ALL_FEEDS}
        return {k: futs[k].result() for k in ALL_FEEDS}


def take(got: dict[str, Bars1m | FeedError], key: str) -> Bars1m:
    """The ``Bars1m`` of ``fetch_all``'s ``key``, raising its FeedError."""
    v = got[key]
    if isinstance(v, BaseException):
        raise v
    return v


def last_complete_close(bars: Bars1m) -> float:
    """Close of the last 1-minute bar that had ended when the feed was read.

    Yahoo serves the minute in progress as the last row of a real-time feed.
    """
    f = bars.frame[bars.frame.index + ONE_MINUTE <= bars.fetched_at]
    if f.empty:
        raise FeedError(f"{bars.symbol}: no complete bar at {bars.fetched_at}")
    return float(f["close"].iloc[-1])


def latency_records(bars: dict[str, Bars1m]) -> pd.DataFrame:
    """One row per symbol: fetch time, last bar, measured and nominal delay."""
    return pd.DataFrame([b.latency_record() for b in bars.values()])


def wait_for_stamp(
    fetch: Any,
    stamp_end: pd.Timestamp,
    *,
    deadline: pd.Timestamp,
    poll_seconds: float = 30.0,
    sleeper: Any | None = None,
    clock: Any | None = None,
) -> Bars1m:
    """Re-fetch until the bar ending at ``stamp_end`` is complete or the deadline passes.

    ``fetch()`` returns a ``Bars1m``; ``sleeper``/``clock`` are injectable.
    Raises ``FeedError`` at the deadline with the last delay measured.
    """
    import time

    sleeper = sleeper or time.sleep
    clock = clock or now_et
    last: Bars1m | None = None
    while True:
        last = fetch()
        if last.stamp_complete(stamp_end):
            return last
        if clock() >= deadline:
            raise FeedError(
                f"{last.symbol}: the bar ending {stamp_end} never arrived before "
                f"{deadline} (last bar start {last.last_bar_start}, delay "
                f"{last.delay_minutes:.1f} min)"
            )
        sleeper(poll_seconds)


__all__ = [
    "ALL_FEEDS",
    "BAR",
    "Bars1m",
    "FeedError",
    "cboe_prints",
    "es_minute_bars",
    "fetch_1m",
    "fetch_all",
    "history_download",
    "last_complete_close",
    "latency_records",
    "now_et",
    "spx_minute_bars",
    "take",
    "wait_for_stamp",
]
