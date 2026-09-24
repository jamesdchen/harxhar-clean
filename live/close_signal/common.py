"""Shared constants and conventions of the close-signal package.

Panel clock: naive US/Eastern, bar-END labelled -- the row at stamp ``tau``
carries the bar ``(tau - 30 min, tau]``.  The grid is the 30-minute clock of
the calendar day (48 stamps, ``00:00 .. 23:30``) and a ROW EXISTS IFF THE BAR
HAD ES PRINTS -- read off the vendor panel: no Saturday, Sunday from 18:30,
Friday to 20:00 (the vendor's ES has prints past the 17:00 CME close and
through the 17:00-18:00 maintenance hour; the free feeds do not, so a free
weekday has 46 rows to the vendor's 48 and a Friday 35 to 41), holiday
sessions end where the prints end (13:00 on a Monday holiday), and never a
row at the spring-forward 02:00.  The arms lag by row, so the extension must
keep to this rule (``forecast._extend``), never a full calendar grid.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

ET = "America/New_York"
GRID = "30min"
BARS_PER_DAY = 48
BAR = pd.Timedelta(minutes=30)
ONE_MINUTE = pd.Timedelta(minutes=1)

#: The ES-derived columns of data/core_stats.parquet, in the panel's order.
CORE_COLS: tuple[str, ...] = (
    "sumret",
    "sumabsret",
    "sumret2",
    "sumret3",
    "sumret4",
    "sumpret2",
    "sumbipow",
    "sumautocov",
    "sumvolume",
    "numobs",
)
#: The Cboe prints of data/vix_and_voldemand.parquet used by the live bucket.
CBOE_COLS: tuple[str, ...] = ("vix", "vvix", "vix3m")

#: Yahoo symbols.
ES_SYMBOL = "ES=F"
SPX_SYMBOL = "^GSPC"
CBOE_SYMBOLS: dict[str, str] = {"vix": "^VIX", "vvix": "^VVIX", "vix3m": "^VIX3M"}

#: Yahoo's published exchange delays (minutes), read 2026-09-23: Cboe indices
#: 15, CME futures 10, the S&P 500 index real-time.  At 15:30:30 the free
#: feeds therefore do NOT carry the 15:00-15:30 ES bar or the 15:30 Cboe
#: prints; the run measures the actual delay and never uses a stale bar as
#: the 15:30 stamp without saying so.
YAHOO_NOMINAL_DELAY_MIN: dict[str, int] = {
    ES_SYMBOL: 10,
    SPX_SYMBOL: 0,
    "^VIX": 15,
    "^VVIX": 15,
    "^VIX3M": 15,
}

#: How the 15:00-15:30 inputs are obtained.
#:   ibkr            real-time ES bars and Cboe prints through live/ibkr's broker
#:                   (the research construction; the reference mode)
#:   free_substitute real-time ^GSPC 1-minute bars stand in for ES on the LAST bar's
#:                   return moments; the Cboe prints are the latest available (~15:15),
#:                   lag journaled.  A MODEL CHANGE: to be validated on the purchased
#:                   history before it is trusted (README).
#:   free_delayed    wait until the 15:30 ES bar and Cboe prints have arrived
#:                   (~15:45) and mark the card LATE.
InputMode = Literal["ibkr", "free_substitute", "free_delayed"]
INPUT_MODES: tuple[str, ...] = ("ibkr", "free_substitute", "free_delayed")

#: The forecast bar and its clocks.
FORECAST_STAMP = "16:00"  # the row forecast: the 15:30-16:00 bar
DECISION_STAMP = "15:30"  # the last stamp whose data the forecast may use
SESSION_OPEN = "09:30"

#: Index option contract facts (Cboe).  XSP is one tenth of SPX, settles to
#: SPX / 10 at the official close, $100 a point, $0.01 ticks.  Listed grids
#: near the money (study 79, OPRA definitions, 873 sessions 2023-03-28 ..
#: 2026-09-22): SPX 5 points on every chain day; XSP 1 point on 839 days, and
#: on 34 non-monthly Fridays (never a month-end, never another weekday) XSP
#: also lists the half strikes at SPX's 25 / 75 strikes -- x2.5 and x7.5 --
#: which were in the nearest-OTM pair on 5 of the 873 days.
INDEX_MULTIPLIER = 100.0
SPX_STRIKE_STEP = 5.0
XSP_STRIKE_STEP = 1.0
#: XSP half strikes, when listed: 2.5 + 5n (XSP points).
XSP_HALF_STRIKE_OFFSET = 2.5
XSP_HALF_STRIKE_PERIOD = 5.0
XSP_SCALE = 0.1


def bar_end_of(ts_start: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Panel stamp (bar END) of a 1-minute bar that STARTS at ``ts_start``.

    The minute starting 15:29 belongs to the bar ending 15:30; the minute
    starting 15:30 belongs to the bar ending 16:00.
    """
    return ts_start.floor(GRID) + BAR


def session_grid(day: pd.Timestamp) -> pd.DatetimeIndex:
    """The 48 stamps of a calendar day, ``00:00 .. 23:30``, naive ET."""
    d = pd.Timestamp(day).normalize()
    return pd.date_range(d, d + pd.Timedelta(hours=23, minutes=30), freq=GRID)


def stamp(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    h, m = hhmm.split(":")
    return pd.Timestamp(day).normalize() + pd.Timedelta(hours=int(h), minutes=int(m))
