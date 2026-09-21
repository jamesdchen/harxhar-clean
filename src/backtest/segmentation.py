"""Time-of-day segmentation for per-segment backtests.

A backtest concern, not a feature transform: ``slice_to_segment`` filters a
prepared DataFrame to bars within a named time-of-day range, and the executor
uses it from :func:`src.backtest.executor._iter_TOD_segment` to run separate
backtests per segment when ``--segment`` is set.
"""

from __future__ import annotations

import pandas as pd


def _hhmm(h: int, m: int) -> int:
    return h * 60 + m


SEGMENT_CHOICES: list[str] = [
    "all",
    "morning",
    "midday",
    "closing",
    "overnight",
    "rth",
    "last30",
]

SEGMENT_DEFINITIONS: dict[str, tuple[int, int]] = {
    "morning": (_hhmm(8, 30), _hhmm(11, 0)),
    "midday": (_hhmm(10, 30), _hhmm(14, 30)),
    "closing": (_hhmm(14, 0), _hhmm(16, 0)),
    "overnight": (_hhmm(16, 30), _hhmm(8, 30)),
    # RTH cash session 09:30-16:00 ET (no overnight bars).
    "rth": (_hhmm(9, 30), _hhmm(16, 0)),
    # Last 30 minutes of RTH: 15:30-16:00 ET (the 15:30→close bar).
    "last30": (_hhmm(15, 30), _hhmm(16, 0)),
}

# Subsection regressions. ``t`` is the bar-END label and both bounds are
# inclusive, so a (label, label) range is exactly one bar a day: ``bar1600``
# is the 15:30-16:00 bar, forecast at 15:30. (``last30`` above therefore holds
# TWO bars, the ones ending 15:30 and 16:00.) Use with ``lag_scope="global"``:
# the lags are built on the full series and only the fitted rows are sliced.
BAR_SEGMENTS: dict[str, tuple[int, int]] = {
    f"bar{h:02d}{m:02d}": (_hhmm(h, m), _hhmm(h, m))
    for h in range(10, 17)
    for m in (0, 30)
    if (h, m) <= (16, 0)
}
# A partition of the 13 regular-hours bars into four blocks.
BLOCK_SEGMENTS: dict[str, tuple[int, int]] = {
    "blk_open": (_hhmm(10, 0), _hhmm(10, 30)),
    "blk_core": (_hhmm(11, 0), _hhmm(14, 0)),
    "blk_release": (_hhmm(14, 30), _hhmm(15, 0)),  # the two bars after 14:00
    "blk_close": (_hhmm(15, 30), _hhmm(16, 0)),
}
SEGMENT_DEFINITIONS.update(BAR_SEGMENTS)
SEGMENT_DEFINITIONS.update(BLOCK_SEGMENTS)
SEGMENT_CHOICES.extend([*BAR_SEGMENTS, *BLOCK_SEGMENTS])


def slice_to_segment(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Filter rows to a time-of-day segment. Handles midnight wrap-around."""
    start, end = SEGMENT_DEFINITIONS[segment]
    minutes = df["t"].dt.hour * 60 + df["t"].dt.minute
    if start <= end:  # start == end is a one-bar segment, not a wrap-around
        mask = (minutes >= start) & (minutes <= end)
    else:
        mask = (minutes >= start) | (minutes <= end)
    return df.loc[mask].reset_index(drop=True)


def compute_segment_train_window(dates: pd.Series, train_window_days: int) -> int:
    """Compute train window in periods using median slots/day for a segment."""
    daily_counts = pd.Series(dates).dt.date.value_counts()
    median_slots = int(daily_counts.median())
    return train_window_days * median_slots
