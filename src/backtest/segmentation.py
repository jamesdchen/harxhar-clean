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


SEGMENT_CHOICES: list[str] = ["all", "morning", "midday", "closing", "overnight"]

SEGMENT_DEFINITIONS: dict[str, tuple[int, int]] = {
    "morning": (_hhmm(8, 30), _hhmm(11, 0)),
    "midday": (_hhmm(10, 30), _hhmm(14, 30)),
    "closing": (_hhmm(14, 0), _hhmm(16, 0)),
    "overnight": (_hhmm(16, 30), _hhmm(8, 30)),
}


def slice_to_segment(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Filter rows to a time-of-day segment. Handles midnight wrap-around."""
    start, end = SEGMENT_DEFINITIONS[segment]
    minutes = df["t"].dt.hour * 60 + df["t"].dt.minute
    if start < end:
        mask = (minutes >= start) & (minutes <= end)
    else:
        mask = (minutes >= start) | (minutes <= end)
    return df.loc[mask].reset_index(drop=True)


def compute_segment_train_window(dates: pd.Series, train_window_days: int) -> int:
    """Compute train window in periods using median slots/day for a segment."""
    daily_counts = pd.Series(dates).dt.date.value_counts()
    median_slots = int(daily_counts.median())
    return train_window_days * median_slots
