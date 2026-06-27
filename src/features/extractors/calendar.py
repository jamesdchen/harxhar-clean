"""Calendar features (DOW dummies, hour, is_overnight).

Pure column extractor — adds calendar columns to a DataFrame; doesn't modify
existing values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> list[str]:
    """Add calendar features. Returns the list of new column names.

    Shared encoding across all models (ridge, xgb, lgbm, rf, pcr):
    5 weekday dummies DOW_0..DOW_4 (Mon-Fri; weekends excluded by
    market-hours filter), int hour (0-23), binary is_overnight, and the
    clock-anchored session-edge gates is_open / is_close.

    `is_overnight` = 1 outside US RTH (09:30-16:00 ET) or on weekends.
    `is_open` = 1 on the 09:30 ET opening bar (hour 9); `is_close` = 1 on the
    16:00 close auction + after-hours, hours 16-19 (capped below 20 to exclude
    dead overnight bars; US after-hours trading ends ~20:00 ET). These mark the
    intraday auction/session-transition regime where HAR vol-persistence sign-
    flips, and feed the upstream HAR x open/close interaction features (executor).
    """
    df["DOW"] = df["t"].dt.dayofweek
    df["hour"] = df["t"].dt.hour
    df["is_overnight"] = ((df["hour"] < 9) | (df["hour"] >= 16) | (df["DOW"] >= 5)).astype(np.int8)
    df["is_open"] = (df["hour"] == 9).astype(np.int8)  # 09:30 ET opening bar (open lobe)
    df["is_close"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(np.int8)  # 16-19 close + after-hours (no dead overnight)
    for d in range(5):  # Mon=0 .. Fri=4 (no weekend bars after market filter)
        df[f"DOW_{d}"] = (df["DOW"] == d).astype(np.int8)
    return [f"DOW_{d}" for d in range(5)] + ["hour", "is_overnight", "is_open", "is_close"]
