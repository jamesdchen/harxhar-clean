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
    market-hours filter), int hour (0-23), and binary is_overnight.

    `is_overnight` = 1 outside US RTH (09:30-16:00 ET) or on weekends.
    """
    df["DOW"] = df["t"].dt.dayofweek
    df["hour"] = df["t"].dt.hour
    df["is_overnight"] = ((df["hour"] < 9) | (df["hour"] >= 16) | (df["DOW"] >= 5)).astype(np.int8)
    for d in range(5):  # Mon=0 .. Fri=4 (no weekend bars after market filter)
        df[f"DOW_{d}"] = (df["DOW"] == d).astype(np.int8)
    return [f"DOW_{d}" for d in range(5)] + ["hour", "is_overnight"]
