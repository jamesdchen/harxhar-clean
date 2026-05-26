"""HAR lag features (rolling means at powers-of-5 lags) and raw lag features.

Pure column extractors — given a target column (and optional exogs), produce
new feature columns. Don't modify existing values.

* :func:`resolve_har_lags` — ``[1, 5, 25, 125, 625, 3125]``
* :func:`generate_har_features` — rolling means at those lags, shifted by 1
* :func:`resolve_pca_lags` — log-spaced lag indices for PCR
* :func:`generate_raw_lag_features` — shifted-lag columns for each PCR lag
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_har_lags(max_lag: int = 3125) -> list[int]:
    """Powers-of-5 lag sequence: [1, 5, 25, 125, 625, 3125]."""
    seq, v = [], 1
    while v <= max_lag:
        seq.append(v)
        v *= 5
    return seq


def generate_har_features(
    df: pd.DataFrame,
    target_col: str = "adj_RV",
    exog_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add rolling-mean HAR features (shifted by 1) for each powers-of-5 lag.

    Features are generated for *target_col* and each column in *exog_cols*.
    Target features are named ``har_ma_{lag}``; exog features ``{col}_ma_{lag}``.
    """
    lags = resolve_har_lags()
    features: dict[str, pd.Series] = {}
    feature_names: list[str] = []
    for col in [target_col] + (exog_cols or []):
        for lag in lags:
            name = f"har_ma_{lag}" if col == target_col else f"{col}_ma_{lag}"
            features[name] = df[col].rolling(window=lag, min_periods=1).mean().shift(1)
            feature_names.append(name)
    feat_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feat_df], axis=1), feature_names


def resolve_pca_lags(max_lag: int = 3125, num_points: int = 20) -> list[int]:
    """Generate log-spaced lag indices from 1 to *max_lag*."""
    raw = np.geomspace(1, max_lag, num=num_points)
    return sorted(set(int(round(v)) for v in raw))


def generate_raw_lag_features(
    df: pd.DataFrame,
    target_col: str = "adj_RV",
    max_lag: int = 3125,
    exog_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create shifted-lag columns for each log-spaced lag.

    Features are generated for *target_col* and each column in *exog_cols*.
    """
    lags = resolve_pca_lags(max_lag)
    features: dict[str, pd.Series] = {}
    feature_names: list[str] = []
    for col in [target_col] + (exog_cols or []):
        for lag in lags:
            name = f"{col}_lag_{lag}"
            features[name] = df[col].shift(lag)
            feature_names.append(name)
    feat_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feat_df], axis=1), feature_names
