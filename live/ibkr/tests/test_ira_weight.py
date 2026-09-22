"""The daily-bar volatility-managed weight: causal, bounded, agrees with the panel's."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from live.ira_weight import (
    PANEL_WEIGHT,
    parkinson_variance,
    weights_from_variance,
)

CACHE = os.path.join("results", "ira_weight", "gspc_ohlc.parquet")
needs_data = pytest.mark.skipif(
    not (os.path.exists(CACHE) and os.path.exists(PANEL_WEIGHT)),
    reason="the cached bars or the panel weight have not been built",
)


def synthetic_bars(n: int, seed: int, shock_from: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.01)
    if shock_from is not None:
        vol[shock_from:] = 0.04
    close = 4000.0 * np.exp(np.cumsum(rng.normal(0, vol)))
    rng_hl = np.abs(rng.normal(0, vol)) + 0.002
    high, low = close * (1 + rng_hl / 2), close * (1 - rng_hl / 2)
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


def test_parkinson_variance_is_positive_and_scales_with_the_range():
    bars = synthetic_bars(500, 0)
    v = parkinson_variance(bars)
    assert (v.dropna() > 0).all()
    wide = bars.copy()
    wide["high"] *= 1.01
    assert (parkinson_variance(wide) > v).all()
    flat = bars.copy()
    flat["high"] = flat["low"]
    assert parkinson_variance(flat).isna().all()


def test_weights_warm_up_then_bound_and_cut_after_a_shock():
    bars = synthetic_bars(900, 1, shock_from=850)
    w = weights_from_variance(parkinson_variance(bars))
    assert len(w) == 901 and w.index[-1] > bars.index[-1]  # tomorrow's row
    assert w["weight"].iloc[:252].isna().all()  # the regression's warm-up
    finite = w["weight"].dropna()
    assert (finite <= 1.0).all() and (finite > 0.0).all()
    assert (
        w["weight"].iloc[-1] < 0.6
    )  # a month of 4x the variance: well below full weight
    # causal: rewriting the future changes nothing before it
    other = bars.copy()
    other.iloc[850:, other.columns.get_loc("high")] *= 1.05
    w2 = weights_from_variance(parkinson_variance(other))
    assert np.allclose(w["weight"].iloc[:850], w2["weight"].iloc[:850], equal_nan=True)


@needs_data
def test_daily_bar_weight_agrees_with_the_panel_weight():
    bars = pd.read_parquet(CACHE)
    w = weights_from_variance(parkinson_variance(bars))
    panel = pd.read_parquet(PANEL_WEIGHT)
    j = w.join(panel, how="inner").dropna(subset=["weight", "vm_weight_panel"])
    assert len(j) > 5000
    corr = float(np.corrcoef(j["weight"], j["vm_weight_panel"])[0, 1])
    assert corr > 0.9, corr
    assert float((j["weight"] - j["vm_weight_panel"]).abs().mean()) < 0.05
    agree = float(((j["weight"] < 1) == (j["vm_weight_panel"] < 1)).mean())
    assert agree > 0.9, agree
