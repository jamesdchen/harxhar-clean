"""The 30-minute moments against their written definitions, on synthetic 1-minute bars."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.common import CORE_COLS, bar_end_of, session_grid  # noqa: E402
from live.close_signal.features import (  # noqa: E402
    assert_parity,
    cboe_stamp_values,
    panel_rows,
    substitute_last_bar,
    thirty_minute_moments,
)


def _bars(start: str, n: int, seed: int = 0, base: float = 5000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min")
    r = rng.normal(0, 3e-4, size=n)
    close = base * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 100.0},
        index=idx,
    )


def test_bar_end_labels_are_bar_end() -> None:
    idx = pd.DatetimeIndex(["2026-09-23 15:29", "2026-09-23 15:30", "2026-09-23 15:59"])
    got = bar_end_of(idx)
    assert list(got.strftime("%H:%M")) == ["15:30", "16:00", "16:00"]


def test_moments_match_the_definitions_on_one_bar() -> None:
    # 31 closes: p_0 at 14:59 then the 30 minutes 15:00..15:29 -> the bar ending 15:30
    bars = _bars("2026-09-23 14:59", 31, seed=1)
    out = thirty_minute_moments(bars).set_index("endbartime")
    row = out.loc[pd.Timestamp("2026-09-23 15:30")]
    r = np.diff(np.log(bars["close"].to_numpy()))
    assert row["numobs"] == 30
    assert row["sumret"] == pytest.approx(r.sum())
    assert row["sumabsret"] == pytest.approx(np.abs(r).sum())
    assert row["sumret2"] == pytest.approx((r**2).sum())
    assert row["sumret3"] == pytest.approx((r**3).sum())
    assert row["sumret4"] == pytest.approx((r**4).sum())
    assert row["sumpret2"] == pytest.approx((r[r > 0] ** 2).sum())
    assert row["sumbipow"] == pytest.approx((np.abs(r[1:]) * np.abs(r[:-1])).sum())
    assert row["sumautocov"] == pytest.approx((r[1:] * r[:-1]).sum())
    assert row["sumvolume"] == pytest.approx(3000.0)
    assert list(out.columns) == list(CORE_COLS)


def test_lag_products_do_not_cross_the_bar_edge() -> None:
    bars = _bars(
        "2026-09-23 14:29", 61, seed=2
    )  # two full bars 14:30-15:00 and 15:00-15:30
    out = thirty_minute_moments(bars).set_index("endbartime")
    r = np.diff(np.log(bars["close"].to_numpy()))
    second = r[30:]  # the 30 returns of the bar ending 15:30
    assert out.loc[pd.Timestamp("2026-09-23 15:30"), "sumbipow"] == pytest.approx(
        (np.abs(second[1:]) * np.abs(second[:-1])).sum()
    )
    # the return across the edge (14:59 -> 15:00) belongs to the second bar's sumret2
    assert out.loc[pd.Timestamp("2026-09-23 15:30"), "sumret2"] == pytest.approx(
        (second**2).sum()
    )


def test_no_return_across_the_session_break_but_across_a_short_gap() -> None:
    """Friday 16:59 -> Sunday 18:00 is a break (no return); a 5-minute no-trade gap is not."""
    fri = _bars("2026-09-18 16:30", 30, seed=5)  # minutes 16:30..16:59 -> bar 17:00
    sun = _bars("2026-09-20 18:00", 30, seed=6, base=fri["close"].iloc[-1] * 1.02)
    sun = sun.drop(sun.index[10:15])  # five minutes without a trade inside the bar
    out = thirty_minute_moments(pd.concat([fri, sun])).set_index("endbartime")
    row = out.loc[pd.Timestamp("2026-09-20 18:30")]
    r = np.diff(np.log(sun["close"].to_numpy()))  # within Sunday only: 24 returns
    assert row["numobs"] == 24  # 25 minutes, the first without a prior
    assert row["sumret2"] == pytest.approx((r**2).sum())  # the +2 % gap jump is absent
    assert row["sumret2"] < 1e-4  # ... it would have been ~4e-4 on its own
    # the 5-minute gap's return (minute 9 -> minute 15) IS in the sum
    assert (np.abs(r) > 0).all()
    # the Friday bar is untouched
    assert out.loc[pd.Timestamp("2026-09-18 17:00"), "numobs"] == 29


def test_cboe_print_is_the_last_close_inside_the_bar() -> None:
    idx = pd.date_range("2026-09-23 15:00", periods=45, freq="1min")
    fr = pd.DataFrame({"close": np.arange(45, dtype=float)}, index=idx)
    stamps = pd.DatetimeIndex(
        ["2026-09-23 15:30", "2026-09-23 16:00", "2026-09-23 16:30"]
    )
    got = cboe_stamp_values({"vix": fr}, stamps)
    assert got["vix"].tolist()[0] == 29.0  # the minute starting 15:29
    assert got["vix"].tolist()[1] == 44.0  # last available inside 15:30-16:00
    assert np.isnan(got["vix"].tolist()[2])
    assert np.isnan(got["vvix"]).all()


def test_panel_rows_cover_the_48_stamps_and_substitute_only_the_asked_bar() -> None:
    day = pd.Timestamp("2026-09-23")
    es = _bars("2026-09-22 18:00", 22 * 60, seed=3)  # through 15:59 next day
    spx = _bars("2026-09-23 09:29", 391, seed=4, base=5001.0)
    rows = panel_rows(es, {"vix": spx.rename(columns={"close": "close"})}, day)
    assert len(rows) == 48 and list(rows["endbartime"]) == list(session_grid(day))
    t = pd.Timestamp("2026-09-23 15:30")
    before = rows.set_index("endbartime").loc[t, "sumret2"]
    sub = substitute_last_bar(rows, spx, t).set_index("endbartime")
    assert sub.loc[t, "sumret2"] != before
    assert np.isnan(sub.loc[t, "sumvolume"]) and np.isnan(sub.loc[t, "numobs"])
    other = pd.Timestamp("2026-09-23 15:00")
    assert (
        sub.loc[other, "sumret2"] == rows.set_index("endbartime").loc[other, "sumret2"]
    )


def test_parity_gate_passes_on_identical_and_fails_on_drift() -> None:
    bars = _bars("2026-09-01 00:00", 48 * 60 * 3, seed=5)
    ours = thirty_minute_moments(bars)
    rep = assert_parity(ours, ours, rel_tol=1e-12, min_rows=100)
    assert (rep["max_rel"].fillna(0) == 0).all()
    bad = ours.copy()
    bad["sumret2"] = bad["sumret2"] * 1.5
    with pytest.raises(AssertionError, match="sumret2"):
        assert_parity(bad, ours, rel_tol=1e-6, min_rows=100)
    with pytest.raises(AssertionError, match="overlapping"):
        assert_parity(ours.head(5), ours.head(5), rel_tol=1e-6, min_rows=100)
