"""The Yahoo-hourly Cboe ingest: stamp placement, carry, no look-ahead, daily fallback, provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.common import CBOE_COLS, ET  # noqa: E402
from live.close_signal.forecast import _extend  # noqa: E402
from live.close_signal.ingest import (  # noqa: E402
    YAHOO_HOURLY_SOURCE,
    cboe_rows_from_yahoo,
    ingest_cboe_yahoo,
    read_yahoo_daily,
    read_yahoo_hourly,
)
from live.close_signal.state import StateStore  # noqa: E402

D0 = pd.Timestamp("2024-02-29")  # a Thursday
D1 = pd.Timestamp("2024-03-01")  # Friday: hourly bars
D2 = pd.Timestamp("2024-03-04")  # Monday: NO hourly bars (a feed hole)


def _hourly(starts: list[str], closes: list[float]) -> pd.DataFrame:
    """Bars indexed by naive-ET bar START (what read_yahoo_hourly returns)."""
    idx = pd.DatetimeIndex([pd.Timestamp(s) for s in starts])
    return pd.DataFrame(
        {
            "open": np.array(closes) - 0.1,
            "high": np.array(closes) + 0.2,
            "low": 0.0,
            "close": closes,
        },
        index=idx,
    )


def _daily(dates: list[str], closes: list[float]) -> pd.Series:
    return pd.Series(closes, index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))


def test_a_bar_s_close_lands_at_the_stamp_the_bar_ends_on() -> None:
    """VIX-style bars start on the hour: the 14:00-15:00 bar's close is the 15:00 stamp."""
    h = {"vix": _hourly(["2024-03-01 14:00", "2024-03-01 15:00"], [16.5, 17.0])}
    rows = cboe_rows_from_yahoo(h, {}, D1, D1).set_index("endbartime")
    assert rows.loc[pd.Timestamp("2024-03-01 15:00"), "vix"] == 16.5
    assert rows.loc[pd.Timestamp("2024-03-01 16:00"), "vix"] == 17.0
    assert rows.loc[pd.Timestamp("2024-03-01 15:00"), "vix_source"] == "hourly"
    # the half-hour stamp between two prints carries the earlier one
    assert rows.loc[pd.Timestamp("2024-03-01 15:30"), "vix"] == 16.5
    assert rows.loc[pd.Timestamp("2024-03-01 15:30"), "vix_source"] == "hourly_carry"
    # after the last print of the day the carry continues to the day's end
    assert rows.loc[pd.Timestamp("2024-03-01 23:30"), "vix"] == 17.0
    assert rows.loc[pd.Timestamp("2024-03-01 23:30"), "vix_source"] == "hourly_carry"


def test_half_hour_bars_land_on_the_half_hour_stamps_including_15_30() -> None:
    """VVIX / VIX3M bars start at :30 from 09:30: the 14:30-15:30 bar's close IS the 15:30 stamp."""
    h = {"vvix": _hourly(["2024-03-01 14:30", "2024-03-01 15:30"], [90.0, 92.0])}
    rows = cboe_rows_from_yahoo(h, {}, D1, D1).set_index("endbartime")
    assert rows.loc[pd.Timestamp("2024-03-01 15:30"), "vvix"] == 90.0
    assert rows.loc[pd.Timestamp("2024-03-01 15:30"), "vvix_source"] == "hourly"
    assert rows.loc[pd.Timestamp("2024-03-01 16:00"), "vvix"] == 90.0  # carry
    assert rows.loc[pd.Timestamp("2024-03-01 16:30"), "vvix"] == 92.0
    assert rows.loc[pd.Timestamp("2024-03-01 16:30"), "vvix_source"] == "hourly"


def test_bars_off_the_half_hour_grid_are_refused() -> None:
    h = {"vix": _hourly(["2024-03-01 14:07"], [16.5])}
    with pytest.raises(ValueError, match="half-hour grid"):
        cboe_rows_from_yahoo(h, {}, D1, D1)


def test_daily_fallback_is_the_previous_close_and_never_the_same_day() -> None:
    h = {"vix": _hourly(["2024-03-01 15:00"], [17.0])}
    d = {"vix": _daily(["2024-02-29", "2024-03-01", "2024-03-04"], [15.0, 99.0, 77.0])}
    rows = cboe_rows_from_yahoo(h, d, D1, D2).set_index("endbartime")
    # before the day's first print: the PREVIOUS day's close (15.0), never the same day's (99.0)
    early = rows.loc[pd.Timestamp("2024-03-01 09:30")]
    assert early["vix"] == 15.0 and early["vix_source"] == "daily_carry"
    # a day with no hourly bars: the previous session's close all day; its own close (77.0) is unusable
    hole = rows.loc[pd.Timestamp("2024-03-04 15:30")]
    assert hole["vix"] == 99.0 and hole["vix_source"] == "daily_carry"
    assert (rows.loc[rows.index.normalize() == D2, "vix"] == 99.0).all()
    # the weekend in between also carries the 03-01 close (the loader drops the closed rows)
    assert rows.loc[pd.Timestamp("2024-03-02 12:00"), "vix"] == 99.0


def test_no_source_at_all_stays_nan_with_an_empty_tag() -> None:
    rows = cboe_rows_from_yahoo({}, {}, D1, D1)
    assert rows[list(CBOE_COLS)].isna().all().all()
    assert (rows["vix_source"] == "").all()


def test_readers_convert_utc_to_naive_et_bar_start_and_dates(tmp_path) -> None:
    # 2024-03-01 15:00 ET = 20:00 UTC (EST)
    hp = tmp_path / "vix_1h_yahoo.parquet"
    pd.DataFrame(
        {
            "t": pd.to_datetime(["2024-03-01 20:00", "2024-03-01 21:00"], utc=True),
            "open": [16.9, 17.1],
            "high": [17.2, 17.3],
            "low": [16.4, 16.8],
            "close": [17.0, 17.2],
            "volume": [0, 0],
        }
    ).to_parquet(hp, index=False)
    h = read_yahoo_hourly(hp)
    assert list(h.index) == [
        pd.Timestamp("2024-03-01 15:00"),
        pd.Timestamp("2024-03-01 16:00"),
    ]
    assert h.index.tz is None and list(h.columns) == ["open", "high", "low", "close"]
    dp = tmp_path / "vix_1d_yahoo.parquet"
    pd.DataFrame(
        {"t": pd.to_datetime(["2024-02-29", "2024-03-01"]), "close": [15.0, 99.0]}
    ).to_parquet(dp, index=False)
    d = read_yahoo_daily(dp)
    assert d.loc[pd.Timestamp("2024-03-01")] == 99.0 and d.index.tz is None
    # ET conversion round-trips through the zone name the package uses
    assert pd.Timestamp("2024-03-01 20:00", tz="UTC").tz_convert(ET).hour == 15


def test_ingest_writes_the_standalone_file_and_the_store_under_the_hourly_source(
    tmp_path,
) -> None:
    hp = tmp_path / "vix_1h_yahoo.parquet"
    pd.DataFrame(
        {
            "t": pd.to_datetime(["2024-03-01 19:00", "2024-03-01 20:00"], utc=True),
            "open": [16.4, 16.9],
            "high": [16.7, 17.2],
            "low": [16.1, 16.4],
            "close": [16.5, 17.0],
            "volume": [0, 0],
        }
    ).to_parquet(hp, index=False)
    dp = tmp_path / "vix_1d_yahoo.parquet"
    pd.DataFrame({"t": pd.to_datetime(["2024-02-29"]), "close": [15.0]}).to_parquet(
        dp, index=False
    )
    store = StateStore(tmp_path / "state")
    info = ingest_cboe_yahoo(
        store,
        {"vix": hp},
        {"vix": dp},
        start=D1,
        end=D1,
        out_path=tmp_path / "state" / "cboe_gap_yahoo.parquet",
    )
    standalone = pd.read_parquet(tmp_path / "state" / "cboe_gap_yahoo.parquet")
    assert {"vix_source", "vvix_source", "vix3m_source"} <= set(standalone.columns)
    assert info["vix:hourly"] == 2 and info["vix:daily_carry"] > 0
    panel = store.load_panel().set_index("endbartime")
    assert (panel["source"] == YAHOO_HOURLY_SOURCE).all()
    assert panel.loc[pd.Timestamp("2024-03-01 15:00"), "vix"] == 16.5
    assert panel.loc[pd.Timestamp("2024-03-01 16:00"), "vix"] == 17.0
    assert np.isnan(panel.loc[pd.Timestamp("2024-03-01 16:00"), "vvix"])
    assert bool(panel["placeholder"].any()) is False


def test_the_vendor_s_own_stamps_are_never_overwritten_by_the_extension() -> None:
    vendor = pd.DataFrame(
        {
            "endbartime": pd.to_datetime(["2024-02-12 15:30", "2024-02-12 16:00"]),
            "vix": [14.0, 14.1],
        }
    )
    ext = pd.DataFrame(
        {
            "endbartime": pd.to_datetime(["2024-02-12 16:00", "2024-02-13 15:30"]),
            "vix": [99.0, 15.5],
            "vvix": [np.nan, 80.0],
            "vix3m": [np.nan, 16.0],
        }
    )
    out = _extend(vendor, ext, ["vix", "vvix", "vix3m"]).set_index("endbartime")
    assert out.loc[pd.Timestamp("2024-02-12 16:00"), "vix"] == 14.1  # vendor wins
    assert (
        out.loc[pd.Timestamp("2024-02-13 15:30"), "vix"] == 15.5
    )  # the gap row is added
