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
    out = _extend(vendor, ext, ["vix", "vvix", "vix3m"], require=None).set_index(
        "endbartime"
    )
    assert out.loc[pd.Timestamp("2024-02-12 16:00"), "vix"] == 14.1  # vendor wins
    assert (
        out.loc[pd.Timestamp("2024-02-13 15:30"), "vix"] == 15.5
    )  # the gap row is added


def test_extension_adds_rows_only_where_es_prints_exist_and_fills_vendor_nan_cells() -> (
    None
):
    """The vendor's row rule, and the vendor's NaN VIX cells taking the free feed's prints."""
    vendor = pd.DataFrame(
        {
            "endbartime": pd.to_datetime(
                ["2024-04-30 15:30", "2024-04-30 16:00", "2024-04-30 16:30"]
            ),
            "sumret2": [1e-6, 2e-6, 3e-6],
            "vix": [np.nan, 14.1, np.nan],
        }
    )
    ext = pd.DataFrame(
        {
            "endbartime": pd.to_datetime(
                [
                    "2024-04-30 15:30",  # vendor stamp, vendor vix NaN -> filled
                    "2024-04-30 16:00",  # vendor stamp, vendor vix finite -> kept
                    "2024-05-01 10:00",  # new stamp with ES -> added
                    "2024-05-04 10:00",  # Saturday: Cboe carry, no ES -> NOT a row
                    "2025-03-09 02:00",  # spring-forward stamp, no ES -> NOT a row
                ]
            ),
            "sumret2": [np.nan, np.nan, 4e-6, np.nan, np.nan],
            "vix": [15.0, 99.0, 15.5, 15.6, 15.7],
        }
    )
    core = _extend(vendor, ext, ["sumret2"]).set_index("endbartime")
    vix = _extend(vendor[["endbartime", "vix", "sumret2"]], ext, ["vix"]).set_index(
        "endbartime"
    )
    assert list(core.index) == list(vix.index)  # the four files share one grid
    assert len(core) == 4 and pd.Timestamp("2024-05-01 10:00") in core.index
    assert pd.Timestamp("2024-05-04 10:00") not in core.index
    assert pd.Timestamp("2025-03-09 02:00") not in core.index
    assert vix.loc[pd.Timestamp("2024-04-30 15:30"), "vix"] == 15.0  # NaN filled
    assert vix.loc[pd.Timestamp("2024-04-30 16:00"), "vix"] == 14.1  # finite kept
    assert core.loc[pd.Timestamp("2024-04-30 15:30"), "sumret2"] == 1e-6
    # the extended grid localizes: no non-existent stamp survived
    pd.DatetimeIndex(core.index).tz_localize(ET)


def _databento_csv(path: Path, fixed_point: bool) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Two contracts, the roll at 10:00 ET; the price level jumps 20 points at the roll."""
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2024-06-13 09:30", tz=ET)
    n = 90
    ts = pd.date_range(start, periods=n, freq="1min")
    r = rng.normal(0, 3e-4, size=n)
    close = 5000.0 * np.exp(np.cumsum(r))
    inst = np.where(np.arange(n) < 30, 1001, 1002)
    close = np.where(inst == 1002, close + 20.0, close)  # the stitched jump
    px = close * 1e9 if fixed_point else close
    df = pd.DataFrame(
        {
            "ts_event": ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "rtype": 32,
            "publisher_id": 1,
            "instrument_id": inst,
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "volume": 100,
            "symbol": "ES.v.0",
        }
    )
    if fixed_point:
        for c in ("open", "high", "low", "close"):
            df[c] = np.round(df[c]).astype("int64")
    df.to_csv(path, index=False)
    bars = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 100.0},
        index=ts.tz_convert(ET).tz_localize(None),
    )
    return bars, ts[30].tz_convert(ET).tz_localize(None)


@pytest.mark.parametrize("fixed_point", [False, True])
def test_databento_es_ingest_builds_moments_per_contract_and_reports_the_seam(
    tmp_path, fixed_point
) -> None:
    from live.close_signal.features import thirty_minute_moments
    from live.close_signal.ingest import ingest_es, read_databento_ohlcv_1m

    csv = tmp_path / "es.csv"
    bars, roll_minute = _databento_csv(csv, fixed_point)
    raw = read_databento_ohlcv_1m(csv)
    assert np.allclose(raw["close"].to_numpy(), bars["close"].to_numpy())  # scaled
    assert raw.index[0] == pd.Timestamp("2024-06-13 09:30")  # UTC -> naive ET
    assert "instrument_id" in raw.columns
    # the store already has the free feed's build of the 10:00 bar (from the SAME
    # 1-minute closes, so the seam report is exact on that stamp) ...
    store = StateStore(tmp_path / "state")
    free = thirty_minute_moments(bars[bars.index < roll_minute])
    store.append_panel(free, source="yahoo_es")
    info = ingest_es(store, csv)
    assert info["roll_key"] == "instrument_id" and info["rolls"] == [str(roll_minute)]
    assert info["overlap_stamps"] == 1 and info["sumret2_n"] == 1
    # fixed-point prices are rounded to 1e-9: the two builds agree to ~1e-10 in log
    assert abs(info["sumret2_logratio_median"]) < (1e-8 if fixed_point else 1e-12)
    panel = store.load_panel().set_index("endbartime")
    assert (panel["source"] == "databento_es").all()  # the purchase overwrote
    # ... and the bar containing the roll carries no stitched jump: its moments
    # equal the two contract pieces built apart, and the 20-point jump is absent
    a = thirty_minute_moments(bars[bars.index < roll_minute]).set_index("endbartime")
    b = thirty_minute_moments(bars[bars.index >= roll_minute]).set_index("endbartime")
    t = pd.Timestamp("2024-06-13 10:30")
    assert np.isclose(
        panel.loc[t, "sumret2"],
        a.get("sumret2", pd.Series(dtype=float)).get(t, 0.0) + b.loc[t, "sumret2"],
    )
    naive = thirty_minute_moments(bars).set_index("endbartime")
    assert panel.loc[t, "sumret2"] < naive.loc[t, "sumret2"] * 0.5  # jump gone
    assert panel.loc[t, "numobs"] == naive.loc[t, "numobs"] - 1  # one minute lost
    # the 10:00 bar (first contract only) is untouched by the roll
    t0 = pd.Timestamp("2024-06-13 10:00")
    assert np.isclose(panel.loc[t0, "sumret2"], naive.loc[t0, "sumret2"])


def test_parent_file_takes_the_previous_days_volume_leader_and_drops_spreads(
    tmp_path,
) -> None:
    """The portal sells every ES contract: the front month is the prior day's volume leader."""
    from live.close_signal.ingest import ingest_es, select_front_by_volume

    rng = np.random.default_rng(3)
    rows = []
    # three ET days; ESM4 leads on day 1, ESU4 on days 2 and 3 -> the FRONT is
    # ESM4 on days 1 and 2 (day 1 takes its own leader), ESU4 on day 3
    lead = {"2024-06-12": "ESM4", "2024-06-13": "ESU4", "2024-06-14": "ESU4"}
    for d, leader in lead.items():
        ts = pd.date_range(f"{d} 09:30", periods=60, freq="1min", tz=ET)
        for sym, iid, base in (("ESM4", 1001, 5000.0), ("ESU4", 1002, 5020.0)):
            close = base * np.exp(np.cumsum(rng.normal(0, 3e-4, size=60)))
            vol = 300 if sym == leader else 100
            for t, c in zip(ts, close):
                rows.append((t, iid, c, vol, sym))
        # a calendar spread, never a candidate
        for t in ts[:5]:
            rows.append((t, 2001, -20.0, 5000, "ESM4-ESU4"))
    df = pd.DataFrame(
        rows, columns=["ts", "instrument_id", "close", "volume", "symbol"]
    )
    df = df.sort_values(["ts", "instrument_id"])
    csv = tmp_path / "es_parent.csv"
    pd.DataFrame(
        {
            "ts_event": df["ts"]
            .dt.tz_convert("UTC")
            .dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "rtype": 32,
            "publisher_id": 1,
            "instrument_id": df["instrument_id"],
            "open": df["close"],
            "high": df["close"],
            "low": df["close"],
            "close": df["close"],
            "volume": df["volume"],
            "symbol": df["symbol"],
        }
    ).to_csv(csv, index=False)
    from live.close_signal.ingest import read_databento_ohlcv_1m

    raw = read_databento_ohlcv_1m(csv)
    assert len(raw) == len(df)  # one row per minute AND instrument, nothing collapsed
    front = select_front_by_volume(raw)
    by_day = front.groupby(front.index.normalize())["symbol"].unique()
    assert list(by_day.loc["2024-06-12"]) == ["ESM4"]
    assert list(by_day.loc["2024-06-13"]) == ["ESM4"]  # yesterday's leader, not today's
    assert list(by_day.loc["2024-06-14"]) == ["ESU4"]
    assert not front["symbol"].str.contains("-").any()
    assert len(front) == 3 * 60
    store = StateStore(tmp_path / "state")
    info = ingest_es(store, csv)
    assert info["selection"] == "front_by_volume" and info["instruments_in_file"] == 3
    assert info["roll_key"] == "instrument_id"
    assert info["rolls"] == [str(pd.Timestamp("2024-06-14 09:30"))]
