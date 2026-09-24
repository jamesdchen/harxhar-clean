"""The XSP definition selector: the expiry day comes from the OCC symbol, not the UTC-midnight stamp."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.pull_databento_xsp import (  # noqa: E402
    et_window,
    quote_file,
    occ_expiry,
    zero_dte,
)


def _defs() -> pd.DataFrame:
    # Databento stamps the expiry DATE at 00:00 UTC (= 20:00 ET the evening before)
    return pd.DataFrame(
        {
            "raw_symbol": [
                "XSP   230330C00400000",
                "XSP   230330P00400000",
                "XSP   230331C00400000",  # tomorrow's contract
                "XSP   230330C00400500",  # a half-point strike
                "XSP   230330X00000000",  # not an option class
            ],
            "instrument_class": ["C", "P", "C", "C", "X"],
            "strike_price": [400.0, 400.0, 400.0, 400.5, float("nan")],
            "expiration": pd.to_datetime(
                ["2023-03-30", "2023-03-30", "2023-03-31", "2023-03-30", "2023-03-30"],
                utc=True,
            ),
        }
    )


def test_occ_symbol_dates_and_the_selector():
    d = _defs()
    assert list(occ_expiry(d["raw_symbol"]).dt.strftime("%Y-%m-%d")) == [
        "2023-03-30",
        "2023-03-30",
        "2023-03-31",
        "2023-03-30",
        "2023-03-30",
    ]
    z = zero_dte(d, pd.Timestamp("2023-03-30"))
    assert sorted(z["raw_symbol"]) == [
        "XSP   230330C00400000",
        "XSP   230330C00400500",
        "XSP   230330P00400000",
    ]
    # the day AFTER: only tomorrow's contract; a Friday is never empty
    assert list(zero_dte(d, pd.Timestamp("2023-03-31"))["raw_symbol"]) == [
        "XSP   230331C00400000"
    ]


def test_quote_window_is_1520_to_1616_et_in_utc():
    lo, hi = et_window(pd.Timestamp("2023-03-30"))  # EDT: UTC-4
    assert (str(lo), str(hi)) == (
        "2023-03-30 19:20:00+00:00",
        "2023-03-30 20:16:00+00:00",
    )
    lo, hi = et_window(pd.Timestamp("2024-01-10"))  # EST: UTC-5
    assert (str(lo), str(hi)) == (
        "2024-01-10 20:20:00+00:00",
        "2024-01-10 21:16:00+00:00",
    )


def test_a_custom_window_gets_its_own_file_outside_the_default_glob(tmp_path) -> None:
    assert quote_file(tmp_path, "2024-01-10").name == "cbbo1m_2024-01-10.parquet"
    f = quote_file(tmp_path, "2024-01-10", ("15:00", "15:21"))
    assert f.name == "cbbo1m-1500-1521_2024-01-10.parquet"
    assert not f.match("cbbo1m_*.parquet")
    lo, hi = et_window(pd.Timestamp("2024-01-10"), ("15:00", "15:21"))  # EST
    assert (lo.hour, lo.minute, hi.hour, hi.minute) == (20, 0, 20, 21)
