"""The operator-maintained FOMC statement dates feed the 14:30 ``fomc release`` flag."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.forecast import fomc_release_rows  # noqa: E402

STATE_CSV = _ROOT / "live" / "close_signal" / "state" / "fomc_statement_dates.csv"


def _stamps(*days: str) -> pd.DatetimeIndex:
    out: list[pd.Timestamp] = []
    for d in days:
        out += pd.date_range(f"{d} 10:00", f"{d} 16:00", freq="30min").tolist()
    return pd.DatetimeIndex(out)


def test_statement_day_flags_its_1430_stamp_only(tmp_path: Path) -> None:
    csv = tmp_path / "fomc.csv"
    csv.write_text("# comment line\n# another\ndate\n2025-09-17\n", encoding="utf-8")
    out = fomc_release_rows(csv, _stamps("2025-09-17", "2025-09-18"))
    flag = out.set_index("endbartime")["fomc release"]
    assert flag.loc["2025-09-17 14:30"] == 1.0
    assert flag.loc["2025-09-18 14:30"] == 0.0  # a session not in the file
    # every other stamp is NaN, as the vendor's releases file has them
    assert flag.drop(["2025-09-17 14:30", "2025-09-18 14:30"]).isna().all()


def test_missing_file_gives_zero_flags(tmp_path: Path) -> None:
    out = fomc_release_rows(tmp_path / "absent.csv", _stamps("2025-09-17"))
    flag = out.set_index("endbartime")["fomc release"]
    assert flag.loc["2025-09-17 14:30"] == 0.0 and flag.notna().sum() == 1


def test_the_shipped_file_parses_and_is_plausible() -> None:
    s = pd.read_csv(STATE_CSV, comment="#")
    t = pd.DatetimeIndex(pd.to_datetime(s["date"]))
    assert len(t) >= 25 and t.is_monotonic_increasing and t.is_unique
    assert (t.dayofweek < 5).all()  # statements fall on weekdays
    gaps = np.diff(t.values).astype("timedelta64[D]").astype(int)
    assert gaps.min() >= 27 and gaps.max() <= 63  # eight meetings a year
    # the vendor feed's last statement (2023-11-01) is followed by the file's first
    assert t.min() == pd.Timestamp("2023-12-13")
    out = fomc_release_rows(STATE_CSV, _stamps("2026-09-16"))
    assert out.set_index("endbartime")["fomc release"].loc["2026-09-16 14:30"] == 1.0
