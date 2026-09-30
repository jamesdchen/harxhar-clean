"""The backfill: recent sessions without ES rows are fetched and appended (2026-09-30)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal import run  # noqa: E402


class _Store:
    def __init__(self, days: list[str]) -> None:
        self.panel = pd.DataFrame(
            {
                "endbartime": [pd.Timestamp(f"{d} 15:30") for d in days],
                "source": ["databento_es"] * len(days),
            }
        )
        self.appended: list[pd.Timestamp] = []

    def load_panel(self) -> pd.DataFrame:
        return self.panel

    def append_panel(self, rows: pd.DataFrame, source: str) -> int:
        assert source == "yahoo_es"
        self.appended.append(pd.Timestamp(rows["day"].iloc[0]))
        return len(rows)


def test_backfill_fills_the_skipped_sessions_only(monkeypatch) -> None:
    # 2026-09-25 (Fri), 09-28 (Mon), 09-29 (Tue) skipped; 09-26/27 are a weekend
    store = _Store(["2026-09-22", "2026-09-23", "2026-09-24", "2026-09-30"])
    fetched: list[pd.Timestamp] = []

    def es(d):  # type: ignore[no-untyped-def]
        fetched.append(pd.Timestamp(d))
        return SimpleNamespace(frame=pd.DataFrame())

    monkeypatch.setattr(run.feeds, "es_minute_bars", es)
    monkeypatch.setattr(run.feeds, "cboe_prints", lambda d: {})
    monkeypatch.setattr(run, "panel_rows", lambda e, c, d: pd.DataFrame({"day": [d]}))
    filled = run.backfill_sessions(store, pd.Timestamp("2026-10-01"))  # type: ignore[arg-type]
    assert filled == ["2026-09-25", "2026-09-28", "2026-09-29"]
    assert store.appended == [pd.Timestamp(d) for d in filled]


def test_backfill_never_raises(monkeypatch) -> None:
    store = _Store(["2026-09-24"])

    def boom(d):  # type: ignore[no-untyped-def]
        raise RuntimeError("feed down")

    monkeypatch.setattr(run.feeds, "es_minute_bars", boom)
    assert run.backfill_sessions(store, pd.Timestamp("2026-09-30")) == []  # type: ignore[arg-type]
