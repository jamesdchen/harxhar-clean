"""The fast path's pieces that run in seconds: the spec reading, the farm's
fallbacks, the parallel feeds and the prep / precompute timing.

The equality of the fast path with the path of record (13 spec processes) is
the job of ``python -m live.close_signal.fastpath_check --check`` (minutes).
"""

from __future__ import annotations

import ast
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(_ROOT / "notebooks"))

from live.close_signal import arms_shared, feeds  # noqa: E402
from live.close_signal.schedule import precompute_plan, prep_plan  # noqa: E402

SPEC = _ROOT / "specs" / "causal_tune_linear.py"


# ------------------------------------------------------------ spec reading --
def test_the_spec_definitions_leave_out_the_demo_cells_and_the_runs() -> None:
    defs, setup, call = arms_shared.spec_parts(SPEC)
    names = {
        n.name for n in defs.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    assert {"RollingTunedLinear", "fit_predict_lin_tuned", "_batch_theta"} <= names
    called = {
        c.func.id
        for n in defs.body
        for c in ast.walk(n)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    # the pinned demo cells and every executor run are gone
    assert not called & {"load_raw_data", "robust_transform", "run_executor"}
    assert not any(isinstance(n, ast.For) for n in defs.body), (
        "the arm loop must not be part of the definitions"
    )
    assigned = [
        t.id
        for n in defs.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    for const in ("HORIZON", "TRAIN_WIN", "SEGMENT", "LAG_SCOPE", "RESULTS_ROOT"):
        assert const in assigned
    assert "df" not in assigned


def test_the_arm_call_is_read_off_the_loop_with_its_setup() -> None:
    _, setup, call = arms_shared.spec_parts(SPEC)
    kws = {k.arg for k in call.keywords}
    assert {
        "fit_predict",
        "data_path",
        "output_file",
        "train_window",
        "exog_cols",
        "segment",
        "lag_scope",
        "prescale",
        "impute_indicate",
        "start",
        "end",
        "halo",
    } <= kws
    assert not call.args
    targets = [ast.unparse(s.targets[0]) for s in setup]  # type: ignore[attr-defined]
    assert targets == ["RollingTunedLinear.grid", "out_csv"]


# ------------------------------------------------------------- the farm --
class _FakeEx:
    """Stands in for src.backtest.executor inside a (thread) worker."""

    @staticmethod
    def robust_transform(df: pd.DataFrame, col: str, **kw: Any) -> Any:
        s = df[col] * 2.0 + (df[kw.get("time_col", "time_of_day")] == "a")
        return s, pd.Series(1.0, index=df.index)

    @staticmethod
    def add_calendar_features(df: pd.DataFrame) -> list[str]:
        df["is_close"] = (df["t"].dt.hour >= 16).astype(np.int8)
        return ["is_close"]

    @staticmethod
    def add_expiry_features(df: pd.DataFrame) -> list[str]:
        df["is_opex"] = (df["t"].dt.day == 18).astype(np.int8)
        df["days_to_opex"] = df["t"].dt.day.astype(float) - 18.0
        return ["is_opex", "days_to_opex"]


def _frame() -> pd.DataFrame:
    t = pd.date_range("2026-09-17 15:00", periods=8, freq="30min")
    return pd.DataFrame(
        {
            "t": t,
            "x": np.arange(8.0),
            "y": np.arange(8.0) ** 2,
            "time_of_day": list("abababab"),
        }
    )


@pytest.fixture()
def pool(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setitem(arms_shared._WORKER, "ex", _FakeEx)
    with ThreadPoolExecutor(2) as ex:
        yield ex


def test_a_farmed_transform_equals_the_inline_one(pool: Any) -> None:
    plan: list[arms_shared.Key] = [
        ("x", (("use_diurnal", True),)),
        ("y", (("use_diurnal", True),)),
    ]
    farm = arms_shared._Farm(pool, 2, plan)
    rt = farm.robust_transform(_FakeEx.robust_transform)
    df = _frame()
    got = [rt(df, c, use_diurnal=True)[0] for c in ("x", "y")]
    want = [_FakeEx.robust_transform(df, c, use_diurnal=True)[0] for c in ("x", "y")]
    assert farm.farmed == ["x", "y"] and farm.inline == []
    for g, w in zip(got, want):
        pd.testing.assert_series_equal(g, w, check_names=False)
    assert farm.calls == plan


def test_a_changed_column_or_new_keywords_run_inline(pool: Any) -> None:
    plan: list[arms_shared.Key] = [
        ("x", (("use_diurnal", True),)),
        ("y", (("use_diurnal", True),)),
    ]
    farm = arms_shared._Farm(pool, 2, plan)
    rt = farm.robust_transform(_FakeEx.robust_transform)
    df = _frame()
    rt(df, "x", use_diurnal=True)  # submits the plan
    df.loc[3, "y"] = -1.0  # the column moved after it was sent
    s, _ = rt(df, "y", use_diurnal=True)
    assert farm.inline == ["y"] and s.iloc[3] == -2.0
    rt(df, "x", use_diurnal=False)  # not in the plan
    assert farm.inline == ["y", "x"]


def test_no_plan_or_farm_off_is_all_inline(pool: Any) -> None:
    for farm in (
        arms_shared._Farm(pool, 2, []),
        arms_shared._Farm(pool, 2, [("x", ())], enabled=False),
    ):
        rt = farm.robust_transform(_FakeEx.robust_transform)
        rt(_frame(), "x")
        assert farm.farmed == [] and farm.inline == ["x"]


def test_the_farmed_expiry_columns_equal_the_extractor(pool: Any) -> None:
    farm = arms_shared._Farm(pool, 2, [("x", ())])
    rt = farm.robust_transform(_FakeEx.robust_transform)
    exp = farm.expiry(_FakeEx.add_expiry_features)
    df = _frame()
    rt(df, "x")  # the first call also sends the stamps
    _FakeEx.add_calendar_features(df)
    names = exp(df)
    ref = _frame()
    _FakeEx.add_calendar_features(ref)
    want = _FakeEx.add_expiry_features(ref)
    assert names == want and "expiry" in farm.farmed
    pd.testing.assert_frame_equal(df, ref)


def test_expiry_runs_inline_when_the_stamps_moved(pool: Any) -> None:
    farm = arms_shared._Farm(pool, 2, [("x", ())])
    rt = farm.robust_transform(_FakeEx.robust_transform)
    exp = farm.expiry(_FakeEx.add_expiry_features)
    df = _frame()
    rt(df, "x")
    df = df.iloc[1:].reset_index(drop=True)
    _FakeEx.add_calendar_features(df)
    exp(df)
    assert farm.inline == ["expiry"]


# ------------------------------------------------------------ the feeds --
def _dl_factory(fail: set[str]) -> Any:
    def dl(symbol: str, start: Any = None, end: Any = None, **_: Any) -> pd.DataFrame:
        if symbol in fail:
            raise RuntimeError("HTTP 429")
        idx = pd.date_range(
            "2026-09-23 15:00", periods=31, freq="1min", tz="America/New_York"
        )
        c = np.linspace(100.0, 101.0, len(idx))
        return pd.DataFrame(
            {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0}, index=idx
        )

    return dl


def test_fetch_all_returns_every_feed_and_keeps_failures_apart() -> None:
    got = feeds.fetch_all(pd.Timestamp("2026-09-23"), downloader=_dl_factory({"^GSPC"}))
    assert set(got) == set(feeds.ALL_FEEDS)
    assert feeds.take(got, "es").symbol == "ES=F"
    assert feeds.take(got, "vix3m").symbol == "^VIX3M"
    with pytest.raises(feeds.FeedError, match="429"):
        feeds.take(got, "spx")


def test_the_last_complete_close_skips_the_minute_in_progress() -> None:
    idx = pd.date_range("2026-09-23 14:58", periods=3, freq="1min")
    b = feeds.Bars1m(
        symbol="^GSPC",
        frame=pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx),
        fetched_at=pd.Timestamp("2026-09-23 15:00:20"),
        nominal_delay_min=0,
    )
    assert feeds.last_complete_close(b) == 2.0  # the 15:00 minute is still open


# ------------------------------------------------- prep / precompute timing --
def _et(h: int, m: int, s: int = 0) -> datetime:
    # 2026-09-24 is EDT (UTC-4)
    return datetime(2026, 9, 24, h + 4, m, s, tzinfo=timezone.utc)


def test_prep_waits_for_1500_posts_until_1525_and_skips_replays() -> None:
    today = date(2026, 9, 24)
    kind, s = prep_plan(today, _et(12, 17))
    assert kind == "sleep" and s == pytest.approx(2 * 3600 + 43 * 60)
    assert prep_plan(today, _et(15, 0))[0] == "now"
    assert prep_plan(today, _et(15, 24, 59))[0] == "now"
    assert prep_plan(today, _et(15, 25, 1))[0] == "skip"
    assert prep_plan(date(2026, 9, 23), _et(12, 17))[0] == "skip"  # a replay


def test_precompute_runs_at_1515_and_not_too_close_to_the_stamp() -> None:
    today = date(2026, 9, 24)
    kind, s = precompute_plan(today, _et(14, 17), "15:30:30")
    assert kind == "sleep" and s == pytest.approx(58 * 60)
    assert precompute_plan(today, _et(15, 20), "15:30:30")[0] == "now"
    assert precompute_plan(today, _et(15, 29), "15:30:30")[0] == "skip"
    assert precompute_plan(date(2026, 9, 23), _et(14, 17), "15:30:30")[0] == "skip"


def test_arm_env_is_the_arm_environment_with_segment_all() -> None:
    env = arms_shared.arm_env("free_vix_only", "ridge", 2000, Path("x"))
    assert env["HPC_KW_SEGMENT"] == "all" and env["HPC_KW_LAG_SCOPE"] == "global"
    assert env["OMP_NUM_THREADS"] == "1" and env["HPC_KW_TRAIN_WIN"] == "2000"
    assert (env["HPC_KW_START"], env["HPC_KW_END"], env["HPC_KW_HALO"]) == (
        "0",
        "-1",
        "0",
    )
    assert str(_ROOT) in env["PYTHONPATH"].split(os.pathsep)


def test_clocks_may_omit_the_seconds() -> None:
    from live.close_signal.schedule import ET_TZ, seconds_until

    now = _et(14, 59, 30).astimezone(ET_TZ)  # seconds_until reads an ET clock
    assert seconds_until("15:00", now) == seconds_until("15:00:00", now) == 30.0
    assert prep_plan(date(2026, 9, 24), now, at="15:00")[0] == "sleep"
