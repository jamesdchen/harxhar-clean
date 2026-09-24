"""The 13-arm yhat table, the deck loader's call shape, and the placeholder comparison.

No network and no spec run: synthetic per-bar CSVs and a fake notebook library.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal import forecast  # noqa: E402
from live.close_signal.common import ET  # noqa: E402

DAYS = pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-06"])


def _bar_clock(bar: str) -> tuple[int, int]:
    return int(bar[3:5]), int(bar[5:7])


def _write_arm(
    path: Path, bar: str, seed: int, winsor_day: int | None = None
) -> pd.DataFrame:
    """A synthetic results_<bar>.csv in the spec's layout; returns the expected rows."""
    rng = np.random.default_rng(seed)
    hh, mm = _bar_clock(bar)
    rows = []
    for k, d in enumerate(DAYS):
        t = d + pd.Timedelta(hours=hh, minutes=mm)
        true_raw = float(rng.uniform(1e-6, 4e-6))
        true_adj = float(rng.uniform(0.7, 1.3))
        pred_adj = float(rng.uniform(0.7, 1.3))
        if winsor_day is not None and k == winsor_day:
            true_raw *= 0.5  # the spec's winsorized target: differs from the panel's RV
        rows.append(
            {
                "date": t,
                "horizon": 1,
                "true_adj": true_adj,
                "pred_adj": pred_adj,
                "true_raw": true_raw,
                "pred_raw": pred_adj**2 * true_raw / true_adj**2,
            }
        )
    # one row the table must drop: a non-positive target
    rows.append(
        {
            "date": DAYS[0] + pd.Timedelta(hours=hh, minutes=mm) - pd.Timedelta(days=7),
            "horizon": 1,
            "true_adj": 0.0,
            "pred_adj": 1.0,
            "true_raw": 0.0,
            "pred_raw": 0.0,
        }
    )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _ext_root(tmp_path: Path, panel_rv: dict[pd.Timestamp, float]) -> Path:
    root = tmp_path / "ext"
    (root / "data").mkdir(parents=True)
    core = pd.DataFrame(
        {
            "endbartime": list(panel_rv.keys()),
            "sumret2": list(panel_rv.values()),
        }
    )
    core.to_parquet(root / "data" / "core_stats.parquet", index=False)
    return root


def test_read_arm_is_build_subsection_yhat_read_arm(tmp_path):
    csv = tmp_path / "results_bar1600.csv"
    raw = _write_arm(csv, "bar1600", seed=1)
    tab = forecast.read_arm(csv)
    kept = raw[(raw["true_adj"] > 0) & (raw["true_raw"] > 0)]
    assert len(tab) == len(kept) == len(DAYS)
    et = pd.DatetimeIndex(kept["date"]).tz_localize(ET)
    assert list(tab["t"]) == list(et.tz_convert("UTC").as_unit("us"))
    assert str(tab["t"].dtype) == "datetime64[us, UTC]"
    assert np.allclose(tab["yhat"], kept["pred_adj"])
    assert np.allclose(tab["baseline"], kept["true_raw"] / kept["true_adj"] ** 2)
    assert np.allclose(tab["rv_raw"], kept["true_raw"])


def test_assemble_stacks_13_bars_with_the_panels_rv(tmp_path):
    arms: dict[str, Path] = {}
    panel_rv: dict[pd.Timestamp, float] = {}
    expected = []
    for i, bar in enumerate(forecast.BARS):
        csv = tmp_path / "arms" / bar / f"results_{bar}.csv"
        raw = _write_arm(csv, bar, seed=10 + i, winsor_day=1)
        arms[bar] = csv
        hh, mm = _bar_clock(bar)
        for k, d in enumerate(DAYS):
            t = d + pd.Timedelta(hours=hh, minutes=mm)
            rv = float(raw.loc[k, "true_raw"]) * (
                2.0 if k == 1 else 1.0
            )  # the panel's RV
            panel_rv[t] = rv
            expected.append(
                (
                    pd.Timestamp(t, tz=ET).tz_convert("UTC").as_unit("us"),
                    float(raw.loc[k, "pred_adj"]),
                    float(raw.loc[k, "true_raw"] / raw.loc[k, "true_adj"] ** 2),
                    rv,
                )
            )
    root = _ext_root(tmp_path, panel_rv)
    tab = forecast.assemble_yhat_table(arms, root)
    exp = (
        pd.DataFrame(expected, columns=["t", "yhat", "baseline", "rv_raw"])
        .sort_values("t")
        .reset_index(drop=True)
    )
    exp["t"] = pd.DatetimeIndex(exp["t"]).as_unit("us")  # the tables' resolution
    assert len(tab) == 13 * len(DAYS)
    assert tab["t"].is_monotonic_increasing and not tab["t"].duplicated().any()
    pd.testing.assert_frame_equal(tab, exp, check_exact=False, rtol=1e-12)
    # the winsorized day carries the PANEL's rv, not the arm's true_raw
    day1 = tab[
        tab["t"].dt.tz_convert(ET).dt.normalize().dt.tz_localize(None) == DAYS[1]
    ]
    raw_1600 = pd.read_csv(arms["bar1600"])
    assert not np.isclose(
        day1[day1["t"].dt.tz_convert(ET).dt.hour == 16]["rv_raw"].iloc[0],
        raw_1600.loc[1, "true_raw"],
    )


def test_with_panel_rv_refuses_a_stamp_the_panel_lacks(tmp_path):
    csv = tmp_path / "results_bar1000.csv"
    _write_arm(csv, "bar1000", seed=3)
    tab = forecast.read_arm(csv)
    rv = forecast.panel_rv(
        _ext_root(tmp_path, {DAYS[0] + pd.Timedelta(hours=10): 1e-6})
    )
    with pytest.raises(RuntimeError, match="no panel realized variance"):
        forecast.with_panel_rv(tab, rv)


def test_recalibrate_calls_the_decks_loader_and_picks_the_session(
    tmp_path, monkeypatch
):
    calls: list[dict] = []

    def fake_loader(tag, path, need_dates, cache, method="mean"):
        calls.append(
            {
                "tag": tag,
                "path": Path(path),
                "need": list(need_dates),
                "cache": Path(cache),
                "method": method,
            }
        )
        idx = pd.DatetimeIndex(DAYS)
        return pd.DataFrame(
            {
                "yhat": [1.0, 1.1, 1.2],
                "baseline": [2e-6, 2e-6, 2e-6],
                "rv_raw": [1e-6, 3e-6, 2e-6],
                "rv_hat": [3e-6, 4e-6, 5e-6],
                "m": [1.0, 1.05, 1.1],
                "s2": [0.1, 0.1, 0.1],
                "yhat_vol": [0.0, 0.0, 0.0],
                "m_vol": [0.0, 0.0, 0.0],
            },
            index=idx,
        )

    fake = types.ModuleType("atm_straddle_lib")
    fake.load_yhat_1530_mz_cached = fake_loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atm_straddle_lib", fake)
    table = pd.DataFrame(
        {
            "t": pd.DatetimeIndex(DAYS + pd.Timedelta(hours=16))
            .tz_localize(ET)
            .tz_convert("UTC"),
            "yhat": [1.0, 1.1, 1.2],
            "baseline": [2e-6] * 3,
            "rv_raw": [1e-6, 3e-6, 2e-6],
        }
    )
    out = forecast.rv_hat_for(DAYS[1], table, _ROOT, tmp_path)
    assert out == {
        "rv_hat": 4e-6,
        "yhat": 1.1,
        "baseline": 2e-6,
        "mz_m": 1.05,
        "mz_s2": 0.1,
    }
    c = calls[0]
    assert c["method"] == "mean" and c["need"] == [DAYS[1]]
    assert c["path"].exists() and c["cache"].is_dir()
    saved = pd.read_parquet(c["path"])
    assert str(saved["t"].dtype).startswith("datetime64[") and "UTC" in str(
        saved["t"].dtype
    )
    with pytest.raises(RuntimeError, match="no 16:00 row"):
        forecast.rv_hat_for(pd.Timestamp("2024-03-07"), table, _ROOT, tmp_path)


def test_table_deviation_and_the_placeholder_comparison():
    t = (
        pd.DatetimeIndex(DAYS + pd.Timedelta(hours=16))
        .tz_localize(ET)
        .tz_convert("UTC")
        .as_unit("us")
    )
    a = pd.DataFrame(
        {
            "t": t,
            "yhat": [1.0, 1.1, 1.2],
            "baseline": [2e-6, 3e-6, 4e-6],
            "rv_raw": [1e-6] * 3,
        }
    )
    b = a.copy()
    dev = forecast.table_deviation(a, b)
    assert (
        dev["rows"] == 3
        and dev["max_rel_yhat"] == 0.0
        and dev["max_rel_baseline"] == 0.0
    )
    # a later row may differ (the future is not part of the claim); rows up to the session must not
    b.loc[2, "yhat"] = 9.0
    upto = DAYS[1] + pd.Timedelta(hours=16)
    assert forecast.table_deviation(a, b, upto=upto)["max_rel_yhat"] == 0.0
    # relative to the second table (the run with the placeholder)
    assert forecast.table_deviation(a, b)["max_rel_yhat"] == pytest.approx(
        (9.0 - 1.2) / 9.0
    )
    b.loc[1, "baseline"] = 3.3e-6
    assert forecast.table_deviation(a, b, upto=upto)[
        "max_rel_baseline"
    ] == pytest.approx(0.3e-6 / 3.3e-6)


def test_bars_and_the_run_arm_environment(tmp_path, monkeypatch):
    assert (
        forecast.BARS[0] == "bar1000"
        and forecast.BARS[-1] == "bar1600"
        and len(forecast.BARS) == 13
    )
    seen: dict = {}

    def fake_call(cmd, cwd, env, stdout, stderr):
        seen.update(env)
        out = (
            Path(env["HPC_RESULT_DIR"])
            / "causal_tune_linear"
            / forecast.ESTIMATOR
            / env["HPC_KW_EXOG_BUCKET"]
        )
        out.mkdir(parents=True)
        (out / f"results_{env['HPC_KW_SEGMENT']}.csv").write_text("date\n")
        return 0

    monkeypatch.setattr(forecast.subprocess, "call", fake_call)
    root = tmp_path / "root"
    (root / "specs").mkdir(parents=True)
    forecast.run_arm(root, tmp_path / "res", segment="bar1430", bucket="live_feasible")
    assert (
        seen["HPC_KW_SEGMENT"] == "bar1430"
        and seen["HPC_KW_EXOG_BUCKET"] == "live_feasible"
    )
    assert (seen["HPC_KW_START"], seen["HPC_KW_END"], seen["HPC_KW_HALO"]) == (
        "0",
        "-1",
        "0",
    )
    assert seen["HPC_KW_LAG_SCOPE"] == "global" and seen["HPC_KW_TRAIN_WIN"] == "2000"
