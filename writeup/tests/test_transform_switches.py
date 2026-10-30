"""Unit tests of the transform switches (src/backtest/executor.py TARGET_* / EXOG_* / FEATURE_WINSOR / PRESCALE,
src/features/transforms/target.py robust_transform(winsor_q, target_map) and rolling_winsorize(q); 2026-10-30,
results/transform_ablation_2026-10-30/RULE.txt section 2 and gate 8(a)).

Every default must leave the transforms as they were: the default calls are compared with the calls the code made
before the switches (the "Was:" lines of the dated comments), on small synthetic frames (no panel load).  Each switch
must change only its own part (target or predictors), and the fixed target_use_diurnal=False path must turn the
target's diurnal adjustment off while src/models/baseline.py's call keeps its former output.  The real-data gate (the
HAR-only design rebuilt md5-equal) is recorded in results/transform_ablation_2026-10-30/invariance/.

Run:  python -m pytest writeup/tests/test_transform_switches.py   (or: python writeup/tests/test_transform_switches.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.backtest.executor as EX  # noqa: E402
import src.data.loading as LD  # noqa: E402
import src.features.transforms.target as TG  # noqa: E402

FLAGS = {
    "TARGET_DIURNAL": True,
    "TARGET_WINSOR": True,
    "TARGET_WINSOR_5_95": False,
    "TARGET_LOG": False,
    "TARGET_LEVEL": False,
    "EXOG_DIURNAL": True,
    "EXOG_SEMANTIC": True,
    "FEATURE_WINSOR": True,
    "PRESCALE": True,
}
TARGET_FLAGS = {"TARGET_DIURNAL": False, "TARGET_WINSOR": False, "TARGET_WINSOR_5_95": True, "TARGET_LOG": True}
TARGET_FLAGS_LEVEL = {"TARGET_LEVEL": True}
EXOG_FLAGS = {"EXOG_DIURNAL": False, "EXOG_SEMANTIC": False, "FEATURE_WINSOR": False}
EXOG = ["sumret2", "sumret", "vix"]  # sqrt by name + diurnal; signed identity + diurnal; log, diurnal excluded
WIN = 240  # the pipeline's target_winsor_window


def frame(seed: int = 0) -> pd.DataFrame:
    """13 regular-hours bars a business day, 2018-09-04 .. 2019-01-31 (the early closes 2018-11-23 and 2018-12-24
    inside, with bars after 13:00 so that B_REGULAR_BARS_ONLY's exclusion acts), a U-shaped intraday pattern, a few
    zero and spiky RV prints; three predictors."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-09-04", "2019-01-31")
    tod = [pd.Timedelta(hours=10) + pd.Timedelta(minutes=30 * k) for k in range(13)]
    t = pd.DatetimeIndex([d + o for d in days for o in tod])
    k = np.tile(np.arange(13), len(days))
    shape = 1.0 + 0.8 * ((k - 6) / 6.0) ** 2
    rv = 1e-6 * shape * np.exp(rng.normal(0, 0.6, len(t)))
    rv[rng.choice(len(t), 40, replace=False)] *= 25.0  # spikes the winsorization clips
    rv[rng.choice(len(t), 5, replace=False)] = 0.0
    df = pd.DataFrame({"t": t, "RV": rv})
    df["sumret2"] = rv * np.exp(rng.normal(0, 0.3, len(t)))
    df["sumret"] = rng.normal(0, 1e-3, len(t)) * np.sqrt(shape)
    df["vix"] = 15.0 + np.cumsum(rng.normal(0, 0.05, len(t)))
    df["time_of_day"] = df["t"].dt.time
    return df


@pytest.fixture
def raw(monkeypatch):
    df = frame()
    monkeypatch.setattr(EX, "load_raw_data", lambda *a, **k: df.copy())
    return df


def lt(exog=EXOG, **kw):
    """load_and_transform as experiments/pool_linear_pretune.py's run_spec_executor calls it (run_executor)."""
    args = dict(
        target_use_diurnal=True,
        target_winsor_window=WIN,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    args.update(kw)
    return EX.load_and_transform("data", list(exog), **args)


def same(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


# ---------------------------------------------------------------------------------------------- the flags
def test_flags_defaults_and_lookup():
    """Every switch is a bool of the executor at its default; pool_linear_pretune's --ex-flag finds it there first
    (spec_section looks in EX, then loading, then target) and no other module has the name."""
    for k, v in FLAGS.items():
        assert isinstance(getattr(EX, k), bool) and getattr(EX, k) is v, k
        assert not hasattr(LD, k) and not hasattr(TG, k), k


# ---------------------------------------------------------------------------------------------- target.py
def test_rolling_winsorize_q_default_and_5_95():
    s = pd.Series(np.random.default_rng(1).standard_t(3, 2000))
    base = TG.rolling_winsorize(s, window=WIN, is_target=True)
    assert same(base, TG.rolling_winsorize(s, window=WIN, is_target=True, q=(TG.WINSOR_LOWER_Q, TG.WINSOR_UPPER_Q)))
    w5 = TG.rolling_winsorize(s, window=WIN, is_target=True, q=(0.05, 0.95))
    assert not same(base, w5)
    assert np.nanmax(np.abs(w5)) <= np.nanmax(np.abs(base))  # 5/95 clips more of the tail


def test_robust_transform_target_map():
    df = frame()
    kw = dict(is_target=True, use_diurnal=True, winsor_window=WIN)
    base, b0 = TG.robust_transform(df, "RV", **kw)
    for m in (None, "sqrt"):  # the name rule's map of "RV" is sqrt
        s, b = TG.robust_transform(df, "RV", target_map=m, **kw)
        assert same(s, base) and same(b, b0), m
    lvl, b1 = TG.robust_transform(df, "RV", target_map="level", is_target=True, use_diurnal=True, winsor_window=0)
    sq, _ = TG.robust_transform(df, "RV", target_map="sqrt", is_target=True, use_diurnal=True, winsor_window=0)
    lg, _ = TG.robust_transform(df, "RV", target_map="log", is_target=True, use_diurnal=True, winsor_window=0)
    assert same(b1, b0)  # B does not depend on the map
    assert same(sq, np.sqrt(lvl)) and same(lg, np.log(lvl.clip(lower=1e-12)))
    assert same(lvl, df["RV"] / b0)
    with pytest.raises(ValueError):
        TG.robust_transform(df, "RV", target_map="cbrt", **kw)


def test_robust_transform_winsor_q_default():
    df = frame()
    a, _ = TG.robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=WIN)
    b, _ = TG.robust_transform(
        df, "RV", is_target=True, use_diurnal=True, winsor_window=WIN, winsor_q=(TG.WINSOR_LOWER_Q, TG.WINSOR_UPPER_Q)
    )
    c, _ = TG.robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=WIN, winsor_q=(0.05, 0.95))
    assert same(a, b) and not same(a, c)


# ---------------------------------------------------------------------------------------------- load_and_transform
def legacy_target(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The target call before 2026-10-30 with target_use_diurnal=True, target_winsor_window=240."""
    kw: dict = {"is_target": True, "use_diurnal": True, "winsor_window": WIN}
    excl = EX.early_close_post_bars(df["t"])
    assert excl.any()
    kw["baseline_exclude"] = excl
    return TG.robust_transform(df, "RV", **kw)


def test_defaults_equal_legacy_calls(raw):
    out, cols = lt()
    a, b = legacy_target(raw)
    assert same(out["adj_RV"], a) and same(out["baseline"], b)
    for c in EXOG:
        ref, _ = TG.robust_transform(raw, c, use_transform=True, use_diurnal=True, diurnal_mode="divide")
        ok = np.isfinite(ref.to_numpy())  # the rows the causal neutral fill leaves alone
        assert ok.sum() > 0.9 * len(ref)
        assert same(out[f"adj_{c}"].to_numpy()[ok], ref.to_numpy()[ok]), c
    # the frame as a whole equals an explicit "every switch at its default" run
    out2, cols2 = lt()
    assert cols == cols2 and out.equals(out2)


def test_target_use_diurnal_false_is_honoured(raw):
    out, _ = lt(target_use_diurnal=False)
    ref, _ = TG.robust_transform(raw, "RV", is_target=True, use_diurnal=False, winsor_window=WIN)
    assert (out["baseline"] == 1.0).all()
    assert same(out["adj_RV"], ref)
    dflt, _ = lt()
    assert not same(out["adj_RV"], dflt["adj_RV"])


def test_baseline_py_call_unchanged(raw):
    """src/models/baseline.py now asks for target_use_diurnal=True, target_b_exclude=False: the former output of its
    target_use_diurnal=False call (diurnal adjustment on -- robust_transform's default -- and no B exclusions)."""
    out, _ = lt(exog=[], target_use_diurnal=True, target_winsor_window=None, target_b_exclude=False)
    ref, b = TG.robust_transform(raw, "RV", is_target=True)  # the former call: {"is_target": True} only
    assert same(out["adj_RV"], ref) and same(out["baseline"], b)
    excl, _ = lt(exog=[], target_use_diurnal=True, target_winsor_window=None)
    assert not same(excl["baseline"], b)  # the B exclusion matters on this frame, so the test has power


@pytest.mark.parametrize("flags", [*({k: v} for k, v in TARGET_FLAGS.items()), TARGET_FLAGS_LEVEL])
def test_target_switch_leaves_predictors(raw, monkeypatch, flags):
    base, cols0 = lt()
    for k, v in flags.items():
        monkeypatch.setattr(EX, k, v)
    out, cols = lt()
    assert cols == cols0
    assert not same(out["adj_RV"], base["adj_RV"])
    for c in cols:
        assert same(out[c], base[c]), c
    if "TARGET_DIURNAL" in flags:
        assert (out["baseline"] == 1.0).all()
    else:
        assert same(out["baseline"], base["baseline"])


def test_target_switch_values(raw, monkeypatch):
    a, b = legacy_target(raw)
    kw: dict = {"is_target": True, "use_diurnal": True, "baseline_exclude": EX.early_close_post_bars(raw["t"])}
    cases = {
        "TARGET_WINSOR": (False, dict(kw, winsor_window=0)),
        "TARGET_WINSOR_5_95": (True, dict(kw, winsor_window=WIN, winsor_q=(0.05, 0.95))),
        "TARGET_LOG": (True, dict(kw, winsor_window=WIN, target_map="log")),
        "TARGET_LEVEL": (True, dict(kw, winsor_window=WIN, target_map="level")),
    }
    for k, (v, rkw) in cases.items():
        monkeypatch.setattr(EX, k, v)
        out, _ = lt(exog=[])
        ref, rb = TG.robust_transform(raw, "RV", **rkw)
        assert same(out["adj_RV"], ref) and same(out["baseline"], rb), k
        monkeypatch.setattr(EX, k, FLAGS[k])
    out, _ = lt(exog=[])
    assert same(out["adj_RV"], a) and same(out["baseline"], b)


@pytest.mark.parametrize("flag", sorted(EXOG_FLAGS))
def test_exog_switch_leaves_target(raw, monkeypatch, flag):
    base, cols0 = lt()
    monkeypatch.setattr(EX, flag, EXOG_FLAGS[flag])
    out, cols = lt()
    assert same(out["adj_RV"], base["adj_RV"]) and same(out["baseline"], base["baseline"])
    changed = [c for c in EXOG if not same(out[f"adj_{c}"], base[f"adj_{c}"])]
    assert changed, flag
    if flag == "EXOG_DIURNAL":  # vix is diurnally excluded by name: untouched
        assert "vix" not in changed and "sumret2" in changed
    if flag == "EXOG_SEMANTIC":  # the maps of sumret2 (sqrt) and vix (log) are gone
        assert {"sumret2", "vix"} <= set(changed)
    ref = {
        "EXOG_DIURNAL": dict(use_transform=True, use_diurnal=False),
        "EXOG_SEMANTIC": dict(use_transform=False, use_diurnal=True),
        "FEATURE_WINSOR": dict(use_transform=True, use_diurnal=True, winsor_window=0),
    }[flag]
    for c in EXOG:
        r, _ = TG.robust_transform(raw, c, diurnal_mode="divide", **ref)
        ok = np.isfinite(r.to_numpy())
        assert same(out[f"adj_{c}"].to_numpy()[ok], r.to_numpy()[ok]), (flag, c)


def test_exclusive_target_flags(raw, monkeypatch):
    monkeypatch.setattr(EX, "TARGET_LOG", True)
    monkeypatch.setattr(EX, "TARGET_LEVEL", True)
    with pytest.raises(AssertionError):
        lt(exog=[])
    monkeypatch.setattr(EX, "TARGET_LOG", False)
    monkeypatch.setattr(EX, "TARGET_LEVEL", False)
    monkeypatch.setattr(EX, "TARGET_WINSOR", False)
    monkeypatch.setattr(EX, "TARGET_WINSOR_5_95", True)
    with pytest.raises(AssertionError):
        lt(exog=[])


# ---------------------------------------------------------------------------------------------- _backtest_and_save
@pytest.mark.parametrize("on", [True, False])
def test_prescale_switch(monkeypatch, tmp_path, on):
    """PRESCALE off: the caller's prescale=True no longer scales (nor runs the two steps on the scaled matrix); on:
    the scaling path runs as before."""
    rng = np.random.default_rng(2)
    n, p = 300, 3
    df = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{j}" for j in range(p)])
    df["adj_RV"] = rng.normal(size=n)
    df["baseline"] = 1.0
    df["t"] = pd.date_range("2020-01-02 10:00", periods=n, freq="30min")
    calls: list[str] = []
    seen: dict = {}

    def scale(X, w, ref_iqr=None, fixed_cols=None):
        calls.append("scale")
        return X * 2.0

    monkeypatch.setattr(EX, "PRESCALE", on)
    monkeypatch.setattr(EX, "_build_scale_guards", lambda *a, **k: calls.append("guards") or (None, None))
    monkeypatch.setattr(EX, "rolling_robust_scale", scale)
    monkeypatch.setattr(EX, "late_start_warmup", lambda X, *a, **k: calls.append("warmup") or X)
    monkeypatch.setattr(EX, "unobserved_neutral", lambda X, *a, **k: calls.append("neutral") or X)
    monkeypatch.setattr(EX, "check_and_report", lambda *a, **k: None)

    def fp(X, y, w, hp):
        seen["X"] = np.array(X)
        return np.zeros(len(X) - w)

    names = [f"f{j}" for j in range(p)]
    EX._backtest_and_save(df, names, fp, {}, 50, 1, 0, -1, 0, str(tmp_path / "r.csv"), prescale=True, burn_in=0)
    raw_x = df[names].to_numpy(float)
    if on:
        assert calls == ["guards", "scale", "warmup", "neutral"] and same(seen["X"], raw_x * 2.0)
    else:
        assert calls == [] and same(seen["X"], raw_x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
