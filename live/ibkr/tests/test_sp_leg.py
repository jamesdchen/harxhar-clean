"""The S&P-leg report (studies 67-68): causal, bounded, a report and not an order."""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pytest

from live.ibkr.config import Config
from live.ibkr.premium_ledger import ClockRecord, PremiumLedger
from live.ibkr.sp_leg import (
    HAR_LONG,
    SP_LEG_MIN_SESSIONS,
    har_variance_forecasts,
    sp_leg_target,
    vol_managed_weight,
)

SEED = os.path.join("results", "live_seed", "premium_ledger.parquet")
needs_seed = pytest.mark.skipif(
    not os.path.exists(SEED), reason="the seed premium ledger has not been built"
)


def ledger_with_variance(rv: np.ndarray, last_day: dt.date) -> PremiumLedger:
    """A ledger whose 10:00 realized remaining-window variance is ``rv``, one value a session."""
    led = PremiumLedger()
    days: list[dt.date] = []
    day = last_day
    while len(days) < len(rv):
        if day.weekday() < 5:
            days.append(day)
        day -= dt.timedelta(days=1)
    days.reverse()
    led.append_session(
        [
            ClockRecord(
                session=d,
                clock="10:00",
                implied_rem_var=float(v),
                realized_rem_var=float(v),
                spot=5000.0,
                premium_mid=20.0,
                kc=5005.0,
                kp=5000.0,
            )
            for d, v in zip(days, rv, strict=True)
        ]
    )
    return led


def test_har_forecasts_are_one_step_ahead_and_warm_up():
    rng = np.random.default_rng(0)
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = 0.9 * x[t - 1] + rng.normal(scale=0.3)
    rv = np.exp(x - 10.0)
    f = har_variance_forecasts(rv, 100)
    assert f.shape == (401,)
    # nothing before HAR_LONG + 100 fitted pairs; the last element is the
    # forecast for the session AFTER the series
    assert (
        np.isnan(f[: HAR_LONG + 100]).all() and np.isfinite(f[HAR_LONG + 100 :]).all()
    )
    # a forecast made from rv[:t] only: changing rv[t:] never changes f[t]
    g = har_variance_forecasts(np.concatenate([rv[:300], rv[300:] * 50.0]), 100)
    assert np.allclose(f[:301], g[:301], equal_nan=True)
    assert not np.allclose(f[301:], g[301:])
    # it tracks the level: a persistent series is forecast near its last value
    assert (
        abs(np.corrcoef(f[HAR_LONG + 100 : 400], np.log(rv[HAR_LONG + 100 :]))[0, 1])
        > 0.7
    )


def test_vol_managed_weight_is_causal_bounded_and_falls_when_variance_rises():
    rng = np.random.default_rng(1)
    base = np.exp(rng.normal(-10.0, 0.2, size=700))
    last = dt.date(2025, 6, 30)
    calm = ledger_with_variance(base, last)
    w_calm, info = vol_managed_weight(calm, last + dt.timedelta(days=1))
    assert info["state"] in ("full", "reduced") and 0.0 < w_calm <= 1.0
    # the same history with the last month at 9x the variance: a lower weight
    shocked = base.copy()
    shocked[-22:] *= 9.0
    hot = ledger_with_variance(shocked, last)
    w_hot, info_hot = vol_managed_weight(hot, last + dt.timedelta(days=1))
    assert info_hot["state"] == "reduced" and w_hot < w_calm and w_hot < 0.7
    # never above 1, and NEVER reads the session being sized: a huge variance
    # ON asof itself changes nothing for asof
    spiked = ledger_with_variance(
        np.concatenate([base, [base[-1] * 1e3]]), last + dt.timedelta(days=1)
    )
    w_same, _ = vol_managed_weight(spiked, last + dt.timedelta(days=1))
    assert w_same == pytest.approx(w_calm)
    assert 0.0 < w_hot <= 1.0


def test_vol_managed_weight_warms_up_at_full_weight():
    short = ledger_with_variance(np.full(300, 1e-5), dt.date(2025, 6, 30))
    w, info = vol_managed_weight(short, dt.date(2025, 7, 1))
    assert w == 1.0 and info["state"] == "warmup"
    assert info["n_sessions"] == 300 and info["n_forecasts"] < SP_LEG_MIN_SESSIONS


def test_sp_leg_target_states_dollars_and_futures_floored():
    rng = np.random.default_rng(2)
    base = np.exp(rng.normal(-10.0, 0.2, size=700))
    base[-22:] *= 9.0
    led = ledger_with_variance(base, dt.date(2025, 6, 30))
    asof = dt.date(2025, 7, 1)
    rep = sp_leg_target(led, asof, "vol_managed", notional=1_000_000.0, spot=6000.0)
    w = rep["weight"]
    assert rep["target_dollars"] == pytest.approx(w * 1_000_000.0)
    es, mes = rep["target_es"], rep["target_mes"]
    assert es == int(np.floor(w * 1_000_000.0 / 300_000.0))
    assert 0 <= mes < 10  # a whole ES is ten MES
    assert es * 300_000.0 + mes * 30_000.0 <= rep["target_dollars"]  # never rounds up
    # the brake rule reads the ledger's own multiplier
    br = sp_leg_target(led, asof, "brake")
    assert br["rule"] == "brake" and br["weight"] in (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="sp-leg rule"):
        sp_leg_target(led, asof, "trend")
    # no notional: the weight alone
    bare = sp_leg_target(led, asof, "vol_managed")
    assert "target_dollars" not in bare and bare["notional_full"] == 0.0


def test_the_config_flags():
    assert Config().sp_leg_rule == "vol_managed" and Config().sp_leg_notional == 0.0
    assert Config.from_args([]).sp_leg_rule == "vol_managed"
    assert Config.from_args(["--sp-leg", "brake"]).sp_leg_rule == "brake"
    assert Config.from_args(["--sp-leg", "off"]).sp_leg_rule == "off"
    cfg = Config.from_args(["--sp-leg-notional", "2500000"])
    assert cfg.sp_leg_notional == 2_500_000.0
    with pytest.raises(SystemExit):
        Config.from_args(["--sp-leg", "trend"])
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sp_leg_notional"):
            Config(sp_leg_notional=bad)
    with pytest.raises(ValueError, match="sp_leg_rule"):
        Config(sp_leg_rule="trend")  # type: ignore[arg-type]


@needs_seed
def test_the_seed_ledger_cuts_the_sp_leg_after_the_april_2025_crash():
    """Study 68 part C's replay: the Monday after the tariff sessions."""
    led = PremiumLedger.load(SEED)
    hot = sp_leg_target(
        led, dt.date(2025, 4, 7), "vol_managed", notional=1_000_000.0, spot=5074.08
    )
    assert hot["state"] == "reduced" and hot["weight"] < 0.5
    assert hot["target_es"] == 1 and hot["target_mes"] == 7
    calm = sp_leg_target(led, dt.date(2025, 2, 3), "vol_managed")
    assert calm["weight"] > hot["weight"]
    # the report is strictly causal: the ledger through 2025-04-04 alone decides 2025-04-07
    frame = led.frame
    cut = PremiumLedger()
    cut.append_session(
        [
            ClockRecord(**{k: r[k] for k in ClockRecord.__dataclass_fields__})  # type: ignore[arg-type]
            for r in frame[frame["session"] < "2025-04-07"]
            .assign(session=lambda f: f["session"].dt.date)
            .to_dict("records")
        ]
    )
    again = sp_leg_target(cut, dt.date(2025, 4, 7), "vol_managed")
    assert again["weight"] == pytest.approx(hot["weight"])
