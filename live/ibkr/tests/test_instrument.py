"""The instrument axis: SPXW (the book of record) and XSP (Mini-SPX, one tenth).

XSP has no tape in this repository.  Its replays here are SYNTHETIC -- the
recorded SPXW session rescaled by 0.1 -- and prove the code path (sizing, the
MES/NES ladder, the S&P-unit ledger, settlement), not anything about XSP.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pytest

from live.ibkr.broker import (
    FAKE_FUTURE_CONID,
    FakeBroker,
    load_replay_day,
    round_to_tick,
    synthetic_xsp_frame,
)
from live.ibkr.config import FUTURES_MULTIPLIER, INSTRUMENTS, Config
from live.ibkr.hedge import (
    ladder_lots,
    ladder_rebalance,
    ladder_residual,
    residual_delta_lots,
    target_lots,
)
from live.ibkr.journal import Journal, read_journal
from live.ibkr.run_day import DayRunner, clock_records
from live.ibkr.sizing import contracts_for, stress_loss_per_contract

ORDINARY = "2025-06-13"  # a plain Friday, not an expiration, not a month-end
TARIFF_MONDAY = "2025-04-07"  # the session after the tariff crash: a wide book
SEED = os.path.join("results", "live_seed", "premium_ledger.parquet")
needs_seed = pytest.mark.skipif(
    not os.path.exists(SEED), reason="the seed premium ledger has not been built"
)


def kinds_of(path, kind):
    return [r["payload"] for r in read_journal(path) if r["kind"] == kind]


# ------------------------------------------------------------ the table ----


def test_the_instrument_table():
    spx, xsp = INSTRUMENTS["spxw"], INSTRUMENTS["xsp"]
    assert (spx.option_symbol, spx.trading_class, spx.index_symbol) == (
        "SPX",
        "SPXW",
        "SPX",
    )
    assert (xsp.option_symbol, xsp.trading_class, xsp.index_symbol) == (
        "XSP",
        "XSP",
        "XSP",
    )
    assert spx.index_scale == 1.0 and xsp.index_scale == 0.1
    assert spx.default_hedge_symbols == ("ES", "MES")
    assert xsp.default_hedge_symbols == ("MES", "NES")
    assert FUTURES_MULTIPLIER == {"ES": 50.0, "MES": 5.0, "NES": 0.5}


def test_config_defaults_and_the_xsp_flag():
    c = Config()
    assert c.instrument == "spxw" and c.hedge_ladder == ("ES", "MES")
    assert c.index_scale == 1.0 and c.ladder_multipliers == (50.0, 5.0)
    x = Config(instrument="xsp")
    assert x.hedge_ladder == ("MES", "NES") and x.index_scale == 0.1
    assert x.ladder_multipliers == (5.0, 0.5)
    # the CLI
    y = Config.from_args(["--instrument", "xsp"])
    assert y.instrument == "xsp" and y.hedge_ladder == ("MES", "NES")
    z = Config.from_args(["--hedge-ladder", "ES,MES,NES"])
    assert z.hedge_ladder == ("ES", "MES", "NES") and z.instrument == "spxw"
    assert Config.from_args([]).hedge_ladder == ("ES", "MES")
    with pytest.raises(SystemExit):
        Config.from_args(["--instrument", "spy"])
    with pytest.raises(ValueError, match="unknown hedge instrument"):
        Config(hedge_symbols=("ES", "ZN"))
    with pytest.raises(ValueError, match="largest contract to the smallest"):
        Config(hedge_symbols=("MES", "ES"))
    with pytest.raises(ValueError, match="repeats"):
        Config(hedge_symbols=("ES", "ES"))
    with pytest.raises(ValueError, match="instrument must be"):
        Config(instrument="spy")  # type: ignore[arg-type]


def test_the_tick_rule_follows_the_instrument():
    c = Config()
    assert c.option_tick_for(1.00) == 0.05 and c.option_tick_for(5.00) == 0.10
    x = Config(instrument="xsp")
    assert x.option_tick_for(0.40) == 0.01 and x.option_tick_for(8.40) == 0.01
    assert round_to_tick(1.234, x.option_tick_for(1.234)) == 1.23
    assert round_to_tick(1.234, c.option_tick_for(1.234)) == 1.25
    # an explicit tick rule is kept
    o = Config(instrument="xsp", option_tick_lo=0.05, option_tick_hi=0.10)
    assert o.option_tick_for(1.00) == 0.05


# ------------------------------------------------------------ sizing -------


def test_sizing_two_xsp_straddles_at_sixty_thousand_and_zero_spxw():
    # the same day in both instruments: XSP is SPXW / 10 in every input
    spx = stress_loss_per_contract(6500.0, 6505.0, 6500.0, 14.0, 0.05)
    xsp = stress_loss_per_contract(650.0, 650.5, 650.0, 1.4, 0.05)
    assert xsp["total"] == pytest.approx(spx["total"] / 10.0)
    assert contracts_for(60_000.0, 0.10, spx["total"]) == 0
    assert contracts_for(60_000.0, 0.10, xsp["total"]) == 1
    # the median SPXW stress of study 65 ($22,422) is $2,242 in XSP: two fit
    assert contracts_for(60_000.0, 0.10, 2_242.2) == 2
    assert contracts_for(60_000.0, 0.10, 22_422.0) == 0


# ------------------------------------------------------------ the ladder ---


def test_the_es_mes_ladder_is_target_lots_bit_for_bit():
    for d in np.linspace(-1.0, 1.0, 2001):
        for n in (1, 2, 3, 4, 7, 10):
            assert ladder_lots(d, n, (50.0, 5.0)) == target_lots(d, n)
    assert ladder_lots(float("nan"), 3, (50.0, 5.0)) == (0, 0)
    with pytest.raises(ValueError):
        ladder_lots(0.3, 1, ())


def test_two_xsp_straddles_at_delta_point_three_hedge_in_one_mes_and_two_nes():
    # dollars of delta per S&P point: 0.3 x 100 x 0.1 x 2 = $6 -> 1 MES ($5) + 2 NES ($1)
    lots = ladder_lots(0.3, 2, (5.0, 0.5), index_scale=0.1)
    assert lots == (1, 2)
    assert ladder_residual(0.3, 2, lots, (5.0, 0.5), index_scale=0.1) == pytest.approx(
        0.0
    )
    # MES alone leaves a quarter of the delta unhedged: one MES is 0.25 straddle-deltas here
    only = ladder_lots(0.3, 2, (5.0,), index_scale=0.1)
    assert only == (1,)
    assert ladder_residual(0.3, 2, only, (5.0,), index_scale=0.1) == pytest.approx(0.05)
    # three rungs on SPXW: 1.3 straddle-deltas on 3 straddles = $390 -> 7 ES, 8 MES, 0 NES
    assert ladder_lots(1.3, 3, (50.0, 5.0, 0.5)) == (7, 8, 0)
    assert ladder_rebalance((1, 2), (0, -1)) == (1, 3)
    with pytest.raises(ValueError):
        ladder_rebalance((1, 2), (0,))
    # the old residual and the ladder's agree on the default ladder
    assert ladder_residual(0.42, 3, (2, 5), (50.0, 5.0)) == pytest.approx(
        residual_delta_lots(0.42, 3, 2, 5)
    )


# ------------------------------------------------------------ the ledger ---


def test_ledger_rows_are_in_sp500_units_whatever_the_instrument():
    observed = [
        {
            "clock": "13:30",
            "S": 650.0,
            "total_vol": 0.004,
            "premium_mid": 1.4,
            "kc": 651.0,
            "kp": 650.0,
        },
        {
            "clock": "14:00",
            "S": 651.0,
            "total_vol": 0.003,
            "premium_mid": 1.1,
            "kc": 651.0,
            "kp": 650.0,
        },
    ]
    xsp_rows = clock_records(dt.date(2025, 6, 13), observed, 652.0, index_scale=0.1)
    spx_obs = [
        {
            **r,
            "S": r["S"] * 10,
            "premium_mid": r["premium_mid"] * 10,
            "kc": r["kc"] * 10,
            "kp": r["kp"] * 10,
        }
        for r in observed
    ]
    spx_rows = clock_records(dt.date(2025, 6, 13), spx_obs, 6520.0)
    assert len(xsp_rows) == len(spx_rows) == 2
    for a, b in zip(xsp_rows, spx_rows):
        assert (a.spot, a.premium_mid, a.kc, a.kp) == pytest.approx(
            (b.spot, b.premium_mid, b.kc, b.kp)
        )
        assert a.implied_rem_var == pytest.approx(
            b.implied_rem_var
        )  # a ratio: no scaling
        assert a.realized_rem_var == pytest.approx(b.realized_rem_var)
    assert xsp_rows[0].spot == pytest.approx(6500.0) and xsp_rows[
        0
    ].premium_mid == pytest.approx(14.0)


# ------------------------------------------------------------ the replay ---


def test_the_synthetic_xsp_frame_is_the_spxw_day_on_the_dollar_grid():
    spx = load_replay_day(ORDINARY, cache_dir=None)
    xsp = synthetic_xsp_frame(spx, 0.1)
    assert xsp.attrs["synthetic_xsp_from_spxw"] is True
    strikes = sorted(set(xsp["strike"].tolist()))
    assert all(abs(k - round(k)) < 1e-9 for k in strikes)  # whole-dollar strikes
    assert (
        len(strikes) * 2 - 2 <= len(set(spx["strike"].tolist())) <= len(strikes) * 2 + 2
    )
    # half the rows survive (every second 5-point strike), the same stamps
    assert 0.4 * len(spx) < len(xsp) < 0.6 * len(spx)
    assert set(xsp["hhmm"]) == set(spx["hhmm"])
    a = xsp.loc[xsp["underlying_price"].notna(), "underlying_price"].iloc[0]
    b = spx.loc[spx["underlying_price"].notna(), "underlying_price"].iloc[0]
    assert a == pytest.approx(b * 0.1)
    # quotes on the $0.01 tick
    assert (((xsp["bid"] * 100).round() - xsp["bid"] * 100).abs() < 1e-6).all()


def _run(tmp_path, date, **kw):
    kw.setdefault("instrument", "xsp")
    kw.setdefault("entry_mode", "fixed")
    kw.setdefault("fixed_entry_clock", "13:30")
    kw.setdefault("capital", 60_000.0)
    kw.setdefault("ledger_path", SEED)
    kw.setdefault("ledger_live_path", str(tmp_path / "live.parquet"))
    kw.setdefault("replay_cache_dir", str(tmp_path / "cache"))
    kw.setdefault("third_friday_size_multiplier", 1.0)
    cfg = Config(replay_date=date, journal_dir=str(tmp_path), **kw)
    broker = FakeBroker(cfg, date)
    broker.connect()
    path = os.path.join(str(tmp_path), date + ".jsonl")
    jr = Journal(path, echo=False)
    runner = DayRunner(cfg, broker, jr)
    try:
        runner.run()
    finally:
        jr.close()
    return cfg, broker, runner, path


@needs_seed
def test_a_synthetic_xsp_ordinary_day_runs_end_to_end_at_sixty_thousand(tmp_path):
    cfg, broker, runner, path = _run(tmp_path, ORDINARY)
    assert broker.synthetic is True
    conf = kinds_of(path, "config")[0]
    assert conf["instrument"] == "xsp" and conf["synthetic_xsp_from_spxw"] is True
    assert conf["hedge_symbols"] == ["MES", "NES"] and conf["index_scale"] == 0.1
    sz = kinds_of(path, "sizing")[0]
    assert sz["instrument"] == "xsp" and sz["n"] == 2  # two XSP under the 10 % rule
    assert 0.08 < sz["fraction_implied"] <= 0.10
    assert 1_500 < sz["loss_total"] < 3_500  # a tenth of an SPXW straddle's stress
    ent = kinds_of(path, "entry")[0]
    assert ent["entered"] and ent["n_straddles"] == 2 and 500 < ent["S"] < 700
    assert ent["K_c"] - ent["K_p"] == pytest.approx(1.0)  # XSP's dollar grid
    fills = kinds_of(path, "fill")
    assert fills[0]["symbol"] == "XSP" and fills[0]["multiplier"] == 100.0
    fut = {f["symbol"] for f in fills if f["what"] in ("futures", "futures_flatten")}
    assert fut <= {"MES", "NES"} and "NES" in fut
    # every rebalance leaves less than a quarter of one MES unhedged per straddle
    for r in kinds_of(path, "rebalance"):
        assert abs(r["residual_delta"]) <= 5.0 / (2 * 100.0 * 0.1) / 2 + 1e-9
        assert set(r["target_lots"]) == {"MES", "NES"}
    flat = kinds_of(path, "flatten")[-1]
    assert flat["residual_lots"] == {"MES": 0, "NES": 0}
    st = kinds_of(path, "settle")[0]
    assert st["n_settled"] == 2 and "x 0.1" in st["note"]
    # the pending file carries the scale the morning needs
    with open(
        os.path.join(str(tmp_path), "pending_settlement.json"), encoding="utf-8"
    ) as fh:
        pending = json.load(fh)
    assert pending["instrument"] == "xsp" and pending["index_scale"] == 0.1
    rows = clock_records(
        dt.date(2025, 6, 13), pending["observed"], st["S_close"], index_scale=0.1
    )
    assert rows and 5_000 < rows[0].spot < 7_000  # S&P units in the ledger


@needs_seed
def test_a_synthetic_xsp_tariff_monday_hedges_in_both_rungs_and_ends_flat(tmp_path):
    cfg, broker, runner, path = _run(tmp_path, TARIFF_MONDAY)
    assert kinds_of(path, "config")[0]["synthetic_xsp_from_spxw"] is True
    sz = kinds_of(path, "sizing")[0]
    assert sz["n"] >= 2 and sz["fraction_implied"] <= 0.10
    tg = kinds_of(path, "target")
    assert any(t["target_lots"]["MES"] != 0 for t in tg)
    assert any(t["target_lots"]["NES"] != 0 for t in tg)
    assert kinds_of(path, "flatten")[-1]["residual_lots"] == {"MES": 0, "NES": 0}
    assert runner.state.n_open == 0 and not any(runner.state.lots.values())
    assert broker.futures_contract("NES").conId == FAKE_FUTURE_CONID["NES"]


@needs_seed
def test_spxw_replays_do_not_see_the_instrument_axis(tmp_path):
    cfg, broker, runner, path = _run(
        tmp_path, ORDINARY, instrument="spxw", capital=1_000_000.0
    )
    assert broker.synthetic is False
    conf = kinds_of(path, "config")[0]
    assert conf["instrument"] == "spxw" and conf["synthetic_xsp_from_spxw"] is False
    assert conf["hedge_symbols"] == ["ES", "MES"]
    tg = kinds_of(path, "target")[0]
    assert {"target_es", "target_mes", "current_es", "qty_mes", "target_lots"} <= set(
        tg
    )
    assert kinds_of(path, "fill")[0]["symbol"] == "SPXW"
