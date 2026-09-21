"""End-to-end replay of the afternoon hold book, and every refusal it must make.

Everything here runs through ``FakeBroker`` against the recorded chain, in
``dry`` mode: no socket is opened and no order object is ever built.  The
tests that matter are the ones the 2026-09-18 code audit found the old suite
structurally blind to -- a refused exit, a partial fill, a stale snapshot, a
mid-session disconnect -- so every one of them drives the runner through a
``FakeFaults`` replay rather than asserting on the happy path.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os

import pytest

from live.ibkr.broker import FakeBroker, FakeFaults, load_replay_day
from live.ibkr.config import (
    AFTERNOON_CLOCKS,
    THIRD_FRIDAY_SIZE_MULTIPLIER,
    Config,
    LiveModeRefused,
)
from live.ibkr.hedge import target_lots
from live.ibkr.journal import DaySummary, Journal, read_journal
from live.ibkr.premium_ledger import ClockRecord, PremiumLedger
from live.ibkr.calendar_guard import (
    CALENDAR_FLAT_REASON,
    CALENDAR_OVERRIDE_REASON,
    THIRD_FRIDAY_REASON,
    is_last_session_of_month,
    is_third_friday_session,
)
from live.ibkr.run_day import (
    DELEVER_FLAT_REASON,
    Abort,
    DayRunner,
    clock_records,
    main,
    previous_trading_day,
    reconcile_day,
    settlement_payoff,
)

WORST = "2023-11-30"  # the hold book's worst day: settlement 0.51% above 15:30
RECENT = "2025-06-20"  # an unseen-sample session, for the selector
#: the FIRST session the shipped ledger gives proposal 50's rule a zero
#: multiplier on -- March 2020 is inside the 252-session warm-up, because the
#: ledger itself starts 2020-01-03 (see the warm-up test below).
DELEVER_ZERO = "2025-04-09"
FIXED = "11:00"  # proposal 41's clock, replayed here as the fixed entry


@pytest.fixture(scope="module")
def frames():
    return {
        d: load_replay_day(d, cache_dir=None) for d in (WORST, RECENT, DELEVER_ZERO)
    }


def make_cfg(tmp_path, date, **kw):
    """A replay config that touches nothing outside ``tmp_path``."""
    kw.setdefault("entry_mode", "fixed")
    kw.setdefault("fixed_entry_clock", FIXED)
    kw.setdefault("n_override", 1)
    kw.setdefault("ledger_path", str(tmp_path / "seed.parquet"))
    kw.setdefault("ledger_live_path", str(tmp_path / "live.parquet"))
    kw.setdefault("replay_cache_dir", str(tmp_path / "cache"))
    # WORST is the last trading session of November 2023: in the default
    # override mode the runner would BUY the 15:30 straddle there instead of
    # running the short book.  These tests are about the SHORT book, so on a
    # month-end the calendar is off unless the test asks for a mode.  On every
    # other date the default mode stands, so the short-book tests on RECENT
    # and DELEVER_ZERO also prove the override is inert off a month-end.
    if is_last_session_of_month(dt.date.fromisoformat(date)):
        kw.setdefault("month_end_mode", "off")
    # RECENT is a monthly-expiration session (the third Friday of June 2025).
    # These tests are about the ordinary-size book, so there the third-Friday
    # multiplier is pinned to 1.0 unless a test asks for one, so they do not move
    # with the default (study 66's).  The third-Friday tests pass their own.
    if is_third_friday_session(dt.date.fromisoformat(date)):
        kw.setdefault("third_friday_size_multiplier", 1.0)
    return Config(replay_date=date, journal_dir=str(tmp_path), **kw)


def run_replay(tmp_path, frames, date, faults=None, broker_kw=None, **cfg_kw):
    cfg = make_cfg(tmp_path, date, **cfg_kw)
    broker = FakeBroker(
        cfg, date, frame=frames[date], faults=faults, **(broker_kw or {})
    )
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), date + ".jsonl"), echo=False)
    runner = DayRunner(cfg, broker, jr)
    try:
        runner.run()
    finally:
        jr.close()
    return cfg, broker, runner, jr.path, DaySummary.from_journal(jr.path)


def kinds_of(path, kind):
    return [r["payload"] for r in read_journal(path) if r["kind"] == kind]


def errors_of(path):
    return {str(p.get("what", "")) for p in kinds_of(path, "error")}


def synthetic_ledger(path, best="13:30", sessions=300, last_day=None):
    """A ledger whose RECENT premium (E2's score) is highest at ``best``.

    Every clock has the same long-run implied/realized level, so the only
    thing that can move E2's argmax is the recent half -- which is the
    migration the selector exists to follow.
    """
    led = PremiumLedger()
    recs = []
    day = (
        dt.date.fromisoformat(last_day)
        if last_day
        else dt.date.fromisoformat(RECENT) - dt.timedelta(days=1)
    )
    k = 0
    while k < sessions:
        if day.weekday() < 5:
            recent = k < 126
            for c in AFTERNOON_CLOCKS:
                ratio = 1.4 if (recent and c == best) else 1.0
                recs.append(
                    ClockRecord(
                        session=day,
                        clock=c,
                        implied_rem_var=1.0e-5 * ratio,
                        realized_rem_var=1.0e-5,
                        spot=5000.0,
                        premium_mid=20.0,
                        kc=5005.0,
                        kp=5000.0,
                    )
                )
            k += 1
        day -= dt.timedelta(days=1)
    led.append_session(recs)
    led.save(str(path))
    return path


# ---------------------------------------------------------------- end to end


def test_the_worst_day_held_to_settlement(tmp_path, frames):
    """2023-11-30 at 11:00, held: the settlement printed 0.51% above 15:30."""
    cfg, _b, _r, path, s = run_replay(tmp_path, frames, WORST, terminal="hold")

    pre = kinds_of(path, "preflight")
    assert len(pre) == 1 and pre[0]["ok"] is True
    assert pre[0]["checks"]["liquid_close"] == "16:00"

    assert s.entered and s.entry_clock == FIXED
    assert s.exit_kind == "settlement" and s.provisional is True
    assert s.entry_premium == pytest.approx(11.20, abs=1e-6)

    # the ladder is every half hour after the entry through 15:30 inclusive
    assert cfg.ladder(FIXED) == (
        "11:30",
        "12:00",
        "12:30",
        "13:00",
        "13:30",
        "14:00",
        "14:30",
        "15:00",
        "15:30",
    )
    assert s.n_rebalances == 1 + len(cfg.ladder(FIXED)) == 10

    # the futures die at the bell, the options cash-settle
    flat = kinds_of(path, "flatten")
    assert flat and flat[-1]["clock"] == "16:00:00"
    assert not s.futures_open and s.straddles_open == 0

    # the day itself: a ~2.5 premium-unit settlement loss
    assert -2.6 < s.pnl_units < -2.2
    assert s.pnl_units == pytest.approx(-2.4805, abs=5e-4)
    assert s.pnl_dollars == pytest.approx(
        s.n_straddles * cfg.index_multiplier * (s.entry_premium - s.exit_price)
        + s.hedge_pnl_dollars,
        abs=1e-6,
    )

    # and the handover file the morning reconciliation reads
    pending = json.loads(
        (tmp_path / "pending_settlement.json").read_text(encoding="utf-8")
    )
    assert pending["date"] == WORST and pending["entry_clock"] == FIXED
    assert [o["clock"] for o in pending["observed"]][0] == FIXED
    assert [o["clock"] for o in pending["observed"]][-1] == "15:30"


def test_the_selector_enters_in_the_afternoon(tmp_path, frames):
    """With a ledger in hand the entry clock is chosen, not configured."""
    seed = synthetic_ledger(tmp_path / "seed.parquet", best="13:30")
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        entry_mode="selector",
        fixed_entry_clock="15:00",  # deliberately NOT the ledger's answer
        ledger_path=str(seed),
        terminal="hold",
        n_override=1,
    )
    sel = kinds_of(path, "selector")[0]
    assert sel["mode"] == "selector" and sel["warmup"] is False
    assert sel["clock"] == "13:30"
    assert set(sel["candidates"]) == set(AFTERNOON_CLOCKS)
    assert s.entry_clock == "13:30"
    assert s.entered and s.exit_kind == "settlement"
    # 14:00 .. 15:30 after a 13:30 entry
    assert s.n_rebalances == 5


def test_the_selector_falls_back_to_the_fixed_clock_while_warming_up(tmp_path, frames):
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        entry_mode="selector",
        fixed_entry_clock="14:00",
        terminal="hold",
    )
    sel = kinds_of(path, "selector")[0]
    assert sel["mode"] == "warmup_fixed" and sel["warmup"] is True
    assert s.entry_clock == "14:00"
    led = kinds_of(path, "ledger")[0]
    assert led["ok"] is False and led["used"] == "seed"


def test_a_stale_live_ledger_falls_back_to_the_seed(tmp_path, frames):
    """The last session must be the previous trading session, or we shout."""
    stale_end = (dt.date.fromisoformat(RECENT) - dt.timedelta(days=30)).isoformat()
    seed = synthetic_ledger(tmp_path / "seed.parquet", best="14:30", last_day=stale_end)
    _cfg, _b, _r, path, _s = run_replay(
        tmp_path, frames, RECENT, entry_mode="selector", ledger_path=str(seed)
    )
    led = kinds_of(path, "ledger")[0]
    assert led["expected_last"] == str(
        previous_trading_day(dt.date.fromisoformat(RECENT))
    )
    assert led["last_session"] != led["expected_last"]
    assert led["ok"] is False and led["used"] == "seed"


# -------------------------------------------- against the shipped seed ledger

SEED = os.path.join("results", "live_seed", "premium_ledger.parquet")
needs_seed = pytest.mark.skipif(
    not os.path.exists(SEED), reason="the seed premium ledger has not been built"
)


@needs_seed
def test_the_seed_ledger_picks_an_afternoon_clock_on_a_2025_session(tmp_path, frames):
    """The integration test: the shipped ledger, the shipped selector."""
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        entry_mode="selector",
        ledger_path=SEED,
        terminal="hold",
        n_override=None,
        capital=1_000_000.0,
    )
    sel = kinds_of(path, "selector")[0]
    assert sel["mode"] == "selector" and sel["warmup"] is False
    assert sel["clock"] in AFTERNOON_CLOCKS
    assert sel["clock"] == "14:30"
    assert s.entry_clock == "14:30" and s.entered
    assert s.n_straddles == 3  # $1m at 10% over a $28,812 stress loss
    assert s.exit_kind == "settlement"
    # the correction factor is live at every clock the book hedged
    factors = [p["correction_factor"] for p in kinds_of(path, "delta")]
    assert factors and all(f is not None and f > 0 for f in factors)


@needs_seed
def test_the_v9_correction_changes_the_hedge_and_only_the_hedge(tmp_path, frames):
    corrected = run_replay(
        tmp_path / "on", frames, WORST, terminal="hold", ledger_path=SEED
    )[4]
    raw = run_replay(
        tmp_path / "off",
        frames,
        WORST,
        terminal="hold",
        ledger_path=SEED,
        delta_correction=False,
    )[4]
    assert corrected.entry_premium == raw.entry_premium
    assert corrected.exit_price == raw.exit_price
    assert corrected.hedge_pnl_points != raw.hedge_pnl_points
    assert corrected.pnl_units == pytest.approx(-2.5279, abs=5e-4)
    assert raw.pnl_units == pytest.approx(-2.4805, abs=5e-4)


# ------------------------------------------------------------------- sizing


def test_sizing_comes_from_the_stress_table(tmp_path, frames):
    capital = 650_000.0
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        WORST,
        n_override=None,
        capital=capital,
        stress_fraction=0.10,
        terminal="hold",
    )
    sz = kinds_of(path, "sizing")[0]
    loss = float(sz["loss_total"])
    assert math.isfinite(loss) and loss > 0
    # the stress loss is the option loss net of what the hedge earns back
    assert sz["loss_side"] in ("+", "-")
    assert loss == pytest.approx(sz["loss_option"] + sz["loss_hedge"], abs=1e-6)
    expect = int(capital * 0.10 / loss)
    assert sz["n"] == expect == s.n_straddles
    assert sz["fraction_implied"] == pytest.approx(expect * loss / capital, abs=1e-9)
    # ten times the capital is ten times the size, up to the hard cap
    _c2, _b2, _r2, p2, s2 = run_replay(
        tmp_path / "big",
        frames,
        WORST,
        n_override=None,
        capital=10 * capital,
        max_straddles=4,
        terminal="hold",
    )
    assert s2.n_straddles == 4
    assert kinds_of(p2, "sizing")[0]["n_from_table"] > 4


def test_n_overrides_the_table_and_logs_the_fraction_it_implies(tmp_path, frames):
    _cfg, _b, _r, path, _s = run_replay(
        tmp_path, frames, WORST, n_override=3, capital=200_000.0, terminal="hold"
    )
    sz = kinds_of(path, "sizing")[0]
    assert sz["n"] == 3 and sz["note"] == "--n override"
    assert sz["fraction_implied"] == pytest.approx(
        3 * sz["loss_total"] / 200_000.0, abs=1e-9
    )
    assert sz["fraction_implied"] > 0.10  # louder than the configured fraction


def test_no_capital_and_no_n_is_refused_in_preflight(tmp_path, frames):
    cfg = make_cfg(tmp_path, WORST, n_override=None, capital=0.0)
    broker = FakeBroker(cfg, WORST, frame=frames[WORST])
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), "nosize.jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert "no --capital and no --n" in str(exc.value)


# ------------------------------------------------------- the safety batch


def test_an_unfilled_exit_never_flattens_the_futures(tmp_path, frames):
    """The one that matters: no naked short straddle into the close (L-1)."""
    _cfg, broker, runner, path, s = run_replay(
        tmp_path,
        frames,
        WORST,
        terminal="flatten",
        faults=FakeFaults(refuse=frozenset({"body_exit"})),
    )
    assert "exit_unfilled" in errors_of(path)
    assert s.straddles_open == 1
    assert s.futures_open and sum(s.futures_open.values()) != 0
    assert any(v for v in broker.positions()["futures"].values())
    # nothing tried to flatten
    assert not [o for o in broker.orders if o["what"] == "futures_flatten"]
    assert "manual_flatten_required" in errors_of(path)
    # and the day reports no P&L rather than a confident wrong one
    assert math.isnan(s.pnl_units)
    assert "still open" in s.incomplete
    assert "MANUAL FLATTEN" not in s.render()  # it is a run-time shout
    assert "STILL OPEN" in s.render()


def test_an_unfilled_kill_combo_leaves_the_hedge_on(tmp_path, frames):
    """Blow the body's quotes out from 13:00, then refuse the buy-back."""
    frame = frames[WORST].copy()
    hit = (frame["hhmm"] >= "13:00") & frame["strike"].isin([4555.0, 4550.0])
    frame.loc[hit, "bid"] = 200.0
    frame.loc[hit, "ask"] = 210.0
    cfg = make_cfg(tmp_path, WORST, terminal="flatten")
    broker = FakeBroker(
        cfg, WORST, frame=frame, faults=FakeFaults(refuse=frozenset({"body_exit"}))
    )
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), "kill.jsonl"), echo=False)
    runner = DayRunner(cfg, broker, jr)
    runner.run()
    jr.close()
    path = jr.path

    kill = kinds_of(path, "kill")
    # the loss stays over the limit, so the runner re-attempts the buy-back at
    # every remaining clock rather than giving up or stripping the hedge
    assert kill and kill[0]["clock"] == "13:00"
    assert [k["clock"] for k in kill] == sorted({k["clock"] for k in kill})
    assert kill[0]["futures_ref"] != kill[0]["S"]  # marked on the FUTURES (L-5)
    assert "exit_unfilled" in errors_of(path)
    # the kill did not take: the hedge is still on and the day kept hedging
    assert not [o for o in broker.orders if o["what"] == "futures_flatten"]
    assert any(v for v in broker.positions()["futures"].values())
    after = [p["clock"] for p in kinds_of(path, "rebalance") if p["clock"] > "13:00"]
    assert after, "the runner must keep hedging an open short straddle"
    summary = DaySummary.from_journal(path)
    # the switch fired, but firing is not flat: the summary says so out loud
    assert summary.killed is True
    assert summary.straddles_open == 1
    assert runner.state.killed is False
    assert math.isnan(summary.pnl_units)
    assert "STILL OPEN" in summary.render()


def test_a_partial_futures_fill_is_journaled_with_the_right_sign(tmp_path, frames):
    _cfg, broker, runner, path, s = run_replay(
        tmp_path,
        frames,
        WORST,
        terminal="hold",
        faults=FakeFaults(partial={"futures": 1}),
    )
    fills = [p for p in kinds_of(path, "fill") if p["what"] == "futures"]
    assert fills, "the entry hedge must have tried"
    first = fills[0]
    assert first["ok"] is False and first["status"] == "Submitted"
    assert abs(first["quantity"]) == 1
    # ... and it is signed by the leg it was working, not by ``ok``
    tgt = [p for p in kinds_of(path, "target") if p["clock"] == first["clock"]][0]
    wanted = tgt["qty_es"] if first["symbol"] == "ES" else tgt["qty_mes"]
    assert abs(wanted) > 1  # the order asked for more than it got
    assert first["quantity"] == math.copysign(1, wanted)
    assert "futures_unfilled" in errors_of(path)
    # and the runner's own book agrees with the broker's
    assert runner.state.lots["MES"] == broker.positions()["futures"].get("MES", 0)


def test_a_non_invertible_vol_never_zeroes_the_hedge(tmp_path, frames):
    """A package mid below intrinsic must not be read as delta = 0 (L-6)."""
    frame = frames[WORST].copy()
    frame.loc[frame["hhmm"] == "14:00", "underlying_price"] = 4400.0
    cfg = make_cfg(tmp_path, WORST, terminal="hold")
    broker = FakeBroker(cfg, WORST, frame=frame)
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), "novol.jsonl"), echo=False)
    DayRunner(cfg, broker, jr).run()
    jr.close()
    path = jr.path

    assert "vol_not_invertible" in errors_of(path)
    at14 = [p for p in kinds_of(path, "delta") if p["clock"] == "14:00"][0]
    assert at14["total_vol"] is not None  # a carried vol, not a NaN
    assert at14["source"] == "carried_sqrt_scaled"
    assert at14["delta_pkg"] != 0.0
    tgt = [p for p in kinds_of(path, "target") if p["clock"] == "14:00"][0]
    assert (tgt["target_es"], tgt["target_mes"]) != (0, 0)


def test_a_dead_vol_with_nothing_to_carry_carries_the_last_delta(tmp_path, frames):
    """The second carry: no invertible and no carryable vol, still not zero."""
    cfg = make_cfg(tmp_path, WORST, terminal="hold")
    broker = FakeBroker(cfg, WORST, frame=frames[WORST])
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), "carry.jsonl"), echo=False)
    runner = DayRunner(cfg, broker, jr)
    runner.preflight()
    assert runner.enter()
    runner.state.total_vol = float("nan")  # nothing carryable
    runner.state.last_delta = 0.42
    lots_before = dict(runner.state.lots)
    now = broker.advance_to("14:00")
    pq = runner._package_quote()
    pq.mid = 1e-9  # below intrinsic: not invertible
    runner.hedge_to_target("14:00", now, 4400.0, pq)
    jr.close()

    assert "delta_not_computable" in errors_of(jr.path)
    at14 = [p for p in kinds_of(jr.path, "delta") if p["clock"] == "14:00"][-1]
    assert at14["delta_pkg"] == pytest.approx(0.42)
    assert at14["delta_source"] == "carried_last_delta"
    assert runner.state.lots != lots_before  # it hedged to 0.42, not to zero


def test_a_stale_snapshot_is_not_traded_on(tmp_path, frames):
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        WORST,
        terminal="hold",
        faults=FakeFaults(stale_at=frozenset({"13:00"})),
    )
    assert "unhealthy_quotes" in errors_of(path)
    assert "13:00" not in [p["clock"] for p in kinds_of(path, "rebalance")]
    assert s.entered  # the rest of the day still ran


def test_a_disconnect_mid_rebalance_rehedges_and_journals_lateness(tmp_path, frames):
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        WORST,
        terminal="hold",
        faults=FakeFaults(disconnect_at=frozenset({"14:00"}), reconnect_delay_s=180.0),
    )
    errs = {(p["what"], p.get("clock")) for p in kinds_of(path, "error")}
    assert ("disconnect", "14:00") in errs
    assert ("late_clock", "14:00") in errs
    late = [p for p in kinds_of(path, "error") if p["what"] == "late_clock"][0]
    assert late["late_s"] == pytest.approx(180.0)
    # it reconnected and rebalanced the clock it was late for
    assert "14:00" in [p["clock"] for p in kinds_of(path, "rebalance")]
    assert s.entered and s.exit_kind == "settlement"


def test_the_summary_identity_holds_with_open_futures_marked_at_the_reference(
    tmp_path, frames
):
    """Options flat, futures stuck open: mark them at the FUTURES price (L-11)."""
    _cfg, broker, runner, path, s = run_replay(
        tmp_path,
        frames,
        WORST,
        terminal="flatten",
        faults=FakeFaults(refuse=frozenset({"futures_flatten"})),
    )
    assert s.exit_kind == "buyback" and s.straddles_open == 0
    assert s.futures_open, "the flatten was refused, so a leg is open"
    mark = kinds_of(path, "mark")[-1]
    assert s.futures_mark == pytest.approx(mark["futures_ref"])
    assert s.futures_mark != 0.0
    # cash from the fills, plus the open lots at the mark
    realized = 0.0
    for p in kinds_of(path, "fill"):
        if str(p["what"]).startswith("futures") and p["quantity"]:
            realized -= p["quantity"] * p["price"] * p["multiplier"]
    open_value = sum(
        q * s.futures_mark * (s.es_multiplier if k == "ES" else s.mes_multiplier)
        for k, q in s.futures_open.items()
    )
    assert s.hedge_pnl_dollars == pytest.approx(realized + open_value, abs=1e-6)
    assert s.pnl_dollars == pytest.approx(
        s.n_straddles * s.index_multiplier * (s.entry_premium - s.exit_price)
        + s.hedge_pnl_dollars,
        abs=1e-6,
    )
    assert "marked at" in s.incomplete
    assert "FUTURES STILL OPEN" in s.render()


def test_a_journal_with_no_config_record_computes_no_pnl(tmp_path):
    path = str(tmp_path / "bare.jsonl")
    jr = Journal(path, echo=False)
    jr.event("fill", what="body_entry", price=10.0, quantity=1, ok=True, order_id=1)
    jr.event("fill", what="body_exit", price=5.0, quantity=1, ok=True, order_id=2)
    jr.close()
    s = DaySummary.from_journal(jr.path)
    assert math.isnan(s.pnl_dollars) and math.isnan(s.es_multiplier)
    assert "no config record" in s.incomplete


def test_a_rerun_rolls_the_journal_instead_of_appending(tmp_path, frames):
    _c1, _b1, _r1, p1, s1 = run_replay(tmp_path, frames, WORST, terminal="hold")
    _c2, _b2, _r2, p2, s2 = run_replay(tmp_path, frames, WORST, terminal="hold")
    assert p1 != p2 and p2.endswith(".1.jsonl")
    assert s2.hedge_pnl_dollars == pytest.approx(s1.hedge_pnl_dollars)
    assert s2.pnl_units == pytest.approx(s1.pnl_units)


def test_a_repeated_fill_record_is_counted_once(tmp_path):
    path = str(tmp_path / "dupe.jsonl")
    jr = Journal(path, echo=False)
    jr.event(
        "config",
        date="2025-01-02",
        book="b",
        mode="dry",
        index_multiplier=100.0,
        es_multiplier=50.0,
        mes_multiplier=5.0,
    )
    for _ in range(2):  # a restart re-journaling the same order
        jr.event(
            "fill",
            what="futures",
            symbol="MES",
            multiplier=5.0,
            price=100.0,
            quantity=2,
            ok=True,
            order_id=77,
        )
    jr.close()
    s = DaySummary.from_journal(jr.path)
    assert s.n_futures_fills == 1
    assert s.futures_open == {"MES": 2}


# --------------------------------------------------------------- reconcile


def test_reconcile_finalises_the_pnl_and_appends_the_ledger(tmp_path, frames):
    cfg, broker, _r, path, provisional = run_replay(
        tmp_path, frames, WORST, terminal="hold"
    )
    assert provisional.provisional is True
    official = float(provisional.exit_price)  # the payoff, re-derived below
    settle_rec = kinds_of(path, "settle")[0]
    assert settle_rec["provisional"] is True

    cfg.settlement = settle_rec["S_close"] + 5.0  # the official print moved
    rc = reconcile_day(cfg)
    assert rc == 0
    final = DaySummary.from_journal(path)
    assert final.exit_kind == "settlement" and final.provisional is False
    want = settlement_payoff(cfg.settlement, final.K_c, final.K_p, None)
    assert final.exit_price == pytest.approx(want)
    assert final.exit_price != pytest.approx(official)
    assert final.pnl_units == pytest.approx(
        (final.entry_premium - want + final.hedge_pnl_points) / final.entry_premium
    )

    led = PremiumLedger.load(cfg.ledger_live_path)
    rows = led.frame
    assert len(rows) and set(rows["clock"]) <= {
        p["clock"] for p in kinds_of(path, "delta")
    }
    assert led.sessions() == [dt.date.fromisoformat(WORST)]
    assert (rows["implied_rem_var"] > 0).all()
    assert (rows["realized_rem_var"] > 0).all()
    # the pending file is consumed, not left to be reconciled twice
    assert not os.path.exists(os.path.join(cfg.journal_dir, "pending_settlement.json"))
    assert reconcile_day(cfg) == 2


def test_clock_records_are_the_remaining_window_variances():
    observed = [
        {
            "clock": "13:30",
            "S": 100.0,
            "total_vol": 0.01,
            "premium_mid": 2.0,
            "kc": 100.0,
            "kp": 100.0,
        },
        {
            "clock": "14:00",
            "S": 101.0,
            "total_vol": 0.008,
            "premium_mid": 1.0,
            "kc": 100.0,
            "kp": 100.0,
        },
    ]
    recs = clock_records(dt.date(2025, 1, 2), observed, 102.0)
    assert [r.clock for r in recs] == ["13:30", "14:00"]
    s1 = math.log(101.0 / 100.0) ** 2
    s2 = math.log(102.0 / 101.0) ** 2
    assert recs[0].realized_rem_var == pytest.approx(s1 + s2)
    assert recs[1].realized_rem_var == pytest.approx(s2)
    assert recs[0].implied_rem_var == pytest.approx(0.01**2)
    assert recs[0].spot == 100.0 and recs[0].premium_mid == 2.0


# ------------------------------------------------------------- the CLI gate


def test_paper_mode_may_not_talk_to_the_live_port():
    with pytest.raises(LiveModeRefused) as exc:
        Config.from_args(["--mode", "paper", "--port", "7496"])
    assert "LIVE listener" in str(exc.value)
    with pytest.raises(LiveModeRefused):
        Config.from_args(["--mode", "paper", "--port", "4001"])
    with pytest.raises(LiveModeRefused):
        Config.from_args(["--port", "7496"])  # dry is not a live consent either
    assert Config.from_args(["--mode", "paper", "--port", "7497"]).port == 7497


def test_live_mode_needs_both_consents_in_either_flag_form(monkeypatch):
    monkeypatch.delenv("HARXHAR_LIVE", raising=False)
    with pytest.raises(LiveModeRefused):
        Config.from_args(["--mode", "live"])
    monkeypatch.setenv("HARXHAR_LIVE", "I_UNDERSTAND")
    with pytest.raises(LiveModeRefused):
        Config(mode="live")  # no explicit --mode live on a command line
    for argv in (["--mode", "live"], ["--mode=live"]):
        cfg = Config.from_args(argv)
        assert cfg.port == 7496 and cfg.live_ack
        assert "L I V E" in cfg.banner()
    with pytest.raises(LiveModeRefused):
        Config.from_args(["--mode=live", "--port", "7497"])


def test_from_args_with_none_reads_sys_argv(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["run_day", "--entry-clock", "14:30", "--terminal", "flatten"]
    )
    cfg = Config.from_args(None)
    assert cfg.fixed_entry_clock == "14:30" and cfg.terminal == "flatten"
    monkeypatch.setattr("sys.argv", ["run_day"])
    assert Config.from_args(None).fixed_entry_clock == "13:30"


def test_the_ladder_is_always_filtered_against_the_terminal_clock():
    hold = Config(terminal="hold")
    assert hold.ladder("13:30")[-1] == "15:30"
    flat = Config(terminal="flatten", exit_clock="15:30")
    assert flat.ladder("13:30")[-1] == "15:00"
    # an explicit ladder is filtered too (L-30)
    custom = Config(terminal="flatten", rebalance_clocks=("14:00", "15:30", "15:45"))
    assert custom.ladder("13:30") == ("14:00",)


def test_an_inert_exit_clock_is_refused_rather_than_ignored():
    """``--exit-clock`` under ``--terminal hold`` was accepted and inert (L-43)."""
    with pytest.raises(ValueError) as exc:
        Config(terminal="hold", exit_clock="15:50")
    assert "inert with --terminal hold" in str(exc.value)
    assert Config(terminal="hold").flatten_clock == "16:00:00"
    assert Config(terminal="flatten", exit_clock="15:50").flatten_clock == "15:50:00"


def test_the_option_tick_follows_the_price_level():
    cfg = Config()
    assert cfg.option_tick_for(2.95) == 0.05
    assert cfg.option_tick_for(3.00) == 0.10
    assert cfg.option_tick_for(11.40) == 0.10
    # and the crossing allowance comes off the live spread, capped
    assert cfg.cross_ticks_for(0.40, 0.10) == 5
    assert cfg.cross_ticks_for(100.0, 0.10) == cfg.max_cross_ticks


# ------------------------------------------------------------------ main()


def test_main_runs_a_replay_and_writes_the_summary(tmp_path, frames):
    rc = main(
        [
            "--date",
            WORST,
            "--entry-mode",
            "fixed",
            "--entry-clock",
            FIXED,
            "--n",
            "1",
            "--journal-dir",
            str(tmp_path),
            "--ledger",
            str(tmp_path / "absent.parquet"),
            "--ledger-live",
            str(tmp_path / "absent_live.parquet"),
            "--month-end-mode",
            "off",  # WORST is a month-end session: run the short book
        ]
    )
    assert rc == 0
    out = os.path.join(str(tmp_path), WORST + "_summary.json")
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob["book"] == "afternoon-hold-to-cash-settlement"
    assert blob["exit_kind"] == "settlement"
    assert blob["pnl_units"] is not None


def test_main_returns_non_zero_with_a_position_open(tmp_path, frames, monkeypatch):
    """A supervisor must not read success off a day that left a leg on (L-34)."""
    import live.ibkr.run_day as rd

    def build(cfg):
        b = FakeBroker(
            cfg,
            cfg.replay_date,
            frame=frames[WORST],
            faults=FakeFaults(refuse=frozenset({"futures_flatten"})),
        )
        return b

    monkeypatch.setattr(rd, "build_broker", build)
    rc = main(
        [
            "--date",
            WORST,
            "--terminal",
            "flatten",
            "--entry-mode",
            "fixed",
            "--entry-clock",
            FIXED,
            "--n",
            "1",
            "--journal-dir",
            str(tmp_path),
            "--ledger",
            str(tmp_path / "absent.parquet"),
            "--no-calendar-guard",  # the alias of --month-end-mode off
        ]
    )
    assert rc == 3


def test_a_half_session_aborts_in_preflight(tmp_path, frames):
    cfg = make_cfg(tmp_path, WORST)
    broker = FakeBroker(cfg, WORST, frame=frames[WORST], liquid_close="13:00")
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), "half.jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert "half session" in str(exc.value)
    s = DaySummary.from_journal(jr.path)
    assert not s.entered and "half session" in s.flat_reason
    assert not [r for r in read_journal(jr.path) if r["kind"] in ("order", "fill")]


def test_an_open_position_aborts_in_preflight(tmp_path, frames):
    cfg = make_cfg(tmp_path, WORST)
    broker = FakeBroker(cfg, WORST, frame=frames[WORST])
    broker.connect()
    broker._futures_qty = {"ES": 2}
    jr = Journal(os.path.join(str(tmp_path), "open.jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert "reconcile manually" in str(exc.value)


def test_wings_requested_but_not_quoted_ends_the_day_flat(tmp_path, frames):
    _cfg, _b, _r, path, s = run_replay(tmp_path, frames, WORST, wings_pct=0.015)
    assert not s.entered and s.flat_reason == "wings_not_quoted"
    assert s.n_rebalances == 0
    assert not [r for r in read_journal(path) if r["kind"] == "fill"]
    assert "FLAT" in s.render()


def test_dry_mode_is_the_default_and_places_no_ib_order(tmp_path, frames):
    cfg, broker, _r, _p, s = run_replay(tmp_path, frames, RECENT, terminal="hold")
    assert cfg.mode == "dry" and s.entered
    assert broker.orders and all("fill" in o for o in broker.orders)
    assert not hasattr(broker, "ib")


# ------------------- the deleveraging candidate (proposal 50, part B2) ---


def quiet_rv_ledger(path, last_day, last_rv, n=20, lo=1e-5, hi=2e-5):
    """A ledger whose 10:00 realized variance is quiet, then ``last_rv``.

    ``n - 1`` quiet sessions set the thresholds and the last one is the
    regime the multiplier is read on: with ``delever_window=1`` the last
    session's variance IS the next session's RV21, and it never enters its
    own percentiles.
    """
    end = dt.date.fromisoformat(last_day)
    step = (hi - lo) / float(n - 2)
    values = [lo + step * i for i in range(n - 1)] + [float(last_rv)]
    led = PremiumLedger()
    led.append_session(
        [
            ClockRecord(
                session=end - dt.timedelta(days=n - 1 - i),
                clock="10:00",
                implied_rem_var=1e-4,
                realized_rem_var=float(v),
                spot=5000.0,
                premium_mid=20.0,
                kc=5005.0,
                kp=5000.0,
            )
            for i, v in enumerate(values)
        ]
    )
    led.save(str(path))
    return str(path)


HALF_KW = dict(delever_window=1, delever_min_sessions=10)


@needs_seed
def test_march_2020_is_inside_the_deleveraging_warm_up() -> None:
    """Why the flat-day test is not 2020-03-16.

    The rule needs 252 prior sessions carrying a trailing variance before its
    expanding percentiles mean anything, and the shipped ledger starts
    2020-01-03 -- so the whole of the COVID crash is warm-up and the rule is
    inactive over it.  The first session it actually zeroes is 2025-04-09.
    """
    led = PremiumLedger.load(SEED)
    mult, info = led.delever_multiplier(dt.date(2020, 3, 16))
    assert (mult, info["state"]) == (1.0, "warmup")
    assert int(info["n_prior"]) < 252
    zeros = [d for d in led.sessions() if led.delever_multiplier(d)[0] == 0.0]
    assert zeros and str(zeros[0]) == DELEVER_ZERO
    print("first zero-multiplier session in the shipped ledger:", zeros[0])


@needs_seed
def test_a_zero_multiplier_ends_the_day_flat_in_preflight(tmp_path, frames):
    """The design candidate's whole point: April 2025 is not traded."""
    cfg = make_cfg(
        tmp_path,
        DELEVER_ZERO,
        fixed_entry_clock="13:30",
        n_override=None,
        capital=1_000_000.0,
        terminal="hold",
        ledger_path=SEED,
    )
    broker = FakeBroker(cfg, DELEVER_ZERO, frame=frames[DELEVER_ZERO])
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), DELEVER_ZERO + ".jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert DELEVER_FLAT_REASON in str(exc.value)
    assert "97.5th percentile" in str(exc.value)

    regime = kinds_of(jr.path, "regime")
    assert len(regime) == 1
    r = regime[0]
    assert r["state"] == "zero" and r["multiplier"] == 0.0
    assert r["rv21"] > r["p_zero"] > r["p_half"]
    assert r["n_prior"] >= 252 and r["enabled"] is True
    assert r["window"] == 21 and r["zero_pct"] == 0.975

    s = DaySummary.from_journal(jr.path)
    assert not s.entered and s.is_flat
    assert DELEVER_FLAT_REASON in s.flat_reason
    assert not kinds_of(jr.path, "order")  # nothing was sent


@needs_seed
def test_no_delever_puts_the_same_session_back_on_at_full_size(tmp_path, frames):
    """``--no-delever`` restores the old size, and the old loss with it."""
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        DELEVER_ZERO,
        fixed_entry_clock="13:30",
        n_override=None,
        capital=1_000_000.0,
        terminal="hold",
        ledger_path=SEED,
        delever=False,
    )
    assert kinds_of(path, "regime")[0]["state"] == "disabled"
    sz = kinds_of(path, "sizing")[0]
    assert (sz["n"], sz["n_stress"], sz["multiplier"]) == (6, 6, 1.0)
    assert s.entered and s.n_straddles == 6
    assert s.pnl_units == pytest.approx(-0.3559, abs=5e-5)


@needs_seed
def test_a_full_regime_leaves_the_2025_06_20_replay_exactly_as_it_was(tmp_path, frames):
    """The rule is inert on a quiet day: the README's worked example stands."""
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        entry_mode="selector",
        ledger_path=SEED,
        terminal="hold",
        n_override=None,
        capital=1_000_000.0,
    )
    r = kinds_of(path, "regime")[0]
    assert (r["state"], r["multiplier"]) == ("full", 1.0)
    assert r["rv21"] < r["p_half"]
    sz = kinds_of(path, "sizing")[0]
    assert (sz["n"], sz["n_stress"], sz["multiplier"]) == (3, 3, 1.0)
    assert s.entry_clock == "14:30" and s.n_straddles == 3
    assert s.pnl_units == pytest.approx(0.2215, abs=5e-5)


def test_a_half_regime_halves_the_stress_size(tmp_path, frames):
    """p90 < RV21 <= p97.5: the table's size, floored after halving."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2025-06-19", 1.94e-5)
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        fixed_entry_clock="13:30",
        n_override=None,
        capital=1_000_000.0,
        terminal="hold",
        ledger_path=seed,
        **HALF_KW,
    )
    r = kinds_of(path, "regime")[0]
    assert (r["state"], r["multiplier"]) == ("half", 0.5)
    assert r["p_half"] < r["rv21"] <= r["p_zero"]
    sz = kinds_of(path, "sizing")[0]
    assert sz["n_stress"] >= 2  # the table wanted more than one
    assert sz["n"] == sz["n_stress"] // 2 and sz["multiplier"] == 0.5
    assert "deleveraged" in sz["note"]
    assert s.entered and s.n_straddles == sz["n"]


def test_an_explicit_n_is_deleveraged_too_and_journals_both_numbers(tmp_path, frames):
    """``--n`` is a FULL-SIZE intent, not an exemption from the regime."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2025-06-19", 1.94e-5)
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        fixed_entry_clock="13:30",
        n_override=3,
        capital=1_000_000.0,
        terminal="hold",
        ledger_path=seed,
        **HALF_KW,
    )
    sz = kinds_of(path, "sizing")[0]
    assert (sz["n_override"], sz["n_stress"], sz["multiplier"], sz["n"]) == (
        3,
        3,
        0.5,
        1,
    )
    assert s.n_straddles == 1
    # and the implied stress fraction is reported for BOTH sizes
    assert sz["fraction_implied_stress"] == pytest.approx(3.0 * sz["fraction_implied"])


def test_one_straddle_deleveraged_to_zero_ends_the_day_flat(tmp_path, frames):
    """floor(1 x 0.5) = 0: flat, and the reason names the rule, not the table."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2025-06-19", 1.94e-5)
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        fixed_entry_clock="13:30",
        n_override=1,
        capital=1_000_000.0,
        terminal="hold",
        ledger_path=seed,
        **HALF_KW,
    )
    entry = kinds_of(path, "entry")[0]
    assert entry["entered"] is False
    assert entry["reason"] == "deleveraging_took_the_size_to_zero_contracts"
    assert entry["multiplier"] == 0.5
    assert not s.entered and s.is_flat
    assert not kinds_of(path, "order")


def test_an_unreadable_ledger_is_not_a_quiet_regime(tmp_path, frames):
    """No ledger, no rule: full size, and the journal says which it was."""
    _cfg, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        WORST,
        terminal="hold",
        ledger_path=str(tmp_path / "not_a_ledger.parquet"),
    )
    r = kinds_of(path, "regime")[0]
    # an ABSENT ledger loads as an empty one: the rule warms up rather than
    # firing.  Either way the multiplier is 1.0 and it is journaled.
    assert r["multiplier"] == 1.0 and r["state"] in ("warmup", "no_ledger")
    assert s.entered


def test_the_delever_flags_and_their_bounds():
    assert Config.from_args([]).delever is True
    cfg = Config.from_args(["--no-delever"])
    assert cfg.delever is False
    assert (cfg.delever_window, cfg.delever_min_sessions) == (21, 252)
    assert (cfg.delever_half_pct, cfg.delever_zero_pct) == (0.90, 0.975)
    with pytest.raises(ValueError, match="delever_half_pct"):
        Config(delever_half_pct=0.99, delever_zero_pct=0.90)
    with pytest.raises(ValueError, match="delever_window"):
        Config(delever_window=0)
    with pytest.raises(ValueError, match="delever_min_sessions"):
        Config(delever_min_sessions=0)


# ------------------ the month-end modes (proposals 54 and 55) ------------

#: Proposal 54's 15:30 long straddle on WORST, bought at the quoted ask and
#: settled at the official close (proposals/54/c_daily.csv, pts_ask / R_ask).
WORST_LONG_ASK = 3.65
WORST_LONG_PTS = 19.149805
WORST_LONG_UNITS = 5.246522
SETTLEMENT_TAPE = os.path.join(
    "results", "atm_straddle_intraday_holdclose", "cache", "gspc_ohlc.parquet"
)
needs_settlement_tape = pytest.mark.skipif(
    not os.path.exists(SETTLEMENT_TAPE), reason="needs the official-close tape"
)
#: The README's worked capital; a tenth of it is the long's loss budget.
CAPITAL = 1_000_000.0
#: The clock the short program is sized at in these replays (fixed, no ledger).
SIZE_CLOCK = "13:30"


def run_override(tmp_path, frames, faults=None, **kw):
    """WORST in the default mode: no short, a long 15:30 straddle."""
    cfg_kw = dict(
        month_end_mode="override",
        fixed_entry_clock=SIZE_CLOCK,
        n_override=None,
        capital=CAPITAL,
        terminal="hold",
    )
    cfg_kw.update(kw)
    return run_replay(tmp_path, frames, WORST, faults=faults, **cfg_kw)


def official_close(day):
    import pandas as pd

    close = pd.read_parquet(SETTLEMENT_TAPE)["close"]
    close.index = pd.to_datetime(close.index).normalize()
    return float(close.loc[pd.Timestamp(day)])


def test_the_override_buys_the_15_30_straddle_and_holds_it(tmp_path, frames):
    cfg, _b, _r, path, s = run_override(tmp_path, frames)

    c = kinds_of(path, "calendar")[0]
    assert (c["mode"], c["decision"], c["override"], c["flat"]) == (
        "override",
        "override",
        True,
        False,
    )
    assert c["hits"] == ["month_end"] and c["long_size"] == "match_short"
    assert c["reason"].startswith(CALENDAR_OVERRIDE_REASON)
    assert "bad_kind" not in c
    pre = kinds_of(path, "preflight")[0]
    assert pre["ok"] is True and pre["checks"]["calendar"]["override"] is True
    assert pre["checks"]["delever_applied"] is False

    # sized like the short program, at the short program's clock
    sz = kinds_of(path, "sizing")[0]
    assert (sz["side"], sz["clock"], sz["rule"]) == ("long", SIZE_CLOCK, "match_short")
    assert sz["n"] == min(sz["n_from_table"], cfg.max_straddles) >= 1
    assert sz["multiplier"] == 1.0 and "short_loss_total" in sz
    assert "loss_total" not in sz  # the short's stress loss is not the long's risk

    # ONE order: a BUY at the quoted ask that may not cross
    orders = kinds_of(path, "order")
    assert len(orders) == 1
    o = orders[0]
    assert (o["what"], o["action"], o["side"], o["max_cross_ticks"]) == (
        "body_entry",
        "BUY",
        "long",
        0,
    )
    entry = [e for e in kinds_of(path, "entry") if e["entered"]]
    assert len(entry) == 1 and entry[0]["clock"] == "15:30"
    assert entry[0]["side"] == "long" and entry[0]["size_clock"] == SIZE_CLOCK
    assert o["limit"] == pytest.approx(entry[0]["pkg_ask"])
    assert entry[0]["outlay_dollars"] <= entry[0]["loss_budget_dollars"]

    # no hedge at all
    fut = [f for f in kinds_of(path, "fill") if str(f["what"]).startswith("fut")]
    assert not fut
    assert s.n_rebalances == 0 and s.n_futures_fills == 0 and not s.futures_open

    # the summary books a LONG: payoff minus premium
    assert s.side == "long" and s.entered and s.exit_kind == "settlement"
    assert s.entry_clock == "15:30" and s.n_straddles == sz["n"]
    assert s.entry_premium == pytest.approx(WORST_LONG_ASK, abs=1e-6)  # float32 tape
    assert s.option_pnl_points == pytest.approx(s.exit_price - s.entry_premium)
    assert s.pnl_units == pytest.approx(s.exit_price / s.entry_premium - 1.0)
    assert s.pnl_dollars == pytest.approx(
        s.n_straddles * cfg.index_multiplier * (s.exit_price - s.entry_premium)
    )
    assert s.is_flat and "LONG" in s.render()
    assert kinds_of(path, "settle")[0]["side"] == "long"

    # the ledger keeps its rows: the short program's clock through 15:30
    pending = json.loads(
        (tmp_path / "pending_settlement.json").read_text(encoding="utf-8")
    )
    assert (pending["side"], pending["decision"]) == ("long", "override")
    assert (pending["entry_clock"], pending["size_clock"]) == ("15:30", SIZE_CLOCK)
    assert [r["clock"] for r in pending["observed"]] == [
        "13:30",
        "14:00",
        "14:30",
        "15:00",
        "15:30",
    ]


@needs_settlement_tape
def test_the_override_reproduces_proposal_54_after_reconcile(tmp_path, frames):
    """At the official close the long is 54's long, to the printed digit."""
    cfg, _b, _r, path, s = run_override(tmp_path, frames, n_override=1)
    assert s.provisional is True
    cfg.settlement = official_close(WORST)
    assert reconcile_day(cfg) == 0
    final = DaySummary.from_journal(path)
    assert final.provisional is False and final.side == "long"
    assert final.option_pnl_points == pytest.approx(WORST_LONG_PTS, abs=1e-6)
    assert final.pnl_units == pytest.approx(WORST_LONG_UNITS, abs=1e-6)
    led = PremiumLedger.load(cfg.ledger_live_path)
    assert led.sessions() == [dt.date.fromisoformat(WORST)]


def test_the_long_is_sized_without_the_deleveraging_multiplier(tmp_path, frames):
    """Half regime: the short would be halved; the long is the short's FULL count."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2023-11-29", 1.94e-5)
    kw = dict(ledger_path=seed, **HALF_KW)
    _c, _b, _r, p_long, s_long = run_override(tmp_path / "long", frames, **kw)
    _c2, _b2, _r2, p_short, _s2 = run_replay(
        tmp_path / "short",
        frames,
        WORST,
        month_end_mode="off",
        fixed_entry_clock=SIZE_CLOCK,
        n_override=None,
        capital=CAPITAL,
        terminal="hold",
        **kw,
    )
    assert kinds_of(p_long, "regime")[0]["multiplier"] == 0.5
    sz_long = kinds_of(p_long, "sizing")[0]
    sz_short = kinds_of(p_short, "sizing")[0]
    assert sz_short["multiplier"] == 0.5 and sz_short["n_stress"] >= 2
    assert sz_short["n"] == sz_short["n_stress"] // 2
    assert sz_long["n"] == sz_short["n_stress"]  # the short program's count, whole
    assert (sz_long["multiplier"], sz_long["multiplier_not_applied"]) == (1.0, 0.5)
    assert s_long.side == "long" and s_long.n_straddles == sz_short["n_stress"]


def test_a_zero_regime_does_not_refuse_an_override_day(tmp_path, frames):
    """The brake is for short-tail risk: it refuses a short day, not the long."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2023-11-29", 1.0e-3)
    kw = dict(ledger_path=seed, **HALF_KW)
    _c, _b, _r, path, s = run_override(tmp_path / "long", frames, **kw)
    assert kinds_of(path, "regime")[0]["state"] == "zero"
    pre = kinds_of(path, "preflight")[0]
    assert pre["ok"] is True and pre["checks"]["delever_applied"] is False
    assert s.side == "long" and s.entered

    cfg = make_cfg(
        tmp_path / "short",
        WORST,
        month_end_mode="off",
        fixed_entry_clock=SIZE_CLOCK,
        n_override=None,
        capital=CAPITAL,
        **kw,
    )
    broker = FakeBroker(cfg, WORST, frame=frames[WORST])
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), "short.jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert DELEVER_FLAT_REASON in str(exc.value)


def test_an_outlay_above_the_loss_budget_is_refused(tmp_path, frames):
    capital = 5_000.0  # a tenth of it is $500: less than three straddles' premium
    cfg, _b, _r, path, s = run_override(tmp_path, frames, n_override=3, capital=capital)
    e = kinds_of(path, "entry")[-1]
    assert e["entered"] is False
    assert e["reason"] == "month_end_long_outlay_exceeds_the_loss_budget"
    assert e["loss_budget_dollars"] == pytest.approx(capital * cfg.stress_fraction)
    assert e["outlay_dollars"] == pytest.approx(
        3 * WORST_LONG_ASK * cfg.index_multiplier
    )
    assert e["outlay_dollars"] > e["loss_budget_dollars"]
    assert not kinds_of(path, "order") and not s.entered and s.is_flat
    assert s.flat_reason == "month_end_long_outlay_exceeds_the_loss_budget"

    # the flat day still hands the morning its rows, and reconciles them
    pending = json.loads(
        (tmp_path / "pending_settlement.json").read_text(encoding="utf-8")
    )
    assert (pending["side"], pending["K_c"], pending["n_straddles"]) == (
        "flat",
        None,
        0,
    )
    assert [r["clock"] for r in pending["observed"]][-1] == "15:30"
    cfg.settlement = float(kinds_of(path, "quote")[-1]["S"])
    assert reconcile_day(cfg) == 0
    assert kinds_of(path, "reconcile")[0]["payoff"] is None
    assert PremiumLedger.load(cfg.ledger_live_path).sessions() == [
        dt.date.fromisoformat(WORST)
    ]


def test_no_capital_means_no_loss_budget_and_no_long(tmp_path, frames):
    _c, _b, _r, path, s = run_override(tmp_path, frames, n_override=1, capital=0.0)
    e = kinds_of(path, "entry")[-1]
    assert e["entered"] is False
    assert e["reason"].startswith("month_end_long_no_loss_budget")
    assert not kinds_of(path, "order") and not s.entered and s.is_flat


def test_no_quote_at_15_30_ends_the_override_flat(tmp_path, frames):
    faults = FakeFaults(stale_at=frozenset({"15:30"}))
    _c, _b, _r, path, s = run_override(tmp_path, frames, faults=faults)
    e = kinds_of(path, "entry")[-1]
    assert e["entered"] is False and e["reason"].startswith("stale_or_halted_quotes")
    assert not kinds_of(path, "order") and not s.entered and s.is_flat
    assert not kinds_of(path, "settle")


def test_no_fill_ends_the_override_flat_and_never_chases(tmp_path, frames):
    faults = FakeFaults(refuse=frozenset({"body_entry"}))
    _c, _b, _r, path, s = run_override(tmp_path, frames, faults=faults)
    orders = kinds_of(path, "order")
    assert len(orders) == 1 and orders[0]["max_cross_ticks"] == 0  # one try only
    fill = kinds_of(path, "fill")[0]
    assert fill["quantity"] == 0 and fill["status"] == "Cancelled"
    assert "month_end_long_unfilled" in errors_of(path)
    e = kinds_of(path, "entry")[-1]
    assert e["entered"] is False and e["reason"] == "month_end_long_unfilled"
    assert not kinds_of(path, "settle") and not s.entered and s.is_flat


def test_a_partial_long_is_held_to_settlement(tmp_path, frames):
    faults = FakeFaults(partial={"body_entry": 1})
    _c, _b, _r, path, s = run_override(tmp_path, frames, faults=faults, n_override=3)
    assert "entry_partial" in errors_of(path)
    assert s.side == "long" and s.n_straddles == 1
    assert s.exit_kind == "settlement" and s.is_flat
    assert kinds_of(path, "settle")[0]["n_settled"] == 1


def test_sit_out_ends_a_month_end_flat_in_preflight(tmp_path, frames):
    """The old guard, kept as a mode."""
    cfg = make_cfg(tmp_path, WORST, month_end_mode="sit_out", terminal="hold")
    broker = FakeBroker(cfg, WORST, frame=frames[WORST])
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), WORST + ".jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert str(exc.value).startswith(CALENDAR_FLAT_REASON)
    assert "last trading session of the month" in str(exc.value)

    c = kinds_of(jr.path, "calendar")[0]
    assert (c["mode"], c["decision"], c["flat"]) == ("sit_out", "sit_out", True)
    assert c["session"] == WORST and c["calendars"] == ["month_end"]
    pre = kinds_of(jr.path, "preflight")[0]
    assert pre["ok"] is False and pre["checks"]["calendar"]["flat"] is True
    # the refusal comes BEFORE the ledger is read: no regime record at all
    assert not kinds_of(jr.path, "regime") and not kinds_of(jr.path, "ledger")

    s = DaySummary.from_journal(jr.path)
    assert not s.entered and s.is_flat
    assert s.flat_reason.startswith(CALENDAR_FLAT_REASON)
    assert not kinds_of(jr.path, "order")


def test_off_puts_the_short_book_back_on_a_month_end(tmp_path, frames):
    _cfg, _b, _r, path, s = run_replay(
        tmp_path, frames, WORST, month_end_mode="off", terminal="hold"
    )
    c = kinds_of(path, "calendar")[0]
    assert (c["enabled"], c["decision"], c["hits"]) == (False, "short", [])
    assert s.side == "short" and s.entered and s.n_straddles == 1
    assert s.pnl_units == pytest.approx(-2.4805, abs=5e-4)  # as the worst-day test


def test_an_ordinary_session_is_the_same_in_every_mode(tmp_path, frames):
    seen = {}
    for mode in ("override", "sit_out", "off"):
        _c, _b, _r, path, s = run_replay(
            tmp_path / mode, frames, RECENT, month_end_mode=mode, terminal="hold"
        )
        c = kinds_of(path, "calendar")[0]
        assert (c["mode"], c["decision"], c["hits"]) == (mode, "short", [])
        seen[mode] = (s.side, s.entry_clock, s.n_straddles, s.pnl_units)
    assert seen["override"] == seen["sit_out"] == seen["off"]
    assert seen["off"][0] == "short"


def test_an_open_position_outranks_the_calendar(tmp_path, frames):
    """A leg left on from a crash is the louder alarm, month-end or not."""
    cfg = make_cfg(tmp_path, WORST, month_end_mode="override", capital=CAPITAL)
    broker = FakeBroker(cfg, WORST, frame=frames[WORST])
    broker.connect()
    broker._futures_qty = {"ES": 2}
    jr = Journal(os.path.join(str(tmp_path), "open_me.jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert "reconcile manually" in str(exc.value)
    assert not kinds_of(jr.path, "calendar")


def test_main_overrides_a_month_end_by_default(tmp_path, frames):
    base = [
        "--date",
        WORST,
        "--entry-mode",
        "fixed",
        "--entry-clock",
        SIZE_CLOCK,
        "--capital",
        str(CAPITAL),
        "--ledger",
        str(tmp_path / "absent.parquet"),
        "--ledger-live",
        str(tmp_path / "absent_live.parquet"),
    ]
    over = tmp_path / "override"
    assert main(base + ["--journal-dir", str(over)]) == 0
    path = os.path.join(str(over), WORST + "_summary.json")
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob["side"] == "long" and blob["exit_kind"] == "settlement"

    sit = tmp_path / "sit_out"
    argv = base + ["--journal-dir", str(sit), "--month-end-mode", "sit_out"]
    assert main(argv) == 2  # a preflight refusal, not a crash and not a leg on
    path = os.path.join(str(sit), WORST + "_summary.json")
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob["flat_reason"].startswith(CALENDAR_FLAT_REASON)


def test_the_month_end_flags():
    assert Config.from_args([]).month_end_mode == "override"
    sit = Config.from_args(["--month-end-mode", "sit_out"])
    assert sit.month_end_mode == "sit_out"
    assert Config.from_args(["--no-calendar-guard"]).month_end_mode == "off"
    both = ["--no-calendar-guard", "--month-end-mode", "off"]
    assert Config.from_args(both).month_end_mode == "off"
    with pytest.raises(SystemExit):
        Config.from_args(["--no-calendar-guard", "--month-end-mode", "override"])
    cfg = Config()
    assert cfg.no_short_calendars == ("month_end",)
    assert cfg.month_end_long_size == "match_short"
    with pytest.raises(ValueError, match="no_short_calendars"):
        Config(no_short_calendars=("month_end", "full_moon"))
    with pytest.raises(ValueError, match="month_end_mode"):
        Config(month_end_mode="guard")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not enabled"):
        Config(month_end_long_size="loss_budget")


# ------------------------- the third-Friday size (proposal 58) ------------

#: RECENT at $1m on the seed ledger: the selector's 14:30, three straddles at
#: the ordinary size (the README's worked replay).
RECENT_TABLE_N = 3


def run_third_friday(tmp_path, frames, multiplier, **kw):
    """RECENT, the monthly-expiration session, sized by the stress table."""
    cfg_kw = dict(
        entry_mode="selector",
        ledger_path=SEED,
        n_override=None,
        capital=CAPITAL,
        terminal="hold",
        third_friday_size_multiplier=multiplier,
    )
    cfg_kw.update(kw)
    return run_replay(tmp_path, frames, RECENT, **cfg_kw)


@needs_seed
def test_the_third_friday_multiplier_scales_the_short_and_its_hedge(tmp_path, frames):
    """x2 on 2025-06-20: three straddles become six, and the lots follow them."""
    _c1, _b1, _r1, p1, s1 = run_third_friday(tmp_path / "x1", frames, 1.0)
    _c2, _b2, _r2, p2, s2 = run_third_friday(tmp_path / "x2", frames, 2.0)

    # the calendar record: flagged both times, applied only when asked
    c1, c2 = kinds_of(p1, "calendar")[0], kinds_of(p2, "calendar")[0]
    assert c1["third_friday"] is True and c2["third_friday"] is True
    assert (c1["third_friday_multiplier"], c1["third_friday_applied"]) == (1.0, 1.0)
    assert (c2["third_friday_multiplier"], c2["third_friday_applied"]) == (2.0, 2.0)
    assert c2["decision"] == "short"
    assert c2["third_friday_reason"].startswith(THIRD_FRIDAY_REASON)
    assert kinds_of(p2, "config")[0]["third_friday_size_multiplier"] == 2.0

    # the sizing record: the count before and after
    z1, z2 = kinds_of(p1, "sizing")[0], kinds_of(p2, "sizing")[0]
    assert (z1["n_before_third_friday"], z1["n"]) == (RECENT_TABLE_N, RECENT_TABLE_N)
    assert (z2["n_before_third_friday"], z2["n"]) == (
        RECENT_TABLE_N,
        2 * RECENT_TABLE_N,
    )
    assert (z2["n_stress"], z2["multiplier"]) == (RECENT_TABLE_N, 1.0)
    assert z2["third_friday"] is True and z2["third_friday_multiplier"] == 2.0
    assert z2["third_friday_capped"] is False and "third Friday x2" in z2["note"]
    assert z2["loss_total"] == z1["loss_total"]  # the same straddle, the same table
    assert z2["fraction_implied"] == pytest.approx(2.0 * z1["fraction_implied"])
    assert s1.entry_clock == s2.entry_clock == "14:30"
    assert (s1.n_straddles, s2.n_straddles) == (RECENT_TABLE_N, 2 * RECENT_TABLE_N)

    # the ES/MES split is recomputed from the new count at every rebalance
    rebal = kinds_of(p2, "rebalance")
    assert rebal
    for r in rebal:
        want = target_lots(r["delta_pkg"], 2 * RECENT_TABLE_N)
        assert (r["target_es"], r["target_mes"]) == want


@needs_seed
def test_a_fractional_multiplier_is_floored(tmp_path, frames):
    _c, _b, _r, path, s = run_third_friday(tmp_path, frames, 1.5)
    sz = kinds_of(path, "sizing")[0]
    assert (sz["n_before_third_friday"], sz["n"]) == (RECENT_TABLE_N, 4)  # 4.5 -> 4
    assert s.n_straddles == 4


@needs_seed
def test_study_66_s_multiplier_floors_and_stays_under_the_cap(tmp_path, frames):
    """1.1, floored: no change below ten straddles, and the cap still binds."""
    four_m = 4.0 * CAPITAL  # 13 straddles in the table at 14:30
    cases = {
        "1m": (dict(), RECENT_TABLE_N, RECENT_TABLE_N, False),  # 3.3 -> 3
        "4m_room": (dict(capital=four_m, max_straddles=20), 13, 14, False),  # 14.3
        "4m_cap": (dict(capital=four_m), 10, 10, True),  # 10 x 1.1 = 11 -> 10
    }
    for name, (kw, before, after, capped) in cases.items():
        _c, _b, _r, path, s = run_third_friday(tmp_path / name, frames, 1.1, **kw)
        sz = kinds_of(path, "sizing")[0]
        assert kinds_of(path, "calendar")[0]["third_friday_applied"] == 1.1
        got = (sz["n_before_third_friday"], sz["n"], sz["third_friday_capped"])
        assert got == (before, after, capped), name
        assert s.n_straddles == after


@needs_seed
def test_max_straddles_caps_the_count_after_the_multiplier(tmp_path, frames):
    _c, _b, _r, path, s = run_third_friday(tmp_path, frames, 2.0, max_straddles=5)
    sz = kinds_of(path, "sizing")[0]
    assert (sz["n_before_third_friday"], sz["n"]) == (RECENT_TABLE_N, 5)
    assert sz["third_friday_capped"] is True and "capped" in sz["note"]
    assert s.n_straddles == 5


def test_an_explicit_n_is_never_multiplied(tmp_path, frames):
    """--n sets the count by hand: the calendar flags the day and applies 1."""
    _c, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        fixed_entry_clock="13:30",
        n_override=2,
        capital=CAPITAL,
        terminal="hold",
        third_friday_size_multiplier=2.0,
    )
    c = kinds_of(path, "calendar")[0]
    assert c["third_friday"] is True and c["third_friday_applied"] == 1.0
    assert "--n sets the size" in c["third_friday_reason"]
    sz = kinds_of(path, "sizing")[0]
    assert (sz["n"], sz["third_friday_multiplier"], sz["note"]) == (
        2,
        1.0,
        "--n override",
    )
    assert s.n_straddles == 2


def test_the_multiplier_scales_the_deleveraged_count(tmp_path, frames):
    """Half regime: 3 at full size, 1 after the brake, 2 after the multiplier."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2025-06-19", 1.94e-5)
    _c, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        fixed_entry_clock="13:30",
        n_override=None,
        capital=CAPITAL,
        terminal="hold",
        ledger_path=seed,
        third_friday_size_multiplier=2.0,
        **HALF_KW,
    )
    sz = kinds_of(path, "sizing")[0]
    assert sz["multiplier"] == 0.5 and sz["n_stress"] == RECENT_TABLE_N
    assert (sz["n_before_third_friday"], sz["n"]) == (1, 2)
    assert "deleveraged x0.5" in sz["note"] and "third Friday x2" in sz["note"]
    assert s.n_straddles == 2


def test_the_brake_at_zero_still_ends_a_third_friday_flat(tmp_path, frames):
    """A zero regime refuses the day in preflight, whatever the multiplier."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2025-06-19", 1.0e-3)
    cfg = make_cfg(
        tmp_path,
        RECENT,
        fixed_entry_clock="13:30",
        n_override=None,
        capital=CAPITAL,
        terminal="hold",
        ledger_path=seed,
        third_friday_size_multiplier=2.0,
        **HALF_KW,
    )
    broker = FakeBroker(cfg, RECENT, frame=frames[RECENT])
    broker.connect()
    jr = Journal(os.path.join(str(tmp_path), "tf_zero.jsonl"), echo=False)
    with pytest.raises(Abort) as exc:
        DayRunner(cfg, broker, jr).run()
    jr.close()
    assert DELEVER_FLAT_REASON in str(exc.value)
    assert kinds_of(jr.path, "calendar")[0]["third_friday_applied"] == 2.0
    assert kinds_of(jr.path, "regime")[0]["multiplier"] == 0.0
    assert not kinds_of(jr.path, "sizing") and not kinds_of(jr.path, "order")


def test_a_count_the_brake_took_to_zero_stays_zero(tmp_path, frames):
    """floor(1 x 0.5) = 0, and 0 x 2 = 0: flat, and the reason names the brake."""
    seed = quiet_rv_ledger(tmp_path / "seed.parquet", "2025-06-19", 1.94e-5)
    _c, _b, _r, path, s = run_replay(
        tmp_path,
        frames,
        RECENT,
        fixed_entry_clock="13:30",
        n_override=None,
        capital=400_000.0,  # one straddle at a ~$30k stress loss
        terminal="hold",
        ledger_path=seed,
        third_friday_size_multiplier=2.0,
        **HALF_KW,
    )
    sz = kinds_of(path, "sizing")[0]
    assert (sz["n_stress"], sz["n_before_third_friday"], sz["n"]) == (1, 0, 0)
    entry = kinds_of(path, "entry")[0]
    assert entry["reason"] == "deleveraging_took_the_size_to_zero_contracts"
    assert not s.entered and not kinds_of(path, "order")


def test_an_ordinary_session_is_not_scaled(tmp_path, frames):
    """2025-04-09 is not an expiration: x2 and x1 are the same run."""
    seen = {}
    for mult in (1.0, 2.0):
        _c, _b, _r, path, s = run_replay(
            tmp_path / str(mult),
            frames,
            DELEVER_ZERO,
            fixed_entry_clock="13:30",
            n_override=None,
            capital=CAPITAL,
            terminal="hold",
            third_friday_size_multiplier=mult,
        )
        c = kinds_of(path, "calendar")[0]
        assert c["third_friday"] is False and c["third_friday_applied"] == 1.0
        sz = kinds_of(path, "sizing")[0]
        assert sz["n_before_third_friday"] == sz["n"] and sz["third_friday"] is False
        seen[mult] = (sz["n"], sz["note"], s.n_straddles, s.pnl_units, s.pnl_dollars)
    assert seen[1.0] == seen[2.0] and seen[1.0][0] >= 1


def test_the_month_end_long_is_not_scaled(tmp_path, frames):
    """A month-end is never an expiration, and the override is read first."""
    _c1, _b1, _r1, p1, s1 = run_override(tmp_path / "x1", frames)
    _c2, _b2, _r2, p2, s2 = run_override(
        tmp_path / "x2", frames, third_friday_size_multiplier=3.0
    )
    c2 = kinds_of(p2, "calendar")[0]
    assert c2["decision"] == "override" and c2["third_friday"] is False
    assert c2["third_friday_applied"] == 1.0
    assert kinds_of(p1, "sizing")[0]["n"] == kinds_of(p2, "sizing")[0]["n"]
    assert s1.n_straddles == s2.n_straddles and s2.side == "long"


@needs_seed
def test_main_takes_the_third_friday_multiplier_from_the_command_line(tmp_path, frames):
    argv = [
        "--date",
        RECENT,
        "--capital",
        str(CAPITAL),
        "--ledger",
        SEED,
        "--ledger-live",
        str(tmp_path / "absent_live.parquet"),
        "--third-friday-multiplier",
        "2",
        "--journal-dir",
        str(tmp_path),
    ]
    assert main(argv) == 0
    with open(
        os.path.join(str(tmp_path), RECENT + "_summary.json"), encoding="utf-8"
    ) as fh:
        blob = json.load(fh)
    assert blob["n_straddles"] == 2 * RECENT_TABLE_N and blob["side"] == "short"


def test_the_third_friday_flags():
    assert Config().third_friday_size_multiplier == THIRD_FRIDAY_SIZE_MULTIPLIER
    assert (
        Config.from_args([]).third_friday_size_multiplier
        == THIRD_FRIDAY_SIZE_MULTIPLIER
    )
    two = Config.from_args(["--third-friday-multiplier", "2"])
    assert two.third_friday_size_multiplier == 2.0
    half = Config.from_args(["--third-friday-multiplier=1.5"])
    assert half.third_friday_size_multiplier == 1.5
    assert Config.from_args(["--no-third-friday"]).third_friday_size_multiplier == 1.0
    both = ["--no-third-friday", "--third-friday-multiplier", "1"]
    assert Config.from_args(both).third_friday_size_multiplier == 1.0
    with pytest.raises(SystemExit):
        Config.from_args(["--no-third-friday", "--third-friday-multiplier", "2"])
    for bad in (0.5, 0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="third_friday_size_multiplier"):
            Config(third_friday_size_multiplier=bad)


def test_the_operator_defaults_third_friday_and_the_pinned_clock():
    # 2026-09-21: third Fridays x1.5 and the entry pinned to the fixed 13:30
    # clock; the selector stays one flag away.
    assert THIRD_FRIDAY_SIZE_MULTIPLIER == 1.5
    for cfg in (Config(), Config.from_args([])):
        assert cfg.entry_mode == "fixed" and cfg.fixed_entry_clock == "13:30"
    assert Config.from_args(["--entry-mode", "selector"]).entry_mode == "selector"
    # 1.5 adds a contract at every count the stress table gives at $1M
    from live.ibkr.sizing import scaled_contracts

    assert [
        scaled_contracts(n, THIRD_FRIDAY_SIZE_MULTIPLIER) for n in (2, 3, 4, 5, 6)
    ] == [
        3,
        4,
        6,
        7,
        9,
    ]
