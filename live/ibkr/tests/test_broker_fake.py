"""FakeBroker: the recorded-session replay, and its refusal to touch a socket.

The replay serves ANY session on the chain (2020-01-03 .. 2025-12-31) at ANY
stamp it carries, because the book's entry clock is now chosen per session
rather than fixed at 11:00.  It also has to be able to go WRONG on demand --
refuse, reject, partially fill, disconnect, go stale -- because the 2026-09-18
audit's four worst findings were all invisible to a fake that always filled.
"""

from __future__ import annotations

import os

import pytest

from live.ibkr.broker import (
    BrokerError,
    FakeBroker,
    FakeFaults,
    FakeOption,
    load_replay_day,
)
from live.ibkr.config import Config
from live.ibkr.strikes import pick_nearest_otm_reason, pick_wings

DAY = "2023-11-30"
RECENT = "2025-06-20"

# The notebook cache (results/atm_straddle_intraday_holdclose/cache/trade_*.parquet)
# records this stamp as K_c=4555, K_p=4550, entry=11.40, bid=11.20, ask=11.60.
ENTRY_MID = 11.40
ENTRY_BID = 11.20
ENTRY_ASK = 11.60


@pytest.fixture(scope="module")
def frame():
    return load_replay_day(DAY, cache_dir=None)


@pytest.fixture(scope="module")
def recent_frame():
    return load_replay_day(RECENT, cache_dir=None)


def make_broker(tmp_path, frame, date=DAY, **kw):
    cfg = Config(
        replay_date=date,
        journal_dir=str(tmp_path),
        replay_cache_dir=str(tmp_path / "cache"),
        **kw.pop("cfg", {}),
    )
    b = FakeBroker(cfg, date, frame=frame, **kw)
    b.connect()
    return b


# ------------------------------------------------------------- any clock


def test_serves_the_recorded_quotes_at_the_11_00_stamp(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("11:00")
    assert b.now_et().strftime("%Y-%m-%d %H:%M") == DAY + " 11:00"
    spot = b.spx_spot()
    assert 4552.0 < spot < 4553.0

    contracts = b.spxw_0dte_contracts("20231130")
    assert len(contracts) >= 20
    quotes = b.quotes(contracts)
    assert all(q.live for q in quotes)
    assert b.quote_health().ok

    body, reason = pick_nearest_otm_reason(quotes, spot)
    assert reason == "" and body is not None
    assert (body.Kc, body.Kp) == (4555.0, 4550.0)
    assert body.entry_mid == pytest.approx(ENTRY_MID, abs=1e-6)
    assert body.entry_bid == pytest.approx(ENTRY_BID, abs=1e-6)
    assert body.entry_ask == pytest.approx(ENTRY_ASK, abs=1e-6)


@pytest.mark.parametrize(
    "clock", ["10:00", "11:30", "13:00", "13:30", "14:30", "15:00", "15:30"]
)
def test_every_stamp_on_the_tape_is_an_entry_clock(tmp_path, recent_frame, clock):
    """The selector may pick any of these, so the replay must serve them all."""
    b = make_broker(tmp_path, recent_frame, date=RECENT)
    b.advance_to(clock)
    spot = b.spx_spot()
    assert spot > 0
    quotes = b.quotes(b.spxw_0dte_contracts("20250620"))
    body, reason = pick_nearest_otm_reason(quotes, spot)
    assert reason == "" and body is not None
    assert body.Kp <= spot <= body.Kc
    assert body.entry_mid > 0


def test_the_16_00_stamp_is_the_settlement_print_not_a_quote(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("15:30")
    at1530 = b.spx_spot()
    assert b.settlement_spot() != at1530
    assert b.official_settlement(DAY) == b.settlement_spot()
    with pytest.raises(BrokerError) as exc:
        b.official_settlement("2023-11-29")
    assert "the replay holds" in str(exc.value)


def test_a_tape_that_stops_before_the_bell_has_no_settlement(tmp_path, frame):
    cut = frame[frame["hhmm"] < "16:00"]
    b = make_broker(tmp_path, cut)
    with pytest.raises(BrokerError) as exc:
        b.settlement_spot()
    assert "no settlement print" in str(exc.value)


def test_an_out_of_range_session_is_refused(tmp_path, frame):
    cfg = Config(replay_date=DAY, journal_dir=str(tmp_path))
    with pytest.raises(BrokerError) as exc:
        FakeBroker(cfg, "2019-12-31", frame=frame)
    assert "outside it" in str(exc.value)


def test_clock_off_the_tape_is_refused(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    with pytest.raises(BrokerError) as exc:
        b.advance_to("15:40")
    assert "30-minute stamps" in str(exc.value)


def test_clock_advances_and_requotes(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("11:00")
    s11 = b.spx_spot()
    b.advance_to("15:30")
    assert b.spx_spot() != s11
    assert (b.now_et().hour, b.now_et().minute) == (15, 30)
    # the 11:00 strikes are still quoted at 15:30 (they must be, to exit)
    bid, ask, mid = b.package_quote(4555.0, 4550.0)
    assert bid == pytest.approx(6.25, abs=1e-6)
    assert ask == pytest.approx(6.70, abs=1e-6)
    assert mid == pytest.approx(6.475, abs=1e-6)


def test_an_unlisted_strike_is_refused_the_way_ib_refuses_it(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("11:00")
    assert b.spxw_0dte_contracts("20231201") == []
    with pytest.raises(BrokerError) as exc:
        b.contract_for(1234.0, "C")
    assert "no qualified contract" in str(exc.value)
    # a fabricated contract still gets the vendor's no-quote sentinel
    q = b.quotes([FakeOption(strike=1234.0, right="C", conId=1)])[0]
    assert (q.bid, q.ask) == (0.0, 0.0)
    assert q.mid != q.mid  # NaN
    assert not q.live


def test_wings_are_not_quoted_in_replay(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("11:00")
    spot = b.spx_spot()
    quotes = b.quotes(b.spxw_0dte_contracts("20231130"))
    body, _ = pick_nearest_otm_reason(quotes, spot)
    assert body is not None
    # The tape serves an ATM band only, so 1.5%-of-spot wings have no quote
    # and the fly cannot be entered off a replay.
    assert pick_wings(quotes, body, spot * 0.015) is None


# ------------------------------------------------------------------ fills


def test_combo_fills_crossed_and_at_the_mid(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("11:00")
    legs = [
        (b.contract_for(4555.0, "C"), 1, "BUY"),
        (b.contract_for(4550.0, "P"), 1, "BUY"),
    ]
    sell = b.place_combo(legs, 1, ENTRY_MID, action="SELL", what="body_entry")
    assert sell.ok and sell.price == pytest.approx(ENTRY_BID, abs=1e-6)
    assert sell.status == "Filled" and sell.order_id > 0
    buy = b.place_combo(legs, 1, ENTRY_MID, action="BUY", what="body_exit")
    assert buy.ok and buy.price == pytest.approx(ENTRY_ASK, abs=1e-6)
    assert buy.order_id != sell.order_id

    b2 = make_broker(tmp_path, frame, fill_mode="mid")
    b2.advance_to("11:00")
    legs2 = [
        (b2.contract_for(4555.0, "C"), 1, "BUY"),
        (b2.contract_for(4550.0, "P"), 1, "BUY"),
    ]
    mid = b2.place_combo(legs2, 1, ENTRY_MID, action="SELL", what="body_entry")
    assert mid.price == pytest.approx(ENTRY_MID, abs=1e-6)


def test_combo_with_an_unquoted_leg_does_not_fill(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("11:00")
    legs = [
        (b.contract_for(4555.0, "C"), 1, "BUY"),
        (FakeOption(strike=1234.0, right="P", conId=2), 1, "BUY"),
    ]
    fill = b.place_combo(legs, 1, 10.0, action="SELL")
    assert not fill.ok and fill.quantity == 0
    assert "no quote" in fill.note and fill.status == "NoQuote"


def test_es_and_mes_are_both_simulated(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    b.advance_to("11:00")
    es = b.place_futures(b.futures_contract("ES"), -2, what="futures")
    mes = b.place_futures(b.futures_contract("MES"), 3, what="futures")
    assert es.symbol == "ES" and es.multiplier == 50.0 and es.quantity == -2
    assert mes.symbol == "MES" and mes.multiplier == 5.0 and mes.quantity == 3
    assert b.positions()["futures"] == {"ES": -2, "MES": 3}
    with pytest.raises(ValueError):
        b.futures_contract("NQ")


def test_the_futures_reference_is_not_the_index(tmp_path, frame):
    """Marking the hedge at SPX is the phantom-P&L bug L-5; the fake shows it."""
    b = make_broker(tmp_path, frame, futures_basis_pts=18.0)
    b.advance_to("11:00")
    assert b.futures_reference("ES") == pytest.approx(b.spx_spot() + 18.0)
    assert b.futures_reference("MES") == b.futures_reference("ES")
    f = b.place_futures(b.futures_contract("ES"), 1)
    assert f.price == pytest.approx(b.spx_spot() + 18.0)


# ----------------------------------------------------------------- faults


def test_an_order_can_be_refused(tmp_path, frame):
    b = make_broker(tmp_path, frame, faults=FakeFaults(refuse=frozenset({"body_exit"})))
    b.advance_to("11:00")
    legs = [
        (b.contract_for(4555.0, "C"), 1, "BUY"),
        (b.contract_for(4550.0, "P"), 1, "BUY"),
    ]
    ok = b.place_combo(legs, 1, ENTRY_MID, action="SELL", what="body_entry")
    assert ok.ok
    bad = b.place_combo(legs, 1, ENTRY_MID, action="BUY", what="body_exit")
    assert not bad.ok and bad.quantity == 0 and bad.status == "Cancelled"
    assert bad.price != bad.price  # NaN: nothing traded


def test_an_order_can_be_rejected_with_a_reason(tmp_path, frame):
    b = make_broker(
        tmp_path, frame, faults=FakeFaults(reject={"body_entry": "no route to SMART"})
    )
    b.advance_to("11:00")
    legs = [(b.contract_for(4555.0, "C"), 1, "BUY")]
    fill = b.place_combo(legs, 1, 5.0, action="SELL", what="body_entry")
    assert not fill.ok and fill.status == "Rejected" and fill.quantity == 0
    assert "no route to SMART" in fill.note


def test_an_order_can_fill_partially_and_keeps_its_sign(tmp_path, frame):
    b = make_broker(tmp_path, frame, faults=FakeFaults(partial={"futures": 2}))
    b.advance_to("11:00")
    fill = b.place_futures(b.futures_contract("MES"), -7, what="futures")
    assert not fill.ok and fill.status == "Submitted"
    assert fill.quantity == -2  # signed by the order, not by ``ok``
    assert b.positions()["futures"] == {"MES": -2}
    # a smaller order than the cap fills whole
    whole = b.place_futures(b.futures_contract("MES"), -1, what="futures")
    assert whole.ok and whole.quantity == -1


def test_the_socket_can_go_down_at_a_clock(tmp_path, frame):
    b = make_broker(
        tmp_path,
        frame,
        faults=FakeFaults(disconnect_at=frozenset({"13:00"}), reconnect_delay_s=120.0),
    )
    b.advance_to("11:00")
    assert b.is_connected()
    b.advance_to("13:00")
    assert not b.is_connected()
    assert b.ensure_connected()
    assert b.now_et().strftime("%H:%M:%S") == "13:02:00"  # 120 s late
    b.advance_to("13:30")  # it goes down once, not forever
    assert b.is_connected()


def test_a_snapshot_can_be_stale(tmp_path, frame):
    b = make_broker(tmp_path, frame, faults=FakeFaults(stale_at=frozenset({"14:00"})))
    b.advance_to("11:00")
    b.quotes(b.spxw_0dte_contracts("20231130"))
    assert b.quote_health().ok
    b.advance_to("14:00")
    b.quotes(b.spxw_0dte_contracts("20231130"))
    health = b.quote_health()
    assert not health.ok and health.stale and "stale" in health.detail


# ------------------------------------------------------------ plumbing


def test_liquid_hours_and_account(tmp_path, frame):
    b = make_broker(tmp_path, frame)
    open_et, close_et = b.liquid_hours(None)
    assert (open_et.hour, open_et.minute) == (9, 30)
    assert (close_et.hour, close_et.minute) == (16, 0)
    half = make_broker(tmp_path, frame, liquid_close="13:00")
    assert half.liquid_hours(None)[1].hour == 13
    assert b.account_summary()["AccountType"] == "REPLAY"
    assert b.account_summary()["AccountId"].startswith("DU")


def test_never_opens_a_socket(tmp_path, frame, monkeypatch):
    """A whole replay session with ``socket.socket`` booby-trapped."""
    import socket

    def boom(*a, **k):  # pragma: no cover - the point is that it is not called
        raise AssertionError("FakeBroker touched the network")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)

    b = make_broker(tmp_path, frame)
    for clock in ("10:00", "11:00", "13:30", "14:30", "15:30", "16:00"):
        b.advance_to(clock)
        b.spx_spot()
        b.quotes(b.spxw_0dte_contracts("20231130"))
        b.place_futures(b.futures_contract("MES"), 1)
    b.settlement_spot()
    assert b.is_connected()
    assert not hasattr(b, "ib")


def test_the_replay_cache_is_keyed_on_the_band_it_was_cut_with(tmp_path):
    """A cache key of the date alone served a stale band (L-25)."""
    cache = str(tmp_path / "replay_cache")
    df1 = load_replay_day(DAY, cache_dir=cache, band_pct=0.01)
    files = os.listdir(cache)
    assert len(files) == 1 and files[0].startswith(DAY + "_")
    # same key -> served from the cache, chain file never touched
    df2 = load_replay_day(DAY, cache_dir=cache, band_pct=0.01)
    assert len(df1) == len(df2)
    # a different band is a different key, so it must NOT be served the old cut
    df3 = load_replay_day(DAY, cache_dir=cache, band_pct=0.02)
    assert len(os.listdir(cache)) == 2
    assert len(df3) > len(df1)
    # and a different chain file cannot be served the old one either
    with pytest.raises(Exception):
        load_replay_day(
            DAY, cache_dir=cache, chain_path="/does/not/exist.parquet", band_pct=0.01
        )


def test_the_replay_cache_is_not_in_the_live_journal_tree(tmp_path, frame):
    cfg = Config(
        replay_date=DAY,
        journal_dir=str(tmp_path / "journal"),
        replay_cache_dir=str(tmp_path / "cache"),
    )
    assert not str(cfg.replay_cache_dir).startswith(str(cfg.journal_dir))
    assert "live_journal" not in Config().replay_cache_dir


def test_a_session_with_no_chain_rows_is_refused(tmp_path):
    with pytest.raises(BrokerError) as exc:
        load_replay_day("2021-12-24", cache_dir=None)
    assert "no SPXW chain rows" in str(exc.value)
