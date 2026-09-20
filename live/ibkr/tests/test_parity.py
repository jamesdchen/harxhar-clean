"""Parity: the live engine must reproduce the backtest's 11:00 hedge tape.

The vendor-implied replay is the same formula on the same inputs, so both the
exit book (buy the straddle back at 15:30) and the hold book (cash settlement)
must agree with ``hold_mark_1100`` to floating point.  A 30-day subsample keeps
this fast; ``python -m live.ibkr.parity`` runs all 865 scored days.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.ibkr.parity import (  # noqa: E402
    ENTRY_CLOCK,
    EXIT_CLOCK,
    PARITY_TOL,
    ParityError,
    _argv_flag,
    _require_close,
    latest_trade_cache,
    parity_report,
    replay_day,
    repo_root,
    require_inputs,
    trade_cache_candidates,
)

SUBSAMPLE_DAYS = 30

_ROOT_REPO = repo_root()
_HAVE_DATA = (
    (_ROOT_REPO / "results" / "atm_straddle_intraday_holdclose" / "cache").is_dir()
    and (
        _ROOT_REPO / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
    ).exists()
    and (_ROOT_REPO / "data" / "spxw_chain.parquet").exists()
)
needs_data = pytest.mark.skipif(not _HAVE_DATA, reason="research data not present")


@pytest.fixture(scope="module")
def report() -> dict:
    return parity_report(n_days=SUBSAMPLE_DAYS, write=False)


@needs_data
def test_exit_book_matches_the_research(report: dict) -> None:
    """Enter 11:00, delta-hedge, buy back at 15:30: identical to hold_mark_1100."""
    assert report["n_days"] == SUBSAMPLE_DAYS
    assert report["exit_vendor"]["max_abs"] < PARITY_TOL


@needs_data
def test_hold_book_matches_the_research(report: dict) -> None:
    """Enter 11:00, delta-hedge, hold through cash settlement."""
    assert report["hold_vendor"]["max_abs"] < PARITY_TOL


@needs_data
def test_entry_premium_is_rebuilt_from_the_leg_quotes(report: dict) -> None:
    assert report["entry_premium_max_abs_diff"] == pytest.approx(0.0, abs=1e-12)


@needs_data
def test_no_day_is_refused_on_the_subsample(report: dict) -> None:
    assert report["refused"] == []


@needs_data
def test_replay_day_hedges_the_short_with_long_futures() -> None:
    """A positive package delta means a LONG futures leg for the short straddle."""
    import pandas as pd

    pkg = pd.read_parquet(latest_trade_cache(_ROOT_REPO))
    pkg["date"] = pd.to_datetime(pkg["date"])
    day = pkg["date"].iloc[0]
    rows = pkg[pkg["date"] == day].copy()
    rows["iv_hourly_used"] = rows["iv_hourly"]
    res = replay_day(rows, n_straddles=1, exit_clock=EXIT_CLOCK, iv_mode="vendor")
    assert res["ok"]
    assert res["stamps"][0]["hhmm"] == ENTRY_CLOCK
    assert res["stamps"][-1]["hhmm"] == EXIT_CLOCK
    # The exit stamp is flattened, so it carries no spot step.
    assert res["stamps"][-1]["dS"] == 0.0
    for st in res["stamps"]:
        assert -1.0 <= st["delta_pkg"] <= 1.0
        assert st["target_futures"] * st["delta_pkg"] >= 0


# -------------------------------------------- the gate, and the refusals -----
def test_parity_tol_is_referenced_and_raises() -> None:
    """PARITY_TOL used to be declared and never read; it is now the gate."""
    assert _require_close("a reproduction", 1.0, 1.0 + 1e-12) <= PARITY_TOL
    with pytest.raises(ParityError, match="1e-09"):
        _require_close("a reproduction", 1.0, 1.0 + 1e-6)


def test_missing_inputs_raise_rather_than_skip(tmp_path: Path) -> None:
    """On a clean clone the gate must fail, not vanish."""
    with pytest.raises(FileNotFoundError) as excinfo:
        require_inputs(tmp_path)
    message = str(excinfo.value)
    assert "option chain" in message
    assert "46 selector timeline" in message


def test_the_deck_flag_does_not_leak_into_argv() -> None:
    """A permanently appended --dh-holdclose broke pytest and any later main()."""
    before = list(sys.argv)
    with _argv_flag("--dh-holdclose"):
        assert "--dh-holdclose" in sys.argv
    assert sys.argv == before
    # ... and it restores argv even when the block raises
    with pytest.raises(RuntimeError), _argv_flag("--dh-holdclose"):
        raise RuntimeError("boom")
    assert sys.argv == before


@needs_data
def test_the_trade_cache_is_picked_by_modification_time() -> None:
    """The names are content hashes: lexicographic order is not recency."""
    candidates = trade_cache_candidates(_ROOT_REPO)
    assert candidates
    assert latest_trade_cache(_ROOT_REPO) == candidates[0]
    times = [q.stat().st_mtime for q in candidates]
    assert times == sorted(times, reverse=True)


def test_no_trade_cache_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "results" / "atm_straddle_intraday_holdclose" / "cache").mkdir(
        parents=True
    )
    with pytest.raises(FileNotFoundError, match="no trade cache"):
        latest_trade_cache(tmp_path)


# ------------------------------------------------------ a hole in the tape ---
def _one_day(clocks: tuple[str, ...]) -> Any:
    """One synthetic session on the full 30-minute grid, spot inside the body."""
    import pandas as pd

    rows = []
    for i, hhmm in enumerate(clocks):
        rows.append(
            {
                "date": pd.Timestamp("2024-06-12"),
                "hhmm": hhmm,
                "S": 5000.0 + 0.3 * i,
                "K_c": 5005.0,
                "K_p": 5000.0,
                "bid_c": 4.0,
                "ask_c": 4.4,
                "bid_p": 4.0,
                "ask_p": 4.4,
                "entry": 8.4,
                "iv_hourly_used": 0.002,
                "S_close": 5004.0,
                "n_live": 40,
            }
        )
    return pd.DataFrame(rows)


def test_a_missing_stamp_is_an_error_not_a_zero_hedge_step() -> None:
    """d_spot = 0.0 on a hole reported a day the engine did not replay."""
    from live.ibkr.parity import SESSION_CLOCKS

    whole = _one_day(SESSION_CLOCKS)
    good = replay_day(whole, exit_clock=EXIT_CLOCK, iv_mode="vendor")
    assert good["ok"]
    assert good["n_gap_steps"] == 0

    holed = whole[whole["hhmm"] != "13:00"]
    bad = replay_day(holed, exit_clock=EXIT_CLOCK, iv_mode="vendor")
    assert not bad["ok"]
    assert bad["reason"] == "stamp_gap"
    # the hole costs two steps: the one into it and the one out of it
    assert bad["n_gap_steps"] == 2


def test_a_hole_outside_the_held_window_is_not_a_gap() -> None:
    """The book only carries the steps between its entry and its exit."""
    from live.ibkr.parity import SESSION_CLOCKS

    holed = _one_day(SESSION_CLOCKS)
    holed = holed[holed["hhmm"] != "10:30"]
    res = replay_day(holed, entry_clock=ENTRY_CLOCK, exit_clock=EXIT_CLOCK)
    assert res["ok"]
    assert res["n_gap_steps"] == 0


def test_a_missing_settlement_is_a_gap_for_the_hold_book_only() -> None:
    from live.ibkr.parity import SESSION_CLOCKS

    whole = _one_day(SESSION_CLOCKS)
    whole = whole.assign(S_close=float("nan"))
    held = replay_day(whole, exit_clock=None, iv_mode="vendor")
    assert not held["ok"]
    assert held["reason"] == "stamp_gap"
    assert held["n_gap_steps"] == 1
    # the exit book never carries a step past 15:30, so it does not care
    flat = replay_day(whole, exit_clock=EXIT_CLOCK, iv_mode="vendor")
    assert flat["ok"]


def test_replay_day_takes_any_entry_clock() -> None:
    """The redesign enters in the afternoon; the replay must follow it there."""
    from live.ibkr.parity import SESSION_CLOCKS

    whole = _one_day(SESSION_CLOCKS)
    res = replay_day(whole, entry_clock="13:30", exit_clock=None)
    assert res["ok"]
    assert res["entry_clock"] == "13:30"
    assert [st["hhmm"] for st in res["stamps"]] == list(SESSION_CLOCKS[7:])
    assert res["exit_source"] == "settlement"
    missing = replay_day(whole, entry_clock="09:30")
    assert not missing["ok"]
    assert missing["reason"] == "no_entry_row"


# ----------------------------------------------------- the ledger's tape -----
def test_realized_remaining_variance_is_the_steps_the_hedge_walks() -> None:
    """From the stamp to 15:30, plus the last step into the settlement print."""
    import numpy as np

    from live.ibkr.parity import realized_remaining_variance

    spot = np.array([[100.0, 110.0, 121.0]])
    chain = {"S": spot, "S_close": np.array([133.1])}
    got = realized_remaining_variance(chain)
    step = np.log(1.1) ** 2
    assert got[0, 2] == pytest.approx(step)
    assert got[0, 1] == pytest.approx(2 * step)
    assert got[0, 0] == pytest.approx(3 * step)


def test_a_hole_in_the_spot_path_leaves_no_remaining_variance() -> None:
    """A partial window is not a shorter window; it is no window."""
    import numpy as np

    from live.ibkr.parity import realized_remaining_variance

    spot = np.array([[100.0, float("nan"), 121.0]])
    got = realized_remaining_variance({"S": spot, "S_close": np.array([133.1])})
    assert np.isnan(got[0, 0])
    assert np.isnan(got[0, 1])
    assert got[0, 2] == pytest.approx(np.log(1.1) ** 2)
    # a missing settlement kills every window, including the last step's
    holed = realized_remaining_variance(
        {"S": np.array([[100.0, 110.0]]), "S_close": np.array([float("nan")])}
    )
    assert np.isnan(holed).all()
