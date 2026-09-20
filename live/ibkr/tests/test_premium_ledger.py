"""The premium ledger: idempotence, the lag, the shared mask and the warm-up."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.ibkr.premium_ledger import (  # noqa: E402
    LEDGER_COLUMNS,
    MIN_SESSIONS,
    ClockRecord,
    PremiumLedger,
    records_from_grids,
)

CLOCK = "11:00"
OTHER = "13:30"
DAY0 = date(2024, 1, 2)


def day(i: int) -> date:
    """A session key; the ledger never reads a calendar, only the row order."""
    return DAY0 + timedelta(days=i)


def rec(
    i: int,
    implied: float = 2.0,
    realized: float = 1.0,
    clock: str = CLOCK,
) -> ClockRecord:
    return ClockRecord(
        session=day(i),
        clock=clock,
        implied_rem_var=implied,
        realized_rem_var=realized,
        spot=5000.0,
        premium_mid=20.0,
        kc=5005.0,
        kp=5000.0,
    )


def ledger(n: int = 10, **kw: float) -> PremiumLedger:
    led = PremiumLedger(min_sessions=1)
    led.append_session([rec(i, **kw) for i in range(n)])  # type: ignore[arg-type]
    return led


# ------------------------------------------------------------------ shape ---
def test_columns_and_sessions() -> None:
    led = ledger(3)
    assert list(led.frame.columns) == list(LEDGER_COLUMNS)
    assert led.sessions() == [day(0), day(1), day(2)]
    assert led.clocks() == [CLOCK]


def test_append_session_is_idempotent_per_session() -> None:
    """Re-appending a session REPLACES its rows; it never duplicates them.

    A session arrives as one batch of records -- every clock of that day -- so
    replacing the session wholesale is what makes a re-run of a day a no-op.
    """
    led = ledger(3)
    before = led.frame
    led.append_session([rec(1)])
    assert led.frame.equals(before)
    assert len(led.frame) == 3
    # a changed record for the same session overwrites, it does not append
    led.append_session([rec(1, implied=9.0)])
    assert len(led.frame) == 3
    row = led.frame.loc[led.frame["session"] == pd.Timestamp(day(1))].iloc[0]
    assert row["implied_rem_var"] == 9.0
    # the batch is the whole session: re-appending it with two clocks replaces
    # the one-clock row with both, and leaves the other sessions alone
    led.append_session([rec(1), rec(1, clock=OTHER)])
    assert len(led.frame) == 4
    assert led.clocks() == [CLOCK, OTHER]
    assert led.sessions() == [day(0), day(1), day(2)]
    # a session the batch does not mention keeps every row it had
    assert (led.frame["session"] == pd.Timestamp(day(0))).sum() == 1


def test_repeated_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="repeated"):
        PremiumLedger().append_session([rec(0), rec(0, implied=3.0)])


def test_a_clock_off_the_half_hour_is_refused() -> None:
    with pytest.raises(ValueError, match="half-hour"):
        PremiumLedger().append_session([rec(0, clock="11:15")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="HH:MM"):
        PremiumLedger().append_session([rec(0, clock="1100")])  # type: ignore[arg-type]


def test_round_trip_through_parquet(tmp_path: Path) -> None:
    led = ledger(5)
    led.save(tmp_path / "sub" / "ledger.parquet")
    back = PremiumLedger.load(tmp_path / "sub" / "ledger.parquet", min_sessions=1)
    assert back.frame.equals(led.frame)
    assert back.sessions() == led.sessions()
    with pytest.raises(FileNotFoundError):
        PremiumLedger.load(tmp_path / "nope.parquet")


# -------------------------------------------------------------------- lag ---
def test_a_session_never_enters_its_own_statistic() -> None:
    """The defining causality property: asof reads strictly earlier rows."""
    led = PremiumLedger(min_sessions=1)
    led.append_session([rec(i, implied=2.0, realized=1.0) for i in range(5)])
    led.append_session([rec(5, implied=1000.0, realized=1.0)])
    # The huge session is invisible to its own asof, and to every earlier one.
    assert led.ratio_of_sums(CLOCK, day(5), None) == pytest.approx(2.0)
    assert led.ratio_of_sums(CLOCK, day(3), None) == pytest.approx(2.0)
    # It enters the next session's.
    assert led.ratio_of_sums(CLOCK, day(6), None) == pytest.approx(
        (5 * 2.0 + 1000.0) / 6.0
    )


def test_ratio_of_sums_is_a_ratio_of_sums_not_a_mean_of_ratios() -> None:
    """One big session dominates the ratio; a mean of ratios would not see it."""
    led = PremiumLedger(min_sessions=1)
    led.append_session([rec(i, implied=1.0, realized=1.0) for i in range(4)])
    led.append_session([rec(4, implied=100.0, realized=400.0)])
    assert led.ratio_of_sums(CLOCK, day(5), None) == pytest.approx(104.0 / 404.0)
    # the mean of the five per-session ratios would be (4 * 1 + 0.25) / 5 = 0.85
    assert led.ratio_of_sums(CLOCK, day(5), None) != pytest.approx(0.85)


def test_window_counts_session_rows() -> None:
    led = PremiumLedger(min_sessions=1)
    led.append_session([rec(i, implied=1.0, realized=1.0) for i in range(10)])
    led.append_session([rec(i, implied=3.0, realized=1.0) for i in range(10, 20)])
    assert led.ratio_of_sums(CLOCK, day(20), 10) == pytest.approx(3.0)
    assert led.ratio_of_sums(CLOCK, day(20), 20) == pytest.approx(2.0)
    assert led.ratio_of_sums(CLOCK, day(20), None) == pytest.approx(2.0)
    assert led.ratio_of_sums(CLOCK, day(20), 5) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="positive"):
        led.ratio_of_sums(CLOCK, day(20), 0)
    with pytest.raises(ValueError, match="positive"):
        led.ratio_of_sums(CLOCK, day(20), -5)


def test_asof_need_not_be_a_ledger_session() -> None:
    """The live path asks on a session the ledger has not recorded yet."""
    led = ledger(5)
    later = day(99)
    assert led.ratio_of_sums(CLOCK, later, None) == pytest.approx(2.0)
    earlier = DAY0 - timedelta(days=1)
    assert isnan(led.ratio_of_sums(CLOCK, earlier, None))


# ------------------------------------------------------------- warm-up -----
def test_warm_up_refuses_until_min_sessions() -> None:
    led = PremiumLedger()  # the default warm-up, 63
    led.append_session([rec(i) for i in range(MIN_SESSIONS + 1)])
    assert isnan(led.ratio_of_sums(CLOCK, day(MIN_SESSIONS - 1), None))
    assert isnan(led.realized_over_implied_mean(CLOCK, day(MIN_SESSIONS - 1), None))
    assert not isnan(led.ratio_of_sums(CLOCK, day(MIN_SESSIONS), None))
    assert not isnan(led.realized_over_implied_mean(CLOCK, day(MIN_SESSIONS), None))
    # the override is per call and does not touch the ledger's own convention
    assert not isnan(led.ratio_of_sums(CLOCK, day(3), None, min_sessions=3))
    assert led.min_sessions == MIN_SESSIONS


def test_the_warm_up_counts_usable_rows_not_calendar_rows() -> None:
    """Rows 0, 2, 4 are usable; rows 1, 3, 5 carry a NaN implied variance."""
    led = PremiumLedger(min_sessions=3)
    led.append_session(
        [rec(i, implied=float("nan") if i % 2 else 2.0) for i in range(6)]
    )
    # four calendar rows behind day(4), but only two of them usable
    assert isnan(led.ratio_of_sums(CLOCK, day(4), None))
    # day(5) has rows 0..4 behind it: three usable, so the warm-up is over
    assert not isnan(led.ratio_of_sums(CLOCK, day(5), None))


# ------------------------------------------------------------- the mask ----
def test_implied_and_realized_share_one_mask() -> None:
    """A session missing either side contributes to NEITHER sum."""
    led = PremiumLedger(min_sessions=1)
    led.append_session([rec(i, implied=2.0, realized=1.0) for i in range(3)])
    led.append_session([rec(3, implied=1000.0, realized=float("nan"))])
    assert led.ratio_of_sums(CLOCK, day(4), None) == pytest.approx(2.0)
    led.append_session([rec(4, implied=float("nan"), realized=1000.0)])
    assert led.ratio_of_sums(CLOCK, day(5), None) == pytest.approx(2.0)


def test_non_positive_variances_are_not_usable() -> None:
    led = PremiumLedger(min_sessions=1)
    led.append_session([rec(i, implied=2.0, realized=1.0) for i in range(3)])
    led.append_session([rec(3, implied=0.0, realized=1.0)])
    led.append_session([rec(4, implied=2.0, realized=-1.0)])
    assert led.ratio_of_sums(CLOCK, day(5), None) == pytest.approx(2.0)


def test_clocks_are_independent() -> None:
    led = PremiumLedger(min_sessions=1)
    led.append_session(
        [rec(i, implied=2.0, realized=1.0) for i in range(4)]
        + [rec(i, implied=1.0, realized=4.0, clock=OTHER) for i in range(4)]
    )
    assert led.ratio_of_sums(CLOCK, day(4), None) == pytest.approx(2.0)
    assert led.ratio_of_sums(OTHER, day(4), None) == pytest.approx(0.25)
    assert isnan(led.ratio_of_sums("15:00", day(4), None))


# ------------------------------------------------------- the V9 factor -----
def test_realized_over_implied_mean_is_a_mean_of_ratios() -> None:
    """The V9 factor is the MEAN of realized/implied, not a ratio of sums."""
    led = PremiumLedger(min_sessions=1)
    led.append_session([rec(0, implied=1.0, realized=1.0)])
    led.append_session([rec(1, implied=100.0, realized=25.0)])
    assert led.realized_over_implied_mean(CLOCK, day(2), None) == pytest.approx(
        (1.0 + 0.25) / 2.0
    )
    assert led.ratio_of_sums(CLOCK, day(2), None) == pytest.approx(101.0 / 26.0)


def test_series_helpers_agree_with_the_scalar_calls() -> None:
    """The whole-tape convenience must be the same call, session by session."""
    led = PremiumLedger(min_sessions=3)
    rng = np.random.default_rng(0)
    led.append_session(
        [
            rec(i, implied=float(rng.uniform(1, 5)), realized=float(rng.uniform(1, 5)))
            for i in range(40)
        ]
    )
    for window in (None, 5, 12):
        a = led.ratio_of_sums_series(CLOCK, window).to_numpy(float)
        b = np.array(
            [led.ratio_of_sums(CLOCK, d, window) for d in led.sessions()], dtype=float
        )
        assert np.array_equal(a, b, equal_nan=True)
        a = led.realized_over_implied_series(CLOCK, window).to_numpy(float)
        b = np.array(
            [led.realized_over_implied_mean(CLOCK, d, window) for d in led.sessions()],
            dtype=float,
        )
        assert np.array_equal(a, b, equal_nan=True)


def test_records_from_grids_keeps_every_cell() -> None:
    days = [day(i) for i in range(3)]
    clocks = [CLOCK, OTHER]
    grid = np.arange(6, dtype=float).reshape(3, 2)
    recs = records_from_grids(days, clocks, grid, grid + 1, grid, grid, grid, grid)
    assert len(recs) == 6
    assert recs[0].session == days[0]
    assert recs[1].clock == OTHER
    assert recs[5].implied_rem_var == 5.0
    assert recs[5].realized_rem_var == 6.0


def test_a_tz_aware_session_key_is_refused() -> None:
    frame = pd.DataFrame(
        {
            "session": pd.to_datetime(["2024-01-02"]).tz_localize("UTC"),
            "clock": ["11:00"],
            "implied_rem_var": [1.0],
            "realized_rem_var": [1.0],
            "spot": [5000.0],
            "premium_mid": [20.0],
            "kc": [5005.0],
            "kp": [5000.0],
        }
    )
    with pytest.raises(ValueError, match="tz-naive"):
        PremiumLedger(frame)


def test_a_missing_column_is_refused() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        PremiumLedger(pd.DataFrame({"session": [], "clock": []}))


# --------------------------------------- the deleveraging candidate (50) ---

SEED_LEDGER = _ROOT / "results" / "live_seed" / "premium_ledger.parquet"
P50_PATH = _ROOT / "writeup" / "intraday_proposals" / "50_cat_replay_repriced.py"
P49_PATH = _ROOT / "writeup" / "intraday_proposals" / "49_cat_replay.py"


def load_p50():
    """Proposal 50 itself, imported by path.  Its body runs; ``main`` does not."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("p50_for_delever_gate", P50_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def p50_rule(rv21: np.ndarray, p50) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Proposal 50's part-B2 weights: its own helper, its own constants."""
    q_half = p50.expanding_pct(rv21, p50.DELEVER_HALF_PCT, p50.DELEVER_MIN_OBS)
    q_zero = p50.expanding_pct(rv21, p50.DELEVER_ZERO_PCT, p50.DELEVER_MIN_OBS)
    w: np.ndarray = np.ones(len(rv21))
    w = np.where(np.isfinite(q_half) & (rv21 > q_half), 0.5, w)
    w = np.where(np.isfinite(q_zero) & (rv21 > q_zero), 0.0, w)
    return w


def rv_ledger(values: list[float], clock: str = "10:00") -> PremiumLedger:
    """A ledger whose 10:00 REALIZED remaining variance is ``values``."""
    led = PremiumLedger(min_sessions=1)
    led.append_session(
        [
            ClockRecord(
                session=day(i),
                clock=clock,
                implied_rem_var=1.0,
                realized_rem_var=float(v),
                spot=5000.0,
                premium_mid=20.0,
                kc=5005.0,
                kp=5000.0,
            )
            for i, v in enumerate(values)
        ]
    )
    return led


def test_the_session_rv_series_is_the_first_clocks_realized_variance() -> None:
    led = rv_ledger([1.0, 2.0, 3.0, float("nan"), 5.0])
    s = led.session_rv_series()
    assert list(s.index) == list(pd.to_datetime([day(i) for i in range(5)]))
    assert s.to_numpy(float)[0] == 1.0
    assert isnan(s.to_numpy(float)[3])  # not finite and positive -> masked
    assert led.session_realized_var(day(2)) == 3.0
    assert isnan(led.session_realized_var(day(3)))
    assert isnan(led.session_realized_var(day(99)))  # not a ledger session


def test_rv21_is_lagged_and_skips_the_blank_sessions() -> None:
    """The window is the ``window`` USABLE sessions STRICTLY BEFORE asof."""
    led = rv_ledger([float(i) for i in range(1, 11)])
    assert isnan(led.rv21(day(2), window=3))  # only two prior sessions
    assert led.rv21(day(3), window=3) == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    assert led.rv21(day(4), window=3) == pytest.approx((2.0 + 3.0 + 4.0) / 3.0)
    # a blank session does not blank the next ``window`` windows: it is skipped
    holed = rv_ledger([1.0, 2.0, float("nan"), 3.0, 4.0])
    assert holed.rv21(day(4), window=3) == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    # asof need not be a ledger session at all: that is the live case
    assert led.rv21(day(500), window=3) == pytest.approx((8.0 + 9.0 + 10.0) / 3.0)
    with pytest.raises(ValueError, match="positive"):
        led.rv21(day(5), window=0)


def test_the_multiplier_is_full_half_and_zero_on_a_synthetic_regime() -> None:
    """One quiet history of 300 sessions, then a session in each state.

    The rule is lagged twice over: the thresholds come from the RV21 of the
    sessions strictly before ``asof``, and RV21 itself is the mean of the
    ``window`` sessions strictly before ``asof``.  With ``window=1`` the last
    appended variance IS ``asof``'s RV21, and it never enters its own
    thresholds -- which is the whole point.
    """
    quiet = [float(v) for v in np.linspace(1.0, 2.0, 300)]

    led = rv_ledger(quiet + [1.5])
    m, info = led.delever_multiplier(day(301), window=1, min_sessions=252)
    assert (m, info["state"]) == (1.0, "full")
    assert info["n_prior"] == 300
    assert float(info["rv21"]) == pytest.approx(1.5)
    p_half, p_zero = float(info["p_half"]), float(info["p_zero"])
    assert p_half < p_zero < 2.0

    led = rv_ledger(quiet + [500.0])
    m, info = led.delever_multiplier(day(301), window=1, min_sessions=252)
    assert (m, info["state"]) == (0.0, "zero")
    assert float(info["p_zero"]) == pytest.approx(p_zero)  # same history

    led = rv_ledger(quiet + [0.5 * (p_half + p_zero)])
    m, info = led.delever_multiplier(day(301), window=1, min_sessions=252)
    assert (m, info["state"]) == (0.5, "half")
    assert p_half < float(info["rv21"]) <= float(info["p_zero"])


def test_the_multiplier_is_full_and_says_warmup_below_the_minimum() -> None:
    led = rv_ledger([float(v) for v in np.linspace(1.0, 2.0, 300)])
    m, info = led.delever_multiplier(day(100), window=1, min_sessions=252)
    # RV21 is finite on day(1)..day(99): 99 of the 252 the thresholds need
    assert (m, info["state"], info["n_prior"]) == (1.0, "warmup", 99)
    assert isnan(float(info["p_half"])) and isnan(float(info["p_zero"]))
    # the RV window itself warming up is a warm-up too, never a zero
    m, info = led.delever_multiplier(day(0), window=21, min_sessions=1)
    assert (m, info["state"]) == (1.0, "warmup")
    assert isnan(float(info["rv21"]))


def test_the_deleveraging_quantiles_are_checked() -> None:
    led = rv_ledger([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="p_half < p_zero"):
        led.delever_multiplier(day(2), p_half=0.99, p_zero=0.90)
    with pytest.raises(ValueError, match="min_sessions"):
        led.delever_multiplier(day(2), min_sessions=0)


@pytest.mark.skipif(
    not (SEED_LEDGER.exists() and P50_PATH.exists() and P49_PATH.exists()),
    reason="the seed ledger and proposals 49/50 are needed for the gate",
)
def test_the_multiplier_matches_proposal_50_session_by_session() -> None:
    """THE GATE.  Proposal 50's own rule, its own helper, the same sessions.

    50 measured the rule on the 30-minute PANEL -- 6,593 sessions, 6,134 full
    / 318 half / 141 zero -- and the live ledger is a different, much shorter
    history, so the COUNTS cannot match: the expanding percentiles start at
    the ledger's first session, 2020-01-03.  What must match is the RULE.
    50's ``expanding_pct`` and its three constants, evaluated on the same
    trailing-variance series, give the ledger's multiplier on every one of
    the 1279 chain sessions.
    """
    p50 = load_p50()
    assert (p50.DELEVER_HALF_PCT, p50.DELEVER_ZERO_PCT, p50.DELEVER_MIN_OBS) == (
        90.0,
        97.5,
        252,
    )
    led = PremiumLedger.load(SEED_LEDGER)
    sessions = led.sessions()
    assert len(sessions) == 1279

    # proposal 49's holiday-safe ``rv_trail``, on the ledger's own RV series
    rv = led.session_rv_series()
    usable = rv.dropna()
    trail = (
        usable.rolling(21, min_periods=21)
        .mean()
        .shift(1)
        .reindex(rv.index, method="ffill")
        .to_numpy(float)
    )
    want = p50_rule(trail, p50)
    mine = np.array([led.delever_multiplier(d)[0] for d in sessions], dtype=float)
    assert np.array_equal(mine, want)

    counts = {v: int((mine == v).sum()) for v in (1.0, 0.5, 0.0)}
    assert counts == {1.0: 1223, 0.5: 38, 0.0: 18}
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])
    assert str(idx[mine == 0.5][0].date()) == "2022-02-22"
    assert str(idx[mine == 0.0][0].date()) == "2025-04-09"


@pytest.mark.skipif(
    not SEED_LEDGER.exists(), reason="the seed ledger is not in this checkout"
)
def test_the_ledgers_session_rv_is_its_own_object_not_the_panels() -> None:
    """The proxy is the 10:00 REMAINING window of the ledger's own tape.

    Proposal 50 read the panel's 13 regular-hours bars (09:30 through 16:00)
    and summed the one-minute squared returns inside each; the ledger sums the
    30-minute squared spot returns of its own tape from 10:00 through the
    settlement print.  Close, not equal -- the README states the difference.
    What makes the ledger's the LIVE object is that the runner has it on every
    session and the research panel stops in 2024.
    """
    led = PremiumLedger.load(SEED_LEDGER)
    rv = led.session_rv_series().dropna()
    assert len(rv) == 1278  # 2020-05-01 carries no usable 10:00 cell
    assert float(rv.min()) > 0.0
    # a sane variance for a six-hour SPX window: sqrt(252 x RV) in (1 %, 400 %)
    ann = np.sqrt(rv.to_numpy(float) * 252.0)
    assert float(ann.min()) > 0.01
    assert float(ann.max()) < 4.0
