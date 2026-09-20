"""The per-clock premium ledger: implied and realized remaining-window variance.

One row per ``(session, clock)``.  At a clock the ledger records

* ``implied_rem_var``  -- the market's variance over the REMAINING session,
  the square of the total volatility bisected out of the nearest-OTM
  straddle's package midpoint at that stamp (``total_vol ** 2``);
* ``realized_rem_var`` -- the variance actually realized over that same
  remaining window: the sum of squared log spot returns over the 30-minute
  steps from the stamp through the 15:30 stamp plus the last step into the
  settlement print.  Same units, same window.

Those two objects, and nothing else, drive the two causal estimators the live
book needs:

``ratio_of_sums``
    The period's summed implied over its summed realized variance, per clock,
    over the sessions strictly BEFORE ``asof``.  This is the estimator
    proposal 43's premium curve reports and proposal 46's selector uses --
    the one that shows the premium walking from the morning to the afternoon
    over 2020-2025.  It is a ratio of sums, never a mean of per-day ratios:
    proposal 44 showed that a mean of logs is monotone in the clock (a
    one-bar remaining window makes ``log(implied/realized)`` explode) and
    picks the last clock on every session.

``realized_over_implied_mean``
    The lagged mean of realized over implied at that clock -- proposal 36's
    V9 bias correction, the one hedge-volatility variant whose era interval
    excludes zero.  ``live.ibkr.pricing.corrected_total_vol`` multiplies the
    implied VARIANCE by this factor before taking the delta.

``delever_multiplier``
    Proposal 50's part-B2 DESIGN CANDIDATE: the size multiplier of a trailing
    realized-variance regime filter.  ``realized_rem_var`` at the first entry
    clock is the session's realized variance proxy; its 21-session lagged mean
    (``rv21``) is compared with its own expanding lagged 90th and 97.5th
    quantiles and the size is halved above the first and zeroed above the
    second.  Proposal 50 evaluated it on its replay and adopted it nowhere.

Every estimator reads strictly earlier sessions: a record for ``asof`` itself
never enters its own statistic.  Both refuse (NaN) until ``min_sessions``
usable sessions have accumulated; nothing is filled, clipped or floored.

A cell is USABLE only when its implied and its realized remaining variance
are both finite and strictly positive, so implied and realized are summed
over exactly the same sessions -- proposal 46's shared mask.  That is what
makes the ratio the one the short actually earns on.

The ledger persists as a parquet, one row per ``(session, clock)``; the seed
built from the research tape lives at ``results/live_seed/premium_ledger.parquet``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DELEVER_HALF_PCT",
    "DELEVER_MIN_SESSIONS",
    "DELEVER_WINDOW",
    "DELEVER_ZERO_PCT",
    "LEDGER_COLUMNS",
    "MIN_SESSIONS",
    "SESSION_RV_CLOCK",
    "ClockRecord",
    "PremiumLedger",
]

#: The repo's warm-up, ``30_skip_day_rule.WARMUP_SESSIONS`` / ``43.MIN_DAYS``.
MIN_SESSIONS = 63

#: The clock whose REALIZED remaining-window variance is read as the session's
#: realized variance: the first entry clock, so the window is 10:00 through the
#: settlement print.  See :meth:`PremiumLedger.session_rv_series`.
SESSION_RV_CLOCK = "10:00"

#: Proposal 50's deleveraging candidate (``50_cat_replay_repriced.py``, part B2,
#: ``DELEVER_HALF_PCT`` / ``DELEVER_ZERO_PCT`` / ``DELEVER_MIN_OBS``), as
#: fractions rather than percents.  A DESIGN CANDIDATE: 50 evaluated it on its
#: replay and adopted it nowhere.
DELEVER_WINDOW = 21
DELEVER_HALF_PCT = 0.90
DELEVER_ZERO_PCT = 0.975
DELEVER_MIN_SESSIONS = 252

#: The ledger's columns, in the order they are written.
LEDGER_COLUMNS: tuple[str, ...] = (
    "session",
    "clock",
    "implied_rem_var",
    "realized_rem_var",
    "spot",
    "premium_mid",
    "kc",
    "kp",
)

_NAN = float("nan")
_FLOAT_COLUMNS: tuple[str, ...] = LEDGER_COLUMNS[2:]


@dataclass(frozen=True)
class ClockRecord:
    """One session's observation at one 30-minute clock.

    ``clock`` is an ``"HH:MM"`` Eastern stamp on the session's own 30-minute
    grid, ``"10:00"`` through ``"15:30"``.  ``implied_rem_var`` and
    ``realized_rem_var`` are variances over the SAME remaining window, so
    their ratio is dimensionless.
    """

    session: date
    clock: str
    implied_rem_var: float
    realized_rem_var: float
    spot: float
    premium_mid: float
    kc: float
    kp: float


def _as_day(value: object) -> np.datetime64:
    """A session key: the midnight stamp of ``value``, tz-naive."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        raise ValueError(f"a session key must be tz-naive, got {value!r}")
    return np.datetime64(ts.normalize().to_datetime64(), "ns")


def _check_clock(clock: str) -> str:
    """``"HH:MM"`` on the half hour, the grid the tape is stamped on."""
    text = str(clock)
    parts = text.split(":")
    if len(parts) != 2 or len(parts[0]) != 2 or len(parts[1]) != 2:
        raise ValueError(f"a clock must be 'HH:MM', got {clock!r}")
    hh, mm = (int(p) for p in parts)
    if not (0 <= hh < 24 and mm in (0, 30)):
        raise ValueError(f"a clock must be a half-hour ET stamp, got {clock!r}")
    return text


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame({name: pd.Series(dtype=float) for name in _FLOAT_COLUMNS})
    frame.insert(0, "clock", pd.Series(dtype=object))
    frame.insert(0, "session", pd.Series(dtype="datetime64[ns]"))
    return frame


class PremiumLedger:
    """The ``(session, clock)`` tape of implied and realized remaining variance.

    ``min_sessions`` is the warm-up both estimators refuse below.  It is an
    attribute rather than an argument of every call so that the ledger, not
    its callers, owns the convention; each estimator still takes a
    keyword-only override for a caller that runs its own grid of warm-ups.
    """

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        *,
        min_sessions: int = MIN_SESSIONS,
    ) -> None:
        self.min_sessions = int(min_sessions)
        self._frame = self._normalise(_empty_frame() if frame is None else frame)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._rv_cache: dict[tuple[int, str], np.ndarray] = {}
        self._sessions: np.ndarray = np.array([], dtype="datetime64[ns]")
        self._reindex()

    # ------------------------------------------------------------- state ---
    @staticmethod
    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in LEDGER_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"the ledger is missing columns {missing}")
        out = frame.loc[:, list(LEDGER_COLUMNS)].copy()
        out["session"] = pd.to_datetime(out["session"]).dt.normalize()
        if isinstance(out["session"].dtype, pd.DatetimeTZDtype):
            raise ValueError("the ledger's session column must be tz-naive")
        out["clock"] = [_check_clock(c) for c in out["clock"].astype(str)]
        for name in _FLOAT_COLUMNS:
            out[name] = out[name].astype(float)
        dup = out.duplicated(subset=["session", "clock"], keep=False)
        if bool(dup.any()):
            bad = out.loc[dup, ["session", "clock"]].head(5).to_dict("records")
            raise ValueError(f"the ledger has repeated (session, clock) rows: {bad}")
        return out.sort_values(["session", "clock"], kind="stable").reset_index(
            drop=True
        )

    def _reindex(self) -> None:
        """Rebuild the per-clock arrays the estimators read."""
        self._cache = {}
        self._rv_cache = {}
        self._sessions = (
            pd.DatetimeIndex(self._frame["session"].unique())
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )

    def _pair(self, clock: str) -> tuple[np.ndarray, np.ndarray]:
        """``(implied, realized)`` aligned to ``self._sessions``, NaN off the mask.

        A session without a row at this clock, or one whose implied or
        realized remaining variance is not finite and strictly positive,
        carries NaN in BOTH arrays -- one shared mask, so every sum below
        runs over exactly the same sessions.
        """
        key = _check_clock(clock)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sub = self._frame.loc[self._frame["clock"] == key]
        n = int(self._sessions.size)
        implied = np.full(n, _NAN)
        realized = np.full(n, _NAN)
        if len(sub):
            pos = np.searchsorted(
                self._sessions, sub["session"].to_numpy(dtype="datetime64[ns]")
            )
            implied[pos] = sub["implied_rem_var"].to_numpy(float)
            realized[pos] = sub["realized_rem_var"].to_numpy(float)
        good = (
            np.isfinite(implied)
            & (implied > 0.0)
            & np.isfinite(realized)
            & (realized > 0.0)
        )
        pair = (
            np.asarray(np.where(good, implied, _NAN), dtype=float),
            np.asarray(np.where(good, realized, _NAN), dtype=float),
        )
        self._cache[key] = pair
        return pair

    def _bounds(self, asof: date, window: int | None) -> tuple[int, int]:
        """The half-open row range of sessions strictly before ``asof``.

        ``window`` is a count of SESSION ROWS, as the research's rolling
        windows are: the rows ``[hi - window, hi)`` where ``hi`` is the number
        of ledger sessions strictly earlier than ``asof``.  ``None`` expands
        over every earlier session.  A row inside the range that is not usable
        contributes nothing but still consumes one of the ``window`` rows --
        the behaviour of ``Series.rolling(window, min_periods=...)``.
        """
        hi = int(np.searchsorted(self._sessions, _as_day(asof), side="left"))
        if window is None:
            return 0, hi
        w = int(window)
        if w <= 0:
            raise ValueError(f"a rolling window must be positive, got {window!r}")
        return max(0, hi - w), hi

    # ------------------------------------------------------------- build ---
    @classmethod
    def load(
        cls, path: Path | str, *, min_sessions: int = MIN_SESSIONS
    ) -> PremiumLedger:
        """Read a ledger parquet written by :meth:`save`."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no premium ledger at {p}")
        return cls(pd.read_parquet(p), min_sessions=min_sessions)

    def save(self, path: Path | str) -> None:
        """Write the ledger, one row per ``(session, clock)``, creating the parent."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._frame.to_parquet(p, index=False)

    def append_session(self, records: Iterable[ClockRecord]) -> None:
        """Add (or replace) every row of the sessions ``records`` touches.

        Idempotent per session: re-appending a session's records replaces that
        session's rows rather than duplicating them, so a re-run of a day is a
        no-op.  Sessions the records do not mention are left alone.
        """
        rows = list(records)
        if not rows:
            return
        new = pd.DataFrame(
            {
                "session": [_as_day(r.session) for r in rows],
                "clock": [_check_clock(r.clock) for r in rows],
                "implied_rem_var": [float(r.implied_rem_var) for r in rows],
                "realized_rem_var": [float(r.realized_rem_var) for r in rows],
                "spot": [float(r.spot) for r in rows],
                "premium_mid": [float(r.premium_mid) for r in rows],
                "kc": [float(r.kc) for r in rows],
                "kp": [float(r.kp) for r in rows],
            }
        )
        touched = set(new["session"].unique())
        keep = self._frame.loc[~self._frame["session"].isin(touched)]
        self._frame = self._normalise(pd.concat([keep, new], ignore_index=True))
        self._reindex()

    # ------------------------------------------------------------- reads ---
    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the ledger's rows, sorted by ``(session, clock)``."""
        return self._frame.copy()

    def sessions(self) -> list[date]:
        """Every session in the ledger, ascending."""
        return [pd.Timestamp(d).date() for d in self._sessions]

    def clocks(self) -> list[str]:
        """Every clock the ledger carries, ascending."""
        return sorted(set(self._frame["clock"].astype(str)))

    def ratio_of_sums(
        self,
        clock: str,
        asof: date,
        window: int | None,
        *,
        min_sessions: int | None = None,
    ) -> float:
        """Summed implied over summed realized remaining variance at ``clock``.

        Over the ``window`` ledger sessions strictly before ``asof``
        (``None`` = expanding).  NaN when fewer than ``min_sessions`` of them
        are usable, or when the realized sum is not strictly positive.
        """
        implied, realized = self._pair(clock)
        lo, hi = self._bounds(asof, window)
        a = implied[lo:hi]
        b = realized[lo:hi]
        keep = np.isfinite(a)
        n = int(keep.sum())
        floor = self.min_sessions if min_sessions is None else int(min_sessions)
        if n < floor:
            return _NAN
        s_realized = float(b[keep].sum())
        if not (isfinite(s_realized) and s_realized > 0.0):
            return _NAN
        return float(a[keep].sum()) / s_realized

    def realized_over_implied_mean(
        self,
        clock: str,
        asof: date,
        window: int | None,
        *,
        min_sessions: int | None = None,
    ) -> float:
        """Proposal 36's V9 factor: the lagged mean of realized over implied.

        The per-session ratio ``realized_rem_var / implied_rem_var`` at
        ``clock``, averaged over the ``window`` ledger sessions strictly
        before ``asof`` (``None`` = expanding).  NaN before ``min_sessions``
        usable sessions.  Multiply the implied VARIANCE by it -- that is what
        ``pricing.corrected_total_vol`` does.
        """
        implied, realized = self._pair(clock)
        lo, hi = self._bounds(asof, window)
        a = implied[lo:hi]
        b = realized[lo:hi]
        keep = np.isfinite(a)
        n = int(keep.sum())
        floor = self.min_sessions if min_sessions is None else int(min_sessions)
        if n < floor:
            return _NAN
        return float(np.mean(b[keep] / a[keep]))

    # --------------------------------------------- whole-tape convenience ---
    def ratio_of_sums_series(
        self,
        clock: str,
        window: int | None,
        *,
        min_sessions: int | None = None,
    ) -> "pd.Series[float]":
        """:meth:`ratio_of_sums` at every ledger session, as one lagged series.

        The same call, session by session -- the parity harness reads the
        whole tape and must get bit-for-bit what the live path would get on
        each of those sessions, so this is a convenience, not a second
        implementation.
        """
        idx = pd.DatetimeIndex(self._sessions)
        return pd.Series(
            [
                self.ratio_of_sums(clock, d, window, min_sessions=min_sessions)
                for d in idx
            ],
            index=idx,
            name=clock,
        )

    def realized_over_implied_series(
        self,
        clock: str,
        window: int | None,
        *,
        min_sessions: int | None = None,
    ) -> "pd.Series[float]":
        """:meth:`realized_over_implied_mean` at every ledger session."""
        idx = pd.DatetimeIndex(self._sessions)
        return pd.Series(
            [
                self.realized_over_implied_mean(
                    clock, d, window, min_sessions=min_sessions
                )
                for d in idx
            ],
            index=idx,
            name=clock,
        )

    # ------------------------------------------------- the regime filter ---
    def session_rv_series(self, *, clock: str = SESSION_RV_CLOCK) -> "pd.Series[float]":
        """The session's realized variance proxy, one value per ledger session.

        The REALIZED remaining-window variance at ``clock`` -- by default the
        first entry clock, ``"10:00"``, whose remaining window is 10:00 through
        the settlement print, so the number is the session's own realized
        variance over (almost all of) the regular session.  A session without a
        usable cell at that clock carries NaN; nothing is filled or floored.

        This is the ledger's SESSION RV PROXY.  It is not the object proposal
        50 measured its deleveraging rule on -- 50 read the 30-minute panel's
        ``sumret2`` summed over the 13 regular-hours bars (09:30 through 16:00,
        one-minute squared returns) -- and the two are close but not equal; the
        difference is stated in ``live/ibkr/README.md``.  The ledger's own
        column is what the live runner has: the panel ends in 2024 and the
        runner never reads it.
        """
        key = _check_clock(clock)
        idx = pd.DatetimeIndex(self._sessions)
        out = np.full(int(idx.size), _NAN)
        sub = self._frame.loc[self._frame["clock"] == key]
        if len(sub):
            pos = np.searchsorted(
                self._sessions, sub["session"].to_numpy(dtype="datetime64[ns]")
            )
            out[pos] = sub["realized_rem_var"].to_numpy(float)
        good = np.isfinite(out) & (out > 0.0)
        return pd.Series(np.where(good, out, _NAN), index=idx, name="session_rv")

    def session_realized_var(
        self, asof: date, *, clock: str = SESSION_RV_CLOCK
    ) -> float:
        """The session RV proxy of ``asof`` itself, NaN when it is not usable.

        A CONTEMPORANEOUS read -- it is the session's own variance, so it is a
        diagnostic, never an input to a decision made on ``asof``.  The causal
        objects are :meth:`rv21` and :meth:`delever_multiplier`, both of which
        read strictly earlier sessions.
        """
        v = self.session_rv_series(clock=clock)
        day = _as_day(asof)
        pos = int(np.searchsorted(self._sessions, day, side="left"))
        if pos >= self._sessions.size or self._sessions[pos] != day:
            return _NAN
        return float(v.to_numpy(float)[pos])

    def _rv21_at_sessions(self, window: int, clock: str) -> np.ndarray:
        """:meth:`rv21` at every ledger session, vectorised and cached."""
        w = int(window)
        if w <= 0:
            raise ValueError(f"the RV window must be positive, got {window!r}")
        key = (w, _check_clock(clock))
        hit = self._rv_cache.get(key)
        if hit is not None:
            return hit
        v = self.session_rv_series(clock=clock).to_numpy(float)
        good = np.isfinite(v)
        # The window is taken over the sessions that HAVE a usable variance --
        # proposal 49's holiday-safe ``rv_trail`` -- so one blank session does
        # not blank the next ``window`` of them.
        csum = np.concatenate(([0.0], np.cumsum(v[good])))
        before = np.cumsum(good) - good.astype(int)  # usable rows strictly before
        out = np.full(v.size, _NAN)
        ok = before >= w
        out[ok] = (csum[before[ok]] - csum[before[ok] - w]) / float(w)
        self._rv_cache[key] = out
        return out

    def rv21(
        self,
        asof: date,
        window: int = DELEVER_WINDOW,
        *,
        clock: str = SESSION_RV_CLOCK,
    ) -> float:
        """The trailing realized variance: the mean RV over ``window`` sessions.

        Over the ``window`` USABLE ledger sessions strictly before ``asof``, so
        it is known before ``asof`` opens.  NaN until that many have
        accumulated.  ``asof`` itself need not be a ledger session: this is the
        number the live runner computes for TODAY off yesterday's tape.
        """
        w = int(window)
        if w <= 0:
            raise ValueError(f"the RV window must be positive, got {window!r}")
        v = self.session_rv_series(clock=clock).to_numpy(float)
        hi = int(np.searchsorted(self._sessions, _as_day(asof), side="left"))
        prior = v[:hi]
        prior = prior[np.isfinite(prior)]
        if prior.size < w:
            return _NAN
        return float(prior[-w:].mean())

    def delever_multiplier(
        self,
        asof: date,
        window: int = DELEVER_WINDOW,
        p_half: float = DELEVER_HALF_PCT,
        p_zero: float = DELEVER_ZERO_PCT,
        min_sessions: int = DELEVER_MIN_SESSIONS,
        *,
        clock: str = SESSION_RV_CLOCK,
    ) -> tuple[float, dict[str, Any]]:
        """Proposal 50's deleveraging candidate, as a size multiplier.

        ``RV21`` is :meth:`rv21` at ``asof``.  Its thresholds are the EXPANDING
        LAGGED ``p_half`` and ``p_zero`` quantiles of RV21 over the ledger
        sessions strictly before ``asof``, taken only once ``min_sessions`` of
        them carry a finite RV21:

            1.0   RV21 <= p_half           full size
            0.5   p_half < RV21 <= p_zero  half size
            0.0   RV21 > p_zero            no position

        Before the threshold has enough history -- or while RV21 itself is
        still warming up -- the rule is INACTIVE and the multiplier is 1.0,
        reported as ``state="warmup"``.  Nothing is clipped or floored.

        Returns ``(multiplier, info)`` with ``info`` carrying ``rv21``,
        ``p_half``, ``p_zero``, ``n_prior`` and ``state``.

        This is a DESIGN CANDIDATE from proposal 50, evaluated on its replay
        and adopted nowhere in the research.
        """
        if not 0.0 < float(p_half) < float(p_zero) < 1.0:
            raise ValueError(
                "the deleveraging quantiles must satisfy 0 < p_half < p_zero "
                f"< 1, got {p_half!r} and {p_zero!r}"
            )
        floor = int(min_sessions)
        if floor < 1:
            raise ValueError(f"min_sessions must be >= 1, got {min_sessions!r}")
        rv = self.rv21(asof, window, clock=clock)
        hi = int(np.searchsorted(self._sessions, _as_day(asof), side="left"))
        hist = self._rv21_at_sessions(window, clock)[:hi]
        hist = hist[np.isfinite(hist)]
        n_prior = int(hist.size)
        info: dict[str, Any] = {
            "rv21": rv,
            "p_half": _NAN,
            "p_zero": _NAN,
            "n_prior": n_prior,
            "state": "warmup",
        }
        if n_prior < floor or not isfinite(rv):
            return 1.0, info
        q_half = float(np.quantile(hist, float(p_half)))
        q_zero = float(np.quantile(hist, float(p_zero)))
        info["p_half"] = q_half
        info["p_zero"] = q_zero
        if rv <= q_half:
            info["state"] = "full"
            return 1.0, info
        if rv <= q_zero:
            info["state"] = "half"
            return 0.5, info
        info["state"] = "zero"
        return 0.0, info


def records_from_grids(
    sessions: Sequence[object],
    clocks: Sequence[str],
    implied_rem_var: np.ndarray,
    realized_rem_var: np.ndarray,
    spot: np.ndarray,
    premium_mid: np.ndarray,
    kc: np.ndarray,
    kp: np.ndarray,
) -> list[ClockRecord]:
    """Every ``(session, clock)`` cell of a research tape, as ledger records.

    The grids are ``(session x clock)`` arrays in the order of ``sessions``
    and ``clocks``.  Nothing is dropped: an unusable cell is recorded as the
    NaN it is, and the estimators mask it.
    """
    out: list[ClockRecord] = []
    for i, day in enumerate(sessions):
        key = pd.Timestamp(day).date()
        for j, clock in enumerate(clocks):
            out.append(
                ClockRecord(
                    session=key,
                    clock=str(clock),
                    implied_rem_var=float(implied_rem_var[i, j]),
                    realized_rem_var=float(realized_rem_var[i, j]),
                    spot=float(spot[i, j]),
                    premium_mid=float(premium_mid[i, j]),
                    kc=float(kc[i, j]),
                    kp=float(kp[i, j]),
                )
            )
    return out
