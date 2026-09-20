"""33 - a path-conditional buy-back for the 11:00 delta-hedged short straddle.

The book of record is proposal 32's: every session sells the nearest-OTM SPX
0DTE straddle at 11:00 ET, delta-hedges the package on the vendor spot every
30 minutes, and buys the straddle back at 15:30 at the QUOTED ask with the
entry taken at the quoted bid (fully crossed); the "mid" variant buys back at
the quoted midpoint against an entry at the midpoint.  Returns are one unit a
day in units of the 11:00 midpoint entry premium.

Proposal 32 found that the losing days are single large one-directional
afternoon bars: the worst half-hour is 113% of a losing day's loss, and the
trending tercile of days (|sum of bar returns| / sum of |bar returns|) earns
an era crossed Sharpe of -1.95 against the choppy tercile's +6.96.  Proposals
30 and 31 found that nothing observable at 11:00 separates those days from the
rest.  This proposal therefore spends information that arrives DURING the
session: a rule that buys the straddle back early, at a stamp, on what the
path has done up to that stamp.

Every rule is causal.  It is decided at a stamp t in 12:00..15:00 using that
stamp's own quotes and spot, the buy-back is at the QUOTED ask of the 11:00
strikes (the midpoint in the mid variant), the delta hedge stops at that same
stamp, and the day's remaining premium is forgone -- the seller is flat from
the stop to the close.

  Family S    (move stop)     buy back at the first stamp where the index has
                              travelled |S_t - S_11:00| / S_11:00 >= k m_11,
                              where m_11 is the "expected move" priced at
                              11:00: the package's total volatility over the
                              remaining five hours, already a return-unit
                              sigma.  k in K_GRID.
  Family S-t  (timed move)    the same with the threshold grown on the clock,
                              k m_11 sqrt(hours elapsed / 5), so an early
                              move must be larger to stop the day.
  Family L    (loss stop)     buy back at the first stamp where the day's
                              mark-to-market loss -- the straddle at the
                              QUOTED midpoint of the 11:00 strikes plus the
                              hedge so far, in units of the entry premium --
                              exceeds L.  L in L_GRID.
  Family C    (signal cross)  buy back at the first stamp where the
                              re-measured window-matched signal of the
                              two-block ridge turns positive (the forecast
                              variance rises above the implied slice) -- the
                              deck's surviving exit rule for the other book,
                              re-measured at every stamp exactly as proposal
                              30 measures it at 11:00.

Two reference rows: never stop (the book itself) and always stop at 15:00
(the exit-clock ladder's 15:00 rung).

The gate is the repo's standard and nothing new is computed until it passes:
proposal 32's published book numbers and its whole exit ladder are reproduced
first; selection is proposal 30's purged, embargoed walk-forward; significance
is proposal 30's run-length-preserving block placebo, in proposal 31's
vectorised form, with the shuffle applied to WHICH DAYS GET STOPPED -- the
count of stopped days, their run lengths and the multiset of stop clocks are
all preserved, so the null is "a rule that stops as often, in runs as long, at
the same clocks, but on days chosen for no reason".  A rule is adopted only if
its out-of-sample calendar-day Sharpe beats the book AND its placebo p-value
is below PLACEBO_ALPHA.

Run:  python writeup/intraday_proposals/33_path_stop.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.ibkr.parity import research_modules  # noqa: E402


def _load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
p30 = _load_module(HERE / "30_skip_day_rule.py", "p33_p30")
p31 = _load_module(HERE / "31_margin_skip_rule.py", "p33_p31")
p32 = _load_module(HERE / "32_loss_anatomy.py", "p33_p32")

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "33"
CHAIN = p32.CHAIN

ENTRY = p32.ENTRY
CLOSE = p32.CLOSE
SESSION: tuple[str, ...] = p32.SESSION
H_REM: tuple[float, ...] = p32.H_REM
ERA0 = p32.ERA0
ANN = p32.ANN

#: The stamps a stop may be taken at: every half hour from 12:00 to 15:00.
#: 11:00 is the entry (no path yet) and 15:30 is the book's own buy-back.
STOP_STAMPS: tuple[str, ...] = (
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
)
#: Family S and S-t: multiples of the 11:00 implied expected move.
K_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
#: Family L: the mark-to-market loss, in units of the entry premium.
L_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5)
#: Family C: the forecast whose window-matched signal is re-measured.
SIGNAL_TAG = "blk2"
#: The reference row taken from the exit-clock ladder.
ALWAYS_CLOCK = "15:00"

#: Proposal 30's constants, reused verbatim so this proposal is scored on the
#: same day sets against the same null: a 63-session warm-up, a 5-day embargo,
#: 2000 block placebos at seed 0, adoption at p < 0.05.
WARMUP_SESSIONS = p30.WARMUP_SESSIONS
EMBARGO_DAYS = p30.EMBARGO_DAYS
N_PLACEBO = p30.N_PLACEBO
PLACEBO_SEED = p30.PLACEBO_SEED
PLACEBO_ALPHA = p30.PLACEBO_ALPHA

#: Published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = p32.GATE_N_DAYS
GATE_CROSSED_WHOLE = p32.GATE_CROSSED_WHOLE
GATE_CROSSED_ERA = p32.GATE_CROSSED_ERA
GATE_TOL = p32.GATE_TOL
#: Proposal 32's exit-clock ladder, daily era, crossed-quoted, as published.
GATE_LADDER_ERA_CROSSED: dict[str, float] = {
    "13:00": 0.902,
    "13:30": 0.688,
    "14:00": 1.004,
    "14:30": 1.325,
    "15:00": 1.787,
    "15:30": 2.513,
}
#: The ladder is published to three decimals, so it is gated at that bar.
LADDER_TOL = 5e-4
#: The tape identities (hedge cumulation, the book's own return) are float paths.
EXACT_TOL = 1e-12

BOOKS: tuple[str, ...] = ("crossed", "mid")
FAMILIES: tuple[str, ...] = (
    "move_stop",
    "move_stop_timed",
    "loss_stop",
    "signal_cross",
)


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(use.to_string(index=False))


# ------------------------------------------------------------- the tape ----
def quote_grids(
    ent: pd.DataFrame, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Package bid / mid / ask of the 11:00 strikes at every 30-minute stamp.

    Proposal 32's ``quote_grids`` with one addition: the per-stamp DEAD mask
    is returned as well, because a stop can only be taken at a stamp whose two
    legs carry a quote.  The chain scan itself is proposal 32's
    ``quotes_at_stamps``, imported and unedited, and the gate below asserts
    that these grids reproduce the published book numbers exactly.
    """
    rows: list[pd.DataFrame] = []
    for cp, col in (("C", "K_c"), ("P", "K_p")):
        for clock in SESSION:
            rows.append(
                pd.DataFrame(
                    {
                        "expiration": pd.DatetimeIndex(dates),
                        "hhmm": clock,
                        "strike": ent[col].to_numpy(float),
                        "cp": cp,
                    }
                )
            )
    keys = pd.concat(rows, ignore_index=True)
    got = p32.quotes_at_stamps(CHAIN, SESSION, keys).set_index(
        ["expiration", "hhmm", "strike", "cp"]
    )
    n, m = len(dates), len(SESSION)
    bid = np.zeros((n, m))
    ask = np.zeros((n, m))
    dead = np.zeros((n, m), dtype=bool)
    for cp, col in (("C", "K_c"), ("P", "K_p")):
        for j, clock in enumerate(SESSION):
            key = pd.MultiIndex.from_arrays(
                [
                    pd.DatetimeIndex(dates),
                    np.repeat(clock, n),
                    ent[col].to_numpy(float),
                    np.repeat(cp, n),
                ]
            )
            leg = got.reindex(key)
            lb = leg["bid"].to_numpy(float)
            la = leg["ask"].to_numpy(float)
            # the vendor's no-quote sentinel is bid == ask == 0 (asl.quote_mid)
            dead[:, j] |= ~np.isfinite(lb) | ~np.isfinite(la) | ((lb == 0) & (la == 0))
            bid[:, j] += lb
            ask[:, j] += la
    print(
        f"chain quotes on the two 11:00 strikes: {int((~dead).all(axis=1).sum())} of "
        f"{n} days live at all {m} stamps; {int(dead.sum())} dead legs in total"
    )
    return bid, 0.5 * (bid + ask), ask, dead


def hedge_cumulated(
    attribution: pd.DataFrame, tape: dict[str, Any], dates: pd.DatetimeIndex
) -> np.ndarray:
    """Hedge P&L accumulated from 11:00 to each stamp, in index points.

    ``cum[i, j]`` is ``sum_{k < j} delta_k (S_{k+1} - S_k)`` -- exactly what
    ``live.ibkr.parity.replay_day`` accumulates when it is told to exit at
    stamp ``j``, since that replay holds no step past its exit stamp.  The
    asserts below check it against the engine's own replays at the six ladder
    clocks and at the book's 15:30 buy-back.
    """
    att = attribution.copy()
    att["date"] = pd.to_datetime(att["date"])
    grid = att.pivot_table(
        index="date", columns="clock", values="hedge_pts", aggfunc="first"
    ).reindex(index=dates, columns=list(SESSION[:-1]))
    assert np.isfinite(grid.to_numpy(float)).all(), "a bar is missing its hedge leg"
    cum = np.zeros((len(dates), len(SESSION)))
    cum[:, 1:] = np.cumsum(grid.to_numpy(float), axis=1)
    assert np.max(np.abs(cum[:, -1] - tape["hedge_ref"])) < EXACT_TOL, "hedge to 15:30"
    for clock, hedge in tape["hedge_clock"].items():
        j = SESSION.index(clock)
        assert np.max(np.abs(cum[:, j] - hedge)) < EXACT_TOL, f"hedge to {clock}"
    print(
        f"hedge cumulation checked against the live engine's own replays at "
        f"{len(tape['hedge_clock'])} exit clocks and at {CLOSE}, to {EXACT_TOL:.0e}"
    )
    return cum


def stop_return_grids(
    tape: dict[str, Any], mid: np.ndarray, ask: np.ndarray, cum: np.ndarray
) -> dict[str, np.ndarray]:
    """The day's return if the straddle is bought back at stamp ``j``.

    The buy-back is the QUOTED ask of the 11:00 strikes against an entry at
    the quoted bid (the crossed book) or the quoted midpoint against an entry
    at the midpoint (the mid variant); the hedge stops at the same stamp and
    the day's remaining premium is forgone.  Column ``-1`` (15:30) is the book
    of record itself.
    """
    em = tape["entry_mid"][:, None]
    eb = tape["entry_bid"][:, None]
    return {
        "crossed": (-(ask - eb) + cum) / em,
        "mid": (-(mid - em) + cum) / em,
    }


# ------------------------------------------------------------- the gate ----
def gate(
    tape: dict[str, Any],
    mid: np.ndarray,
    ask: np.ndarray,
    dates: pd.DatetimeIndex,
    grids: dict[str, np.ndarray],
    day: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce proposal 32's books and its whole exit ladder, then stop."""
    p32.gate(tape, mid, ask, dates)
    ladder = p32.part_b(tape, mid, ask, dates)
    era_x = ladder[(ladder["variant"] == "crossed") & (ladder["sample"] == "daily_era")]
    print("\nGATE - proposal 32's exit-clock ladder, daily era, crossed-quoted")
    for clock, want in GATE_LADDER_ERA_CROSSED.items():
        have = float(era_x.loc[era_x["exit_clock"] == clock, "Sharpe_ann"].iloc[0])
        assert abs(have - want) < LADDER_TOL, (clock, have, want)
        print(f"  exit {clock:<12s} {have:>14.9f}  reference {want:.3f}  OK")

    # the stop grid must reproduce the book at 15:30 and the ladder at 15:00
    r_x = pd.Series(grids["crossed"][:, -1], index=dates)
    for label, have, want in (
        ("crossed-quoted Sharpe_ann whole", p32.sharpe_ann(r_x), GATE_CROSSED_WHOLE),
        (
            "crossed-quoted Sharpe_ann era",
            p32.sharpe_ann(r_x[r_x.index >= ERA0]),
            GATE_CROSSED_ERA,
        ),
    ):
        assert abs(have - want) < GATE_TOL, (label, have, want)
        print(f"  stop grid, {label:<32s} {have:>16.12f}  reference {want:.12f}  OK")
    for book in BOOKS:
        r_book = day[f"r_{book}"].to_numpy(float)
        assert np.max(np.abs(grids[book][:, -1] - r_book)) < EXACT_TOL, book
        j = SESSION.index(ALWAYS_CLOCK)
        r_stop = pd.Series(grids[book][:, j], index=dates)
        want = float(
            ladder.loc[
                (ladder["variant"] == book)
                & (ladder["sample"] == "daily_era")
                & (ladder["exit_clock"] == ALWAYS_CLOCK),
                "Sharpe_ann",
            ].iloc[0]
        )
        have = p32.sharpe_ann(r_stop[r_stop.index >= ERA0])
        assert abs(have - want) < EXACT_TOL, (book, have, want)
        print(
            f"  stop grid, {book:<8s} 15:30 column is the book to {EXACT_TOL:.0e}; "
            f"{ALWAYS_CLOCK} column is the ladder rung {have:.9f}  OK"
        )
    print("GATE PASSED")
    return ladder


def gate_placebo(r: np.ndarray) -> None:
    """Proposal 31's vectorised block shuffle draws proposal 30's own masks."""
    probe = np.zeros(r.size, dtype=bool)
    probe[::3] = True
    probe[5:9] = True
    a = p31.block_shuffled_masks(probe, N_PLACEBO, np.random.default_rng(PLACEBO_SEED))
    b = p30.block_shuffled_masks(probe, N_PLACEBO, np.random.default_rng(PLACEBO_SEED))
    assert (a == b).all(), "the vectorised block shuffle does not reproduce p30's draws"
    print(
        f"GATE - placebo: {N_PLACEBO} vectorised block-shuffled masks identical to "
        f"proposal 30's draw for draw, seed {PLACEBO_SEED}  OK"
    )


# ----------------------------------------------------------- the signal ----
def stamp_signals(
    pkg: pd.DataFrame, dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The window-matched signal at EVERY stamp, built as proposal 30 builds it.

    Proposal 30's ``signals_1100`` computes ``s_matched = rv_hat - IV_hr^2
    h_rem w_slice`` on every stamp of the trade cache and then keeps the 11:00
    row; this keeps all of them.  The remaining-share profile ``w_slice`` is
    proposal 30's ``panel_profile``, expanding with a 63-session warm-up and
    shifted one session, so the value used at a stamp is a function of
    strictly earlier sessions.  The gate asserts that the 11:00 column of this
    grid is proposal 30's own 11:00 signal, day for day -- and proposal 30's
    construction in turn asserts itself against the notebook's persisted 11:00
    buy share.
    """
    asl = p30.asl
    work0 = pkg.copy()
    work0["t"] = pd.to_datetime(work0["timestamp"], utc=True)
    work0["date"] = pd.to_datetime(work0["date"])
    clocks = sorted(work0["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    paths = asl.yhat_paths(p30.REPO)
    base = asl.load_yhat_panel(paths[SIGNAL_TAG]).set_index("t")
    _, w_slice = p30.panel_profile(base, clocks)
    fc = base.reset_index()[["t", "rv_hat", "in_fit"]].copy()
    fc["t"] = pd.to_datetime(fc["t"], utc=True) - pd.Timedelta(minutes=30)
    work = work0.merge(fc, on="t", how="left").dropna(subset=["R", "rv_hat"])
    assert bool(work["in_fit"].all()), "a joined trade bar is outside the fit mask"
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work["w_slice"] = w_slice.stack().reindex(mi).to_numpy()
    work["h_rem"] = work["hhmm"].map(n_rem).astype(float) * 0.5
    work["slice"] = (
        work["iv_hourly"].astype(float) ** 2 * work["h_rem"] * work["w_slice"]
    )
    work["s_matched"] = work["rv_hat"] - work["slice"]
    grid = work.pivot_table(
        index="date", columns="hhmm", values="s_matched", aggfunc="first"
    )
    sig = grid.reindex(index=dates, columns=list(SESSION))

    ref = p30.signals_1100(pkg, dates)[f"s_{SIGNAL_TAG}"]
    a = sig[ENTRY].to_numpy(float)
    b = ref.to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    assert (np.isfinite(a) == np.isfinite(b)).all(), "the 11:00 signal masks disagree"
    assert np.array_equal(a[ok], b[ok]), "the 11:00 signal does not reproduce p30's"
    print(
        f"GATE - the all-stamp signal grid reproduces proposal 30's 11:00 "
        f"{SIGNAL_TAG} signal exactly on all {int(ok.sum())} days that carry one  OK"
    )
    return sig, grid


def censored_grid(panel: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """True where a stamp's vendor implied volatility sits on the solver node.

    The window-matched signal is built from the vendor's hourly implied
    volatility, so a stamp whose call or put leg is censored carries no usable
    implied slice and the signal rule takes no exit there.  The flag is the
    research's own ``on_vendor_node``, imported from the rule-table module.
    """
    _, rule_mod = research_modules(ROOT)
    flag = rule_mod.on_vendor_node(
        panel["impl_volatility_c"]
    ) | rule_mod.on_vendor_node(panel["impl_volatility_p"])
    tab = panel.assign(_cen=np.asarray(flag, dtype=float)).pivot_table(
        index="date", columns="hhmm", values="_cen", aggfunc="max"
    )
    out = (
        tab.reindex(index=dates, columns=list(SESSION)).fillna(1.0).to_numpy(float) > 0
    )
    print(
        f"censored implied volatility: {int(out.sum())} of {out.size} stamp-cells on "
        f"the 11:00 book's days (a censored stamp takes no signal exit)"
    )
    return out


# -------------------------------------------------------------- the rules ---
def first_stop(
    trigger: np.ndarray, usable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The first stop stamp at which the trigger fires and the stamp is quoted.

    ``trigger`` and ``usable`` are day x stamp over SESSION.  A stamp outside
    STOP_STAMPS, or one whose two legs carry no quote, cannot be traded, so
    the rule simply moves on to the next stamp; a day on which nothing fires
    runs to the book's 15:30 buy-back.  Returns the stopped mask and the
    column index of the stop (-1 where the day is not stopped).
    """
    fire = trigger & usable
    allowed = np.zeros(len(SESSION), dtype=bool)
    for c in STOP_STAMPS:
        allowed[SESSION.index(c)] = True
    fire = fire & allowed[None, :]
    stopped = fire.any(axis=1)
    stop_j = np.where(stopped, np.argmax(fire, axis=1), -1)
    return stopped, stop_j.astype(int)


def build_rules(
    tape: dict[str, Any],
    day: pd.DataFrame,
    mid: np.ndarray,
    cum: np.ndarray,
    sig: pd.DataFrame,
    censored: np.ndarray,
    usable: np.ndarray,
) -> list[dict[str, Any]]:
    """Every candidate rule: its family, its parameter and its stop tape."""
    spot = tape["spot"]
    em = tape["entry_mid"][:, None]
    n = spot.shape[0]
    move = np.abs(spot - spot[:, [0]]) / spot[:, [0]]
    m11 = day["s_1100"].to_numpy(float)[:, None]
    elapsed = np.array([0.5 * j for j in range(len(SESSION))])[None, :]
    clock_scale = np.sqrt(elapsed / H_REM[0])
    mtm = (-(mid - em) + cum) / em
    s_grid = sig.to_numpy(float)

    rules: list[dict[str, Any]] = [
        {
            "family": "reference",
            "param": float("nan"),
            "name": "never stop (the book of record, buy back at 15:30)",
            "stopped": np.zeros(n, dtype=bool),
            "stop_j": np.full(n, -1, dtype=int),
        },
        {
            "family": "reference",
            "param": float("nan"),
            "name": f"always stop at {ALWAYS_CLOCK} (the exit-clock ladder rung)",
            "stopped": np.ones(n, dtype=bool),
            "stop_j": np.full(n, SESSION.index(ALWAYS_CLOCK), dtype=int),
        },
    ]
    for k in K_GRID:
        stopped, stop_j = first_stop(move >= k * m11, usable)
        rules.append(
            {
                "family": "move_stop",
                "param": float(k),
                "name": f"buy back when |S_t - S_11:00| / S_11:00 >= {k:.2f} m_11",
                "stopped": stopped,
                "stop_j": stop_j,
            }
        )
    for k in K_GRID:
        stopped, stop_j = first_stop(move >= k * m11 * clock_scale, usable)
        rules.append(
            {
                "family": "move_stop_timed",
                "param": float(k),
                "name": (
                    f"buy back when |S_t - S_11:00| / S_11:00 >= {k:.2f} m_11 "
                    f"sqrt(hours elapsed / 5)"
                ),
                "stopped": stopped,
                "stop_j": stop_j,
            }
        )
    for level in L_GRID:
        stopped, stop_j = first_stop(mtm <= -level, usable)
        rules.append(
            {
                "family": "loss_stop",
                "param": float(level),
                "name": (
                    f"buy back when the mark-to-market day loss exceeds "
                    f"{level:.2f} entry premia"
                ),
                "stopped": stopped,
                "stop_j": stop_j,
            }
        )
    trig = np.isfinite(s_grid) & (s_grid > 0) & ~censored
    stopped, stop_j = first_stop(trig, usable)
    rules.append(
        {
            "family": "signal_cross",
            "param": 0.0,
            "name": (
                f"buy back when the window-matched signal of {SIGNAL_TAG} turns "
                f"positive at a stamp"
            ),
            "stopped": stopped,
            "stop_j": stop_j,
        }
    )
    return rules


def realised(rule: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    """The day's return under a rule: the stop column where it stops, else 15:30."""
    j = np.where(rule["stopped"], rule["stop_j"], len(SESSION) - 1)
    return grid[np.arange(grid.shape[0]), j]


def rule_stats(
    r: np.ndarray,
    r_book: np.ndarray,
    stopped: np.ndarray,
    stop_j: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Calendar-day statistics of one book under one stop rule.

    The rule never sits a day out -- it only shortens the day -- so every rule
    is scored on the same calendar days and the mean is directly comparable
    with the book's own.
    """
    ok = np.isfinite(r) & np.isfinite(r_book)
    rr = r[ok]
    bb = r_book[ok]
    ss = stopped[ok]
    jj = stop_j[ok]
    dd = dates[ok]
    sd = float(rr.std(ddof=1)) if rr.size >= 2 else float("nan")
    mean = float(rr.mean()) if rr.size else float("nan")
    live = bool(np.isfinite(sd)) and sd > 0.0
    worst_i = int(np.argmin(rr)) if rr.size else -1
    hours = np.where(ss, 11.0 + 0.5 * jj, np.nan)
    return {
        "n_days": int(rr.size),
        "n_stopped": int(ss.sum()),
        "frac_stopped": float(ss.mean()) if rr.size else float("nan"),
        "mean_stop_clock_et": float(np.nanmean(hours)) if ss.any() else float("nan"),
        "mean_hours_held": float(np.nanmean(hours - 11.0))
        if ss.any()
        else float("nan"),
        "mean_calendar": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if live else float("nan"),
        "t": float(np.sqrt(rr.size)) * mean / sd if live else float("nan"),
        "MaxDD": p30.maxdd(rr),
        "worst_day": float(rr.min()) if rr.size else float("nan"),
        "worst_date": str(pd.Timestamp(dd[worst_i]).date()) if worst_i >= 0 else "",
        "mean_stopped_under_rule": float(rr[ss].mean()) if ss.any() else float("nan"),
        "mean_stopped_under_book": float(bb[ss].mean()) if ss.any() else float("nan"),
        "saved_per_stopped_day": float((rr[ss] - bb[ss]).mean())
        if ss.any()
        else float("nan"),
        "mean_not_stopped": float(rr[~ss].mean()) if (~ss).any() else float("nan"),
    }


def decomposition_rows(
    rule: dict[str, Any],
    r: np.ndarray,
    r_book: np.ndarray,
    stopped: np.ndarray,
    book: str,
    sample: str,
) -> dict[str, Any]:
    """On the stopped days: the stop's return against the 15:30 counterfactual."""
    ok = np.isfinite(r) & np.isfinite(r_book) & stopped
    rr = r[ok]
    bb = r_book[ok]
    helped = rr > bb
    hurt = rr < bb
    return {
        "book": book,
        "sample": sample,
        "family": rule["family"],
        "rule": rule["name"],
        "n_stopped": int(rr.size),
        "mean_at_stop": float(rr.mean()) if rr.size else float("nan"),
        "mean_counterfactual_1530": float(bb.mean()) if rr.size else float("nan"),
        "mean_difference": float((rr - bb).mean()) if rr.size else float("nan"),
        "sum_difference": float((rr - bb).sum()) if rr.size else float("nan"),
        "n_helped": int(helped.sum()),
        "mean_helped_at_stop": float(rr[helped].mean())
        if helped.any()
        else float("nan"),
        "mean_helped_counterfactual": float(bb[helped].mean())
        if helped.any()
        else float("nan"),
        "sum_helped": float((rr - bb)[helped].sum()) if helped.any() else float("nan"),
        "n_hurt": int(hurt.sum()),
        "mean_hurt_at_stop": float(rr[hurt].mean()) if hurt.any() else float("nan"),
        "mean_hurt_counterfactual": float(bb[hurt].mean())
        if hurt.any()
        else float("nan"),
        "sum_hurt": float((rr - bb)[hurt].sum()) if hurt.any() else float("nan"),
        "n_tied": int(rr.size - helped.sum() - hurt.sum()),
    }


# ------------------------------------------------------- the walk-forward ---
def walk_forward(
    series: np.ndarray, stops: np.ndarray, clocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expanding, embargoed rule choice; only the out-of-sample day is scored.

    Proposal 30's ``walk_forward`` with one substitution: a candidate's
    in-window series is its own realised return (a stopped day earns the stop's
    P&L) instead of proposal 30's sat-out zero.  Everything else is unchanged
    -- for day i the training window is days [0, i - EMBARGO_DAYS), the
    candidate with the best in-window calendar-day Sharpe is applied to day i,
    candidate 0 is "never stop" and ``np.argmax`` keeps the first maximum so a
    tie goes to the book, and days whose training window is shorter than
    WARMUP_SESSIONS are not scored.  ``series`` is candidates x days.
    """
    n = series.shape[1]
    curves = np.vstack([p30.expanding_sharpe(row) for row in series])
    curves = np.where(np.isfinite(curves), curves, -np.inf)
    scored = np.zeros(n, dtype=bool)
    pick = np.full(n, -1, dtype=int)
    out = np.full(n, np.nan)
    stopped = np.zeros(n, dtype=bool)
    stop_j = np.full(n, -1, dtype=int)
    for i in range(n):
        k = i - EMBARGO_DAYS
        if k < WARMUP_SESSIONS:
            continue
        c = int(np.argmax(curves[:, k]))
        pick[i] = c
        out[i] = series[c, i]
        stopped[i] = bool(stops[c, i])
        stop_j[i] = int(clocks[c, i])
        scored[i] = True
    return scored, out, stopped, stop_j, pick


# ------------------------------------------------------------- the placebo --
def placebo_row(
    r_book: np.ndarray,
    grid: np.ndarray,
    stopped: np.ndarray,
    stop_j: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Where the rule's Sharpe falls among run-length-preserving stop shuffles.

    Proposal 30's block placebo (in proposal 31's vectorised form, asserted
    draw for draw against proposal 30's own) shuffles WHICH DAYS are stopped:
    each draw stops exactly as many days, in runs of exactly the same lengths.
    The rule's own ordered list of stop clocks is then laid on the drawn days
    in date order, so the multiset of stop clocks is preserved too and the
    only thing the draw changes is WHICH days get stopped.  A drawn day whose
    stop stamp carries no quote keeps the book's return.
    """
    ok = np.isfinite(r_book)
    rb = r_book[ok]
    gg = grid[ok]
    mm = np.asarray(stopped, dtype=bool)[ok]
    jj = np.asarray(stop_j, dtype=int)[ok]
    r_rule = np.where(mm, gg[np.arange(gg.shape[0]), np.where(mm, jj, -1)], rb)
    s_rule = p30.sharpe(r_rule)
    if mm.sum() == 0 or mm.all():
        return {
            "n_placebo": 0,
            "sharpe_rule": s_rule,
            "placebo_mean": float("nan"),
            "placebo_sd": float("nan"),
            "placebo_q05": float("nan"),
            "placebo_q50": float("nan"),
            "placebo_q95": float("nan"),
            "pct_rank": float("nan"),
            "p_value": float("nan"),
        }
    masks = p31.block_shuffled_masks(mm, N_PLACEBO, rng)
    order = jj[mm]
    cols = np.nonzero(masks)[1].reshape(N_PLACEBO, order.size)
    cal = np.repeat(rb[None, :], N_PLACEBO, axis=0)
    drawn = gg[cols, order[None, :]]
    cal[np.arange(N_PLACEBO)[:, None], cols] = np.where(
        np.isfinite(drawn), drawn, rb[cols]
    )
    mean = cal.mean(axis=1)
    sd = cal.std(axis=1, ddof=1)
    s_plac = np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0) * ANN, np.nan)
    good = s_plac[np.isfinite(s_plac)]
    ge = int((good >= s_rule).sum())
    return {
        "n_placebo": int(good.size),
        "sharpe_rule": s_rule,
        "placebo_mean": float(good.mean()),
        "placebo_sd": float(good.std(ddof=1)),
        "placebo_q05": float(np.quantile(good, 0.05)),
        "placebo_q50": float(np.quantile(good, 0.50)),
        "placebo_q95": float(np.quantile(good, 0.95)),
        "pct_rank": float((good < s_rule).mean()),
        "p_value": float((1 + ge) / (1 + good.size)),
    }


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, ent, dates = p32.load_panel()
    assert len(dates) == GATE_N_DAYS, (len(dates), GATE_N_DAYS)
    tape = p32.build_tape(panel, ent, dates)
    _, mid, ask, dead = quote_grids(ent, dates)
    assert np.max(np.abs(mid[:, 0] - tape["entry_mid"])) < EXACT_TOL, "11:00 mid"
    attribution, day = p32.attribute(tape, mid, ask, dates)
    cum = hedge_cumulated(attribution, tape, dates)
    grids = stop_return_grids(tape, mid, ask, cum)
    gate(tape, mid, ask, dates, grids, day)
    gate_placebo(day["r_crossed"].to_numpy(float))

    sig, _ = stamp_signals(
        pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1]), dates
    )
    censored = censored_grid(panel, dates)
    usable = (~dead) & np.isfinite(ask) & (ask > 0) & np.isfinite(mid) & (mid > 0)
    rules = build_rules(tape, day, mid, cum, sig, censored, usable)
    print(
        f"\n{len(rules)} rules: {len(K_GRID)} move stops, {len(K_GRID)} timed move "
        f"stops, {len(L_GRID)} loss stops, 1 signal cross, 2 reference rows; stop "
        f"stamps {STOP_STAMPS[0]}..{STOP_STAMPS[-1]}; m_11 median "
        f"{float(day['s_1100'].median()):.5f} in return units"
    )

    era = np.asarray(dates >= ERA0, dtype=bool)
    samples: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )
    ret = {bk: {r["name"]: realised(r, grids[bk]) for r in rules} for bk in BOOKS}
    book_ret = {bk: grids[bk][:, -1] for bk in BOOKS}
    base = rules[0]

    # --------------------------------------------------------- rules.csv ----
    rows: list[dict[str, Any]] = []
    for bk in BOOKS:
        for smp, m in samples:
            for rule in rules:
                row: dict[str, Any] = {
                    "book": bk,
                    "sample": smp,
                    "family": rule["family"],
                    "param": rule["param"],
                    "rule": rule["name"],
                }
                row.update(
                    rule_stats(
                        ret[bk][rule["name"]][m],
                        book_ret[bk][m],
                        rule["stopped"][m],
                        rule["stop_j"][m],
                        dates[m],
                    )
                )
                rows.append(row)
    rules_df = pd.DataFrame(rows)
    rcols = [
        "family",
        "param",
        "rule",
        "n_days",
        "n_stopped",
        "frac_stopped",
        "mean_stop_clock_et",
        "mean_calendar",
        "sd",
        "Sharpe_ann",
        "t",
        "MaxDD",
        "worst_day",
        "worst_date",
        "mean_stopped_under_rule",
        "mean_stopped_under_book",
        "saved_per_stopped_day",
        "mean_not_stopped",
    ]
    rules_df.to_csv(OUT / "rules.csv", index=False)
    for bk in BOOKS:
        for smp, _ in samples:
            sub = rules_df[(rules_df["book"] == bk) & (rules_df["sample"] == smp)]
            show(sub[rcols], f"rules.csv  |  book {bk}  |  sample {smp}")

    # ------------------------------------------------- decomposition.csv ----
    drows = [
        decomposition_rows(
            rule,
            ret[bk][rule["name"]][m],
            book_ret[bk][m],
            rule["stopped"][m],
            bk,
            smp,
        )
        for bk in BOOKS
        for smp, m in samples
        for rule in rules
        if rule["family"] != "reference" or rule["stopped"].any()
    ]
    dec_df = pd.DataFrame(drows)
    dcols = [
        "family",
        "rule",
        "n_stopped",
        "mean_at_stop",
        "mean_counterfactual_1530",
        "mean_difference",
        "sum_difference",
        "n_helped",
        "mean_helped_at_stop",
        "mean_helped_counterfactual",
        "sum_helped",
        "n_hurt",
        "mean_hurt_at_stop",
        "mean_hurt_counterfactual",
        "sum_hurt",
        "n_tied",
    ]
    dec_df.to_csv(OUT / "decomposition.csv", index=False)
    for bk in BOOKS:
        for smp, _ in samples:
            sub = dec_df[(dec_df["book"] == bk) & (dec_df["sample"] == smp)]
            show(
                sub[dcols],
                f"decomposition.csv  |  the stopped days against their 15:30 "
                f"counterfactual  |  book {bk}  |  sample {smp}",
            )

    # ----------------------------------------------------------- oos.csv ----
    orows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    for bk in BOOKS:
        for fam in (*FAMILIES, "all_families"):
            cands = [base] + [
                c
                for c in rules
                if c["family"] == fam
                or (fam == "all_families" and c["family"] in FAMILIES)
            ]
            series = np.vstack([ret[bk][c["name"]] for c in cands])
            stops = np.vstack([c["stopped"] for c in cands])
            clocks = np.vstack([c["stop_j"] for c in cands])
            scored, wf_r, wf_stop, wf_j, pick = walk_forward(series, stops, clocks)
            selected[(bk, fam)] = (scored, wf_r, wf_stop, wf_j)
            names = np.array([c["name"] for c in cands])
            for smp, m in samples:
                sel = scored & m
                if not sel.any():
                    continue
                st = rule_stats(
                    wf_r[sel], book_ret[bk][sel], wf_stop[sel], wf_j[sel], dates[sel]
                )
                bs = rule_stats(
                    book_ret[bk][sel],
                    book_ret[bk][sel],
                    base["stopped"][sel],
                    base["stop_j"][sel],
                    dates[sel],
                )
                counts = pd.Series(names[pick[sel]]).value_counts()
                orows.append(
                    {
                        "block": "walk_forward_selected",
                        "book": bk,
                        "sample": smp,
                        "family": fam,
                        "param": float("nan"),
                        "rule": (
                            f"{fam}: rule chosen on an expanding window, "
                            f"{EMBARGO_DAYS}-day embargo"
                        ),
                        **st,
                        "Sharpe_book": bs["Sharpe_ann"],
                        "delta_Sharpe": st["Sharpe_ann"] - bs["Sharpe_ann"],
                        "beats_book": bool(st["Sharpe_ann"] > bs["Sharpe_ann"]),
                        "modal_pick": str(counts.index[0]),
                        "modal_pick_share": float(counts.iloc[0] / counts.sum()),
                        "null_share": float(counts.get(base["name"], 0) / counts.sum()),
                    }
                )
        scored_any = walk_forward(
            np.vstack([ret[bk][base["name"]]]),
            np.vstack([base["stopped"]]),
            np.vstack([base["stop_j"]]),
        )[0]
        for smp, m in samples:
            sel = scored_any & m
            bs = rule_stats(
                book_ret[bk][sel],
                book_ret[bk][sel],
                base["stopped"][sel],
                base["stop_j"][sel],
                dates[sel],
            )
            for rule in rules:
                st = rule_stats(
                    ret[bk][rule["name"]][sel],
                    book_ret[bk][sel],
                    rule["stopped"][sel],
                    rule["stop_j"][sel],
                    dates[sel],
                )
                orows.append(
                    {
                        "block": "fixed_rule_oos",
                        "book": bk,
                        "sample": smp,
                        "family": rule["family"],
                        "param": rule["param"],
                        "rule": rule["name"],
                        **st,
                        "Sharpe_book": bs["Sharpe_ann"],
                        "delta_Sharpe": st["Sharpe_ann"] - bs["Sharpe_ann"],
                        "beats_book": bool(st["Sharpe_ann"] > bs["Sharpe_ann"]),
                        "modal_pick": rule["name"],
                        "modal_pick_share": 1.0,
                        "null_share": float("nan"),
                    }
                )
    oos_df = pd.DataFrame(orows)
    oos_df.to_csv(OUT / "oos.csv", index=False)
    ocols = [
        "block",
        "family",
        "param",
        "rule",
        "n_days",
        "n_stopped",
        "frac_stopped",
        "mean_stop_clock_et",
        "mean_calendar",
        "sd",
        "Sharpe_ann",
        "Sharpe_book",
        "delta_Sharpe",
        "beats_book",
        "t",
        "MaxDD",
        "worst_day",
        "modal_pick",
        "modal_pick_share",
        "null_share",
    ]
    print(
        f"\nout-of-sample day set: expanding training window, {EMBARGO_DAYS}-day "
        f"embargo, minimum {WARMUP_SESSIONS} training days, so the first "
        f"{WARMUP_SESSIONS + EMBARGO_DAYS} days are never scored"
    )
    for bk in BOOKS:
        for smp, _ in samples:
            sub = oos_df[(oos_df["book"] == bk) & (oos_df["sample"] == smp)]
            show(sub[ocols], f"oos.csv  |  book {bk}  |  sample {smp}")

    # ------------------------------------------------------- placebo.csv ----
    rng = np.random.default_rng(PLACEBO_SEED)
    prows: list[dict[str, Any]] = []
    for bk in BOOKS:
        scored_any = walk_forward(
            np.vstack([ret[bk][base["name"]]]),
            np.vstack([base["stopped"]]),
            np.vstack([base["stop_j"]]),
        )[0]
        for smp, m in samples:
            for day_set, sel in (("full", m), ("oos", m & scored_any)):
                for rule in rules:
                    if rule["family"] == "reference":
                        continue
                    prows.append(
                        {
                            "book": bk,
                            "sample": smp,
                            "day_set": day_set,
                            "block": "fixed_rule",
                            "family": rule["family"],
                            "param": rule["param"],
                            "rule": rule["name"],
                            **placebo_row(
                                book_ret[bk][sel],
                                grids[bk][sel],
                                rule["stopped"][sel],
                                rule["stop_j"][sel],
                                rng,
                            ),
                        }
                    )
            for fam in (*FAMILIES, "all_families"):
                wf_scored, _, wf_stop, wf_j = selected[(bk, fam)]
                sel = m & wf_scored
                prows.append(
                    {
                        "book": bk,
                        "sample": smp,
                        "day_set": "oos",
                        "block": "walk_forward_selected",
                        "family": fam,
                        "param": float("nan"),
                        "rule": (
                            f"{fam}: rule chosen on an expanding window, "
                            f"{EMBARGO_DAYS}-day embargo"
                        ),
                        **placebo_row(
                            book_ret[bk][sel],
                            grids[bk][sel],
                            wf_stop[sel],
                            wf_j[sel],
                            rng,
                        ),
                    }
                )
    plac_df = pd.DataFrame(prows)
    plac_df.to_csv(OUT / "placebo.csv", index=False)
    pcols = [
        "day_set",
        "block",
        "family",
        "param",
        "rule",
        "n_placebo",
        "sharpe_rule",
        "placebo_mean",
        "placebo_sd",
        "placebo_q05",
        "placebo_q50",
        "placebo_q95",
        "pct_rank",
        "p_value",
    ]
    print(
        f"\nplacebo: {N_PLACEBO} run-length-preserving shuffles of the stopped-day "
        f"mask per rule (same stopped count, same run lengths, the rule's own stop "
        f"clocks laid on the drawn days in date order), seed {PLACEBO_SEED}; "
        f"p = (1 + #(placebo Sharpe >= rule Sharpe)) / (1 + n_placebo)"
    )
    for bk in BOOKS:
        for smp, _ in samples:
            sub = plac_df[(plac_df["book"] == bk) & (plac_df["sample"] == smp)]
            show(sub[pcols], f"placebo.csv  |  book {bk}  |  sample {smp}")

    # --------------------------------------------------- trend_tercile.csv ---
    q1, q2 = day["trend_ratio"].quantile([1 / 3, 2 / 3]).to_numpy()
    trend = (day["trend_ratio"] > q2).to_numpy()
    trows: list[dict[str, Any]] = []
    for bk in BOOKS:
        for smp, m in samples:
            sel = m & trend
            for rule in rules:
                rr = ret[bk][rule["name"]][sel]
                bb = book_ret[bk][sel]
                trows.append(
                    {
                        "book": bk,
                        "sample": smp,
                        "family": rule["family"],
                        "param": rule["param"],
                        "rule": rule["name"],
                        "n_trend_days": int(sel.sum()),
                        "n_stopped": int(rule["stopped"][sel].sum()),
                        "frac_stopped": float(rule["stopped"][sel].mean()),
                        "mean_rule": float(rr.mean()),
                        "mean_book": float(bb.mean()),
                        "mean_difference": float((rr - bb).mean()),
                        "Sharpe_rule": p30.sharpe(rr),
                        "Sharpe_book": p30.sharpe(bb),
                        "worst_rule": float(rr.min()),
                        "worst_book": float(bb.min()),
                    }
                )
    trend_df = pd.DataFrame(trows)
    trend_df.to_csv(OUT / "trend_tercile.csv", index=False)
    tcols = [
        "family",
        "param",
        "rule",
        "n_trend_days",
        "n_stopped",
        "frac_stopped",
        "mean_rule",
        "mean_book",
        "mean_difference",
        "Sharpe_rule",
        "Sharpe_book",
        "worst_rule",
        "worst_book",
    ]
    print(
        f"\ntrend tercile: proposal 32's own cut of |sum of bar returns| / sum of "
        f"|bar returns| on the full sample, tercile cut points {q1:.6f} / {q2:.6f}; "
        f"the top tercile is the population the stop is meant to protect"
    )
    for bk in BOOKS:
        for smp, _ in samples:
            sub = trend_df[(trend_df["book"] == bk) & (trend_df["sample"] == smp)]
            show(sub[tcols], f"trend_tercile.csv  |  book {bk}  |  sample {smp}")

    # ------------------------------------------------------------- gate ----
    key = ["book", "sample", "family", "rule"]
    g = oos_df[oos_df["family"] != "reference"].merge(
        plac_df.loc[plac_df["day_set"] == "oos", [*key, "p_value", "sharpe_rule"]],
        on=key,
        how="left",
    )
    assert not g["p_value"].isna().all(), "the gate lost its placebo leg in the merge"
    same = g["sharpe_rule"].notna() & g["Sharpe_ann"].notna()
    assert np.allclose(
        g.loc[same, "sharpe_rule"], g.loc[same, "Sharpe_ann"], rtol=0, atol=EXACT_TOL
    ), "the placebo and the out-of-sample table disagree on the rule's own Sharpe"
    g["adopted"] = g["beats_book"] & (g["p_value"] < PLACEBO_ALPHA)
    g.to_csv(OUT / "gate.csv", index=False)
    show(
        g[
            [
                "book",
                "sample",
                "block",
                "family",
                "param",
                "rule",
                "Sharpe_ann",
                "Sharpe_book",
                "beats_book",
                "p_value",
                "adopted",
            ]
        ],
        f"gate.csv  |  adopted = out-of-sample beats the book AND placebo "
        f"p < {PLACEBO_ALPHA}",
    )
    prim = g[(g["book"] == "crossed") & (g["block"] == "fixed_rule_oos")]
    n_cells = int(len(prim))
    print(f"\nrules adopted on any book or sample: {int(g['adopted'].sum())}")
    print(
        f"rules adopted on the primary book (exit-at-15:30 crossed-quoted): "
        f"{int(prim['adopted'].sum())} of {n_cells} cells"
    )
    for fam in (*FAMILIES, "all_families"):
        sub = g[
            (g["book"] == "crossed")
            & (g["block"] == "walk_forward_selected")
            & (g["family"] == fam)
        ]
        for _, r in sub.iterrows():
            verdict = "BEATS THE BOOK" if r["beats_book"] else "DOES NOT BEAT THE BOOK"
            print(
                f"  family verdict, crossed {r['sample']:<9s} {fam:<16s} "
                f"walk-forward {r['Sharpe_ann']:7.3f} against the book "
                f"{r['Sharpe_book']:7.3f}  {verdict}"
            )
    n_par = len(K_GRID) + len(K_GRID) + len(L_GRID) + 1
    print(
        f"\nmultiple testing: {n_par} parameterisations across {len(FAMILIES)} "
        f"families were tried, each scored on 2 books x 2 samples, so the gate above "
        f"reads {n_par * 2} cells on the primary book alone and a 5% level expects "
        f"{n_par * 2 * PLACEBO_ALPHA:.1f} false adoptions there by chance."
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
