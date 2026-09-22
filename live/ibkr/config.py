"""Run configuration for the live 0DTE delta-hedged straddle book.

The book this executes is the **afternoon hold** book (audit 2026-09-18,
sections 4-7): the morning premium is gone on the 2024-25 sessions and the
afternoon/settlement premium is what remains, so the runner

* enters in the AFTERNOON, at a fixed clock (``entry_mode="fixed"``, the
  default, ``13:30``) or at the clock a *causal* premium selector chooses from
  the ledger each session (``entry_mode="selector"``);
* delta-hedges every 30 minutes through 15:30 with ES for the bulk and MES
  for the remainder, from the implied volatility corrected by the trailing
  per-clock realized-over-implied factor;
* HOLDS the body to cash settlement (``terminal="hold"``), flattens the
  futures at 16:00:00 ET, and reconciles the official settlement the next
  morning with ``--reconcile``;
* sizes by the stress table -- contracts = capital x fraction / the dollar
  loss of one hedged straddle in a 5 % last-bar jump -- never by margin,
  then scales that size by proposal 50's DELEVERAGING CANDIDATE: full size
  while the trailing realized variance sits at or below its own expanding
  lagged 90th percentile, half above it, flat above the 97.5th
  (``delever``, off with ``--no-delever``);
* on the no-short calendars -- today the last trading session of a month,
  where proposal 54 measured the short book losing into the month-end close
  -- does not sell; by default (``month_end_mode="override"``) it BUYS the
  15:30 nearest-OTM straddle instead, one per straddle the short program
  would have sold, and holds it to cash settlement (``--month-end-mode
  override|sit_out|off``; ``--no-calendar-guard`` is ``off``);
* on the monthly-expiration session -- the third Friday, or the session
  before it when that Friday is a holiday -- multiplies the short program's
  contract count by ``third_friday_size_multiplier`` (proposal 58), after the
  deleveraging multiplier and under ``max_straddles``
  (``--third-friday-multiplier``; ``--no-third-friday`` is 1.0).  The default,
  1.1, is study 66's pre-registered lower end of the Kelly ratio.
* writes the book on SPX Weeklys (``instrument="spxw"``) or, one tenth the
  size for a small account, on Cboe Mini-SPX options (``--instrument xsp``,
  hedged in MES then the E-nano NES); see ``INSTRUMENTS`` and README 1.9.

``terminal="flatten"`` keeps the old exit-at-15:30 variant; it is the
optional variant now, not the book of record.

Nothing in this module talks to a socket.  ``Config`` is a plain dataclass
so that the replay harness and the tests can build one without IB.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Literal, TextIO

from live.ibkr.calendar_guard import MONTH_END_MODES, NO_SHORT_CALENDARS

Mode = Literal["dry", "paper", "live"]
Instrument = Literal["spxw", "xsp"]
EntryMode = Literal["selector", "fixed"]
SpLegRule = Literal["vol_managed", "brake", "off"]
Terminal = Literal["hold", "flatten"]
MonthEndMode = Literal["override", "sit_out", "off"]

#: How the month-end long is sized.  ``match_short``: one long straddle per
#: straddle the short program would have sold that session (the stress table at
#: the configured capital and fraction, at the clock the selector picked,
#: WITHOUT the deleveraging multiplier).  That is the variant the evidence was
#: measured for.  Sizing the long to the same loss budget as the short -- a
#: median 28x more contracts on the 70 month-ends, since the long can lose only
#: its premium -- is described in the README and deliberately NOT offered: its
#: liquidity is untested.
MONTH_END_LONG_SIZES: tuple[str, ...] = ("match_short",)

#: Proposal 58: on the monthly-expiration session (the third Friday, or the
#: session before it when that Friday is a holiday) the short program's contract
#: count -- the stress table's, after the deleveraging multiplier -- is multiplied
#: by this, floored to whole contracts and capped at ``max_straddles``.  1.0 is
#: NO CHANGE.
# 1.5 is the OPERATOR'S CHOICE (2026-09-21), between study 66's two numbers: the
# pre-registered lower end of the Kelly ratio (third Friday / other sessions),
# 1.1, which floored changes no count below ten straddles, and the point ratio,
# 1.86.  1.5 is the smallest multiplier that adds a contract at every count the
# stress table gives at $1M (2 -> 3, 3 -> 4, 4 -> 6, 5 -> 7, 6 -> 9); a 5 % jump
# on a third Friday then costs up to 15 % of capital at the default fraction
# (writeup/intraday_proposals/66_kelly_sizing.py, results/
# atm_straddle_intraday_holdclose/proposals/66/c_multiplier.csv).
THIRD_FRIDAY_SIZE_MULTIPLIER = 1.5

#: Every 30-minute stamp the research tape carries.  16:00 is a settlement
#: print, never a quote, so it is not in here.
SESSION_STAMPS: tuple[str, ...] = (
    "10:00",
    "10:30",
    "11:00",
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

#: The last stamp the book hedges on (hold variant).
LAST_HEDGE_CLOCK = "15:30"

#: The stamp the month-end override buys its straddle at: the research's
#: 15:30 stamp, the last one the tape quotes before the settlement print.
MONTH_END_LONG_CLOCK = LAST_HEDGE_CLOCK

#: The bell: the hold book flattens its futures here and the options settle.
SETTLEMENT_CLOCK = "16:00:00"

#: Clocks the selector may enter at: proposal 43's ENTRIES (10:00..15:00).
ENTRY_CLOCKS: tuple[str, ...] = SESSION_STAMPS[:-1]

#: The default candidate set: the afternoon, where the premium now lives
#: (proposal 43: six afternoon clocks positive per contract with t > 2 on the
#: 413 unseen sessions; fixed 11:00 t 1.0).
AFTERNOON_CLOCKS: tuple[str, ...] = (
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
)

#: Exit clocks the flatten variant is defined for.  Only 15:30 lands on the
#: research tape; 15:40 and 15:50 are supported but unvalidated (see README).
ALLOWED_EXIT_CLOCKS: tuple[str, ...] = ("15:30", "15:40", "15:50")

#: Exit clocks the FakeBroker replay can serve (the tape is 30-minute).
REPLAYABLE_EXIT_CLOCKS: tuple[str, ...] = ("15:30",)

#: Dollars per S&P 500 INDEX point, by hedge instrument: the E-mini (ES),
#: the Micro E-mini (MES) and the E-nano (NES, CME, listed 2026-08-24, one
#: tenth of a Micro).  Every future is on the S&P 500 index itself, whatever
#: option instrument the book trades.
FUTURES_MULTIPLIER: dict[str, float] = {"ES": 50.0, "MES": 5.0, "NES": 0.5}


@dataclass(frozen=True)
class InstrumentSpec:
    """The option contract the book is written on.

    ``index_scale`` is the instrument's index per S&P 500 point (1 for SPX,
    0.1 for XSP, whose index is one tenth of the S&P 500 and whose settlement
    is one tenth of the official SPX close).  Prices, strikes, spot and the
    stress table are all in the instrument's own points; only the futures
    hedge and the premium ledger, both kept in S&P 500 units, need the scale.
    """

    key: str
    label: str
    option_symbol: str  #: the IB ``Option.symbol`` / ``BAG`` symbol
    trading_class: str  #: the IB ``tradingClass`` (SPXW for the weeklys; XSP)
    index_symbol: str  #: the IB ``Index`` whose print is the spot
    index_scale: float
    option_tick_lo: float
    option_tick_hi: float
    option_tick_break: float
    default_hedge_symbols: tuple[str, ...]


INSTRUMENTS: dict[str, InstrumentSpec] = {
    # SPXW: $0.05 below $3.00 and $0.10 at/above it (L-8).
    "spxw": InstrumentSpec(
        key="spxw",
        label="SPX Weeklys (SPXW)",
        option_symbol="SPX",
        trading_class="SPXW",
        index_symbol="SPX",
        index_scale=1.0,
        option_tick_lo=0.05,
        option_tick_hi=0.10,
        option_tick_break=3.00,
        default_hedge_symbols=("ES", "MES"),
    ),
    # XSP (Cboe Mini-SPX, 1/10 of the S&P 500, $100 multiplier): the minimum
    # tick is $0.01 for every series; PM-settled to one tenth of the official
    # SPX close; expiring weeklys stop trading at 16:00 ET like SPXW.  One
    # XSP straddle is a tenth of an SPX straddle, so the ladder is MES then
    # NES.
    "xsp": InstrumentSpec(
        key="xsp",
        label="Mini-SPX (XSP)",
        option_symbol="XSP",
        trading_class="XSP",
        index_symbol="XSP",
        index_scale=0.1,
        option_tick_lo=0.01,
        option_tick_hi=0.01,
        option_tick_break=3.00,
        default_hedge_symbols=("MES", "NES"),
    ),
}

DEFAULT_PORT: dict[str, int] = {"dry": 7497, "paper": 7497, "live": 7496}

#: TWS 7496 and IB Gateway 4001 are the LIVE listeners; 7497 / 4002 are paper.
#: The gate keys on BOTH the mode string and the port, so ``--mode paper
#: --port 7496`` is refused rather than silently routed to the live account.
LIVE_PORTS: tuple[int, ...] = (7496, 4001)
PAPER_PORTS: tuple[int, ...] = (7497, 4002)

LIVE_ENV_VAR = "HARXHAR_LIVE"
LIVE_ENV_VALUE = "I_UNDERSTAND"
LIVE_TYPED_ACK = "SHORT GAMMA"

BANNER = """
################################################################################
#  L I V E   M O D E  --  REAL ORDERS GO TO A REAL IB ACCOUNT                  #
#                                                                              #
#  Short {instrument} 0DTE straddle held to CASH SETTLEMENT, delta-hedged
#  with {ladder}.  Loss is not bounded by the premium received and the last
#  bar is unhedged.
#  n={n}  entry={entry} ET  terminal={terminal}  kill={kill} premium units
#  port={port} ({portkind})
################################################################################
"""


def parse_clock(text: str) -> dtime:
    """Parse ``HH:MM`` or ``HH:MM:SS`` (ET wall clock) into a ``time``."""
    parts = text.split(":")
    if len(parts) == 2:
        h, m, s = int(parts[0]), int(parts[1]), 0
    elif len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError("not a clock: " + repr(text))
    return dtime(h, m, s)


class LiveModeRefused(RuntimeError):
    """Raised when live mode is requested without the explicit consents."""


@dataclass
class Config:
    """Everything the runner needs; no IB object is held here."""

    mode: Mode = "dry"
    host: str = "127.0.0.1"
    port: int | None = None
    client_id: int = 17
    account: str | None = None

    # -- the decision layer ---------------------------------------------
    #: PINNED to the fixed clock (operator's choice, 2026-09-21): every costed
    #: number for this book (studies 64-67) is the fixed 13:30 entry; proposal 46
    #: adopted the selector in 0 of 14 cells.  ``--entry-mode selector`` restores it.
    entry_mode: EntryMode = "fixed"
    fixed_entry_clock: str = "13:30"
    selector_window: int = 252
    selector_min_sessions: int = 63
    candidate_clocks: tuple[str, ...] = AFTERNOON_CLOCKS
    ledger_path: str = os.path.join("results", "live_seed", "premium_ledger.parquet")
    ledger_live_path: str = os.path.join(
        "results", "live_journal", "premium_ledger_live.parquet"
    )
    #: V9: hedge off the implied vol corrected by the trailing per-clock
    #: realized-over-implied factor.  ``None`` window = expanding.
    delta_correction: bool = True
    delta_correction_window: int | None = None

    # -- the deleveraging candidate (proposal 50, part B2) ---------------
    #: Halve the size when the trailing realized variance is above its own
    #: expanding lagged ``delever_half_pct`` quantile and drop the day when it
    #: is above ``delever_zero_pct``.  A DESIGN CANDIDATE: proposal 50 measured
    #: it on its replay and adopted it nowhere.  ``--no-delever`` turns it off.
    delever: bool = True
    delever_window: int = 21
    delever_half_pct: float = 0.90
    delever_zero_pct: float = 0.975
    delever_min_sessions: int = 252

    # -- the month-end override (proposals 54 and 55) --------------------
    #: What a session on one of the named no-short calendars
    #: (``live.ibkr.calendar_guard``) does: ``override`` buys the 15:30
    #: straddle instead of selling, ``sit_out`` ends the day flat in
    #: preflight, ``off`` trades the short book.  Only ``month_end`` has
    #: evidence behind it today.
    month_end_mode: MonthEndMode = "override"
    month_end_long_size: str = "match_short"
    no_short_calendars: tuple[str, ...] = ("month_end",)

    # -- the third-Friday size (proposal 58) ------------------------------
    #: Multiplies the short program's contract count on the monthly-expiration
    #: session; read after the month-end calendar, which it never overrides.
    third_friday_size_multiplier: float = THIRD_FRIDAY_SIZE_MULTIPLIER
    #: The S&P-leg REPORT of a joint account (studies 67-68, ``sp_leg.py``):
    #: ``"vol_managed"`` (default; the one rule that met the written criterion
    #: at the timing the runner can deliver), ``"brake"`` (the insurance book's
    #: own multiplier; offered), or ``"off"``.  A report, never an order.
    sp_leg_rule: SpLegRule = "vol_managed"
    #: The S&P leg at FULL weight, in dollars; 0 reports the weight alone.
    sp_leg_notional: float = 0.0

    # -- the book --------------------------------------------------------
    terminal: Terminal = "hold"
    exit_clock: str = "15:30"  # flatten variant only
    rebalance_clocks: tuple[str, ...] | None = None  # None = derive from entry
    wings_pct: float | None = None

    # -- sizing ----------------------------------------------------------
    capital: float = 0.0
    stress_fraction: float = 0.10
    stress_jump: float = 0.05
    max_straddles: int = 10
    #: ``--n``: overrides the stress table.  ``None`` = size from the table.
    n_override: int | None = None

    # -- the option instrument and the hedge ladder ----------------------
    #: ``spxw`` (the book of record) or ``xsp`` (Mini-SPX, one tenth the
    #: size: the instrument a ~$60k account can run two straddles of under
    #: the 10 % stress rule).  See ``INSTRUMENTS`` and README 1.9.
    instrument: Instrument = "spxw"
    #: The futures the hedge is laid in, largest first; each rung takes the
    #: whole contracts it can (toward zero) and the last rung rounds the rest.
    #: ``None`` = the instrument's default (SPXW: ES then MES; XSP: MES then
    #: NES).  ``--hedge-ladder ES,MES,NES``.
    hedge_symbols: tuple[str, ...] | None = None
    es_multiplier: float = FUTURES_MULTIPLIER["ES"]
    mes_multiplier: float = FUTURES_MULTIPLIER["MES"]
    nes_multiplier: float = FUTURES_MULTIPLIER["NES"]
    index_multiplier: float = 100.0

    # -- order handling --------------------------------------------------
    passive_wait_s: float = 20.0
    #: The CEILING on crossing.  The working cap is sized off the live
    #: spread: ``ceil(spread / tick) + 1``, capped here (see L-2).
    max_cross_ticks: int = 12
    #: The option tick rule.  ``None`` = the instrument's (SPXW: $0.05 below
    #: $3.00 and $0.10 at/above it, L-8; XSP: $0.01 for every series).
    option_tick_lo: float | None = None
    option_tick_hi: float | None = None
    option_tick_break: float | None = None
    futures_tick: float = 0.25
    #: A snapshot older than this, halted, or crossed, is not tradeable.
    max_quote_age_s: float = 10.0

    max_loss_units: float = 6.0
    max_residual_delta: float = 0.6
    disconnect_grace_s: float = 90.0

    journal_dir: str = os.path.join("results", "live_journal")
    #: The replay cache lives OUTSIDE the live journal tree (L-25).
    replay_cache_dir: str = os.path.join("results", "live_replay_cache")
    tz: str = "America/New_York"

    # -- replay / plumbing ----------------------------------------------
    replay_date: str | None = None
    live_ack: bool = False
    strike_band_pct: float = 0.02
    #: replay only: fill combos crossed (bid/ask) or at the mid.
    replay_fill: str = "cross"
    #: ``--reconcile``: finalise yesterday and append its ledger records.
    reconcile: bool = False
    #: the official settlement, when the broker cannot supply one.
    settlement: float | None = None

    _derived: dict[str, object] = field(default_factory=dict, repr=False)

    # -- construction ----------------------------------------------------
    def __post_init__(self) -> None:
        if self.mode not in ("dry", "paper", "live"):
            raise ValueError("mode must be dry|paper|live, got " + repr(self.mode))
        if self.mode == "live":
            env = os.environ.get(LIVE_ENV_VAR, "")
            if env != LIVE_ENV_VALUE or not self.live_ack:
                raise LiveModeRefused(
                    "live mode refused: it needs BOTH --mode live passed "
                    "explicitly on the command line AND "
                    + LIVE_ENV_VAR
                    + "="
                    + LIVE_ENV_VALUE
                    + " in the environment. (explicit flag="
                    + repr(self.live_ack)
                    + ", env="
                    + repr(env)
                    + ")"
                )

        if self.port is None:
            self.port = DEFAULT_PORT[self.mode]
        # The second half of the live gate: the PORT, not just the mode.
        if int(self.port) in LIVE_PORTS and self.mode != "live":
            raise LiveModeRefused(
                "port "
                + str(self.port)
                + " is a LIVE listener (TWS 7496 / IB Gateway 4001) but the "
                "mode is "
                + repr(self.mode)
                + ": refusing. Pass --mode live with "
                + LIVE_ENV_VAR
                + "="
                + LIVE_ENV_VALUE
                + ", or point --port at a paper listener "
                + repr(PAPER_PORTS)
                + "."
            )
        if self.mode == "live" and int(self.port) not in LIVE_PORTS:
            raise LiveModeRefused(
                "--mode live with port "
                + str(self.port)
                + ": live mode must talk to a live listener "
                + repr(LIVE_PORTS)
                + ". Refusing rather than guessing which account this is."
            )

        if self.entry_mode not in ("selector", "fixed"):
            raise ValueError(
                "entry_mode must be selector|fixed, got " + repr(self.entry_mode)
            )
        if self.terminal not in ("hold", "flatten"):
            raise ValueError(
                "terminal must be hold|flatten, got " + repr(self.terminal)
            )

        for name, mult in (
            ("es_multiplier", self.es_multiplier),
            ("mes", self.mes_multiplier),
            ("nes_multiplier", self.nes_multiplier),
        ):
            if not (math.isfinite(mult) and mult > 0):
                raise ValueError(name + " must be a positive number of dollars/point")
        if self.es_multiplier < self.mes_multiplier:
            raise ValueError("es_multiplier must be the larger of the two")
        if self.mes_multiplier < self.nes_multiplier:
            raise ValueError("mes_multiplier must be larger than nes_multiplier")

        if self.instrument not in INSTRUMENTS:
            raise ValueError(
                "instrument must be one of "
                + repr(tuple(INSTRUMENTS))
                + ", got "
                + repr(self.instrument)
            )
        spec = INSTRUMENTS[self.instrument]
        # The option tick rule follows the instrument unless the caller set it.
        if self.option_tick_lo is None:
            self.option_tick_lo = spec.option_tick_lo
        if self.option_tick_hi is None:
            self.option_tick_hi = spec.option_tick_hi
        if self.option_tick_break is None:
            self.option_tick_break = spec.option_tick_break
        for name, tick in (
            ("option_tick_lo", self.option_tick_lo),
            ("option_tick_hi", self.option_tick_hi),
            ("option_tick_break", self.option_tick_break),
        ):
            if not (math.isfinite(float(tick)) and float(tick) > 0):
                raise ValueError(name + " must be a positive number")
        ladder = tuple(
            str(s).upper()
            for s in (
                spec.default_hedge_symbols
                if self.hedge_symbols is None
                else self.hedge_symbols
            )
        )
        if not ladder:
            raise ValueError("hedge_symbols is empty")
        for sym in ladder:
            if sym not in FUTURES_MULTIPLIER:
                raise ValueError(
                    "unknown hedge instrument "
                    + repr(sym)
                    + "; the ladder takes "
                    + repr(tuple(FUTURES_MULTIPLIER))
                )
        if len(set(ladder)) != len(ladder):
            raise ValueError("hedge_symbols repeats a symbol: " + repr(ladder))
        mults = [self.multiplier(s) for s in ladder]
        if any(a <= b for a, b in zip(mults[:-1], mults[1:])):
            raise ValueError(
                "hedge_symbols must run from the largest contract to the "
                "smallest, got " + repr(ladder)
            )
        self.hedge_symbols = ladder

        parse_clock(self.fixed_entry_clock)
        self.candidate_clocks = tuple(self.candidate_clocks)
        if not self.candidate_clocks:
            raise ValueError("candidate_clocks is empty")
        for c in self.candidate_clocks:
            if c not in ENTRY_CLOCKS:
                raise ValueError(
                    "candidate clock "
                    + repr(c)
                    + " is not a 30-minute stamp the tape carries "
                    + repr(ENTRY_CLOCKS)
                )
        if self.fixed_entry_clock not in ENTRY_CLOCKS:
            raise ValueError(
                "fixed_entry_clock "
                + repr(self.fixed_entry_clock)
                + " is not one of "
                + repr(ENTRY_CLOCKS)
            )

        if self.exit_clock not in ALLOWED_EXIT_CLOCKS:
            raise ValueError(
                "exit_clock must be one of "
                + repr(ALLOWED_EXIT_CLOCKS)
                + ", got "
                + repr(self.exit_clock)
            )
        if self.terminal == "hold" and self.exit_clock != ALLOWED_EXIT_CLOCKS[0]:
            raise ValueError(
                "--exit-clock "
                + repr(self.exit_clock)
                + " is inert with --terminal hold (the body cash-settles and "
                "only the futures are flattened, at 16:00:00). Pass "
                "--terminal flatten if you meant the exit book."
            )
        if self.n_override is not None and self.n_override < 1:
            raise ValueError("--n must be >= 1")
        if self.max_straddles < 1:
            raise ValueError("max_straddles must be >= 1")
        if not 0.0 < self.stress_fraction <= 1.0:
            raise ValueError("stress_fraction must be a fraction in (0, 1]")
        if not 0.0 < self.stress_jump < 1.0:
            raise ValueError("stress_jump must be a fraction in (0, 1)")
        if self.capital < 0.0:
            raise ValueError("capital must be >= 0")
        if self.wings_pct is not None and not 0.0 < self.wings_pct < 0.2:
            raise ValueError("wings_pct must be a fraction of spot in (0, 0.2)")
        if self.max_cross_ticks < 0:
            raise ValueError("max_cross_ticks (the ceiling) must be >= 0")

        if self.delever_window < 1:
            raise ValueError("delever_window must be >= 1 session")
        if not 0.0 < self.delever_half_pct < self.delever_zero_pct < 1.0:
            raise ValueError(
                "the deleveraging quantiles must satisfy 0 < delever_half_pct "
                "< delever_zero_pct < 1, got "
                + repr(self.delever_half_pct)
                + " and "
                + repr(self.delever_zero_pct)
            )
        if self.delever_min_sessions < 1:
            raise ValueError("delever_min_sessions must be >= 1")

        self.no_short_calendars = tuple(self.no_short_calendars)
        unknown = [n for n in self.no_short_calendars if n not in NO_SHORT_CALENDARS]
        if unknown:
            raise ValueError(
                "unknown no_short_calendars "
                + repr(unknown)
                + "; registered: "
                + repr(sorted(NO_SHORT_CALENDARS))
            )
        if self.month_end_mode not in MONTH_END_MODES:
            raise ValueError(
                "month_end_mode must be one of "
                + repr(MONTH_END_MODES)
                + ", got "
                + repr(self.month_end_mode)
            )
        if self.month_end_long_size not in MONTH_END_LONG_SIZES:
            raise ValueError(
                "month_end_long_size must be one of "
                + repr(MONTH_END_LONG_SIZES)
                + ", got "
                + repr(self.month_end_long_size)
                + " (sizing the long to the loss budget is documented in the "
                "README and not enabled: a median 28x the contracts, untested for "
                "liquidity)"
            )
        tf = float(self.third_friday_size_multiplier)
        if not (math.isfinite(tf) and tf >= 1.0):
            raise ValueError(
                "third_friday_size_multiplier must be a finite number >= 1 (1 is "
                "no change; the rule sells MORE on the monthly-expiration "
                "session), got " + repr(self.third_friday_size_multiplier)
            )
        self.third_friday_size_multiplier = tf
        if self.sp_leg_rule not in ("vol_managed", "brake", "off"):
            raise ValueError(
                "sp_leg_rule must be vol_managed|brake|off, got "
                + repr(self.sp_leg_rule)
            )
        notional = float(self.sp_leg_notional)
        if not (math.isfinite(notional) and notional >= 0.0):
            raise ValueError(
                "sp_leg_notional must be a finite number >= 0 (dollars of the "
                "S&P leg at full weight; 0 reports the weight alone), got "
                + repr(self.sp_leg_notional)
            )
        self.sp_leg_notional = notional

        if self.rebalance_clocks is not None:
            self.rebalance_clocks = tuple(self.rebalance_clocks)
            for c in self.rebalance_clocks:
                parse_clock(c)

        if self.replay_date is not None and self.terminal == "flatten":
            if self.exit_clock not in REPLAYABLE_EXIT_CLOCKS:
                raise ValueError(
                    "--date replay supports exit_clock in "
                    + repr(REPLAYABLE_EXIT_CLOCKS)
                    + " only: the research tape is stamped every 30 minutes, so "
                    + repr(self.exit_clock)
                    + " has no recorded quotes. Run that exit clock forward in "
                    "dry/paper mode instead."
                )

    # -- derived views ---------------------------------------------------
    @property
    def hold_to_settle(self) -> bool:
        return self.terminal == "hold"

    @property
    def book_name(self) -> str:
        if self.hold_to_settle:
            return "afternoon-hold-to-cash-settlement"
        return "afternoon-exit-at-" + self.exit_clock

    @property
    def is_replay(self) -> bool:
        return self.replay_date is not None

    @property
    def flatten_clock(self) -> str:
        """When the FUTURES leg dies.  Hold: the bell.  Flatten: the exit."""
        return SETTLEMENT_CLOCK if self.hold_to_settle else self.exit_clock + ":00"

    def multiplier(self, symbol: str) -> float:
        sym = str(symbol).upper()
        if sym == "ES":
            return float(self.es_multiplier)
        if sym == "MES":
            return float(self.mes_multiplier)
        if sym == "NES":
            return float(self.nes_multiplier)
        raise ValueError("unknown hedge instrument " + repr(symbol))

    @property
    def spec(self) -> InstrumentSpec:
        return INSTRUMENTS[self.instrument]

    @property
    def index_scale(self) -> float:
        """The instrument's index per S&P 500 point (1 for SPX, 0.1 for XSP)."""
        return float(self.spec.index_scale)

    @property
    def hedge_ladder(self) -> tuple[str, ...]:
        """The hedge futures, largest first (resolved in ``__post_init__``)."""
        assert self.hedge_symbols is not None
        return tuple(self.hedge_symbols)

    @property
    def ladder_multipliers(self) -> tuple[float, ...]:
        return tuple(self.multiplier(s) for s in self.hedge_ladder)

    def option_tick_for(self, price: float) -> float:
        """The instrument's tick: SPXW $0.05 / $0.10 either side of $3.00, XSP $0.01."""
        lo, hi, brk = self.option_tick_lo, self.option_tick_hi, self.option_tick_break
        assert lo is not None and hi is not None and brk is not None
        p = abs(float(price))
        if not math.isfinite(p):
            return float(hi)
        return float(lo) if p < float(brk) else float(hi)

    def cross_ticks_for(self, spread: float, tick: float) -> int:
        """Crossing allowance sized off the LIVE spread, capped by the ceiling.

        ``ceil(spread / tick) + 1`` is enough to walk from the mid to the far
        touch and one tick through it; the config ceiling bounds it.
        """
        if not (math.isfinite(spread) and spread > 0 and tick > 0):
            return int(self.max_cross_ticks)
        want = int(math.ceil(float(spread) / float(tick))) + 1
        return int(max(0, min(want, int(self.max_cross_ticks))))

    def ladder(self, entry_clock: str) -> tuple[str, ...]:
        """Rebalance clocks after ``entry_clock``, in order.

        Hold: every stamp after the entry through 15:30 inclusive.
        Flatten: every stamp after the entry and strictly before the exit.
        An explicit ``rebalance_clocks`` is honoured but ALWAYS filtered
        against the terminal clock (L-30), never only when it is the default.
        """
        pool = (
            self.rebalance_clocks
            if self.rebalance_clocks is not None
            else SESSION_STAMPS
        )
        limit = LAST_HEDGE_CLOCK if self.hold_to_settle else self.exit_clock
        out = (
            [c for c in pool if entry_clock < c <= limit]
            if self.hold_to_settle
            else [c for c in pool if entry_clock < c < limit]
        )
        return tuple(out)

    def banner(self) -> str:
        port = int(self.port or 0)
        return BANNER.format(
            instrument=self.spec.trading_class,
            ladder="+".join(self.hedge_ladder),
            n=self.n_override if self.n_override is not None else "sized by stress",
            entry=self.fixed_entry_clock
            if self.entry_mode == "fixed"
            else "selector(" + ",".join(self.candidate_clocks) + ")",
            terminal=self.terminal,
            kill=self.max_loss_units,
            port=port,
            portkind="LIVE" if port in LIVE_PORTS else "paper",
        )

    # -- CLI -------------------------------------------------------------
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog="python -m live.ibkr.run_day",
            description=(
                "Run one session of the afternoon delta-hedged nearest-OTM SPX "
                "0DTE straddle, held to cash settlement (default) or flattened "
                "at --exit-clock."
            ),
        )
        p.add_argument("--mode", choices=("dry", "paper", "live"), default="dry")
        p.add_argument(
            "--instrument",
            choices=tuple(INSTRUMENTS),
            default="spxw",
            dest="instrument",
            help="the option contract the book is written on: spxw (default) or "
            "xsp (Mini-SPX, one tenth the size; README 1.9)",
        )
        p.add_argument(
            "--hedge-ladder",
            default=None,
            dest="hedge_ladder",
            help="comma-separated futures for the hedge, largest first, e.g. "
            "ES,MES or MES,NES (default: the instrument's; NES is the E-nano)",
        )
        p.add_argument(
            "--n",
            type=int,
            default=None,
            dest="n_override",
            help="override the stress sizing with this many straddles (the "
            "implied stress fraction is journaled)",
        )
        p.add_argument("--capital", type=float, default=0.0)
        p.add_argument(
            "--stress-fraction", type=float, default=0.10, dest="stress_fraction"
        )
        p.add_argument("--stress-jump", type=float, default=0.05, dest="stress_jump")
        p.add_argument("--max-straddles", type=int, default=10, dest="max_straddles")
        p.add_argument(
            "--entry-mode",
            choices=("selector", "fixed"),
            default="fixed",
            dest="entry_mode",
        )
        p.add_argument(
            "--entry-clock",
            default="13:30",
            dest="fixed_entry_clock",
            help="the clock used when --entry-mode fixed, and the fall-back "
            "while the selector is warming up",
        )
        p.add_argument(
            "--candidates",
            choices=("afternoon", "all"),
            default="afternoon",
            dest="candidates",
            help="the clocks the selector may pick from",
        )
        p.add_argument(
            "--selector-window", type=int, default=252, dest="selector_window"
        )
        p.add_argument(
            "--selector-min-sessions",
            type=int,
            default=63,
            dest="selector_min_sessions",
        )
        p.add_argument("--ledger", default=None, dest="ledger_path")
        p.add_argument("--ledger-live", default=None, dest="ledger_live_path")
        p.add_argument(
            "--no-delta-correction",
            action="store_false",
            dest="delta_correction",
            default=True,
            help="hedge off the raw implied volatility instead of the "
            "realized-over-implied corrected one",
        )
        p.add_argument(
            "--no-delever",
            action="store_false",
            dest="delever",
            default=True,
            help="size at full stress every session: turn OFF proposal 50's "
            "deleveraging candidate (half size above the expanding lagged "
            "90th percentile of the trailing realized variance, flat above "
            "the 97.5th)",
        )
        p.add_argument(
            "--month-end-mode",
            choices=MONTH_END_MODES,
            default=None,
            dest="month_end_mode",
            help="the last trading session of a month: 'override' (default) "
            "sells nothing and BUYS the 15:30 straddle, one per straddle the "
            "short program would have sold, held to settlement; 'sit_out' ends "
            "the day flat in preflight; 'off' trades the short book (proposals "
            "54 and 55)",
        )
        p.add_argument(
            "--no-calendar-guard",
            action="store_true",
            dest="no_calendar_guard",
            default=False,
            help="alias of --month-end-mode off: trade the short book every session",
        )
        p.add_argument(
            "--third-friday-multiplier",
            type=float,
            default=None,
            dest="third_friday_multiplier",
            help="on the monthly-expiration session (the third Friday, or the "
            "session before it when that Friday is a holiday) multiply the short "
            "program's contract count by this, after the deleveraging multiplier, "
            "floored, and capped at --max-straddles; --n is never multiplied "
            "(proposal 58; default "
            + f"{THIRD_FRIDAY_SIZE_MULTIPLIER:g}"
            + ", study 66's pre-registered lower end of the Kelly ratio)",
        )
        p.add_argument(
            "--no-third-friday",
            action="store_true",
            dest="no_third_friday",
            default=False,
            help="alias of --third-friday-multiplier 1: the ordinary size every "
            "session",
        )
        p.add_argument(
            "--sp-leg",
            choices=("vol_managed", "brake", "off"),
            default="vol_managed",
            dest="sp_leg_rule",
            help="the S&P-leg target the runner REPORTS for a joint account "
            "(studies 67-68): vol_managed (default), brake, or off",
        )
        p.add_argument(
            "--sp-leg-notional",
            type=float,
            default=0.0,
            dest="sp_leg_notional",
            help="dollars of the S&P leg at full weight; the report then states "
            "the target in dollars and in ES / MES (0: the weight alone)",
        )
        p.add_argument(
            "--terminal",
            choices=("hold", "flatten"),
            default="hold",
            dest="terminal",
        )
        p.add_argument(
            "--wings",
            type=float,
            default=None,
            dest="wings_pct",
            help="wing width as a fraction of spot, e.g. 0.015 for the "
            "defined-risk iron fly",
        )
        p.add_argument(
            "--date",
            default=None,
            dest="replay_date",
            help="YYYY-MM-DD: replay that session off the recorded chain "
            "through FakeBroker (no network)",
        )
        p.add_argument("--client-id", type=int, default=17, dest="client_id")
        p.add_argument("--account", default=None)
        p.add_argument(
            "--exit-clock",
            choices=ALLOWED_EXIT_CLOCKS,
            default="15:30",
            dest="exit_clock",
            help="flatten variant only",
        )
        p.add_argument(
            "--journal-dir",
            default=os.path.join("results", "live_journal"),
            dest="journal_dir",
        )
        p.add_argument("--port", type=int, default=None)
        p.add_argument(
            "--fill",
            choices=("cross", "mid"),
            default="cross",
            dest="replay_fill",
            help="replay only: fill the combos crossed (bid/ask) or at the mid",
        )
        p.add_argument(
            "--reconcile",
            action="store_true",
            help="finalise the pending session against the official "
            "settlement and append its records to the live ledger",
        )
        p.add_argument(
            "--settlement",
            type=float,
            default=None,
            help="--reconcile: the official settlement, when the broker "
            "statement cannot supply one",
        )
        return p

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> Config:
        """Build from ``argv``; ``None`` means ``sys.argv[1:]`` (L-29)."""
        argv = list(argv) if argv is not None else list(sys.argv[1:])
        parser = cls.build_parser()
        ns = parser.parse_args(argv)
        month_end_mode: MonthEndMode = ns.month_end_mode or "override"
        if ns.no_calendar_guard:
            if ns.month_end_mode not in (None, "off"):
                parser.error(
                    "--no-calendar-guard is --month-end-mode off; it contradicts "
                    "--month-end-mode " + str(ns.month_end_mode)
                )
            month_end_mode = "off"
        third_friday = (
            THIRD_FRIDAY_SIZE_MULTIPLIER
            if ns.third_friday_multiplier is None
            else float(ns.third_friday_multiplier)
        )
        if ns.no_third_friday:
            if ns.third_friday_multiplier not in (None, 1.0):
                parser.error(
                    "--no-third-friday is --third-friday-multiplier 1; it "
                    "contradicts --third-friday-multiplier "
                    + f"{ns.third_friday_multiplier:g}"
                )
            third_friday = 1.0
        # The equals form is a passed flag too (L-17).
        passed_mode = any(a == "--mode" or a.startswith("--mode=") for a in argv)
        live_ack = ns.mode == "live" and passed_mode
        ladder = (
            None
            if ns.hedge_ladder is None
            else tuple(
                x.strip().upper() for x in str(ns.hedge_ladder).split(",") if x.strip()
            )
        )
        cfg = cls(
            mode=ns.mode,
            instrument=ns.instrument,
            hedge_symbols=ladder,
            port=ns.port,
            client_id=ns.client_id,
            account=ns.account,
            entry_mode=ns.entry_mode,
            fixed_entry_clock=ns.fixed_entry_clock,
            selector_window=ns.selector_window,
            selector_min_sessions=ns.selector_min_sessions,
            candidate_clocks=ENTRY_CLOCKS
            if ns.candidates == "all"
            else AFTERNOON_CLOCKS,
            delta_correction=ns.delta_correction,
            delever=ns.delever,
            month_end_mode=month_end_mode,
            third_friday_size_multiplier=third_friday,
            sp_leg_rule=ns.sp_leg_rule,
            sp_leg_notional=ns.sp_leg_notional,
            terminal=ns.terminal,
            exit_clock=ns.exit_clock,
            wings_pct=ns.wings_pct,
            capital=ns.capital,
            stress_fraction=ns.stress_fraction,
            stress_jump=ns.stress_jump,
            max_straddles=ns.max_straddles,
            n_override=ns.n_override,
            journal_dir=ns.journal_dir,
            replay_date=ns.replay_date,
            live_ack=live_ack,
            reconcile=ns.reconcile,
            settlement=ns.settlement,
        )
        if ns.ledger_path:
            cfg.ledger_path = ns.ledger_path
        if ns.ledger_live_path:
            cfg.ledger_live_path = ns.ledger_live_path
        cfg.replay_fill = ns.replay_fill
        return cfg


def typed_live_ack(stream: TextIO | None = None) -> bool:
    """Third consent: a typed confirmation when a human is at the terminal.

    Non-interactive callers (cron, a supervisor) never reach the prompt, so
    the env var + explicit flag remain the only two gates there; a human at a
    TTY has to type the phrase.  Returns True when the session may proceed.
    """
    fh: TextIO = stream if stream is not None else sys.stdin
    try:
        interactive = bool(fh.isatty())
    except Exception:  # pragma: no cover - a detached stdin is not a TTY
        interactive = False
    if not interactive:
        return True
    print(
        "Type "
        + repr(LIVE_TYPED_ACK)
        + " to send real orders (anything else aborts): ",
        end="",
        flush=True,
    )
    try:
        got = fh.readline().strip()
    except Exception:  # pragma: no cover - a closed stdin is a refusal
        return False
    return got == LIVE_TYPED_ACK
