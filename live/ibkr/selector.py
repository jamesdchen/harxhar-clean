"""The causal entry-clock selector (proposal 46, estimator E2).

The research's reading, in one sentence: over 2020-2025 the premium walked
from the morning to the afternoon -- implied over realized remaining-window
variance went 1.17 -> 0.86 at 10:00, 1.04 -> 0.77 at 11:00 and 0.87 -> 1.10
at 14:00 -- so a book that always enters at 11:00 is trading a premium that
is no longer there, while afternoon entries held to settlement are positive
per contract with t > 2 on the 413 sessions the deck never saw.

E2 is the causal way to follow that migration.  For each candidate entry
clock ``c``,

    score_c(asof) = ratio_of_sums(c, asof, window)
                    / ratio_of_sums(c, asof, None)

-- the clock's premium over the trailing ``window`` sessions RELATIVE TO ITS
OWN long-run level, both legs read off sessions strictly before ``asof``.
The pick is the argmax, ties to the earlier clock.

Why relative, and not the raw cross-clock ratio (E1): the level of
implied-over-realized differs by clock for reasons that have nothing to do
with where the premium is -- a one-bar remaining window is a different object
from a six-bar one.  Dividing each clock by its own expanding level removes
that fixed effect and leaves the movement, which is what proposal 43 showed
migrating.  Proposal 46 found E2 the defensible causal choice; it tracks the
migration about a quarter late, and its rolling-126 cells are the two with
t > 2.  It does NOT beat fixed 11:00 with an interval excluding zero, and it
does not beat fixed 13:30: the morning/afternoon split is established, the
exact clock is inside per-contract noise.  The selector is therefore a
defensible rule, not a demonstrated improvement, and the live engine treats
it as such.

Expanding over expanding is identically 1 at every clock, so ``window`` must
be a real rolling window; the call refuses ``None``.

During the warm-up -- fewer than ``min_sessions`` usable sessions behind
every candidate clock -- there is no pick and the book stays flat, exactly as
proposals 43, 44 and 46 treat it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from math import isfinite

from .premium_ledger import MIN_SESSIONS, PremiumLedger

__all__ = ["CLOCKS", "DEFAULT_WINDOW", "pick_entry_clock", "score_entry_clocks"]

#: The entry clocks the selector may pick: 10:00 .. 15:00.  15:30 is on the
#: ledger (its remaining window is the last step into the settlement) but is
#: never an entry -- there is no session left to hedge.
CLOCKS: tuple[str, ...] = (
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
)

#: Proposal 46's rolling window on the numerator (126 is the other cell run).
DEFAULT_WINDOW = 252


def score_entry_clocks(
    ledger: PremiumLedger,
    asof: date,
    window: int = DEFAULT_WINDOW,
    min_sessions: int = MIN_SESSIONS,
    candidate_clocks: Sequence[str] = CLOCKS,
) -> dict[str, float]:
    """E2's score at every candidate clock, NaN where the clock is warming up."""
    if window is None:  # type: ignore[comparison-overlap]
        raise ValueError(
            "E2 needs a rolling window: the expanding ratio of sums over itself "
            "is identically 1.0 at every clock, so the argmax would be an "
            "artefact of the tie rule rather than a pick"
        )
    w = int(window)
    if w <= 0:
        raise ValueError(f"a rolling window must be positive, got {window!r}")
    if w < int(min_sessions):
        raise ValueError(
            f"the rolling window {w} is shorter than the warm-up "
            f"{int(min_sessions)}, so the numerator can never carry enough "
            f"usable sessions and the selector would stay flat for ever"
        )
    scores: dict[str, float] = {}
    for clock in candidate_clocks:
        recent = ledger.ratio_of_sums(clock, asof, w, min_sessions=min_sessions)
        level = ledger.ratio_of_sums(clock, asof, None, min_sessions=min_sessions)
        scores[str(clock)] = (
            recent / level
            if isfinite(recent) and isfinite(level) and level > 0.0
            else float("nan")
        )
    return scores


def pick_entry_clock(
    ledger: PremiumLedger,
    asof: date,
    window: int = DEFAULT_WINDOW,
    min_sessions: int = MIN_SESSIONS,
    candidate_clocks: Sequence[str] = CLOCKS,
) -> tuple[str | None, dict[str, float]]:
    """The entry clock for ``asof``, and E2's score at every candidate.

    Returns ``(None, scores)`` during the warm-up, when no candidate clock
    yet carries a finite score -- the book is flat that session.  Ties go to
    the earlier clock, which is what ``numpy.nanargmax`` does on the
    research's score matrix.
    """
    scores = score_entry_clocks(ledger, asof, window, min_sessions, candidate_clocks)
    best: str | None = None
    best_score = float("-inf")
    for clock in candidate_clocks:
        value = scores[str(clock)]
        if isfinite(value) and value > best_score:
            best, best_score = str(clock), value
    return best, scores
