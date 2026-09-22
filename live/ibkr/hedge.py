"""Futures hedge arithmetic for the short straddle.

The book is SHORT one 0DTE SPX straddle per lot, delta-hedged every 30
minutes from the 11:00 entry.  The short's hedge is the negative of the long
package's, so it holds ``+delta_pkg`` units of index per straddle: a positive
package delta means a LONG futures position.

Units.  One SPX option is on ``SPX_INDEX_MULTIPLIER`` dollars per index
point; one E-mini (ES) future is on ``ES_MULTIPLIER``, one Micro (MES) on
``MES_MULTIPLIER``.  So the continuous hedge of ``n`` straddles is
``delta_pkg * SPX_INDEX_MULTIPLIER * n / futures_multiplier`` contracts --
``2 * delta * n`` in ES, ``20 * delta * n`` in MES.

Granularity, stated once and correctly.  Against ``n`` straddles one ES is
``ES_MULTIPLIER / (SPX_INDEX_MULTIPLIER * n) = 0.5 / n`` straddle-deltas and
one MES is ``0.05 / n``, so rounding to a whole contract leaves at most half
of that -- ``0.25 / n`` on ES, ``0.025 / n`` on MES.  ES at ``n = 4`` is
therefore ``0.0625`` of residual delta, **2.5 times coarser** than MES at
``n = 1`` (``0.025``), not equal to it: parity with MES-at-one needs ES at
``n = 10``.  The fee half of the trade-off is the real one -- at ``n = 4``,
4 ES tickets cost about $3.40 against $10.00 for 40 MES.  ``target_lots``
below buys both: ES for the bulk, MES for the remainder.

The ladder.  ``ladder_lots`` is the same allocation for ANY ordered list of
futures, largest first: every rung but the last takes the whole contracts it
can toward zero, the last rung rounds the remainder.  ``target_lots`` is the
two-rung ES/MES case of it, bit for bit.  With the E-nano (NES, $0.50 a
point) a third rung exists, and the Mini-SPX book (XSP, one tenth of an SPX
straddle: ``index_scale`` 0.1) hedges in MES then NES.  The futures are all
on the S&P 500 index, so the dollars of delta per S&P point are
``delta_pkg * index_multiplier * index_scale * n``.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

__all__ = [
    "ES_MULTIPLIER",
    "MES_MULTIPLIER",
    "NES_MULTIPLIER",
    "SPX_INDEX_MULTIPLIER",
    "ladder_lots",
    "ladder_rebalance",
    "ladder_residual",
    "rebalance_lots",
    "rebalance_qty",
    "residual_delta",
    "residual_delta_lots",
    "target_futures",
    "target_lots",
]

# Dollars per index point.
SPX_INDEX_MULTIPLIER = 100.0
ES_MULTIPLIER = 50.0
MES_MULTIPLIER = 5.0
NES_MULTIPLIER = 0.5  # the E-nano S&P 500 (CME, from 2026-08-24)


def ladder_lots(
    delta_pkg: float,
    n_straddles: int,
    multipliers: Sequence[float],
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
    index_scale: float = 1.0,
) -> tuple[int, ...]:
    """Lots per rung against ``n_straddles`` short straddles, largest rung first.

    The package carries ``delta_pkg * index_multiplier * index_scale *
    n_straddles`` dollars of delta per S&P 500 point (``index_scale`` is the
    option's index per S&P point: 1 for SPX, 0.1 for XSP).  Every rung but the
    last takes the whole contracts it can TOWARD ZERO, so a big contract never
    overshoots; the last rung rounds what is left to the nearest contract.
    Signed: positive is LONG futures.  A non-finite delta hedges nothing.
    """
    mults = [float(m) for m in multipliers]
    if not mults:
        raise ValueError("the hedge ladder is empty")
    d = float(delta_pkg)
    if not isfinite(d):
        return tuple(0 for _ in mults)
    dollars = d * float(index_multiplier) * float(index_scale) * int(n_straddles)
    out: list[int] = []
    for m in mults[:-1]:
        lots = int(dollars / m)  # truncation toward zero
        out.append(lots)
        dollars = dollars - lots * m
    out.append(int(round(dollars / mults[-1])))
    return tuple(out)


def ladder_rebalance(target: Sequence[int], current: Sequence[int]) -> tuple[int, ...]:
    """Lots to trade on each rung to move from ``current`` to ``target``."""
    if len(target) != len(current):
        raise ValueError("ladder lengths differ")
    return tuple(int(t) - int(c) for t, c in zip(target, current))


def ladder_residual(
    delta_pkg: float,
    n_straddles: int,
    lots: Sequence[int],
    multipliers: Sequence[float],
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
    index_scale: float = 1.0,
) -> float:
    """Unhedged package delta left by ``lots`` on the ladder, per straddle.

    In the index-delta units of ONE straddle (of the option's own index), so
    it is directly comparable with ``package_delta``; positive means the
    futures legs are short of the delta the straddles carry.
    """
    if len(lots) != len(multipliers):
        raise ValueError("ladder lengths differ")
    dollars = sum(int(q) * float(m) for q, m in zip(lots, multipliers))
    hedged = dollars / (int(n_straddles) * float(index_multiplier) * float(index_scale))
    return float(delta_pkg) - hedged


def target_futures(
    delta_pkg: float,
    n_straddles: int,
    futures_multiplier: float = ES_MULTIPLIER,
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
) -> int:
    """Futures contracts to hold against ``n_straddles`` short straddles.

    Positive is LONG futures (the short straddle's hedge when the package
    delta is positive).  Rounded to a whole contract with Python's
    round-half-to-even; an exact half-contract target is a measure-zero tie
    and either side of it is one contract of residual delta.
    """
    d = float(delta_pkg)
    if not isfinite(d):
        return 0
    return int(
        round(
            d * float(index_multiplier) * int(n_straddles) / float(futures_multiplier)
        )
    )


def rebalance_qty(target: int, current: int) -> int:
    """Contracts to trade to move the futures position from ``current`` to ``target``."""
    return int(target) - int(current)


def residual_delta(
    delta_pkg: float,
    n_straddles: int,
    futures_pos: int,
    futures_multiplier: float = ES_MULTIPLIER,
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
) -> float:
    """Unhedged package delta left after ``futures_pos``, per straddle.

    In the index-delta units of ONE straddle, so it is directly comparable
    with ``package_delta``: zero is perfectly hedged, positive means the
    futures leg is short of the delta the straddle carries.
    """
    hedged = (
        int(futures_pos)
        * float(futures_multiplier)
        / (int(n_straddles) * float(index_multiplier))
    )
    return float(delta_pkg) - hedged


def target_lots(
    delta_pkg: float,
    n_straddles: int,
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
    es_multiplier: float = ES_MULTIPLIER,
    mes_multiplier: float = MES_MULTIPLIER,
) -> tuple[int, int]:
    """``(es, mes)`` lots to hold against ``n_straddles`` short straddles.

    The package carries ``delta_pkg * index_multiplier * n_straddles`` dollars
    of index delta per point.  ES takes the bulk -- the integer part TOWARD
    ZERO of that divided by ``es_multiplier``, so the big contract never
    overshoots -- and MES takes the remainder, rounded to the nearest whole
    contract.  Both are signed and positive means LONG futures, the short
    straddle's hedge when the package delta is positive.

    Residual delta after the pair is at most half an MES, ``0.025 / n``
    straddle-deltas; see the module docstring for why that is the granularity
    argument and 'MES up to about four straddles, then ES' was not.

    A non-finite delta hedges nothing: ``(0, 0)``, and the caller's residual
    alarm sees the whole package delta unhedged rather than a silent zero.
    """
    es, mes = ladder_lots(
        delta_pkg,
        n_straddles,
        (float(es_multiplier), float(mes_multiplier)),
        index_multiplier=index_multiplier,
    )
    return es, mes


def rebalance_lots(
    target: tuple[int, int], current: tuple[int, int]
) -> tuple[int, int]:
    """``(es, mes)`` to trade to move from ``current`` lots to ``target``."""
    return int(target[0]) - int(current[0]), int(target[1]) - int(current[1])


def residual_delta_lots(
    delta_pkg: float,
    n_straddles: int,
    es: int,
    mes: int,
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
    es_multiplier: float = ES_MULTIPLIER,
    mes_multiplier: float = MES_MULTIPLIER,
) -> float:
    """Unhedged package delta left after ``es`` ES and ``mes`` MES, per straddle.

    In the index-delta units of ONE straddle, directly comparable with
    ``package_delta``: zero is perfectly hedged, positive means the futures
    legs are short of the delta the straddle carries.
    """
    hedged = (int(es) * float(es_multiplier) + int(mes) * float(mes_multiplier)) / (
        int(n_straddles) * float(index_multiplier)
    )
    return float(delta_pkg) - hedged
