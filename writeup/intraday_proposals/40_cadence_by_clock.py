"""40 - the hedge cadence by the clock: is a sparser middle of the day better?

The book of record is proposal 32's: every session sells the nearest-OTM SPX
0DTE straddle at 11:00 ET, delta-hedges the package on the vendor spot at
every 30-minute stamp, and buys the straddle back at 15:30 at the QUOTED ask
with the entry taken at the quoted bid (fully crossed).  Returns are one unit
a day in units of the 11:00 midpoint entry premium.  The secondary book holds
the straddle through the 16:00 cash settlement and hedges the 15:30-16:00
stretch as well (``parity.replay_day`` with ``exit_clock=None``).

Proposal 39 asked whether hedging MORE often helps and answered no: the
within-bar path mean-reverts, so the 1-minute bound's gross Sharpe sits below
the 30-minute book's and proposal 29's 60-minute cadence halves the gap.  On
the 30-minute tape the only remaining direction is SPARSER, and proposal 32's
per-bar attribution says the day is not flat: the convexity bill is U-shaped
by clock (worst 11:00-11:30 and 14:00-14:30, mildest 12:30-13:00) while the
realised per-bar P&L is front-loaded (11:30 best, 13:00 dead).  This proposal
asks whether that shape is worth trading: rebalance sparsely in the quiet
middle and densely at the edges.

THE MECHANICS.  A rule names the stamps among 11:30, 12:00, ..., 15:00 at
which the book rebalances.  The entry hedge at 11:00 is always run; a SKIPPED
stamp is one at which no trade happens, so the delta set at the last
rebalance is HELD across it.  The exit book flattens at 15:30 (never a
rebalance stamp: the futures leg is closed there), and the hold book's ladder
runs to 15:30 -- that stamp is a rebalance for the hold book under every
rule, because the hold book still carries the 15:30-16:00 bar -- and flattens
at the settlement.  Nothing else about either book changes: the option leg,
the entry and the buy-back price are the book's own.

THE RULES (skips are the stamps at which no trade happens):

    R0  every stamp                         the book of record
    R1  skip 12:30                          hold 12:00's delta to 13:00
    R2  skip 12:30, 13:00                    hold 12:00's delta to 13:30
    R3  skip 12:00, 13:00, 14:00             hourly through the middle
    R4  skip 11:30                           CONTROL: sparse at the morning edge
    R5  skip 14:30                           CONTROL: sparse at the afternoon edge
    R6  rebalance only 11:00, 12:00, 13:30, 15:00   the diurnal-U schedule
    R7  rebalance only 11:00, 12:00, 13:00, 14:00, 15:00   hourly (proposal 29)

WHAT A SKIP DOES, EXACTLY.  Over bar ``j`` (stamp ``j`` to stamp ``j + 1``)
proposal 32 splits the package's frozen-volatility value change into the
hedge leg and the convexity term,

    hedge_j    = D_j (S_{j+1} - S_j)
    convex_j   = V(s_j, S_{j+1}) - V(s_j, S_j) - hedge_j

and the SHORT books the pair as ``convexity_pts_j = hedge_j - dV_frozen_j``,
which is proposal 32's published ``convexity_pts`` column.  A rule replaces
``D_j`` by the delta ``D_src(j)`` set at the last rebalance stamp, so
``dV_frozen_j`` -- a property of the tape alone -- is UNCHANGED and the whole
effect of the skip on that bar is

    change_j = (D_src(j) - D_j) (S_{j+1} - S_j),

which is both the change in the bar's convexity+hedge pair and, summed over
the affected bars and divided by the entry premium, the change in the day's
gross return.  The script asserts that identity to 1e-12 for both books.
Only bars whose STARTING stamp is skipped are affected; the hold book's
15:30-16:00 bar never is.

Costs are the repo's: ``bp x S x |delta traded|``, charged at 0.2 / 0.5 / 1.0
basis points, each rebalance charged at its own stamp's spot and the entry
and the flatten at theirs (proposal 32's convention, reproduced in proposal
39).  The verdict is proposal 39's bootstrap: 2000 circular block draws,
block 21, seed 0, the 2.5-97.5 percentile interval of the paired Sharpe
difference against R0.

GATE, all before anything new is computed: n = 865 days; the exit book's
crossed-quoted Sharpe 2.2525467 whole / 2.5131322 era; the hold book's
crossed Sharpe 3.582391; both books' R0 hedge legs against the live engine's
own ``parity.replay_day`` to 1e-12; R0's turnover against proposal 39's
2.055879 whole / 2.124129 era delta units a day; proposal 32's trend/choppy
tercile cut points 0.2076 / 0.4782; and the R0 convexity+hedge grid against
proposal 32's published ``convexity_pts`` to 1e-12.

Run:  python writeup/intraday_proposals/40_cadence_by_clock.py
"""

from __future__ import annotations

import importlib.util
import sys
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
p39 = _load_module(HERE / "39_hedge_density.py", "p40_p39")
p35 = p39.p35
p32 = p39.p32

from live.ibkr.pricing import package_price  # noqa: E402

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "40"

SESSION: tuple[str, ...] = p32.SESSION
H_REM: tuple[float, ...] = p32.H_REM
ERA0 = p32.ERA0

#: Bars of the primary (exit 15:30) book and of the hold book.
N_BAR_EXIT = len(SESSION) - 1
N_BAR_HOLD = N_BAR_EXIT + 1
#: The stamps a rule may skip: everything strictly between the 11:00 entry and
#: the 15:30 exit.  11:00 is always hedged; 15:30 is the exit book's flatten
#: and the hold book's last rebalance, under every rule.
SKIPPABLE: tuple[str, ...] = SESSION[1:-1]
LAST_STAMP = SESSION[-1]

#: The eight schedules, as the stamps at which NO trade happens.  R0 is the
#: book of record; R4 and R5 are the edge controls; R7 is proposal 29's
#: 60-minute reference.
RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("R0", "every stamp (the book of record)", ()),
    ("R1", "skip 12:30", ("12:30",)),
    ("R2", "skip 12:30 and 13:00", ("12:30", "13:00")),
    (
        "R3",
        "skip 12:00, 13:00, 14:00 (hourly through the middle)",
        ("12:00", "13:00", "14:00"),
    ),
    ("R4", "CONTROL skip 11:30 (sparse at the morning edge)", ("11:30",)),
    ("R5", "CONTROL skip 14:30 (sparse at the afternoon edge)", ("14:30",)),
    (
        "R6",
        "rebalance only 11:00, 12:00, 13:30, 15:00 (diurnal-U)",
        ("11:30", "12:30", "13:00", "14:00", "14:30"),
    ),
    (
        "R7",
        "hourly 11:00, 12:00, 13:00, 14:00, 15:00 (proposal 29's 60 minutes)",
        ("11:30", "12:30", "13:30", "14:30"),
    ),
)
BOOK_RULE = RULES[0][0]
#: The rules the multiple-comparison statement counts (everything but R0).
N_TRIED = len(RULES) - 1

#: Turnover charges, basis points of S x |delta traded| (proposal 39's grid).
COST_BP_GRID: tuple[float, ...] = p39.COST_BP_GRID
#: The cost level the headline net numbers and the bootstrap are read at.
HEADLINE_BP = p39.HEADLINE_BP

#: The deck's circular block bootstrap: 2000 draws, 21-day blocks, seed 0.
BOOT_B = p39.BOOT_B
BOOT_BLOCK = p39.BOOT_BLOCK
BOOT_SEED = p39.BOOT_SEED
CI_PCT: tuple[float, float] = p39.CI_PCT
#: The interval's nominal size, for the multiple-comparison statement.
CI_ALPHA = (CI_PCT[0] + (100.0 - CI_PCT[1])) / 100.0

#: Published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = p32.GATE_N_DAYS
GATE_CROSSED_WHOLE = p32.GATE_CROSSED_WHOLE
GATE_CROSSED_ERA = p32.GATE_CROSSED_ERA
GATE_HOLD_CROSSED = 3.582391
GATE_TOL = p32.GATE_TOL
#: Proposal 39's own 30-minute turnover of the primary book, delta units a day.
GATE_TURNOVER_WHOLE = 2.055879
GATE_TURNOVER_ERA = 2.124129
GATE_TURNOVER_TOL = 1e-5
#: Proposal 32's full-sample trend/choppy tercile cut points, as published.
GATE_TREND_LO = 0.2076
GATE_TREND_HI = 0.4782
TREND_TOL = 5e-5
#: Tape identities are float paths, not bars.
EXACT_TOL = 1e-12


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 90):
        print(use.to_string(index=False))


def write(
    df: pd.DataFrame, name: str, title: str, cols: list[str] | None = None
) -> None:
    """Persist a table under the proposal's own directory and print it."""
    df.to_csv(OUT / name, index=False)
    show(df, f"{title}   [{name}]", cols)


# --------------------------------------------------------- the schedules ----
def rebalance_stamps(skip: tuple[str, ...], hold_last: bool) -> list[int]:
    """Stamp indices at which the rule trades, in order, entry first.

    ``hold_last`` adds the 15:30 stamp, which the hold book rebalances at
    under every rule (it still carries the 15:30-16:00 bar) and the exit book
    never does (it flattens there).
    """
    reb = [0] + [j for j, c in enumerate(SESSION) if c in SKIPPABLE and c not in skip]
    if hold_last:
        reb.append(len(SESSION) - 1)
    return reb


def frozen_value_change(tape: dict[str, Any]) -> np.ndarray:
    """``dV_frozen[i, j] = V(s_j, S_{j+1}) - V(s_j, S_j)``, in index points.

    The package's Black-76 value change over the bar at the stamp's OWN total
    volatility -- proposal 32's ``v_move - v_now`` -- which is a property of
    the tape and of the 11:00 strikes alone and so is the same under every
    rebalance schedule.  It is rebuilt here from the live engine rather than
    read off, so the gate's reproduction of proposal 32's published
    ``convexity_pts`` is a real one.
    """
    spot = tape["spot"]
    stot = tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])
    kc = tape["Kc"]
    kp = tape["Kp"]
    n = spot.shape[0]
    out = np.zeros((n, N_BAR_EXIT))
    for i in range(n):
        for j in range(N_BAR_EXIT):
            s_j = stot[i, j]
            out[i, j] = package_price(
                s_j, spot[i, j + 1], kc[i], kp[i]
            ) - package_price(s_j, spot[i, j], kc[i], kp[i])
    return out


def held_source(reb: list[int], n_bars: int) -> np.ndarray:
    """For each bar, the stamp index whose delta the book holds over it."""
    src = np.zeros(n_bars, dtype=int)
    cur = 0
    reb_set = set(reb)
    for j in range(n_bars):
        if j in reb_set:
            cur = j
        src[j] = cur
    return src


# ------------------------------------------------------------- the books ----
def exit_spec(g: dict[str, Any], delta: np.ndarray) -> dict[str, Any]:
    """The primary book: entry 11:00 quoted bid, buy-back 15:30 quoted ask."""
    spot = g["spot"]
    return {
        "book": "exit_1530_crossed",
        "spot": spot,
        "delta": delta,
        "dS": spot[:, 1:] - spot[:, :-1],
        "n_bars": N_BAR_EXIT,
        "opt_leg": -(g["ask"][:, -1] - g["eb"]),
        "em": g["em"],
        "spot_flat": spot[:, -1],
        "hedge_ref": g["tape"]["hedge_ref"],
        "hold_last": False,
    }


def hold_spec(g: dict[str, Any], delta: np.ndarray) -> dict[str, Any]:
    """The secondary book: the straddle held through the 16:00 cash settlement."""
    spot = g["spot"]
    s_close = g["tape"]["S_close"]
    return {
        "book": "hold_settlement_crossed",
        "spot": spot,
        "delta": delta,
        "dS": np.concatenate(
            [spot[:, 1:] - spot[:, :-1], (s_close - spot[:, -1])[:, None]], axis=1
        ),
        "n_bars": N_BAR_HOLD,
        "opt_leg": -(g["tape"]["settle"] - g["eb"]),
        "em": g["em"],
        "spot_flat": s_close,
        "hedge_ref": g["tape"]["hedge_hold"],
        "hold_last": True,
    }


def rule_series(sp: dict[str, Any], skip: tuple[str, ...]) -> dict[str, Any]:
    """Day return, hedge turnover and traded notional under one schedule.

    The delta held over bar ``j`` is the one set at the last rebalance stamp
    at or before ``j``; the hedge leg is the sum of those deltas against the
    bar moves.  The turnover is the entry hedge, one trade at each rebalance
    stamp (the change from the delta the book was carrying), and the flatten.
    Each trade is charged at its own stamp's spot, which is proposal 32's
    convention and proposal 39's.
    """
    delta = sp["delta"]
    spot = sp["spot"]
    reb = rebalance_stamps(skip, sp["hold_last"])
    src = held_source(reb, sp["n_bars"])
    held = delta[:, src]
    hedge = (held * sp["dS"]).sum(axis=1)
    turn = np.abs(delta[:, 0])
    notional = turn * spot[:, 0]
    for a, b in zip(reb[:-1], reb[1:], strict=True):
        traded = np.abs(delta[:, b] - delta[:, a])
        turn = turn + traded
        notional = notional + traded * spot[:, b]
    pos = np.abs(delta[:, reb[-1]])
    return {
        "ret": (sp["opt_leg"] + hedge) / sp["em"],
        "hedge_pts": hedge,
        "held": held,
        "turnover": turn + pos,
        "notional": notional + pos * sp["spot_flat"],
        "reb": reb,
        "src": src,
        "n_trades": len(reb) + 1,
    }


# -------------------------------------------------------------- the gate ----
def gate(
    g: dict[str, Any],
    specs: list[dict[str, Any]],
    base: dict[str, dict[str, Any]],
    conv: np.ndarray,
    day: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, float, float]:
    """Reproduce every published number; nothing new runs until this passes."""
    era = np.asarray(dates >= ERA0, dtype=bool)
    print("\nGATE - published reference numbers")
    assert len(dates) == GATE_N_DAYS, len(dates)
    print(f"  n whole sample {len(dates)}  reference {GATE_N_DAYS}  OK")
    print(f"  daily-0DTE era ({ERA0.date()} onward): {int(era.sum())} days")

    for sp in specs:
        b = base[sp["book"]]
        d_h = float(np.max(np.abs(b["hedge_pts"] - sp["hedge_ref"])))
        assert d_h < EXACT_TOL, (sp["book"], d_h)
        print(
            f"  {sp['book']:<24s} R0 hedge leg = sum delta_src(j) dS_j over "
            f"{sp['n_bars']} bars, max |difference| against replay_day "
            f"{d_h:.3e}  OK"
        )

    r_x = pd.Series(base["exit_1530_crossed"]["ret"], index=dates)
    r_h = pd.Series(base["hold_settlement_crossed"]["ret"], index=dates)
    for label, have, want in (
        ("exit crossed Sharpe_ann whole", p32.sharpe_ann(r_x), GATE_CROSSED_WHOLE),
        ("exit crossed Sharpe_ann era", p32.sharpe_ann(r_x[era]), GATE_CROSSED_ERA),
        ("hold crossed Sharpe_ann whole", p32.sharpe_ann(r_h), GATE_HOLD_CROSSED),
    ):
        assert abs(have - want) < GATE_TOL, (label, have, want)
        print(f"  {label:<32s} {have:>16.12f}  reference {want:.12f}  OK")

    t_all = base["exit_1530_crossed"]["turnover"]
    for label, have, want in (
        ("R0 turnover whole", float(t_all.mean()), GATE_TURNOVER_WHOLE),
        ("R0 turnover era", float(t_all[era].mean()), GATE_TURNOVER_ERA),
    ):
        assert abs(have - want) < GATE_TURNOVER_TOL, (label, have, want)
        print(
            f"  {label:<32s} {have:>16.9f}  reference {want:.6f} delta units a "
            f"day (proposal 39)  OK"
        )

    d_c = float(np.max(np.abs(conv - g["conv_ref"])))
    assert d_c < EXACT_TOL, d_c
    print(
        f"  the R0 convexity+hedge pair (hedge_j - dV_frozen_j) reproduces "
        f"proposal 32's published convexity_pts on all {conv.size} day-bar "
        f"cells to {d_c:.3e}  OK"
    )

    q1, q2 = day["trend_ratio"].quantile([1 / 3, 2 / 3]).to_numpy()
    assert abs(float(q1) - GATE_TREND_LO) < TREND_TOL, q1
    assert abs(float(q2) - GATE_TREND_HI) < TREND_TOL, q2
    lab = np.full(len(dates), "2_mixed", dtype=object)
    tr = day["trend_ratio"].to_numpy(float)
    lab[tr <= q1] = "1_choppy"
    lab[tr > q2] = "3_trend"
    print(
        f"  proposal 32's shape terciles of |sum of bar returns| / sum |bar "
        f"returns|: cut points {float(q1):.6f} / {float(q2):.6f}  reference "
        f"{GATE_TREND_LO:.4f} / {GATE_TREND_HI:.4f}  OK; "
        f"{int((lab == '1_choppy').sum())} choppy, "
        f"{int((lab == '2_mixed').sum())} mixed, "
        f"{int((lab == '3_trend').sum())} trend days"
    )
    print("GATE PASSED")
    return lab, float(q1), float(q2)


# -------------------------------------------------------------- the rows ----
def rule_row(
    sp: dict[str, Any],
    tag: str,
    name: str,
    skip: tuple[str, ...],
    out: dict[str, Any],
    sample: str,
    mask: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """One (book, sample, rule) line: the deck's statistics plus the costs."""
    ret = out["ret"][mask]
    em = sp["em"][mask]
    notional = out["notional"][mask]
    row = p32.stats(pd.Series(ret, index=dates[mask]), sp["book"], sample)
    row["rule"] = tag
    row["name"] = name
    row["n_skipped_stamps"] = len(skip)
    row["skipped"] = "|".join(skip) if skip else "none"
    row["rebalance_stamps"] = "|".join(SESSION[j] for j in out["reb"])
    row["n_trades"] = out["n_trades"]
    row["turnover"] = float(out["turnover"][mask].mean())
    row["notional_pts"] = float(notional.mean())
    p39.cost_columns(row, ret, notional, em)
    net = ret - HEADLINE_BP * 1e-4 * notional / em
    s_net = pd.Series(net, index=dates[mask])
    row["t_net_headline"] = (
        sqrt(s_net.size) * float(s_net.mean()) / float(s_net.std(ddof=1))
    )
    row["MaxDD_net_headline"] = p32.maxdd(s_net)
    row["worst_day_net_headline"] = float(s_net.min())
    return row


def skipped_bar_rows(
    sp: dict[str, Any],
    tag: str,
    name: str,
    skip: tuple[str, ...],
    out: dict[str, Any],
    base: dict[str, Any],
    conv: np.ndarray,
    dv_frozen: np.ndarray,
    sample: str,
    mask: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """What the skipped rebalances cost or saved on the bars they touch.

    The affected bars are exactly those whose STARTING stamp is skipped; on
    each the convexity+hedge pair moves by ``(D_src(j) - D_j) dS_j`` and
    ``dV_frozen_j`` does not move at all, so the sum over them is the whole
    change in the day's gross return.  Both are reported: the per-day sum in
    units of the entry premium (with its t), and the per-bar mean in index
    points next to the book's own level on the same bars.
    """
    delta = sp["delta"]
    d_s = sp["dS"]
    nb = N_BAR_EXIT
    change = (out["held"][:, :nb] - delta[:, :nb]) * d_s[:, :nb]
    conv_rule = conv + change
    bars = [j for j in range(nb) if SESSION[j] in skip]
    em = sp["em"][mask]
    per_day = (
        change[mask][:, bars].sum(axis=1) / em if bars else np.zeros(int(mask.sum()))
    )
    n = int(per_day.size)
    sd = float(per_day.std(ddof=1)) if n >= 2 else float("nan")
    ok = bool(np.isfinite(sd)) and sd > 0.0
    head: dict[str, Any] = {
        "book": sp["book"],
        "sample": sample,
        "rule": tag,
        "name": name,
        "skipped": "|".join(skip) if skip else "none",
        "n_days": n,
        "n_skipped_bars_per_day": len(bars),
        "mean_change_prem_per_day": float(per_day.mean()),
        "sd_change_prem_per_day": sd,
        "t_change": sqrt(n) * float(per_day.mean()) / sd if ok else float("nan"),
        "mean_change_pts_per_bar": (
            float(change[mask][:, bars].mean()) if bars else 0.0
        ),
        "mean_book_pair_pts_per_bar": (
            float(conv[mask][:, bars].mean()) if bars else float("nan")
        ),
        "mean_rule_pair_pts_per_bar": (
            float(conv_rule[mask][:, bars].mean()) if bars else float("nan")
        ),
    }
    rows: list[dict[str, Any]] = []
    for j in bars:
        c = change[mask][:, j]
        cp = c / em
        sd_j = float(cp.std(ddof=1))
        rows.append(
            {
                "book": sp["book"],
                "sample": sample,
                "rule": tag,
                "skipped_stamp": SESSION[j],
                "bar": f"{SESSION[j]}-{SESSION[j + 1]}",
                "n_days": n,
                "mean_change_pts": float(c.mean()),
                "mean_change_prem": float(cp.mean()),
                "t_change": (
                    sqrt(n) * float(cp.mean()) / sd_j if sd_j > 0.0 else float("nan")
                ),
                "mean_book_pair_pts": float(conv[mask][:, j].mean()),
                "mean_rule_pair_pts": float(conv_rule[mask][:, j].mean()),
                "mean_dV_frozen_pts": float(dv_frozen[mask][:, j].mean()),
            }
        )
    # the identity: the change in the day's gross return IS that sum
    d_id = float(
        np.max(
            np.abs((out["ret"] - base["ret"])[mask] - (change[mask].sum(axis=1) / em))
        )
    )
    assert d_id < EXACT_TOL, (sp["book"], tag, sample, d_id)
    head["identity_max_abs_diff"] = d_id
    return head, rows


# ------------------------------------------------------------ statistics ----
def boot_row(
    book: str, sample: str, tag: str, leg: str, a: np.ndarray, b: np.ndarray
) -> dict[str, Any]:
    """Proposal 39's paired circular block bootstrap of Sharpe(rule) - Sharpe(R0)."""
    raw = p39.boot_sharpe_diff(a, b)
    return {
        "book": book,
        "sample": sample,
        "rule": tag,
        "leg": leg,
        "n_days": raw["n_days"],
        "Sharpe_rule": raw["Sharpe_dense"],
        "Sharpe_R0": raw["Sharpe_30"],
        "diff": raw["diff"],
        "ci_lo": raw["ci_lo"],
        "ci_hi": raw["ci_hi"],
        "excludes_zero": raw["excludes_zero"],
        "B": raw["B"],
        "block": raw["block"],
        "seed": raw["seed"],
    }


def tercile_rows(
    sp: dict[str, Any],
    tag: str,
    out: dict[str, Any],
    lab: np.ndarray,
    sample: str,
    mask: np.ndarray,
    dates: pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    """Proposal 32's shape terciles, per rule, on the primary book."""
    rows: list[dict[str, Any]] = []
    for cls in ("1_choppy", "2_mixed", "3_trend"):
        m = mask & (lab == cls)
        ret = out["ret"][m]
        em = sp["em"][m]
        notional = out["notional"][m]
        net = ret - HEADLINE_BP * 1e-4 * notional / em
        s = pd.Series(ret, index=dates[m])
        rows.append(
            {
                "book": sp["book"],
                "sample": sample,
                "rule": tag,
                "tercile": cls,
                "n": int(m.sum()),
                "mean_gross": float(s.mean()),
                "sd_gross": float(s.std(ddof=1)),
                "Sharpe_gross": p32.sharpe_ann(s),
                "t_gross": sqrt(int(m.sum())) * float(s.mean()) / float(s.std(ddof=1)),
                "turnover": float(out["turnover"][m].mean()),
                f"mean_net_{HEADLINE_BP:g}bp": float(net.mean()),
                f"Sharpe_net_{HEADLINE_BP:g}bp": p32.sharpe_ann(net),
            }
        )
    return rows


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------ gates first ----
    g = p35.book_grids()
    dates = g["dates"]
    delta, gamma = p39.stamp_greeks(g["tape"])
    d_gam = float(np.max(np.abs(gamma - g["gamma"])))
    assert d_gam < 1e-15, d_gam
    attribution, day = p32.attribute(g["tape"], g["mid"], g["ask"], dates)
    att = attribution.copy()
    att["date"] = pd.to_datetime(att["date"])
    g["conv_ref"] = (
        att.pivot_table(
            index="date", columns="clock", values="convexity_pts", aggfunc="first"
        )
        .reindex(index=dates, columns=list(SESSION[:-1]))
        .to_numpy(float)
    )
    assert np.isfinite(g["conv_ref"]).all(), "a bar is missing its convexity"

    specs = [exit_spec(g, delta), hold_spec(g, delta)]
    base = {sp["book"]: rule_series(sp, ()) for sp in specs}
    sp_exit = specs[0]
    d_s = sp_exit["dS"]
    # The pair the rules move is hedge_j - dV_frozen_j, which IS proposal 32's
    # published convexity_pts.  dV_frozen is rebuilt from the live engine, so
    # the gate's reproduction of that column is a real one, and it does not
    # move under any rule.
    dv_frozen = frozen_value_change(g["tape"])
    conv = delta[:, :N_BAR_EXIT] * d_s[:, :N_BAR_EXIT] - dv_frozen

    lab, q1, q2 = gate(g, specs, base, conv, day, dates)
    print(
        f"  the package gamma written out by proposal 39 matches proposal 35's "
        f"to {d_gam:.3e} on all {gamma.size} stamp-cells  OK"
    )

    era = np.asarray(dates >= ERA0, dtype=bool)
    samples: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )

    # --------------------------------------------------- the schedules ----
    sched: list[dict[str, Any]] = []
    for tag, name, skip in RULES:
        reb_x = rebalance_stamps(skip, False)
        reb_h = rebalance_stamps(skip, True)
        sched.append(
            {
                "rule": tag,
                "name": name,
                "skipped": "|".join(skip) if skip else "none",
                "n_skipped": len(skip),
                "exit_rebalance_stamps": "|".join(SESSION[j] for j in reb_x),
                "exit_n_rebalances": len(reb_x) - 1,
                "exit_n_trades": len(reb_x) + 1,
                "hold_rebalance_stamps": "|".join(SESSION[j] for j in reb_h),
                "hold_n_rebalances": len(reb_h) - 1,
                "hold_n_trades": len(reb_h) + 1,
                "exit_held_delta_by_bar": "|".join(
                    SESSION[j] for j in held_source(reb_x, N_BAR_EXIT)
                ),
            }
        )
    write(
        pd.DataFrame(sched),
        "schedules.csv",
        "The eight schedules.  A trade count is the entry hedge, one trade per "
        f"rebalance stamp after it, and the flatten ({LAST_STAMP} for the exit "
        "book, the settlement for the hold book)",
    )

    # ------------------------------------------------------- the rules ----
    series: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    terc: list[dict[str, Any]] = []
    for sp in specs:
        for tag, name, skip in RULES:
            out = rule_series(sp, skip)
            series[(sp["book"], tag)] = out
            for smp, m in samples:
                rows.append(rule_row(sp, tag, name, skip, out, smp, m, dates))
                h, br = skipped_bar_rows(
                    sp,
                    tag,
                    name,
                    skip,
                    out,
                    base[sp["book"]],
                    conv,
                    dv_frozen,
                    smp,
                    m,
                )
                head_rows.append(h)
                bar_rows.extend(br)
                if sp is sp_exit:
                    terc.extend(tercile_rows(sp, tag, out, lab, smp, m, dates))

    rule_cols = [
        "book",
        "sample",
        "rule",
        "skipped",
        "n_trades",
        "n",
        "mean",
        "sd",
        "Sharpe_ann",
        "t",
        "MaxDD",
        "worst_day",
        "worst_date",
        "turnover",
        *[f"Sharpe_net_{bp:g}bp" for bp in COST_BP_GRID],
        *[f"cost_prem_{bp:g}bp" for bp in COST_BP_GRID],
    ]
    rules_df = pd.DataFrame(rows)
    rules_df.to_csv(OUT / "rules.csv", index=False)
    for sp in specs:
        for smp, _m in samples:
            sub = rules_df[
                (rules_df["book"] == sp["book"]) & (rules_df["sample"] == smp)
            ]
            show(
                sub,
                f"Per-rule book statistics  |  book {sp['book']}  |  sample "
                f"{smp}   [rules.csv]",
                rule_cols,
            )
    show(
        rules_df,
        "Per-rule net drawdown and worst day at the headline "
        f"{HEADLINE_BP:g} bp charge   [rules.csv]",
        [
            "book",
            "sample",
            "rule",
            "turnover",
            f"mean_net_{HEADLINE_BP:g}bp",
            f"Sharpe_net_{HEADLINE_BP:g}bp",
            "t_net_headline",
            "MaxDD_net_headline",
            "worst_day_net_headline",
        ],
    )

    # ------------------------------------------ what the skips did do ----
    head_df = pd.DataFrame(head_rows)
    write(
        head_df,
        "skipped_bars.csv",
        "The change in the convexity+hedge pair on the bars the rule's skips "
        "touch, per day in units of the entry premium (this IS the change in "
        "the day's gross return: dV_frozen does not move)",
        [
            "book",
            "sample",
            "rule",
            "skipped",
            "n_days",
            "n_skipped_bars_per_day",
            "mean_change_prem_per_day",
            "sd_change_prem_per_day",
            "t_change",
            "mean_change_pts_per_bar",
            "mean_book_pair_pts_per_bar",
            "mean_rule_pair_pts_per_bar",
            "identity_max_abs_diff",
        ],
    )
    write(
        pd.DataFrame(bar_rows),
        "skipped_bars_by_clock.csv",
        "The same change, bar by bar: what holding the previous delta across "
        "each skipped stamp did on that bar alone",
    )

    # ------------------------------------------------- the bootstrap ----
    boots: list[dict[str, Any]] = []
    for sp in specs:
        b0 = base[sp["book"]]
        for tag, _name, _skip in RULES:
            if tag == BOOK_RULE:
                continue
            out = series[(sp["book"], tag)]
            for smp, m in samples:
                em = sp["em"][m]
                c0 = HEADLINE_BP * 1e-4 * b0["notional"][m] / em
                c1 = HEADLINE_BP * 1e-4 * out["notional"][m] / em
                boots.append(
                    boot_row(sp["book"], smp, tag, "gross", out["ret"][m], b0["ret"][m])
                )
                boots.append(
                    boot_row(
                        sp["book"],
                        smp,
                        tag,
                        f"net {HEADLINE_BP:g} bp",
                        out["ret"][m] - c1,
                        b0["ret"][m] - c0,
                    )
                )
    boot_df = pd.DataFrame(boots)
    boot_df.to_csv(OUT / "bootstrap.csv", index=False)
    for sp in specs:
        for smp, _m in samples:
            sub = boot_df[(boot_df["book"] == sp["book"]) & (boot_df["sample"] == smp)]
            show(
                sub,
                f"Sharpe(rule) - Sharpe(R0), {BOOT_B} circular block draws, block "
                f"{BOOT_BLOCK}, seed {BOOT_SEED}, {CI_PCT[0]}-{CI_PCT[1]} "
                f"percentile interval  |  book {sp['book']}  |  sample {smp}"
                f"   [bootstrap.csv]",
            )

    # ---------------------------------------------------- the terciles ----
    write(
        pd.DataFrame(terc),
        "terciles.csv",
        f"Proposal 32's shape terciles of |sum of bar returns| / sum |bar "
        f"returns| (full-sample cut {q1:.4f} / {q2:.4f}), per rule, primary "
        f"book",
    )

    # ------------------------------------------------------ the verdict ----
    prim = sp_exit["book"]
    ver: list[dict[str, Any]] = []
    for tag, name, skip in RULES:
        if tag == BOOK_RULE:
            continue
        hit = boot_df[
            (boot_df["book"] == prim)
            & (boot_df["sample"] == "daily_era")
            & (boot_df["rule"] == tag)
        ].set_index("leg")
        gr = hit.loc["gross"]
        nt = hit.loc[f"net {HEADLINE_BP:g} bp"]
        both = bool(gr["excludes_zero"]) and bool(nt["excludes_zero"])
        if not both:
            verdict = "does NOT improve (the gate is not met)"
        elif gr["diff"] > 0.0 and nt["diff"] > 0.0:
            verdict = (
                "IMPROVES (both intervals exclude zero, both differences positive)"
            )
        elif gr["diff"] < 0.0 and nt["diff"] < 0.0:
            verdict = (
                "DEGRADES (both intervals exclude zero, both differences negative)"
            )
        else:
            verdict = "does NOT improve (both intervals exclude zero, signs disagree)"
        ver.append(
            {
                "rule": tag,
                "name": name,
                "skipped": "|".join(skip) if skip else "none",
                "book": prim,
                "sample": "daily_era",
                "Sharpe_R0_gross": gr["Sharpe_R0"],
                "Sharpe_rule_gross": gr["Sharpe_rule"],
                "diff_gross": gr["diff"],
                "ci_lo_gross": gr["ci_lo"],
                "ci_hi_gross": gr["ci_hi"],
                "gross_excludes_zero": bool(gr["excludes_zero"]),
                "Sharpe_R0_net": nt["Sharpe_R0"],
                "Sharpe_rule_net": nt["Sharpe_rule"],
                "diff_net": nt["diff"],
                "ci_lo_net": nt["ci_lo"],
                "ci_hi_net": nt["ci_hi"],
                "net_excludes_zero": bool(nt["excludes_zero"]),
                "gate_both_exclude_zero": both,
                "verdict": verdict,
            }
        )
    ver_df = pd.DataFrame(ver)
    write(
        ver_df,
        "verdict.csv",
        f"The gate: a rule improves only if the daily-era interval against R0 "
        f"on the primary book ({prim}) excludes zero for BOTH the gross and "
        f"the net-at-{HEADLINE_BP:g}-bp Sharpe difference",
    )
    n_pass = int(ver_df["gate_both_exclude_zero"].sum())
    n_gross = int(ver_df["gross_excludes_zero"].sum())
    n_net = int(ver_df["net_excludes_zero"].sum())
    print(
        f"\nVERDICT: {N_TRIED} rules were tried (R1-R7).  At the "
        f"{CI_ALPHA:.0%} level a single {CI_PCT[1] - CI_PCT[0]:g}% interval "
        f"excludes zero by chance with probability {CI_ALPHA:.2f}, so "
        f"{N_TRIED * CI_ALPHA:.2f} of {N_TRIED} rules would be expected to "
        f"clear one leg by chance alone.  {n_gross} of {N_TRIED} cleared the "
        f"gross leg, {n_net} of {N_TRIED} cleared the net leg, and "
        f"{n_pass} of {N_TRIED} cleared BOTH, which is the gate."
    )
    for _, r in ver_df.iterrows():
        print(
            f"  {r['rule']}  {r['name']:<62s} gross {r['diff_gross']:+.3f} "
            f"[{r['ci_lo_gross']:+.3f}, {r['ci_hi_gross']:+.3f}]  net "
            f"{r['diff_net']:+.3f} [{r['ci_lo_net']:+.3f}, "
            f"{r['ci_hi_net']:+.3f}]  ->  {r['verdict']}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
