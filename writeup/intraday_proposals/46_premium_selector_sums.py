"""46 - the entry clock chosen by the trailing RATIO OF SUMS of the premium.

Proposal 44 chose the entry clock by the trailing mean of

    q_{c,d} = ln( implied remaining-window variance at clock c on session d
                  / realised remaining-window variance from c to the close )

and degenerated: a one-bar remaining window makes ln(implied/realised)
explode, so the mean of logs is monotone increasing in the clock in every
calendar period and the selector picks the last entry clock 15:00 on
1216 of 1216 non-warm-up sessions, at all four lookbacks.  Proposal 43's
premium curve -- the one that DID show the migration (implied over realised
10:00 1.17 -> 0.86, 11:00 1.04 -> 0.77, 14:00 0.87 -> 1.10 from 2020 to
2025) -- is not a mean of logs but a RATIO OF SUMS: the period's summed
implied remaining variance over its summed realised remaining variance, per
clock.  This proposal makes that estimator causal and asks it to choose.

Two pre-registered estimators, both on I_{c,d} (implied remaining-window
variance, the square of the total volatility bisected out of the straddle
midpoint at clock c) and R_{c,d} (realised remaining-window variance, the
summed squared 30-minute log returns from c through the settlement print):

  E1  ratio of sums, cross-clock
      score_c(d) = sum_{d' in window(d)} I_{c,d'} / sum_{d' in window(d)}
      R_{c,d'} over PRIOR sessions (shift 1); pick argmax_c.
      Windows: expanding (min 63), rolling 126, rolling 252, EWMA
      half-life 63 applied to both sums.
  E2  ratio of sums, per-clock relative
      score_c(d) = [rolling-window ratio of sums for c] / [expanding ratio
      of sums for c up to d-1] -- the clock's current premium relative to
      its own long-run level; pick argmax_c.  Windows for the numerator:
      rolling 126, rolling 252, EWMA 63 (expanding over expanding is
      identically 1, so three windows, not four).

Two terminal treatments (hold to the settlement; flatten at 15:30 at the
quoted ask).  Cells: E1 4 x 2 + E2 3 x 2 = 14.

References on the same sessions: fixed 11:00, fixed 13:30, fixed 15:00
(44's degenerate pick), 43's P&L-Sharpe selector at rolling 252, and the
ex-post best fixed clock of each year (an in-sample ceiling, labelled).

The instrument, the tape, the guards, the mid-inversion, the hedge, both
books, the bootstrap, the placebo and the gate arithmetic are proposals 43
and 44's: this script imports 44 by path (which imports 43 by path) and
calls their builders.  It never runs 43's or 44's ``main`` and never writes
into their outputs.

Gates, in order (all before any new number):
  GATE V  43's vectorised package-volatility inversion against
          live.ibkr.pricing.invert_total_vol on every cell of the tape
          (printed by 43.build_chain).
  GATE 0  43's own gate: the 11:00 clock reproduces proposal 41's chain
          build day by day, in-sample Sharpes 2.2530773 / 2.5133605.
  GATE 1  43's own gate: the 11:00 flatten book on the 413 unseen
          sessions, mean +0.025014 premium units, HAC t +1.243,
          -0.0205 index points per contract.
  GATE 2  44's gate: fixed 13:30 on the unseen sample, per contract, hold
          +1.173260 (HAC t +3.210432) and flatten +0.641280 (+2.167573).
  GATE 3  44's gate: 43's P&L-Sharpe selector, 1 of 6 cells improves, and
          it is flatten / rolling 252, +0.4882594 [+0.0211200, +0.9985324].
  GATE N  the NEGATIVE CONTROL: 44's mean-of-logs estimator still picks
          15:00 on every non-warm-up session, fraction 1.0 at all four
          lookbacks.
  GATE C  cross-check, reported not asserted: the per-period ratio of sums
          built here against 43.premium_curve's implied_over_realised.
  GATE 4  the adoption gate of THIS proposal, per cell, on the 413 unseen
          sessions, per contract: the circular-block bootstrap CI of the
          paired daily difference against fixed 11:00 must exclude zero on
          the positive side, AND the cell's mean must not sit below fixed
          13:30 by more than the half-width of its own CI against 13:30,
          AND the per-quarter-histogram clock placebo p must be < 0.05.
          14 cells; under the null the 5% expectation is 0.7 cells.

Run:  python writeup/intraday_proposals/46_premium_selector_sums.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited.

    The module is registered in ``sys.modules`` under ``name`` so that the
    ProcessPoolExecutor inside 43 (and 44's placebo worker) can pickle its
    worker by module name, and so that a spawned child -- which re-imports
    this file and therefore re-runs this line -- finds the same registration.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p44 = _load_module(HERE / "44_premium_ratio_selector.py", "p46_p44")
p43 = p44.p43
asl = p43.asl

OUT: Path = p43.HOLD / "proposals" / "46"

CLOCKS: tuple[str, ...] = p43.CLOCKS
ENTRIES: tuple[str, ...] = p43.ENTRIES
REF_CLOCK: str = p43.REF_CLOCK  # 11:00
ALT_CLOCK: str = p43.ALT_CLOCK  # 13:30
LAST_CLOCK: str = ENTRIES[-1]  # 15:00, 44's degenerate pick
TREATMENTS: tuple[str, ...] = p43.TREATMENTS
UNITS: tuple[tuple[str, str], ...] = p43.UNITS
PTS_UNIT: str = p44.PTS_UNIT
DECK_START: str = p43.DECK_START
DECK_END: str = p43.DECK_END
OOS_START: str = p43.OOS_START
OOS_END: str = p43.OOS_END
HEDGE_COST_BP: float = p43.HEDGE_COST_BP

MIN_DAYS: int = p43.MIN_DAYS  # 63, the repo's warm-up
HALFLIFE: int = p44.HALFLIFE  # 63 = atm_straddle_lib.MZ_HALFLIFE_DAYS
#: (label, window): 0 = expanding, > 0 = rolling, -1 = EWMA at HALFLIFE.
WINDOWS_E1: tuple[tuple[str, int], ...] = (
    ("expanding", 0),
    ("rolling 126", 126),
    ("rolling 252", 252),
    (f"ewma hl {HALFLIFE}", -1),
)
#: E2's numerator windows; expanding over expanding is identically 1.
WINDOWS_E2: tuple[tuple[str, int], ...] = (
    ("rolling 126", 126),
    ("rolling 252", 252),
    (f"ewma hl {HALFLIFE}", -1),
)
ESTIMATORS: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
    ("E1 ratio of sums", WINDOWS_E1),
    ("E2 relative ratio of sums", WINDOWS_E2),
)
N_RULES = len(WINDOWS_E1) + len(WINDOWS_E2)  # 7 distinct pick sequences
N_CELLS = N_RULES * len(TREATMENTS)  # 14

BOOT_B: int = p43.BOOT_B  # 2000
BOOT_BLOCK: int = p43.BOOT_BLOCK  # 21
BOOT_SEED: int = p43.BOOT_SEED  # 0
PLACEBO_B: int = p44.PLACEBO_B  # 2000
PLACEBO_SEED: int = p44.PLACEBO_SEED  # 0
PLACEBO_ALPHA: float = p44.PLACEBO_ALPHA  # 0.05
MAX_LAG_Q: int = p44.MAX_LAG_Q  # 4 quarters searched against the ceiling
AFTERNOON_FROM: str = p44.AFTERNOON_FROM  # 12:30
T_GATE: float = p43.T_GATE  # 2.0
DEGENERATE_FRAC = 0.95  # "one clock > 95% of sessions" = degenerate
N_WORKERS = 8


# ------------------------------------------------- the two variance tapes ---
def implied_realised(
    ch: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """I_{c,d} and R_{c,d}: implied and realised remaining-window variance.

    Identical construction to 44.premium_log_ratio, kept as the two matrices
    instead of their log ratio: implied at clock c is the square of the total
    volatility bisected out of that stamp's straddle midpoint, realised is the
    sum of squared log returns of the 30-minute spot tape from c through 15:30
    plus the last step into the settlement print.  A cell is kept only when
    BOTH are finite and strictly positive, so the two matrices share one mask
    and every ratio of sums below is taken over the same sessions.
    """
    s = ch["S"]
    nxt = np.full_like(s, np.nan)
    nxt[:, :-1] = s[:, 1:]
    nxt[:, -1] = ch["S_close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        step2 = np.where(
            np.isfinite(s) & np.isfinite(nxt) & (s > 0) & (nxt > 0),
            np.log(np.where(s > 0, nxt / s, 1.0)) ** 2,
            np.nan,
        )
    rv = np.full_like(s, np.nan)
    for k in range(len(CLOCKS)):
        whole = np.isfinite(step2[:, k:]).all(axis=1)
        rv[:, k] = np.where(whole, np.nansum(step2[:, k:], axis=1), np.nan)
    iv = ch["tot"] ** 2
    good = np.isfinite(iv) & (iv > 0.0) & np.isfinite(rv) & (rv > 0.0)
    n_e = len(ENTRIES)
    g = good[:, :n_e]
    diag = {
        "cells": int(g.size),
        "finite": int(g.sum()),
        "no_implied": int((~np.isfinite(iv[:, :n_e]) | (iv[:, :n_e] <= 0)).sum()),
        "no_realised": int((~np.isfinite(rv[:, :n_e]) | (rv[:, :n_e] <= 0)).sum()),
    }
    cols = list(ENTRIES)
    i_df = pd.DataFrame(
        np.where(g, iv[:, :n_e], np.nan), index=ch["dates"], columns=cols
    )
    r_df = pd.DataFrame(
        np.where(g, rv[:, :n_e], np.nan), index=ch["dates"], columns=cols
    )
    return i_df, r_df, diag


def ratio_of_sums(i_df: pd.DataFrame, r_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """sum(I) / sum(R) over PRIOR sessions (shift 1), per clock.

    window == 0 expands with a MIN_DAYS minimum; window > 0 rolls with the
    same minimum; window == -1 weights both sums exponentially at HALFLIFE
    with the same minimum.  The EWMA case is written as the ratio of the two
    exponentially weighted MEANS: I and R share one NaN mask by construction,
    so the two weight normalisers are the same number and cancel, leaving
    exactly the ratio of the two exponentially weighted sums.  pandas'
    defaults ``adjust=True`` / ``ignore_na=False`` are kept.  All four
    windows warm up on the same session, so they are comparable day for day.
    """
    out = pd.DataFrame(index=i_df.index, columns=list(ENTRIES), dtype=float)
    for c in ENTRIES:
        i_s, r_s = i_df[c], r_df[c]
        if window == 0:
            si = i_s.expanding(min_periods=MIN_DAYS).sum()
            sr = r_s.expanding(min_periods=MIN_DAYS).sum()
        elif window > 0:
            si = i_s.rolling(window, min_periods=MIN_DAYS).sum()
            sr = r_s.rolling(window, min_periods=MIN_DAYS).sum()
        else:
            si = i_s.ewm(halflife=HALFLIFE, min_periods=MIN_DAYS).mean()
            sr = r_s.ewm(halflife=HALFLIFE, min_periods=MIN_DAYS).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            out[c] = (si / sr).shift(1)
    return out


def relative_ratio_of_sums(
    i_df: pd.DataFrame, r_df: pd.DataFrame, window: int
) -> pd.DataFrame:
    """E2: the window's ratio of sums over the clock's own expanding ratio.

    Both legs are already lagged one session by ``ratio_of_sums``, so the
    score compares the clock's current premium with its own long-run level
    as of the previous close.
    """
    num = ratio_of_sums(i_df, r_df, window)
    den = ratio_of_sums(i_df, r_df, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den.replace(0.0, np.nan)


def pick_argmax(score: pd.DataFrame) -> tuple["pd.Series[Any]", "pd.Series[float]"]:
    """argmax_c of the score; ties to the earlier clock (np.nanargmax).

    A warm-up session (every clock still NaN) is flat, exactly as
    43.trailing_pick and 44.pick_from_trailing treat it.
    """
    arr = score.to_numpy(float)
    all_nan = np.isnan(arr).all(axis=1)
    picks: list[Any] = []
    best = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if all_nan[i]:
            picks.append(pd.NA)
            continue
        j = int(np.nanargmax(arr[i]))
        best[i] = arr[i, j]
        picks.append(ENTRIES[j])
    return (
        pd.Series(picks, index=score.index, dtype=object),
        pd.Series(best, index=score.index),
    )


# ------------------------------------------------------------------ gates ---
def gate_negative_control(
    ch: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, "pd.Series[Any]"]]:
    """GATE N: 44's mean-of-logs estimator is still degenerate at 15:00."""
    q44, qd = p44.premium_log_ratio(ch)
    rows = []
    picks44: dict[str, "pd.Series[Any]"] = {}
    for name, w in p44.LOOKBACKS:
        pk, _ = p44.pick_from_trailing(p44.trailing_q(q44, w), False)
        picks44[name] = pk
        live = pk.dropna()
        frac = float((live == LAST_CLOCK).sum()) / max(int(live.size), 1)
        rows.append(
            {
                "estimator": "44 mean of logs",
                "lookback": name,
                "n_non_warmup": int(live.size),
                f"n_at_{LAST_CLOCK}": int((live == LAST_CLOCK).sum()),
                f"frac_argmax_is_{LAST_CLOCK}": frac,
            }
        )
        assert abs(frac - 1.0) < 1e-12, (name, frac)
    tab = pd.DataFrame(rows)
    print(
        f"GATE N  negative control -- 44's mean-of-logs estimator on the same tape "
        f"({qd['finite']} of {qd['cells']} finite q cells): the argmax is the last "
        f"entry clock {LAST_CLOCK} on "
        + ", ".join(
            f"{r['lookback']} {r[f'n_at_{LAST_CLOCK}']}/{r['n_non_warmup']} "
            f"(frac {r[f'frac_argmax_is_{LAST_CLOCK}']:.4f})"
            for _, r in tab.iterrows()
        )
    )
    return tab, picks44


def gate_cross_check(
    ch: dict[str, Any], idx: pd.DatetimeIndex, i_df: pd.DataFrame, r_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GATE C: this script's per-period ratio of sums against 43's curve.

    Reported, not asserted: 43.premium_curve keeps a cell whenever implied and
    realised are both finite, while this script additionally requires both to
    be strictly positive, so the two masks can differ by the handful of cells
    without a mid-inverted implied variance.
    """
    curve = p43.premium_curve(ch, idx)
    masks = p43.period_masks(idx)
    mine = pd.DataFrame(
        {
            pname: [
                float(i_df.loc[m, c].sum() / r_df.loc[m, c].sum())
                if float(r_df.loc[m, c].sum()) > 0
                else float("nan")
                for c in ENTRIES
            ]
            for pname, m in masks.items()
        },
        index=list(ENTRIES),
    )
    ref = curve.pivot(index="clock", columns="period", values="implied_over_realised")
    ref = ref.reindex(index=list(ENTRIES), columns=list(mine.columns))
    dev = (mine - ref).abs()
    print(
        "\nGATE C (cross-check, reported not asserted)  implied over realised "
        "remaining-window variance as a RATIO OF SUMS, by clock x period -- the "
        "estimator 43's premium curve reports and the one the selectors below "
        f"use.  Max abs deviation from 43.premium_curve {float(dev.max().max()):.3e} "
        f"over {int(dev.notna().sum().sum())} cells; max deviation by period: "
        + ", ".join(f"{k} {float(v):.1e}" for k, v in dev.max().items())
    )
    print(mine.to_string(float_format=lambda v: f"{v:,.4f}"))
    return mine, dev


# -------------------------------------------------------------- reporting ---
def modal_table(
    picks: dict[str, "pd.Series[Any]"], keys: pd.Index, labels: np.ndarray
) -> pd.DataFrame:
    """Modal clock per group: "clock (count / sessions in the group)"."""
    tab = pd.DataFrame(index=pd.Index(sorted(set(keys)), name=keys.name))
    for lab, pk in picks.items():
        col = []
        for g in tab.index:
            m = np.asarray(labels == g)
            v = pk[m].dropna()
            n_g = int(m.sum())
            col.append(
                f"{v.mode().iloc[0]} ({int((v == v.mode().iloc[0]).sum())}/{n_g})"
                if len(v)
                else f"flat (0/{n_g})"
            )
        tab[lab] = col
    return tab


def modal_clock(pk: "pd.Series[Any]", m: np.ndarray) -> str:
    v = pk[m].dropna()
    return str(v.mode().iloc[0]) if len(v) else "flat"


def main() -> None:  # noqa: PLR0912, PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 140)
    pd.set_option("display.max_rows", 2000)
    t_start = time.time()

    # -------------------------------------------------------------- tape --
    stamp, _px = p43.session_stamps()
    half = p43.half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    print(
        "46: the entry clock chosen by the trailing RATIO OF SUMS of implied over "
        "realised remaining-window variance.  Tape, guards, mid-inversion, hedge, "
        "both books, bootstrap, placebo and gate arithmetic are proposals 43 and "
        f"44's, imported by path from {HERE / '44_premium_ratio_selector.py'}"
    )
    print(
        f"chain sessions {len(stamp)}; half sessions dropped {len(half)}; sessions "
        f"scored {len(sessions)} "
        f"{pd.Timestamp(sessions.min()).date()}..{pd.Timestamp(sessions.max()).date()}"
    )
    ch = p43.build_chain(stamp, sessions)  # prints GATE V
    idx = pd.DatetimeIndex(ch["dates"])
    books = p43.clock_books(ch)

    # ------------------------------------------------------------- gates --
    p43.gate_zero(books, ch)
    p43.gate_one(books)
    p44.gate_two(books)
    g3 = p44.gate_three(books, idx)
    gate_n, picks44 = gate_negative_control(ch)

    samples: dict[str, np.ndarray] = {
        "deck in-sample": np.asarray(idx <= pd.Timestamp(DECK_END)),
        "unseen": np.asarray(idx >= pd.Timestamp(OOS_START)),
        "all": np.ones(len(idx), bool),
    }
    print(
        "\nsamples: "
        + ", ".join(f"{k} n={int(v.sum())}" for k, v in samples.items())
        + f" (deck {DECK_START}..{DECK_END}, unseen {OOS_START}..{OOS_END})"
    )

    # ------------------------------------------------- the variance tapes --
    i_df, r_df, diag = implied_realised(ch)
    print(
        f"\n--- A1. I (implied) and R (realised) remaining-window variance per "
        f"(clock, session): {diag['cells']} cells ({len(idx)} sessions x "
        f"{len(ENTRIES)} entry clocks), kept {diag['finite']}, cells without a "
        f"mid-inverted implied variance {diag['no_implied']}, cells without a "
        f"complete realised remaining window {diag['no_realised']}; the two "
        "matrices share one mask"
    )
    ratio_period, ratio_dev = gate_cross_check(ch, idx, i_df, r_df)
    masks = p43.period_masks(idx)
    print("\n--- A2. n kept cells by clock x period")
    print(
        pd.DataFrame(
            {
                pname: [int(i_df.loc[m, c].notna().sum()) for c in ENTRIES]
                for pname, m in masks.items()
            },
            index=list(ENTRIES),
        ).to_string()
    )

    # ----------------------------------------------------- the selectors --
    scores: dict[tuple[str, str], pd.DataFrame] = {}
    for est, windows in ESTIMATORS:
        for wname, w in windows:
            scores[(est, wname)] = (
                ratio_of_sums(i_df, r_df, w)
                if est.startswith("E1")
                else relative_ratio_of_sums(i_df, r_df, w)
            )
    rule_keys = list(scores)
    assert len(rule_keys) == N_RULES, len(rule_keys)
    picks: dict[tuple[str, str], "pd.Series[Any]"] = {}
    bests: dict[tuple[str, str], "pd.Series[float]"] = {}
    for k, sc in scores.items():
        picks[k], bests[k] = pick_argmax(sc)
    rule_label = {k: f"{k[0]} | {k[1]}" for k in rule_keys}

    hold_df = pd.DataFrame({c: books[c]["hold"] for c in ENTRIES})
    p26_lookback = "rolling 252"
    p26_pick = p43.trailing_pick(hold_df, dict(p43.LOOKBACKS)[p26_lookback])

    counts = pd.DataFrame(
        {
            rule_label[k]: [int((picks[k] == c).sum()) for c in ENTRIES]
            + [int(picks[k].isna().sum())]
            for k in rule_keys
        },
        index=list(ENTRIES) + ["flat (warm-up)"],
    )
    counts[f"43 P&L-Sharpe | {p26_lookback}"] = [
        int((p26_pick == c).sum()) for c in ENTRIES
    ] + [int(p26_pick.isna().sum())]
    counts.to_csv(OUT / "pick_counts.csv")
    print(f"\n--- B1. sessions at each clock, by rule (all {len(idx)} sessions)")
    print(counts.to_string())

    # B2 degeneracy: is any estimator again a single clock?
    deg_rows = []
    for k in rule_keys:
        live = picks[k].dropna()
        frac = {c: float((live == c).sum()) / max(int(live.size), 1) for c in ENTRIES}
        top = max(frac, key=lambda c: frac[c])
        deg_rows.append(
            {
                "rule": rule_label[k],
                "estimator": k[0],
                "window": k[1],
                "n_non_warmup": int(live.size),
                "n_distinct_clocks": int(live.nunique()),
                "modal_clock": top,
                "modal_fraction": frac[top],
                "DEGENERATE_gt_95pct": bool(frac[top] > DEGENERATE_FRAC),
            }
        )
    live26 = p26_pick.dropna()
    frac26 = {c: float((live26 == c).sum()) / max(int(live26.size), 1) for c in ENTRIES}
    top26 = max(frac26, key=lambda c: frac26[c])
    deg_rows.append(
        {
            "rule": f"43 P&L-Sharpe | {p26_lookback}",
            "estimator": "43 P&L Sharpe",
            "window": p26_lookback,
            "n_non_warmup": int(live26.size),
            "n_distinct_clocks": int(live26.nunique()),
            "modal_clock": top26,
            "modal_fraction": frac26[top26],
            "DEGENERATE_gt_95pct": bool(frac26[top26] > DEGENERATE_FRAC),
        }
    )
    for name, _w in p44.LOOKBACKS:
        l44 = picks44[name].dropna()
        f44 = {c: float((l44 == c).sum()) / max(int(l44.size), 1) for c in ENTRIES}
        t44 = max(f44, key=lambda c: f44[c])
        deg_rows.append(
            {
                "rule": f"44 mean of logs | {name}",
                "estimator": "44 mean of logs",
                "window": name,
                "n_non_warmup": int(l44.size),
                "n_distinct_clocks": int(l44.nunique()),
                "modal_clock": t44,
                "modal_fraction": f44[t44],
                "DEGENERATE_gt_95pct": bool(f44[t44] > DEGENERATE_FRAC),
            }
        )
    degen = pd.DataFrame(deg_rows)
    degen.to_csv(OUT / "degeneracy.csv", index=False)
    new_rows = degen["estimator"].str.startswith("E")
    n_deg = int(degen.loc[new_rows, "DEGENERATE_gt_95pct"].sum())
    print(
        f"\n--- B2. degeneracy screen (one clock on more than "
        f"{DEGENERATE_FRAC:.0%} of non-warm-up sessions): {n_deg} of "
        f"{N_RULES} new rules are degenerate"
    )
    print(degen.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # B3 per-year argmax fractions
    years = sorted(set(idx.year))
    yr_rows = []
    for k in rule_keys:
        for yr in years:
            m = np.asarray(idx.year == yr)
            v = picks[k][m]
            n_y = max(int(m.sum()), 1)
            row: dict[str, Any] = {
                "rule": rule_label[k],
                "estimator": k[0],
                "window": k[1],
                "year": int(yr),
                "n_sessions": int(m.sum()),
            }
            for c in ENTRIES:
                row[f"frac_{c}"] = float((v == c).sum()) / n_y
            row["frac_flat"] = float(v.isna().sum()) / n_y
            row["modal_clock"] = modal_clock(picks[k], m)
            yr_rows.append(row)
    argmax_year = pd.DataFrame(yr_rows)
    argmax_year.to_csv(OUT / "argmax_fraction_by_year.csv", index=False)
    print(
        "\n--- B3. fraction of sessions whose argmax is each clock, per year, per rule"
    )
    print(argmax_year.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # B4 modal clock per quarter, and the ex-post ceiling
    quarters = pd.Index(idx.to_period("Q").astype(str), name="quarter")
    qarr = np.asarray(quarters)
    modal_q = modal_table(
        {rule_label[k]: picks[k] for k in rule_keys}
        | {f"43 P&L-Sharpe | {p26_lookback}": p26_pick},
        quarters,
        qarr,
    )
    modal_q.to_csv(OUT / "modal_clock_by_quarter.csv")
    print("\n--- B4. modal chosen clock per quarter (count of that clock / sessions)")
    print(modal_q.to_string())

    best_q: dict[str, dict[str, str]] = {}
    for treat in TREATMENTS:
        mat = pd.DataFrame(
            {c: p43.fixed_series(books, c, treat, "pts", "pnl") for c in ENTRIES}
        )
        best_q[treat] = {
            qq: str(mat[np.asarray(qarr == qq)].mean().idxmax())
            for qq in sorted(set(qarr))
        }
    best_q_tab = pd.DataFrame(best_q).reindex(index=sorted(set(qarr)))
    best_q_tab.index.name = "quarter"
    best_q_tab.to_csv(OUT / "ex_post_best_clock_by_quarter.csv")
    print(
        "\n--- B5. the ex-post best fixed clock of each quarter by per-contract "
        "mean (IN-SAMPLE CEILING, never a rule)"
    )
    print(best_q_tab.to_string())

    ceilings: dict[str, dict[int, str]] = {}
    for treat in TREATMENTS:
        for unit, uname in UNITS:
            _s, chosen = p43.ex_post_best(books, idx, treat, unit)
            ceilings[f"{treat}|{uname}"] = chosen
    yr_best = pd.DataFrame(index=pd.Index(years, name="year"))
    for treat in TREATMENTS:
        mat = pd.DataFrame(
            {c: p43.fixed_series(books, c, treat, "pts", "pnl") for c in ENTRIES}
        )
        yr_best[f"ex-post best {treat} by mean pts"] = [
            str(mat[np.asarray(idx.year == yr)].mean().idxmax()) for yr in years
        ]
        yr_best[f"ex-post best {treat} by Sharpe pts"] = [
            ceilings[f"{treat}|{PTS_UNIT}"][int(yr)] for yr in years
        ]
    for k in rule_keys:
        yr_best[rule_label[k]] = [
            modal_clock(picks[k], np.asarray(idx.year == yr)) for yr in years
        ]
    yr_best[f"43 P&L-Sharpe | {p26_lookback}"] = [
        modal_clock(p26_pick, np.asarray(idx.year == yr)) for yr in years
    ]
    yr_best.to_csv(OUT / "ex_post_best_clock_by_year.csv")
    print(
        "\n--- B6. modal chosen clock by YEAR beside the ex-post best fixed clock "
        "of that year (both selection criteria; the ceiling is in-sample)"
    )
    print(yr_best.to_string())

    # B7 pairwise agreement
    all_pick = {rule_label[k]: picks[k] for k in rule_keys}
    all_pick[f"43 P&L-Sharpe | {p26_lookback}"] = p26_pick
    keys_a = list(all_pick)
    agree = pd.DataFrame(index=pd.Index(keys_a, name="rule"), dtype=float)
    for a in keys_a:
        agree[a] = [
            float(
                (
                    all_pick[a].fillna("flat").to_numpy()
                    == all_pick[b].fillna("flat").to_numpy()
                ).mean()
            )
            for b in keys_a
        ]
    agree.to_csv(OUT / "rule_agreement.csv")
    distinct = {
        tuple(all_pick[k].fillna("flat").to_numpy().tolist())
        for k in rule_label.values()
    }
    print(
        f"\n--- B7. pairwise agreement between rules over all {len(idx)} sessions "
        f"(warm-up counted as 'flat'); distinct pick sequences among the "
        f"{N_RULES} new rules: {len(distinct)}"
    )
    print(agree.to_string(float_format=lambda v: f"{v:,.4f}"))

    # timeline
    timeline = pd.DataFrame({"quarter": qarr}, index=idx)
    for k in rule_keys:
        timeline[f"pick_{rule_label[k]}"] = picks[k].astype(object)
        timeline[f"score_{rule_label[k]}"] = bests[k]
    timeline[f"p26_pick_{p26_lookback}"] = p26_pick.astype(object)
    for c in ENTRIES:
        timeline[f"E1_expanding_score_{c}"] = scores[("E1 ratio of sums", "expanding")][
            c
        ]
    for c in ENTRIES:
        timeline[f"I_{c}"] = i_df[c]
        timeline[f"R_{c}"] = r_df[c]
    timeline.to_csv(OUT / "selector_timeline.csv")

    # ------------------------------------------------------------- rules --
    cell_keys = [
        (est, wname, treat) for (est, wname) in rule_keys for treat in TREATMENTS
    ]
    assert len(cell_keys) == N_CELLS, len(cell_keys)
    cell_label = {k: f"{k[0]} | {k[1]} | {k[2]}" for k in cell_keys}
    rows: list[dict[str, Any]] = []
    for est, wname, treat in cell_keys:
        rows += p44.rule_rows(
            books,
            cell_label[(est, wname, treat)],
            est,
            wname,
            treat,
            "-",
            picks[(est, wname)],
            samples,
        )
    for treat in TREATMENTS:
        for c in (REF_CLOCK, ALT_CLOCK, LAST_CLOCK):
            rows += p44.rule_rows(
                books,
                f"fixed {c}",
                "fixed",
                "-",
                treat,
                "-",
                p44.fixed_pick(idx, c),
                samples,
            )
        rows += p44.rule_rows(
            books,
            f"43 P&L-Sharpe selector {p26_lookback}",
            "43 selector",
            p26_lookback,
            treat,
            "-",
            p26_pick,
            samples,
        )
    rules = pd.DataFrame(rows)

    ceil_rows = []
    for treat in TREATMENTS:
        series = {}
        for unit, _uname in UNITS:
            s, _chosen = p43.ex_post_best(books, idx, treat, unit)
            series[unit] = s
        for sname, m in samples.items():
            row = {
                "book": "ex-post best fixed clock per year (IN-SAMPLE CEILING)",
                "family": "ceiling",
                "lookback": "-",
                "treatment": treat,
                "variant": "-",
                "sample": sname,
                "n_sessions": int(m.sum()),
            }
            for unit, uname in UNITS:
                st = p43.stats_row(series[unit][m])
                row[f"mean_{unit}"] = st["mean"]
                row[f"Sharpe_ann_{unit}"] = st["Sharpe_ann"]
                row[f"t_hac_{unit}"] = st["t_hac"]
                row[f"MaxDD_{unit}"] = st["MaxDD"]
                row[f"worst_{unit}"] = st["worst"]
                row[f"worst_date_{unit}"] = st["worst_date"]
                row[f"unit_name_{unit}"] = uname
            ceil_rows.append(row)
    rules = pd.concat([rules, pd.DataFrame(ceil_rows)], ignore_index=True)
    rules.to_csv(OUT / "rules.csv", index=False)

    head = [
        "book",
        "sample",
        "n_traded",
        "frac_traded",
        "mean_pts",
        "Sharpe_ann_pts",
        "t_hac_pts",
        "MaxDD_pts",
        "worst_pts",
        "mean_units",
        "Sharpe_ann_units",
        "t_hac_units",
        "MaxDD_units",
        "worst_units",
        "mean_turnover_pts",
        "mean_cost_pts",
        "mean_charged_pts",
        "Sharpe_charged_pts",
        "mean_cost_units",
        "mean_charged_units",
        "Sharpe_charged_units",
    ]
    head = [h for h in head if h in rules.columns]
    for treat in TREATMENTS:
        sub = rules.loc[rules["treatment"] == treat]
        print(
            f"\n--- C1. {treat}: every book on every sample (per contract beside "
            "premium units; the hedge cost is reported and charged at "
            f"{HEDGE_COST_BP * 1e4:.1f} bp of S per index unit traded)"
        )
        print(sub[head].to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    frac_cols = [f"frac_{c}" for c in ENTRIES] + ["frac_flat"]
    print("\n--- C2. fraction of sessions at each clock, unseen sample")
    print(
        rules.loc[rules["sample"] == "unseen", ["book", "treatment", *frac_cols]]
        .dropna(subset=frac_cols[:1])
        .drop_duplicates(subset=["book", "treatment"])
        .to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )
    print("\n--- C3. the ex-post best fixed clock of each year (in-sample ceiling)")
    for kk, chosen in ceilings.items():
        print(f"  {kk:36s} " + ", ".join(f"{y} {c}" for y, c in chosen.items()))

    # ------------------------------------------------------------ paired --
    t0 = time.time()
    ref_names: tuple[str, ...] = (
        f"fixed {REF_CLOCK}",
        f"fixed {ALT_CLOCK}",
        f"fixed {LAST_CLOCK}",
        f"43 P&L-Sharpe {p26_lookback}",
    )
    ref_series: dict[tuple[str, str, str], "pd.Series[float]"] = {}
    for treat in TREATMENTS:
        for unit, _u in UNITS:
            for c in (REF_CLOCK, ALT_CLOCK, LAST_CLOCK):
                ref_series[(f"fixed {c}", treat, unit)] = p43.fixed_series(
                    books, c, treat, unit, "pnl"
                )
            ref_series[(f"43 P&L-Sharpe {p26_lookback}", treat, unit)] = (
                p43.pick_series(books, p26_pick, treat, unit, "pnl")
            )
    p_rows = []
    for est, wname, treat in cell_keys:
        lab = cell_label[(est, wname, treat)]
        for unit, uname in UNITS:
            s = p43.pick_series(books, picks[(est, wname)], treat, unit, "pnl")
            for rname in ref_names:
                base = ref_series[(rname, treat, unit)]
                for sname, m in samples.items():
                    p_rows.append(
                        {
                            "book": lab,
                            "estimator": est,
                            "window": wname,
                            "treatment": treat,
                            "unit": uname,
                            "reference": rname,
                            "sample": sname,
                            **p44.paired_full(f"{lab} - {rname}", s[m], base[m]),
                        }
                    )
    paired = pd.DataFrame(p_rows)
    print(f"\npaired bootstrap: {len(paired)} rows in {time.time() - t0:.1f} s")
    pcols = [
        "book",
        "unit",
        "reference",
        "sample",
        "n",
        "mean_diff",
        "sd_diff",
        "t_hac",
        "t_lag",
        "d_Sharpe",
    ]
    bcols = [
        "book",
        "unit",
        "reference",
        "sample",
        "n",
        "mean_diff",
        "ci_lo",
        "ci_hi",
        "ci_halfwidth",
        "boot_mean",
        "boot_sd",
        "excludes_zero_positive",
    ]
    paired[pcols].to_csv(OUT / "paired.csv", index=False)
    paired[bcols].assign(B=BOOT_B, block=BOOT_BLOCK, seed=BOOT_SEED).to_csv(
        OUT / "bootstrap.csv", index=False
    )
    print(f"\n--- D1. paired daily differences, per contract ({PTS_UNIT})")
    print(
        paired.loc[paired["unit"] == PTS_UNIT, pcols].to_string(
            index=False, float_format=lambda v: f"{v:,.6f}"
        )
    )
    print(
        f"\n--- D2. circular-block bootstrap of the same differences (B={BOOT_B}, "
        f"block {BOOT_BLOCK}, seed {BOOT_SEED}), per contract"
    )
    print(
        paired.loc[paired["unit"] == PTS_UNIT, bcols].to_string(
            index=False, float_format=lambda v: f"{v:,.6f}"
        )
    )
    print("\n--- D3. the same two tables in premium units")
    print(
        paired.loc[paired["unit"] == "premium", pcols].to_string(
            index=False, float_format=lambda v: f"{v:,.6f}"
        )
    )
    print(
        paired.loc[paired["unit"] == "premium", bcols].to_string(
            index=False, float_format=lambda v: f"{v:,.6f}"
        )
    )
    id_rows = paired.loc[
        (paired["reference"] == f"fixed {LAST_CLOCK}")
        & (paired["unit"] == PTS_UNIT)
        & (paired["sample"] == "unseen")
    ]
    print(
        f"\n--- D4. the paired difference against fixed {LAST_CLOCK} (44's "
        f"degenerate pick) on the unseen sample, per contract, over "
        f"{len(id_rows)} cells: max |mean| "
        f"{float(id_rows['mean_diff'].abs().max()):.3e}, min |mean| "
        f"{float(id_rows['mean_diff'].abs().min()):.3e} (a zero row would mean the "
        f"rule IS fixed {LAST_CLOCK} on that sample)"
    )

    # ----------------------------------------------------------- placebo --
    unseen = samples["unseen"]
    qcode = pd.factorize(qarr[unseen])[0]
    jobs = []
    for est, wname, treat in cell_keys:
        pnl = np.column_stack(
            [
                p43.fixed_series(books, c, treat, "pts", "pnl").to_numpy(float)[unseen]
                for c in ENTRIES
            ]
        )
        pk = picks[(est, wname)][unseen]
        assign = np.array(
            [ENTRIES.index(v) if v in ENTRIES else -1 for v in pk], dtype=int
        )
        jobs.append((cell_label[(est, wname, treat)], pnl, assign, qcode))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(jobs), N_WORKERS)) as pool:
        placebo = pd.DataFrame(list(pool.map(p44._placebo_cell, jobs)))
    placebo.to_csv(OUT / "placebo.csv", index=False)
    print(
        f"\n--- E1. placebo: {PLACEBO_B} clock sequences per cell with the rule's own "
        "per-quarter clock histogram (flat included), unseen sample, per contract; "
        f"p = (1 + #{{placebo >= rule}}) / (1 + {PLACEBO_B}); seed {PLACEBO_SEED}; "
        f"{len(jobs)} cells on {min(len(jobs), N_WORKERS)} worker processes in "
        f"{time.time() - t0:.1f} s.  A placebo sd of 0 means the null has no power "
        "on that cell (its per-quarter histogram is a point mass)."
    )
    print(placebo.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    n_powerless = int((placebo["placebo_sd_pts"] <= 0.0).sum())
    print(
        f"placebo sd: min {float(placebo['placebo_sd_pts'].min()):.6f}, max "
        f"{float(placebo['placebo_sd_pts'].max()):.6f}; cells with sd 0 (powerless "
        f"null) {n_powerless} of {len(placebo)}"
    )

    # -------------------------------------------------------------- gate --
    gate_rows = []
    for est, wname, treat in cell_keys:
        lab = cell_label[(est, wname, treat)]
        sel = paired.loc[
            (paired["book"] == lab)
            & (paired["unit"] == PTS_UNIT)
            & (paired["sample"] == "unseen")
        ].set_index("reference")
        r11 = sel.loc[f"fixed {REF_CLOCK}"]
        r13 = sel.loc[f"fixed {ALT_CLOCK}"]
        pl = placebo.loc[placebo["book"] == lab].iloc[0]
        c1 = bool(r11["excludes_zero_positive"])
        c2 = bool(float(r13["mean_diff"]) >= -float(r13["ci_halfwidth"]))
        c3 = bool(float(pl["p_value"]) < PLACEBO_ALPHA)
        uns = rules.loc[(rules["book"] == lab) & (rules["sample"] == "unseen")].iloc[0]
        gate_rows.append(
            {
                "book": lab,
                "estimator": est,
                "window": wname,
                "treatment": treat,
                "n_unseen": int(uns["n_sessions"]),
                "mean_pts_unseen": float(uns["mean_pts"]),
                "t_hac_pts_unseen": float(uns["t_hac_pts"]),
                "mean_units_unseen": float(uns["mean_units"]),
                "mean_charged_pts_unseen": float(uns["mean_charged_pts"]),
                "diff_vs_1100": float(r11["mean_diff"]),
                "ci_lo_vs_1100": float(r11["ci_lo"]),
                "ci_hi_vs_1100": float(r11["ci_hi"]),
                "cond1_ci_excludes_zero_positive": c1,
                "diff_vs_1330": float(r13["mean_diff"]),
                "ci_lo_vs_1330": float(r13["ci_lo"]),
                "ci_hi_vs_1330": float(r13["ci_hi"]),
                "ci_halfwidth_vs_1330": float(r13["ci_halfwidth"]),
                "cond2_not_below_1330_by_halfwidth": c2,
                "placebo_p": float(pl["p_value"]),
                "placebo_sd_pts": float(pl["placebo_sd_pts"]),
                "cond3_placebo_p_lt_0.05": c3,
                "ADOPTED": bool(c1 and c2 and c3),
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT / "gate.csv", index=False)
    n_adopt = int(gate["ADOPTED"].sum())
    print(
        f"\nGATE 4  {len(gate)} cells tried (E1 {len(WINDOWS_E1)} windows x "
        f"{len(TREATMENTS)} treatments + E2 {len(WINDOWS_E2)} windows x "
        f"{len(TREATMENTS)} treatments), unseen sample, per contract.  A cell is "
        f"adopted only if (1) its bootstrap CI against fixed {REF_CLOCK} excludes "
        f"zero on the positive side, (2) its mean is not below fixed {ALT_CLOCK} by "
        f"more than the half-width of its own CI against {ALT_CLOCK}, and (3) its "
        f"placebo p < {PLACEBO_ALPHA}.  Under the null the 5% expectation is "
        f"{PLACEBO_ALPHA * len(gate):.1f} cells."
    )
    print(gate.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    for _, r in gate.iterrows():
        print(
            f"  {str(r['book']):52s} vs {REF_CLOCK} {r['diff_vs_1100']:+.4f} "
            f"[{r['ci_lo_vs_1100']:+.4f}, {r['ci_hi_vs_1100']:+.4f}] "
            f"{'PASS' if r['cond1_ci_excludes_zero_positive'] else 'fail'}; vs "
            f"{ALT_CLOCK} {r['diff_vs_1330']:+.4f} (half-width "
            f"{r['ci_halfwidth_vs_1330']:.4f}) "
            f"{'PASS' if r['cond2_not_below_1330_by_halfwidth'] else 'fail'}; placebo "
            f"p {r['placebo_p']:.4f} (sd {r['placebo_sd_pts']:.4f}) "
            f"{'PASS' if r['cond3_placebo_p_lt_0.05'] else 'fail'}  -> "
            + ("ADOPTED" if r["ADOPTED"] else "not adopted")
        )
    print(
        f"GATE 4  {n_adopt} of {len(gate)} cells adopted (5% expectation "
        f"{PLACEBO_ALPHA * len(gate):.1f}) -> "
        + ("PASS" if n_adopt > 0 else "FAIL: no cell is adopted")
    )

    # ---------------------------------------------------- family verdict --
    fam_rows = []
    for est, windows in ESTIMATORS:
        for treat in TREATMENTS:
            deck = {
                wname: float(
                    rules.loc[
                        (rules["book"] == cell_label[(est, wname, treat)])
                        & (rules["sample"] == "deck in-sample"),
                        "Sharpe_ann_pts",
                    ].iloc[0]
                )
                for wname, _ in windows
            }
            best_deck = max(deck.values())
            tied = [w for w, v in deck.items() if abs(v - best_deck) < 1e-12]
            # ties -> the simplest window, which is the first declared
            # (expanding for E1; E2 has no expanding numerator, so rolling 126)
            pick_w = next(w for w, _ in windows if w in tied)
            lab = cell_label[(est, pick_w, treat)]
            uns = rules.loc[
                (rules["book"] == lab) & (rules["sample"] == "unseen")
            ].iloc[0]
            row = {
                "estimator": est,
                "treatment": treat,
                "chosen_window": pick_w,
                "criterion": "best deck in-sample Sharpe_ann per contract",
                "n_windows_tied_at_best": len(tied),
                **{f"deck_Sharpe_{k}": v for k, v in deck.items()},
                "unseen_n": int(uns["n_sessions"]),
                "unseen_mean_pts": float(uns["mean_pts"]),
                "unseen_Sharpe_pts": float(uns["Sharpe_ann_pts"]),
                "unseen_t_hac_pts": float(uns["t_hac_pts"]),
                "unseen_mean_charged_pts": float(uns["mean_charged_pts"]),
                "unseen_mean_units": float(uns["mean_units"]),
                "unseen_Sharpe_units": float(uns["Sharpe_ann_units"]),
                "unseen_mean_charged_units": float(uns["mean_charged_units"]),
            }
            for rname, short in (
                (f"fixed {REF_CLOCK}", "1100"),
                (f"fixed {ALT_CLOCK}", "1330"),
            ):
                rr = paired.loc[
                    (paired["book"] == lab)
                    & (paired["unit"] == PTS_UNIT)
                    & (paired["sample"] == "unseen")
                    & (paired["reference"] == rname)
                ].iloc[0]
                row[f"diff_vs_{short}_pts"] = float(rr["mean_diff"])
                row[f"ci_lo_vs_{short}"] = float(rr["ci_lo"])
                row[f"ci_hi_vs_{short}"] = float(rr["ci_hi"])
                row[f"t_hac_vs_{short}"] = float(rr["t_hac"])
                row[f"improves_vs_{short}"] = bool(rr["excludes_zero_positive"])
            fam_rows.append(row)
    family = pd.DataFrame(fam_rows)
    family.to_csv(OUT / "family_verdict.csv", index=False)
    print(
        "\n--- F1. family verdict: within each (estimator, treatment) the window "
        "with the best DECK in-sample per-contract Sharpe is chosen causally (ties "
        "to the simplest, the first declared window), then read once on the unseen "
        f"sample against fixed {REF_CLOCK} and fixed {ALT_CLOCK}"
    )
    print(family.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # -------------------------------------------------- the one paragraph --
    print("\n--- G. one paragraph, numbers only")
    aft = ENTRIES.index(AFTERNOON_FROM)
    qs = sorted(set(qarr))
    lag_rows = []
    for k in rule_keys:
        pk = picks[k]
        mi = np.array(
            [
                float(ENTRIES.index(modal_clock(pk, np.asarray(qarr == qq))))
                if modal_clock(pk, np.asarray(qarr == qq)) in ENTRIES
                else float("nan")
                for qq in qs
            ]
        )
        first_aft = next(
            (qs[i] for i in range(len(qs)) if np.isfinite(mi[i]) and mi[i] >= aft),
            "never",
        )
        n_aft = int(np.nansum(mi >= aft))
        lags = []
        for treat in TREATMENTS:
            bi = np.array(
                [float(ENTRIES.index(best_q[treat][qq])) for qq in qs], dtype=float
            )
            best_lag, best_err = 0, float("inf")
            for lag in range(MAX_LAG_Q + 1):
                mi_l = mi[lag:] if lag else mi
                bi_l = bi[: len(bi) - lag] if lag else bi
                ok = np.isfinite(mi_l) & np.isfinite(bi_l)
                err = (
                    float(np.abs(mi_l[ok] - bi_l[ok]).mean())
                    if ok.any()
                    else float("inf")
                )
                if err < best_err:
                    best_lag, best_err = lag, err
            lags.append((treat, best_lag, best_err))
        yr_match = sum(
            1
            for yr in years
            if modal_clock(pk, np.asarray(idx.year == yr))
            == yr_best.loc[yr, "ex-post best hold by mean pts"]
        )
        lag_rows.append(
            {
                "rule": rule_label[k],
                "mean_modal_clock_index": float(np.nanmean(mi)),
                "first_afternoon_modal_quarter": first_aft,
                "afternoon_modal_quarters": n_aft,
                "n_quarters": len(qs),
                "years_matching_ex_post_best_hold": yr_match,
                "n_years": len(years),
                **{f"best_lag_q_{t}": lag for t, lag, _ in lags},
                **{f"mean_clock_gap_{t}": err for t, _, err in lags},
            }
        )
        print(
            f"  {rule_label[k]:34s} modal clock index (0 = {ENTRIES[0]}, {aft} = "
            f"{AFTERNOON_FROM}, {len(ENTRIES) - 1} = {LAST_CLOCK}): mean "
            f"{float(np.nanmean(mi)):.2f}, first afternoon-modal quarter "
            f"{first_aft}, afternoon-modal quarters {n_aft}/{len(qs)}, years whose "
            f"modal clock equals the ex-post best hold clock {yr_match}/{len(years)}"
            "; best lag vs the per-quarter ex-post ceiling "
            + ", ".join(
                f"{t} {lag} quarters (mean |clock index| gap {err:.2f})"
                for t, lag, err in lags
            )
        )
    lag_tab = pd.DataFrame(lag_rows)
    lag_tab.to_csv(OUT / "migration_lag.csv", index=False)

    pos = rules.loc[
        (rules["sample"] == "unseen")
        & (rules["family"].isin([e for e, _ in ESTIMATORS]))
        & (rules["mean_pts"] > 0.0)
        & (rules["t_hac_pts"] > T_GATE)
    ]
    print(
        f"  cells positive per contract with HAC t > {T_GATE:.0f} on the unseen "
        f"sample: {len(pos)} of {N_CELLS}"
        + (
            ""
            if not len(pos)
            else " -- "
            + ", ".join(
                f"{r['book']} {float(r['mean_pts']):+.4f} pts (t "
                f"{float(r['t_hac_pts']):+.3f})"
                for _, r in pos.iterrows()
            )
        )
    )
    for treat in TREATMENTS:
        for c in (REF_CLOCK, ALT_CLOCK, LAST_CLOCK):
            rr = rules.loc[
                (rules["book"] == f"fixed {c}")
                & (rules["treatment"] == treat)
                & (rules["sample"] == "unseen")
            ].iloc[0]
            print(
                f"    reference fixed {c} {treat:8s} unseen mean "
                f"{float(rr['mean_pts']):+.6f} pts (Sharpe "
                f"{float(rr['Sharpe_ann_pts']):+.6f}, HAC t "
                f"{float(rr['t_hac_pts']):+.6f}), charged "
                f"{float(rr['mean_charged_pts']):+.6f} pts, premium units "
                f"{float(rr['mean_units']):+.6f} (Sharpe "
                f"{float(rr['Sharpe_ann_units']):+.6f})"
            )

    # ----------------------------------------------------------- on disk --
    g3.to_csv(OUT / "gate3_p26_reproduction.csv", index=False)
    gate_n.to_csv(OUT / "gate_negative_control_44.csv", index=False)
    ratio_period.to_csv(OUT / "ratio_of_sums_by_clock_period.csv")
    ratio_dev.to_csv(OUT / "ratio_of_sums_vs_43_deviation.csv")
    written = [
        "selector_timeline.csv",
        "rules.csv",
        "paired.csv",
        "bootstrap.csv",
        "placebo.csv",
        "gate.csv",
        "family_verdict.csv",
        "degeneracy.csv",
        "argmax_fraction_by_year.csv",
        "modal_clock_by_quarter.csv",
        "rule_agreement.csv",
        "migration_lag.csv",
        "pick_counts.csv",
        "ex_post_best_clock_by_quarter.csv",
        "ex_post_best_clock_by_year.csv",
        "ratio_of_sums_by_clock_period.csv",
        "ratio_of_sums_vs_43_deviation.csv",
        "gate3_p26_reproduction.csv",
        "gate_negative_control_44.csv",
    ]
    print(f"\nwrote {OUT}: " + ", ".join(written))
    print(f"total runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
