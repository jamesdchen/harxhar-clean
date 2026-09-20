"""44 - choosing the entry clock by the trailing PREMIUM, not by trailing P&L.

Proposal 43 asked whether the entry clock can be chosen causally with
proposal 26's selector: the clock with the best trailing Sharpe of the
hold-to-close always-short book.  One of its six cells survived the
bootstrap gate.  This proposal replaces the P&L observable by the PRICE
observable the same tape already carries:

    q_{c,d} = ln( implied remaining-window variance at clock c on session d
                  / realised remaining-window variance from c to the close )

which is 43's ``a_premium_curve`` construction taken per session instead of
per calendar period.  q is the log premium the short collects at that clock
on that session; the selector enters, on session d, at the clock whose
trailing MEAN of q over PRIOR sessions is highest.

Four lookbacks (expanding with a 63-session minimum, rolling 126, rolling
252, EWMA half-life 63 = atm_straddle_lib.MZ_HALFLIFE_DAYS), two terminal
treatments (hold to the settlement, flatten at 15:30 at the quoted ask) and
two variants (unconditional; thresholded, which trades only when the
selected clock's trailing mean q is positive) = 16 cells per sample.

The instrument, the tape, the guards, the mid-inversion, the hedge and both
books are proposal 43's -- this script imports 43 by path and calls its
builders; it never re-runs 43's ``main`` and never writes into 43's outputs.

Gates, in order (all before any new number):
  GATE V  43's vectorised package-volatility inversion against
          live.ibkr.pricing.invert_total_vol on every cell (printed by
          43.build_chain).
  GATE 0  43's own gate: the 11:00 clock reproduces proposal 41's chain
          build day by day and its in-sample Sharpes 2.2530773 / 2.5133605.
  GATE 1  43's own gate: the 11:00 flatten book on the 413 unseen sessions,
          mean +0.025014 premium units, HAC t +1.243, -0.0205 points.
  GATE 2  the two fixed benchmarks on the unseen sample, per contract:
          13:30 hold +1.1733 pts/day (HAC t +3.2104), 13:30 flatten
          +0.6413 (t +2.1676).
  GATE 3  43's part-B verdict reproduced: 1 of 6 P&L-Sharpe selector cells
          improves, and it is flatten / rolling 252, per contract
          +0.48826 with the bootstrap CI [+0.02112, +0.99853].
  GATE 4  the adoption gate of THIS proposal, per cell, on the unseen
          sample, per contract: the circular-block bootstrap CI of the
          paired daily difference against fixed 11:00 must exclude zero on
          the positive side, AND the cell's mean must not sit below fixed
          13:30 by more than the half-width of its own CI against 13:30,
          AND the per-quarter-histogram placebo p must be below 0.05.
          16 cells; under the null the 5% expectation is 0.8 cells.

Run:  python writeup/intraday_proposals/44_premium_ratio_selector.py
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
    ProcessPoolExecutor inside 43 can pickle its own worker by module name
    (and so that a spawned child, which re-imports this file as ``__main__``,
    finds the same registration).
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p43 = _load_module(HERE / "43_causal_entry_over_time.py", "p44_p43")
asl = p43.asl

OUT: Path = p43.HOLD / "proposals" / "44"

CLOCKS: tuple[str, ...] = p43.CLOCKS
ENTRIES: tuple[str, ...] = p43.ENTRIES
CLOSE: str = p43.CLOSE
REF_CLOCK: str = p43.REF_CLOCK  # 11:00
ALT_CLOCK: str = p43.ALT_CLOCK  # 13:30
TREATMENTS: tuple[str, ...] = p43.TREATMENTS
UNITS: tuple[tuple[str, str], ...] = p43.UNITS
PTS_UNIT = "index points per contract"
DECK_START: str = p43.DECK_START
DECK_END: str = p43.DECK_END
OOS_START: str = p43.OOS_START
OOS_END: str = p43.OOS_END
ANN: float = p43.ANN
HEDGE_COST_BP: float = p43.HEDGE_COST_BP

# Stated grids.
MIN_DAYS: int = p43.MIN_DAYS  # 63, the repo's warm-up
HALFLIFE: int = int(asl.MZ_HALFLIFE_DAYS)  # 63
#: (label, window): 0 = expanding, > 0 = rolling, -1 = EWMA at HALFLIFE.
LOOKBACKS: tuple[tuple[str, int], ...] = (
    ("expanding", 0),
    ("rolling 126", 126),
    ("rolling 252", 252),
    (f"ewma hl {HALFLIFE}", -1),
)
VARIANTS: tuple[str, ...] = ("unconditional", "thresholded")
N_CELLS = len(LOOKBACKS) * len(TREATMENTS) * len(VARIANTS)  # 16
BOOT_B: int = p43.BOOT_B  # 2000
BOOT_BLOCK: int = p43.BOOT_BLOCK  # 21
BOOT_SEED: int = p43.BOOT_SEED  # 0
CI_LO_PCT: float = p43.CI_LO_PCT
CI_HI_PCT: float = p43.CI_HI_PCT
PLACEBO_B = 2000
PLACEBO_SEED = 0
PLACEBO_ALPHA = 0.05
MAX_LAG_Q = 4  # quarters searched when timing the rule against the ceiling
AFTERNOON_FROM = "12:30"  # the first afternoon entry clock in 43's split

# GATE targets: 43's published numbers.
GATE_INS_WHOLE: float = p43.GATE_INS_WHOLE
GATE_1330 = {"hold": (1.1733, 3.2104), "flatten": (0.6413, 2.1676)}
GATE_1330_TOL = 5e-4
GATE_P26_LOOKBACK = "rolling 252"
GATE_P26_TREAT = "flatten"
GATE_P26_MEAN = 0.48826
GATE_P26_CI = (0.02112, 0.99853)
GATE_P26_TOL = 5e-5
GATE_P26_PASS = 1
GATE_P26_CELLS = 6


# ------------------------------------------------------------- statistics ---
_BOOT_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    """The (B, n) circular-block index array, one per length, seed BOOT_SEED."""
    if n not in _BOOT_IDX:
        rng = np.random.default_rng(BOOT_SEED)
        _BOOT_IDX[n] = asl.circular_block_bootstrap_idx(rng, n, BOOT_BLOCK, BOOT_B)
    return _BOOT_IDX[n]


def paired_full(
    label: str, a: "pd.Series[float]", b: "pd.Series[float]"
) -> dict[str, Any]:
    """a - b: mean, HAC t, and the circular-block bootstrap of the mean.

    Identical construction to 43.paired_row (same B, block, seed and
    percentiles); the bootstrap draws are also summarised so that paired.csv
    and bootstrap.csv can be written from one pass.
    """
    d = (pd.Series(a) - pd.Series(b)).dropna()
    t, lag = asl.newey_west_t(d)
    v = d.to_numpy(float)
    if v.size < 2:
        lo = hi = b_mean = b_sd = float("nan")
    else:
        draws = v[boot_idx(v.size)].mean(axis=1)
        lo = float(np.percentile(draws, CI_LO_PCT))
        hi = float(np.percentile(draws, CI_HI_PCT))
        b_mean = float(draws.mean())
        b_sd = float(draws.std(ddof=1))
    return {
        "difference": label,
        "n": int(d.size),
        "mean_diff": float(d.mean()) if d.size else float("nan"),
        "sd_diff": float(d.std(ddof=1)) if d.size >= 2 else float("nan"),
        "t_hac": t,
        "t_lag": lag,
        "Sharpe_ann_diff": p43.sharpe(d),
        "d_Sharpe": p43.sharpe(a) - p43.sharpe(b),
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_halfwidth": 0.5 * (hi - lo),
        "boot_mean": b_mean,
        "boot_sd": b_sd,
        "excludes_zero_positive": bool(np.isfinite(lo) and lo > 0.0),
    }


# ------------------------------------------------------- the q observable ---
def premium_log_ratio(ch: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """q_{c,d} = ln(implied / realised remaining-window variance), per session.

    Implied at clock c is the square of the total volatility bisected out of
    that stamp's straddle midpoint (43's ``tot``); realised is the sum of
    squared log returns of the 30-minute spot tape from c through 15:30 plus
    the last step into the settlement print -- exactly 43.premium_curve's two
    quantities, kept per session instead of averaged over a period.
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
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(
            good, np.log(np.where(good, iv / np.where(rv > 0, rv, 1.0), 1.0)), np.nan
        )
    n_e = len(ENTRIES)
    diag = {
        "cells": int(good[:, :n_e].size),
        "finite": int(good[:, :n_e].sum()),
        "no_implied": int((~np.isfinite(iv[:, :n_e]) | (iv[:, :n_e] <= 0)).sum()),
        "no_realised": int((~np.isfinite(rv[:, :n_e]) | (rv[:, :n_e] <= 0)).sum()),
    }
    return pd.DataFrame(q[:, :n_e], index=ch["dates"], columns=list(ENTRIES)), diag


def trailing_q(q: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing mean of q per clock over PRIOR sessions (shift 1).

    window == 0 expands with a MIN_DAYS minimum; window > 0 rolls with the
    same minimum; window == -1 is the EWMA at HALFLIFE with the same minimum,
    so all four lookbacks warm up on the same session and are comparable day
    for day.  pandas' default ``adjust=True`` / ``ignore_na=False`` are kept.
    """
    out = pd.DataFrame(index=q.index, columns=list(ENTRIES), dtype=float)
    for c in ENTRIES:
        s = q[c]
        if window == 0:
            m = s.expanding(min_periods=MIN_DAYS).mean()
        elif window > 0:
            m = s.rolling(window, min_periods=MIN_DAYS).mean()
        else:
            m = s.ewm(halflife=HALFLIFE, min_periods=MIN_DAYS).mean()
        out[c] = m.shift(1)
    return out


def pick_from_trailing(
    trail: pd.DataFrame, threshold: bool
) -> tuple["pd.Series[Any]", "pd.Series[float]"]:
    """Highest trailing mean q; ties to the earlier clock (np.nanargmax).

    ``threshold`` additionally refuses the day when the selected clock's
    trailing mean q is not strictly positive.  A refused day and a warm-up
    day are both flat, exactly as 43.pick_series treats them.
    """
    arr = trail.to_numpy(float)
    all_nan = np.isnan(arr).all(axis=1)
    picks: list[Any] = []
    qbar = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if all_nan[i]:
            picks.append(pd.NA)
            continue
        j = int(np.nanargmax(arr[i]))
        qbar[i] = arr[i, j]
        picks.append(pd.NA if (threshold and not arr[i, j] > 0.0) else ENTRIES[j])
    return (
        pd.Series(picks, index=trail.index, dtype=object),
        pd.Series(qbar, index=trail.index),
    )


# ------------------------------------------------------------- the books ----
def rule_field(
    books: dict[str, Any], pick: "pd.Series[Any]", field: str
) -> "pd.Series[float]":
    """Any per-clock field of the tape, selected by the rule; flat days 0.0."""
    idx = pd.DatetimeIndex(pick.index)
    out = pd.Series(0.0, index=idx)
    for c in ENTRIES:
        m = (pick == c).to_numpy()
        if m.any():
            v = books[c][field].reindex(idx).to_numpy(float)
            out.iloc[np.where(m)[0]] = np.nan_to_num(v[m], nan=0.0)
    return out


def traded_mask(
    books: dict[str, Any], pick: "pd.Series[Any]", treat: str
) -> np.ndarray:
    """Days the rule names a clock AND that clock's book is priceable."""
    idx = pd.DatetimeIndex(pick.index)
    out = np.zeros(len(idx), bool)
    for c in ENTRIES:
        m = (pick == c).to_numpy()
        if m.any():
            v = books[c][treat].reindex(idx).to_numpy(float)
            out |= m & np.isfinite(v)
    return out


def fixed_pick(idx: pd.DatetimeIndex, clock: str) -> "pd.Series[Any]":
    return pd.Series([clock] * len(idx), index=idx, dtype=object)


def clock_fraction(pick: "pd.Series[Any]", m: np.ndarray) -> dict[str, float]:
    v = pick[m]
    n = max(int(len(v)), 1)
    out = {f"frac_{c}": float((v == c).sum()) / n for c in ENTRIES}
    out["frac_flat"] = float(v.isna().sum()) / n
    return out


def rule_rows(
    books: dict[str, Any],
    label: str,
    family: str,
    lookback: str,
    treat: str,
    variant: str,
    pick: "pd.Series[Any]",
    samples: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """One row per sample: both units, both cost columns, the clock mix."""
    gross = {u: p43.pick_series(books, pick, treat, u, "pnl") for u, _ in UNITS}
    cost = {u: p43.pick_series(books, pick, treat, u, "cost") for u, _ in UNITS}
    turn = rule_field(books, pick, f"turnover_{treat}_pts")
    traded = traded_mask(books, pick, treat)
    rows = []
    for sname, m in samples.items():
        row: dict[str, Any] = {
            "book": label,
            "family": family,
            "lookback": lookback,
            "treatment": treat,
            "variant": variant,
            "sample": sname,
            "n_sessions": int(m.sum()),
            "n_traded": int((traded & m).sum()),
            "frac_traded": float((traded & m).sum()) / max(int(m.sum()), 1),
            "mean_turnover_pts": float(turn[m].mean()),
        }
        for unit, uname in UNITS:
            g = gross[unit][m]
            c = cost[unit][m]
            st = p43.stats_row(g)
            suf = unit
            row[f"mean_{suf}"] = st["mean"]
            row[f"sd_{suf}"] = st["sd"]
            row[f"Sharpe_ann_{suf}"] = st["Sharpe_ann"]
            row[f"t_hac_{suf}"] = st["t_hac"]
            row[f"t_lag_{suf}"] = st["t_lag"]
            row[f"MaxDD_{suf}"] = st["MaxDD"]
            row[f"worst_{suf}"] = st["worst"]
            row[f"worst_date_{suf}"] = st["worst_date"]
            row[f"mean_cost_{suf}"] = float(c.mean())
            row[f"mean_charged_{suf}"] = float((g - c).mean())
            row[f"Sharpe_charged_{suf}"] = p43.sharpe(g - c)
            row[f"unit_name_{suf}"] = uname
        row.update(clock_fraction(pick, m))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- placebo ---
def _assign_mean(pnl: np.ndarray, assign: np.ndarray) -> float:
    """Mean daily per-contract P&L of a clock assignment; -1 is a flat day."""
    take = np.maximum(assign, 0)
    v = pnl[np.arange(len(assign)), take]
    return float(np.where(assign >= 0, v, 0.0).mean())


def _placebo_cell(
    job: tuple[str, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """PLACEBO_B clock sequences with the rule's own per-quarter histogram.

    Within each calendar quarter the rule's clock labels (flat included) are
    permuted across that quarter's sessions, so every placebo keeps the
    rule's drift from morning to afternoon and scrambles only which session
    gets which clock.
    """
    label, pnl, assign, qcode = job
    rule = _assign_mean(pnl, assign)
    groups = [np.where(qcode == g)[0] for g in np.unique(qcode)]
    support = [int(len(np.unique(assign[g]))) for g in groups]
    rng = np.random.default_rng(PLACEBO_SEED)
    draws = np.empty(PLACEBO_B)
    perm = assign.copy()
    for b in range(PLACEBO_B):
        for g in groups:
            perm[g] = assign[rng.permutation(g)]
        draws[b] = _assign_mean(pnl, perm)
    ge = int((draws >= rule).sum())
    return {
        "book": label,
        "n_sessions": int(len(assign)),
        "n_quarters": int(len(groups)),
        "mean_labels_per_quarter": float(np.mean(support)),
        "quarters_with_one_label": int(sum(1 for s in support if s == 1)),
        "B": PLACEBO_B,
        "rule_mean_pts": rule,
        "placebo_mean_pts": float(draws.mean()),
        "placebo_sd_pts": float(draws.std(ddof=1)),
        "placebo_p05_pts": float(np.percentile(draws, 5.0)),
        "placebo_p95_pts": float(np.percentile(draws, 95.0)),
        "n_placebo_ge_rule": ge,
        "p_value": (1.0 + ge) / (1.0 + PLACEBO_B),
    }


# ------------------------------------------------------------------ gates ---
def gate_two(books: dict[str, Any]) -> None:
    """The 13:30 benchmark on the unseen sample, per contract."""
    for treat, (mean_t, t_t) in GATE_1330.items():
        s = p43.clock_series(books, ALT_CLOCK, treat, "pts", "pnl")
        m = (s.index >= pd.Timestamp(OOS_START)) & (s.index <= pd.Timestamp(OOS_END))
        v = s[m].dropna()
        t, lag = asl.newey_west_t(v)
        assert abs(float(v.mean()) - mean_t) < GATE_1330_TOL, float(v.mean())
        assert abs(t - t_t) < GATE_1330_TOL, t
        print(
            f"GATE 2  fixed {ALT_CLOCK} {treat:8s} on the unseen sample: n "
            f"{int(v.size)}, mean {float(v.mean()):+.6f} index points per contract "
            f"(target {mean_t:+.4f}), HAC t {t:+.6f} (lag {lag}, target {t_t:+.4f})"
        )


def gate_three(books: dict[str, Any], idx: pd.DatetimeIndex) -> pd.DataFrame:
    """43's part B reproduced: 1 of 6 P&L-Sharpe cells improves, and which."""
    hold_df = pd.DataFrame({c: books[c]["hold"] for c in ENTRIES})
    picks = {name: p43.trailing_pick(hold_df, w) for name, w in p43.LOOKBACKS}
    m = np.asarray(idx >= pd.Timestamp(OOS_START))
    rows = []
    for treat in TREATMENTS:
        base = p43.fixed_series(books, REF_CLOCK, treat, "pts", "pnl")
        for name, _ in p43.LOOKBACKS:
            s = p43.pick_series(books, picks[name], treat, "pts", "pnl")
            rows.append(
                {
                    "treatment": treat,
                    "lookback": name,
                    **paired_full(f"p26 {name} - fixed {REF_CLOCK}", s[m], base[m]),
                }
            )
    tab = pd.DataFrame(rows)
    n_pass = int(tab["excludes_zero_positive"].sum())
    hit = tab.loc[
        (tab["treatment"] == GATE_P26_TREAT) & (tab["lookback"] == GATE_P26_LOOKBACK)
    ].iloc[0]
    assert len(tab) == GATE_P26_CELLS, len(tab)
    assert n_pass == GATE_P26_PASS, n_pass
    assert abs(float(hit["mean_diff"]) - GATE_P26_MEAN) < GATE_P26_TOL, hit["mean_diff"]
    assert abs(float(hit["ci_lo"]) - GATE_P26_CI[0]) < GATE_P26_TOL, hit["ci_lo"]
    assert abs(float(hit["ci_hi"]) - GATE_P26_CI[1]) < GATE_P26_TOL, hit["ci_hi"]
    print(
        f"GATE 3  43's P&L-Sharpe selector on the unseen sample, per contract: "
        f"{n_pass} of {len(tab)} cells improve (target {GATE_P26_PASS}), and it is "
        f"{GATE_P26_TREAT} / {GATE_P26_LOOKBACK}: mean {float(hit['mean_diff']):+.7f} "
        f"pts (target {GATE_P26_MEAN:+.5f}), CI [{float(hit['ci_lo']):+.7f}, "
        f"{float(hit['ci_hi']):+.7f}] (target [{GATE_P26_CI[0]:+.5f}, "
        f"{GATE_P26_CI[1]:+.5f}])"
    )
    print(tab.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    return tab


# ------------------------------------------------------------------- main ---
def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.max_rows", 500)
    t_start = time.time()

    # -------------------------------------------------------------- tape --
    stamp, _px = p43.session_stamps()
    half = p43.half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    print(
        f"44: entry-clock selection on the trailing log premium.  Tape, guards, "
        f"mid-inversion, hedge and both books are proposal 43's, imported by path "
        f"from {HERE / '43_causal_entry_over_time.py'}"
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
    gate_two(books)
    g3 = gate_three(books, idx)

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

    # ------------------------------------------------------ the q surface --
    q, qdiag = premium_log_ratio(ch)
    print(
        f"\n--- A1. q = ln(implied / realised remaining-window variance) per "
        f"(clock, session): {qdiag['cells']} cells ({len(idx)} sessions x "
        f"{len(ENTRIES)} entry clocks), finite {qdiag['finite']}, cells without a "
        f"mid-inverted implied variance {qdiag['no_implied']}, cells without a "
        f"complete realised remaining window {qdiag['no_realised']}"
    )
    period_m = p43.period_masks(idx)
    q_period = pd.DataFrame(
        {
            pname: [float(q.loc[m, c].mean()) for c in ENTRIES]
            for pname, m in period_m.items()
        },
        index=list(ENTRIES),
    )
    print("\n--- A2. mean q by clock x period (log premium; > 0 = implied above)")
    print(q_period.to_string(float_format=lambda v: f"{v:,.4f}"))
    q_n = pd.DataFrame(
        {
            pname: [int(q.loc[m, c].notna().sum()) for c in ENTRIES]
            for pname, m in period_m.items()
        },
        index=list(ENTRIES),
    )
    print("\n--- A3. n finite q by clock x period")
    print(q_n.to_string())

    # ----------------------------------------------------- the selectors --
    trails = {name: trailing_q(q, w) for name, w in LOOKBACKS}
    picks: dict[tuple[str, str], "pd.Series[Any]"] = {}
    qbars: dict[str, "pd.Series[float]"] = {}
    for name, _ in LOOKBACKS:
        for variant in VARIANTS:
            pk, qb = pick_from_trailing(trails[name], variant == "thresholded")
            picks[(name, variant)] = pk
            if variant == "unconditional":
                qbars[name] = qb
    hold_df = pd.DataFrame({c: books[c]["hold"] for c in ENTRIES})
    p26_picks = {name: p43.trailing_pick(hold_df, w) for name, w in p43.LOOKBACKS}

    counts = pd.DataFrame(
        {
            f"{name} | {variant}": [
                int((picks[(name, variant)] == c).sum()) for c in ENTRIES
            ]
            + [int(picks[(name, variant)].isna().sum())]
            for name, _ in LOOKBACKS
            for variant in VARIANTS
        },
        index=list(ENTRIES) + ["flat"],
    )
    print("\n--- B1. sessions at each clock, by lookback and variant (all 1279)")
    print(counts.to_string())

    # B1b: how many DISTINCT rules the 4 lookbacks x 2 variants actually are.
    keys8 = [(name, variant) for name, _ in LOOKBACKS for variant in VARIANTS]
    agree = pd.DataFrame(index=[f"{a} | {b}" for a, b in keys8], dtype=float)
    for a in keys8:
        agree[f"{a[0]} | {a[1]}"] = [
            float(
                (
                    picks[a].fillna("flat").to_numpy()
                    == picks[b].fillna("flat").to_numpy()
                ).mean()
            )
            for b in keys8
        ]
    distinct = {tuple(picks[k].fillna("flat").to_numpy().tolist()) for k in keys8}
    refused = {
        name: int(
            (
                picks[(name, "thresholded")].isna()
                & picks[(name, "unconditional")].notna()
            ).sum()
        )
        for name, _ in LOOKBACKS
    }
    print(
        f"\n--- B1b. distinct pick sequences among the {len(keys8)} "
        f"(lookback, variant) rules: {len(distinct)}; sessions the threshold "
        "refuses, by lookback: " + ", ".join(f"{k} {v}" for k, v in refused.items())
    )
    print(agree.to_string(float_format=lambda v: f"{v:,.4f}"))
    last = ENTRIES[-1]
    argmax_last = {
        name: float((picks[(name, "unconditional")].dropna() == last).sum())
        / max(int(picks[(name, "unconditional")].notna().sum()), 1)
        for name, _ in LOOKBACKS
    }
    print(
        f"fraction of non-warm-up sessions whose trailing-mean-q argmax is the "
        f"last entry clock {last}: "
        + ", ".join(f"{k} {v:.4f}" for k, v in argmax_last.items())
    )

    quarters = pd.Index(idx.to_period("Q").astype(str), name="quarter")
    timeline = pd.DataFrame({"quarter": np.asarray(quarters)}, index=idx)
    for name, _ in LOOKBACKS:
        timeline[f"pick_{name}"] = picks[(name, "unconditional")].astype(object)
        timeline[f"qbar_{name}"] = qbars[name]
        timeline[f"pick_thresholded_{name}"] = picks[(name, "thresholded")].astype(
            object
        )
    for name, _ in p43.LOOKBACKS:
        timeline[f"p26_pick_{name}"] = p26_picks[name].astype(object)
    for c in ENTRIES:
        timeline[f"q_{c}"] = q[c]
    timeline.to_csv(OUT / "selector_timeline.csv")

    def modal_table(get: dict[str, "pd.Series[Any]"]) -> pd.DataFrame:
        tab = pd.DataFrame(index=pd.Index(sorted(set(quarters)), name="quarter"))
        for lab, pk in get.items():
            col = []
            for qq in tab.index:
                v = pk[np.asarray(quarters == qq)].dropna()
                col.append(
                    f"{v.mode().iloc[0]} ({int((v == v.mode().iloc[0]).sum())}/"
                    f"{int(np.asarray(quarters == qq).sum())})"
                    if len(v)
                    else f"flat (0/{int(np.asarray(quarters == qq).sum())})"
                )
            tab[lab] = col
        return tab

    modal_u = modal_table(
        {name: picks[(name, "unconditional")] for name, _ in LOOKBACKS}
    )
    modal_t = modal_table({name: picks[(name, "thresholded")] for name, _ in LOOKBACKS})
    modal_26 = modal_table({f"p26 {n}": p26_picks[n] for n, _ in p43.LOOKBACKS})
    print(
        "\n--- B2. modal chosen clock per quarter, UNCONDITIONAL "
        "(count of that clock / sessions in the quarter)"
    )
    print(modal_u.to_string())
    print("\n--- B3. modal chosen clock per quarter, THRESHOLDED (q-bar > 0)")
    print(modal_t.to_string())
    print("\n--- B4. modal chosen clock per quarter, 43's P&L-Sharpe selector")
    print(modal_26.to_string())

    # the ex-post best fixed clock of each quarter, by per-contract mean
    pts_by_clock = pd.DataFrame(
        {c: p43.clock_series(books, c, "hold", "pts", "pnl") for c in ENTRIES}
    )
    best_q: dict[str, dict[str, str]] = {}
    for treat in TREATMENTS:
        mat = pd.DataFrame(
            {c: p43.fixed_series(books, c, treat, "pts", "pnl") for c in ENTRIES}
        )
        best_q[treat] = {
            qq: str(mat[np.asarray(quarters == qq)].mean().idxmax())
            for qq in sorted(set(quarters))
        }
    best_q_tab = pd.DataFrame(best_q).reindex(index=sorted(set(quarters)))
    best_q_tab.index.name = "quarter"
    print(
        "\n--- B5. the ex-post best fixed clock of each quarter by per-contract "
        "mean (IN-SAMPLE CEILING, never a rule)"
    )
    print(best_q_tab.to_string())
    print(
        "\n--- B6. mean per-contract P&L by clock x period, hold (index points) -- "
        "the surface the ceiling reads"
    )
    print(
        pd.DataFrame(
            {
                pname: [float(pts_by_clock.loc[m, c].mean()) for c in ENTRIES]
                for pname, m in period_m.items()
            },
            index=list(ENTRIES),
        ).to_string(float_format=lambda v: f"{v:,.4f}")
    )

    # ------------------------------------------------------------- rules --
    cell_keys = [
        (name, treat, variant)
        for name, _ in LOOKBACKS
        for treat in TREATMENTS
        for variant in VARIANTS
    ]
    assert len(cell_keys) == N_CELLS, len(cell_keys)
    cell_label = {k: f"q-selector {k[0]} | {k[1]} | {k[2]}" for k in cell_keys}
    rows: list[dict[str, Any]] = []
    for name, treat, variant in cell_keys:
        rows += rule_rows(
            books,
            cell_label[(name, treat, variant)],
            "q-selector",
            name,
            treat,
            variant,
            picks[(name, variant)],
            samples,
        )
    for treat in TREATMENTS:
        for c in (REF_CLOCK, ALT_CLOCK, ENTRIES[-1]):
            rows += rule_rows(
                books,
                f"fixed {c}",
                "fixed",
                "-",
                treat,
                "-",
                fixed_pick(idx, c),
                samples,
            )
        for name, _ in p43.LOOKBACKS:
            rows += rule_rows(
                books,
                f"p26 Sharpe selector {name}",
                "p26 selector",
                name,
                treat,
                "unconditional",
                p26_picks[name],
                samples,
            )
    rules = pd.DataFrame(rows)

    # the ex-post ceiling, per year, both treatments and units (43's builder)
    ceil_rows = []
    ceilings: dict[str, dict[int, str]] = {}
    for treat in TREATMENTS:
        series = {}
        for unit, uname in UNITS:
            s, chosen = p43.ex_post_best(books, idx, treat, unit)
            series[unit] = s
            ceilings[f"{treat}|{uname}"] = chosen
        for sname, m in samples.items():
            row: dict[str, Any] = {
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
    ]
    for treat in TREATMENTS:
        sub = rules.loc[rules["treatment"] == treat]
        print(
            f"\n--- C1. {treat}: every book on every sample (per contract beside "
            "premium units; cost reported and charged at "
            f"{HEDGE_COST_BP * 1e4:.1f} bp of S per index unit traded)"
        )
        print(sub[head].to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    frac_cols = [f"frac_{c}" for c in ENTRIES] + ["frac_flat"]
    print("\n--- C2. fraction of sessions at each clock, unseen sample")
    print(
        rules.loc[rules["sample"] == "unseen", ["book", "treatment", *frac_cols]]
        .drop_duplicates(subset=["book", "treatment"])
        .to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )
    print("\n--- C3. the ex-post best fixed clock of each year (in-sample ceiling)")
    for k, chosen in ceilings.items():
        print(f"  {k:36s} " + ", ".join(f"{y} {c}" for y, c in chosen.items()))

    # ------------------------------------------------------------ paired --
    t0 = time.time()
    p_rows = []
    for name, treat, variant in cell_keys:
        pk = picks[(name, variant)]
        for unit, uname in UNITS:
            s = p43.pick_series(books, pk, treat, unit, "pnl")
            for ref in (REF_CLOCK, ALT_CLOCK, ENTRIES[-1]):
                base = p43.fixed_series(books, ref, treat, unit, "pnl")
                for sname, m in samples.items():
                    p_rows.append(
                        {
                            "book": cell_label[(name, treat, variant)],
                            "lookback": name,
                            "treatment": treat,
                            "variant": variant,
                            "unit": uname,
                            "reference": f"fixed {ref}",
                            "sample": sname,
                            **paired_full(
                                f"{cell_label[(name, treat, variant)]} - fixed {ref}",
                                s[m],
                                base[m],
                            ),
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
    print(
        f"\n--- D1. paired daily differences, per contract "
        f"({PTS_UNIT}), against fixed {REF_CLOCK} and fixed {ALT_CLOCK}"
    )
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
        (paired["reference"] == f"fixed {ENTRIES[-1]}")
        & (paired["unit"] == PTS_UNIT)
        & (paired["sample"] == "unseen")
    ]
    print(
        f"\n--- D4. the paired difference against fixed {ENTRIES[-1]} on the unseen "
        f"sample, per contract, over {len(id_rows)} cells: max |mean| "
        f"{float(id_rows['mean_diff'].abs().max()):.3e}, max sd "
        f"{float(id_rows['sd_diff'].abs().max()):.3e} (a zero row means the rule IS "
        f"fixed {ENTRIES[-1]} on that sample)"
    )

    # ----------------------------------------------------------- placebo --
    unseen = samples["unseen"]
    qcode = pd.factorize(np.asarray(quarters)[unseen])[0]
    jobs = []
    for name, treat, variant in cell_keys:
        pnl = np.column_stack(
            [
                p43.fixed_series(books, c, treat, "pts", "pnl").to_numpy(float)[unseen]
                for c in ENTRIES
            ]
        )
        pk = picks[(name, variant)][unseen]
        assign = np.array(
            [ENTRIES.index(v) if v in ENTRIES else -1 for v in pk], dtype=int
        )
        jobs.append((cell_label[(name, treat, variant)], pnl, assign, qcode))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
        placebo = pd.DataFrame(list(pool.map(_placebo_cell, jobs)))
    placebo.to_csv(OUT / "placebo.csv", index=False)
    print(
        f"\n--- E1. placebo: {PLACEBO_B} clock sequences per cell with the rule's own "
        f"per-quarter clock histogram (flat included), unseen sample, per contract; "
        f"p = (1 + #{{placebo >= rule}}) / (1 + {PLACEBO_B}); seed {PLACEBO_SEED}; "
        f"{len(jobs)} cells on {min(len(jobs), 8)} worker processes in "
        f"{time.time() - t0:.1f} s"
    )
    print(placebo.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # -------------------------------------------------------------- gate --
    gate_rows = []
    for name, treat, variant in cell_keys:
        lab = cell_label[(name, treat, variant)]
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
        gate_rows.append(
            {
                "book": lab,
                "lookback": name,
                "treatment": treat,
                "variant": variant,
                "mean_pts_unseen": float(
                    rules.loc[
                        (rules["book"] == lab)
                        & (rules["treatment"] == treat)
                        & (rules["sample"] == "unseen"),
                        "mean_pts",
                    ].iloc[0]
                ),
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
                "cond3_placebo_p_lt_0.05": c3,
                "ADOPTED": bool(c1 and c2 and c3),
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT / "gate.csv", index=False)
    n_adopt = int(gate["ADOPTED"].sum())
    print(
        f"\nGATE 4  {len(gate)} cells tried ({len(LOOKBACKS)} lookbacks x "
        f"{len(TREATMENTS)} treatments x {len(VARIANTS)} variants), unseen sample, "
        f"per contract.  A cell is adopted only if (1) its bootstrap CI against "
        f"fixed {REF_CLOCK} excludes zero on the positive side, (2) its mean is not "
        f"below fixed {ALT_CLOCK} by more than the half-width of its own CI against "
        f"{ALT_CLOCK}, and (3) its placebo p < {PLACEBO_ALPHA}.  Under the null the "
        f"5% expectation is {PLACEBO_ALPHA * len(gate):.1f} cells."
    )
    print(gate.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    for _, r in gate.iterrows():
        print(
            f"  {r['book']:52s} vs {REF_CLOCK} {r['diff_vs_1100']:+.4f} "
            f"[{r['ci_lo_vs_1100']:+.4f}, {r['ci_hi_vs_1100']:+.4f}] "
            f"{'PASS' if r['cond1_ci_excludes_zero_positive'] else 'fail'}; vs "
            f"{ALT_CLOCK} {r['diff_vs_1330']:+.4f} (half-width "
            f"{r['ci_halfwidth_vs_1330']:.4f}) "
            f"{'PASS' if r['cond2_not_below_1330_by_halfwidth'] else 'fail'}; placebo "
            f"p {r['placebo_p']:.4f} "
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
    for treat in TREATMENTS:
        for variant in VARIANTS:
            deck = {
                name: float(
                    rules.loc[
                        (rules["book"] == cell_label[(name, treat, variant)])
                        & (rules["sample"] == "deck in-sample"),
                        "Sharpe_ann_pts",
                    ].iloc[0]
                )
                for name, _ in LOOKBACKS
            }
            best_deck = max(deck.values())
            n_tied = int(sum(1 for v in deck.values() if abs(v - best_deck) < 1e-12))
            pick_lb = max(deck, key=lambda k: deck[k])
            lab = cell_label[(pick_lb, treat, variant)]
            r11 = paired.loc[
                (paired["book"] == lab)
                & (paired["unit"] == PTS_UNIT)
                & (paired["sample"] == "unseen")
                & (paired["reference"] == f"fixed {REF_CLOCK}")
            ].iloc[0]
            uns = rules.loc[
                (rules["book"] == lab) & (rules["sample"] == "unseen")
            ].iloc[0]
            fam_rows.append(
                {
                    "treatment": treat,
                    "variant": variant,
                    "chosen_lookback": pick_lb,
                    "criterion": "best deck in-sample Sharpe_ann per contract",
                    "n_lookbacks_tied_at_best": n_tied,
                    **{f"deck_Sharpe_{k}": v for k, v in deck.items()},
                    "unseen_mean_pts": float(uns["mean_pts"]),
                    "unseen_Sharpe_pts": float(uns["Sharpe_ann_pts"]),
                    "unseen_t_hac_pts": float(uns["t_hac_pts"]),
                    "unseen_mean_units": float(uns["mean_units"]),
                    "unseen_Sharpe_units": float(uns["Sharpe_ann_units"]),
                    "diff_vs_1100_pts": float(r11["mean_diff"]),
                    "ci_lo": float(r11["ci_lo"]),
                    "ci_hi": float(r11["ci_hi"]),
                    "t_hac_diff": float(r11["t_hac"]),
                    "improves_vs_1100": bool(r11["excludes_zero_positive"]),
                }
            )
    family = pd.DataFrame(fam_rows)
    family.to_csv(OUT / "family_verdict.csv", index=False)
    print(
        "\n--- F1. family verdict: within each (treatment, variant) the lookback "
        "with the best DECK in-sample per-contract Sharpe is chosen causally, then "
        "read once on the unseen sample against fixed 11:00"
    )
    print(family.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # -------------------------------------------------- the one paragraph --
    print("\n--- G. one paragraph, numbers only")
    aft = ENTRIES.index(AFTERNOON_FROM)
    qs = sorted(set(quarters))
    for name, _ in LOOKBACKS:
        pk = picks[(name, "unconditional")]
        modal_i: list[float] = []
        for qq in qs:
            v = pk[np.asarray(quarters == qq)].dropna()
            modal_i.append(
                float(ENTRIES.index(str(v.mode().iloc[0]))) if len(v) else float("nan")
            )
        mi = np.array(modal_i)
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
        mean_clock = float(np.nanmean(mi))
        print(
            f"  {name:14s} modal clock index (0 = {ENTRIES[0]}, {aft} = "
            f"{AFTERNOON_FROM}): mean {mean_clock:.2f}, first afternoon-modal quarter "
            f"{first_aft}, afternoon-modal quarters {n_aft}/{len(qs)}; best lag vs "
            "the per-quarter ex-post ceiling "
            + ", ".join(
                f"{t} {lag} quarters (mean |clock index| gap {err:.2f})"
                for t, lag, err in lags
            )
        )
    m25 = np.asarray(idx.year == 2025)
    print(f"  2025 ({int(m25.sum())} sessions):")
    for name, _ in LOOKBACKS:
        for treat in TREATMENTS:
            u = p43.pick_series(
                books, picks[(name, "unconditional")], treat, "pts", "pnl"
            )[m25]
            th = p43.pick_series(
                books, picks[(name, "thresholded")], treat, "pts", "pnl"
            )[m25]
            n_u = int(
                traded_mask(books, picks[(name, "unconditional")], treat)[m25].sum()
            )
            n_t = int(
                traded_mask(books, picks[(name, "thresholded")], treat)[m25].sum()
            )
            print(
                f"    {name:14s} {treat:8s} unconditional n {n_u} mean "
                f"{float(u.mean()):+.4f} pts (Sharpe {p43.sharpe(u):+.4f}); "
                f"thresholded n {n_t} mean {float(th.mean()):+.4f} pts (Sharpe "
                f"{p43.sharpe(th):+.4f}); sessions the threshold refuses "
                f"{int(m25.sum()) - n_t}"
            )
    qb25 = {name: float(qbars[name][m25].mean()) for name, _ in LOOKBACKS}
    real25: dict[str, float] = {}
    for name, _ in LOOKBACKS:
        pk25 = picks[(name, "unconditional")][m25]
        vals = [
            float(q.loc[d, c]) if c in ENTRIES else np.nan
            for d, c in zip(idx[m25], pk25, strict=True)
        ]
        real25[name] = float(np.nanmean(vals))
    print(
        "  mean trailing q-bar at the selected clock in 2025: "
        + ", ".join(f"{k} {v:+.4f}" for k, v in qb25.items())
    )
    print(
        "  mean REALISED q at the selected clock in 2025: "
        + ", ".join(f"{k} {v:+.4f}" for k, v in real25.items())
    )

    g3.to_csv(OUT / "gate3_p26_reproduction.csv", index=False)
    counts.to_csv(OUT / "pick_counts.csv")
    modal_u.to_csv(OUT / "modal_clock_by_quarter_unconditional.csv")
    modal_t.to_csv(OUT / "modal_clock_by_quarter_thresholded.csv")
    modal_26.to_csv(OUT / "modal_clock_by_quarter_p26.csv")
    best_q_tab.to_csv(OUT / "ex_post_best_clock_by_quarter.csv")
    q_period.to_csv(OUT / "q_mean_by_clock_period.csv")
    print(
        f"\nwrote {OUT}: selector_timeline.csv, rules.csv, paired.csv, "
        "bootstrap.csv, placebo.csv, gate.csv, family_verdict.csv, "
        "gate3_p26_reproduction.csv, pick_counts.csv, "
        "modal_clock_by_quarter_unconditional.csv, "
        "modal_clock_by_quarter_thresholded.csv, modal_clock_by_quarter_p26.csv, "
        "ex_post_best_clock_by_quarter.csv, q_mean_by_clock_period.csv"
    )
    print(f"total runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
