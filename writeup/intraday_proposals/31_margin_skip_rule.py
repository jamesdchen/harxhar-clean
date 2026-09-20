"""31 - a margin of safety on the 11:00 signal: how rich is rich enough?

Proposal 30 asked which days the seller should sit out and answered with the
SIGN of the window-matched signal.  This proposal replaces the sign with a
MARGIN.  At 11:00 the seller compares the implied variance slice it is paid
for against the forecast of the realized variance over that same slice, and
sells only when the implied sits at least theta above the forecast in
log-volatility terms.  The premium is the paper's own price-form quantity,

    v_f = 0.5 * ln( iv_var_matched / rv_hat_f )  =  ln( IV / sqrt(RV_hat_f) )

which experiments/spxw_delta_hedged_legs.py:311 writes as log(iv / mvol_tag)
and experiments/spxw_dh_regime_exit.py:132 turns into a position by selling
when it exceeds EXIT_THETA = 0.10.  Here iv_var_matched = IV_hr^2 h_rem
w_slice is the SAME 11:00 slice that proposal 30's s_matched subtracts the
forecast from, so "v_f >= 0" and "s_matched <= 0" are the same event: theta
= 0 reproduces proposal 30's forecast skip rule exactly, and the gate below
asserts it day by day and number by number against the persisted table.

  Rule(f, theta):  SELL the straddle at 11:00 iff v_f >= theta, else sit out.

Nothing is ever long the straddle, so a sat-out day contributes exactly zero
and every rule is scored on the same CALENDAR days.  A day with no forecast
or no implied slice SELLS, which is proposal 30's always-sell default.

Two reference rows keep every day:

  always sell     the book itself (theta = -infinity);
  size by margin  position = clip(v, 0, cap) / the median of the past
                  WARMUP_SESSIONS days' clipped v, both lagged and so
                  causal, with cap = the expanding lagged
                  SIZE_CAP_QUANTILE of v - a weakly rich day is then
                  traded small instead of skipped.  Scored per calendar
                  day by Sharpe (invariant to the overall size) AND by
                  the mean, which is not.

The gate is proposal 30's, imported and unchanged: the three books are
rebuilt and asserted against their published numbers, the 11:00 signal is
asserted against the notebook's persisted buy share, and nothing new is
computed until both pass.  Selection uses proposal 30's purged, embargoed
walk-forward; significance uses proposal 30's run-length-preserving block
placebo, re-implemented here only to VECTORISE it - the draws are asserted
identical to proposal 30's own, seed for seed.

Run:  python writeup/intraday_proposals/31_margin_skip_rule.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import atm_straddle_lib as asl  # noqa: E402


# The repo's read-only import: a script is imported by path, never edited.
# (The same six lines as proposal 30's _load_module, copied because they are
# what loads proposal 30 itself.)
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
p30 = _load_module(HERE / "30_skip_day_rule.py", "p31_p30")
p27 = _load_module(HERE / "27_tradeable_polish.py", "p31_p27")

REPO = p30.REPO
HOLD = p30.HOLD
OUT = HOLD / "proposals" / "31"

ANN = p30.ANN
ENTRY = p30.ENTRY
# proposal 30's constants, reused verbatim so the two proposals are scored on
# the same day sets against the same null: a 63-session warm-up, a 5-day
# embargo, 2000 block placebos at seed 0, adoption at p < 0.05.
WARMUP_SESSIONS = p30.WARMUP_SESSIONS
EMBARGO_DAYS = p30.EMBARGO_DAYS
N_PLACEBO = p30.N_PLACEBO
PLACEBO_SEED = p30.PLACEBO_SEED
PLACEBO_ALPHA = p30.PLACEBO_ALPHA

# The margin grid, stated once and never tuned inside the script.  0 is the
# sign rule proposal 30 already tested; 0.10 is the threshold the paper's
# delta-hedged book runs at (EXIT_THETA in experiments/spxw_dh_regime_exit.py).
THETA_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30)

# The sizing row's only free choice: the quantile of v that caps the
# position, taken expanding and lagged like every other statistic here.
SIZE_CAP_QUANTILE = 0.95

BOOK_ORDER = ("exit_crossed_quoted", "exit_mid_model", "hold_settle")
SELECTIONS = ("full_grid", "theta_only_blk2", "theta_only_a0")


# ------------------------------------------------------------- the rows ----
def row_stats(r: np.ndarray, w: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """Calendar-day statistics of one book under one position series.

    w is the position: 1 on a day the rule sells, 0 on a day it sits out,
    and for the sizing row any non-negative number.  The scored series is
    w * r on EVERY calendar day, so a sat-out day is a zero and the mean is
    directly comparable with always-sell.  With w in {0, 1} this returns
    proposal 30's rule_stats exactly, which the gate asserts.
    """
    ok = np.isfinite(r) & np.isfinite(w)
    rr = np.asarray(r, dtype=float)[ok]
    ww = np.asarray(w, dtype=float)[ok]
    dd = dates[ok]
    cal = rr * ww
    traded = ww > 0
    sd = float(cal.std(ddof=1)) if cal.size >= 2 else float("nan")
    mean = float(cal.mean()) if cal.size else float("nan")
    live = bool(sd and np.isfinite(sd) and sd > 0)
    worst_i = int(np.argmin(cal)) if cal.size else -1
    return {
        "n_days": int(cal.size),
        "n_traded": int(traded.sum()),
        "n_satout": int((~traded).sum()),
        "frac_satout": float((~traded).mean()) if cal.size else float("nan"),
        "mean_traded": float(cal[traded].mean()) if traded.any() else float("nan"),
        "mean_calendar": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if live else float("nan"),
        "t": float(np.sqrt(cal.size)) * mean / sd if live else float("nan"),
        "MaxDD": p30.maxdd(cal),
        "worst_day": float(cal.min()) if cal.size else float("nan"),
        "worst_date": str(pd.Timestamp(dd[worst_i]).date()) if worst_i >= 0 else "",
        "mean_satout": float(rr[~traded].mean()) if (~traded).any() else float("nan"),
        "mean_weight": float(ww.mean()) if ww.size else float("nan"),
        "mean_weight_traded": float(ww[traded].mean())
        if traded.any()
        else float("nan"),
    }


def band_stats(r: np.ndarray, v: np.ndarray, lo: float, hi: float) -> dict:
    """The book on the days with v in [lo, hi) - what the next theta removes."""
    m = np.isfinite(r) & np.isfinite(v) & (v >= lo) & (v < hi)
    return {
        "band_lo": lo,
        "band_hi": hi,
        "n_band": int(m.sum()),
        "mean_band": float(np.asarray(r, dtype=float)[m].mean())
        if m.any()
        else float("nan"),
    }


# ------------------------------------------------------------ the family ----
def margin(sig: pd.DataFrame, tag: str) -> pd.Series:
    """v_f = 0.5 ln(iv_var_matched / rv_hat_f) on the book's own days.

    proposal 30's signals_1100 carries the matched implied slice and
    s_matched = rv_hat - slice for every forecast, so rv_hat = slice +
    s_matched and no part of the 11:00 stamp is rebuilt here.
    """
    slice_ = sig["slice"].astype(float)
    rv_hat = slice_ + sig[f"s_{tag}"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = 0.5 * np.log(slice_ / rv_hat)
    return pd.Series(np.asarray(v, dtype=float), index=sig.index, name=f"v_{tag}")


def cell_weight(v: pd.Series, theta: float) -> np.ndarray:
    """1 on the days the rule sells, 0 on the days it sits out.

    SELL iff v >= theta.  A day whose v is not finite - no forecast, no
    implied slice, a censored implied that leaves the ratio undefined -
    sells, which is the always-sell default proposal 30 uses.
    """
    a = v.to_numpy(float)
    sell = ~(np.isfinite(a) & (a < theta))
    return sell.astype(float)


def size_by_margin(v: pd.Series) -> tuple[np.ndarray, dict]:
    """position = clip(v, 0, cap) / median of the past 63 days' clipped v.

    cap is the expanding lagged SIZE_CAP_QUANTILE of v and the median is a
    trailing WARMUP_SESSIONS-day median of the clipped series, shifted one
    day: both are functions of strictly earlier sessions, so the position is
    known at 11:00.  While the cap is still warming up the clip has no upper
    arm (no invented level); while the median is still warming up, or is not
    positive, or v is not finite, the position is the always-sell unit 1.
    """
    cap = p30.expanding_lagged_quantile(v, SIZE_CAP_QUANTILE, WARMUP_SESSIONS)
    vc0 = v.clip(lower=0.0)
    vc = pd.Series(
        np.minimum(vc0.to_numpy(float), cap.fillna(np.inf).to_numpy(float)),
        index=v.index,
    )
    denom = vc.rolling(WARMUP_SESSIONS, min_periods=WARMUP_SESSIONS).median().shift(1)
    usable = denom.notna() & (denom > 0) & v.notna()
    w = (vc / denom).where(usable, 1.0)
    info = {
        "n_warmup_unit": int(denom.isna().sum()),
        "n_zero_denom_unit": int((denom.notna() & (denom <= 0)).sum()),
        "n_capped": int((vc0 > cap).fillna(False).sum()),
        "mean_weight": float(w.mean()),
        "q95_weight": float(w.quantile(0.95)),
        "max_weight": float(w.max()),
        "median_cap": float(cap.median(skipna=True)),
        "median_denom": float(denom.median(skipna=True)),
    }
    return w.to_numpy(float), info


def build_cells(
    sig: pd.DataFrame, idx: pd.DatetimeIndex
) -> tuple[list[dict], list[dict]]:
    """Every (forecast, theta) cell, plus the two rows that keep every day."""
    vs = {tag: margin(sig, tag) for tag in asl.MODEL_ORDER}
    cells: list[dict] = []
    for tag in asl.MODEL_ORDER:
        v = vs[tag]
        for k, theta in enumerate(THETA_GRID):
            hi = THETA_GRID[k + 1] if k + 1 < len(THETA_GRID) else float("inf")
            w = cell_weight(v, float(theta))
            cells.append(
                {
                    "family": "margin",
                    "forecast": tag,
                    "theta": float(theta),
                    "name": f"sell only when v >= {theta:.2f}, {asl.YHAT_LABEL[tag]}",
                    "v": v.to_numpy(float),
                    "w": w,
                    "skip": w == 0.0,
                    "band": (float(theta), float(hi)),
                    "info": {},
                }
            )
    refs: list[dict] = [
        {
            "family": "reference",
            "forecast": "",
            "theta": float("nan"),
            "name": "always sell (no margin)",
            "v": np.full(len(idx), np.nan),
            "w": np.ones(len(idx), dtype=float),
            "skip": np.zeros(len(idx), dtype=bool),
            "band": (float("nan"), float("nan")),
            "info": {},
        }
    ]
    for tag in asl.MODEL_ORDER:
        w, info = size_by_margin(vs[tag])
        refs.append(
            {
                "family": "reference",
                "forecast": tag,
                "theta": float("nan"),
                "name": f"size by margin, {asl.YHAT_LABEL[tag]}",
                "v": vs[tag].to_numpy(float),
                "w": w,
                "skip": w == 0.0,
                "band": (float("nan"), float("nan")),
                "info": info,
            }
        )
    return cells, refs


# ------------------------------------------------ the vectorised placebo ----
def block_shuffled_masks(
    skip: np.ndarray, n_draw: int, rng: np.random.Generator
) -> np.ndarray:
    """proposal 30's block placebo, same draws, without its per-draw Python.

    The mask is cut into maximal runs; the sat-out run lengths and the traded
    run lengths are each permuted and re-interleaved from the state the real
    mask starts in, so every draw sits out exactly as many days in runs of
    exactly the same lengths.  The permutations are drawn in proposal 30's
    order from the same Generator, so the masks are identical draw for draw
    (the gate asserts it); only the interleave and the fill are vectorised,
    the fill through a first-difference array.
    """
    m = np.asarray(skip, dtype=bool)
    n = m.size
    edges = np.flatnonzero(np.diff(m.astype(np.int8)) != 0) + 1
    runs = np.split(np.arange(n), edges)
    lens = np.array([len(r) for r in runs])
    states = np.array([bool(m[r[0]]) for r in runs])
    len_on = lens[states]
    len_off = lens[~states]
    first = bool(m[0])
    n_on, n_off = int(len_on.size), int(len_off.size)
    on = np.empty((n_draw, n_on), dtype=np.int64)
    off = np.empty((n_draw, n_off), dtype=np.int64)
    for b in range(n_draw):
        on[b] = rng.permutation(len_on)
        off[b] = rng.permutation(len_off)
    n_runs = n_on + n_off
    pos_on = np.arange(0 if first else 1, n_runs, 2)
    pos_off = np.arange(1 if first else 0, n_runs, 2)
    assert pos_on.size == n_on and pos_off.size == n_off, "the runs do not alternate"
    lmat = np.empty((n_draw, n_runs), dtype=np.int64)
    lmat[:, pos_on] = on
    lmat[:, pos_off] = off
    ends = np.cumsum(lmat, axis=1)
    starts = ends - lmat
    rows = np.repeat(np.arange(n_draw)[:, None], n_on, axis=1)
    d = np.zeros((n_draw, n + 1), dtype=np.int16)
    np.add.at(d, (rows, starts[:, pos_on]), 1)
    np.add.at(d, (rows, ends[:, pos_on]), -1)
    out = np.cumsum(d, axis=1)[:, :n] > 0
    assert (out.sum(axis=1) == m.sum()).all(), (
        "a placebo mask changed the sat-out count"
    )
    return out


def placebo_row(r: np.ndarray, skip: np.ndarray, rng: np.random.Generator) -> dict:
    """Where the rule's calendar-day Sharpe falls among block-shuffled skips."""
    ok = np.isfinite(r)
    rr = np.asarray(r, dtype=float)[ok]
    mm = np.asarray(skip, dtype=bool)[ok]
    if mm.sum() == 0 or mm.all():
        return {
            "n_placebo": 0,
            "sharpe_rule": p30.sharpe(np.where(mm, 0.0, rr)),
            "placebo_mean": float("nan"),
            "placebo_sd": float("nan"),
            "placebo_q05": float("nan"),
            "placebo_q50": float("nan"),
            "placebo_q95": float("nan"),
            "pct_rank": float("nan"),
            "p_value": float("nan"),
        }
    s_rule = p30.sharpe(np.where(mm, 0.0, rr))
    masks = block_shuffled_masks(mm, N_PLACEBO, rng)
    cal = np.where(masks, 0.0, rr[None, :])
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


# ---------------------------------------------------------------- gate ----
def gate_margin(
    cells: list[dict], books: dict, idx: pd.DatetimeIndex, sig: pd.DataFrame
) -> None:
    """theta = 0 IS proposal 30's sign rule, cell by cell and number by number."""
    p30_rules = [c for c in p30.build_rules(sig, idx, p27) if c["family"] == "forecast"]
    by_name = {c["name"]: c for c in p30_rules}
    ref = pd.read_csv(OUT.parent / "30" / "rules.csv")
    era = np.asarray(idx >= p27.DAILY_0DTE, dtype=bool)
    samples = (("whole", np.ones(len(idx), dtype=bool)), ("daily_era", era))
    n_checked = 0
    for tag in asl.MODEL_ORDER:
        cell = next(c for c in cells if c["forecast"] == tag and c["theta"] == 0.0)
        name30 = f"sit out when s_matched(11:00) > 0, {asl.YHAT_LABEL[tag]}"
        m30 = by_name[name30]["skip"]
        assert (cell["skip"] == m30).all(), (
            f"{tag}: theta=0 is not proposal 30's sign rule"
        )
        for bk in BOOK_ORDER:
            r = books[bk].to_numpy(float)
            for smp, m in samples:
                mine = row_stats(r[m], cell["w"][m], idx[m])
                theirs = p30.rule_stats(r[m], m30[m], idx[m])
                for k, want in theirs.items():
                    have = mine[k]
                    if isinstance(want, str):
                        assert have == want, (tag, bk, smp, k, have, want)
                    elif np.isfinite(want):
                        assert abs(have - want) < 1e-12, (tag, bk, smp, k, have, want)
                    else:
                        assert not np.isfinite(have), (tag, bk, smp, k, have, want)
                row = ref[
                    (ref["book"] == bk)
                    & (ref["sample"] == smp)
                    & (ref["rule"] == name30)
                ]
                assert len(row) == 1, (bk, smp, name30)
                for k in (
                    "n_days",
                    "n_traded",
                    "mean_calendar",
                    "sd",
                    "Sharpe_ann",
                    "t",
                    "MaxDD",
                ):
                    assert abs(float(row.iloc[0][k]) - mine[k]) < 1e-12, (bk, smp, k)
                n_checked += 1
    print(
        f"\nGATE - margin at theta = 0: all {len(asl.MODEL_ORDER)} forecasts reproduce proposal "
        f"30's sign rule day for day, and {n_checked} book x sample rows of "
        f"proposals/30/rules.csv to 1e-12"
    )

    # the vectorised placebo draws the SAME masks as proposal 30's loop
    probe = next(c for c in cells if c["forecast"] == "blk2" and c["theta"] == 0.0)[
        "skip"
    ]
    a = block_shuffled_masks(probe, N_PLACEBO, np.random.default_rng(PLACEBO_SEED))
    b = p30.block_shuffled_masks(probe, N_PLACEBO, np.random.default_rng(PLACEBO_SEED))
    assert (a == b).all(), (
        "the vectorised placebo does not reproduce proposal 30's draws"
    )
    r0 = books["exit_crossed_quoted"].to_numpy(float)
    pa = placebo_row(r0, probe, np.random.default_rng(PLACEBO_SEED))
    pb = p30.placebo_row(r0, probe, np.random.default_rng(PLACEBO_SEED))
    for k2, want2 in pb.items():
        assert abs(pa[k2] - want2) < 1e-12, (k2, pa[k2], want2)
    print(
        f"GATE - placebo: {N_PLACEBO} vectorised draws identical to proposal 30's, seed "
        f"{PLACEBO_SEED}, and the p-value it renders matches to 1e-12"
    )
    print("GATE PASSED\n")


# ---------------------------------------------------------------- main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    books, idx = p30.gate(p27, pkg)
    sig = p30.signals_1100(pkg, idx)
    cells, refs = build_cells(sig, idx)
    gate_margin(cells, books, idx, sig)

    era = np.asarray(idx >= p27.DAILY_0DTE, dtype=bool)
    samples = (("whole", np.ones(len(idx), dtype=bool)), ("daily_era", era))
    v_blk2 = margin(sig, "blk2").to_numpy(float)
    qs = np.nanquantile(v_blk2, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"margin v = 0.5 ln(implied slice / forecast) at {ENTRY}, block-diagonal ridge: finite on "
        f"{int(np.isfinite(v_blk2).sum())} of {len(idx)} days, quantiles 5/25/50/75/95% "
        + "/".join(f"{q:+.4f}" for q in qs)
    )
    print(
        f"\nsize by margin: position = clip(v, 0, cap) / the trailing {WARMUP_SESSIONS}-day median "
        f"of the clipped v, both lagged; cap = the expanding lagged "
        f"{SIZE_CAP_QUANTILE:.0%} quantile of v.  The median is exactly zero whenever at least "
        f"half of the trailing window is not rich, and the position is then the always-sell unit; "
        f"where it is positive it is not bounded away from zero, so the position is not bounded "
        f"above.  Both counts, and the size the definition actually produces, are stated here."
    )
    for ref in refs:
        if ref["info"]:
            i = ref["info"]
            print(
                f"  {ref['name']:<46s} unit size on {i['n_warmup_unit']} warm-up days and "
                f"{i['n_zero_denom_unit']} zero-median days, capped on {i['n_capped']}; "
                f"size mean {i['mean_weight']:.3f}, 95% {i['q95_weight']:.3f}, "
                f"max {i['max_weight']:.3f}; median cap {i['median_cap']:.4f}, "
                f"median denominator {i['median_denom']:.4f}"
            )

    # ------------------------------------------------------- rules.csv ----
    rows = []
    for bk in BOOK_ORDER:
        r_full = books[bk].to_numpy(float)
        for smp, m in samples:
            for c in cells + refs:
                lo, hi = c["band"]
                row = {
                    "book": bk,
                    "sample": smp,
                    "family": c["family"],
                    "forecast": c["forecast"],
                    "theta": c["theta"],
                    "rule": c["name"],
                }
                row.update(row_stats(r_full[m], c["w"][m], idx[m]))
                row.update(band_stats(r_full[m], c["v"][m], lo, hi))
                rows.append(row)
    rules_df = pd.DataFrame(rows)
    rules_df.to_csv(OUT / "rules.csv", index=False)
    rcols = [
        "family",
        "forecast",
        "theta",
        "n_traded",
        "frac_satout",
        "mean_traded",
        "mean_calendar",
        "sd",
        "Sharpe_ann",
        "t",
        "MaxDD",
        "worst_day",
        "mean_satout",
        "band_lo",
        "band_hi",
        "n_band",
        "mean_band",
        "mean_weight",
    ]
    for bk in BOOK_ORDER:
        for smp, _ in samples:
            sub = rules_df[(rules_df["book"] == bk) & (rules_df["sample"] == smp)]
            p30.show(sub[rcols], f"rules.csv  |  book {bk}  |  sample {smp}")

    # --------------------------------------------------------- oos.csv ----
    null_w = np.ones(len(idx), dtype=float)
    null = {
        "family": "reference",
        "forecast": "",
        "theta": float("nan"),
        "name": "always sell (no margin)",
        "skip": np.zeros(len(idx), dtype=bool),
        "w": null_w,
    }
    grids = {
        "full_grid": cells,
        "theta_only_blk2": [c for c in cells if c["forecast"] == "blk2"],
        "theta_only_a0": [c for c in cells if c["forecast"] == "a0"],
    }
    oos_rows = []
    selected: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for bk in BOOK_ORDER:
        r_full = books[bk].to_numpy(float)
        scored_any, _, _ = p30.walk_forward(r_full, [null])
        for sel_name in SELECTIONS:
            cands = [null] + grids[sel_name]
            scored, wf_skip, pick = p30.walk_forward(r_full, cands)
            selected[(bk, sel_name)] = (scored, wf_skip)
            names = np.array([c["name"] for c in cands])
            for smp, m in samples:
                sel = scored & m
                if not sel.any():
                    continue
                st = row_stats(r_full[sel], (~wf_skip[sel]).astype(float), idx[sel])
                base = row_stats(r_full[sel], null_w[sel], idx[sel])
                picks = pd.Series(names[pick[sel]]).value_counts()
                oos_rows.append(
                    {
                        "block": "walk_forward_selected",
                        "book": bk,
                        "sample": smp,
                        "selection": sel_name,
                        "family": "margin",
                        "forecast": "",
                        "theta": float("nan"),
                        "rule": f"{sel_name}: (forecast, theta) chosen on an expanding window, "
                        f"{EMBARGO_DAYS}-day embargo",
                        **st,
                        "Sharpe_always_sell": base["Sharpe_ann"],
                        "delta_Sharpe": st["Sharpe_ann"] - base["Sharpe_ann"],
                        "mean_always_sell": base["mean_calendar"],
                        "beats_always_sell": bool(
                            st["Sharpe_ann"] > base["Sharpe_ann"]
                        ),
                        "modal_pick": str(picks.index[0]),
                        "modal_pick_share": float(picks.iloc[0] / picks.sum()),
                        "frac_null_chosen": float((pick[sel] == 0).mean()),
                    }
                )
        for smp, m in samples:
            sel = scored_any & m
            base = row_stats(r_full[sel], null_w[sel], idx[sel])
            for c in cells + refs:
                st = row_stats(r_full[sel], c["w"][sel], idx[sel])
                oos_rows.append(
                    {
                        "block": "fixed_cell_oos",
                        "book": bk,
                        "sample": smp,
                        "selection": "",
                        "family": c["family"],
                        "forecast": c["forecast"],
                        "theta": c["theta"],
                        "rule": c["name"],
                        **st,
                        "Sharpe_always_sell": base["Sharpe_ann"],
                        "delta_Sharpe": st["Sharpe_ann"] - base["Sharpe_ann"],
                        "mean_always_sell": base["mean_calendar"],
                        "beats_always_sell": bool(
                            st["Sharpe_ann"] > base["Sharpe_ann"]
                        ),
                        "modal_pick": c["name"],
                        "modal_pick_share": 1.0,
                        "frac_null_chosen": float("nan"),
                    }
                )
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT / "oos.csv", index=False)
    ocols = [
        "block",
        "selection",
        "family",
        "forecast",
        "theta",
        "n_days",
        "n_traded",
        "frac_satout",
        "mean_calendar",
        "mean_always_sell",
        "sd",
        "Sharpe_ann",
        "Sharpe_always_sell",
        "delta_Sharpe",
        "beats_always_sell",
        "t",
        "MaxDD",
        "worst_day",
        "modal_pick",
        "modal_pick_share",
        "frac_null_chosen",
    ]
    print(
        f"\nout-of-sample day set: expanding training window, {EMBARGO_DAYS}-day embargo, "
        f"minimum {WARMUP_SESSIONS} training days, so the first "
        f"{WARMUP_SESSIONS + EMBARGO_DAYS} days are never scored; the selection grids are "
        f"{len(cells)} cells + the null (full_grid) and {len(THETA_GRID)} + the null (theta only)"
    )
    for bk in BOOK_ORDER:
        for smp, _ in samples:
            sub = oos_df[(oos_df["book"] == bk) & (oos_df["sample"] == smp)]
            p30.show(sub[ocols], f"oos.csv  |  book {bk}  |  sample {smp}")

    # ----------------------------------------------------- placebo.csv ----
    rng = np.random.default_rng(PLACEBO_SEED)
    prows = []
    for bk in BOOK_ORDER:
        r_full = books[bk].to_numpy(float)
        scored_any, _, _ = p30.walk_forward(r_full, [null])
        for smp, m in samples:
            for day_set, sel in (("full", m), ("oos", m & scored_any)):
                for c in cells:
                    prows.append(
                        {
                            "book": bk,
                            "sample": smp,
                            "day_set": day_set,
                            "block": "fixed_cell",
                            "selection": "",
                            "family": c["family"],
                            "forecast": c["forecast"],
                            "theta": c["theta"],
                            "rule": c["name"],
                            **placebo_row(r_full[sel], c["skip"][sel], rng),
                        }
                    )
            for sel_name in SELECTIONS:
                wf_scored, wf = selected[(bk, sel_name)]
                sel = m & wf_scored
                prows.append(
                    {
                        "book": bk,
                        "sample": smp,
                        "day_set": "oos",
                        "block": "walk_forward_selected",
                        "selection": sel_name,
                        "family": "margin",
                        "forecast": "",
                        "theta": float("nan"),
                        "rule": f"{sel_name}: (forecast, theta) chosen on an expanding window, "
                        f"{EMBARGO_DAYS}-day embargo",
                        **placebo_row(r_full[sel], wf[sel], rng),
                    }
                )
    plac_df = pd.DataFrame(prows)
    plac_df.to_csv(OUT / "placebo.csv", index=False)
    pcols = [
        "day_set",
        "block",
        "selection",
        "forecast",
        "theta",
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
        f"\nplacebo: {N_PLACEBO} block-shuffled skip masks per cell (same sat-out count, same "
        f"run-length multiset), seed {PLACEBO_SEED}; "
        f"p = (1 + #(placebo Sharpe >= rule Sharpe)) / (1 + n_placebo).  The sizing row has no "
        f"placebo: its null is not a permutation of a skip mask."
    )
    for bk in BOOK_ORDER:
        for smp, _ in samples:
            sub = plac_df[(plac_df["book"] == bk) & (plac_df["sample"] == smp)]
            p30.show(sub[pcols], f"placebo.csv  |  book {bk}  |  sample {smp}")

    # ---------------------------------------------------------- gate ----
    key = ["book", "sample", "family", "rule"]
    cell_oos = oos_df[
        (oos_df["block"] == "fixed_cell_oos") & (oos_df["family"] == "margin")
    ]
    pl = plac_df.loc[(plac_df["day_set"] == "oos") & (plac_df["block"] == "fixed_cell")]
    g = cell_oos.merge(
        pl[key + ["p_value", "pct_rank", "sharpe_rule"]], on=key, how="left"
    )
    assert not g["p_value"].isna().any(), "a cell lost its placebo leg in the merge"
    same = g["sharpe_rule"].notna() & g["Sharpe_ann"].notna()
    assert np.allclose(
        g.loc[same, "sharpe_rule"], g.loc[same, "Sharpe_ann"], rtol=0, atol=1e-12
    ), "the placebo and the out-of-sample table disagree on the cell's own Sharpe"
    g["adopted"] = g["beats_always_sell"] & (g["p_value"] < PLACEBO_ALPHA)
    g.to_csv(OUT / "gate.csv", index=False)
    gcols = [
        "book",
        "sample",
        "forecast",
        "theta",
        "n_traded",
        "Sharpe_ann",
        "Sharpe_always_sell",
        "delta_Sharpe",
        "beats_always_sell",
        "p_value",
        "adopted",
    ]
    for bk in BOOK_ORDER:
        for smp, _ in samples:
            sub = g[(g["book"] == bk) & (g["sample"] == smp)]
            p30.show(
                sub[gcols],
                f"GATE cells  |  book {bk}  |  sample {smp}  |  adopted = out-of-sample beats "
                f"always-sell AND placebo p < {PLACEBO_ALPHA}",
            )

    fam = oos_df[oos_df["block"] == "walk_forward_selected"].merge(
        plac_df.loc[plac_df["block"] == "walk_forward_selected"][
            ["book", "sample", "selection", "p_value"]
        ],
        on=["book", "sample", "selection"],
        how="left",
    )
    fam.to_csv(OUT / "gate_family.csv", index=False)
    p30.show(
        fam[
            [
                "book",
                "sample",
                "selection",
                "Sharpe_ann",
                "Sharpe_always_sell",
                "delta_Sharpe",
                "beats_always_sell",
                "p_value",
                "modal_pick",
                "modal_pick_share",
                "frac_null_chosen",
            ]
        ],
        "GATE family  |  the walk-forward selection is adopted only if it beats always-sell",
    )

    n_cells = len(cells)
    n_gated = n_cells * len(BOOK_ORDER) * len(samples)
    print(
        f"\nmultiple testing: {n_cells} cells ({len(asl.MODEL_ORDER)} forecasts x "
        f"{len(THETA_GRID)} thresholds) are gated on each of {len(BOOK_ORDER)} books x "
        f"{len(samples)} samples, {n_gated} gated cells in all.  At a {PLACEBO_ALPHA:.0%} bar, "
        f"and if every cell were null, {n_cells * PLACEBO_ALPHA:.1f} cells would pass the placebo "
        f"leg by chance on one book and sample and {n_gated * PLACEBO_ALPHA:.1f} across the whole "
        f"table; nothing here is corrected for that."
    )
    prim = g[(g["book"] == "exit_crossed_quoted") & g["adopted"]]
    print(
        f"\ncells adopted on any book or sample: {int(g['adopted'].sum())} of {len(g)}"
    )
    print(f"cells adopted on the primary book (exit_crossed_quoted): {int(len(prim))}")
    if len(prim):
        p30.show(prim[gcols], "adopted on the primary book")
    else:
        print("no margin cell survives the gate on the primary book")
    fam_prim = fam[fam["book"] == "exit_crossed_quoted"]
    print(
        "walk-forward family verdict on the primary book: "
        + "; ".join(
            f"{r.selection} ({r.sample}) "
            f"{'beats' if r.beats_always_sell else 'does not beat'} always-sell, "
            f"delta {r.delta_Sharpe:+.4f}"
            for r in fam_prim.itertuples()
        )
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
