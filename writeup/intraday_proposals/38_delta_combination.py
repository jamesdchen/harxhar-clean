"""38 - the hedge delta under the combination's view of THIS bar.

Proposal 36 asked which volatility the delta of the 11:00 straddle is computed
with, and found exactly one causal answer that the era bootstrap separates
from the book: V9, the implied remaining-window variance multiplied by the
expanding, one-session-lagged mean of realized over implied at that stamp.
V9 is a per-clock CONSTANT discount - a number that knows the hour and
nothing about the day.  Proposal 35 built, and gated, a causal NEXT-BAR
forecaster: the per-clock log-fit combination

    f_comb(t, d) = exp(a_c) rv_hat(t, d)^{b_c} slice(t, d)^{1 - b_c}
                   read as a level combination with weight w = clip(b_c, 0, 1),

whose coefficients are fitted on strictly earlier sessions with the repo's
63-session warm-up and lagged one session (35's ``C2_log_fit_weight``), and
whose market counterpart at the same stamp is the implied next-bar slice

    slice(t, d) = iv_hourly(t, d)^2 h_rem(t) w_slice(t, d).

This proposal asks whether the DAY-LEVEL information in that combination
moves the hedge.  The object it carries into the delta is the ratio

    r(t, d) = f_comb(t, d) / slice(t, d),

the combination's view of how rich or lean THIS bar is relative to the way
the market has priced it: r = 1 is "as priced", r > 1 "the model expects more
variance over the next thirty minutes than the slice is charging".  Both
pieces are measurable at the stamp and both carry 35's own warm-up, so r is
causal by construction; the cells that have no r (the warm-up, and the stamps
whose vendor implied volatility sat on the solver's bracket node) fall back to
the variant's own base and are counted on every row.

The variants (all causal at t)
------------------------------
The book's stamp variance is ``var0 = sig^2 h_rem`` with ``sig`` the
re-inverted implied total volatility, and V9's is that variance times the
expanding lagged 63-day mean of realized-over-implied at that stamp.

  V0   implied         the book.
  V9   implied_bc      proposal 36's reference, reproduced here to the float
                       path before anything new is computed.
  V11  V9 x (1 + lam (r - 1)),  lam in 0.25, 0.50, 0.75, 1.00
                       the constant per-clock discount modulated by today's
                       bar-level richness and shrunk toward V9; positive for
                       every r > 0 and every lam <= 1, so no clip is needed.
  V12  V9 x clip(r, 1/3, 3)^lam,  lam in 0.50, 1.00
                       the multiplicative version, with the bound stated
                       rather than fitted.
  V13  V0 x r          the ratio applied to the RAW implied variance, without
                       V9's discount - the row that separates the two effects.
                       Its base is V0, so its warm-up cells fall back to V0;
                       falling back to V9 there would put the discount back in.
  V14  V9 x rbar       r replaced by its expanding, one-session-lagged mean at
                       that clock: a smoothed, day-INVARIANT version of the
                       same object.  If V14 moves the book and V11/V12 do not,
                       the gain was a second constant, not day-specific news.

Nine variants are scored against the book V0; the eight that carry r or rbar
are also scored against V9, which is the bar the verdict reads.  r at stamp t
is applied to the delta held over the bar [t, t + 30], the bar the
combination forecasts.

What is reported
----------------
On the primary book (the crossed-quoted 15:30 exit) and the two secondary
books (mid exit, hold to settlement), whole sample and daily-0DTE era: mean,
sd, annualised Sharpe, t, MaxDD, worst day and its date, hedge turnover, the
repo's 0.5 bp hedge charge (REPORTED on every row and charged to none), and
proposal 32's trend / choppy tercile means.  ``pairs.csv`` carries the paired
daily difference against V0 AND against V9 with a HAC t; ``bootstrap.csv``
carries the circular block bootstrap (B = 2000, block 21, seed 0) of the
Sharpe difference against V0 and against V9.  ``r_by_clock.csv`` reports the
distribution of r itself - 10th percentile, median, 90th percentile per stamp
- so the reader can see how far the combination actually moves the discount
from one day to the next, and how often the V12 bound binds.

The gate
--------
Nothing new is computed until three published objects are reproduced: (1)
proposal 32's book (865 expirations, crossed Sharpe 2.252546754993 whole and
2.513132263902 era), its hedge identity against ``live.ibkr.parity`` and its
shape terciles; (2) proposal 36's V9 - its era Sharpe difference and bootstrap
interval, its whole-sample pair, and both HAC t statistics; (3) proposal 35's
pooled QLIKE of the combination, whole and era.

The verdict bar
---------------
A variant "improves on V9" only if its era bootstrap interval of the Sharpe
difference AGAINST V9, on the primary book, excludes zero.  The interval is
uncorrected for multiplicity and the script says what that costs.

Run:  python writeup/intraday_proposals/38_delta_combination.py
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
p35 = _load_module(HERE / "35_combined_forecast_toggle.py", "p38_p35")
p36 = _load_module(HERE / "36_remaining_window_honest.py", "p38_p36")
p30 = p36.p30
p32 = p36.p32
asl = p30.asl

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "38"

ENTRY = p32.ENTRY
CLOSE = p32.CLOSE
SESSION: tuple[str, ...] = p32.SESSION
H_REM: tuple[float, ...] = p32.H_REM
ERA0 = p32.ERA0

#: The panel's twelve scored trade bars, bar-START labelled (proposal 36's).
TRADE_CLOCKS: tuple[str, ...] = p36.TRADE_CLOCKS
#: The forecast panel the combination and the book are both built on.
HEADLINE_TAG = p36.HEADLINE_TAG
#: Proposal 35's causal combination, and its market counterpart at the stamp.
COMBINATION = "C2_log_fit_weight"
SLICE = "implied slice"

#: The repo's expanding warm-up, 63 sessions.
WARMUP = p36.WARMUP

#: V11's shrinkage grid: V9 x (1 + lam (r - 1)).
LAM_V11: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
#: V12's exponent grid: V9 x clip(r, R_CLIP_LO, R_CLIP_HI)^lam.
LAM_V12: tuple[float, ...] = (0.5, 1.0)
#: V12's stated bound on the richness ratio.
R_CLIP_LO = 1.0 / 3.0
R_CLIP_HI = 3.0

#: The books this proposal scores.  The first is the primary.
BOOKS: tuple[str, ...] = p36.BOOKS
PRIMARY_BOOK = p36.PRIMARY_BOOK
SAMPLES: tuple[str, ...] = p36.SAMPLES
#: The sample the verdict is read on.
VERDICT_SAMPLE = p36.VERDICT_SAMPLE
#: The variant the verdict is read against.
VERDICT_REFERENCE = "V9_implied_bc"
#: The book itself, the other reference every difference is also reported against.
BOOK_REFERENCE = "V0_implied"

#: The deck's circular block bootstrap: 2000 draws, 21-day blocks, seed 0.
BOOT_B = p36.BOOT_B
BOOT_BLOCK = p36.BOOT_BLOCK
BOOT_SEED = p36.BOOT_SEED
CI_PCT: tuple[float, float] = p36.CI_PCT
#: The percentiles the richness ratio is reported at.
R_PCT: tuple[float, float, float] = (10.0, 50.0, 90.0)

#: Published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = p32.GATE_N_DAYS
GATE_TOL = p32.GATE_TOL
#: Proposal 36's V9 on the primary book
#: (results/atm_straddle_intraday_holdclose/proposals/36/bootstrap.csv, pairs.csv).
GATE_V9_DSHARPE_WHOLE = 0.1309569244741494
GATE_V9_CI_WHOLE: tuple[float, float] = (0.05266980309665783, 0.21936214098613388)
GATE_V9_DSHARPE_ERA = 0.17555194627816872
GATE_V9_CI_ERA: tuple[float, float] = (0.06328949647804978, 0.3005113715778494)
GATE_V9_HAC_T_WHOLE = 2.7390318414673698
GATE_V9_HAC_T_ERA = 2.540921506042825
#: Proposal 35's pooled QLIKE of the combination
#: (results/atm_straddle_intraday_holdclose/proposals/35/a_qlike_pooled.csv).
GATE_QLIKE_COMB_WHOLE = 0.15051372940605923
GATE_QLIKE_COMB_ERA = 0.138980797841277
#: Every gated object here is rebuilt on the SAME float path as the published
#: one, so the bar is the float-path bar, not the printed-precision bar.
EXACT_TOL = 1e-12


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(use.to_string(index=False))


def write(df: pd.DataFrame, name: str, title: str) -> None:
    """Persist a table under the proposal's own directory."""
    df.to_csv(OUT / name, index=False)
    print(f"\nwrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


# --------------------------------------------------------------- the grids --
def remaining_grid(dates: pd.DatetimeIndex, cols: list[str]) -> np.ndarray:
    """The variance realized from each book stamp to the 16:00 close.

    Proposal 36's object, rebuilt by its own recipe: the panel's realized bar
    variances on the twelve bar-start-labelled trade clocks, reverse cumulated
    inclusively, then read on the book's days and stamps.
    """
    base = asl.load_yhat_panel(asl.yhat_paths(ROOT)[HEADLINE_TAG]).set_index("t")
    clocks = list(TRADE_CLOCKS)
    prof, _ = p30.panel_profile(base, clocks)
    assert list(prof.columns) == clocks, list(prof.columns)
    rv = prof[clocks].to_numpy(float)
    rem = pd.DataFrame(
        rv[:, ::-1].cumsum(axis=1)[:, ::-1], index=prof.index, columns=clocks
    )
    out = rem.reindex(index=dates, columns=cols).to_numpy(float)
    assert np.isfinite(out).all(), "a book stamp has no realized remaining window"
    print(
        f"\nthe realized remaining window: {len(prof)} panel sessions "
        f"{prof.index.min().date()} .. {prof.index.max().date()} on "
        f"{len(clocks)} trade bars, read on the book's {out.shape[0]} days x "
        f"{out.shape[1]} stamps"
    )
    return out


def censored_grid(
    panel: pd.DataFrame, dates: pd.DatetimeIndex, cols: list[str]
) -> np.ndarray:
    """The stamps whose vendor implied volatility sat on the solver's node.

    Proposal 36's mask, unchanged: a variant says nothing at such a stamp and
    hedges with its own base there.
    """
    _, rule_mod = research_modules(ROOT)
    flag = rule_mod.on_vendor_node(
        panel["impl_volatility_c"]
    ) | rule_mod.on_vendor_node(panel["impl_volatility_p"])
    out = (
        panel.assign(_cen=np.asarray(flag, dtype=float))
        .pivot_table(index="date", columns="hhmm", values="_cen", aggfunc="max")
        .reindex(index=dates, columns=cols)
        .fillna(1.0)
        .to_numpy(float)
        > 0
    )
    print(
        f"\ncensored implied volatility: {int(out.sum())} of {out.size} stamp-cells "
        f"on the book's days (those stamps hedge with the variant's own base)"
    )
    return out


def richness_grids(
    dates: pd.DatetimeIndex, cols: list[str]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Proposal 35's combination and implied slice, and the ratio r they define.

    ``35_combined_forecast_toggle.py``'s own grid builder is called, so the
    combination is the gated object and not a re-implementation: the per-clock
    log fit on strictly earlier sessions, the repo's 63-session minimum,
    lagged one session.  Returns r on the book's grid, its expanding lagged
    per-clock mean, and the two source grids for the gate.
    """
    work, clocks, _ = p35.forecast_frame()
    fc, _, y = p35.combination_grids(work, clocks)
    comb = fc[COMBINATION].reindex(index=dates, columns=cols).to_numpy(float)
    sli = fc[SLICE].reindex(index=dates, columns=cols).to_numpy(float)
    ok = np.isfinite(comb) & np.isfinite(sli) & (comb > 0.0) & (sli > 0.0)
    r = np.where(ok, comb / np.where(ok, sli, 1.0), np.nan)
    rbar = p36.expanding_lagged_mean(r, WARMUP)
    print(
        f"\nthe richness ratio r = {COMBINATION} / {SLICE} on the book's grid: "
        f"{int(np.isfinite(r).sum())} of {r.size} cells carry one (the rest are "
        f"proposal 35's {WARMUP}-session warm-up), first on "
        f"{pd.Timestamp(dates[int(np.argmax(np.isfinite(r).any(axis=1)))]).date()}; "
        f"its expanding lagged per-clock mean rbar covers "
        f"{int(np.isfinite(rbar).sum())} cells"
    )
    return r, rbar, fc[COMBINATION], y


def richness_table(
    r: np.ndarray, rbar: np.ndarray, dates: pd.DatetimeIndex, cols: list[str]
) -> pd.DataFrame:
    """How far the combination moves the discount, stamp by stamp."""
    era = np.asarray(dates >= ERA0, dtype=bool)
    masks: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )
    rows: list[dict[str, Any]] = []
    for sample, m in masks:
        for j, clock in enumerate(cols):
            v = r[m, j]
            v = v[np.isfinite(v)]
            b = rbar[m, j]
            b = b[np.isfinite(b)]
            pct = np.percentile(v, list(R_PCT)) if v.size else np.full(3, np.nan)
            rows.append(
                {
                    "sample": sample,
                    "clock": clock,
                    "h_rem": H_REM[j],
                    "n_cells_with_r": int(v.size),
                    "n_cells_without_r": int(m.sum()) - int(v.size),
                    f"r_p{R_PCT[0]:.0f}": float(pct[0]),
                    "r_median": float(pct[1]),
                    f"r_p{R_PCT[2]:.0f}": float(pct[2]),
                    "r_mean": float(v.mean()) if v.size else float("nan"),
                    "share_r_above_1": float((v > 1.0).mean())
                    if v.size
                    else float("nan"),
                    "share_r_below_clip_lo": float((v < R_CLIP_LO).mean())
                    if v.size
                    else float("nan"),
                    "share_r_above_clip_hi": float((v > R_CLIP_HI).mean())
                    if v.size
                    else float("nan"),
                    "rbar_median": float(np.median(b)) if b.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------- the variants ----
def total_vol_variants(
    var0: np.ndarray,
    v9: np.ndarray,
    n_fallback_v9: int,
    r: np.ndarray,
    rbar: np.ndarray,
    censored: np.ndarray,
) -> list[dict[str, Any]]:
    """Every total-volatility grid the hedge delta is tried with.

    Each variant falls back to its OWN base wherever its multiplier is
    missing, non-positive or the stamp's vendor implied volatility sat on the
    solver's bracket node - V9 for the rows that modulate V9, V0 for V13,
    whose whole purpose is to carry the ratio WITHOUT V9's discount.  Every
    fallback is counted on the row.
    """
    ok_r = np.isfinite(r) & (r > 0.0) & ~censored
    ok_b = np.isfinite(rbar) & (rbar > 0.0) & ~censored
    rr = np.where(ok_r, r, 1.0)
    bb = np.where(ok_b, rbar, 1.0)
    out: list[dict[str, Any]] = [
        {
            "variant": BOOK_REFERENCE,
            "family": "V0 implied",
            "lam": float("nan"),
            "base": "-",
            "description": "the book: the re-inverted implied total volatility",
            "variance": var0,
            "n_fallback": 0,
        },
        {
            "variant": VERDICT_REFERENCE,
            "family": "V9 bias-corrected implied",
            "lam": float("nan"),
            "base": BOOK_REFERENCE,
            "description": (
                f"proposal 36's reference: the implied remaining-window variance "
                f"times its expanding lagged {WARMUP}-day realized-over-implied "
                f"mean at that stamp"
            ),
            "variance": v9,
            "n_fallback": n_fallback_v9,
        },
    ]
    for lam in LAM_V11:
        out.append(
            {
                "variant": f"V11_shrunk_{lam:.2f}",
                "family": "V11 shrunk richness",
                "lam": lam,
                "base": VERDICT_REFERENCE,
                "description": f"V9 x (1 + {lam:.2f} (r - 1))",
                "variance": v9 * ((1.0 - lam) + lam * rr),
                "n_fallback": int((~ok_r).sum()),
            }
        )
    for lam in LAM_V12:
        out.append(
            {
                "variant": f"V12_power_{lam:.2f}",
                "family": "V12 bounded power",
                "lam": lam,
                "base": VERDICT_REFERENCE,
                "description": (
                    f"V9 x clip(r, {R_CLIP_LO:.4f}, {R_CLIP_HI:.4f})^{lam:.2f}"
                ),
                "variance": v9 * np.clip(rr, R_CLIP_LO, R_CLIP_HI) ** lam,
                "n_fallback": int((~ok_r).sum()),
            }
        )
    out.append(
        {
            "variant": "V13_raw_implied_x_r",
            "family": "V13 raw implied x richness",
            "lam": 1.0,
            "base": BOOK_REFERENCE,
            "description": "V0 x r (the ratio without V9's per-clock discount)",
            "variance": var0 * rr,
            "n_fallback": int((~ok_r).sum()),
        }
    )
    out.append(
        {
            "variant": "V14_smoothed_r",
            "family": "V14 smoothed richness",
            "lam": 1.0,
            "base": VERDICT_REFERENCE,
            "description": (
                f"V9 x rbar, rbar the expanding lagged per-clock mean of r "
                f"(minimum {WARMUP} sessions) - day-invariant by construction"
            ),
            "variance": v9 * bb,
            "n_fallback": int((~ok_b).sum()),
        }
    )
    for row in out:
        v = np.asarray(row["variance"], dtype=float)
        assert np.isfinite(v).all() and (v > 0.0).all(), row["variant"]
        row["total_vol"] = np.sqrt(v)
    return out


# ------------------------------------------------------------------ gates ----
def gate_v9(
    r0: dict[str, np.ndarray], r9: dict[str, np.ndarray], era: np.ndarray
) -> None:
    """Reproduce proposal 36's V9 before the new variants are built."""
    print("\nGATE - proposal 36's V9 on the primary book")
    for sample, m, d_want, ci_want, hac_want in (
        (
            "whole",
            np.ones(era.size, dtype=bool),
            GATE_V9_DSHARPE_WHOLE,
            GATE_V9_CI_WHOLE,
            GATE_V9_HAC_T_WHOLE,
        ),
        (
            "daily_era",
            era,
            GATE_V9_DSHARPE_ERA,
            GATE_V9_CI_ERA,
            GATE_V9_HAC_T_ERA,
        ),
    ):
        got = p36.sharpe_diff_ci(r9[PRIMARY_BOOK][m], r0[PRIMARY_BOOK][m])
        pair = p36.paired_row(r9[PRIMARY_BOOK][m], r0[PRIMARY_BOOK][m])
        assert abs(got["dSharpe"] - d_want) < EXACT_TOL, (sample, got["dSharpe"])
        assert abs(got["boot_lo"] - ci_want[0]) < EXACT_TOL, (sample, got["boot_lo"])
        assert abs(got["boot_hi"] - ci_want[1]) < EXACT_TOL, (sample, got["boot_hi"])
        assert abs(pair["hac_t_diff"] - hac_want) < EXACT_TOL, (
            sample,
            pair["hac_t_diff"],
        )
        print(
            f"  {sample:<10s} n {got['n']:>4d}  dSharpe vs V0 {got['dSharpe']:+.12f} "
            f"[{got['boot_lo']:+.12f}, {got['boot_hi']:+.12f}]  HAC t "
            f"{pair['hac_t_diff']:+.12f}  reference {d_want:+.6f} "
            f"[{ci_want[0]:+.6f}, {ci_want[1]:+.6f}] / {hac_want:+.6f}  OK"
        )


def gate_combination(comb: pd.DataFrame, y: pd.DataFrame) -> None:
    """Reproduce proposal 35's pooled QLIKE of the combination."""
    dates = pd.DatetimeIndex(y.index)
    yv = y.to_numpy(float)
    cv = comb.reindex(index=dates, columns=list(y.columns)).to_numpy(float)
    era = np.asarray(dates >= ERA0, dtype=bool)
    print(f"\nGATE - proposal 35's pooled QLIKE of {COMBINATION}")
    for sample, m, want in (
        ("whole", np.ones(len(dates), dtype=bool), GATE_QLIKE_COMB_WHOLE),
        ("daily_era", era, GATE_QLIKE_COMB_ERA),
    ):
        q, n = p35.qlike_mean(
            np.where(m[:, None], yv, np.nan).ravel(),
            np.where(m[:, None], cv, np.nan).ravel(),
        )
        assert abs(q - want) < EXACT_TOL, (sample, q, want)
        print(
            f"  {sample:<10s} pooled QLIKE {q:>18.12f} on {n} bars  reference "
            f"{want:.12f}  OK"
        )
    print("GATE PASSED")


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, ent, dates = p32.load_panel()
    assert len(dates) == GATE_N_DAYS, (len(dates), GATE_N_DAYS)
    tape = p32.build_tape(panel, ent, dates)
    _, mid, ask = p32.quote_grids(ent, dates)
    assert np.max(np.abs(mid[:, 0] - tape["entry_mid"])) < EXACT_TOL, "11:00 mid"
    _, day = p32.attribute(tape, mid, ask, dates)

    cols = list(SESSION)
    era = np.asarray(dates >= ERA0, dtype=bool)
    masks: dict[str, np.ndarray] = {
        "whole": np.ones(len(dates), dtype=bool),
        "daily_era": era,
    }
    var0 = (tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])) ** 2
    legs0 = p36.hedge_legs(np.sqrt(var0), tape)
    lab, q1, q2 = p36.gate(tape, mid, ask, dates, day, legs0)

    # ---- V9, proposal 36's reference, rebuilt on its own float path --------
    realized = remaining_grid(dates, cols)
    censored = censored_grid(panel, dates, cols)
    scale_imp = p36.expanding_lagged_mean(realized / var0, WARMUP)
    bc = np.where(np.isfinite(scale_imp) & (scale_imp > 0.0), scale_imp * var0, np.nan)
    ok_v9 = np.isfinite(bc) & (bc > 0.0) & ~censored
    v9 = np.where(ok_v9, bc, var0)
    legs9 = p36.hedge_legs(np.sqrt(v9), tape)
    r0 = p36.book_returns(tape, mid, ask, legs0)
    r9 = p36.book_returns(tape, mid, ask, legs9)
    print(
        f"\nV9: the expanding lagged {WARMUP}-day realized-over-implied scale "
        f"covers {int(ok_v9.sum())} of {ok_v9.size} stamp-cells; the other "
        f"{int((~ok_v9).sum())} hedge with V0"
    )
    gate_v9(r0, r9, era)

    # ---- the combination's bar-level richness ------------------------------
    r, rbar, comb, y = richness_grids(dates, cols)
    gate_combination(comb, y)

    rtab = richness_table(r, rbar, dates, cols)
    write(rtab, "r_by_clock.csv", "the richness ratio, stamp by stamp")
    for sample in SAMPLES:
        show(
            rtab[rtab["sample"] == sample],
            f"r_by_clock.csv  |  sample {sample}  |  r = {COMBINATION} / {SLICE}; "
            f"r = 1 is 'as the market priced this bar'",
        )

    # ---- the variants ------------------------------------------------------
    variants = total_vol_variants(var0, v9, int((~ok_v9).sum()), r, rbar, censored)
    n_tried = len(variants) - 1
    print(
        f"\n{len(variants)} total-volatility grids: the book V0 and {n_tried} causal "
        f"variants ({VERDICT_REFERENCE}, {len(LAM_V11)} V11 rungs, {len(LAM_V12)} V12 "
        f"rungs, V13, V14); r at a stamp is applied to the delta held over that "
        f"stamp's own thirty-minute bar"
    )
    legs = {v["variant"]: p36.hedge_legs(v["total_vol"], tape) for v in variants}
    rets = {
        v["variant"]: p36.book_returns(tape, mid, ask, legs[v["variant"]])
        for v in variants
    }
    refs: tuple[str, ...] = (BOOK_REFERENCE, VERDICT_REFERENCE)

    vrows: list[dict[str, Any]] = []
    for v in variants:
        name = v["variant"]
        for bk in BOOKS:
            turn = p36.book_turnover(legs[name], bk)
            cost = p36.book_cost(legs[name], bk) / tape["entry_mid"]
            ret = rets[name][bk]
            for sample, m in masks.items():
                row: dict[str, Any] = {
                    "variant": name,
                    "family": v["family"],
                    "lam": v["lam"],
                    "base": v["base"],
                    "causal": True,
                    "description": v["description"],
                    "book": bk,
                    "sample": sample,
                    "n_fallback_stamps": v["n_fallback"],
                }
                row.update(p36.stat_row(ret[m], dates[m]))
                for ref in refs:
                    d = ret - rets[ref][bk]
                    row[f"sd_diff_vs_{ref[:2]}"] = (
                        float(np.std(d[m], ddof=1))
                        if int(m.sum()) >= 2
                        else float("nan")
                    )
                    row[f"mean_abs_diff_vs_{ref[:2]}"] = float(np.mean(np.abs(d[m])))
                row["mean_turnover"] = float(np.mean(turn[m]))
                row["mean_hedge_cost_prem"] = float(np.mean(cost[m]))
                for cls, tg in (("1_choppy", "choppy"), ("3_trend", "trend")):
                    sel = m & (lab == cls)
                    row[f"mean_{tg}"] = float(np.mean(ret[sel]))
                    row[f"Sharpe_{tg}"] = p32.sharpe_ann(pd.Series(ret[sel]))
                    row[f"n_{tg}"] = int(sel.sum())
                vrows.append(row)
    variants_df = pd.DataFrame(vrows)
    write(variants_df, "variants.csv", "the hedge volatilities, scored")
    vcols = [
        "variant",
        "n",
        "mean",
        "sd",
        "Sharpe_ann",
        "t",
        "MaxDD",
        "worst_day",
        "worst_date",
        "sd_diff_vs_V0",
        "sd_diff_vs_V9",
        "mean_abs_diff_vs_V9",
        "mean_turnover",
        "mean_hedge_cost_prem",
        "n_fallback_stamps",
        "mean_choppy",
        "Sharpe_choppy",
        "mean_trend",
        "Sharpe_trend",
    ]
    for bk in BOOKS:
        for sample in SAMPLES:
            sub = variants_df[
                (variants_df["book"] == bk) & (variants_df["sample"] == sample)
            ]
            show(sub, f"variants.csv  |  book {bk}  |  sample {sample}", vcols)
    show(
        variants_df[
            ["variant", "family", "lam", "base", "description", "n_fallback_stamps"]
        ].drop_duplicates("variant"),
        "variants.csv  |  what each variant puts inside package_delta, and the "
        "base its warm-up cells fall back to",
    )
    print(
        f"\nmean_hedge_cost_prem is the repo's {p32.HEDGE_COST_BP} bp charge on "
        f"S x |delta traded|, in units of the entry premium.  It is REPORTED, not "
        f"taken out of any book: every Sharpe, mean and difference in these tables "
        f"is gross of it, exactly as the published books are."
    )
    print(
        f"\nshape terciles used for the choppy / trend columns: "
        f"{q1:.6f} / {q2:.6f} of |sum of bar returns| / sum |bar returns|"
    )

    # ---- the paired daily differences --------------------------------------
    prows: list[dict[str, Any]] = []
    for v in variants:
        for ref in refs:
            if v["variant"] in (BOOK_REFERENCE, ref):
                continue
            for bk in BOOKS:
                for sample, m in masks.items():
                    row = {
                        "variant": v["variant"],
                        "family": v["family"],
                        "lam": v["lam"],
                        "reference": ref,
                        "book": bk,
                        "sample": sample,
                    }
                    row.update(
                        p36.paired_row(rets[v["variant"]][bk][m], rets[ref][bk][m])
                    )
                    prows.append(row)
    pairs_df = pd.DataFrame(prows)
    write(pairs_df, "pairs.csv", "the paired daily difference against V0 and V9")
    pcols = [
        "variant",
        "n",
        "mean_diff",
        "sd_diff",
        "t_diff",
        "hac_t_diff",
        "hac_lag",
        "frac_days_improved",
        "frac_days_unchanged",
    ]
    for ref in refs:
        for bk in BOOKS:
            for sample in SAMPLES:
                sub = pairs_df[
                    (pairs_df["reference"] == ref)
                    & (pairs_df["book"] == bk)
                    & (pairs_df["sample"] == sample)
                ]
                show(
                    sub,
                    f"pairs.csv  |  against {ref}  |  book {bk}  |  sample {sample}",
                    pcols,
                )

    # ---- the bootstrap of the Sharpe difference ----------------------------
    brows: list[dict[str, Any]] = []
    for v in variants:
        for ref in refs:
            if v["variant"] in (BOOK_REFERENCE, ref):
                continue
            for bk in BOOKS:
                for sample, m in masks.items():
                    row = {
                        "variant": v["variant"],
                        "family": v["family"],
                        "lam": v["lam"],
                        "reference": ref,
                        "book": bk,
                        "sample": sample,
                        "B": BOOT_B,
                        "block": BOOT_BLOCK,
                        "seed": BOOT_SEED,
                        "ci_pct_lo": CI_PCT[0],
                        "ci_pct_hi": CI_PCT[1],
                    }
                    row.update(
                        p36.sharpe_diff_ci(rets[v["variant"]][bk][m], rets[ref][bk][m])
                    )
                    if (
                        ref == VERDICT_REFERENCE
                        and bk == PRIMARY_BOOK
                        and sample == VERDICT_SAMPLE
                    ):
                        row["verdict"] = (
                            "improves on V9"
                            if row["ci_excludes_zero"] and row["dSharpe"] > 0
                            else "no evidence"
                        )
                    else:
                        row["verdict"] = ""
                    brows.append(row)
    boot_df = pd.DataFrame(brows)
    write(boot_df, "bootstrap.csv", "the bootstrap interval of the Sharpe difference")
    bcols = [
        "variant",
        "n",
        "dSharpe",
        "boot_lo",
        "boot_hi",
        "pct_draws_positive",
        "ci_excludes_zero",
        "verdict",
    ]
    for ref in refs:
        for bk in BOOKS:
            for sample in SAMPLES:
                sub = boot_df[
                    (boot_df["reference"] == ref)
                    & (boot_df["book"] == bk)
                    & (boot_df["sample"] == sample)
                ]
                show(
                    sub,
                    f"bootstrap.csv  |  against {ref}  |  book {bk}  |  sample "
                    f"{sample}  |  {BOOT_B} circular block draws, block "
                    f"{BOOT_BLOCK}, seed {BOOT_SEED}",
                    bcols,
                )

    # ---- the verdict -------------------------------------------------------
    verdicts = boot_df[
        (boot_df["reference"] == VERDICT_REFERENCE)
        & (boot_df["book"] == PRIMARY_BOOK)
        & (boot_df["sample"] == VERDICT_SAMPLE)
    ]
    show(
        verdicts,
        f"VERDICT  |  primary book {PRIMARY_BOOK}, sample {VERDICT_SAMPLE}, against "
        f"{VERDICT_REFERENCE}: 'improves on V9' only where the "
        f"{int(CI_PCT[1] - CI_PCT[0])}% bootstrap interval of the Sharpe difference "
        f"excludes zero",
        ["variant", "family", "lam", "dSharpe", "boot_lo", "boot_hi", "verdict"],
    )
    n_tested = int(len(verdicts))
    n_improve = int((verdicts["verdict"] == "improves on V9").sum())
    alpha = 1.0 - (CI_PCT[1] - CI_PCT[0]) / 100.0
    print(
        f"\n{n_tried} causal variants were tried; {n_tested} of them carry r or rbar "
        f"and are tested against {VERDICT_REFERENCE} on the primary book in the era "
        f"({VERDICT_REFERENCE} itself is the reference, not a test).  "
        f"Improving on V9 by the stated bar: {n_improve}.  "
        f"No evidence: {n_tested - n_improve}."
    )
    print(
        f"The intervals are uncorrected for multiplicity: at a "
        f"{100.0 * alpha:.0f}% level, {n_tested} independent tests return "
        f"{n_tested * alpha:.2f} false positives in expectation, and at least one "
        f"with probability {1.0 - (1.0 - alpha) ** n_tested:.3f}.  A single interval "
        f"that just excludes zero is therefore not, on its own, evidence."
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
