"""57 - trading the 15:30 sign(s) straddle only when the forecasts agree.

How this was found (stated because it decides how far to trust it): study 54
Part A decomposed the deck's 15:30 trade and noticed, in-sample and without a
pre-registered rule, that on the 262 of 866 days where the eight forecasts'
signs disagree the ridge's trade earns nothing (0.11 mid / -0.16 crossed) and
that trading only on the 604 agreement days scores 1.59 / 1.20 against the
deck's 1.34 / 0.87; the paired bootstrap of that difference was +0.33 crossed,
interval [-0.32, +1.00].  This proposal asks whether that is a real second layer
or a by-product of which eight forecasts happen to be on disk.  Everything below
was fixed before a number was computed.

The trade.  The deck's own: at 15:30 hold sign(s) = sign(rv_hat - iv_var) of the
nearest-OTM SPXW 0DTE straddle to the 4pm cash settlement, one unit a day, in
units of the midpoint entry premium; "crossed" buys at the quoted ask and sells
at the quoted bid, a fractional position paying the touch on its own side.  All
eight forecasts share the same 866 days, the same quotes and the same implied
slice; every forecast is issued AT 15:30 for the 15:30-16:00 bar (Part E checks
that against the forecast tables).

The forecast sets, chosen in advance because the set is itself a choice:
  S8  all eight tags of atm_straddle_lib.MODEL_ORDER.
  S6  the six that are not near-duplicates.  Dropped: `blk2_inc`, the same ridge
      on the earlier panel; and `lasso_t`, the tuned lasso on the earlier panel
      - the fixed lasso `lasso_f` is kept because it sits on the panel of record
      like the ridge, and the tuned sparse family on the earlier panel is still
      represented by `enet`.
  S4  one forecast per family: ridge `blk2`, baseline OLS `a0`, XGBoost `xgb`,
      LightGBM `lgbm`.

The rules, none with a free parameter (q0 is the ridge's sign):
  R1  q0 when every forecast in the set has the same sign, flat otherwise.
  R2  the average of the set's signs (a fractional position).
  R3  the majority vote; a tied vote is flat.  (Study 54's exploratory vote sent
      ties short; that variant is printed for S8 only, as R3s, to tie the two.)
  R4  q0 when at most ONE other forecast in the set disagrees with the ridge,
      flat otherwise (so a ridge that is the lone dissenter does not trade).

For every set x rule: days traded, buy share, mean, mid and crossed Sharpe, max
drawdown, crossed Sharpe by year, and tail capture (of the 20 best long-straddle
days, how many the rule is long / short / flat).

Is abstaining informative, or is it only trading fewer days?
  PLACEBO   write the rule as q = w * q0 with w = q * q0 the day's weight on the
            ridge's trade (1 = follow it, 0 = abstain, -1 = oppose it).  The
            days with w != 1 are cut into maximal runs; the deviating runs and
            the following runs are each permuted and re-interleaved from the
            state the real series starts in, every deviating run carrying its
            own weights with it.  Every draw therefore deviates on exactly as
            many days, in runs of exactly the same lengths, with exactly the
            same weights; only WHICH days changes.  For R1 and R4 this is
            proposal 31's block_shuffled_masks, and GATE P asserts the draws
            are identical.  B = 2000, seed 0,
            p = (1 + #{placebo Sharpe >= rule Sharpe}) / (1 + B), mid and crossed.
  BOOTSTRAP paired circular-block bootstrap (block 21, B = 2000, seed 0) of the
            Sharpe difference against plain ridge sign(s), mid and crossed.
  BAND      the mechanical comparison: abstain on the SAME NUMBER of days as R1
            (and as R4), choosing the days with the smallest |rv_hat - iv_var| /
            iv_var of the ridge itself.  The count is matched ex post, so the
            band is a diagnostic, not a tradeable rule.  If it does as well,
            agreement is only a proxy for "the signal is near zero".

Interaction with study 54's month-end result (flags read from its c_daily.csv):
the agreement rule alone, the ridge forced long on the last trading day of the
month alone, and both layers together.

Reading.  A cell PASSES only if, on the crossed series, its paired interval
against plain sign(s) excludes zero on the positive side AND its placebo p is
below 0.05.  12 cells (3 sets x 4 rules); the bootstrap leg alone passes about
0.025 x 12 = 0.3 of them by chance.  Everything is in-sample: the forecasts end
on 2024-04-30, so there is no unseen day for any rule here.

Gates:
  GATE 0  ridge sign(s) 1.338322 mid / 0.869588 crossed on 866 days.
  GATE 1  study 54 Part A reproduces: 604 agreement days, 1.589 / 1.199.
  GATE P  the weight placebo equals proposal 31's masks, draw for draw, on R1(S8).

Run:  python writeup/intraday_proposals/57_forecast_agreement_filter.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = DECK / "proposals" / "57"
P54_DAILY = DECK / "proposals" / "54" / "c_daily.csv"

RIDGE = "blk2"
SETS: dict[str, tuple[str, ...]] = {
    "S8": tuple(asl.MODEL_ORDER),
    "S6": tuple(t for t in asl.MODEL_ORDER if t not in ("blk2_inc", "lasso_t")),
    "S4": ("blk2", "a0", "xgb", "lgbm"),
}
RULES: tuple[str, ...] = ("R1", "R2", "R3", "R4")
ABSTAIN_RULES: tuple[str, ...] = ("R1", "R4")

ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0
N_PLACEBO, PLACEBO_SEED, PLACEBO_ALPHA = 2000, 0, 0.05
TOP_DAYS = 20
LAST_BAR_MINUTES = 16 * 60  # bar-END label of the 15:30-16:00 bar

GATE_RIDGE = (1.338322, 0.869588)
GATE_RIDGE_TOL = 1e-6
GATE_AGREE_DAYS = 604
GATE_AGREE = (1.589, 1.199)
GATE_AGREE_TOL = 5e-4  # study 54 printed three decimals


def _load_module(path: Path, name: str) -> ModuleType:
    """The repo's read-only import: a script is imported by path, never edited."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ helpers --
def sharpe_rows(x: np.ndarray) -> np.ndarray:
    """Annualized Sharpe along the last axis of a (d, n) or (n,) array."""
    a = np.atleast_2d(np.asarray(x, float))
    sd = a.std(axis=1, ddof=1)
    return np.where(sd > 0, a.mean(axis=1) / np.where(sd > 0, sd, 1.0) * ANN, np.nan)


def sharpe(x: np.ndarray) -> float:
    return float(sharpe_rows(x)[0])


def maxdd(x: np.ndarray) -> float:
    """Worst peak-to-trough of the cumulative SUM path, peak seeded at 0."""
    path = np.cumsum(np.asarray(x, float))
    peak = np.maximum(np.maximum.accumulate(path), 0.0)
    return float((path - peak).min())


def write(df: pd.DataFrame, name: str, title: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"wrote {OUT / name}  ({len(df)} rows) - {title}")


class Tape:
    """The deck's 15:30 day axis: returns, touches, every forecast's sign."""

    def __init__(self) -> None:
        books = {
            t: pd.read_parquet(DECK / f"daily_{t}.parquet").sort_index()
            for t in asl.MODEL_ORDER
        }
        d = books[RIDGE]
        self.books = books
        self.dates = pd.DatetimeIndex(pd.to_datetime(d.index))
        self.n = len(d)
        for t, b in books.items():
            assert b.index.equals(d.index), f"{t}: a different day axis"
            for col in ("R", "iv_var", "exit"):
                same = np.array_equal(b[col].to_numpy(float), d[col].to_numpy(float))
                assert same, (t, col)
        self.r = d["R"].to_numpy(float)
        ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
        bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
        ex = d["exit"].to_numpy(float)
        self.cr_long = ex / ask - 1.0
        self.cr_short = ex / bid - 1.0
        iv = d["iv_var"].to_numpy(float)
        self.sign = {
            t: np.where(b["rv_hat"].to_numpy(float) > iv, 1.0, -1.0)
            for t, b in books.items()
        }
        self.q0 = self.sign[RIDGE]
        self.s_rel = np.abs(d["rv_hat"].to_numpy(float) - iv) / iv
        self.top = np.argsort(self.r)[::-1][:TOP_DAYS]
        self.year = self.dates.year.to_numpy()

    def mid(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, float) * self.r

    def crossed(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        return np.where(q > 0.0, q * self.cr_long, q * self.cr_short)

    def position(self, set_name: str, rule: str) -> np.ndarray:
        stack = np.array([self.sign[t] for t in SETS[set_name]])
        q0 = self.q0
        if rule == "R1":
            return np.where(np.abs(stack.sum(axis=0)) == len(stack), q0, 0.0)
        if rule == "R2":
            return stack.mean(axis=0)
        if rule == "R3":
            return np.sign(stack.sum(axis=0))
        if rule == "R3s":  # study 54's exploratory vote: a tie is a selling day
            return np.where(stack.sum(axis=0) > 0, 1.0, -1.0)
        if rule == "R4":
            others = stack[[t != RIDGE for t in SETS[set_name]]]
            dissent = (others != q0[None, :]).sum(axis=0)
            return np.where(dissent <= 1, q0, 0.0)
        raise ValueError(rule)


def boot_index(n: int) -> np.ndarray:
    return asl.circular_block_bootstrap_idx(
        np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
    )


def paired_ci(
    a: np.ndarray, b: np.ndarray, idx: np.ndarray
) -> tuple[float, float, float]:
    """Sharpe(a) - Sharpe(b): point estimate and the percentile interval."""
    d = sharpe_rows(a[idx]) - sharpe_rows(b[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return sharpe(a) - sharpe(b), float(lo), float(hi)


# --------------------------------------------------------------- placebo -----
def shuffled_weights(
    w: np.ndarray, n_draw: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_draw, n) weight paths: deviating runs permuted, weights riding along."""
    w = np.asarray(w, float)
    dev = w != 1.0
    n = w.size
    edges = np.flatnonzero(np.diff(dev.astype(np.int8)) != 0) + 1
    runs = np.split(np.arange(n), edges)
    states = [bool(dev[r[0]]) for r in runs]
    on_runs = [r for r, s in zip(runs, states, strict=True) if s]
    off_runs = [r for r, s in zip(runs, states, strict=True) if not s]
    first = bool(dev[0])
    out = np.ones((n_draw, n))
    for b in range(n_draw):
        # proposal 31's order: the deviating runs are permuted first
        p_on = rng.permutation(len(on_runs))
        p_off = rng.permutation(len(off_runs))
        pos, k_on, k_off, state = 0, 0, 0, first
        while pos < n:
            if state:
                r = on_runs[p_on[k_on]]
                k_on += 1
                out[b, pos : pos + r.size] = w[r]
            else:
                r = off_runs[p_off[k_off]]
                k_off += 1
            pos += r.size
            state = not state
    return out


def placebo_p(tape: Tape, q: np.ndarray, seed_key: list[int]) -> dict[str, float]:
    w = q * tape.q0
    dev = w != 1.0
    if not dev.any() or bool(dev.all()):
        return {
            "placebo_p_mid": np.nan,
            "placebo_p_crossed": np.nan,
            "placebo_q50_crossed": np.nan,
        }
    ws = shuffled_weights(
        w, N_PLACEBO, np.random.default_rng([PLACEBO_SEED, *seed_key])
    )
    qs = ws * tape.q0[None, :]
    s_mid = sharpe_rows(qs * tape.r[None, :])
    s_cr = sharpe_rows(
        np.where(qs > 0.0, qs * tape.cr_long[None, :], qs * tape.cr_short[None, :])
    )
    return {
        "placebo_p_mid": float(
            (1 + int((s_mid >= sharpe(tape.mid(q))).sum())) / (1 + N_PLACEBO)
        ),
        "placebo_p_crossed": float(
            (1 + int((s_cr >= sharpe(tape.crossed(q))).sum())) / (1 + N_PLACEBO)
        ),
        "placebo_q50_crossed": float(np.nanmedian(s_cr)),
    }


def gate_placebo(tape: Tape) -> None:
    p31 = _load_module(HERE / "31_margin_skip_rule.py", "p57_p31")
    q = tape.position("S8", "R1")
    abstain = q == 0.0
    ref = p31.block_shuffled_masks(
        abstain, 200, np.random.default_rng([PLACEBO_SEED, 99])
    )
    mine = shuffled_weights(q * tape.q0, 200, np.random.default_rng([PLACEBO_SEED, 99]))
    assert np.array_equal(mine == 0.0, ref), "weight placebo != proposal 31's masks"
    print(
        "GATE P  the weight placebo equals proposal 31's block_shuffled_masks on R1(S8), 200 draws"
    )


# ---------------------------------------------------------------- the cells --
def cell_row(
    tape: Tape, q: np.ndarray, idx: np.ndarray, seed_key: list[int]
) -> dict[str, Any]:
    mid, cr = tape.mid(q), tape.crossed(q)
    base_mid, base_cr = tape.mid(tape.q0), tape.crossed(tape.q0)
    d_mid, lo_mid, hi_mid = paired_ci(mid, base_mid, idx)
    d_cr, lo_cr, hi_cr = paired_ci(cr, base_cr, idx)
    rec: dict[str, Any] = {
        "days_traded": int((q != 0.0).sum()),
        "buy_share": float((q > 0.0).mean()),
        "differs_from_ridge_days": int((q != tape.q0).sum()),
        "mean_mid": float(mid.mean()),
        "mean_crossed": float(cr.mean()),
        "Sharpe_mid": sharpe(mid),
        "Sharpe_crossed": sharpe(cr),
        "MaxDD_mid": maxdd(mid),
        "MaxDD_crossed": maxdd(cr),
        f"top{TOP_DAYS}_long": int((q[tape.top] > 0.0).sum()),
        f"top{TOP_DAYS}_short": int((q[tape.top] < 0.0).sum()),
        f"top{TOP_DAYS}_flat": int((q[tape.top] == 0.0).sum()),
        "dSharpe_mid": d_mid,
        "dS_mid_lo": lo_mid,
        "dS_mid_hi": hi_mid,
        "dSharpe_crossed": d_cr,
        "dS_crossed_lo": lo_cr,
        "dS_crossed_hi": hi_cr,
    }
    rec.update(placebo_p(tape, q, seed_key))
    rec["passes"] = bool(
        rec["dS_crossed_lo"] > 0.0 and rec["placebo_p_crossed"] < PLACEBO_ALPHA
    )
    return rec


def by_year(tape: Tape, q: np.ndarray) -> dict[str, float]:
    cr = tape.crossed(q)
    return {str(y): sharpe(cr[tape.year == y]) for y in sorted(set(tape.year))}


# ----------------------------------------------------- causality (Part E) ----
def causality_task(tag: str) -> dict[str, Any]:
    """The day book's forecast is the table row stamped 16:00 (issued at 15:30)."""
    book = pd.read_parquet(DECK / f"daily_{tag}.parquet").sort_index()
    days = pd.DatetimeIndex(pd.to_datetime(book.index))
    panel = asl.load_yhat_panel_mz(asl.yhat_paths(ROOT)[tag])
    pday = pd.DatetimeIndex(pd.to_datetime(panel["date"]))
    out: dict[str, Any] = {
        "forecast": tag,
        "entry_stamps": ",".join(sorted(set(book["hhmm_c"].astype(str)))),
    }
    for label, mins in (("16:00", LAST_BAR_MINUTES), ("16:30", LAST_BAR_MINUTES + 30)):
        m = panel["mins"].to_numpy() == mins
        row = pd.Series(panel.loc[m, "rv_hat"].to_numpy(float), index=pday[m])
        row = row[~row.index.duplicated()].reindex(days).to_numpy(float)
        rel = np.abs(row / book["rv_hat"].to_numpy(float) - 1.0)
        out[f"max_rel_diff_vs_row_{label}"] = float(np.nanmax(rel))
        out[f"days_matched_row_{label}"] = int(np.isfinite(rel).sum())
    return out


# ------------------------------------------------------------------- main ----
def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 200)
    t_start = time.time()
    tape = Tape()
    idx = boot_index(tape.n)

    got = (sharpe(tape.mid(tape.q0)), sharpe(tape.crossed(tape.q0)))
    assert abs(got[0] - GATE_RIDGE[0]) < GATE_RIDGE_TOL, got
    assert abs(got[1] - GATE_RIDGE[1]) < GATE_RIDGE_TOL, got
    print(
        f"GATE 0  ridge sign(s), {tape.n} days: {got[0]:.6f} mid / {got[1]:.6f} crossed"
    )
    q_agree = tape.position("S8", "R1")
    agree = (
        int((q_agree != 0.0).sum()),
        sharpe(tape.mid(q_agree)),
        sharpe(tape.crossed(q_agree)),
    )
    assert agree[0] == GATE_AGREE_DAYS, agree
    assert abs(agree[1] - GATE_AGREE[0]) < GATE_AGREE_TOL, agree
    assert abs(agree[2] - GATE_AGREE[1]) < GATE_AGREE_TOL, agree
    print(
        f"GATE 1  study 54 Part A: {agree[0]} agreement days, {agree[1]:.3f} mid / "
        f"{agree[2]:.3f} crossed"
    )
    gate_placebo(tape)

    # ------------------------------------------------ A. agreement matrix --
    tags = list(asl.MODEL_ORDER)
    mat = pd.DataFrame(
        [[float((tape.sign[a] == tape.sign[b]).mean()) for b in tags] for a in tags],
        index=tags,
        columns=tags,
    )
    mat.insert(0, "buy_share", [float((tape.sign[t] > 0).mean()) for t in tags])
    write(
        mat.reset_index(names="forecast"),
        "a_agreement_matrix.csv",
        "pairwise sign agreement",
    )
    print("\n--- A. share of days two forecasts have the same sign")
    print(mat.round(3).to_string())
    for name, members in SETS.items():
        stack = np.array([tape.sign[t] for t in members])
        unanimous = np.abs(stack.sum(axis=0)) == len(stack)
        print(
            f"   {name} {members}: unanimous on {int(unanimous.sum())} of {tape.n} days"
        )

    # -------------------------------------------------------- B. the rules --
    rows, years = [], []
    base = {"set": "-", "rule": "ridge sign(s)"}
    rows.append(base | cell_row(tape, tape.q0, idx, [0, 0]))
    years.append(base | by_year(tape, tape.q0))
    cells = [(s, r) for s in SETS for r in RULES] + [("S8", "R3s")]
    for i_cell, (s, r) in enumerate(cells):
        q = tape.position(s, r)
        rows.append({"set": s, "rule": r} | cell_row(tape, q, idx, [1, i_cell]))
        years.append({"set": s, "rule": r} | by_year(tape, q))
    rules = pd.DataFrame(rows)
    write(rules, "b_rules.csv", "every set x rule against plain ridge sign(s)")
    write(pd.DataFrame(years), "b_by_year.csv", "crossed Sharpe by year")
    show = [
        "set",
        "rule",
        "days_traded",
        "buy_share",
        "differs_from_ridge_days",
        "Sharpe_mid",
        "Sharpe_crossed",
        "MaxDD_crossed",
        f"top{TOP_DAYS}_long",
        f"top{TOP_DAYS}_short",
        f"top{TOP_DAYS}_flat",
        "dSharpe_crossed",
        "dS_crossed_lo",
        "dS_crossed_hi",
        "placebo_p_mid",
        "placebo_p_crossed",
        "placebo_q50_crossed",
        "passes",
    ]
    print(
        "\n--- B. the rules (dSharpe and its interval are against plain ridge sign(s))"
    )
    print(rules[show].round(3).to_string(index=False))
    print("\n--- B2. crossed Sharpe by year")
    print(pd.DataFrame(years).round(2).to_string(index=False))
    counted = rules[rules["rule"].isin(RULES)]
    print(
        f"\ncells {len(counted)} (3 sets x 4 rules); passes {int(counted['passes'].sum())}; "
        f"the bootstrap leg alone gives about {0.025 * len(counted):.1f} by chance; cells with a "
        f"positive crossed dSharpe {int((counted['dSharpe_crossed'] > 0).sum())}"
    )

    # ------------------------------------------ C. the equal-size band ------
    band_rows = []
    order = np.argsort(tape.s_rel, kind="stable")
    for i_cell, (s, r) in enumerate((s, r) for s in SETS for r in ABSTAIN_RULES):
        q = tape.position(s, r)
        abstain = q == 0.0
        k = int(abstain.sum())
        band_abstain = np.zeros(tape.n, bool)
        band_abstain[order[:k]] = True
        q_band = np.where(band_abstain, 0.0, tape.q0)
        d_cr, lo, hi = paired_ci(tape.crossed(q), tape.crossed(q_band), idx)
        pb = placebo_p(tape, q_band, [2, i_cell])
        band_rows.append(
            {
                "set": s,
                "rule": r,
                "days_abstained": k,
                "overlap_share": float((abstain & band_abstain).sum() / k)
                if k
                else np.nan,
                "median_rel_signal_abstained": float(np.median(tape.s_rel[abstain])),
                "median_rel_signal_traded": float(np.median(tape.s_rel[~abstain])),
                "agreement_Sharpe_mid": sharpe(tape.mid(q)),
                "agreement_Sharpe_crossed": sharpe(tape.crossed(q)),
                "band_Sharpe_mid": sharpe(tape.mid(q_band)),
                "band_Sharpe_crossed": sharpe(tape.crossed(q_band)),
                "band_placebo_p_crossed": pb["placebo_p_crossed"],
                "agreement_minus_band_crossed": d_cr,
                "diff_lo": lo,
                "diff_hi": hi,
                f"band_top{TOP_DAYS}_flat": int(band_abstain[tape.top].sum()),
            }
        )
    band = pd.DataFrame(band_rows)
    write(
        band, "c_equal_size_band.csv", "agreement against a no-trade band of equal size"
    )
    print(
        "\n--- C. abstaining on the same NUMBER of days, smallest |rv_hat - iv_var| / iv_var"
    )
    print(band.round(3).to_string(index=False))

    # ------------------------------------------ D. month-end interaction ----
    flags = pd.read_csv(P54_DAILY, index_col=0, parse_dates=True)["month_end"].astype(
        bool
    )
    me = flags.reindex(tape.dates)
    assert not me.isna().any(), "a deck day is missing from study 54's daily file"
    me_arr = me.to_numpy(bool)
    layers = {
        "ridge sign(s)": tape.q0,
        "month-ends forced long": np.where(me_arr, 1.0, tape.q0),
    }
    for s in ("S8", "S4"):
        for r in ABSTAIN_RULES:
            q = tape.position(s, r)
            layers[f"{r}({s})"] = q
            layers[f"{r}({s}) + month-ends forced long"] = np.where(me_arr, 1.0, q)
    d_rows = []
    q_me = layers["month-ends forced long"]
    for name, q in layers.items():
        d0, lo0, hi0 = paired_ci(tape.crossed(q), tape.crossed(tape.q0), idx)
        d1, lo1, hi1 = paired_ci(tape.crossed(q), tape.crossed(q_me), idx)
        d_rows.append(
            {
                "book": name,
                "days_traded": int((q != 0.0).sum()),
                "Sharpe_mid": sharpe(tape.mid(q)),
                "Sharpe_crossed": sharpe(tape.crossed(q)),
                "MaxDD_crossed": maxdd(tape.crossed(q)),
                "vs_ridge": d0,
                "vs_ridge_lo": lo0,
                "vs_ridge_hi": hi0,
                "vs_month_end_layer": d1,
                "vs_month_end_lo": lo1,
                "vs_month_end_hi": hi1,
            }
        )
    inter = pd.DataFrame(d_rows)
    write(inter, "d_month_end_interaction.csv", "agreement layer x month-end layer")
    print(
        f"\n--- D. with study 54's month-end layer ({int(me_arr.sum())} month-ends on the deck days)"
    )
    print(inter.round(3).to_string(index=False))
    disagree = tape.position("S8", "R1") == 0.0
    print(
        f"   month-ends among the S8 disagreement days: {int((me_arr & disagree).sum())} of "
        f"{int(disagree.sum())}; among the agreement days {int((me_arr & ~disagree).sum())} of "
        f"{int((~disagree).sum())}"
    )

    # --------------------------------------------------- E. causality -------
    with ProcessPoolExecutor(max_workers=len(tags)) as pool:
        caus = pd.DataFrame(list(pool.map(causality_task, tags)))
    write(
        caus,
        "e_causality.csv",
        "day-book forecast against the table rows stamped 16:00 and 16:30",
    )
    print(
        "\n--- E. every day book enters at 15:30 and reads the row stamped 16:00 (issued at 15:30)"
    )
    print(caus.to_string(index=False))
    print(f"\ntotal runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
