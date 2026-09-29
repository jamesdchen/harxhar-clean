"""Kelly-type weights w_t = f(VRP_t) for the 15:30 last-30-min straddle trade (checklist A3, W2, 2026-09-29).

The question. sign(s) holds one unit of straddle premium every day: long the straddle when the forecast of the
15:30-16:00 realized variance exceeds the implied variance (s > 0), short otherwise. The rv_iv notebook §10 ("Sizing by
the size of the forecast gap") already showed that sizing in proportion to s loses to sign(s) and that the rank of |s|
ties it. This script asks the Kelly version: set the weight each day from a causal estimate of the payoff of the trade
GIVEN the day's gap s_t, at the growth-optimal (Kelly) fraction, and compare it with sign(s) and with always-short,
on Sharpe ratio (scale-free) and on compounded growth (what Kelly maximises).

Straddle: nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position, entered at
15:30 ET and cash-settled at the official close. No option chain is read: the inputs are the notebook's own daily
books ``results/atm_straddle_0dte_1530/daily_<tag>.parquet`` (written by §9 of ``notebooks/_write_0dte_nb.py``), one
per forecast, 866 days each: the entry midpoint, bid and ask of the two legs, the settlement payout, the straddle
return R = exit/entry - 1, the implied variance and the recalibrated forecast. The gap is s_t = rv_hat_t - iv_var_t
(both known at 15:30). SCORER: the forecasts are the NOTEBOOK's (the §7 recalibration fitted on all 13 session bars),
the scorer of §10 and §15; the research scorer's 16:00-bar recalibration is not used or mixed in here. Forecasts: the
paper's eight (asl.MODEL_ORDER) and the seven per-bar linear ones (asl.SUBSAMPLE_ORDER), as in §10.

Weights. w_t is the signed share of wealth deployed as straddle premium on day t (w > 0 long, w < 0 short, 0 flat),
the unit of §15. Per unit of premium, a long returns r_L and a short r_S:
    midpoint fill: r_L = R,               r_S = -R
    crossed fill:  r_L = exit/ask - 1,    r_S = 1 - exit/bid     (buy at the ask, sell at the bid; §10 / §13)
and the day's return on wealth is x_t = w_t r_L,t (w > 0) or |w_t| r_S,t (w < 0). Each fill estimates its Kelly
fraction from its own per-unit returns (a bettor who pays the spread estimates the payoff net of it), so at the
crossed fill a cell whose long and short returns are both negative on average is sized ZERO.

Every estimate at day t uses days u < t only (R_t is not known at 15:30; s_t is). The Kelly fraction of a return X
is its mean over its variance, f = mu/sigma^2 (the "mean-variance" Kelly), or the exact argmax_f mean log(1 + f X)
over the past payoff distribution (the "log-optimal" Kelly). Rules (the full-Kelly fraction f*; the traded weight is
w = sign(f*) min(kappa |f*|, cap of that side), kappa in {1, 1/2, 1/4}):
    (d)  always short, Kelly                 short only, f* = max(mean/var of r_S, 0) over every past day
    sign(s), Kelly-levered                   one fraction for the sign(s) book: f* = sign(s_t) max(mean/var, 0)
                                             of the past sign(s) returns (the flat rule with a Kelly-sized stake)
    (b)  Kelly by sign of s, log-optimal     the past days with the same sign of s as today; the exact log-optimal
                                             fraction of their realised per-unit returns, on whichever side (long
                                             or short) grows faster; 0 if neither side has a positive mean
    (b') Kelly by sign of s, mean-variance   the same days, mu/sigma^2 on the better side
    (a)  Kelly by sign x |s| bin             the past days with the same sign of s, split into N_BINS equal-count
                                             bins of |s| (edges from those past days); today's cell = its sign and
                                             bin; mu/sigma^2 of the cell on the better side
    (a)  Kelly by regression on s, |s|       least squares of the past per-unit return on [1, s/rms, |s|/rms]
                                             (rms over the window); mu_hat(s_t) from the fit, sigma^2 from the past
                                             same-sign residuals. Its two sides share one intercept, so it cannot
                                             represent the jump at s = 0 that the sign rule trades; hence also
    (a)  Kelly by regression on sign(s), s, |s|   the same with an indicator 1{s > 0} (one intercept per sign), which
                                             nests the sign rule (b') when the slopes are zero
    (c)  proportional s/mean|s|, sign(s) Kelly stake   w* = f*(sign(s), Kelly-levered) x |s_t| / mean |s| of the
                                             window's earlier days: the sign(s) Kelly stake on average, redistributed
                                             across days in proportion to the gap. (§10's own scale s/rms cannot be
                                             Kelly-levered causally: the book's early sizes, up to 29 when rms rests
                                             on a few days, make its past mean negative through the whole sample.)
Unit references (no Kelly): always short (-1), sign(s) (+-1), and (c) proportional s/rms (§10's rule (a), a gate);
their growth is at §15's fixed 3 % of wealth.

Named constants.
    WARMUP = 252      one year of prior deck days before the first scored day: §10's first block and §18's warm-up;
                      at 252 days the smaller (long) side has ~100 days, so a tercile cell holds ~30+ days
    N_BINS = 3        terciles of |s| within each sign (robustness: 2 and 5, and terciles of the relative gap s/iv)
    windows           expanding (every earlier deck day; the deck is 866 days, so a 2000-session window would be the
                      same thing); robustness: trailing 504 (two years) and 252 (one year) deck days
    caps              "history ruin bound" (primary): no fraction above 1/(worst per-unit loss on that side over every
                      earlier deck day) -- the largest stake at which the account would have survived its own
                      worst day; a long straddle loses at most its premium, so the long cap is 1. Robustness:
                      "exchange margin": the short side at most P_t/m_t (the §13 Cboe strategy-based short-straddle
                      margin m_t, in index points, fully collateralised by the account; P_t the premium at the
                      fill), and never above the ruin bound; "none".
    PAIR_BLOCK/B/SEED = 21 / 2000 / 0, the notebook's paired circular block bootstrap (§10); the index set is
                      rng([0, n_scored]), exactly the set §10's sized-rule table draws for its 614-day block.
    F_FIXED = 0.03    §15's fixed share of wealth, for the unit references' growth.

Metrics on the 614 scored days (the 866 deck days after the warm-up; identical rows for every rule): mean daily return
on wealth, Sharpe ratio (sqrt(PERIODS_PER_YEAR = 252) as the library), annualised log-growth g = 252 mean log(1 + x)
(compounded as §15 does), terminal wealth, max drawdown of compounded wealth (fraction of the running peak), worst
day, mean |w| (the premium traded per day as a share of wealth -- every position opens at 15:30 and settles at 16:00),
mean |w_t - w_{t-1}|, share of days sized zero, long, short, opposite to sign(s), at the cap. Paired bootstrap 95 %
percentile intervals: Sharpe difference vs sign(s) and vs always short (unit rules; scale-free), growth difference vs
sign(s) Kelly-levered and vs always-short Kelly at the same kappa (unit references: vs their 3 % twins), and vs sign(s)
at the fixed 3 %.

Gates: the 15 books share the 866 days; R == exit/entry - 1, s == rv_hat - iv_var, 0 < bid <= entry <= ask on every
day; sign(s) on the 866 days reproduces §10's rule table Sharpe ratio for every forecast; on the 614-day block the
sign(s) and proportional rows reproduce §10's vrp_sized_rules.csv (Sharpe ratios at both fills and the proportional
rule's paired interval, to 1e-9); causality: perturbing R (days >= k) and s (days > k) by x50 leaves every fraction and
cap on days <= k unchanged, for every rule, window and fill.

Outputs (results/close_kelly/): kelly_rules.csv (primary spec: one row per forecast x rule x kappa x fill),
kelly_robustness.csv (every window x cap, plus the bin variants), kelly_bins.csv (the conditional mu, sigma by cell at
the last refit), kelly_summary.csv (across the 15 forecasts), kelly_wealth.png, kelly_f_vs_s.png.

Run: python experiments/close_kelly_sizing.py  (process-parallel across forecasts, at most 4 workers)
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

DECK = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = ROOT / "results" / "close_kelly"
TAGS = list(asl.MODEL_ORDER) + list(asl.SUBSAMPLE_ORDER)
FIG_TAGS = ("blk2", "sub_live_ridge")  # the two forecasts §10 prints in full
ANN = float(asl.PERIODS_PER_YEAR)
WARMUP = int(asl.PERIODS_PER_YEAR)  # §10 VS_WARMUP: one year of prior deck days
N_BINS = 3
BINS_ROBUST = (2, 5)
KAPPA = {"full": 1.0, "half": 0.5, "quarter": 0.25}
WINDOWS: dict[str, int | None] = {
    "expanding": None,
    "trailing 504": 504,
    "trailing 252": 252,
}
CAPS = ("history ruin bound", "exchange margin", "none")
PRIMARY_WINDOW, PRIMARY_CAP = "expanding", "history ruin bound"
PAIR_BLOCK, PAIR_B, PAIR_SEED = 21, 2000, 0  # the notebook's §10 bootstrap
F_FIXED = 0.03  # §15
MAX_WORKERS = 4
LOG_OPT_ITERS = (
    64  # bisection halvings of a bracket of width <= 1: past double precision
)
GATE_TOL = 1e-9
PERTURB = 50.0  # the causality gate multiplies the future by this factor
FILLS = ("mid", "crossed")
SYMLOG_LIN = 0.01  # figure only: x linear inside +-0.01 (a fifth of the ridge's median |s/rms|, 0.055), log outside
SYMLOG_TICKS = (-1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0)

# rule keys -> professor-facing names
R_AS = "(d) always short, Kelly"
R_SS = "sign(s), Kelly-levered"
R_BLOG = "(b) Kelly by sign of s, log-optimal"
R_BMV = "(b') Kelly by sign of s, mean-variance"
R_A1 = f"(a) Kelly by sign x |s| {N_BINS} bins"
R_A2 = "(a) Kelly by regression on s, |s|"
R_A2S = "(a) Kelly by regression on sign(s), s, |s|"
R_C = "(c) proportional s/mean|s|, sign(s) Kelly stake"
KELLY_RULES = (R_AS, R_SS, R_BLOG, R_BMV, R_A1, R_A2, R_A2S, R_C)
U_AS, U_SS, U_C = "always short", "sign(s)", "(c) proportional s/rms (unit, §10)"
UNIT_RULES = (U_AS, U_SS, U_C)


def bin_rule(n_bins: int, relative: bool = False) -> str:
    return f"(a) Kelly by sign x |s{'/iv' if relative else ''}| {n_bins} bins"


ROBUST_BIN_RULES = tuple(bin_rule(b) for b in BINS_ROBUST) + (
    bin_rule(N_BINS, relative=True),
)


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def load_book(tag: str) -> pd.DataFrame:
    px = pd.read_parquet(DECK / f"daily_{tag}.parquet").sort_index()
    # the stored leg quotes are float32: sum them in their own dtype, then widen -- the notebook's arithmetic
    # (§10: (px["bid_c"] + px["bid_p"]).to_numpy(float)), so the crossed fills match it to the bit
    px["bid"] = (px["bid_c"] + px["bid_p"]).astype(float)
    px["ask"] = (px["ask_c"] + px["ask_p"]).astype(float)
    return px


def check_book(tag: str, px: pd.DataFrame) -> None:
    s = px["signal"].to_numpy(float)
    assert np.array_equal(s, (px["rv_hat"] - px["iv_var"]).to_numpy(float)), tag
    R = px["R"].to_numpy(float)
    assert np.allclose(
        R,
        px["exit"].to_numpy(float) / px["entry"].to_numpy(float) - 1.0,
        rtol=0,
        atol=1e-12,
    ), tag
    bid, ask, ent = (
        px["bid"].to_numpy(float),
        px["ask"].to_numpy(float),
        px["entry"].to_numpy(float),
    )
    assert (bid > 0).all() and (bid <= ent).all() and (ent <= ask).all(), tag
    assert np.isfinite(s).all() and np.isfinite(R).all(), tag


def per_unit(px: pd.DataFrame, fill: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit-premium return of a long (r_L) and of a short (r_S) straddle at the fill."""
    ex = px["exit"].to_numpy(float)
    if fill == "mid":
        R = px["R"].to_numpy(float)
        return R, -R
    return ex / px["ask"].to_numpy(float) - 1.0, 1.0 - ex / px["bid"].to_numpy(float)


def margin_ratio(px: pd.DataFrame, fill: str) -> np.ndarray:
    """P_t / m_t: the short premium over the Cboe short-straddle margin at the fill, both known at 15:30."""
    prem = px["entry"].to_numpy(float) if fill == "mid" else px["bid"].to_numpy(float)
    m = np.array(
        [
            asl.cboe_short_straddle_margin_points(S, kc, kp, p)
            for S, kc, kp, p in zip(px["S"], px["K_c"], px["K_p"], prem, strict=True)
        ]
    )
    return prem / m


def wealth_ret(w: np.ndarray, rL: np.ndarray, rS: np.ndarray) -> np.ndarray:
    return np.where(w > 0, w * rL, np.where(w < 0, -w * rS, 0.0))


def rms_prior(s: np.ndarray) -> np.ndarray:
    """§10's scale: root mean square of s over every earlier deck day (NaN on day 0)."""
    k = np.arange(len(s))
    n_prior = np.where(k > 0, k, np.nan)
    return np.sqrt(np.concatenate([[np.nan], np.cumsum(s**2)[:-1]]) / n_prior)


# ----------------------------------------------------------------------------------------------------------------
# Kelly fractions
# ----------------------------------------------------------------------------------------------------------------
def mv_side(r: np.ndarray) -> float:
    """mean/variance Kelly fraction of one side, 0 when the mean is not positive."""
    if len(r) < 2:
        return 0.0
    m = float(r.mean())
    v = float(r.var(ddof=1))
    return m / v if (m > 0 and v > 0) else 0.0


def mv_two_sided(rL: np.ndarray, rS: np.ndarray) -> tuple[float, bool]:
    """Signed mean-variance Kelly fraction on the better side; flag when both sides had a positive mean."""
    fL, fS = mv_side(rL), mv_side(rS)
    both = fL > 0 and fS > 0
    return (fL if fL >= fS else -fS), both


def log_opt_side(r: np.ndarray) -> tuple[float, float]:
    """argmax_{f >= 0} mean log(1 + f r) and its value; (0, 0) when the mean is not positive."""
    if len(r) < 2 or not float(r.mean()) > 0:
        return 0.0, 0.0
    worst = float(-r.min())
    assert worst > 0, "a straddle side with no losing day"
    lo, hi = 0.0, 1.0 / worst
    for _ in range(LOG_OPT_ITERS):
        mid = 0.5 * (lo + hi)
        if float(np.mean(r / (1.0 + mid * r))) > 0:
            lo = mid
        else:
            hi = mid
    return lo, float(np.mean(np.log1p(lo * r)))


def log_two_sided(rL: np.ndarray, rS: np.ndarray) -> float:
    fL, gL = log_opt_side(rL)
    fS, gS = log_opt_side(rS)
    return fL if gL >= gS else -fS


def bin_cell(a_past: np.ndarray, a_today: float, n_bins: int) -> np.ndarray:
    """Mask of the past days in today's equal-count bin (edges = quantiles of the past values)."""
    edges = np.quantile(a_past, np.arange(1, n_bins) / n_bins)
    return np.searchsorted(edges, a_past, side="left") == np.searchsorted(
        edges, a_today, side="left"
    )


def full_kelly(
    s: np.ndarray, iv: np.ndarray, rL: np.ndarray, rS: np.ndarray, window: int | None
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Full-Kelly signed fractions of every rule on every day >= WARMUP (NaN before), from days < t only."""
    n = len(s)
    rules = list(KELLY_RULES) + list(ROBUST_BIN_RULES)
    f = {r: np.full(n, np.nan) for r in rules}
    flags = {"both_sides_positive": 0, "min_cell": 10**9}
    signbook = np.where(s > 0, rL, rS)  # the sign(s) book, per unit
    for t in range(WARMUP, n):
        lo = 0 if window is None else max(0, t - window)
        sl = slice(lo, t)
        sP, rLP, rSP = s[sl], rL[sl], rS[sl]
        today_long = s[t] > 0
        same = (sP > 0) == today_long
        # (d) always short
        f[R_AS][t] = -mv_side(rSP)
        # sign(s), one Kelly fraction for its book
        f[R_SS][t] = (1.0 if today_long else -1.0) * max(mv_side(signbook[sl]), 0.0)
        # (b) / (b') by sign
        f[R_BLOG][t] = log_two_sided(rLP[same], rSP[same])
        f[R_BMV][t], both = mv_two_sided(rLP[same], rSP[same])
        flags["both_sides_positive"] += int(both)
        # (a) bins of |s| (and of |s|/iv) within the sign
        for rule, nb, a in (
            [(R_A1, N_BINS, np.abs(s))]
            + [(bin_rule(b), b, np.abs(s)) for b in BINS_ROBUST]
            + [(bin_rule(N_BINS, relative=True), N_BINS, np.abs(s) / iv)]
        ):
            aP = a[sl][same]
            cell = bin_cell(aP, float(a[t]), nb)
            if rule == R_A1:
                flags["min_cell"] = min(flags["min_cell"], int(cell.sum()))
            f[rule][t], both = mv_two_sided(rLP[same][cell], rSP[same][cell])
            flags["both_sides_positive"] += int(both)
        # (a) regressions on [1, s/rms, |s|/rms] (literal) and on [1, 1{s>0}, s/rms, |s|/rms] (nests the sign
        # rule: one intercept per sign), rms over the window; residual variance over the past same-sign days
        rms_w = float(np.sqrt(np.mean(sP**2)))
        zP, z_t = sP / rms_w, s[t] / rms_w
        for rule, X, x_t in (
            (
                R_A2,
                np.column_stack([np.ones(len(sP)), zP, np.abs(zP)]),
                np.array([1.0, z_t, abs(z_t)]),
            ),
            (
                R_A2S,
                np.column_stack(
                    [np.ones(len(sP)), (sP > 0).astype(float), zP, np.abs(zP)]
                ),
                np.array([1.0, float(today_long), z_t, abs(z_t)]),
            ),
        ):
            bL = np.linalg.lstsq(X, rLP, rcond=None)[0]
            bS = np.linalg.lstsq(X, rSP, rcond=None)[0]
            muL, muS = float(x_t @ bL), float(x_t @ bS)
            vL = float((rLP - X @ bL)[same].var(ddof=1))
            vS = float((rSP - X @ bS)[same].var(ddof=1))
            fL = muL / vL if muL > 0 else 0.0
            fS = muS / vS if muS > 0 else 0.0
            flags["both_sides_positive"] += int(fL > 0 and fS > 0)
            f[rule][t] = fL if fL >= fS else -fS
        # (c) proportional to s at the sign(s) Kelly stake: s over the mean |s| of the window (average size one)
        f[R_C][t] = f[R_SS][t] * abs(s[t]) / float(np.mean(np.abs(sP)))
    return f, flags


def caps_for(
    kind: str, rL: np.ndarray, rS: np.ndarray, pm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(long cap, short cap) on every day, from earlier days (the ruin bound) and the day's own quote (margin)."""
    n = len(rL)
    if kind == "none":
        return np.full(n, np.inf), np.full(n, np.inf)
    worstL = np.concatenate([[np.nan], np.maximum.accumulate(-rL)[:-1]])
    worstS = np.concatenate([[np.nan], np.maximum.accumulate(-rS)[:-1]])
    capL, capS = 1.0 / worstL, 1.0 / worstS
    if kind == "exchange margin":
        capS = np.minimum(capS, pm)
    return capL, capS


def apply(
    fstar: np.ndarray, kappa: float, capL: np.ndarray, capS: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """w = sign(f*) min(kappa |f*|, cap of that side); also the at-cap mask."""
    a = kappa * np.abs(fstar)
    cap = np.where(fstar > 0, capL, capS)
    w = np.sign(fstar) * np.minimum(a, cap)
    at_cap = (fstar != 0) & (a >= cap)
    return w, at_cap


# ----------------------------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------------------------
def boot_idx(n: int) -> np.ndarray:
    return asl.circular_block_bootstrap_idx(
        np.random.default_rng([PAIR_SEED, n]), n, PAIR_BLOCK, PAIR_B
    )


def sharpe(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd * np.sqrt(ANN)) if sd > 0 else float("nan")


def boot_sharpe(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = x[idx]
    sd = d.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return d.mean(axis=1) / sd * np.sqrt(ANN)


def log_growth(x: np.ndarray) -> np.ndarray | None:
    """Daily log wealth factors, or None when a day takes wealth to zero or below (ruin)."""
    if (x <= -1.0).any():
        return None
    return np.log1p(x)


def reading(lo: float, hi: float) -> str:
    # §10's _interval_reading: a bound within 5 % of the width from zero is a knife edge
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    width = hi - lo
    edge = min(abs(lo), abs(hi)) < 0.05 * width
    if lo > 0 or hi < 0:
        return "knife-edge, excludes zero" if edge else "excludes zero"
    return "knife-edge, includes zero" if edge else "includes zero"


def pct_ci(a: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(a).all():
        return float("nan"), float("nan")
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def book_stats(
    x: np.ndarray,
    w: np.ndarray,
    xg: np.ndarray,
    sgn: np.ndarray,
    at_cap: np.ndarray | None,
    days: pd.DatetimeIndex,
) -> dict:
    """x: the daily return at the rule's weight (Sharpe, mean); xg: the return at the growth weight."""
    ell = log_growth(xg)
    wealth = np.cumprod(1.0 + xg)
    ruined = ell is None
    rec = {
        "days": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "Sharpe": sharpe(x),
        "mean per unit |w|": float(np.mean(x) / np.mean(np.abs(w)))
        if np.mean(np.abs(w)) > 0
        else float("nan"),
        "g_ann": float(ANN * np.mean(ell)) if ell is not None else float("-inf"),
        "terminal wealth": float(wealth[-1]) if not ruined else 0.0,
        "maxDD_frac": float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
        if not ruined
        else -1.0,
        "worst day factor": float(np.min(1.0 + xg)),
        "worst day": str(days[int(np.argmin(xg))].date()),
        "ruined": bool(ruined),
        "ruin day": str(days[int(np.argmax(xg <= -1.0))].date()) if ruined else "",
        "mean |w|": float(np.mean(np.abs(w))),
        "max |w|": float(np.max(np.abs(w))),
        "mean |dw|": float(np.mean(np.abs(np.diff(w)))),
        "share zero": float(np.mean(w == 0)),
        "share long": float(np.mean(w > 0)),
        "share short": float(np.mean(w < 0)),
        "share opposite sign(s)": float(np.mean(w * sgn < 0)),
        "share at cap": float(np.mean(at_cap)) if at_cap is not None else float("nan"),
    }
    return rec


def score_spec(
    tag: str,
    fill: str,
    window: str,
    cap: str,
    fstar: dict[str, np.ndarray],
    rL: np.ndarray,
    rS: np.ndarray,
    s: np.ndarray,
    pm: np.ndarray,
    days: pd.DatetimeIndex,
    idx: np.ndarray,
    rules: tuple[str, ...],
) -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    sc = slice(WARMUP, len(s))
    rLs, rSs, ds = rL[sc], rS[sc], days[sc]
    sgn = np.where(s[sc] > 0, 1.0, -1.0)
    capL, capS = caps_for(cap, rL, rS, pm)
    capL, capS = capL[sc], capS[sc]
    # unit references
    q10 = (s / rms_prior(s))[sc]
    units = {U_AS: -np.ones_like(sgn), U_SS: sgn, U_C: q10}
    ux = {k: wealth_ret(w, rLs, rSs) for k, w in units.items()}
    ugx = {k: wealth_ret(F_FIXED * w, rLs, rSs) for k, w in units.items()}
    bs_unit = {k: boot_sharpe(v, idx) for k, v in ux.items()}
    ug_ell = {k: log_growth(v) for k, v in ugx.items()}
    rows: list[dict] = []
    weights: dict[tuple[str, str], np.ndarray] = {}

    def comparisons(
        x: np.ndarray, xg: np.ndarray, ref_g: dict[str, tuple[np.ndarray, float] | None]
    ) -> dict:
        out: dict = {}
        bsx = boot_sharpe(x, idx)
        for lab, ref in (("sign(s)", U_SS), ("always short", U_AS)):
            d = sharpe(x) - sharpe(ux[ref])
            lo, hi = pct_ci(bsx - bs_unit[ref])
            out.update(
                {
                    f"dSharpe vs {lab}": d,
                    f"lo vs {lab}": lo,
                    f"hi vs {lab}": hi,
                    f"reading vs {lab}": reading(lo, hi),
                }
            )
        ell = log_growth(xg)
        bg = ANN * ell[idx].mean(axis=1) if ell is not None else None
        g = float(ANN * ell.mean()) if ell is not None else float("-inf")
        for lab, rg in ref_g.items():
            key = f"dg vs {lab}"
            if bg is None or rg is None or not np.isfinite(rg[1]):
                out.update(
                    {
                        key: float("nan"),
                        f"lo {key[3:]}": float("nan"),
                        f"hi {key[3:]}": float("nan"),
                        f"reading {key[3:]}": "ruined",
                    }
                )
                continue
            lo, hi = pct_ci(bg - rg[0])
            out.update(
                {
                    key: g - rg[1],
                    f"lo {key[3:]}": lo,
                    f"hi {key[3:]}": hi,
                    f"reading {key[3:]}": reading(lo, hi),
                }
            )
        return out

    def ref_pair(ell: np.ndarray | None) -> tuple[np.ndarray, float] | None:
        if ell is None:
            return None
        return ANN * ell[idx].mean(axis=1), float(ANN * ell.mean())

    fixed_ref = ref_pair(ug_ell[U_SS])
    for k in UNIT_RULES:
        w = units[k]
        rec = {
            "tag": tag,
            "forecast": asl.YHAT_LABEL[tag],
            "fill": fill,
            "window": window,
            "cap": cap,
            "rule": k,
            "kelly": "unit (growth at 3%)",
        }
        rec.update(book_stats(ux[k], w, ugx[k], sgn, None, ds))
        refs = {
            "sign(s) Kelly-levered": fixed_ref,
            "always-short Kelly": ref_pair(ug_ell[U_AS]),
            "sign(s) at 3%": fixed_ref,
        }
        rec.update(comparisons(ux[k], ugx[k], refs))
        rows.append(rec)
        weights[(k, "unit")] = w
    for kname, kap in KAPPA.items():
        wk = {r: apply(fstar[r][sc], kap, capL, capS) for r in rules if r in fstar}
        xk = {r: wealth_ret(v[0], rLs, rSs) for r, v in wk.items()}
        refs = {
            "sign(s) Kelly-levered": ref_pair(log_growth(xk[R_SS])),
            "always-short Kelly": ref_pair(log_growth(xk[R_AS])),
            "sign(s) at 3%": fixed_ref,
        }
        for r in rules:
            w, at_cap = wk[r]
            assert np.isfinite(w).all(), (tag, fill, window, cap, r)
            assert (np.where(w > 0, w <= capL + 1e-15, True)).all() and (
                np.where(w < 0, -w <= capS + 1e-15, True)
            ).all()
            rec = {
                "tag": tag,
                "forecast": asl.YHAT_LABEL[tag],
                "fill": fill,
                "window": window,
                "cap": cap,
                "rule": r,
                "kelly": kname,
            }
            rec.update(book_stats(xk[r], w, xk[r], sgn, at_cap, ds))
            rec.update(comparisons(xk[r], xk[r], refs))
            rows.append(rec)
            weights[(r, kname)] = w
    return rows, weights


def bins_table(
    tag: str,
    fill: str,
    s: np.ndarray,
    rL: np.ndarray,
    rS: np.ndarray,
    days: pd.DatetimeIndex,
) -> list[dict]:
    """The conditional mean and sd by cell at the last refit: every deck day before the last one (expanding)."""
    t = len(s) - 1
    sP, rLP, rSP = s[:t], rL[:t], rS[:t]
    out = []
    for side, mask in (
        ("s > 0 (sign(s) long)", sP > 0),
        ("s <= 0 (sign(s) short)", sP <= 0),
    ):
        a = np.abs(sP[mask])
        edges = np.quantile(a, np.arange(1, N_BINS) / N_BINS)
        b = np.searchsorted(edges, a, side="left")
        bounds = np.concatenate([[a.min()], edges, [a.max()]])
        cells = [("all (sign only)", np.ones(len(a), bool), a.min(), a.max())]
        cells += [
            (f"|s| bin {k + 1} of {N_BINS}", b == k, bounds[k], bounds[k + 1])
            for k in range(N_BINS)
        ]
        for name, c, lo, hi in cells:
            xL, xS = rLP[mask][c], rSP[mask][c]
            fmv, _ = mv_two_sided(xL, xS)
            out.append(
                {
                    "tag": tag,
                    "forecast": asl.YHAT_LABEL[tag],
                    "fill": fill,
                    "refit on": str(days[t].date()),
                    "history days": t,
                    "side": side,
                    "cell": name,
                    "|s| from": float(lo),
                    "|s| to": float(hi),
                    "n": int(c.sum()),
                    "mean r_long": float(xL.mean()),
                    "sd r_long": float(xL.std(ddof=1)),
                    "mean r_short": float(xS.mean()),
                    "sd r_short": float(xS.std(ddof=1)),
                    "hit rate long (r_long > 0)": float((xL > 0).mean()),
                    "full-Kelly f (mean-variance, signed)": fmv,
                    "full-Kelly f (log-optimal, signed)": log_two_sided(xL, xS),
                }
            )
    return out


# ----------------------------------------------------------------------------------------------------------------
# worker: one forecast
# ----------------------------------------------------------------------------------------------------------------
def run_forecast(tag: str) -> dict:
    px = load_book(tag)
    check_book(tag, px)
    days = pd.DatetimeIndex(px.index)
    s, iv = px["signal"].to_numpy(float), px["iv_var"].to_numpy(float)
    idx = boot_idx(len(s) - WARMUP)
    prim_rules = tuple(KELLY_RULES)
    rob_rules = tuple(KELLY_RULES) + ROBUST_BIN_RULES
    rows: list[dict] = []
    fig: dict = {}
    bins: list[dict] = []
    flags: dict = {}
    for fill in FILLS:
        rL, rS = per_unit(px, fill)
        pm = margin_ratio(px, fill)
        bins += bins_table(tag, fill, s, rL, rS, days)
        for wname, wlen in WINDOWS.items():
            fstar, fl = full_kelly(s, iv, rL, rS, wlen)
            flags[(fill, wname)] = fl
            for cap in CAPS:
                prim = wname == PRIMARY_WINDOW and cap == PRIMARY_CAP
                r, w = score_spec(
                    tag,
                    fill,
                    wname,
                    cap,
                    fstar,
                    rL,
                    rS,
                    s,
                    pm,
                    days,
                    idx,
                    rob_rules if prim else prim_rules,
                )
                for rec in r:
                    rec["primary"] = prim and rec["rule"] not in ROBUST_BIN_RULES
                rows += r
                if prim and tag in FIG_TAGS:
                    fig[fill] = {
                        "w": w,
                        "fstar": {k: v[WARMUP:] for k, v in fstar.items()},
                    }
        fig.setdefault("pm", {})[fill] = pm[WARMUP:]
    fig["days"] = days[WARMUP:]
    fig["s_over_rms"] = (s / rms_prior(s))[WARMUP:]
    fig["rL_mid"] = px["R"].to_numpy(float)[WARMUP:]
    sgn = np.where(s > 0, 1.0, -1.0)
    sh866 = sharpe(sgn * px["R"].to_numpy(float))
    return {
        "tag": tag,
        "rows": rows,
        "bins": bins,
        "fig": fig if tag in FIG_TAGS else None,
        "flags": flags,
        "sharpe866": sh866,
        "max_abs_q": float(np.nanmax(np.abs(s / rms_prior(s))[WARMUP:])),
        "days": days,
    }


# ----------------------------------------------------------------------------------------------------------------
# gates run in the main process
# ----------------------------------------------------------------------------------------------------------------
def causality_gate(tag: str) -> str:
    px = load_book(tag)
    s, iv = px["signal"].to_numpy(float), px["iv_var"].to_numpy(float)
    k = WARMUP + (len(s) - WARMUP) // 2
    rng = np.random.default_rng(1)
    msgs = []
    for fill in FILLS:
        rL, rS = per_unit(px, fill)
        pm = margin_ratio(px, fill)
        rL2, rS2, s2, iv2 = rL.copy(), rS.copy(), s.copy(), iv.copy()
        rL2[k:] = rL[k:] * PERTURB * rng.standard_normal(len(s) - k)
        rS2[k:] = rS[k:] * PERTURB * rng.standard_normal(len(s) - k)
        s2[k + 1 :] = s[k + 1 :] * PERTURB * rng.standard_normal(len(s) - k - 1)
        iv2[k + 1 :] = iv[k + 1 :] * PERTURB * rng.uniform(0.5, 1.5, len(s) - k - 1)
        for wname, wlen in WINDOWS.items():
            f1, _ = full_kelly(s, iv, rL, rS, wlen)
            f2, _ = full_kelly(s2, iv2, rL2, rS2, wlen)
            n_moved = 0
            for r in f1:
                a, b = f1[r][: k + 1], f2[r][: k + 1]
                assert np.array_equal(a, b, equal_nan=True), (fill, wname, r)
                n_moved += int(
                    not np.array_equal(f1[r][k + 1 :], f2[r][k + 1 :], equal_nan=True)
                )
            assert n_moved > 0, (
                fill,
                wname,
                "the perturbation never reached the estimators",
            )
            msgs.append(
                f"{fill}/{wname}: later fractions move in {n_moved} of {len(f1)} rules"
            )
        for cap in CAPS:
            c1, c2 = caps_for(cap, rL, rS, pm), caps_for(cap, rL2, rS2, pm)
            for a, b in zip(c1, c2, strict=True):
                assert np.array_equal(a[: k + 1], b[: k + 1], equal_nan=True), (
                    fill,
                    cap,
                )
    return (
        f"causality gate ({tag}): per-unit returns x{PERTURB:g}-perturbed on days >= {k} and s, iv on days > {k} "
        f"(k = {px.index[k].date()}); every fraction and cap on days <= {k} unchanged for every rule, window, "
        f"cap and fill ({'; '.join(msgs)})"
    )


def notebook_gates(rules: pd.DataFrame, res: list[dict]) -> list[str]:
    msgs = []
    ref866 = pd.read_csv(DECK / "rule_by_strategy_sign_s.csv", index_col=0)[
        "Sharpe_ann"
    ]
    for r in res:
        lab = asl.YHAT_LABEL[r["tag"]]
        assert abs(r["sharpe866"] - float(ref866.loc[lab])) < GATE_TOL, (
            r["tag"],
            r["sharpe866"],
            ref866.loc[lab],
        )
    msgs.append(
        f"gate: sign(s) midpoint Sharpe on the 866 days equals §10's rule table for all {len(res)} forecasts"
    )
    vs = pd.read_csv(DECK / "vrp_sized_rules.csv")
    vs = vs[vs["block"] == f"after {WARMUP}-day warm-up"]
    prim = rules[rules["primary"]]
    n_chk = 0
    for tag in TAGS:
        mine = prim[(prim["tag"] == tag)]
        for vrule, myrule in (("sign(s)", U_SS), ("(a) proportional s/rms", U_C)):
            ref = vs[(vs["tag"] == tag) & (vs["rule"] == vrule)].iloc[0]
            for fill, col in (("mid", "mid"), ("crossed", "crossed")):
                m = mine[(mine["rule"] == myrule) & (mine["fill"] == fill)].iloc[0]
                assert int(m["days"]) == int(ref["days"]), (tag, vrule)
                assert abs(m["Sharpe"] - ref[f"Sharpe {col}"]) < GATE_TOL, (
                    tag,
                    vrule,
                    fill,
                )
                if vrule != "sign(s)":
                    for a, b in (
                        ("dSharpe vs sign(s)", f"dSharpe {col}"),
                        ("lo vs sign(s)", f"lo {col}"),
                        ("hi vs sign(s)", f"hi {col}"),
                    ):
                        assert abs(m[a] - ref[b]) < GATE_TOL, (
                            tag,
                            vrule,
                            fill,
                            a,
                            m[a],
                            ref[b],
                        )
                n_chk += 1
    msgs.append(
        f"gate: on the {int(vs['days'].iloc[0])}-day block, sign(s) and the proportional rule reproduce §10's "
        f"vrp_sized_rules.csv ({n_chk} forecast x rule x fill rows: Sharpe ratios, and the proportional rule's "
        f"difference and paired interval, to {GATE_TOL:g}) -- same days, same fills, same bootstrap draws"
    )
    return msgs


# ----------------------------------------------------------------------------------------------------------------
# summaries and figures
# ----------------------------------------------------------------------------------------------------------------
def _med(x: pd.Series) -> float:
    """Median ignoring NaN (a ruined path's -inf growth counts as the lowest value); NaN, without a warning,
    when every value is NaN."""
    x = x.astype(float).dropna()
    return float(x.median()) if len(x) else float("nan")


def summarise(rules: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (win, cap, fill, rule, kel), g in rules.groupby(
        ["window", "cap", "fill", "rule", "kelly"], sort=False
    ):
        rec = {
            "window": win,
            "cap": cap,
            "fill": fill,
            "rule": rule,
            "kelly": kel,
            "forecasts": len(g),
        }
        for lab in ("sign(s)", "always short"):
            d = g[f"dSharpe vs {lab}"]
            rec.update(
                {
                    f"dSharpe vs {lab}: >0": int((d > 0).sum()),
                    f"dSharpe vs {lab}: CI>0": int((g[f"lo vs {lab}"] > 0).sum()),
                    f"dSharpe vs {lab}: CI<0": int((g[f"hi vs {lab}"] < 0).sum()),
                    f"dSharpe vs {lab}: median": _med(d),
                    f"dSharpe vs {lab}: min": float(d.min()),
                    f"dSharpe vs {lab}: max": float(d.max()),
                }
            )
        for lab in ("sign(s) Kelly-levered", "always-short Kelly", "sign(s) at 3%"):
            d = g[f"dg vs {lab}"]
            rec.update(
                {
                    f"dg vs {lab}: >0": int((d > 0).sum()),
                    f"dg vs {lab}: CI>0": int((g[f"lo vs {lab}"] > 0).sum()),
                    f"dg vs {lab}: CI<0": int((g[f"hi vs {lab}"] < 0).sum()),
                    f"dg vs {lab}: median": _med(d),
                }
            )
        rec.update(
            {
                "Sharpe median": _med(g["Sharpe"]),
                "g_ann median": _med(g["g_ann"]),
                "ruined": int(g["ruined"].sum()),
                "maxDD_frac median": _med(g["maxDD_frac"]),
                "mean |w| median": _med(g["mean |w|"]),
                "share zero median": _med(g["share zero"]),
                "share opposite sign(s) median": _med(g["share opposite sign(s)"]),
                "share at cap median": _med(g["share at cap"]),
            }
        )
        out.append(rec)
    return pd.DataFrame(out)


# categorical slots of the reference palette, in fixed order (validated: scripts/validate_palette.js)
C_SIGN, C_BIN, C_BLOG, C_SSK, C_AS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#4a3aa7",
    "#e34948",
)


def plot_wealth(figs: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(
        len(FIG_TAGS), 2, figsize=(12, 7.2), sharex=True, sharey=True
    )
    lines = [
        (U_SS, "unit", "sign(s) at a fixed 3% of wealth (§15)", C_SIGN, "-", 1.4),
        (R_SS, "half", "sign(s), ½-Kelly-levered", C_SSK, "-", 1.1),
        (R_A1, "half", f"(a) ½-Kelly by sign x |s| {N_BINS} bins", C_BIN, "-", 1.1),
        (R_BLOG, "half", "(b) ½-Kelly by sign of s, log-optimal", C_BLOG, "-", 1.1),
        (R_AS, "half", "(d) always short, ½-Kelly", C_AS, "-", 1.1),
    ]
    for i, tag in enumerate(FIG_TAGS):
        fg = figs[tag]
        for j, fill in enumerate(FILLS):
            ax = axes[i, j]
            rL = fg[fill]["rL"]
            rS = fg[fill]["rS"]
            for rule, kel, lab, c, ls, lw in lines:
                w = fg[fill]["w"][(rule, kel)]
                w = F_FIXED * w if kel == "unit" else w
                wealth = np.cumprod(1.0 + wealth_ret(w, rL, rS))
                ax.plot(fg["days"], wealth, ls, color=c, lw=lw, label=lab)
                ax.annotate(
                    f"{wealth[-1]:.2f}×",
                    (fg["days"][-1], wealth[-1]),
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=7,
                    va="center",
                    color="#52514e",
                )
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([0.1, 0.25, 0.5, 1, 2, 4, 8]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.axhline(1.0, color="#c3c2b7", lw=0.6)
            ax.grid(axis="y", color="#e1e0d9", lw=0.5)
            ax.set_title(
                f"{asl.YHAT_LABEL[tag]} — {'midpoint' if fill == 'mid' else 'crossed spread'} fills",
                fontsize=9,
                color="#0b0b0b",
            )
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel("wealth multiple (log scale)", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle(
        "15:30 straddle, last-30-min trade: compounded wealth on the 614 days after the one-year warm-up "
        "(weights from earlier days only; caps: history ruin bound)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_f_vs_s(fg: dict, tag: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(R_A1, C_BIN), (R_A2S, C_SIGN), (R_BLOG, C_BLOG), (R_C, C_SSK)]
    fig, axes = plt.subplots(
        2, len(panels), figsize=(13, 6.2), sharex=True, sharey="row"
    )
    x = fg["s_over_rms"]
    for i, fill in enumerate(FILLS):
        for j, (rule, c) in enumerate(panels):
            ax = axes[i, j]
            w = fg[fill]["w"][(rule, "full")]
            ax.scatter(x, w, s=9, color=c, alpha=0.55, linewidths=0)
            ax.set_xscale("symlog", linthresh=SYMLOG_LIN)
            ax.set_xticks(SYMLOG_TICKS)
            ax.set_xticklabels([f"{v:g}" for v in SYMLOG_TICKS], fontsize=7)
            ax.minorticks_off()
            ax.axhline(0.0, color="#c3c2b7", lw=0.6)
            ax.axvline(0.0, color="#c3c2b7", lw=0.6)
            ax.grid(color="#e1e0d9", lw=0.4)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            n0 = int((w == 0).sum())
            ax.set_title(
                f"{rule}\n{'midpoint' if fill == 'mid' else 'crossed'}; {n0} of {len(w)} days at 0",
                fontsize=8,
                color="#0b0b0b",
            )
            if j == 0:
                ax.set_ylabel(
                    "full-Kelly weight $w_t$ (share of wealth as premium)", fontsize=8
                )
    fig.supxlabel(
        rf"forecast gap $s_t/\mathrm{{rms}}_t$ (rms of $s$ over earlier days, as §10); symmetric-log axis, "
        rf"linear inside $\pm${SYMLOG_LIN:g}",
        fontsize=8,
    )
    fig.suptitle(
        f"{asl.YHAT_LABEL[tag]}: full-Kelly weight against the day's forecast gap, 614 scored days "
        "(positive = long the straddle; capped at the history ruin bound)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = None
    for tag in TAGS:
        px = load_book(tag)
        check_book(tag, px)
        common = px.index if common is None else common
        assert px.index.equals(common), tag
    assert common is not None
    print(
        f"gate: the {len(TAGS)} books share the same {len(common)} days ({common.min().date()} -> "
        f"{common.max().date()}); R == exit/entry - 1, s == rv_hat - iv_var, 0 < bid <= entry <= ask on every day"
    )
    print(
        f"scored: the {len(common) - WARMUP} days after the {WARMUP}-day warm-up "
        f"({common[WARMUP].date()} -> {common.max().date()}), identical rows for every rule"
    )

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        res = list(pool.map(run_forecast, TAGS))
    allrows = pd.DataFrame([r for x in res for r in x["rows"]])
    for msg in notebook_gates(allrows, res):
        print(msg)
    print(causality_gate("blk2"))
    for x in res:
        fl = x["flags"]
        both = sum(v["both_sides_positive"] for v in fl.values())
        mc = min(v["min_cell"] for v in fl.values())
        print(
            f"  {x['tag']:<15} max |s/rms| on the scored days {x['max_abs_q']:.2f}; smallest {N_BINS}-bin cell "
            f"{mc} days (any window); days where both sides had a positive mean {both}"
        )

    prim = allrows[allrows["primary"]].drop(columns=["primary"]).reset_index(drop=True)
    prim.to_csv(OUT / "kelly_rules.csv", index=False)
    rob_cols = [
        "tag",
        "fill",
        "window",
        "cap",
        "rule",
        "kelly",
        "days",
        "Sharpe",
        "dSharpe vs sign(s)",
        "lo vs sign(s)",
        "hi vs sign(s)",
        "dSharpe vs always short",
        "lo vs always short",
        "hi vs always short",
        "g_ann",
        "dg vs sign(s) Kelly-levered",
        "lo vs sign(s) Kelly-levered",
        "hi vs sign(s) Kelly-levered",
        "dg vs always-short Kelly",
        "lo vs always-short Kelly",
        "hi vs always-short Kelly",
        "maxDD_frac",
        "ruined",
        "ruin day",
        "mean |w|",
        "share zero",
        "share at cap",
    ]
    # six significant digits and the tag only (labels in kelly_rules.csv) keep the file small; kelly_rules.csv
    # carries the primary rows at full precision
    allrows.drop(columns=["primary"])[rob_cols].to_csv(
        OUT / "kelly_robustness.csv", index=False, float_format="%.6g"
    )
    pd.DataFrame([b for x in res for b in x["bins"]]).to_csv(
        OUT / "kelly_bins.csv", index=False
    )
    summ = summarise(allrows.drop(columns=["primary"]))
    summ.to_csv(OUT / "kelly_summary.csv", index=False)

    # the shape of the conditional mean by |s| tercile at the last refit (midpoint): is the edge rising in |s|?
    bins_df = pd.DataFrame([b for x in res for b in x["bins"]])
    bm = bins_df[bins_df["fill"] == "mid"].pivot_table(
        index="tag", columns=["side", "cell"], values="mean r_long"
    )
    lng, sht = "s > 0 (sign(s) long)", "s <= 0 (sign(s) short)"
    t1, t2, t3 = (f"|s| bin {k} of {N_BINS}" for k in (1, 2, N_BINS))
    L1, L2, L3 = bm[(lng, t1)], bm[(lng, t2)], bm[(lng, t3)]
    S1, S2, S3 = bm[(sht, t1)], bm[(sht, t2)], bm[(sht, t3)]
    print(
        f"conditional mean of the long straddle's midpoint return by |s| tercile, last refit ({len(bm)} forecasts):"
        f" long side (s > 0): top tercile the lowest of the three in {int(((L3 < L1) & (L3 < L2)).sum())},"
        f" negative in {int((L3 < 0).sum())}; rising across the terciles in {int(((L1 < L2) & (L2 < L3)).sum())}."
        f" Short side (s <= 0): the middle tercile pays the short most in {int(((S2 < S1) & (S2 < S3)).sum())};"
        f" the top tercile pays it most in {int(((S3 < S1) & (S3 < S2)).sum())}"
    )
    kel = allrows[~allrows["kelly"].str.startswith("unit")]
    print(
        f"every spec (windows x caps x bins x kappa x fill x forecast): {len(kel)} Kelly rows; Sharpe interval vs "
        f"sign(s) above zero in {int((kel['lo vs sign(s)'] > 0).sum())}, below zero in "
        f"{int((kel['hi vs sign(s)'] < 0).sum())}; growth interval vs sign(s) at 3% above zero in "
        f"{int((kel['lo vs sign(s) at 3%'] > 0).sum())}, below in {int((kel['hi vs sign(s) at 3%'] < 0).sum())}; "
        f"ruined paths {int(kel['ruined'].sum())} (history ruin bound: "
        f"{int(kel.loc[kel['cap'] == 'history ruin bound', 'ruined'].sum())}, days "
        f"{sorted(set(kel.loc[kel['ruined'] & (kel['cap'] == 'history ruin bound'), 'ruin day']))})"
    )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v: .3f}")
    ps = summ[(summ["window"] == PRIMARY_WINDOW) & (summ["cap"] == PRIMARY_CAP)]
    cols = [
        "fill",
        "rule",
        "kelly",
        "Sharpe median",
        "dSharpe vs sign(s): >0",
        "dSharpe vs sign(s): CI>0",
        "dSharpe vs sign(s): CI<0",
        "dSharpe vs sign(s): median",
        "dSharpe vs always short: CI>0",
        "dSharpe vs always short: median",
        "g_ann median",
        "dg vs sign(s) Kelly-levered: CI>0",
        "dg vs sign(s) Kelly-levered: CI<0",
        "dg vs sign(s) Kelly-levered: median",
        "ruined",
        "maxDD_frac median",
        "mean |w| median",
        "share zero median",
    ]
    print(
        f"\nPRIMARY (window {PRIMARY_WINDOW}, cap {PRIMARY_CAP}, {N_BINS} bins): across the {len(TAGS)} forecasts"
    )
    print(ps[cols].to_string(index=False))
    for tag in FIG_TAGS:
        print(f"\n{asl.YHAT_LABEL[tag]} (primary spec)")
        c2 = [
            "fill",
            "rule",
            "kelly",
            "Sharpe",
            "dSharpe vs sign(s)",
            "lo vs sign(s)",
            "hi vs sign(s)",
            "dSharpe vs always short",
            "lo vs always short",
            "hi vs always short",
            "g_ann",
            "dg vs sign(s) Kelly-levered",
            "lo vs sign(s) Kelly-levered",
            "hi vs sign(s) Kelly-levered",
            "dg vs sign(s) at 3%",
            "lo vs sign(s) at 3%",
            "hi vs sign(s) at 3%",
            "maxDD_frac",
            "worst day",
            "ruined",
            "mean |w|",
            "mean |dw|",
            "share zero",
            "share opposite sign(s)",
            "share at cap",
        ]
        print(prim[prim["tag"] == tag][c2].to_string(index=False))
    rc = [
        "window",
        "cap",
        "fill",
        "rule",
        "kelly",
        "Sharpe median",
        "dSharpe vs sign(s): CI>0",
        "dSharpe vs sign(s): CI<0",
        "dSharpe vs sign(s): median",
        "g_ann median",
        "dg vs sign(s) Kelly-levered: median",
        "ruined",
        "maxDD_frac median",
        "share at cap median",
    ]
    rs = summ[(summ["kelly"].isin(["half"])) & summ["rule"].isin(list(KELLY_RULES))]
    print("\nROBUSTNESS, ½-Kelly, across the forecasts: every window x cap")
    print(rs[rc].to_string(index=False))
    bn = summ[
        (summ["rule"].isin(ROBUST_BIN_RULES + (R_A1,)))
        & (summ["window"] == PRIMARY_WINDOW)
        & (summ["cap"] == PRIMARY_CAP)
    ]
    print("\nROBUSTNESS, bins")
    print(bn[rc].to_string(index=False))

    figs = {}
    for x in res:
        if x["fig"] is None:
            continue
        fg = x["fig"]
        px = load_book(x["tag"])
        for fill in FILLS:
            rL, rS = per_unit(px, fill)
            fg[fill]["rL"], fg[fill]["rS"] = rL[WARMUP:], rS[WARMUP:]
        figs[x["tag"]] = fg
    plot_wealth(figs, OUT / "kelly_wealth.png")
    plot_f_vs_s(figs["blk2"], "blk2", OUT / "kelly_f_vs_s.png")
    print(
        "\nsaved",
        *(
            OUT / n
            for n in (
                "kelly_rules.csv",
                "kelly_robustness.csv",
                "kelly_bins.csv",
                "kelly_summary.csv",
                "kelly_wealth.png",
                "kelly_f_vs_s.png",
            )
        ),
    )


if __name__ == "__main__":
    main()
