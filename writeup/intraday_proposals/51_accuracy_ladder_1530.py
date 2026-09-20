"""51 - the accuracy ladder of the 0DTE straddle trade at 15:30 and at 15:00.

The trade of record is the deck's (``notebooks/_write_0dte_nb.py``, sections
4-10): at 15:30 ET the nearest-OTM SPXW 0DTE call and put are bought or sold at
the quoted midpoint, one unit of entry premium a day, and settle in cash against
the official close.  The signal is

    s_t = rv_hat_t - slice_t,

the recalibrated forecast of the bar the trade holds minus the implied slice of
the same window, and the published rule is sign(s): q = +1 when s > 0, q = -1
otherwise (s = 0 is a selling day, the deck's convention).  The "unit-median
VRP" rule sizes by the magnitude,

    q = sign(s) * min(|s| / median_{u < t} |s_u|, 3),

with an expanding, lagged median and a 63-session warm-up on which the rule is
flat - proposal 19's ``um_leverage``, the construction proposal 02 tabulated.
No unit-median table is persisted on any of these frames, so that rule is
constructed here and is NOT gated against a stored row; sign(s) and always short
are, in every book.

FIVE BOOKS are carried, all on the deck's 866 expiration days:

  1530_deck       the deck's own 15:30 book.  Crossed fill: entry at the touch,
                  cash settlement free, return per CROSSED price
                  (``asl.crossed_premium_return``, section 13 of the deck).
  1530_unhedged   the same 15:30 window in the intraday notebook's frame
                  (``results/atm_straddle_intraday``).  Identical positions;
                  the crossed column divides by the MIDPOINT entry premium
                  instead, which is the intraday tables' convention.
  1530_dh         15:30 entry, held to cash settlement, delta-hedged on the
                  vendor spot every 30 minutes
                  (``results/atm_straddle_intraday_holdclose``).
  1500_unhedged   15:00 entry, ONE-BAR hold: exit thirty minutes later at the
                  next stamp in the same two strikes (the intraday notebook's
                  per-clock rule).
  1500_dh         15:00 entry, held to cash settlement (two bars), delta-hedged
                  every 30 minutes.

The two clocks are compared inside a convention (1500_unhedged against
1530_unhedged, 1500_dh against 1530_dh); 1530_deck is the anchor the paper
quotes.

Nothing new runs before the gate passes.  The gate reproduces, by calling
``notebooks/atm_straddle_lib`` and
``writeup/make_rule_by_strategy_intraday_tex`` exactly as the decks do:

  * the deck's ``rule_by_strategy_sign_s.csv``, ``_always_short.csv``,
    ``_sign_s_flat_on_event_days.csv``, ``rule_table_blk2.csv`` and the mid and
    crossed rows of ``pnl_variants_blk2.csv`` - 8 forecasts x 14 columns each;
  * the intraday per-clock tables ``rule_by_strategy/{1500,1530}`` and the
    delta-hedged ``rule_by_strategy_dh/{1500,1530}``: the sign(s) panel for all
    eight forecasts and the always-short row, every column including
    ``Sharpe_crossed`` and ``n_crossed``;
  * the intraday builder's own 15:30-against-the-deck gate;
  * the 15:30 and 15:00 rows of the ridge in
    ``forecast_qlike_by_clock_blk2.csv``;
  * this script's numpy tape (one (R, cr_long, cr_short) triple per book)
    against the builder's ``daily_series`` bar for bar, and its numpy summary
    statistics against ``asl.rule_row``.

Part A - the identity, and the bias.  Proposal 35's combined next-bar
forecaster (C2 log-fit weights, imported by path) is f_comb = w rv_hat +
(1 - w) slice with w in (0, 1], so f_comb - slice = w (rv_hat - slice) and
sign(s) cannot move: the part asserts that day by day at both clocks.  A SCALE
change can move it, so three named rungs are carried in every book:

  (a) the ridge as the deck uses it - the stored y-space forecast mapped to a
      variance by the deck's second-order Mincer-Zarnowitz map
      (``asl.second_order_raw``, method "mean": a flat 250-SESSION window,
      weighted least squares of the unwinsorized sqrt(RV/B) on the stored yhat
      with weights 1 / max(yhat, q10)^2, and RV_hat = (m^2 + sigma^2_hat) B);
  (b) proposal 35's C2 combination at that clock;
  (c) (b) with a CAUSAL recalibration on top: multiplied by the expanding,
      one-session-lagged mean of realized / forecast with a 63-session minimum,
      proposal 34's V5 calibration.

For each of the three: QLIKE, the MZ intercept and slope of realized on
forecast, the mean ratio realized / forecast (both the mean of the ratio and
the ratio of the means), the sign agreement with (a) day by day, the sign(s)
trade's Sharpe mid and crossed, the paired daily difference against (a) with
its HAC t and a circular-block bootstrap interval, and the unit-median VRP row.

Part B - the accuracy ladder.  For each book and each of four forecasts a
family of tunable accuracy for the traded bar,

    f_lambda = exp((1 - lambda) ln rv_hat + lambda ln rv_realized),

lambda = 1 the oracle and lambda = 0 the model (both taken exactly, not through
the exp/log round trip; the round-trip error is printed), and a degraded family

    f_-mu = exp(ln rv_hat + mu e),   e ~ N(0, sigma_e^2) i.i.d.,

with sigma_e the model's own log residual standard deviation, over 200 seeded
draws held as one (200, n_days) matrix (mean and standard deviation across
draws reported).  Every rung carries QLIKE, the sign hit rate of (f - slice)
against (realized - slice), the MZ intercept and slope, the mean ratio, and the
trade.  The half-way readings place (a), (b) and (c) on that curve by QLIKE.

Part C - per year, and the crossed cliff.

Speed and determinism.  The per-clock tapes and the eight forecast panels are
built ONCE; every rung is a matrix operation over days; the independent
(book x forecast) cells run in a ``ProcessPoolExecutor``; and the 2000-draw
circular-block bootstrap is evaluated from a multiplicity-count matrix cached
per SAMPLE LENGTH and seeded on (BOOT_SEED, n), so the result does not depend
on the number of workers and is byte-identical across runs.  Forming each
draw's mean and variance from the counts is algebraically the deck's
``_boot_sharpe`` (it agrees with the gather form to 1.3e-15) and lets one
matrix product score all 200 noise draws at once.

Run:  python writeup/intraday_proposals/51_accuracy_ladder_1530.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

DECK = ROOT / "results" / "atm_straddle_0dte_1530"
INTRADAY = ROOT / "results" / "atm_straddle_intraday"
HOLDCLOSE = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = DECK / "proposals" / "51"

#: The gate's tolerance on every persisted number.
GATE_TOL = 1e-9
#: The intraday builder's own tolerance against the deck (float32 quote noise
#: in the two trade caches); its gate() uses 1e-6 and so does this script when
#: it compares ACROSS the two frames rather than within one.
CROSS_FRAME_TOL = 1e-6
#: sqrt(252), the library's per-trade-day annualization.
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
#: The deck's summary columns, in the deck's order.
COLS: tuple[str, ...] = (
    "n",
    "mean",
    "std",
    "min",
    "25%",
    "50%",
    "75%",
    "max",
    "skew",
    "ex_kurt",
    "t_mean",
    "Sharpe_ann",
    "n_buy",
    "pct_buy",
)

#: The circular moving-block bootstrap, the deck's settings.
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0

#: The unit-median VRP rule (proposal 19's ``um_leverage``).
UM_CAP = 3.0
UM_MIN_HIST = 63
#: Proposal 34's V5 calibration warm-up (the repo's 63-session standard).
CALIB_WARMUP = 63

#: The noise family: draws, and the base seed of the worker-independent RNG.
N_NOISE = 200
NOISE_SEED = 51

#: The forecasts the ladder is built on; the ridge is the headline.
LADDER_TAGS: tuple[str, ...] = ("blk2", "a0", "xgb", "lgbm")
HEADLINE_TAG = "blk2"
#: The ladder's rungs.  {0, 0.1, ..., 1.0} is the declared grid; 0.25 and 0.75
#: are added because Part C reads lambda = 0.25.
LAMBDAS: tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
)
DECLARED_LAMBDAS: tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(11))
MUS: tuple[float, ...] = (0.25, 0.5, 1.0)
#: Part C's own grid: where does the crossed trade first clear zero.
FINE_LAMBDAS: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(21))
YEARS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)
PART_C_LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

#: (book key, clock, delta-hedged, source frame, persisted reference directory)
CONTEXTS: tuple[tuple[str, str, bool, str, Path | None], ...] = (
    ("1530_deck", "15:30", False, "deck", None),
    (
        "1530_unhedged",
        "15:30",
        False,
        "intraday",
        INTRADAY / "rule_by_strategy" / "1530",
    ),
    ("1530_dh", "15:30", True, "dh", HOLDCLOSE / "rule_by_strategy_dh" / "1530"),
    (
        "1500_unhedged",
        "15:00",
        False,
        "intraday",
        INTRADAY / "rule_by_strategy" / "1500",
    ),
    ("1500_dh", "15:00", True, "dh", HOLDCLOSE / "rule_by_strategy_dh" / "1500"),
)
CONTEXT_KEYS: tuple[str, ...] = tuple(c[0] for c in CONTEXTS)
#: The three named rungs of Part A, in order.
NAMED: tuple[str, ...] = (
    "(a) ridge, deck MZ map",
    "(b) p35 C2 combination",
    "(c) (b) + causal recalibration",
)

_COUNTS_CACHE: dict[int, np.ndarray] = {}


# ------------------------------------------------------------------ helpers --
def boot_counts(n: int) -> np.ndarray:
    """(B, n) multiplicity counts of the deck's circular moving-block bootstrap.

    The block index is drawn by ``asl.circular_block_bootstrap_idx`` from
    ``default_rng([BOOT_SEED, n])`` - one index per SAMPLE LENGTH, so every
    rung scored on the same number of days shares it (the deck's "one set of
    resampled day indices, reused by every portfolio") and nothing depends on
    which worker draws it.  The counts carry the same information as the index:
    a draw's mean is (1/n) sum_i c_i x_i and its ddof=1 variance is
    (n/(n-1))((1/n) sum_i c_i x_i^2 - mean^2).
    """
    hit = _COUNTS_CACHE.get(n)
    if hit is None:
        idx = asl.circular_block_bootstrap_idx(
            np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
        )
        flat = (idx + n * np.arange(BOOT_B)[:, None]).ravel()
        hit = np.bincount(flat, minlength=BOOT_B * n).reshape(BOOT_B, n).astype(float)
        _COUNTS_CACHE[n] = hit
    return hit


def as_rows(x: np.ndarray) -> np.ndarray:
    """A (d, n) view of a (n,) or (d, n) array."""
    a = np.asarray(x, float)
    return a[None, :] if a.ndim == 1 else a


def sharpe_rows(X: np.ndarray) -> np.ndarray:
    """mean / std(ddof=1) * sqrt(252) along the day axis."""
    A = as_rows(X)
    n = A.shape[1]
    mu = A.mean(axis=1)
    sd = A.std(axis=1, ddof=1) if n >= 2 else np.full(A.shape[0], np.nan)
    return np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0) * ANN, np.nan)


def t_rows(X: np.ndarray) -> np.ndarray:
    """The deck's plain t of the mean: sqrt(n) * mean / std(ddof=1)."""
    A = as_rows(X)
    n = A.shape[1]
    sd = A.std(axis=1, ddof=1)
    return np.where(
        sd > 0, A.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(n), np.nan
    )


def maxdd_rows(X: np.ndarray) -> np.ndarray:
    """Worst peak-to-trough of the cumulative SUM path, peak seeded at 0."""
    A = as_rows(X)
    path = np.cumsum(A, axis=1)
    peak = np.maximum(np.maximum.accumulate(path, axis=1), 0.0)
    return (path - peak).min(axis=1)


def qlike_rows(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Patton's QLIKE, y / f - log(y / f) - 1, averaged over the day axis."""
    A = as_rows(F)
    r = y[None, :] / A
    return (r - np.log(r) - 1.0).mean(axis=1)


def ols_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(intercept, slope) of the univariate OLS of y on each row of X."""
    A = as_rows(X)
    n = A.shape[1]
    xm = A.mean(axis=1)
    ym = float(y.mean())
    cov = (A * y[None, :]).sum(axis=1) / n - xm * ym
    var = (A * A).sum(axis=1) / n - xm * xm
    slope = np.where(var > 0, cov / np.where(var > 0, var, 1.0), np.nan)
    return ym - slope * xm, slope


def um_sizes_rows(S: np.ndarray) -> np.ndarray:
    """The unit-median VRP position for each row of the signal matrix S.

    Proposal 19's ``um_leverage``: sign(s) x clip(|s| / expanding lagged
    median, cap), the median over STRICTLY prior days with a UM_MIN_HIST
    minimum, so the rule is flat (q = 0) on the warm-up; s = 0 is a selling
    day, the deck's convention.
    """
    A = as_rows(S)
    absa = np.abs(A).T
    med = (
        pd.DataFrame(absa)
        .expanding(min_periods=UM_MIN_HIST)
        .median()
        .shift(1)
        .to_numpy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ell = absa / med
    ok = np.isfinite(absa) & np.isfinite(ell)
    ell = np.where(ok, np.minimum(ell, UM_CAP), 0.0)
    return (np.where(A.T > 0.0, 1.0, -1.0) * ell).T


def um_defined(s: np.ndarray) -> np.ndarray:
    """True where the unit-median scale exists (the rule is out of warm-up)."""
    a = np.abs(np.asarray(s, float))
    med = pd.Series(a).expanding(min_periods=UM_MIN_HIST).median().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.isfinite(a) & np.isfinite(a / med)


def boot_sharpe_rows(X: np.ndarray) -> np.ndarray:
    """(B, d) annualized Sharpe of each row of X on each bootstrap resample."""
    A = as_rows(X)
    n = A.shape[1]
    C = boot_counts(n)
    m1 = C @ A.T / n
    m2 = C @ (A * A).T / n
    var = (m2 - m1 * m1) * (n / (n - 1.0))
    return np.where(var > 0, m1 / np.sqrt(np.where(var > 0, var, 1.0)) * ANN, np.nan)


def pct_ci(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def interval_reading(lo: float, hi: float) -> str:
    """The deck's reading of a bootstrap interval (section 10)."""
    width = hi - lo
    edge = min(abs(lo), abs(hi)) < 0.05 * width
    if lo > 0 or hi < 0:
        return "knife-edge, excludes zero" if edge else "excludes zero"
    return "knife-edge, includes zero" if edge else "includes zero"


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """a minus b on the same days: mean difference, plain and HAC t, dSharpe."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = int(len(d))
    sd = float(d.std(ddof=1))
    t_hac, lag = asl.newey_west_t(d)
    boot = boot_sharpe_rows(a)[:, 0] - boot_sharpe_rows(b)[:, 0]
    lo, hi = pct_ci(boot)
    hat = float(sharpe_rows(a)[0] - sharpe_rows(b)[0])
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "t_plain": float(d.mean() / sd * np.sqrt(n)) if sd > 0 else float("nan"),
        "t_hac": float(t_hac),
        "hac_lag": int(lag),
        "dSharpe": hat,
        "pct_lo": lo,
        "pct_hi": hi,
        "basic_lo": 2.0 * hat - hi,
        "basic_hi": 2.0 * hat - lo,
        "interval": interval_reading(lo, hi),
    }


def show(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(df.to_string())


def write(df: pd.DataFrame, name: str, title: str, index: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=index)
    print(f"\nwrote {path}  ({len(df)} rows) - {title}")


def load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------- rungs and cells --
def lambda_forecast(value: float, rv_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """exp((1 - lambda) ln rv_hat + lambda ln y), endpoints taken exactly."""
    if value == 0.0:
        return rv_hat.copy()[None, :]
    if value == 1.0:
        return y.copy()[None, :]
    return np.exp((1.0 - value) * np.log(rv_hat) + value * np.log(y))[None, :]


def noise_forecast(
    value: float, rv_hat: np.ndarray, sigma: float, key: tuple[int, ...]
) -> np.ndarray:
    """(N_NOISE, n) degraded forecasts exp(ln rv_hat + mu e), e ~ N(0, sigma^2).

    The RNG is seeded on (NOISE_SEED, book, forecast, rung) alone, so a draw is
    the same whichever worker computes it and however many workers there are.
    """
    rng = np.random.default_rng([NOISE_SEED, *key])
    z = rng.standard_normal((N_NOISE, len(rv_hat)))
    return np.exp(np.log(rv_hat)[None, :] + value * sigma * z)


def trade_rows(F: np.ndarray, tape: dict[str, Any]) -> dict[str, np.ndarray]:
    """The sign(s) and unit-median positions and their mid/crossed returns."""
    A = as_rows(F)
    s = A - tape["slice"][None, :]
    q = np.where(s > 0.0, 1.0, -1.0)
    q_um = um_sizes_rows(s)
    cl, cs = tape["cr_long"][None, :], tape["cr_short"][None, :]
    return {
        "s": s,
        "q": q,
        "mid": q * tape["R"][None, :],
        "crossed": q * np.where(q > 0.0, cl, cs),
        "um_mid": q_um * tape["R"][None, :],
        "um_crossed": q_um * np.where(q_um > 0.0, cl, cs),
    }


def cell_stats(F: np.ndarray, tape: dict[str, Any]) -> dict[str, np.ndarray]:
    """Every per-draw number a rung reports (arrays of length d)."""
    A = as_rows(F)
    y, sl = tape["rv_raw"], tape["slice"]
    tr = trade_rows(A, tape)
    a_lvl, b_lvl = ols_rows(A, y)
    a_log, b_log = ols_rows(np.log(A), np.log(y))
    out: dict[str, np.ndarray] = {
        "QLIKE": qlike_rows(y, A),
        "hit_sign": ((A > sl[None, :]) == (y > sl)[None, :]).mean(axis=1),
        "MZ_intercept": a_lvl,
        "MZ_slope": b_lvl,
        "MZ_intercept_log": a_log,
        "MZ_slope_log": b_log,
        "mean_ratio": (y[None, :] / A).mean(axis=1),
        "ratio_of_means": float(y.mean()) / A.mean(axis=1),
        "pct_buy": 100.0 * (tr["q"] > 0.0).mean(axis=1),
    }
    for fill in ("mid", "crossed"):
        x = tr[fill]
        out[f"mean_{fill}"] = x.mean(axis=1)
        out[f"Sharpe_{fill}"] = sharpe_rows(x)
        out[f"t_{fill}"] = t_rows(x)
        out[f"hit_{fill}"] = (x > 0.0).mean(axis=1)
        out[f"MaxDD_{fill}"] = maxdd_rows(x)
        u = tr[f"um_{fill}"]
        out[f"um_mean_{fill}"] = u.mean(axis=1)
        out[f"um_Sharpe_{fill}"] = sharpe_rows(u)
    return out


def summarize(stats: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean and across-draw standard deviation of every per-draw statistic."""
    rec: dict[str, float] = {}
    for k, v in stats.items():
        a = np.asarray(v, float)
        rec[k] = float(a.mean())
        rec[f"{k}_sd_draws"] = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return rec


def cell_task(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One (book, forecast) cell: the whole ladder plus Part C's fine grid."""
    tape = spec["tape"]
    y = tape["rv_raw"]
    rv_hat = tape["rv_hat"]
    sigma = float(np.std(np.log(y) - np.log(rv_hat), ddof=1))
    base = trade_rows(lambda_forecast(0.0, rv_hat, y), tape)
    base_boot = {f: boot_sharpe_rows(base[f])[:, 0] for f in ("mid", "crossed")}
    recs: list[dict[str, Any]] = []
    families: list[tuple[str, float, np.ndarray]] = [
        ("lambda", float(v), lambda_forecast(float(v), rv_hat, y)) for v in LAMBDAS
    ]
    families += [
        (
            "noise",
            float(v),
            noise_forecast(
                float(v), rv_hat, sigma, (spec["book_id"], spec["tag_id"], i + 1)
            ),
        )
        for i, v in enumerate(MUS)
    ]
    for family, value, F in families:
        rec: dict[str, Any] = {
            "context": spec["context"],
            "clock": spec["clock"],
            "forecast": spec["tag"],
            "family": family,
            "rung": value,
            "n_days": int(len(y)),
            "n_draws": int(as_rows(F).shape[0]),
            "sigma_log_resid": sigma,
        }
        rec.update(summarize(cell_stats(F, tape)))
        tr = trade_rows(F, tape)
        for fill in ("mid", "crossed"):
            bs = boot_sharpe_rows(tr[fill]).mean(axis=1)
            lo, hi = pct_ci(bs - base_boot[fill])
            rec[f"dSharpe_{fill}"] = rec[f"Sharpe_{fill}"] - float(
                sharpe_rows(base[fill])[0]
            )
            rec[f"dSharpe_{fill}_lo"] = lo
            rec[f"dSharpe_{fill}_hi"] = hi
            rec[f"dSharpe_{fill}_reading"] = interval_reading(lo, hi)
            slo, shi = pct_ci(bs)
            rec[f"Sharpe_{fill}_lo"] = slo
            rec[f"Sharpe_{fill}_hi"] = shi
            rec[f"Sharpe_{fill}_reading"] = interval_reading(slo, shi)
        rec["declared_grid"] = bool(
            family == "lambda" and any(abs(value - g) < 1e-12 for g in DECLARED_LAMBDAS)
        )
        recs.append(rec)
    return recs


def cliff_task(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Part C: the fine lambda grid, per year and with the crossed interval."""
    tape = spec["tape"]
    y, rv_hat, year = tape["rv_raw"], tape["rv_hat"], tape["year"]
    recs: list[dict[str, Any]] = []
    for value in FINE_LAMBDAS:
        F = lambda_forecast(float(value), rv_hat, y)
        tr = trade_rows(F, tape)
        cr, mid = tr["crossed"][0], tr["mid"][0]
        lo, hi = pct_ci(boot_sharpe_rows(cr)[:, 0])
        rec: dict[str, Any] = {
            "context": spec["context"],
            "clock": spec["clock"],
            "forecast": spec["tag"],
            "lambda": float(value),
            "QLIKE": float(qlike_rows(y, F)[0]),
            "hit_sign": float(((F[0] > tape["slice"]) == (y > tape["slice"])).mean()),
            "Sharpe_mid": float(sharpe_rows(mid)[0]),
            "Sharpe_crossed": float(sharpe_rows(cr)[0]),
            "lo": lo,
            "hi": hi,
            "reading": interval_reading(lo, hi),
            "t_crossed": float(t_rows(cr)[0]),
            "t_hac_crossed": float(asl.newey_west_t(cr)[0]),
            "in_part_c_grid": bool(any(abs(value - g) < 1e-12 for g in PART_C_LAMBDAS)),
        }
        for yr in YEARS:
            m = year == yr
            rec[f"crossed_{yr}"] = float(sharpe_rows(cr[m])[0])
            rec[f"mid_{yr}"] = float(sharpe_rows(mid[m])[0])
            rec[f"n_{yr}"] = int(m.sum())
        recs.append(rec)
    return recs


# ------------------------------------------------------------------- frames --
def build_intraday_frames() -> dict[str, Any]:
    """The two intraday trade frames, the eight panels, and the builder module.

    ``writeup/make_rule_by_strategy_intraday_tex`` is imported by its own module
    name so that the panel loader it hands to a ProcessPoolExecutor stays
    picklable.  The delta-hedged frame needs the same module with
    ``--dh-holdclose`` in argv (the flag is read at import), so a second copy is
    executed with that argv and its panel loader is replaced by the cached
    panels: the eight Mincer-Zarnowitz panels are built ONCE.
    """
    t0 = time.time()
    mk: Any = importlib.import_module("make_rule_by_strategy_intraday_tex")
    panels = mk.load_panels_parallel()
    mk.load_panels_parallel = lambda: panels
    work_un, clocks, deck = mk.build_work()
    argv = list(sys.argv)
    try:
        sys.argv = [*argv, "--dh-holdclose"]
        mk_dh = load_module(
            ROOT / "writeup" / "make_rule_by_strategy_intraday_tex.py", "p51_mk_dh"
        )
    finally:
        sys.argv = argv
    assert mk_dh.DH_HOLDCLOSE, "the delta-hedged copy did not see --dh-holdclose"
    mk_dh.load_panels_parallel = lambda: panels
    work_dh, clocks_dh, _ = mk_dh.build_work()
    work_dh = mk_dh.attach_long_dh(work_dh)
    assert clocks_dh == clocks, (clocks_dh, clocks)
    # the realized variance of the bar each forecast is issued for: the panel
    # row stamped t + 30 (bar-end labels), the same join the builder uses
    pan = panels[HEADLINE_TAG][["t", "rv_raw"]].copy()
    pan["t"] = pd.to_datetime(pan["t"], utc=True) - pd.Timedelta(minutes=30)
    pan = pan.rename(columns={"rv_raw": "rv_next"})
    for w in (work_un, work_dh):
        w["rv_next"] = w[["t"]].merge(pan, on="t", how="left")["rv_next"].to_numpy()
        assert bool(np.isfinite(w["rv_next"]).all()), (
            "a scored bar has no realized variance"
        )
    print(f"intraday frames built in {time.time() - t0:.1f} s")
    return {
        "mk": mk,
        "mk_dh": mk_dh,
        "panels": panels,
        "work": {"intraday": work_un, "dh": work_dh},
        "module": {"intraday": mk, "dh": mk_dh},
        "clocks": clocks,
        "deck": deck,
    }


def intraday_tape(fr: dict[str, Any], source: str, clock: str) -> dict[str, Any]:
    """One book's numpy tape: the day axis at one clock in one trade frame."""
    work = fr["work"][source]
    m = (work["hhmm"].to_numpy() == clock).astype(bool)
    w = work[m]
    entry = w["entry"].to_numpy(float)
    ask_e = (w["ask_c"] + w["ask_p"]).to_numpy(float)
    bid_e = (w["bid_c"] + w["bid_p"]).to_numpy(float)
    ask_x = (w["ask_c_nxt"] + w["ask_p_nxt"]).to_numpy(float)
    bid_x = (w["bid_c_nxt"] + w["bid_p_nxt"]).to_numpy(float)
    is_last = w["is_last"].to_numpy(dtype=bool)
    exit_ = w["exit"].to_numpy(float)
    hedge = (
        w["hedge_long"].to_numpy(float)
        if "hedge_long" in w.columns
        else np.zeros(len(w))
    )
    # crossed_points, restricted to one clock: nothing is held across the
    # boundary (the neighbouring stamps carry q = 0), so the entry is the touch
    # and the exit is the next stamp's touch unless the bar cash-settles.
    exit_long = np.where(is_last, exit_, bid_x)
    exit_short = np.where(is_last, exit_, ask_x)
    cr_long = np.where(ask_e > 0.0, (exit_long - ask_e + hedge) / entry, np.nan)
    cr_short = np.where(bid_e > 0.0, (exit_short - bid_e + hedge) / entry, np.nan)
    dates = pd.DatetimeIndex(pd.to_datetime(w["date"]))
    tape = {
        "dates": dates,
        "year": dates.year.to_numpy(),
        "R": w["R"].to_numpy(float),
        "cr_long": cr_long,
        "cr_short": cr_short,
        "rv_raw": w["rv_next"].to_numpy(float),
        "slice": w["slice"].to_numpy(float),
        "entry": entry,
        "mask": m,
        "source": source,
        "rv_hat_all": {t: w["rv_hat_" + t].to_numpy(float) for t in asl.MODEL_ORDER},
    }
    return tape


def deck_tape(fr: dict[str, Any]) -> dict[str, Any]:
    """The deck's own 15:30 book, from its persisted day books."""
    books = {
        tag: pd.read_parquet(DECK / f"daily_{tag}.parquet") for tag in asl.MODEL_ORDER
    }
    common = None
    for tag in asl.MODEL_ORDER:
        idx = books[tag].index
        common = idx if common is None else common.intersection(idx)
    assert common is not None
    common = pd.DatetimeIndex(common).sort_values()
    px = books["blk2"].loc[common]
    ask = (px["ask_c"].astype(float) + px["ask_p"].astype(float)).to_numpy(float)
    bid = (px["bid_c"].astype(float) + px["bid_p"].astype(float)).to_numpy(float)
    exit_ = px["exit"].to_numpy(float)
    intr = intraday_tape(fr, "intraday", "15:30")
    assert bool(pd.DatetimeIndex(intr["dates"]).equals(common)), (
        "the deck and intraday day axes differ"
    )
    return {
        "books": books,
        "dates": common,
        "year": common.year.to_numpy(),
        "R": px["R"].to_numpy(float),
        "cr_long": np.where(
            ask > 0.0, exit_ / np.where(ask > 0.0, ask, 1.0) - 1.0, np.nan
        ),
        "cr_short": np.where(
            bid > 0.0, exit_ / np.where(bid > 0.0, bid, 1.0) - 1.0, np.nan
        ),
        "rv_raw": intr["rv_raw"],
        "slice": px["iv_var"].to_numpy(float),
        "entry": px["entry"].to_numpy(float),
        "bid": bid,
        "ask": ask,
        "exit": exit_,
        "source": "deck",
        "rv_hat_all": {
            tag: books[tag].loc[common, "rv_hat"].to_numpy(float)
            for tag in asl.MODEL_ORDER
        },
    }


# -------------------------------------------------------------------- gates --
def gate_deck_tables(tape: dict[str, Any]) -> None:
    """Reproduce every persisted deck rule row by calling the library."""
    books, common = tape["books"], tape["dates"]
    worst = 0.0
    for name, fname in (
        ("sign(s)", "rule_by_strategy_sign_s.csv"),
        (
            "sign(s), flat on event days",
            "rule_by_strategy_sign_s_flat_on_event_days.csv",
        ),
    ):
        ref = pd.read_csv(DECK / fname, index_col=0)
        got = {}
        for tag in asl.MODEL_ORDER:
            px = books[tag]
            sizes = asl.rule_sizes(px, ROOT)
            got[asl.YHAT_LABEL[tag]] = asl.rule_row(
                (sizes[name] * px["R"]).loc[common], sizes[name].loc[common]
            )
        tab = pd.DataFrame(got).T[list(COLS)]
        assert list(tab.index) == list(ref.index), (name, list(ref.index))
        d = float(np.max(np.abs(tab.to_numpy(float) - ref[list(COLS)].to_numpy(float))))
        worst = max(worst, d)
        assert d < GATE_TOL, (fname, d)
        print(
            f"  {fname:<48s} {len(ref)} rows x {len(COLS)} cols, max |diff| {d:.3e}  OK"
        )

    px = books["blk2"]
    sizes = asl.rule_sizes(px, ROOT)
    ref = pd.read_csv(DECK / "rule_by_strategy_always_short.csv", index_col=0)
    row = asl.rule_row(
        (sizes["always short"] * px["R"]).loc[common], sizes["always short"].loc[common]
    )
    d = float(
        np.max(
            np.abs(row[list(COLS)].to_numpy(float) - ref[list(COLS)].to_numpy(float)[0])
        )
    )
    worst = max(worst, d)
    assert d < GATE_TOL, ("always short", d)
    print(
        f"  {'rule_by_strategy_always_short.csv':<48s} 1 row x {len(COLS)} cols, max |diff| {d:.3e}  OK"
    )

    ref = pd.read_csv(DECK / "rule_table_blk2.csv", index_col=0)
    tab = pd.DataFrame(
        {
            n: asl.rule_row((sizes[n] * px["R"]).loc[common], sizes[n].loc[common])
            for n in asl.RULE_ORDER
        }
    ).T[list(COLS)]
    d = float(
        np.max(
            np.abs(
                tab.to_numpy(float)
                - ref.loc[list(asl.RULE_ORDER), list(COLS)].to_numpy(float)
            )
        )
    )
    worst = max(worst, d)
    assert d < GATE_TOL, ("rule_table_blk2", d)
    print(
        f"  {'rule_table_blk2.csv':<48s} {len(ref)} rows x {len(COLS)} cols, max |diff| {d:.3e}  OK"
    )

    # the crossed convention, and the tape that carries it
    ref = pd.read_csv(DECK / "pnl_variants_blk2.csv")
    bid = pd.Series(tape["bid"], index=common)
    ask = pd.Series(tape["ask"], index=common)
    for name in asl.RULE_ORDER:
        q = sizes[name].loc[common]
        signq = np.sign(q.replace(0, -1.0))
        lib = (
            asl.crossed_premium_return(signq, px["exit"].loc[common], bid, ask)
            * q.abs()
        )
        for variant, series in (
            ("mid premium R", q * px["R"].loc[common]),
            ("crossed spread", lib),
        ):
            got = asl.rule_row(series, q)
            rr = ref[(ref["rule"] == name) & (ref["variant"] == variant)].iloc[0]
            d = max(abs(float(got[c]) - float(rr[c])) for c in COLS)
            worst = max(worst, d)
            assert d < GATE_TOL, (name, variant, d)
        qa = q.to_numpy(float)
        mine = qa * np.where(qa > 0.0, tape["cr_long"], tape["cr_short"])
        dh = float(np.max(np.abs(mine - lib.to_numpy(float))))
        assert dh == 0.0, (name, dh)
    print(
        f"  {'pnl_variants_blk2.csv':<48s} {len(asl.RULE_ORDER)} rules x 2 fills x "
        f"{len(COLS)} cols, max |diff| {worst:.3e}  OK"
    )
    print(
        "  the deck tape's crossed return equals asl.crossed_premium_return exactly on all three rules  OK"
    )
    for tag in asl.MODEL_ORDER:
        d = float(
            (asl.rule_sizes(books[tag], ROOT)["sign(s)"] - books[tag]["pos"])
            .abs()
            .max()
        )
        assert d == 0.0, (tag, d)
    print(
        f"  the stored 'pos' column equals asl.rule_sizes['sign(s)'] on all {len(asl.MODEL_ORDER)} forecasts  OK"
    )
    # the numpy statistics against the library's, on the deck's own row
    q = sizes["sign(s)"].loc[common]
    lib_row = asl.rule_row(q * px["R"].loc[common], q)
    mid = (q * px["R"].loc[common]).to_numpy(float)
    for label, mine_v, ref_v in (
        ("mean", float(mid.mean()), float(lib_row["mean"])),
        ("t_mean", float(t_rows(mid)[0]), float(lib_row["t_mean"])),
        ("Sharpe_ann", float(sharpe_rows(mid)[0]), float(lib_row["Sharpe_ann"])),
    ):
        d = abs(mine_v - ref_v)
        assert d < GATE_TOL, (label, d)
        print(
            f"  numpy {label:<12s} {mine_v:.12f} vs library {ref_v:.12f}, |diff| {d:.3e}  OK"
        )


def gate_intraday(fr: dict[str, Any], tapes: dict[str, dict[str, Any]]) -> None:
    """The intraday per-clock tables, the builder's own gate, and the tapes."""
    tables: dict[str, Any] = {}
    for source in ("intraday", "dh"):
        mod = fr["module"][source]
        tabs = mod.build_tables(fr["work"][source], fr["clocks"])
        tables[source] = tabs
        mod.gate(fr["work"][source], fr["deck"], tabs)
    for key, clock, _hedged, source, ref_dir in CONTEXTS:
        if ref_dir is None:
            continue
        mod = fr["module"][source]
        work = fr["work"][source]
        tabs = tables[source][clock.replace(":", "")]
        worst = 0.0
        for stem in ("sign_s", "always_short"):
            ref = pd.read_csv(ref_dir / f"rule_by_strategy_{stem}.csv", index_col=0)
            got = tabs[stem]
            assert list(got.index) == list(ref.index), (key, stem)
            d = float((got[ref.columns] - ref).abs().to_numpy(float).max())
            worst = max(worst, d)
            assert d < GATE_TOL, (key, stem, d)
        print(
            f"  {key:<14s} {ref_dir.relative_to(ROOT).as_posix():<58s} "
            f"sign(s) 8 rows + always short, max |diff| {worst:.3e}  OK"
        )
        # the tape must reproduce the builder's own daily series bar for bar
        tape = tapes[key]
        m = tape["mask"]
        for tag in asl.MODEL_ORDER:
            q_full = mod.positions(work, tag)
            r_ref, _sz, rc_ref = mod.daily_series(work, q_full, m, False)
            q = q_full[m]
            mid = q * tape["R"]
            cr = q * np.where(q > 0.0, tape["cr_long"], tape["cr_short"])
            dm = float(np.max(np.abs(mid - r_ref.to_numpy(float))))
            dc = float(np.nanmax(np.abs(cr - rc_ref.to_numpy(float))))
            assert dm == 0.0 and dc == 0.0, (key, tag, dm, dc)
        print(
            f"  {key:<14s} tape reproduces daily_series on all {len(asl.MODEL_ORDER)} forecasts, mid and crossed, exactly  OK"
        )


def gate_qlike(tapes: dict[str, dict[str, Any]]) -> None:
    """The ridge's own QLIKE at both clocks, against the stored per-clock tables.

    ``rv_raw`` and ``rv_hat`` are panel quantities, so a clock's QLIKE does not
    depend on which trade frame the day axis came from; both stored tables are
    therefore checked against the same numbers.
    """
    sources = {
        "15:00": tapes["1500_unhedged"],
        "15:30": tapes["1530_unhedged"],
    }
    for csv in (
        INTRADAY / "forecast_qlike_by_clock_blk2.csv",
        HOLDCLOSE / "forecast_qlike_by_clock_blk2.csv",
    ):
        ref = pd.read_csv(csv)
        for clock, src in sources.items():
            row = ref[(ref["hhmm"] == clock) & (ref["forecast"] == "ridge")]
            assert len(row) == 1, (csv.name, clock, len(row))
            y, f = src["rv_raw"], src["rv_hat_all"][HEADLINE_TAG]
            q = float(qlike_rows(y, f)[0])
            n = int(len(y))
            corr = float(pd.Series(y).corr(pd.Series(f)))
            ratio = float(y.mean() / f.mean())
            r = row.iloc[0]
            for label, mine, ref_v in (
                ("n", float(n), float(r["n"])),
                ("QLIKE", q, float(r["QLIKE"])),
                ("corr", corr, float(r["corr"])),
                ("mean RV / mean f", ratio, float(r["mean RV / mean f"])),
            ):
                d = abs(mine - ref_v)
                assert d < GATE_TOL, (csv.name, clock, label, d)
            print(
                f"  {csv.parent.name:<34s} {clock} ridge: n {n}, QLIKE {q:.12f}, corr "
                f"{corr:.12f}, mean RV / mean f {ratio:.12f} - all four match  OK"
            )


def gate_ladder_base(tapes: dict[str, dict[str, Any]]) -> None:
    """The ladder's lambda = 0 rung IS the published sign(s) row, in every book."""
    deck_mid = pd.read_csv(DECK / "rule_by_strategy_sign_s.csv", index_col=0)
    pv = pd.read_csv(DECK / "pnl_variants_blk2.csv")
    deck_cr = float(
        pv[(pv["rule"] == "sign(s)") & (pv["variant"] == "crossed spread")].iloc[0][
            "Sharpe_ann"
        ]
    )
    for key, _clock, _h, _s, ref_dir in CONTEXTS:
        tape = tapes[key]
        n_cr = 0
        for tag in asl.MODEL_ORDER:
            f = tape["rv_hat_all"][tag]
            tr = trade_rows(f[None, :], tape)
            sh_mid = float(sharpe_rows(tr["mid"])[0])
            sh_cr = float(sharpe_rows(tr["crossed"])[0])
            if ref_dir is None:
                want_mid = float(deck_mid.loc[asl.YHAT_LABEL[tag], "Sharpe_ann"])
                if tag == HEADLINE_TAG:
                    assert abs(sh_cr - deck_cr) < GATE_TOL, (key, tag, sh_cr, deck_cr)
                    n_cr += 1
            else:
                ref = pd.read_csv(ref_dir / "rule_by_strategy_sign_s.csv", index_col=0)
                want_mid = float(ref.loc[asl.YHAT_LABEL[tag], "Sharpe_ann"])
                want_cr = float(ref.loc[asl.YHAT_LABEL[tag], "Sharpe_crossed"])
                assert abs(sh_cr - want_cr) < GATE_TOL, (
                    key,
                    tag,
                    "crossed",
                    sh_cr,
                    want_cr,
                )
                n_cr += 1
            assert abs(sh_mid - want_mid) < GATE_TOL, (
                key,
                tag,
                "mid",
                sh_mid,
                want_mid,
            )
        print(
            f"  {key:<14s} lambda = 0 reproduces the published sign(s) Sharpe on all "
            f"{len(asl.MODEL_ORDER)} forecasts at the midpoint and on {n_cr} of them "
            f"at the crossed spread  OK"
        )


def run_gates(fr: dict[str, Any], tapes: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("GATE - every persisted table this script stands on, before anything new")
    print("=" * 78)
    gate_deck_tables(tapes["1530_deck"])
    gate_intraday(fr, tapes)
    gate_qlike(tapes)
    gate_ladder_base(tapes)
    for key in CONTEXT_KEYS:
        t = tapes[key]
        bad = int(
            np.sum(~np.isfinite(t["cr_long"])) + np.sum(~np.isfinite(t["cr_short"]))
        )
        print(
            f"  {key:<14s} n = {len(t['R'])} days, {t['dates'].min().date()} -> "
            f"{t['dates'].max().date()}; days a crossed fill cannot price: {bad}"
        )
        assert bad == 0, (key, bad)
    print("\nGATE PASSED")


# --------------------------------------------------- Part A: identity, bias --
def causal_scale(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Proposal 34's V5: the expanding, one-session-lagged mean of y / f."""
    r = pd.Series(y / f)
    return r.expanding(min_periods=CALIB_WARMUP).mean().shift(1).to_numpy()


def part_a(
    tapes: dict[str, dict[str, Any]], w_clock: dict[str, pd.Series]
) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("PART A - the convex blend cannot move sign(s); a scale change can")
    print("=" * 78)
    id_rows: list[dict[str, Any]] = []
    named_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    um_rows: dict[str, pd.Series] = {}
    marks: dict[str, dict[str, float]] = {}
    defined_masks: dict[str, np.ndarray] = {}
    for key, clock, _h, _s, _r in CONTEXTS:
        tape = tapes[key]
        rv_hat = tape["rv_hat_all"][HEADLINE_TAG]
        sl, y = tape["slice"], tape["rv_raw"]
        w = w_clock[clock].reindex(tape["dates"]).to_numpy(float)
        live = np.isfinite(w)
        n_live = int(live.sum())
        f_b = np.where(live, w * rv_hat + (1.0 - w) * sl, np.nan)
        s_r, s_b = rv_hat - sl, f_b - sl
        algebra = float(np.max(np.abs(s_b[live] - w[live] * s_r[live])))
        same = (s_b[live] > 0.0) == (s_r[live] > 0.0)
        assert np.all(w[live] > 0.0) and np.all(w[live] <= 1.0), (
            key,
            "weight left (0, 1]",
        )
        assert bool(same.all()), (key, "the identity failed")
        id_rows.append(
            {
                "context": key,
                "clock": clock,
                "n_days": int(len(y)),
                "n_weighted": n_live,
                "w_min": float(np.nanmin(w)),
                "w_median": float(np.nanmedian(w)),
                "w_mean": float(np.nanmean(w)),
                "w_max": float(np.nanmax(w)),
                "max_abs_identity_gap": algebra,
                "sign_agreements": int(same.sum()),
                "sign_disagreements": int((~same).sum()),
            }
        )
        # (c) the causal recalibration of (b)
        scale = causal_scale(y, np.where(live, f_b, np.nan))
        f_c = f_b * scale
        defined = (
            live
            & np.isfinite(scale)
            & um_defined(s_r)
            & um_defined(s_b)
            & um_defined(f_c - sl)
        )
        n_def = int(defined.sum())
        defined_masks[key] = defined
        sub = {
            k: (
                v[defined]
                if isinstance(v, np.ndarray) and v.shape[:1] == (len(y),)
                else v
            )
            for k, v in tape.items()
            if k not in ("rv_hat_all", "books", "mask")
        }
        sub["dates"] = tape["dates"][defined]
        sub["year"] = sub["dates"].year.to_numpy()
        fs = {
            NAMED[0]: rv_hat[defined],
            NAMED[1]: f_b[defined],
            NAMED[2]: f_c[defined],
        }
        base_q = np.where(fs[NAMED[0]] - sub["slice"] > 0.0, 1.0, -1.0)
        base_tr = trade_rows(fs[NAMED[0]][None, :], sub)
        marks[key] = {}
        for name, f in fs.items():
            st = cell_stats(f[None, :], sub)
            tr = trade_rows(f[None, :], sub)
            q = tr["q"][0]
            rec: dict[str, Any] = {
                "context": key,
                "clock": clock,
                "rung": name,
                "n_days": n_def,
                "n_book_days": int(len(y)),
                "sign_agreement_with_a": float(np.mean(q == base_q)),
                "n_sign_flips_vs_a": int(np.sum(q != base_q)),
            }
            rec.update({k: float(np.asarray(v)[0]) for k, v in st.items()})
            named_rows.append(rec)
            marks[key][name] = float(st["QLIKE"][0])
            for fill in ("mid", "crossed"):
                um_rows[f"{key} | {name} | unit-median VRP {fill}"] = asl.rule_row(
                    pd.Series(tr[f"um_{fill}"][0], index=sub["dates"]),
                    pd.Series(um_sizes_rows(tr["s"])[0], index=sub["dates"]),
                )
            if name == NAMED[0]:
                continue
            for fill in ("mid", "crossed"):
                prec: dict[str, Any] = {
                    "context": key,
                    "clock": clock,
                    "fill": fill,
                    "comparison": f"{name} minus {NAMED[0]}",
                    "rule": "sign(s)",
                }
                prec.update(paired(tr[fill][0], base_tr[fill][0]))
                pair_rows.append(prec)
                qrec: dict[str, Any] = {
                    "context": key,
                    "clock": clock,
                    "fill": fill,
                    "comparison": f"{name} minus {NAMED[0]}",
                    "rule": "unit-median VRP",
                }
                qrec.update(paired(tr[f"um_{fill}"][0], base_tr[f"um_{fill}"][0]))
                pair_rows.append(qrec)
        # the control, on exactly the same days: (c) sells nearly every day, so
        # the row the reader needs next to it is always short
        q_as = -np.ones(n_def)
        ctrl: dict[str, Any] = {
            "context": key,
            "clock": clock,
            "rung": "control: always short",
            "n_days": n_def,
            "n_book_days": int(len(y)),
            "sign_agreement_with_a": float(np.mean(q_as == base_q)),
            "n_sign_flips_vs_a": int(np.sum(q_as != base_q)),
            "pct_buy": 0.0,
        }
        for fill in ("mid", "crossed"):
            x = q_as * (sub["R"] if fill == "mid" else sub["cr_short"])
            ctrl[f"mean_{fill}"] = float(x.mean())
            ctrl[f"Sharpe_{fill}"] = float(sharpe_rows(x)[0])
            ctrl[f"t_{fill}"] = float(t_rows(x)[0])
            ctrl[f"hit_{fill}"] = float((x > 0).mean())
            ctrl[f"MaxDD_{fill}"] = float(maxdd_rows(x)[0])
        named_rows.append(ctrl)
        print(
            f"  {key:<14s} clock {clock}: the C2 weight exists on {n_live} of "
            f"{len(y)} days (median {np.nanmedian(w):.6f}, range "
            f"{np.nanmin(w):.6f}-{np.nanmax(w):.6f}); identity gap {algebra:.3e}; "
            f"sign agreements {int(same.sum())}/{n_live}; the three named rungs are "
            f"scored on the {n_def} days all three and their unit-median scales exist"
        )
    ident = pd.DataFrame(id_rows)
    show(ident.set_index(["context", "clock"]), "the identity, book by book")
    write(
        ident, "a_identity.csv", "sign(s) is invariant to the convex blend", index=False
    )

    named = pd.DataFrame(named_rows)
    cols = [
        "context",
        "clock",
        "rung",
        "n_days",
        "QLIKE",
        "MZ_intercept",
        "MZ_slope",
        "mean_ratio",
        "ratio_of_means",
        "hit_sign",
        "sign_agreement_with_a",
        "n_sign_flips_vs_a",
        "pct_buy",
        "Sharpe_mid",
        "Sharpe_crossed",
        "um_Sharpe_mid",
        "um_Sharpe_crossed",
        "t_mid",
        "MaxDD_mid",
    ]
    show(
        named[cols].set_index(["context", "rung"]).round(6),
        "the three named rungs: accuracy, bias, and the trade",
    )
    write(
        named,
        "a_named_rungs.csv",
        "(a) deck ridge, (b) p35 C2 combination, (c) (b) + causal recalibration",
        index=False,
    )

    pair = pd.DataFrame(pair_rows)
    show(
        pair.set_index(["context", "rule", "fill", "comparison"])[
            [
                "n",
                "mean_diff",
                "t_plain",
                "t_hac",
                "hac_lag",
                "dSharpe",
                "pct_lo",
                "pct_hi",
                "interval",
            ]
        ].round(6),
        "paired daily difference against (a), HAC t and the circular-block bootstrap interval",
    )
    write(
        pair,
        "a_named_paired.csv",
        "the paired tests on (b) and (c) against (a)",
        index=False,
    )

    um = pd.DataFrame(um_rows).T[list(COLS)]
    show(um.round(6), "the unit-median VRP row of each named rung, mid and crossed")
    write(um, "a_named_unit_median.csv", "the magnitude rule under (a), (b) and (c)")
    return {
        "marks": marks,
        "named": named,
        "defined": defined_masks,
        "n_paired_ci": int(len(pair)),
    }


# ---------------------------------------------------- Part B: the ladder --
def part_b(tapes: dict[str, dict[str, Any]], a_out: dict[str, Any]) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("PART B - the accuracy ladder, book by book")
    print("=" * 78)
    specs: list[dict[str, Any]] = []
    for book_id, (key, clock, _h, _s, _r) in enumerate(CONTEXTS):
        tape = tapes[key]
        for tag_id, tag in enumerate(LADDER_TAGS):
            lean = {
                k: tape[k]
                for k in ("rv_raw", "slice", "R", "cr_long", "cr_short", "year")
            }
            lean["rv_hat"] = tape["rv_hat_all"][tag]
            specs.append(
                {
                    "context": key,
                    "clock": clock,
                    "tag": tag,
                    "book_id": book_id,
                    "tag_id": tag_id,
                    "tape": lean,
                }
            )
    print(
        f"  {len(specs)} (book x forecast) cells, each {len(LAMBDAS)} lambda rungs + "
        f"{len(MUS)} noise rungs of {N_NOISE} draws = "
        f"{len(specs) * (len(LAMBDAS) + len(MUS))} ladder cells"
    )
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        recs = [r for chunk in pool.map(cell_task, specs) for r in chunk]
    print(f"  ladder computed in {time.time() - t0:.1f} s")
    lad = pd.DataFrame(recs)
    order = {k: i for i, k in enumerate(CONTEXT_KEYS)}
    torder = {t: i for i, t in enumerate(LADDER_TAGS)}
    lad["_c"] = lad["context"].map(order)
    lad["_t"] = lad["forecast"].map(torder)
    lad = (
        lad.sort_values(["_c", "_t", "family", "rung"])
        .drop(columns=["_c", "_t"])
        .reset_index(drop=True)
    )
    write(lad, "b_ladder.csv", "one row per (book, forecast, rung)", index=False)

    curve = [
        "QLIKE",
        "hit_sign",
        "MZ_intercept",
        "MZ_slope",
        "mean_ratio",
        "pct_buy",
        "mean_mid",
        "t_mid",
        "hit_mid",
        "MaxDD_mid",
        "Sharpe_mid",
        "Sharpe_crossed",
        "um_Sharpe_mid",
        "um_Sharpe_crossed",
        "dSharpe_mid",
        "dSharpe_mid_lo",
        "dSharpe_mid_hi",
        "dSharpe_crossed",
        "dSharpe_crossed_lo",
        "dSharpe_crossed_hi",
    ]
    for key in CONTEXT_KEYS:
        sub = lad[(lad["context"] == key) & (lad["forecast"] == HEADLINE_TAG)]
        show(
            sub.set_index(["family", "rung"])[curve].round(6),
            f"{key}: the ridge's ladder - QLIKE -> hit rate -> Sharpe (mid, crossed), "
            f"with the bootstrap interval on the Sharpe difference against lambda = 0",
        )
    two_clock = lad[
        (lad["family"] == "lambda") & (lad["forecast"] == HEADLINE_TAG)
    ].pivot_table(
        index="rung",
        columns="context",
        values=["QLIKE", "hit_sign", "Sharpe_mid", "Sharpe_crossed"],
    )
    show(
        two_clock.round(6), "the two-clock table: the ridge's ladder in all five books"
    )

    # the always-short control, on the same days, in every book
    as_rows_tab = {}
    for key in CONTEXT_KEYS:
        tape = tapes[key]
        q = -np.ones(len(tape["R"]))
        mid = q * tape["R"]
        cr = q * np.where(q > 0.0, tape["cr_long"], tape["cr_short"])
        as_rows_tab[key] = pd.Series(
            {
                "n": float(len(mid)),
                "mean_mid": float(mid.mean()),
                "Sharpe_mid": float(sharpe_rows(mid)[0]),
                "t_mid": float(t_rows(mid)[0]),
                "hit_mid": float((mid > 0).mean()),
                "MaxDD_mid": float(maxdd_rows(mid)[0]),
                "mean_crossed": float(cr.mean()),
                "Sharpe_crossed": float(sharpe_rows(cr)[0]),
                "MaxDD_crossed": float(maxdd_rows(cr)[0]),
            }
        )
    control = pd.DataFrame(as_rows_tab).T
    show(control.round(6), "the always-short control, on the same days, in every book")
    write(
        control, "b_always_short.csv", "the control every ladder rung is read against"
    )

    # ------------------------------------------------ the half-way readings --
    def read_curve(
        lam: np.ndarray,
        qk: np.ndarray,
        sh_mid: np.ndarray,
        sh_cr: np.ndarray,
        hit: np.ndarray,
        target: float,
        **extra: Any,
    ) -> dict[str, Any]:
        o = np.argsort(qk)
        return {
            **extra,
            "target_QLIKE": target,
            "QLIKE_monotone_in_lambda": bool(np.all(np.diff(qk) < 0)),
            "target_inside_ladder": bool(qk.min() <= target <= qk.max()),
            "lambda_at_mark": float(np.interp(target, qk[o], lam[o])),
            "Sharpe_mid_at_mark": float(np.interp(target, qk[o], sh_mid[o])),
            "Sharpe_crossed_at_mark": float(np.interp(target, qk[o], sh_cr[o])),
            "hit_sign_at_mark": float(np.interp(target, qk[o], hit[o])),
        }

    rows = []
    for key in CONTEXT_KEYS:
        tape = tapes[key]
        q_slice = float(qlike_rows(tape["rv_raw"], tape["slice"][None, :])[0])
        for tag in LADDER_TAGS:
            sub = lad[
                (lad["context"] == key)
                & (lad["forecast"] == tag)
                & (lad["family"] == "lambda")
            ].sort_values("rung")
            rows.append(
                read_curve(
                    sub["rung"].to_numpy(float),
                    sub["QLIKE"].to_numpy(float),
                    sub["Sharpe_mid"].to_numpy(float),
                    sub["Sharpe_crossed"].to_numpy(float),
                    sub["hit_sign"].to_numpy(float),
                    q_slice,
                    context=key,
                    forecast=tag,
                    mark="implied slice's QLIKE",
                    frame=f"all {int(sub['n_days'].iloc[0])} days",
                )
            )
        # the named rungs live on a restricted frame, so they are placed on a
        # ladder measured on those same days - never on the full-sample curve
        mask = a_out["defined"][key]
        rt = {
            k: (
                v[mask]
                if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape
                else v
            )
            for k, v in tapes[key].items()
            if k not in ("rv_hat_all", "books", "mask", "dates")
        }
        rv_hat = tapes[key]["rv_hat_all"][HEADLINE_TAG][mask]
        y = rt["rv_raw"]
        lam = np.array(LAMBDAS, float)
        qk = np.empty(len(lam))
        sh_mid = np.empty(len(lam))
        sh_cr = np.empty(len(lam))
        hit = np.empty(len(lam))
        for i, v in enumerate(lam):
            F = lambda_forecast(float(v), rv_hat, y)
            tr = trade_rows(F, rt)
            qk[i] = float(qlike_rows(y, F)[0])
            sh_mid[i] = float(sharpe_rows(tr["mid"])[0])
            sh_cr[i] = float(sharpe_rows(tr["crossed"])[0])
            hit[i] = float(((F[0] > rt["slice"]) == (y > rt["slice"])).mean())
        nm_tab = a_out["named"]
        for nm in NAMED:
            rec = read_curve(
                lam,
                qk,
                sh_mid,
                sh_cr,
                hit,
                a_out["marks"][key][nm],
                context=key,
                forecast=HEADLINE_TAG,
                mark=f"{nm}'s QLIKE",
                frame=f"the {int(mask.sum())} named-rung days",
            )
            act = nm_tab[(nm_tab["context"] == key) & (nm_tab["rung"] == nm)].iloc[0]
            for fill in ("mid", "crossed"):
                rec[f"Sharpe_{fill}_actual"] = float(act[f"Sharpe_{fill}"])
                rec[f"Sharpe_{fill}_actual_minus_ladder"] = (
                    float(act[f"Sharpe_{fill}"]) - rec[f"Sharpe_{fill}_at_mark"]
                )
            rows.append(rec)
    half = pd.DataFrame(rows)
    show(
        half.set_index(["context", "forecast", "mark", "frame"]).round(6),
        "the half-way readings (linear interpolation along the ladder's QLIKE axis; "
        "the named rungs are read on a ladder rebuilt on their own days)",
    )
    write(
        half,
        "b_halfway.csv",
        "the ladder read at the slice's and the named rungs' QLIKE",
        index=False,
    )
    gap = half[half["Sharpe_mid_actual"].notna()]
    show(
        gap.set_index(["context", "mark"])[
            [
                "lambda_at_mark",
                "Sharpe_mid_at_mark",
                "Sharpe_mid_actual",
                "Sharpe_mid_actual_minus_ladder",
                "Sharpe_crossed_at_mark",
                "Sharpe_crossed_actual",
                "Sharpe_crossed_actual_minus_ladder",
            ]
        ].round(6),
        "what the accuracy gain predicts against what the named rung actually trades",
    )
    print(
        "\nthe implied slice as a forecast is the DEGENERATE rung: f = slice makes "
        "s = 0 on every day, and s = 0 is a selling day, so the rule collapses to the "
        "always-short control printed above. It is not a point on the f_lambda ladder, "
        "and the row above only says which lambda has the slice's QLIKE, not what the "
        "slice itself trades."
    )
    return lad


# -------------------------------------------- Part C: per year, and the cliff --
def part_c(tapes: dict[str, dict[str, Any]]) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("PART C - the ladder per year, and how much accuracy the crossed trade needs")
    print("=" * 78)
    specs = []
    for key, clock, _h, _s, _r in CONTEXTS:
        tape = tapes[key]
        for tag in LADDER_TAGS:
            lean = {
                k: tape[k]
                for k in ("rv_raw", "slice", "R", "cr_long", "cr_short", "year")
            }
            lean["rv_hat"] = tape["rv_hat_all"][tag]
            specs.append({"context": key, "clock": clock, "tag": tag, "tape": lean})
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        cliff = pd.DataFrame(
            [r for chunk in pool.map(cliff_task, specs) for r in chunk]
        )
    print(f"  {len(cliff)} fine-grid cells computed in {time.time() - t0:.1f} s")
    order = {k: i for i, k in enumerate(CONTEXT_KEYS)}
    torder = {t: i for i, t in enumerate(LADDER_TAGS)}
    cliff["_c"] = cliff["context"].map(order)
    cliff["_t"] = cliff["forecast"].map(torder)
    cliff = (
        cliff.sort_values(["_c", "_t", "lambda"])
        .drop(columns=["_c", "_t"])
        .reset_index(drop=True)
    )
    write(
        cliff,
        "c_crossed_cliff.csv",
        "the fine lambda grid, per year and with the crossed interval",
        index=False,
    )

    per_year = cliff[cliff["in_part_c_grid"]].copy()
    ycols = [
        "context",
        "forecast",
        "lambda",
        *[f"crossed_{y}" for y in YEARS],
        "Sharpe_crossed",
        "Sharpe_mid",
    ]
    show(
        per_year[ycols].set_index(["context", "forecast", "lambda"]).round(6),
        f"sign(s) CROSSED Sharpe by year at lambda in {PART_C_LAMBDAS} (last two columns: the whole sample, crossed and mid)",
    )
    write(
        per_year[ycols],
        "c_per_year.csv",
        "the crossed sign(s) Sharpe by year",
        index=False,
    )
    t0tape = tapes[CONTEXT_KEYS[0]]
    print(
        "  days per year: "
        + ", ".join(f"{y}: {int((t0tape['year'] == y).sum())}" for y in YEARS)
    )

    print("\nthe first rung whose crossed 95% bootstrap interval excludes zero:")
    for key in CONTEXT_KEYS:
        for tag in LADDER_TAGS:
            sub = cliff[
                (cliff["context"] == key) & (cliff["forecast"] == tag)
            ].sort_values("lambda")
            clear = sub[sub["lo"] > 0.0]
            if len(clear):
                r = clear.iloc[0]
                print(
                    f"  {key:<14s} {tag:<5s} lambda = {float(r['lambda']):.2f}: QLIKE "
                    f"{float(r['QLIKE']):.6f}, hit rate {float(r['hit_sign']):.6f}, Sharpe "
                    f"crossed {float(r['Sharpe_crossed']):.6f}, interval "
                    f"[{float(r['lo']):.6f}, {float(r['hi']):.6f}], t {float(r['t_crossed']):.6f}, "
                    f"HAC t {float(r['t_hac_crossed']):.6f}"
                )
            else:
                print(
                    f"  {key:<14s} {tag:<5s} no rung on the grid has an interval excluding zero"
                )
    return cliff


def bookkeeping(lad: pd.DataFrame, cliff: pd.DataFrame, a_out: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(
        "MULTIPLICITY - how many intervals this script reports, and the 5% expectation"
    )
    print("=" * 78)
    n_ladder_ci = 2 * len(lad)  # dSharpe mid and crossed per ladder rung
    n_level_ci = 2 * len(lad)  # the Sharpe level interval, mid and crossed
    n_cliff_ci = len(cliff)  # the crossed level interval on the fine grid
    n_paired_ci = int(a_out["n_paired_ci"])  # (b), (c) against (a): 2 fills x 2 rules
    total = n_ladder_ci + n_level_ci + n_cliff_ci + n_paired_ci
    print(
        f"  ladder Sharpe-difference intervals (mid + crossed)          {n_ladder_ci:5d}"
    )
    print(
        f"  ladder Sharpe-level intervals (mid + crossed)               {n_level_ci:5d}"
    )
    print(
        f"  crossed-cliff level intervals (fine lambda grid)            {n_cliff_ci:5d}"
    )
    print(
        f"  paired intervals on the named rungs (2 fills x 2 rules)     {n_paired_ci:5d}"
    )
    print(f"  TOTAL intervals reported                                    {total:5d}")
    print(
        f"  at the nominal 5% level, {0.05 * total:.1f} of them would be expected to "
        f"exclude zero by chance alone if every underlying effect were zero; the "
        f"intervals are NOT corrected for multiplicity and no rung is adopted on one."
    )
    exc = int(
        (lad["dSharpe_mid_lo"] > 0).sum()
        + (lad["dSharpe_mid_hi"] < 0).sum()
        + (lad["dSharpe_crossed_lo"] > 0).sum()
        + (lad["dSharpe_crossed_hi"] < 0).sum()
    )
    print(
        f"  ladder Sharpe-difference intervals that do exclude zero:     {exc:5d} of {n_ladder_ci}"
    )
    cl = int((cliff["lo"] > 0).sum() + (cliff["hi"] < 0).sum())
    print(
        f"  crossed-cliff intervals that do exclude zero:                {cl:5d} of {n_cliff_ci}"
    )


def main() -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"repo: {ROOT}")
    print(f"outputs: {OUT}")
    fr = build_intraday_frames()
    tapes: dict[str, dict[str, Any]] = {}
    for key, clock, _h, source, _r in CONTEXTS:
        tapes[key] = (
            deck_tape(fr) if source == "deck" else intraday_tape(fr, source, clock)
        )
    run_gates(fr, tapes)

    t0 = time.time()
    p35 = load_module(
        ROOT / "writeup" / "intraday_proposals" / "35_combined_forecast_toggle.py",
        "p51_p35",
    )
    work35, clocks35, _prof = p35.forecast_frame()
    _fc, weights, _grid = p35.combination_grids(work35, clocks35)
    w_clock = {c: weights["C2_log_fit_weight"][c] for c in ("15:00", "15:30")}
    print(f"proposal 35's C2 weights read in {time.time() - t0:.1f} s")

    a_out = part_a(tapes, w_clock)
    lad = part_b(tapes, a_out)
    cliff = part_c(tapes)
    bookkeeping(lad, cliff, a_out)
    print(f"\ntotal runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
