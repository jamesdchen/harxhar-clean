"""35 - a combined next-bar forecaster, and a per-bar exposure toggle that pays.

The book of record is proposal 32's: every session sells the nearest-OTM SPX
0DTE straddle at 11:00 ET, delta-hedges the package on the vendor spot every
30 minutes, and buys the straddle back at 15:30 at the QUOTED ask with the
entry taken at the quoted bid (fully crossed).  Returns are one unit a day in
units of the 11:00 midpoint entry premium.

Part A asks a forecasting question and nothing else.  At clock t the deck
carries two forecasts of the NEXT bar's realized variance: the block-diagonal
ridge's ``rv_hat`` and the implied slice ``iv_next30_matched =
IV_hr^2 h_rem w_slice``.  The notebook's own per-clock QLIKE table (section
8c, ``forecast_qlike_by_clock_blk2.csv``) says the ridge wins 8 of the 12
clocks and the slice wins the other four.  A forecaster that knew WHICH clock
it was standing at would beat both.  The proposal builds that forecaster
causally:

    f_comb(t, d) = w(t, d) rv_hat(t, d) + (1 - w(t, d)) slice(t, d)

with the per-clock weight chosen from STRICTLY PRIOR sessions only, on an
expanding window with the repo's 63-session warm-up, lagged one session:

  C1 inverse QLIKE    w = Q_slice / (Q_ridge + Q_slice), the two past
                      per-clock QLIKEs measured on the common past bars;
  C2 log-fit weight   w = the slope b of the per-clock constrained log
                      regression ln y = a + b ln rv_hat + (1 - b) ln slice,
                      bounded to the unit simplex the combination lives on;
  C2L log combination the fitted object itself, exp(a) rv_hat^b slice^(1-b) --
                      a combination in logs, not in levels;
  C3 switch           w = 1 when the ridge's past QLIKE is the lower, else 0.

Part B spends that forecast.  The book keeps its 11:00 strikes; at each stamp
t in 11:30..15:00 it decides the bar [t, t + 30] only.  If the signal calls
the bar hot the seller is FLAT for it -- buy the straddle back at the QUOTED
ask at t, re-sell at the QUOTED bid at t + 30 (and simply stay flat to 15:30
if every remaining bar is hot) -- and otherwise stays short.  The hedge
follows the position: no hedge over a flat bar, re-hedged at the re-sell.
Every crossing is charged at the quoted ask/bid.  Three signal families:

  M   margin, ridge   flat when s_t = rv_hat_t - slice_t > theta slice_t,
                      theta in THETA_GRID (theta = 0 is plain sign(s));
  MC  margin, comb    the same with Part A's best combination in place of
                      rv_hat;
  G   expected gain   flat only when the forecast's edge covers the spread:
                      0.5 Gamma_t S_t^2 (f_t - slice_t) > crossing cost,
                      with Gamma the Black-76 package gamma at the stamp's
                      implied total volatility.  The causal cell prices the
                      crossing at twice the half-spread QUOTED AT t; the cell
                      that uses the realised second half-spread at t + 30 is
                      the brief's own and is NON-CAUSAL, so it is reported as
                      a ceiling and never enters the walk-forward.

The gate is the repo's standard and nothing new is computed until it passes:
proposal 32's published book numbers (n = 865, crossed-quoted Sharpe 2.2525467
whole / 2.5131322 era), the per-bar hedge leg against ``parity.replay_day``'s
own replays to 1e-12, the 36 numbers of the notebook's per-clock QLIKE table
to 1e-9, and the 11:00 column of the all-stamp signal grid against proposal
30's own 11:00 signal, exactly.  Selection is proposal 30's purged, embargoed
walk-forward (63-session minimum, 5-day embargo); significance is a placebo
that shuffles WHICH BARS a day is flat on, preserving that day's flat-bar
count and its flat-run lengths exactly.  A rule is adopted only if its
out-of-sample Sharpe beats the book AND its placebo p-value is below
PLACEBO_ALPHA.

Run:  python writeup/intraday_proposals/35_combined_forecast_toggle.py
"""

from __future__ import annotations

import importlib.util
import sys
from math import pi, sqrt
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
p30 = _load_module(HERE / "30_skip_day_rule.py", "p35_p30")
p32 = _load_module(HERE / "32_loss_anatomy.py", "p35_p32")
p33 = _load_module(HERE / "33_path_stop.py", "p35_p33")
asl = p30.asl

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "35"

ENTRY = p32.ENTRY
CLOSE = p32.CLOSE
SESSION: tuple[str, ...] = p32.SESSION
H_REM: tuple[float, ...] = p32.H_REM
ERA0 = p32.ERA0
ANN = p32.ANN

#: The stamps a bar decision is taken at: the bar [t, t + 30] for each of
#: these t.  11:00 is the entry (the first bar is always short) and 15:30 is
#: the book's own buy-back, so the decisions are SESSION[1:-1].
DECISION_STAMPS: tuple[str, ...] = SESSION[1:-1]
#: The bars a day carries (11:00-11:30 ... 15:00-15:30).
N_BARS = len(SESSION) - 1
#: The number of those bars a rule may turn flat.
N_DEC = len(DECISION_STAMPS)

#: The forecast the combination is built on, and the ones carried as context.
HEADLINE_TAG = "blk2"
CONTEXT_TAGS: tuple[str, ...] = ("a0", "xgb", "lgbm")
#: The margin families' grid: flat when s_t exceeds theta times the slice.
THETA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

#: Proposal 30's constants, reused verbatim so this proposal is scored on the
#: same day sets against the same standard.
WARMUP_SESSIONS = p30.WARMUP_SESSIONS
EMBARGO_DAYS = p30.EMBARGO_DAYS
N_PLACEBO = p30.N_PLACEBO
PLACEBO_SEED = p30.PLACEBO_SEED
PLACEBO_ALPHA = p30.PLACEBO_ALPHA

#: The deck's circular block bootstrap: 2000 draws, 21-day blocks, seed 0.
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
#: The percentile interval every bootstrap verdict reads.
CI_PCT: tuple[float, float] = (2.5, 97.5)
#: Placebo draws are computed in day-chunks of this many bootstrap rows.
PLACEBO_CHUNK = 250

#: Published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = p32.GATE_N_DAYS
GATE_CROSSED_WHOLE = p32.GATE_CROSSED_WHOLE
GATE_CROSSED_ERA = p32.GATE_CROSSED_ERA
GATE_TOL = p32.GATE_TOL
#: The notebook's own per-clock QLIKE table, reproduced cell for cell.
GATE_QLIKE_CSV = HOLD / "forecast_qlike_by_clock_blk2.csv"
GATE_QLIKE_TOL = 1e-9
#: Tape identities (the hedge leg, the book's own return) are float paths.
EXACT_TOL = 1e-12
#: The per-day toggle identity, in units of the entry premium.
SUM_TOL = 1e-9


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
    print(f"wrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


# ------------------------------------------------------- causal estimators --
def _prior_cumsum(v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sum of ``v`` over the STRICTLY PRIOR valid days, one entry per day."""
    filled = np.where(valid, np.nan_to_num(v, nan=0.0), 0.0)
    return np.concatenate(([0.0], np.cumsum(filled)))[:-1]


def _prior_count(valid: np.ndarray) -> np.ndarray:
    """Count of STRICTLY PRIOR valid days, one entry per day."""
    return np.concatenate(([0.0], np.cumsum(valid.astype(float))))[:-1]


def prior_mean(v: np.ndarray, valid: np.ndarray, min_count: int) -> np.ndarray:
    """Mean of ``v`` over STRICTLY PRIOR valid days, NaN until ``min_count``.

    The same shape as every other expanding statistic in this line of
    proposals (``30_skip_day_rule.expanding_lagged_quantile``): an expanding
    mean with a minimum count, lagged one session, so the value used on day i
    is a function of days before i only.
    """
    csum = _prior_cumsum(v, valid)
    cnt = _prior_count(valid)
    return np.where(
        cnt >= float(min_count), csum / np.where(cnt > 0.0, cnt, 1.0), np.nan
    )


def prior_ols(
    x: np.ndarray, z: np.ndarray, valid: np.ndarray, min_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Intercept and slope of z on x, fitted on the STRICTLY PRIOR valid days.

    One pass of cumulative sums, lagged one session, with the same
    ``min_count`` warm-up as every other expanding statistic here.  A window
    whose x carries no dispersion yields NaN rather than a divide.
    """
    cnt = _prior_count(valid)
    sx = _prior_cumsum(x, valid)
    sz = _prior_cumsum(z, valid)
    sxx = _prior_cumsum(x * x, valid)
    sxz = _prior_cumsum(x * z, valid)
    good = cnt >= float(min_count)
    safe = np.where(cnt > 0.0, cnt, 1.0)
    den = sxx - sx * sx / safe
    num = sxz - sx * sz / safe
    slope = np.where(good & (den > 0.0), num / np.where(den > 0.0, den, 1.0), np.nan)
    inter = (sz - np.nan_to_num(slope) * sx) / safe
    return np.where(np.isfinite(slope), inter, np.nan), slope


def qlike_terms(y: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Patton's QLIKE per observation, and the mask of scorable observations.

    ``q = y / f - log(y / f) - 1``, exactly ``_qlike_mean`` of section 8c of
    ``notebooks/_write_0dte_intraday_T_nb.py``, which scores only the cells
    whose realized variance and forecast are both finite and strictly
    positive.
    """
    ok = np.isfinite(y) & np.isfinite(f) & (y > 0.0) & (f > 0.0)
    r = np.where(ok, y / np.where(ok, f, 1.0), np.nan)
    return r - np.log(r) - 1.0, ok


def qlike_mean(y: np.ndarray, f: np.ndarray) -> tuple[float, int]:
    """The notebook's ``_qlike_mean``: the mean of the scorable QLIKE terms."""
    q, ok = qlike_terms(y, f)
    if int(ok.sum()) < 2:
        return float("nan"), 0
    return float(np.mean(q[ok])), int(ok.sum())


def norm_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal density."""
    return np.exp(-0.5 * z * z) / sqrt(2.0 * pi)


def package_gamma(
    total_vol: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """Black-76 package gamma of the call at ``kc`` plus the put at ``kp``.

    ``live.ibkr.pricing.package_delta`` is ``N(d1c) + N(d1p) - 1`` with
    ``d1 = (log(F / K) + s^2 / 2) / s`` and r = 0, so its derivative in the
    forward is ``[phi(d1c) + phi(d1p)] / (F s)``.  Zero wherever the delta
    itself is zero (non-positive or non-finite total volatility), which is
    exactly the branch the engine takes.
    """
    ok = np.isfinite(total_vol) & (total_vol > 0.0) & np.isfinite(f) & (f > 0.0)
    s = np.where(ok, total_vol, 1.0)
    ff = np.where(ok, f, 1.0)
    d1c = (np.log(ff / kc) + 0.5 * s * s) / s
    d1p = (np.log(ff / kp) + 0.5 * s * s) / s
    return np.where(ok, (norm_pdf(d1c) + norm_pdf(d1p)) / (ff * s), 0.0)


# ------------------------------------------------------- Part A: the panel --
def forecast_frame() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """The notebook's own scored bar frame, and its remaining-share profile.

    Rebuilds section 5b/8c of ``notebooks/_write_0dte_intraday_T_nb.py`` from
    the same trade cache and the same forecast panels: the trade bar at clock
    t joins the bar-end-labelled panel row stamped t + 30, which carries the
    forecast issued AT t for the bar [t, t + 30] and that bar's own realized
    variance ``rv_raw``.  Returns the frame, its clock list and the panel's
    realized profile.
    """
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    pkg["date"] = pd.to_datetime(pkg["date"])
    paths = asl.yhat_paths(ROOT)
    tags = (HEADLINE_TAG, *CONTEXT_TAGS)
    panels = {tag: asl.load_yhat_panel(paths[tag]).set_index("t") for tag in tags}

    base = panels[HEADLINE_TAG].reset_index()[["t", "rv_hat", "rv_raw", "in_fit"]]
    base = base.copy()
    base["t"] = pd.to_datetime(base["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(base, on="t", how="left").dropna(subset=["R", "rv_hat"])
    assert bool(work["in_fit"].all()), "a joined trade bar is outside the fit mask"
    for tag in CONTEXT_TAGS:
        other = panels[tag].reset_index()[["t", "rv_hat"]].copy()
        other["t"] = pd.to_datetime(other["t"], utc=True) - pd.Timedelta(minutes=30)
        work = work.merge(
            other.rename(columns={"rv_hat": f"rv_hat_{tag}"}), on="t", how="left"
        )

    clocks = sorted(work["hhmm"].unique())
    prof, w_slice = p30.panel_profile(panels[HEADLINE_TAG], clocks)
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work["w_slice"] = w_slice.stack().reindex(mi).to_numpy()
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    work["h_rem"] = work["hhmm"].map(n_rem).astype(float) * 0.5
    work["iv_var_raw"] = work["iv_hourly"].astype(float) ** 2
    work["slice"] = work["iv_var_raw"] * work["h_rem"] * work["w_slice"]
    naive = prof[clocks].expanding(min_periods=WARMUP_SESSIONS).mean().shift(1)
    work["rv_naive"] = naive.stack().reindex(mi).to_numpy()
    print(
        f"forecast frame: {len(work)} scored bars on "
        f"{int(work['date'].nunique())} sessions, clocks {clocks[0]}..{clocks[-1]} "
        f"({len(clocks)} of them); the remaining-share profile is fitted on "
        f"{int(prof.index.size)} panel sessions"
    )
    return work, clocks, prof


def gate_qlike(work: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's per-clock QLIKE table; nothing new runs first."""
    rows: list[dict[str, Any]] = []
    cols = {
        "ridge": "rv_hat",
        "naive clock mean": "rv_naive",
        "implied slice": "slice",
    }
    for hh, g in work.groupby("hhmm", sort=True):
        y = g["rv_raw"].to_numpy(float)
        for name, col in cols.items():
            f = g[col].to_numpy(float)
            q, n = qlike_mean(y, f)
            ok = np.isfinite(y) & np.isfinite(f) & (y > 0.0) & (f > 0.0)
            corr = float(pd.Series(y[ok]).corr(pd.Series(f[ok])))
            rows.append(
                {
                    "hhmm": str(hh),
                    "forecast": name,
                    "n": n,
                    "QLIKE": q,
                    "corr": corr,
                    "mean RV / mean f": float(y[ok].mean() / f[ok].mean()),
                }
            )
    got = pd.DataFrame(rows)
    ref = pd.read_csv(GATE_QLIKE_CSV)
    merged = ref.merge(got, on=["hhmm", "forecast"], suffixes=("_ref", "_got"))
    assert len(merged) == len(ref) == 36, (len(merged), len(ref))
    print(f"\nGATE - the notebook's per-clock QLIKE table, {GATE_QLIKE_CSV.name}")
    for col in ("n", "QLIKE", "corr", "mean RV / mean f"):
        d = float(
            np.max(
                np.abs(
                    merged[f"{col}_ref"].to_numpy(float)
                    - merged[f"{col}_got"].to_numpy(float)
                )
            )
        )
        assert d < GATE_QLIKE_TOL, (col, d)
        print(f"  {col:<18s} 12 clocks x 3 forecasts, max |difference| {d:.3e}  OK")
    wins = got[got["forecast"] != "naive clock mean"].pivot(
        index="hhmm", columns="forecast", values="QLIKE"
    )
    ridge_wins = [
        c for c in wins.index if wins.loc[c, "ridge"] < wins.loc[c, "implied slice"]
    ]
    slice_wins = [c for c in wins.index if c not in ridge_wins]
    print(
        f"  the ridge is the lower QLIKE at {len(ridge_wins)} of {len(wins)} clocks; "
        f"the implied slice at {', '.join(slice_wins)}"
    )
    print("GATE PASSED")
    return got


# -------------------------------------------------- Part A: the combination --
def combination_grids(
    work: pd.DataFrame, clocks: list[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Every candidate next-bar forecast on the (session, clock) grid.

    The weights are per clock and causal: each is a function of that clock's
    STRICTLY PRIOR sessions only, on an expanding window with a
    ``WARMUP_SESSIONS`` minimum, lagged one session.  Returns the forecast
    grids, the weight grids, and the realized grid.
    """
    dates = pd.DatetimeIndex(sorted(work["date"].unique()))

    def grid(col: str) -> pd.DataFrame:
        return work.pivot_table(
            index="date", columns="hhmm", values=col, aggfunc="first"
        ).reindex(index=dates, columns=clocks)

    y = grid("rv_raw")
    f_r = grid("rv_hat")
    f_s = grid("slice")
    yv, rv, sv = y.to_numpy(float), f_r.to_numpy(float), f_s.to_numpy(float)

    w_inv = np.full(yv.shape, np.nan)
    w_log = np.full(yv.shape, np.nan)
    w_sw = np.full(yv.shape, np.nan)
    f_log = np.full(yv.shape, np.nan)
    b_log = np.full(yv.shape, np.nan)
    for j in range(len(clocks)):
        q_r, ok_r = qlike_terms(yv[:, j], rv[:, j])
        q_s, ok_s = qlike_terms(yv[:, j], sv[:, j])
        both = ok_r & ok_s
        past_r = prior_mean(q_r, both, WARMUP_SESSIONS)
        past_s = prior_mean(q_s, both, WARMUP_SESSIONS)
        tot = past_r + past_s
        w_inv[:, j] = np.where(
            np.isfinite(tot) & (tot > 0.0),
            past_s / np.where(tot > 0.0, tot, 1.0),
            np.nan,
        )
        w_sw[:, j] = np.where(
            np.isfinite(tot), np.where(past_r < past_s, 1.0, 0.0), np.nan
        )
        lx = np.log(np.where(both, rv[:, j], 1.0)) - np.log(
            np.where(both, sv[:, j], 1.0)
        )
        lz = np.log(np.where(both, yv[:, j], 1.0)) - np.log(
            np.where(both, sv[:, j], 1.0)
        )
        a, b = prior_ols(lx, lz, both, WARMUP_SESSIONS)
        b_log[:, j] = b
        w_log[:, j] = np.clip(b, 0.0, 1.0)
        usable = (
            np.isfinite(a) & np.isfinite(b) & np.isfinite(rv[:, j]) & (rv[:, j] > 0.0)
        )
        usable &= np.isfinite(sv[:, j]) & (sv[:, j] > 0.0)
        f_log[:, j] = np.where(
            usable,
            np.exp(
                np.nan_to_num(a)
                + np.nan_to_num(b) * np.log(np.where(usable, rv[:, j], 1.0))
                + (1.0 - np.nan_to_num(b)) * np.log(np.where(usable, sv[:, j], 1.0))
            ),
            np.nan,
        )

    weights = {
        "C1_inverse_qlike": pd.DataFrame(w_inv, index=dates, columns=clocks),
        "C2_log_fit_weight": pd.DataFrame(w_log, index=dates, columns=clocks),
        "C3_switch": pd.DataFrame(w_sw, index=dates, columns=clocks),
    }
    fc: dict[str, pd.DataFrame] = {
        "ridge": f_r,
        "implied slice": f_s,
        "naive clock mean": grid("rv_naive"),
    }
    for tag in CONTEXT_TAGS:
        fc[f"ridge {tag}"] = grid(f"rv_hat_{tag}")
    for name, wdf in weights.items():
        wv = wdf.to_numpy(float)
        fc[name] = pd.DataFrame(wv * rv + (1.0 - wv) * sv, index=dates, columns=clocks)
    fc["C2L_log_combination"] = pd.DataFrame(f_log, index=dates, columns=clocks)
    weights["C2L_log_combination_slope"] = pd.DataFrame(
        b_log, index=dates, columns=clocks
    )
    print(
        f"\ncombination grids: {len(dates)} sessions x {len(clocks)} clocks; the "
        f"weights are expanding, lagged one session, minimum {WARMUP_SESSIONS} "
        f"prior scorable sessions at that clock"
    )
    return fc, weights, y


#: The combinations the verdict ranks (the rest of ``fc`` is context).
COMBINATIONS: tuple[str, ...] = (
    "C1_inverse_qlike",
    "C2_log_fit_weight",
    "C2L_log_combination",
    "C3_switch",
)
#: The two forecasts a combination has to beat.
BASELINES: tuple[str, ...] = ("ridge", "implied slice")


def part_a(
    fc: dict[str, pd.DataFrame], weights: dict[str, pd.DataFrame], y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """QLIKE by clock and pooled, the weights, and the bootstrap verdict."""
    dates = pd.DatetimeIndex(y.index)
    clocks = list(y.columns)
    era = np.asarray(dates >= ERA0, dtype=bool)
    samples: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )
    yv = y.to_numpy(float)

    # the common set: every cell on which every ranked forecast is scorable
    common = np.isfinite(yv) & (yv > 0.0)
    for name in (*BASELINES, *COMBINATIONS):
        v = fc[name].to_numpy(float)
        common &= np.isfinite(v) & (v > 0.0)

    rows: list[dict[str, Any]] = []
    for name, f in fc.items():
        fv = f.to_numpy(float)
        for smp, m in samples:
            for cset, cmask in (("own", None), ("common", common)):
                for j, c in enumerate(clocks):
                    sel = m.copy()
                    yy = np.where(sel, yv[:, j], np.nan)
                    ff = np.where(sel, fv[:, j], np.nan)
                    if cmask is not None:
                        keep = cmask[:, j] & sel
                        yy = np.where(keep, yv[:, j], np.nan)
                        ff = np.where(keep, fv[:, j], np.nan)
                    q, n = qlike_mean(yy, ff)
                    ok = np.isfinite(yy) & np.isfinite(ff) & (yy > 0.0) & (ff > 0.0)
                    rows.append(
                        {
                            "sample": smp,
                            "cell_set": cset,
                            "hhmm": c,
                            "forecast": name,
                            "n": n,
                            "QLIKE": q,
                            "bias_mean_y_over_mean_f": float(
                                yy[ok].mean() / ff[ok].mean()
                            )
                            if int(ok.sum())
                            else float("nan"),
                        }
                    )
    by_clock = pd.DataFrame(rows)

    prows: list[dict[str, Any]] = []
    for name, f in fc.items():
        fv = f.to_numpy(float)
        for smp, m in samples:
            for cset, cmask in (("own", None), ("common", common)):
                keep = np.repeat(m[:, None], len(clocks), axis=1)
                if cmask is not None:
                    keep = keep & cmask
                yy = np.where(keep, yv, np.nan).ravel()
                ff = np.where(keep, fv, np.nan).ravel()
                q, n = qlike_mean(yy, ff)
                ok = np.isfinite(yy) & np.isfinite(ff) & (yy > 0.0) & (ff > 0.0)
                prows.append(
                    {
                        "sample": smp,
                        "cell_set": cset,
                        "forecast": name,
                        "n_bars": n,
                        "QLIKE_pooled": q,
                        "bias_mean_y_over_mean_f": float(yy[ok].mean() / ff[ok].mean())
                        if int(ok.sum())
                        else float("nan"),
                    }
                )
    pooled = pd.DataFrame(prows)

    wrows: list[dict[str, Any]] = []
    for name, wdf in weights.items():
        wv = wdf.to_numpy(float)
        for smp, m in samples:
            for j, c in enumerate(clocks):
                col = wv[m, j]
                fin = col[np.isfinite(col)]
                wrows.append(
                    {
                        "sample": smp,
                        "weight_rule": name,
                        "hhmm": c,
                        "n_days_with_weight": int(fin.size),
                        "median": float(np.median(fin)) if fin.size else float("nan"),
                        "mean": float(fin.mean()) if fin.size else float("nan"),
                        "share_at_1": float((fin >= 1.0).mean())
                        if fin.size
                        else float("nan"),
                        "share_at_0": float((fin <= 0.0).mean())
                        if fin.size
                        else float("nan"),
                    }
                )
    weight_tab = pd.DataFrame(wrows)

    # ---------------------------------------------------- the bootstrap ----
    rank = pooled[(pooled["sample"] == "whole") & (pooled["cell_set"] == "common")]
    rank = rank[rank["forecast"].isin(COMBINATIONS)].sort_values("QLIKE_pooled")
    best = str(rank["forecast"].iloc[0])
    brows: list[dict[str, Any]] = []
    for smp, m in samples:
        keep = np.repeat(m[:, None], len(clocks), axis=1) & common
        n_j = keep.sum(axis=1).astype(float)
        for cand in COMBINATIONS:
            qa = np.where(keep, qlike_terms(yv, fc[cand].to_numpy(float))[0], 0.0)
            for ref in BASELINES:
                qb = np.where(keep, qlike_terms(yv, fc[ref].to_numpy(float))[0], 0.0)
                d_day = (qa - qb).sum(axis=1)
                use = n_j > 0.0
                brows.append(
                    {
                        "sample": smp,
                        "combination": cand,
                        "against": ref,
                        "is_best": cand == best,
                        **boot_ratio_ci(d_day[use], n_j[use]),
                    }
                )
    boot = pd.DataFrame(brows)
    return by_clock, pooled, weight_tab, boot, best


def boot_ratio_ci(num_day: np.ndarray, den_day: np.ndarray) -> dict[str, Any]:
    """Circular block bootstrap of a pooled per-bar mean difference.

    The estimator is the ratio of sums -- the summed QLIKE difference over
    the summed bar count -- so resampling DAYS in blocks keeps a day's bars
    together and the pooled statistic is exactly the one the table reports.
    """
    n = int(num_day.size)
    hat = float(num_day.sum() / den_day.sum())
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng(BOOT_SEED), n, BOOT_BLOCK, BOOT_B
    )
    draws = num_day[idx].sum(axis=1) / den_day[idx].sum(axis=1)
    lo, hi = (float(v) for v in np.percentile(draws, list(CI_PCT)))
    return {
        "n_days": n,
        "n_bars": int(den_day.sum()),
        "delta_QLIKE": hat,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
        "improves": bool(hi < 0.0),
        "B": BOOT_B,
        "block": BOOT_BLOCK,
        "seed": BOOT_SEED,
    }


# ------------------------------------------------------- Part B: the book --
def clock_grid(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """A forecast panel column on the (session, trade clock) grid.

    Proposal 34's transform, itself proposal 30's: the panel is bar-end
    labelled, so the row stamped t is the trade clock t - 30 minutes, and only
    the fit mask's rows are kept.
    """
    pf = panel.reset_index()
    pf = pf[pf["in_fit"].to_numpy(dtype=bool)].copy()
    clk = pd.to_datetime(pf["t"], utc=True).dt.tz_convert(
        "America/New_York"
    ) - pd.Timedelta(minutes=30)
    pf["pdate"] = clk.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clk.dt.strftime("%H:%M")
    return pf.pivot_table(
        index="pdate", columns="phhmm", values=col, aggfunc="mean"
    ).sort_index()


def book_grids() -> dict[str, Any]:
    """The 11:00 book's tape, its quotes, its per-bar hedge and its signals.

    Everything is proposal 32's and proposal 33's, imported and unedited: the
    replayed tape, the chain quotes of the 11:00 strikes at all ten stamps,
    the per-bar attribution whose hedge leg is asserted against the live
    engine's own replays, and the all-stamp window-matched signal whose 11:00
    column is asserted against proposal 30's.
    """
    panel, ent, dates = p32.load_panel()
    assert len(dates) == GATE_N_DAYS, (len(dates), GATE_N_DAYS)
    tape = p32.build_tape(panel, ent, dates)
    bid, mid, ask, dead = p33.quote_grids(ent, dates)
    assert not dead.any(), "a stamp of the 11:00 strikes carries no quote"
    assert np.max(np.abs(mid[:, 0] - tape["entry_mid"])) < EXACT_TOL, "11:00 mid"
    attribution, day = p32.attribute(tape, mid, ask, dates)
    p33.hedge_cumulated(attribution, tape, dates)

    att = attribution.copy()
    att["date"] = pd.to_datetime(att["date"])
    hedge = (
        att.pivot_table(
            index="date", columns="clock", values="hedge_pts", aggfunc="first"
        )
        .reindex(index=dates, columns=list(SESSION[:-1]))
        .to_numpy(float)
    )
    assert np.isfinite(hedge).all(), "a bar is missing its hedge leg"
    d_hedge = float(np.max(np.abs(hedge.sum(axis=1) - tape["hedge_ref"])))
    assert d_hedge < EXACT_TOL, d_hedge
    print(
        f"per-bar hedge grid: {hedge.shape[0]} days x {hedge.shape[1]} bars; its row "
        f"sums match the live engine's own {CLOSE} replay to {d_hedge:.3e}"
    )

    # ---- the signal grids, built exactly as proposal 30 builds the 11:00 one
    paths = asl.yhat_paths(ROOT)
    base = asl.load_yhat_panel(paths[HEADLINE_TAG]).set_index("t")
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    clocks = sorted(pkg["hhmm"].unique())
    _, w_full = p30.panel_profile(base, clocks)
    cols = list(SESSION)
    rv_hat = (
        clock_grid(base, "rv_hat").reindex(index=dates, columns=cols).to_numpy(float)
    )
    w_book = w_full.reindex(index=dates, columns=cols).to_numpy(float)
    pn = panel.copy()
    pn["date"] = pd.to_datetime(pn["date"])
    iv_raw = (
        pn.pivot_table(
            index="date", columns="hhmm", values="iv_hourly", aggfunc="first"
        )
        .reindex(index=dates, columns=cols)
        .to_numpy(float)
    )
    slice_book = iv_raw**2 * np.asarray(H_REM)[None, :] * w_book

    ref = p30.signals_1100(pkg, dates)
    got = rv_hat[:, 0] - slice_book[:, 0]
    want = ref[f"s_{HEADLINE_TAG}"].to_numpy(float)
    assert (np.isfinite(got) == np.isfinite(want)).all(), (
        "the 11:00 signal masks differ"
    )
    ok = np.isfinite(got)
    assert np.array_equal(got[ok], want[ok]), (
        "the 11:00 signal does not reproduce p30's"
    )
    print(
        f"GATE - the all-stamp signal grid reproduces proposal 30's {ENTRY} "
        f"{HEADLINE_TAG} signal on all {int(ok.sum())} days that carry one, "
        f"exactly  OK"
    )

    tot_vol = tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])
    gamma = package_gamma(
        tot_vol, tape["spot"], tape["Kc"][:, None], tape["Kp"][:, None]
    )
    return {
        "dates": dates,
        "tape": tape,
        "day": day,
        "bid": bid,
        "mid": mid,
        "ask": ask,
        "hedge": hedge,
        "rv_hat": rv_hat,
        "slice": slice_book,
        "w": w_book,
        "gamma": gamma,
        "spot": tape["spot"],
        "em": tape["entry_mid"],
        "eb": tape["entry_bid"],
    }


# ------------------------------------------------------- Part B: the maths --
def toggle_terms(mask: np.ndarray, g: dict[str, Any]) -> dict[str, np.ndarray]:
    """What a flat-bar mask does to the book, in index points.

    ``mask[..., i]`` is True when the bar starting at ``DECISION_STAMPS[i]``
    is held FLAT.  Entering a flat run buys the straddle back at the QUOTED
    ASK of that stamp; leaving one re-sells at the QUOTED BID of that stamp; a
    run that reaches 15:30 simply stays flat, so the book's own 15:30 ask is
    never paid.  The hedge is dropped over every flat bar.  Returns the change
    against the book of record and the half-spreads actually crossed.
    """
    ask, bid, mid, hedge = g["ask"], g["bid"], g["mid"], g["hedge"]
    prev = np.concatenate(
        [np.zeros(mask.shape[:-1] + (1,), dtype=bool), mask[..., :-1]], axis=-1
    )
    buy = mask & ~prev
    sell = ~mask & prev
    a_in = ask[:, 1 : N_DEC + 1]
    b_in = bid[:, 1 : N_DEC + 1]
    m_in = mid[:, 1 : N_DEC + 1]
    h_in = hedge[:, 1 : N_DEC + 1]
    last = mask[..., -1]
    delta = (
        -(h_in * mask).sum(axis=-1)
        - (a_in * buy).sum(axis=-1)
        + (b_in * sell).sum(axis=-1)
        + ask[:, -1] * last
    )
    cost = ((a_in - m_in) * buy).sum(axis=-1) + ((m_in - b_in) * sell).sum(axis=-1)
    extra = cost - (ask[:, -1] - mid[:, -1]) * last
    return {
        "delta_pts": delta,
        "cost_pts": cost,
        "extra_cost_pts": extra,
        "n_cross": (buy.sum(axis=-1) + sell.sum(axis=-1)).astype(float),
        "n_runs": buy.sum(axis=-1).astype(float),
        "n_flat": mask.sum(axis=-1).astype(float),
    }


def bar_short_pnl(g: dict[str, Any]) -> np.ndarray:
    """Each bar's P&L to the short, marked at the quoted midpoint, in premia.

    ``-(mid_{j+1} - mid_j) + hedge_j`` over the 11:00 entry premium, on the
    DECISION bars only (the first bar is always short, so it is never given
    up).  This is what a flat bar gives up, and the gate's identity ties it to
    the toggle's own return.
    """
    mid = g["mid"]
    full = (-(mid[:, 1:] - mid[:, :-1]) + g["hedge"]) / g["em"][:, None]
    return full[:, 1:]


def book_return(g: dict[str, Any]) -> np.ndarray:
    """The book of record: short every bar, bought back at the 15:30 ask."""
    return (-(g["ask"][:, -1] - g["eb"]) + g["hedge"].sum(axis=1)) / g["em"]


def rule_return(mask: np.ndarray, g: dict[str, Any]) -> np.ndarray:
    """The day's crossed-quoted return under a flat-bar mask, in premia."""
    book = -(g["ask"][:, -1] - g["eb"]) + g["hedge"].sum(axis=1)
    return (book + toggle_terms(mask, g)["delta_pts"]) / g["em"]


# ------------------------------------------------------- Part B: the rules --
def build_rules(
    g: dict[str, Any], f_comb: np.ndarray, best: str
) -> list[dict[str, Any]]:
    """Every candidate flat-bar mask, in families.

    A stamp with no signal -- the vendor's implied volatility on a leg is a
    censored solver node, so the bar carries no implied slice -- stays SHORT,
    which is the book's own default, and is counted.  Only the decision stamps
    11:30..15:00 are ever turned flat.
    """
    dec = slice(1, N_DEC + 1)
    sl = g["slice"][:, dec]
    rv = g["rv_hat"][:, dec]
    fc = f_comb[:, dec]
    gam = g["gamma"][:, dec]
    spot = g["spot"][:, dec]
    ask, bid, mid = g["ask"], g["bid"], g["mid"]
    half_now = (ask - mid)[:, dec]
    half_next = (mid - bid)[:, 2 : N_DEC + 2]
    live_r = np.isfinite(sl) & (sl > 0.0) & np.isfinite(rv)
    live_c = live_r & np.isfinite(fc)

    rules: list[dict[str, Any]] = [
        {
            "family": "reference",
            "param": float("nan"),
            "name": "never toggle (the book of record: short every bar to 15:30)",
            "flat": np.zeros((sl.shape[0], N_DEC), dtype=bool),
            "causal": True,
            "n_no_signal": 0,
        }
    ]
    for theta in THETA_GRID:
        rules.append(
            {
                "family": "margin_ridge",
                "param": float(theta),
                "name": (
                    f"flat on the bar when rv_hat_t - slice_t > {theta:.2f} slice_t "
                    f"(ridge)"
                ),
                "flat": live_r & ((rv - sl) > theta * sl),
                "causal": True,
                "n_no_signal": int((~live_r).sum()),
            }
        )
    for theta in THETA_GRID:
        rules.append(
            {
                "family": "margin_comb",
                "param": float(theta),
                "name": (
                    f"flat on the bar when f_comb_t - slice_t > {theta:.2f} slice_t "
                    f"({best})"
                ),
                "flat": live_c & ((fc - sl) > theta * sl),
                "causal": True,
                "n_no_signal": int((~live_c).sum()),
            }
        )
    gain_r = 0.5 * gam * spot**2 * (rv - sl)
    gain_c = 0.5 * gam * spot**2 * (fc - sl)
    cost_causal = 2.0 * half_now
    cost_actual = half_now + half_next
    for tag, gain, live in (("ridge", gain_r, live_r), (best, gain_c, live_c)):
        rules.append(
            {
                "family": "expected_gain",
                "param": float("nan"),
                "name": (
                    f"flat when 0.5 Gamma_t S_t^2 ({tag} - slice_t) exceeds twice the "
                    f"half-spread quoted at t (causal)"
                ),
                "flat": live & np.isfinite(gain) & (gain > cost_causal),
                "causal": True,
                "n_no_signal": int((~live).sum()),
            }
        )
        rules.append(
            {
                "family": "expected_gain",
                "param": float("nan"),
                "name": (
                    f"flat when 0.5 Gamma_t S_t^2 ({tag} - slice_t) exceeds the two "
                    f"REALISED half-spreads (ask_t - mid_t) + (mid_t+30 - bid_t+30) "
                    f"(NON-CAUSAL)"
                ),
                "flat": live & np.isfinite(gain) & (gain > cost_actual),
                "causal": False,
                "n_no_signal": int((~live).sum()),
            }
        )
    return rules


def standalone_exit_row(g: dict[str, Any]) -> dict[str, Any]:
    """The deck's own 15:30 rule, carried as a reference on a different axis.

    The standalone (``writeup/make_dh_causal_standalone_tex.py``) buys the
    straddle back at 15:30 when that model's signal is positive there, and
    otherwise cash-settles.  It decides the TERMINAL treatment, not a bar, so
    it is not a toggle and it is reported beside the toggles, never inside
    their walk-forward.  Here both legs are the fully crossed quoted book: the
    exit pays the 15:30 quoted ask, the hold carries to cash settlement.
    """
    s15 = g["rv_hat"][:, -1] - g["slice"][:, -1]
    exit_flag = np.isfinite(s15) & (s15 > 0.0)
    r_exit = book_return(g)
    r_hold = (-(g["tape"]["settle"] - g["eb"]) + g["tape"]["hedge_hold"]) / g["em"]
    return {
        "family": "reference",
        "param": float("nan"),
        "name": (
            "15:30 exit if s>0, else cash settlement (the standalone's rule; it "
            "moves the terminal treatment, not a bar)"
        ),
        "returns": np.where(exit_flag, r_exit, r_hold),
        "flat": np.zeros((len(g["dates"]), N_DEC), dtype=bool),
        "exit_share": float(exit_flag.mean()),
    }


# ------------------------------------------------------- Part B: the stats --
def rule_stats(
    r: np.ndarray,
    r_book: np.ndarray,
    flat: np.ndarray,
    terms: dict[str, np.ndarray],
    bar_short: np.ndarray,
    em: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Calendar-day statistics of the toggled book, and the toggle's own bill.

    A toggle rule never sits a day out -- it only flattens bars inside a day
    that is traded anyway -- so every rule is scored on the same calendar days
    and its mean is directly comparable with the book's.  The two bar columns
    are the counterfactual: what the book earned on the bars the rule went
    flat on, against what the rule earned there (nothing, less the crossing
    bill of those flat runs, spread over their bars).
    """
    n = int(r.size)
    sd = float(r.std(ddof=1)) if n >= 2 else float("nan")
    mean = float(r.mean()) if n else float("nan")
    live = bool(np.isfinite(sd)) and sd > 0.0
    n_flat = float(flat.sum())
    extra_prem = terms["extra_cost_pts"] / em
    cost_prem = terms["cost_pts"] / em
    without = float(bar_short[flat].mean()) if n_flat else float("nan")
    with_toggle = float(-extra_prem.sum() / n_flat) if n_flat else float("nan")
    return {
        "n_days": n,
        "n_days_toggled": int((terms["n_flat"] > 0).sum()),
        "toggles_per_day": float(terms["n_cross"].mean()) if n else float("nan"),
        "flat_runs_per_day": float(terms["n_runs"].mean()) if n else float("nan"),
        "frac_bars_flat": float(n_flat / (n * N_DEC)) if n else float("nan"),
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if live else float("nan"),
        "t": float(np.sqrt(n)) * mean / sd if live else float("nan"),
        "MaxDD": p30.maxdd(r),
        "worst_day": float(r.min()) if n else float("nan"),
        "worst_date": str(pd.Timestamp(dates[int(np.argmin(r))]).date()) if n else "",
        "cross_cost_per_day_prem": float(cost_prem.mean()) if n else float("nan"),
        "extra_cost_per_day_prem": float(extra_prem.mean()) if n else float("nan"),
        "n_flat_bars": int(n_flat),
        "mean_bar_without_toggle": without,
        "mean_bar_with_toggle": with_toggle,
        "mean_bar_saved": with_toggle - without,
        "total_saved_prem": float((r - r_book).sum()),
    }


def plain_stats(
    r: np.ndarray, dates: pd.DatetimeIndex, r_book: np.ndarray
) -> dict[str, Any]:
    """The same row for a reference series that carries no flat bars."""
    n = int(r.size)
    sd = float(r.std(ddof=1)) if n >= 2 else float("nan")
    mean = float(r.mean()) if n else float("nan")
    live = bool(np.isfinite(sd)) and sd > 0.0
    return {
        "n_days": n,
        "n_days_toggled": 0,
        "toggles_per_day": float("nan"),
        "flat_runs_per_day": float("nan"),
        "frac_bars_flat": float("nan"),
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if live else float("nan"),
        "t": float(np.sqrt(n)) * mean / sd if live else float("nan"),
        "MaxDD": p30.maxdd(r),
        "worst_day": float(r.min()) if n else float("nan"),
        "worst_date": str(pd.Timestamp(dates[int(np.argmin(r))]).date()) if n else "",
        "cross_cost_per_day_prem": float("nan"),
        "extra_cost_per_day_prem": float("nan"),
        "n_flat_bars": 0,
        "mean_bar_without_toggle": float("nan"),
        "mean_bar_with_toggle": float("nan"),
        "mean_bar_saved": float("nan"),
        "total_saved_prem": float((r - r_book).sum()),
    }


# ------------------------------------------------ Part B: the walk-forward --
def walk_forward(
    series: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expanding, embargoed rule choice; only the out-of-sample day is scored.

    Proposal 30's walk-forward, unchanged in shape: for day i the training
    window is days [0, i - EMBARGO_DAYS), the candidate with the best
    in-window calendar-day Sharpe is applied to day i, candidate 0 is the book
    and ``np.argmax`` keeps the first maximum so a tie goes to the book, and
    days whose training window is shorter than WARMUP_SESSIONS are not scored.
    ``series`` is candidates x days, ``masks`` candidates x days x bars.
    """
    n = series.shape[1]
    curves = np.vstack([p30.expanding_sharpe(row) for row in series])
    curves = np.where(np.isfinite(curves), curves, -np.inf)
    scored = np.zeros(n, dtype=bool)
    pick = np.full(n, -1, dtype=int)
    out = np.full(n, np.nan)
    sel = np.zeros((n, masks.shape[2]), dtype=bool)
    for i in range(n):
        k = i - EMBARGO_DAYS
        if k < WARMUP_SESSIONS:
            continue
        c = int(np.argmax(curves[:, k]))
        pick[i] = c
        out[i] = series[c, i]
        sel[i] = masks[c, i]
        scored[i] = True
    return scored, out, sel, pick


# ---------------------------------------------------- Part B: the placebo --
def _run_signature(m: np.ndarray) -> tuple[int, ...]:
    """The sorted multiset of flat-run lengths of one day's bar mask."""
    out: list[int] = []
    run = 0
    for v in m:
        if v:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return tuple(sorted(out))


def _mask_table() -> tuple[np.ndarray, dict[tuple[int, ...], np.ndarray]]:
    """Every bar mask a day can carry, grouped by its flat-run signature.

    A day has N_DEC decidable bars, so there are 2 ** N_DEC masks in all; two
    masks are exchangeable under the null exactly when they have the same
    multiset of flat-run lengths (hence the same flat-bar count and the same
    number of crossings).  Enumerating them makes the placebo draw uniform on
    that exchangeability class rather than on an approximation of it.
    """
    bits = np.arange(1 << N_DEC)[:, None] >> np.arange(N_DEC)[None, :]
    all_masks = (bits & 1).astype(bool)
    groups: dict[tuple[int, ...], list[int]] = {}
    for k in range(all_masks.shape[0]):
        groups.setdefault(_run_signature(all_masks[k]), []).append(k)
    return all_masks, {s: np.asarray(v, dtype=int) for s, v in groups.items()}


_ALL_MASKS, _MASK_GROUPS = _mask_table()


def shuffled_bar_masks(
    flat: np.ndarray, n_draw: int, rng: np.random.Generator
) -> np.ndarray:
    """``n_draw`` masks that flatten the same bars per day, on other bars.

    Each day is redrawn uniformly from the masks that share its flat-run
    signature, so every draw flattens exactly as many bars that day, in runs
    of exactly the same lengths, and crosses the spread exactly as often -- it
    only flattens different bars.  A day the rule leaves alone is left alone
    in every draw.
    """
    n_days = flat.shape[0]
    sigs = [_run_signature(flat[i]) for i in range(n_days)]
    by: dict[tuple[int, ...], list[int]] = {}
    for i, s in enumerate(sigs):
        by.setdefault(s, []).append(i)
    out = np.zeros((n_draw, n_days, N_DEC), dtype=bool)
    for s in sorted(by, key=lambda t: (len(t), t)):
        idx = np.asarray(by[s], dtype=int)
        pool = _MASK_GROUPS[s]
        pick = rng.integers(0, pool.size, size=(n_draw, idx.size))
        out[:, idx, :] = _ALL_MASKS[pool[pick]]
    assert (out.sum(axis=2) == flat.sum(axis=1)[None, :]).all(), (
        "a placebo draw changed a day's flat-bar count"
    )
    return out


def subset_grids(g: dict[str, Any], sel: np.ndarray) -> dict[str, Any]:
    """The book's grids restricted to a day set, for the placebo's arithmetic."""
    keys = ("bid", "mid", "ask", "hedge", "em", "eb")
    return {k: g[k][sel] for k in keys}


def placebo_row(
    flat: np.ndarray, g: dict[str, Any], sel: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    """Where the rule's Sharpe falls among bar-shuffled draws of itself."""
    sub = subset_grids(g, sel)
    mask = flat[sel]
    r_rule = rule_return(mask, sub)
    s_rule = p30.sharpe(r_rule)
    if not mask.any() or bool(mask.all()):
        return {
            "n_placebo": 0,
            "sharpe_rule": s_rule,
            "placebo_mean": float("nan"),
            "placebo_sd": float("nan"),
            "placebo_q05": float("nan"),
            "placebo_q50": float("nan"),
            "placebo_q95": float("nan"),
            "pct_rank": float("nan"),
            "p_value": float("nan"),
        }
    draws = shuffled_bar_masks(mask, N_PLACEBO, rng)
    book = -(sub["ask"][:, -1] - sub["eb"]) + sub["hedge"].sum(axis=1)
    sharpes = np.full(N_PLACEBO, np.nan)
    for a in range(0, N_PLACEBO, PLACEBO_CHUNK):
        blk = draws[a : a + PLACEBO_CHUNK]
        rr = (book[None, :] + toggle_terms(blk, sub)["delta_pts"]) / sub["em"][None, :]
        mean = rr.mean(axis=1)
        sd = rr.std(axis=1, ddof=1)
        sharpes[a : a + blk.shape[0]] = np.where(
            sd > 0.0, mean / np.where(sd > 0.0, sd, 1.0) * ANN, np.nan
        )
    good = sharpes[np.isfinite(sharpes)]
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


# --------------------------------------------------------- Part B: the gate --
def gate_book(g: dict[str, Any], rules: list[dict[str, Any]]) -> None:
    """Proposal 32's published book numbers, then the toggle's own identity."""
    p32.gate(g["tape"], g["mid"], g["ask"], g["dates"])
    r_book = book_return(g)
    d_book = float(np.max(np.abs(r_book - g["day"]["r_crossed"].to_numpy(float))))
    assert d_book < EXACT_TOL, d_book
    never = rules[0]
    d_never = float(np.max(np.abs(rule_return(never["flat"], g) - r_book)))
    assert d_never < EXACT_TOL, d_never
    print(
        f"\nGATE - the toggle arithmetic on the book of record\n"
        f"  the never-toggle mask reproduces proposal 32's crossed book to "
        f"{max(d_book, d_never):.3e}  OK"
    )
    bar_short = bar_short_pnl(g)
    worst = 0.0
    for rule in rules:
        flat = rule["flat"]
        terms = toggle_terms(flat, g)
        lhs = (rule_return(flat, g) - r_book).sum()
        rhs = -bar_short[flat].sum() - (terms["extra_cost_pts"] / g["em"]).sum()
        worst = max(worst, float(abs(lhs - rhs)))
    assert worst < SUM_TOL, worst
    print(
        f"  on every one of the {len(rules)} masks, the day-return change equals "
        f"minus the flat bars' midpoint P&L minus the extra half-spreads crossed, "
        f"to {worst:.3e} of the entry premium  OK"
    )
    probe = np.zeros((4, N_DEC), dtype=bool)
    probe[0, 2:4] = True
    probe[1, 0] = True
    probe[1, 5:8] = True
    probe[2, :] = True
    drawn = shuffled_bar_masks(probe, 64, np.random.default_rng(PLACEBO_SEED))
    for i in range(probe.shape[0]):
        want = _run_signature(probe[i])
        assert all(_run_signature(drawn[b, i]) == want for b in range(64)), i
    print(
        f"  placebo draws preserve each day's flat-run signature exactly "
        f"(4 probe days x 64 draws, seed {PLACEBO_SEED})  OK"
    )
    print("GATE PASSED")


#: The toggle families the walk-forward selects inside.
FAMILIES: tuple[str, ...] = ("margin_ridge", "margin_comb", "expected_gain")

RULE_COLS = [
    "family",
    "param",
    "rule",
    "causal",
    "n_days",
    "n_days_toggled",
    "toggles_per_day",
    "flat_runs_per_day",
    "frac_bars_flat",
    "mean",
    "sd",
    "Sharpe_ann",
    "t",
    "MaxDD",
    "worst_day",
    "worst_date",
    "cross_cost_per_day_prem",
    "extra_cost_per_day_prem",
    "n_flat_bars",
    "mean_bar_without_toggle",
    "mean_bar_with_toggle",
    "mean_bar_saved",
    "total_saved_prem",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ================================================== Part A ==============
    work, clocks, _prof = forecast_frame()
    gate_qlike(work)
    fc, weights, y = combination_grids(work, clocks)
    by_clock, pooled, weight_tab, boot, best = part_a(fc, weights, y)

    write(by_clock, "a_qlike_by_clock.csv", "next-bar QLIKE, clock by clock")
    write(pooled, "a_qlike_pooled.csv", "next-bar QLIKE, pooled over clocks")
    write(weight_tab, "a_weights.csv", "the causal combination weights")
    write(boot, "a_bootstrap.csv", "the pooled QLIKE difference, block bootstrapped")

    ranked = (
        *BASELINES,
        *COMBINATIONS,
        "naive clock mean",
        *[f"ridge {t}" for t in CONTEXT_TAGS],
    )
    for smp in ("whole", "daily_era"):
        for cset in ("own", "common"):
            sub = by_clock[
                (by_clock["sample"] == smp)
                & (by_clock["cell_set"] == cset)
                & (by_clock["forecast"].isin(ranked))
            ]
            tab = sub.pivot(index="hhmm", columns="forecast", values="QLIKE")[
                list(ranked)
            ]
            show(
                tab.reset_index(),
                f"a_qlike_by_clock.csv  |  QLIKE by clock  |  sample {smp}  |  "
                f"cells {cset}",
            )
            bias = sub.pivot(
                index="hhmm", columns="forecast", values="bias_mean_y_over_mean_f"
            )[list(ranked)]
            show(
                bias.reset_index(),
                f"a_qlike_by_clock.csv  |  bias mean(y) / mean(f)  |  sample {smp}"
                f"  |  cells {cset}",
            )
    for smp in ("whole", "daily_era"):
        for cset in ("own", "common"):
            sub = pooled[(pooled["sample"] == smp) & (pooled["cell_set"] == cset)]
            show(
                sub.sort_values("QLIKE_pooled"),
                f"a_qlike_pooled.csv  |  sample {smp}  |  cells {cset}",
                ["forecast", "n_bars", "QLIKE_pooled", "bias_mean_y_over_mean_f"],
            )
    for smp in ("whole", "daily_era"):
        show(
            weight_tab[weight_tab["sample"] == smp],
            f"a_weights.csv  |  the causal per-clock weight on the ridge  |  "
            f"sample {smp}",
        )
    show(
        boot,
        f"a_bootstrap.csv  |  pooled QLIKE(combination) - QLIKE(reference), "
        f"{BOOT_B} circular block draws, block {BOOT_BLOCK}, seed {BOOT_SEED}, "
        f"{CI_PCT[0]}-{CI_PCT[1]} percentile interval",
    )
    ver = boot[(boot["combination"] == best) & (boot["sample"] == "whole")]
    improves = bool(ver["improves"].all())
    print(
        f"\nPart A verdict: the best combination on the common cells is {best}; "
        f"against both references its interval "
        f"{'EXCLUDES' if improves else 'DOES NOT EXCLUDE'} zero on the improving "
        f"side, so it "
        f"{'IMPROVES' if improves else 'DOES NOT IMPROVE'} on the stated gate "
        f"(whole sample)."
    )
    for _, row in ver.iterrows():
        print(
            f"  vs {row['against']:<14s} delta QLIKE {row['delta_QLIKE']:+.6f}  "
            f"[{row['ci_lo']:+.6f}, {row['ci_hi']:+.6f}]  "
            f"n_bars {int(row['n_bars'])}"
        )

    # ================================================== Part B ==============
    g = book_grids()
    dates = g["dates"]
    cols = list(SESSION)
    comb = fc[best].reindex(index=dates, columns=cols).to_numpy(float)
    sl_a = fc["implied slice"].reindex(index=dates, columns=cols).to_numpy(float)
    shared = np.isfinite(sl_a) & np.isfinite(g["slice"])
    d_slice = float(np.max(np.abs(sl_a[shared] - g["slice"][shared])))
    assert d_slice < EXACT_TOL, d_slice
    print(
        f"\nPart A's implied slice and the book's agree on all "
        f"{int(shared.sum())} shared stamp-cells to {d_slice:.3e}; the book's "
        f"combination grid carries {int(np.isfinite(comb).sum())} of "
        f"{comb.size} cells (the rest are the expanding weight's warm-up and "
        f"stay SHORT)"
    )

    rules = build_rules(g, comb, best)
    gate_book(g, rules)

    r_book = book_return(g)
    bar_short = bar_short_pnl(g)
    era = np.asarray(dates >= ERA0, dtype=bool)
    samples: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )
    ret = {rule["name"]: rule_return(rule["flat"], g) for rule in rules}
    stand = standalone_exit_row(g)

    n_cells = len([r for r in rules if r["family"] != "reference"])
    n_causal = len([r for r in rules if r["family"] != "reference" and r["causal"]])
    print(
        f"\n{n_cells} toggle cells tried ({n_causal} of them causal, the other "
        f"{n_cells - n_causal} priced on a half-spread not yet quoted at the "
        f"decision), across {len(FAMILIES)} families, each scored on "
        f"{len(samples)} samples; decision stamps "
        f"{DECISION_STAMPS[0]}..{DECISION_STAMPS[-1]} ({N_DEC} bars a day)"
    )
    for rule in rules:
        if rule["family"] == "reference":
            continue
        print(
            f"  {rule['family']:<14s} {rule['name'][:96]:<96s} "
            f"no-signal stamps {rule['n_no_signal']:>5d}"
        )

    # ------------------------------------------------------- b_rules.csv ----
    rows: list[dict[str, Any]] = []
    for smp, m in samples:
        for rule in rules:
            terms = toggle_terms(rule["flat"][m], subset_grids(g, m))
            rows.append(
                {
                    "sample": smp,
                    "family": rule["family"],
                    "param": rule["param"],
                    "rule": rule["name"],
                    "causal": rule["causal"],
                    **rule_stats(
                        ret[rule["name"]][m],
                        r_book[m],
                        rule["flat"][m],
                        terms,
                        bar_short[m],
                        g["em"][m],
                        dates[m],
                    ),
                }
            )
        rows.append(
            {
                "sample": smp,
                "family": stand["family"],
                "param": stand["param"],
                "rule": stand["name"],
                "causal": True,
                **plain_stats(stand["returns"][m], dates[m], r_book[m]),
            }
        )
    rules_df = pd.DataFrame(rows)
    write(rules_df, "b_rules.csv", "every toggle rule on the full sample")
    for smp, _ in samples:
        show(
            rules_df[rules_df["sample"] == smp],
            f"b_rules.csv  |  the crossed-quoted book, every crossing charged  |  "
            f"sample {smp}",
            RULE_COLS,
        )
    print(
        f"\nthe standalone's 15:30 rule exits on "
        f"{stand['exit_share']:.4f} of days; it changes the TERMINAL treatment "
        f"(buy back at 15:30 against cash settlement), not a bar, so it is a "
        f"reference row and never a walk-forward candidate"
    )

    # --------------------------------------------------------- b_oos.csv ----
    base = rules[0]
    orows: list[dict[str, Any]] = []
    selected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fam in (*FAMILIES, "all_families"):
        cands = [base] + [
            c
            for c in rules
            if c["causal"]
            and c["family"] != "reference"
            and (c["family"] == fam or fam == "all_families")
        ]
        series = np.vstack([ret[c["name"]] for c in cands])
        masks = np.stack([c["flat"] for c in cands])
        scored, wf_r, wf_mask, pick = walk_forward(series, masks)
        selected[fam] = (scored, wf_r, wf_mask)
        names = np.array([c["name"] for c in cands])
        for smp, m in samples:
            sel = scored & m
            if not sel.any():
                continue
            st = rule_stats(
                wf_r[sel],
                r_book[sel],
                wf_mask[sel],
                toggle_terms(wf_mask[sel], subset_grids(g, sel)),
                bar_short[sel],
                g["em"][sel],
                dates[sel],
            )
            bs = plain_stats(r_book[sel], dates[sel], r_book[sel])
            counts = pd.Series(names[pick[sel]]).value_counts()
            orows.append(
                {
                    "block": "walk_forward_selected",
                    "sample": smp,
                    "family": fam,
                    "param": float("nan"),
                    "rule": (
                        f"{fam}: cell chosen on an expanding window, "
                        f"{EMBARGO_DAYS}-day embargo"
                    ),
                    "causal": True,
                    **st,
                    "Sharpe_book": bs["Sharpe_ann"],
                    "delta_Sharpe": st["Sharpe_ann"] - bs["Sharpe_ann"],
                    "beats_book": bool(st["Sharpe_ann"] > bs["Sharpe_ann"]),
                    "modal_pick": str(counts.index[0]),
                    "modal_pick_share": float(counts.iloc[0] / counts.sum()),
                    "null_share": float(counts.get(base["name"], 0) / counts.sum()),
                    "n_candidates": len(cands),
                }
            )
    scored_any = selected["all_families"][0]
    for smp, m in samples:
        sel = scored_any & m
        bs = plain_stats(r_book[sel], dates[sel], r_book[sel])
        for rule in rules:
            st = rule_stats(
                ret[rule["name"]][sel],
                r_book[sel],
                rule["flat"][sel],
                toggle_terms(rule["flat"][sel], subset_grids(g, sel)),
                bar_short[sel],
                g["em"][sel],
                dates[sel],
            )
            orows.append(
                {
                    "block": "fixed_rule_oos",
                    "sample": smp,
                    "family": rule["family"],
                    "param": rule["param"],
                    "rule": rule["name"],
                    "causal": rule["causal"],
                    **st,
                    "Sharpe_book": bs["Sharpe_ann"],
                    "delta_Sharpe": st["Sharpe_ann"] - bs["Sharpe_ann"],
                    "beats_book": bool(st["Sharpe_ann"] > bs["Sharpe_ann"]),
                    "modal_pick": rule["name"],
                    "modal_pick_share": 1.0,
                    "null_share": float("nan"),
                    "n_candidates": 1,
                }
            )
    oos_df = pd.DataFrame(orows)
    write(oos_df, "b_oos.csv", "the purged, embargoed out-of-sample table")
    ocols = [
        "block",
        "family",
        "param",
        "rule",
        "causal",
        "n_days",
        "toggles_per_day",
        "frac_bars_flat",
        "mean",
        "sd",
        "Sharpe_ann",
        "Sharpe_book",
        "delta_Sharpe",
        "beats_book",
        "t",
        "MaxDD",
        "worst_day",
        "cross_cost_per_day_prem",
        "mean_bar_without_toggle",
        "mean_bar_with_toggle",
        "mean_bar_saved",
        "modal_pick",
        "modal_pick_share",
        "null_share",
    ]
    print(
        f"\nout-of-sample day set: expanding training window, {EMBARGO_DAYS}-day "
        f"embargo, minimum {WARMUP_SESSIONS} training days, so the first "
        f"{WARMUP_SESSIONS + EMBARGO_DAYS} days are never scored "
        f"({int(scored_any.sum())} of {len(dates)} days scored)"
    )
    for smp, _ in samples:
        show(oos_df[oos_df["sample"] == smp], f"b_oos.csv  |  sample {smp}", ocols)

    # ----------------------------------------------------- b_placebo.csv ----
    rng = np.random.default_rng(PLACEBO_SEED)
    prows: list[dict[str, Any]] = []
    for smp, m in samples:
        for day_set, sel in (("full", m), ("oos", m & scored_any)):
            for rule in rules:
                if rule["family"] == "reference":
                    continue
                prows.append(
                    {
                        "sample": smp,
                        "day_set": day_set,
                        "block": "fixed_rule",
                        "family": rule["family"],
                        "param": rule["param"],
                        "rule": rule["name"],
                        "causal": rule["causal"],
                        **placebo_row(rule["flat"], g, sel, rng),
                    }
                )
        for fam in (*FAMILIES, "all_families"):
            wf_scored, _, wf_mask = selected[fam]
            sel = m & wf_scored
            prows.append(
                {
                    "sample": smp,
                    "day_set": "oos",
                    "block": "walk_forward_selected",
                    "family": fam,
                    "param": float("nan"),
                    "rule": (
                        f"{fam}: cell chosen on an expanding window, "
                        f"{EMBARGO_DAYS}-day embargo"
                    ),
                    "causal": True,
                    **placebo_row(wf_mask, g, sel, rng),
                }
            )
    plac_df = pd.DataFrame(prows)
    write(plac_df, "b_placebo.csv", "the bar-shuffled placebo")
    pcols = [
        "day_set",
        "block",
        "family",
        "param",
        "rule",
        "causal",
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
        f"\nplacebo: {N_PLACEBO} draws per rule, seed {PLACEBO_SEED}; each draw "
        f"redraws every day's flat bars uniformly among the masks with that day's "
        f"flat-run signature, so the flat-bar count, the run lengths and the "
        f"number of crossings are all preserved and only WHICH bars are flat "
        f"changes; p = (1 + #(placebo Sharpe >= rule Sharpe)) / (1 + n_placebo)"
    )
    for smp, _ in samples:
        show(
            plac_df[plac_df["sample"] == smp], f"b_placebo.csv  |  sample {smp}", pcols
        )

    # -------------------------------------------------------- b_gate.csv ----
    key = ["sample", "block_key", "family", "rule"]
    left = oos_df.copy()
    left["block_key"] = left["block"].replace({"fixed_rule_oos": "fixed_rule"})
    right = plac_df.loc[plac_df["day_set"] == "oos"].copy()
    right["block_key"] = right["block"]
    gate_df = left[left["family"] != "reference"].merge(
        right[[*key, "p_value", "sharpe_rule"]],
        on=key,
        how="left",
    )
    assert not gate_df["p_value"].isna().all(), "the gate lost its placebo leg"
    same = gate_df["sharpe_rule"].notna() & gate_df["Sharpe_ann"].notna()
    assert np.allclose(
        gate_df.loc[same, "sharpe_rule"],
        gate_df.loc[same, "Sharpe_ann"],
        rtol=0.0,
        atol=EXACT_TOL,
    ), "the placebo and the out-of-sample table disagree on the rule's own Sharpe"
    gate_df["adopted"] = (
        gate_df["beats_book"] & (gate_df["p_value"] < PLACEBO_ALPHA) & gate_df["causal"]
    )
    write(gate_df, "b_gate.csv", "adoption: out-of-sample beats the book AND p < alpha")
    show(
        gate_df,
        f"b_gate.csv  |  adopted = causal AND out-of-sample beats the book AND "
        f"placebo p < {PLACEBO_ALPHA}",
        [
            "sample",
            "block",
            "family",
            "param",
            "rule",
            "causal",
            "Sharpe_ann",
            "Sharpe_book",
            "beats_book",
            "p_value",
            "adopted",
        ],
    )
    fixed = gate_df[gate_df["block_key"] == "fixed_rule"]
    print(f"\nrules adopted on any sample: {int(gate_df['adopted'].sum())}")
    print(
        f"fixed-rule cells read by the gate: {len(fixed)} "
        f"({int(fixed['causal'].sum())} causal); adopted {int(fixed['adopted'].sum())}"
    )
    for fam in (*FAMILIES, "all_families"):
        sub = gate_df[
            (gate_df["block"] == "walk_forward_selected") & (gate_df["family"] == fam)
        ]
        for _, row in sub.iterrows():
            verdict = (
                "BEATS THE BOOK" if row["beats_book"] else "DOES NOT BEAT THE BOOK"
            )
            print(
                f"  family verdict, {row['sample']:<9s} {fam:<14s} walk-forward "
                f"{row['Sharpe_ann']:7.3f} against the book "
                f"{row['Sharpe_book']:7.3f}  {verdict}  (placebo p "
                f"{row['p_value']:.4f})"
            )
    print(
        f"\nmultiple testing: {n_cells} toggle cells were tried across "
        f"{len(FAMILIES)} families, each scored on {len(samples)} samples, so the "
        f"gate above reads {n_cells * len(samples)} fixed-rule cells and a "
        f"{PLACEBO_ALPHA:.0%} level expects "
        f"{n_cells * len(samples) * PLACEBO_ALPHA:.1f} false adoptions there by "
        f"chance."
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
