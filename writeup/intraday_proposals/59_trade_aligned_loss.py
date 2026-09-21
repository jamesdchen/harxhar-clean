"""59 - a loss function aligned with the 15:30 sign(s) straddle trade.

Why.  Every forecast in the deck was tuned and ranked on QLIKE.  Studies 53-56
moved QLIKE at the 15:30 forecast by 2-17% without moving the trade, and study
56 found that QLIKE and tail capture rank the eight forecasts almost
oppositely.  QLIKE was always a proxy; it stays the paper's headline loss.  This
proposal asks what should sit beside it, and what the trade should be tuned on.

The trade.  At 15:30 the position is q_t = sign(f_t - slice_t), f the forecast
of the 15:30-16:00 bar's realized variance and slice the variance the
nearest-OTM SPX 0DTE straddle implies for that bar; it earns q_t R_t, R the
long straddle's return to cash settlement in units of its midpoint premium.

Two candidate losses, both checked here before they are used.

  ECONOMIC loss   EL_t = |R_t| 1{q_t != sign(R_t)}.  The identity
                  mean(q R) = mean|R| - 2 mean(EL) holds exactly (GATE G4), so
                  minimising EL is maximising the trade's mean P&L.  It scores
                  the DECISION.  It is not a function of realized variance: the
                  payoff depends on the net move of the half hour, not on the
                  sum of squared one-minute returns, so no variance forecast
                  can be "consistent" for it.  Part A measures that gap
                  (how often sign(y - slice) and sign(R) disagree).
  ELEMENTARY      ES_t = |y_t/slice_t - 1| 1{f_t and y_t on opposite sides of
  score at the    slice_t}, y the realized variance.  This is the elementary
  slice           score of the mean functional (Ehm, Gneiting, Jordan and
                  Krueger 2016) at the threshold theta_t = slice_t, in units of
                  the slice.  Every consistent scoring function for the mean is
                  a mixture of elementary scores over the threshold; for QLIKE
                  = y/f - ln(y/f) - 1 (the Bregman function -ln) the mixing
                  density is 1/theta^2, i.e. in slice units
                      QLIKE(f, y) = integral ES_c(f, y) c^-2 dc,
                  ES_c the same score at theta = c x slice (GATE G3 checks this
                  numerically, day by day).  QLIKE spreads its weight over every
                  threshold; the trade only ever uses c = 1.  ES keeps
                  consistency for the mean, which the paper needs, and puts all
                  the weight at the decision threshold.
  A prediction    An unweighted classifier of 1{y > slice} targets
  to check        P(y > slice) > 1/2 (a median-type statement); weighting each
                  day by |y/slice - 1| targets E[y] > slice (the mean).  The
                  repo found earlier that a median-type recalibration is right
                  more often and earns less.  If the reasoning is right, in
                  Part B the unweighted fit has the higher hit rate and the
                  weighted fits the higher P&L.

PART A - evaluation, the 866 deck days, the eight forecasts.  QLIKE, ES at the
slice, EL, the hit rate of sign(f - slice) against sign(y - slice) and against
sign(R), tail capture (long on the 10 / 20 / 50 best long-straddle days), mid
and crossed Sharpe.  The Murphy diagram: mean ES_c for c = 2^(k/8), k = -16..16
(1/4 to 4, c = 1 on the grid), one curve per forecast, CSV and PNG.  Then: which
loss ranks forecasts the way the trade's crossed Sharpe does?  Eight forecasts
cannot settle that, so the set is extended with proposal 51's two ladders on the
ridge - the blend toward the realized value (its 12 rungs with lambda > 0) and
the multiplicative-noise degradation (its 3 rungs, 200 seeded draws each,
statistics averaged over draws).  Spearman correlation of minus each loss with
the crossed Sharpe over (i) the eight, (ii) the 16 ladder points including the
ridge, (iii) the union, with a circular-block bootstrap interval over days
(block 21, B = 2000, seed 0).  The economic loss is an affine function of the
mid P&L, so its correlation with the Sharpe is high by construction; it is
reported as the ceiling, not as a finding.

PART B - training on the aligned loss, causal throughout.  A weighted logistic
regression, unpenalised, fitted on strictly prior deck days (expanding, minimum
252 days, REFIT EVERY SESSION: a fit is a few Newton steps on at most 865 rows,
so there is no cadence to choose), long iff the fitted probability exceeds 1/2.
Feature sets, fixed here: F1 ln(f_ridge/slice); F2 ln(f_k/slice) for the ridge,
the baseline, XGBoost and LightGBM; F3 = F2 plus the month-end and third-Friday
dummies (studies 54 and 58).  Targets and weights: (u) 1{y > slice}, unweighted;
(e) 1{y > slice}, weight |y/slice - 1|; (p) 1{R > 0}, weight |R|; (s) plain
sign(s) on the same days.  A day whose weight is exactly zero carries no
information under that weighting and drops out of that fit by rule.  A dummy
whose flagged training days all fall in one class has no finite coefficient
(quasi-separation): it is left out of that day's fit by rule, and the count is
reported.  A fit that does not converge falls back to sign(s) for that day and
is counted.  For F1 the fit is a threshold rule f/slice > c*, c* = exp(-a/b);
its path is reported, because "which c should the trade use" is the question a
loss function answers and sign(s) answers it with c = 1 by fiat.
A cell PASSES only if the paired circular-block bootstrap interval of its
crossed Sharpe minus plain sign(s)'s, on the same days, excludes zero on the
positive side.  9 cells; about 0.2 pass by chance.  Power, stated in advance:
614 scored days, and a trade whose P&L sits in about 20 days of 866.  A null
here is a null about this sample, not about the loss.

GATES
  G1  ridge sign(s) on the 866 deck days: 1.338322 mid / 0.869588 crossed.
  G2  ridge QLIKE at the 15:30 forecast 0.1100 (proposals 35 and 53), with y the
      forecast table's 16:00-row realized variance, whose rv_hat must equal the
      deck's.
  G3  QLIKE equals the c^-2 mixture of elementary scores, day by day.
  G4  mean(q R) = mean|R| - 2 mean(EL) for every forecast.

Run:  python writeup/intraday_proposals/59_trade_aligned_loss.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = DECK / "proposals" / "59"
FLAGS = DECK / "proposals" / "56" / "a_calendar_flags.csv"

ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0
CLOSE_MIN = 16 * 60
TOP_K = (10, 20, 50)
MURPHY_C = tuple(2.0 ** (k / 8.0) for k in range(-16, 17))  # 1/4 .. 4, c = 1 on it
MIN_TRAIN = 252
FAMILIES = ("blk2", "a0", "xgb", "lgbm")
CAL_DUMMIES = ("month-end", "third Friday")
NOISE_BOOK_ID = 59  # this study's own seed namespace for proposal 51's noise rungs
NEWTON_MAX_ITER, NEWTON_TOL = 100, 1e-10  # solver settings, not model parameters
G3_GRID = 4001  # quadrature points per day for gate G3 (a numerical setting)

GATE_SHARPE = (1.338322, 0.869588)
GATE_TOL = 1e-6
GATE_QLIKE, GATE_QLIKE_TOL = 0.1100, 5e-5
GATE_IDENTITY_TOL = 1e-12


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ scores ---
def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def elementary(f_over_s: np.ndarray, y_over_s: np.ndarray, c: float) -> np.ndarray:
    """|y/s - c| 1{min(f, y)/s <= c < max(f, y)/s}: the mean's elementary score."""
    lo = np.minimum(f_over_s, y_over_s)
    hi = np.maximum(f_over_s, y_over_s)
    return np.abs(y_over_s - c) * ((lo <= c) & (c < hi))


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(x)
    return a.mean(axis=1) / a.std(axis=1, ddof=1) * ANN


def boot_counts(n: int) -> np.ndarray:
    """(B, n) multiplicities of the deck's circular moving-block bootstrap."""
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
    )
    flat = (idx + n * np.arange(BOOT_B)[:, None]).ravel()
    return np.bincount(flat, minlength=BOOT_B * n).reshape(BOOT_B, n).astype(float)


def resampled_mean(x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(d, B) resampled means of each row of x (d, n)."""
    return np.atleast_2d(x) @ counts.T / counts.shape[1]


def resampled_sharpe(x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n = counts.shape[1]
    a = np.atleast_2d(x)
    m1 = a @ counts.T / n
    m2 = (a * a) @ counts.T / n
    var = (m2 - m1 * m1) * (n / (n - 1.0))
    return m1 / np.sqrt(var) * ANN


# ------------------------------------------------------------------- tape ----
def load_tape() -> dict[str, Any]:
    books = {
        t: pd.read_parquet(DECK / f"daily_{t}.parquet").sort_index()
        for t in asl.MODEL_ORDER
    }
    d = books["blk2"]
    days = pd.DatetimeIndex(pd.to_datetime(d.index))
    for b in books.values():
        assert pd.DatetimeIndex(pd.to_datetime(b.index)).equals(days)
    panel = asl.load_yhat_panel_mz(asl.yhat_paths(ROOT)["blk2"])
    close = panel[panel["mins"] == CLOSE_MIN]
    close = close.set_index(pd.DatetimeIndex(pd.to_datetime(close["date"])))
    close = close.reindex(days)
    y = close["rv_raw"].to_numpy(float)
    f_ridge = d["rv_hat"].to_numpy(float)
    dev = float(np.max(np.abs(close["rv_hat"].to_numpy(float) / f_ridge - 1.0)))
    assert np.isfinite(y).all() and (y > 0).all(), "a deck day has no realized variance"
    ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
    bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
    ex = d["exit"].to_numpy(float)
    tape = {
        "days": days,
        "y": y,
        "slice": d["iv_var"].to_numpy(float),
        "R": d["R"].to_numpy(float),
        "cl": ex / ask - 1.0,
        "cs": ex / bid - 1.0,
        "f": {t: b["rv_hat"].to_numpy(float) for t, b in books.items()},
        "panel_vs_deck_rv_hat": dev,
    }
    return tape


def pnl(q: np.ndarray, tape: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(mid, crossed) daily P&L of positions q (rows = forecasts or draws)."""
    a = np.atleast_2d(q)
    return a * tape["R"], a * np.where(a > 0, tape["cl"], tape["cs"])


def day_scores(f: np.ndarray, tape: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-day score arrays for each row of f (d, n)."""
    a = np.atleast_2d(f)
    y, s, r = tape["y"], tape["slice"], tape["R"]
    q = np.where(a > s, 1.0, -1.0)
    mid, cr = pnl(q, tape)
    wrong_r = q != np.where(r > 0, 1.0, -1.0)
    return {
        "q": q,
        "QLIKE": qlike(y, a),
        "ES": elementary(a / s, (y / s)[None, :], 1.0),
        "EL": np.abs(r) * wrong_r,
        "hit_var": (a > s) == (y > s),
        "hit_pay": ~wrong_r,
        "mid": mid,
        "crossed": cr,
    }


def gates(tape: dict[str, Any]) -> None:
    sc = day_scores(tape["f"]["blk2"], tape)
    got = (float(sharpe_rows(sc["mid"])[0]), float(sharpe_rows(sc["crossed"])[0]))
    assert abs(got[0] - GATE_SHARPE[0]) < GATE_TOL, got
    assert abs(got[1] - GATE_SHARPE[1]) < GATE_TOL, got
    print(
        f"G1  ridge sign(s): {got[0]:.6f} mid / {got[1]:.6f} crossed on {len(tape['R'])} days"
    )
    ql = float(sc["QLIKE"].mean())
    assert abs(ql - GATE_QLIKE) < GATE_QLIKE_TOL, ql
    print(
        f"G2  ridge QLIKE at the 15:30 forecast {ql:.4f}; the forecast table's 16:00-row "
        f"rv_hat vs the deck's: max relative difference {tape['panel_vs_deck_rv_hat']:.1e}"
    )
    # G3: QLIKE = integral of ES_c c^-2 dc.  Between a = min(f, y)/s and b = max the
    # integrand is |y/s - c| / c^2; integrate it on a fine log grid per day.
    fs, ys = tape["f"]["blk2"] / tape["slice"], tape["y"] / tape["slice"]
    lo, hi = np.minimum(fs, ys), np.maximum(fs, ys)
    u = np.linspace(0.0, 1.0, G3_GRID)
    c = lo[:, None] * (hi / lo)[:, None] ** u[None, :]
    integrand = np.abs(ys[:, None] - c) / c**2
    integral = np.trapezoid(integrand, c, axis=1)
    dev = float(np.max(np.abs(integral - sc["QLIKE"][0])))
    assert dev < GATE_QLIKE_TOL, dev
    print(
        f"G3  QLIKE equals the c^-2 mixture of elementary scores, day by day: max abs gap {dev:.1e}"
    )
    for t, f in tape["f"].items():
        s = day_scores(f, tape)
        gap = abs(
            float(s["mid"].mean())
            - (float(np.abs(tape["R"]).mean()) - 2 * float(s["EL"].mean()))
        )
        assert gap < GATE_IDENTITY_TOL, (t, gap)
    print("G4  mean(q R) = mean|R| - 2 mean(EL) for all eight forecasts")


# ----------------------------------------------------------------- part A ----
def tail_capture(q: np.ndarray, r: np.ndarray) -> dict[str, float]:
    order = np.argsort(r)[::-1]
    return {
        f"long_on_top{k}": float(
            (np.atleast_2d(q)[:, order[:k]] > 0).sum(axis=1).mean()
        )
        for k in TOP_K
    }


def part_a(tape: dict[str, Any], p51: ModuleType) -> dict[str, pd.DataFrame]:
    r, y, s = tape["R"], tape["y"], tape["slice"]
    n = len(r)
    counts = boot_counts(n)
    members: list[dict[str, Any]] = []
    for t in asl.MODEL_ORDER:
        members.append(
            {"name": asl.YHAT_LABEL[t], "kind": "forecast", "F": tape["f"][t][None, :]}
        )
    ridge = tape["f"]["blk2"]
    for lam in p51.LAMBDAS:
        if lam == 0.0:
            continue
        members.append(
            {
                "name": f"ridge blended toward realized, lambda {lam:g}",
                "kind": "ladder",
                "F": p51.lambda_forecast(float(lam), ridge, y),
            }
        )
    sigma = float(np.std(np.log(y) - np.log(ridge), ddof=1))
    for i, mu in enumerate(p51.MUS):
        members.append(
            {
                "name": f"ridge with noise, mu {mu:g}",
                "kind": "ladder",
                "F": p51.noise_forecast(
                    float(mu), ridge, sigma, (NOISE_BOOK_ID, 0, i + 1)
                ),
            }
        )

    rows, boot = [], {}
    for m in members:
        sc = day_scores(m["F"], tape)
        rec: dict[str, Any] = {
            "forecast": m["name"],
            "kind": m["kind"],
            "draws": m["F"].shape[0],
            "QLIKE": float(sc["QLIKE"].mean()),
            "ES_at_slice": float(sc["ES"].mean()),
            "economic_loss": float(sc["EL"].mean()),
            "hit_vs_variance": float(sc["hit_var"].mean()),
            "hit_vs_payoff": float(sc["hit_pay"].mean()),
            "buy_share": float((sc["q"] > 0).mean()),
            **tail_capture(sc["q"], r),
            "mean_mid": float(sc["mid"].mean()),
            "Sharpe_mid": float(sharpe_rows(sc["mid"]).mean()),
            "Sharpe_crossed": float(sharpe_rows(sc["crossed"]).mean()),
        }
        rows.append(rec)
        boot[m["name"]] = {
            "QLIKE": resampled_mean(sc["QLIKE"], counts).mean(axis=0),
            "ES_at_slice": resampled_mean(sc["ES"], counts).mean(axis=0),
            "economic_loss": resampled_mean(sc["EL"], counts).mean(axis=0),
            "miss_vs_variance": 1.0
            - resampled_mean(sc["hit_var"].astype(float), counts).mean(axis=0),
            "Sharpe_crossed": resampled_sharpe(sc["crossed"], counts).mean(axis=0),
        }
    table = pd.DataFrame(rows)

    # how far is the variance target from the payoff?
    agree = float(((y > s) == (r > 0)).mean())
    q_var, q_pay = np.where(y > s, 1.0, -1.0), np.where(r > 0, 1.0, -1.0)
    gap = pd.DataFrame(
        [
            {
                "oracle": "sign(realized variance - slice)",
                "agrees_with_sign_R": agree,
                "Sharpe_mid": float(sharpe_rows(pnl(q_var, tape)[0])[0]),
                "Sharpe_crossed": float(sharpe_rows(pnl(q_var, tape)[1])[0]),
                **tail_capture(q_var, r),
            },
            {
                "oracle": "sign(R)",
                "agrees_with_sign_R": 1.0,
                "Sharpe_mid": float(sharpe_rows(pnl(q_pay, tape)[0])[0]),
                "Sharpe_crossed": float(sharpe_rows(pnl(q_pay, tape)[1])[0]),
                **tail_capture(q_pay, r),
            },
        ]
    )

    # Murphy diagram, the eight forecasts
    mur = []
    for t in asl.MODEL_ORDER:
        fs, ys = tape["f"][t] / s, y / s
        for c in MURPHY_C:
            mur.append(
                {
                    "forecast": asl.YHAT_LABEL[t],
                    "c": c,
                    "mean_ES": float(elementary(fs, ys, c).mean()),
                }
            )
    murphy = pd.DataFrame(mur)

    # which loss ranks the way the crossed Sharpe does?
    sets = {
        "(i) the eight forecasts": [
            m["name"] for m in members if m["kind"] == "forecast"
        ],
        "(ii) ridge + its 15 ladder rungs": [asl.YHAT_LABEL["blk2"]]
        + [m["name"] for m in members if m["kind"] == "ladder"],
        "(iii) union": [m["name"] for m in members],
    }
    losses = ("QLIKE", "ES_at_slice", "miss_vs_variance", "economic_loss")
    rk = []
    point = table.set_index("forecast").assign(
        miss_vs_variance=lambda d: 1.0 - d["hit_vs_variance"]
    )
    for sname, names in sets.items():
        sh_pt = point.loc[names, "Sharpe_crossed"].to_numpy()
        sh_b = np.stack([boot[nm]["Sharpe_crossed"] for nm in names], axis=1)  # (B, k)
        r_sh = stats.rankdata(sh_b, axis=1)
        for loss in losses:
            rho = float(
                stats.spearmanr(-point.loc[names, loss].to_numpy(), sh_pt).statistic
            )
            lb = np.stack([boot[nm][loss] for nm in names], axis=1)
            r_l = stats.rankdata(-lb, axis=1)
            a = r_l - r_l.mean(axis=1, keepdims=True)
            b = r_sh - r_sh.mean(axis=1, keepdims=True)
            rb = (a * b).sum(axis=1) / np.sqrt(
                (a * a).sum(axis=1) * (b * b).sum(axis=1)
            )
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            rk.append(
                {
                    "set": sname,
                    "k": len(names),
                    "loss": loss,
                    "spearman_with_crossed_Sharpe": rho,
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )
    return {"table": table, "gap": gap, "murphy": murphy, "rank": pd.DataFrame(rk)}


def plot_murphy(murphy: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for name, g in murphy.groupby("forecast", sort=False):
        ax.plot(g["c"], g["mean_ES"], label=name, linewidth=1.3)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(
        "threshold as a multiple of the implied slice, c  (the trade uses c = 1)"
    )
    ax.set_ylabel("mean elementary score, slice units")
    ax.set_title("Murphy diagram, 15:30 forecast of the last half hour's variance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "a_murphy.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- part B ----
def weighted_logit(
    x: np.ndarray, t: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Unpenalised weighted logistic MLE by Newton's method; (beta, converged)."""
    beta = np.zeros(x.shape[1])
    for _ in range(NEWTON_MAX_ITER):
        eta = x @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = x.T @ (w * (t - p))
        hess = (x * (w * p * (1.0 - p))[:, None]).T @ x
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return beta, False
        beta = beta + step
        if not np.isfinite(beta).all():
            return beta, False
        if float(np.max(np.abs(step))) < NEWTON_TOL:
            return beta, True
    return beta, False


def fit_cell(job: dict[str, Any]) -> dict[str, Any]:
    x_all, n_cont = job["X"], job["n_continuous"]
    target, weight, q_ref = job["target"], job["weight"], job["q_ref"]
    n = len(target)
    q = np.full(n, np.nan)
    c_star = np.full(n, np.nan)
    slope_pos = np.full(n, np.nan)
    dropped_dummy = fallback = 0
    for day in range(MIN_TRAIN, n):
        use = weight[:day] > 0.0
        xs, ts, ws = x_all[:day][use], target[:day][use], weight[:day][use]
        keep = list(range(1 + n_cont))
        for j in range(1 + n_cont, x_all.shape[1]):
            flagged = xs[:, j] > 0
            if flagged.any() and 0.0 < ts[flagged].mean() < 1.0:
                keep.append(j)
            else:
                dropped_dummy += 1
        beta, ok = weighted_logit(xs[:, keep], ts, ws)
        if not ok:
            fallback += 1
            q[day] = q_ref[day]
            continue
        q[day] = 1.0 if float(x_all[day, keep] @ beta) > 0.0 else -1.0
        if n_cont == 1 and len(keep) == 2 and beta[1] != 0.0:
            c_star[day] = float(np.exp(-beta[0] / beta[1]))
            slope_pos[day] = float(beta[1] > 0.0)
    return {
        "key": job["key"],
        "q": q,
        "c_star": c_star,
        "slope_pos": slope_pos,
        "dropped_dummy": dropped_dummy,
        "fallback": fallback,
    }


def part_b(tape: dict[str, Any]) -> dict[str, pd.DataFrame]:
    r, y, s, days = tape["R"], tape["y"], tape["slice"], tape["days"]
    n = len(r)
    flags = pd.read_csv(FLAGS, index_col=0, parse_dates=True).reindex(days)[
        list(CAL_DUMMIES)
    ]
    assert flags.notna().all().all(), "a deck day is missing from study 56's calendar"
    lf = {t: np.log(tape["f"][t] / s) for t in FAMILIES}
    ones = np.ones(n)
    designs = {
        "F1 ridge": (np.column_stack([ones, lf["blk2"]]), 1),
        "F2 four families": (
            np.column_stack([ones, *[lf[t] for t in FAMILIES]]),
            len(FAMILIES),
        ),
        "F3 four families + month-end, third Friday": (
            np.column_stack([ones, *[lf[t] for t in FAMILIES], flags.to_numpy(float)]),
            len(FAMILIES),
        ),
    }
    up_var, up_pay = (y > s).astype(float), (r > 0).astype(float)
    weightings = {
        "(u) unweighted 1{y > slice}": (up_var, np.ones(n)),
        "(e) ES-weighted 1{y > slice}": (up_var, np.abs(y / s - 1.0)),
        "(p) payoff-weighted 1{R > 0}": (up_pay, np.abs(r)),
    }
    q_ref = np.where(tape["f"]["blk2"] > s, 1.0, -1.0)
    jobs = [
        {
            "key": (dn, wn),
            "X": x,
            "n_continuous": nc,
            "target": tg,
            "weight": wt,
            "q_ref": q_ref,
        }
        for dn, (x, nc) in designs.items()
        for wn, (tg, wt) in weightings.items()
    ]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        res = {r_["key"]: r_ for r_ in pool.map(fit_cell, jobs)}

    ev = np.arange(n) >= MIN_TRAIN
    sub = {
        k: (v[ev] if isinstance(v, np.ndarray) else v)
        for k, v in tape.items()
        if k in ("R", "cl", "cs", "y", "slice")
    }
    n_ev = int(ev.sum())
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([BOOT_SEED, n_ev]), n_ev, BOOT_BLOCK, BOOT_B
    )
    yr = days[ev].year
    base_mid, base_cr = (p_[0] for p_ in pnl(q_ref[ev], sub))
    rows, years, cpath = [], [], []

    def describe(
        name_d: str, name_w: str, q: np.ndarray, extra: dict[str, Any]
    ) -> None:
        mid, cr = (p_[0] for p_ in pnl(q, sub))
        d_b = np.array(
            [sharpe_rows(cr[i])[0] - sharpe_rows(base_cr[i])[0] for i in idx]
        )
        lo, hi = np.percentile(d_b, [2.5, 97.5])
        rows.append(
            {
                "features": name_d,
                "weighting": name_w,
                "n_days": n_ev,
                "buy_share": float((q > 0).mean()),
                "hit_vs_variance": float(((q > 0) == (sub["y"] > sub["slice"])).mean()),
                "hit_vs_payoff": float(((q > 0) == (sub["R"] > 0)).mean()),
                **tail_capture(q, sub["R"]),
                "mean_mid": float(mid.mean()),
                "Sharpe_mid": float(sharpe_rows(mid)[0]),
                "Sharpe_crossed": float(sharpe_rows(cr)[0]),
                "d_crossed_vs_sign_s": float(
                    sharpe_rows(cr)[0] - sharpe_rows(base_cr)[0]
                ),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "passes": bool(lo > 0.0),
                **extra,
            }
        )
        for yy in sorted(set(yr)):
            m = yr == yy
            years.append(
                {
                    "features": name_d,
                    "weighting": name_w,
                    "year": int(yy),
                    "n": int(m.sum()),
                    "Sharpe_crossed": float(sharpe_rows(cr[m])[0]),
                }
            )

    describe(
        "-",
        "(s) plain sign(s)",
        q_ref[ev],
        {"dropped_dummy_fits": 0, "fallback_days": 0},
    )
    for (dn, wn), out in res.items():
        describe(
            dn,
            wn,
            out["q"][ev],
            {
                "dropped_dummy_fits": out["dropped_dummy"],
                "fallback_days": out["fallback"],
            },
        )
        if dn.startswith("F1"):
            c = out["c_star"][ev]
            cpath.append(
                {
                    "weighting": wn,
                    "c_star_first": float(c[0]),
                    "c_star_median": float(np.nanmedian(c)),
                    "c_star_last": float(c[-1]),
                    "c_star_min": float(np.nanmin(c)),
                    "c_star_max": float(np.nanmax(c)),
                    "share_slope_positive": float(np.nanmean(out["slope_pos"][ev])),
                }
            )
    return {
        "table": pd.DataFrame(rows),
        "years": pd.DataFrame(years),
        "c_star": pd.DataFrame(cpath),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    tape = load_tape()
    gates(tape)
    p51 = _load(HERE / "51_accuracy_ladder_1530.py", "p51_accuracy_ladder")

    a = part_a(tape, p51)
    a["table"].to_csv(OUT / "a_scores.csv", index=False)
    a["gap"].to_csv(OUT / "a_variance_vs_payoff.csv", index=False)
    a["murphy"].to_csv(OUT / "a_murphy.csv", index=False)
    a["rank"].to_csv(OUT / "a_rank_correlation.csv", index=False)
    plot_murphy(a["murphy"])
    print("\nA1  every forecast and ladder rung under each loss")
    print(a["table"].round(4).to_string(index=False))
    print("\nA2  how far the variance target is from the payoff")
    print(a["gap"].round(3).to_string(index=False))
    at1 = a["murphy"][np.isclose(a["murphy"]["c"], 1.0)].sort_values("mean_ES")
    print("\nA3  Murphy diagram at c = 1 (the trade's threshold), best first")
    print(at1.round(4).to_string(index=False))
    wide = a["murphy"].pivot(index="c", columns="forecast", values="mean_ES")
    best = wide.idxmin(axis=1)
    print(
        "    best forecast by threshold: "
        + "; ".join(f"c={c:.2f}: {best.loc[c]}" for c in wide.index[::4])
    )
    print("\nA4  Spearman of (minus the loss) with the crossed Sharpe")
    print(a["rank"].round(3).to_string(index=False))

    b = part_b(tape)
    b["table"].to_csv(OUT / "b_trained_rules.csv", index=False)
    b["years"].to_csv(OUT / "b_by_year.csv", index=False)
    b["c_star"].to_csv(OUT / "b_threshold_path.csv", index=False)
    print(
        f"\nB1  weighted logistic rules on the {int(b['table']['n_days'].iloc[0])} post-warm-up days"
    )
    print(b["table"].round(3).to_string(index=False))
    print("\nB2  the fitted threshold c* (long iff f_ridge/slice > c*), F1")
    print(b["c_star"].round(3).to_string(index=False))
    print("\nB3  crossed Sharpe by year")
    print(
        b["years"]
        .pivot_table(
            index=["features", "weighting"], columns="year", values="Sharpe_crossed"
        )
        .round(2)
        .to_string()
    )
    t = b["table"].set_index(["features", "weighting"])
    held = 0
    for dn in [k for k in t.index.get_level_values(0).unique() if k != "-"]:
        u = t.loc[(dn, "(u) unweighted 1{y > slice}")]
        e = t.loc[(dn, "(e) ES-weighted 1{y > slice}")]
        p_ = t.loc[(dn, "(p) payoff-weighted 1{R > 0}")]
        ok = bool(
            u["hit_vs_variance"] >= e["hit_vs_variance"]
            and max(e["mean_mid"], p_["mean_mid"]) > u["mean_mid"]
        )
        held += ok
        print(
            f"    prediction [{dn}]: hit(u) {u['hit_vs_variance']:.3f} vs hit(e) {e['hit_vs_variance']:.3f}; "
            f"mean P&L (u) {u['mean_mid']:+.4f}, (e) {e['mean_mid']:+.4f}, (p) {p_['mean_mid']:+.4f} -> "
            f"{'held' if ok else 'did not hold'}"
        )
    n_cells = int((b["table"]["weighting"] != "(s) plain sign(s)").sum())
    print(
        f"\ncells {n_cells}; passes {int(b['table']['passes'].sum())}; about {0.025 * n_cells:.1f} expected by "
        f"chance; the weighted-vs-unweighted prediction held in {held} of 3 feature sets"
    )


if __name__ == "__main__":
    main()
