"""Proposal 13 - the terminal move: forecasting r^2 instead of RV.

A PRE-REGISTERED four-cell study. The cells T0-T3 below were fixed before any
number was seen; nothing is selected from a grid.

THE IDEA. The 15:30 straddle held to cash settlement pays on the terminal move
|S_close - K|, that is on the squared 30-minute return r^2, not on the sum of
squared one-minute returns RV. The forecast pipeline and the deck's
recalibration target RV. Under a martingale E[r^2] = E[RV], but realizations
differ, so a map calibrated to RV is a biased forecast of the terminal variance
and the sign rule compares the wrong quantity with the implied.

THE CELLS.

  T0 current (the control). rv_hat from the RV-targeted Mincer-Zarnowitz map -
  the deck's - and q = sign(rv_hat - iv_var).

  T1 terminal-variance target. The same weighted Mincer-Zarnowitz map (same
  weights 1/max(yhat, q10)^2, same 250-session window, same session fit mask
  10:30-16:00) with the target y = sqrt(r^2 / B) in place of sqrt(RV / B),
  giving tv_hat = (m^2 + s2) B as the forecast of the bar's squared terminal
  return; q = sign(tv_hat - iv_var).

  T2 expected payoff against premium. With tv_hat, the expected payoff of the
  ACTUAL nearest-OTM package (call at K_c, put at K_p, index at S_1530) under a
  normal terminal log return with variance tv_hat, in closed form (Black-76 at
  zero rate, forward = S_1530, total variance tv_hat). Long when the expected
  payoff exceeds the midpoint premium, short otherwise. The hurdle variant is
  long only above the ask, short only below the bid, flat in between. The same
  two rules are also run on rv_hat as a secondary.

  T3 terminal-move regressors. An expanding, prior-days-only least-squares fit
  (minimum 250 sessions, refit every day) of log r^2 at the 16:00 stamp on
  log tv_hat plus five F_t-measurable features known at 15:30:
    x1  |r| over 15:00-15:30 (the stamp-15:30 bar's terminal log return);
    x2  the day's net log return from the 10:00 stamp to the 15:30 stamp;
    x3  log of the day's realized variance so far (stamps 10:30..15:30) over
        its trailing prior-days-only mean;
    x4  the distance of S_1530 to the nearest 25-point strike as a fraction of
        the implied move S_1530 * sqrt(iv_var) (pinning);
    x5  a month-end / FOMC indicator (asl.fomc_and_monthend; NA read as other).
  The forecast is exp(fitted log r^2 + half the in-window residual variance)
  and q = sign(forecast - iv_var). This is the only cell with more than one
  degree of freedom; the feature list is pre-registered as written above and
  nothing was added to it.

  Oracles (reference, not trades). sign(RV - iv_var) and sign(r^2 - iv_var) at
  15:30: their agreement, and how well each predicts the sign of the straddle's
  actual return R.

FILLS. The midpoint case enters at the quoted midpoint. The crossed case pays
the touch at entry - buy at the ask, sell at the bid - and cash settlement pays
no exit spread (asl.crossed_premium_return).

STANDING RULE. Nothing is adopted unless it beats T0 at the crossed spread with
an interval excluding zero AND improves calibration against r^2.

Outputs: CSV and PNG under results/atm_straddle_intraday/proposals/13/.
Every number in 13_terminal_move.md is printed by this script.

Run:  python writeup/intraday_proposals/13_terminal_move.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _bootstrap_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "notebooks" / "atm_straddle_lib.py").exists():
            return q
    raise FileNotFoundError("repo root not found from " + str(start))


sys.path.insert(0, str(_bootstrap_repo(Path(__file__)) / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
DECK_DIR = REPO / "results" / "atm_straddle_0dte_1530"
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "13"

# The main repository's minute-bar summary panel: sumret is the bar's terminal
# log return, sumret2 the bar's realized variance (identical to the forecast
# panel's rv_raw), on naive-ET bar-end stamps stored as strings.
CORE_STATS = Path("C:/Users/james/CC Allowed/harxhar-clean/data/core_stats.parquet")

PRIMARY = "blk2"
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
PLACEBO_DRAWS = 2000
N_CUTS = 10
T3_MIN_DAYS = 250  # prior sessions before the terminal-move regression fires
TRAIL_MIN = 63  # prior sessions before a trailing mean is used
STRIKE_GRID = 25.0  # the pinning grid in index points
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
FILLS = ("mid", "crossed")

# The deck's own rule table for the block-diagonal ridge on its 866 days.
GATE = {
    "sign_mean": 0.094736,
    "sign_t": 2.480957,
    "sign_sharpe": 1.338322,
    "as_sharpe": 0.203779,
}
GATE_TOL = 1e-6

CELLS = [
    "T0 sign(rv_hat - iv)",
    "T1 sign(tv_hat - iv)",
    "T2 payoff > mid (tv_hat)",
    "T2h payoff vs touch (tv_hat)",
    "T2b payoff > mid (rv_hat)",
    "T2bh payoff vs touch (rv_hat)",
    "T3 sign(f3 - iv)",
]


def tick(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def hdr(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


# ------------------------------------------------------------------- panel


def load_core() -> pd.DataFrame:
    core = pd.read_parquet(CORE_STATS)
    core["et_naive"] = pd.to_datetime(core["endbartime"])
    return core[["et_naive", "sumret", "sumret2", "numobs"]]


def build_panel(path: Path, core: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The forecast panel with the terminal move attached, plus the fit mask.

    Both recalibrations run on this frame: the deck's (target RV) and the
    terminal one (target r^2). Only the target column differs, so the weights,
    the window, the mask and the session ranking are identical by construction.
    """
    df, rth = asl._panel_frame(path)
    df["et_naive"] = df["et"].dt.tz_localize(None)
    n0 = len(df)
    df = df.merge(core, on="et_naive", how="left")
    if len(df) != n0:
        raise ValueError("core_stats join changed the panel's row count")
    if int(df.loc[rth, "sumret"].isna().sum()):
        raise ValueError("core_stats does not cover every fit-mask row")
    gap = float((df["rv_raw"] - df["sumret2"]).abs().max())
    if gap != 0.0:
        raise ValueError(f"rv_raw and sumret2 disagree by {gap}")
    df["r2"] = df["sumret"] ** 2
    df["mins"] = (df["et"].dt.hour * 60 + df["et"].dt.minute).to_numpy()
    return df, rth


def recalibrate(df: pd.DataFrame, rth: np.ndarray, target: str) -> pd.DataFrame:
    """The library's weighted Mincer-Zarnowitz map with the given target column."""
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    f, m, s2 = asl.second_order_mz(
        df["yhat"].to_numpy(float),
        df[target].to_numpy(float),
        df["baseline"].to_numpy(float),
        day_codes,
        len(uniq),
        need_days=None,
        fit_mask=rth,
        method="mean",
    )
    out = pd.DataFrame({"f": f, "m": m, "s2": s2}, index=df.index)
    out.loc[df["early_close"].to_numpy(bool), :] = np.nan
    return out


# -------------------------------------------------------------- statistics


def safe_t(x) -> float:
    v = pd.Series(x).astype(float).dropna()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * np.sqrt(len(v))) if sd > 0 else float("nan")


def sharpe(x) -> float:
    v = pd.Series(x).astype(float).dropna().to_numpy()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * ANN) if sd > 0 else float("nan")


def qlike(f, y) -> pd.Series:
    """QLIKE loss y/f - log(y/f) - 1; rows with y <= 0 or f <= 0 are NaN."""
    f = pd.Series(f).astype(float)
    y = pd.Series(y).astype(float)
    y.index = f.index
    ok = (f > 0) & (y > 0)
    r = y.where(ok) / f.where(ok)
    return r - np.log(r) - 1.0


def dm_t(loss_a, loss_b) -> tuple[float, int]:
    """Autocorrelation-robust t of mean(loss_a - loss_b); negative favours a."""
    d = (pd.Series(loss_a).astype(float) - pd.Series(loss_b).astype(float)).dropna()
    return asl.newey_west_t(d)


_BOOT_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    """One circular-block index array per sample size, shared across cells."""
    if n not in _BOOT_IDX:
        _BOOT_IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng(SEED), n, BOOT_BLOCK, BOOT_B
        )
    return _BOOT_IDX[n]


def paired_stats(base: pd.Series, cell: pd.Series) -> dict[str, float]:
    """cell - base on the common days: mean, plain t, HAC t, Sharpe difference."""
    j = pd.concat([base.rename("a"), cell.rename("b")], axis=1).dropna()
    x = j["b"].to_numpy(float)
    y = j["a"].to_numpy(float)
    d = x - y
    n = len(d)
    hac, lag = asl.newey_west_t(pd.Series(d))
    hat = (x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1)) * ANN
    idx = boot_idx(n)
    xa, ya = x[idx], y[idx]
    ds = (
        xa.mean(axis=1) / xa.std(axis=1, ddof=1)
        - ya.mean(axis=1) / ya.std(axis=1, ddof=1)
    ) * ANN
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {
        "n": float(n),
        "mean_diff": float(d.mean()),
        "t_diff": safe_t(pd.Series(d)),
        "hac_t_diff": float(hac),
        "hac_lag": float(lag),
        "dSharpe": float(hat),
        "pct_draws_positive": float(100.0 * (ds > 0).mean()),
        "pctile_lo": float(lo),
        "pctile_hi": float(hi),
        "basic_lo": float(2.0 * hat - hi),
        "basic_hi": float(2.0 * hat - lo),
    }


def placebo_rate_matched(
    long_r: pd.Series, short_r: pd.Series, pos: pd.Series
) -> dict[str, float]:
    """Sharpe percentile of the real sign among random signs at its own long rate."""
    j = pd.concat(
        [long_r.rename("l"), short_r.rename("s"), pos.rename("q")], axis=1
    ).dropna()
    ell = j["l"].to_numpy(float)
    sh = j["s"].to_numpy(float)
    q = j["q"].to_numpy(float)
    real = np.where(q > 0, ell, np.where(q < 0, sh, 0.0))
    real_sh = float(real.mean() / real.std(ddof=1) * ANN)
    n = len(ell)
    p_long = float((q > 0).mean())
    p_flat = float((q == 0).mean())
    rng = np.random.default_rng(SEED)
    u = rng.random((PLACEBO_DRAWS, n))
    # rate-matched: the same long / short / flat shares, permuted across days
    draw = np.where(
        u < p_long, 1.0, np.where(u < p_long + (1.0 - p_flat - p_long), -1.0, 0.0)
    )
    r = np.where(draw > 0, ell[None, :], np.where(draw < 0, sh[None, :], 0.0))
    s_draw = r.mean(axis=1) / r.std(axis=1, ddof=1) * ANN
    return {
        "n": float(n),
        "long_share_pct": 100.0 * p_long,
        "flat_share_pct": 100.0 * p_flat,
        "Sharpe_real": real_sh,
        "pctile": float(100.0 * (s_draw < real_sh).mean()),
        "placebo_median": float(np.median(s_draw)),
        "placebo_p05": float(np.percentile(s_draw, 5)),
        "placebo_p95": float(np.percentile(s_draw, 95)),
    }


# ------------------------------------------------------------- expected payoff


def package_expected_payoff(
    s_spot: pd.Series, k_c: pd.Series, k_p: pd.Series, var: pd.Series
) -> pd.Series:
    """E[(S_T-K_c)^+ + (K_p-S_T)^+] with log(S_T/S) ~ N(-var/2, var).

    Black-76 at a zero rate with forward F = S_1530 and total variance var; the
    same closed form the library prices the package with (asl._bsm_package_price).
    """
    out = []
    for s, kc, kp, v in zip(
        s_spot.to_numpy(float),
        k_c.to_numpy(float),
        k_p.to_numpy(float),
        var.to_numpy(float),
        strict=True,
    ):
        if not (
            np.isfinite(s) and np.isfinite(kc) and np.isfinite(kp) and np.isfinite(v)
        ):
            out.append(np.nan)
            continue
        out.append(asl._bsm_package_price(float(np.sqrt(max(v, 0.0))), s, kc, kp))
    return pd.Series(out, index=s_spot.index)


# ------------------------------------------------------------------ features


def terminal_features(df: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """The F_t-measurable terminal-move features, one row per traded day."""
    ses = df[df["mins"].between(10 * 60 + 30, 15 * 60 + 30)].copy()
    piv_r = ses.pivot_table(index="date", columns="mins", values="sumret")
    piv_v = ses.pivot_table(index="date", columns="mins", values="rv_raw")
    abs_last = piv_r[15 * 60 + 30].abs()
    net_1000_1530 = piv_r.sum(axis=1, min_count=1)  # stamps 10:30..15:30
    rv_sofar = piv_v.sum(axis=1, min_count=1)
    feat = pd.DataFrame(
        {"abs_last": abs_last, "net_ret": net_1000_1530, "rv_sofar": rv_sofar}
    ).reindex(days)
    trail = feat["rv_sofar"].expanding(min_periods=TRAIL_MIN).mean().shift(1)
    feat["log_rv_sofar_rel"] = np.log(feat["rv_sofar"] / trail)
    return feat


def expanding_lr(
    y: pd.Series, X: pd.DataFrame, min_days: int
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Expanding prior-days-only OLS.

    Returns (fitted value, coefficient path, in-window residual variance,
    in-window mean of exp(residual)). The last column is Duan's smearing
    factor; it is a DIAGNOSTIC only - the pre-registered T3 forecast is
    exp(fit + s2/2), the Gaussian retransformation.
    """
    ok = y.notna() & X.notna().all(axis=1)
    yv = y.to_numpy(float)
    Xv = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    n = len(y)
    fit = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    smear = np.full(n, np.nan)
    coefs = np.full((n, Xv.shape[1]), np.nan)
    okv = ok.to_numpy(bool)
    for i in range(n):
        if not okv[i]:
            continue
        m = okv[:i]
        if int(m.sum()) < min_days:
            continue
        A = Xv[:i][m]
        b = yv[:i][m]
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = b - A @ beta
        coefs[i] = beta
        s2[i] = float(res @ res) / len(res)  # ddof 0, the library's convention
        smear[i] = float(np.mean(np.exp(res)))
        fit[i] = float(Xv[i] @ beta)
    names = ["const", *list(X.columns)]
    return (
        pd.Series(fit, index=y.index),
        pd.DataFrame(coefs, index=y.index, columns=names),
        pd.Series(s2, index=y.index),
        pd.Series(smear, index=y.index),
    )


# ----------------------------------------------------------------- the book


def build_book(tag: str, core: pd.DataFrame) -> dict:
    """Every per-day series the cells need, for one forecast tag."""
    path = asl.yhat_paths(REPO)[tag]
    df, rth = build_panel(path, core)
    rv_map = recalibrate(df, rth, "rv_raw")
    tv_map = recalibrate(df, rth, "r2")
    close_row = (df["mins"] == 16 * 60) & ~df["early_close"].to_numpy(bool)
    per_day = pd.DataFrame(
        {
            "rv_hat_recon": rv_map["f"],
            "tv_hat": tv_map["f"],
            "tv_m": tv_map["m"],
            "tv_s2": tv_map["s2"],
            "rv_raw": df["rv_raw"],
            "r2": df["r2"],
            "sumret": df["sumret"],
        }
    )[close_row.to_numpy()]
    per_day.index = df.loc[close_row.to_numpy(), "date"].to_numpy()
    per_day = per_day[~per_day.index.duplicated()]

    deck = pd.read_parquet(DECK_DIR / f"daily_{tag}.parquet")
    px = deck.join(per_day, how="left")
    px["bid_entry"] = px["bid_c"].astype(float) + px["bid_p"].astype(float)
    px["ask_entry"] = px["ask_c"].astype(float) + px["ask_p"].astype(float)
    return {
        "tag": tag,
        "df": df,
        "rth": rth,
        "rv_map": rv_map,
        "tv_map": tv_map,
        "px": px,
    }


def cell_positions(px: pd.DataFrame, feat_out: dict | None) -> pd.DataFrame:
    """The pre-registered positions, one column per cell."""
    iv = px["iv_var"].astype(float)
    q = pd.DataFrame(index=px.index)
    q[CELLS[0]] = np.where(px["rv_hat"].astype(float) > iv, 1.0, -1.0)
    q[CELLS[1]] = np.where(px["tv_hat"].astype(float) > iv, 1.0, -1.0)
    for name, col in ((CELLS[2], "ep_tv"), (CELLS[4], "ep_rv")):
        q[name] = np.where(px[col].astype(float) > px["entry"].astype(float), 1.0, -1.0)
    for name, col in ((CELLS[3], "ep_tv"), (CELLS[5], "ep_rv")):
        e = px[col].astype(float)
        q[name] = np.where(
            e > px["ask_entry"].astype(float),
            1.0,
            np.where(e < px["bid_entry"].astype(float), -1.0, 0.0),
        )
    if feat_out is None:
        q[CELLS[6]] = 0.0
    else:
        f3 = feat_out["f3"]
        q[CELLS[6]] = np.where(f3.isna(), 0.0, np.where(f3 > iv, 1.0, -1.0))
    return q


def returns_for(px: pd.DataFrame, q: pd.Series, fill: str) -> pd.Series:
    if fill == "mid":
        return q.astype(float) * px["R"].astype(float)
    return asl.crossed_premium_return(
        q.astype(float), px["exit"], px["bid_entry"], px["ask_entry"]
    )


def points_for(px: pd.DataFrame, q: pd.Series, fill: str) -> pd.Series:
    ex = px["exit"].astype(float)
    if fill == "mid":
        return q.astype(float) * (ex - px["entry"].astype(float))
    qq = q.astype(float)
    return pd.Series(
        np.where(
            qq > 0,
            ex - px["ask_entry"].astype(float),
            np.where(qq < 0, px["bid_entry"].astype(float) - ex, 0.0),
        ),
        index=px.index,
    )


# ==================================================================== main


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    print(f"proposal 13 - the terminal move.  repo {REPO}")
    print(
        "PRE-REGISTERED: cells T0, T1, T2 (with its hurdle variant and its rv_hat "
        "secondary) and T3 were fixed before any number was seen."
    )

    core = load_core()
    tick(t0, f"core_stats loaded ({len(core)} rows)")
    books = {tag: build_book(tag, core) for tag in asl.MODEL_ORDER}
    tick(t0, "recalibrations built for all eight forecasts")
    B = books[PRIMARY]
    px = B["px"].copy()
    df, rth = B["df"], B["rth"]

    # ------------------------------------------------------------ 0. the gate
    hdr("0. GATE - the deck's rule table for the block-diagonal ridge")
    sizes = asl.rule_sizes(px)
    gate_tab = pd.DataFrame(
        {
            name: asl.rule_row(sizes[name] * px["R"], sizes[name])
            for name in ["always short", "sign(s)"]   # the deck's two standing rules
        }
    ).T
    print(gate_tab.to_string())
    got = {
        "sign_mean": float(gate_tab.loc["sign(s)", "mean"]),
        "sign_t": float(gate_tab.loc["sign(s)", "t_mean"]),
        "sign_sharpe": float(gate_tab.loc["sign(s)", "Sharpe_ann"]),
        "as_sharpe": float(gate_tab.loc["always short", "Sharpe_ann"]),
    }
    gate_rows = [
        {
            "figure": k,
            "target": GATE[k],
            "reproduced": got[k],
            "abs_diff": abs(got[k] - GATE[k]),
        }
        for k in GATE
    ]
    gate_rows.append(
        {
            "figure": "rv_hat rebuilt from the panel",
            "target": 0.0,
            "reproduced": float((px["rv_hat"] - px["rv_hat_recon"]).abs().max()),
            "abs_diff": float((px["rv_hat"] - px["rv_hat_recon"]).abs().max()),
        }
    )
    gate_rows.append(
        {
            "figure": "deck pos vs sign(rv_hat - iv_var)",
            "target": 0.0,
            "reproduced": float(
                (px["pos"] - np.where(px["rv_hat"] > px["iv_var"], 1.0, -1.0))
                .abs()
                .sum()
            ),
            "abs_diff": float(
                (px["pos"] - np.where(px["rv_hat"] > px["iv_var"], 1.0, -1.0))
                .abs()
                .sum()
            ),
        }
    )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT / "13_gate.csv", index=False)
    print(gate.to_string(index=False))
    worst = float(gate["abs_diff"].max())
    print(f"n days {len(px)};  worst gate difference {worst:.3e}")
    if worst > GATE_TOL:
        raise SystemExit(f"GATE FAILED: worst difference {worst:.3e} > {GATE_TOL}")
    print("GATE PASSED.")

    # ------------------------------------------ 1. the RV / r^2 bias
    hdr("1. THE BIAS - realized variance against the squared terminal move")
    ses = df[rth].copy()
    ses["hhmm"] = ses["et"].dt.strftime("%H:%M")
    ses["year"] = ses["et"].dt.year
    by_clock = ses.groupby("hhmm").apply(
        lambda x: pd.Series(
            {
                "n": float(len(x)),
                "mean_RV": float(x["rv_raw"].mean()),
                "mean_r2": float(x["r2"].mean()),
                "ratio_of_means": float(x["rv_raw"].mean() / x["r2"].mean()),
                "median_ratio": float(
                    (x["rv_raw"] / x["r2"].where(x["r2"] > 0)).median()
                ),
                "pct_r2_gt_RV": float(100.0 * (x["r2"] > x["rv_raw"]).mean()),
                "n_zero_r2": float((x["r2"] == 0).sum()),
            }
        ),
        include_groups=False,
    )
    by_clock.to_csv(OUT / "13_bias_by_clock.csv")
    print("mean(RV) / mean(r^2) by clock, all session bars (bar-END stamps):")
    print(by_clock.round(6).to_string())
    by_year = ses.groupby("year").apply(
        lambda x: pd.Series(
            {
                "n": float(len(x)),
                "ratio_all_bars": float(x["rv_raw"].mean() / x["r2"].mean()),
            }
        ),
        include_groups=False,
    )
    close_bars = ses[ses["hhmm"] == "16:00"]
    by_year["ratio_close_bar"] = close_bars.groupby("year").apply(
        lambda x: float(x["rv_raw"].mean() / x["r2"].mean()), include_groups=False
    )
    by_year.to_csv(OUT / "13_bias_by_year.csv")
    print("\nmean(RV) / mean(r^2) by year:")
    print(by_year.round(4).to_string())
    pooled = float(ses["rv_raw"].mean() / ses["r2"].mean())
    close_ratio = float(close_bars["rv_raw"].mean() / close_bars["r2"].mean())
    deck_ratio = float(px["rv_raw"].mean() / px["r2"].mean())
    print(
        f"\npooled over all {len(ses)} session bars: {pooled:.4f}"
        f"\nthe 16:00 stamp alone ({len(close_bars)} bars): {close_ratio:.4f}"
        f"\nthe {len(px)} traded bars: {deck_ratio:.4f}"
    )

    # the tape check the prompt asks for
    lr_close = np.log(px["S_close"].astype(float) / px["S"].astype(float))
    lr_tape = np.log(px["S_1600_tape"].astype(float) / px["S"].astype(float))
    tape = pd.DataFrame(
        [
            {
                "against": "log(S_close / S_1530), official close",
                "n": float(lr_close.notna().sum()),
                "mean_abs_diff": float((px["sumret"] - lr_close).abs().mean()),
                "median_abs_diff": float((px["sumret"] - lr_close).abs().median()),
                "max_abs_diff": float((px["sumret"] - lr_close).abs().max()),
                "corr": float(px["sumret"].corr(lr_close)),
            },
            {
                "against": "log(S_1600_tape / S_1530), chain tape",
                "n": float(lr_tape.notna().sum()),
                "mean_abs_diff": float((px["sumret"] - lr_tape).abs().mean()),
                "median_abs_diff": float((px["sumret"] - lr_tape).abs().median()),
                "max_abs_diff": float((px["sumret"] - lr_tape).abs().max()),
                "corr": float(px["sumret"].corr(lr_tape)),
            },
        ]
    )
    tape.to_csv(OUT / "13_tape_check.csv", index=False)
    print("\nsumret at the 16:00 stamp against the price path (the futures bar")
    print("and the cash close differ slightly):")
    print(tape.to_string(index=False))
    print(
        f"sd of sumret {float(px['sumret'].std()):.6f}, sd of log(S_close/S_1530) {float(lr_close.std()):.6f}"
    )

    # -------------------------------------------------- 2. T1 and calibration
    hdr("2. T1 - the terminal-variance map, and calibration against r^2")
    cal_rows = []
    for label, frame, fcol_rv, fcol_tv, ycol in (
        ("866 traded bars", px, "rv_hat", "tv_hat", "r2"),
        (
            "all session bars",
            df[rth]
            .join(B["rv_map"]["f"].rename("rv_f"))
            .join(B["tv_map"]["f"].rename("tv_f")),
            "rv_f",
            "tv_f",
            "r2",
        ),
    ):
        f_rv = frame[fcol_rv].astype(float)
        f_tv = frame[fcol_tv].astype(float)
        y = frame[ycol].astype(float)
        q_rv, q_tv = qlike(f_rv, y), qlike(f_tv, y)
        both = q_rv.notna() & q_tv.notna()
        t, lag = dm_t(q_tv[both], q_rv[both])
        cal_rows.append(
            {
                "frame": label,
                "n_scored": float(both.sum()),
                "n_dropped_zero_r2": float(int((y <= 0).sum())),
                "QLIKE rv_hat": float(q_rv[both].mean()),
                "QLIKE tv_hat": float(q_tv[both].mean()),
                "DM t (tv - rv)": float(t),
                "DM lag": float(lag),
                "mean rv_hat / mean r2": float(f_rv[both].mean() / y[both].mean()),
                "mean tv_hat / mean r2": float(f_tv[both].mean() / y[both].mean()),
                "mean tv_hat / mean rv_hat": float(
                    f_tv[both].mean() / f_rv[both].mean()
                ),
            }
        )
        # against RV, for reference
        qr = qlike(f_rv, frame["rv_raw"].astype(float))
        qt = qlike(f_tv, frame["rv_raw"].astype(float))
        b2 = qr.notna() & qt.notna()
        t2, lag2 = dm_t(qt[b2], qr[b2])
        cal_rows.append(
            {
                "frame": label + " (target RV, reference)",
                "n_scored": float(b2.sum()),
                "n_dropped_zero_r2": 0.0,
                "QLIKE rv_hat": float(qr[b2].mean()),
                "QLIKE tv_hat": float(qt[b2].mean()),
                "DM t (tv - rv)": float(t2),
                "DM lag": float(lag2),
                "mean rv_hat / mean r2": float(
                    f_rv[b2].mean() / frame["rv_raw"].astype(float)[b2].mean()
                ),
                "mean tv_hat / mean r2": float(
                    f_tv[b2].mean() / frame["rv_raw"].astype(float)[b2].mean()
                ),
                "mean tv_hat / mean rv_hat": float(f_tv[b2].mean() / f_rv[b2].mean()),
            }
        )
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "13_calibration.csv", index=False)
    print("QLIKE of each map against the squared terminal move r^2 (and, for")
    print("reference, against RV). A negative DM t favours the terminal map.")
    print(cal.round(6).to_string(index=False))

    # ---------------------------------------------------- 3. expected payoffs
    hdr("3. T2 - the expected payoff of the actual package")
    px["ep_tv"] = package_expected_payoff(px["S"], px["K_c"], px["K_p"], px["tv_hat"])
    px["ep_rv"] = package_expected_payoff(px["S"], px["K_c"], px["K_p"], px["rv_hat"])
    px["ep_iv"] = package_expected_payoff(px["S"], px["K_c"], px["K_p"], px["iv_var"])
    pr = px["ep_iv"] / px["entry"]
    print(
        "pricing check: the same closed form at the quoted implied variance "
        "reproduces the package midpoint,\n  ratio mean "
        f"{float(pr.mean()):.4f}, 5th-95th {float(pr.quantile(0.05)):.4f}-{float(pr.quantile(0.95)):.4f}"
    )
    ep = pd.DataFrame(
        {
            "mean_entry_mid": [float(px["entry"].mean())],
            "mean_bid": [float(px["bid_entry"].mean())],
            "mean_ask": [float(px["ask_entry"].mean())],
            "mean_E_payoff_tv": [float(px["ep_tv"].mean())],
            "mean_E_payoff_rv": [float(px["ep_rv"].mean())],
            "mean_realised_exit": [float(px["exit"].mean())],
            "median_half_spread_pct": [
                float(
                    (0.5 * (px["ask_entry"] - px["bid_entry"]) / px["entry"]).median()
                    * 100.0
                )
            ],
        }
    )
    ep.to_csv(OUT / "13_expected_payoff.csv", index=False)
    print(ep.round(4).to_string(index=False))

    # ------------------------------------------------------ 4. T3 regression
    hdr("4. T3 - the terminal-move regression")
    days = pd.DatetimeIndex(px.index)
    feat = terminal_features(df, days)
    cal_flags = asl.fomc_and_monthend(days, REPO)
    event = cal_flags["is_event"].fillna(False).astype(float)
    px["pin"] = (
        px["S"].astype(float)
        - STRIKE_GRID * np.round(px["S"].astype(float) / STRIKE_GRID)
    ).abs() / (px["S"].astype(float) * np.sqrt(px["iv_var"].astype(float)))
    X = pd.DataFrame(
        {
            "log_tv_hat": np.log(px["tv_hat"].astype(float)),
            "abs_ret_1500_1530": feat["abs_last"],
            "net_ret_1000_1530": feat["net_ret"],
            "log_rv_sofar_rel": feat["log_rv_sofar_rel"],
            "pin_frac_implied_move": px["pin"],
            "event_fomc_or_monthend": event,
        },
        index=days,
    )
    y3 = np.log(px["r2"].astype(float))
    if not np.isfinite(y3).all():
        raise SystemExit("log r^2 is not finite on every traded day")
    fit3, coef3, s2_3, smear3 = expanding_lr(y3, X, T3_MIN_DAYS)
    f3 = np.exp(fit3 + 0.5 * s2_3)  # the pre-registered retransformation
    f3s = np.exp(fit3) * smear3  # POST-HOC diagnostic only (see section 4b)
    feat_out = {"f3": f3, "coef": coef3, "s2": s2_3, "X": X}
    print(
        f"design: {len(X.columns)} regressors plus a constant; the fit fires on "
        f"{int(f3.notna().sum())} of {len(px)} days\n"
        f"first traded day {f3.dropna().index.min().date()}, last {f3.dropna().index.max().date()}"
    )
    print("\nfeature summary (traded days):")
    print(
        X.describe()
        .T[["count", "mean", "std", "min", "50%", "max"]]
        .round(6)
        .to_string()
    )
    print(f"event days (FOMC or month-end): {int(event.sum())} of {len(event)}")

    # coefficient stability: the expanding path's yearly means, and a per-year
    # in-sample fit with a heteroskedasticity-robust t (a diagnostic, not traded)
    cp = coef3.dropna()
    path_year = cp.groupby(cp.index.year).mean()
    path_year.to_csv(OUT / "13_t3_coef_path_by_year.csv")
    print("\nthe expanding fit's coefficients, averaged within each year:")
    print(path_year.round(4).to_string())

    rows = []
    Xd = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    names = ["const", *list(X.columns)]
    for yr, idx in pd.Series(range(len(days)), index=days).groupby(days.year):
        sl = idx.to_numpy()
        A, b = Xd[sl], y3.to_numpy(float)[sl]
        good = np.isfinite(A).all(axis=1) & np.isfinite(b)
        A, b = A[good], b[good]
        if len(b) <= len(names) + 2:
            continue
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = b - A @ beta
        xtx_inv = np.linalg.pinv(A.T @ A)
        meat = A.T @ (A * (res**2)[:, None])
        se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
        for k, nm in enumerate(names):
            rows.append(
                {
                    "year": int(yr),
                    "n": len(b),
                    "term": nm,
                    "coef": float(beta[k]),
                    "t_robust": float(beta[k] / se[k]) if se[k] > 0 else float("nan"),
                }
            )
    t3coef = pd.DataFrame(rows)
    t3coef.to_csv(OUT / "13_t3_coef_by_year.csv", index=False)
    print("\nper-year in-sample coefficients (a stability check, never traded):")
    print(
        t3coef.pivot(index="term", columns="year", values="coef")
        .reindex(names)
        .round(3)
        .to_string()
    )
    print("\nthe same, heteroskedasticity-robust t-statistics:")
    print(
        t3coef.pivot(index="term", columns="year", values="t_robust")
        .reindex(names)
        .round(2)
        .to_string()
    )
    sign_stab = (
        t3coef.groupby("term")["coef"]
        .apply(
            lambda s: pd.Series(
                {
                    "years": len(s),
                    "same_sign_years": int(max((s > 0).sum(), (s < 0).sum())),
                }
            )
        )
        .unstack()
        .reindex(names)
    )
    print("\nsign stability across years:")
    print(sign_stab.to_string())

    # T3 calibration against T1 on T3's own days
    t3days = f3.dropna().index
    q_t1 = qlike(px.loc[t3days, "tv_hat"], px.loc[t3days, "r2"])
    q_t3 = qlike(f3.loc[t3days], px.loc[t3days, "r2"])
    q_t0 = qlike(px.loc[t3days, "rv_hat"], px.loc[t3days, "r2"])
    t31, lag31 = dm_t(q_t3, q_t1)
    t30, lag30 = dm_t(q_t3, q_t0)
    inc = pd.DataFrame(
        [
            {
                "frame": f"T3's {len(t3days)} days",
                "QLIKE T0 rv_hat": float(q_t0.mean()),
                "QLIKE T1 tv_hat": float(q_t1.mean()),
                "QLIKE T3": float(q_t3.mean()),
                "DM t (T3 - T1)": float(t31),
                "DM t (T3 - T0)": float(t30),
                "DM lag": float(lag31),
                "R2 of the expanding fit (log r^2)": float(
                    1.0
                    - np.var(y3.loc[t3days] - fit3.loc[t3days]) / np.var(y3.loc[t3days])
                ),
            }
        ]
    )
    inc.to_csv(OUT / "13_t3_incremental.csv", index=False)
    print("\nincremental calibration of T3:")
    print(inc.round(6).to_string(index=False))
    if lag30 != lag31:
        print(f"(the T3-T0 test uses lag {lag30})")

    # 4b. Why T3's level is what it is. POST-HOC DIAGNOSTIC, NOT A CELL.
    print(
        "\n--- 4b. the retransformation, a POST-HOC DIAGNOSTIC (not a "
        "pre-registered cell, not adoptable) ---"
    )
    q_t3s = qlike(f3s.loc[t3days], px.loc[t3days, "r2"])
    diag = pd.DataFrame(
        [
            {
                "mean in-window residual variance s2": float(s2_3.loc[t3days].mean()),
                "Gaussian factor exp(s2/2)": float(
                    np.exp(0.5 * s2_3.loc[t3days]).mean()
                ),
                "smearing factor mean exp(residual)": float(smear3.loc[t3days].mean()),
                "pi^2/2 (variance of log of a squared standard normal)": float(
                    np.pi**2 / 2
                ),
                "mean f3 / mean r2": float(
                    f3.loc[t3days].mean() / px.loc[t3days, "r2"].mean()
                ),
                "mean f3_smeared / mean r2": float(
                    f3s.loc[t3days].mean() / px.loc[t3days, "r2"].mean()
                ),
                "QLIKE f3 (pre-registered)": float(q_t3.mean()),
                "QLIKE f3 smeared (diagnostic)": float(q_t3s.mean()),
                "pct long, f3": float(
                    100.0 * (f3.loc[t3days] > px.loc[t3days, "iv_var"]).mean()
                ),
                "pct long, f3 smeared": float(
                    100.0 * (f3s.loc[t3days] > px.loc[t3days, "iv_var"]).mean()
                ),
            }
        ]
    )
    q_smear = pd.Series(
        np.where(f3s.loc[t3days] > px.loc[t3days, "iv_var"], 1.0, -1.0), index=t3days
    )
    for fill in FILLS:
        diag[f"Sharpe smeared, {fill}"] = sharpe(
            returns_for(px.loc[t3days], q_smear, fill)
        )
    diag.to_csv(OUT / "13_t3_retransformation.csv", index=False)
    print(diag.T.round(6).to_string(header=False))
    print(
        "log r^2 is the log of a squared normal, not a normal, so the Gaussian\n"
        "retransformation exp(s2/2) is the wrong constant and inflates the level."
    )

    # ---------------------------------------------------------- 5. oracles
    hdr("5. ORACLES - reference only, not trades")
    q_rv_or = pd.Series(
        np.where(px["rv_raw"] > px["iv_var"], 1.0, -1.0), index=px.index
    )
    q_r2_or = pd.Series(np.where(px["r2"] > px["iv_var"], 1.0, -1.0), index=px.index)
    sgn_R = np.sign(px["R"].astype(float))
    orows = []
    for nm, q in (("sign(RV - iv_var)", q_rv_or), ("sign(r^2 - iv_var)", q_r2_or)):
        orows.append(
            {
                "oracle": nm,
                "n": float(len(q)),
                "pct_long": float(100.0 * (q > 0).mean()),
                "corr with sign(R)": float(np.corrcoef(q, sgn_R)[0, 1]),
                "hit rate q R > 0": float((q * px["R"] > 0).mean()),
                "Sharpe mid": sharpe(q * px["R"]),
                "Sharpe crossed": sharpe(
                    asl.crossed_premium_return(
                        q, px["exit"], px["bid_entry"], px["ask_entry"]
                    )
                ),
            }
        )
    orows.append(
        {
            "oracle": "agreement of the two oracles",
            "n": float(len(px)),
            "pct_long": float("nan"),
            "corr with sign(R)": float("nan"),
            "hit rate q R > 0": float((q_rv_or == q_r2_or).mean()),
            "Sharpe mid": float("nan"),
            "Sharpe crossed": float("nan"),
        }
    )
    orc = pd.DataFrame(orows)
    orc.to_csv(OUT / "13_oracles.csv", index=False)
    print(orc.round(4).to_string(index=False))
    print(
        "\nmean r^2 / iv_var "
        f"{float((px['r2'] / px['iv_var']).mean()):.4f}, median {float((px['r2'] / px['iv_var']).median()):.4f};"
        f"\nmean RV / iv_var {float((px['rv_raw'] / px['iv_var']).mean()):.4f}, "
        f"median {float((px['rv_raw'] / px['iv_var']).median()):.4f}"
    )

    # ------------------------------------------------------- 6. the scoring
    hdr("6. THE CELLS - rule tables at both fills")
    q_all = cell_positions(px, feat_out)
    q_all.to_csv(OUT / "13_positions.csv")
    n_untrade = {
        c: asl.crossed_untradeable_count(q_all[c], px["bid_entry"], px["ask_entry"])
        for c in CELLS
    }
    print("rows a crossed fill cannot price:", n_untrade)

    rets: dict[tuple[str, str], pd.Series] = {}
    ptss: dict[tuple[str, str], pd.Series] = {}
    tab_rows = []
    for fill in FILLS:
        for c in CELLS:
            r = returns_for(px, q_all[c], fill)
            rets[(c, fill)] = r
            ptss[(c, fill)] = points_for(px, q_all[c], fill)
            row = asl.rule_row(r, q_all[c])
            n_diff = int((q_all[c] != q_all[CELLS[0]]).sum())
            tab_rows.append(
                {
                    "cell": c,
                    "fill": fill,
                    "n": float(row["n"]),
                    "mean": float(row["mean"]),
                    "t": float(row["t_mean"]),
                    "Sharpe": float(row["Sharpe_ann"]),
                    "n_buy": float(row["n_buy"]),
                    "pct_buy": float(row["pct_buy"]),
                    "n_flat": float((q_all[c] == 0).sum()),
                    "days_differ_from_T0": float(n_diff),
                    "hit_rate": float((q_all[c] * px["R"] > 0).mean()),
                    "mean_pts": float(ptss[(c, fill)].mean()),
                    "Sharpe_pts": sharpe(ptss[(c, fill)]),
                }
            )
    cells_tab = pd.DataFrame(tab_rows)
    cells_tab.to_csv(OUT / "13_cells.csv", index=False)
    for fill in FILLS:
        print(
            f"\n--- {fill} fills, all {len(px)} days (T3 sits flat in its warm-up) ---"
        )
        print(
            cells_tab[cells_tab["fill"] == fill]
            .drop(columns=["fill"])
            .round(6)
            .to_string(index=False)
        )

    # the same on T3's support, so every cell is read on one frame
    sub = px.loc[t3days]
    sub_rows = []
    for fill in FILLS:
        for c in CELLS:
            r = rets[(c, fill)].loc[t3days]
            row = asl.rule_row(r, q_all[c].loc[t3days])
            sub_rows.append(
                {
                    "cell": c,
                    "fill": fill,
                    "n": float(row["n"]),
                    "mean": float(row["mean"]),
                    "t": float(row["t_mean"]),
                    "Sharpe": float(row["Sharpe_ann"]),
                    "n_buy": float(row["n_buy"]),
                    "pct_buy": float(row["pct_buy"]),
                    "days_differ_from_T0": float(
                        (q_all[c].loc[t3days] != q_all[CELLS[0]].loc[t3days]).sum()
                    ),
                    "mean_pts": float(ptss[(c, fill)].loc[t3days].mean()),
                }
            )
    cells_sub = pd.DataFrame(sub_rows)
    cells_sub.to_csv(OUT / "13_cells_t3_support.csv", index=False)
    for fill in FILLS:
        print(f"\n--- {fill} fills, T3's {len(sub)} days ---")
        print(
            cells_sub[cells_sub["fill"] == fill]
            .drop(columns=["fill"])
            .round(6)
            .to_string(index=False)
        )

    # ------------------------------------------------- 7. paired differences
    hdr("7. PAIRED DIFFERENCES against T0")
    pr_rows = []
    for fill in FILLS:
        for c in CELLS[1:]:
            idx = t3days if c == CELLS[6] else px.index
            st = paired_stats(rets[(CELLS[0], fill)].loc[idx], rets[(c, fill)].loc[idx])
            pr_rows.append({"cell": c, "fill": fill, **st})
    paired = pd.DataFrame(pr_rows)
    paired.to_csv(OUT / "13_paired.csv", index=False)
    print(
        "cell - T0, daily, on each cell's own tradeable days. HAC t at lag "
        f"floor(1.5 n^(1/3)); Sharpe difference by circular block bootstrap, "
        f"block {BOOT_BLOCK}, B = {BOOT_B}, rng({SEED}), draws shared across cells."
    )
    print(paired.round(4).to_string(index=False))

    # ------------------------------------------------------------ 8. placebo
    hdr("8. PLACEBO - random signs at each cell's own rate")
    beat = []
    for c in CELLS[1:]:
        row = paired[(paired["cell"] == c) & (paired["fill"] == "crossed")]
        idx = t3days if c == CELLS[6] else px.index
        s_cell = sharpe(rets[(c, "crossed")].loc[idx])
        s_t0 = sharpe(rets[(CELLS[0], "crossed")].loc[idx])
        if s_cell > s_t0:
            beat.append(c)
        print(
            f"  {c:30s} crossed Sharpe {s_cell:+.4f} vs T0 {s_t0:+.4f} "
            f"(dSharpe {float(row['dSharpe'].iloc[0]):+.4f})"
        )
    print("cells beating T0 at the crossed spread:", beat if beat else "none")
    pl_rows = []
    long_r = px["exit"] / px["ask_entry"] - 1.0
    short_r = 1.0 - px["exit"] / px["bid_entry"]
    long_m = px["R"].astype(float)
    short_m = -px["R"].astype(float)
    for c in beat:
        idx = t3days if c == CELLS[6] else px.index
        for fill, lr_, sr_ in (("mid", long_m, short_m), ("crossed", long_r, short_r)):
            pl_rows.append(
                {
                    "cell": c,
                    "fill": fill,
                    **placebo_rate_matched(
                        lr_.loc[idx], sr_.loc[idx], q_all[c].loc[idx]
                    ),
                }
            )
    placebo = (
        pd.DataFrame(pl_rows)
        if pl_rows
        else pd.DataFrame(columns=["cell", "fill", "n", "Sharpe_real", "pctile"])
    )
    placebo.to_csv(OUT / "13_placebo.csv", index=False)
    print(
        placebo.round(4).to_string(index=False)
        if len(placebo)
        else "(no cell qualifies)"
    )

    # ------------------------------- 8b. the post-hoc diagnostic, scored the same way
    hdr("8b. THE POST-HOC RETRANSFORMATION DIAGNOSTIC - not a pre-registered cell")
    print(
        "Reported so the reader can see what the T3 regressors do once the level\n"
        "is fixed. It is NOT adoptable: the standing rule needs an improvement in\n"
        "calibration as well, and this forecast is worse-calibrated than T0."
    )
    legs = {"mid": (long_m, short_m), "crossed": (long_r, short_r)}
    dg_rows = []
    for fill in FILLS:
        r = returns_for(px.loc[t3days], q_smear, fill)
        row = asl.rule_row(r, q_smear)
        st = paired_stats(rets[(CELLS[0], fill)].loc[t3days], r)
        lr_, sr_ = legs[fill]
        pl = placebo_rate_matched(lr_.loc[t3days], sr_.loc[t3days], q_smear)
        dg_rows.append(
            {
                "fill": fill,
                "n": float(row["n"]),
                "mean": float(row["mean"]),
                "t": float(row["t_mean"]),
                "Sharpe": float(row["Sharpe_ann"]),
                "pct_buy": float(row["pct_buy"]),
                "days_differ_from_T0": float(
                    (q_smear != q_all[CELLS[0]].loc[t3days]).sum()
                ),
                "dSharpe_vs_T0": st["dSharpe"],
                "pctile_lo": st["pctile_lo"],
                "pctile_hi": st["pctile_hi"],
                "basic_lo": st["basic_lo"],
                "basic_hi": st["basic_hi"],
                "hac_t_diff": st["hac_t_diff"],
                "placebo_pctile": pl["pctile"],
            }
        )
    diag_trade = pd.DataFrame(dg_rows)
    diag_trade.to_csv(OUT / "13_t3_retransformation_trade.csv", index=False)
    print(diag_trade.round(4).to_string(index=False))

    # -------------------------------------------------------- 9. causality
    hdr("9. CAUSALITY - nothing decided at 15:30 may read the bar it trades")
    viol_tv = viol_f3 = viol_pos = 0
    cut_rows = []
    dates = df["date"].to_numpy()
    close_mask = (df["mins"] == 16 * 60).to_numpy()
    cut_days = list(
        pd.Series(t3days).iloc[
            np.linspace(0, len(t3days) - 1, N_CUTS).round().astype(int)
        ]
    )
    for d in cut_days:
        pert = df.copy()
        # everything at or after day d's 16:00 stamp: the traded bar itself and
        # every later row. Nothing at or before 15:30 on day d is touched.
        after = (dates > np.datetime64(d)) | (close_mask & (dates == np.datetime64(d)))
        pert.loc[after, "rv_raw"] = pert.loc[after, "rv_raw"] * 3.0
        pert.loc[after, "sumret"] = pert.loc[after, "sumret"] * 3.0
        pert["r2"] = pert["sumret"] ** 2
        tv_p = recalibrate(pert, rth, "r2")
        cr = close_mask & ~df["early_close"].to_numpy(bool)
        tvp = pd.Series(tv_p["f"].to_numpy()[cr], index=df.loc[cr, "date"].to_numpy())
        tvp = tvp[~tvp.index.duplicated()]
        d_tv = abs(float(tvp.loc[d]) - float(px.loc[d, "tv_hat"]))
        px_p = px.copy()
        px_p["tv_hat"] = tvp.reindex(px_p.index)
        feat_p = terminal_features(pert, days)
        Xp = X.copy()
        Xp["log_tv_hat"] = np.log(px_p["tv_hat"].astype(float))
        Xp["abs_ret_1500_1530"] = feat_p["abs_last"]
        Xp["net_ret_1000_1530"] = feat_p["net_ret"]
        Xp["log_rv_sofar_rel"] = feat_p["log_rv_sofar_rel"]
        y3p = np.log(
            pd.Series(
                pert.loc[cr, "r2"].to_numpy(), index=pert.loc[cr, "date"].to_numpy()
            )
            .groupby(level=0)
            .first()
            .reindex(days)
            .astype(float)
        )
        fit_p, _, s2p, _ = expanding_lr(y3p, Xp, T3_MIN_DAYS)
        f3p = np.exp(fit_p + 0.5 * s2p)
        d_f3 = abs(float(f3p.loc[d]) - float(f3.loc[d]))
        q0 = float(np.where(f3.loc[d] > px.loc[d, "iv_var"], 1.0, -1.0))
        q1 = float(np.where(f3p.loc[d] > px.loc[d, "iv_var"], 1.0, -1.0))
        viol_tv += int(d_tv > 0)
        viol_f3 += int(d_f3 > 0)
        viol_pos += int(q0 != q1)
        cut_rows.append(
            {
                "cut": str(pd.Timestamp(d).date()),
                "d_tv_hat": d_tv,
                "d_f3": d_f3,
                "pos_moved": float(q0 != q1),
            }
        )
    caus = pd.DataFrame(cut_rows)
    caus.to_csv(OUT / "13_causality.csv", index=False)
    print(caus.to_string(index=False))
    print(
        f"tv_hat moved on {viol_tv} of {N_CUTS} cut days; the T3 forecast on "
        f"{viol_f3} of {N_CUTS}; the T3 position on {viol_pos} of {N_CUTS}."
    )
    if viol_tv or viol_f3 or viol_pos:
        raise SystemExit("CAUSALITY ASSERTION FAILED")
    print("CAUSALITY ASSERTION PASSED.")

    # ----------------------------------------------------- 10. all eight tags
    hdr("10. ALL EIGHT FORECASTS (secondary)")
    all_rows = []
    for tag in asl.MODEL_ORDER:
        bk = books[tag]
        p = bk["px"].copy()
        p["ep_tv"] = package_expected_payoff(p["S"], p["K_c"], p["K_p"], p["tv_hat"])
        p["ep_rv"] = package_expected_payoff(p["S"], p["K_c"], p["K_p"], p["rv_hat"])
        dd = pd.DatetimeIndex(p.index)
        ft = terminal_features(bk["df"], dd)
        cf = asl.fomc_and_monthend(dd, REPO)
        pin = (
            p["S"].astype(float)
            - STRIKE_GRID * np.round(p["S"].astype(float) / STRIKE_GRID)
        ).abs() / (p["S"].astype(float) * np.sqrt(p["iv_var"].astype(float)))
        Xt = pd.DataFrame(
            {
                "log_tv_hat": np.log(p["tv_hat"].astype(float)),
                "abs_ret_1500_1530": ft["abs_last"],
                "net_ret_1000_1530": ft["net_ret"],
                "log_rv_sofar_rel": ft["log_rv_sofar_rel"],
                "pin_frac_implied_move": pin,
                "event_fomc_or_monthend": cf["is_event"].fillna(False).astype(float),
            },
            index=dd,
        )
        yt = np.log(p["r2"].astype(float))
        fit_t, _, s2t, _ = expanding_lr(yt, Xt, T3_MIN_DAYS)
        qt = cell_positions(p, {"f3": np.exp(fit_t + 0.5 * s2t)})
        qcal = qlike(p["tv_hat"], p["r2"])
        rcal = qlike(p["rv_hat"], p["r2"])
        dmt, _ = dm_t(qcal, rcal)
        for fill in FILLS:
            base = returns_for(p, qt[CELLS[0]], fill)
            for c in CELLS:
                r = returns_for(p, qt[c], fill)
                idx = r.index if c != CELLS[6] else r.index[qt[c] != 0]
                st = paired_stats(base.loc[idx], r.loc[idx]) if c != CELLS[0] else {}
                all_rows.append(
                    {
                        "tag": tag,
                        "cell": c,
                        "fill": fill,
                        "n": float(len(idx)),
                        "Sharpe": sharpe(r.loc[idx]),
                        "t": safe_t(r.loc[idx]),
                        "pct_buy": float(100.0 * (qt[c].loc[idx] > 0).mean()),
                        "days_differ_from_T0": float(
                            (qt[c].loc[idx] != qt[CELLS[0]].loc[idx]).sum()
                        ),
                        "dSharpe_vs_T0": st.get("dSharpe", 0.0),
                        "pctile_lo": st.get("pctile_lo", float("nan")),
                        "pctile_hi": st.get("pctile_hi", float("nan")),
                        "QLIKE tv_hat": float(qcal.mean()),
                        "QLIKE rv_hat": float(rcal.mean()),
                        "DM t (tv - rv)": float(dmt),
                    }
                )
        tick(t0, f"tag {tag} scored")
    all_tags = pd.DataFrame(all_rows)
    all_tags.to_csv(OUT / "13_all_tags.csv", index=False)
    print(
        all_tags[all_tags["cell"].isin([CELLS[1], CELLS[6]])]
        .round(4)
        .to_string(index=False)
    )
    print("\ncalibration across the eight forecasts (traded bars):")
    print(
        all_tags.drop_duplicates("tag")[
            ["tag", "QLIKE tv_hat", "QLIKE rv_hat", "DM t (tv - rv)"]
        ]
        .round(6)
        .to_string(index=False)
    )

    # ------------------------------------------------------------ 11. figures
    hdr("11. FIGURES")
    q_cal = {c: float("nan") for c in CELLS}
    q_cal[CELLS[0]] = float(qlike(px["rv_hat"], px["r2"]).mean())
    q_cal[CELLS[1]] = float(qlike(px["tv_hat"], px["r2"]).mean())
    q_cal[CELLS[2]] = q_cal[CELLS[3]] = q_cal[CELLS[1]]
    q_cal[CELLS[4]] = q_cal[CELLS[5]] = q_cal[CELLS[0]]
    q_cal[CELLS[6]] = float(qlike(f3, px["r2"]).mean())
    print("pooled QLIKE against r^2 by cell (the choice of the figure's second line):")
    for c in CELLS:
        print(f"  {c:30s} {q_cal[c]:.6f}")
    # SELECTION RULE, stated before the numbers were read: the figure's second
    # line is the cell other than T0 with the lowest pooled QLIKE against r^2;
    # cells sharing a forecast tie exactly, and a tie is broken toward the cell
    # whose position differs from T0 on the most days - never on the return.
    n_diff_map = {c: int((q_all[c] != q_all[CELLS[0]]).sum()) for c in CELLS}
    best = min(CELLS[1:], key=lambda c: (q_cal[c], -n_diff_map[c]))
    print(
        f"best-calibrated cell (chosen on QLIKE, not on Sharpe; ties broken on "
        f"days differing from T0): {best}"
    )

    daily = pd.DataFrame(
        {
            f"{c} | {fill}": ptss[(c, fill)]
            for c in (CELLS[0], best, CELLS[1])
            for fill in FILLS
        }
    )
    daily.to_csv(OUT / "13_daily_pnl_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=True)
    for ax, fill in zip(axes, FILLS, strict=True):
        ax.plot(
            px.index,
            ptss[(CELLS[0], fill)].cumsum(),
            lw=1.3,
            label=f"T0 (control) - {CELLS[0]}",
        )
        ax.plot(
            px.index,
            ptss[(best, fill)].cumsum(),
            lw=1.3,
            label=f"best-calibrated cell (on QLIKE): {best}",
        )
        if best != CELLS[1]:
            ax.plot(
                px.index,
                ptss[(CELLS[1], fill)].cumsum(),
                lw=1.1,
                ls="--",
                color="0.35",
                label=f"{CELLS[1]} (the terminal-variance map)",
            )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(f"{fill} fills", fontsize=10)
        ax.set_ylabel("cumulative index points per straddle", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=8)
    fig.suptitle(
        "Proposal 13: cumulative points, T0 against the best-calibrated cell "
        "(chosen on QLIKE against r^2, not on Sharpe)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "13_cum_points.png", dpi=120, bbox_inches="tight")
    print("saved", OUT / "13_cum_points.png")

    fig2, ax = plt.subplots(figsize=(7.6, 3.6))
    bc = by_clock.reset_index()
    ax.bar(bc["hhmm"], bc["ratio_of_means"], color="#4C72B0")
    ax.axhline(1.0, color="0.3", lw=1.0, ls="--")
    ax.set_ylabel("mean(RV) / mean(r$^2$)", fontsize=9)
    ax.set_xlabel("bar-end stamp", fontsize=9)
    ax.set_title(
        "Realized variance against the squared terminal move, by clock "
        f"(all {len(ses)} session bars)",
        fontsize=10,
    )
    ax.tick_params(labelsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "13_bias_by_clock.png", dpi=120, bbox_inches="tight")
    print("saved", OUT / "13_bias_by_clock.png")

    tick(t0, "done")


if __name__ == "__main__":
    main()
