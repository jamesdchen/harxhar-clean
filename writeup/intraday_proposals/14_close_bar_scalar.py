"""Proposal 14 - the close-bar scalar: one trailing number between RV and r^2.

A PRE-REGISTERED ONE-CELL study. The cell below was fixed before any number
was seen. Nothing is selected from a grid; there is exactly one window, one
scalar and one position rule.

THE FACT (report 13). On the traded 16:00-stamp bars mean(RV) / mean(r^2) is
0.68 - RV is the realized variance, the sum of squared one-minute returns and
the target the forecast is calibrated to; r^2 is the squared thirty-minute
terminal return, what the settled straddle actually pays on. Mid-day bars run
1.05-1.20 and the 16:00 stamp is below 1 in 19 of 24 years. So rv_hat is biased
LOW as a forecast of the close bar's terminal variance by roughly a third, and
the sign rule compares the implied - which prices the terminal move - against
it.

THE ONE CELL.

  c_t   the trailing ratio of means on prior 16:00-stamp session bars,
        c_t = mean(r^2 over the previous 250 sessions' close bars)
            / mean(RV over the same bars),
        computed with prior sessions only (the 250-session window is shifted by
        one session; a minimum of 250 sessions is required). Before that the
        cell sits flat, q = 0, and those days are reported. One scalar a day,
        low-noise by construction.

  tv_hat_t = c_t * rv_hat_t, the deck's recalibrated forecast rescaled and
        otherwise unchanged.

  q_t   = sign(tv_hat_t - iv_var_t): +1 long, -1 short, and s = 0 is short.
        The 15:30 nearest-OTM package, cash-settled - exactly the deck's trade.

COMPARATORS. T0, the deck's rule q = sign(rv_hat - iv_var), and always short.
The primary forecast is blk2, the block-diagonal ridge; all eight tags in
asl.MODEL_ORDER are carried as secondary rows in one CSV.

FILLS. The midpoint case enters at the quoted midpoint. The crossed case pays
the touch at entry - long pays the ask, short receives the bid - and cash
settlement pays no exit spread (asl.crossed_premium_return).

STANDING RULE. Nothing is adopted unless it beats T0 at the crossed spread with
an interval excluding zero AND calibrates better against r^2.

NO VARIANTS. One window, no per-clock version, no shrinkage. Nothing was added
after a number was seen.

Outputs: CSV and PNG under results/atm_straddle_intraday/proposals/14/.
Every number in 14_close_bar_scalar.md is printed by this script.

Run:  python writeup/intraday_proposals/14_close_bar_scalar.py
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
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "14"

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
C_WINDOW = 250  # prior 16:00-stamp sessions in the scalar's window
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
FILLS = ("mid", "crossed")
CLOSE_MINS = 16 * 60

CELL = "cell  sign(c * rv_hat - iv)"
T0 = "T0    sign(rv_hat - iv)"
AS = "always short"
RULES = [CELL, T0, AS]

# The deck's own rule table for the block-diagonal ridge on its 866 days.
GATE = {
    "sign_mean": 0.094736,
    "sign_t": 2.480957,
    "sign_sharpe": 1.338322,
    "as_sharpe": 0.203779,
}
GATE_TOL = 1e-6


def tick(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def hdr(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


# ------------------------------------------------------------------- panel


def load_core() -> pd.DataFrame:
    core = pd.read_parquet(CORE_STATS)
    core["et_naive"] = pd.to_datetime(core["endbartime"])
    return core[["et_naive", "sumret", "sumret2"]]


def build_panel(path: Path, core: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The forecast panel with the terminal move attached, plus the fit mask."""
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


def close_bars(df: pd.DataFrame, rth: np.ndarray) -> pd.DataFrame:
    """One row per 16:00-stamp session bar: the close bar's RV and r^2."""
    m = rth & (df["mins"].to_numpy() == CLOSE_MINS)
    cb = df.loc[m, ["date", "rv_raw", "r2", "sumret"]].copy()
    cb = cb.set_index("date").sort_index()
    if cb.index.has_duplicates:
        raise ValueError("duplicate 16:00 stamps in the session close bars")
    return cb


def recalibrate(df: pd.DataFrame, rth: np.ndarray) -> pd.Series:
    """The library's weighted Mincer-Zarnowitz map on the RV target (the deck's)."""
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    f, _m, _s2 = asl.second_order_mz(
        df["yhat"].to_numpy(float),
        df["rv_raw"].to_numpy(float),
        df["baseline"].to_numpy(float),
        day_codes,
        len(uniq),
        need_days=None,
        fit_mask=rth,
        method="mean",
    )
    out = pd.Series(f, index=df.index)
    out[df["early_close"].to_numpy(bool)] = np.nan
    return out


# ------------------------------------------------------------- THE SCALAR


def close_bar_scalar(cb: pd.DataFrame, window: int = C_WINDOW) -> pd.Series:
    """c_t = mean(r^2) / mean(RV) over the previous `window` close bars.

    Prior sessions only: the rolling window is closed at t-1 by the shift, and
    a full `window` of prior sessions is required. This is the ONLY window in
    the study; no other was run.
    """
    r2 = cb["r2"].rolling(window, min_periods=window).mean().shift(1)
    rv = cb["rv_raw"].rolling(window, min_periods=window).mean().shift(1)
    return (r2 / rv.where(rv > 0)).rename("c")


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


# ----------------------------------------------------------------- the book


def positions(px: pd.DataFrame) -> pd.DataFrame:
    """The cell, the control and always short, one column each."""
    iv = px["iv_var"].astype(float)
    tv = px["tv_hat"].astype(float)
    q = pd.DataFrame(index=px.index)
    q[CELL] = np.where(tv.isna(), 0.0, np.where(tv > iv, 1.0, -1.0))
    q[T0] = np.where(px["rv_hat"].astype(float) > iv, 1.0, -1.0)
    q[AS] = -1.0
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


def build_book(tag: str, cb: pd.DataFrame, c: pd.Series) -> pd.DataFrame:
    """The deck's day frame for one forecast, with c_t and tv_hat attached."""
    deck = pd.read_parquet(DECK_DIR / f"daily_{tag}.parquet")
    px = deck.join(cb[["rv_raw", "r2", "sumret"]], how="left")
    px["c"] = c.reindex(px.index)
    px["tv_hat"] = px["c"] * px["rv_hat"].astype(float)
    px["bid_entry"] = px["bid_c"].astype(float) + px["bid_p"].astype(float)
    px["ask_entry"] = px["ask_c"].astype(float) + px["ask_p"].astype(float)
    if int(px["r2"].isna().sum()):
        raise ValueError(f"{tag}: the close bar is missing on some traded day")
    return px


# ==================================================================== main


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    print(f"proposal 14 - the close-bar scalar.  repo {REPO}")
    print(
        "PRE-REGISTERED: one cell, one window, one scalar. Fixed before any "
        "number was seen; no variant was run."
    )

    core = load_core()
    tick(t0, f"core_stats loaded ({len(core)} rows)")
    df, rth = build_panel(asl.yhat_paths(REPO)[PRIMARY], core)
    cb = close_bars(df, rth)
    tick(t0, f"panel built ({len(df)} rows, {len(cb)} session close bars)")

    # the close bar's RV and r^2 are the tape, not the forecast: identical on
    # every forecast panel, which is what lets one scalar serve all eight tags
    for tag in asl.MODEL_ORDER:
        if tag == PRIMARY:
            continue
        d2, r2m = build_panel(asl.yhat_paths(REPO)[tag], core)
        cb2 = close_bars(d2, r2m)
        if not cb2.index.equals(cb.index):
            raise ValueError(f"{tag}: session close bars differ from {PRIMARY}")
        gap = float((cb2["rv_raw"] - cb["rv_raw"]).abs().max()) + float(
            (cb2["r2"] - cb["r2"]).abs().max()
        )
        if gap != 0.0:
            raise ValueError(f"{tag}: close-bar tape differs by {gap}")
    tick(t0, "close-bar tape identical on all eight forecast panels")

    c = close_bar_scalar(cb)
    px = build_book(PRIMARY, cb, c)

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
    rv_recon = recalibrate(df, rth)
    close_row = ((df["mins"] == CLOSE_MINS) & ~df["early_close"]).to_numpy(bool)
    recon = pd.Series(
        rv_recon.to_numpy()[close_row],
        index=df.loc[close_row, "date"].to_numpy(),
    )
    recon = recon[~recon.index.duplicated()].reindex(px.index)
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
    d_recon = float((px["rv_hat"] - recon).abs().max())
    d_pos = float(
        (px["pos"] - np.where(px["rv_hat"] > px["iv_var"], 1.0, -1.0)).abs().sum()
    )
    gate_rows.append(
        {
            "figure": "rv_hat rebuilt from the panel",
            "target": 0.0,
            "reproduced": d_recon,
            "abs_diff": d_recon,
        }
    )
    gate_rows.append(
        {
            "figure": "deck pos vs sign(rv_hat - iv_var)",
            "target": 0.0,
            "reproduced": d_pos,
            "abs_diff": d_pos,
        }
    )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT / "14_gate.csv", index=False)
    print(gate.to_string(index=False))
    worst = float(gate["abs_diff"].max())
    print(f"n days {len(px)};  worst gate difference {worst:.3e}")
    if worst > GATE_TOL:
        raise SystemExit(f"GATE FAILED: worst difference {worst:.3e} > {GATE_TOL}")
    print("GATE PASSED.")

    # -------------------------------------------------------------- 1. the fact
    hdr("1. THE FACT - mean(RV) / mean(r^2) at the bar the trade owns")
    ses = df[rth].copy()
    ses["hhmm"] = ses["et"].dt.strftime("%H:%M")
    by_clock = ses.groupby("hhmm").apply(
        lambda x: pd.Series(
            {
                "n": float(len(x)),
                "ratio_of_means": float(x["rv_raw"].mean() / x["r2"].mean()),
            }
        ),
        include_groups=False,
    )
    close_ratio = float(cb["rv_raw"].mean() / cb["r2"].mean())
    traded_ratio = float(px["rv_raw"].mean() / px["r2"].mean())
    day = ses[ses["hhmm"] != "16:00"]
    day_ratio = float(day["rv_raw"].mean() / day["r2"].mean())
    yr = cb.groupby(cb.index.year).apply(
        lambda x: float(x["rv_raw"].mean() / x["r2"].mean())
    )
    fact = pd.DataFrame(
        [
            {
                "frame": "all session bars, 11 daytime stamps",
                "n": float(len(day)),
                "mean_RV_over_mean_r2": day_ratio,
            },
            {
                "frame": "the 16:00 stamp, all sessions",
                "n": float(len(cb)),
                "mean_RV_over_mean_r2": close_ratio,
            },
            {
                "frame": "the 16:00 stamp, the traded days",
                "n": float(len(px)),
                "mean_RV_over_mean_r2": traded_ratio,
            },
        ]
    )
    fact["implied_c"] = 1.0 / fact["mean_RV_over_mean_r2"]
    fact.to_csv(OUT / "14_fact.csv", index=False)
    print(by_clock.round(4).to_string())
    print()
    print(fact.round(6).to_string(index=False))
    print(
        f"\nthe 16:00 stamp is below 1 in {int((yr < 1).sum())} of {len(yr)} years "
        f"({int(yr.index.min())}-{int(yr.index.max())}); daytime stamps range "
        f"{float(by_clock.drop(index='16:00')['ratio_of_means'].min()):.3f}-"
        f"{float(by_clock.drop(index='16:00')['ratio_of_means'].max()):.3f}."
    )

    # ---------------------------------------------------------- 2. the scalar
    hdr("2. THE SCALAR c_t - description, not a cell")
    c_px = px["c"]
    n_flat = int(c_px.isna().sum())
    print(
        f"c_t needs {C_WINDOW} prior 16:00-stamp sessions. The first traded day "
        f"sits at session {int(cb.index.get_indexer(px.index[:1])[0]) + 1} of "
        f"{len(cb)}, so the warm-up is already spent:\n"
        f"  days with c_t: {int(c_px.notna().sum())} of {len(px)};  days flat "
        f"(q = 0): {n_flat}."
    )
    c_year = (
        c_px.groupby(c_px.index.year)
        .agg(n="size", mean="mean", min="min", max="max")
        .rename_axis("year")
    )
    c_year["full_sample_c"] = 1.0 / traded_ratio
    c_year.to_csv(OUT / "14_c_by_year.csv")
    print("\nc_t by year (traded days):")
    print(c_year.round(4).to_string())
    c_all = pd.DataFrame({"c": c}).dropna()
    c_all.to_csv(OUT / "14_c_series.csv")
    print(
        f"\nc_t over the traded days: mean {float(c_px.mean()):.4f}, min "
        f"{float(c_px.min()):.4f}, max {float(c_px.max()):.4f}, sd "
        f"{float(c_px.std()):.4f}; over all {len(c_all)} sessions with a value: "
        f"mean {float(c_all['c'].mean()):.4f}, min {float(c_all['c'].min()):.4f}, "
        f"max {float(c_all['c'].max()):.4f}."
    )
    print(
        f"the full-sample scalar the fact implies is 1 / {traded_ratio:.4f} = "
        f"{1.0 / traded_ratio:.4f} on the traded bars and 1 / {close_ratio:.4f} = "
        f"{1.0 / close_ratio:.4f} on all 16:00 stamps; c_t is above 1 on "
        f"{float(100.0 * (c_px > 1).mean()):.1f}% of traded days."
    )

    # ------------------------------------------------------------- 3. the cell
    hdr("3. THE CELL - what the scalar does to the position")
    q_all = positions(px)
    q_all.to_csv(OUT / "14_positions.csv")
    qc, q0 = q_all[CELL], q_all[T0]
    n_diff = int((qc != q0).sum())
    short_to_long = int(((q0 < 0) & (qc > 0)).sum())
    long_to_short = int(((q0 > 0) & (qc < 0)).sum())
    n_short0 = int((q0 < 0).sum())
    n_long0 = int((q0 > 0).sum())
    flips = pd.DataFrame(
        [
            {
                "n_days": float(len(px)),
                "T0 long": float(n_long0),
                "T0 short": float(n_short0),
                "cell long": float((qc > 0).sum()),
                "cell short": float((qc < 0).sum()),
                "cell flat": float((qc == 0).sum()),
                "days differ": float(n_diff),
                "pct differ": 100.0 * n_diff / len(px),
                "short -> long": float(short_to_long),
                "pct of T0 shorts": 100.0 * short_to_long / max(n_short0, 1),
                "long -> short": float(long_to_short),
            }
        ]
    )
    flips.to_csv(OUT / "14_flips.csv", index=False)
    print(flips.T.round(4).to_string(header=False))
    n_untrade = {
        r: asl.crossed_untradeable_count(q_all[r], px["bid_entry"], px["ask_entry"])
        for r in RULES
    }
    print("rows a crossed fill cannot price:", n_untrade)

    # ------------------------------------------------------- 4. calibration
    hdr("4. CALIBRATION against r^2 on the traded bars")
    cal_rows = []
    targets = (
        ("r^2, the terminal variance", px["r2"]),
        ("RV, reference", px["rv_raw"]),
    )
    for label, y in targets:
        q_rv = qlike(px["rv_hat"], y)
        q_tv = qlike(px["tv_hat"], y)
        both = q_rv.notna() & q_tv.notna()
        t, lag = dm_t(q_tv[both], q_rv[both])
        cal_rows.append(
            {
                "target": label,
                "n_scored": float(both.sum()),
                "QLIKE rv_hat": float(q_rv[both].mean()),
                "QLIKE tv_hat": float(q_tv[both].mean()),
                "DM t (tv - rv)": float(t),
                "DM lag": float(lag),
                "mean rv_hat / mean y": float(
                    px.loc[both, "rv_hat"].mean() / y[both].mean()
                ),
                "mean tv_hat / mean y": float(
                    px.loc[both, "tv_hat"].mean() / y[both].mean()
                ),
            }
        )
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "14_calibration.csv", index=False)
    print(
        "QLIKE = y/f - log(y/f) - 1. A NEGATIVE Diebold-Mariano t favours the\n"
        "rescaled forecast; the t is autocorrelation-robust at lag "
        "floor(1.5 n^(1/3))."
    )
    print(cal.round(6).to_string(index=False))

    # ---------------------------------------------------------- 5. the oracle
    hdr("5. THE ORACLE - what the scalar can do at best")
    iv = px["iv_var"].astype(float)
    o_rv = pd.Series(np.where(px["rv_raw"] > iv, 1.0, -1.0), index=px.index)
    o_crv = pd.Series(np.where(px["c"] * px["rv_raw"] > iv, 1.0, -1.0), index=px.index)
    o_r2 = pd.Series(np.where(px["r2"] > iv, 1.0, -1.0), index=px.index)
    sgn_r = np.sign(px["R"].astype(float))
    orows = []
    for nm, q in (
        ("sign(RV - iv_var)", o_rv),
        ("sign(c * RV - iv_var)", o_crv),
        ("sign(r^2 - iv_var)", o_r2),
    ):
        orows.append(
            {
                "oracle": nm,
                "n": float(len(q)),
                "pct_long": float(100.0 * (q > 0).mean()),
                "agree with sign(r^2 - iv) %": float(100.0 * (q == o_r2).mean()),
                "corr with sign(R)": float(np.corrcoef(q, sgn_r)[0, 1]),
                "hit rate q R > 0": float((q * px["R"] > 0).mean()),
                "Sharpe mid": sharpe(q * px["R"]),
                "Sharpe crossed": sharpe(
                    asl.crossed_premium_return(
                        q, px["exit"], px["bid_entry"], px["ask_entry"]
                    )
                ),
            }
        )
    orc = pd.DataFrame(orows)
    orc.to_csv(OUT / "14_oracles.csv", index=False)
    print("Neither is a trade: both read the bar they price.")
    print(orc.round(4).to_string(index=False))

    # ------------------------------------------------------ 6. the rule tables
    hdr("6. THE RULE TABLES at both fills")
    rets: dict[tuple[str, str], pd.Series] = {}
    ptss: dict[tuple[str, str], pd.Series] = {}
    frames = {"all traded days": px.index, "days with c_t": px.index[c_px.notna()]}
    tab_rows = []
    for fill in FILLS:
        for r in RULES:
            rets[(r, fill)] = returns_for(px, q_all[r], fill)
            ptss[(r, fill)] = points_for(px, q_all[r], fill)
    for fname, idx in frames.items():
        for fill in FILLS:
            for r in RULES:
                row = asl.rule_row(rets[(r, fill)].loc[idx], q_all[r].loc[idx])
                tab_rows.append(
                    {
                        "frame": fname,
                        "rule": r,
                        "fill": fill,
                        "n": float(row["n"]),
                        "mean": float(row["mean"]),
                        "t": float(row["t_mean"]),
                        "Sharpe": float(row["Sharpe_ann"]),
                        "pct_buy": float(row["pct_buy"]),
                        "n_flat": float((q_all[r].loc[idx] == 0).sum()),
                        "days_differ_from_T0": float(
                            (q_all[r].loc[idx] != q_all[T0].loc[idx]).sum()
                        ),
                        "hit_rate": float(
                            (q_all[r].loc[idx] * px.loc[idx, "R"] > 0).mean()
                        ),
                        "mean_pts": float(ptss[(r, fill)].loc[idx].mean()),
                    }
                )
    rules_tab = pd.DataFrame(tab_rows)
    rules_tab.to_csv(OUT / "14_rules.csv", index=False)
    for fname in frames:
        for fill in FILLS:
            sel = (rules_tab["frame"] == fname) & (rules_tab["fill"] == fill)
            print(f"\n--- {fname}, {fill} fills ---")
            print(
                rules_tab[sel]
                .drop(columns=["frame", "fill"])
                .round(6)
                .to_string(index=False)
            )

    # ------------------------------------------------- 7. paired differences
    hdr("7. PAIRED DIFFERENCE, cell - T0")
    pr_rows = []
    for fname, idx in frames.items():
        for fill in FILLS:
            st = paired_stats(rets[(T0, fill)].loc[idx], rets[(CELL, fill)].loc[idx])
            pr_rows.append({"frame": fname, "fill": fill, **st})
    paired = pd.DataFrame(pr_rows)
    paired.to_csv(OUT / "14_paired.csv", index=False)
    print(
        "cell - T0, daily. HAC t at lag floor(1.5 n^(1/3)); Sharpe difference "
        f"by circular block bootstrap,\nblock {BOOT_BLOCK}, B = {BOOT_B}, "
        f"rng({SEED}), draws shared."
    )
    print(paired.round(4).to_string(index=False))

    # ------------------------------------------------------------ 8. placebo
    hdr("8. PLACEBO - random signs at the cell's own long rate")
    long_r = px["exit"] / px["ask_entry"] - 1.0
    short_r = 1.0 - px["exit"] / px["bid_entry"]
    long_m = px["R"].astype(float)
    short_m = -px["R"].astype(float)
    legs = {"mid": (long_m, short_m), "crossed": (long_r, short_r)}
    pl_rows = []
    for fill in FILLS:
        lr_, sr_ = legs[fill]
        pl_rows.append(
            {
                "cell": CELL,
                "fill": fill,
                **placebo_rate_matched(lr_, sr_, q_all[CELL]),
            }
        )
    placebo = pd.DataFrame(pl_rows)
    placebo.to_csv(OUT / "14_placebo.csv", index=False)
    print(f"{PLACEBO_DRAWS} draws, rng({SEED}), rate-matched to the cell's sign.")
    print(placebo.round(4).to_string(index=False))

    # -------------------------------------------------------- 9. causality
    hdr("9. CAUSALITY - day d's own close bar may not move day d's scalar")
    cut_days = list(
        pd.Series(px.index).iloc[
            np.linspace(0, len(px) - 1, N_CUTS).round().astype(int)
        ]
    )
    viol_c = viol_q = 0
    later_moved = n_with_later = 0
    cut_rows = []
    cb_idx = cb.index
    for d in cut_days:
        pert = cb.copy()
        after = cb_idx >= d  # day d's own close bar and every later one
        pert.loc[after, "rv_raw"] = pert.loc[after, "rv_raw"] * 3.0
        pert.loc[after, "sumret"] = pert.loc[after, "sumret"] * 3.0
        pert["r2"] = pert["sumret"] ** 2
        c_p = close_bar_scalar(pert)
        d_c = abs(float(c_p.loc[d]) - float(c.loc[d]))
        tv_p = float(c_p.loc[d]) * float(px.loc[d, "rv_hat"])
        q_p = float(np.where(tv_p > float(px.loc[d, "iv_var"]), 1.0, -1.0))
        q_r = float(q_all.loc[d, CELL])
        nxt = c_p.index[c_p.index > d]
        d_next = (
            float((c_p.loc[nxt] - c.loc[nxt]).abs().max()) if len(nxt) else float("nan")
        )
        viol_c += int(d_c > 0)
        viol_q += int(q_p != q_r)
        # the last cut is the panel's final session and has no later sessions
        n_with_later += int(len(nxt) > 0)
        later_moved += int(len(nxt) > 0 and d_next > 0)
        cut_rows.append(
            {
                "cut": str(pd.Timestamp(d).date()),
                "d_c_on_the_cut_day": d_c,
                "pos_moved_on_the_cut_day": float(q_p != q_r),
                "max_d_c_on_later_sessions": d_next,
            }
        )
    caus = pd.DataFrame(cut_rows)
    caus.to_csv(OUT / "14_causality.csv", index=False)
    print(
        "Day d's close-bar RV and terminal return are tripled, together with "
        "every later\nsession's, and the scalar is rebuilt."
    )
    print(caus.to_string(index=False))
    print(
        f"c_t moved on the cut day in {viol_c} of {N_CUTS} cuts; the cell's "
        f"position in {viol_q} of {N_CUTS};\nlater sessions moved in "
        f"{later_moved} of the {n_with_later} cuts that have later sessions, "
        "so the perturbation has teeth."
    )
    if viol_c or viol_q:
        raise SystemExit("CAUSALITY ASSERTION FAILED")
    if later_moved != n_with_later:
        raise SystemExit("CAUSALITY TEST IS TOOTHLESS: no later session moved")
    print("CAUSALITY ASSERTION PASSED.")

    # ----------------------------------------------------- 10. all eight tags
    hdr("10. ALL EIGHT FORECASTS (secondary)")
    all_rows = []
    for tag in asl.MODEL_ORDER:
        p = build_book(tag, cb, c)
        qt = positions(p)
        q_rv_t = qlike(p["rv_hat"], p["r2"])
        q_tv_t = qlike(p["tv_hat"], p["r2"])
        both = q_rv_t.notna() & q_tv_t.notna()
        dmt, dml = dm_t(q_tv_t[both], q_rv_t[both])
        for fill in FILLS:
            base = returns_for(p, qt[T0], fill)
            for r in RULES:
                rr = returns_for(p, qt[r], fill)
                st = paired_stats(base, rr) if r != T0 else {}
                all_rows.append(
                    {
                        "tag": tag,
                        "rule": r,
                        "fill": fill,
                        "n": float(rr.notna().sum()),
                        "mean": float(rr.mean()),
                        "t": safe_t(rr),
                        "Sharpe": sharpe(rr),
                        "pct_buy": float(100.0 * (qt[r] > 0).mean()),
                        "days_differ_from_T0": float((qt[r] != qt[T0]).sum()),
                        "dSharpe_vs_T0": st.get("dSharpe", 0.0),
                        "pctile_lo": st.get("pctile_lo", float("nan")),
                        "pctile_hi": st.get("pctile_hi", float("nan")),
                        "basic_lo": st.get("basic_lo", float("nan")),
                        "basic_hi": st.get("basic_hi", float("nan")),
                        "hac_t_diff": st.get("hac_t_diff", float("nan")),
                        "QLIKE rv_hat vs r2": float(q_rv_t[both].mean()),
                        "QLIKE tv_hat vs r2": float(q_tv_t[both].mean()),
                        "DM t (tv - rv)": float(dmt),
                        "DM lag": float(dml),
                    }
                )
    all_tags = pd.DataFrame(all_rows)
    all_tags.to_csv(OUT / "14_all_tags.csv", index=False)
    print("the cell against its control on each forecast, crossed spread:")
    print(
        all_tags[(all_tags["rule"] != AS) & (all_tags["fill"] == "crossed")][
            [
                "tag",
                "rule",
                "Sharpe",
                "pct_buy",
                "days_differ_from_T0",
                "dSharpe_vs_T0",
                "pctile_lo",
                "pctile_hi",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\ncalibration against r^2 across the eight forecasts (traded bars):")
    print(
        all_tags.drop_duplicates("tag")[
            ["tag", "QLIKE rv_hat vs r2", "QLIKE tv_hat vs r2", "DM t (tv - rv)"]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nthe cell's Sharpe at both fills, all eight:")
    print(
        all_tags[all_tags["rule"] == CELL]
        .pivot(index="tag", columns="fill", values="Sharpe")
        .reindex(asl.MODEL_ORDER)
        .round(4)
        .to_string()
    )
    tick(t0, "eight forecasts scored")

    # ------------------------------------------------------------ 11. figure
    hdr("11. FIGURE")
    daily = pd.DataFrame(
        {f"{r} | {fill}": ptss[(r, fill)] for r in RULES for fill in FILLS}
    )
    daily["c"] = c_px
    daily.to_csv(OUT / "14_daily_pnl_points.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for ax, fill in zip(axes[:2], FILLS, strict=True):
        ax.plot(
            px.index,
            ptss[(T0, fill)].cumsum(),
            lw=1.3,
            color="#4C72B0",
            label="T0 (control), sign(rv_hat - iv)",
        )
        ax.plot(
            px.index,
            ptss[(CELL, fill)].cumsum(),
            lw=1.3,
            color="#C44E52",
            label="cell, sign(c * rv_hat - iv)",
        )
        ax.plot(
            px.index,
            ptss[(AS, fill)].cumsum(),
            lw=1.0,
            ls="--",
            color="0.45",
            label="always short",
        )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(f"{fill} fills", fontsize=10)
        ax.set_ylabel("cumulative index points per straddle", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=8)
    ax = axes[2]
    ax.plot(c_all.index, c_all["c"], lw=1.0, color="#55A868")
    ax.axhline(1.0, color="0.3", lw=0.9, ls=":")
    ax.axhline(
        1.0 / traded_ratio,
        color="#C44E52",
        lw=0.9,
        ls="--",
        label=f"full sample on the traded bars {1.0 / traded_ratio:.2f}",
    )
    ax.axvspan(px.index.min(), px.index.max(), color="0.85", alpha=0.6, zorder=0)
    ax.set_title(f"c_t, the {C_WINDOW}-session close-bar scalar", fontsize=10)
    ax.set_ylabel("mean(r$^2$) / mean(RV), prior sessions", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.suptitle(
        "Proposal 14: the close-bar scalar against the deck's rule "
        "(shaded band = the 866 traded days)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "14_cum_points.png", dpi=120, bbox_inches="tight")
    print("saved", OUT / "14_cum_points.png")
    tick(t0, "done")


if __name__ == "__main__":
    main()
