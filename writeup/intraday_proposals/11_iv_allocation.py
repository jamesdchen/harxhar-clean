"""Proposal 11 - how the remaining implied variance is allocated across the day.

A PRE-REGISTERED four-variant study, not a search. The four share rules were
fixed before any of them was scored; nothing below is selected from a grid.

CAVEAT, up front. The intraday deck prices the next 30-minute bar by taking a
SHARE of the remaining-session implied variance:

    IV_rem_t = iv_hourly_t^2 * h_t        (h_t = hours from t to the close)
    slice_t  = IV_rem_t * w_t             (w_t = the share of what is left)

A share reallocates a fixed quantity. It cannot add one bit of day-by-day
information: on any given day it moves variance between clocks and the total
implied is whatever the option market says it is. So the question a better
share can answer is a CALIBRATION question - is the bar-level price right at
each clock - and only after that a trading question. The scoring below is in
that order: calibration by clock first, the sign rule second.

The four variants (all F_t-measurable; every trailing statistic uses prior days
only with shift(1) and a 63-session minimum):

  V0  unconditional.  w = (trailing expanding per-clock mean of realized bar
      variance) / (the remaining-session sum of those means).  This is the
      deck's current share; the gate below reproduces the notebook's rule
      table from it before anything else is reported.

  V1  event-conditional.  Separate trailing per-clock profiles for FOMC
      statement days, month-end sessions, third-Friday / quad-witching
      expirations and all other days, each shrunk toward the unconditional
      profile with weight n_cond / (n_cond + 20).

  V2  market-implied.  On each PRIOR day the forward implied variance of bar t
      is IV_rem_t - IV_rem_{t+1}, floored at zero; the trailing expanding
      per-clock mean of (forward / IV_rem) is the market's own average
      allocation, renormalized so the shares of the remaining clocks sum to one.

  V3  same-day-conditioned.  A per-clock linear regression, fit on prior days
      only, of the realized share of the remaining variance taken by this bar
      on the log of today's cumulative realized variance so far relative to its
      trailing per-clock expectation, the log of iv_hourly_t relative to its
      trailing per-clock mean, and day-of-week indicators; predicted share
      clipped to (0.02, 0.98) and renormalized across the remaining clocks.

At 15:30 there is one bar left, so w = 1 for every variant by construction and
the settlement leg is IDENTICAL across variants. Every difference reported here
lives in the daytime legs.

Fills. The midpoint case uses the cached midpoint return R. The crossed case is
the deck's convention: a long buys at ask_entry and sells at the next stamp's
bid, a short sells at bid_entry and buys back at the next stamp's ask, the
15:30 bar cash-settles at exit_settle with no exit spread, and a re-pick that
lands on the same two strikes with the same sign is a hold, not a round trip.

Outputs: CSVs and one figure under results/atm_straddle_intraday/proposals/11/.
Every number in 11_iv_allocation.md is printed by this script.

Run:  python writeup/intraday_proposals/11_iv_allocation.py
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
import statsmodels.api as sm  # noqa: E402


def _bootstrap_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "notebooks" / "atm_straddle_lib.py").exists():
            return q
    raise FileNotFoundError("repo root not found from " + str(start))


sys.path.insert(0, str(_bootstrap_repo(Path(__file__)) / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
CACHE_DIR = REPO / "results" / "atm_straddle_intraday" / "cache"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "11"

MIN_DAYS = 63  # the deck's standing warm-up for a trailing per-clock statistic
SHRINK_K = 20.0  # V1 shrinkage: weight n / (n + K) on the conditional profile
CLIP_LO, CLIP_HI = 0.02, 0.98  # V3 predicted-share clip
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
PLACEBO_DRAWS = 200
N_CUTS = 10  # perturbation cut points for the causality assertion

# The gate: the intraday notebook's rule table, block-diagonal ridge, 866 days.
GATE = {"always short": 1.9020389, "sign(s)": 1.7205907, "hybrid": 3.1714284}
GATE_TOL = 1e-3

VARIANTS = ["V0", "V1", "V2", "V3"]
VLABEL = {
    "V0": "V0 unconditional",
    "V1": "V1 event-conditional",
    "V2": "V2 market-implied",
    "V3": "V3 same-day-conditioned",
}


# --------------------------------------------------------------------------
# frame
# --------------------------------------------------------------------------
def build_work() -> pd.DataFrame:
    """The intraday notebook's scored frame: 12 clocks x the deck's 866 days.

    The panel is bar-END labelled, so the forecast issued at t for the bar
    [t, t+30] - and that bar's own realized variance - sit on the row stamped
    t+30. The join shifts the panel stamps back one bar.
    """
    caches = sorted(CACHE_DIR.glob("trade_*.parquet"))
    if not caches:
        raise FileNotFoundError(f"no trade cache under {CACHE_DIR}")
    cache = caches[-1]
    pkg = pd.read_parquet(cache)
    pan = asl.load_yhat_panel_mz(asl.yhat_paths(REPO)["blk2"])
    pan["date"] = pan["et"].dt.normalize().dt.tz_localize(None)

    p2 = pan.set_index("t")[["rv_raw", "rv_hat", "in_fit"]].reset_index()
    p2["t"] = pd.to_datetime(p2["t"], utc=True) - pd.Timedelta(minutes=30)
    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    w = pkg.merge(p2, on="t", how="left").dropna(subset=["R", "rv_hat"])
    w["et"] = w["t"].dt.tz_convert("America/New_York")
    w["hhmm"] = w["et"].dt.strftime("%H:%M")
    w["date"] = w["et"].dt.normalize().dt.tz_localize(None)
    assert bool(w["in_fit"].all()), "a joined trade bar is outside the smear's fit mask"

    deck = pd.read_parquet(DECK)
    w = w[w["date"].isin(pd.DatetimeIndex(deck.index))].copy()
    w = w.sort_values(["date", "t"]).reset_index(drop=True)

    # The 09:30-10:00 bar (panel stamp 10:00) is the only realized session bar
    # that is not itself a trade bar; it starts the same-day cumulative sum V3
    # conditions on.
    pre = pan.loc[pan["mins"] == 600, ["date", "rv_raw"]].rename(
        columns={"rv_raw": "rv_pre"}
    )
    w = w.merge(pre, on="date", how="left")

    w["iv_var_raw"] = w["iv_hourly"].astype(float) ** 2
    clocks = sorted(w["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    w["h_rem"] = w["hhmm"].map(n_rem).astype(float) * 0.5
    # the cached touch prices are the two legs' quotes (float32 storage)
    assert bool((w["bid_entry"] - (w["bid_c"] + w["bid_p"])).abs().max() < 1e-4)
    assert bool((w["ask_entry"] - (w["ask_c"] + w["ask_p"])).abs().max() < 1e-4)
    print(
        f"trade cache {cache.name}: bars {len(w)}, days {w['date'].nunique()}, "
        f"clocks {len(clocks)} ({clocks[0]}..{clocks[-1]})"
    )
    return w


def _pivot(w: pd.DataFrame, col: str, clocks: list[str]) -> pd.DataFrame:
    return (
        w.pivot_table(index="date", columns="hhmm", values=col, aggfunc="mean")
        .reindex(columns=clocks)
        .sort_index()
    )


def _share_from_profile(prof: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    """w_t = m_t / sum_{j>=t} m_j: the share of the remaining sum at clock t."""
    rem = prof[clocks[::-1]].cumsum(axis=1)[clocks]
    return prof / rem


def _alloc_from_conditional(p: np.ndarray) -> np.ndarray:
    """Shares-of-remaining p_c -> the allocation a_c = p_c prod_{j<c}(1-p_j)."""
    surv = np.concatenate([[1.0], np.cumprod(1.0 - p)[:-1]])
    return p * surv


# --------------------------------------------------------------------------
# the four shares
# --------------------------------------------------------------------------
def shares_v0(w: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    prof = _pivot(w, "rv_raw", clocks)
    prof_exp = prof.expanding(min_periods=MIN_DAYS).mean().shift(1)
    return _share_from_profile(prof_exp, clocks)


def day_categories(dates: pd.DatetimeIndex) -> pd.Series:
    """FOMC / month-end / third-Friday / other, first match in that order.

    is_fomc is a nullable boolean and is pd.NA beyond the release file's
    knowledge horizon; NA is treated as "other", never as "FOMC".
    """
    flags = asl.fomc_and_monthend(dates, REPO)
    is_fomc = flags["is_fomc"].fillna(False).astype(bool).to_numpy()
    is_me = flags["is_me"].astype(bool).to_numpy()
    d = pd.DatetimeIndex(dates)
    is_tf = (d.dayofweek == 4) & (d.day >= 15) & (d.day <= 21)
    cat = np.where(
        is_fomc,
        "fomc",
        np.where(is_me, "monthend", np.where(is_tf, "thirdfri", "other")),
    )
    return pd.Series(cat, index=dates, name="cat")


def shares_v1(w: pd.DataFrame, clocks: list[str], cat: pd.Series) -> pd.DataFrame:
    prof = _pivot(w, "rv_raw", clocks)
    prof_exp = prof.expanding(min_periods=MIN_DAYS).mean().shift(1)
    c = cat.reindex(prof.index)
    num = pd.DataFrame(np.nan, index=prof.index, columns=prof.columns)
    for name in sorted(pd.unique(c.dropna())):
        mask = (c == name).to_numpy()
        sub = prof.where(pd.Series(mask, index=prof.index), other=np.nan)
        m_cond = sub.expanding(min_periods=1).mean().shift(1)
        n_cond = sub.expanding(min_periods=1).count().shift(1)
        lam = (n_cond / (n_cond + SHRINK_K)).fillna(0.0)
        lam = lam.where(m_cond.notna(), 0.0)
        blend = lam * m_cond.fillna(0.0) + (1.0 - lam) * prof_exp
        num = num.mask(np.broadcast_to(mask[:, None], num.shape), blend)
    num = num.where(prof_exp.notna())  # the unconditional warm-up gates all four
    return _share_from_profile(num, clocks)


def shares_v2(w: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    """The market's own average allocation of the remaining implied variance."""
    ivrem = _pivot(w, "iv_var_raw", clocks) * _pivot(w, "h_rem", clocks)
    nxt = ivrem.shift(-1, axis=1)
    nxt[clocks[-1]] = 0.0  # nothing is left after the settlement bar
    fwd = (ivrem - nxt).clip(lower=0.0)
    ratio = (fwd / ivrem).where(ivrem > 0)
    r_exp = ratio.expanding(min_periods=MIN_DAYS).mean().shift(1)
    a = np.full(r_exp.shape, np.nan)
    ok = r_exp.notna().all(axis=1).to_numpy()
    if ok.any():
        a[ok] = np.apply_along_axis(
            _alloc_from_conditional, 1, r_exp.to_numpy()[ok].clip(0.0, 1.0)
        )
    alloc = pd.DataFrame(a, index=r_exp.index, columns=clocks)
    return _share_from_profile(alloc, clocks)


def _v3_features(w: pd.DataFrame, clocks: list[str]) -> dict[str, pd.DataFrame]:
    """F_t regressors and the prior-day target for the per-clock regressions."""
    rv = _pivot(w, "rv_raw", clocks)
    ivh = _pivot(w, "iv_hourly", clocks)
    pre = w.groupby("date")["rv_pre"].first().reindex(rv.index)

    # Today's cumulative realized variance strictly BEFORE the bar being
    # priced: the 09:30-10:00 bar plus every trade bar already closed.
    cum = rv.cumsum(axis=1).shift(1, axis=1)
    cum[clocks[0]] = 0.0
    cum = cum.add(pre, axis=0)
    cum_exp = cum.expanding(min_periods=MIN_DAYS).mean().shift(1)
    ivh_exp = ivh.expanding(min_periods=MIN_DAYS).mean().shift(1)

    x1 = np.log(cum.where(cum > 0) / cum_exp.where(cum_exp > 0))
    x2 = np.log(ivh.where(ivh > 0) / ivh_exp.where(ivh_exp > 0))
    rv_rem = rv[clocks[::-1]].cumsum(axis=1)[clocks]
    y = (rv / rv_rem).where(rv_rem > 0)
    return {"x1": x1, "x2": x2, "y": y}


def shares_v3(w: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    f = _v3_features(w, clocks)
    x1, x2, y = f["x1"], f["x2"], f["y"]
    dates = pd.DatetimeIndex(x1.index)
    dow = dates.dayofweek.to_numpy()
    dum = np.stack([(dow == k).astype(float) for k in (1, 2, 3, 4)], axis=1)
    n_days, n_cl, p = len(dates), len(clocks), 7

    x = np.empty((n_days, n_cl, p))
    x[:, :, 0] = 1.0
    x[:, :, 1] = x1.to_numpy()
    x[:, :, 2] = x2.to_numpy()
    x[:, :, 3:] = dum[:, None, :]
    yy = y.to_numpy()
    good = np.isfinite(x).all(axis=2) & np.isfinite(yy)

    beta = np.full((n_days, n_cl, p), np.nan)
    for j in range(n_cl):
        g = np.zeros((p, p))
        b = np.zeros(p)
        n = 0
        for i in range(n_days):
            if n >= MIN_DAYS:
                beta[i, j] = np.linalg.lstsq(g, b, rcond=None)[0]
            if good[i, j]:
                xi = x[i, j]
                g += np.outer(xi, xi)
                b += xi * yy[i, j]
                n += 1

    # At clock t the only day state known is the one measured at t, so every
    # remaining clock's regression is evaluated at clock t's regressors.
    pred = np.einsum("itp,ijp->itj", x, beta)  # [day, evaluated at t, clock j]
    pred = np.clip(pred, CLIP_LO, CLIP_HI)
    ok = np.isfinite(pred).all(axis=2) & np.isfinite(x).all(axis=2)

    out = np.full((n_days, n_cl), np.nan)
    for t in range(n_cl):
        tail = pred[:, t, t:]
        alloc_sum = 1.0 - np.prod(1.0 - tail, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = pred[:, t, t] / alloc_sum
        out[:, t] = np.where(ok[:, t] & (alloc_sum > 0), v, np.nan)
    return pd.DataFrame(out, index=x1.index, columns=clocks)


def build_shares(
    w: pd.DataFrame, clocks: list[str], cat: pd.Series
) -> dict[str, pd.DataFrame]:
    return {
        "V0": shares_v0(w, clocks),
        "V1": shares_v1(w, clocks, cat),
        "V2": shares_v2(w, clocks),
        "V3": shares_v3(w, clocks),
    }


def attach_slices(
    w: pd.DataFrame, shares: dict[str, pd.DataFrame], clocks: list[str]
) -> pd.DataFrame:
    mi = pd.MultiIndex.from_arrays([w["date"], w["hhmm"]])
    out = w.copy()
    for name, sh in shares.items():
        out[f"w_{name}"] = sh.stack().reindex(mi).to_numpy()
        out[f"slice_{name}"] = out["iv_var_raw"] * out["h_rem"] * out[f"w_{name}"]
        out[f"s_{name}"] = out["rv_hat"] - out[f"slice_{name}"]
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def _sh(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * np.sqrt(asl.PERIODS_PER_YEAR))


def _t_plain(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(len(x)) * x.mean() / x.std(ddof=1))


def _t_hac(x) -> float:
    """Autocorrelation-robust t of the mean; lag floor(1.5 n^(1/3))."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.allclose(x, x[0]):
        return float("nan")
    lag = int(np.floor(1.5 * len(x) ** (1.0 / 3.0)))
    fit = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(fit.tvalues[0])


def _dd_points(d) -> float:
    c = np.nan_to_num(np.asarray(d, float)).cumsum()
    return float((c - np.maximum.accumulate(c)).min())


def _qlike(rv: np.ndarray, fc: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((fc > 0) & (rv > 0), rv / fc, np.nan)
        return r - np.log(r) - 1.0


def _boot_dsharpe(
    a, b, block: int = BOOT_BLOCK, boot: int = BOOT_B, seed: int = SEED
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = asl.circular_block_bootstrap_idx(rng, len(a), block, boot)
    root = np.sqrt(asl.PERIODS_PER_YEAR)

    def shr(v):
        return v.mean(axis=1) / v.std(axis=1, ddof=1) * root

    d = shr(a[idx]) - shr(b[idx])
    lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
    hat = _sh(a) - _sh(b)
    return {
        "dSharpe": hat,
        "pct_lo": lo,
        "pct_hi": hi,
        "basic_lo": 2 * hat - hi,
        "basic_hi": 2 * hat - lo,
    }


# --------------------------------------------------------------------------
# fills
# --------------------------------------------------------------------------
class Fills:
    """Midpoint and crossed-spread P&L in index points, the deck's convention."""

    def __init__(self, w: pd.DataFrame):
        self.w = w
        self.entry = w["entry"].to_numpy(float)
        self.exit = w["exit"].to_numpy(float)
        self.ret = w["R"].to_numpy(float)
        self.bid_e = w["bid_entry"].to_numpy(float)
        self.ask_e = w["ask_entry"].to_numpy(float)
        self.bid_x = (w["bid_c_nxt"] + w["bid_p_nxt"]).to_numpy(float)
        self.ask_x = (w["ask_c_nxt"] + w["ask_p_nxt"]).to_numpy(float)
        self.is_last = w["is_last"].to_numpy(bool)
        self.same_k = (
            (w["K_c"].shift(-1) == w["K_c"])
            & (w["K_p"].shift(-1) == w["K_p"])
            & (w["date"].shift(-1) == w["date"])
            & ~w["is_last"]
        ).to_numpy(bool)

    def mid(self, q: np.ndarray):
        r = q * self.ret
        return r, r * self.entry, np.zeros(len(q))

    def crossed(self, q: np.ndarray):
        q = np.asarray(q, float)
        long, short = q > 0, q < 0
        nxt_q = np.append(q[1:], 0.0)
        hold = self.same_k & (np.sign(nxt_q) == np.sign(q)) & (q != 0)
        held_in = np.concatenate([[False], hold[:-1]])
        entry_px = np.where(
            held_in,
            self.entry,
            np.where(long, self.ask_e, np.where(short, self.bid_e, self.entry)),
        )
        exit_px = np.where(
            self.is_last,
            self.exit,
            np.where(
                hold,
                self.exit,
                np.where(long, self.bid_x, np.where(short, self.ask_x, self.exit)),
            ),
        )
        untradeable = ~held_in & (
            (long & ~(self.ask_e > 0)) | (short & ~(self.bid_e > 0))
        )
        pts = np.where(untradeable, np.nan, q * (exit_px - entry_px))
        active = (q != 0).astype(float)
        ncross = active * (
            (~held_in).astype(float) + ((~self.is_last) & (~hold)).astype(float)
        )
        return pts / self.entry, pts, ncross

    def daily(self, x: np.ndarray) -> pd.Series:
        s = pd.Series(np.asarray(x, float), index=self.w.index)
        return s.groupby(self.w["date"]).sum()


def score_rule(f: Fills, q: np.ndarray, fill: str) -> dict[str, float | str]:
    ret, pts, ncross = f.mid(q) if fill == "mid" else f.crossed(q)
    d_ret, d_pts, d_cross = f.daily(ret), f.daily(pts), f.daily(ncross)
    return {
        "Sharpe": _sh(d_ret),
        "mean/day": float(d_ret.mean()),
        "t": _t_plain(d_ret.to_numpy()),
        "t_HAC": _t_hac(d_ret.to_numpy()),
        "crossings/day": float(d_cross.mean()),
        "maxDD_points": _dd_points(d_pts.to_numpy()),
        "mean pts/day": float(d_pts.mean()),
        "pct long": 100.0 * float((q > 0).mean()),
        "pct flat": 100.0 * float((q == 0).mean()),
        "n_days": int(len(d_ret)),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    w = build_work()
    clocks = sorted(w["hhmm"].unique())
    dates = pd.DatetimeIndex(sorted(w["date"].unique()))
    cat = day_categories(dates)
    print("\nday categories (first match: FOMC, then month-end, then third Friday):")
    print(cat.value_counts().to_string())

    shares = build_shares(w, clocks, cat)
    w = attach_slices(w, shares, clocks)

    # ---------------- gate ----------------
    f = Fills(w)
    hhmm = w["hhmm"].to_numpy()
    s0 = w["s_V0"].to_numpy(float)
    q0 = np.where(np.isfinite(s0), np.where(s0 > 0, 1.0, -1.0), 0.0)
    rows = [
        ("always short", _sh(f.daily(-f.ret)), GATE["always short"]),
        ("sign(s) V0", _sh(f.daily(q0 * f.ret)), GATE["sign(s)"]),
        (
            "hybrid: always short, sign(s) at 15:30",
            _sh(f.daily(np.where(hhmm == clocks[-1], q0, -1.0) * f.ret)),
            GATE["hybrid"],
        ),
    ]
    gate = pd.DataFrame(rows, columns=["rule", "reproduced", "notebook"]).set_index(
        "rule"
    )
    gate["abs diff"] = (gate["reproduced"] - gate["notebook"]).abs()
    print("\n=== GATE: the notebook's rule table under V0 (midpoint, 866 days) ===")
    print(gate.to_string(float_format=lambda x: f"{x:.7f}"))
    gate.to_csv(OUT / "11_gate.csv")
    assert bool((gate["abs diff"] < GATE_TOL).all()), (
        "gate failed: V0 is not the deck's share"
    )
    print(f"gate passed: every rule within {GATE_TOL:g} of the notebook")

    # ---------------- warm-up ----------------
    warm = []
    for v in VARIANTS:
        miss = ~np.isfinite(w[f"w_{v}"].to_numpy(float))
        nosig = ~np.isfinite(w[f"s_{v}"].to_numpy(float))
        warm.append(
            {
                "variant": VLABEL[v],
                "warm-up bars (no share)": int(miss.sum()),
                "warm-up dates": int(w.loc[miss, "date"].nunique()),
                "bars with no signal": int(nosig.sum()),
                "days with a signal": int(w.loc[~nosig, "date"].nunique()),
            }
        )
    warm_t = pd.DataFrame(warm).set_index("variant")
    print(
        "\n=== warm-up (a bar with no share sits flat, q = 0; the zero stays in the daily sum) ==="
    )
    print(warm_t.to_string())
    warm_t.to_csv(OUT / "11_warmup.csv")

    # common support: the bars every variant prices
    fin = np.isfinite(w["rv_raw"].to_numpy(float))
    for v in VARIANTS:
        fin &= np.isfinite(w[f"slice_{v}"].to_numpy(float))
    com_days = int(w.loc[fin, "date"].nunique())
    print(
        f"\ncommon support (every variant prices the bar): {int(fin.sum())} bars on "
        f"{com_days} days of {w['date'].nunique()}"
    )

    # ---------------- 1. calibration ----------------
    rv = w["rv_raw"].to_numpy(float)
    cal_rows = []
    for v in VARIANTS:
        sl = w[f"slice_{v}"].to_numpy(float)
        wv = w[f"w_{v}"].to_numpy(float)
        for c in clocks:
            m = fin & (hhmm == c)
            cal_rows.append(
                {
                    "variant": VLABEL[v],
                    "clock": c,
                    "n": int(m.sum()),
                    "mean RV / mean slice": float(rv[m].mean() / sl[m].mean()),
                    "median RV/slice": float(np.median(rv[m] / sl[m])),
                    "mean w": float(wv[m].mean()),
                }
            )
    cal = pd.DataFrame(cal_rows)
    order = [VLABEL[v] for v in VARIANTS]
    cal_p = cal.pivot(index="clock", columns="variant", values="mean RV / mean slice")[
        order
    ]
    med_p = cal.pivot(index="clock", columns="variant", values="median RV/slice")[order]
    w_p = cal.pivot(index="clock", columns="variant", values="mean w")[order]
    print("\n=== 1a. calibration by clock: mean(RV_bar) / mean(slice), 1.0 = fair ===")
    print(cal_p.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== 1b. median(RV_bar / slice) by clock ===")
    print(med_p.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== 1c. mean share w by clock ===")
    print(w_p.to_string(float_format=lambda x: f"{x:.4f}"))
    cal.to_csv(OUT / "11_calibration_by_clock.csv", index=False)
    w_p.to_csv(OUT / "11_shares_by_clock.csv")

    disp = pd.DataFrame(
        {
            "sd across clocks of mean(RV)/mean(slice)": cal_p.std(ddof=1),
            "range across clocks": cal_p.max() - cal_p.min(),
            "mean abs deviation from 1": (cal_p - 1.0).abs().mean(),
        }
    )
    print(
        "\n=== 1d. pooled calibration statistic (flat across clocks is the target) ==="
    )
    print(disp.to_string(float_format=lambda x: f"{x:.4f}"))
    disp.to_csv(OUT / "11_calibration_dispersion.csv")

    ql = {v: _qlike(rv, w[f"slice_{v}"].to_numpy(float)) for v in VARIANTS}
    q_rows = []
    for c in [*clocks, "pooled"]:
        m = fin if c == "pooled" else (fin & (hhmm == c))
        row: dict[str, float | str | int] = {"clock": c, "n": int(m.sum())}
        for v in VARIANTS:
            row[f"QLIKE {v}"] = float(np.nanmean(ql[v][m]))
        for v in VARIANTS[1:]:
            row[f"DM t {v}-V0"] = _t_hac(ql[v][m] - ql["V0"][m])
        q_rows.append(row)
    qt = pd.DataFrame(q_rows).set_index("clock")
    print(
        "\n=== 1e. QLIKE of the slice as a forecast of RV_bar, and the paired "
        "Diebold-Mariano t against V0 (autocorrelation-robust, lag floor(1.5 n^(1/3)); "
        "a negative t favours the variant) ==="
    )
    print(qt.to_string(float_format=lambda x: f"{x:.4f}"))
    qt.to_csv(OUT / "11_qlike_by_clock.csv")

    # ---------------- 2. the sign rule ----------------
    q_of = {}
    for v in VARIANTS:
        s = w[f"s_{v}"].to_numpy(float)
        q_of[v] = np.where(np.isfinite(s), np.where(s > 0, 1.0, -1.0), 0.0)
    last = hhmm == clocks[-1]
    both = last.copy()
    for v in VARIANTS:
        both &= np.isfinite(w[f"w_{v}"].to_numpy(float))
    for v in VARIANTS[1:]:
        assert float(np.nanmax(np.abs(w.loc[last, f"w_{v}"] - 1.0))) == 0.0, (
            "15:30 w != 1"
        )
        assert np.array_equal(q_of[v][both], q_of["V0"][both]), "15:30 positions differ"
    print(
        f"\nthe 15:30 share is exactly 1 for every variant ({int(last.sum())} settlement "
        f"bars, max |w - 1| = 0), so the settlement leg is identical wherever all four "
        f"are warm ({int(both.sum())} bars) and the variants differ only in the daytime "
        f"legs. The warm-ups differ, so on the full frame the later variants sit flat at "
        f"15:30 on {int((last & ~both).sum())} early days where V0 already trades."
    )

    frames = (("full 866 days", np.ones(len(w), bool)), ("common support", fin))
    sign_rows = []
    for frame_name, keep in frames:
        wk = w[keep].reset_index(drop=True)
        fk = Fills(wk)
        hk = wk["hhmm"].to_numpy()
        for v in VARIANTS:
            for fill in ("mid", "crossed"):
                r = score_rule(fk, q_of[v][keep], fill)
                r.update(
                    {"frame": frame_name, "rule": f"sign(s) {VLABEL[v]}", "fill": fill}
                )
                sign_rows.append(r)
        for fill in ("mid", "crossed"):
            r = score_rule(fk, -np.ones(len(wk)), fill)
            r.update({"frame": frame_name, "rule": "always short", "fill": fill})
            sign_rows.append(r)
            r = score_rule(fk, np.where(hk == clocks[-1], q_of["V0"][keep], -1.0), fill)
            r.update(
                {
                    "frame": frame_name,
                    "rule": "hybrid: always short, sign(s) at 15:30 (identical V0-V3)",
                    "fill": fill,
                }
            )
            sign_rows.append(r)
    sign = pd.DataFrame(sign_rows).set_index(["frame", "fill", "rule"]).sort_index()
    cols = [
        "Sharpe",
        "mean/day",
        "t",
        "t_HAC",
        "crossings/day",
        "maxDD_points",
        "pct long",
        "pct flat",
        "n_days",
    ]
    print("\n=== 2. the sign rule: pooled daily-sum Sharpe x sqrt(252) ===")
    print(sign[cols].to_string(float_format=lambda x: f"{x:+.3f}"))
    sign.to_csv(OUT / "11_sign_rule.csv")

    pl_rows = []
    for v in VARIANTS:
        sl = w[f"slice_{v}"].to_numpy(float)
        for c in clocks:
            m = fin & (hhmm == c)
            pl_rows.append(
                {
                    "variant": VLABEL[v],
                    "clock": c,
                    "pct long (rule)": 100.0 * float((q_of[v][m] > 0).mean()),
                    "pct short (oracle)": 100.0 * float((rv[m] < sl[m]).mean()),
                }
            )
    pl = pd.DataFrame(pl_rows)
    long_p = pl.pivot(index="clock", columns="variant", values="pct long (rule)")[order]
    orc_p = pl.pivot(index="clock", columns="variant", values="pct short (oracle)")[
        order
    ]
    print("\n=== 2b. % long by clock, sign(s) under each variant (common support) ===")
    print(long_p.to_string(float_format=lambda x: f"{x:.1f}"))
    print(
        "\n=== 2c. oracle short rate by clock, %(RV_bar < slice) under each variant ==="
    )
    print(orc_p.to_string(float_format=lambda x: f"{x:.1f}"))
    pl.to_csv(OUT / "11_by_clock_rates.csv", index=False)

    # ---------------- 3. paired differences ----------------
    pair_rows = []
    for frame_name, keep in frames:
        wk = w[keep].reset_index(drop=True)
        fk = Fills(wk)
        for fill in ("mid", "crossed"):
            book = fk.mid if fill == "mid" else fk.crossed
            d0 = fk.daily(book(q_of["V0"][keep])[0])
            for v in VARIANTS[1:]:
                d1 = fk.daily(book(q_of[v][keep])[0])
                diff = (d1 - d0).to_numpy()
                ci = _boot_dsharpe(d1.to_numpy(), d0.to_numpy())
                pair_rows.append(
                    {
                        "frame": frame_name,
                        "fill": fill,
                        "variant": VLABEL[v],
                        "Sharpe V0": _sh(d0),
                        "Sharpe variant": _sh(d1),
                        "mean daily diff": float(np.nanmean(diff)),
                        "t plain": _t_plain(diff),
                        "t HAC": _t_hac(diff),
                        **{
                            k: ci[k]
                            for k in (
                                "dSharpe",
                                "pct_lo",
                                "pct_hi",
                                "basic_lo",
                                "basic_hi",
                            )
                        },
                    }
                )
    pair = pd.DataFrame(pair_rows).set_index(["frame", "fill", "variant"]).sort_index()
    print(
        f"\n=== 3. paired differences against V0 on the daily sums (circular block "
        f"bootstrap, block {BOOT_BLOCK}, B {BOOT_B}, rng({SEED}), shared draws) ==="
    )
    print(pair.to_string(float_format=lambda x: f"{x:+.4f}"))
    pair.to_csv(OUT / "11_paired.csv")

    # ---------------- 4. placebo ----------------
    base_cross = float(
        pair.loc[("common support", "crossed", VLABEL["V1"]), "Sharpe V0"]
    )
    cross_sh = {
        v: float(pair.loc[("common support", "crossed", VLABEL[v]), "Sharpe variant"])
        for v in VARIANTS[1:]
    }
    beaters = [v for v in VARIANTS[1:] if cross_sh[v] > base_cross]
    plc_rows: list[dict] = []
    if not beaters:
        print(
            f"\n=== 4. placebo: not run. No variant beats V0 at the crossed spread "
            f"(V0 {base_cross:+.3f}; "
            + ", ".join(f"{v} {cross_sh[v]:+.3f}" for v in VARIANTS[1:])
            + ") ==="
        )
        plc_rows.append(
            {
                "variant": "none",
                "note": "no variant beats V0 at the crossed spread; the placebo is not run",
                "V0 Sharpe crossed": base_cross,
            }
        )
    else:
        rng = np.random.default_rng(SEED)
        wk = w[fin].reset_index(drop=True)
        fk = Fills(wk)
        mi = pd.MultiIndex.from_arrays([wk["date"], wk["hhmm"]])
        iv_h = wk["iv_var_raw"].to_numpy(float) * wk["h_rem"].to_numpy(float)
        rvhat = wk["rv_hat"].to_numpy(float)
        for v in beaters:
            sh_v = shares[v]
            a = np.apply_along_axis(
                lambda p: _alloc_from_conditional(np.asarray(p, float)),
                1,
                sh_v.to_numpy(),
            )
            draws = np.empty(PLACEBO_DRAWS)
            for k in range(PLACEBO_DRAWS):
                perm = rng.permuted(
                    np.tile(np.arange(len(clocks)), (a.shape[0], 1)), axis=1
                )
                ap = np.take_along_axis(a, perm, axis=1)
                shp = _share_from_profile(
                    pd.DataFrame(ap, index=sh_v.index, columns=clocks), clocks
                )
                wv = shp.stack().reindex(mi).to_numpy()
                s = rvhat - iv_h * wv
                qq = np.where(np.isfinite(s), np.where(s > 0, 1.0, -1.0), 0.0)
                draws[k] = _sh(fk.daily(fk.crossed(qq)[0]))
            pct = 100.0 * float(np.mean(draws < cross_sh[v]))
            plc_rows.append(
                {
                    "variant": VLABEL[v],
                    "real Sharpe crossed": cross_sh[v],
                    "placebo mean": float(np.nanmean(draws)),
                    "placebo p95": float(np.nanpercentile(draws, 95)),
                    "percentile of the real variant": pct,
                    "draws": PLACEBO_DRAWS,
                }
            )
        print(
            f"\n=== 4. placebo: {PLACEBO_DRAWS} reshufflings of the shares across "
            f"clocks within each day (the day's allocation still sums to one) ==="
        )
        print(pd.DataFrame(plc_rows).to_string(index=False))
    pd.DataFrame(plc_rows).to_csv(OUT / "11_placebo.csv", index=False)

    # ---------------- 5. causality ----------------
    rng = np.random.default_rng(SEED)
    base = {v: shares[v].copy() for v in VARIANTS}
    cut_days = rng.choice(np.arange(200, len(dates)), size=N_CUTS, replace=False)
    cut_cl = rng.integers(0, len(clocks) - 1, size=N_CUTS)
    caus_rows = []
    for k in range(N_CUTS):
        d = dates[cut_days[k]]
        c = clocks[cut_cl[k]]
        w2 = w.copy()
        hit = (w2["date"] == d) & (w2["hhmm"] == c)
        w2.loc[hit, "rv_raw"] = w2.loc[hit, "rv_raw"] * 10.0
        sh2 = build_shares(w2, clocks, cat)
        row = {"date": str(d.date()), "perturbed clock": c}
        for v in ("V0", "V1", "V2"):
            a = base[v].loc[d].to_numpy(float)
            b = sh2[v].loc[d].to_numpy(float)
            row[f"{v} shares moved"] = int(np.nansum(np.abs(a - b) > 1e-12)) + int(
                (np.isnan(a) != np.isnan(b)).sum()
            )
        a = base["V3"].loc[d].to_numpy(float)
        b = sh2["V3"].loc[d].to_numpy(float)
        j = clocks.index(c)
        moved = (np.abs(a - b) > 1e-12) | (np.isnan(a) != np.isnan(b))
        row["V3 moved at or before the perturbed bar"] = int(moved[: j + 1].sum())
        row["V3 moved after it"] = int(moved[j + 1 :].sum())
        row["later bars"] = len(clocks) - 1 - j
        caus_rows.append(row)
    caus = pd.DataFrame(caus_rows)
    print(
        "\n=== 5. causality: multiply one bar's realized variance by 10 and rebuild "
        "every share; nothing used on that day may move ==="
    )
    print(caus.to_string(index=False))
    caus.to_csv(OUT / "11_causality.csv", index=False)
    for v in ("V0", "V1", "V2"):
        assert int(caus[f"{v} shares moved"].sum()) == 0, (
            f"{v} moved on the perturbed day"
        )
    assert int(caus["V3 moved at or before the perturbed bar"].sum()) == 0, (
        "V3 moved at or before the perturbed bar"
    )
    assert int(caus["V3 moved after it"].sum()) > 0, "V3's same-day term is dead"
    print(
        "assert passed: V0, V1 and V2 shares on the perturbed day are untouched "
        f"(0 of {N_CUTS * len(clocks)} share cells each); V3 moves only AFTER the "
        f"perturbed bar - {int(caus['V3 moved after it'].sum())} of "
        f"{int(caus['later bars'].sum())} later bars, "
        f"0 of {int((caus['later bars'].map(lambda x: len(clocks) - x)).sum())} "
        "bars at or before it"
    )

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for v in VARIANTS:
        ax.plot(clocks, cal_p[VLABEL[v]].to_numpy(), marker="o", label=VLABEL[v])
    ax.axhline(1.0, color="0.4", lw=1.0, ls="--")
    ax.set_ylabel("mean(RV bar) / mean(slice)")
    ax.set_xlabel("entry clock (ET)")
    ax.set_title(
        "Calibration of the implied slice by clock, four share variants\n"
        f"{int(fin.sum())} bars on {com_days} days; 1.0 = fair, above 1 = cheap slice"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "11_calibration_by_clock.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {OUT}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
