"""Proposal 10 - the ideas that were never run: Check 0 and books A-G.

Check 0 comes first and decides the frame of everything after it. On the
intraday notebook's own cached 10,387-bar work frame it separates two oracles
that the diagnosis has been quoting as if they were one:

  (1) sign(RV_bar - slice)          the published oracle, slice = IV^2 h w
  (3) sign(RV_rem - IV^2 h)         remaining realized against remaining implied

and asks whether the daytime edge is "always short, you just could not see it"
or "the oracle is mixed and you need a nowcast". Definitions (2) and (4) are
the tenor-mismatch control and the flat-share control. The slice-oracle is then
costed at the crossed spread, which has never been done.

The books, in the order they are run, killed as soon as the crossed figure is
non-positive or the placebo falls below its 95th percentile:

  A  nowcast of the current bar from elapsed sub-bar realized variance
  B  a synthesized 30-minute implied, from the implied term structure, not a
     realized-profile slice of it
  C  a hurdle: trade only when the expected edge clears k half-spreads
  D  sparse clocks, three daytime trades at most, not twelve
  E  a 0DTE-against-1DTE calendar as a market-priced 30-minute window
  F  fading the implied rather than realized-minus-implied
  G  the 15:00 always-short bar stacked on the 15:30 sign(s) settlement

Every book is F_t-measurable: no same-bar realized variance, no same-day
profile share, no 15:30 information on an earlier bar. Every trailing statistic
is an expanding mean over prior days only, lagged one day.

Fills. The midpoint case marks entry and exit at the quoted midpoints. The
crossed case pays the touch on both sides of every crossing, with the intraday
notebook's hold-through exemption: a re-pick that lands on the same two strikes
while the position keeps its sign is a hold, not a round trip. The 15:30 bar
cash-settles at the official close and pays no exit spread. Per-premium returns
divide the crossed points by the midpoint entry premium, which is the intraday
notebook's own convention.

Outputs: CSV and PNG under results/atm_straddle_intraday/proposals/10/.
Every number in 10_unrun.md is printed by this script.

Run:  python writeup/intraday_proposals/10_unrun.py
"""

from __future__ import annotations

import argparse
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
INTRA = REPO / "results" / "atm_straddle_intraday"
CACHE = INTRA / "cache"
OUT = INTRA / "proposals" / "10"
CHAIN = REPO / "data" / "spxw_chain.parquet"
CORE = REPO / "data" / "core_stats.parquet"
SPOT = REPO / "data" / "spxw_spot.parquet"

PROFILE_MIN_DAYS = 63  # the intraday notebook's standing warm-up for the profile
SEED = 0
PLACEBO_DRAWS = 200  # the brief's pre-registered entry-pattern placebo
SIGN_PLACEBO_DRAWS = 2000  # the rate-matched sign placebo, where a sign is the content
BOOT_B = 2000  # circular block bootstrap draws for a Sharpe difference
BOOT_BLOCK = 21  # block length, the deck's convention
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
CLOSE = "15:30"
HURDLE_K = (1.0, 1.5, 2.0)
HURDLE_K_PREREG = 1.5
SPARSE_CLOCKS = ("11:30", "12:30", "15:00")  # pre-registered before any per-clock read
SPARSE_MAX = 3  # at most three daytime trades in the honest sparse set

GATE_TOL = 1e-6

CHECK0_DEFS = {
    "1 sign(RV_bar - slice)": "the published oracle: the bar against IV^2 h w",
    "2 sign(RV_bar - IV_rem)": "tenor mismatch: the bar against the whole remainder",
    "3 sign(RV_rem - IV_rem)": "remaining against remaining, the honest remainder",
    "4 sign(RV_bar - IV^2 h)": "flat share: the bar against an equal 30-minute slice",
}


def tick(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78, flush=True)


# ----------------------------------------------------------------- the frame


def load_trade_cache() -> tuple[pd.DataFrame, Path]:
    """The intraday notebook's own cached package frame (trade_*.parquet)."""
    cands = sorted(CACHE.glob("trade_*.parquet"))
    if len(cands) != 1:
        raise FileNotFoundError(
            f"expected exactly one trade cache in {CACHE}, found {[c.name for c in cands]}"
        )
    return pd.read_parquet(cands[0]), cands[0]


def build_work(t0: float) -> tuple[pd.DataFrame, dict[str, object]]:
    """The notebook's scored frame: 10,387 bars, 866 days, twelve clocks.

    Built exactly as sections 4-5b of notebooks/_write_0dte_intraday_nb.py build
    it: the cached packages, the bar-end-labelled forecast panel shifted back one
    bar so the trade bar at t joins the forecast issued at t, and the causal
    diurnal share w (expanding per-clock mean of realized bar variance over prior
    days only, minimum 63 days, lagged one day).
    """
    pkg, cache_path = load_trade_cache()
    tick(t0, f"trade cache {cache_path.name}: {len(pkg):,} packages")

    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    tick(t0, f"forecast panel: {len(panel):,} stamps")

    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    pm = panel.set_index("t")[
        ["rv_hat", "rv_raw", "in_fit", "early_close"]
    ].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left")
    n_pre = len(work)
    work = work.dropna(subset=["R", "rv_hat"]).copy()
    if not bool(work["in_fit"].all()):
        raise AssertionError("a joined trade bar sits outside the smear's fit mask")

    work["iv_var_raw"] = work["iv_hourly"].astype(float) ** 2
    work["iv_var_chris"] = work["iv_var_raw"] * 0.5

    prof = work.pivot_table(
        index="date", columns="hhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    prof_exp = prof.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    clocks = sorted(work["hhmm"].unique())
    rem_sum = prof_exp[clocks[::-1]].cumsum(axis=1)[clocks]
    w_slice = prof_exp / rem_sum
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work["w_slice"] = w_slice.stack().reindex(mi).to_numpy()
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    work["n_rem"] = work["hhmm"].map(n_rem).astype(float)
    work["h_rem"] = work["n_rem"] * 0.5
    work["iv_rem"] = work["iv_var_raw"] * work["h_rem"]
    work["slice"] = work["iv_rem"] * work["w_slice"]
    work["s_matched"] = work["rv_hat"] - work["slice"]

    work = work.sort_values("t").reset_index(drop=True)
    work["half_spread"] = 0.5 * (
        work["ask_entry"].astype(float) - work["bid_entry"].astype(float)
    )
    # the variance still to run, realized, from this bar to the close inclusive
    work["rv_rem"] = (
        work.iloc[::-1].groupby("date")["rv_raw"].cumsum().iloc[::-1].to_numpy()
    )

    # The second frame is the notebook's own: a day belongs to it when the
    # sign(s) rule can trade at all, that is when at least one bar carries a
    # matched signal. The profile needs 63 prior sessions and one further date
    # has a single clock still short of them, so 64 dates carry an incomplete
    # profile while 803 days carry a signal.
    sig_days = set(work.loc[np.isfinite(work["s_matched"]), "date"].unique())
    work["post_warmup"] = work["date"].isin(sig_days)

    chk = work[(work["hhmm"] == CLOSE) & np.isfinite(work["slice"])]
    collapse = float(
        (chk["slice"] / work.loc[chk.index, "iv_var_chris"] - 1.0).abs().max()
    )

    meta = {
        "cache": cache_path.name,
        "bars": len(work),
        "days": int(work["date"].nunique()),
        "days_post_warmup": int(work.loc[work["post_warmup"], "date"].nunique()),
        "clocks": clocks,
        "dropped_at_join": n_pre - len(work),
        "collapse_1530": collapse,
        "bars_no_signal": int((~np.isfinite(work["s_matched"])).sum()),
        "dates_incomplete_profile": int(
            work.loc[~np.isfinite(work["w_slice"]), "date"].nunique()
        ),
    }
    tick(
        t0,
        f"work frame: {meta['bars']:,} bars on {meta['days']} days "
        f"({meta['days_post_warmup']} with a signal); "
        f"15:30 collapse |slice/(IV^2/2) - 1| max {collapse:.2e}",
    )
    return work, meta


# ----------------------------------------------------- fills and book scoring


class Book:
    """The crossed-spread machinery of the intraday notebook, on one frame."""

    def __init__(self, work: pd.DataFrame) -> None:
        self.w = work
        self.entry = work["entry"].to_numpy(float)
        self.exit = work["exit"].to_numpy(float)
        self.ask_e = work["ask_entry"].to_numpy(float)
        self.bid_e = work["bid_entry"].to_numpy(float)
        self.ask_x = (work["ask_c_nxt"] + work["ask_p_nxt"]).to_numpy(float)
        self.bid_x = (work["bid_c_nxt"] + work["bid_p_nxt"]).to_numpy(float)
        self.is_last = work["is_last"].to_numpy(bool)
        self.R = work["R"].to_numpy(float)
        self.date = work["date"]
        self.same_k = (
            (work["K_c"].shift(-1) == work["K_c"])
            & (work["K_p"].shift(-1) == work["K_p"])
            & (work["date"].shift(-1) == work["date"])
            & ~work["is_last"]
        ).to_numpy(bool)

    def crossed(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """(points, crossings, untradeable bars) at the touch, per bar."""
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
        return pts, ncross, int(untradeable.sum())

    def daily(self, q: np.ndarray, fill: str) -> pd.Series:
        q = np.asarray(q, float)
        if fill == "mid":
            r = pd.Series(q * self.R, index=self.w.index)
        else:
            pts, _, _ = self.crossed(q)
            r = pd.Series(pts / self.entry, index=self.w.index)
        return r.groupby(self.date).sum()

    def crossings_per_day(self, q: np.ndarray) -> float:
        _, nc, _ = self.crossed(q)
        return float(pd.Series(nc, index=self.w.index).groupby(self.date).sum().mean())


def sharpe(x) -> float:
    v = pd.Series(x).astype(float).dropna().to_numpy()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * ANN) if sd > 0 else float("nan")


def t_stat(x) -> float:
    v = pd.Series(x).astype(float).dropna().to_numpy()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * np.sqrt(len(v))) if sd > 0 else float("nan")


def max_dd(x) -> float:
    c = pd.Series(x).astype(float).fillna(0.0).cumsum()
    return float((c - c.cummax()).min())


def book_row(bk: Book, q: np.ndarray, name: str, post: pd.Series) -> dict[str, object]:
    """One book, both fills, both frames, per unit of midpoint entry premium."""
    q = np.asarray(q, float)
    out: dict[str, object] = {"book": name}
    keep_post = post.to_numpy(bool)
    for fill in ("mid", "crossed"):
        d = bk.daily(q, fill)
        d_post = bk.daily(np.where(keep_post, q, 0.0), fill)
        d_post = d_post[d_post.index.isin(bk.date[keep_post].unique())]
        out[f"Sharpe_{fill}_866"] = sharpe(d)
        out[f"mean_day_{fill}_866"] = float(d.mean())
        out[f"t_{fill}_866"] = t_stat(d)
        out[f"maxDD_{fill}_866"] = max_dd(d)
        out[f"n_days_{fill}_866"] = float(len(d))
        out[f"Sharpe_{fill}_803"] = sharpe(d_post)
        out[f"mean_day_{fill}_803"] = float(d_post.mean())
        out[f"t_{fill}_803"] = t_stat(d_post)
        out[f"maxDD_{fill}_803"] = max_dd(d_post)
        out[f"n_days_{fill}_803"] = float(len(d_post))
    _, nc, n_untr = bk.crossed(q)
    out["crossings_day"] = bk.crossings_per_day(q)
    out["trades_day"] = float(
        pd.Series((q != 0).astype(float), index=bk.w.index)
        .groupby(bk.date)
        .sum()
        .mean()
    )
    out["pct_bars_active"] = 100.0 * float((q != 0).mean())
    out["pct_long_of_active"] = (
        100.0 * float((q > 0).sum() / max((q != 0).sum(), 1)) if (q != 0).any() else 0.0
    )
    out["untradeable_bars"] = float(n_untr)
    return out


def entry_pattern_placebo(
    bk: Book,
    q: np.ndarray,
    fixed: np.ndarray | None = None,
    draws: int = PLACEBO_DRAWS,
    seed: int = SEED,
) -> dict[str, float]:
    """Sharpe percentile among random entry patterns with the same trade count.

    Each day keeps the book's own multiset of non-zero positions and the same
    number of trades; only WHICH bars carry them is redrawn, uniformly among
    that day's free bars. `fixed` marks bars whose position is part of the
    question rather than part of the pattern - the settlement leg, when the
    candidate is a set of daytime entries added to it - and those bars keep
    their position in every draw. A book with no free bar left to move has one
    pattern and the row reads NaN.
    """
    q = np.asarray(q, float)
    fx = np.zeros(len(q), bool) if fixed is None else np.asarray(fixed, bool)
    idx_by_day: list[np.ndarray] = []
    pos_by_day: list[np.ndarray] = []
    codes, _ = pd.factorize(bk.date)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order], prepend=-1, append=-1))
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        free = rows[~fx[rows]]
        idx_by_day.append(free)
        pos_by_day.append(q[free][q[free] != 0])
    if all(len(p) == len(r) for p, r in zip(pos_by_day, idx_by_day)):
        return {
            "draws": float(draws),
            "Sharpe_real_mid": sharpe(bk.daily(q, "mid")),
            "Sharpe_real_crossed": sharpe(bk.daily(q, "crossed")),
            "pctile_mid": float("nan"),
            "pctile_crossed": float("nan"),
            "note_degenerate": 1.0,
        }
    rng = np.random.default_rng(seed)
    real_mid = sharpe(bk.daily(q, "mid"))
    real_cr = sharpe(bk.daily(q, "crossed"))
    s_mid = np.empty(draws)
    s_cr = np.empty(draws)
    base = np.where(fx, q, 0.0)
    for b in range(draws):
        qd = base.copy()
        for rows, pos in zip(idx_by_day, pos_by_day):
            if len(pos) == 0:
                continue
            pick = rng.choice(len(rows), size=len(pos), replace=False)
            pick.sort()
            qd[rows[pick]] = pos
        s_mid[b] = sharpe(bk.daily(qd, "mid"))
        s_cr[b] = sharpe(bk.daily(qd, "crossed"))
    return {
        "draws": float(draws),
        "Sharpe_real_mid": real_mid,
        "Sharpe_real_crossed": real_cr,
        "pctile_mid": float(100.0 * np.mean(s_mid < real_mid)),
        "pctile_crossed": float(100.0 * np.mean(s_cr < real_cr)),
        "placebo_median_mid": float(np.median(s_mid)),
        "placebo_median_crossed": float(np.median(s_cr)),
        "placebo_p95_mid": float(np.percentile(s_mid, 95)),
        "placebo_p95_crossed": float(np.percentile(s_cr, 95)),
        "note_degenerate": 0.0,
    }


def paired_stats(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """B - A on the common days: mean, plain t, autocorrelation-robust t, dSharpe.

    The Sharpe difference is bootstrapped with a circular block resample of the
    daily series (block 21, 2,000 draws, rng(0)), the deck's own convention.
    """
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    x = j["b"].to_numpy(float)
    y = j["a"].to_numpy(float)
    d = x - y
    n = len(d)
    hac_t, lag = asl.newey_west_t(pd.Series(d))
    hat = (x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1)) * ANN
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng(SEED), n, BOOT_BLOCK, BOOT_B
    )
    xa, ya = x[idx], y[idx]
    ds = (
        xa.mean(axis=1) / xa.std(axis=1, ddof=1)
        - ya.mean(axis=1) / ya.std(axis=1, ddof=1)
    ) * ANN
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {
        "n": float(n),
        "mean_diff": float(d.mean()),
        "t_diff": t_stat(pd.Series(d)),
        "robust_t_diff": float(hac_t),
        "robust_lag": float(lag),
        "dSharpe": float(hat),
        "pct_draws_positive": float(100.0 * (ds > 0).mean()),
        "pctile_lo": float(lo),
        "pctile_hi": float(hi),
        "basic_lo": float(2.0 * hat - hi),
        "basic_hi": float(2.0 * hat - lo),
    }


def sign_placebo(
    bk: Book, q: np.ndarray, draws: int = SIGN_PLACEBO_DRAWS, seed: int = SEED
) -> dict[str, float]:
    """Sharpe percentile among random signs at the book's own long rate.

    The active bars are held fixed; only the sign is redrawn. This is the right
    placebo when the book's content IS a sign rather than a choice of clocks.
    """
    q = np.asarray(q, float)
    act = q != 0
    if not act.any() or (q[act] > 0).all() or (q[act] < 0).all():
        return {
            "draws": float(draws),
            "long_share_pct": 100.0 * float((q[act] > 0).mean()) if act.any() else 0.0,
            "pctile_mid": float("nan"),
            "pctile_crossed": float("nan"),
            "note_degenerate": 1.0,
        }
    rng = np.random.default_rng(seed)
    p_long = float((q[act] > 0).mean())
    real_mid = sharpe(bk.daily(q, "mid"))
    real_cr = sharpe(bk.daily(q, "crossed"))
    s_mid = np.empty(draws)
    s_cr = np.empty(draws)
    for b in range(draws):
        qd = np.zeros_like(q)
        qd[act] = np.where(rng.random(int(act.sum())) < p_long, 1.0, -1.0)
        s_mid[b] = sharpe(bk.daily(qd, "mid"))
        s_cr[b] = sharpe(bk.daily(qd, "crossed"))
    return {
        "draws": float(draws),
        "long_share_pct": 100.0 * p_long,
        "Sharpe_real_mid": real_mid,
        "Sharpe_real_crossed": real_cr,
        "pctile_mid": float(100.0 * np.mean(s_mid < real_mid)),
        "pctile_crossed": float(100.0 * np.mean(s_cr < real_cr)),
        "placebo_median_mid": float(np.median(s_mid)),
        "placebo_median_crossed": float(np.median(s_cr)),
        "note_degenerate": 0.0,
    }


# ---------------------------------------------------------------------- gate


def run_gate(bk: Book, work: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the intraday notebook's own two rule tables from this frame."""
    pos_m = np.where(np.isfinite(work["s_matched"]), np.sign(work["s_matched"]), 0.0)
    pos_m = np.where((pos_m == 0) & np.isfinite(work["s_matched"]), -1.0, pos_m)
    is_close = (work["hhmm"] == CLOSE).to_numpy(bool)
    qs = {
        "always short": -np.ones(len(work)),
        "always short, flat at 15:30": np.where(is_close, 0.0, -1.0),
        "sign(s)": pos_m,
        "always short, sign(s) close": np.where(is_close, pos_m, -1.0),
    }
    tgt_mid = pd.read_csv(INTRA / "rule_table_intraday_blk2.csv", index_col=0)
    tgt_cr = pd.read_csv(INTRA / "rule_table_intraday_crossed_blk2.csv", index_col=0)
    rows = []
    for name, q in qs.items():
        d_mid = bk.daily(q, "mid")
        d_cr = bk.daily(q, "crossed")
        rows.append(
            {
                "rule": name,
                "Sharpe mid": sharpe(d_mid),
                "target Sharpe mid": float(tgt_mid.loc[name, "Sharpe_ann"]),
                "mean/day mid": float(d_mid.mean()),
                "target mean/day mid": float(tgt_mid.loc[name, "mean_daily"]),
                "Sharpe crossed": sharpe(d_cr),
                "target Sharpe crossed": float(
                    tgt_cr.loc[name, "Sharpe crossed-spread"]
                ),
                "crossings/day": bk.crossings_per_day(q),
                "target crossings/day": float(tgt_cr.loc[name, "crossings/day"]),
            }
        )
    g = pd.DataFrame(rows).set_index("rule")
    g["max |gap|"] = np.maximum.reduce(
        [
            (g["Sharpe mid"] - g["target Sharpe mid"]).abs(),
            (g["mean/day mid"] - g["target mean/day mid"]).abs(),
            (g["Sharpe crossed"] - g["target Sharpe crossed"]).abs(),
            (g["crossings/day"] - g["target crossings/day"]).abs(),
        ]
    )
    worst = float(g["max |gap|"].max())
    print(g.to_string(float_format=lambda x: f"{x:+.6f}"))
    print(f"worst reproduction gap against the notebook's tables: {worst:.2e}")
    if worst > GATE_TOL:
        raise AssertionError(f"gate failed: worst gap {worst:.3e} > {GATE_TOL:.0e}")
    print("GATE PASSED")
    return g


def assert_causal(work: pd.DataFrame) -> pd.DataFrame:
    """Every trailing input moves only on prior days: a perturbation check.

    Ten cut days spread across the frame. On each, realized bar variance is
    tripled on that day and every later day and the trailing profile share, the
    trailing implied ratio and the trailing implied drift are recomputed; a
    causal statistic on the cut day cannot move.
    """
    clocks = sorted(work["hhmm"].unique())
    days = pd.DatetimeIndex(sorted(work.loc[work["post_warmup"], "date"].unique()))
    cuts = days[np.linspace(0, len(days) - 1, 10).astype(int)]
    rows = []
    prof0 = work.pivot_table(
        index="date", columns="hhmm", values="rv_raw", aggfunc="mean"
    )
    prof0 = prof0.sort_index()[clocks]
    for cut in cuts:
        prof = prof0.copy()
        prof.loc[prof.index >= cut] = prof.loc[prof.index >= cut] * 3.0
        pe = prof.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
        rs = pe[clocks[::-1]].cumsum(axis=1)[clocks]
        w_new = (pe / rs).loc[cut]
        pe0 = prof0.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
        rs0 = pe0[clocks[::-1]].cumsum(axis=1)[clocks]
        w_old = (pe0 / rs0).loc[cut]
        gap = float(np.nanmax(np.abs(w_new.to_numpy() - w_old.to_numpy())))
        rows.append({"cut": cut.date().isoformat(), "max |dw| on the cut day": gap})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    viol = int((out["max |dw| on the cut day"] > 0).sum())
    print(f"perturbation violations: {viol} of {len(out)}")
    if viol:
        raise AssertionError("the trailing profile share is not causal")
    return out


# ------------------------------------------------------------------- Check 0


def check0(work: pd.DataFrame, bk: Book) -> dict[str, pd.DataFrame]:
    """The four oracles, by clock, on both frames; then the slice-oracle costed."""
    w = work
    gaps = {
        "1 sign(RV_bar - slice)": w["rv_raw"] - w["slice"],
        "2 sign(RV_bar - IV_rem)": w["rv_raw"] - w["iv_rem"],
        "3 sign(RV_rem - IV_rem)": w["rv_rem"] - w["iv_rem"],
        "4 sign(RV_bar - IV^2 h)": w["rv_raw"] - w["iv_var_chris"],
    }
    ratios = {
        "1 sign(RV_bar - slice)": w["rv_raw"] / w["slice"],
        "2 sign(RV_bar - IV_rem)": w["rv_raw"] / w["iv_rem"],
        "3 sign(RV_rem - IV_rem)": w["rv_rem"] / w["iv_rem"],
        "4 sign(RV_bar - IV^2 h)": w["rv_raw"] / w["iv_var_chris"],
    }
    rows = []
    for frame, mask in (
        ("866 all days", np.ones(len(w), bool)),
        ("803 signal days", w["post_warmup"].to_numpy(bool)),
    ):
        for name, g in gaps.items():
            r = ratios[name]
            for clk, sub in pd.DataFrame(
                {"hhmm": w["hhmm"], "gap": g, "ratio": r, "keep": mask}
            ).groupby("hhmm"):
                s = sub.loc[sub["keep"] & np.isfinite(sub["gap"])]
                n = len(s)
                if n == 0:
                    continue
                short = float((s["gap"] < 0).mean())
                rows.append(
                    {
                        "frame": frame,
                        "definition": name,
                        "hhmm": clk,
                        "n": n,
                        "pct_short": 100.0 * short,
                        "pct_long": 100.0 * (1.0 - short),
                        "mean_gap": float(s["gap"].mean()),
                        "t_mean_gap": t_stat(s["gap"]),
                        "mean_ratio": float(
                            s["ratio"].replace([np.inf, -np.inf], np.nan).mean()
                        ),
                        "median_ratio": float(
                            s["ratio"].replace([np.inf, -np.inf], np.nan).median()
                        ),
                    }
                )
    by_clock = pd.DataFrame(rows)

    pooled = (
        by_clock.groupby(["frame", "definition"])
        .apply(
            lambda g: pd.Series(
                {
                    "n": g["n"].sum(),
                    "pct_short": float((g["pct_short"] * g["n"]).sum() / g["n"].sum()),
                    "mean_ratio": float(
                        (g["mean_ratio"] * g["n"]).sum() / g["n"].sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # item 5: the gap of definition (1) in variance units and in premium units.
    # A one-bar variance surprise adds to the variance still to run, and an
    # at-the-money package prices the square root of that, so the first-order
    # premium response is dP/P = dV / (2 V) with V = IV^2 h the remaining
    # implied variance. That is the gap in units of the premium itself.
    g1 = (w["rv_raw"] - w["slice"]).astype(float)
    prem_units = g1 / (2.0 * w["iv_rem"].astype(float))
    q_rows = []
    for clk, sub in pd.DataFrame(
        {"hhmm": w["hhmm"], "var": g1, "prem": prem_units, "keep": w["post_warmup"]}
    ).groupby("hhmm"):
        s = sub.loc[sub["keep"] & np.isfinite(sub["var"])]
        q_rows.append(
            {
                "hhmm": clk,
                "n": len(s),
                "mean_var": float(s["var"].mean()),
                "p10_var": float(s["var"].quantile(0.10)),
                "p50_var": float(s["var"].quantile(0.50)),
                "p90_var": float(s["var"].quantile(0.90)),
                "mean_prem": float(s["prem"].mean()),
                "p10_prem": float(s["prem"].quantile(0.10)),
                "p50_prem": float(s["prem"].quantile(0.50)),
                "p90_prem": float(s["prem"].quantile(0.90)),
            }
        )
    quant = pd.DataFrame(q_rows)

    # the slice-oracle, costed. It peeks at the bar's realized variance and is
    # not a book; it is the ceiling, and the question is whether the ceiling
    # survives the touch.
    cost_rows = []
    for name, g in gaps.items():
        q = np.where(np.isfinite(g), np.where(g > 0, 1.0, -1.0), 0.0)
        row = book_row(bk, q, "oracle " + name, work["post_warmup"])
        pl = sign_placebo(bk, q)
        row["sign_placebo_pctile_mid"] = pl["pctile_mid"]
        row["sign_placebo_pctile_crossed"] = pl["pctile_crossed"]
        cost_rows.append(row)
    # per clock, definition (1), both fills
    q1 = np.where(
        np.isfinite(gaps["1 sign(RV_bar - slice)"]),
        np.sign(gaps["1 sign(RV_bar - slice)"]),
        0.0,
    )
    pts_cr, _, _ = bk.crossed(q1)
    per_clock = []
    for clk, sub in pd.DataFrame(
        {
            "hhmm": w["hhmm"],
            "mid": q1 * w["R"].to_numpy(float),
            "crossed": pts_cr / w["entry"].to_numpy(float),
        }
    ).groupby("hhmm"):
        per_clock.append(
            {
                "hhmm": clk,
                "n": len(sub),
                "Sharpe mid": sharpe(sub["mid"]),
                "Sharpe crossed": sharpe(sub["crossed"]),
                "mean/bar mid": float(sub["mid"].mean()),
                "mean/bar crossed": float(sub["crossed"].mean()),
            }
        )
    oracle_cost = pd.DataFrame(cost_rows).set_index("book")
    oracle_clock = pd.DataFrame(per_clock)

    print("pooled short rate by definition (weighted over clocks)")
    print(pooled.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("by clock, 803 signal days (pct_short, mean gap, t of the mean gap)")
    piv = by_clock[by_clock["frame"] == "803 signal days"].pivot(
        index="hhmm", columns="definition", values="pct_short"
    )
    print(piv.to_string(float_format=lambda x: f"{x:6.2f}"))
    print()
    print("mean realized / priced by clock, 803 signal days")
    piv_r = by_clock[by_clock["frame"] == "803 signal days"].pivot(
        index="hhmm", columns="definition", values="mean_ratio"
    )
    print(piv_r.to_string(float_format=lambda x: f"{x:6.3f}"))
    print()
    print(
        "the gap of definition (1), variance units and premium units, 803 signal days"
    )
    print(quant.to_string(index=False, float_format=lambda x: f"{x:12.6g}"))
    print()
    print("the four oracles as books, per unit of midpoint premium")
    cols = [
        "Sharpe_mid_866",
        "Sharpe_crossed_866",
        "Sharpe_mid_803",
        "Sharpe_crossed_803",
        "mean_day_mid_866",
        "mean_day_crossed_866",
        "crossings_day",
        "pct_long_of_active",
        "sign_placebo_pctile_mid",
        "sign_placebo_pctile_crossed",
    ]
    print(oracle_cost[cols].to_string(float_format=lambda x: f"{x:+.4f}"))
    print()
    print("the published oracle (1) by clock, both fills")
    print(oracle_clock.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    sub = by_clock[by_clock["frame"] == "803 signal days"]
    for name, marker in (
        ("1 sign(RV_bar - slice)", "o"),
        ("3 sign(RV_rem - IV_rem)", "s"),
    ):
        s = sub[sub["definition"] == name].sort_values("hhmm")
        ax.plot(s["hhmm"], s["pct_short"], marker=marker, label=name)
    ax.axhline(50.0, color="0.5", lw=0.8, ls="--")
    ax.axhline(80.0, color="0.7", lw=0.8, ls=":")
    ax.set_ylabel("percent of bars the oracle is short")
    ax.set_xlabel("clock (entry stamp, ET)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Check 0: the slice oracle is mixed, the remaining-session oracle is short"
    )
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "10_check0_pct_short.png", dpi=140)
    plt.close(fig)

    return {
        "by_clock": by_clock,
        "pooled": pooled,
        "quantiles": quant,
        "oracle_cost": oracle_cost,
        "oracle_clock": oracle_clock,
    }


# ------------------------------------------------- causal trailing statistics


def trailing_clock_mean(
    work: pd.DataFrame, values: pd.Series, min_days: int = PROFILE_MIN_DAYS
) -> pd.Series:
    """Expanding per-clock mean over PRIOR days only, lagged one day.

    Same construction as the notebook's diurnal profile: pivot by (date, clock),
    expanding mean with a minimum day count, shifted one day, restacked onto the
    bars. Nothing on day d enters day d's value.
    """
    tab = pd.DataFrame(
        {
            "date": work["date"],
            "hhmm": work["hhmm"],
            "v": pd.to_numeric(values, errors="coerce"),
        }
    )
    piv = tab.pivot_table(
        index="date", columns="hhmm", values="v", aggfunc="mean"
    ).sort_index()
    exp = piv.expanding(min_periods=min_days).mean().shift(1)
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    return pd.Series(exp.stack().reindex(mi).to_numpy(), index=work.index)


# ------------------------------------------------------------------- book A


def book_a(work: pd.DataFrame) -> pd.DataFrame:
    """A nowcast of the current bar needs sub-bar data. This repository has none."""
    ts = pd.read_parquet(CHAIN, columns=["timestamp"])["timestamp"]
    et = pd.to_datetime(ts, utc=True).dt.tz_convert("America/New_York")
    mins = sorted((et.dt.hour * 60 + et.dt.minute).unique())
    step_chain = int(np.min(np.diff(mins))) if len(mins) > 1 else 0
    core = pd.read_parquet(CORE, columns=["endbartime", "numobs"])
    ct = pd.to_datetime(core["endbartime"])
    dt = ct.diff().dropna()
    step_core = int(dt[dt > pd.Timedelta(0)].min().total_seconds() // 60)
    spot = pd.read_parquet(SPOT, columns=["timestamp"])["timestamp"]
    st = pd.to_datetime(spot, utc=True).diff().dropna()
    step_spot = int(st[st > pd.Timedelta(0)].min().total_seconds() // 60)
    rows = [
        {
            "source": "data/spxw_chain.parquet (quotes)",
            "finest spacing (minutes)": step_chain,
            "stamps per session": len(mins),
        },
        {
            "source": "data/core_stats.parquet (realized moments)",
            "finest spacing (minutes)": step_core,
            "stamps per session": int(core["numobs"].max()),
        },
        {
            "source": "data/spxw_spot.parquet (underlying)",
            "finest spacing (minutes)": step_spot,
            "stamps per session": np.nan,
        },
    ]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(
        "the realized panel stores per-bar sums over at most "
        f"{int(core['numobs'].max())} intra-bar observations; the observations themselves "
        "are not in this repository, and no quote, spot or realized series is finer "
        f"than {step_chain} minutes."
    )
    print(
        "A is UNCONSTRUCTIBLE: an elapsed-realized nowcast of the bar in flight would "
        "have to read the bar's own first minutes, and the only object with that "
        "content here is the finished bar. Using it is the peek the oracle already "
        "measures. Skipped rather than leaked."
    )
    return out


# ------------------------------------------------------------------- book B


def book_b(
    work: pd.DataFrame, bk: Book
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """A synthesized 30-minute implied: the term structure, not a realized slice.

    At t the package midpoint is a price for the whole window still to run. The
    market's price of the next thirty minutes is what that price gives up when
    the clock advances one bar and the implied moves the way it usually moves at
    this clock:

        V_t = IV_t^2 h_t,   E[V_{t+1}] = (g_t IV_t)^2 (h_t - 1/2),
        sigma^2_syn = V_t - E[V_{t+1}] = IV_t^2 (h_t - g_t^2 (h_t - 1/2)),

    with g_t the trailing per-clock mean of IV_{t+1}/IV_t over prior days only.
    At 15:30, h_t = 1/2 and the second term vanishes: the synthetic collapses to
    IV^2/2, the deck's own close-trade implied, exactly as the realized-profile
    slice does. In points, the same model prices the expected decay

        D_t = P(IV_t sqrt(h_t)) - P(g_t IV_t sqrt(h_t - 1/2))

    of the SAME strikes on an unchanged underlying, which is what book C spends
    against the half-spread.
    """
    w = work
    iv = w["iv_hourly"].astype(float)
    nxt_iv = iv.shift(-1).where(
        (w["date"].shift(-1) == w["date"]) & ~w["is_last"].to_numpy(bool)
    )
    ratio = (nxt_iv / iv).replace([np.inf, -np.inf], np.nan)
    g = trailing_clock_mean(w, ratio)
    g = g.where(
        np.isfinite(g), 1.0
    )  # 15:30 has no next bar; its term is multiplied by zero
    h = w["h_rem"].astype(float)
    v_now = (iv**2) * h
    v_next = (g * iv) ** 2 * (h - 0.5)
    syn = v_now - v_next

    # the same model in points: the expected one-bar decay of the held strikes
    s_now = iv * np.sqrt(h)
    s_next = g * iv * np.sqrt(np.maximum(h - 0.5, 0.0))
    price_now = np.array(
        [
            asl._bsm_package_price(float(a), float(s), float(kc), float(kp))
            if np.isfinite(a)
            else np.nan
            for a, s, kc, kp in zip(s_now, w["S"], w["K_c"], w["K_p"])
        ]
    )
    price_next = np.array(
        [
            asl._bsm_package_price(float(a), float(s), float(kc), float(kp))
            if np.isfinite(a)
            else np.nan
            for a, s, kc, kp in zip(s_next, w["S"], w["K_c"], w["K_p"])
        ]
    )
    decay = pd.Series(price_now - price_next, index=w.index)

    # gates: the model reprices the quoted midpoint, and the synthetic collapses
    pricing_ratio = pd.Series(price_now, index=w.index) / w["entry"].astype(float)
    close = (w["hhmm"] == CLOSE).to_numpy(bool)
    collapse = float((syn[close] / w.loc[close, "iv_var_chris"] - 1.0).abs().max())
    n_nonpos = int((syn <= 0).sum())

    s_syn = w["rv_hat"].astype(float) - syn
    q_syn = np.where(np.isfinite(s_syn), np.where(s_syn > 0, 1.0, -1.0), 0.0)
    q_w = np.where(
        np.isfinite(w["s_matched"]), np.where(w["s_matched"] > 0, 1.0, -1.0), 0.0
    )

    rows = [
        book_row(
            bk, q_syn, "B sign(rv_hat - synthetic 30-min implied)", w["post_warmup"]
        ),
        book_row(
            bk,
            q_w,
            "sign(s) on the realized-profile slice (the deck)",
            w["post_warmup"],
        ),
    ]
    tab = pd.DataFrame(rows).set_index("book")

    # collinearity: is the synthetic just the remaining implied, or the slice?
    ok = np.isfinite(syn) & np.isfinite(w["slice"]) & np.isfinite(w["iv_rem"])
    coll = pd.DataFrame(
        [
            {
                "pair": "synthetic vs IV^2 h w (the realized-profile slice)",
                "corr level": float(np.corrcoef(syn[ok], w.loc[ok, "slice"])[0, 1]),
                "corr log": float(
                    np.corrcoef(
                        np.log(syn[ok & (syn > 0)]),
                        np.log(w.loc[ok & (syn > 0), "slice"]),
                    )[0, 1]
                ),
                "median ratio": float((syn[ok] / w.loc[ok, "slice"]).median()),
            },
            {
                "pair": "synthetic vs IV^2 h (the whole remainder)",
                "corr level": float(np.corrcoef(syn[ok], w.loc[ok, "iv_rem"])[0, 1]),
                "corr log": float(
                    np.corrcoef(
                        np.log(syn[ok & (syn > 0)]),
                        np.log(w.loc[ok & (syn > 0), "iv_rem"]),
                    )[0, 1]
                ),
                "median ratio": float((syn[ok] / w.loc[ok, "iv_rem"]).median()),
            },
        ]
    )
    share_syn = (syn / w["iv_rem"]).groupby(w["hhmm"]).median()
    share_w = w["w_slice"].groupby(w["hhmm"]).median()
    shares = pd.DataFrame(
        {"implied share (synthetic / IV^2 h)": share_syn, "realized share w": share_w}
    )
    shares["ratio"] = shares.iloc[:, 0] / shares.iloc[:, 1]

    print(
        f"model gate: BSM package price at the vendor implied over the remaining window "
        f"reproduces the quoted midpoint, median ratio {float(pricing_ratio.median()):.4f}, "
        f"5th-95th {float(pricing_ratio.quantile(0.05)):.4f}-{float(pricing_ratio.quantile(0.95)):.4f}"
    )
    print(
        f"collapse gate: at 15:30 the synthetic equals IV^2/2, max |ratio - 1| = {collapse:.2e}"
    )
    print(f"synthetic slices that are not positive: {n_nonpos} of {len(w)}")
    print(
        f"trailing implied ratio g by clock (median): \n{g.groupby(w['hhmm']).median().round(4).to_string()}"
    )
    print()
    print("the two shares of the remaining implied variance, by clock (medians)")
    print(shares.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("collinearity of the synthetic with what it replaces")
    print(coll.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print()
    cols = [
        "Sharpe_mid_866",
        "Sharpe_crossed_866",
        "Sharpe_mid_803",
        "Sharpe_crossed_803",
        "mean_day_mid_866",
        "mean_day_crossed_866",
        "crossings_day",
        "pct_long_of_active",
    ]
    print(tab[cols].to_string(float_format=lambda x: f"{x:+.4f}"))
    pl = sign_placebo(bk, q_syn)
    print(
        f"rate-matched sign placebo, {int(pl['draws'])} draws, long share "
        f"{pl['long_share_pct']:.2f}%: percentile {pl['pctile_mid']:.1f} mid, "
        f"{pl['pctile_crossed']:.1f} crossed"
    )
    tab["sign_placebo_pctile_mid"] = [pl["pctile_mid"], np.nan]
    tab["sign_placebo_pctile_crossed"] = [pl["pctile_crossed"], np.nan]
    diag = pd.concat(
        [
            coll.assign(kind="collinearity"),
            shares.reset_index()
            .rename(columns={"hhmm": "pair"})
            .assign(kind="share by clock"),
        ],
        ignore_index=True,
    )
    return tab, diag, pd.Series(syn, index=w.index), decay


# ------------------------------------------------------------------- book C


def book_c(work: pd.DataFrame, bk: Book, decay: pd.Series) -> pd.DataFrame:
    """A hurdle: trade only when the expected edge clears k quoted half-spreads.

    The signal s is a variance mispricing. A one-bar variance surprise adds to
    the variance still to run and an at-the-money package prices its square
    root, so the first-order premium response to |s| is

        E_t = P_t |s_t| / (2 V_t),   V_t = IV_t^2 h_t,

    in index points. (The brief's |s| x premium is not a points quantity; this
    is that comparison made dimensionally honest, and it is the same first-order
    map Check 0 uses for its premium-unit column.) The always-short line is run
    against two readings of "expected decay". The first is book B's frozen-spot
    decay D_t, which is the package's theta over the bar and not its expected
    profit: under the market's own implied the package is a martingale and the
    gamma term exactly offsets the theta, so D_t overstates what a short can
    expect. The second is the honest one - the trailing per-clock mean of the
    always-short profit in points over prior days only, lagged one day - which
    is what the market has actually been paying at this clock. k = 1.5 is
    pre-registered; k = 1 and k = 2 are reported beside it.
    """
    w = work
    half = w["half_spread"].astype(float)
    edge_sign = (
        w["entry"].astype(float)
        * w["s_matched"].abs()
        / (2.0 * w["iv_rem"].astype(float))
    )
    q_sign = np.where(
        np.isfinite(w["s_matched"]), np.where(w["s_matched"] > 0, 1.0, -1.0), 0.0
    )
    short_pts = -(w["exit"].astype(float) - w["entry"].astype(float))
    decay_hat = trailing_clock_mean(w, short_pts)
    rows = []
    for k in HURDLE_K:
        keep_s = (edge_sign > k * half).to_numpy(bool) & np.isfinite(edge_sign)
        rows.append(
            book_row(
                bk,
                np.where(keep_s, q_sign, 0.0),
                f"C sign(s) with hurdle k={k}",
                w["post_warmup"],
            )
        )
        keep_d = (decay > k * half).to_numpy(bool) & np.isfinite(decay)
        rows.append(
            book_row(
                bk,
                np.where(keep_d, -1.0, 0.0),
                f"C always short, theta hurdle k={k}",
                w["post_warmup"],
            )
        )
        keep_h = (decay_hat > k * half).to_numpy(bool) & np.isfinite(decay_hat)
        rows.append(
            book_row(
                bk,
                np.where(keep_h, -1.0, 0.0),
                f"C always short, trailing-decay hurdle k={k}",
                w["post_warmup"],
            )
        )
    rows.append(book_row(bk, q_sign, "C sign(s) unfiltered", w["post_warmup"]))
    rows.append(
        book_row(bk, -np.ones(len(w)), "C always short unfiltered", w["post_warmup"])
    )
    tab = pd.DataFrame(rows).set_index("book")

    print(
        f"median expected edge {float(edge_sign.median()):.4f} pts against a median "
        f"half-spread of {float(half.median()):.4f} pts "
        f"({float((half / w['entry']).median() * 100):.2f}% of the midpoint premium)"
    )
    print(
        f"median frozen-spot decay (theta) {float(decay.median()):.4f} pts; median "
        f"trailing realized short profit {float(decay_hat.median()):.4f} pts; the "
        f"realized always-short profit is {float(short_pts.mean()):.4f} pts a bar - "
        "the theta is not an expectation, the gamma term offsets it"
    )
    for k in HURDLE_K:
        print(
            f"k={k}: sign(s) keeps {float((edge_sign > k * half).mean()) * 100:.1f}% of bars, "
            f"the theta hurdle keeps {float((decay > k * half).mean()) * 100:.1f}%, "
            f"the trailing-decay hurdle keeps {float((decay_hat > k * half).mean()) * 100:.1f}%"
        )
    cols = [
        "Sharpe_mid_866",
        "Sharpe_crossed_866",
        "Sharpe_mid_803",
        "Sharpe_crossed_803",
        "mean_day_crossed_866",
        "trades_day",
        "crossings_day",
    ]
    print(tab[cols].to_string(float_format=lambda x: f"{x:+.4f}"))

    prereg = f"C always short, trailing-decay hurdle k={HURDLE_K_PREREG}"
    keep_h = (decay_hat > HURDLE_K_PREREG * half).to_numpy(bool) & np.isfinite(
        decay_hat
    )
    pl = entry_pattern_placebo(bk, np.where(keep_h, -1.0, 0.0))
    print(
        f"entry-pattern placebo on the pre-registered cell ({prereg}), "
        f"{int(pl['draws'])} draws: percentile {pl['pctile_mid']:.1f} mid, "
        f"{pl['pctile_crossed']:.1f} crossed"
    )
    tab.loc[prereg, "placebo_pctile_mid"] = pl["pctile_mid"]
    tab.loc[prereg, "placebo_pctile_crossed"] = pl["pctile_crossed"]
    prereg_s = f"C sign(s) with hurdle k={HURDLE_K_PREREG}"
    keep_s = (edge_sign > HURDLE_K_PREREG * half).to_numpy(bool) & np.isfinite(
        edge_sign
    )
    pl_s = entry_pattern_placebo(bk, np.where(keep_s, q_sign, 0.0))
    print(
        f"entry-pattern placebo on {prereg_s}: percentile {pl_s['pctile_mid']:.1f} mid, "
        f"{pl_s['pctile_crossed']:.1f} crossed"
    )
    tab.loc[prereg_s, "placebo_pctile_mid"] = pl_s["pctile_mid"]
    tab.loc[prereg_s, "placebo_pctile_crossed"] = pl_s["pctile_crossed"]
    return tab, decay_hat


def _q_hurdle_short(work: pd.DataFrame, decay: pd.Series, k: float) -> np.ndarray:
    keep = (decay > k * work["half_spread"].astype(float)).to_numpy(bool) & np.isfinite(
        decay
    )
    return np.where(keep, -1.0, 0.0)


# ------------------------------------------------------------------- book D


def book_d(work: pd.DataFrame, bk: Book) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sparse clocks: three daytime trades at most, plus the settlement leg.

    Two sets. The pre-registered one is {11:30, 12:30, 15:00} always short,
    fixed before any per-clock Sharpe in report 03 was re-read, with sign(s) at
    15:30. The honest one is chosen on the first half of the frame by the only
    criterion that is about costs rather than about returns - a clock's
    break-even half-spread must exceed the half-spread actually quoted there -
    and evaluated on the second half.
    """
    w = work
    is_close = (w["hhmm"] == CLOSE).to_numpy(bool)
    pos_m = np.where(
        np.isfinite(w["s_matched"]), np.where(w["s_matched"] > 0, 1.0, -1.0), 0.0
    )

    def q_for(clocks: tuple[str, ...], with_close: bool = True) -> np.ndarray:
        day = w["hhmm"].isin(clocks).to_numpy(bool) & ~is_close
        q = np.where(day, -1.0, 0.0)
        if with_close:
            q = np.where(is_close, pos_m, q)
        return q

    rows = [
        book_row(
            bk,
            q_for(SPARSE_CLOCKS),
            "D pre-registered {11:30,12:30,15:00} + sign(s) close",
            w["post_warmup"],
        ),
        book_row(
            bk, q_for(()), "D settlement leg only: sign(s) at 15:30", w["post_warmup"]
        ),
        book_row(
            bk,
            np.where(is_close, pos_m, -1.0),
            "D the deck's hybrid (all twelve bars)",
            w["post_warmup"],
        ),
    ]

    # the honest set, chosen on the first half
    days = pd.DatetimeIndex(sorted(w["date"].unique()))
    split = days[len(days) // 2]
    first = (w["date"] < split).to_numpy(bool)
    second = ~first
    sel_rows = []
    for clk in sorted(set(w["hhmm"]) - {CLOSE}):
        m = first & (w["hhmm"] == clk).to_numpy(bool)
        q = np.where(m, -1.0, 0.0)
        d = bk.daily(q, "mid")
        d = d[d.index < split]
        _, nc, _ = bk.crossed(q)
        ncd = pd.Series(nc, index=w.index).groupby(w["date"]).sum()
        ncd = ncd[ncd.index < split]
        be = float(d.mean() / ncd.mean() * 100.0) if float(ncd.mean()) > 0 else np.nan
        hs = float((w.loc[m, "half_spread"] / w.loc[m, "entry"]).median() * 100.0)
        sel_rows.append(
            {
                "hhmm": clk,
                "break-even half-spread % prem (first half)": be,
                "median half-spread % prem (first half)": hs,
                "margin": be - hs,
            }
        )
    sel = pd.DataFrame(sel_rows).sort_values("margin", ascending=False)
    chosen = tuple(sel.loc[sel["margin"] > 0, "hhmm"].head(SPARSE_MAX))
    print("clock selection on the first half (days before " + str(split.date()) + ")")
    print(sel.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(
        "clocks whose break-even half-spread exceeds the quoted one:", chosen or "none"
    )

    if chosen:
        q_h = q_for(chosen)
        rows.append(
            book_row(
                bk,
                np.where(second, q_h, 0.0),
                f"D honest {chosen} + sign(s) close (second half)",
                w["post_warmup"],
            )
        )
        rows.append(
            book_row(
                bk,
                np.where(second, np.where(is_close, pos_m, 0.0), 0.0),
                "D settlement leg only (second half)",
                w["post_warmup"],
            )
        )
    else:
        rows.append(
            book_row(
                bk,
                np.where(second, np.where(is_close, pos_m, 0.0), 0.0),
                "D settlement leg only (second half)",
                w["post_warmup"],
            )
        )
    tab = pd.DataFrame(rows).set_index("book")
    cols = [
        "Sharpe_mid_866",
        "Sharpe_crossed_866",
        "Sharpe_mid_803",
        "Sharpe_crossed_803",
        "mean_day_crossed_866",
        "trades_day",
        "crossings_day",
    ]
    print(tab[cols].to_string(float_format=lambda x: f"{x:+.4f}"))
    # The settlement leg is held fixed: the question is which DAYTIME clocks,
    # not whether the close trade matters.
    pl = entry_pattern_placebo(bk, q_for(SPARSE_CLOCKS), fixed=is_close)
    print(
        f"entry-pattern placebo on the pre-registered set (settlement leg held fixed, "
        f"three daytime entries redrawn), {int(pl['draws'])} draws: percentile "
        f"{pl['pctile_mid']:.1f} mid, {pl['pctile_crossed']:.1f} crossed; median draw "
        f"{pl.get('placebo_median_mid', float('nan')):+.3f} mid, "
        f"{pl.get('placebo_median_crossed', float('nan')):+.3f} crossed"
    )
    name0 = "D pre-registered {11:30,12:30,15:00} + sign(s) close"
    tab.loc[name0, "placebo_pctile_mid"] = pl["pctile_mid"]
    tab.loc[name0, "placebo_pctile_crossed"] = pl["pctile_crossed"]
    # against the surviving trade, paired on the same days
    paired = {}
    for fill in ("mid", "crossed"):
        a = bk.daily(q_for(()), fill)
        b = bk.daily(q_for(SPARSE_CLOCKS), fill)
        st = paired_stats(a, b)
        paired[fill] = st
        print(
            f"D minus the settlement leg alone, {fill}: mean {st['mean_diff']:+.5f}/day, "
            f"t {st['t_diff']:+.2f}, autocorrelation-robust t {st['robust_t_diff']:+.2f} "
            f"(lag {int(st['robust_lag'])}), dSharpe {st['dSharpe']:+.4f}, percentile 95% "
            f"[{st['pctile_lo']:+.3f}, {st['pctile_hi']:+.3f}], basic 95% "
            f"[{st['basic_lo']:+.3f}, {st['basic_hi']:+.3f}]"
        )
        for k, v in st.items():
            tab.loc[name0, f"vs_close_{fill}_{k}"] = v
    return tab, sel


# ------------------------------------------------------------------- book E


def book_e() -> pd.DataFrame:
    """A 0DTE-against-1DTE calendar needs a second expiration. The chain has one."""
    df = pd.read_parquet(CHAIN, columns=["expiration", "timestamp"])
    et = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    lag = (
        pd.to_datetime(df["expiration"]).dt.normalize()
        - et.dt.normalize().dt.tz_localize(None)
    ).dt.days
    vc = lag.value_counts().sort_index()
    out = vc.rename("rows").rename_axis("days to expiration").reset_index()
    print(out.to_string(index=False))
    print(
        "E is UNCONSTRUCTIBLE: every row of data/spxw_chain.parquet expires on its own "
        "trade date, so there is no 1DTE or next-weekly quote at any 30-minute stamp and "
        "no calendar spread to price. A traded claim on the front window cannot be built "
        "from this chain."
    )
    return out


# ------------------------------------------------------------------- book F


def book_f(work: pd.DataFrame, bk: Book) -> pd.DataFrame:
    """Fade the implied rather than realized-minus-implied.

    The next mark's change in implied volatility is not F_t. Two causal
    stand-ins: the trailing per-clock mean drift of the implied over prior days,
    and the day's own last thirty-minute implied change. The package is long
    vega, so a book that expects the implied to fall is SHORT the package:
    q = sign(expected change in implied). Both stand-ins size the same package
    as always short, so the question is whether either is a different book -
    measured as position agreement, not asserted.
    """
    w = work
    iv = w["iv_hourly"].astype(float)
    d_iv_next = (
        iv.shift(-1).where(
            (w["date"].shift(-1) == w["date"]) & ~w["is_last"].to_numpy(bool)
        )
        - iv
    )
    d_iv_last = iv - iv.shift(1).where(w["date"].shift(1) == w["date"])
    drift = trailing_clock_mean(w, d_iv_next)

    q_short = -np.ones(len(w))
    q_drift = np.where(np.isfinite(drift), np.sign(drift), 0.0)
    q_fade = np.where(np.isfinite(d_iv_last), -np.sign(d_iv_last), 0.0)
    q_follow = np.where(np.isfinite(d_iv_last), np.sign(d_iv_last), 0.0)
    q_peek = np.where(np.isfinite(d_iv_next), np.sign(d_iv_next), 0.0)

    books = {
        "F short when the trailing implied drift is negative": q_drift,
        "F fade the last 30-min implied change": q_fade,
        "F follow the last 30-min implied change": q_follow,
        "F always short the package (the comparator)": q_short,
        "F peeking: trade the next implied change (NOT F_t)": q_peek,
    }
    rows = [book_row(bk, q, n, w["post_warmup"]) for n, q in books.items()]
    tab = pd.DataFrame(rows).set_index("book")
    for n, q in books.items():
        act = q != 0
        tab.loc[n, "agreement with always short %"] = (
            100.0 * float((q[act] < 0).mean()) if act.any() else np.nan
        )
    print("trailing implied drift by clock (median, vendor hourly units)")
    print(
        drift.groupby(w["hhmm"]).median().to_string(float_format=lambda x: f"{x:+.6f}")
    )
    print(
        "clocks whose trailing drift is negative: "
        f"{int((drift.groupby(w['hhmm']).median() < 0).sum())} of {w['hhmm'].nunique()}"
    )
    cols = [
        "Sharpe_mid_866",
        "Sharpe_crossed_866",
        "Sharpe_mid_803",
        "Sharpe_crossed_803",
        "mean_day_crossed_866",
        "crossings_day",
        "agreement with always short %",
    ]
    print(tab[cols].to_string(float_format=lambda x: f"{x:+.4f}"))
    return tab


# ------------------------------------------------------------------- book G


def book_g(work: pd.DataFrame, bk: Book) -> pd.DataFrame:
    """Do the 15:00 always-short bar and the 15:30 sign(s) settlement stack?"""
    w = work
    is_close = (w["hhmm"] == CLOSE).to_numpy(bool)
    is_1500 = (w["hhmm"] == "15:00").to_numpy(bool)
    pos_m = np.where(
        np.isfinite(w["s_matched"]), np.where(w["s_matched"] > 0, 1.0, -1.0), 0.0
    )
    books = {
        "G line 1 only: 15:00 always short, 30-min hold": np.where(is_1500, -1.0, 0.0),
        "G line 2 only: 15:30 sign(s), cash-settled": np.where(is_close, pos_m, 0.0),
        "G both lines": np.where(is_1500, -1.0, np.where(is_close, pos_m, 0.0)),
    }
    rows = [book_row(bk, q, n, w["post_warmup"]) for n, q in books.items()]
    tab = pd.DataFrame(rows).set_index("book")
    q1 = books["G line 1 only: 15:00 always short, 30-min hold"]
    q2 = books["G line 2 only: 15:30 sign(s), cash-settled"]
    q12 = books["G both lines"]
    d1 = bk.daily(q1, "crossed")
    d2 = bk.daily(q2, "crossed")
    d12 = bk.daily(q12, "crossed")
    k1500 = w.loc[is_1500, ["date", "K_c", "K_p"]].set_index("date")
    k1530 = w.loc[is_close, ["date", "K_c", "K_p"]].set_index("date")
    sgn = pd.Series(pos_m[is_close], index=w.loc[is_close, "date"])
    j = k1500.join(k1530, how="inner", lsuffix="_a", rsuffix="_b")
    netted = float(
        (
            (j["K_c_a"] == j["K_c_b"])
            & (j["K_p_a"] == j["K_p_b"])
            & (sgn.reindex(j.index) < 0)
        ).mean()
    )
    print(
        f"the joint book is CHEAPER than the sum of the two lines by "
        f"{float((d12 - d1 - d2).mean()):+.5f} per unit of premium a day: on "
        f"{100.0 * netted:.1f}% of days the 15:30 pick lands on the 15:00 strikes with "
        "the same sign, and that boundary is a hold, not a round trip"
    )
    cols = [
        "Sharpe_mid_866",
        "Sharpe_crossed_866",
        "Sharpe_mid_803",
        "Sharpe_crossed_803",
        "mean_day_mid_866",
        "mean_day_crossed_866",
        "crossings_day",
        "maxDD_crossed_866",
    ]
    print(tab[cols].to_string(float_format=lambda x: f"{x:+.4f}"))
    for fill in ("mid", "crossed"):
        st = paired_stats(bk.daily(q2, fill), bk.daily(q12, fill))
        print(
            f"both lines minus line 2 alone, {fill}: mean {st['mean_diff']:+.5f}/day, "
            f"t {st['t_diff']:+.2f}, autocorrelation-robust t {st['robust_t_diff']:+.2f} "
            f"(lag {int(st['robust_lag'])}), dSharpe {st['dSharpe']:+.4f}, draws positive "
            f"{st['pct_draws_positive']:.1f}%, percentile 95% [{st['pctile_lo']:+.3f}, "
            f"{st['pctile_hi']:+.3f}], basic 95% [{st['basic_lo']:+.3f}, {st['basic_hi']:+.3f}]"
        )
        for k, v in st.items():
            tab.loc["G both lines", f"vs_line2_{fill}_{k}"] = v
    # Is 15:00 the clock, or would any daytime bar do? One daytime short a day,
    # redrawn among the eleven daytime bars, settlement leg held fixed.
    pl = entry_pattern_placebo(bk, q12, fixed=is_close)
    print(
        f"entry-pattern placebo on the joint book (settlement leg fixed, the one daytime "
        f"short redrawn among the eleven daytime bars), {int(pl['draws'])} draws: "
        f"percentile {pl['pctile_mid']:.1f} mid, {pl['pctile_crossed']:.1f} crossed; "
        f"median draw {pl.get('placebo_median_mid', float('nan')):+.3f} mid, "
        f"{pl.get('placebo_median_crossed', float('nan')):+.3f} crossed"
    )
    tab.loc["G both lines", "placebo_pctile_mid"] = pl["pctile_mid"]
    tab.loc["G both lines", "placebo_pctile_crossed"] = pl["pctile_crossed"]
    return tab


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Proposal 10 - Check 0 and the books that were never run."
    )
    ap.add_argument(
        "--out", default=str(OUT), help="output directory for CSVs and figures"
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    globals()["OUT"] = out

    t0 = time.time()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    rule("Frame and gate")
    work, meta = build_work(t0)
    bk = Book(work)
    print(
        f"cache {meta['cache']}: {meta['bars']:,} bars, {meta['days']} days, "
        f"{meta['days_post_warmup']} with a signal, "
        f"{meta['bars_no_signal']} bars with no signal on "
        f"{meta['dates_incomplete_profile']} dates with an incomplete profile, "
        f"{meta['dropped_at_join']:,} packages dropped at the forecast join"
    )
    gate = run_gate(bk, work)
    gate.to_csv(out / "10_gate.csv")
    caus = assert_causal(work)
    caus.to_csv(out / "10_causality.csv", index=False)

    rule("Check 0 - is the oracle almost always short before 15:30?")
    c0 = check0(work, bk)
    c0["by_clock"].to_csv(out / "10_check0_by_clock.csv", index=False)
    c0["pooled"].to_csv(out / "10_check0_pooled.csv", index=False)
    c0["quantiles"].to_csv(out / "10_check0_gap_quantiles.csv", index=False)
    c0["oracle_cost"].to_csv(out / "10_check0_oracle_cost.csv")
    c0["oracle_clock"].to_csv(out / "10_check0_oracle_by_clock.csv", index=False)

    rule("A - nowcast the bar in flight")
    a = book_a(work)
    a.to_csv(out / "10_A_unconstructible.csv", index=False)

    rule("B - synthesize a 30-minute implied, do not slice one")
    b_tab, b_diag, syn, decay = book_b(work, bk)
    b_tab.to_csv(out / "10_B_synthetic.csv")
    b_diag.to_csv(out / "10_B_diagnostics.csv", index=False)

    rule("C - a hurdle against the quoted half-spread")
    c_tab, decay_hat = book_c(work, bk, decay)
    c_tab.to_csv(out / "10_C_hurdle.csv")

    rule("D - sparse clocks, not twelve")
    d_tab, d_sel = book_d(work, bk)
    d_tab.to_csv(out / "10_D_sparse.csv")
    d_sel.to_csv(out / "10_D_clock_selection.csv", index=False)

    rule("E - a 0DTE against 1DTE calendar as a market 30-minute window")
    e = book_e()
    e.to_csv(out / "10_E_unconstructible.csv", index=False)

    rule("F - fade the implied, not realized minus implied")
    f_tab = book_f(work, bk)
    f_tab.to_csv(out / "10_F_fade_iv.csv")

    rule("G - the 15:00 bar stacked on the settlement leg")
    g_tab = book_g(work, bk)
    g_tab.to_csv(out / "10_G_two_line.csv")

    rule("Every book against the pre-registered comparison")
    allb = pd.concat(
        [c0["oracle_cost"], b_tab, c_tab, d_tab, f_tab, g_tab], axis=0, sort=False
    )
    bench = float(
        g_tab.loc["G line 2 only: 15:30 sign(s), cash-settled", "Sharpe_crossed_866"]
    )
    bench803 = float(
        g_tab.loc["G line 2 only: 15:30 sign(s), cash-settled", "Sharpe_crossed_803"]
    )
    # Books that read the future are listed for the ceiling they mark, never as
    # candidates: the four oracles read the bar's own realized variance and the
    # peeking vega line reads the next mark's implied volatility.
    allb["F_t"] = [
        not (str(i).startswith("oracle") or "peeking" in str(i)) for i in allb.index
    ]
    allb["beats 15:30 sign(s) crossed (866)"] = allb["Sharpe_crossed_866"] > bench
    allb["beats 15:30 sign(s) crossed (803)"] = allb["Sharpe_crossed_803"] > bench803
    allb["positive crossed (866)"] = allb["Sharpe_crossed_866"] > 0
    cols = [
        "F_t",
        "Sharpe_mid_866",
        "Sharpe_crossed_866",
        "Sharpe_mid_803",
        "Sharpe_crossed_803",
        "mean_day_crossed_866",
        "t_crossed_866",
        "maxDD_crossed_866",
        "crossings_day",
        "positive crossed (866)",
        "beats 15:30 sign(s) crossed (866)",
    ]
    print(
        f"pre-registered comparison: 15:30 sign(s) alone at the crossed spread, "
        f"{bench:+.4f} on 866 days and {bench803:+.4f} on 803"
    )
    print(allb[cols].to_string(float_format=lambda x: f"{x:+.4f}"))
    allb.to_csv(out / "10_books.csv")

    positives = [
        i
        for i in allb.index
        if bool(allb.loc[i, "F_t"]) and bool(allb.loc[i, "positive crossed (866)"])
    ]
    survivors = [
        i
        for i in allb.index
        if bool(allb.loc[i, "F_t"])
        and bool(allb.loc[i, "beats 15:30 sign(s) crossed (866)"])
        and bool(allb.loc[i, "beats 15:30 sign(s) crossed (803)"])
    ]
    print()
    print(
        "F_t-measurable books with a positive crossed Sharpe: "
        + (", ".join(positives) if positives else "NONE")
    )
    print(
        "of those, ones whose point estimate exceeds 15:30 sign(s) alone on both frames: "
        + (", ".join(survivors) if survivors else "NONE")
    )
    for s in survivors:
        lo = allb.loc[s].get("vs_line2_crossed_pctile_lo", np.nan)
        hi = allb.loc[s].get("vs_line2_crossed_pctile_hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            reading = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
            print(
                f"  {s}: paired dSharpe against 15:30 sign(s) alone at the crossed spread, "
                f"percentile 95% [{lo:+.3f}, {hi:+.3f}] - {reading}"
            )
    if not survivors:
        print("The instruction stands: trade 15:30 sign(s) only.")

    # one figure: the crossed cumulative of the surviving trade against the books
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    is_close = (work["hhmm"] == CLOSE).to_numpy(bool)
    pos_m = np.where(
        np.isfinite(work["s_matched"]), np.where(work["s_matched"] > 0, 1.0, -1.0), 0.0
    )
    curves = {
        "15:30 sign(s) alone": np.where(is_close, pos_m, 0.0),
        "the deck's hybrid, twelve bars": np.where(is_close, pos_m, -1.0),
        "B synthetic 30-min implied": np.where(
            np.isfinite(work["rv_hat"] - syn),
            np.where((work["rv_hat"] - syn) > 0, 1.0, -1.0),
            0.0,
        ),
        f"C always short, trailing-decay hurdle k={HURDLE_K_PREREG}": _q_hurdle_short(
            work, decay_hat, HURDLE_K_PREREG
        ),
        "D pre-registered sparse clocks": np.where(
            work["hhmm"].isin(SPARSE_CLOCKS).to_numpy(bool) & ~is_close,
            -1.0,
            np.where(is_close, pos_m, 0.0),
        ),
        "G 15:00 always short + 15:30 sign(s)": np.where(
            (work["hhmm"] == "15:00").to_numpy(bool),
            -1.0,
            np.where(is_close, pos_m, 0.0),
        ),
    }
    for name, q in curves.items():
        d = bk.daily(q, "crossed")
        ax.plot(d.index, d.cumsum(), lw=1.2, label=name)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel("cumulative return per unit of entry premium")
    ax.set_title("At the crossed spread, on the notebook's own frame")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "10_cum_crossed.png", dpi=140)
    plt.close(fig)
    tick(t0, f"wrote {len(list(out.glob('10_*')))} artefacts to {out}")


if __name__ == "__main__":
    main()
