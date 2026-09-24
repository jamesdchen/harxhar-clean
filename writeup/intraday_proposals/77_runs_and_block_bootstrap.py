"""Study 77 -- run lengths and a block bootstrap of the month-end long and the joint long-only book.

The 22 % month-end fraction was argued from an iid resample of the deck's 52
month-end returns.  This study replaces it: the observed run-length
distribution of losers and winners against independence, a CIRCULAR BLOCK
bootstrap of the monthly series (blocks of 3 and 6 month-ends) and of the
daily joint long-only close book (blocks of 21 sessions), the joint book's
growth-optimal and blow-up-safe fraction pairs, and the pessimistic
sensitivities of study 76.

Series (both decks; crossed = bought at the ask; hold to cash settlement):
  (i)   the month-end long, one return per month-end (the deck's last session
        of each calendar month);
  (ii)  the general long leg on s > 0 days;
  (iii) the JOINT long-only close book per session: month-end -> the month-end
        long at f_me; else s > 0 -> the general long at f_long; else flat;
        the full sign(s) book (long at the ask on s > 0, short at the bid on
        s <= 0, both at f_long) beside it for reference.

Gates: study 71's loader (R = exit/entry - 1 and s = rv_hat - iv_var
reproduce; bid <= entry <= ask), the month-end count against the deck's
calendar, and the block-1 bootstrap reproducing the iid resample.

Outputs: results/atm_straddle_intraday_holdclose/proposals/77/*.csv.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "77"
TAGS = ("sub_live_ridge", "blk2")
RECORD_TAG = "sub_live_ridge"
SEED = 77
N_PATHS = 20_000
N_PATHS_GRID = 5_000
YEARS = 5
MONTHS_PER_YEAR = 12
SESSIONS_PER_YEAR = int(asl.PERIODS_PER_YEAR)
MONTH_BLOCKS = (1, 3, 6)  # 1 = the iid resample of the earlier table
DAY_BLOCK = 21
N_IID_RUNS = 20_000
BOOT_B = 2_000
F_ME = (0.10, 0.15, 0.22, 0.30)
F_LONG = (0.016, 0.033)  # study 76's f*/4 and f*/2 of the general long leg
F_ME_GRID = tuple(np.round(np.arange(0.05, 0.55, 0.05), 2))
F_LONG_GRID = (0.0, 0.01, 0.02, 0.033, 0.05, 0.066)
DD_LEVELS = (0.25, 0.50, 0.75)
DEEP_DD = 0.39  # three average losers in a row at 22 %
DD_WINDOW_MONTHS = 12
MAX_LAG = 3
EDGE = 1e-12
IID_REFERENCE = {  # the table quoted for the 22 % decision (iid resample, f_me = 0.22; growth = log rate)
    "growth": 0.76,
    "dd50": 0.60,
    "dd75": 0.07,
    "loss1y": 0.16,
}
IID_TOL = 0.05


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p71 = _load(HERE / "71_conviction_sizing.py", "p71_conviction")
p76 = _load(HERE / "76_kelly_long_leg.py", "p76_kelly_long")


# ------------------------------------------------------------------ runs ----
def run_lengths(loss: np.ndarray) -> tuple[list[int], list[int]]:
    """Lengths of consecutive loser runs and consecutive winner runs."""
    losers: list[int] = []
    winners: list[int] = []
    n = 0
    cur: bool | None = None
    for v in loss:
        b = bool(v)
        if cur is None or b == cur:
            n += 1
            cur = b
        else:
            (losers if cur else winners).append(n)
            n, cur = 1, b
    if cur is not None:
        (losers if cur else winners).append(n)
    return losers, winners


def longest_run(loss: np.ndarray) -> int:
    losers, _ = run_lengths(loss)
    return max(losers) if losers else 0


def wald_wolfowitz(loss: np.ndarray) -> dict[str, float]:
    """Runs test on the loss indicator: fewer runs than independence = clustering."""
    x = np.asarray(loss, bool)
    n1, n0 = int(x.sum()), int((~x).sum())
    n = n1 + n0
    runs = 1 + int((x[1:] != x[:-1]).sum())
    mu = 2.0 * n1 * n0 / n + 1.0
    var = (mu - 1.0) * (mu - 2.0) / (n - 1.0)
    z = (runs - mu) / np.sqrt(var) if var > 0 else float("nan")
    return {
        "runs": float(runs),
        "expected_runs": float(mu),
        "z": float(z),
        "p_two_sided": float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
        "p_clustering_one_sided": float(norm.cdf(z))
        if np.isfinite(z)
        else float("nan"),
    }


def iid_longest_run(p_loss: float, n: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.random((N_IID_RUNS, n)) < p_loss
    # longest run of True per row
    out: np.ndarray = np.zeros(N_IID_RUNS, int)
    cur: np.ndarray = np.zeros(N_IID_RUNS, int)
    for j in range(n):
        cur = np.where(draws[:, j], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = float((x**2).sum())
    if denom <= 0 or lag >= len(x):
        return float("nan")
    return float((x[lag:] * x[:-lag]).sum() / denom)


def block_ci(
    x: np.ndarray, stat, block: int, rng: np.random.Generator
) -> tuple[float, float]:
    n = len(x)
    idx = asl.circular_block_bootstrap_idx(rng, n, block, BOOT_B)
    vals = np.array([stat(x[idx[b]]) for b in range(BOOT_B)], float)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------ paths ---
def block_paths(
    x: np.ndarray, n_steps: int, block: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular block bootstrap of a series into n_paths x n_steps draws."""
    n = len(x)
    n_blocks = -(-n_steps // block)
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(
        n_paths, -1
    )[:, :n_steps]
    return x[idx]


def path_stats(
    step_ret: np.ndarray, steps_per_year: int, dd_window: int | None = None
) -> dict[str, float]:
    """Growth, terminal wealth, drawdowns, one-year loss odds of wealth paths.

    ``step_ret`` is the per-step fractional P&L (already multiplied by f).
    """
    step = 1.0 + step_ret
    ruined = (step <= 0).any(axis=1)
    lg = np.log(np.maximum(step, EDGE))
    cum = np.cumsum(lg, axis=1)
    wealth = np.exp(cum)
    peak = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)
    dd = 1.0 - np.exp(cum - peak)
    worst = dd.max(axis=1)
    final = np.where(ruined, 0.0, wealth[:, -1])
    y1 = np.exp(cum[:, steps_per_year - 1])
    n_years = step_ret.shape[1] / steps_per_year
    out = {
        "growth_median_per_yr": float(np.median(final) ** (1.0 / n_years) - 1.0),
        # the continuously compounded rate (the number quoted in the 22 % table) and its exp
        "loggrowth_per_yr": float(lg.mean() * steps_per_year),
        "growth_meanlog_per_yr": float(np.exp(lg.mean() * steps_per_year) - 1.0),
        "terminal_median": float(np.median(final)),
        "terminal_p05": float(np.quantile(final, 0.05)),
        "prob_ruin": float(ruined.mean()),
        "p_loss_1y": float((y1 < 1.0).mean()),
        "p05_1y": float(np.quantile(y1, 0.05)),
    }
    for lvl in DD_LEVELS:
        out[f"p_dd_gt_{int(lvl * 100)}"] = float((worst >= lvl).mean())
    if dd_window is not None:
        # worst peak-to-trough inside any window of dd_window steps
        w = dd_window
        n_steps = cum.shape[1]
        deep = np.zeros(cum.shape[0], bool)
        for s in range(0, n_steps - w + 1):
            seg = cum[:, s : s + w]
            seg_peak = np.maximum.accumulate(seg, axis=1)
            seg_dd = 1.0 - np.exp(seg - seg_peak)
            deep |= seg_dd.max(axis=1) >= DEEP_DD
        out[f"p_dd_ge_{int(DEEP_DD * 100)}_in_{w}"] = float(deep.mean())
    return out


def longest_loser_in_paths(step_ret: np.ndarray) -> dict[str, float]:
    loss = step_ret < 0
    out: np.ndarray = np.zeros(loss.shape[0], int)
    cur: np.ndarray = np.zeros(loss.shape[0], int)
    for j in range(loss.shape[1]):
        cur = np.where(loss[:, j], cur + 1, 0)
        out = np.maximum(out, cur)
    return {
        "longest_loser_run_median": float(np.median(out)),
        "longest_loser_run_p95": float(np.percentile(out, 95)),
    }


# ------------------------------------------------------------------ inputs --
def load(tag: str) -> pd.DataFrame:
    d = p71.load_deck(tag)
    fl = p71.calendar_flags(d.index).reindex(d.index)
    d["month_end"] = fl["month_end"].to_numpy(bool)
    d["buy"] = d["s"] > 0
    return d


def joint_daily(d: pd.DataFrame, f_me: float, f_long: float) -> np.ndarray:
    """Per-session fractional P&L of the long-only close book."""
    r = d["r_ask"].to_numpy(float)
    me = d["month_end"].to_numpy(bool)
    buy = d["buy"].to_numpy(bool)
    x = np.zeros(len(d))
    x[me] = f_me * r[me]
    gen = buy & ~me
    x[gen] = f_long * r[gen]
    return x


def sign_book_daily(d: pd.DataFrame, f: float) -> np.ndarray:
    """The full sign(s) book at fraction f: long at the ask on s > 0, short at the bid otherwise."""
    buy = d["buy"].to_numpy(bool)
    return np.where(
        buy, f * d["r_ask"].to_numpy(float), -f * d["r_bid"].to_numpy(float)
    )


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    fmt = "{:.4f}".format
    rng = np.random.default_rng(SEED)

    decks = {t: load(t) for t in TAGS}
    d0 = decks[RECORD_TAG]
    me_n = int(d0["month_end"].sum())
    # every calendar month on the deck contributes exactly one month-end session
    months = d0.index.to_period("M")
    assert me_n == months.nunique(), (me_n, months.nunique())
    print(
        f"deck {len(d0)} sessions {d0.index.min().date()} .. {d0.index.max().date()}: "
        f"{me_n} month-ends (one per calendar month; the live README's 70 replayed month-ends "
        f"run to 2025-12 on the ledger window, the deck stops 2024-04-30)"
    )
    for t, d in decks.items():
        print(
            f"GATE {t}: R, s, bid <= entry <= ask reproduce on {len(d)} days; month-ends {int(d['month_end'].sum())}, "
            f"buys {int(d['buy'].sum())}, month-end long at the ask: hit {float((d.loc[d['month_end'], 'r_ask'] > 0).mean()):.3f} "
            f"mean {float(d.loc[d['month_end'], 'r_ask'].mean()):+.3f}; general long: hit "
            f"{float((d.loc[d['buy'], 'r_ask'] > 0).mean()):.3f} mean {float(d.loc[d['buy'], 'r_ask'].mean()):+.3f}"
        )

    # ---------------------------------------------------------------- A -----
    print("\nA  run lengths against independence")
    rows_runs, rows_ac, rows_year = [], [], []
    for t, d in decks.items():
        series = {
            "month-end long": (d.loc[d["month_end"], "r_ask"].to_numpy(float), 3),
            "general long leg (s > 0)": (
                d.loc[d["buy"], "r_ask"].to_numpy(float),
                DAY_BLOCK,
            ),
        }
        for label, (r, block) in series.items():
            loss = r < 0
            losers, winners = run_lengths(loss)
            p_loss = float(loss.mean())
            iid = iid_longest_run(p_loss, len(r), rng)
            ww = wald_wolfowitz(loss)
            obs_longest = longest_run(loss)
            row = {
                "deck": t,
                "series": label,
                "n": len(r),
                "p_loss": p_loss,
                "longest_loser_run": obs_longest,
                "longest_winner_run": max(winners) if winners else 0,
                "p_iid_longest_ge_observed": float((iid >= obs_longest).mean()),
                "iid_longest_median": float(np.median(iid)),
                "iid_longest_p95": float(np.percentile(iid, 95)),
                **{
                    f"loser_runs_len_{k}": int(sum(1 for v in losers if v == k))
                    for k in range(1, 6)
                },
                "loser_runs_len_6plus": int(sum(1 for v in losers if v >= 6)),
                **{
                    f"winner_runs_len_{k}": int(sum(1 for v in winners if v == k))
                    for k in range(1, 4)
                },
                "winner_runs_len_4plus": int(sum(1 for v in winners if v >= 4)),
                **{f"ww_{k}": v for k, v in ww.items()},
            }
            # analytic: expected number of loser runs of length >= k under independence
            for k in (2, 3, 4):
                row[f"iid_expected_loser_runs_ge_{k}"] = float(
                    (len(r) - k + 1) * p_loss**k * (1 - p_loss)
                    + p_loss**k  # a run may end at the sequence's end
                )
                row[f"obs_loser_runs_ge_{k}"] = int(sum(1 for v in losers if v >= k))
            rows_runs.append(row)
            for lag in range(1, MAX_LAG + 1):
                for name, x in (("R", r), ("loss", loss.astype(float))):
                    lo, hi = block_ci(
                        x, lambda v, lag=lag: autocorr(v, lag), block, rng
                    )
                    rows_ac.append(
                        {
                            "deck": t,
                            "series": label,
                            "variable": name,
                            "lag": lag,
                            "autocorr": autocorr(x, lag),
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )
        me = d[d["month_end"]]
        for yr, g in me.groupby(me.index.year):
            rows_year.append(
                {
                    "deck": t,
                    "year": int(yr),
                    "n": len(g),
                    "hit": float((g["r_ask"] > 0).mean()),
                    "mean_R": float(g["r_ask"].mean()),
                    "sum_R": float(g["r_ask"].sum()),
                }
            )
    a_runs = pd.DataFrame(rows_runs)
    a_ac = pd.DataFrame(rows_ac)
    a_year = pd.DataFrame(rows_year)
    a_runs.to_csv(OUT / "a_runs.csv", index=False)
    a_ac.to_csv(OUT / "a_autocorr.csv", index=False)
    a_year.to_csv(OUT / "a_monthend_hit_by_year.csv", index=False)
    cols = [
        "deck",
        "series",
        "n",
        "p_loss",
        "longest_loser_run",
        "p_iid_longest_ge_observed",
        "iid_longest_median",
        "iid_longest_p95",
        "obs_loser_runs_ge_2",
        "iid_expected_loser_runs_ge_2",
        "obs_loser_runs_ge_3",
        "iid_expected_loser_runs_ge_3",
        "ww_runs",
        "ww_expected_runs",
        "ww_z",
        "ww_p_two_sided",
        "ww_p_clustering_one_sided",
    ]
    print(a_runs[cols].to_string(index=False, float_format=fmt))
    print(
        "\n   autocorrelation with block-bootstrap intervals (blocks 3 monthly / 21 daily)"
    )
    print(a_ac.to_string(index=False, float_format=fmt))
    print("\n   month-end long, hit by year")
    print(a_year.to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- B -----
    print("\nB  the month-end long alone: block bootstrap of the monthly series")
    rows_b = []
    for t, d in decks.items():
        r_me = d.loc[d["month_end"], "r_ask"].to_numpy(float)
        for block in MONTH_BLOCKS:
            for f in F_ME:
                x = block_paths(r_me, YEARS * MONTHS_PER_YEAR, block, N_PATHS, rng) * f
                st = path_stats(x, MONTHS_PER_YEAR, dd_window=DD_WINDOW_MONTHS)
                st.update(longest_loser_in_paths(x))
                rows_b.append({"deck": t, "block_months": block, "f_me": f, **st})
    b = pd.DataFrame(rows_b)
    b.to_csv(OUT / "b_monthend_block_paths.csv", index=False)
    # gate: the block-1 row reproduces the quoted iid table at f_me = 0.22
    g = b[
        (b["deck"] == RECORD_TAG) & (b["block_months"] == 1) & (b["f_me"] == 0.22)
    ].iloc[0]
    got = {
        "growth": g["loggrowth_per_yr"],
        "dd50": g["p_dd_gt_50"],
        "dd75": g["p_dd_gt_75"],
        "loss1y": g["p_loss_1y"],
    }
    dev = max(abs(got[k] - IID_REFERENCE[k]) for k in got)
    assert dev < IID_TOL, (got, IID_REFERENCE)
    print(
        f"GATE  block 1 at f_me 0.22 reproduces the quoted iid table within {IID_TOL:g}: "
        + ", ".join(f"{k} {got[k]:.3f} (quoted {IID_REFERENCE[k]:.2f})" for k in got)
    )
    bcols = [
        "deck",
        "block_months",
        "f_me",
        "loggrowth_per_yr",
        "growth_meanlog_per_yr",
        "terminal_median",
        "terminal_p05",
        "p_dd_gt_25",
        "p_dd_gt_50",
        "p_dd_gt_75",
        "p_loss_1y",
        "p05_1y",
        "longest_loser_run_median",
        "longest_loser_run_p95",
        f"p_dd_ge_{int(DEEP_DD * 100)}_in_{DD_WINDOW_MONTHS}",
    ]
    print(b[bcols].to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- C -----
    print(
        "\nC  the joint long-only close book: daily block bootstrap (block 21 sessions)"
    )
    rows_c = []
    n_steps = YEARS * SESSIONS_PER_YEAR
    for t, d in decks.items():
        me = d["month_end"].to_numpy(bool)
        gen = d["buy"].to_numpy(bool) & ~me
        for f_me in F_ME:
            for f_long in F_LONG:
                x = joint_daily(d, f_me, f_long)
                n = len(x)
                n_blocks = -(-n_steps // DAY_BLOCK)
                starts = rng.integers(0, n, size=(N_PATHS, n_blocks))
                idx = (
                    (starts[:, :, None] + np.arange(DAY_BLOCK)[None, None, :]) % n
                ).reshape(N_PATHS, -1)[:, :n_steps]
                paths_x = x[idx]
                st = path_stats(paths_x, SESSIONS_PER_YEAR)
                lg = np.log(np.maximum(1.0 + paths_x, EDGE))
                tot = lg.sum()
                share_me = float(lg[me[idx]].sum() / tot) if tot != 0 else float("nan")
                # legs' contributions by calendar month in the ORIGINAL series
                lg0 = np.log(np.maximum(1.0 + x, EDGE))
                per_month = (
                    pd.DataFrame(
                        {"me": np.where(me, lg0, 0.0), "gen": np.where(gen, lg0, 0.0)},
                        index=d.index.to_period("M"),
                    )
                    .groupby(level=0)
                    .sum()
                )
                corr = float(per_month["me"].corr(per_month["gen"]))
                me_loser = per_month["me"] < 0
                p_gen_loss = float((per_month["gen"] < 0).mean())
                p_gen_loss_given_me_loser = (
                    float((per_month.loc[me_loser, "gen"] < 0).mean())
                    if me_loser.any()
                    else float("nan")
                )
                rows_c.append(
                    {
                        "deck": t,
                        "book": "long-only",
                        "f_me": f_me,
                        "f_long": f_long,
                        **st,
                        "share_growth_month_ends": share_me,
                        "corr_legs_by_month": corr,
                        "p_gen_month_loss": p_gen_loss,
                        "p_gen_month_loss_given_me_loser": p_gen_loss_given_me_loser,
                    }
                )
        for f_long in F_LONG:
            x = sign_book_daily(d, f_long)
            paths_x = block_paths(x, n_steps, DAY_BLOCK, N_PATHS, rng)
            st = path_stats(paths_x, SESSIONS_PER_YEAR)
            rows_c.append(
                {
                    "deck": t,
                    "book": "sign(s) both legs (reference)",
                    "f_me": np.nan,
                    "f_long": f_long,
                    **st,
                }
            )
    c = pd.DataFrame(rows_c)
    c.to_csv(OUT / "c_joint_block_paths.csv", index=False)
    ccols = [
        "deck",
        "book",
        "f_me",
        "f_long",
        "growth_meanlog_per_yr",
        "terminal_median",
        "terminal_p05",
        "p_dd_gt_50",
        "p_dd_gt_75",
        "p_loss_1y",
        "p05_1y",
        "share_growth_month_ends",
        "corr_legs_by_month",
        "p_gen_month_loss",
        "p_gen_month_loss_given_me_loser",
    ]
    print(c[ccols].to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- D -----
    print(
        "\nD  Kelly of the joint book on the (f_me, f_long) grid (block 21, "
        f"{N_PATHS_GRID} paths): growth-optimal and blow-up-safe (max p05 terminal)"
    )
    rows_d = []
    for t, d in decks.items():
        for f_me in F_ME_GRID:
            for f_long in F_LONG_GRID:
                x = joint_daily(d, float(f_me), float(f_long))
                paths_x = block_paths(x, n_steps, DAY_BLOCK, N_PATHS_GRID, rng)
                st = path_stats(paths_x, SESSIONS_PER_YEAR)
                rows_d.append(
                    {"deck": t, "f_me": float(f_me), "f_long": float(f_long), **st}
                )
    dgrid = pd.DataFrame(rows_d)
    dgrid.to_csv(OUT / "d_kelly_grid.csv", index=False)
    for t in TAGS:
        sub = dgrid[dgrid["deck"] == t]
        best_g = sub.loc[sub["growth_meanlog_per_yr"].idxmax()]
        best_p05 = sub.loc[sub["terminal_p05"].idxmax()]
        print(
            f"  {t}: growth-optimal (f_me, f_long) = ({best_g['f_me']:.2f}, {best_g['f_long']:.3f}) "
            f"growth {best_g['growth_meanlog_per_yr']:+.1%}/yr, p05 {best_g['terminal_p05']:.2f}x, "
            f"P(DD>50%) {best_g['p_dd_gt_50']:.2f} | blow-up-safe (max p05) = "
            f"({best_p05['f_me']:.2f}, {best_p05['f_long']:.3f}) growth {best_p05['growth_meanlog_per_yr']:+.1%}/yr, "
            f"p05 {best_p05['terminal_p05']:.2f}x, P(DD>50%) {best_p05['p_dd_gt_50']:.2f}"
        )
    # the growth surface along f_me at f_long = 0.033 for the record deck
    sub = dgrid[(dgrid["deck"] == RECORD_TAG) & (dgrid["f_long"] == 0.033)]
    print("  growth by f_me at f_long 0.033 (" + RECORD_TAG + "):")
    print(
        sub[
            [
                "f_me",
                "growth_meanlog_per_yr",
                "terminal_p05",
                "p_dd_gt_50",
                "p_dd_gt_75",
                "p_loss_1y",
            ]
        ].to_string(index=False, float_format=fmt)
    )

    # ---------------------------------------------------------------- E -----
    print(
        "\nE  sensitivity (record deck): mean month-end R shrunk by 1 SE; the two largest month-ends removed"
    )
    rows_e = []
    d = d0.copy()
    r_me = d.loc[d["month_end"], "r_ask"].to_numpy(float)
    se = p76.se_mean(r_me)
    variants = {
        "as observed": d,
        "month-end mean - 1 SE": None,
        "two largest month-ends removed": None,
    }
    dd1 = d.copy()
    dd1.loc[dd1["month_end"], "r_ask"] = dd1.loc[dd1["month_end"], "r_ask"] - se
    variants["month-end mean - 1 SE"] = dd1
    dd2 = d.copy()
    top2 = dd2.loc[dd2["month_end"], "r_ask"].nlargest(2).index
    dd2 = dd2.drop(index=top2)
    variants["two largest month-ends removed"] = dd2
    for name, dv in variants.items():
        assert dv is not None
        r_v = dv.loc[dv["month_end"], "r_ask"].to_numpy(float)
        for block in (3,):
            for f in F_ME:
                x = block_paths(r_v, YEARS * MONTHS_PER_YEAR, block, N_PATHS, rng) * f
                st = path_stats(x, MONTHS_PER_YEAR, dd_window=DD_WINDOW_MONTHS)
                rows_e.append(
                    {
                        "variant": name,
                        "book": "month-end alone",
                        "block": block,
                        "f_me": f,
                        "f_long": np.nan,
                        "kelly_f_me": p76.kelly(r_v),
                        **st,
                    }
                )
        for f_me in F_ME:
            x = joint_daily(dv, f_me, 0.033)
            paths_x = block_paths(x, n_steps, DAY_BLOCK, N_PATHS, rng)
            st = path_stats(paths_x, SESSIONS_PER_YEAR)
            rows_e.append(
                {
                    "variant": name,
                    "book": "joint, f_long 0.033",
                    "block": DAY_BLOCK,
                    "f_me": f_me,
                    "f_long": 0.033,
                    "kelly_f_me": np.nan,
                    **st,
                }
            )
    e = pd.DataFrame(rows_e)
    e.to_csv(OUT / "e_sensitivity.csv", index=False)
    ecols = [
        "variant",
        "book",
        "f_me",
        "kelly_f_me",
        "growth_meanlog_per_yr",
        "terminal_median",
        "terminal_p05",
        "p_dd_gt_50",
        "p_dd_gt_75",
        "p_loss_1y",
    ]
    print(e[ecols].to_string(index=False, float_format=fmt))

    # ------------------------------------------------------------ verdict ----
    a0 = a_runs[
        (a_runs["deck"] == RECORD_TAG) & (a_runs["series"] == "month-end long")
    ].iloc[0]
    a1 = a_runs[
        (a_runs["deck"] == RECORD_TAG) & (a_runs["series"].str.startswith("general"))
    ].iloc[0]
    b3 = b[
        (b["deck"] == RECORD_TAG) & (b["block_months"] == 3) & (b["f_me"] == 0.22)
    ].iloc[0]
    b6 = b[
        (b["deck"] == RECORD_TAG) & (b["block_months"] == 6) & (b["f_me"] == 0.22)
    ].iloc[0]
    b1 = b[
        (b["deck"] == RECORD_TAG) & (b["block_months"] == 1) & (b["f_me"] == 0.22)
    ].iloc[0]
    sub = dgrid[dgrid["deck"] == RECORD_TAG]
    bg = sub.loc[sub["growth_meanlog_per_yr"].idxmax()]
    bp = sub.loc[sub["terminal_p05"].idxmax()]
    print(
        "\nVERDICT  (A) losers cluster? month-end long: longest loser run "
        f"{int(a0['longest_loser_run'])} (iid P(>= observed) {a0['p_iid_longest_ge_observed']:.2f}), "
        f"runs test z {a0['ww_z']:+.2f} p {a0['ww_p_two_sided']:.2f}; general long leg: longest {int(a1['longest_loser_run'])} "
        f"(iid P {a1['p_iid_longest_ge_observed']:.2f}), runs test z {a1['ww_z']:+.2f} p {a1['ww_p_two_sided']:.2f}; "
        f"(B) month-end long at 0.22: P(DD > 50%) iid {b1['p_dd_gt_50']:.2f} -> block-3 {b3['p_dd_gt_50']:.2f} / block-6 {b6['p_dd_gt_50']:.2f}, "
        f"P(DD > 75%) {b1['p_dd_gt_75']:.2f} -> {b3['p_dd_gt_75']:.2f} / {b6['p_dd_gt_75']:.2f}, "
        f"1-yr P(loss) {b1['p_loss_1y']:.2f} -> {b3['p_loss_1y']:.2f} / {b6['p_loss_1y']:.2f}, "
        f"growth {b1['growth_meanlog_per_yr']:+.0%} -> {b3['growth_meanlog_per_yr']:+.0%} / {b6['growth_meanlog_per_yr']:+.0%}; "
        f"(D) joint book growth-optimal (f_me, f_long) = ({bg['f_me']:.2f}, {bg['f_long']:.3f}), "
        f"blow-up-safe = ({bp['f_me']:.2f}, {bp['f_long']:.3f})."
    )


if __name__ == "__main__":
    main()
