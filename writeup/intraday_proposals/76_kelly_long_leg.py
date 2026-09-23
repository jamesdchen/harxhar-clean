"""Study 76 -- Kelly of the LONG leg of sign(s): the Heaviside part 1{s > 0}.

The 15:30 sign(s) rule is q = sign(rv_hat - iv_var).  Its long half is the
Heaviside indicator 1{s > 0}: on those days the book BUYS the nearest-OTM
0DTE straddle at the ask and holds it to cash settlement.  Per premium dollar
the return R = exit / ask - 1 is bounded below at -1 (the straddle expires
worthless) and fat-tailed above, so the long leg is the textbook Kelly
object: growth g(f) = E log(1 + f R) with f the fraction of capital paid as
premium, Kelly f* the root of E[R / (1 + f R)] = 0 (study 66's machinery).
The live runner's premium sizing (2026-09-23) pays the stress fraction, 10 %
of capital, on the month-end long; this study asks where that sits.

Questions, stated before the results:
  A  f* of the long leg per buy day, crossed (at the ask) as the number of
     record and mid beside it; half-Kelly; growth at f*, f*/2 and the live
     10 %; day-block bootstrap interval on f*; the pessimistic f* with the
     mean R shrunk by one and two standard errors; both halves; both decks.
  B  Fragility: f* without the top 5 / 10 / 20 R days; with extra worthless
     expiries (R = -1) injected at the observed rate and twice it; the
     sensitivity of f* to the mean R.  How much of f* is the 20 days.
  C  The month-end long on its own (the override the runner trades), the
     third-Friday and FOMC-day buys for comparison (small n, said so).
  D  The short leg's Kelly BOUND for the record: its loss per premium dollar
     is unbounded, so f < 1 / |min R_short|.
  E  10,000 five-year day-block paths at f in {f*/4, f*/2, f*, 0.10, 0.20}:
     terminal wealth, drawdown probabilities, and the growth-optimal f
     re-estimated on the paths as a gate on the machinery.
  F  f* within the top and bottom tercile of the trailing QLIKE_63 (studies
     72 / 75) -- descriptive, not a rule.

Gates: R = exit / entry - 1 and s = rv_hat - iv_var reproduce on both decks
(study 71's loader); bid <= entry <= ask on every day; the path ensemble's
growth-optimal f agrees with the analytic f*.

Outputs: results/atm_straddle_intraday_holdclose/proposals/76/
  a_kelly.csv, b_fragility.csv, c_calendar.csv, d_short_bound.csv,
  e_paths.csv, f_regime.csv
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
from live.ibkr.config import Config  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "76"
TAGS = ("sub_live_ridge", "blk2")
RECORD_TAG = "sub_live_ridge"
LIVE_FRACTION = float(Config().stress_fraction)  # the runner's premium fraction
BOOT_B, BLOCK, SEED = 2000, 21, 76
SPLIT = pd.Timestamp("2022-01-01")
HALVES = (("2020-21", None, SPLIT), ("2022-24", SPLIT, None))
TOP_REMOVE = (5, 10, 20)
TAIL_MULTIPLES = (1.0, 2.0)
WORTHLESS_TOL = 1e-9  # exit == 0 -> R at the ask == -1 exactly
YEARS = 5
N_PATHS = 10_000
DD_LEVELS = (0.25, 0.50)
F_GRID = np.linspace(0.0, 1.0, 201)[1:]  # the growth-optimal f on the paths
GATE_PATH_TOL = 0.05  # |f*_paths - f*| on a 0.005 grid
TRAIL_QLIKE = 63
N_T = 3
EDGE = 1e-12


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p66 = _load(HERE / "66_kelly_sizing.py", "p66_kelly")
p71 = _load(HERE / "71_conviction_sizing.py", "p71_conviction")
p72 = _load(HERE / "72_vol_surprise_clustering.py", "p72_surprise")


# ------------------------------------------------------------------ kelly ---
def kelly(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    if r.size == 0:
        return float("nan")
    return float(p66.kelly(r, np.ones(r.size)))


def growth_per_day(r: np.ndarray, f: float) -> float:
    return float(p66.growth(np.asarray(r, float), np.ones(len(r)), f))


def annual(g_per_buy: float, buys_per_year: float) -> float:
    """Annual growth rate from the log growth per buy day and the realised buy rate."""
    return (
        float(np.exp(g_per_buy * buys_per_year) - 1.0)
        if np.isfinite(g_per_buy)
        else float("nan")
    )


def se_mean(r: np.ndarray) -> float:
    return float(np.std(r, ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")


def pessimistic_kelly(r: np.ndarray, k: float) -> float:
    return kelly(r - k * se_mean(r))


def block_ci_kelly(
    daily: np.ndarray, mask: np.ndarray, seed_tag: int
) -> tuple[float, float]:
    """Day-block bootstrap of the whole session series; f* on the resampled buys."""
    n = daily.size
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, seed_tag, n]), n, BLOCK, BOOT_B
    )
    fs = np.empty(BOOT_B)
    for b in range(BOOT_B):
        rows = idx[b]
        r = daily[rows][mask[rows]]
        fs[b] = kelly(r) if r.size else np.nan
    lo, hi = np.nanpercentile(fs, [2.5, 97.5])
    return float(lo), float(hi)


def sensitivity(r: np.ndarray) -> float:
    """d f* / d(mean R), by a symmetric shift of one standard error."""
    d = se_mean(r)
    return float((kelly(r + d) - kelly(r - d)) / (2.0 * d))


# ------------------------------------------------------------------ inputs --
def load(tag: str) -> pd.DataFrame:
    d = p71.load_deck(tag)
    rv = p71.realized_last_bar(tag).reindex(d.index)
    assert rv.notna().all(), f"{tag}: realized last bar missing on the deck"
    q = rv / d["rv_hat"]
    d["qlike"] = q - np.log(q) - 1.0
    d["qlike_trail"] = p72.trailing(d["qlike"], TRAIL_QLIKE)
    d["buy"] = d["s"] > 0
    return d


def kelly_row(
    label: str, tag: str, price: str, r: np.ndarray, buys_per_year: float
) -> dict[str, float | str | int]:
    f = kelly(r)
    return {
        "deck": tag,
        "price": price,
        "sample": label,
        "n_buys": int(r.size),
        "hit": float((r > 0).mean()),
        "mean_R": float(r.mean()),
        "kelly_f": f,
        "half_kelly": f / 2.0,
        "ann_growth_at_f": annual(growth_per_day(r, f), buys_per_year),
        "ann_growth_at_half": annual(growth_per_day(r, f / 2.0), buys_per_year),
        "ann_growth_at_live": annual(growth_per_day(r, LIVE_FRACTION), buys_per_year),
        "live_over_kelly": LIVE_FRACTION / f if f > 0 else float("inf"),
        "kelly_pess_1se": pessimistic_kelly(r, 1.0),
        "kelly_pess_2se": pessimistic_kelly(r, 2.0),
    }


# ------------------------------------------------------------------- paths --
def path_table(
    daily: np.ndarray, f: float, rng: np.random.Generator
) -> dict[str, float]:
    """N_PATHS five-year circular-block paths of the DAILY long-leg return at f."""
    n_days = YEARS * int(asl.PERIODS_PER_YEAR)
    n_blocks = -(-n_days // BLOCK)
    starts = rng.integers(0, daily.size, size=(N_PATHS, n_blocks))
    idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % daily.size).reshape(
        N_PATHS, -1
    )[:, :n_days]
    step = 1.0 + f * daily[idx]
    ruined = (step <= 0).any(axis=1)
    cum = np.cumsum(np.log(np.maximum(step, EDGE)), axis=1)
    peak = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)
    worst = (np.exp(cum - peak) - 1.0).min(axis=1)
    final = np.where(ruined, 0.0, np.exp(cum[:, -1]))
    out = {
        "f": f,
        "median_terminal_wealth": float(np.median(final)),
        "p05_terminal_wealth": float(np.quantile(final, 0.05)),
        "median_annual_growth": float(np.median(final) ** (1.0 / YEARS) - 1.0),
        "p05_annual_growth": float(np.quantile(final, 0.05) ** (1.0 / YEARS) - 1.0),
        "prob_below_start": float((final < 1.0).mean()),
        "prob_ruin": float(ruined.mean()),
    }
    for lvl in DD_LEVELS:
        out[f"prob_drawdown_{int(lvl * 100)}pct"] = float((worst <= -lvl).mean())
    return out


def growth_optimal_on_paths(daily: np.ndarray, rng: np.random.Generator) -> float:
    """argmax over F_GRID of the mean log terminal wealth on one path ensemble."""
    n_days = YEARS * int(asl.PERIODS_PER_YEAR)
    n_blocks = -(-n_days // BLOCK)
    starts = rng.integers(0, daily.size, size=(N_PATHS, n_blocks))
    idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % daily.size).reshape(
        N_PATHS, -1
    )[:, :n_days]
    x = daily[idx]
    best_f, best_g = float("nan"), -np.inf
    for f in F_GRID:
        step = 1.0 + f * x
        if (step <= 0).any():
            break
        g = float(np.log(step).sum(axis=1).mean())
        if g > best_g:
            best_f, best_g = float(f), g
    return best_f


# -------------------------------------------------------------------- main --
def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    fmt = "{:.4f}".format

    decks = {t: load(t) for t in TAGS}
    d0 = decks[RECORD_TAG]
    flags = p71.calendar_flags(d0.index)
    n_sessions = len(d0)
    years = n_sessions / asl.PERIODS_PER_YEAR
    print(
        f"deck: {n_sessions} sessions {d0.index.min().date()} .. {d0.index.max().date()} "
        f"({years:.2f} years at {asl.PERIODS_PER_YEAR:.0f}/yr); live premium fraction {LIVE_FRACTION:g}"
    )
    for t, d in decks.items():
        print(
            f"GATE {t}: R and s reproduce, bid <= entry <= ask on {len(d)} days; "
            f"buys {int(d['buy'].sum())}, hit at the ask {float((d.loc[d['buy'], 'r_ask'] > 0).mean()):.3f}"
        )

    # ---------------------------------------------------------------- A -----
    rows_a = []
    for t, d in decks.items():
        for price, col in (("crossed", "r_ask"), ("mid", "r_mid")):
            for label, lo, hi in (("all", None, None), *HALVES):
                m = d["buy"].to_numpy().copy()
                idx = d.index
                window = np.ones(len(idx), bool)
                if lo is not None:
                    window &= np.asarray(idx >= lo)
                if hi is not None:
                    window &= np.asarray(idx < hi)
                m &= window
                r = d.loc[m, col].to_numpy(float)
                n_sess = int(window.sum())
                bpy = r.size / (n_sess / asl.PERIODS_PER_YEAR)
                row = kelly_row(label, t, price, r, bpy)
                row["buys_per_year"] = bpy
                if label == "all":
                    lo_ci, hi_ci = block_ci_kelly(
                        d[col].to_numpy(float), m, seed_tag=hash((t, price)) % 10_000
                    )
                    row["kelly_ci_lo"], row["kelly_ci_hi"] = lo_ci, hi_ci
                rows_a.append(row)
    a = pd.DataFrame(rows_a)
    a.to_csv(OUT / "a_kelly.csv", index=False)
    print(
        "\nA  Kelly of the long leg per buy day (f = fraction of capital paid as premium)"
    )
    print(a.to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- B -----
    rows_b = []
    for t, d in decks.items():
        r = d.loc[d["buy"], "r_ask"].to_numpy(float)
        f_all = kelly(r)
        rows_b.append(
            {
                "deck": t,
                "variant": "all buys",
                "n": r.size,
                "kelly_f": f_all,
                "share_of_f": 1.0,
            }
        )
        order = np.argsort(r)[::-1]
        for k in TOP_REMOVE:
            rk = r[order[k:]]
            fk = kelly(rk)
            rows_b.append(
                {
                    "deck": t,
                    "variant": f"top {k} R days removed",
                    "n": rk.size,
                    "kelly_f": fk,
                    "share_of_f": fk / f_all if f_all > 0 else float("nan"),
                }
            )
        worthless = float((r <= -1.0 + WORTHLESS_TOL).mean())
        for mult in TAIL_MULTIPLES:
            rr, ww = p66.with_tail(r, np.array([-1.0]), worthless * mult)
            fw = float(p66.kelly(rr, ww))
            rows_b.append(
                {
                    "deck": t,
                    "variant": f"worthless expiries injected at {mult:g}x the observed rate ({worthless:.3f})",
                    "n": r.size,
                    "kelly_f": fw,
                    "share_of_f": fw / f_all if f_all > 0 else float("nan"),
                }
            )
        rows_b.append(
            {
                "deck": t,
                "variant": "d f* / d(mean R) (per unit of mean R)",
                "n": r.size,
                "kelly_f": sensitivity(r),
                "share_of_f": float("nan"),
            }
        )
    b = pd.DataFrame(rows_b)
    b.to_csv(OUT / "b_fragility.csv", index=False)
    print("\nB  fragility of f* (crossed) to the tail")
    print(b.to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- C -----
    rows_c = []
    for t, d in decks.items():
        fl = flags.reindex(d.index)
        sets = {
            "month-end long (every month-end, the override)": fl["month_end"].to_numpy(
                bool
            ),
            "third-Friday buys (s > 0)": (
                fl["third_friday"].to_numpy(bool) & d["buy"].to_numpy()
            ),
            "FOMC-day buys (s > 0)": (
                fl["fomc_day"].to_numpy(bool) & d["buy"].to_numpy()
            ),
        }
        for label, m in sets.items():
            r = d.loc[m, "r_ask"].to_numpy(float)
            if r.size < 2:
                continue
            f = kelly(r)
            lo_ci, hi_ci = block_ci_kelly(
                d["r_ask"].to_numpy(float), m, seed_tag=hash((t, label)) % 10_000
            )
            rows_c.append(
                {
                    "deck": t,
                    "set": label,
                    "n": r.size,
                    "hit": float((r > 0).mean()),
                    "mean_R": float(r.mean()),
                    "kelly_f": f,
                    "kelly_ci_lo": lo_ci,
                    "kelly_ci_hi": hi_ci,
                    "kelly_pess_1se": pessimistic_kelly(r, 1.0),
                    "kelly_pess_2se": pessimistic_kelly(r, 2.0),
                    "live_over_kelly": LIVE_FRACTION / f if f > 0 else float("inf"),
                    "small_n": bool(r.size < 30),
                }
            )
    c = pd.DataFrame(rows_c)
    c.to_csv(OUT / "c_calendar.csv", index=False)
    print("\nC  the calendar longs on their own (crossed)")
    print(c.to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- D -----
    rows_d = []
    for t, d in decks.items():
        sell = ~d["buy"].to_numpy()
        r_short = -d.loc[sell, "r_bid"].to_numpy(
            float
        )  # sold at the bid, per premium dollar
        worst = float(r_short.min())
        rows_d.append(
            {
                "deck": t,
                "n_sells": int(sell.sum()),
                "hit": float((r_short > 0).mean()),
                "mean_R_short": float(r_short.mean()),
                "min_R_short": worst,
                "feasible_bound_f": 1.0 / abs(worst),
                "kelly_f_short": kelly(r_short),
            }
        )
    dd = pd.DataFrame(rows_d)
    dd.to_csv(OUT / "d_short_bound.csv", index=False)
    print(
        "\nD  the short leg's bound (for the record; sold at the bid, per premium dollar)"
    )
    print(dd.to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- E -----
    daily = np.where(d0["buy"].to_numpy(), d0["r_ask"].to_numpy(float), 0.0)
    f_star = float(
        a[
            (a["deck"] == RECORD_TAG)
            & (a["price"] == "crossed")
            & (a["sample"] == "all")
        ]["kelly_f"].iloc[0]
    )
    rng = np.random.default_rng([SEED, 5])
    f_paths = growth_optimal_on_paths(daily, np.random.default_rng([SEED, 6]))
    gate_ok = abs(f_paths - f_star) <= GATE_PATH_TOL
    print(
        f"\nGATE E  growth-optimal f on {N_PATHS} five-year paths {f_paths:.3f} vs analytic f* {f_star:.3f}: "
        + ("agree" if gate_ok else "DISAGREE")
    )
    assert gate_ok, (f_paths, f_star)
    rows_e = [
        path_table(daily, f, rng)
        for f in (f_star / 4, f_star / 2, f_star, LIVE_FRACTION, 0.20)
    ]
    e = pd.DataFrame(rows_e)
    e.insert(0, "label", ["f*/4", "f*/2", "f*", f"live {LIVE_FRACTION:g}", "0.20"])
    e.to_csv(OUT / "e_paths.csv", index=False)
    print(
        f"\nE  {N_PATHS} five-year day-block paths of the long leg ({RECORD_TAG}, crossed)"
    )
    print(e.to_string(index=False, float_format=fmt))

    # ---------------------------------------------------------------- F -----
    rows_f = []
    for t, d in decks.items():
        b_ = d[d["buy"] & d["qlike_trail"].notna()]
        edges = np.quantile(b_["qlike_trail"], np.linspace(0, 1, N_T + 1))
        terc = np.clip(
            np.searchsorted(edges, b_["qlike_trail"], side="right") - 1, 0, N_T - 1
        )
        for k, label in (
            (0, "bottom tercile of trailing QLIKE_63"),
            (N_T - 1, "top tercile of trailing QLIKE_63"),
        ):
            r = b_.loc[terc == k, "r_ask"].to_numpy(float)
            rows_f.append(
                {
                    "deck": t,
                    "set": label,
                    "n": r.size,
                    "hit": float((r > 0).mean()),
                    "mean_R": float(r.mean()),
                    "kelly_f": kelly(r),
                }
            )
    ff = pd.DataFrame(rows_f)
    ff.to_csv(OUT / "f_regime.csv", index=False)
    print("\nF  f* by trailing-surprise tercile (descriptive; crossed)")
    print(ff.to_string(index=False, float_format=fmt))

    # ------------------------------------------------------------ verdict ----
    rec = a[
        (a["deck"] == RECORD_TAG) & (a["price"] == "crossed") & (a["sample"] == "all")
    ].iloc[0]
    me = c[(c["deck"] == RECORD_TAG) & (c["set"].str.startswith("month-end"))].iloc[0]
    top20 = b[
        (b["deck"] == RECORD_TAG) & (b["variant"] == "top 20 R days removed")
    ].iloc[0]
    print(
        "\nVERDICT  (A) long leg of sign(s), crossed, "
        f"{RECORD_TAG}: f* = {rec['kelly_f']:.3f} [{rec['kelly_ci_lo']:.3f}, {rec['kelly_ci_hi']:.3f}], "
        f"pessimistic {rec['kelly_pess_1se']:.3f} / {rec['kelly_pess_2se']:.3f} (1 / 2 SE); the live "
        f"{LIVE_FRACTION:g} is {rec['live_over_kelly']:.2f} x f* -- "
        + (
            "UNDER"
            if rec["live_over_kelly"] < 0.5
            else "AT"
            if rec["live_over_kelly"] <= 1.0
            else "OVER"
        )
        + f" Kelly; (B) without the 20 largest days f* = {top20['kelly_f']:.3f} ({top20['share_of_f']:.0%} of f*); "
        f"(C) month-end long f* = {me['kelly_f']:.3f} [{me['kelly_ci_lo']:.3f}, {me['kelly_ci_hi']:.3f}], "
        f"pessimistic {me['kelly_pess_1se']:.3f} / {me['kelly_pess_2se']:.3f}, live = {me['live_over_kelly']:.2f} x f*."
    )


if __name__ == "__main__":
    main()
