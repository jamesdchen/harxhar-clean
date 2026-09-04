"""Proposal 08 - one straddle, held from an early stamp to 15:30, then sign(s).

Read-only study. Nothing here is wired into any notebook; the script rebuilds the
0DTE package frame from ``data/spxw_chain.parquet`` with the shared library
helpers, joins the deck's own per-day file
``results/atm_straddle_0dte_1530/daily_blk2.parquet`` (866 scored days, the
block-diagonal ridge forecast) and scores one construction per entry stamp.

The construction. Short the nearest out-of-the-money call and put at an entry
stamp E in {10:00, ..., 15:00}, hold THOSE strikes to 15:30, and at 15:30 read
the deck's signal s for that day (issued at 15:30 against the 15:30
at-the-money implied variance; it is never recomputed here).

  * s <= 0 (short): keep the position to cash settlement at the official close.
    One crossing.
  * s > 0, variant FLAT: buy the held strikes back at 15:30 and stay flat.
    Two crossings.
  * s > 0, variant FLIP: buy the held strikes back at 15:30 and buy the 15:30
    nearest out-of-the-money straddle, settled at the close. Three crossings.

Comparators on the same days: (a) short at E and hold to settlement
unconditionally, (b) short at E and close at 15:30 unconditionally, (c) the
deck's close-only sign(s), (d) the deck's always short at 15:30.

CAVEAT, stated up front: a held straddle is not the sum of re-picked one-bar
shorts. The held strikes drift away from the index while a re-picked package
keeps its premium fresh, so the two paths are different portfolios over the same
bars. Section 4 measures the gap instead of assuming it away.

Fills. The midpoint case sells and buys at the quoted midpoint. The crossed
case pays the touch at every crossing: sell at the bid, buy at the ask; cash
settlement pays no exit spread. Per-premium returns divide the day's points by
the entry price actually paid at E - the midpoint in the midpoint case, the bid
in the crossed case - which is the convention of asl.crossed_premium_return.

Outputs: CSV and PNG under results/atm_straddle_intraday/proposals/08/.
Every number in 08_hold_to_close.md is printed by this script.

Run:  python writeup/intraday_proposals/08_hold_to_close.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _find_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "notebooks" / "atm_straddle_lib.py").exists():
            return q
    raise FileNotFoundError("repo root not found from " + str(start))


REPO = _find_repo(Path(__file__))
sys.path.insert(0, str(REPO / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

CHAIN = REPO / "data" / "spxw_chain.parquet"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
GSPC_DIR = REPO / "results" / "atm_straddle_0dte_1530" / "cache"
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "08"
CACHE = OUT / "cache"

# 30-minute stamps the trade may read. 9:30 has no vendor underlying price and
# live mids on only ~40% of days, so it forms no straddle; the 16:00 stamp is
# after the last decision and is excluded from the frame outright (gate 1).
STAMP_MIN = list(range(10 * 60, 15 * 60 + 30 + 1, 30))
CLOSE_MIN = 15 * 60 + 30
ENTRY_MIN = [10 * 60, 11 * 60, 12 * 60, 13 * 60, 14 * 60, 15 * 60]

SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
PLACEBO_DRAWS = 2000
ANN = float(np.sqrt(252.0))
SPACING_WINDOW = (
    50.0  # points either side of the 15:30 index used to measure the strike grid
)

VARIANTS = ["hold+flat", "hold+flip", "(a) hold uncond", "(b) close uncond"]
COMPARATORS = ["(c) close sign(s)", "(d) close always short"]
FILLS = ("mid", "crossed")


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def tick(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- frame build


def build_packages(
    t0: float, use_cache: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """(packages, quotes, refused cells, half-session dates) on the 10:00-15:30 stamps."""
    CACHE.mkdir(parents=True, exist_ok=True)
    st = CHAIN.stat()
    key = f"{st.st_size}_{st.st_mtime_ns}"
    p_pkg = CACHE / f"pkg_{key}.parquet"
    p_qte = CACHE / f"quotes_{key}.parquet"
    p_drp = CACHE / f"refused_{key}.csv"
    p_hlf = CACHE / f"half_sessions_{key}.csv"
    if use_cache and all(p.exists() for p in (p_pkg, p_qte, p_drp, p_hlf)):
        tick(t0, f"package cache hit {p_pkg.name}")
        half = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(p_hlf)["date"]))
        return pd.read_parquet(p_pkg), pd.read_parquet(p_qte), pd.read_csv(p_drp), half

    cols = [
        "expiration",
        "strike",
        "cp",
        "timestamp",
        "bid",
        "ask",
        "underlying_price",
        "hours_to_expiration",
        "impl_volatility",
    ]
    raw = pd.read_parquet(CHAIN, columns=cols)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["expiration"] = pd.to_datetime(raw["expiration"])
    raw["cp"] = raw["cp"].astype(str).str.upper().str[0]
    codes, uts = pd.factorize(raw["timestamp"])
    uet = pd.DatetimeIndex(uts).tz_convert("America/New_York")
    raw["et"] = uet.take(codes)
    raw["et_date"] = uet.normalize().take(codes)
    ecodes, uexp = pd.factorize(raw["expiration"])
    uexp_d = (
        pd.DatetimeIndex(uexp)
        .tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        .normalize()
    )
    raw["exp_date"] = uexp_d.take(ecodes)
    chain = raw[raw["et_date"] == raw["exp_date"]].copy()
    del raw
    tick(t0, f"0DTE rows {len(chain):,}")

    # Regular hours, then the shared half-session rule (hours_to_expiration <= 0
    # at the 15:30 stamp): those days go whole.
    mins_all = chain["et"].dt.hour * 60 + chain["et"].dt.minute
    chain = chain[(mins_all >= 9 * 60 + 30) & (mins_all <= 16 * 60)].copy()
    chain, half = asl.drop_early_close(chain)
    tick(
        t0,
        f"regular hours + half-session drop: {len(chain):,} rows, {len(half)} days dropped",
    )

    # strftime on eight million rows is the slow path; the clock label is an integer here.
    mins = (chain["et"].dt.hour * 60 + chain["et"].dt.minute).to_numpy()
    chain = chain[np.isin(mins, STAMP_MIN)].copy()
    chain["min"] = (chain["et"].dt.hour * 60 + chain["et"].dt.minute).to_numpy()
    chain["date"] = chain["et"].dt.normalize().dt.tz_localize(None)
    tick(t0, f"stamps 10:00-15:30: {len(chain):,} rows")

    live = chain.assign(mid=asl.quote_mid(chain["bid"], chain["ask"]).to_numpy())
    n_sentinel = int(live["mid"].isna().sum())
    live = live[np.isfinite(live["mid"]) & (live["mid"] > 0)].copy()
    spot = asl.stamp_spot(live, ["expiration", "timestamp"])
    stamps_all = pd.MultiIndex.from_frame(
        chain[["expiration", "timestamp"]].drop_duplicates()
    )
    stamps_live = pd.MultiIndex.from_frame(
        live[["expiration", "timestamp"]].drop_duplicates()
    )
    dead = stamps_all.difference(stamps_live)
    pkg, dropped = asl.pick_nearest_otm_guarded(
        live[
            [
                "expiration",
                "timestamp",
                "strike",
                "cp",
                "bid",
                "ask",
                "mid",
                "impl_volatility",
            ]
        ],
        spot,
        keys=("expiration", "timestamp"),
    )
    pkg = asl.attach_iv_hourly_as_30min(pkg)
    et_pkg = pd.to_datetime(pkg["timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    pkg["min"] = (et_pkg.dt.hour * 60 + et_pkg.dt.minute).to_numpy()
    pkg["date"] = et_pkg.dt.normalize().dt.tz_localize(None)
    keep = [
        "date",
        "expiration",
        "min",
        "S",
        "K_c",
        "K_p",
        "entry",
        "bid_entry",
        "ask_entry",
        "mid_c",
        "mid_p",
        "bid_c",
        "ask_c",
        "bid_p",
        "ask_p",
        "same_strike",
        "iv_var",
    ]
    pkg = pkg[keep].sort_values(["date", "min"]).reset_index(drop=True)

    dropped = dropped.copy()
    if len(dropped):
        et_d = pd.to_datetime(dropped["timestamp"], utc=True).dt.tz_convert(
            "America/New_York"
        )
        dropped["date"] = et_d.dt.normalize().dt.tz_localize(None)
        dropped["hhmm"] = [hhmm(int(m)) for m in (et_d.dt.hour * 60 + et_d.dt.minute)]
    dead_et = pd.to_datetime(dead.get_level_values(1), utc=True).tz_convert(
        "America/New_York"
    )
    dead_rows = pd.DataFrame(
        {
            "date": dead_et.normalize().tz_localize(None),
            "hhmm": [hhmm(int(m)) for m in (dead_et.hour * 60 + dead_et.minute)],
            "reason": "no_live_quote",
            "n_live": 0,
        }
    )
    cols_d = ["date", "hhmm", "reason", "n_live", "S", "K_c", "K_p", "gap"]
    refused = pd.concat([dropped, dead_rows], ignore_index=True).reindex(columns=cols_d)
    refused = refused.sort_values(["date", "hhmm"]).reset_index(drop=True)

    quotes = live[["date", "min", "strike", "cp", "bid", "ask", "mid"]].copy()
    quotes["strike"] = quotes["strike"].astype(float)
    tick(
        t0,
        f"packages {len(pkg):,} | refused cells {len(refused)} | "
        f"no-quote rows held out of the live frame {n_sentinel:,}",
    )

    for old in CACHE.glob("*"):
        old.unlink()
    pkg.to_parquet(p_pkg)
    quotes.to_parquet(p_qte)
    refused.to_csv(p_drp, index=False)
    pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in half]}).to_csv(
        p_hlf, index=False
    )
    return pkg, quotes, refused, half


def package_marks(quotes: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """bid/ask/mid of the package (K_c, K_p) at (date, min), one row per key row."""
    left_c = key[["date", "min", "K_c"]].rename(columns={"K_c": "strike"})
    left_p = key[["date", "min", "K_p"]].rename(columns={"K_p": "strike"})
    qc = quotes.loc[quotes["cp"] == "C", ["date", "min", "strike", "bid", "ask", "mid"]]
    qp = quotes.loc[quotes["cp"] == "P", ["date", "min", "strike", "bid", "ask", "mid"]]
    got_c = left_c.merge(qc, on=["date", "min", "strike"], how="left")
    got_p = left_p.merge(qp, on=["date", "min", "strike"], how="left")
    out = pd.DataFrame(index=key.index)
    for f in ("bid", "ask", "mid"):
        out[f] = got_c[f].to_numpy(dtype=float) + got_p[f].to_numpy(dtype=float)
    return out


def load_official_close() -> tuple[pd.Series, Path]:
    """The deck's cached ^GSPC official close, whichever file name the deck last wrote."""
    cands = sorted(GSPC_DIR.glob("gspc_close*.parquet"))
    plain = GSPC_DIR / "gspc_close.parquet"
    path = plain if plain.exists() else (cands[-1] if cands else plain)
    px = pd.read_parquet(path)["close"].astype(float)
    px.index = pd.DatetimeIndex(px.index).normalize()
    return px, path


def settle_held(
    days: pd.DatetimeIndex, k_c: pd.Series, k_p: pd.Series, s_close: pd.Series
) -> pd.Series:
    """Cash settlement of the held strikes at the official close (asl.settle_package)."""
    df = pd.DataFrame(
        {"expiration": days, "K_c": k_c.to_numpy(float), "K_p": k_p.to_numpy(float)}
    )
    out = asl.settle_package(df, s_close)
    return pd.Series(out["exit"].to_numpy(float), index=days)


# ---------------------------------------------------------------- statistics


def safe_t(x: pd.Series) -> float:
    v = x.astype(float).dropna()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * np.sqrt(len(v))) if sd > 0 else float("nan")


def stat_row(
    pnl_pts: pd.Series,
    ret_prem: pd.Series,
    crossings: pd.Series,
    prem_crossed: pd.Series,
    hs_paid: pd.Series,
) -> dict[str, float]:
    r = ret_prem.astype(float).dropna()
    p = pnl_pts.astype(float).dropna()
    n = int(len(r))
    sd = float(r.std(ddof=1))
    sd_p = float(p.std(ddof=1))
    cum = p.cumsum()
    mean_p = float(p.mean())
    xr = float(crossings.astype(float).dropna().mean())
    qbar = float(prem_crossed.astype(float).dropna().mean())
    hbar = float(hs_paid.astype(float).dropna().mean())
    return {
        "n": float(n),
        "mean_prem": float(r.mean()),
        "t_prem": safe_t(r),
        "Sharpe_prem": float(r.mean() / sd * ANN) if sd > 0 else float("nan"),
        "mean_pts": mean_p,
        "t_pts": safe_t(p),
        "Sharpe_pts": float(mean_p / sd_p * ANN) if sd_p > 0 else float("nan"),
        "maxDD_pts": float((cum - cum.cummax()).min()),
        "worst_pts": float(p.min()),
        "crossings_per_day": xr,
        "premium_crossed_pts": qbar,
        "half_spread_paid_pts": hbar,
        "be_half_spread_pct_prem": 100.0 * mean_p / qbar if qbar > 0 else float("nan"),
        "realized_half_spread_pct_prem": 100.0 * hbar / qbar
        if qbar > 0
        else float("nan"),
        "be_half_spread_pts_per_crossing": mean_p / xr if xr > 0 else float("nan"),
    }


def block_boot_delta(
    a: pd.Series, b: pd.Series, rng: np.random.Generator
) -> dict[str, float]:
    """Circular block bootstrap of Sharpe(a) - Sharpe(b) on the common days."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    x = j["a"].to_numpy(float)
    y = j["b"].to_numpy(float)
    n = len(x)
    hat = (x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1)) * ANN
    idx = asl.circular_block_bootstrap_idx(rng, n, BOOT_BLOCK, BOOT_B)
    xa = x[idx]
    ya = y[idx]
    d = (
        xa.mean(axis=1) / xa.std(axis=1, ddof=1)
        - ya.mean(axis=1) / ya.std(axis=1, ddof=1)
    ) * ANN
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "n_common": float(n),
        "dSharpe": float(hat),
        "pct_draws_positive": float(100.0 * (d > 0).mean()),
        "pctile_lo": float(lo),
        "pctile_hi": float(hi),
        "basic_lo": float(2.0 * hat - hi),
        "basic_hi": float(2.0 * hat - lo),
    }


def placebo_percentiles(
    short_branch: pd.Series, long_branch: pd.Series, is_short: pd.Series
) -> dict[str, float]:
    """Sharpe percentile of the real construction among random 15:30 signs."""
    s = short_branch.to_numpy(float)
    ell = long_branch.to_numpy(float)
    q = is_short.to_numpy(bool)
    ok = np.isfinite(s) & np.isfinite(ell)
    s, ell, q = s[ok], ell[ok], q[ok]
    real = np.where(q, s, ell)
    real_sh = float(real.mean() / real.std(ddof=1) * ANN)
    n = len(s)
    p_short = float(q.mean())
    out: dict[str, float] = {
        "n": float(n),
        "Sharpe_real": real_sh,
        "short_share_pct": 100.0 * p_short,
    }
    for tag, p in (("p50", 0.5), ("rate", p_short)):
        rng = np.random.default_rng(SEED)
        draw = rng.random((PLACEBO_DRAWS, n)) < p
        r = np.where(draw, s[None, :], ell[None, :])
        sh = r.mean(axis=1) / r.std(axis=1, ddof=1) * ANN
        out[f"pctile_{tag}"] = float(100.0 * (sh < real_sh).mean())
        out[f"median_{tag}"] = float(np.median(sh))
        out[f"p95_{tag}"] = float(np.percentile(sh, 95))
    return out


# ------------------------------------------------------------- the trade grid


def entry_context(
    pkg: pd.DataFrame, quotes: pd.DataFrame, days: pd.DatetimeIndex
) -> dict[str, dict[str, pd.Series]]:
    """Per entry stamp: the entry package and the 15:30 marks of the HELD strikes.

    Nothing in here reads a field dated after 15:30 (gate 1).
    """
    ctx: dict[str, dict[str, pd.Series]] = {}
    for e_min in ENTRY_MIN:
        pe = pkg[pkg["min"] == e_min].set_index("date").reindex(days)
        key = pd.DataFrame(
            {
                "date": days,
                "min": CLOSE_MIN,
                "K_c": pe["K_c"].to_numpy(),
                "K_p": pe["K_p"].to_numpy(),
            }
        )
        held = package_marks(quotes, key)
        held.index = days
        ctx[hhmm(e_min)] = {
            "K_c": pe["K_c"].astype(float),
            "K_p": pe["K_p"].astype(float),
            "S_E": pe["S"].astype(float),
            "mid_E": pe["entry"].astype(float),
            "bid_E": pe["bid_entry"].astype(float),
            "ask_E": pe["ask_entry"].astype(float),
            "mid_1530": held["mid"].astype(float),
            "bid_1530": held["bid"].astype(float),
            "ask_1530": held["ask"].astype(float),
        }
    return ctx


def points_pnl_grid(
    ctx: dict[str, dict[str, pd.Series]],
    days: pd.DatetimeIndex,
    is_short: pd.Series,
    s_close: pd.Series,
    entry_1530: pd.Series,
    bid_e1530: pd.Series,
    ask_e1530: pd.Series,
    settle_1530: pd.Series,
    pos: pd.Series,
) -> dict[tuple[str, str], pd.Series]:
    """Points of P&L per straddle for every construction and both fills."""
    out: dict[tuple[str, str], pd.Series] = {}
    for tag, c in ctx.items():
        settle_e = settle_held(days, c["K_c"], c["K_p"], s_close)
        short_mid = c["mid_E"] - settle_e
        short_cr = c["bid_E"] - settle_e
        flat_mid = c["mid_E"] - c["mid_1530"]
        flat_cr = c["bid_E"] - c["ask_1530"]
        flip_mid = c["mid_E"] - c["mid_1530"] - entry_1530 + settle_1530
        flip_cr = c["bid_E"] - c["ask_1530"] - ask_e1530 + settle_1530
        out[(f"{tag} hold+flat", "mid")] = short_mid.where(is_short, flat_mid)
        out[(f"{tag} hold+flat", "crossed")] = short_cr.where(is_short, flat_cr)
        out[(f"{tag} hold+flip", "mid")] = short_mid.where(is_short, flip_mid)
        out[(f"{tag} hold+flip", "crossed")] = short_cr.where(is_short, flip_cr)
        out[(f"{tag} (a) hold uncond", "mid")] = short_mid
        out[(f"{tag} (a) hold uncond", "crossed")] = short_cr
        out[(f"{tag} (b) close uncond", "mid")] = flat_mid
        out[(f"{tag} (b) close uncond", "crossed")] = flat_cr
    out[("(c) close sign(s)", "mid")] = pos * (settle_1530 - entry_1530)
    out[("(c) close sign(s)", "crossed")] = pd.Series(
        np.where(pos > 0, settle_1530 - ask_e1530, bid_e1530 - settle_1530), index=days
    )
    out[("(d) close always short", "mid")] = -(settle_1530 - entry_1530)
    out[("(d) close always short", "crossed")] = bid_e1530 - settle_1530
    return out


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Proposal 08 - hold one straddle to 15:30, then sign(s)."
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="rebuild the package frame from the chain",
    )
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    print("=" * 110)
    print(
        "Proposal 08 - short one nearest-OTM straddle at E, hold the strikes to 15:30, then sign(s) at 15:30"
    )
    print("=" * 110)
    print(
        "CAVEAT: a held straddle is not the sum of re-picked one-bar shorts; section 4 measures the gap."
    )
    print(
        f"Forecast: {asl.YHAT_LABEL['blk2']}. Rule: sign(s). Fills: midpoint and crossed spread."
    )

    pkg, quotes, refused, half = build_packages(t0, use_cache=not args.no_cache)
    deck = pd.read_parquet(DECK)
    deck.index = pd.DatetimeIndex(deck.index).normalize()
    days = pd.DatetimeIndex(deck.index)
    close_px, close_path = load_official_close()

    print("\n--- 0. Sample and the deck's rule table ---")
    # The deck's per-day file is rebuilt by the notebooks; fingerprint the version
    # this run scored so every number below is traceable to one of them.
    deck_st = DECK.stat()
    deck_fp = hashlib.sha1(
        np.ascontiguousarray(deck[["rv_hat", "signal", "pos", "R"]].to_numpy(float))
    ).hexdigest()[:12]
    print(
        f"deck file: {DECK.name} {deck_st.st_size} bytes, written "
        f"{pd.Timestamp(deck_st.st_mtime, unit='s', tz='UTC').tz_convert('Asia/Taipei'):%Y-%m-%d %H:%M:%S %Z}, "
        f"forecast fingerprint {deck_fp}"
    )
    print(f"official close cache: {close_path.name} ({len(close_px)} sessions)")
    print(f"deck days {len(days)} from {days.min().date()} to {days.max().date()}")
    print(
        f"half sessions dropped by the shared rule ({len(half)}): "
        f"{', '.join(d.strftime('%Y-%m-%d') for d in half)}"
    )
    pkg = pkg[pkg["date"].isin(days)].copy()
    quotes = quotes[quotes["date"].isin(days)].copy()
    print(
        f"packages on the deck's days: {len(pkg):,} on {pkg['date'].nunique()} days, "
        f"{len(STAMP_MIN)} stamps 10:00-15:30"
    )
    size = asl.rule_sizes(deck)
    for name in asl.RULE_ORDER:
        row = asl.rule_row(size[name] * deck["R"], size[name])
        print(
            f"  deck {name:<13s} Sharpe {row['Sharpe_ann']:.4f}  t {row['t_mean']:.4f}  "
            f"n {int(row['n'])}  buy {row['pct_buy']:.2f}%"
        )
    nb_cross = (
        REPO
        / "results"
        / "atm_straddle_intraday"
        / "rule_table_intraday_crossed_blk2.csv"
    )
    if nb_cross.exists():
        nb = pd.read_csv(nb_cross).set_index("rule")
        print(
            "what the intraday re-pick costs (the notebook's own table, for context):"
        )
        print(
            nb[
                [
                    "Sharpe mid",
                    "Sharpe crossed-spread",
                    "crossings/day",
                    "break-even half-spread % prem",
                ]
            ]
            .round(4)
            .to_string()
        )

    pos = deck["pos"].astype(float)
    is_short = pos <= 0
    settle_1530 = deck["exit"].astype(float)
    entry_1530 = deck["entry"].astype(float)
    bid_e1530 = deck["bid_c"].astype(float) + deck["bid_p"].astype(float)
    ask_e1530 = deck["ask_c"].astype(float) + deck["ask_p"].astype(float)
    s_close = close_px.reindex(days)

    # -------------------------------------------------------------- gate 1
    print("\n--- 1. Gates: reproduction and causality ---")
    print(
        f"latest stamp anywhere in the frame: {hhmm(int(pkg['min'].max()))} "
        f"(the 16:00 stamp is excluded outright, so its censored implied volatility is never read)"
    )
    p15 = pkg[pkg["min"] == CLOSE_MIN].set_index("date").reindex(days)
    for col in ("K_c", "K_p", "S"):
        print(
            f"  independent 15:30 pick vs the deck's, {col}: max |difference| "
            f"{float((p15[col] - deck[col]).abs().max()):.3e}"
        )
    print(
        f"  independent 15:30 midpoint entry vs the deck's: max |difference| "
        f"{float((p15['entry'] - deck['entry']).abs().max()):.3e}"
    )
    print(
        f"  independent 15:30 implied variance vs the deck's: max |difference| "
        f"{float((p15['iv_var'] - deck['iv_var']).abs().max()):.3e}"
    )
    print(
        f"  the deck's signal equals rv_hat - iv_var: max |difference| "
        f"{float((deck['signal'] - (deck['rv_hat'] - deck['iv_var'])).abs().max()):.3e}"
    )
    print(
        f"  the cached official close equals the deck's own S_close on every day: max |difference| "
        f"{float((s_close - deck['S_close'].astype(float)).abs().max()):.3e}"
    )
    print(
        f"  settling the 15:30 strikes at that close reproduces the deck's exit: max |difference| "
        f"{float((settle_held(days, p15['K_c'].astype(float), p15['K_p'].astype(float), s_close) - settle_1530).abs().max()):.3e}"
    )
    print(
        f"  hours_to_expiration on the signal's row: "
        f"{sorted(set(deck['hours_to_expiration_c'].unique()))} - the signal is a 15:30 object"
    )

    ctx = entry_context(pkg, quotes, days)
    grid = points_pnl_grid(
        ctx, days, is_short, s_close, entry_1530, bid_e1530, ask_e1530, settle_1530, pos
    )
    rng_c = np.random.default_rng(SEED)
    s_perm = pd.Series(
        s_close.to_numpy(float)[rng_c.permutation(len(days))], index=days
    )
    grid_perm = points_pnl_grid(
        ctx, days, is_short, s_perm, entry_1530, bid_e1530, ask_e1530, settle_1530, pos
    )
    n_long_tot = n_long_chg = n_short_tot = n_short_chg = 0
    for tag in ctx:
        a = grid[(f"{tag} hold+flat", "mid")]
        b = grid_perm[(f"{tag} hold+flat", "mid")]
        ok = a.notna()
        n_long_tot += int((ok & ~is_short).sum())
        n_long_chg += int((ok & ~is_short & (a != b)).sum())
        n_short_tot += int((ok & is_short).sum())
        n_short_chg += int((ok & is_short & (a != b)).sum())
    print(
        "  every decision input (entry price, held strikes, 15:30 marks, the deck's position) is built "
        "in entry_context, which reads no stamp after 15:30."
    )
    print(
        f"  permuting the official close across days: FLAT days that end flat at 15:30 change on "
        f"{n_long_chg} of {n_long_tot}; days that run to settlement change on "
        f"{n_short_chg} of {n_short_tot}."
    )

    # ------------------------------------------------- per-premium and crossings
    prem: dict[tuple[str, str], pd.Series] = {}
    cross: dict[str, pd.Series] = {}
    crossed_prem: dict[str, pd.Series] = {}
    hs_paid: dict[str, pd.Series] = {}
    branches: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
    hs_1530 = 0.5 * (ask_e1530 - bid_e1530)
    for tag, c in ctx.items():
        have = c["mid_E"].notna()
        hs_e_pts = 0.5 * (c["ask_E"] - c["bid_E"])
        hs_h_pts = 0.5 * (c["ask_1530"] - c["bid_1530"])
        for v in VARIANTS:
            nm = f"{tag} {v}"
            prem[(nm, "mid")] = grid[(nm, "mid")] / c["mid_E"]
            prem[(nm, "crossed")] = grid[(nm, "crossed")] / c["bid_E"]
        cross[f"{tag} hold+flat"] = pd.Series(
            np.where(is_short, 1.0, 2.0), index=days
        ).where(have)
        cross[f"{tag} hold+flip"] = pd.Series(
            np.where(is_short, 1.0, 3.0), index=days
        ).where(have)
        cross[f"{tag} (a) hold uncond"] = pd.Series(1.0, index=days).where(have)
        cross[f"{tag} (b) close uncond"] = pd.Series(2.0, index=days).where(have)
        crossed_prem[f"{tag} hold+flat"] = c["mid_E"].where(
            is_short, c["mid_E"] + c["mid_1530"]
        )
        crossed_prem[f"{tag} hold+flip"] = c["mid_E"].where(
            is_short, c["mid_E"] + c["mid_1530"] + entry_1530
        )
        crossed_prem[f"{tag} (a) hold uncond"] = c["mid_E"]
        crossed_prem[f"{tag} (b) close uncond"] = c["mid_E"] + c["mid_1530"]
        hs_paid[f"{tag} hold+flat"] = hs_e_pts.where(is_short, hs_e_pts + hs_h_pts)
        hs_paid[f"{tag} hold+flip"] = hs_e_pts.where(
            is_short, hs_e_pts + hs_h_pts + hs_1530
        )
        hs_paid[f"{tag} (a) hold uncond"] = hs_e_pts
        hs_paid[f"{tag} (b) close uncond"] = hs_e_pts + hs_h_pts
        settle_e = settle_held(days, c["K_c"], c["K_p"], s_close)
        for v, long_mid, long_cr in (
            ("hold+flat", c["mid_E"] - c["mid_1530"], c["bid_E"] - c["ask_1530"]),
            (
                "hold+flip",
                c["mid_E"] - c["mid_1530"] - entry_1530 + settle_1530,
                c["bid_E"] - c["ask_1530"] - ask_e1530 + settle_1530,
            ),
        ):
            branches[(f"{tag} {v}", "mid")] = (
                (c["mid_E"] - settle_e) / c["mid_E"],
                long_mid / c["mid_E"],
            )
            branches[(f"{tag} {v}", "crossed")] = (
                (c["bid_E"] - settle_e) / c["bid_E"],
                long_cr / c["bid_E"],
            )
    prem[("(c) close sign(s)", "mid")] = grid[("(c) close sign(s)", "mid")] / entry_1530
    prem[("(c) close sign(s)", "crossed")] = grid[
        ("(c) close sign(s)", "crossed")
    ] / pd.Series(np.where(pos > 0, ask_e1530, bid_e1530), index=days)
    prem[("(d) close always short", "mid")] = (
        grid[("(d) close always short", "mid")] / entry_1530
    )
    prem[("(d) close always short", "crossed")] = (
        grid[("(d) close always short", "crossed")] / bid_e1530
    )
    for nm in COMPARATORS:
        cross[nm] = pd.Series(1.0, index=days)
        crossed_prem[nm] = entry_1530
        hs_paid[nm] = hs_1530
    ref_c = asl.crossed_premium_return(pos, settle_1530, bid_e1530, ask_e1530)
    print(
        f"  comparator (c) crossed reproduces asl.crossed_premium_return: max |difference| "
        f"{float((prem[('(c) close sign(s)', 'crossed')] - ref_c).abs().max()):.3e}; "
        f"untradeable crossed rows {asl.crossed_untradeable_count(pos, bid_e1530, ask_e1530)}"
    )
    # The short branch does NOTHING at 15:30: the position opened at E is carried
    # to settlement on its own strikes, so the day has one crossing, the entry.
    # That makes FLAT, FLIP and comparator (a) the same series on those days.
    print(
        f"  the short branch is the deck's own sign: (pos <= 0) equals (signal <= 0) on all days: "
        f"{bool(((pos <= 0) == (deck['signal'].astype(float) <= 0)).all())}; "
        f"short days {int(is_short.sum())}, long days {int((~is_short).sum())}"
    )
    hold_gap = 0.0
    for tag in ctx:
        for fill in FILLS:
            a = grid[(f"{tag} hold+flat", fill)][is_short]
            b = grid[(f"{tag} hold+flip", fill)][is_short]
            u = grid[(f"{tag} (a) hold uncond", fill)][is_short]
            hold_gap = max(
                hold_gap,
                float((a - u).abs().max()),
                float((b - u).abs().max()),
            )
    print(
        f"  on short days no 15:30 transaction is booked: FLAT, FLIP and comparator (a) are the "
        f"same series, max |difference| {hold_gap:.3e} points, and the day carries one crossing"
    )
    gap = max(
        float((grid[(nm, "mid")] - hs_paid[nm] - grid[(nm, "crossed")]).abs().max())
        for nm in hs_paid
    )
    print(
        f"  spread identity, crossed points == midpoint points minus the half-spreads paid: "
        f"max |difference| {gap:.3e} over all {len(hs_paid)} constructions"
    )
    flip_gap = max(
        float(
            (
                grid[(f"{tag} hold+flip", "mid")]
                - grid[(f"{tag} (b) close uncond", "mid")]
                - grid[("(c) close sign(s)", "mid")]
            )[~is_short]
            .abs()
            .max()
        )
        for tag in ctx
    )
    print(
        f"  on the {100.0 * float((~is_short).mean()):.2f}% of days the signal is long, FLIP is exactly "
        f"comparator (b) plus comparator (c): max |difference| {flip_gap:.3e} points"
    )

    # -------------------------------------------------------------- results grid
    order = [f"{hhmm(e)} {v}" for e in ENTRY_MIN for v in VARIANTS] + COMPARATORS
    rows: list[dict[str, object]] = []
    for nm in order:
        for fill in FILLS:
            rows.append(
                {
                    "construction": nm,
                    "fill": fill,
                    **stat_row(
                        grid[(nm, fill)],
                        prem[(nm, fill)],
                        cross[nm],
                        crossed_prem[nm],
                        hs_paid[nm],
                    ),
                }
            )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "08_results_grid.csv", index=False)
    wide = res.pivot(index="construction", columns="fill")
    show = pd.DataFrame(
        {
            "n": wide[("n", "mid")].astype(int),
            "mean_prem": wide[("mean_prem", "mid")],
            "t_prem": wide[("t_prem", "mid")],
            "Sharpe_mid": wide[("Sharpe_prem", "mid")],
            "Sharpe_crossed": wide[("Sharpe_prem", "crossed")],
            "mean_pts": wide[("mean_pts", "mid")],
            "mean_pts_cr": wide[("mean_pts", "crossed")],
            "Sharpe_pts": wide[("Sharpe_pts", "mid")],
            "maxDD_pts": wide[("maxDD_pts", "mid")],
            "worst_pts": wide[("worst_pts", "mid")],
            "cross/day": wide[("crossings_per_day", "mid")],
            "be_half_%": wide[("be_half_spread_pct_prem", "mid")],
            "real_half_%": wide[("realized_half_spread_pct_prem", "mid")],
        }
    ).reindex(order)
    print("\n--- 2. Results grid: every entry stamp, every variant, both fills ---")
    print(
        "(mean_prem, t_prem, Sharpe_mid, Sharpe_crossed are per unit of the entry premium paid at E;"
    )
    print(
        " mean_pts, maxDD_pts, worst_pts are index points per straddle; be_half_% is the uniform"
    )
    print(
        " proportional half-spread per crossing that zeroes the midpoint mean, real_half_% the"
    )
    print(" half-spread the quotes actually charge on the same crossings.)")
    print(show.round(4).to_string())

    # ----------------------------------------------------- 15:30 diagnostics
    q15 = quotes[quotes["min"] == CLOSE_MIN]
    spacing_map: dict[pd.Timestamp, float] = {}
    for dt, g in q15.groupby("date"):
        s_here = float(p15.loc[dt, "S"])
        ks = np.sort(g["strike"].unique())
        ks = ks[np.abs(ks - s_here) <= SPACING_WINDOW]
        spacing_map[dt] = float(np.median(np.diff(ks))) if len(ks) > 2 else float("nan")
    spacing = pd.Series(spacing_map).reindex(days)
    print("\n--- 3. What the held strikes look like at 15:30 ---")
    print(
        f"measured near-the-money strike spacing at 15:30: median {float(spacing.median()):.1f} points; "
        f"{int((spacing == spacing.median()).sum())} of {len(spacing)} days at that value"
    )
    diag_rows: list[dict[str, object]] = []
    for tag, c in ctx.items():
        centre = 0.5 * (c["K_c"] + c["K_p"])
        dist = (p15["S"].astype(float) - centre).abs()
        in_strikes = dist / spacing
        delta_proxy = np.maximum(p15["S"].astype(float) - c["K_c"], 0.0) - np.maximum(
            c["K_p"] - p15["S"].astype(float), 0.0
        )
        hs_e = 0.5 * (c["ask_E"] - c["bid_E"]) / c["mid_E"]
        hs_h = 0.5 * (c["ask_1530"] - c["bid_1530"]) / c["mid_1530"]
        diag_rows.append(
            {
                "entry": tag,
                "n": int(c["mid_E"].notna().sum()),
                "premium_E_median_pts": float(c["mid_E"].median()),
                "held_at_1530_median_pts": float(c["mid_1530"].median()),
                "fresh_1530_median_pts": float(entry_1530.median()),
                "dist_mean_pts": float(dist.mean()),
                "dist_median_pts": float(dist.median()),
                "dist_p90_pts": float(dist.quantile(0.90)),
                "dist_mean_strikes": float(in_strikes.mean()),
                "share_gt_1_strike_%": float(100.0 * (in_strikes > 1.0).mean()),
                "share_gt_2_strikes_%": float(100.0 * (in_strikes > 2.0).mean()),
                "delta_proxy_mean_pts": float(delta_proxy.mean()),
                "abs_delta_proxy_mean_pts": float(delta_proxy.abs().mean()),
                "delta_proxy_zero_%": float(100.0 * (delta_proxy.abs() < 1e-12).mean()),
                "held_eq_1530_strikes_%": float(
                    100.0
                    * (
                        (c["K_c"] == p15["K_c"].astype(float))
                        & (c["K_p"] == p15["K_p"].astype(float))
                    ).mean()
                ),
                "half_spread_E_median_%": float(100.0 * hs_e.median()),
                "half_spread_held_1530_median_%": float(100.0 * hs_h.median()),
            }
        )
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "08_diagnostics_1530.csv", index=False)
    print(diag.round(3).to_string(index=False))
    hs_fresh = 0.5 * (ask_e1530 - bid_e1530) / entry_1530
    print(
        f"median half-spread of the fresh 15:30 package: {float(100.0 * hs_fresh.median()):.3f}% of premium"
    )

    # ------------------------------------------------------ drift decomposition
    nxt = pkg[["date", "min", "K_c", "K_p", "entry"]].copy()
    nxt = nxt[nxt["min"] < CLOSE_MIN].copy()
    nxt["min"] = nxt["min"] + 30
    mk = package_marks(quotes, nxt[["date", "min", "K_c", "K_p"]])
    nxt["min"] = nxt["min"] - 30
    nxt["bar_pts"] = nxt["entry"].astype(float) - mk["mid"].to_numpy(dtype=float)
    nxt["bar_prem"] = nxt["bar_pts"] / nxt["entry"].astype(float)
    bar_pts = nxt.pivot_table(index="date", columns="min", values="bar_pts").reindex(
        days
    )
    bar_prem = nxt.pivot_table(index="date", columns="min", values="bar_prem").reindex(
        days
    )
    drift_rows: list[dict[str, object]] = []
    for e_min in ENTRY_MIN:
        tag = hhmm(e_min)
        c = ctx[tag]
        held_pts = c["mid_E"] - c["mid_1530"]
        held_prem = held_pts / c["mid_E"]
        bars = [m for m in STAMP_MIN if e_min <= m < CLOSE_MIN]
        rep_pts = bar_pts[bars].sum(axis=1, min_count=len(bars))
        rep_prem = bar_prem[bars].sum(axis=1, min_count=len(bars))
        d_pts = (held_pts - rep_pts).dropna()
        d_prem = (held_prem - rep_prem).dropna()
        drift_rows.append(
            {
                "entry": tag,
                "bars": len(bars),
                "n": int(len(d_pts)),
                "held_mean_pts": float(held_pts.mean()),
                "repick_mean_pts": float(rep_pts.mean()),
                "drift_mean_pts": float(d_pts.mean()),
                "drift_t_pts": safe_t(d_pts),
                "held_sd_pts": float(held_pts.std(ddof=1)),
                "repick_sd_pts": float(rep_pts.std(ddof=1)),
                "held_mean_prem": float(held_prem.mean()),
                "repick_mean_prem": float(rep_prem.mean()),
                "drift_mean_prem": float(d_prem.mean()),
                "drift_t_prem": safe_t(d_prem),
                "max_abs_drift_pts": float(d_pts.abs().max()),
            }
        )
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(OUT / "08_drift_decomposition.csv", index=False)
    print(
        "\n--- 4. Drift: the held short from E to 15:30 vs the sum of re-picked one-bar shorts ---"
    )
    print(
        "(both are shorts over the same bars at midpoints; the difference is the drift cost or benefit)"
    )
    print(drift.round(4).to_string(index=False))

    # --------------------------------------------------------------- placebo
    pl_rows: list[dict[str, object]] = []
    for e_min in ENTRY_MIN:
        for v in ("hold+flat", "hold+flip"):
            nm = f"{hhmm(e_min)} {v}"
            for fill in FILLS:
                sb, lb = branches[(nm, fill)]
                pl_rows.append(
                    {
                        "construction": nm,
                        "fill": fill,
                        **placebo_percentiles(sb, lb, is_short),
                    }
                )
    placebo = pd.DataFrame(pl_rows)
    placebo.to_csv(OUT / "08_placebo_random_sign.csv", index=False)
    print(
        f"\n--- 5. Placebo: the 15:30 sign replaced by a random sign ({PLACEBO_DRAWS} draws, rng({SEED})) ---"
    )
    print(
        "(pctile_p50 draws short with probability one half; pctile_rate matches the rule's own short share)"
    )
    print(placebo.round(3).to_string(index=False))

    # ------------------------------------------------------------- bootstrap
    bt_rows: list[dict[str, object]] = []
    for nm in order:
        if nm == "(c) close sign(s)":
            continue
        for fill in FILLS:
            bt_rows.append(
                {
                    "construction": nm,
                    "fill": fill,
                    **block_boot_delta(
                        prem[(nm, fill)],
                        prem[("(c) close sign(s)", fill)],
                        np.random.default_rng(SEED),
                    ),
                }
            )
    boot = pd.DataFrame(bt_rows)
    boot.to_csv(OUT / "08_bootstrap_vs_close.csv", index=False)
    print(
        f"\n--- 6. Circular block bootstrap of the Sharpe difference against comparator (c) "
        f"(block {BOOT_BLOCK}, B={BOOT_B}, rng({SEED})) ---"
    )
    print(boot.round(4).to_string(index=False))

    # ------------------------------------------------------------- anomalies
    anomalies: list[dict[str, object]] = []
    ref = refused.copy()
    ref["date"] = pd.to_datetime(ref["date"])
    for tag, c in ctx.items():
        for dt in days[c["mid_E"].isna()]:
            why = ref[(ref["date"] == dt) & (ref["hhmm"] == tag)]
            anomalies.append(
                {
                    "entry": tag,
                    "date": dt.strftime("%Y-%m-%d"),
                    "reason": str(why["reason"].iloc[0])
                    if len(why)
                    else "no package formed",
                    "n_live": float(why["n_live"].iloc[0])
                    if len(why)
                    else float("nan"),
                }
            )
    anom = pd.DataFrame(anomalies)
    anom.to_csv(OUT / "08_unconstructible_days.csv", index=False)
    print("\n--- 7. Days that cannot be constructed, and why ---")
    print(anom.to_string(index=False) if len(anom) else "none")
    ref_deck = ref[ref["date"].isin(days)]
    print("\nrefused cells over the whole 10:00-15:30 grid on the deck's days:")
    print(ref_deck.to_string(index=False) if len(ref_deck) else "none")
    n_held_missing = sum(
        int((ctx[t]["mid_E"].notna() & ctx[t]["mid_1530"].isna()).sum()) for t in ctx
    )
    print(
        f"entry-stamp days whose held strikes have no 15:30 midpoint: {n_held_missing}"
    )

    # ------------------------------------------------------------ daily series
    daily = pd.DataFrame(
        {f"{nm} | {fill}": grid[(nm, fill)] for nm in order for fill in FILLS}
    )
    daily.index.name = "date"
    daily.to_csv(OUT / "08_daily_pnl_points.csv")

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, v in zip(axes, ("hold+flat", "hold+flip")):
        for i, e_min in enumerate(ENTRY_MIN):
            nm = f"{hhmm(e_min)} {v}"
            ax.plot(
                days,
                grid[(nm, "mid")].fillna(0.0).cumsum(),
                lw=1.1,
                color=cmap(i / (len(ENTRY_MIN) - 1)),
                label=f"E = {hhmm(e_min)}",
            )
        ax.plot(
            days,
            grid[("(c) close sign(s)", "mid")].cumsum(),
            lw=2.0,
            color="crimson",
            label="(c) close sign(s)",
        )
        ax.plot(
            days,
            grid[("(d) close always short", "mid")].cumsum(),
            lw=1.2,
            color="0.35",
            ls="--",
            label="(d) close always short",
        )
        ax.set_title(
            f"short at E, hold to 15:30, then sign(s): {v} (midpoint fills)",
            fontsize=10,
        )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("date")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative P&L, index points per straddle")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=8, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(OUT / "08_cum_points.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\nwrote:")
    for p in sorted(OUT.glob("08_*")):
        print("  ", p.relative_to(REPO))
    tick(t0, "done")


if __name__ == "__main__":
    main()
