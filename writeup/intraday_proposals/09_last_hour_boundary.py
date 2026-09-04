"""Proposal 09 - the last hour: the close trade against a 15:00 entry.

A PRE-REGISTERED two-cell study. Two cells, fixed before the data were read,
nothing selected from a grid.

  Cell A (15:30). The deck's close trade exactly: position sign(s) from the
  block-diagonal ridge per-day file, nearest out-of-the-money straddle picked
  at 15:30, cash-settled at the official close.

  Cell B (15:00). The nearest out-of-the-money straddle picked at 15:00,
  position sign(s_rem) with s_rem = rv_hat_rem - iv_var_rem, where
  rv_hat_rem is the fresh one-bar forecast for [15:00, 15:30] (the panel row
  stamped 15:30, issued at 15:00) divided by w_1500, the trailing share of the
  15:00-15:30 bar in the remaining hour's variance; and
  iv_var_rem = iv_hourly(15:00)^2 x hours_to_expiration(15:00) = 1.0 h. The
  SAME strikes are held to cash settlement at the official close: one
  crossing, no 15:30 transaction of any kind.

Comparators on the same days: always short at 15:00 held to settlement, and
always short at 15:30 (the deck's row).

CAVEAT, up front. The 15:00 cell was flagged by proposal 08 as the best of
twenty-four cells. This file is its pre-registered replication on two cells,
not a fresh search; the multiplicity statement below says so.

Fills. The midpoint case enters at the quoted midpoint. The crossed case pays
the touch at entry - buy at the ask, sell at the bid - and cash settlement pays
no exit spread (asl.crossed_premium_return).

Outputs: CSV and PNG under results/atm_straddle_intraday/proposals/09/.
Every number in 09_last_hour_boundary.md is printed by this script.

Run:  python writeup/intraday_proposals/09_last_hour_boundary.py
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


def _bootstrap_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "notebooks" / "atm_straddle_lib.py").exists():
            return q
    raise FileNotFoundError("repo root not found from " + str(start))


sys.path.insert(0, str(_bootstrap_repo(Path(__file__)) / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
CHAIN = REPO / "data" / "spxw_chain.parquet"
DECK_DIR = REPO / "results" / "atm_straddle_0dte_1530"
GSPC_DIR = DECK_DIR / "cache"
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "09"
CACHE = OUT / "cache"

STAMP_MIN = list(range(10 * 60, 15 * 60 + 30 + 1, 30))
CLOSE_MIN = 15 * 60 + 30  # cell A's entry stamp; the deck's stamp
ENTRY_MIN = 15 * 60  # cell B's entry stamp
PROFILE_MIN_DAYS = 63  # the intraday notebook's standing warm-up for the profile

SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
PLACEBO_DRAWS = 2000
N_CUTS = 10  # perturbation cut points for the causality assertion
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
FILLS = ("mid", "crossed")

CELL_A = "A 15:30 sign(s)"
CELL_B = "B 15:00 sign(s_rem)"
COMP_1500 = "always short 15:00 to settlement"
COMP_1530 = "always short 15:30 (deck)"
ORDER = [CELL_A, CELL_B, COMP_1500, COMP_1530]

# The gate: the deck's own rule table for the block-diagonal ridge on 866 days.
GATE = {
    "sign_mean": 0.094736,
    "sign_t": 2.480957,
    "sign_sharpe": 1.338322,
    "as_sharpe": 0.203779,
}
GATE_TOL = 1e-6


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def tick(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- frame build


def attach_iv_deck(pkg: pd.DataFrame) -> pd.DataFrame:
    """The deck's own hourly implied volatility of the package, stamp by stamp.

    The mean of the two quoted legs, replaced on a day whose vendor solve
    returned a bracket node by the volatility that reproduces the package
    midpoint over the window still to run (asl.bsm_invert_package_vol,
    asl.hourly_iv_from_total_vol). This is the deck's `iv_hourly_15_30`
    generalized from 0.5 h to the stamp's own hours to expiration; at 15:30
    it is that function exactly, which is what the gate needs.
    """
    out = pkg.copy()
    iv_c = pd.to_numeric(out["impl_volatility_c"], errors="coerce").astype(float)
    iv_p = pd.to_numeric(out["impl_volatility_p"], errors="coerce").astype(float)
    on_node_c = (asl.censor_vendor_iv(iv_c).isna() & iv_c.notna()).to_numpy()
    on_node_p = (asl.censor_vendor_iv(iv_p).isna() & iv_p.notna()).to_numpy()
    out["iv_capped"] = on_node_c | on_node_p
    quoted = pd.concat([iv_c, iv_p], axis=1).mean(axis=1)
    cap = out["iv_capped"] & (out["entry"] > 0)
    hourly = quoted.copy()
    if bool(cap.any()):
        inv = [
            asl.bsm_invert_package_vol(s, kc, kp, m, hours_remaining=float(h))
            for s, kc, kp, m, h in zip(
                out.loc[cap, "S"],
                out.loc[cap, "K_c"],
                out.loc[cap, "K_p"],
                out.loc[cap, "entry"],
                out.loc[cap, "hte"],
            )
        ]
        hourly.loc[cap] = [
            asl.hourly_iv_from_total_vol(v, hours_remaining=float(h))
            for v, h in zip(inv, out.loc[cap, "hte"])
        ]
    out["iv_hourly_deck"] = hourly
    out["iv_var_deck"] = (hourly / np.sqrt(2.0)) ** 2
    return out


def build_packages(
    t0: float, use_cache: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """(packages, quotes, refused cells, half-session dates) on the 10:00-15:30 stamps."""
    CACHE.mkdir(parents=True, exist_ok=True)
    st = CHAIN.stat()
    key = f"{st.st_size}_{st.st_mtime_ns}_v2"
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

    mins_all = chain["et"].dt.hour * 60 + chain["et"].dt.minute
    chain = chain[(mins_all >= 9 * 60 + 30) & (mins_all <= 16 * 60)].copy()
    chain, half = asl.drop_early_close(chain)
    tick(
        t0,
        f"regular hours + half-session drop: {len(chain):,} rows, {len(half)} days dropped",
    )

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
                "hours_to_expiration",
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
    pkg["hte"] = pkg[["hours_to_expiration_c", "hours_to_expiration_p"]].mean(axis=1)
    pkg = attach_iv_deck(pkg)
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
        "hte",
        "iv_hourly",
        "iv_var",
        "iv_capped",
        "iv_hourly_deck",
        "iv_var_deck",
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


def settle_rows(exp_days, k_c, k_p, s_close: pd.Series) -> np.ndarray:
    """Cash settlement of the given strikes at the official close (asl.settle_package)."""
    df = pd.DataFrame(
        {
            "expiration": pd.DatetimeIndex(
                pd.to_datetime(pd.Series(exp_days).to_numpy())
            ),
            "K_c": np.asarray(k_c, dtype=float),
            "K_p": np.asarray(k_p, dtype=float),
        }
    )
    out = asl.settle_package(df, s_close)
    return out["exit"].to_numpy(float)


# ------------------------------------------------------------ the work frame


def build_work(
    pkg: pd.DataFrame, quotes: pd.DataFrame, panel: pd.DataFrame, s_close: pd.Series
) -> pd.DataFrame:
    """The intraday notebook's scored frame: a return and a fresh forecast per bar.

    Exit = the next stamp's midpoint of the SAME strikes for bars 10:00-15:00,
    cash settlement at the official close for the 15:30 bar. The forecast is
    joined bar-end labelled: the trade bar at t takes the panel row stamped
    t + 30, which is the forecast issued at t for the bar actually held, and
    that row's realized variance is the bar's own.
    """
    p = pkg.sort_values(["date", "min"]).reset_index(drop=True).copy()
    p["nxt_min"] = p.groupby("date")["min"].shift(-1)
    p["is_last"] = p["nxt_min"].isna()
    key = p.loc[~p["is_last"], ["date", "nxt_min", "K_c", "K_p"]].rename(
        columns={"nxt_min": "min"}
    )
    key["min"] = key["min"].astype(int)
    mk = package_marks(quotes, key)
    p["exit_mark"] = np.nan
    p.loc[~p["is_last"], "exit_mark"] = mk["mid"].to_numpy(float)
    p["exit_settle"] = settle_rows(p["date"], p["K_c"], p["K_p"], s_close)
    p["exit"] = np.where(p["is_last"], p["exit_settle"], p["exit_mark"])
    keep = np.isfinite(p["entry"]) & np.isfinite(p["exit"]) & (p["entry"] > 0)
    p = p[keep].copy()
    p["R"] = p["exit"] / p["entry"] - 1.0

    pm = panel.loc[
        panel["mins"].isin([m + 30 for m in STAMP_MIN]),
        ["date", "mins", "rv_hat", "rv_raw", "in_fit"],
    ].copy()
    pm["min"] = pm["mins"].astype(int) - 30
    work = p.merge(pm.drop(columns=["mins"]), on=["date", "min"], how="left")
    work = work.dropna(subset=["R", "rv_hat"]).reset_index(drop=True)
    return work


def profile_share(work: pd.DataFrame) -> tuple[pd.Series, float, list[int]]:
    """(w at the 15:00 stamp, hours to the close at 15:00, the clock ladder).

    The intraday notebook's own share: the expanding per-clock mean of realized
    bar variance over PRIOR days only (min 63, lagged one day), divided by the
    same profile summed over the clocks from this one to the close.
    """
    clocks = sorted(int(c) for c in work["min"].unique())
    prof = work.pivot_table(
        index="date", columns="min", values="rv_raw", aggfunc="mean"
    ).sort_index()
    prof = prof[clocks]
    prof_exp = prof.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    rem_sum = prof_exp[clocks[::-1]].cumsum(axis=1)[clocks]
    w_slice = prof_exp / rem_sum
    h_rem = (len(clocks) - clocks.index(ENTRY_MIN)) * 0.5
    return w_slice[ENTRY_MIN].astype(float), float(h_rem), clocks


# ---------------------------------------------------------------- statistics


def safe_t(x: pd.Series) -> float:
    v = x.astype(float).dropna()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * np.sqrt(len(v))) if sd > 0 else float("nan")


def sharpe(x: pd.Series | np.ndarray) -> float:
    v = pd.Series(x).astype(float).dropna().to_numpy()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * ANN) if sd > 0 else float("nan")


def stat_row(
    pnl_pts: pd.Series, ret_prem: pd.Series, pos: pd.Series
) -> dict[str, float]:
    r = ret_prem.astype(float).dropna()
    p = pnl_pts.astype(float).dropna()
    q = pos.astype(float).reindex(r.index)
    cum = p.cumsum()
    n = int(len(r))
    n_buy = int((q > 0).sum())
    return {
        "n": float(n),
        "mean_prem": float(r.mean()),
        "t_prem": safe_t(r),
        "Sharpe_prem": sharpe(r),
        "mean_pts": float(p.mean()),
        "t_pts": safe_t(p),
        "Sharpe_pts": sharpe(p),
        "n_buy": float(n_buy),
        "pct_buy": 100.0 * n_buy / n if n else float("nan"),
        "maxDD_pts": float((cum - cum.cummax()).min()),
        "worst_pts": float(p.min()),
    }


def paired_stats(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """B - A on the common days: mean, plain t, HAC t, and the Sharpe difference."""
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
        "t_diff": safe_t(pd.Series(d)),
        "hac_t_diff": float(hac_t),
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
    """Sharpe percentile of the real sign among random signs at the rule's own long rate."""
    j = pd.concat(
        [long_r.rename("l"), short_r.rename("s"), pos.rename("q")], axis=1
    ).dropna()
    ell = j["l"].to_numpy(float)
    sh = j["s"].to_numpy(float)
    q = j["q"].to_numpy(float) > 0
    real = np.where(q, ell, sh)
    real_sh = float(real.mean() / real.std(ddof=1) * ANN)
    n = len(ell)
    p_long = float(q.mean())
    rng = np.random.default_rng(SEED)
    draw = rng.random((PLACEBO_DRAWS, n)) < p_long
    r = np.where(draw, ell[None, :], sh[None, :])
    s_draw = r.mean(axis=1) / r.std(axis=1, ddof=1) * ANN
    return {
        "n": float(n),
        "long_share_pct": 100.0 * p_long,
        "Sharpe_real": real_sh,
        "pctile": float(100.0 * (s_draw < real_sh).mean()),
        "placebo_median": float(np.median(s_draw)),
        "placebo_p05": float(np.percentile(s_draw, 5)),
        "placebo_p95": float(np.percentile(s_draw, 95)),
    }


# --------------------------------------------------------------- the two cells


def build_cells(
    pkg: pd.DataFrame,
    quotes: pd.DataFrame,
    work: pd.DataFrame,
    deck: pd.DataFrame,
    s_close: pd.Series,
) -> dict[str, pd.Series]:
    """Every per-day series both cells need, on the deck's day index."""
    days = pd.DatetimeIndex(deck.index)
    p15 = pkg[pkg["min"] == CLOSE_MIN].set_index("date").reindex(days)
    p00 = pkg[pkg["min"] == ENTRY_MIN].set_index("date").reindex(days)
    w_1500, h_rem, clocks = profile_share(work)
    w = w_1500.reindex(days)

    rv15 = work.loc[work["min"] == CLOSE_MIN].set_index("date")["rv_hat"].reindex(days)
    rv00 = work.loc[work["min"] == ENTRY_MIN].set_index("date")["rv_hat"].reindex(days)

    # Cell A: the deck's own object, rebuilt from the chain and the panel.
    iv_var_a = p15["iv_var_deck"].astype(float)
    signal_a = rv15 - iv_var_a
    pos_a = pd.Series(np.where(signal_a.to_numpy(float) > 0, 1.0, -1.0), index=days)
    settle_a = pd.Series(settle_rows(days, p15["K_c"], p15["K_p"], s_close), index=days)
    entry_a = p15["entry"].astype(float)

    # Cell B: the remaining-hour reading at 15:00.
    iv_var_b = (p00["iv_hourly"].astype(float) ** 2) * h_rem
    rv_hat_rem = rv00 / w
    signal_b = rv_hat_rem - iv_var_b
    pos_b = pd.Series(np.where(signal_b.to_numpy(float) > 0, 1.0, -1.0), index=days)
    pos_b = pos_b.where(np.isfinite(signal_b.to_numpy(float)))
    settle_b = pd.Series(settle_rows(days, p00["K_c"], p00["K_p"], s_close), index=days)
    entry_b = p00["entry"].astype(float)

    # The 15:30 mark of the strikes bought at 15:00 - the decomposition's hinge.
    key = pd.DataFrame(
        {
            "date": days,
            "min": CLOSE_MIN,
            "K_c": p00["K_c"].to_numpy(),
            "K_p": p00["K_p"].to_numpy(),
        }
    )
    mark = package_marks(quotes, key)
    mark.index = days

    return {
        "days": pd.Series(days, index=days),
        "w_1500": w,
        "h_rem": pd.Series(h_rem, index=days),
        "clocks": pd.Series([len(clocks)] * len(days), index=days),
        "rv_hat_1530": rv15,
        "rv_hat_1500": rv00,
        "rv_hat_rem": rv_hat_rem,
        "iv_var_1530": iv_var_a,
        "iv_var_rem": iv_var_b,
        "signal_1530": signal_a,
        "signal_rem": signal_b,
        "pos_A": pos_a,
        "pos_B": pos_b,
        "entry_A": entry_a,
        "bid_A": p15["bid_entry"].astype(float),
        "ask_A": p15["ask_entry"].astype(float),
        "settle_A": settle_a,
        "entry_B": entry_b,
        "bid_B": p00["bid_entry"].astype(float),
        "ask_B": p00["ask_entry"].astype(float),
        "settle_B": settle_b,
        "mark_B_1530": mark["mid"].astype(float),
        "S_1530": p15["S"].astype(float),
        "S_1500": p00["S"].astype(float),
        "K_c_A": p15["K_c"].astype(float),
        "K_p_A": p15["K_p"].astype(float),
        "K_c_B": p00["K_c"].astype(float),
        "K_p_B": p00["K_p"].astype(float),
        "hte_1500": p00["hte"].astype(float),
        "hte_1530": p15["hte"].astype(float),
    }


def series_for(
    c: dict[str, pd.Series], name: str
) -> tuple[pd.Series, dict[str, pd.Series], dict[str, pd.Series]]:
    """(position, points by fill, per-premium return by fill) for one construction."""
    if name in (CELL_A, COMP_1530):
        q = c["pos_A"] if name == CELL_A else pd.Series(-1.0, index=c["pos_A"].index)
        q = q.where(c["entry_A"].notna())
        entry, bid, ask, ex = c["entry_A"], c["bid_A"], c["ask_A"], c["settle_A"]
    else:
        q = c["pos_B"] if name == CELL_B else pd.Series(-1.0, index=c["pos_B"].index)
        q = q.where(c["entry_B"].notna())
        entry, bid, ask, ex = c["entry_B"], c["bid_B"], c["ask_B"], c["settle_B"]
    pts = {
        "mid": q * (ex - entry),
        "crossed": pd.Series(
            np.where(q.to_numpy(float) > 0, ex - ask, bid - ex), index=q.index
        ).where(q.notna()),
    }
    ret = {
        "mid": q * (ex / entry - 1.0),
        "crossed": asl.crossed_premium_return(q.fillna(0.0), ex, bid, ask).where(
            q.notna()
        ),
    }
    return q, pts, ret


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Proposal 09 - the last hour: the close trade against a 15:00 entry."
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
        "Proposal 09 - two pre-registered cells: the close trade at 15:30 (A) and the "
        "remaining-hour reading at 15:00 (B)"
    )
    print("=" * 110)
    print(
        "CAVEAT: the 15:00 cell was flagged by proposal 08 as the best of 24 cells. "
        "This is its pre-registered replication on two cells."
    )
    print(
        "MULTIPLICITY: two pre-registered cells, nothing selected. No grid is searched "
        "in this file; the eight-tag table is a robustness display of the same two cells."
    )
    print(f"Forecast: {asl.YHAT_LABEL['blk2']}. Fills: midpoint and crossed spread.")

    pkg, quotes, refused, half = build_packages(t0, use_cache=not args.no_cache)
    deck = pd.read_parquet(DECK_DIR / "daily_blk2.parquet")
    deck.index = pd.DatetimeIndex(deck.index).normalize()
    days = pd.DatetimeIndex(deck.index)
    close_px, close_path = load_official_close()
    s_close = close_px.reindex(days)

    print("\n--- 0. Sample ---")
    deck_st = (DECK_DIR / "daily_blk2.parquet").stat()
    deck_fp = hashlib.sha1(
        np.ascontiguousarray(deck[["rv_hat", "signal", "pos", "R"]].to_numpy(float))
    ).hexdigest()[:12]
    print(
        f"deck file: daily_blk2.parquet {deck_st.st_size} bytes, forecast fingerprint {deck_fp}"
    )
    print(f"official close cache: {close_path.name} ({len(close_px)} sessions)")
    print(f"deck days {len(days)} from {days.min().date()} to {days.max().date()}")
    print(f"realized trade days a year on this frame: {asl.trades_per_year(days):.1f}")
    print(
        f"half sessions dropped by the shared rule ({len(half)}): "
        f"{', '.join(d.strftime('%Y-%m-%d') for d in half)}"
    )
    pkg = pkg[pkg["date"].isin(days)].copy()
    quotes = quotes[quotes["date"].isin(days)].copy()
    print(
        f"packages on the deck's days: {len(pkg):,} on {pkg['date'].nunique()} days, "
        f"{len(STAMP_MIN)} stamps 10:00-15:30 (the 16:00 stamp is excluded outright)"
    )
    hte = pkg.groupby("min")["hte"].median()
    print("hours to expiration by stamp (chain median):")
    print(
        "  "
        + ", ".join(f"{hhmm(int(m))} {v:.2f}" for m, v in hte.items())
        + f"  -> the 15:00 entry has {float(hte.loc[ENTRY_MIN]):.2f} h to the close"
    )

    panel = asl.load_yhat_panel_mz(asl.yhat_paths(REPO)["blk2"])
    work = build_work(pkg, quotes, panel, s_close)
    tick(t0, f"work frame {len(work):,} bars on {work['date'].nunique()} days")
    assert bool(work["in_fit"].all()), (
        "a joined trade bar is outside the smear's fit mask"
    )
    print("every joined bar is inside the recalibration's session fit mask (in_fit)")
    same_days = set(work["date"].unique()) == set(days)
    print(
        f"the scored frame's dates equal the deck's {len(days)} days: {same_days} "
        f"({work['date'].nunique()} dates)"
    )

    c = build_cells(pkg, quotes, work, deck, s_close)

    # ------------------------------------------------------------------ gate
    print(
        "\n--- 1. GATE: the deck's rule table, rebuilt from the chain and the panel ---"
    )
    for col, mine in (
        ("K_c", c["K_c_A"]),
        ("K_p", c["K_p_A"]),
        ("S", c["S_1530"]),
        ("entry", c["entry_A"]),
        ("iv_var", c["iv_var_1530"]),
        ("rv_hat", c["rv_hat_1530"]),
        ("exit", c["settle_A"]),
    ):
        print(
            f"  independent 15:30 {col:<7s} vs the deck's: max |difference| "
            f"{float((mine - deck[col].astype(float)).abs().max()):.3e}"
        )
    r_a = c["settle_A"] / c["entry_A"] - 1.0
    print(
        f"  independent per-premium return vs the deck's R: max |difference| "
        f"{float((r_a - deck['R'].astype(float)).abs().max()):.3e}"
    )
    print(
        f"  independent position vs the deck's pos: days that differ "
        f"{int((c['pos_A'] != deck['pos'].astype(float)).sum())}"
    )
    gate_rows: list[dict[str, float | str]] = []
    for name, q in (
        ("always short", pd.Series(-1.0, index=days)),
        ("sign(s)", c["pos_A"]),
    ):
        gate_row = asl.rule_row(q * r_a, q)
        gate_rows.append(
            {
                "rule": name,
                "n": float(gate_row["n"]),
                "mean": float(gate_row["mean"]),
                "t": float(gate_row["t_mean"]),
                "Sharpe": float(gate_row["Sharpe_ann"]),
                "pct_buy": float(gate_row["pct_buy"]),
            }
        )
        print(
            f"  reconstruction {name:<13s} n {int(gate_row['n'])}  "
            f"mean {gate_row['mean']:.6f}  t {gate_row['t_mean']:.6f}  "
            f"Sharpe {gate_row['Sharpe_ann']:.6f}  buy {gate_row['pct_buy']:.4f}%"
        )
    got = {
        "sign_mean": float(gate_rows[1]["mean"]),
        "sign_t": float(gate_rows[1]["t"]),
        "sign_sharpe": float(gate_rows[1]["Sharpe"]),
        "as_sharpe": float(gate_rows[0]["Sharpe"]),
    }
    for k, target in GATE.items():
        d = abs(got[k] - target)
        print(f"  gate {k:<12s} target {target:.6f}  got {got[k]:.6f}  |diff| {d:.2e}")
        assert d < GATE_TOL, f"GATE FAILED on {k}: {got[k]} vs {target}"
    assert float(gate_rows[1]["n"]) == 866.0, (
        "GATE FAILED: the deck frame is not 866 days"
    )
    print("  GATE PASSED: the reconstruction is the deck's close trade.")
    pd.DataFrame(gate_rows).to_csv(OUT / "09_gate.csv", index=False)

    # ------------------------------------------------------- cell B and causality
    print("\n--- 2. Cell B: the remaining-hour reading at 15:00 ---")
    w = c["w_1500"]
    n_w = int(w.notna().sum())
    n_iv = int(c["iv_var_rem"].notna().sum())
    n_b = int(c["pos_B"].notna().sum())
    print(
        f"  w_1500 = the 15:00-15:30 bar's share of the remaining hour's variance, "
        f"expanding mean over prior days (min {PROFILE_MIN_DAYS}, lagged one day)"
    )
    print(
        f"  median w_1500 {float(w.median()):.4f}; 5th-95th "
        f"{float(w.quantile(0.05)):.4f}-{float(w.quantile(0.95)):.4f}; "
        f"days with a share {n_w} of {len(days)} (the rest are the warm-up)"
    )
    print(
        f"  iv_var_rem = iv_hourly(15:00)^2 x h_rem with h_rem = {float(c['h_rem'].iloc[0]):.1f} h; "
        f"days with an uncensored 15:00 implied volatility {n_iv}"
    )
    print(
        f"  days with a 15:00 package {int(c['entry_B'].notna().sum())}; "
        f"days cell B can trade {n_b}; days it cannot {len(days) - n_b} "
        f"(warm-up {int((~w.notna() & c['entry_B'].notna()).sum())}, "
        f"censored implied volatility {int((w.notna() & ~c['iv_var_rem'].notna()).sum())}, "
        f"no package {int(c['entry_B'].isna().sum())})"
    )
    print(
        f"  median rv_hat_rem/iv_var_rem {float((c['rv_hat_rem'] / c['iv_var_rem']).median()):.4f}; "
        f"cell B buys on {100.0 * float((c['pos_B'] > 0).sum()) / max(n_b, 1):.2f}% of its days"
    )

    print(
        "\n--- 2b. Causality: the 15:00 decision cannot see day d's realized values ---"
    )
    yhat_p = panel["yhat"].to_numpy(float)
    rv_raw_p = panel["rv_raw"].to_numpy(float)
    base_p = panel["baseline"].to_numpy(float)
    day_codes, uniq = pd.factorize(panel["date"], sort=True)
    lo_m, hi_m = asl.FIT_MASK_MINUTES
    rth = (
        (panel["mins"] >= lo_m)
        & (panel["mins"] <= hi_m)
        & (~panel["early_close"])
        & panel["session_date"]
    ).to_numpy()
    code_of = {pd.Timestamp(d): k for k, d in enumerate(uniq)}
    b_days = pd.DatetimeIndex(c["pos_B"].dropna().index)
    cuts = b_days[np.linspace(0, len(b_days) - 1, N_CUTS).astype(int)]
    stamp_1530 = (panel["mins"] == CLOSE_MIN).to_numpy()
    n_viol_w = n_viol_rv = 0
    for d in cuts:
        # (i) the profile share: triple realized variance on day d and every later day
        pert = work.copy()
        pert.loc[pert["date"] >= d, "rv_raw"] = (
            pert.loc[pert["date"] >= d, "rv_raw"] * 3.0
        )
        w_p, _, _ = profile_share(pert)
        if not np.isclose(float(w_p.loc[d]), float(w.loc[d]), rtol=0.0, atol=1e-15):
            n_viol_w += 1
        # (ii) the forecast: triple realized variance on every panel row from day d's
        # 15:30 stamp on (the bar the 15:00 decision is about, and everything after)
        cut_row = int(np.flatnonzero((panel["date"] == d).to_numpy() & stamp_1530)[0])
        rv_p = rv_raw_p.copy()
        rv_p[cut_row:] *= 3.0
        code = code_of[pd.Timestamp(d)]
        rv_new, _, _ = asl.second_order_mz(
            yhat_p,
            rv_p,
            base_p,
            day_codes,
            len(uniq),
            need_days={code},
            fit_mask=rth,
        )
        if not np.isclose(
            float(rv_new[cut_row]), float(c["rv_hat_1500"].loc[d]), rtol=1e-12, atol=0.0
        ):
            n_viol_rv += 1
    print(
        f"  {N_CUTS} cut days from {cuts[0].date()} to {cuts[-1].date()}: "
        f"tripling realized variance on day d and after moves w_1500(d) on {n_viol_w} of {N_CUTS}, "
        f"and rv_hat_1500(d) on {n_viol_rv} of {N_CUTS}"
    )
    assert n_viol_w == 0 and n_viol_rv == 0, "CAUSALITY VIOLATED"
    print(
        "  the third input is the 15:00 quote itself (the two picked legs' vendor implied "
        "volatility and midpoints), which is a 15:00 object by construction"
    )
    print(
        "  CAUSALITY ASSERTED: day d's position does not move when day d's realized values do."
    )

    # ------------------------------------------------------- the results table
    pos_map: dict[str, pd.Series] = {}
    pts_map: dict[tuple[str, str], pd.Series] = {}
    ret_map: dict[tuple[str, str], pd.Series] = {}
    for nm in ORDER:
        q, pts, ret = series_for(c, nm)
        pos_map[nm] = q
        for fill in FILLS:
            pts_map[(nm, fill)] = pts[fill]
            ret_map[(nm, fill)] = ret[fill]

    common = pd.DatetimeIndex(
        days[
            pd.concat([ret_map[(nm, "mid")] for nm in ORDER], axis=1)
            .notna()
            .all(axis=1)
        ]
    )
    print(
        f"\n--- 3. The two cells and the two comparators, {len(common)} common days "
        f"({common.min().date()} to {common.max().date()}) ---"
    )
    rows: list[dict[str, object]] = []
    for frame_name, ix in (("common", common), ("deck", days)):
        for nm in ORDER:
            if frame_name == "deck" and nm in (CELL_B, COMP_1500):
                continue
            for fill in FILLS:
                rows.append(
                    {
                        "frame": frame_name,
                        "construction": nm,
                        "fill": fill,
                        **stat_row(
                            pts_map[(nm, fill)].reindex(ix),
                            ret_map[(nm, fill)].reindex(ix),
                            pos_map[nm].reindex(ix),
                        ),
                    }
                )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "09_cells.csv", index=False)
    show = res.set_index(["frame", "construction", "fill"])[
        [
            "n",
            "mean_prem",
            "t_prem",
            "Sharpe_prem",
            "mean_pts",
            "t_pts",
            "Sharpe_pts",
            "n_buy",
            "pct_buy",
            "maxDD_pts",
            "worst_pts",
        ]
    ]
    print(
        "(mean_prem/t_prem/Sharpe_prem are per unit of the entry premium actually paid; "
        "mean_pts/t_pts/Sharpe_pts and the drawdown are index points per straddle)"
    )
    print(show.round(4).to_string())

    # ---------------------------------------------------------- paired B - A
    print("\n--- 4. The paired difference B - A on the same days ---")
    pair_rows: list[dict[str, object]] = []
    for fill in FILLS:
        for unit, book in (("per premium", ret_map), ("points", pts_map)):
            pair_rows.append(
                {
                    "fill": fill,
                    "unit": unit,
                    **paired_stats(
                        book[(CELL_A, fill)].reindex(common),
                        book[(CELL_B, fill)].reindex(common),
                    ),
                }
            )
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(OUT / "09_paired.csv", index=False)
    print(
        f"(HAC t at lag floor(1.5 n^(1/3)); Sharpe difference by circular block bootstrap, "
        f"block {BOOT_BLOCK}, B = {BOOT_B}, rng({SEED}))"
    )
    print(pair.round(4).to_string(index=False))

    # --------------------------------------------- signals and the realized sign
    print(
        "\n--- 5. The two signals against each other and against the realized sign ---"
    )
    rv_1530 = (
        panel.loc[panel["mins"] == CLOSE_MIN].set_index("date")["rv_raw"].reindex(days)
    )
    rv_1600 = (
        panel.loc[panel["mins"] == 16 * 60].set_index("date")["rv_raw"].reindex(days)
    )
    rv_rem_1500 = rv_1530 + rv_1600
    oracle_1500 = np.sign(rv_rem_1500 - c["iv_var_rem"])
    oracle_1530 = np.sign(rv_1600 - c["iv_var_1530"])
    agree = (
        (np.sign(c["signal_rem"]) == np.sign(c["signal_1530"]))
        .where(c["signal_rem"].notna() & c["signal_1530"].notna())
        .reindex(common)
    )
    sig_rows: list[dict[str, object]] = []
    for sname, q, oname, orc in (
        ("15:00 sign(s_rem)", c["pos_B"], "[15:00,16:00]", oracle_1500),
        ("15:30 sign(s)", c["pos_A"], "[15:30,16:00]", oracle_1530),
        ("15:00 sign(s_rem)", c["pos_B"], "[15:30,16:00]", oracle_1530),
        ("15:30 sign(s)", c["pos_A"], "[15:00,16:00]", oracle_1500),
    ):
        j = pd.concat([q.rename("q"), orc.rename("o")], axis=1).reindex(common).dropna()
        hit = float((np.sign(j["q"]) == np.sign(j["o"])).mean())
        sig_rows.append(
            {
                "signal": sname,
                "window": oname,
                "n": int(len(j)),
                "hit_rate": hit,
                "signal_long_pct": 100.0 * float((j["q"] > 0).mean()),
                "oracle_long_pct": 100.0 * float((j["o"] > 0).mean()),
            }
        )
    sig = pd.DataFrame(sig_rows)
    sig["sign_agreement_15_00_vs_15_30_pct"] = 100.0 * float(agree.mean())
    sig.to_csv(OUT / "09_signal_agreement.csv", index=False)
    print(
        f"sign agreement between the 15:00 and the 15:30 signal on the {len(common)} common "
        f"days: {100.0 * float(agree.mean()):.2f}%"
    )
    print(
        "hit rate = the share of days the position matches sign(remaining realized variance "
        "- remaining implied variance) over the stated window"
    )
    print(sig.round(4).to_string(index=False))

    # ------------------------------------------------------------- placebo (B)
    print(
        f"\n--- 6. Random-sign placebo for cell B ({PLACEBO_DRAWS} draws, rng({SEED}), "
        f"rate-matched to B's long share) ---"
    )
    pl_rows: list[dict[str, object]] = []
    long_mid = (c["settle_B"] / c["entry_B"] - 1.0).reindex(common)
    long_cr = (c["settle_B"] / c["ask_B"] - 1.0).reindex(common)
    short_cr = (1.0 - c["settle_B"] / c["bid_B"]).reindex(common)
    for fill, ell, sh in (
        ("mid", long_mid, -long_mid),
        ("crossed", long_cr, short_cr),
    ):
        pl_rows.append(
            {"fill": fill, **placebo_rate_matched(ell, sh, c["pos_B"].reindex(common))}
        )
    placebo = pd.DataFrame(pl_rows)
    placebo.to_csv(OUT / "09_placebo.csv", index=False)
    print(placebo.round(4).to_string(index=False))

    # ------------------------------------------------------------ decomposition
    print("\n--- 7. Cell B decomposed into its two bars, in index points ---")
    q_b = pos_map[CELL_B].reindex(common)
    mark = c["mark_B_1530"].reindex(common)
    entry_mid = c["entry_B"].reindex(common)
    bid_b = c["bid_B"].reindex(common)
    ask_b = c["ask_B"].reindex(common)
    settle_b = c["settle_B"].reindex(common)
    is_long = q_b.to_numpy(float) > 0
    paid = pd.Series(np.where(is_long, ask_b, bid_b), index=common)
    legs: dict[tuple[str, str], pd.Series] = {
        ("mid", "15:00-15:30"): q_b * (mark - entry_mid),
        ("mid", "15:30-16:00"): q_b * (settle_b - mark),
        ("crossed", "15:00-15:30"): q_b * (mark - paid),
        ("crossed", "15:30-16:00"): q_b * (settle_b - mark),
    }
    dec_rows: list[dict[str, object]] = []
    for fill in FILLS:
        tot = pts_map[(CELL_B, fill)].reindex(common)
        gap = float(
            (legs[(fill, "15:00-15:30")] + legs[(fill, "15:30-16:00")] - tot)
            .abs()
            .max()
        )
        for bar in ("15:00-15:30", "15:30-16:00"):
            v = legs[(fill, bar)].dropna()
            dec_rows.append(
                {
                    "fill": fill,
                    "bar": bar,
                    "n": int(len(v)),
                    "mean_pts": float(v.mean()),
                    "t_pts": safe_t(v),
                    "sd_pts": float(v.std(ddof=1)),
                    "share_of_total_pct": 100.0 * float(v.mean()) / float(tot.mean()),
                    "Sharpe_pts": sharpe(v),
                    "additivity_max_abs_gap_pts": gap,
                }
            )
        dec_rows.append(
            {
                "fill": fill,
                "bar": "total",
                "n": int(tot.notna().sum()),
                "mean_pts": float(tot.mean()),
                "t_pts": safe_t(tot),
                "sd_pts": float(tot.std(ddof=1)),
                "share_of_total_pct": 100.0,
                "Sharpe_pts": sharpe(tot),
                "additivity_max_abs_gap_pts": gap,
            }
        )
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(OUT / "09_decomposition.csv", index=False)
    print(
        "(the crossed case charges the whole entry half-spread to the first bar, which is "
        "where it is paid; cash settlement pays no exit spread)"
    )
    print(dec.round(4).to_string(index=False))

    # -------------------------------------------------------- eight-tag display
    print("\n--- 8. The same two cells under all eight forecasts (secondary) ---")
    tag_rows: list[dict[str, object]] = []
    for tag in asl.MODEL_ORDER:
        dpath = DECK_DIR / f"daily_{tag}.parquet"
        ypath = asl.yhat_paths(REPO)[tag]
        if not dpath.exists() or not ypath.exists():
            print(
                f"  {tag}: missing {'deck file' if not dpath.exists() else 'forecast table'}"
            )
            continue
        dk = pd.read_parquet(dpath)
        dk.index = pd.DatetimeIndex(dk.index).normalize()
        pn = asl.load_yhat_panel_mz(ypath)
        wk = build_work(
            pkg[pkg["date"].isin(dk.index)], quotes, pn, close_px.reindex(dk.index)
        )
        ct = build_cells(
            pkg[pkg["date"].isin(dk.index)], quotes, wk, dk, close_px.reindex(dk.index)
        )
        pm: dict[str, pd.Series] = {}
        rm: dict[tuple[str, str], pd.Series] = {}
        for nm in ORDER:
            q, _, ret = series_for(ct, nm)
            pm[nm] = q
            for fill in FILLS:
                rm[(nm, fill)] = ret[fill]
        cm = pd.DatetimeIndex(
            dk.index[
                pd.concat([rm[(nm, "mid")] for nm in ORDER], axis=1).notna().all(axis=1)
            ]
        )
        tag_row: dict[str, object] = {
            "tag": tag,
            "forecast": asl.YHAT_LABEL[tag],
            "n_deck": int(len(dk)),
            "n_common": int(len(cm)),
            "pct_buy_A": 100.0 * float((pm[CELL_A].reindex(cm) > 0).mean()),
            "pct_buy_B": 100.0 * float((pm[CELL_B].reindex(cm) > 0).mean()),
        }
        for nm, short in (
            (CELL_A, "A"),
            (CELL_B, "B"),
            (COMP_1500, "AS1500"),
            (COMP_1530, "AS1530"),
        ):
            for fill in FILLS:
                tag_row[f"Sharpe_{short}_{fill}"] = sharpe(rm[(nm, fill)].reindex(cm))
        for fill in FILLS:
            st = paired_stats(
                rm[(CELL_A, fill)].reindex(cm), rm[(CELL_B, fill)].reindex(cm)
            )
            tag_row[f"dSharpe_{fill}"] = st["dSharpe"]
            tag_row[f"dS_pctile_lo_{fill}"] = st["pctile_lo"]
            tag_row[f"dS_pctile_hi_{fill}"] = st["pctile_hi"]
            tag_row[f"dS_basic_lo_{fill}"] = st["basic_lo"]
            tag_row[f"dS_basic_hi_{fill}"] = st["basic_hi"]
            tag_row[f"hac_t_{fill}"] = st["hac_t_diff"]
        tag_rows.append(tag_row)
        tick(t0, f"tag {tag} done")
    tags = pd.DataFrame(tag_rows)
    tags.to_csv(OUT / "09_all_tags.csv", index=False)
    print(
        tags[
            [
                "tag",
                "n_common",
                "Sharpe_A_mid",
                "Sharpe_B_mid",
                "dSharpe_mid",
                "dS_pctile_lo_mid",
                "dS_pctile_hi_mid",
                "Sharpe_A_crossed",
                "Sharpe_B_crossed",
                "dSharpe_crossed",
                "dS_pctile_lo_crossed",
                "dS_pctile_hi_crossed",
                "pct_buy_B",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # ------------------------------------------------------------ daily series
    daily = pd.DataFrame(
        {f"{nm} | {fill}": pts_map[(nm, fill)] for nm in ORDER for fill in FILLS}
    )
    daily["pos_A"] = pos_map[CELL_A]
    daily["pos_B"] = pos_map[CELL_B]
    daily["w_1500"] = c["w_1500"]
    daily["signal_1530"] = c["signal_1530"]
    daily["signal_rem_1500"] = c["signal_rem"]
    daily.index.name = "date"
    daily.to_csv(OUT / "09_daily_pnl_points.csv")

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for ax, fill in zip(axes, FILLS):
        for nm, col, lw in (
            (CELL_A, "crimson", 2.0),
            (CELL_B, "#1f4e79", 2.0),
            (COMP_1500, "0.35", 1.1),
            (COMP_1530, "0.60", 1.1),
        ):
            ax.plot(
                common,
                pts_map[(nm, fill)].reindex(common).cumsum(),
                lw=lw,
                color=col,
                ls="--" if nm.startswith("always") else "-",
                label=nm,
            )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(
            f"cumulative points, {fill} fills ({len(common)} days)", fontsize=10
        )
        ax.set_xlabel("date")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative P&L, index points per straddle")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT / "09_cum_points.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------- verdict
    lo_c = float(
        pair.loc[
            (pair["fill"] == "crossed") & (pair["unit"] == "per premium"), "pctile_lo"
        ].iloc[0]
    )
    hi_c = float(
        pair.loc[
            (pair["fill"] == "crossed") & (pair["unit"] == "per premium"), "pctile_hi"
        ].iloc[0]
    )
    blo_c = float(
        pair.loc[
            (pair["fill"] == "crossed") & (pair["unit"] == "per premium"), "basic_lo"
        ].iloc[0]
    )
    bhi_c = float(
        pair.loc[
            (pair["fill"] == "crossed") & (pair["unit"] == "per premium"), "basic_hi"
        ].iloc[0]
    )
    excl = (lo_c > 0.0 or hi_c < 0.0) and (blo_c > 0.0 or bhi_c < 0.0)
    print("\n--- 9. Verdict ---")
    print(
        f"crossed-spread Sharpe difference B - A: percentile 95% [{lo_c:+.4f}, {hi_c:+.4f}], "
        f"basic 95% [{blo_c:+.4f}, {bhi_c:+.4f}]"
    )
    if excl:
        print(
            "the interval excludes zero at the crossed spread: THE HEAD START IS REAL."
        )
    else:
        print(
            "the interval includes zero at the crossed spread: THE FRONTIER IS THE LAST BAR "
            "AND CELL B IS NOT ADOPTED."
        )

    print("\nrefused cells on the deck's days at the two entry stamps:")
    rf = refused.copy()
    rf["date"] = pd.to_datetime(rf["date"])
    rf = rf[rf["date"].isin(days) & rf["hhmm"].isin([hhmm(ENTRY_MIN), hhmm(CLOSE_MIN)])]
    print(rf.to_string(index=False) if len(rf) else "none")

    print("\nwrote:")
    for p in sorted(OUT.glob("09_*")):
        print("  ", p.relative_to(REPO))
    tick(t0, "done")


if __name__ == "__main__":
    main()
