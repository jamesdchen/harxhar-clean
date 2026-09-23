"""Study 74 -- tail-day capture: is there a better way to catch the days that carry sign(s)?

The 15:30 sign(s) P&L lives on about 20 of 866 days.  The forecaster minimises
QLIKE (a mean), the sign rule compares means, and studies 70-72 re-graded the
same information (logistic / OLS of R, sizing by s, the trailing surprise)
without touching the tail.  This study asks the tail question directly.

A. DIAGNOSTIC.  Tail days three ways (the 20 largest long-straddle returns R;
   R > 1, the straddle at least doubles; the top 2.5% of |R|): how many did
   sign(s) buy, what they were, what share of sign(s)'s P&L they carry; then
   the ex-ante separability of tail days, one 15:30-observable feature at a
   time (AUC with a day-block bootstrap interval, hit rate in the feature's
   top decile).
B. THE CLASSIFIER RULE (pre-registered).  Logistic (L2, standardised) on a
   feature list FIXED before results: s, QLIKE_63, isur_63, log VIX,
   log(vix3m / vix), the 15:00-15:30 bar's variance over the day's earlier
   bars, |day return so far|, month-end, third Friday, FOMC day.  Expanding
   causal fit, warm-up 252 sessions, refit every 21.  Rules: buy when
   p-hat > p*, the break-even p* = -E[R | not tail] / (E[R | tail] - E[R | not
   tail]) from the trailing conditional means at the ask; buy when p-hat is
   above its trailing top quintile; sign(s) AND p-hat above its trailing median;
   sign(s) OR the break-even rule.  Short leg -1 otherwise, as in sign(s).
   Compared to sign(s) at EQUAL AVERAGE LONG EXPOSURE (the long size is scaled
   so the mean long exposure over the scored days matches).  Supported =
   crossed Sharpe beats sign(s) with the day-block bootstrap interval on the
   daily P&L difference excluding zero AND a within-year placebo (p-hat
   permuted within calendar year, 200 draws) at p < 0.05 AND the direction
   holding in both halves (2021 | 2022-24).
C. STRUCTURE ON TAIL DAYS.  On the buy days of each rule, the straddle against
   the 1% and 2% OTM strangles (nearest listed strikes at 15:30, mids and the
   ask), settlement payoff, per unit premium and per unit of spot.
D. THE RAW-FORECAST RULE (pre-registered because study 72 flagged it): sign(
   yhat^2 x baseline - iv_var) on the smear-free forecast vs the recalibrated
   sign(s), both decks, same days; placebo = the raw-minus-recalibrated gap
   permuted within year; both halves; which buys it drops by s-quintile.

Everything causal: trailing measures end the previous session; the classifier
sees only prior days; thresholds are trailing.  Inputs: the two decks
(sub_live_ridge, blk2), the forecast tables, core_stats (RTH bars), the VIX
prints at 15:30 (feed ends 2024-02-12: days without a VIX print drop out of the
classifier's scored set), the release calendar, and one pyarrow-filtered chain
read restricted to the deck's 15:30 stamps for part C.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    sys.path.insert(0, str(p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "74"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
CORE = ROOT / "data" / "core_stats.parquet"
VIX = ROOT / "data" / "vix_and_voldemand.parquet"
CHAIN = ROOT / "data" / "spxw_chain.parquet"
TAGS = ("sub_live_ridge", "blk2")
WARMUP = 252
REFIT = 21
TRAIL = 252
TOP_DAYS = 20
TAIL_Q = 0.975  # the top 2.5% of |R|
DOUBLE = 1.0  # R > 1: the straddle at least doubles
N_PLACEBO = 200
SEED = 74
SPLIT = pd.Timestamp("2022-01-01")
AUC_BOOT_B, AUC_BOOT_BLOCK = 2000, 21
AUC_ADMIT = 0.60  # a feature "separates" if its AUC interval clears 0.5 and the point is beyond 0.6 / 0.4
RTH_CLOCKS = [
    f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)
]  # bar-end labels 10:00..15:30
# 1% and 2% were pre-registered; 0.5% was ADDED AFTER the first run showed the 1% and
# 2% wings are tick-priced at 30 minutes to the close (0.23 bp of spot, ~4 sigma):
# they expired worthless on every buy day of every rule.  Reported, not a criterion.
WINGS = (0.005, 0.01, 0.02)


def _wtag(w: float) -> str:
    return "w" + f"{100 * w:g}".replace(".", "p")


def _wname(w: float) -> str:
    return f"strangle {100 * w:g}% OTM"


CLASSIFIER_FEATURES = [  # fixed before part B's results were seen
    "s",
    "qlike_63",
    "isur_63",
    "log_vix",
    "vix_slope",
    "log_last_bar_ratio",
    "abs_day_ret",
    "month_end",
    "third_friday",
    "fomc_day",
]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P71 = _load(HERE / "71_conviction_sizing.py", "p71_conviction")
P72 = _load(HERE / "72_vol_surprise_clustering.py", "p72_surprise")
sharpe, book, max_drawdown, top_days, block_ci = (
    P71.sharpe,
    P71.book,
    P71.max_drawdown,
    P71.top_days,
    P71.block_ci,
)


# ------------------------------------------------------------------ inputs --
def deck_full(tag: str) -> pd.DataFrame:
    """Study 71's gated deck frame plus the straddle geometry part C needs."""
    d = P71.load_deck(tag)
    raw = pd.read_parquet(DECK / f"daily_{tag}.parquet").sort_index()
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index)).normalize()
    for c in ("S", "S_close", "K_c", "K_p", "entry", "timestamp_c"):
        d[c] = raw[c].to_numpy()
    d["ask"] = (raw["ask_c"] + raw["ask_p"]).to_numpy(float)
    d["exit"] = raw["exit"].to_numpy(float)
    d["entry_over_spot"] = d["entry"].to_numpy(float) / d["S"].to_numpy(float)
    return d


def vix_family_1530(days: pd.DatetimeIndex) -> pd.DataFrame:
    v = pd.read_parquet(VIX, columns=["endbartime", "vix", "vvix", "vix3m"])
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    at = v["endbartime"].dt.strftime("%H:%M") == "15:30"
    f = v.loc[at].set_index(v.loc[at, "endbartime"].dt.normalize())[
        ["vix", "vvix", "vix3m"]
    ]
    return f[~f.index.duplicated()].reindex(days).astype(float)


def intraday_features(days: pd.DatetimeIndex) -> pd.DataFrame:
    """From the RTH bars ending 10:00..15:30 (bar-end labels; all known at 15:30)."""
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret", "sumret2"])
    c["endbartime"] = pd.to_datetime(c["endbartime"])
    c["day"] = c["endbartime"].dt.normalize()
    c["hhmm"] = c["endbartime"].dt.strftime("%H:%M")
    c = c[c["day"].isin(days) & c["hhmm"].isin(RTH_CLOCKS)]
    rv = c.pivot_table(index="day", columns="hhmm", values="sumret2").reindex(days)
    rt = c.pivot_table(index="day", columns="hhmm", values="sumret").reindex(days)
    rv = rv.reindex(columns=RTH_CLOCKS)
    rt = rt.reindex(columns=RTH_CLOCKS)
    last = rv["15:30"]
    earlier = rv[RTH_CLOCKS[:-1]].mean(axis=1)
    path = np.cumsum(rt.to_numpy(float), axis=1)
    path0 = np.column_stack([np.zeros(len(rt)), path])
    out = pd.DataFrame(index=days)
    out["rv_last_bar"] = last
    out["last_bar_ratio"] = last / earlier
    out["log_last_bar_ratio"] = np.log(out["last_bar_ratio"])
    out["day_ret"] = rt.sum(axis=1)
    out["abs_day_ret"] = out["day_ret"].abs()
    out["day_range"] = np.nanmax(path0, axis=1) - np.nanmin(path0, axis=1)
    return out


def calendar(days: pd.DatetimeIndex) -> pd.DataFrame:
    f = P71.calendar_flags(days).astype(bool)
    f["quarter_end"] = f["month_end"] & days.month.isin([3, 6, 9, 12])
    f["fomc_next"] = np.r_[False, f["fomc_day"].to_numpy()[:-1]]
    f["dow"] = days.dayofweek
    return f


def build(tag: str) -> pd.DataFrame:
    d = deck_full(tag)
    days = d.index
    rv = P71.realized_last_bar(tag).reindex(days)
    raw = P72.raw_forecast(tag).reindex(days)
    assert rv.notna().all() and raw.notna().all(), (
        f"{tag}: realized / raw forecast missing on deck days"
    )
    vf = vix_family_1530(days)
    m = P72.build_measures(
        d[["s", "iv_var", "rv_hat", "r_mid", "r_ask", "r_bid"]], rv, raw, vf["vix"]
    )
    x = pd.concat([d, intraday_features(days), calendar(days)], axis=1)
    x["rv"] = rv
    x["raw"] = raw
    x["qlike_63"] = m["qlike_63"]
    x["isur_63"] = m["isur_63"]
    x["vix"], x["vvix"], x["vix3m"] = vf["vix"], vf["vvix"], vf["vix3m"]
    x["log_vix"] = np.log(x["vix"])
    x["vix_slope"] = np.log(x["vix3m"] / x["vix"])
    x["s_rel"] = x["s"] / x["iv_var"]
    x["last_bar_over_iv"] = np.log(x["rv_last_bar"] / x["iv_var"])
    # gate: the 15:30-16:00 realized variance from the forecast table is the panel's own
    y16 = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    y16["endbartime"] = pd.to_datetime(y16["endbartime"])
    at = y16["endbartime"].dt.strftime("%H:%M") == "16:00"
    core16 = (
        y16.loc[at]
        .set_index(y16.loc[at, "endbartime"].dt.normalize())["sumret2"]
        .reindex(days)
    )
    dev = float(np.nanmax(np.abs(core16.to_numpy(float) / rv.to_numpy(float) - 1.0)))
    assert dev < 1e-6, f"{tag}: realized last bar differs from core_stats by {dev:.1e}"
    return x


# ------------------------------------------------------------------- part A --
def tail_sets(x: pd.DataFrame) -> dict[str, pd.Index]:
    r = x["r_mid"]
    return {
        "top20_R": top_days(r, TOP_DAYS),
        "R_gt_1": r.index[r > DOUBLE],
        "top2.5pct_absR": r.index[r.abs() >= r.abs().quantile(TAIL_Q)],
    }


def auc(score: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(score)
    s, yy = score[ok], y[ok].astype(bool)
    n1, n0 = int(yy.sum()), int((~yy).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(score)
    s, yy = score[ok], y[ok]
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, s.size]), s.size, AUC_BOOT_BLOCK, AUC_BOOT_B
    )
    vals = np.array([auc(s[i], yy[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


CANDIDATES = [
    "s",
    "s_rel",
    "qlike_63",
    "isur_63",
    "vix",
    "vvix",
    "vix3m",
    "vix_slope",
    "log_last_bar_ratio",
    "last_bar_over_iv",
    "day_ret",
    "abs_day_ret",
    "day_range",
    "entry_over_spot",
    "month_end",
    "quarter_end",
    "third_friday",
    "fomc_day",
    "fomc_next",
    "dow",
]


def part_a(
    x: pd.DataFrame, tag: str, sets: dict[str, pd.Index]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pos0 = np.where(x["s"] > 0, 1.0, -1.0)
    _, cr0 = book(
        pos0,
        x["r_mid"].to_numpy(float),
        x["r_ask"].to_numpy(float),
        x["r_bid"].to_numpy(float),
    )
    cr0 = pd.Series(cr0, index=x.index)
    rows, days_rows = [], []
    for name, idx in sets.items():
        bought = x.loc[idx, "s"] > 0
        rows.append(
            {
                "tag": tag,
                "definition": name,
                "n_tail_days": len(idx),
                "sign_bought": int(bought.sum()),
                "share_of_sign_crossed_pnl": float(cr0.loc[idx].sum() / cr0.sum()),
                "mean_R_tail": float(x.loc[idx, "r_mid"].mean()),
                "month_end": int(x.loc[idx, "month_end"].sum()),
                "third_friday": int(x.loc[idx, "third_friday"].sum()),
                "fomc_day": int(x.loc[idx, "fomc_day"].sum()),
                "median_vix": float(x.loc[idx, "vix"].median()),
            }
        )
        if name == "top20_R":
            for d in idx:
                days_rows.append(
                    {
                        "date": d.date(),
                        "R": float(x.at[d, "r_mid"]),
                        "sign_bought": bool(x.at[d, "s"] > 0),
                        "s_rel": float(x.at[d, "s_rel"]),
                        "vix": float(x.at[d, "vix"]),
                        "abs_day_ret_pct": float(100 * x.at[d, "abs_day_ret"]),
                        "last_bar_ratio": float(x.at[d, "last_bar_ratio"]),
                        "month_end": bool(x.at[d, "month_end"]),
                        "third_friday": bool(x.at[d, "third_friday"]),
                        "fomc_day": bool(x.at[d, "fomc_day"]),
                        "dow": int(x.at[d, "dow"]),
                    }
                )
    aucs = []
    for name, idx in sets.items():
        y = x.index.isin(idx).astype(float)
        for f in CANDIDATES:
            sc = x[f].to_numpy(float)
            a = auc(sc, y)
            lo, hi = auc_ci(sc, y)
            ok = np.isfinite(sc)
            top = sc >= np.nanquantile(sc, 0.9)
            aucs.append(
                {
                    "tag": tag,
                    "definition": name,
                    "feature": f,
                    "n": int(ok.sum()),
                    "AUC": a,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "separates": bool(
                        (a >= AUC_ADMIT and lo > 0.5)
                        or (a <= 1 - AUC_ADMIT and hi < 0.5)
                    ),
                    "hit_in_top_decile": float(y[top & ok].mean())
                    if (top & ok).any()
                    else np.nan,
                    "base_rate": float(y[ok].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(days_rows), pd.DataFrame(aucs)


# ------------------------------------------------------------------- part B --
def causal_probs(x: pd.DataFrame, y: np.ndarray, feats: list[str]) -> np.ndarray:
    """Expanding L2 logistic (C = 1, standardised in the fit window), refit every REFIT sessions after WARMUP."""
    X = x[feats].to_numpy(float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = len(x)
    p = np.full(n, np.nan)
    model = None
    mu = sd = None
    for t in range(WARMUP, n):
        if (t - WARMUP) % REFIT == 0:
            fit = ok.copy()
            fit[t:] = False
            if fit.sum() < WARMUP // 2 or y[fit].sum() < 2 or (1 - y[fit]).sum() < 2:
                model = None
                continue
            mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0, ddof=1)
            sd = np.where(sd > 0, sd, 1.0)
            model = LogisticRegression(C=1.0, max_iter=1000).fit(
                (X[fit] - mu) / sd, y[fit]
            )
        if model is not None and ok[t]:
            p[t] = model.predict_proba(((X[t] - mu) / sd)[None, :])[0, 1]
    return p


def trailing_quantile(v: pd.Series, q: float) -> pd.Series:
    return v.rolling(TRAIL, min_periods=TRAIL // 2).quantile(q).shift(1)


def break_even(x: pd.DataFrame, tail: np.ndarray) -> pd.Series:
    """p* such that p E[R_ask | tail] + (1 - p) E[R_ask | not tail] = 0, from trailing means ending the previous session."""
    r = x["r_ask"]
    e_tail = (
        pd.Series(np.where(tail == 1, r, np.nan), index=x.index)
        .expanding(min_periods=5)
        .mean()
        .shift(1)
    )
    e_not = (
        pd.Series(np.where(tail == 0, r, np.nan), index=x.index)
        .expanding(min_periods=WARMUP // 2)
        .mean()
        .shift(1)
    )
    return (-e_not / (e_tail - e_not)).where((e_tail > 0) & (e_not < 0))


def rule_positions(
    x: pd.DataFrame, p: pd.Series, pstar: pd.Series
) -> dict[str, np.ndarray]:
    s = x["s"].to_numpy(float)
    sign = s > 0
    pv = p.to_numpy(float)
    top_q = trailing_quantile(p, 0.8).to_numpy(float)
    med = trailing_quantile(p, 0.5).to_numpy(float)
    be = pv > pstar.to_numpy(float)
    return {
        "sign(s)": sign,
        "tail: p-hat > break-even p*": be,
        "tail: p-hat > trailing top quintile": pv > top_q,
        "sign(s) AND p-hat > trailing median": sign & (pv > med),
        "sign(s) OR break-even": sign | be,
    }


def exposure_match(buy: np.ndarray, base_buy: np.ndarray) -> np.ndarray:
    """Long size scaled so the mean long exposure over the scored days matches sign(s); short leg -1."""
    if buy.sum() == 0:
        return np.where(buy, 0.0, -1.0)
    size = base_buy.mean() / buy.mean()
    return np.where(buy, size, -1.0)


def evaluate_rules(
    x: pd.DataFrame,
    p: pd.Series,
    pstar: pd.Series,
    scored: np.ndarray,
    tag: str,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = x.index[scored]
    d = x.loc[idx]
    rules = {k: v[scored] for k, v in rule_positions(x, p, pstar).items()}
    base_buy = rules["sign(s)"]
    pos0 = np.where(base_buy, 1.0, -1.0)
    _, cr0 = book(
        pos0,
        d["r_mid"].to_numpy(float),
        d["r_ask"].to_numpy(float),
        d["r_bid"].to_numpy(float),
    )
    base_top = top_days(pd.Series(cr0, index=idx))
    years = idx.year.to_numpy()
    rng = np.random.default_rng([SEED, 1])
    rows, daily = [], pd.DataFrame({"sign(s)": cr0}, index=idx)
    for name, buy in rules.items():
        pos = exposure_match(buy, base_buy) if name != "sign(s)" else pos0
        st = P72.stats(pos, d, idx, base_top)
        _, cr = book(
            pos,
            d["r_mid"].to_numpy(float),
            d["r_ask"].to_numpy(float),
            d["r_bid"].to_numpy(float),
        )
        daily[name] = cr
        row = {
            "tag": tag,
            "target": label,
            "rule": name,
            **st,
            "dSharpe_crossed": st["Sharpe_crossed"] - sharpe(cr0),
        }
        if name != "sign(s)":
            lo, hi = block_ci(cr - cr0)
            h1, h2 = idx < SPLIT, idx >= SPLIT
            row.update(
                {
                    "d_ci_lo": lo,
                    "d_ci_hi": hi,
                    "dSharpe_half1": sharpe(cr[h1]) - sharpe(cr0[h1])
                    if h1.sum() > 20
                    else np.nan,
                    "dSharpe_half2": sharpe(cr[h2]) - sharpe(cr0[h2])
                    if h2.sum() > 20
                    else np.nan,
                }
            )
            # placebo: p-hat permuted within calendar year, the rule recomputed on the permuted series
            actual = st["Sharpe_crossed"]
            hits = 0
            for _ in range(N_PLACEBO):
                pp = p.copy()
                vals = pp.to_numpy(float).copy()
                for yv in np.unique(years):
                    sel = np.where(scored & (x.index.year == yv))[0]
                    vals[sel] = rng.permutation(vals[sel])
                pp = pd.Series(vals, index=x.index)
                buy_p = rule_positions(x, pp, pstar)[name][scored]
                pos_p = exposure_match(buy_p, base_buy)
                _, cr_p = book(
                    pos_p,
                    d["r_mid"].to_numpy(float),
                    d["r_ask"].to_numpy(float),
                    d["r_bid"].to_numpy(float),
                )
                hits += sharpe(cr_p) >= actual
            row["placebo_p"] = hits / N_PLACEBO
            row["supported"] = bool(
                lo > 0
                and row["placebo_p"] < 0.05
                and row["dSharpe_half1"] > 0
                and row["dSharpe_half2"] > 0
            )
        rows.append(row)
    return pd.DataFrame(rows), daily


# ------------------------------------------------------------------- part C --
def wing_legs(x: pd.DataFrame) -> pd.DataFrame:
    """Nearest listed OTM strangle legs at the deck's 15:30 stamps, from one filtered chain read."""
    stamps = pd.DatetimeIndex(x["timestamp_c"]).unique()
    q = pd.read_parquet(
        CHAIN,
        columns=["expiration", "timestamp", "strike", "cp", "bid", "ask"],
        filters=[("timestamp", "in", list(stamps))],
    )
    q["timestamp"] = pd.to_datetime(q["timestamp"], utc=True)
    q["strike"] = q["strike"].astype(float)
    q["cp"] = q["cp"].astype(str).str.upper().str[0]
    q = q[~((q["bid"] == 0) & (q["ask"] == 0))]
    q["mid"] = 0.5 * (q["bid"].astype(float) + q["ask"].astype(float))
    out = pd.DataFrame(index=x.index)
    by_stamp = {ts: g for ts, g in q.groupby("timestamp")}
    for w in WINGS:
        kc: list[float] = []
        kp: list[float] = []
        mc: list[float] = []
        mp: list[float] = []
        ac: list[float] = []
        ap: list[float] = []
        for d, ts, spot in zip(
            x.index, pd.DatetimeIndex(x["timestamp_c"]), x["S"].to_numpy(float)
        ):
            g = by_stamp.get(ts)
            vals = [np.nan] * 6
            if g is not None:
                c = (
                    g[(g["cp"] == "C") & (g["strike"] >= spot * (1 + w))]
                    .sort_values("strike")
                    .head(1)
                )
                p_ = (
                    g[(g["cp"] == "P") & (g["strike"] <= spot * (1 - w))]
                    .sort_values("strike")
                    .tail(1)
                )
                if len(c) and len(p_):
                    vals = [
                        float(c["strike"].iloc[0]),
                        float(p_["strike"].iloc[0]),
                        float(c["mid"].iloc[0]),
                        float(p_["mid"].iloc[0]),
                        float(c["ask"].iloc[0]),
                        float(p_["ask"].iloc[0]),
                    ]
            for lst, v in zip((kc, kp, mc, mp, ac, ap), vals):
                lst.append(v)
        tag = _wtag(w)
        out[f"{tag}_K_c"], out[f"{tag}_K_p"] = kc, kp
        out[f"{tag}_entry"] = np.array(mc) + np.array(mp)
        out[f"{tag}_ask"] = np.array(ac) + np.array(ap)
        sc = x["S_close"].to_numpy(float)
        out[f"{tag}_pay"] = np.maximum(sc - np.array(kc), 0.0) + np.maximum(
            np.array(kp) - sc, 0.0
        )
    return out


def part_c(
    x: pd.DataFrame,
    wings: pd.DataFrame,
    buys: dict[str, np.ndarray],
    scored: np.ndarray,
    tag: str,
) -> pd.DataFrame:
    rows = []
    idx = x.index[scored]
    spot = x.loc[idx, "S"].to_numpy(float)
    structures = {
        "straddle": (
            x.loc[idx, "entry"].to_numpy(float),
            x.loc[idx, "ask"].to_numpy(float),
            x.loc[idx, "exit"].to_numpy(float),
        ),
    }
    for w in WINGS:
        t = _wtag(w)
        structures[_wname(w)] = (
            wings.loc[idx, f"{t}_entry"].to_numpy(float),
            wings.loc[idx, f"{t}_ask"].to_numpy(float),
            wings.loc[idx, f"{t}_pay"].to_numpy(float),
        )
    for rule, buy in buys.items():
        b = buy[scored]
        for name, (entry, ask, pay) in structures.items():
            ok = b & np.isfinite(entry) & np.isfinite(ask) & (entry > 0) & (ask > 0)
            r_mid = pay[ok] / entry[ok] - 1.0
            r_ask = pay[ok] / ask[ok] - 1.0
            pnl_spot = (pay[ok] - ask[ok]) / spot[ok] * 1e4  # bp of spot, at the ask
            rows.append(
                {
                    "tag": tag,
                    "rule": rule,
                    "structure": name,
                    "buy_days": int(ok.sum()),
                    "median_premium_bp_spot": float(
                        np.median(ask[ok] / spot[ok] * 1e4)
                    ),
                    "hit_ask": float((r_ask > 0).mean()),
                    "mean_R_mid": float(r_mid.mean()),
                    "mean_R_ask": float(r_ask.mean()),
                    "Sharpe_R_ask": sharpe(r_ask),
                    "mean_pnl_bp_spot_ask": float(pnl_spot.mean()),
                    "Sharpe_pnl_bp_spot": sharpe(pnl_spot),
                    "top20_share_pnl_spot": float(
                        np.sort(pnl_spot)[-TOP_DAYS:].sum() / pnl_spot.sum()
                    )
                    if pnl_spot.sum() > 0 and ok.sum() > TOP_DAYS
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- part D --
def part_d(
    x: pd.DataFrame, scored: np.ndarray, tag: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = x.index[scored]
    d = x.loc[idx]
    s_rec = d["s"].to_numpy(float)
    gap = (d["raw"] - d["rv_hat"]).to_numpy(float)
    s_raw = s_rec + gap
    pos0 = np.where(s_rec > 0, 1.0, -1.0)
    pos1 = np.where(s_raw > 0, 1.0, -1.0)
    r_mid, r_ask, r_bid = (d[c].to_numpy(float) for c in ("r_mid", "r_ask", "r_bid"))
    _, cr0 = book(pos0, r_mid, r_ask, r_bid)
    _, cr1 = book(pos1, r_mid, r_ask, r_bid)
    base_top = top_days(pd.Series(cr0, index=idx))
    rows = []
    for name, pos, cr in (
        ("recalibrated sign(s)", pos0, cr0),
        ("raw sign(yhat^2 baseline - iv_var)", pos1, cr1),
    ):
        st = P72.stats(pos, d, idx, base_top)
        rows.append({"tag": tag, "rule": name, **st})
    lo, hi = block_ci(cr1 - cr0)
    years = idx.year.to_numpy()
    rng = np.random.default_rng([SEED, 4])
    actual = sharpe(cr1)
    hits = 0
    for _ in range(N_PLACEBO):
        g = gap.copy()
        for yv in np.unique(years):
            sel = years == yv
            g[sel] = rng.permutation(g[sel])
        pos_p = np.where(s_rec + g > 0, 1.0, -1.0)
        _, cr_p = book(pos_p, r_mid, r_ask, r_bid)
        hits += sharpe(cr_p) >= actual
    h1, h2 = idx < SPLIT, idx >= SPLIT
    rows[1].update(
        {
            "dSharpe_crossed": sharpe(cr1) - sharpe(cr0),
            "d_ci_lo": lo,
            "d_ci_hi": hi,
            "placebo_p": hits / N_PLACEBO,
            "dSharpe_half1": sharpe(cr1[h1]) - sharpe(cr0[h1]),
            "dSharpe_half2": sharpe(cr1[h2]) - sharpe(cr0[h2]),
            "median_raw_over_recal": float(np.median(d["raw"] / d["rv_hat"])),
        }
    )
    rows[1]["supported"] = bool(
        lo > 0
        and rows[1]["placebo_p"] < 0.05
        and rows[1]["dSharpe_half1"] > 0
        and rows[1]["dSharpe_half2"] > 0
    )
    # which sign(s) buys the raw rule drops, by study 71's s-quintile (quintiles over the buys)
    buys = s_rec > 0
    qn = pd.qcut(pd.Series(s_rec[buys]), 5, labels=False) + 1
    dropped = ~(s_raw[buys] > 0)
    dq = pd.DataFrame(
        {
            "s_quintile": qn.to_numpy(),
            "dropped": dropped,
            "r_ask": r_ask[buys],
        }
    )
    by_q = dq.groupby("s_quintile").agg(
        buys=("dropped", "size"),
        dropped=("dropped", "sum"),
        mean_r_ask_dropped=(
            "r_ask",
            lambda v: (
                float(v[dq.loc[v.index, "dropped"]].mean())
                if dq.loc[v.index, "dropped"].any()
                else np.nan
            ),
        ),
        mean_r_ask_kept=(
            "r_ask",
            lambda v: (
                float(v[~dq.loc[v.index, "dropped"]].mean())
                if (~dq.loc[v.index, "dropped"]).any()
                else np.nan
            ),
        ),
    )
    by_q["tag"] = tag
    return pd.DataFrame(rows), by_q.reset_index()


# --------------------------------------------------------------------- main --
def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    frames = {tag: build(tag) for tag in TAGS}
    x0 = frames[TAGS[0]]
    print(
        f"GATE  {len(x0)} deck days; R identical across decks: "
        f"{np.allclose(frames['sub_live_ridge']['r_mid'], frames['blk2']['r_mid'])}; "
        f"VIX print at 15:30 on {int(x0['vix'].notna().sum())} days (feed ends {x0.index[x0['vix'].notna()].max().date()}); "
        f"intraday bars complete on {int(x0['last_bar_ratio'].notna().sum())} days"
    )
    sets = tail_sets(x0)
    print(
        "CRITERIA (stated before results): a feature separates tail days if AUC >= 0.60 (or <= 0.40) with the day-block "
        "interval clear of 0.5; a rule is supported if exposure-matched crossed Sharpe beats sign(s) with the bootstrap "
        "interval on the daily difference excluding zero AND within-year placebo p < 0.05 AND both halves positive."
    )

    # ---- A
    cap, top, aucs = [], None, []
    for tag, x in frames.items():
        c, t, a = part_a(x, tag, sets)
        cap.append(c)
        aucs.append(a)
        if top is None:
            top = t
    cap_df, auc_df = pd.concat(cap), pd.concat(aucs)
    assert top is not None
    top["sign_bought_blk2"] = [
        bool(frames["blk2"].at[pd.Timestamp(d), "s"] > 0) for d in top["date"]
    ]
    cap_df.to_csv(OUT / "a_capture.csv", index=False)
    top.to_csv(OUT / "a_top20_days.csv", index=False)
    auc_df.to_csv(OUT / "a_auc.csv", index=False)
    print("\nA  tail days and how many sign(s) bought")
    print(cap_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nA  the 20 largest long-straddle returns (sign_bought = sub_live_ridge)")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    a1 = auc_df[
        (auc_df["tag"] == "sub_live_ridge") & (auc_df["definition"] == "R_gt_1")
    ].sort_values("AUC", ascending=False)
    print(
        "\nA  ex-ante separability of R > 1 days, one feature at a time (sub_live_ridge; base rate in the last column)"
    )
    print(
        a1.drop(columns=["tag", "definition"]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )
    adm = a1[a1["separates"]]["feature"].tolist()
    print("features that separate:", adm if adm else "none")
    if len(adm) > 1:
        corr = x0[adm].corr(method="spearman")
        corr.to_csv(OUT / "a_corr.csv")
        print(corr.round(2).to_string())

    # ---- B
    all_rules: list[pd.DataFrame] = []
    buys_for_c: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for tag, x in frames.items():
        y_double = (x["r_mid"] > DOUBLE).to_numpy(float)
        thr = x["r_mid"].expanding(min_periods=WARMUP).quantile(TAIL_Q).shift(1)
        y_tail = np.where(thr.notna(), (x["r_mid"] > thr).astype(float), np.nan)
        for label, y in (("R > 1", y_double), ("R > trailing 97.5% quantile", y_tail)):
            p = pd.Series(causal_probs(x, y, CLASSIFIER_FEATURES), index=x.index)
            pstar = break_even(x, y)
            scored = p.notna().to_numpy() & pstar.notna().to_numpy()
            rules, daily = evaluate_rules(x, p, pstar, scored, tag, label)
            all_rules.append(rules)
            daily.to_csv(
                OUT / f"b_daily_{tag}_{'double' if label == 'R > 1' else 'q975'}.csv"
            )
            if label == "R > 1":
                buys_for_c[tag] = (
                    {k: v for k, v in rule_positions(x, p, pstar).items()},
                    scored,
                )
                print(
                    f"\nB  {tag}: scored {int(scored.sum())} days {x.index[scored].min().date()} .. {x.index[scored].max().date()}; "
                    f"tail base rate {np.nanmean(y[scored]):.3f}; median p-hat {np.nanmedian(p[scored]):.3f}; "
                    f"median break-even p* {np.nanmedian(pstar[scored]):.3f}"
                )
    rules_df = pd.concat(all_rules)
    rules_df.to_csv(OUT / "b_rules.csv", index=False)
    cols = [
        "tag",
        "target",
        "rule",
        "days",
        "buys",
        "pct_buy",
        "hit_buys",
        "Sharpe_mid",
        "Sharpe_crossed",
        "dSharpe_crossed",
        "d_ci_lo",
        "d_ci_hi",
        "placebo_p",
        "dSharpe_half1",
        "dSharpe_half2",
        "top20_share",
        "top20_kept_from_sign",
        "max_dd_crossed",
        "supported",
    ]
    print("\nB  classifier rules vs sign(s), exposure-matched, short leg -1")
    print(
        rules_df[[c for c in cols if c in rules_df.columns]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )

    # ---- C
    wings = wing_legs(x0)
    wings.to_csv(OUT / "c_wing_legs.csv")
    cs = []
    for tag, (buys, scored) in buys_for_c.items():
        keep = {
            k: v
            for k, v in buys.items()
            if k
            in (
                "sign(s)",
                "tail: p-hat > break-even p*",
                "tail: p-hat > trailing top quintile",
            )
        }
        cs.append(part_c(frames[tag], wings, keep, scored, tag))
    c_df = pd.concat(cs)
    c_df.to_csv(OUT / "c_wings.csv", index=False)
    print(
        "\nC  straddle vs OTM strangles on each rule's buy days (long leg only; premium and P&L at the ask)"
    )
    print(c_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- D
    ds, dq = [], []
    for tag, x in frames.items():
        scored = np.arange(len(x)) >= WARMUP
        a, b = part_d(x, scored, tag)
        ds.append(a)
        dq.append(b)
    d_df, dq_df = pd.concat(ds), pd.concat(dq)
    d_df.to_csv(OUT / "d_raw_rule.csv", index=False)
    dq_df.to_csv(OUT / "d_dropped_by_quintile.csv", index=False)
    print(
        "\nD  the raw-forecast sign rule vs the recalibrated sign(s), same days from the 252-session warm-up"
    )
    print(d_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "\nD  which sign(s) buys the raw rule drops, by s-quintile (quintiles over the buys)"
    )
    print(dq_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- verdict
    c20 = cap_df[
        (cap_df["tag"] == "sub_live_ridge") & (cap_df["definition"] == "top20_R")
    ].iloc[0]
    b_best = (
        rules_df[
            (rules_df["tag"] == "sub_live_ridge")
            & (rules_df["target"] == "R > 1")
            & (rules_df["rule"] != "sign(s)")
        ]
        .sort_values("dSharpe_crossed", ascending=False)
        .iloc[0]
    )
    c_sign = c_df[
        (c_df["tag"] == "sub_live_ridge") & (c_df["rule"] == "sign(s)")
    ].set_index("structure")
    d1 = d_df[
        (d_df["tag"] == "sub_live_ridge") & (d_df["rule"].str.startswith("raw"))
    ].iloc[0]
    print(
        f"\nVERDICT  (A) tail separable ex ante: sign(s) bought {int(c20['sign_bought'])} of the 20 largest days "
        f"({100 * c20['share_of_sign_crossed_pnl']:.0f}% of its P&L); features that separate R > 1 days: {adm if adm else 'none'}; "
        f"(B) classifier rule SUPPORTED: {bool(b_best['supported'])} (best {b_best['rule']}: dSharpe crossed {b_best['dSharpe_crossed']:+.2f}, "
        f"interval [{b_best['d_ci_lo']:+.3f}, {b_best['d_ci_hi']:+.3f}], placebo p {b_best['placebo_p']:.3f}, halves {b_best['dSharpe_half1']:+.2f} / {b_best['dSharpe_half2']:+.2f}); "
        f"(C) wings on sign(s) buys: Sharpe of P&L per unit spot straddle {c_sign.loc['straddle', 'Sharpe_pnl_bp_spot']:.2f} vs "
        f"0.5% {c_sign.loc[_wname(0.005), 'Sharpe_pnl_bp_spot']:.2f} / 1% {c_sign.loc[_wname(0.01), 'Sharpe_pnl_bp_spot']:.2f} / 2% {c_sign.loc[_wname(0.02), 'Sharpe_pnl_bp_spot']:.2f} "
        f"(hit at the ask {c_sign.loc['straddle', 'hit_ask']:.2f} / {c_sign.loc[_wname(0.005), 'hit_ask']:.2f} / {c_sign.loc[_wname(0.01), 'hit_ask']:.2f} / {c_sign.loc[_wname(0.02), 'hit_ask']:.2f}); "
        f"(D) raw-forecast rule SUPPORTED: {bool(d1['supported'])} (dSharpe crossed {d1['dSharpe_crossed']:+.2f}, interval "
        f"[{d1['d_ci_lo']:+.3f}, {d1['d_ci_hi']:+.3f}], placebo p {d1['placebo_p']:.3f}, halves {d1['dSharpe_half1']:+.2f} / {d1['dSharpe_half2']:+.2f})"
    )


if __name__ == "__main__":
    main()
