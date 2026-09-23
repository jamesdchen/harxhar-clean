"""Study 72 -- vol-surprise clustering and the sign(s) long leg.

The lead from study 71: the only monotone grading of the long leg was the
forecast's own trailing 63-session realized QLIKE -- the quintile where the
model had recently been WORST had hit 0.49, mean crossed +0.36 and 71% of the
long P&L (Spearman +0.87, both halves, both decks).  Two readings compete:
(1) a regime proxy -- the forecast is worst when realized variance is high and
jumpy, and that is when long straddles pay; (2) a genuine forecast lag -- when
realized has been running above the forecast the forecast is biased low,
sign(s) under-buys, and correcting the bias fixes the buys.  This study
separates them and tests the pre-registered rules that follow.

A. WHAT IS THE SURPRISE MEASURING?  Trailing measures, each ending the PREVIOUS
   session (rolling mean, shift 1): (a) unsigned, the trailing 21- and 63-session
   mean QLIKE of rv_hat against the realized 15:30-16:00 variance (study 71's
   object); (b) signed forecast surprise, the trailing mean of log(rv / rv_hat);
   (c) signed implied surprise, the trailing mean of log(rv / iv_var) -- the
   trade's own recent realized-minus-implied; (d) the trailing long-leg hit rate
   and mean crossed return over the previous 63 sessions' buys; (e) the VIX level
   at the 15:30 stamp and its 21-session log change (Cboe prints, feed ends
   2024-02-12).  Correlation matrix on the buy days; each measure's grading of
   the long leg by quintile (hit, mean crossed, share of long P&L, Spearman of
   the quintile index against hit and mean), both decks, whole sample and the two
   halves.  Regime check: the long leg graded by (a) WITHIN VIX-level terciles
   and by the VIX level within (a)-terciles (3 x 3 tables).

B. MECHANISM.  A causal multiplicative bias correction of the forecast,
   rv_hat_adj = rv_hat * exp(trailing 21- or 63-session mean log(rv / rv_hat)),
   then sign(rv_hat_adj - iv_var).  Reported against sign(s): pct_buy, hit,
   Sharpe mid / crossed, tail capture.  The same correction on the smear-free
   forecast (yhat^2 x baseline from the forecast table, the prediction before
   the production recalibration) and that raw forecast's own sign rule.

C. PRE-REGISTERED RULES (short leg -1 unchanged; long-leg exposure rescaled by
   one constant to the same mean as sign(s) over the scored days -- the constant
   never chooses days, the causal rule does; warm-up 252; windows 21 and 63):
     (i)   gate: long only when (a) is in its top trailing-252 tercile;
     (ii)  gate: long only when the signed forecast surprise (b) > 0;
     (iii) size = 2 x rank of (a) among the trailing 252 sessions (mean 1);
     (iv)  size 2 / 1 / 0.5 by trailing-252 tercile of (a), divided by 7/6;
     (v)   the bias-corrected sign rule of part B, at size 1 and exposure-matched.
   CRITERION: SUPPORTED if the crossed Sharpe beats sign(s) with the circular-
   block-bootstrap 95% interval on the daily crossed P&L difference excluding
   zero, AND beats a placebo (the surprise series permuted within calendar year,
   200 draws, sizes re-derived) at p < 0.05, AND the direction holds in both
   halves (2020-21 scored part, 2022-24).  Beside: mean, pct_buy, hit, tail
   capture (share of P&L from the top-20 days; how many of sign(s)'s top-20 are
   kept), max drawdown in premium units, buys.

D. Robustness: by year; part A's grading with 2020-03 .. 2020-06 excluded (the
   scored rules start after the warm-up, in 2021, so the exclusion touches part
   A only); the long leg alone (no short leg).

Gates: study 71's (signal == rv_hat - iv_var, R == exit/entry - 1, bid <= entry
<= ask, both decks on the same 866 days); realized last-bar variance and the
raw forecast matched on every deck day; VIX at 15:30 on 812 of 866 days.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "72"
VIX = ROOT / "data" / "vix_and_voldemand.parquet"
TAGS = ("sub_live_ridge", "blk2")
WINDOWS = (21, 63)
WARMUP = 252
TRAIL = 252  # sessions in the rank / tercile window
HIT_WINDOW = 63
MIN_BUYS_HIT = 10
N_Q = 5
N_T = 3
TOP_DAYS = 20
N_PLACEBO = 200
SEED = 72
SPLIT = pd.Timestamp("2022-01-01")
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"))


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P71 = _load(HERE / "71_conviction_sizing.py", "p71_conviction")
sharpe, book, max_drawdown, top_days, block_ci = (
    P71.sharpe,
    P71.book,
    P71.max_drawdown,
    P71.top_days,
    P71.block_ci,
)


# ------------------------------------------------------------------ inputs --
def raw_forecast(tag: str) -> pd.Series:
    """The smear-free forecast of the 15:30-16:00 bar: yhat^2 x baseline at the 16:00 row."""
    y = pd.read_parquet(asl.yhat_paths(ROOT)[tag], columns=["t", "yhat", "baseline"])
    et = pd.to_datetime(y["t"], utc=True).dt.tz_convert("America/New_York")
    last = et.dt.strftime("%H:%M") == "16:00"
    s = pd.Series(
        y.loc[last, "yhat"].to_numpy(float) ** 2
        * y.loc[last, "baseline"].to_numpy(float),
        index=pd.DatetimeIndex(et[last].dt.tz_localize(None)).normalize(),
    )
    return s[~s.index.duplicated()]


def vix_1530(days: pd.DatetimeIndex) -> pd.Series:
    v = pd.read_parquet(VIX, columns=["endbartime", "vix"])
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    at = v["endbartime"].dt.strftime("%H:%M") == "15:30"
    s = pd.Series(
        v.loc[at, "vix"].to_numpy(float), index=v.loc[at, "endbartime"].dt.normalize()
    )
    return s[~s.index.duplicated()].reindex(days)


def trailing(x: pd.Series, w: int) -> pd.Series:
    """Mean over the previous w sessions, ending the previous session."""
    return x.rolling(w, min_periods=w).mean().shift(1)


def build_measures(
    d: pd.DataFrame, rv: pd.Series, raw: pd.Series, vix: pd.Series
) -> pd.DataFrame:
    m = pd.DataFrame(index=d.index)
    q = rv / d["rv_hat"]
    qlike = q - np.log(q) - 1.0
    lsur = np.log(rv / d["rv_hat"])
    lsur_raw = np.log(rv / raw)
    isur = np.log(rv / d["iv_var"])
    for w in WINDOWS:
        m[f"qlike_{w}"] = trailing(qlike, w)
        m[f"fsur_{w}"] = trailing(lsur, w)
        m[f"fsur_raw_{w}"] = trailing(lsur_raw, w)
        m[f"isur_{w}"] = trailing(isur, w)
    buy = d["s"] > 0
    hit = pd.Series(
        np.where(buy, (d["r_mid"] > 0).astype(float), np.nan), index=d.index
    )
    r_long = pd.Series(np.where(buy, d["r_ask"], np.nan), index=d.index)
    m[f"hit_{HIT_WINDOW}"] = (
        hit.rolling(HIT_WINDOW, min_periods=MIN_BUYS_HIT).mean().shift(1)
    )
    m[f"meanR_{HIT_WINDOW}"] = (
        r_long.rolling(HIT_WINDOW, min_periods=MIN_BUYS_HIT).mean().shift(1)
    )
    m["vix"] = vix
    m["vix_chg_21"] = np.log(vix / vix.shift(21))
    return m


# ------------------------------------------------------------------- part A --
def grading(
    d: pd.DataFrame, m: pd.DataFrame, tag: str, sample: str, mask: np.ndarray
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    buys = (d["s"] > 0).to_numpy() & mask
    r_ask = d["r_ask"].to_numpy(float)
    hit = (d["r_mid"].to_numpy(float) > 0).astype(float)
    tot = float(r_ask[buys].sum())
    rows = []
    for col in m.columns:
        v = m[col].to_numpy(float)
        ok = buys & np.isfinite(v)
        if ok.sum() < 5 * N_Q:
            continue
        q = pd.qcut(v[ok], N_Q, labels=False, duplicates="drop")
        idx = np.flatnonzero(ok)
        hits, means = [], []
        for k in range(N_Q):
            sel = idx[q == k]
            hits.append(float(hit[sel].mean()))
            means.append(float(r_ask[sel].mean()))
            rows.append(
                {
                    "tag": tag,
                    "sample": sample,
                    "measure": col,
                    "bin": f"Q{k + 1}",
                    "n": int(sel.size),
                    "hit": hits[-1],
                    "mean_crossed": means[-1],
                    "share_long_pnl": float(r_ask[sel].sum() / tot)
                    if tot != 0
                    else np.nan,
                    "rho_hit": np.nan,
                    "rho_mean": np.nan,
                }
            )
        ks = np.arange(1, N_Q + 1)
        rows[-N_Q]["rho_hit"] = float(spearmanr(ks, hits).statistic)
        rows[-N_Q]["rho_mean"] = float(spearmanr(ks, means).statistic)
    return pd.DataFrame(rows)


def regime_tables(d: pd.DataFrame, m: pd.DataFrame, tag: str) -> pd.DataFrame:
    """The long leg by tercile of qlike_63 within VIX terciles and vice versa."""
    buys = (d["s"] > 0).to_numpy()
    ok = (
        buys
        & np.isfinite(m["qlike_63"].to_numpy(float))
        & np.isfinite(m["vix"].to_numpy(float))
    )
    sub = pd.DataFrame(
        {
            "q": pd.qcut(
                m.loc[ok, "qlike_63"], N_T, labels=[f"surprise T{k}" for k in (1, 2, 3)]
            ),
            "v": pd.qcut(
                m.loc[ok, "vix"], N_T, labels=[f"VIX T{k}" for k in (1, 2, 3)]
            ),
            "hit": (d.loc[ok, "r_mid"] > 0).astype(float),
            "r": d.loc[ok, "r_ask"],
        }
    )
    g = sub.groupby(["v", "q"], observed=True).agg(
        n=("r", "size"), hit=("hit", "mean"), mean_crossed=("r", "mean")
    )
    g["tag"] = tag
    return g.reset_index()


# ------------------------------------------------------------------- rules --
def gate_top_tercile(meas: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        if t < WARMUP:
            continue
        if s[t] <= 0:
            out[t] = 0.0
            continue
        past = meas[max(0, t - TRAIL) : t]
        past = past[np.isfinite(past)]
        if past.size < N_T or not np.isfinite(meas[t]):
            continue
        out[t] = float(meas[t] >= np.quantile(past, 2.0 / 3.0))
    return out


def gate_positive(meas: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.full(len(s), np.nan)
    ok = np.isfinite(meas)
    out[WARMUP:] = np.where(s[WARMUP:] > 0, (meas[WARMUP:] > 0).astype(float), 0.0)
    out[~ok] = np.nan
    return out


def size_rank(meas: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        if t < WARMUP:
            continue
        if s[t] <= 0:
            out[t] = 0.0
            continue
        past = meas[max(0, t - TRAIL) : t]
        past = past[np.isfinite(past)]
        if past.size < N_T or not np.isfinite(meas[t]):
            continue
        out[t] = 2.0 * np.sum(past <= meas[t]) / past.size
    return out


def size_tercile(meas: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.full(len(s), np.nan)
    levels = np.array([0.5, 1.0, 2.0]) / (3.5 / 3.0)
    for t in range(len(s)):
        if t < WARMUP:
            continue
        if s[t] <= 0:
            out[t] = 0.0
            continue
        past = meas[max(0, t - TRAIL) : t]
        past = past[np.isfinite(past)]
        if past.size < N_T or not np.isfinite(meas[t]):
            continue
        edges = np.quantile(past, [1.0 / 3.0, 2.0 / 3.0])
        out[t] = levels[int(np.searchsorted(edges, meas[t], side="right"))]
    return out


def bias_corrected_positions(
    fsur: np.ndarray, rv_hat: np.ndarray, iv_var: np.ndarray
) -> np.ndarray:
    """sign(rv_hat * exp(fsur) - iv_var): +1 long, -1 short; NaN before the warm-up or without a surprise."""
    adj = rv_hat * np.exp(fsur)
    pos = np.where(adj > iv_var, 1.0, -1.0)
    pos[:WARMUP] = np.nan
    pos[~np.isfinite(fsur)] = np.nan
    return pos


# ---------------------------------------------------------------- evaluate --
def stats(
    pos: np.ndarray, d: pd.DataFrame, idx: pd.DatetimeIndex, base_top: pd.Index | None
) -> dict:
    r_mid, r_ask, r_bid = (
        d.loc[idx, c].to_numpy(float) for c in ("r_mid", "r_ask", "r_bid")
    )
    mid, cr = book(pos, r_mid, r_ask, r_bid)
    buys = pos > 0
    top = top_days(pd.Series(cr, index=idx))
    out = {
        "days": int(len(pos)),
        "buys": int(buys.sum()),
        "pct_buy": float(100 * buys.mean()),
        "mean_size_buys": float(pos[buys].mean()) if buys.any() else np.nan,
        "hit_buys": float((r_mid[buys] > 0).mean()) if buys.any() else np.nan,
        "Sharpe_mid": sharpe(mid),
        "Sharpe_crossed": sharpe(cr),
        "mean_crossed": float(cr.mean()),
        "top20_share": float(cr[np.isin(idx, top)].sum() / cr.sum())
        if cr.sum() != 0
        else np.nan,
        "top20_kept_from_sign": int(len(top.intersection(base_top)))
        if base_top is not None
        else TOP_DAYS,
        "max_dd_crossed": max_drawdown(cr),
        "worst_crossed": float(cr.min()),
    }
    return out


def evaluate_rules(
    d: pd.DataFrame, m: pd.DataFrame, tag: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = d["s"].to_numpy(float)
    rv_hat, iv_var = d["rv_hat"].to_numpy(float), d["iv_var"].to_numpy(float)

    def rule_sizes(meas_by_name: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Long-leg sizes (NaN = not scored) for every rule, from the measure series given."""
        out: dict[str, np.ndarray] = {}
        for w in WINDOWS:
            q, f = meas_by_name[f"qlike_{w}"], meas_by_name[f"fsur_{w}"]
            out[f"(i) gate: qlike_{w} top trailing tercile"] = gate_top_tercile(q, s)
            out[f"(ii) gate: fsur_{w} > 0"] = gate_positive(f, s)
            out[f"(iii) rank of qlike_{w}"] = size_rank(q, s)
            out[f"(iv) tercile 2/1/0.5 of qlike_{w}"] = size_tercile(q, s)
        return out

    def rule_positions(meas_by_name: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for w in WINDOWS:
            out[f"(v) sign(rv_hat e^fsur_{w} - iv_var)"] = bias_corrected_positions(
                meas_by_name[f"fsur_{w}"], rv_hat, iv_var
            )
        return out

    meas = {c: m[c].to_numpy(float) for c in m.columns}
    sizes = rule_sizes(meas)
    posrules = rule_positions(meas)
    scored = np.arange(len(s)) >= WARMUP
    for v in list(sizes.values()) + list(posrules.values()):
        scored &= np.isfinite(v)
    idx = d.index[scored]
    years = idx.year.to_numpy()
    half1, half2 = np.asarray(idx < SPLIT), np.asarray(idx >= SPLIT)
    r_mid, r_ask, r_bid = (
        d.loc[idx, c].to_numpy(float) for c in ("r_mid", "r_ask", "r_bid")
    )
    base_pos = np.where(s[scored] > 0, 1.0, -1.0)
    base_mid, base_cr = book(base_pos, r_mid, r_ask, r_bid)
    base_top = top_days(pd.Series(base_cr, index=idx))
    n_long_base = float(base_pos[base_pos > 0].sum())
    rng = np.random.default_rng([SEED, len(idx)])

    def positions_from_sizes(sz: np.ndarray) -> tuple[np.ndarray, float]:
        """Long sizes on s > 0 days, -1 on the others, one constant so the long exposure equals sign(s)'s."""
        long = np.where(s[scored] > 0, sz[scored], 0.0)
        c = n_long_base / long.sum() if long.sum() > 0 else np.nan
        return np.where(s[scored] > 0, long * c, -1.0), c

    def matched(pos_rule: np.ndarray) -> tuple[np.ndarray, float]:
        long = np.where(pos_rule > 0, 1.0, 0.0)
        c = n_long_base / long.sum() if long.sum() > 0 else np.nan
        return np.where(pos_rule > 0, c, -1.0), c

    rows, daily = [], pd.DataFrame({"s": s[scored], "sign_crossed": base_cr}, index=idx)
    rows.append(
        {"tag": tag, "rule": "sign(s), size 1", "exposure_scale": 1.0}
        | stats(base_pos, d, idx, None)
        | {
            "dSharpe_crossed": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "placebo_p": np.nan,
            "dSharpe_2020_21": 0.0,
            "dSharpe_2022_24": 0.0,
            "supported": "",
        }
    )

    def add_rule(name: str, pos: np.ndarray, c: float, placebo_fn) -> None:
        mid, cr = book(pos, r_mid, r_ask, r_bid)
        daily[f"crossed_{name}"] = cr
        daily[f"pos_{name}"] = pos
        d_sh = sharpe(cr) - sharpe(base_cr)
        lo, hi = block_ci(cr - base_cr)
        pl = np.array([placebo_fn() for _ in range(N_PLACEBO)])
        p_val = float(np.mean(pl >= sharpe(cr)))
        d1 = sharpe(cr[half1]) - sharpe(base_cr[half1]) if half1.sum() > 20 else np.nan
        d2 = sharpe(cr[half2]) - sharpe(base_cr[half2]) if half2.sum() > 20 else np.nan
        supported = bool(lo > 0 and p_val < 0.05 and d1 > 0 and d2 > 0)
        rows.append(
            {"tag": tag, "rule": name, "exposure_scale": c}
            | stats(pos, d, idx, base_top)
            | {
                "dSharpe_crossed": d_sh,
                "ci_lo": lo,
                "ci_hi": hi,
                "placebo_p": p_val,
                "dSharpe_2020_21": d1,
                "dSharpe_2022_24": d2,
                "supported": str(supported),
            }
        )

    def shuffled_measures() -> dict[str, np.ndarray]:
        out = {}
        yrs = d.index.year.to_numpy()
        for k, v in meas.items():
            vv = v.copy()
            for y in np.unique(yrs):
                sel = np.flatnonzero(yrs == y)
                vv[sel] = rng.permutation(v[sel])
            out[k] = vv
        return out

    for name, sz in sizes.items():
        pos, c = positions_from_sizes(sz)

        def placebo(name=name) -> float:
            sm = shuffled_measures()
            p_sz = rule_sizes(sm)[name]
            p_pos, _ = positions_from_sizes(np.where(np.isfinite(p_sz), p_sz, 0.0))
            return sharpe(book(p_pos, r_mid, r_ask, r_bid)[1])

        add_rule(name, pos, c, placebo)
    for name, pr in posrules.items():
        pos1 = pr[scored]

        def placebo1(name=name) -> float:
            sm = shuffled_measures()
            p = rule_positions(sm)[name][scored]
            p = np.where(np.isfinite(p), p, -1.0)
            return sharpe(book(p, r_mid, r_ask, r_bid)[1])

        add_rule(name + ", size 1", pos1, 1.0, placebo1)
        posm, cm = matched(pos1)

        def placebo2(name=name) -> float:
            sm = shuffled_measures()
            p = rule_positions(sm)[name][scored]
            p = np.where(np.isfinite(p), p, -1.0)
            pm, _ = matched(p)
            return sharpe(book(pm, r_mid, r_ask, r_bid)[1])

        add_rule(name + ", exposure-matched", posm, cm, placebo2)

    # D: long leg alone (no short), by rule
    long_rows = []
    for col in [c for c in daily.columns if c.startswith("pos_")]:
        pos = daily[col].to_numpy(float)
        lp = np.where(pos > 0, pos, 0.0)
        _, cr = book(lp, r_mid, r_ask, r_bid)
        long_rows.append(
            {
                "tag": tag,
                "rule": col[4:],
                "long_only_Sharpe_crossed": sharpe(cr),
                "long_only_mean_crossed": float(cr.mean()),
                "long_only_top20_share": float(
                    cr[np.isin(idx, top_days(pd.Series(cr, index=idx)))].sum()
                    / cr.sum()
                )
                if cr.sum() != 0
                else np.nan,
            }
        )
    lp0 = np.where(base_pos > 0, 1.0, 0.0)
    _, cr0 = book(lp0, r_mid, r_ask, r_bid)
    long_rows.insert(
        0,
        {
            "tag": tag,
            "rule": "sign(s), size 1",
            "long_only_Sharpe_crossed": sharpe(cr0),
            "long_only_mean_crossed": float(cr0.mean()),
            "long_only_top20_share": float(
                cr0[np.isin(idx, top_days(pd.Series(cr0, index=idx)))].sum() / cr0.sum()
            ),
        },
    )
    # D: by year, crossed Sharpe of every book
    by_year = (
        daily[
            [
                c
                for c in daily.columns
                if c.startswith("crossed_") or c == "sign_crossed"
            ]
        ]
        .groupby(years)
        .agg(lambda x: sharpe(x.to_numpy(float)))
    )
    by_year.index.name = "year"
    by_year["tag"] = tag
    return pd.DataFrame(rows), pd.DataFrame(long_rows), by_year.reset_index()


# ------------------------------------------------------------------- part B --
def bias_table(
    d: pd.DataFrame, m: pd.DataFrame, raw: pd.Series, tag: str
) -> pd.DataFrame:
    s = d["s"].to_numpy(float)
    rv_hat, iv_var = d["rv_hat"].to_numpy(float), d["iv_var"].to_numpy(float)
    rawv = raw.reindex(d.index).to_numpy(float)
    cands = {
        "sign(s) [rv_hat]": np.where(s > 0, 1.0, -1.0),
        "sign(raw - iv_var) [smear-free]": np.where(rawv > iv_var, 1.0, -1.0),
    }
    for w in WINDOWS:
        cands[f"sign(rv_hat e^fsur_{w} - iv_var)"] = bias_corrected_positions(
            m[f"fsur_{w}"].to_numpy(float), rv_hat, iv_var
        )
        cands[f"sign(raw e^fsur_raw_{w} - iv_var)"] = bias_corrected_positions(
            m[f"fsur_raw_{w}"].to_numpy(float), rawv, iv_var
        )
    scored = np.arange(len(s)) >= WARMUP
    for v in cands.values():
        scored &= np.isfinite(v)
    idx = d.index[scored]
    r_mid, r_ask, r_bid = (
        d.loc[idx, c].to_numpy(float) for c in ("r_mid", "r_ask", "r_bid")
    )
    base_top = top_days(
        pd.Series(
            book(cands["sign(s) [rv_hat]"][scored], r_mid, r_ask, r_bid)[1], index=idx
        )
    )
    rows = []
    for name, pos in cands.items():
        st = stats(pos[scored], d, idx, base_top)
        st["agree_with_sign_s"] = float(
            np.mean(pos[scored] == cands["sign(s) [rv_hat]"][scored])
        )
        rows.append({"tag": tag, "rule": name} | st)
    # the bias itself: how often is the trailing signed surprise positive, and how big
    for w in WINDOWS:
        f = m[f"fsur_{w}"].to_numpy(float)[scored]
        rows.append(
            {
                "tag": tag,
                "rule": f"[fsur_{w} on scored days: share > 0 {np.mean(f > 0):.3f}, median exp {np.exp(np.median(f)):.3f}, IQR exp {np.exp(np.quantile(f, 0.25)):.3f}-{np.exp(np.quantile(f, 0.75)):.3f}]",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main --
def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    decks = {tag: P71.load_deck(tag) for tag in TAGS}
    assert decks["sub_live_ridge"].index.equals(decks["blk2"].index)
    days = decks["sub_live_ridge"].index
    vix = vix_1530(days)
    measures, raws = {}, {}
    for tag, d in decks.items():
        rv = P71.realized_last_bar(tag).reindex(days)
        raw = raw_forecast(tag).reindex(days)
        assert rv.notna().all() and raw.notna().all(), tag
        raws[tag] = raw
        measures[tag] = build_measures(d, rv, raw, vix)
    print(
        f"GATE  {len(days)} deck days {days[0].date()} .. {days[-1].date()} (study 71's deck gates re-run on load); realized last-bar variance and "
        f"the smear-free forecast matched on every day; VIX at 15:30 on {int(vix.notna().sum())} days (feed ends {vix.dropna().index.max().date()}); "
        f"corr(log raw, log rv_hat) {np.corrcoef(np.log(raws['sub_live_ridge']), np.log(decks['sub_live_ridge']['rv_hat']))[0, 1]:.3f}"
    )

    # A
    m = measures["sub_live_ridge"]
    d = decks["sub_live_ridge"]
    buys = (d["s"] > 0).to_numpy()
    corr = m[buys].corr(method="spearman")
    corr.to_csv(OUT / "a_measures_corr.csv")
    print(
        "\nA  Spearman correlation of the trailing measures on the buy days (sub_live_ridge)"
    )
    print(corr.to_string(float_format=lambda v: f"{v:+.2f}"))
    grades = []
    for tag in TAGS:
        dd, mm = decks[tag], measures[tag]
        allm = np.ones(len(dd), bool)
        for sample, mask in (
            ("all", allm),
            ("2020-21", np.asarray(dd.index < SPLIT)),
            ("2022-24", np.asarray(dd.index >= SPLIT)),
            (
                "ex 2020-03..06",
                ~np.asarray((dd.index >= COVID[0]) & (dd.index <= COVID[1])),
            ),
        ):
            grades.append(grading(dd, mm, tag, sample, mask))
    g = pd.concat(grades, ignore_index=True)
    g.to_csv(OUT / "a_grading.csv", index=False)
    mono = g[g["bin"] == "Q1"][["tag", "sample", "measure", "rho_hit", "rho_mean"]]
    q_view = g[(g.tag == "sub_live_ridge") & (g["sample"] == "all")].pivot_table(
        index="measure", columns="bin", values="hit"
    )
    s_view = g[(g.tag == "sub_live_ridge") & (g["sample"] == "all")].pivot_table(
        index="measure", columns="bin", values="share_long_pnl"
    )
    print(
        "\nA  long leg by quintile of each measure -- HIT RATE (sub_live_ridge, all buys)"
    )
    print(q_view.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nA  long leg by quintile of each measure -- SHARE OF LONG P&L (crossed)")
    print(s_view.to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\nA  Spearman of quintile index vs hit / mean crossed, by deck and sample")
    print(
        mono.pivot_table(
            index=["measure"], columns=["tag", "sample"], values="rho_hit"
        ).to_string(float_format=lambda v: f"{v:+.2f}")
    )
    reg = pd.concat(
        [regime_tables(decks[t], measures[t], t) for t in TAGS], ignore_index=True
    )
    reg.to_csv(OUT / "a_regime_3x3.csv", index=False)
    print(
        "\nA  regime check (sub_live_ridge): the long leg by surprise tercile WITHIN VIX tercile -- hit / mean crossed / n"
    )
    r = reg[reg.tag == "sub_live_ridge"]
    print(
        r.pivot_table(index="v", columns="q", values="hit", observed=True).to_string(
            float_format=lambda v: f"{v:.2f}"
        )
    )
    print(
        r.pivot_table(
            index="v", columns="q", values="mean_crossed", observed=True
        ).to_string(float_format=lambda v: f"{v:+.2f}")
    )
    print(r.pivot_table(index="v", columns="q", values="n", observed=True).to_string())

    # B
    b = pd.concat(
        [bias_table(decks[t], measures[t], raws[t], t) for t in TAGS], ignore_index=True
    )
    b.to_csv(OUT / "b_bias_correction.csv", index=False)
    print(
        f"\nB  causal bias correction of the forecast, scored after the {WARMUP}-session warm-up; positions +1 / -1"
    )
    print(b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # C + D
    c_all, long_all, year_all = [], [], []
    for tag in TAGS:
        c, lo, yr = evaluate_rules(decks[tag], measures[tag], tag)
        c_all.append(c)
        long_all.append(lo)
        year_all.append(yr)
    c_df = pd.concat(c_all, ignore_index=True)
    c_df.to_csv(OUT / "c_rules.csv", index=False)
    long_df = pd.concat(long_all, ignore_index=True)
    long_df.to_csv(OUT / "d_long_only.csv", index=False)
    year_df = pd.concat(year_all, ignore_index=True)
    year_df.to_csv(OUT / "d_by_year.csv", index=False)
    print(
        f"\nC  pre-registered rules, causal, warm-up {WARMUP}, long exposure matched to sign(s) by one constant, short leg -1; crossed, per unit premium"
    )
    cols = [
        "tag",
        "rule",
        "exposure_scale",
        "buys",
        "pct_buy",
        "hit_buys",
        "Sharpe_mid",
        "Sharpe_crossed",
        "dSharpe_crossed",
        "ci_lo",
        "ci_hi",
        "placebo_p",
        "dSharpe_2020_21",
        "dSharpe_2022_24",
        "top20_share",
        "top20_kept_from_sign",
        "max_dd_crossed",
        "supported",
    ]
    print(c_df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nD  the long leg alone (no short), crossed")
    print(long_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nD  crossed Sharpe by year (sub_live_ridge)")
    yv = year_df[year_df.tag == "sub_live_ridge"].drop(columns="tag").set_index("year")
    yv.columns = [c.replace("crossed_", "") for c in yv.columns]
    print(yv.T.to_string(float_format=lambda v: f"{v:.2f}"))

    live = c_df[c_df.tag == "sub_live_ridge"]
    sup = live[live.supported == "True"]
    best = live.iloc[1:].sort_values("dSharpe_crossed", ascending=False).iloc[0]
    mono_live = mono[
        (mono.tag == "sub_live_ridge") & (mono["sample"] == "all")
    ].set_index("measure")
    print(
        f"\nVERDICT  (1) what grades the long leg: trailing QLIKE_63 Spearman(hit) {mono_live.loc['qlike_63', 'rho_hit']:+.2f}, signed forecast surprise fsur_63 "
        f"{mono_live.loc['fsur_63', 'rho_hit']:+.2f}, implied surprise isur_63 {mono_live.loc['isur_63', 'rho_hit']:+.2f}, VIX level {mono_live.loc['vix', 'rho_hit']:+.2f}; "
        f"(2) any pre-registered rule SUPPORTED: {'False' if sup.empty else 'True: ' + ', '.join(sup.rule)} (best {best.rule}: dSharpe crossed {best.dSharpe_crossed:+.2f}, "
        f"interval [{best.ci_lo:+.4f}, {best.ci_hi:+.4f}], placebo p {best.placebo_p:.3f}, halves {best.dSharpe_2020_21:+.2f} / {best.dSharpe_2022_24:+.2f})"
    )


if __name__ == "__main__":
    main()
