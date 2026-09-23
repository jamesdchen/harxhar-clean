"""Study 71 -- conviction sizing of the sign(s) long leg.

The question (user, 2026-09-23): the long side of sign(s) is right about 40% of
the time; can the trade be scaled so that it bets harder when the chance of
being right is higher?  The sizing ledger of 2026-09-01 answered "the sign
split is the edge, no slope" on the block ridge; this study re-asks it on the
per-bar live-feasible ridge (deck tag sub_live_ridge; blk2 alongside for
continuity), on the 866 deck days, at the crossed spread.

A. Calibration of the long leg (descriptive).  On days with s > 0 (s = rv_hat
   - iv_var, the deck's ``signal``): hit rate P(R > 0), mean return at mid and
   crossed, and the share of the long leg's P&L, by quintile of s, of s/iv_var
   (the relative gap), of the forecast's trailing 63-session realized QLIKE (a
   causal confidence proxy: lower = the forecast has been good lately), and by
   the calendar flags month-end / third Friday / FOMC day; whole sample and
   2020-21 vs 2022-24.  Full-sample quintiles here (this part describes; it
   does not trade).

B. Pre-registered sizing rules for the long leg; the short leg stays at -1.
   Every rule is causal (trailing 252 sessions, expanding fits after a 252-
   session warm-up, refit every 21) and is normalised to unit MEAN size over
   buys by construction (deterministic constants, no look-ahead), so the
   comparison is at equal average long exposure:
     (i)   rank:  size = 2 x rank of s among the trailing 252 sessions'
           positive s (uniform ranks average 1/2);
     (ii)  top quintile: size = (1 + 1{s above the trailing 80th percentile of
           positive s}) / 1.2;
     (iii) Kelly by quintile: at each refit, quintile edges of all prior
           positive s and, per quintile, mean / variance of the crossed long
           return on those prior buys; size = max(mean/var, 0) rescaled to
           mean 1 across the five quintiles;
     (iv)  calendar conviction: size 2 on a month-end close, 1 otherwise,
           divided by (1 + 1/21) -- the book's month-end override in disguise.
   CRITERION (stated before the results): a rule is SUPPORTED if its crossed
   Sharpe beats sign(s) with a circular-block-bootstrap 95% interval on the
   daily crossed P&L difference excluding zero AND it beats a placebo (its
   sizes permuted among the buy days within each calendar year, 200 draws) at
   p < 0.05.  Reported beside: mean, tail capture (share of P&L from the top
   20 days and how many of sign(s)'s top-20 days it keeps), max drawdown in
   premium units, buys, realised mean size.

C. If nothing grades: the hit-rate spread across quintiles that a rank rule
   would need to add 0.2 of crossed Sharpe, by simulation on the observed
   long-leg return distribution.

Gates: signal == rv_hat - iv_var and R == exit/entry - 1 on every deck day;
bid <= entry <= ask on every day; the two decks share the 866 days.
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
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "71"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
RELEASES = ROOT / "data" / "releases.parquet"
TAGS = ("sub_live_ridge", "blk2")
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
WARMUP = 252  # sessions before the first scored day of a causal rule
REFIT = 21  # sessions between Kelly refits
TRAIL_RANK = 252  # sessions in the rank / top-quintile window
TRAIL_QLIKE = 63  # sessions in the confidence proxy
N_Q = 5
TOP_DAYS = 20
BOOT_B, BOOT_BLOCK, SEED = 2000, 21, 0
N_PLACEBO = 200
SPLIT = pd.Timestamp("2022-01-01")
GATE_TOL = 1e-9


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * ANN) if sd > 0 else float("nan")


def book(
    pos: np.ndarray, r_mid: np.ndarray, r_ask: np.ndarray, r_bid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Signed size -> returns at mid and crossed, per unit premium (longs pay the ask, shorts receive the bid)."""
    mid = pos * r_mid
    crossed = np.where(pos > 0, pos * r_ask, np.where(pos < 0, pos * r_bid, 0.0))
    return mid, crossed


def max_drawdown(x: np.ndarray) -> float:
    c = np.cumsum(x)
    return float(np.max(np.maximum.accumulate(c) - c))


def top_days(x: pd.Series, k: int = TOP_DAYS) -> pd.Index:
    return x.sort_values(ascending=False).index[:k]


def block_ci(diff: np.ndarray) -> tuple[float, float]:
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, diff.size]), diff.size, BOOT_BLOCK, BOOT_B
    )
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------ inputs --
def load_deck(tag: str) -> pd.DataFrame:
    d = pd.read_parquet(DECK / f"daily_{tag}.parquet").sort_index()
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index)).normalize()
    assert np.allclose(d["rv_hat"] - d["iv_var"], d["signal"], atol=GATE_TOL), tag
    assert np.allclose(d["exit"] / d["entry"] - 1.0, d["R"], atol=GATE_TOL), tag
    ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
    bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
    entry = d["entry"].to_numpy(float)
    assert (
        np.all(bid <= entry + GATE_TOL)
        and np.all(entry <= ask + GATE_TOL)
        and np.all(bid > 0)
    ), tag
    out = pd.DataFrame(
        {
            "s": d["signal"].to_numpy(float),
            "iv_var": d["iv_var"].to_numpy(float),
            "rv_hat": d["rv_hat"].to_numpy(float),
            "r_mid": d["R"].to_numpy(float),
            "r_ask": d["exit"].to_numpy(float) / ask - 1.0,
            "r_bid": d["exit"].to_numpy(float) / bid - 1.0,
        },
        index=d.index,
    )
    return out


def realized_last_bar(tag: str) -> pd.Series:
    """Realized variance of the 15:30-16:00 bar by session, from the forecast table."""
    y = pd.read_parquet(asl.yhat_paths(ROOT)[tag], columns=["t", "rv_raw"])
    et = pd.to_datetime(y["t"], utc=True).dt.tz_convert("America/New_York")
    last = et.dt.strftime("%H:%M") == "16:00"
    s = pd.Series(
        y.loc[last, "rv_raw"].to_numpy(float),
        index=pd.DatetimeIndex(et[last].dt.tz_localize(None)).normalize(),
    )
    return s[~s.index.duplicated()]


def calendar_flags(days: pd.DatetimeIndex) -> pd.DataFrame:
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p56 = _load(HERE / "56_close_calendar_family.py", "p56_close_calendar")
    p58 = _load(HERE / "58_third_friday_short.py", "p58_third_friday")
    stamp, _ = p43.session_stamps()
    cal = p56.session_calendar(pd.DatetimeIndex(stamp.index))
    types = p56.day_types(cal)
    me = pd.Series(types["month-end"].to_numpy(), index=cal)
    tf = pd.Series(p58.nth_weekday_sessions(cal, 4, 3), index=cal)
    rel = pd.read_parquet(RELEASES, columns=["endbartime", "fomc release"])
    rel["endbartime"] = pd.to_datetime(rel["endbartime"])
    fomc_days = pd.DatetimeIndex(
        rel.loc[
            pd.to_numeric(rel["fomc release"], errors="coerce").fillna(0) > 0,
            "endbartime",
        ]
        .dt.normalize()
        .unique()
    )
    return pd.DataFrame(
        {
            "month_end": me.reindex(days).fillna(False).to_numpy(bool),
            "third_friday": tf.reindex(days).fillna(False).to_numpy(bool),
            "fomc_day": days.isin(fomc_days),
        },
        index=days,
    )


# ------------------------------------------------------------------- part A --
def calibration(d: pd.DataFrame, flags: pd.DataFrame, tag: str) -> pd.DataFrame:
    buys = d[d["s"] > 0].copy()
    buys["hit"] = buys["r_mid"] > 0
    rows = []

    def add(
        kind: str, label: str, m: np.ndarray, sample: str, total_crossed: float
    ) -> None:
        g = buys[m]
        rows.append(
            {
                "tag": tag,
                "sample": sample,
                "grading": kind,
                "bin": label,
                "n": int(len(g)),
                "hit": float(g["hit"].mean()) if len(g) else np.nan,
                "mean_mid": float(g["r_mid"].mean()) if len(g) else np.nan,
                "mean_crossed": float(g["r_ask"].mean()) if len(g) else np.nan,
                "share_long_pnl_crossed": float(g["r_ask"].sum() / total_crossed)
                if len(g)
                else np.nan,
            }
        )

    gradings = {
        "s": buys["s"],
        "s_over_iv": buys["s"] / buys["iv_var"],
        "trailing_qlike_63": buys["qlike_trail"],
    }
    for sample, m_s in (
        ("all", np.ones(len(buys), bool)),
        ("2020-21", np.asarray(buys.index < SPLIT)),
        ("2022-24", np.asarray(buys.index >= SPLIT)),
    ):
        tot = float(buys.loc[m_s, "r_ask"].sum())
        for kind, v in gradings.items():
            ok = m_s & np.isfinite(v.to_numpy(float))
            q = pd.qcut(v[ok], N_Q, labels=False, duplicates="drop")
            for k in range(N_Q):
                mk = np.zeros(len(buys), bool)
                mk[np.flatnonzero(ok)[q.to_numpy() == k]] = True
                add(kind, f"Q{k + 1}", mk, sample, tot)
        for flag in flags.columns:
            f = flags.loc[buys.index, flag].to_numpy(bool)
            add(flag, "on", m_s & f, sample, tot)
            add(flag, "off", m_s & ~f, sample, tot)
    return pd.DataFrame(rows)


def monotone_report(cal: pd.DataFrame) -> pd.DataFrame:
    """Spearman of quintile index against hit and mean crossed, per grading and sample."""
    from scipy.stats import spearmanr

    rows = []
    for (tag, sample, kind), g in cal[cal["bin"].str.startswith("Q")].groupby(
        ["tag", "sample", "grading"], sort=False
    ):
        k = g["bin"].str[1:].astype(int).to_numpy()
        rows.append(
            {
                "tag": tag,
                "sample": sample,
                "grading": kind,
                "rho_hit": float(spearmanr(k, g["hit"]).statistic),
                "rho_mean_crossed": float(spearmanr(k, g["mean_crossed"]).statistic),
                "hit_Q1": float(g["hit"].iloc[0]),
                "hit_Q5": float(g["hit"].iloc[-1]),
                "mean_crossed_Q1": float(g["mean_crossed"].iloc[0]),
                "mean_crossed_Q5": float(g["mean_crossed"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- part B --
def size_rank(s: np.ndarray) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        if s[t] <= 0:
            out[t] = 0.0
            continue
        past = s[max(0, t - TRAIL_RANK) : t]
        pos = past[past > 0]
        if t < WARMUP or pos.size < N_Q:
            continue
        out[t] = 2.0 * (np.sum(pos <= s[t]) / pos.size)
    return out


def size_top_quintile(s: np.ndarray) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        if s[t] <= 0:
            out[t] = 0.0
            continue
        past = s[max(0, t - TRAIL_RANK) : t]
        pos = past[past > 0]
        if t < WARMUP or pos.size < N_Q:
            continue
        out[t] = (1.0 + float(s[t] > np.quantile(pos, 0.8))) / 1.2
    return out


def size_kelly(s: np.ndarray, r_long: np.ndarray) -> np.ndarray:
    """Expanding, refit every REFIT sessions after WARMUP: quintile edges of prior positive s and
    max(mean/var, 0) of the crossed long return per quintile, rescaled to mean 1 over quintiles."""
    out = np.full(len(s), np.nan)
    edges = None
    w = None
    for t in range(len(s)):
        if t >= WARMUP and (t - WARMUP) % REFIT == 0:
            past_s, past_r = s[:t], r_long[:t]
            m = past_s > 0
            if m.sum() >= 5 * N_Q:
                edges = np.quantile(past_s[m], np.linspace(0, 1, N_Q + 1)[1:-1])
                q = np.searchsorted(edges, past_s[m], side="right")
                w = np.zeros(N_Q)
                for k in range(N_Q):
                    rk = past_r[m][q == k]
                    if rk.size >= 2 and rk.var(ddof=1) > 0:
                        w[k] = max(rk.mean() / rk.var(ddof=1), 0.0)
                w = w / w.mean() if w.mean() > 0 else np.ones(N_Q)
        if s[t] <= 0:
            out[t] = 0.0
            continue
        if edges is None or w is None:
            continue
        out[t] = w[int(np.searchsorted(edges, s[t], side="right"))]
    return out


def size_calendar(s: np.ndarray, month_end: np.ndarray) -> np.ndarray:
    out = np.where(s > 0, np.where(month_end, 2.0, 1.0) / (1.0 + 1.0 / 21.0), 0.0)
    out[:WARMUP] = np.nan  # scored on the same days as the causal rules
    return out


def evaluate(
    d: pd.DataFrame, sizes: dict[str, np.ndarray], tag: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = d["s"].to_numpy(float)
    r_mid, r_ask, r_bid = (d[c].to_numpy(float) for c in ("r_mid", "r_ask", "r_bid"))
    scored = np.ones(len(d), bool)
    for v in sizes.values():
        scored &= np.isfinite(v)
    idx = d.index[scored]
    base_pos = np.where(s > 0, 1.0, -1.0)[scored]
    base_mid, base_cr = book(base_pos, r_mid[scored], r_ask[scored], r_bid[scored])
    base_top = top_days(pd.Series(base_cr, index=idx))
    years = idx.year.to_numpy()
    rng = np.random.default_rng([SEED, 71])
    rows = []
    daily = pd.DataFrame(
        {
            "s": s[scored],
            "r_mid": r_mid[scored],
            "r_ask": r_ask[scored],
            "r_bid": r_bid[scored],
            "sign_crossed": base_cr,
        },
        index=idx,
    )

    def row(name: str, pos: np.ndarray, mid: np.ndarray, cr: np.ndarray) -> dict:
        buys = pos > 0
        top = top_days(pd.Series(cr, index=idx))
        return {
            "tag": tag,
            "rule": name,
            "days": int(len(pos)),
            "buys": int(buys.sum()),
            "mean_size_buys": float(pos[buys].mean()) if buys.any() else np.nan,
            "Sharpe_mid": sharpe(mid),
            "Sharpe_crossed": sharpe(cr),
            "mean_crossed": float(cr.mean()),
            "top20_share": float(cr[np.isin(idx, top)].sum() / cr.sum())
            if cr.sum() != 0
            else np.nan,
            "top20_kept_from_sign": int(len(top.intersection(base_top))),
            "max_dd_crossed": max_drawdown(cr),
            "worst_crossed": float(cr.min()),
        }

    rows.append(
        row("sign(s), size 1", base_pos, base_mid, base_cr)
        | {
            "dSharpe_crossed": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "placebo_p": np.nan,
            "supported": "",
        }
    )
    for name, v in sizes.items():
        sz = v[scored]
        pos = np.where(s[scored] > 0, sz, -1.0)
        mid, cr = book(pos, r_mid[scored], r_ask[scored], r_bid[scored])
        daily[f"size_{name}"] = pos
        daily[f"crossed_{name}"] = cr
        d_sh = sharpe(cr) - sharpe(base_cr)
        lo, hi = block_ci(cr - base_cr)
        # placebo: the same sizes dealt at random among the buy days within each calendar year
        buys = np.flatnonzero(pos > 0)
        pl = np.empty(N_PLACEBO)
        for b in range(N_PLACEBO):
            pp = pos.copy()
            for y in np.unique(years[buys]):
                sel = buys[years[buys] == y]
                pp[sel] = rng.permutation(pos[sel])
            _, pc = book(pp, r_mid[scored], r_ask[scored], r_bid[scored])
            pl[b] = sharpe(pc)
        p_val = float(np.mean(pl >= sharpe(cr)))
        supported = bool((lo > 0) and (p_val < 0.05))
        rows.append(
            row(name, pos, mid, cr)
            | {
                "dSharpe_crossed": d_sh,
                "ci_lo": lo,
                "ci_hi": hi,
                "placebo_p": p_val,
                "supported": str(supported),
            }
        )
    return pd.DataFrame(rows), daily


# ------------------------------------------------------------------- part C --
def required_spread(daily: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Hit-rate spread across quintiles a rank rule would need for +0.2 crossed Sharpe.

    The observed long-leg crossed returns (on the scored days) are dealt into five
    equal groups so that the hit rate rises linearly across groups by ``spread``
    from Q1 to Q5 (wins and losses drawn without replacement); the groups get rank
    sizes 0.2, 0.6, 1.0, 1.4, 1.8 (mean 1); the shorts are unchanged.  200 deals per
    spread; the Sharpe gain is averaged.
    """
    s = daily["s"].to_numpy(float)
    r_ask, r_bid = daily["r_ask"].to_numpy(float), daily["r_bid"].to_numpy(float)
    buys = s > 0
    long_r = r_ask[buys]
    short_cr = -r_bid[~buys]
    wins, losses = long_r[long_r > 0], long_r[long_r <= 0]
    p_bar = wins.size / long_r.size
    sizes_q = 2.0 * (np.arange(N_Q) + 0.5) / N_Q
    base = np.concatenate([long_r, short_cr])
    base_sh = sharpe(base)
    rng = np.random.default_rng([SEED, 710])
    rows = []
    n_per = long_r.size // N_Q
    for spread in np.arange(0.0, 0.41, 0.05):
        hit_q = np.clip(
            p_bar + spread * (np.arange(N_Q) - (N_Q - 1) / 2) / (N_Q - 1), 0.0, 1.0
        )
        n_win_q = np.round(hit_q * n_per).astype(int)
        if n_win_q.sum() > wins.size or (n_per * N_Q - n_win_q.sum()) > losses.size:
            continue
        gains = np.empty(N_PLACEBO)
        for b in range(N_PLACEBO):
            w_perm = rng.permutation(wins)
            l_perm = rng.permutation(losses)
            sized = []
            iw = il = 0
            for k in range(N_Q):
                nw = n_win_q[k]
                nl = n_per - nw
                grp = np.concatenate([w_perm[iw : iw + nw], l_perm[il : il + nl]])
                iw += nw
                il += nl
                sized.append(sizes_q[k] * grp)
            gains[b] = sharpe(np.concatenate([*sized, short_cr])) - base_sh
        rows.append(
            {
                "tag": tag,
                "hit_spread_Q5_minus_Q1": float(spread),
                "hit_Q1": float(hit_q[0]),
                "hit_Q5": float(hit_q[-1]),
                "mean_dSharpe_crossed": float(gains.mean()),
                "p10": float(np.percentile(gains, 10)),
                "p90": float(np.percentile(gains, 90)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main --
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    decks = {tag: load_deck(tag) for tag in TAGS}
    assert decks["sub_live_ridge"].index.equals(decks["blk2"].index)
    days = decks["sub_live_ridge"].index
    flags = calendar_flags(days)
    print(
        f"GATE  {len(days)} deck days {days[0].date()} .. {days[-1].date()}; signal == rv_hat - iv_var, R == exit/entry - 1, "
        f"bid <= entry <= ask on every day of both decks; flags: month-end {int(flags.month_end.sum())}, third Friday "
        f"{int(flags.third_friday.sum())}, FOMC day {int(flags.fomc_day.sum())}"
    )
    for tag, d in decks.items():
        rv = realized_last_bar(tag).reindex(days)
        q = rv / d["rv_hat"]
        d["qlike"] = q - np.log(q) - 1.0
        d["qlike_trail"] = (
            d["qlike"].rolling(TRAIL_QLIKE, min_periods=TRAIL_QLIKE).mean().shift(1)
        )
        n_buy = int((d["s"] > 0).sum())
        print(
            f"{tag}: buys {n_buy} of {len(d)} ({100 * n_buy / len(d):.1f}%), hit on buys {(d.loc[d.s > 0, 'r_mid'] > 0).mean():.3f}, "
            f"hit on shorts {(d.loc[d.s <= 0, 'r_mid'] < 0).mean():.3f}; realized last-bar variance matched on {int(rv.notna().sum())} days"
        )

    # A
    cal = pd.concat(
        [calibration(d, flags, tag) for tag, d in decks.items()], ignore_index=True
    )
    cal.to_csv(OUT / "a_calibration.csv", index=False)
    mono = monotone_report(cal)
    mono.to_csv(OUT / "a_monotone.csv", index=False)
    print("\nA  long leg (s > 0) by quintile / flag -- sub_live_ridge, whole sample")
    show = cal[(cal.tag == "sub_live_ridge") & (cal["sample"] == "all")]
    print(
        show.drop(columns=["tag", "sample"]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )
    print(
        "\nA  is any grading monotone?  Spearman of quintile index vs hit / mean crossed"
    )
    print(mono.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # B
    b_all, dailies = [], []
    for tag, d in decks.items():
        s = d["s"].to_numpy(float)
        sizes = {
            "(i) rank of s, trailing 252": size_rank(s),
            "(ii) top trailing quintile x2": size_top_quintile(s),
            "(iii) Kelly by s-quintile, expanding": size_kelly(
                s, d["r_ask"].to_numpy(float)
            ),
            "(iv) month-end x2": size_calendar(s, flags["month_end"].to_numpy(bool)),
        }
        b, daily = evaluate(d, sizes, tag)
        b_all.append(b)
        dailies.append(daily.assign(tag=tag))
    b_df = pd.concat(b_all, ignore_index=True)
    b_df.to_csv(OUT / "b_rules.csv", index=False)
    pd.concat(dailies).to_csv(OUT / "b_daily.csv")
    print(
        f"\nB  sizing rules on the long leg, causal, scored after a {WARMUP}-session warm-up; short leg -1; per unit premium"
    )
    print(b_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # C
    c_all = [required_spread(daily, tag) for daily, tag in zip(dailies, TAGS)]
    c_df = pd.concat(c_all, ignore_index=True)
    c_df.to_csv(OUT / "c_required_spread.csv", index=False)
    print(
        "\nC  what calibration a rank rule would need: hit-rate spread Q5 - Q1 vs the crossed Sharpe it adds (simulated on the observed returns)"
    )
    print(c_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    live = b_df[b_df.tag == "sub_live_ridge"]
    sup = live[live.supported == "True"]
    best = live.iloc[1:].sort_values("dSharpe_crossed", ascending=False).iloc[0]
    m_live = mono[
        (mono.tag == "sub_live_ridge")
        & (mono["sample"] == "all")
        & (mono.grading == "s")
    ].iloc[0]
    print(
        f"\nVERDICT  (1) does the long leg grade by s: hit Q1 {m_live.hit_Q1:.3f} -> Q5 {m_live.hit_Q5:.3f}, Spearman {m_live.rho_hit:+.2f} "
        f"(mean crossed {m_live.rho_mean_crossed:+.2f}); (2) any sizing rule SUPPORTED by the pre-registered criterion: "
        f"{'False' if sup.empty else 'True: ' + ', '.join(sup.rule)} (best {best.rule}: dSharpe crossed {best.dSharpe_crossed:+.2f}, "
        f"interval [{best.ci_lo:+.4f}, {best.ci_hi:+.4f}] on the daily difference, placebo p {best.placebo_p:.3f})"
    )


if __name__ == "__main__":
    main()
