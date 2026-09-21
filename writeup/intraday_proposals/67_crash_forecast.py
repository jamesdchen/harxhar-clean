"""67 - the crash: can it be forecast, and what keeps the books small when it comes?

The S&P held on the same capital as the insurance book added about 10 points a
year over 2020-2025; a crash hits both at once, and that is when margin calls
come.  Can the crash be forecast?

"Forecast" has two meanings that have to be kept apart:
  SIZE   how big tomorrow's move will be - the variance, which the paper shows
         is forecastable;
  SIGN   which way it goes, and the FIRST shock out of a calm market.
The insurance leg loses on size in either direction; the S&P leg loses on sign;
a margin call is a multi-day S&P drawdown.

Data: the S&P 500's daily close (^GSPC, 1998-2024; it equals the chain's
settlement close on every 2020-2025 session) for the S&P leg; the panel (ES
futures, 30-minute bars, 1998-01-05..2024-04-30) for regular-hours realized
variance only - its summed bars track the index's close-to-close return with
correlation 0.98 on matched sessions and miss by 2-3% on the March-2020 limit
days, so the index's own close is used (a first run on the summed bars was
withdrawn); VIX (2006-05..2024-02) and VIX3M (2009-08..); study 50's replay
of the 13:30 insurance book, 1998-2024 (its 13:30->16:00 bar path tracks the
chain's spot, correlation 0.998 on 866 sessions).

Signals, all known at the PRIOR session's close, each turned into its causal
expanding percentile (share of strictly earlier values at or below it, 252
earlier values minimum):
  RV21  trailing 21-session mean realized variance (the brake's own input)
  RV5   trailing 5-session mean
  HAR   expanding-OLS HAR forecast of the session's log realized variance
        (previous session, week, month)
  VIX   VIX at the prior close
  TS    VIX / VIX3M at the prior close (above 1 = inverted term structure)
  DD    depth of the S&P below its trailing 252-session high

Events:
  E1  S&P crash day: close-to-close return in the worst 1% of sessions
  E1f E1 first shocks: E1 days with no E1 day in the previous 21 sessions
  E2  insurance crash day: the replay's full-size loss (fraction of the index)
      in the worst 1%
  E3  margin-call run: the S&P's next-20-session return in the worst 1% of
      windows, dated at the window's first session

  A  forecastability.  For each event x signal: the AUC of the causal
     percentile, and the LIFT - the share of the event's loss that falls in the
     signal's top quintile (percentile >= 0.8) over that state's share of days -
     with 21-session circular block-bootstrap intervals (B 2000, seed 0).
     Episodes: the ten worst non-overlapping 20-session S&P runs, with each
     signal's percentile on the run's first session.
  B  size, not sign.  By quintile of the signal: the share of the largest 2% of
     moves (|return|) that are down, and the mean session return.
  C  a better brake for the insurance leg?  Study 50's rule (half size above
     the expanding 90th percentile, zero above the 97.5th) driven by each signal
     instead of RV21, on the replay, each on its own sample beside RV21 on the
     same sample: days halved / zeroed, share of premium given up, and at the
     live 10% of capital per stress unit the worst day, the 1-in-200 day and the
     worst 20-session run.  The replay's P&L enters with its mean replaced by
     the realized book's (its tails are an upper bound, its mean is not the
     book's).
  D  the S&P leg.  Buy and hold against: VM volatility-managed (weight =
     min(1, expanding median of the HAR-forecast volatility / today's)); MA200
     (hold above the 200-session moving average of prior closes, cash below);
     BRK the insurance brake's RV21 rule applied to the S&P; TSI cash when
     VIX / VIX3M > 1 (from 2009-08; held before).  Cash earns nothing (this
     understates the rules).  Whole sample and halves: annual growth, vol,
     Sharpe, max drawdown, worst 20-session run; each rule's loss inside seven
     dated S&P drawdowns; the paired block-bootstrap interval of the Sharpe
     difference.  The JOINT book, the S&P leg plus the insurance leg at 10%
     with its brake: worst day, worst 20-session run, max drawdown, on the
     replay 1998-2024 and on the realized tape 2020-01..2024-04.

Written before running:
  A  a signal FORECASTS an event for sizing if the lift is at least 2 with its
     bootstrap lower bound above 1.
  C  a signal REPLACES RV21 in the brake if on its own sample it lowers both
     the 1-in-200 day and the worst 20-session run in both halves of that
     sample and gives up no more premium than RV21.
  D  a rule MITIGATES the crash if it lowers both the max drawdown and the
     worst 20-session run in both halves of 1998-2024 and its Sharpe-difference
     interval against buy and hold is not entirely below zero.

GATES  study 50's replay and brake counts (study 65's load_replay); the RV21
       brake rebuilt here equals the replay's; the ^GSPC close equals the
       chain's settlement close on all 1,279 sessions.

Run:  python writeup/intraday_proposals/67_crash_forecast.py
"""

from __future__ import annotations

import bisect
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
from live.ibkr.premium_ledger import PremiumLedger  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "67"
PANEL = ROOT / "data" / "core_stats.parquet"
VIXP = ROOT / "data" / "vix_and_voldemand.parquet"
# ^GSPC daily closes (yfinance, auto-adjusted), fetched once into the study's folder
GSPC = OUT / "gspc_close.parquet"
GATE_CLOSE_TOL = 1e-9
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
MIN_OBS = 252
EVENT_Q = 0.01
BIG_MOVE_Q = 0.02
TOP = 0.8  # top quintile of the causal percentile
RUN = 20
FIRST_GAP = 21
N_EPISODES = 10
MA = 200
HIGH_WINDOW = 252
B, BLOCK, SEED = 2000, 21, 0
TAPE_START = "2020-01-01"
DRAWDOWNS = (  # S&P 500 closing peak -> trough
    ("2000-02 dot-com", "2000-03-24", "2002-10-09"),
    ("2007-09 financial crisis", "2007-10-09", "2009-03-09"),
    ("2011 downgrade", "2011-04-29", "2011-10-03"),
    ("2015-16", "2015-05-21", "2016-02-11"),
    ("2018 Q4", "2018-09-20", "2018-12-24"),
    ("2020 covid", "2020-02-19", "2020-03-23"),
    ("2022", "2022-01-03", "2022-10-12"),
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def causal_pct(x: np.ndarray) -> np.ndarray:
    """Share of strictly earlier finite values at or below each value."""
    out = np.full(len(x), np.nan)
    seen: list[float] = []
    for i, v in enumerate(x):
        if np.isfinite(v):
            if len(seen) >= MIN_OBS:
                out[i] = bisect.bisect_right(seen, v) / len(seen)
            bisect.insort(seen, float(v))
    return out


def har_forecast(rv: pd.Series) -> pd.Series:
    """Expanding-OLS HAR forecast of log rv_t from information through t-1."""
    x = np.log(rv.to_numpy(float))
    lvl = rv.to_numpy(float)
    n = len(x)
    feats = np.full((n, 4), np.nan)
    for t in range(22, n):
        feats[t] = (
            1.0,
            x[t - 1],
            np.log(lvl[t - 5 : t].mean()),
            np.log(lvl[t - 22 : t].mean()),
        )
    ok = np.isfinite(feats).all(axis=1) & np.isfinite(x)
    xx = np.zeros((4, 4))
    xy = np.zeros(4)
    k = 0
    out = np.full(n, np.nan)
    for t in range(n):
        if ok[t] and k >= MIN_OBS:
            out[t] = feats[t] @ np.linalg.solve(xx, xy)
        if ok[t]:
            xx += np.outer(feats[t], feats[t])
            xy += feats[t] * x[t]
            k += 1
    return pd.Series(out, index=rv.index)


def maxdd(r: np.ndarray) -> float:
    w = np.cumprod(1.0 + r)
    return float((w / np.maximum.accumulate(np.maximum(w, 1.0)) - 1.0).min())


def worst_run(r: np.ndarray, n: int = RUN) -> float:
    lg = np.concatenate([[0.0], np.cumsum(np.log1p(r))])
    return float(np.expm1((lg[n:] - lg[:-n]).min())) if len(r) >= n else np.nan


def summary(r: np.ndarray) -> dict[str, float]:
    return {
        "ann_growth": float(np.prod(1.0 + r) ** (asl.PERIODS_PER_YEAR / len(r)) - 1),
        "ann_vol": float(r.std(ddof=1) * ANN),
        "Sharpe": float(r.mean() / r.std(ddof=1) * ANN),
        "max_drawdown": maxdd(r),
        "worst_20": worst_run(r),
        "worst_day": float(r.min()),
    }


def auc_rows(p: np.ndarray, e: np.ndarray) -> np.ndarray:
    """AUC of p for e, row by row (Mann-Whitney with average ranks)."""
    rk = rankdata(p, axis=1)
    n1 = e.sum(axis=1)
    n0 = e.shape[1] - n1
    return ((rk * e).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def panel_daily(p49: ModuleType) -> pd.DataFrame:
    pan = p49.panel_bars()
    full = pan["bar_count"][pan["bar_count"] == p49.N_RTH_BARS].index
    raw = pd.read_parquet(PANEL, columns=["endbartime", "sumret"]).dropna()
    t = pd.to_datetime(raw["endbartime"]).to_numpy()
    closes = (full + pd.Timedelta(hours=16)).to_numpy()
    pos = np.searchsorted(closes, t, side="left")
    ok = pos < len(closes)
    lr = np.bincount(
        pos[ok], weights=raw["sumret"].to_numpy(float)[ok], minlength=len(closes)
    )
    d = pd.DataFrame({"lr": lr}, index=full).iloc[1:]
    rv = pan["rv_day"].reindex(d.index)
    d["rv"] = rv
    d["RV21"] = pan["rv_trail"].reindex(d.index)
    d["RV5"] = rv.rolling(5, min_periods=5).mean().shift(1)
    d["HAR"] = har_forecast(rv)
    v = pd.read_parquet(VIXP, columns=["endbartime", "vix", "vix3m"])
    v["t"] = pd.to_datetime(v["endbartime"])
    v = v[v["t"].dt.strftime("%H:%M") <= "16:00"]
    vd = v.groupby(v["t"].dt.normalize())[["vix", "vix3m"]].last()
    vd = vd.reindex(d.index)
    d["VIX"] = vd["vix"].shift(1)
    d["TS"] = (vd["vix"] / vd["vix3m"]).shift(1)
    # The S&P leg is the index's own close.  The panel's summed bars are NOT a
    # price path: sumret drops the return across missing minutes (halts,
    # limit-locked nights), so the sum of a session's 48 bars misses the moves
    # that matter most (kept as panel_lr for the record).
    gs = pd.read_parquet(GSPC)["close"]
    d = d.rename(columns={"lr": "panel_lr"})
    d = d[d.index.isin(gs.index)]
    lvl = gs.reindex(d.index)
    d["lr"] = np.log(lvl).diff()
    d = d.iloc[1:]
    lvl = lvl.iloc[1:]
    d["DD"] = -(lvl / lvl.rolling(HIGH_WINDOW, min_periods=1).max() - 1.0).shift(1)
    d["above_ma"] = (
        (lvl > lvl.rolling(MA, min_periods=MA).mean())
        .astype(float)
        .where(lvl.rolling(MA, min_periods=MA).mean().notna())
        .shift(1)
    )
    d["r"] = np.expm1(d["lr"])
    return d


SIGNALS = ("RV21", "RV5", "HAR", "VIX", "TS", "DD")


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p49 = _load(HERE / "49_cat_replay.py", "p49_cat_replay")
    p50 = _load(HERE / "50_cat_replay_repriced.py", "p50_cat_replay")
    p65 = _load(HERE / "65_capital_from_history.py", "p65_capital")
    d = panel_daily(p49)
    n = len(d)
    print(
        f"panel: {n} full sessions {d.index[0].date()}..{d.index[-1].date()}, "
        f"VIX from {d['VIX'].first_valid_index().date()}, VIX3M from {d['TS'].first_valid_index().date()}"
    )
    rep = p65.load_replay(p50)
    rep = rep.set_index("session")
    spot = rep["premium_pts"] / rep["premium_over_spot"]

    real = p65.load_realized(p43)
    real["m"] = p65.realized_brake(real, rep.reset_index())
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    spx = pd.Series(ch["S_close"], index=ch["dates"])
    gs = pd.read_parquet(GSPC)["close"].reindex(spx.index)
    dev = float((spx / gs - 1.0).abs().max())
    assert dev < GATE_CLOSE_TOL, dev
    both = d.index.intersection(spx.index)
    corr = float(np.corrcoef(d.loc[both, "lr"], d.loc[both, "panel_lr"])[0, 1])
    print(
        f"GATE  ^GSPC close = the chain's settlement close on {len(spx)} sessions (max rel diff {dev:.1e}); "
        f"for the record, the panel's summed bars against it: correlation {corr:.3f} on {len(both)} sessions"
    )

    pct = pd.DataFrame(
        {s: causal_pct(d[s].to_numpy(float)) for s in SIGNALS}, index=d.index
    )

    # ------------------------------------------------------------ events ----
    lr = d["lr"].to_numpy()
    e1 = lr <= np.quantile(lr, EVENT_Q)
    prior = (
        pd.Series(e1.astype(float))
        .rolling(FIRST_GAP, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
        .to_numpy()
    )
    e1f = e1 & (prior == 0)
    fwd = pd.Series(lr).rolling(RUN).sum().shift(-(RUN - 1)).to_numpy()
    e3 = np.isfinite(fwd) & (fwd <= np.nanquantile(fwd, EVENT_Q))
    loss2 = rep["loss_frac"].reindex(d.index).to_numpy(float)
    e2 = np.isfinite(loss2) & (loss2 >= np.nanquantile(loss2, 1 - EVENT_Q))
    events = {
        "E1 S&P crash day": (e1, -lr, np.ones(n, bool)),
        "E1f first shock": (e1f, -lr, np.ones(n, bool)),
        "E2 insurance crash day": (e2, loss2, np.isfinite(loss2)),
        "E3 20-session run start": (e3, -fwd, np.isfinite(fwd)),
    }
    print(
        f"events: E1 {int(e1.sum())} days (return <= {np.quantile(lr, EVENT_Q):+.2%}), first shocks {int(e1f.sum())}; "
        f"E2 {int(e2.sum())}; E3 {int(e3.sum())} window starts"
    )

    # ------------------------------------------------------------- A --------
    rows = []
    for ename, (ev, loss, dom) in events.items():
        for s in SIGNALS:
            p = pct[s].to_numpy()
            m = dom & np.isfinite(p)
            pp, ee, ll = p[m], ev[m], np.where(ev[m], loss[m], 0.0)
            if ee.sum() < 2:
                continue
            top = pp >= TOP
            idx = asl.circular_block_bootstrap_idx(
                np.random.default_rng([SEED, len(pp)]), len(pp), BLOCK, B
            )
            top_b, ll_b, ee_b = top[idx], ll[idx], ee[idx]
            good = ee_b.sum(axis=1) >= 1
            with np.errstate(invalid="ignore", divide="ignore"):
                lift_b = ((ll_b * top_b).sum(axis=1) / ll_b.sum(axis=1)) / top_b.mean(
                    axis=1
                )
            auc_b = auc_rows(pp[idx][good], ee_b[good])
            lift = float((ll * top).sum() / ll.sum() / top.mean())
            rows.append(
                {
                    "event": ename,
                    "signal": s,
                    "days": int(m.sum()),
                    "events": int(ee.sum()),
                    "AUC": float(auc_rows(pp[None, :], ee[None, :])[0]),
                    "AUC_lo": float(np.nanpercentile(auc_b, 2.5)),
                    "AUC_hi": float(np.nanpercentile(auc_b, 97.5)),
                    "top_share_of_days": float(top.mean()),
                    "top_share_of_events": float(top[ee].mean()),
                    "top_share_of_loss": float((ll * top).sum() / ll.sum()),
                    "lift": lift,
                    "lift_lo": float(np.nanpercentile(lift_b[good], 2.5)),
                    "lift_hi": float(np.nanpercentile(lift_b[good], 97.5)),
                    "FORECASTS": bool(
                        lift >= 2 and np.nanpercentile(lift_b[good], 2.5) > 1
                    ),
                }
            )
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "a_forecastability.csv", index=False)
    print(
        "\nA  forecastability: AUC of the causal percentile; LIFT = share of the event's loss in the top quintile / share of days there"
    )
    print(a.round(3).to_string(index=False))

    # episodes
    f = pd.Series(fwd, index=d.index)
    chosen: list[int] = []
    fv = f.to_numpy().copy()
    while len(chosen) < N_EPISODES and np.isfinite(fv).any():
        i = int(np.nanargmin(fv))
        chosen.append(i)
        fv[max(0, i - RUN + 1) : i + RUN] = np.nan
    ep = pd.DataFrame(
        [
            {
                "run_start": str(d.index[i].date()),
                "run_return": float(np.expm1(f.iloc[i])),
                "worst_day_in_run": str(
                    d.index[i + int(np.argmin(lr[i : i + RUN]))].date()
                ),
                **{f"pct_{s}_at_start": float(pct[s].iloc[i]) for s in SIGNALS},
                **{
                    f"pct_{s}_eve_of_worst": float(
                        pct[s].iloc[i + int(np.argmin(lr[i : i + RUN]))]
                    )
                    for s in ("RV21", "VIX")
                },
            }
            for i in chosen
        ]
    )
    ep.to_csv(OUT / "a_episodes.csv", index=False)
    print(
        f"\n   the {N_EPISODES} worst non-overlapping {RUN}-session S&P runs; causal percentile of each signal on the run's first session"
    )
    print(ep.round(2).to_string(index=False))

    # ------------------------------------------------------------- B --------
    big = np.abs(lr) >= np.quantile(np.abs(lr), 1 - BIG_MOVE_Q)
    rows_b = []
    for s in SIGNALS:
        p = pct[s].to_numpy()
        q = np.floor(np.clip(p, 0, 1 - 1e-12) * 5)
        for k in range(5):
            m = q == k
            rows_b.append(
                {
                    "signal": s,
                    "quintile": k + 1,
                    "days": int(m.sum()),
                    "big_moves": int((m & big).sum()),
                    "big_share_down": float((lr[m & big] < 0).mean())
                    if (m & big).any()
                    else np.nan,
                    "mean_return_annual": float(
                        d["r"].to_numpy()[m].mean() * asl.PERIODS_PER_YEAR
                    ),
                    "vol_annual": float(d["r"].to_numpy()[m].std(ddof=1) * ANN),
                }
            )
    bq = pd.DataFrame(rows_b)
    bq.to_csv(OUT / "b_size_not_sign.csv", index=False)
    print(
        f"\nB  by signal quintile: the largest {BIG_MOVE_Q:.0%} of moves and the share that are down; the session's mean return and vol"
    )
    print(bq.round(3).to_string(index=False))

    # ------------------------------------------------------------- C --------
    stress_frac = float(
        (real["stress_dollars"] / (real["S"] * p65.SPX_INDEX_MULTIPLIER)).median()
    )
    mu_real = float((real["net_pts"] / real["S"]).mean())
    pnl_frac = rep["pnl_pts"] / spot
    adj = (
        pnl_frac - pnl_frac.mean() + mu_real
    )  # the book's realized mean, the replay's shape
    cap = (
        p65.FRACTION * adj / stress_frac
    )  # one full-size day at 10%, fraction of capital
    rep_sig = d[list(SIGNALS)].reindex(rep.index)
    rep_sig["RV21"] = rep["rv21_trail"]
    m_rv = p65.brake(rep["rv21_trail"].to_numpy(float), p50)
    assert np.array_equal(m_rv, rep["m"].to_numpy()), (
        "RV21 brake differs from the replay's"
    )
    prem = rep["premium_over_spot"].to_numpy(float)
    rows_c = []
    for s in SIGNALS:
        x = rep_sig[s].to_numpy(float)
        dom = np.isfinite(x)
        xs = x[dom]
        m_s = np.full(len(x), np.nan)
        m_s[dom] = p65.brake(xs, p50)
        dom &= np.isfinite(m_s)
        # the signal's percentile needs MIN_OBS earlier values: score from there
        first = np.flatnonzero(
            np.isfinite(
                p50.expanding_pct(xs, p50.DELEVER_HALF_PCT, p50.DELEVER_MIN_OBS)
            )
        )
        start = np.flatnonzero(dom)[first[0]] if len(first) else len(x)
        scored = np.zeros(len(x), bool)
        scored[start:] = dom[start:]
        idx_s = np.flatnonzero(scored)
        halves = (
            ("first half", idx_s[: len(idx_s) // 2]),
            ("second half", idx_s[len(idx_s) // 2 :]),
            ("whole", idx_s),
        )
        for rule, mm in (("RV21 (live brake)", m_rv), (f"{s} brake", m_s)):
            if s == "RV21" and rule != "RV21 (live brake)":
                continue
            for hname, ii in halves:
                pnl = mm[ii] * cap.to_numpy()[ii]
                rows_c.append(
                    {
                        "sample_of": s,
                        "from": str(rep.index[ii[0]].date()),
                        "part": hname,
                        "rule": rule,
                        "sessions": len(ii),
                        "half": int((mm[ii] == 0.5).sum()),
                        "zero": int((mm[ii] == 0).sum()),
                        "premium_given_up": float(
                            ((1 - mm[ii]) * prem[ii]).sum() / prem[ii].sum()
                        ),
                        "worst_day": float(pnl.min()),
                        "one_in_200": float(np.percentile(pnl, 0.5)),
                        "worst_20": float(pd.Series(pnl).rolling(RUN).sum().min()),
                    }
                )
    c = pd.DataFrame(rows_c)
    c.to_csv(OUT / "c_brake_by_signal.csv", index=False)
    print(
        "\nC  the insurance brake driven by each signal, replay, 10% of capital per stress unit (fractions of capital)"
    )
    print(c.round(4).to_string(index=False))
    verdict_c = []
    for s in SIGNALS[1:]:
        cc = c[c["sample_of"] == s]
        ok_h = []
        for part in ("first half", "second half"):
            base = cc[(cc["part"] == part) & (cc["rule"] == "RV21 (live brake)")].iloc[
                0
            ]
            alt = cc[(cc["part"] == part) & (cc["rule"] == f"{s} brake")].iloc[0]
            ok_h.append(
                alt["one_in_200"] > base["one_in_200"]
                and alt["worst_20"] > base["worst_20"]
            )
        wb = cc[(cc["part"] == "whole") & (cc["rule"] == "RV21 (live brake)")].iloc[0]
        wa = cc[(cc["part"] == "whole") & (cc["rule"] == f"{s} brake")].iloc[0]
        verdict_c.append(
            {
                "signal": s,
                "tail_better_both_halves": all(ok_h),
                "no_more_premium": bool(
                    wa["premium_given_up"] <= wb["premium_given_up"]
                ),
                "REPLACES_RV21": bool(
                    all(ok_h) and wa["premium_given_up"] <= wb["premium_given_up"]
                ),
            }
        )
    print(pd.DataFrame(verdict_c).to_string(index=False))

    # C2 (added after C's replay result was seen, so a check, not a test): the
    # realized live book with the RV21 brake and with the RV5 brake.  As in
    # study 65, the panel's brake (history from 1998) through 2024-04-30 and
    # the live ledger's own after it (window 5 in place of 21).
    ledger = PremiumLedger.load(p65.SEED_LEDGER)
    m5 = (
        pd.Series(p65.brake(d["RV5"].to_numpy(float), p50), index=d.index)
        .reindex(real.index)
        .to_numpy(float)
    )
    for i in np.flatnonzero(~np.isfinite(m5)):
        m5[i] = ledger.delever_multiplier(real.index[i].date(), window=5)[0]
    c54r = pd.read_csv(p65.P54_DAILY, index_col=0, parse_dates=True).reindex(real.index)
    me_r = c54r["month_end"].fillna(False).astype(bool).to_numpy()
    rows_c2 = []
    for label, mm in (("RV21 brake (live)", real["m"].to_numpy()), ("RV5 brake", m5)):
        pts_r = np.where(
            me_r, c54r["pts_ask"].to_numpy(float), mm * real["net_pts"].to_numpy()
        )
        x = (
            p65.FRACTION
            * pts_r
            * p65.SPX_INDEX_MULTIPLIER
            / real["stress_dollars"].to_numpy()
        )
        for per, mask in (
            ("deck 2020-01..2024-04", real.index <= "2024-04-30"),
            ("HOLDOUT 2024-05..2025-12", real.index >= "2024-05-01"),
            ("all", np.ones(len(real), bool)),
        ):
            rows_c2.append(
                {
                    "brake": label,
                    "sample": per,
                    "half": int((mm[mask] == 0.5).sum()),
                    "zero": int((mm[mask] == 0).sum()),
                    **p65.evaluate(x[mask], real.index[mask]),
                }
            )
    c2 = pd.DataFrame(rows_c2)
    c2.to_csv(OUT / "c2_realized_book_by_brake.csv", index=False)
    print(
        "\nC2  the realized live book (net of 0.5 bp, month-end override, 10%) with each brake, fractions of capital"
    )
    print(c2.round(4).to_string(index=False))

    # ------------------------------------------------------------- D --------
    r = d["r"].to_numpy()
    sig_vol = np.sqrt(np.exp(d["HAR"].to_numpy()))
    med = pd.Series(sig_vol).expanding(min_periods=MIN_OBS).median().shift(1).to_numpy()
    w_vm = np.where(
        np.isfinite(med) & np.isfinite(sig_vol), np.minimum(1.0, med / sig_vol), 1.0
    )
    am = d["above_ma"].to_numpy(float)
    w_ma = np.where(np.isfinite(am), am, 1.0)
    w_brk = p65.brake(d["RV21"].to_numpy(float), p50)
    w_brk = np.where(np.isfinite(w_brk), w_brk, 1.0)
    ts = d["TS"].to_numpy()
    w_ts = np.where(np.isfinite(ts) & (ts > 1.0), 0.0, 1.0)
    rules = {
        "buy and hold": np.ones(n),
        "VM volatility-managed": w_vm,
        "MA200 trend": w_ma,
        "BRK brake on the S&P": w_brk,
        "TSI term-structure": w_ts,
    }
    mid = n // 2
    parts = {
        "whole": slice(0, n),
        f"to {d.index[mid - 1].date()}": slice(0, mid),
        f"from {d.index[mid].date()}": slice(mid, n),
    }
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, 67]), n, BLOCK, B
    )
    rb = r[idx]
    s_bh = rb.mean(axis=1) / rb.std(axis=1, ddof=1) * ANN
    rows_d = []
    for name, w in rules.items():
        x = w * r
        xb = w[idx] * rb
        dsh = xb.mean(axis=1) / xb.std(axis=1, ddof=1) * ANN - s_bh
        for pname, sl in parts.items():
            rows_d.append(
                {
                    "rule": name,
                    "part": pname,
                    "mean_weight": float(w[sl].mean()),
                    **summary(x[sl]),
                    **(
                        {
                            "dSharpe_lo": float(np.percentile(dsh, 2.5)),
                            "dSharpe_hi": float(np.percentile(dsh, 97.5)),
                        }
                        if pname == "whole"
                        else {}
                    ),
                }
            )
    dd_tab = pd.DataFrame(rows_d)
    dd_tab.to_csv(OUT / "d_sp_leg_rules.csv", index=False)
    print(
        "\nD  the S&P leg (index close-to-close, price only, cash at zero), fractions"
    )
    print(dd_tab.round(3).to_string(index=False))
    named = pd.DataFrame(
        {
            name: {
                lab: float(np.prod(1 + (w * r)[(d.index > a0) & (d.index <= b0)]) - 1)
                for lab, a0, b0 in DRAWDOWNS
            }
            for name, w in rules.items()
        }
    )
    named.to_csv(OUT / "d_named_drawdowns.csv")
    print(
        "\n   each rule's return inside the dated S&P drawdowns (peak close -> trough close)"
    )
    print(named.round(3).to_string())
    verdict_d = []
    for name in list(rules)[1:]:
        ok_p = []
        for pname in list(parts)[1:]:
            bh = dd_tab[
                (dd_tab["rule"] == "buy and hold") & (dd_tab["part"] == pname)
            ].iloc[0]
            ru = dd_tab[(dd_tab["rule"] == name) & (dd_tab["part"] == pname)].iloc[0]
            ok_p.append(
                ru["max_drawdown"] > bh["max_drawdown"]
                and ru["worst_20"] > bh["worst_20"]
            )
        hi = dd_tab[(dd_tab["rule"] == name) & (dd_tab["part"] == "whole")][
            "dSharpe_hi"
        ].iloc[0]
        verdict_d.append(
            {
                "rule": name,
                "smaller_drawdowns_both_halves": all(ok_p),
                "Sharpe_not_worse": bool(hi > 0),
                "MITIGATES": bool(all(ok_p) and hi > 0),
            }
        )
    print(pd.DataFrame(verdict_d).to_string(index=False))

    # joint book: S&P leg + insurance leg (10%, its brake)
    ins = (rep["m"] * cap).reindex(d.index).fillna(0.0).to_numpy()
    rows_j = []
    for name, w in rules.items():
        j = w * r + ins
        rows_j.append(
            {
                "book": f"{name} + insurance (replay tails)",
                "sample": "1998-2024",
                **summary(j),
            }
        )
    # realized 2020-01 .. 2024-04: the live book's own days
    c54 = pd.read_csv(p65.P54_DAILY, index_col=0, parse_dates=True).reindex(real.index)
    me = c54["month_end"].fillna(False).astype(bool).to_numpy()
    pts = np.where(
        me,
        c54["pts_ask"].to_numpy(float),
        real["m"].to_numpy() * real["net_pts"].to_numpy(),
    )
    live = pd.Series(
        p65.FRACTION
        * pts
        * p65.SPX_INDEX_MULTIPLIER
        / real["stress_dollars"].to_numpy(),
        index=real.index,
    )
    tape = d.index[d.index >= TAPE_START]
    live_t = live.reindex(tape).fillna(0.0).to_numpy()
    tm = d.index >= TAPE_START
    for name, w in rules.items():
        rows_j.append(
            {
                "book": f"{name} alone",
                "sample": "tape 2020-01..2024-04",
                **summary((w * r)[tm]),
            }
        )
        rows_j.append(
            {
                "book": f"{name} + insurance (realized)",
                "sample": "tape 2020-01..2024-04",
                **summary((w * r)[tm] + live_t),
            }
        )
    jt = pd.DataFrame(rows_j)
    jt.to_csv(OUT / "d_joint_book.csv", index=False)
    print(
        "\n   the JOINT book on one capital: S&P leg + insurance at 10% of capital per stress unit, brake on"
    )
    print(jt.round(3).to_string(index=False))

    # Is the replay leg a usable LEVEL for the joint book?  Where both exist,
    # the replayed insurance leg against the realized one, same sizing.
    ins_s = pd.Series(ins, index=d.index)
    live_s = pd.Series(live_t, index=tape)
    rows_k = []
    for lab, a0, b0 in (
        *[x for x in DRAWDOWNS if x[1] >= TAPE_START],
        ("tape 2020-01..2024-04", TAPE_START, str(d.index[-1].date())),
    ):
        mk = (
            (tape > a0) & (tape <= b0) if a0 != TAPE_START else np.ones(len(tape), bool)
        )
        a_, b_ = ins_s.reindex(tape).to_numpy()[mk], live_s.to_numpy()[mk]
        rows_k.append(
            {
                "window": lab,
                "replay_leg_return": float(np.prod(1 + a_) - 1),
                "realized_leg_return": float(np.prod(1 + b_) - 1),
                "replay_leg_max_drawdown": maxdd(a_),
                "realized_leg_max_drawdown": maxdd(b_),
            }
        )
    kt = pd.DataFrame(rows_k)
    kt.to_csv(OUT / "d_replay_vs_realized_leg.csv", index=False)
    print(
        "\n   CHECK the replayed insurance leg against the realized one where both exist (same sizing): the 1998-2024 "
        "joint rows are an upper bound on losses, not a level"
    )
    print(kt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
