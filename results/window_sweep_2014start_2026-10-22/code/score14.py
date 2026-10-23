"""Part 2 of the 2014-start window sweep (read-only): out-of-sample accuracy 2014-2024 by window and the sign strategy.

Runs: the six windows' run directories (forecasts.parquet, scores.csv, frozen_penalties.json, scores_all_bars.csv),
evaluation from 2014-01-02.  Modules: the committed ones of the run commit (shadow tree: writeup/ src/ notebooks/
experiments/ from git archive; results/ data/ junctions to the main worktree for the untracked inputs).

(a) QLIKE 2014-01-02 .. 2024-04-30.  Days: rolling_eval_2026_10_15.history() days in the span (the trading days with a
    positive closing-half-hour print, early closes dropped -- the day rule of the score stage's 1,082 days) on which
    every run forecasts.  16:00: the closing interval's QLIKE of each day (pooled_linear_13.at16: the 15:30 forecast
    against the bar's RV).  Every bar: every interval of those days with RV > 0 (every forecast is finite and > 0,
    pooled_linear_13's check); the day's average loss.  Tests: src.evaluation.diebold_mariano.dm_test on the daily
    loss difference (automatic lag, HLN): W -> next W, the first W -> W (same arm), and full vs HAR (same W).
    The run's own 1,082-day closing-interval QLIKE (scores.csv) is reproduced to 1e-10 (pooled_linear_13's check).
(b) The sign strategy on the 866 option days (2020-2024, unchanged) exactly as make_rehaul_economics.add_pooled_linear
    (lin_16), on the 614 post-warmup days: D6 (make_rehaul_economics.sharpe) with hac_inference.d6_diff_test (block 10,
    LW_BOOT, D6_SEED), mid and bid-ask; D1 (sharpe_days_d1) with the Ledoit-Wolf p; the pooled 614-day ratio; LLSW.

usage: python score14.py <shadow tree> <out prefix> W=<run dir> [W=<run dir> ...]   (W as days, e.g. 1000=...)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
MAIN = Path(sys.argv[1])
for p in (MAIN, MAIN / "notebooks", MAIN / "writeup", MAIN / "experiments"):
    sys.path.insert(0, str(p))

import hac_inference as hi  # noqa: E402
import make_rehaul_economics as E  # noqa: E402
import pooled_linear_13 as pl13  # noqa: E402
import rolling_eval_2026_10_15 as RE  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

for m in (E, pl13, RE):
    assert Path(m.__file__).resolve().is_relative_to(MAIN.resolve()), m.__file__
OUTP = sys.argv[2]
RUNS = {int(a.split("=", 1)[0]): Path(a.split("=", 1)[1]) for a in sys.argv[3:]}
WS = sorted(RUNS)
EST = ("enet", "lasso", "ridge")
SPAN = (pd.Timestamp(os.environ.get("SCORE_FIRST", "2014-01-02")), pd.Timestamp("2024-04-30"))  # the sweep's first day
t0 = time.time()


def ann252(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0))


def set_run(run: Path) -> pd.DataFrame:
    """Point pooled_linear_13 at a 2014-start run: its bar count and walk start replace the 2012 run's constants."""
    f = pd.read_parquet(run / "forecasts.parquet", columns=["model", "predictor_set", "date", "plain", "rv_actual"])
    n = f.groupby(["model", "predictor_set"]).size()
    assert n.nunique() == 1, n.to_dict()
    t = pd.to_datetime(f["date"])
    pl13.RUN, pl13.FC = run, run / "forecasts.parquet"
    pl13.N_BARS_ARM, pl13.WALK_START, pl13.FIRST_DAY = int(n.iloc[0]), t.min(), SPAN[0]
    pl13._CACHE.clear()
    pl13._frame()  # pooled_linear_13's checks: arms, bar count, finite positive plain, penalties, scores.csv to 1e-10
    return f


mt, model_set, series0, fam0 = E.load()
days866 = series0[E.OLS].index
pw = E.pw_days()
assert len(days866) == 866 and len(pw) == 614
daily = pd.read_parquet(E.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"])
daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
ref = daily[daily["key"] == E.OLS].set_index("day").sort_index()
ref.index = ref.index.astype(days866.dtype)
ref = ref.reindex(days866)
iv, rv866 = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
hist = RE.history()
hdays = hist.index[(hist.index >= SPAN[0]) & (hist.index <= SPAN[1])]
print(f"loaded: {time.time() - t0:.0f} s; {len(hdays)} history days in {SPAN[0].date()} .. {SPAN[1].date()}", flush=True)

L16, LBAR, econ = {}, {}, []
for W in WS:
    run = RUNS[W]
    f = set_run(run)
    arms = [(e, s) for e in EST for s in pl13.SETS]
    for e, s in arms:
        a = pl13.at16(e, s)
        a = a[(a.index >= SPAN[0]) & (a.index <= SPAN[1]) & a.index.isin(hdays) & (a["rv"] > 0)]
        x = (a["rv"] / a["plain"]).to_numpy(float)
        L16[(W, e, s)] = pd.Series(x - np.log(x) - 1.0, index=a.index)
        g = f[(f["model"] == e) & (f["predictor_set"] == s)]
        tt = pd.to_datetime(g["date"])
        k = tt.dt.normalize().isin(hdays).to_numpy() & (g["rv_actual"].to_numpy(float) > 0)
        r = g["rv_actual"].to_numpy(float)[k] / g["plain"].to_numpy(float)[k]
        q = pd.Series(r - np.log(r) - 1.0, index=pd.DatetimeIndex(tt[k]))
        LBAR[(W, e, s)] = (q, q.groupby(q.index.normalize()).mean())
    # (b) the sign strategy
    series, fam = dict(series0), dict(fam0)
    E.LW_CACHE.clear()
    if hasattr(E, "BOOT_CACHE"):
        E.BOOT_CACHE.clear()
    E.add_pooled_linear(series, fam)
    for e in EST:
        h, fu = series[f"poolpt13_{e}_baseline"], series[f"poolpt13_{e}_all_features"]
        ph, pf = E.lin_16(e, "baseline", days866, rv866), E.lin_16(e, "all_features", days866, rv866)
        out = dict(W=W, model=e)
        for col, lab in (("mid", "mid"), ("x", "ba")):
            hp, fp_ = E.post_warmup(h[col]), E.post_warmup(fu[col])
            sh, sf = E.sharpe(h[col]), E.sharpe(fu[col])
            bt = hi.d6_diff_test(fp_, hp, pw)
            assert abs(bt["delta"] - (sf - sh)) < 1e-10
            ll = hi.llsw_test(fp_ - hp)
            out |= {f"d6_{lab}_har": sh, f"d6_{lab}_full": sf, f"dd6_{lab}": sf - sh, f"d6_boot_p_{lab}": bt["p"],
                    f"d1_{lab}_har": E.sharpe_days_d1(hp), f"d1_{lab}_full": E.sharpe_days_d1(fp_),
                    f"lw_p_{lab}": hi.lw_sharpe_test(fp_, hp, block=hi.LW_BLOCK)["p"],
                    f"llsw_t_{lab}": ll["t"], f"llsw_p_{lab}": ll["p"],
                    f"sr614_{lab}_har": ann252(hp), f"sr614_{lab}_full": ann252(fp_)}
            out[f"dd1_{lab}"] = out[f"d1_{lab}_full"] - out[f"d1_{lab}_har"]
            out[f"dsr614_{lab}"] = out[f"sr614_{lab}_full"] - out[f"sr614_{lab}_har"]
        for lab, g, p in (("har", h, ph), ("full", fu, pf)):
            out[f"long_share_{lab}"] = float((E.post_warmup(g["q"]) > 0).mean())
            out[f"mean_log_plain_rv_{lab}"] = float(np.log(p[E.WARMUP:] / rv866[E.WARMUP:]).mean())
        econ.append(out)
    print(f"W={W}: scored ({time.time() - t0:.0f} s)", flush=True)

# common days and intervals over every run and arm
d16 = None
for v in L16.values():
    d16 = v.index if d16 is None else d16.intersection(v.index)
for k in L16:
    L16[k] = L16[k].reindex(d16)
ib = None
for q, _ in LBAR.values():
    ib = q.index if ib is None else ib.intersection(q.index)
for k in LBAR:
    q = LBAR[k][0].reindex(ib)
    LBAR[k] = (q, q.groupby(q.index.normalize()).mean())
dbar = next(iter(LBAR.values()))[1].index
assert all(v[1].index.equals(dbar) for v in LBAR.values())
print(f"common: {len(d16)} days (16:00), {len(ib)} intervals on {len(dbar)} days (every bar)")


def dm(la: pd.Series, lb: pd.Series) -> dict:
    """b against a: dm_test(b, a) on the daily loss (negative = b lower)."""
    assert la.index.equals(lb.index)
    return dm_test(lb.to_numpy(), la.to_numpy())


rows = []
for e in EST:
    for s in pl13.SETS:
        for i, W in enumerate(WS):
            q16, qb = L16[(W, e, s)], LBAR[(W, e, s)]
            row = dict(model=e, set=s, W=W, n_days16=len(q16), qlike16=float(q16.mean()), n_bars=len(qb[0]),
                       qlike_bar=float(qb[0].mean()))
            if i:
                Wp = WS[i - 1]
                for lab, a_, b_ in (("16", L16[(Wp, e, s)], q16), ("bar", LBAR[(Wp, e, s)][1], qb[1])):
                    base = float(L16[(Wp, e, s)].mean()) if lab == "16" else float(LBAR[(Wp, e, s)][0].mean())
                    cur = row[f"qlike{'16' if lab == '16' else '_bar'}"]
                    r = dm(a_, b_)
                    row |= {f"step_pct_{lab}": 100 * (cur / base - 1), f"step_dm_{lab}": r["dm"], f"step_p_{lab}": r["p"]}
                for lab, a_, b_ in (("16", L16[(WS[0], e, s)], q16), ("bar", LBAR[(WS[0], e, s)][1], qb[1])):
                    base = float(L16[(WS[0], e, s)].mean()) if lab == "16" else float(LBAR[(WS[0], e, s)][0].mean())
                    cur = row[f"qlike{'16' if lab == '16' else '_bar'}"]
                    r = dm(a_, b_)
                    row |= {f"vs1_pct_{lab}": 100 * (cur / base - 1), f"vs1_dm_{lab}": r["dm"], f"vs1_p_{lab}": r["p"]}
            rows.append(row)
R = pd.DataFrame(rows)
gaps = []
for e in EST:
    for W in WS:
        g = dict(model=e, W=W)
        for lab, ah, af in (("16", L16[(W, e, "baseline")], L16[(W, e, "all_features")]),
                            ("bar", LBAR[(W, e, "baseline")][1], LBAR[(W, e, "all_features")][1])):
            mh = float(ah.mean()) if lab == "16" else float(LBAR[(W, e, "baseline")][0].mean())
            mf = float(af.mean()) if lab == "16" else float(LBAR[(W, e, "all_features")][0].mean())
            r = dm(ah, af)
            g |= {f"gap_pct_{lab}": 100 * (mf / mh - 1), f"gap_dm_{lab}": r["dm"], f"gap_p_{lab}": r["p"]}
        gaps.append(g)
G, EC = pd.DataFrame(gaps), pd.DataFrame(econ)
R.to_csv(f"{OUTP}_qlike.csv", index=False)
G.to_csv(f"{OUTP}_gap.csv", index=False)
EC.to_csv(f"{OUTP}_econ.csv", index=False)
LAB = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net"}
SL = {"baseline": "HAR", "all_features": "full"}
for e in EST:
    print(f"\n{LAB[e]}: QLIKE {SPAN[0].date()} .. {SPAN[1].date()} ({len(d16)} days; every bar {len(ib)} intervals)")
    print("| set | W | 16:00 QLIKE | Δ% step (DM p) | Δ% vs first W (DM p) | every-bar QLIKE | Δ% step (DM p) | "
          "Δ% vs first W (DM p) |")
    print("|---|---|---|---|---|---|---|---|")
    for s in pl13.SETS:
        for _, x in R[(R.model == e) & (R.set == s)].iterrows():
            c = lambda k, lab: "—" if np.isnan(x.get(f"{k}_pct_{lab}", np.nan)) else f"{x[f'{k}_pct_{lab}']:+.2f} ({x[f'{k}_p_{lab}']:.3f})"  # noqa: E731
            print(f"| {SL[s]} | {int(x.W):,} | {x.qlike16:.5f} | {c('step', '16')} | {c('vs1', '16')} | "
                  f"{x.qlike_bar:.5f} | {c('step', 'bar')} | {c('vs1', 'bar')} |")
    print("| full vs HAR | W | 16:00 gap % (DM p) | every-bar gap % (DM p) |")
    for _, x in G[G.model == e].iterrows():
        print(f"| gap | {int(x.W):,} | {x.gap_pct_16:+.2f} ({x.gap_p_16:.3f}) | {x.gap_pct_bar:+.2f} ({x.gap_p_bar:.3f}) |")
    print(f"\n{LAB[e]}: sign strategy, 614 post-warmup option days (2021-08 .. 2024-04)")
    print("| W | D6 HAR | D6 full | ΔD6 (boot p) | ΔD6 b/a (boot p) | ΔD1 (LW p) | ΔSR614 | LLSW t (p) | long HAR / full |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _, x in EC[EC.model == e].iterrows():
        print(f"| {int(x.W):,} | {x.d6_mid_har:.2f} | {x.d6_mid_full:.2f} | {x.dd6_mid:+.2f} ({x.d6_boot_p_mid:.3f}) | "
              f"{x.dd6_ba:+.2f} ({x.d6_boot_p_ba:.3f}) | {x.dd1_mid:+.2f} ({x.lw_p_mid:.3f}) | {x.dsr614_mid:+.2f} | "
              f"{x.llsw_t_mid:.2f} ({x.llsw_p_mid:.3f}) | {x.long_share_har:.3f} / {x.long_share_full:.3f} |")
print(f"done in {time.time() - t0:.0f} s")
