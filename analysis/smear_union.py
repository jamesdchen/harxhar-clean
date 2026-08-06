"""§34.10 (repo record) + §34.10b/c: the span law's next tests on the second moment.

§34.10 verdict (scratchpad first-run): ridge over the union dictionary (10 named functionals +
the full daily panel snapshot) BEATS the ten names, DM +3.12 (2020+ +0.50) under the corrected
instrument — per-refit CAUSAL standardization (training mu/sd, z clip ±8), log-forecast clipped
to the training target range ±1, per-refit causal Duan factor, RidgeCV LOO-GCV α. This module
is the reproducible record of that run, plus two pre-registered extensions:

§34.10b — ERA DECOMPOSITION of union-vs-named (the +3.12 is historical; where does it live?).
    No gate; attribution. Loss-diff series saved for the family accounting.

§34.10d (ROLLING=1) — WINDOW/CADENCE for the union: the union's +3.12 is historical and its
    fits are EXPANDING-window at 63d cadence — but the mean channel's window law (§25: longer
    LOSES) and cadence law both say recency wins. Arms: union and named each refit on a
    trailing 756d window at 21d cadence, vs their expanding/63d versions (same instrument).
    Gates: rolling vs expanding DM >= +2.0 per dictionary; the registered target is the 2020+
    number (a revival there could change the production smear; full-span alone cannot).
    Recorded lean: rolling helps the union more than the names (543 nonstationary columns
    should age faster than 10 stationary functionals), but whether it clears the gate is
    genuinely uncertain.

§34.10c — UNION EXPANSION: does the span law convert previously-unconvertible dictionary
    blocks once they sit inside the union under GCV shrinkage?
    (c1) + quarticity/HARQ block (4 cols, §34.12's A block — uplift was real, conversion
         failed standalone). Recorded lean: genuinely uncertain — the span law predicts the
         earlier failure was dictionary construction, the §34.12b check says the content is
         redundant with the means. Gate vs union1: DM >= +2.0.
    (c2) + transmission daily aggregates (mean and mean-|.| of the 40 arrow forecasts per
         day, 80 cols) — the corrected-instrument retry of §37c's dead transmission-smear.
         Recorded lean: FAIL (37c died at −1.99, though under the old instrument). Gate vs
         union1: DM >= +2.0.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from sklearn.linear_model import RidgeCV  # noqa: E402

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t  # noqa: E402
from analysis.synthesis import _p  # noqa: E402

ALPHAS = np.logspace(0, 5, 21)


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    f = np.full(n, np.nan)
    f[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    e2 = (p.y - f) ** 2
    day = ts.dt.normalize()
    day_codes = pd.factorize(day)[0]
    late = (ts >= HOLDOUT).to_numpy()

    a_day = pd.Series(e2).groupby(day.values).mean()
    la = np.log(a_day + 1e-12)
    y = la.to_numpy()
    nd = len(y)
    r_day = pd.Series(p.X[:, p.names.index("adj_sumret_ma_1")].astype(np.float64)).groupby(day.values).sum()
    rs = (r_day - r_day.mean()) / (r_day.std() + 1e-12)
    l1s = la.shift(1)
    neg = rs.clip(upper=0)
    named = pd.DataFrame({
        "l1": l1s, "m5": l1s.rolling(5).mean(), "m21": l1s.rolling(21).mean(),
        "r": rs.shift(1), "absr": rs.abs().shift(1),
        "rneg": neg.shift(1), "r2": (rs ** 2).shift(1),
        "r_x_l1": rs.shift(1) * (l1s - l1s.mean()),
        "neg5": neg.rolling(5).mean().shift(1), "neg21": neg.rolling(21).mean().shift(1),
    })
    day_last = np.flatnonzero(pd.Series(day.values).ne(pd.Series(day.values).shift(-1)))
    Pd = pd.DataFrame(p.X[day_last], columns=p.names).shift(1)
    union1 = pd.concat([named.reset_index(drop=True), Pd.reset_index(drop=True)], axis=1)

    rolling_stage = bool(os.environ.get("ROLLING"))

    # §34.10e (ATTR=1): name the span — which union columns does the GCV ridge actually use?
    if os.environ.get("ATTR"):
        Xraw = union1.to_numpy(dtype=float)
        cols = list(union1.columns)
        coefs = []
        for start in range(3 * 252 + 63, nd, 252):
            tr = np.flatnonzero(np.isfinite(y) & (np.arange(nd) < start))
            if len(tr) < 400:
                continue
            mu = np.nanmean(Xraw[tr], 0)
            sd = np.nanstd(Xraw[tr], 0) + 1e-12
            Ztr = np.clip(np.nan_to_num((Xraw[tr] - mu) / sd), -8, 8)
            mdl = RidgeCV(alphas=ALPHAS).fit(Ztr, y[tr])
            coefs.append(mdl.coef_)
            print(f"  refit @{start}: alpha {mdl.alpha_:.0f}", flush=True)
        Cf = np.array(coefs)
        mean_abs = np.abs(Cf).mean(0)
        sign_con = np.abs(np.sign(Cf).mean(0))
        order = np.argsort(-mean_abs)[:20]
        print("\n  top-20 union columns by mean |coef| (sign consistency):", flush=True)
        for i in order:
            print(f"    {cols[i]:32s} {mean_abs[i]:.4f}  sign {sign_con[i]:.2f}", flush=True)
        named_share = np.abs(Cf[:, :10]).sum(1) / np.abs(Cf).sum(1)
        print(f"\n  named-10 share of total |coef|: {named_share.mean():.3f} "
              f"(last refit {named_share[-1]:.3f})", flush=True)
        return

    # §34.10c1: quarticity/HARQ block (dict-battery A block, causal shifts)
    def dmean(col):
        return pd.Series(p.X[:, p.names.index(col)].astype(np.float64)).groupby(day.values).mean()

    q4 = np.log1p(dmean("adj_sumret4_ma_1") - dmean("adj_sumret4_ma_1").min())
    bp = np.log1p(dmean("adj_sumbipow_ma_1") - dmean("adj_sumbipow_ma_1").min())
    jump = dmean("har_ma_1") - dmean("adj_sumbipow_ma_1")
    quart = pd.DataFrame({"q4": q4.shift(1), "bp": bp.shift(1), "jump": jump.shift(1),
                          "harq": np.sqrt(np.maximum(q4, 0)).shift(1) * (l1s - l1s.mean())})
    quart.index = range(nd)
    union_q = pd.concat([union1, quart], axis=1)

    # §34.10c2: transmission daily aggregates from the q=40 arrow forecasts
    from analysis.pool_width import _frame_q
    from analysis.trans_exploit import _trans_block
    F40 = _trans_block(_frame_q(p, 40), n, lag=1, refresh_days=63)
    Fd = pd.DataFrame(F40).groupby(day.values).mean()
    Fda = pd.DataFrame(np.abs(F40)).groupby(day.values).mean()
    trans_d = pd.concat([Fd.shift(1).reset_index(drop=True).add_prefix("g"),
                         Fda.shift(1).reset_index(drop=True).add_prefix("ag")], axis=1)
    union_t = pd.concat([union1, trans_d], axis=1)

    mfin = np.isfinite(y)

    def expanding_ahat(Xraw, ridge=True, window=None, cadence=63):
        out = np.full(nd, np.nan)
        for start in range(3 * 252 + 63, nd, cadence):
            lo = 0 if window is None else max(0, start - window)
            tr = np.flatnonzero(mfin & (np.arange(nd) >= lo) & (np.arange(nd) < start))
            if len(tr) < 400:
                continue
            seg = np.arange(start, min(start + cadence, nd))
            mu = np.nanmean(Xraw[tr], 0)
            sd = np.nanstd(Xraw[tr], 0) + 1e-12
            Ztr = np.clip(np.nan_to_num((Xraw[tr] - mu) / sd), -8, 8)
            Zte = np.clip(np.nan_to_num((Xraw[seg] - mu) / sd), -8, 8)
            if ridge:
                mdl = RidgeCV(alphas=ALPHAS).fit(Ztr, y[tr])  # LOO-GCV
                fit_tr, fit_te = mdl.predict(Ztr), mdl.predict(Zte)
            else:
                b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Ztr], y[tr], rcond=None)[0]
                fit_tr = np.c_[np.ones(len(tr)), Ztr] @ b
                fit_te = np.c_[np.ones(len(seg)), Zte] @ b
            fit_te = np.clip(fit_te, y[tr].min() - 1.0, y[tr].max() + 1.0)
            duan = np.mean(np.exp(np.clip(y[tr] - fit_tr, -10, 10)))
            out[seg] = np.exp(fit_te) * duan
        return out

    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    rel = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    ss = np.ones(n)
    for s in np.unique(slot):
        mm = slot == s
        cs = pd.Series(np.where(mm, rel, np.nan)).expanding(min_periods=100).mean().shift(48)
        ss[mm] = cs.to_numpy()[mm]
    ss = np.where(np.isfinite(ss), ss, 1.0)
    m0 = np.isfinite(f) & np.isfinite(p.y) & (p.baseline > 0)
    tr_raw = p.y ** 2 * p.baseline

    def q_of_daily(ahat):
        sm = day.map(dict(zip(la.index, ahat))).to_numpy(dtype=float) * ss
        ok = m0 & np.isfinite(sm) & (sm > 0)
        pr = (f ** 2 + sm) * p.baseline
        q = np.full(n, np.nan)
        okk = ok & (pr > 0) & (tr_raw > 0)
        r = tr_raw[okk] / pr[okk]
        q[okk] = r - np.log(r) - 1.0
        return q

    from analysis.armcache import memo

    if rolling_stage:
        qs = {}
        for name, df, ridge, win, cad in (
                ("named-exp", named, False, None, 63),
                ("named-roll", named, False, 756, 21),
                ("union-exp", union1, True, None, 63),
                ("union-roll", union1, True, 756, 21)):
            qs[name] = memo(f"su_roll_{name}",
                            lambda df=df, ridge=ridge, win=win, cad=cad:
                            q_of_daily(expanding_ahat(df.to_numpy(dtype=float), ridge=ridge,
                                                      window=win, cadence=cad)))
            print(f"  {name:12s} QLIKE {np.nanmean(qs[name]):.5f}", flush=True)
        for a, b, label in (("named-exp", "named-roll", "named: rolling vs expanding"),
                            ("union-exp", "union-roll", "union: rolling vs expanding"),
                            ("named-exp", "union-roll", "union-roll vs named-exp (production q)")):
            d = qs[a] - qs[b]
            md = np.isfinite(d)
            g = _hac_mean_t(d[md], 480)
            print(f"  {label:40s} DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480):+.2f})  "
                  f"{'PASS' if g >= 2.0 else 'FAIL'}", flush=True)
        return

    qs = {}
    for name, (df, ridge) in (("named", (named, False)), ("union1", (union1, True)),
                              ("union+quart", (union_q, True)), ("union+trans", (union_t, True))):
        qs[name] = memo(f"su_{name}",
                        lambda df=df, ridge=ridge:
                        q_of_daily(expanding_ahat(df.to_numpy(dtype=float), ridge=ridge)))
        print(f"  {name:12s} QLIKE {np.nanmean(qs[name]):.5f}", flush=True)

    def dm(a, b, label, gate=None):
        d = qs[a] - qs[b]
        md = np.isfinite(d)
        g = _hac_mean_t(d[md], 480)
        tag = "" if gate is None else f"  {'PASS' if g >= gate else 'FAIL'}"
        print(f"  {label:36s} DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480):+.2f}){tag}",
              flush=True)
        return d

    print("\n§34.10 record + gates:", flush=True)
    d_un = dm("named", "union1", "union1 vs named (the span verdict)")
    dm("union1", "union+quart", "c1 +quarticity vs union1", gate=2.0)
    dm("union1", "union+trans", "c2 +transmission vs union1", gate=2.0)

    print("\n§34.10b era decomposition of union1-vs-named (ΔQLIKE ×1e-4, positive = union wins):",
          flush=True)
    yr = ts.dt.year.to_numpy()
    for lo, hi in ((2008, 2011), (2012, 2015), (2016, 2019), (2020, 2021), (2022, 2026)):
        m = np.isfinite(d_un) & (yr >= lo) & (yr <= hi)
        if m.sum() > 1000:
            print(f"  {lo}-{hi}: {np.nanmean(d_un[m]) * 1e4:+.2f}  (t {_hac_mean_t(d_un[m], 480):+.2f})",
                  flush=True)
    np.savez_compressed(_p("smear_union_lossdiffs.npz"),
                        d_union_named=np.where(np.isfinite(d_un), d_un, np.nan),
                        **{f"q_{k.replace('+', '_')}": v for k, v in qs.items()})


if __name__ == "__main__":
    main()
