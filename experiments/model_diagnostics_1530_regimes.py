"""Regime changes, rolling SHAP, and the lasso's VIX path for the 15:30 forecast.

Reads the captures written by ``experiments/model_diagnostics_1530.py capture``
(the spec's own class, coefficients recorded at every refit):

  capture_bar1600.npz                  ridge, penalty re-chosen every 250 refits (arm of record)
  capture_bar1600_ridge_fixed100.npz   ridge, penalty held at 100    (capture ridge 100)
  capture_bar1600_ridge_fixed1000.npz  ridge, penalty held at 1000   (capture ridge 1000)
  capture_bar1600_reclasso.npz         the spec's recursive-lasso arm, same bucket/bar/window
                                       (capture reclasso)

1. Breaks.  Known-date Chow tests (Wald form, Newey-West covariance, chi2(p)) at the
   candidate dates and the unknown-date sup-Wald test of Andrews (1993), 15% trimming,
   critical values simulated from its limiting distribution.  Regressions: the 16:00
   target on the forecast (Mincer-Zarnowitz) and on the series contributions (all
   1,469 refits); the day's return per premium on a constant and on the contributions
   (866 deck days).  The same tests on the fixed-penalty counterfactuals separate
   breaks made by the penalty switching from breaks in the data.
2. Rolling SHAP.  For a linear model with the window mean as baseline the SHAP value
   of input j is beta_j (x_j - window mean_j): checked against the shap library.
   Series shares of mean |SHAP| over trailing 63- and 126-session windows.
3. The lasso: gate against the stored forecast, then which VIX-family columns carry a
   nonzero weight, when, with what sign and size, and whether they enter or leave at
   a penalty re-choice, a between-tune re-anchor, or in the warm path (the data).

Run:  python experiments/model_diagnostics_1530_regimes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "notebooks"))
import model_diagnostics_1530 as md  # noqa: E402

OUT = md.OUT
ET = md.ET
TUNE_PER = 250  # specs/causal_tune_linear.py: the penalty is re-chosen every 250 solves
TRIM = 0.15  # Andrews (1993) standard trimming
SIM_REPS = 20000
SIM_GRID = 2000
SEED = 20260924
ROLL_WINDOWS = (
    63,
    126,
)  # a quarter and a half-year of sessions (a choice; the second is the check)
N_TOP = 6  # series shown individually; the rest are summed
VIX_FAMILY = ("vix", "vvix", "vix3m")


# --------------------------------------------------------------------------- helpers
def load(tag: str | None) -> dict:
    f = OUT / ("capture_bar1600.npz" if tag is None else f"capture_bar1600_{tag}.npz")
    z = np.load(f, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["names"] = [str(v) for v in z["names"]]
    st = pd.DatetimeIndex(pd.to_datetime(np.asarray(z["date"]).astype(str)))
    assert st.tz is None and ((st.hour == 16) & (st.minute == 0)).all()
    out["stamp"] = st.as_unit("ns")
    out["day"] = st.normalize().as_unit("ns")
    th, X, mu = z["th"], z["x"], z["mu"]
    out["C"] = th[:, :-1] * (X - mu)
    out["level"] = th[:, -1] + np.einsum("nk,nk->n", th[:, :-1], mu)
    out["Z"] = th[:, :-1] * z["sd"]
    return out


def stored_16(name: str) -> pd.Series:
    tab = pd.read_parquet(REPO / "results" / "spxw_pnl" / name)
    t = pd.DatetimeIndex(tab["t"]).tz_convert(ET).tz_localize(None).as_unit("ns")
    m = (t.hour == 16) & (t.minute == 0)
    return pd.Series(tab["yhat"].to_numpy(float)[m], index=t[m])


def nw_lag(n: int) -> int:
    """Newey-West (1994) rule: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_cov(Z: np.ndarray, e: np.ndarray, L: int) -> np.ndarray:
    u = Z * e[:, None]
    S = u.T @ u
    for lag in range(1, L + 1):
        G = u[lag:].T @ u[:-lag]
        S += (1.0 - lag / (L + 1.0)) * (G + G.T)
    B = np.linalg.pinv(Z.T @ Z)
    return B @ S @ B


def wald_split(y: np.ndarray, X: np.ndarray, D: np.ndarray, L: int) -> float:
    """Wald statistic for H0: the coefficients on X are the same where D is True (HAC)."""
    Z = np.column_stack([X, X * D[:, None]])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    V = hac_cov(Z, e, L)
    p = X.shape[1]
    d, Vd = b[p:], V[p:, p:]
    return float(d @ np.linalg.solve(Vd, d))


def chow_predictive(
    y: np.ndarray, X: np.ndarray, D: np.ndarray
) -> tuple[float, int, int]:
    """Chow (1960) predictive test for a subsample D too short to fit on its own:
    F = ((RSS_all - RSS_out) / n_in) / (RSS_out / (n_out - p)) ~ F(n_in, n_out - p) under H0."""

    def rss(yy, XX):
        b, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        return float(((yy - XX @ b) ** 2).sum())

    n_in, n_out = int(D.sum()), int((~D).sum())
    Fs = ((rss(y, X) - rss(y[~D], X[~D])) / n_in) / (
        rss(y[~D], X[~D]) / (n_out - X.shape[1])
    )
    return Fs, n_in, n_out - X.shape[1]


def sup_wald_path(
    y: np.ndarray, X: np.ndarray, L: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    lo, hi = int(np.ceil(TRIM * n)), int(np.floor((1 - TRIM) * n))
    taus = np.arange(lo, hi + 1)
    W = np.array([wald_split(y, X, np.arange(n) >= k, L) for k in taus])
    return taus, W


def sim_supwald(p: int, rng: np.random.Generator) -> np.ndarray:
    """Draws of sup_{pi in [TRIM, 1-TRIM]} |B(pi) - pi B(1)|^2 / (pi (1-pi)), B a p-dim Brownian motion."""
    pis = np.arange(1, SIM_GRID + 1) / SIM_GRID
    keep = (pis >= TRIM) & (pis <= 1 - TRIM)
    out = np.empty(SIM_REPS)
    chunk = max(1, int(1e7 // (SIM_GRID * p)))
    for a in range(0, SIM_REPS, chunk):
        m = min(chunk, SIM_REPS - a)
        Bm = np.cumsum(rng.standard_normal((m, SIM_GRID, p)), axis=1) / np.sqrt(
            SIM_GRID
        )
        br = Bm[:, keep, :] - pis[keep][None, :, None] * Bm[:, -1:, :]
        q = (br**2).sum(-1) / (pis[keep] * (1 - pis[keep]))[None, :]
        out[a : a + m] = q.max(1)
    return out


# --------------------------------------------------------------------------- main
def main() -> None:  # noqa: C901 - one linear report
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import atm_straddle_lib as asl  # type: ignore

    lines: list[str] = []
    J: dict = {}

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    R = load(None)
    F = {100.0: load("ridge_fixed100"), 1000.0: load("ridge_fixed1000")}
    LA = load("reclasso")
    names = R["names"]
    n = len(R["pred"])
    for o in (*F.values(), LA):
        assert o["names"] == names and (o["stamp"] == R["stamp"]).all()

    # ---- GATES
    g = {}
    for arm, o, fname in (
        ("ridge", R, "yhat_sub_ridge_live_feasible.parquet"),
        ("lasso", LA, "yhat_sub_lasso_live_feasible.parquet"),
    ):
        s = stored_16(fname)
        c = pd.Series(o["pred"], index=o["stamp"])
        common = s.index.intersection(c.index)
        rel = np.abs(c.loc[common].to_numpy() - s.loc[common].to_numpy()) / np.abs(
            s.loc[common].to_numpy()
        )
        rec = np.einsum("nk,nk->n", o["th"][:, :-1], o["x"]) + o["th"][:, -1]
        g[arm] = dict(
            stored=len(s),
            common=len(common),
            maxrel=float(rel.max()),
            recon=float(np.max(np.abs(rec - o["pred"]) / np.abs(o["pred"]))),
        )
        say(
            f"GATE {arm}: stored 16:00 rows {len(s)}, captured {len(c)}, common {len(common)}; max rel diff "
            f"{rel.max():.1e}; coef.x + intercept vs prediction {g[arm]['recon']:.1e}"
        )
        if len(common) != len(s) or rel.max() > 1e-6:
            raise SystemExit(f"GATE {arm} FAILED -- not interpreted")
    # the fixed-penalty runs equal the arm of record wherever its penalty is the same
    for a, o in F.items():
        m = R["alpha"] == a
        rel = np.abs(o["pred"][m] - R["pred"][m]) / np.abs(R["pred"][m])
        g[f"fixed{a:g}"] = dict(n_same=int(m.sum()), maxrel=float(rel.max()))
        say(
            f"GATE fixed penalty {a:g}: equals the arm of record on its {int(m.sum())} sessions at that "
            f"penalty to {rel.max():.1e}; differs elsewhere (max rel {np.max(np.abs(o['pred'][~m] - R['pred'][~m]) / np.abs(R['pred'][~m])):.2f})"
        )
        assert rel.max() < 1e-9
    J["gates"] = g

    # ---- the calendar: penalty switches, COVID, every-day expirations
    # the arm of record's capture predates the solve counter; the fixed-penalty runs share its refits
    assert (F[100.0]["n_solve"] == F[1000.0]["n_solve"]).all() and (
        F[100.0]["n_solve"] == np.arange(1, n + 1)
    ).all()
    R["n_solve"] = F[1000.0]["n_solve"]
    tune = (R["n_solve"] % TUNE_PER) == 1
    sw = np.flatnonzero(np.diff(R["alpha"]) != 0) + 1
    assert tune[sw].all(), "penalty changes only at a tune"
    d = pd.read_parquet(
        REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
    )
    d.index = pd.DatetimeIndex(d.index).as_unit("ns")
    me = pd.read_csv(
        REPO / "results" / "atm_straddle_0dte_1530" / "monthend_1530_longstraddle.csv",
        index_col=0,
        parse_dates=True,
    )["month_end"]
    me.index = pd.DatetimeIndex(me.index).as_unit("ns")
    d["month_end"] = me.reindex(d.index).astype(bool)
    d["pnl_mid"] = d["pos"] * d["R"]
    bid = d["bid_c"].astype(float) + d["bid_p"].astype(float)
    ask = d["ask_c"].astype(float) + d["ask_p"].astype(float)
    d["pnl_x"] = asl.crossed_premium_return(d["pos"], d["exit"], bid, ask)
    assert asl.crossed_untradeable_count(d["pos"], bid, ask) == 0
    cal = R["day"]
    pre = d.index[d.index < pd.Timestamp("2020-03-01")]
    peak = (
        d.loc[pre, "S"].astype(float).idxmax()
    )  # the S&P high before the crash (on the deck)
    covid_lo = cal[cal > peak][0]
    covid_hi = cal[cal < pd.Timestamp("2020-05-01")][
        -1
    ]  # "Feb-Apr 2020": through the end of April
    after_covid = cal[cal > covid_hi][0]
    in_deck = cal.isin(d.index)
    tue = cal[(cal.dayofweek == 1) & (cal >= d.index.min()) & (cal <= d.index.max())]
    thu = cal[(cal.dayofweek == 3) & (cal >= d.index.min()) & (cal <= d.index.max())]
    tue_bad, thu_bad = tue[~tue.isin(d.index)], thu[~thu.isin(d.index)]
    everyday = tue[tue > tue_bad.max()][0]
    thu_start = thu[thu > thu_bad.max()][0]
    first_tt = d.index[d.index.dayofweek.isin([1, 3])][0]
    cand = [
        (f"penalty {R['alpha'][k - 1]:g} to {R['alpha'][k]:g}", cal[k], "penalty")
        for k in sw
    ]
    cand += [
        ("COVID crash start", covid_lo, "data"),
        ("COVID crash end", after_covid, "data"),
        ("every-day expirations", everyday, "data"),
    ]
    cand = sorted(cand, key=lambda r: r[1])
    say("candidate dates: " + "; ".join(f"{a} {b.date()}" for a, b, _ in cand))
    say(
        f"COVID episode: {covid_lo.date()} .. {covid_hi.date()} (S&P high on the deck {peak.date()}); "
        f"first Tuesday/Thursday deck day {first_tt.date()} (a holiday-shifted expiry); every Tuesday session "
        f"is a deck day from {everyday.date()}, every Thursday from {thu_start.date()}"
    )
    J["dates"] = dict(
        cand=[(a, str(b.date()), k) for a, b, k in cand],
        covid=[str(covid_lo.date()), str(covid_hi.date())],
        peak=str(peak.date()),
        first_tt=str(first_tt.date()),
        everyday=str(everyday.date()),
        thu_start=str(thu_start.date()),
        switches=[str(cal[k].date()) for k in sw],
        alphas=[float(R["alpha"][k]) for k in sw],
        alpha0=float(R["alpha"][0]),
    )

    # ---- series groups (the source series of Figure 1)
    active = np.flatnonzero((R["sd"] > 0).any(0) & (np.abs(R["th"][:, :-1]) > 0).any(0))
    grp = np.array([md.parse_feature(x)[0] for x in names])
    gset = sorted(set(grp[active]))
    Gc = {
        k: {g_: o["C"][:, grp == g_].sum(1) for g_ in gset}
        for k, o in (("actual", R), (100.0, F[100.0]), (1000.0, F[1000.0]))
    }
    order = sorted(gset, key=lambda g_: -Gc["actual"][g_].std(ddof=1))
    top = order[:N_TOP]
    rest = order[N_TOP:]
    stem = {g_: md.var_stem(g_, names) for g_ in gset}
    say(
        "series by contribution sd on all refits: "
        + ", ".join(f"{stem[g_]} {Gc['actual'][g_].std(ddof=1):.3f}" for g_ in order)
    )

    def contrib_design(key):
        cols = [Gc[key][g_] for g_ in top] + [
            np.sum([Gc[key][g_] for g_ in rest], axis=0)
        ]
        return np.column_stack([np.ones(n)] + cols)

    # ---- 1(a) weights: how much of each switch-day step is the penalty
    topcols = [
        names[j] for j in sorted(active, key=lambda j: -R["C"][:, j].std(ddof=1))[:8]
    ]
    vix_top = [c for c in names if c.startswith("adj_vix_ma_")]
    vix_top = sorted(
        [c for c in vix_top if names.index(c) in active],
        key=lambda c: -R["C"][:, names.index(c)].std(ddof=1),
    )[:1]
    show_cols = topcols[:7] + [c for c in vix_top if c not in topcols[:7]]
    srows = []
    dZ = np.diff(R["Z"], axis=0)
    for k in sw:
        a_new, a_old = R["alpha"][k], R["alpha"][k - 1]
        for c in show_cols + ["ALL"]:
            if c == "ALL":
                act = np.linalg.norm(R["Z"][k] - R["Z"][k - 1])
                pen = np.linalg.norm(F[a_new]["Z"][k] - F[a_old]["Z"][k])
                dat = np.linalg.norm(F[a_old]["Z"][k] - F[a_old]["Z"][k - 1])
                other = np.linalg.norm(dZ, axis=1)
            else:
                j = names.index(c)
                act = R["Z"][k, j] - R["Z"][k - 1, j]
                pen = (
                    F[a_new]["Z"][k, j] - F[a_old]["Z"][k, j]
                )  # same data, the two penalties
                dat = (
                    F[a_old]["Z"][k, j] - F[a_old]["Z"][k - 1, j]
                )  # same penalty, one more day of data
                other = np.abs(dZ[:, j])
            other = np.delete(other, sw - 1)
            srows.append(
                dict(
                    date=str(cal[k].date()),
                    switch=f"{a_old:g} to {a_new:g}",
                    column=c,
                    step=act,
                    penalty_part=pen,
                    data_part=dat,
                    max_abs_step_other_days=other.max(),
                    median_abs_step_other_days=np.median(other),
                )
            )
    ST = pd.DataFrame(srows)
    ST.to_csv(OUT / "regimes_weight_steps_at_penalty_switches.csv", index=False)
    for k in sw:
        r = ST[(ST.date == str(cal[k].date())) & (ST.column == "ALL")].iloc[0]
        say(
            f"(1a) {r.date} penalty {r.switch}: all-weight step |dZ| {r.step:.4f} = penalty {r.penalty_part:.4f} "
            f"(+ one day of data {r.data_part:.4f}); largest step on any other day {r.max_abs_step_other_days:.4f}, "
            f"median {r.median_abs_step_other_days:.5f}"
        )
    J["steps"] = ST.to_dict("records")

    # ---- 1 break tests
    Lr, Ld = nw_lag(n), nw_lag(len(d))
    ii = pd.Series(np.arange(n), index=cal).reindex(d.index).to_numpy(int)
    y_t = R["true_adj"]
    regs = []  # (id, label, sample, y, X, penalty-version)
    for key in ("actual", 100.0, 1000.0):
        pr = R["pred"] if key == "actual" else F[float(key)]["pred"]
        regs.append(
            (
                f"A0_{key}",
                "target on forecast",
                "refits",
                y_t,
                np.column_stack([np.ones(n), pr]),
                key,
            )
        )
        regs.append(
            (
                f"A1_{key}",
                "target on series contributions",
                "refits",
                y_t,
                contrib_design(key),
                key,
            )
        )
    pm, px = d["pnl_mid"].to_numpy(float), d["pnl_x"].to_numpy(float)
    one = np.ones((len(d), 1))
    regs.append(("B0_mid", "mean return per premium (mid)", "deck", pm, one, "actual"))
    regs.append(
        ("B0_crossed", "mean return per premium (crossed)", "deck", px, one, "actual")
    )
    for key in ("actual", 1000.0):
        regs.append(
            (
                f"B1_{key}",
                "return (mid) on series contributions",
                "deck",
                pm,
                contrib_design(key)[ii],
                key,
            )
        )
    rng = np.random.default_rng(SEED)
    crit: dict = {}
    for p in sorted({r[4].shape[1] for r in regs}):
        crit[p] = sim_supwald(p, rng)
        say(
            f"sup-Wald limiting distribution, p={p}, trim {TRIM}: 90/95/99% "
            + "/".join(f"{np.quantile(crit[p], q):.2f}" for q in (0.90, 0.95, 0.99))
            + f" ({SIM_REPS} draws, {SIM_GRID}-step grid)"
        )
    say("  (published p=1, 15% trim, 5%: 8.58 Bai-Perron 2003, 8.85 Andrews 2003)")
    krows, srows2, paths = [], [], {}
    for rid, label, sample, y, X, key in regs:
        idx = cal if sample == "refits" else d.index
        L = Lr if sample == "refits" else Ld
        p = X.shape[1]
        for cname, cdate, kind in cand:
            if cdate <= idx[0] or cdate > idx[-1]:
                krows.append(
                    dict(
                        reg=rid,
                        label=label,
                        sample=sample,
                        penalty=str(key),
                        date=str(cdate.date()),
                        candidate=cname,
                        kind=kind,
                        n=len(y),
                        p=p,
                        wald=np.nan,
                        pval=np.nan,
                        n_after=0,
                        test="before the sample",
                    )
                )
                continue
            D = np.asarray(idx >= cdate)
            short = (
                min(D.mean(), 1 - D.mean()) < TRIM
            )  # the sup-Wald trimming: each side >= 15%
            W = np.nan if short else wald_split(y, X, D, L)
            krows.append(
                dict(
                    reg=rid,
                    label=label,
                    sample=sample,
                    penalty=str(key),
                    date=str(cdate.date()),
                    candidate=cname,
                    kind=kind,
                    n=len(y),
                    p=p,
                    wald=W,
                    pval=np.nan if short else stats.chi2.sf(W, p),
                    n_after=int(D.sum()),
                    test="one side < 15% of the sample" if short else "Chow (HAC Wald)",
                )
            )
        # the COVID episode: coefficients inside Feb-Apr 2020 vs outside
        D = np.asarray((idx >= covid_lo) & (idx <= covid_hi))
        Fs, d1, d2 = chow_predictive(y, X, D)
        krows.append(
            dict(
                reg=rid,
                label=label,
                sample=sample,
                penalty=str(key),
                date=f"{covid_lo.date()}..{covid_hi.date()}",
                candidate="COVID episode (inside vs outside)",
                kind="data",
                n=len(y),
                p=p,
                wald=Fs,
                pval=stats.f.sf(Fs, d1, d2),
                n_after=int(D.sum()),
                test=f"Chow predictive F({d1}, {d2})",
            )
        )
        taus, Wp = sup_wald_path(y, X, L)
        paths[rid] = (idx[taus], Wp)
        kmax = int(np.argmax(Wp))
        srows2.append(
            dict(
                reg=rid,
                label=label,
                sample=sample,
                penalty=str(key),
                n=len(y),
                p=p,
                hac_lag=L,
                sup_wald=Wp[kmax],
                date_max=str(idx[taus[kmax]].date()),
                range=f"{idx[taus[0]].date()}..{idx[taus[-1]].date()}",
                crit95=float(np.quantile(crit[p], 0.95)),
                pval=float((crit[p] >= Wp[kmax]).mean()),
            )
        )
    KB = pd.DataFrame(krows)
    SB = pd.DataFrame(srows2)
    KB.to_csv(OUT / "regimes_breaks_known_dates.csv", index=False)
    SB.to_csv(OUT / "regimes_breaks_supwald.csv", index=False)
    pd.set_option("display.width", 250)
    say(f"HAC lags (Newey-West 1994 rule): refits {Lr}, deck days {Ld}")
    for _, r in SB.iterrows():
        say(
            f"(1) sup-Wald {r.reg:12s} {r.label:40s} n={r.n} p={r.p}: {r.sup_wald:7.2f} at {r.date_max} "
            f"(5% crit {r.crit95:.2f}, p {r.pval:.3f}); dates tested {r.range}"
        )
    for _, r in KB.iterrows():
        say(
            f"(1) {r.reg:12s} {r.candidate:34s} {r.date:22s}: stat={r.wald:7.2f} p={r.pval:.3g}  [{r.test}; n={r.n}, p={r.p}]"
        )
    J["sup"] = SB.to_dict("records")
    J["known"] = KB.to_dict("records")

    # penalty-driven vs data: a switch-date break in the arm of record that the fixed-penalty runs do not show
    verdict = []
    for base in ("A0", "A1"):
        for cname, cdate in [(c[0], str(c[1].date())) for c in cand] + [
            (
                "COVID episode (inside vs outside)",
                f"{covid_lo.date()}..{covid_hi.date()}",
            )
        ]:
            ps = {
                key: KB[(KB.reg == f"{base}_{key}") & (KB.date == cdate)].pval.iloc[0]
                for key in ("actual", "100.0", "1000.0")
            }
            rej = {k: bool(v < 0.05) for k, v in ps.items()}
            v = (
                "data (both held penalties reject)"
                if rej["100.0"] and rej["1000.0"]
                else "none"
                if not any(rej.values())
                else "penalty (only the arm of record rejects)"
                if not (rej["100.0"] or rej["1000.0"])
                else "depends on the penalty held"
            )
            verdict.append(
                dict(
                    reg=base,
                    candidate=cname,
                    date=cdate,
                    p_actual=ps["actual"],
                    p_fixed100=ps["100.0"],
                    p_fixed1000=ps["1000.0"],
                    verdict=v,
                )
            )
    VD = pd.DataFrame(verdict)
    VD.to_csv(OUT / "regimes_breaks_penalty_vs_data.csv", index=False)
    say(VD.round(4).to_string())

    # ---- 2 rolling SHAP
    # (i) the identity: the shap library's LinearExplainer (window mean as the one background row,
    # interventional) against beta_j (x_j - window mean_j), on the refits at the candidate dates
    import shap

    samp = sorted(
        {0, n - 1, *[int(np.searchsorted(cal, c[1])) for c in cand if c[1] <= cal[-1]]}
    )
    dmax, smax = 0.0, 0.0
    for k in samp:
        coef, b0 = R["th"][k, :-1], R["th"][k, -1]
        ex = shap.LinearExplainer(
            (coef, b0), shap.maskers.Independent(R["mu"][k][None, :], max_samples=1)
        )
        sv = np.asarray(ex.shap_values(R["x"][k][None, :]))[0]
        dmax = max(dmax, float(np.max(np.abs(sv - R["C"][k]))))
        smax = max(
            smax,
            abs(float(sv.sum() + ex.expected_value - R["pred"][k])) / abs(R["pred"][k]),
        )
    say(
        f"(2) SHAP check on {len(samp)} refits: max |shap - beta (x - window mean)| {dmax:.1e}; "
        f"shap sum + baseline vs forecast {smax:.1e} (relative)"
    )
    J["shap"] = dict(n=len(samp), dmax=dmax, smax=smax)
    Aabs = pd.DataFrame({g_: np.abs(Gc["actual"][g_]) for g_ in gset}, index=cal)
    shares = {}
    for w in ROLL_WINDOWS:
        rm = Aabs.rolling(w, min_periods=w).mean()
        shares[w] = rm.div(rm.sum(1), axis=0)
        shares[w].to_csv(OUT / f"regimes_shap_share_rolling{w}.csv")
    both = shares[ROLL_WINDOWS[1]].notna().all(1) & shares[ROLL_WINDOWS[0]].notna().all(
        1
    )
    agree = shares[ROLL_WINDOWS[0]][both].idxmax(1) == shares[ROLL_WINDOWS[1]][
        both
    ].idxmax(1)
    dom = {w: shares[w].dropna().idxmax(1).value_counts() for w in ROLL_WINDOWS}
    for w in ROLL_WINDOWS:
        say(
            f"(2) rolling {w}: dominant series share of sessions: "
            + ", ".join(
                f"{stem[g_]} {c / dom[w].sum():.0%}" for g_, c in dom[w].items()
            )
        )
    say(
        f"(2) the {ROLL_WINDOWS[0]}- and {ROLL_WINDOWS[1]}-session windows name the same dominant series on {agree.mean():.1%} of {len(agree)} sessions"
    )
    # regime table: mean |SHAP| share inside each regime (no rolling)
    edges = (
        [cal[0]]
        + [c[1] for c in cand if c[1] > cal[0]]
        + [cal[-1] + pd.Timedelta(days=1)]
    )
    edges = sorted(set(edges))
    # the COVID episode ends at covid_hi (the next edge is after_covid, already in cand)
    rrows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (cal >= a) & (cal < b)
        sh = Aabs[m].mean()
        sh = sh / sh.sum()
        o = sh.sort_values(ascending=False)
        rrows.append(
            dict(
                start=str(a.date()),
                end=str(cal[m][-1].date()),
                sessions=int(m.sum()),
                deck_days=int(np.asarray(in_deck)[m].sum()),
                penalty="/".join(f"{v:g}" for v in sorted(set(R["alpha"][m]))),
                **{f"top{i + 1}": stem[o.index[i]] for i in range(3)},
                **{f"share{i + 1}": o.iloc[i] for i in range(3)},
                **{f"s_{g_}": sh[g_] for g_ in order},
            )
        )
        A1k = pd.DataFrame({g_: np.abs(Gc[1000.0][g_]) for g_ in gset}, index=cal)[
            m
        ].mean()
        A1k = (A1k / A1k.sum()).sort_values(ascending=False)
        rrows[-1].update({f"held1000_top{i + 1}": stem[A1k.index[i]] for i in range(3)})
        rrows[-1].update({f"held1000_share{i + 1}": A1k.iloc[i] for i in range(3)})
    RT = pd.DataFrame(rrows)
    RT.to_csv(OUT / "regimes_shap_share_by_regime.csv", index=False)
    say(
        RT[
            [
                "start",
                "end",
                "sessions",
                "deck_days",
                "penalty",
                "top1",
                "share1",
                "top2",
                "share2",
                "top3",
                "share3",
            ]
        ]
        .round(3)
        .to_string()
    )
    say(
        "(2) penalty held at 1000: "
        + RT[
            [
                "start",
                "held1000_top1",
                "held1000_share1",
                "held1000_top2",
                "held1000_share2",
                "held1000_top3",
                "held1000_share3",
            ]
        ]
        .round(3)
        .to_string()
    )
    evc = None
    J["regimes"] = RT.to_dict("records")

    # ---- 3 the lasso and the VIX family
    tuneL = (LA["n_solve"] % TUNE_PER) == 1
    swL = np.flatnonzero(np.diff(LA["alpha"]) != 0) + 1
    assert tuneL[swL].all()
    reseedL = np.r_[False, np.diff(LA["n_reseed"]) > 0]
    say(
        f"(3) lasso penalty: {LA['alpha'][0]:g} from {cal[0].date()}"
        + "".join(f", {LA['alpha'][k]:g} from {cal[k].date()}" for k in swL)
        + f"; between-tune re-anchors on {', '.join(str(c.date()) for c in cal[reseedL])}; "
        f"nonzero weights per refit min {int((LA['th'][:, :-1] != 0).sum(1).min())}, median "
        f"{int(np.median((LA['th'][:, :-1] != 0).sum(1)))}, max {int((LA['th'][:, :-1] != 0).sum(1).max())}"
    )
    J["lasso"] = dict(
        alpha0=float(LA["alpha"][0]),
        switches=[(str(cal[k].date()), float(LA["alpha"][k])) for k in swL],
        reseeds=[str(c.date()) for c in cal[reseedL]],
        nnz=[
            int((LA["th"][:, :-1] != 0).sum(1).min()),
            int(np.median((LA["th"][:, :-1] != 0).sum(1))),
            int((LA["th"][:, :-1] != 0).sum(1).max()),
        ],
    )
    vcols = [c for c in names if md.parse_feature(c)[0] in VIX_FAMILY]
    nzL = LA["th"][:, :-1] != 0
    deck_m = np.asarray(in_deck)
    vrows, erows = [], []
    for c in vcols:
        j = names.index(c)
        nz = nzL[:, j]
        z_ = LA["Z"][:, j]
        ev = np.flatnonzero(np.diff(nz.astype(int)) != 0) + 1
        for k in ev:
            why = (
                ("penalty re-chosen" if k in swL else "tune (same penalty)")
                if tuneL[k]
                else ("between-tune re-anchor" if reseedL[k] else "warm update (data)")
            )
            near = min(cand, key=lambda r: abs((cal[k] - r[1]).days))
            erows.append(
                dict(
                    column=c,
                    date=str(cal[k].date()),
                    event="enters" if nz[k] else "leaves",
                    why=why,
                    nearest_candidate=near[0],
                    candidate_date=str(near[1].date()),
                    sessions_from_candidate=int(
                        np.searchsorted(cal, cal[k]) - np.searchsorted(cal, near[1])
                    ),
                    std_weight_after=z_[k],
                )
            )
        vrows.append(
            dict(
                column=c,
                sessions_nonzero=int(nz.sum()),
                deck_days_nonzero=int((nz & deck_m).sum()),
                frac_positive_when_nonzero=float((z_[nz] > 0).mean())
                if nz.any()
                else np.nan,
                mean_std_weight_when_nonzero=float(z_[nz].mean())
                if nz.any()
                else np.nan,
                max_abs_std_weight=float(np.abs(z_).max()),
                first_nonzero=str(cal[nz][0].date()) if nz.any() else "",
                last_nonzero=str(cal[nz][-1].date()) if nz.any() else "",
                n_events=len(ev),
                ridge_mean_std_weight_deck=float(R["Z"][deck_m, j].mean()),
            )
        )
    # the lasso with its penalty held at the post-2021-06-21 value from the start: was VIX in the data before?
    lo_a = float(LA["alpha"][-1])
    LF = load(f"reclasso_fixed{lo_a:g}")
    assert LF["names"] == names and (LF["stamp"] == R["stamp"]).all()
    msame = LA["alpha"] == lo_a
    relf = np.max(
        np.abs(LF["pred"][msame] - LA["pred"][msame]) / np.abs(LA["pred"][msame])
    )
    say(
        f"GATE lasso with the penalty held at {lo_a:g}: equals the lasso of record on its {int(msame.sum())} sessions "
        f"at that penalty to {relf:.1e}"
    )
    assert relf < 1e-9
    J["gates"]["lasso_fixed"] = dict(
        alpha=lo_a, n_same=int(msame.sum()), maxrel=float(relf)
    )
    nzF = LF["th"][:, :-1] != 0
    pre_m = ~msame
    for r in vrows:
        j = names.index(r["column"])
        r["held_sessions_nonzero_before_switch"] = int(nzF[pre_m, j].sum())
        r["held_first_nonzero"] = (
            str(cal[nzF[:, j]][0].date()) if nzF[:, j].any() else ""
        )
        r["held_mean_std_weight_before_switch"] = (
            float(LF["Z"][pre_m & nzF[:, j], j].mean())
            if (pre_m & nzF[:, j]).any()
            else np.nan
        )
    J["lasso"]["held"] = dict(alpha=lo_a, n_before=int(pre_m.sum()))
    VT = pd.DataFrame(vrows)
    EV = pd.DataFrame(erows)
    VT.to_csv(OUT / "regimes_lasso_vix_columns.csv", index=False)
    EV.to_csv(OUT / "regimes_lasso_vix_events.csv", index=False)
    say(
        VT[(VT.sessions_nonzero > 0) | (VT.held_sessions_nonzero_before_switch > 0)]
        .round(4)
        .to_string()
    )
    say(
        EV.round(4).to_string()
        if len(EV)
        else "(3) no VIX-family column changes status"
    )
    evc = EV.groupby("why").size()
    say(
        "(3) VIX-family entries/exits by cause: "
        + ", ".join(f"{k} {v}" for k, v in evc.items())
    )
    warm = EV[EV.why == "warm update (data)"]
    J["lasso"]["events_by_cause"] = {k: int(v) for k, v in evc.items()}
    J["lasso"]["warm_max_abs_std_weight_after"] = float(
        warm.std_weight_after.abs().max()
    )
    pre_sw = np.asarray(cal < cal[swL[0]])
    vj = [names.index(c) for c in vcols]
    J["lasso"]["pre_switch_max_abs_vix_std_weight"] = float(
        np.abs(LA["Z"][pre_sw][:, vj]).max()
    )
    say(
        f"(3) before {cal[swL[0]].date()} the largest |std weight| of any VIX-family column in the lasso of record: "
        f"{J['lasso']['pre_switch_max_abs_vix_std_weight']:.4f}; after a warm-path entry at most "
        f"{J['lasso']['warm_max_abs_std_weight_after']:.4f}"
    )
    say(
        f"(3) VIX-family columns ever nonzero: {int((VT.sessions_nonzero > 0).sum())} of {len(VT)}; "
        f"sessions with at least one nonzero: {int(nzL[:, [names.index(c) for c in vcols]].any(1).sum())} of {n}"
    )
    # the lasso's other weights: which series it keeps, share of |contribution|
    LAabs = pd.DataFrame(
        {g_: np.abs(LA["C"][:, grp == g_].sum(1)) for g_ in gset}, index=cal
    )
    lsh = LAabs[deck_m].mean()
    lsh = (lsh / lsh.sum()).sort_values(ascending=False)
    say(
        "(3) lasso mean |SHAP| share on deck days: "
        + ", ".join(f"{stem[g_]} {v:.1%}" for g_, v in lsh.head(8).items())
    )
    rsh = Aabs[deck_m].mean()
    rsh = (rsh / rsh.sum()).sort_values(ascending=False)
    say(
        "(3) ridge mean |SHAP| share on deck days: "
        + ", ".join(f"{stem[g_]} {v:.1%}" for g_, v in rsh.head(8).items())
    )
    J["lasso"]["share"] = {stem[g_]: float(v) for g_, v in lsh.items()}
    J["ridge_share_deck"] = {stem[g_]: float(v) for g_, v in rsh.items()}
    J["vix_columns"] = VT.to_dict("records")
    J["vix_events"] = EV.to_dict("records")
    rho = np.corrcoef(LA["pred"], R["pred"])[0, 1]
    say(f"(3) corr(lasso forecast, ridge forecast) on all refits {rho:.3f}")
    J["lasso"]["corr_ridge"] = float(rho)

    # ---- figures
    deck_lo, deck_hi = d.index.min(), d.index.max()

    def marks(ax, labels=False):
        for cname, cdate, kind in cand:
            ax.axvline(
                cdate,
                color="#b2182b" if kind == "penalty" else "#2166ac",
                lw=0.7,
                ls="--" if kind == "penalty" else ":",
            )
        ax.axvspan(covid_lo, covid_hi, color="#2166ac", alpha=0.08, lw=0)
        ax.axvspan(deck_lo, deck_hi, color="0.93", zorder=0)

    # (a) weights with the fixed-penalty counterfactuals
    nc = 4
    nr = int(np.ceil(len(show_cols) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, 2.1 * nr + 0.5), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for a_i, c in enumerate(show_cols):
        ax = axes[a_i]
        j = names.index(c)
        marks(ax)
        ax.plot(
            cal,
            F[100.0]["Z"][:, j],
            color="#1b9e77",
            lw=0.8,
            label="penalty held at 100",
        )
        ax.plot(
            cal,
            F[1000.0]["Z"][:, j],
            color="#d95f02",
            lw=0.8,
            label="penalty held at 1000",
        )
        ax.plot(
            cal,
            R["Z"][:, j],
            color="k",
            lw=1.2,
            ls=(0, (1, 1.5)),
            label="arm of record",
        )
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.axhline(0, color="0.6", lw=0.4)
    for ax in axes[len(show_cols) :]:
        ax.axis("off")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "regimes_weights_counterfactual.png", dpi=150)
    plt.close(fig)

    # (b) sup-Wald paths
    show = [
        ("A0_actual", "A0_1000.0", "A0_100.0", "target on forecast (all refits)"),
        (
            "A1_actual",
            "A1_1000.0",
            "A1_100.0",
            "target on series contributions (all refits)",
        ),
        ("B0_mid", "B0_crossed", None, "mean return per premium (deck days)"),
        ("B1_actual", "B1_1000.0", None, "return (mid) on contributions (deck days)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2), sharex=True)
    for ax, (r1, r2, r3, ttl) in zip(axes.ravel(), show):
        marks(ax)
        for rid_opt, col, lab in (
            (r1, "k", None),
            (r2, "#d95f02", None),
            (r3, "#1b9e77", None),
        ):
            if rid_opt is None:
                continue
            x_, w_ = paths[rid_opt]
            lab = {
                "actual": "arm of record",
                "1000.0": "penalty held at 1000",
                "100.0": "penalty held at 100",
                "mid": "mid",
                "crossed": "crossed",
            }[rid_opt.split("_", 1)[1]]
            ax.plot(x_, w_, color=col, lw=0.9, label=lab)
        p = SB[SB.reg == r1].p.iloc[0]
        ax.axhline(np.quantile(crit[p], 0.95), color="0.3", lw=0.8, ls="-.")
        ax.set_title(ttl + f" (p={p}; dash-dot = 5% critical value)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "regimes_supwald_paths.png", dpi=150)
    plt.close(fig)

    # (c) rolling SHAP shares
    cols8 = order[:N_TOP]
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    fig, axes = plt.subplots(
        len(ROLL_WINDOWS), 1, figsize=(11, 2.4 * len(ROLL_WINDOWS) + 0.6), sharex=True
    )
    for ax, w in zip(axes, ROLL_WINDOWS):
        sh = shares[w]
        stack = [sh[g_].to_numpy() for g_ in cols8] + [sh[rest].sum(1).to_numpy()]
        ax.stackplot(
            cal,
            *stack,
            colors=palette + ["0.8"],
            labels=[stem[g_] for g_ in cols8] + ["all other series"],
            lw=0,
        )
        for cname, cdate, kind in cand:
            ax.axvline(cdate, color="k", lw=0.7, ls="--" if kind == "penalty" else ":")
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"share, {w}-session window", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].legend(
        fontsize=6, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.35), frameon=False
    )
    fig.tight_layout()
    fig.savefig(OUT / "regimes_shap_share.png", dpi=150)
    plt.close(fig)

    # (d) the lasso's VIX-family weights
    lag_cols: dict[int | None, str] = dict(
        zip(md.LAGS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharex=True)
    for ax, gname in zip(axes, VIX_FAMILY):
        marks(ax)
        for k in swL:
            ax.axvline(cal[k], color="#b2182b", lw=1.2, alpha=0.5)
        for c in vcols:
            gg, kk, lag = md.parse_feature(c)
            if gg != gname:
                continue
            j = names.index(c)
            if not nzL[:, j].any():
                continue
            ax.plot(cal, LA["Z"][:, j], color=lag_cols.get(lag, "k"), lw=1.0, label=c)
            ax.plot(
                cal, LF["Z"][:, j], color=lag_cols.get(lag, "k"), lw=0.6, ls=(0, (2, 2))
            )
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(f"lasso: {md.var_stem(gname, names)}", fontsize=8)
        ax.tick_params(labelsize=6)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=5.5, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "regimes_lasso_vix.png", dpi=150)
    plt.close(fig)

    J["n"] = n
    J["deck_n"] = len(d)
    J["top"] = [stem[g_] for g_ in top]
    J["rest"] = [stem[g_] for g_ in rest]
    J["show_cols"] = show_cols
    J["hac"] = dict(refits=Lr, deck=Ld)
    J["crit95"] = {int(p): float(np.quantile(v, 0.95)) for p, v in crit.items()}
    J["dom"] = {
        int(w): {stem[g_]: int(c) for g_, c in dom[w].items()} for w in ROLL_WINDOWS
    }
    J["agree"] = float(agree.mean())
    J["agree_n"] = int(len(agree))
    (OUT / "regimes_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "regimes.json").write_text(
        json.dumps(J, indent=1, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
