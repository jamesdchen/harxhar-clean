"""Scores of the trees' three-way training-sample comparison (results/trees_train3way_2014start_w3000/RULE.txt section
6), run on CARC after the merges (experiments/trees_train3way.py merge): experiments/linear_train3way_score.py's
conventions with the tree classes in place of the linear models.

Days: experiments/rolling_eval_2026_10_15.history() days in 2014-01-02 .. 2024-04-30 (2,579: positive closing print,
early closes dropped).
(a) every RTH bar (every arm but CLOSE): the 13 bars of those days with RV > 0, common to every variant scored; QLIKE
    (mean over bars), MSE x 1e10, R2_OOS against the per-clock expanding mean of the clock's prior RV (positive finite
    prints, early closes dropped; at 16:00 RE.ct_benchmark, asserted); DM on the daily 13-bar mean loss.
(b) 16:00 (every arm): RE.metrics with RE.ct_benchmark; DM (RE.dm_pair: QLIKE and squared error, a minus b).
Full vs HAR within each arm and class.  Checks: the RTH13 rows (the templates' forecasts, md5-checked) reproduce the
templates' scores/metrics.csv (QLIKE and MSE every bar and at 16:00, bar and day counts) to 1e-12 and their full-vs-HAR
QLIKE DM (dm.csv) to 1e-9; every arm's rv_actual, true_adj and B equal RTH13's at common rows.  Arms or classes not
merged yet are left out (their pairs skipped), so the script also gives interim tables.

usage: python experiments/trees_train3way_score.py <runs dir (holds <ARM>/forecasts.parquet)> <out dir>
  -> <out dir>/: qlike_bar.csv qlike_16.csv dm.csv gap.csv f16.parquet tables.txt
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rolling_eval_2026_10_15 as RE  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

RUNS, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SPAN = (pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"))
EST = ("lgbm", "xgb", "rf_skx")
SETS = ("baseline", "all_features")
MODELS = [(e, s) for e in EST for s in SETS]
TPL = {  # the templates' forecasts (the RTH13 arm, reused) and the repo's md5 of each file
    REPO / "results" / "pooled_trees_final_2014start_w3000": ("7f15d5a5d422ad7af90ebe945e8a42b6", ("lgbm", "xgb")),
    REPO / "results" / "pooled_rf_skx_2014start_w3000": ("f68016d41ec3cbda92cd9d2ace21f3c5", ("rf_skx",)),
}
ARMS = ("ALL48", "RTH13", "CLOSE", "ALL48_N3k", "RTH13_N3k", "ALL48_N39k")
BARV = ("ALL48", "RTH13", "ALL48_N3k", "RTH13_N3k", "ALL48_N39k")  # every RTH bar
COLS = ["model", "predictor_set", "date", "true_adj", "B", "plain", "rv_actual"]
PAIRS16 = [
    ("ALL48", "RTH13"),
    ("CLOSE", "RTH13"),
    ("CLOSE", "ALL48"),
    ("ALL48_N3k", "RTH13_N3k"),
    ("CLOSE", "RTH13_N3k"),
    ("CLOSE", "ALL48_N3k"),
    ("ALL48_N39k", "RTH13"),
    ("ALL48_N3k", "ALL48"),
    ("RTH13_N3k", "RTH13"),
    ("ALL48_N39k", "ALL48"),
]
PAIRSB = [
    ("ALL48", "RTH13"),
    ("ALL48_N3k", "RTH13_N3k"),
    ("ALL48_N39k", "RTH13"),
    ("ALL48_N3k", "ALL48"),
    ("RTH13_N3k", "RTH13"),
    ("ALL48_N39k", "ALL48"),
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def prep(f: pd.DataFrame) -> pd.DataFrame:
    f = f[COLS].copy()
    f["date"] = pd.to_datetime(f["date"])
    f["day"] = f["date"].dt.normalize().astype("datetime64[ns]")
    f["clock"] = f["date"].dt.strftime("%H:%M")
    return f


F: dict[str, pd.DataFrame] = {}
parts = []
for d, (h, ests) in TPL.items():
    p = d / "forecasts.parquet"
    assert md5(p) == h, f"the reused template forecasts {p} are not the repo's file"
    f = pd.read_parquet(p)
    parts.append(f[f["model"].isin(ests)])
F["RTH13"] = prep(pd.concat(parts, ignore_index=True))
for v in ARMS:
    p = RUNS / v / "forecasts.parquet"
    if v != "RTH13" and p.is_file():
        F[v] = prep(pd.read_parquet(p))
HAVE = {(v, m, s) for v, f in F.items() for m, s in f.groupby(["model", "predictor_set"]).size().index}
for v, f in F.items():
    n = f.groupby(["model", "predictor_set"]).size()
    print(v, dict(n), f["date"].min(), f["date"].max(), flush=True)
assert all((("RTH13", m, s) in HAVE) for m, s in MODELS), "RTH13 incomplete"

hist = RE.history()
hdays = hist.index[(hist.index >= SPAN[0]) & (hist.index <= SPAN[1])]
print(f"{len(hdays)} history days in {SPAN[0].date()} .. {SPAN[1].date()}", flush=True)
bench16 = RE.ct_benchmark(hist, hdays)

# per-clock benchmark (every bar), as experiments/linear_train3way_score.py
core = pd.read_parquet(REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
et = pd.to_datetime(core["endbartime"])
clock = et.dt.strftime("%H:%M")
BENCH: dict[str, pd.Series] = {}
for c in sorted({f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {"16:30"}):
    k = (clock == c).to_numpy()
    s_ = pd.Series(core.loc[k, "sumret2"].to_numpy(float), index=RE._day_index(et[k])).sort_index()
    assert not s_.index.duplicated().any(), c
    s_ = s_[np.isfinite(s_.to_numpy())]
    h_ = s_[(s_.to_numpy() > 0) & ~s_.index.isin(RE.EARLY)]
    prior = h_.cumsum().shift(1) / pd.Series(np.arange(len(h_), dtype=float), index=h_.index)
    BENCH[c] = prior.replace([np.inf], np.nan)
b16 = BENCH["16:00"].reindex(hdays)
assert np.allclose(b16.to_numpy(), bench16.to_numpy(), rtol=1e-12, atol=0), "16:00 benchmark != RE.ct_benchmark"

# common rows agree with RTH13: truth, target, B
ref = F["RTH13"][(F["RTH13"].model == "lgbm") & (F["RTH13"].predictor_set == "baseline")].set_index("date")
for v, f in F.items():
    for (m, s), g in f.groupby(["model", "predictor_set"]):
        g = g.set_index("date")
        i = g.index.intersection(ref.index)
        for c in ("rv_actual", "true_adj", "B"):
            a_, b_ = g.loc[i, c].to_numpy(float), ref.loc[i, c].to_numpy(float)
            same = np.isclose(a_, b_, rtol=1e-12, atol=0) | (np.isnan(a_) & np.isnan(b_))
            assert same.all(), (v, m, s, c, int((~same).sum()))
    print(f"{v}: rv_actual, true_adj, B equal to RTH13's on the common rows", flush=True)

L16, P16, LB = {}, {}, {}
rv16 = hist.reindex(hdays)
for v, m, s in sorted(HAVE):
    f = F[v]
    g = f[(f.model == m) & (f.predictor_set == s) & (f.clock == "16:00")].set_index("day").sort_index()
    assert g.index.is_unique, (v, m, s)
    g = g.reindex(hdays)
    p = g["plain"]
    assert np.isfinite(p.to_numpy()).all() and (p > 0).all(), (v, m, s)
    assert np.allclose(g["rv_actual"].to_numpy(float), rv16.to_numpy(float), rtol=1e-12, atol=0), (v, m, s)
    P16[(v, m, s)] = p
    x = (rv16 / p).to_numpy(float)
    L16[(v, m, s)] = pd.Series(x - np.log(x) - 1.0, index=hdays)
    if v in BARV:
        g = f[(f.model == m) & (f.predictor_set == s) & f.day.isin(hdays) & (f.rv_actual > 0)].sort_values("date")
        r = g["rv_actual"].to_numpy(float) / g["plain"].to_numpy(float)
        bn = np.array([BENCH[c].get(d, np.nan) for c, d in zip(g["clock"], g["day"])])
        LB[(v, m, s)] = pd.DataFrame(
            {
                "q": r - np.log(r) - 1.0,
                "se": (g["rv_actual"].to_numpy(float) - g["plain"].to_numpy(float)) ** 2,
                "be": (g["rv_actual"].to_numpy(float) - bn) ** 2,
                "day": g["day"].to_numpy(),
            },
            index=pd.DatetimeIndex(g["date"]),
        )
ib: pd.Index | None = None
for xx in LB.values():
    ib = xx.index if ib is None else ib.intersection(xx.index)
assert ib is not None
for k_ in LB:
    n0 = len(LB[k_])
    LB[k_] = LB[k_].reindex(ib)
    assert np.isfinite(LB[k_].to_numpy(float)[:, :3]).all(), k_
    assert n0 == len(ib), (k_, n0, len(ib))  # every variant forecasts every scored bar
print(f"every bar: {len(ib)} common intervals on {pd.DatetimeIndex(ib).normalize().nunique()} days", flush=True)

r16, rbar = [], []
for v, m, s in sorted(HAVE):
    mt = RE.metrics(rv16, P16[(v, m, s)], bench16)
    r16.append(
        dict(
            variant=v,
            model=m,
            set=s,
            n_days=len(hdays),
            qlike=mt["qlike"],
            mse_e10=mt["mse"] * 1e10,
            r2_oos_pct=mt["r2_oos_pct"],
        )
    )
    if v in BARV:
        xx = LB[(v, m, s)]
        rbar.append(
            dict(
                variant=v,
                model=m,
                set=s,
                n_bars=len(xx),
                n_days=int(xx["day"].nunique()),
                qlike=float(xx["q"].mean()),
                mse_e10=float(xx["se"].mean() * 1e10),
                r2_oos_pct=float(100.0 * (1.0 - xx["se"].sum() / xx["be"].sum())),
            )
        )
Q16, QB = pd.DataFrame(r16), pd.DataFrame(rbar)


def daily(v: str, m: str, s: str, col: str) -> pd.Series:
    return LB[(v, m, s)].groupby("day")[col].mean()


dms = []
for m, s in MODELS:
    for a, b in PAIRS16:
        if (a, m, s) not in HAVE or (b, m, s) not in HAVE:
            continue
        r = RE.dm_pair(rv16, P16[(a, m, s)], P16[(b, m, s)])
        qa, qb = L16[(a, m, s)].mean(), L16[(b, m, s)].mean()
        dms.append(dict(target="16:00", model=m, set=s, a=a, b=b, qlike_a=qa, qlike_b=qb, pct=100 * (qa / qb - 1), **r))
    for a, b in PAIRSB:
        if (a, m, s) not in HAVE or (b, m, s) not in HAVE:
            continue
        q = dm_test(daily(a, m, s, "q").to_numpy(), daily(b, m, s, "q").to_numpy())
        e = dm_test(daily(a, m, s, "se").to_numpy(), daily(b, m, s, "se").to_numpy())
        qa, qb = LB[(a, m, s)]["q"].mean(), LB[(b, m, s)]["q"].mean()
        dms.append(
            dict(
                target="every bar",
                model=m,
                set=s,
                a=a,
                b=b,
                qlike_a=qa,
                qlike_b=qb,
                pct=100 * (qa / qb - 1),
                dm_qlike=q["dm"],
                dm_qlike_p=q["p"],
                dm_qlike_mean_diff=q["mean_diff"],
                dm_se=e["dm"],
                dm_se_p=e["p"],
                dm_T=q["T"],
                dm_hac_lag=q.get("hac_lag", -1),
            )
        )
DM = pd.DataFrame(dms)
gaps = []
for v in F:
    for e_ in EST:
        h, fu = "baseline", "all_features"
        if (v, e_, h) not in HAVE or (v, e_, fu) not in HAVE:
            continue
        r = RE.dm_pair(rv16, P16[(v, e_, fu)], P16[(v, e_, h)])
        qh, qf = L16[(v, e_, h)].mean(), L16[(v, e_, fu)].mean()
        gaps.append(
            dict(target="16:00", variant=v, model=e_, qlike_har=qh, qlike_full=qf, pct=100 * (qf / qh - 1), **r)
        )
        if v in BARV:
            q = dm_test(daily(v, e_, fu, "q").to_numpy(), daily(v, e_, h, "q").to_numpy())
            e = dm_test(daily(v, e_, fu, "se").to_numpy(), daily(v, e_, h, "se").to_numpy())
            qh, qf = LB[(v, e_, h)]["q"].mean(), LB[(v, e_, fu)]["q"].mean()
            gaps.append(
                dict(
                    target="every bar",
                    variant=v,
                    model=e_,
                    qlike_har=qh,
                    qlike_full=qf,
                    pct=100 * (qf / qh - 1),
                    dm_qlike=q["dm"],
                    dm_qlike_p=q["p"],
                    dm_qlike_mean_diff=q["mean_diff"],
                    dm_se=e["dm"],
                    dm_se_p=e["p"],
                    dm_T=q["T"],
                    dm_hac_lag=q.get("hac_lag", -1),
                )
            )
GAP = pd.DataFrame(gaps)

f16 = []
for v, m, s in sorted(HAVE):
    f = F[v]
    g = f[(f.model == m) & (f.predictor_set == s) & (f.clock == "16:00") & (f.day >= SPAN[0])]
    f16.append(
        pd.DataFrame(
            dict(
                variant=v,
                model=m,
                predictor_set=s,
                day=g["day"].to_numpy(),
                plain=g["plain"].to_numpy(),
                rv_actual=g["rv_actual"].to_numpy(),
            )
        )
    )
pd.concat(f16, ignore_index=True).to_parquet(OUT / "f16.parquet", index=False)
for name, df in (("qlike_16", Q16), ("qlike_bar", QB), ("dm", DM), ("gap", GAP)):
    df.to_csv(OUT / f"{name}.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
with open(OUT / "tables.txt", "w") as fh:
    fh.write(
        f"arms {sorted(F)}; days {len(hdays)}; every bar {len(ib)} intervals\n\n16:00\n{Q16.to_string(index=False)}\n\n"
    )
    fh.write(f"every bar\n{QB.to_string(index=False)}\n\nDM\n{DM.to_string(index=False)}\n\n")
    fh.write(f"full vs HAR\n{GAP.to_string(index=False)}\n")
print((OUT / "tables.txt").read_text())

# ---------------------------------------------------------------------------- check against the templates' scores
chk = []
for d in TPL:
    M = pd.read_csv(d / "scores" / "metrics.csv")
    D = pd.read_csv(d / "scores" / "dm.csv")
    M = M[(M.W == 3000) & (M.family == "tree") & M.model.isin(EST)]
    for _, r in M.iterrows():
        a16 = Q16[(Q16.variant == "RTH13") & (Q16.model == r.model) & (Q16.set == r.set)].iloc[0]
        ab = QB[(QB.variant == "RTH13") & (QB.model == r.model) & (QB.set == r.set)].iloc[0]
        assert int(ab.n_bars) == int(r.n_bars) and int(ab.n_days) == int(r.n_days), (r.model, r.set)
        for mine, theirs in (
            (ab.qlike, r.qlike),
            (a16.qlike, r.qlike_1600),
            (ab.mse_e10, r.mse_e10),
            (a16.mse_e10, r.mse_1600_e10),
        ):
            assert abs(mine - theirs) <= 1e-12 * max(1.0, abs(theirs)), (r.model, r.set, mine, theirs)
        chk.append(dict(model=r.model, set=r.set, r2_bar_mine=ab.r2_oos_pct, r2_bar_template=r.r2_oos_pct))
    D = D[(D.W == 3000) & (D.comparison == "all features vs HAR benchmark") & (D.loss == "qlike") & D.model.isin(EST)]
    for _, r in D.iterrows():
        x = GAP[(GAP.variant == "RTH13") & (GAP.model == r.model)]
        tg = "16:00" if r.bars == "16:00" else "every bar"
        got = x[x.target == tg].iloc[0]
        assert abs(got.dm_qlike - r.dm) <= 1e-9 and abs(got.dm_qlike_p - r.p) <= 1e-9, (r.model, tg, got.dm_qlike, r.dm)
print(
    "check: RTH13 reproduces the templates' metrics.csv (QLIKE, MSE, every bar and 16:00) and the full-vs-HAR QLIKE DM",
    flush=True,
)

pd.DataFrame(chk).to_csv(OUT / "check_r2.csv", index=False)
print(pd.DataFrame(chk).to_string(index=False))
