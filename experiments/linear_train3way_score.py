"""Scores of the three-way training-sample comparison (results/linear_train3way_2014start_w3000/RULE.txt section 6),
run on CARC after the walks and the per-arm export (experiments/linear_train3way.py export).

Days: experiments/rolling_eval_2026_10_15.history() days in 2014-01-02 .. 2024-04-30 (2,579: positive closing print,
early closes dropped), as results/window_sweep_2014start_2026-10-22/code/score14.py.
(a) every RTH bar (ALL48, RTH13 and their 16:00-tuned variants): the 13 bars of those days with RV > 0; QLIKE (mean over
    bars), MSE x 1e10, R2_OOS against the per-clock expanding mean of the clock's prior RV (positive finite prints,
    early closes dropped; at 16:00 RE.ct_benchmark, asserted); DM on the daily 13-bar mean loss.
(b) 16:00 (every variant): RE.metrics with RE.ct_benchmark; DM (RE.dm_pair: QLIKE and squared error, a minus b).
Full vs HAR within each variant and estimator.  Checks: RTH13 reproduces part2/qlike.csv and part2/gap.csv (w3000) to
1e-12; every variant's rv_actual, true_adj and B agree at common rows.  Diagnostic: the 16:00 rows of the ALL48 and
CLOSE_NAT designs against RTH13's (the arm-specific scaler), and CLOSE's against RTH13's (identical by construction).

usage: python experiments/linear_train3way_score.py <results dir> <template run dir (w3000)>
  -> <results dir>/scores/: qlike_bar.csv qlike_16.csv dm.csv gap.csv f16.parquet design_diag.csv tables.txt
"""

from __future__ import annotations

import hashlib
import json
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

RES, REF = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT = RES / "scores"
OUT.mkdir(exist_ok=True)
SPAN = (pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"))
REF_MD5 = "4e5c7c1771b66ed251cf0f79774e7fd3"  # the repo's w3000/forecasts.parquet (RULE.txt gate iv)
MODELS = [("ols", "baseline")] + [(e, s) for s in ("baseline", "all_features") for e in ("ridge", "lasso", "enet")]
EST = ("ridge", "lasso", "enet")
SRC = {
    "ALL48": RES / "ALL48" / "forecasts_rth.parquet",
    "RTH13": REF / "forecasts.parquet",
    "CLOSE": RES / "CLOSE" / "forecasts_rth.parquet",
    "CLOSE_NAT": RES / "CLOSE_NAT" / "forecasts_rth.parquet",
    "ALL48@16": RES / "ALL48" / "forecasts16_rth.parquet",
    "RTH13@16": RES / "RTH13" / "forecasts16_rth.parquet",
}
BAR = ("ALL48", "RTH13", "ALL48@16", "RTH13@16")  # variants with every RTH bar
COLS = ["model", "predictor_set", "alpha", "date", "true_adj", "B", "plain", "rv_actual"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


assert md5(SRC["RTH13"]) == REF_MD5, "the reused RTH13 forecasts are not the repo's file"
hist = RE.history()
hdays = hist.index[(hist.index >= SPAN[0]) & (hist.index <= SPAN[1])]
print(f"{len(hdays)} history days in {SPAN[0].date()} .. {SPAN[1].date()}", flush=True)
bench16 = RE.ct_benchmark(hist, hdays)

# ---------------------------------------------------------------------------- per-clock benchmark (every bar)
core = pd.read_parquet(REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
et = pd.to_datetime(core["endbartime"])
clock = et.dt.strftime("%H:%M")
BENCH: dict[str, pd.Series] = {}
for c in sorted({f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {"16:30"}):
    k = (clock == c).to_numpy()
    s = pd.Series(core.loc[k, "sumret2"].to_numpy(float), index=RE._day_index(et[k])).sort_index()
    assert not s.index.duplicated().any(), c
    s = s[np.isfinite(s.to_numpy())]
    h = s[(s.to_numpy() > 0) & ~s.index.isin(RE.EARLY)]
    prior = h.cumsum().shift(1) / pd.Series(np.arange(len(h), dtype=float), index=h.index)
    BENCH[c] = prior.replace([np.inf], np.nan)
b16 = BENCH["16:00"].reindex(hdays)
assert np.allclose(b16.to_numpy(), bench16.to_numpy(), rtol=1e-12, atol=0), "16:00 benchmark != RE.ct_benchmark"


def load(v: str) -> pd.DataFrame:
    f = pd.read_parquet(SRC[v], columns=COLS)
    f["date"] = pd.to_datetime(f["date"])
    f["day"] = f["date"].dt.normalize().astype("datetime64[ns]")
    f["clock"] = f["date"].dt.strftime("%H:%M")
    return f


F = {v: load(v) for v in SRC}
for v, f in F.items():
    n = f.groupby(["model", "predictor_set"]).size()
    assert set(n.index) == {(m, s) for m, s in MODELS}, (v, n)
    print(v, len(f), f["date"].min(), f["date"].max(), flush=True)

# common rows agree: truth, target and B
ref = F["RTH13"][(F["RTH13"].model == "ols")].set_index("date")
for v, f in F.items():
    g = f[f.model == "ols"].set_index("date")
    i = g.index.intersection(ref.index)
    i = i[(i >= SPAN[0])]
    for c in ("rv_actual", "true_adj", "B"):
        a, b = g.loc[i, c].to_numpy(float), ref.loc[i, c].to_numpy(float)
        same = np.isclose(a, b, rtol=1e-12, atol=0) | (np.isnan(a) & np.isnan(b))
        assert same.all(), (v, c, int((~same).sum()))
    print(f"{v}: rv_actual, true_adj, B equal to RTH13's on {len(i)} common rows from 2014", flush=True)


# ---------------------------------------------------------------------------- losses
def at16(v: str, m: str, s: str) -> pd.DataFrame:
    f = F[v]
    g = f[(f.model == m) & (f.predictor_set == s) & (f.clock == "16:00")].set_index("day").sort_index()
    assert g.index.is_unique, (v, m, s)
    return g.reindex(hdays)


L16, P16, LB = {}, {}, {}
rv16 = hist.reindex(hdays)
for v in F:
    for m, s in MODELS:
        g = at16(v, m, s)
        p = g["plain"]
        assert np.isfinite(p.to_numpy()).all() and (p > 0).all(), (v, m, s)
        assert np.allclose(g["rv_actual"].to_numpy(float), rv16.to_numpy(float), rtol=1e-12, atol=0), (v, m, s)
        P16[(v, m, s)] = p
        x = (rv16 / p).to_numpy(float)
        L16[(v, m, s)] = pd.Series(x - np.log(x) - 1.0, index=hdays)
        if v in BAR:
            f = F[v]
            g = f[(f.model == m) & (f.predictor_set == s) & f.day.isin(hdays) & (f.rv_actual > 0)]
            g = g.sort_values("date")
            r = g["rv_actual"].to_numpy(float) / g["plain"].to_numpy(float)
            q = r - np.log(r) - 1.0
            se = (g["rv_actual"].to_numpy(float) - g["plain"].to_numpy(float)) ** 2
            bn = np.array([BENCH[c].get(d, np.nan) for c, d in zip(g["clock"], g["day"])])
            LB[(v, m, s)] = pd.DataFrame(
                {"q": q, "se": se, "be": (g["rv_actual"].to_numpy(float) - bn) ** 2, "day": g["day"].to_numpy()},
                index=pd.DatetimeIndex(g["date"]),
            )
ib: pd.Index | None = None
for x in LB.values():
    ib = x.index if ib is None else ib.intersection(x.index)
assert ib is not None
for k in LB:
    LB[k] = LB[k].reindex(ib)
    assert np.isfinite(LB[k]["be"].to_numpy()).all(), k
print(f"every bar: {len(ib)} common intervals on {pd.DatetimeIndex(ib).normalize().nunique()} days", flush=True)

# ---------------------------------------------------------------------------- tables
r16, rbar = [], []
for v in F:
    for m, s in MODELS:
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
        if v in BAR:
            x = LB[(v, m, s)]
            rbar.append(
                dict(
                    variant=v,
                    model=m,
                    set=s,
                    n_bars=len(x),
                    n_days=int(x["day"].nunique()),
                    qlike=float(x["q"].mean()),
                    mse_e10=float(x["se"].mean() * 1e10),
                    r2_oos_pct=float(100.0 * (1.0 - x["se"].sum() / x["be"].sum())),
                )
            )
Q16, QB = pd.DataFrame(r16), pd.DataFrame(rbar)


def daily(v: str, m: str, s: str, col: str) -> pd.Series:
    x = LB[(v, m, s)]
    return x.groupby("day")[col].mean()


PAIRS16 = [
    ("ALL48", "RTH13"),
    ("CLOSE", "RTH13"),
    ("CLOSE", "ALL48"),
    ("CLOSE_NAT", "RTH13"),
    ("CLOSE_NAT", "CLOSE"),
    ("ALL48@16", "RTH13"),
    ("RTH13@16", "RTH13"),
    ("ALL48@16", "ALL48"),
]
PAIRSB = [("ALL48", "RTH13"), ("ALL48@16", "RTH13"), ("RTH13@16", "RTH13"), ("ALL48@16", "ALL48")]
dms = []
for m, s in MODELS:
    for a, b in PAIRS16:
        r = RE.dm_pair(rv16, P16[(a, m, s)], P16[(b, m, s)])
        qa, qb = L16[(a, m, s)].mean(), L16[(b, m, s)].mean()
        dms.append(dict(target="16:00", model=m, set=s, a=a, b=b, qlike_a=qa, qlike_b=qb, pct=100 * (qa / qb - 1), **r))
    for a, b in PAIRSB:
        qa_d, qb_d = daily(a, m, s, "q"), daily(b, m, s, "q")
        sa_d, sb_d = daily(a, m, s, "se"), daily(b, m, s, "se")
        q = dm_test(qa_d.to_numpy(), qb_d.to_numpy())
        e = dm_test(sa_d.to_numpy(), sb_d.to_numpy())
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
        h, fu = ("baseline", "all_features")
        r = RE.dm_pair(rv16, P16[(v, e_, fu)], P16[(v, e_, h)])
        qh, qf = L16[(v, e_, h)].mean(), L16[(v, e_, fu)].mean()
        gaps.append(
            dict(target="16:00", variant=v, model=e_, qlike_har=qh, qlike_full=qf, pct=100 * (qf / qh - 1), **r)
        )
        if v in BAR:
            q = dm_test(daily(v, e_, fu, "q").to_numpy(), daily(v, e_, h, "q").to_numpy())
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
                    dm_T=q["T"],
                    dm_hac_lag=q.get("hac_lag", -1),
                )
            )
GAP = pd.DataFrame(gaps)

# ---------------------------------------------------------------------------- checks against the window sweep's part 2
p2 = pd.read_csv(REPO / "results" / "window_sweep_2014start_2026-10-22" / "part2" / "qlike.csv")
p2 = p2[p2.W == 3000]
g2 = pd.read_csv(REPO / "results" / "window_sweep_2014start_2026-10-22" / "part2" / "gap.csv")
g2 = g2[g2.W == 3000]
for _, r in p2.iterrows():
    a = Q16[(Q16.variant == "RTH13") & (Q16.model == r.model) & (Q16.set == r.set)].iloc[0]
    b = QB[(QB.variant == "RTH13") & (QB.model == r.model) & (QB.set == r.set)].iloc[0]
    assert abs(a.qlike - r.qlike16) <= 1e-12 and abs(b.qlike - r.qlike_bar) <= 1e-12, (r.model, r.set)
    assert int(a.n_days) == int(r.n_days16) and int(b.n_bars) == int(r.n_bars), (r.model, r.set)
for _, r in g2.iterrows():
    x = GAP[(GAP.variant == "RTH13") & (GAP.model == r.model)]
    a, b = x[x.target == "16:00"].iloc[0], x[x.target == "every bar"].iloc[0]
    assert abs(a.pct - r.gap_pct_16) <= 1e-9 and abs(a.dm_qlike - r.gap_dm_16) <= 1e-9, r.model
    assert abs(b.pct - r.gap_pct_bar) <= 1e-9 and abs(b.dm_qlike - r.gap_dm_bar) <= 1e-9, r.model
print("check: RTH13 reproduces part2/qlike.csv and part2/gap.csv (w3000)", flush=True)

# ---------------------------------------------------------------------------- the 16:00 forecasts (economics, local)
f16 = []
for v in F:
    for m, s in MODELS:
        f = F[v]
        g = f[(f.model == m) & (f.predictor_set == s) & (f.clock == "16:00") & (f.day >= SPAN[0])]
        f16.append(
            pd.DataFrame(
                dict(
                    variant=v,
                    model=m,
                    predictor_set=s,
                    alpha=g["alpha"].to_numpy(),
                    day=g["day"].to_numpy(),
                    plain=g["plain"].to_numpy(),
                    rv_actual=g["rv_actual"].to_numpy(),
                )
            )
        )
pd.concat(f16, ignore_index=True).to_parquet(OUT / "f16.parquet", index=False)


# ---------------------------------------------------------------------------- design diagnostic (16:00 rows)
def design16(arm: str, s: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    dz = RES / arm / "_work" / f"design_{s}"
    names = json.loads((dz / "design.json").read_text())["names"]
    d = pd.to_datetime(pd.read_csv(dz / "dates_all.csv", dtype=str)["date"])
    k = np.flatnonzero(((d.dt.strftime("%H:%M") == "16:00") & (d >= SPAN[0])).to_numpy())
    X = np.load(dz / "X.npy", mmap_mode="r")
    y = np.load(dz / "y.npy")
    W = int(json.loads((dz / "design.json").read_text())["config"]["window"])
    b = np.load(dz / "b_oos.npy")
    kb = k[k >= W]
    return (
        pd.DataFrame(np.asarray(X[k]), index=pd.DatetimeIndex(d.iloc[k]), columns=names),
        pd.Series(y[k], index=pd.DatetimeIndex(d.iloc[k])),
        pd.Series(b[kb - W], index=pd.DatetimeIndex(d.iloc[kb])),
    )


diag = []
HAVE = all((RES / a / "_work" / "design_baseline").exists() for a in ("RTH13", "CLOSE", "ALL48", "CLOSE_NAT"))
for s in ("baseline", "all_features") if HAVE else ():
    base, yb, bb = design16("RTH13", s)
    for arm in ("CLOSE", "ALL48", "CLOSE_NAT"):
        X, y, b = design16(arm, s)
        i = base.index.intersection(X.index)
        j = bb.index.intersection(b.index)
        same = dict(
            y_identical=bool(np.array_equal(y.reindex(i).to_numpy(), yb.reindex(i).to_numpy())),
            b_identical=bool(np.array_equal(b.reindex(j).to_numpy(), bb.reindex(j).to_numpy())),
        )
        for c in base.columns:
            u, w = base.loc[i, c].to_numpy(), X.loc[i, c].to_numpy()
            kind = (
                "har"
                if c.startswith("har_ma_")
                else "avail"
                if "_avail_ma_" in c
                else "value"
                if c.startswith("adj_")
                else "calendar"
            )
            sd = float(np.std(u))
            diag.append(
                dict(
                    set=s,
                    arm=arm,
                    column=c,
                    kind=kind,
                    rows=len(i),
                    **same,
                    equal_share=float(np.mean(u == w)),
                    corr=float(np.corrcoef(u, w)[0, 1]) if np.std(u) > 0 and np.std(w) > 0 else np.nan,
                    mean_abs_diff_over_sd=float(np.mean(np.abs(u - w)) / sd) if sd > 0 else np.nan,
                )
            )
DG = pd.DataFrame(
    diag,
    columns=[
        "set",
        "arm",
        "column",
        "kind",
        "rows",
        "y_identical",
        "b_identical",
        "equal_share",
        "corr",
        "mean_abs_diff_over_sd",
    ],
)

for name, df in (("qlike_16", Q16), ("qlike_bar", QB), ("dm", DM), ("gap", GAP), ("design_diag", DG)):
    df.to_csv(OUT / f"{name}.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
with open(OUT / "tables.txt", "w") as fh:
    fh.write(f"days {len(hdays)}; every bar {len(ib)} intervals\n\n16:00\n{Q16.to_string(index=False)}\n\n")
    fh.write(f"every bar\n{QB.to_string(index=False)}\n\nDM\n{DM.to_string(index=False)}\n\n")
    fh.write(f"full vs HAR\n{GAP.to_string(index=False)}\n\n")
    fh.write("design diagnostic (16:00 rows from 2014, against RTH13): median over columns\n")
    fh.write(DG.groupby(["set", "arm", "kind"])[["equal_share", "corr", "mean_abs_diff_over_sd"]].median().to_string())
print((OUT / "tables.txt").read_text())
