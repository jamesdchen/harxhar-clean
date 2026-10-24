"""Scores of the close-only window sweep at 16:00 (results/closeonly_wsweep_2014start/RULE.txt section 4), run on CARC
after the walks: experiments/linear_train3way_score.py's 16:00 conventions per window W.

Days: experiments/rolling_eval_2026_10_15.history() days in 2014-01-02 .. 2024-04-30 (2,579).  Per W and model: QLIKE,
MSE x 1e10, R2_OOS (RE.metrics with RE.ct_benchmark).  Diebold-Mariano: RE.dm_pair (dm_test as is: HLN, Bartlett, lag 8
at T = 2,579) on the daily losses, QLIKE and squared error, a minus b: each W against the previous W (step), against
W = 1,000 and against W = 3,000; full vs HAR within each W and estimator.  Checks: rv_actual, true_adj and B of every W
equal at the scored days; W = 3,000 reproduces the three-way CLOSE rows of qlike_16.csv (to 1e-12) and of gap.csv at
16:00 (pct and DM to 1e-9).

usage: python experiments/closewin_score.py linear|trees <runs dir> <ref scores dir (three-way qlike_16.csv, gap.csv)>
         <out dir>
  linear: <runs>/w<W>/forecasts_rth.parquet (+ frozen_penalties.json); trees: <runs>/CLOSE_w<W>/forecasts.parquet
  -> <out>/qlike16.csv dm.csv gap.csv f16.parquet check.json tables_qlike.txt
"""

from __future__ import annotations

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
from closewin import W_FIRST, W_REF, WS  # noqa: E402

FAM = sys.argv[1]
RUNS, REF, OUT = (Path(x).resolve() for x in sys.argv[2:5])
OUT.mkdir(parents=True, exist_ok=True)
SPAN = (pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"))
if FAM == "linear":
    EST: tuple[str, ...] = ("ridge", "lasso", "enet")
    MODELS = [("ols", "baseline")] + [(e, s) for s in ("baseline", "all_features") for e in EST]
    SRC = {W: RUNS / f"w{W}" / "forecasts_rth.parquet" for W in WS}
else:
    assert FAM == "trees", FAM
    EST = ("lgbm", "xgb", "rf_skx")
    MODELS = [(e, s) for e in EST for s in ("baseline", "all_features")]
    SRC = {W: RUNS / f"CLOSE_w{W}" / "forecasts.parquet" for W in WS}

hist = RE.history()
hdays = hist.index[(hist.index >= SPAN[0]) & (hist.index <= SPAN[1])]
print(f"{len(hdays)} history days in {SPAN[0].date()} .. {SPAN[1].date()}", flush=True)
bench16 = RE.ct_benchmark(hist, hdays)
rv16 = hist.reindex(hdays)

F: dict[int, pd.DataFrame] = {}
for W, p in SRC.items():
    f = pd.read_parquet(p)
    f["date"] = pd.to_datetime(f["date"])
    f["day"] = f["date"].dt.normalize().astype("datetime64[ns]")
    f = f[f["date"].dt.strftime("%H:%M") == "16:00"]
    n = f.groupby(["model", "predictor_set"]).size()
    assert set(n.index) == set(MODELS), (W, n)
    F[W] = f
    print(f"W={W}: {len(f)} 16:00 rows, {f['date'].min()} .. {f['date'].max()}", flush=True)

P16: dict = {}
G16: dict = {}
for W in WS:
    for m, s in MODELS:
        g = F[W][(F[W].model == m) & (F[W].predictor_set == s)].set_index("day").sort_index()
        assert g.index.is_unique, (W, m, s)
        G16[(W, m, s)] = g[g.index >= SPAN[0]]  # (the linear walks start at a 250-solve boundary before 2014)
        g = g.reindex(hdays)
        p = g["plain"]
        assert np.isfinite(p.to_numpy()).all() and (p > 0).all(), (W, m, s)
        assert np.allclose(g["rv_actual"].to_numpy(float), rv16.to_numpy(float), rtol=1e-12, atol=0), (W, m, s)
        P16[(W, m, s)] = p
# the truth, the target and B agree across W at every 16:00 row from 2014
for W in WS:
    for m, s in MODELS:
        a, b = G16[(W, m, s)], G16[(W_REF, MODELS[0][0], MODELS[0][1])]
        i = a.index.intersection(b.index)
        assert len(i) == len(a) == len(b), (W, m, s, len(i), len(a), len(b))
        for c in ("rv_actual", "true_adj", "B"):
            u, v = a.loc[i, c].to_numpy(float), b.loc[i, c].to_numpy(float)
            same = np.isclose(u, v, rtol=1e-12, atol=0) | (np.isnan(u) & np.isnan(v))
            assert same.all(), (W, m, s, c, int((~same).sum()))
print(f"rv_actual, true_adj, B equal across W on the {len(G16[(W_REF, *MODELS[0])])} 16:00 rows from 2014", flush=True)


def ql(W: int, m: str, s: str) -> float:
    x = (rv16 / P16[(W, m, s)]).to_numpy(float)
    return float(np.mean(x - np.log(x) - 1.0))


pen: dict = {}
if FAM == "linear":
    for W in WS:
        fr = json.loads((RUNS / f"w{W}" / "frozen_penalties.json").read_text())
        pen[W] = {(m, s): fr[m][s] for m, s in MODELS if m != "ols"}
rows = []
for W in WS:
    for m, s in MODELS:
        mt = RE.metrics(rv16, P16[(W, m, s)], bench16)
        rows.append(
            dict(
                W=W,
                model=m,
                set=s,
                alpha=pen.get(W, {}).get((m, s), np.nan),
                n_days=len(hdays),
                qlike=mt["qlike"],
                mse_e10=mt["mse"] * 1e10,
                r2_oos_pct=mt["r2_oos_pct"],
            )
        )
Q = pd.DataFrame(rows)
dms = []
for m, s in MODELS:
    for i, W in enumerate(WS):
        for kind, Wb in (("step", WS[i - 1] if i else None), ("vs1000", W_FIRST), ("vs3000", W_REF)):
            if Wb is None or Wb == W:
                continue
            r = RE.dm_pair(rv16, P16[(W, m, s)], P16[(Wb, m, s)])
            qa, qb = ql(W, m, s), ql(Wb, m, s)
            dms.append(dict(kind=kind, model=m, set=s, W=W, W_b=Wb, qlike=qa, qlike_b=qb, pct=100 * (qa / qb - 1), **r))
DM = pd.DataFrame(dms)
gaps = []
for W in WS:
    for e in EST:
        r = RE.dm_pair(rv16, P16[(W, e, "all_features")], P16[(W, e, "baseline")])
        qh, qf = ql(W, e, "baseline"), ql(W, e, "all_features")
        gaps.append(dict(W=W, model=e, qlike_har=qh, qlike_full=qf, pct=100 * (qf / qh - 1), **r))
GAP = pd.DataFrame(gaps)

# ---------------------------------------------------------------------------- W = 3,000 against the three-way CLOSE
chk: dict = {}
rq = pd.read_csv(REF / "qlike_16.csv")
rq = rq[rq.variant == "CLOSE"]
rg = pd.read_csv(REF / "gap.csv")
rg = rg[(rg.variant == "CLOSE") & (rg.target == "16:00")]
assert len(rq) == len(MODELS) and len(rg) == len(EST), (len(rq), len(rg))
for _, r in rq.iterrows():
    a = Q[(Q.W == W_REF) & (Q.model == r.model) & (Q.set == r.set)].iloc[0]
    d = {c: float(abs(a[c] - r[c])) for c in ("qlike", "mse_e10", "r2_oos_pct")}
    good = all(v <= 1e-12 * max(1.0, abs(float(r[c]))) for c, v in d.items()) and int(a.n_days) == int(r.n_days)
    chk[f"qlike16/{r.model}/{r.set}"] = dict(mine=float(a.qlike), ref=float(r.qlike), absdiff=d, pass_=bool(good))
for _, r in rg.iterrows():
    a = GAP[(GAP.W == W_REF) & (GAP.model == r.model)].iloc[0]
    good = (
        abs(a.pct - r.pct) <= 1e-9 and abs(a.dm_qlike - r.dm_qlike) <= 1e-9 and abs(a.dm_qlike_p - r.dm_qlike_p) <= 1e-9
    )
    chk[f"gap16/{r.model}"] = dict(
        mine=[float(a.pct), float(a.dm_qlike)], ref=[float(r.pct), float(r.dm_qlike)], pass_=bool(good)
    )
chk["pass"] = all(v["pass_"] for v in chk.values())
(OUT / "check.json").write_text(json.dumps(chk, indent=1))
print(f"CHECK (W = 3,000 = the three-way CLOSE): {'PASS' if chk['pass'] else 'FAIL'}", flush=True)

# ---------------------------------------------------------------------------- the 16:00 forecasts (economics, local)
f16 = []
for W in WS:
    for m, s in MODELS:
        g = G16[(W, m, s)]
        f16.append(
            pd.DataFrame(
                dict(
                    W=W,
                    model=m,
                    predictor_set=s,
                    day=g.index.to_numpy(),
                    plain=g["plain"].to_numpy(),
                    rv_actual=g["rv_actual"].to_numpy(),
                )
            )
        )
pd.concat(f16, ignore_index=True).to_parquet(OUT / "f16.parquet", index=False)
for name, df in (("qlike16", Q), ("dm", DM), ("gap", GAP)):
    df.to_csv(OUT / f"{name}.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
(OUT / "tables_qlike.txt").write_text(
    f"{FAM}: close-only training, 16:00, {len(hdays)} days\n\n{Q.to_string(index=False)}\n\n"
    f"DM\n{DM.to_string(index=False)}\n\nfull vs HAR\n{GAP.to_string(index=False)}\n\ncheck {json.dumps(chk)}\n"
)
print((OUT / "tables_qlike.txt").read_text())
if not chk["pass"]:
    sys.exit(3)
