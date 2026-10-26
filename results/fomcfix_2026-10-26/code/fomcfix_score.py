"""Scores of results/fomcfix_2026-10-26 (RULE.txt sections 4-6, 7(a), 7(b), addendum A), on the laptop.

Table 1's scoring (writeup/make_rehaul_oos_rerun.py: core, evaluation_days, scored_set, loss, daily, slot_benchmark,
imported) and the sign strategy's D6 (writeup/make_rehaul_economics.py: load, sharpe, post_warmup, the quote book's
straddle; writeup/hac_inference.d6_diff_test), as results/window_sweep_2014start_2026-10-22/code/score14.py and
results/pooled_trees_final_2014start_w3000/code/score_trees_final.py compute them.  Modules and the economic inputs come
from the main worktree (--main, read only; its *.py asserted unmodified against its HEAD).

Inputs (compact forecasts: model, predictor_set, date, pred_adj, B; plain = pred_adj^2 x B)
  --lin CELL=PATH   linear cells (A = the final run's w3000 forecasts.parquet; B, C, D = collect_linear.py output).  A
                    cell without HAR-benchmark rows (B) takes cell A's (RULE.txt section 2, after gate G3).
  --trees CELL=PATH tree cells: A = "templates" (results/pooled_trees_final_2014start_w3000 and
                    results/pooled_rf_skx_2014start_w3000 forecasts); B = the merged cell-B walks (full set; the HAR rows
                    the templates', section 6).
  --extra NAME=PATH|MODEL  a full-set arm of 7(a) / 7(b) (model column chosen by MODEL), scored against cell B's HAR
                    benchmark and cell B's full arm of the class.
Scored set: Table 1's 33,527 bars (cell A's linear panel, asserted), the same for every arm (asserted: every arm
forecasts every one of them); 16:00: the 2,579 days.

Outputs (--out): metrics.csv, dm.csv, d6.csv, effects.csv, materiality.json, gates.json, tables.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lin", action="append", default=[])
ap.add_argument("--trees", action="append", default=[])
ap.add_argument("--extra", action="append", default=[])
A_ = ap.parse_args()
MAIN = Path(A_.main).resolve()
HERE = Path(__file__).resolve().parents[3]
for p in (MAIN, MAIN / "notebooks", MAIN / "writeup", MAIN / "experiments"):
    sys.path.insert(0, str(p))
r = subprocess.run(["git", "-C", str(MAIN), "diff", "--quiet", "HEAD", "--", "*.py"])
assert r.returncode == 0, f"{MAIN}: *.py modified against HEAD"
MAIN_HEAD = subprocess.run(["git", "-C", str(MAIN), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

import hac_inference as hi  # noqa: E402
import make_rehaul_economics as E  # noqa: E402
import make_rehaul_oos_rerun as MR  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

for _m in (hi, E, MR):
    assert Path(_m.__file__).resolve().is_relative_to(MAIN), _m.__file__
OUT = Path(A_.out)
OUT.mkdir(parents=True, exist_ok=True)
FIRST, LAST, NDAYS = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"), 2579
CK = (pd.Timestamp("2020-01-03"), pd.Timestamp("2024-04-30"), 1082)
LIN = ("ridge", "lasso", "enet")
TREES = ("lgbm", "xgb", "rf")
SETS = ("baseline", "all_features")
W3000 = HERE / "results" / "window_sweep_2014start_2026-10-22" / "w3000"
TPL_TREES = HERE / "results" / "pooled_trees_final_2014start_w3000"
TPL_RF = HERE / "results" / "pooled_rf_skx_2014start_w3000"
T1 = HERE / "results" / "oos_final_2014start" / "table1_w3000"
PART2 = HERE / "results" / "window_sweep_2014start_2026-10-22" / "part2"

rv = MR.core()
days = MR.evaluation_days(rv, FIRST, LAST, NDAYS)


def frames(path: Path, models=None, rename=None) -> dict:
    """(model, set) -> frame by bar end: pred_adj, B, plain = pred_adj^2 x B, rv (core_stats)."""
    f = pd.read_parquet(path)
    if rename:
        f["model"] = f["model"].replace(rename)
    out = {}
    for (m, s), g in f.groupby(["model", "predictor_set"]):
        if models and m not in models:
            continue
        g = g.set_index(pd.DatetimeIndex(pd.to_datetime(g["date"])).as_unit("ns")).sort_index()
        assert not g.index.duplicated().any(), (path, m, s)
        plain = g["pred_adj"].to_numpy(float) ** 2 * g["B"].to_numpy(float)
        if "plain" in g.columns:  # the stored forecast is pred_adj^2 x B
            assert np.allclose(plain, g["plain"].to_numpy(float), rtol=1e-12, atol=0), (path, m, s)
            plain = g["plain"].to_numpy(float)
        out[(m, s)] = pd.DataFrame(
            {
                "pred_adj": g["pred_adj"].to_numpy(float),
                "B": g["B"].to_numpy(float),
                "plain": plain,
                "rv": rv.reindex(g.index).to_numpy(float),
            },
            index=g.index,
        )
    return out


def kv(xs: list[str]) -> dict:
    return dict(x.split("=", 1) for x in xs)


LINP = kv(A_.lin)
TRP = kv(A_.trees)
PANEL: dict = {}  # (cell, family, model, set) -> frame
for c, p in LINP.items():
    for (m, s), f in frames(Path(p)).items():
        PANEL[(c, "linear", m, s)] = f
for c in LINP:
    if c != "A" and not any(k[0] == c and k[3] == "baseline" for k in PANEL):
        for (c0, fam, m, s), f in list(PANEL.items()):
            if c0 == "A" and fam == "linear" and s == "baseline":
                PANEL[(c, fam, m, s)] = f  # cell B: the HAR benchmark rows and the OLS are cell A's (gate G3)
for c, p in TRP.items():
    if p == "templates":
        src = {
            **frames(TPL_TREES / "forecasts.parquet"),
            **frames(TPL_RF / "forecasts.parquet", rename={"rf_skx": "rf"}),
        }
    else:
        src = frames(Path(p), rename={"rf_skx": "rf"})
        tpl = {
            **frames(TPL_TREES / "forecasts.parquet"),
            **frames(TPL_RF / "forecasts.parquet", rename={"rf_skx": "rf"}),
        }
        for (m, s), f in tpl.items():
            if s == "baseline":
                src.setdefault((m, s), f)  # section 6: the HAR-benchmark trees are not rerun
    for (m, s), f in src.items():
        PANEL[(c, "tree", m, s)] = f
for e in A_.extra:
    name, rest = e.split("=", 1)
    path, model = rest.rsplit("|", 1)
    g = frames(Path(path))
    for (m, s), f in g.items():
        if m == model and s == "all_features":
            PANEL[(name, "extra", m, s)] = f

# the scored set: cell A's linear panel (Table 1's), the same bars for every arm
base = {(m, s): f for (c, fam, m, s), f in PANEL.items() if c == "A" and fam == "linear"}
idx = MR.scored_set(base, rv, days)
assert len(idx) == 33527, len(idx)
for k, f in PANEL.items():
    p = f["plain"].reindex(idx).to_numpy()
    assert np.isfinite(p).all() and (p > 0).all(), (k, int((~np.isfinite(p)).sum()))
on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
assert len(on16) == NDAYS
ck = on16[(on16.normalize() >= CK[0]) & (on16.normalize() <= CK[1])]
assert len(ck) == CK[2]
bench = MR.slot_benchmark(rv, idx)
y = rv.reindex(idx).to_numpy(float)
den = float(((y - bench.to_numpy()) ** 2).sum())
print(f"scored set: {len(idx)} bars on {len(days)} days; 16:00 {len(on16)}; arms {len(PANEL)}", flush=True)

LOSS, met = {}, []
for k, f in PANEL.items():
    q, se = MR.loss(f, idx, rv)
    q16, se16 = MR.loss(f, on16, rv)
    qc, _ = MR.loss(f, ck, rv)
    LOSS[k] = dict(bar=(q, se), c16=(q16, se16))
    met.append(
        dict(
            cell=k[0],
            family=k[1],
            model=k[2],
            set=k[3],
            n_bars=len(idx),
            n_days16=len(on16),
            qlike_bar=float(q.mean()),
            mse_bar_e10=float(se.mean() * 1e10),
            r2_oos_pct=float(100.0 * (1.0 - se.sum() / den)),
            qlike_16=float(q16.mean()),
            mse_16_e10=float(se16.mean() * 1e10),
            qlike_16_2020=float(qc.mean()),
        )
    )
M = pd.DataFrame(met)


def dm_pair(ka: tuple, kb: tuple, what: str) -> list[dict]:
    out = []
    for li, lname in ((0, "qlike"), (1, "mse")):
        for sp in ("bar", "c16"):
            la, lb = LOSS[ka][sp][li], LOSS[kb][sp][li]
            if sp == "bar":
                la, lb = MR.daily(la, idx), MR.daily(lb, idx)
            rr = dm_test(la, lb)
            out.append(
                dict(
                    cell=ka[0],
                    family=ka[1],
                    model=ka[2],
                    set=ka[3],
                    vs_cell=kb[0],
                    vs_set=kb[3],
                    comparison=what,
                    span=sp,
                    loss=lname,
                    mean_a=float(np.mean(la)),
                    mean_b=float(np.mean(lb)),
                    pct=float(100.0 * (np.mean(la) / np.mean(lb) - 1.0)),
                    dm=float(rr["dm"]),
                    p=float(rr["p"]),
                    mean_diff=float(rr["mean_diff"]),
                    T=int(rr["T"]),
                    hac_lag=int(rr["hac_lag"]),
                )
            )
    return out


dms = []
for c, fam, m, s in PANEL:
    if s == "all_features" and fam in ("linear", "tree"):
        dms += dm_pair((c, fam, m, s), (c, fam, m, "baseline"), "full vs HAR")
    if fam == "linear" and c not in ("A", "B", "C", "D") and m != "ols" and ("B", fam, m, s) in PANEL:
        # a re-tuned cell (RULE.txt 7(b)): each arm against the frozen arm of cell B
        dms += dm_pair((c, fam, m, s), ("B", fam, m, s), "vs frozen (cell B)")
    if fam == "extra":
        dms += dm_pair((c, fam, m, s), ("B", "linear", m, "baseline"), "full vs HAR")
        dms += dm_pair((c, fam, m, s), ("B", "linear", m, "all_features"), "vs frozen full (cell B)")
D = pd.DataFrame(dms)

# ---- D6, the sign strategy (make_rehaul_economics.add_pooled_linear / add_pooled_trees)
mt, model_set, series0, fam0 = E.load()
days866 = series0[E.OLS].index
pw = E.pw_days()
assert len(days866) == 866 and len(pw) == 614
dly = pd.read_parquet(E.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"])
dly["day"] = pd.to_datetime(dly["day"]).dt.normalize()
ref = dly[dly["key"] == E.OLS].set_index("day").sort_index()
ref.index = ref.index.astype(days866.dtype)
ref = ref.reindex(days866)
iv, rv866 = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
bo = series0[E.OLS]
RET = {}
for k, f in PANEL.items():
    t16 = f.index[(f.index.hour == 16) & (f.index.minute == 0)]
    a = f.loc[t16, ["plain", "rv"]]
    a.index = pd.DatetimeIndex(a.index.normalize()).astype(days866.dtype)
    a = a.reindex(days866)
    p = a["plain"].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all(), k
    assert np.allclose(a["rv"].to_numpy(float), rv866, rtol=1e-12, atol=0.0), k
    q = np.where(p > iv, 1.0, -1.0)
    RET[k] = dict(
        p=p,
        q=q,
        mid=q * bo["R"].to_numpy(float),
        x=E.gui._crossed_return(q, bo["ex"].to_numpy(), bo["bid"].to_numpy(), bo["ask"].to_numpy()),
    )
d6 = []
for k in PANEL:
    c, fam, m, s = k
    if s != "all_features":
        continue
    hk = (c, fam, m, "baseline") if fam != "extra" else ("B", "linear", m, "baseline")
    h, fu = RET[hk], RET[k]
    o = dict(cell=c, family=fam, model=m)
    for col, lab in (("mid", "mid"), ("x", "ba")):
        E.LW_CACHE.clear()
        if hasattr(E, "BOOT_CACHE"):
            E.BOOT_CACHE.clear()
        hp, fp_ = E.post_warmup(h[col]), E.post_warmup(fu[col])
        sh, sf = E.sharpe(h[col]), E.sharpe(fu[col])
        bt = hi.d6_diff_test(fp_, hp, pw)
        assert abs(bt["delta"] - (sf - sh)) < 1e-10
        o |= {f"d6_{lab}_har": sh, f"d6_{lab}_full": sf, f"dd6_{lab}": sf - sh, f"d6_boot_p_{lab}": bt["p"]}
    o["long_share_full"] = float((E.post_warmup(fu["q"]) > 0).mean())
    d6.append(o)
D6 = pd.DataFrame(d6)

# ---- gates S (cell A against the printed final numbers)
gates: dict = {"main_head": MAIN_HEAD, "checks": [], "fail": []}


def chk(name: str, got: float, want: float, tol: float) -> None:
    ok = bool(np.isclose(got, want, rtol=tol, atol=0)) if want != 0 else abs(got) <= tol
    gates["checks"].append(dict(name=name, got=got, want=want, rel=float(abs(got / want - 1)) if want else None, ok=ok))
    if not ok:
        gates["fail"].append(name)


if "A" in LINP:
    t1 = pd.read_csv(T1 / "metrics.csv")
    t1d = pd.read_csv(T1 / "dm.csv")
    gap = pd.read_csv(PART2 / "gap.csv")
    ql = pd.read_csv(PART2 / "qlike.csv")
    ec = pd.read_csv(PART2 / "econ.csv")
    for m in ("ols", *LIN):
        for s in SETS if m != "ols" else ("baseline",):
            r0 = M[(M.cell == "A") & (M.family == "linear") & (M.model == m) & (M.set == s)].iloc[0]
            w = t1[(t1.model == m) & (t1.set == s)].iloc[0]
            chk(f"A {m}/{s} qlike bar", r0.qlike_bar, w.qlike, 1e-12)
            chk(f"A {m}/{s} mse bar", r0.mse_bar_e10 * 1e-10, w.mse, 1e-12)
            chk(f"A {m}/{s} qlike 16:00 2020-24", r0.qlike_16_2020, w.qlike_1600, 1e-12)
            if m != "ols":
                q0 = ql[(ql.model == m) & (ql.set == s) & (ql.W == 3000)].iloc[0]
                chk(f"A {m}/{s} qlike 16:00 2014-24", r0.qlike_16, q0.qlike16, 1e-12)
    for m in LIN:
        dd = D[(D.cell == "A") & (D.family == "linear") & (D.model == m) & (D.loss == "qlike")]
        db, d16 = dd[dd.span == "bar"].iloc[0], dd[dd.span == "c16"].iloc[0]
        w = t1d[
            (t1d.model == m)
            & (t1d.set == "all_features")
            & (t1d.comparison == "all features vs HAR benchmark")
            & (t1d.loss == "qlike")
        ].iloc[0]
        chk(f"A {m} DM bar", db.dm, w.dm, 1e-9)
        chk(f"A {m} DM p bar", db.p, w.p, 1e-9)
        g = gap[(gap.model == m) & (gap.W == 3000)].iloc[0]
        chk(f"A {m} DM 16:00", d16.dm, g.gap_dm_16, 1e-9)
        chk(f"A {m} DM p 16:00", d16.p, g.gap_p_16, 1e-9)
        chk(f"A {m} gap % bar", db.pct, g.gap_pct_bar, 1e-9)
        e0 = D6[(D6.cell == "A") & (D6.family == "linear") & (D6.model == m)].iloc[0]
        we = ec[(ec.model == m) & (ec.W == 3000)].iloc[0]
        for col in ("d6_mid_har", "d6_mid_full", "d6_ba_har", "d6_ba_full", "d6_boot_p_mid", "d6_boot_p_ba"):
            chk(f"A {m} {col}", e0[col], we[col], 1e-9)
if "A" in TRP:
    for run, ests in ((TPL_TREES, ("lgbm", "xgb")), (TPL_RF, ("rf_skx",))):
        tm = pd.read_csv(run / "scores" / "metrics.csv")
        td = pd.read_csv(run / "scores" / "dm.csv")
        te = pd.read_csv(run / "scores" / "econ.csv")
        for e in ests:
            m = {"rf_skx": "rf"}.get(e, e)
            for s in SETS:
                r0 = M[(M.cell == "A") & (M.family == "tree") & (M.model == m) & (M.set == s)].iloc[0]
                w = tm[(tm.model == e) & (tm.set == s) & (tm.W == 3000)].iloc[0]
                chk(f"A {e}/{s} qlike bar", r0.qlike_bar, w.qlike, 1e-12)
                chk(f"A {e}/{s} qlike 16:00", r0.qlike_16, w.qlike_1600, 1e-12)
            for sp, bars in (("bar", "every bar"), ("c16", "16:00")):
                d0 = D[(D.cell == "A") & (D.family == "tree") & (D.model == m) & (D.loss == "qlike") & (D.span == sp)]
                w = td[
                    (td.model == e)
                    & (td.set == "all_features")
                    & (td.comparison == "all features vs HAR benchmark")
                    & (td.loss == "qlike")
                    & (td.bars == bars)
                    & (td.W == 3000)
                ].iloc[0]
                chk(f"A {e} DM {sp}", d0.iloc[0].dm, w.dm, 1e-9)
            e0 = D6[(D6.cell == "A") & (D6.family == "tree") & (D6.model == m)].iloc[0]
            we = te[(te.model == e) & (te.W == 3000)].iloc[0]
            for col in ("d6_mid_har", "d6_mid_full", "d6_ba_har", "d6_ba_full", "d6_boot_p_mid", "d6_boot_p_ba"):
                chk(f"A {e} {col}", e0[col], we[col], 1e-9)
gates["pass"] = not gates["fail"]


# ---- the summary per (cell, family, class): full and HAR QLIKE, % vs HAR, DM, D6
def summary(c: str, fam: str, m: str) -> dict:
    o: dict = dict(cell=c, family=fam, model=m)
    hk = (c, fam, m, "baseline") if fam != "extra" else ("B", "linear", m, "baseline")
    for sp, col in (("bar", "qlike_bar"), ("c16", "qlike_16")):
        fu = M[(M.cell == c) & (M.family == fam) & (M.model == m) & (M.set == "all_features")].iloc[0]
        ha = M[(M.cell == hk[0]) & (M.family == hk[1]) & (M.model == m) & (M.set == "baseline")].iloc[0]
        dd = D[(D.cell == c) & (D.family == fam) & (D.model == m) & (D.comparison == "full vs HAR") & (D.span == sp)]
        dq, dmse = dd[dd.loss == "qlike"].iloc[0], dd[dd.loss == "mse"].iloc[0]
        o |= {
            f"qlike_full_{sp}": fu[col],
            f"qlike_har_{sp}": ha[col],
            f"pct_vs_har_{sp}": dq.pct,
            f"dm_{sp}": dq.dm,
            f"p_{sp}": dq.p,
            f"mse_full_{sp}_e10": fu[col.replace("qlike", "mse") + "_e10"],
            f"mse_har_{sp}_e10": ha[col.replace("qlike", "mse") + "_e10"],
            f"dm_mse_{sp}": dmse.dm,
            f"p_mse_{sp}": dmse.p,
        }
    e = D6[(D6.cell == c) & (D6.family == fam) & (D6.model == m)].iloc[0]
    o |= {
        k: e[k]
        for k in (
            "d6_mid_full",
            "d6_mid_har",
            "dd6_mid",
            "d6_boot_p_mid",
            "d6_ba_full",
            "d6_ba_har",
            "dd6_ba",
            "d6_boot_p_ba",
        )
    }
    return o


keys = sorted({(c, fam, m) for (c, fam, m, s) in PANEL if s == "all_features"})
S = pd.DataFrame([summary(*k) for k in keys])
QTY = [c for c in S.columns if c not in ("cell", "family", "model")]


def eff(fam: str, m: str, a: str, b: str) -> dict | None:
    x, y_ = S[(S.cell == a) & (S.family == fam) & (S.model == m)], S[(S.cell == b) & (S.family == fam) & (S.model == m)]
    if x.empty or y_.empty:
        return None
    return {q: float(x.iloc[0][q] - y_.iloc[0][q]) for q in QTY}


effs = []
for fam, ms in (("linear", LIN), ("tree", TREES)):
    for m in ms:
        e = {k: eff(fam, m, *k.split("-")) for k in ("B-A", "D-C", "C-A", "D-B")}
        for k, v in e.items():
            if v is not None:
                effs.append(dict(family=fam, model=m, effect=k, **v))
        if e["D-C"] is not None and e["B-A"] is not None:
            effs.append(dict(family=fam, model=m, effect="(D-C)-(B-A)", **{q: e["D-C"][q] - e["B-A"][q] for q in QTY}))
EF = pd.DataFrame(effs)


def material(new: str, old: str, fam: str = "linear") -> dict:
    out: dict = {"cells": f"{new} vs {old}", "classes": {}, "material": False}
    for m in LIN if fam == "linear" else TREES:
        a = S[(S.cell == old) & (S.family == fam) & (S.model == m)]
        b = S[(S.cell == new) & (S.family == fam) & (S.model == m)]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        r: dict = {}
        for sp in ("bar", "c16"):
            r[f"sign_flip_{sp}"] = bool(np.sign(a[f"dm_{sp}"]) != np.sign(b[f"dm_{sp}"]))
            r[f"cross_5pct_{sp}"] = bool((a[f"p_{sp}"] < 0.05) != (b[f"p_{sp}"] < 0.05))
            r[f"qlike_full_change_pct_{sp}"] = float(100.0 * (b[f"qlike_full_{sp}"] / a[f"qlike_full_{sp}"] - 1.0))
            r[f"qlike_change_gt_0.5pct_{sp}"] = bool(abs(r[f"qlike_full_change_pct_{sp}"]) > 0.5)
        r["d6_mid_full_change"] = float(b["d6_mid_full"] - a["d6_mid_full"])
        r["d6_ba_full_change"] = float(b["d6_ba_full"] - a["d6_ba_full"])
        r["d6_change_gt_0.10"] = bool(abs(r["d6_mid_full_change"]) > 0.10 or abs(r["d6_ba_full_change"]) > 0.10)
        r["material"] = any(v for k, v in r.items() if isinstance(v, bool))
        out["classes"][m] = r
        out["material"] = out["material"] or r["material"]
    return out


mat = {
    "fomc_B_vs_A": material("B", "A"),
    "fomc_D_vs_C": material("D", "C"),
    "holiday_C_vs_A_information": material("C", "A"),
    "holiday_D_vs_B_information": material("D", "B"),
    "trees_B_vs_A_information": material("B", "A", "tree"),
}
M.to_csv(OUT / "metrics.csv", index=False)
D.to_csv(OUT / "dm.csv", index=False)
D6.to_csv(OUT / "d6.csv", index=False)
S.to_csv(OUT / "summary.csv", index=False)
EF.to_csv(OUT / "effects.csv", index=False)
(OUT / "materiality.json").write_text(json.dumps(mat, indent=1))
(OUT / "gates.json").write_text(json.dumps(gates, indent=1, default=float))

# ---- tables
NAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
T = [
    f"# fomcfix scores (main worktree {MAIN_HEAD[:7]}); Table 1's {len(idx):,} bars on {len(days):,} days, 16:00 on "
    f"{len(on16):,} days",
    "",
    "QLIKE full / HAR, % vs HAR = 100 (full / HAR - 1), DM (HLN) full vs HAR (negative = full lower) with p; D6 Sharpe "
    "(614 post-warmup option days) full / HAR mid, bid/ask full, bootstrap p of full - HAR (mid).",
    "",
    "| family | class | cell | QLIKE full bar | HAR bar | % bar | DM bar (p) | QLIKE full 16:00 | HAR 16:00 | % 16:00 | "
    "DM 16:00 (p) | D6 full / HAR mid | D6 b/a full / HAR | ΔD6 mid (p) |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for _, x in S.sort_values(["family", "model", "cell"]).iterrows():
    T.append(
        f"| {x.family} | {NAME.get(x.model, x.model)} | {x.cell} | {x.qlike_full_bar:.5f} | {x.qlike_har_bar:.5f} | "
        f"{x.pct_vs_har_bar:+.2f} | {x.dm_bar:+.2f} ({x.p_bar:.3f}) | {x.qlike_full_c16:.5f} | {x.qlike_har_c16:.5f} | "
        f"{x.pct_vs_har_c16:+.2f} | {x.dm_c16:+.2f} ({x.p_c16:.3f}) | {x.d6_mid_full:.2f} / {x.d6_mid_har:.2f} | "
        f"{x.d6_ba_full:.2f} / {x.d6_ba_har:.2f} | {x.dd6_mid:+.2f} ({x.d6_boot_p_mid:.3f}) |"
    )
T += [
    "",
    "## Effects (differences of the quantities above)",
    "",
    "| family | class | effect | ΔQLIKE full bar | Δ% vs HAR bar | ΔDM bar | ΔQLIKE full 16:00 | Δ% vs HAR 16:00 | "
    "ΔDM 16:00 | ΔD6 mid full | ΔD6 mid HAR |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for _, x in EF.iterrows():
    T.append(
        f"| {x.family} | {NAME.get(x.model, x.model)} | {x.effect} | {x.qlike_full_bar:+.6f} | {x.pct_vs_har_bar:+.3f} | "
        f"{x.dm_bar:+.3f} | {x.qlike_full_c16:+.6f} | {x.pct_vs_har_c16:+.3f} | {x.dm_c16:+.3f} | "
        f"{x.d6_mid_full:+.3f} | {x.d6_mid_har:+.3f} |"
    )
T += [
    "",
    "## Materiality (RULE.txt section 5)",
    "",
    "```",
    json.dumps(mat, indent=1),
    "```",
    "",
    f"## Gates S: {'PASS' if gates['pass'] else 'FAIL ' + str(gates['fail'])} ({len(gates['checks'])} checks)",
]
(OUT / "tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
print("\n".join(T))
sys.exit(0 if gates["pass"] else 3)
