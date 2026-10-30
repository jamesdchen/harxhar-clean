"""Part A scorer of the transform ablation (results/transform_ablation_2026-10-30/RULE.txt sections 4-7, addenda D-E):
every design variant's walk forecasts against the incumbent V0, on the laptop.  Written 2026-10-30 before any variant
forecast exists.

Inputs (--runs, default results/transform_ablation_2026-10-30/runs): one directory a variant, named by its key
(V1 .. V10, V3f; a suffix _frz = the walk with the incumbent's frozen penalties, RULE.txt 6(b), same inverse map):
  <key>/forecasts*.parquet           confirmation forecasts (the linear driver's score stage writes forecasts.parquet;
                                     the LightGBM merge any other name starting with "forecasts"), columns model,
                                     predictor_set, date, pred_adj, B; rows from 2014-01-02 are scored
  <key>/forecasts_sel2013*.parquet   selection forecasts of 2013 (the LightGBM 2013 walk merged), same columns
  <key>/_work/pretune/<set>/<model>  the linear pre-tune of fold 3 (fold.npz, ext_*/fold.npz) with
                                     <key>/frozen_penalties.json and <key>/_work/design_<set> (y.npy, b_oos.npy,
                                     dates_oos.csv): the validation forecasts of 2013 at the selected penalty,
                                     pred_adj = y + resid (pool_linear_pretune.py pretune stores resid = pred - y)
  <key>/_work/design_<set>/design.json  when present, its config.ex_flags must hold the key's flags
                                     (experiments/transform_design_gate.py VARIANTS)
V0 is the paper's final run (the Table 1 files of experiments/transform_backtransform.py FILES) unless <runs>/V0 holds
forecasts.

Inverse maps (RULE.txt section 2): sqrt pred^2 x B; log (V3, V3f) exp(pred) x B; level (V8, V9) max(pred, 0.01) x B,
the floored bars counted.  Scorings: P = the inverse map; C = c_d x P with c_d the trailing mean of RV / P over the 250
sessions before d (lagged one session; identity with fewer than 20; the c of transform_backtransform.py).
Confirmation: Table 1's 33,527 bars (asserted; the bar set of V0's 13 forecasts), spans all / ord / ev / c16 /
c16_2020 (asserted counts); QLIKE and squared error on actual RV; DM of each variant against V0 within the class
(dm_test: HLN, daily average loss on the every-bar spans, bar losses on the 16:00 spans); the verdicts of section 7
(span all, P and C); the MCS (model_confidence_set, resample=True, B 10,000, seed 20261030, daily average QLIKE of
the 2,579 days) per class and scoring over the primary family V0 .. V8 + V3f (blocks 10 at alpha 0.10 and 0.25, 5 and
21 at 0.10) and the secondary family (every own-penalty variant present).
Selection (2013): QLIKE (P) on the 2013 bars (RV > 0; days with a positive 16:00 print, early closes dropped; the bars
every variant of the class forecasts), DM against V0 where V0 has 2013 forecasts, and the LightGBM pick of section 6:
the three variants of the primary family other than V0 with the lowest 2013 QLIKE of lgbm/full.

usage: python experiments/transform_score.py [--runs D] [--out results/transform_ablation_2026-10-30/scores]
                                             [--variants V1,V3f,...] [--mcs-B 10000]
       -> metrics.csv, dm.csv, verdicts.csv, mcs.csv, sel2013_metrics.csv, sel2013_dm.csv, selection.json, checks.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_score as AS  # noqa: E402
import transform_backtransform as TB  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402
from src.evaluation.model_confidence_set import model_confidence_set  # noqa: E402
from transform_design_gate import VARIANTS  # noqa: E402

RUNS = ROOT / "results" / "transform_ablation_2026-10-30" / "runs"
OUT = ROOT / "results" / "transform_ablation_2026-10-30" / "scores"
FIRST = TB.FIRST
SEL_FIRST, SEL_LAST = pd.Timestamp("2013-01-02"), pd.Timestamp("2013-12-31")
LEVEL_FLOOR = 0.01
L_SESS, MIN_HIST = TB.L_SESS, TB.MIN_HIST
PRIMARY = ("V0", "V1", "V2", "V3", "V3f", "V4", "V5", "V6", "V7", "V8")
LOO = tuple(v for v in PRIMARY if v != "V0")
MCS_SEED, MCS_RUNS = 20261030, ((10, 0.10), (10, 0.25), (5, 0.10), (21, 0.10))
EVERY_BAR = ("all", "ord", "ev")
KEY = re.compile(r"^(V\d+f?)(_frz)?$")
SET_ARM = TB.SET_ARM
EST_OF = {"ridge": "ridge", "lasso": "lasso", "enet": "enet"}


def say(msg: str) -> None:
    print(msg, flush=True)


def base_key(key: str) -> str:
    m = KEY.match(key)
    assert m and m.group(1) in VARIANTS, key
    return m.group(1)


def inverse_kind(key: str) -> str:
    fl = VARIANTS[base_key(key)][1]
    assert not (fl.get("TARGET_LOG") and fl.get("TARGET_LEVEL")), key
    return "log" if fl.get("TARGET_LOG") else ("level" if fl.get("TARGET_LEVEL") else "sqrt")


def to_rv(pred: np.ndarray, B: np.ndarray, kind: str) -> tuple[np.ndarray, int]:
    """(forecast of RV, bars floored) under the variant's inverse map."""
    if kind == "sqrt":
        return pred**2 * B, 0
    if kind == "log":
        return np.exp(pred) * B, 0
    low = np.isfinite(pred) & (pred < LEVEL_FLOOR)
    return np.maximum(pred, LEVEL_FLOOR) * B, int(low.sum())


def scalar_c(f: np.ndarray, rv: np.ndarray, sess: np.ndarray, n_sess: int) -> np.ndarray:
    """c_d x f with c_d = mean(RV / f) over the 250 sessions before d (identity with < MIN_HIST sessions): the c of
    transform_backtransform.corrections for any forecast f (tested equal there for f = pred^2 B)."""
    u = np.isfinite(f) & (f > 0) & np.isfinite(rv) & (rv > 0)
    r = np.where(u, rv / np.where(u, f, 1.0), 0.0)
    cn = np.concatenate([[0.0], np.cumsum(np.bincount(sess[u], minlength=n_sess))])
    cr = np.concatenate([[0.0], np.cumsum(np.bincount(sess[u], weights=r[u], minlength=n_sess))])
    nW = TB._window(cn, sess, L_SESS)
    with np.errstate(invalid="ignore", divide="ignore"):
        cW = TB._window(cr, sess, L_SESS) / nW
    on = np.minimum(np.arange(n_sess), L_SESS)[sess] >= MIN_HIST
    return np.where(on & (nW > 0), cW, 1.0) * f


def read_forecasts(paths: list[Path]) -> dict[tuple[str, str], pd.DataFrame]:
    """(class, set) -> frame indexed by bar end with pred, B (sorted, unique)."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for p in paths:
        f = pd.read_parquet(p, columns=["model", "predictor_set", "date", "pred_adj", "B"])
        for (m, s), g in f.groupby(["model", "predictor_set"]):
            k = ({"rf_skx": "rf"}.get(str(m), str(m)), SET_ARM[str(s)])
            g = g.sort_values("date")
            idx = pd.DatetimeIndex(g["date"])
            assert idx.is_unique, (p, k)
            assert k not in out, (p, k, "twice")
            out[k] = pd.DataFrame({"pred": g["pred_adj"].to_numpy(float), "B": g["B"].to_numpy(float)}, index=idx)
    return out


def pretune_sel(run: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """The linear fold-3 validation forecasts at each model's selected penalty (frozen_penalties.json), from the
    pre-tune outputs and the design caches of the run (module docstring)."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    fz = run / "frozen_penalties.json"
    if not fz.is_file():
        return out
    fr = json.loads(fz.read_text())
    for s in ("baseline", "all_features"):
        dz = run / "_work" / f"design_{s}"
        if not (dz / "y.npy").is_file():
            continue
        y = np.load(dz / "y.npy")
        b = np.load(dz / "b_oos.npy")
        d = pd.DatetimeIndex(pd.read_csv(dz / "dates_oos.csv", parse_dates=["date"])["date"])
        W = len(y) - len(d)
        for m in ("ridge", "lasso", "enet"):
            base = run / "_work" / "pretune" / s / m
            zs = [np.load(p) for p in [base / "fold.npz", *sorted(base.glob("ext_*/fold.npz"))] if p.is_file()]
            if not zs or fr.get(m, {}).get(s) is None:
                continue
            grid = np.concatenate([z["grid"] for z in zs])
            res = np.concatenate([z["resid"] for z in zs], axis=1)  # (folds, grid, rows)
            j = np.flatnonzero(np.isclose(grid, float(fr[m][s]), rtol=1e-12, atol=0))
            assert len(j) == 1, (run, s, m, fr[m][s], grid)
            v0, v1 = (int(x) for x in zs[0]["val"][-1])  # the last fold (fold 3 under --first-fold 3)
            rows = np.arange(v0, v1)
            pred = y[rows] + res[-1, j[0], :]
            out[(m, SET_ARM[s])] = pd.DataFrame({"pred": pred, "B": b[rows - W]}, index=d[rows - W])
    return out


def design_flags_ok(run: Path, key: str) -> list[str]:
    """Each design.json of the run holds the key's flags in config.ex_flags (empty list = fine or no design)."""
    bad = []
    for dj in sorted((run / "_work").glob("design_*/design.json")) if (run / "_work").is_dir() else []:
        fl = json.loads(dj.read_text()).get("config", {}).get("ex_flags", {})
        want = VARIANTS[base_key(key)][1]
        if any(fl.get(k) != v for k, v in want.items()):
            bad.append(f"{key}: {dj.parent.name} ex_flags {fl} lack {want}")
        extra = {k for k in fl if k.startswith(("TARGET_", "EXOG_", "FEATURE_WINSOR", "PRESCALE"))} - set(want)
        if any(fl[k] != getattr_default(k) for k in extra):
            bad.append(f"{key}: {dj.parent.name} sets switches beyond the variant's: { {k: fl[k] for k in extra} }")
    return bad


def getattr_default(name: str):
    import src.backtest.executor as EX

    return getattr(EX, name)


def eval_days_between(rv: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DatetimeIndex:
    """ablate_score.evaluation_days' rule on [lo, hi]: days with a positive 16:00 print, early closes dropped."""
    import atm_straddle_lib as asl

    early = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(early)
    days = d[keep]
    return days[(days >= lo) & (days <= hi)]


def dm_rows(x1, x2, ix, every_bar: bool) -> dict:
    r = dm_test(AS.daily(x1, ix), AS.daily(x2, ix)) if every_bar else dm_test(x1, x2)
    return dict(dm=float(r["dm"]), p=float(r["p"]), mean_diff=float(r["mean_diff"]), T=int(r["T"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variants", default="", help="comma-separated keys; default every key directory under --runs")
    ap.add_argument("--mcs-B", type=int, default=10_000, help="bootstrap replicates of the MCS (RULE.txt: 10,000)")
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    checks: dict = {"fail": [], "rule": "results/transform_ablation_2026-10-30/RULE.txt sections 4-7, addenda D-E"}
    rv_s = AS.core_rv()
    days = AS.evaluation_days(rv_s)
    ev = AS.event_days(days)
    # ---------------------------------------------------------------- the panel: (key, class, set) -> pred, B
    keys = [k for k in a.variants.split(",") if k] or sorted(
        p.name for p in runs.glob("*") if p.is_dir() and KEY.match(p.name)
    )
    keys = ["V0", *[k for k in keys if k != "V0"]]
    conf: dict[str, dict] = {}
    sel: dict[str, dict] = {}
    for k in keys:
        run = runs / k
        base_key(k)
        fc = sorted(p for p in run.glob("forecasts*.parquet") if not p.name.startswith("forecasts_sel2013"))
        if k == "V0" and not fc:
            fc = list(TB.FILES)  # the paper's final run
        if not fc:
            checks["fail"].append(f"{k}: no forecasts")
            continue
        conf[k] = read_forecasts(fc)
        sel[k] = {**read_forecasts(sorted(run.glob("forecasts_sel2013*.parquet"))), **pretune_sel(run)}
        checks["fail"] += design_flags_ok(run, k)
    assert "V0" in conf and len(conf["V0"]) == 13, sorted(conf.get("V0", {}))
    # ---------------------------------------------------------------- Table 1's bars and the spans
    v0plain = {
        c: pd.Series(f["pred"].to_numpy() ** 2 * f["B"].to_numpy(), index=f.index) for c, f in conf["V0"].items()
    }
    idx = AS.scored(v0plain, rv_s, days)
    assert len(idx) == TB.N_SCORED, len(idx)
    on_ev = pd.DatetimeIndex(idx.normalize()).isin(ev)
    c16 = (idx.hour == 16) & (idx.minute == 0)
    spans = {
        "all": np.ones(len(idx), bool),
        "ord": ~on_ev,
        "ev": on_ev,
        "c16": c16,
        "c16_2020": c16 & (idx.normalize() >= AS.SUB_FIRST),
    }
    counts = {k: (int(v.sum()), int(pd.DatetimeIndex(idx[v].normalize()).nunique())) for k, v in spans.items()}
    assert counts == {
        "all": (TB.N_SCORED, AS.N_DAYS),
        "ord": (TB.N_SCORED - counts["ev"][0], AS.N_DAYS - AS.N_EVENT),
        "ev": (counts["ev"][0], AS.N_EVENT),
        "c16": (AS.N_DAYS, AS.N_DAYS),
        "c16_2020": (AS.SUB_DAYS, AS.SUB_DAYS),
    }, counts
    checks["spans"] = {k: dict(bars=b, days=d) for k, (b, d) in counts.items()}
    rv_idx = rv_s.reindex(idx).to_numpy(float)
    # ---------------------------------------------------------------- forecasts on the bars, both scorings
    F: dict[tuple[str, str, str, str], np.ndarray] = {}  # (key, cls, set, scoring) -> forecast on idx
    floored: dict[str, int] = {}
    for k, panel in conf.items():
        kind = inverse_kind(k)
        for (cls, arm), f in panel.items():
            f = f[f.index >= FIRST]
            ref = conf["V0"].get((cls, arm))
            if ref is not None and not f.index.equals(ref[ref.index >= FIRST].index):
                checks["fail"].append(f"{k} {cls}/{arm}: bar index differs from V0's (2014-01-02 on)")
                continue
            p, nfl = to_rv(f["pred"].to_numpy(), f["B"].to_numpy(), kind)
            floored[f"{k}/{cls}/{arm}"] = nfl
            sd = pd.DatetimeIndex(f.index.normalize())
            sessions = sd.unique()
            cfc = scalar_c(p, rv_s.reindex(f.index).to_numpy(float), sessions.get_indexer(sd), len(sessions))
            pos = f.index.get_indexer(idx)
            if (pos < 0).any():
                checks["fail"].append(f"{k} {cls}/{arm}: {int((pos < 0).sum())} of Table 1's bars missing")
                continue
            for sc, v in (("P", p), ("C", cfc)):
                x = v[pos]
                if not (np.isfinite(x) & (x > 0)).all():
                    checks["fail"].append(f"{k} {cls}/{arm} {sc}: non-positive or non-finite forecasts on scored bars")
                    continue
                F[(k, cls, arm, sc)] = x
    checks["level_floored_bars"] = {k: v for k, v in floored.items() if v}

    def ql(x: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return TB.qlike_se(x[msk], rv_idx[msk])

    met, dms = [], []
    for (k, cls, arm, sc), x in sorted(F.items()):
        for sp, msk in spans.items():
            q, se = ql(x, msk)
            q0 = ql(F[("V0", cls, arm, sc)], msk)[0] if ("V0", cls, arm, sc) in F else None
            met.append(
                dict(
                    key=k,
                    variant=base_key(k),
                    frz=k.endswith("_frz"),
                    inverse=inverse_kind(k),
                    cls=cls,
                    set=arm,
                    scoring=sc,
                    span=sp,
                    n_bars=int(msk.sum()),
                    qlike=float(q.mean()),
                    mse=float(se.mean()),
                    qlike_pct_vs_v0=np.nan if q0 is None else 100.0 * (q.mean() / q0.mean() - 1.0),
                )
            )
            if k == "V0" or q0 is None:
                continue
            se0 = ql(F[("V0", cls, arm, sc)], msk)[1]
            for lname, (x1, x2) in (("qlike", (q, q0)), ("mse", (se, se0))):
                dms.append(
                    dict(key=k, cls=cls, set=arm, scoring=sc, span=sp, loss=lname, vs="V0")
                    | dm_rows(x1, x2, idx[msk], sp in EVERY_BAR)
                )
    M, D = pd.DataFrame(met), pd.DataFrame(dms)
    # gate: V0's plain rows reproduce Table 1 (QLIKE, MSE on all; the 16:00 QLIKE on 2020-2024) when V0 = the paper's
    t1 = pd.read_csv(TB.TABLE1)
    n_t1 = 0
    for _, r in t1.iterrows():
        mine = M[
            (M["key"] == "V0") & (M["cls"] == r["model"]) & (M["set"] == SET_ARM[r["set"]]) & (M["scoring"] == "P")
        ]
        mine = mine.set_index("span")
        for col, sp, ref in (
            ("qlike", "all", r["qlike"]),
            ("mse", "all", r["mse"]),
            ("qlike", "c16_2020", r["qlike_1600"]),
        ):
            n_t1 += 1
            if not np.isclose(float(mine.loc[sp, col]), ref, rtol=1e-12, atol=0):
                checks["fail"].append(
                    f"V0 vs Table 1 {r['model']}/{r['set']} {col} {sp}: {mine.loc[sp, col]!r} vs {ref!r}"
                )
    checks["table1_values_checked"] = n_t1
    # ---------------------------------------------------------------- verdicts (section 7)
    ver = []
    if len(D):
        q = D[(D["loss"] == "qlike") & (D["span"] == "all")].set_index(["key", "cls", "set", "scoring"])
        mm = M[M["span"] == "all"].set_index(["key", "cls", "set", "scoring"])
        for k in [k for k in conf if base_key(k) in LOO or base_key(k) in ("V9", "V10")]:
            for cls, arm in sorted(conf[k]):
                if not all((k, cls, arm, sc) in q.index for sc in ("P", "C")):
                    continue
                row = dict(key=k, cls=cls, set=arm)
                sig = {}
                for sc in ("P", "C"):
                    d = q.loc[(k, cls, arm, sc)]
                    qv, qv0 = float(mm.loc[(k, cls, arm, sc), "qlike"]), float(mm.loc[("V0", cls, arm, sc), "qlike"])
                    row[f"v0_vs_variant_pct_{sc}"] = 100.0 * (qv0 / qv - 1.0)  # < 0: V0 (the transform on) is better
                    row[f"dm_{sc}"], row[f"p_{sc}"] = float(d["dm"]), float(d["p"])
                    sig[sc] = 1 if (d["p"] < 0.05 and d["mean_diff"] > 0) else (-1 if d["p"] < 0.05 else 0)
                row["verdict"] = (
                    "helps"
                    if sig["P"] == sig["C"] == 1
                    else ("hurts" if sig["P"] == sig["C"] == -1 else "no robust effect")
                )
                row["verdict_applies"] = base_key(k) in LOO and not k.endswith("_frz")
                ver.append(row)
    V = pd.DataFrame(ver)
    # ---------------------------------------------------------------- MCS (section 4)
    mcs = []
    classes = sorted({(c, s) for (_k, c, s, _sc) in F})
    for cls, arm in classes:
        for sc in ("P", "C"):
            own = [k for k in conf if not k.endswith("_frz") and (k, cls, arm, sc) in F]
            fams = {"primary": [k for k in own if base_key(k) in PRIMARY], "secondary": own}
            for fam, members in fams.items():
                if len(members) < 2 or (fam == "secondary" and members == fams["primary"]):
                    continue
                Ld = {k: AS.daily(ql(F[(k, cls, arm, sc)], spans["all"])[0], idx) for k in members}
                for block, alpha in MCS_RUNS:
                    r = model_confidence_set(Ld, alpha=alpha, B=a.mcs_B, block=block, seed=MCS_SEED, resample=True)
                    mcs.append(
                        dict(
                            cls=cls,
                            set=arm,
                            scoring=sc,
                            family=fam,
                            block=block,
                            alpha=alpha,
                            B=a.mcs_B,
                            T=int(r["T"]),
                            members=",".join(members),
                            mcs=",".join(r["mcs"]),
                            v0_in="V0" in r["mcs"],
                            eliminated=",".join(r["eliminated"]),
                            pvals=json.dumps({k: round(float(v), 6) for k, v in r["pvals"].items()}),
                        )
                    )
    # ---------------------------------------------------------------- selection (2013)
    sel_days = eval_days_between(rv_s, SEL_FIRST, SEL_LAST)
    smet, sdm, pick = [], [], {}
    for cls, arm in sorted({c for p in sel.values() for c in p}):
        have = {k: p[(cls, arm)] for k, p in sel.items() if (cls, arm) in p}
        common = None
        for k, f in have.items():
            fx, _ = to_rv(f["pred"].to_numpy(), f["B"].to_numpy(), inverse_kind(k))
            ok = f.index[np.isfinite(fx) & (fx > 0)]
            common = ok if common is None else common.intersection(ok)
        assert common is not None
        common = common[pd.DatetimeIndex(common.normalize()).isin(sel_days)]
        t_ = rv_s.reindex(common).to_numpy(float)
        common = common[(t_ > 0) & np.isfinite(t_)].sort_values()
        y = rv_s.reindex(common).to_numpy(float)
        L13 = {}
        for k, f in have.items():
            fx, _ = to_rv(f["pred"].to_numpy(), f["B"].to_numpy(), inverse_kind(k))
            fx = pd.Series(fx, index=f.index).reindex(common).to_numpy()
            L13[k] = TB.qlike_se(fx, y)[0]
            smet.append(
                dict(
                    key=k,
                    cls=cls,
                    set=arm,
                    n_bars=len(common),
                    n_days=int(pd.DatetimeIndex(common.normalize()).nunique()),
                    qlike=float(L13[k].mean()),
                )
            )
        if "V0" in L13:
            for k in L13:
                if k != "V0":
                    sdm.append(dict(key=k, cls=cls, set=arm, vs="V0") | dm_rows(L13[k], L13["V0"], common, True))
        if (cls, arm) == ("lgbm", "full"):
            ranked = sorted(
                (float(v.mean()), k) for k, v in L13.items() if base_key(k) in LOO and not k.endswith("_frz")
            )
            pick = dict(
                rule="RULE.txt section 6: the three variants of the primary family other than V0 with the lowest 2013 "
                "QLIKE (plain) of lgbm/full",
                ranking=[dict(key=k, qlike=q_) for q_, k in ranked],
                chosen=[k for _q, k in ranked[:3]],
                n_bars=len(common),
            )
    checks["sel2013_days"] = len(sel_days)
    checks["pass"] = not checks["fail"]
    out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "metrics.csv", index=False)
    D.to_csv(out / "dm.csv", index=False)
    V.to_csv(out / "verdicts.csv", index=False)
    pd.DataFrame(mcs).to_csv(out / "mcs.csv", index=False)
    pd.DataFrame(smet).to_csv(out / "sel2013_metrics.csv", index=False)
    pd.DataFrame(sdm).to_csv(out / "sel2013_dm.csv", index=False)
    (out / "selection.json").write_text(json.dumps(pick, indent=1))
    checks["keys"] = list(conf)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    say(
        f"scored {len(conf)} variant(s) {list(conf)} on {len(idx)} bars; checks "
        f"{'PASS' if checks['pass'] else 'FAIL'} {checks['fail'][:5]}"
    )
    return 0 if checks["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
