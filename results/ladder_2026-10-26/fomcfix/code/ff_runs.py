"""The k-ladder (W = 3,000) and the window x feature-count grid on the FOMC-PATCHED design (RULE.txt ADDENDUM 3).

Reuses the ladder's code (results/ladder_2026-10-26/code, staged next to this file) with the window, the design
directory, the work directory and (linear) the fomcfix code tree set per call.  Everything else is the ladder's.

    python ff_runs.py gate-design --wdays W                     gate D (patched vs pre-patch design of that W)
    python ff_runs.py rank-trees  --wdays W --est lgbm|xgb|rf_skx
    python ff_runs.py run-trees   --wdays W --est E --part i/n [--secondary 1]
    python ff_runs.py rank-linear --wdays W
    python ff_runs.py run-linear  --wdays W --model ridge|lasso|enet [--secondary 1] [--workers N]
-> FF_WORK/w<W>/{gate_design.json, rank_*.json, trees/<est>/k###.npz, linear/<model>/<rung>/k###.npz}
The linear full rung also stores theta and the scales of the walk state at each origin (gamma for the patched SHAP
study's M2: ridge theta x s, lasso / elastic net theta; fi_by_window's fiw_replay.state / fiw_reduce.gam_of).
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
LADDER_CODE = Path(os.environ.get("LADDER_CODE", str(HERE.parents[1] / "code")))
sys.path.insert(0, str(LADDER_CODE))
import ladder_common as C  # noqa: E402

FF_WORK = Path(os.environ.get("FF_WORK", "/home1/jc_905/ladder_fomcfix/work"))
FF_CODE = Path(os.environ.get("FF_CODE", "/home1/jc_905/fomcfix_lin/code"))  # the fomcfix code tree (linear)
PATCHED = {
    3000: "/home1/jc_905/fomcfix_lin/runs/B/_work/design_all_features",
    1000: "/home1/jc_905/ladder_fomcfix/designs/w1000/_work/design_all_features",
    2000: "/home1/jc_905/ladder_fomcfix/designs/w2000/_work/design_all_features",
    3500: "/home1/jc_905/ladder_fomcfix/designs/w3500/_work/design_all_features",
}
PREPATCH = "/scratch1/jc_905/fi_window/w{W}/design_all_features"  # X md5 of RULE addendum 1 (3)
FOMC_FLAGS = ["FOMC_SCHEDULE_2020_FIX=1", "FOMC_PROXIMITY_NO_WINSOR=1"]
FOMC_PROX = ("fomc_until_inv", "fomc_since_inv")
FROZEN_RULE = {
    1000: {"ridge": 1e4, "lasso": 1e-3, "enet": 1e-2},
    2000: {"ridge": 1e3, "lasso": 1e-4, "enet": 1e-3},
    3000: {"ridge": 1e3, "lasso": 1e-4, "enet": 1e-3},
    3500: {"ridge": 1e3, "lasso": 1e-4, "enet": 1e-3},
}


def set_window(wd: int) -> Path:
    d = Path(PATCHED[wd])
    C.W_DAYS, C.W = wd, wd * C.PPD
    C.TREE_D = C.LIN_D = d
    C.WORK = FF_WORK / f"w{wd}"
    C.WORK.mkdir(parents=True, exist_ok=True)
    C.FROZEN = dict(FROZEN_RULE[wd])
    os.environ["LADDER_WORK"] = str(C.WORK)
    sec = os.environ.get("FF_SECONDARY", "")
    if sec:  # the patched secondary (look-ahead) importances (RULE addendum 3 (2))
        imp_all = json.loads(Path(sec).read_text())
        C.secondary_order = lambda model, tie: C.order_from_importance(imp_all[model], tie)  # type: ignore[assignment]
    return d


def origins(wd: int) -> pd.DataFrame:
    f = C.HERE / "fiw_origins.csv"
    assert hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == C.ORIGINS_SHA256
    o = pd.read_csv(f)
    dd = C.oos_dates(C.TREE_D)
    assert len(dd) == C.N_ROWS - C.W, (len(dd), C.N_ROWS - C.W)
    o["oos_row"] = o[f"oos_row_w{wd}"].astype(int)
    o["r"] = C.W + o["oos_row"]
    assert (o["r"] == 39000 + o["oos_row_w3000"]).all()
    o["r_end"] = list(o["r"].iloc[1:]) + [C.N_ROWS]
    assert (pd.DatetimeIndex(dd[o["oos_row"]]) == pd.to_datetime(o["origin"])).all()
    assert (o["r_end"] - o["r"]).sum() == 33970 and o["r"].iloc[0] == C.R_FIRST
    return o[["k", "origin", "oos_row", "r", "r_end"]].reset_index(drop=True)


def sets_for(model: str, order: list[str], secondary: bool) -> tuple[dict, dict, dict]:
    sets = C.rung_sets(C.primary_order(model, order), C.secondary_order(model, order) if secondary else None)
    fit, alias = C.unique_sets(sets)
    return sets, fit, alias


# ----------------------------------------------------------------------------------------------------------- design
def gate_design(a) -> None:
    """Gate D: the patched design of W against the pre-patch one: names equal; y, b_oos, dates_oos md5-equal; X
    bit-equal except the 12 value averages adj_fomc_{until,since}_inv_ma_k."""
    d_new, d_old = Path(PATCHED[a.wdays]), Path(PREPATCH.format(W=a.wdays))
    out: dict[str, Any] = {"wdays": a.wdays, "patched": str(d_new), "prepatch": str(d_old)}
    for f in ("y.npy", "b_oos.npy", "dates_oos.csv", "X.npy"):
        out[f"md5_{f}"] = [C.md5(d_new / f), C.md5(d_old / f)]
    n_new, n_old = C.names(d_new), C.names(d_old)
    out["names_equal"] = n_new == n_old
    Xn = np.load(d_new / "X.npy", mmap_mode="r")
    Xo = np.load(d_old / "X.npy", mmap_mode="r")
    allowed = {f"adj_{s}_ma_{k}" for s in FOMC_PROX for k in (1, 5, 25, 125, 625, 3125)}
    diff_cols, bad = [], []
    for j, c in enumerate(n_new):
        if not np.array_equal(np.asarray(Xn[:, j]), np.asarray(Xo[:, j])):
            diff_cols.append(c)
            if c not in allowed:
                bad.append(c)
    out["X_shape"] = list(Xn.shape)
    out["differing_columns"] = diff_cols
    out["differing_outside_allowed"] = bad
    out["pass"] = bool(
        out["names_equal"]
        and not bad
        and all(out[f"md5_{f}"][0] == out[f"md5_{f}"][1] for f in ("y.npy", "b_oos.npy", "dates_oos.csv"))
    )
    (C.WORK / "gate_design.json").write_text(json.dumps(out, indent=1))
    C.say(f"gate D W {a.wdays}: pass {out['pass']}; {len(diff_cols)} columns differ; outside allowed {bad}")


# ------------------------------------------------------------------------------------------------------------ trees
def tree_design() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.load(C.TREE_D / "X.npy", mmap_mode="r")
    y = np.load(C.TREE_D / "y.npy")
    assert X.shape == (C.N_ROWS, 564) and len(y) == C.N_ROWS
    return X, y, C.names(C.TREE_D)


def rank_trees(a) -> None:
    import ladder_trees as LT

    R = LT.load_runner()
    X, y, nm = tree_design()
    order = C.series_list(nm)
    r0 = C.rank_origin_row(C.TREE_D)
    Xw = np.ascontiguousarray(X[r0 - C.W : r0], dtype=np.float64)
    yw = np.asarray(y[r0 - C.W : r0], dtype=np.float64)
    Xn = np.ascontiguousarray(X[r0 : C.R_FIRST], dtype=np.float64)
    cfg, make = R.model_factory(a.est, LT.THREADS)
    m = make()
    m.fit(Xw, yw)
    if a.est == "rf_skx":
        m.set_params(n_jobs=1)
    pred = np.asarray(m.predict(Xn), dtype=np.float64)
    phi, base = LT.shap_values(a.est, m, Xn, 16 if a.est == "rf_skx" else 1)
    add = phi.sum(1) + base - pred
    tol = 1e-5 * np.maximum(1.0, np.abs(pred)) if a.est == "xgb" else np.full(len(pred), 1e-9)
    ser = np.array([C.series_of(c) for c in nm], dtype=object)
    imp = {s: float(np.abs(phi[:, ser == s].sum(1)).mean()) for s in order}
    out = dict(
        est=a.est,
        wdays=a.wdays,
        design=str(C.TREE_D),
        x_md5=C.md5(C.TREE_D / "X.npy"),
        r0=r0,
        importance={a.est: imp},
        add_err_max=float(np.abs(add).max()),
        add_pass=bool((np.abs(add) <= tol).all()),
        cpu=R.cpu_model(),
        host=platform.node(),
        slurm=os.environ.get("SLURM_JOB_ID", ""),
    )
    tmp = C.WORK / f"rank_{a.est}.tmp.json"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, C.WORK / f"rank_{a.est}.json")
    C.say(
        f"W {a.wdays} rank {a.est}: additivity {out['add_err_max']:.1e} {out['add_pass']}; top 10 {C.order_from_importance(imp, order)[:10]}"
    )


def run_trees(a) -> None:
    import ladder_trees as LT

    R = LT.load_runner()
    X, y, nm = tree_design()
    order = C.series_list(nm)
    o = origins(a.wdays)
    sets, fit, alias = sets_for(a.est, order, bool(a.secondary))
    cols = {rid: C.columns_for(nm, ss) for rid, ss in fit.items()}
    out = C.WORK / "trees" / a.est
    out.mkdir(parents=True, exist_ok=True)
    tag = "sec" if a.secondary else "pri"
    (out / f"sets_{tag}_part{a.part.replace('/', 'of')}.json").write_text(
        json.dumps(dict(sets=sets, alias=alias, fitted=list(fit)), indent=1)
    )
    i, n = (int(v) for v in a.part.split("/"))
    ks = [int(k) for k in np.array_split(o["k"].to_numpy(), n)[i - 1]]
    cfg, make = R.model_factory(a.est, LT.THREADS)
    C.say(f"W {a.wdays} {a.est} {cfg['name']}: sets {list(fit)}; origins {ks[0]}..{ks[-1]}; design {C.TREE_D}")
    for k in ks:
        f = out / f"k{k:03d}.npz"
        prev: dict[str, Any] = dict(np.load(f)) if f.is_file() else {}
        todo = [rid for rid in fit if f"pred_{rid}" not in prev]
        if not todo:
            continue
        row = o.loc[o["k"] == k].iloc[0]
        r, r_end = int(row["r"]), int(row["r_end"])
        Xw = np.ascontiguousarray(X[r - C.W : r], dtype=np.float64)
        yw = np.asarray(y[r - C.W : r], dtype=np.float64)
        Xn = np.ascontiguousarray(X[r:r_end], dtype=np.float64)
        rec: dict[str, Any] = dict(prev) if prev else dict(k=k, r=r, r_end=r_end, wdays=a.wdays)
        secs = {}
        for rid in todo:
            t0 = time.time()
            c = cols[rid]
            m = make()
            m.fit(np.ascontiguousarray(Xw[:, c]), yw)
            if a.est == "rf_skx":
                m.set_params(n_jobs=1)
            Xp = np.ascontiguousarray(Xn[:, c])
            rec[f"p0_{rid}"] = np.asarray(m.predict(Xp[:1]), dtype=np.float64)[0]
            rec[f"pred_{rid}"] = np.asarray(m.predict(Xp), dtype=np.float64)
            rec[f"cols_{rid}"] = np.array(sorted(fit[rid]))
            secs[rid] = time.time() - t0
        rec["cpu"] = R.cpu_model()
        tmp = out / f"k{k:03d}.tmp.npz"
        np.savez_compressed(tmp, **rec)
        os.replace(tmp, f)
        C.say(f"W {a.wdays} {a.est} k={k:3d}: " + " ".join(f"{rid} {s:.1f}s" for rid, s in secs.items()))


# ----------------------------------------------------------------------------------------------------------- linear
def _lin_init(wd: int, sec: str) -> None:
    import ladder_linear as LL

    if sec:
        os.environ["FF_SECONDARY"] = sec
    set_window(wd)
    LL.CODE = FF_CODE
    LL.EX_FLAGS = list(LL.EX_FLAGS) + FOMC_FLAGS
    LL.init_worker()


def lin_job(arg: tuple) -> dict:
    """ladder_linear.job, plus (full rung) the walk state's theta and scales (fiw_replay.state)."""
    import ladder_linear as LL

    m, rid, cols, k, r, r_end, keep = arg
    t0 = time.time()
    X = LL.subset(rid, cols)
    preds, o, g, Cls = LL.replay(m, rid, cols, r - C.W)
    block = np.array([float(o.predict_one(np.asarray(X[j], dtype=np.float64))) for j in range(r, r_end)])
    assert block[0] == preds[-1], (block[0], preds[-1])
    rec: dict[str, Any] = dict(m=m, rid=rid, k=k, r=r, r_end=r_end, g=g, pred=block, n_cols=len(cols))
    rec["kkt_reseeds"] = len(getattr(Cls, "kkt_reseeds", []))
    rec["binom_reseeds"] = len(getattr(Cls, "std_binom_reseeds", []))
    rec["evid_reseeds"] = len(getattr(Cls, "std_evid_reseeds", []))
    if keep:
        rec["replay"] = preds
        rec["theta"] = np.asarray(o._th, dtype=np.float64).copy()
        if m == "ridge":
            n = len(o._y)
            var = o._sd_s2 / n - (o._sd_s1 / n) ** 2
            s = np.sqrt(np.maximum(var, 0.0))
            s[o._sd_run >= n] = 0.0
            rec["sc"] = s
        else:
            rec["sc"] = np.asarray(o._sc, dtype=np.float64).copy()
    rec["sec"] = time.time() - t0
    out = C.WORK / "linear" / m / rid
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"k{k:03d}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **rec)
    os.replace(tmp, out / f"k{k:03d}.npz")
    return dict(m=m, rid=rid, k=k, sec=rec["sec"])


def rank_linear(a) -> None:
    import ladder_linear as LL

    sec = os.environ.get("FF_SECONDARY", "")
    with cf.ProcessPoolExecutor(max_workers=3, initializer=_lin_init, initargs=(a.wdays, sec)) as ex:
        res = list(ex.map(LL.rank_job, C.LINEAR))
    out = dict(
        wdays=a.wdays,
        design=str(C.LIN_D),
        x_md5=C.md5(C.LIN_D / "X.npy"),
        importance={r["m"]: r["importance"] for r in res},
        detail={r["m"]: {k: v for k, v in r.items() if k != "importance"} for r in res},
        frozen=C.FROZEN,
        plp_sha256=hashlib.sha256((FF_CODE / "experiments" / "pool_linear_pretune.py").read_bytes()).hexdigest(),
        host=platform.node(),
        env={k: os.environ.get(k, "") for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "SLURM_JOB_ID")},
    )
    tmp = C.WORK / "rank_linear.tmp.json"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, C.WORK / "rank_linear.json")
    order = C.series_list(C.names(C.LIN_D))
    for r in res:
        C.say(f"W {a.wdays} rank {r['m']}: top 10 {C.order_from_importance(r['importance'], order)[:10]}")


def run_linear(a) -> None:
    nm = C.names(C.LIN_D)
    order = C.series_list(nm)
    o = origins(a.wdays)
    sets, fit, alias = sets_for(a.model, order, bool(a.secondary))
    out = C.WORK / "linear" / a.model
    out.mkdir(parents=True, exist_ok=True)
    tag = "sec" if a.secondary else "pri"
    (out / f"sets_{tag}.json").write_text(json.dumps(dict(sets=sets, alias=alias, fitted=list(fit)), indent=1))
    jobs = []
    for rid, ss in fit.items():
        cols = C.columns_for(nm, ss)
        for row in o.itertuples():
            if (out / rid / f"k{int(row.k):03d}.npz").is_file():
                continue
            jobs.append((a.model, rid, cols, int(row.k), int(row.r), int(row.r_end), rid == "full"))
    C.say(f"W {a.wdays} {a.model} alpha {C.FROZEN[a.model]}: sets {list(fit)}; {len(jobs)} jobs, {a.workers} workers")
    sec = os.environ.get("FF_SECONDARY", "")
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_lin_init, initargs=(a.wdays, sec)) as ex:
        futs = [ex.submit(lin_job, j) for j in jobs]
        for n_, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            if n_ % 100 == 0 or n_ == len(futs):
                C.say(f"{n_}/{len(futs)} jobs, {time.time() - t0:.0f}s (last {r['rid']} k={r['k']} {r['sec']:.1f}s)")
    (out / f"run_info_{tag}.json").write_text(
        json.dumps(
            dict(
                model=a.model,
                wdays=a.wdays,
                frozen=C.FROZEN[a.model],
                design=str(C.LIN_D),
                n_jobs=len(jobs),
                wall=time.time() - t0,
                host=platform.node(),
                plp_sha256=hashlib.sha256(
                    (FF_CODE / "experiments" / "pool_linear_pretune.py").read_bytes()
                ).hexdigest(),
                env={
                    k: os.environ.get(k, "") for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "SLURM_JOB_ID")
                },
            ),
            indent=1,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gate-design", "rank-trees", "run-trees", "rank-linear", "run-linear"))
    ap.add_argument("--wdays", type=int, required=True, choices=(1000, 2000, 3000, 3500))
    ap.add_argument("--est", choices=C.TREES)
    ap.add_argument("--model", choices=C.LINEAR)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--secondary", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.mode.endswith("linear"):
        import ladder_linear as LL

        LL.CODE = FF_CODE
        LL.EX_FLAGS = list(LL.EX_FLAGS) + FOMC_FLAGS
    set_window(a.wdays)
    {
        "gate-design": gate_design,
        "rank-trees": rank_trees,
        "run-trees": run_trees,
        "rank-linear": rank_linear,
        "run-linear": run_linear,
    }[a.mode](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
