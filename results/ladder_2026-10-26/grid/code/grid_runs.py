"""Window x feature-count grid (RULE.txt ADDENDUM 1): the primary-ranking ladder at W = 1,000 / 2,000 / 3,500 days.

Reuses the ladder's code (results/ladder_2026-10-26/code, staged next to this file) with the window set per call:
ladder_common's module constants (W_DAYS, W, design directory, WORK, frozen penalties) are set before any use, and the
origins are taken from fiw_origins.csv's oos_row_w<W> column (the same 126 dates and design rows at every W).

    python grid_runs.py rank-trees --est lgbm|xgb --wdays W
    python grid_runs.py run-trees  --est lgbm|xgb --wdays W --part i/n
    python grid_runs.py rank-linear --wdays W [--workers 3]
    python grid_runs.py run-linear  --model ridge|lasso|enet --wdays W [--workers N]
-> GRID_WORK/w<W>/{rank_<est>.json, rank_linear.json, trees/<est>/k###.npz, linear/<model>/<rung>/k###.npz}
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

GRID_WORK = Path(os.environ.get("GRID_WORK", "/scratch1/jc_905/ladder_grid/work"))
X_MD5 = {
    1000: "493ef22384b51eaadcdaeb78a79b5b80",
    2000: "624fec4924f6e890e88fd069fc1dc395",
    3000: "374bea97f35e33a02404dc47c82c3138",
    3500: "118dbe77b10791077f0d5675a8640b5a",
}
DESIGN_DIRS = (
    "/scratch1/jc_905/fi_window/w{W}/design_all_features",
    "/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w{W}/_work/design_all_features",
)
FROZEN_FILE = "/scratch1/jc_905/fi_window/w{W}/frozen_penalties.json"
FROZEN_RULE = {  # RULE addendum 1 (4); asserted against the W's frozen_penalties.json
    1000: {"ridge": 1e4, "lasso": 1e-3, "enet": 1e-2},
    2000: {"ridge": 1e3, "lasso": 1e-4, "enet": 1e-3},
    3000: {"ridge": 1e3, "lasso": 1e-4, "enet": 1e-3},
    3500: {"ridge": 1e3, "lasso": 1e-4, "enet": 1e-3},
}
_MD5_CACHE: dict[str, str] = {}


def design_dir(wd: int) -> Path:
    """The first candidate directory whose X.npy has the W's md5 (RULE addendum 1 (3))."""
    for pat in DESIGN_DIRS:
        d = Path(pat.format(W=wd))
        x = d / "X.npy"
        if not x.is_file():
            continue
        h = _MD5_CACHE.get(str(x)) or C.md5(x)
        _MD5_CACHE[str(x)] = h
        if h == X_MD5[wd]:
            return d
        C.say(f"{x}: md5 {h} != {X_MD5[wd]}; next candidate")
    raise SystemExit(f"no design with X md5 {X_MD5[wd]} for W {wd}")


def set_window(wd: int) -> Path:
    """Point ladder_common at window W (module constants read at call time)."""
    d = design_dir(wd)
    C.W_DAYS, C.W = wd, wd * C.PPD
    C.TREE_D = C.LIN_D = d
    C.WORK = GRID_WORK / f"w{wd}"
    C.WORK.mkdir(parents=True, exist_ok=True)
    fz = json.loads(Path(FROZEN_FILE.format(W=wd)).read_text())
    C.FROZEN = {m: float(fz[m]["all_features"]) for m in C.LINEAR}
    assert C.FROZEN == FROZEN_RULE[wd], (C.FROZEN, FROZEN_RULE[wd])
    os.environ["LADDER_WORK"] = str(C.WORK)
    return d


def origins(wd: int) -> pd.DataFrame:
    """The 126 origins at W: design rows r = 13 W + oos_row_w<W> (the same rows at every W, asserted)."""
    f = C.HERE / "fiw_origins.csv"
    assert hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == C.ORIGINS_SHA256
    o = pd.read_csv(f)
    dd = C.oos_dates(C.TREE_D)
    assert len(dd) == C.N_ROWS - C.W, (len(dd), C.N_ROWS - C.W)
    o["oos_row"] = o[f"oos_row_w{wd}"].astype(int)
    o["r"] = C.W + o["oos_row"]
    assert (o["r"] == 39000 + o["oos_row_w3000"]).all(), "design rows differ across W"
    o["r_end"] = list(o["r"].iloc[1:]) + [C.N_ROWS]
    assert (pd.DatetimeIndex(dd[o["oos_row"]]) == pd.to_datetime(o["origin"])).all()
    assert (o["r_end"] - o["r"]).sum() == 33970 and o["r"].iloc[0] == C.R_FIRST
    return o[["k", "origin", "oos_row", "r", "r_end"]].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------- trees
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
    pred = np.asarray(m.predict(Xn), dtype=np.float64)
    phi, base = LT.shap_values(a.est, m, Xn, 1)
    add = phi.sum(1) + base - pred
    tol = 1e-5 * np.maximum(1.0, np.abs(pred)) if a.est == "xgb" else np.full(len(pred), 1e-9)
    ser = np.array([C.series_of(c) for c in nm], dtype=object)
    imp = {s: float(np.abs(phi[:, ser == s].sum(1)).mean()) for s in order}
    live = Xw.min(axis=0) != Xw.max(axis=0)
    out = dict(
        est=a.est,
        wdays=a.wdays,
        design=str(C.TREE_D),
        r0=r0,
        window=[r0 - C.W, r0],
        explained_rows=[r0, C.R_FIRST],
        importance={a.est: imp},
        add_err_max=float(np.abs(add).max()),
        add_pass=bool((np.abs(add) <= tol).all()),
        live_series=int(sum(bool(live[ser == s].any()) for s in order)),
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
    sets = C.rung_sets(C.primary_order(a.est, order), None)
    fit, alias = C.unique_sets(sets)
    cols = {rid: C.columns_for(nm, ss) for rid, ss in fit.items()}
    out = C.WORK / "trees" / a.est
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sets_part{a.part.replace('/', 'of')}.json").write_text(
        json.dumps(dict(sets=sets, alias=alias, fitted=list(fit)), indent=1)
    )
    i, n = (int(v) for v in a.part.split("/"))
    ks = [int(k) for k in np.array_split(o["k"].to_numpy(), n)[i - 1]]
    cfg, make = R.model_factory(a.est, LT.THREADS)
    C.say(f"W {a.wdays} {a.est} {cfg['name']}: sets {list(fit)}; origins {ks[0]}..{ks[-1]}; design {C.TREE_D}")
    for k in ks:
        f = out / f"k{k:03d}.npz"
        if f.is_file():
            continue
        row = o.loc[o["k"] == k].iloc[0]
        r, r_end = int(row["r"]), int(row["r_end"])
        Xw = np.ascontiguousarray(X[r - C.W : r], dtype=np.float64)
        yw = np.asarray(y[r - C.W : r], dtype=np.float64)
        Xn = np.ascontiguousarray(X[r:r_end], dtype=np.float64)
        rec: dict[str, Any] = dict(k=k, r=r, r_end=r_end, wdays=a.wdays)
        secs = {}
        for rid in fit:
            t0 = time.time()
            c = cols[rid]
            m = make()
            m.fit(np.ascontiguousarray(Xw[:, c]), yw)
            Xp = np.ascontiguousarray(Xn[:, c])
            rec[f"p0_{rid}"] = np.asarray(m.predict(Xp[:1]), dtype=np.float64)[0]
            rec[f"pred_{rid}"] = np.asarray(m.predict(Xp), dtype=np.float64)
            secs[rid] = time.time() - t0
        rec["secs"] = json.dumps(secs)
        rec["cpu"] = R.cpu_model()
        tmp = out / f"k{k:03d}.tmp.npz"
        np.savez_compressed(tmp, **rec)
        os.replace(tmp, f)
        C.say(f"W {a.wdays} {a.est} k={k:3d}: " + " ".join(f"{rid} {s:.1f}s" for rid, s in secs.items()))


# ------------------------------------------------------------------------------------------------------------ linear
def _lin_init(wd: int) -> None:
    import ladder_linear as LL

    set_window(wd)
    LL.init_worker()


def rank_linear(a) -> None:
    import ladder_linear as LL

    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_lin_init, initargs=(a.wdays,)) as ex:
        res = list(ex.map(LL.rank_job, C.LINEAR))
    out = dict(
        wdays=a.wdays,
        design=str(C.LIN_D),
        importance={r["m"]: r["importance"] for r in res},
        detail={r["m"]: {k: v for k, v in r.items() if k != "importance"} for r in res},
        frozen=C.FROZEN,
        r_bb=C.R_BB,
        host=platform.node(),
        env={k: os.environ.get(k, "") for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "SLURM_JOB_ID")},
    )
    tmp = C.WORK / "rank_linear.tmp.json"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, C.WORK / "rank_linear.json")
    order = C.series_list(C.names(C.LIN_D))
    for r in res:
        C.say(
            f"W {a.wdays} rank {r['m']}: live {r['live_series']}; top 10 {C.order_from_importance(r['importance'], order)[:10]}"
        )


def run_linear(a) -> None:
    import ladder_linear as LL

    assert (
        hashlib.sha256((LL.CODE / "experiments" / "pool_linear_pretune.py").read_bytes()).hexdigest() == LL.PLP_SHA256
    )
    nm = C.names(C.LIN_D)
    order = C.series_list(nm)
    o = origins(a.wdays)
    sets = C.rung_sets(C.primary_order(a.model, order), None)
    fit, alias = C.unique_sets(sets)
    out = C.WORK / "linear" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "sets.json").write_text(json.dumps(dict(sets=sets, alias=alias, fitted=list(fit)), indent=1))
    jobs = []
    for rid, ss in fit.items():
        cols = C.columns_for(nm, ss)
        for row in o.itertuples():
            if (out / rid / f"k{int(row.k):03d}.npz").is_file():
                continue
            jobs.append((a.model, rid, cols, int(row.k), int(row.r), int(row.r_end), rid == "full"))
    C.say(f"W {a.wdays} {a.model} alpha {C.FROZEN[a.model]}: sets {list(fit)}; {len(jobs)} jobs, {a.workers} workers")
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_lin_init, initargs=(a.wdays,)) as ex:
        futs = [ex.submit(LL.job, j) for j in jobs]
        for n_, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            if n_ % 100 == 0 or n_ == len(futs):
                C.say(f"{n_}/{len(futs)} jobs, {time.time() - t0:.0f}s (last {r['rid']} k={r['k']} {r['sec']:.1f}s)")
    (out / "run_info.json").write_text(
        json.dumps(
            dict(
                model=a.model,
                wdays=a.wdays,
                frozen=C.FROZEN[a.model],
                design=str(C.LIN_D),
                n_jobs=len(jobs),
                wall=time.time() - t0,
                host=platform.node(),
                env={
                    k: os.environ.get(k, "") for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "SLURM_JOB_ID")
                },
            ),
            indent=1,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rank-trees", "run-trees", "rank-linear", "run-linear"))
    ap.add_argument("--wdays", type=int, required=True, choices=(1000, 2000, 3500))
    ap.add_argument("--est", choices=("lgbm", "xgb"))
    ap.add_argument("--model", choices=C.LINEAR)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    set_window(a.wdays)
    {"rank-trees": rank_trees, "run-trees": run_trees, "rank-linear": rank_linear, "run-linear": run_linear}[a.mode](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
