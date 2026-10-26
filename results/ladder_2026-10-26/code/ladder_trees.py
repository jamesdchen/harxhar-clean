"""Tree rungs of the feature-count ladder (RULE.txt): the primary ranking's pre-sample SHAP fit and the monthly refits.

    python ladder_trees.py rank --est lgbm|xgb|rf_skx [--procs 16]
        one full-set fit on [r_0 - W, r_0) (r_0 = 2013-01-02 10:00), path-dependent TreeSHAP on the 2013 rows,
        I_s = mean |sum of the series' 12 SHAP values| -> WORK/rank_<est>.json
    python ladder_trees.py run --est E --part i/n
        for each origin of the part and each distinct rung set (full, har, P<k>, S<k>): fit on [r_k - W, r_k) restricted
        to the rung's columns, forecast the block [r_k, r_{k+1}) -> WORK/trees/<est>/k###.npz (atomic; resumable)

CARC: python = /home1/jc_905/trees_final/venv_skx, PYTHONPATH = /home1/jc_905/tree_shap/pylib (shap 0.49.1); the runner
experiments/pooled_trees_rerun_13.py at 3cbb85f from /home1/jc_905/tree_shap/runner (sha256 of the git blobs checked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

import ladder_common as C

RUNNER_DIR = Path(os.environ.get("LADDER_RUNNER_DIR", "/home1/jc_905/tree_shap/runner"))
SHAP_WORK = Path(os.environ.get("LADDER_SHAP_WORK", "/home1/jc_905/tree_shap/work"))
RUNNER_SHA = "3cbb85f"
BLOB_SHA256 = {
    "experiments/pooled_trees_rerun_13.py": "c2a5b45724f8ef1105484e8d7db41c974766b36cc8daa29b581905d2af2bbad8",
    "cluster/pooltree_lgbm_config.json": "1349f96a9ae69198db9f754c07993c729fe02b6c5a0a45c66d0ce4dae3e80d0d",
    "cluster/xgbtop_xgb_config.json": "961ca7096386aba2aad9fa54a50a7be5c9a711b4399fa80b49c8ec6f6854f3dc",
    "cluster/pooltree_rf400_config.json": "ba5c99b02b9359e070813a7ed8a421cb96205176a7dc91aa55c6b0288f164567",
}
THREADS = 8
_EX: Any = None
_XN: Any = None


def load_runner() -> types.ModuleType:
    for f, h in BLOB_SHA256.items():
        assert hashlib.sha256((RUNNER_DIR / f).read_bytes()).hexdigest() == h, f
    src = (RUNNER_DIR / "experiments" / "pooled_trees_rerun_13.py").read_bytes()
    mod = types.ModuleType("pooled_trees_rerun_13_at_" + RUNNER_SHA)
    mod.__file__ = str(RUNNER_DIR / "experiments" / "pooled_trees_rerun_13.py")
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"pooled_trees_rerun_13.py@{RUNNER_SHA}", "exec"), mod.__dict__)
    return mod


def load_design() -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = C.TREE_D
    want = [ln.split()[0] for ln in (d / "MD5SUMS").read_text().splitlines() if ln.strip().endswith("X.npy")]
    assert want == [C.X_MD5], want
    X = np.load(d / "X.npy", mmap_mode="r")
    y = np.load(d / "y.npy")
    assert X.shape == (C.N_ROWS, 564) and len(y) == C.N_ROWS
    return X, y, C.names(d)


def _rf_chunk(idx: np.ndarray) -> np.ndarray:
    return np.asarray(_EX.shap_values(_XN[idx], check_additivity=False), dtype=np.float64)


def shap_values(est: str, m, Xn: np.ndarray, procs: int) -> tuple[np.ndarray, np.ndarray]:
    """phi (n x M), phi_0 (n,): as results/tree_shap_w3000_2014start_2026-10-23/code/shap_run.py."""
    global _EX, _XN
    M = Xn.shape[1]
    if est == "lgbm":
        c = m.booster_.predict(Xn, pred_contrib=True, num_threads=16)
        return c[:, :M].astype(np.float64), c[:, M].astype(np.float64)
    if est == "xgb":
        import xgboost as xgb

        c = m.get_booster().predict(xgb.DMatrix(Xn, nthread=16), pred_contribs=True)
        return c[:, :M].astype(np.float64), c[:, M].astype(np.float64)
    import shap

    ex = shap.TreeExplainer(m)
    assert ex.feature_perturbation == "tree_path_dependent", ex.feature_perturbation
    assert ex.model.thresholds.dtype == np.float64, ex.model.thresholds.dtype
    ex.model.input_dtype = np.float64  # the oneDAL forest routes rows on float64 values (tree SHAP study, G2(b))
    if procs > 1 and platform.system() == "Linux":
        _EX, _XN = ex, Xn
        chunks = [c for c in np.array_split(np.arange(len(Xn)), 2 * procs) if len(c)]
        with multiprocessing.get_context("fork").Pool(procs) as pool:
            sv = np.vstack(pool.map(_rf_chunk, chunks))
        _EX, _XN = None, None
    else:
        sv = np.asarray(ex.shap_values(Xn, check_additivity=False), dtype=np.float64)
    base = np.full(len(Xn), float(np.asarray(ex.expected_value).ravel()[0]))
    return sv, base


def rank(a) -> None:
    R = load_runner()
    X, y, nm = load_design()
    order = C.series_list(nm)
    r0 = C.rank_origin_row(C.TREE_D)
    Xw = np.ascontiguousarray(X[r0 - C.W : r0], dtype=np.float64)
    yw = np.asarray(y[r0 - C.W : r0], dtype=np.float64)
    Xn = np.ascontiguousarray(X[r0 : C.R_FIRST], dtype=np.float64)
    cfg, make = R.model_factory(a.est, THREADS)
    t0 = time.time()
    m = make()
    m.fit(Xw, yw)
    if a.est == "rf_skx":
        m.set_params(n_jobs=1)
    pred = np.asarray(m.predict(Xn), dtype=np.float64)
    t_fit = time.time() - t0
    phi, base = shap_values(a.est, m, Xn, a.procs)
    add = phi.sum(1) + base - pred
    tol = 1e-5 * np.maximum(1.0, np.abs(pred)) if a.est == "xgb" else np.full(len(pred), 1e-9)
    ser = np.array([C.series_of(c) for c in nm], dtype=object)
    imp = {s: float(np.abs(phi[:, ser == s].sum(1)).mean()) for s in order}
    live = Xw.min(axis=0) != Xw.max(axis=0)
    out = dict(
        est=a.est,
        config=cfg["name"],
        sha=cfg["sha"],
        r0=r0,
        window=[r0 - C.W, r0],
        explained_rows=[r0, C.R_FIRST],
        n_explained=len(Xn),
        importance={a.est: imp},
        har_mean_abs=float(np.abs(phi[:, ser == None].sum(1)).mean()),  # noqa: E711
        add_err_max=float(np.abs(add).max()),
        add_pass=bool((np.abs(add) <= tol).all()),
        live_series=int(sum(bool(live[ser == s].any()) for s in order)),
        t_fit=t_fit,
        t_total=time.time() - t0,
        cpu=R.cpu_model(),
        host=platform.node(),
        versions=R.versions(),
        slurm=os.environ.get("SLURM_JOB_ID", ""),
    )
    C.WORK.mkdir(parents=True, exist_ok=True)
    tmp = C.WORK / f"rank_{a.est}.tmp.json"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, C.WORK / f"rank_{a.est}.json")
    top = C.order_from_importance(imp, order)[:10]
    C.say(f"rank {a.est}: additivity {out['add_err_max']:.1e} pass {out['add_pass']}; top 10 {top}")


def run(a) -> None:
    R = load_runner()
    X, y, nm = load_design()
    order = C.series_list(nm)
    o = C.origins(C.TREE_D)
    sets = C.rung_sets(C.primary_order(a.est, order), C.secondary_order(a.est, order))
    fit, alias = C.unique_sets(sets)
    cols = {rid: C.columns_for(nm, ss) for rid, ss in fit.items()}
    out = C.WORK / "trees" / a.est
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(sets=sets, alias=alias, fitted=list(fit))
    (out / f"sets_part{a.part.replace('/', 'of')}.json").write_text(json.dumps(meta, indent=1))
    i, n = (int(v) for v in a.part.split("/"))
    ks = [int(k) for k in np.array_split(o["k"].to_numpy(), n)[i - 1]]
    cfg, make = R.model_factory(a.est, THREADS)
    C.say(f"{a.est} {cfg['name']} {cfg['sha']}: {len(fit)} fitted sets {list(fit)}; origins {ks[0]}..{ks[-1]}")
    for k in ks:
        f = out / f"k{k:03d}.npz"
        if f.is_file():
            continue
        row = o.loc[o["k"] == k].iloc[0]
        r, r_end = int(row["r"]), int(row["r_end"])
        Xw = np.ascontiguousarray(X[r - C.W : r], dtype=np.float64)
        yw = np.asarray(y[r - C.W : r], dtype=np.float64)
        Xn = np.ascontiguousarray(X[r:r_end], dtype=np.float64)
        rec: dict[str, Any] = dict(k=k, r=r, r_end=r_end)
        secs = {}
        for rid in fit:  # "full" first (the gate)
            t0 = time.time()
            c = cols[rid]
            m = make()
            m.fit(np.ascontiguousarray(Xw[:, c]), yw)
            if a.est == "rf_skx":
                m.set_params(n_jobs=1)
            Xp = np.ascontiguousarray(Xn[:, c])
            rec[f"p0_{rid}"] = np.asarray(m.predict(Xp[:1]), dtype=np.float64)[0]
            rec[f"pred_{rid}"] = np.asarray(m.predict(Xp), dtype=np.float64)
            secs[rid] = time.time() - t0
        sf = SHAP_WORK / a.est / f"k{k:03d}.npz"
        if sf.is_file():
            z = np.load(sf)
            rec["shap_pred_equal"] = bool(np.array_equal(z["pred"], rec["pred_full"]))
            rec["shap_pred_maxabs"] = float(np.abs(z["pred"] - rec["pred_full"]).max())
            rec["shap_p0_equal"] = bool(float(z["pred_origin"]) == float(rec["p0_full"]))
        rec["secs"] = json.dumps(secs)
        rec["cpu"] = R.cpu_model()
        rec["host"] = platform.node()
        tmp = out / f"k{k:03d}.tmp.npz"
        np.savez_compressed(tmp, **rec)
        os.replace(tmp, f)
        C.say(
            f"{a.est} k={k:3d} {str(row['origin'])[:10]} rows {r_end - r}: "
            + " ".join(f"{rid} {s:.1f}s" for rid, s in secs.items())
            + (f" | shap pred equal {rec['shap_pred_equal']}" if "shap_pred_equal" in rec else "")
        )
    (out / f"run_info_part{a.part.replace('/', 'of')}.json").write_text(
        json.dumps(
            dict(
                est=a.est,
                config=cfg["name"],
                sha=cfg["sha"],
                part=a.part,
                versions=R.versions(),
                cpu=R.cpu_model(),
                host=platform.node(),
                slurm=os.environ.get("SLURM_JOB_ID", ""),
                task=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            ),
            indent=1,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rank", "run"))
    ap.add_argument("--est", required=True, choices=C.TREES)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    rank(a) if a.mode == "rank" else run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
