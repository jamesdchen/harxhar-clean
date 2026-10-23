"""Refit each frozen tree model at the 126 origins (the walk's every-bar refit), TreeSHAP (path-dependent) on the
origin's neighbourhood rows, SHAP interaction values on the M5 subsample (LightGBM, XGBoost).  RULE_shap.txt.

    python shap_run.py --est rf_skx [--part 1/3] [--ks 0,1] [--threads 8] [--procs 15] [--shap-threads 16]

-> WORK/<est>/k###.npz (compressed; resumable: an origin whose file exists is skipped).  Per origin: the prediction
at the origin row (float64; the gate), the model's predictions on the neighbourhood, phi (n x 564, float32), the 48
variable sums of phi (float64), phi_0, additivity errors, the columns live in the window and used by the model, for
LightGBM / XGBoost the 48 x 48 variable interaction matrices of the M5 rows plus column-level main / interaction sums,
for the forest the mean of the estimators_ predictions; at C.REPEAT_KS a second refit (prediction and SHAP compared
bit for bit: G2(a)).  The forest's TreeSHAP runs in --procs forked processes (shap holds the GIL in one process).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import time
from typing import Any

import numpy as np

import shap_common as C

_EX: Any = None
_XN: Any = None


def _rf_chunk(idx: np.ndarray) -> np.ndarray:
    return np.asarray(
        _EX.shap_values(_XN[idx], check_additivity=False), dtype=np.float64
    )


def used_columns(est: str, m, M: int) -> np.ndarray:
    u = np.zeros(M, dtype=bool)
    if est == "lgbm":
        u[np.asarray(m.booster_.feature_importance("split")) > 0] = True
    elif est == "xgb":
        for f in m.get_booster().get_score(importance_type="weight"):
            u[int(f[1:])] = True
    else:
        for t in m.estimators_:
            f = t.tree_.feature
            u[f[f >= 0]] = True
    return u


def tree_mean_f64(m, X: np.ndarray) -> np.ndarray:
    """Mean over m.estimators_ of each tree's leaf value, rows routed on float64 values (x <= threshold)."""
    out = np.zeros(len(X))
    r = np.arange(len(X))
    for e in m.estimators_:
        t = e.tree_
        node = np.zeros(len(X), dtype=np.int64)
        while True:
            inner = t.children_left[node] != -1
            if not inner.any():
                break
            ni = node[inner]
            left = X[r[inner], t.feature[ni]] <= t.threshold[ni]
            node[inner] = np.where(left, t.children_left[ni], t.children_right[ni])
        out += t.value[node, 0, 0]
    return out / len(m.estimators_)


def shap_values(est: str, m, Xn: np.ndarray, threads: int, procs: int = 1):
    """phi (n x M, float64), phi_0 (n,), and for the forest the mean of the estimators_ predictions (what shap
    explains)."""
    global _EX, _XN
    M = Xn.shape[1]
    if est == "lgbm":
        c = m.booster_.predict(Xn, pred_contrib=True, num_threads=threads)
        return c[:, :M].astype(np.float64), c[:, M].astype(np.float64), None
    if est == "xgb":
        import xgboost as xgb

        c = m.get_booster().predict(
            xgb.DMatrix(Xn, nthread=threads), pred_contribs=True
        )
        return c[:, :M].astype(np.float64), c[:, M].astype(np.float64), None
    import shap

    ex = shap.TreeExplainer(m)  # tree_path_dependent (no background data)
    assert ex.feature_perturbation == "tree_path_dependent", ex.feature_perturbation
    # 2026-10-23 (gate G2(b) on the first CARC run, array 13324817, cancelled): shap's scikit-learn path casts X to
    # float32 (as scikit-learn trees predict), but the oneDAL forest routes a row on its float64 value (x <= threshold,
    # float64 thresholds); design values within float32 rounding of a threshold went the other way (additivity errors
    # up to 1.5e-2).  shap evaluated in float64 reproduces the oneDAL prediction (toy forest: 2e-14).
    assert ex.model.thresholds.dtype == np.float64, ex.model.thresholds.dtype
    ex.model.input_dtype = np.float64
    tree_mean = tree_mean_f64(
        m, Xn
    )  # what shap explains; also converts estimators_ before a fork
    if procs > 1 and platform.system() == "Linux":
        _EX, _XN = ex, Xn
        chunks = [c for c in np.array_split(np.arange(len(Xn)), 2 * procs) if len(c)]
        with multiprocessing.get_context("fork").Pool(procs) as pool:
            sv = np.vstack(pool.map(_rf_chunk, chunks))
        _EX, _XN = None, None
    else:
        sv = np.asarray(ex.shap_values(Xn, check_additivity=False), dtype=np.float64)
    base = np.full(len(Xn), float(np.asarray(ex.expected_value).ravel()[0]))
    return sv, base, tree_mean


def interactions(est: str, m, Xs: np.ndarray, threads: int) -> np.ndarray:
    """SHAP interaction values (n x M x M, float64), path-dependent."""
    M = Xs.shape[1]
    if est == "xgb":
        import xgboost as xgb

        c = m.get_booster().predict(
            xgb.DMatrix(Xs, nthread=threads), pred_interactions=True
        )
        return c[:, :M, :M].astype(np.float64)
    import shap

    ex = shap.TreeExplainer(m.booster_)
    return np.asarray(ex.shap_interaction_values(Xs), dtype=np.float64)[:, :M, :M]


def one_origin(a, R, make, X, y, P, row, ii) -> dict:
    r, r_end = int(row["r"]), int(row["r_end"])
    t0 = time.time()
    Xw = np.ascontiguousarray(X[r - C.W : r], dtype=np.float64)
    yw = np.asarray(y[r - C.W : r], dtype=np.float64)
    Xn = np.ascontiguousarray(X[r:r_end], dtype=np.float64)
    # the runner's fit_predict, step by step (the model is kept for SHAP): make(); fit; rf n_jobs = 1; predict
    m = make()
    m.fit(Xw, yw)
    if a.est in ("rf", "rf_skx"):
        m.set_params(n_jobs=1)
    p0 = np.asarray(m.predict(Xn[:1]), dtype=np.float64)[0]
    t_fit = time.time() - t0
    pred = np.asarray(m.predict(Xn), dtype=np.float64)
    M = Xn.shape[1]
    t1 = time.time()
    phi, base, tree_mean = shap_values(a.est, m, Xn, a.shap_threads, a.procs)
    t_shap = time.time() - t1
    rec = dict(
        k=int(row["k"]),
        r=r,
        r_end=r_end,
        pred_origin=p0,
        pred=pred,
        phi=phi.astype(np.float32),
        phi_group=phi @ P.T,
        base=base,
        add_err=phi.sum(1) + base - pred,
        live=Xw.min(axis=0) != Xw.max(axis=0),
        used=used_columns(a.est, m, M),
        t_fit=t_fit,
        t_shap=t_shap,
        phi_full=phi,
    )
    if tree_mean is not None:
        rec["tree_mean"] = tree_mean
    if a.est in ("lgbm", "xgb"):
        t2 = time.time()
        Phi = interactions(a.est, m, Xn[ii], a.shap_threads)
        rec["inter_rows"] = ii
        rec["inter_sum_err"] = Phi.sum(2) - phi[ii]
        rec["G"] = P @ Phi @ P.T  # G_iAB = sum_{j in A, l in B} Phi_ijl
        dg = np.abs(np.einsum("njj->nj", Phi))
        rec["col_main_abs"] = dg.sum(1)
        rec["col_inter_abs"] = np.abs(Phi).sum((1, 2)) - dg.sum(1)
        rec["t_inter"] = time.time() - t2
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, choices=C.EST)
    ap.add_argument("--ks", default="")
    ap.add_argument(
        "--part", default="1/1", help="i/n: the i-th of n contiguous blocks of origins"
    )
    ap.add_argument(
        "--threads", type=int, default=C.THREADS, help="fit threads (the walk's 8)"
    )
    ap.add_argument("--shap-threads", type=int, default=C.THREADS)
    ap.add_argument(
        "--procs", type=int, default=1, help="forest TreeSHAP processes (Linux fork)"
    )
    a = ap.parse_args()
    R = C.load_runner()
    t = C.column_table()
    P = C.group_matrix(t)
    o = C.origins()
    irows = C.inter_rows(o)
    if a.ks:
        ks = [int(k) for k in a.ks.split(",")]
    else:
        i, n = (int(v) for v in a.part.split("/"))
        ks = [int(k) for k in np.array_split(o["k"].to_numpy(), n)[i - 1]]
    X = np.load(C.DESIGN / "_work" / "design_all_features" / "X.npy", mmap_mode="r")
    y = np.load(C.DESIGN / "_work" / "design_all_features" / "y.npy")
    assert X.shape == (84747, 564) and len(y) == 84747
    out = C.WORK / a.est
    out.mkdir(parents=True, exist_ok=True)
    cfg, make = R.model_factory(a.est, a.threads)
    C.say(
        f"{a.est}: {cfg['name']} {cfg['sha']}, fit threads {a.threads}, shap threads {a.shap_threads}, procs "
        f"{a.procs}, origins {ks[0]}..{ks[-1]} ({len(ks)}) -> {out}; cpu {R.cpu_model()}"
    )
    for k in ks:
        f = out / f"k{k:03d}.npz"
        if f.is_file():
            continue
        free = shutil.disk_usage(out).free
        if (
            free < 400e6
        ):  # the laptop's C: filled up once (2026-10-23 02:28): stop before it does again
            raise SystemExit(
                f"only {free / 1e6:.0f} MB free on {out}; stopping before origin {k}"
            )
        row = o.loc[o["k"] == k].iloc[0]
        rec = one_origin(a, R, make, X, y, P, row, irows[k])
        if k in C.REPEAT_KS:  # G2(a) self-consistency (all three models)
            rec2 = one_origin(a, R, make, X, y, P, row, irows[k])
            rec["repeat_pred_equal"] = bool(np.array_equal(rec["pred"], rec2["pred"]))
            rec["repeat_phi_equal"] = bool(
                np.array_equal(rec["phi_full"], rec2["phi_full"])
            )
            C.say(
                f"  repeat k={k}: predictions equal {rec['repeat_pred_equal']}, SHAP equal {rec['repeat_phi_equal']}"
            )
        rec.pop("phi_full")
        rec["cpu"] = R.cpu_model()
        rec["host"] = platform.node()
        tmp = out / f"k{k:03d}.tmp.npz"
        np.savez_compressed(tmp, **rec)
        os.replace(tmp, f)
        C.say(
            f"{a.est} k={k:3d} {str(row['origin'])[:10]} rows {rec['r_end'] - rec['r']}: fit {rec['t_fit']:.1f}s "
            f"shap {rec['t_shap']:.1f}s"
            + (f" inter {rec['t_inter']:.1f}s" if "t_inter" in rec else "")
            + f" | add {np.abs(rec['add_err']).max():.1e} | live {rec['live'].sum()} used {rec['used'].sum()}"
        )
    info = dict(
        est=a.est,
        config=cfg["name"],
        sha=cfg["sha"],
        threads=a.threads,
        shap_threads=a.shap_threads,
        procs=a.procs,
        part=a.part,
        runner_sha=C.RUNNER_SHA,
        runner_src_sha256=R.RUNNER_SRC_SHA256,
        versions=R.versions(),
        shap=__import__("shap").__version__,
        cpu=R.cpu_model(),
        host=platform.node(),
        slurm=os.environ.get("SLURM_JOB_ID", ""),
        task=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    )
    (out / f"run_info_{a.part.replace('/', 'of')}.json").write_text(
        json.dumps(info, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
