"""Pre-test benchmark of three random-forest engines on the final 2014-start design (owner's question, 2026-10-23).

Engines (one task = engine x predictor set x window, on one node):
  A     scikit-learn RandomForestRegressor, the frozen configuration rf400_rf_opt_00 constructed and fitted exactly as
        the walk does (experiments/pooled_trees_rerun_13.py model_factory / fit_predict: seed 42, n_jobs = T to fit,
        n_jobs = 1 to predict); env harxhar-r13.
  B     Intel Extension for Scikit-learn (sklearnex.ensemble.RandomForestRegressor), the same frozen parameters
        (bootstrap, max_samples, max_features, max_depth, min_samples_leaf, n_estimators, random_state 42), its
        default histogram split (max_bins 256); a separate venv (harxhar-r13 untouched).
  B2048 the same with max_bins = 2048.
  C     LightGBM boosting="rf" mapped from the frozen configuration: 485 trees, feature_fraction_bynode = 0.9013
        (= max_features), bagging_fraction = 0.632 with bagging_freq = 1 (rows without replacement), min_data_in_leaf
        = 12 (= min_samples_leaf), max_depth = 20 with num_leaves = 4095 (deep trees; the depth and the leaf minimum
        bind), deterministic = True, force_col_wise = True, seed 42, max_bin 255 (default); env harxhar-r13.
Rows: pre-test fold 3 of the 2014-start design -- fit [S - 3,575 - W, S - 3,575), validation [S - 3,250, S) = the
3,250 bars of 2013 (fold_fit14.py's rows; S = 50,777, the first 2014-01-02 bar).  No row on or after 2014-01-02.
Each task fits twice at T threads (timing; determinism: the two validation forecasts bit for bit) and, with
--one-thread, once at 1 thread (thread invariance); scores the validation rows (MSE on the adjusted scale, QLIKE of
plain = pred^2 x B against the bar's RV) and saves the forecasts.

usage: python bench_rf.py --repo R --design <dir holding _work/design_<set>> --window-days W --set S --engine E
                          --threads T --out O [--one-thread]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PPD, FOLD_N, EMB = 13, 3250, 325
S_WANT = 50777


class _Log(logging.Handler):
    """Collects sklearnex's dispatch messages (accelerated vs fallback)."""

    def __init__(self) -> None:
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.msgs.append(record.getMessage())


def build(engine: str, threads: int, PT):
    """(make, info): a factory of the engine's estimator at T threads and a description of its configuration."""
    cfg = json.loads(PT.CONFIGS["rf"].read_text(encoding="utf-8"))
    p = dict(cfg["params"])
    if engine == "A":
        _, make = PT.model_factory("rf", threads)
        return make, {"config": cfg["name"], "params": p, "engine": "sklearn"}
    if engine.startswith("B"):
        import daal4py
        import sklearnex
        from sklearnex.ensemble import RandomForestRegressor as SKX

        kw = dict(p, random_state=42, n_jobs=threads)
        if engine == "B2048":
            kw["max_bins"] = 2048
        info = {
            "config": cfg["name"],
            "params": kw,
            "engine": "sklearnex",
            "sklearnex": sklearnex.__version__,
            "daal4py": getattr(daal4py, "__version__", ""),
        }
        return (lambda: SKX(**kw)), info
    if engine == "C":
        import lightgbm as lgb

        kw = {
            "boosting_type": "rf",
            "n_estimators": p["n_estimators"],
            "feature_fraction_bynode": p["max_features"],
            "bagging_fraction": 0.632,
            "bagging_freq": 1,
            "min_child_samples": p["min_samples_leaf"],
            "max_depth": p["max_depth"],
            "num_leaves": 4095,
            "deterministic": True,
            "force_col_wise": True,
            "histogram_pool_size": 4096.0,
            "random_state": 42,
            "num_threads": threads,
            "verbosity": -1,
        }
        return (lambda: lgb.LGBMRegressor(**kw)), {
            "config": "lgbm rf mapped from " + cfg["name"],
            "params": kw,
            "engine": "lightgbm",
            "lightgbm": lgb.__version__,
        }
    raise SystemExit(f"engine {engine}")


def tree_stats(engine: str, m) -> dict:
    """Mean leaves and depth of the fitted trees (sklearn / sklearnex estimators_, LightGBM dump)."""
    try:
        if engine == "C":
            d = m.booster_.dump_model()["tree_info"]
            return {
                "trees": len(d),
                "mean_leaves": float(np.mean([t["num_leaves"] for t in d])),
            }
        est = m.estimators_
        return {
            "trees": len(est),
            "mean_leaves": float(np.mean([e.tree_.n_leaves for e in est])),
            "mean_depth": float(np.mean([e.tree_.max_depth for e in est])),
        }
    except Exception as exc:  # noqa: BLE001  (a statistic only)
        return {"tree_stats_error": repr(exc)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--window-days", type=int, required=True)
    ap.add_argument("--set", choices=("baseline", "all_features"), required=True)
    ap.add_argument("--engine", choices=("A", "B", "B2048", "C"), required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--one-thread", action="store_true")
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out)
    sys.path.insert(0, str(repo / "experiments"))
    import pooled_trees_rerun_13 as PT

    lg = _Log()
    logging.getLogger("sklearnex").addHandler(lg)
    logging.getLogger("sklearnex").setLevel(logging.INFO)
    Wr = a.window_days * PPD
    root = Path(a.design)
    meta = PT.check(root, a.set, a.window_days)
    dz = PT.design_dir(root, a.set)
    X = np.load(dz / "X.npy", mmap_mode="r")
    y, b = np.load(dz / "y.npy"), np.load(dz / "b_oos.npy")
    d = PT.oos_dates(root, a.set)
    S = Wr + int(np.flatnonzero(d >= pd.Timestamp("2014-01-02"))[0])
    assert S == S_WANT, S
    f0, f1, v0, v1 = S - FOLD_N - EMB - Wr, S - FOLD_N - EMB, S - FOLD_N, S
    assert f0 >= 0 and v0 - Wr >= 0
    Xf = np.ascontiguousarray(X[f0:f1], dtype=np.float64)
    Xv = np.ascontiguousarray(X[v0:v1], dtype=np.float64)
    yf, yv = y[f0:f1], y[v0:v1]
    idx = np.arange(v0, v1) - Wr
    dates, B = d[idx], b[idx]
    assert (
        str(dates[0]) == "2013-01-02 15:30:00"
        and str(dates[-1]) == "2013-12-31 16:00:00"
    ), (dates[0], dates[-1])
    rv = PT.rv_bars().reindex(dates).to_numpy(float)
    runs = [("T", a.threads), ("T2", a.threads)] + ([("1", 1)] if a.one_thread else [])
    rec: dict = {
        "engine": a.engine,
        "set": a.set,
        "window_days": a.window_days,
        "n_columns": int(X.shape[1]),
        "fit_rows": [f0, f1],
        "val_rows": [v0, v1],
        "val_first": str(dates[0]),
        "val_last": str(dates[-1]),
        "host": platform.node(),
        "cpu": PT.cpu_model(),
        "versions": PT.versions(),
        "x_columns_check": len(meta["names"]),
        "slurm_job": os.environ.get(
            "SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")
        ),
        "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "runs": [],
    }
    preds = {}
    for tag, T in runs:
        make, info = build(a.engine, T, PT)
        rec["config"] = info
        m = make()
        t0, c0 = time.time(), time.process_time()
        m.fit(Xf, yf)
        fit_s, fit_cpu = time.time() - t0, time.process_time() - c0
        if a.engine in ("A", "B", "B2048"):
            m.set_params(
                n_jobs=1
            )  # the walk's one-thread predict (pooled_trees_rerun_13.fit_predict)
        t1 = time.time()
        pr = np.asarray(m.predict(Xv), dtype=np.float64)
        pred_s = time.time() - t1
        t2 = time.time()
        np.asarray(m.predict(Xv[:1]), dtype=np.float64)
        pred1_s = time.time() - t2
        preds[tag] = pr
        plain = pr**2 * B
        ok = np.isfinite(rv) & (rv > 0) & np.isfinite(plain) & (plain > 0)
        r = rv[ok] / plain[ok]
        rec["runs"].append(
            {
                "run": tag,
                "threads": T,
                "fit_sec": round(fit_s, 3),
                "fit_cpu_sec": round(fit_cpu, 3),
                "predict_val_sec": round(pred_s, 3),
                "predict_one_row_sec": round(pred1_s, 4),
                "refit_sec": round(fit_s + pred1_s, 3),
                "mse_adj": float(np.mean((pr - yv) ** 2)),
                "qlike": float(np.mean(r - np.log(r) - 1.0)),
                "n_qlike": int(ok.sum()),
                "pred_md5": hashlib.md5(pr.tobytes()).hexdigest(),
                **tree_stats(a.engine, m),
            }
        )
        print(
            f"{a.engine} {a.set} w{a.window_days} run {tag} ({T} threads): fit {fit_s:.1f} s, mse {rec['runs'][-1]['mse_adj']:.6f} "
            f"qlike {rec['runs'][-1]['qlike']:.6f} md5 {rec['runs'][-1]['pred_md5'][:10]}",
            flush=True,
        )
    rec["same_twice"] = bool(np.array_equal(preds["T"], preds["T2"]))
    rec["max_abs_diff_twice"] = float(np.max(np.abs(preds["T"] - preds["T2"])))
    if "1" in preds:
        rec["same_1_vs_T"] = bool(np.array_equal(preds["T"], preds["1"]))
        rec["max_abs_diff_1_vs_T"] = float(np.max(np.abs(preds["T"] - preds["1"])))
    rec["sklearnex_log"] = sorted(set(lg.msgs))[:20]
    tag = f"{a.engine}_{a.set}_w{a.window_days}_{'xeon' if 'Intel' in rec['cpu'] else 'epyc'}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": dates,
            "true_adj": yv,
            "B": B,
            "rv_actual": rv,
            **{f"pred_{k}": v for k, v in preds.items()},
        }
    ).to_parquet(out / f"{tag}.parquet", index=False)
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(
        json.dumps(
            {k: rec[k] for k in ("same_twice", "max_abs_diff_twice") if k in rec}
        ),
        rec.get("same_1_vs_T"),
        rec["sklearnex_log"][:3],
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
