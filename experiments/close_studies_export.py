"""Committed export of the 2026-10-03 / 10-04 close studies: forecasts, targets, catalogue, model internals.

The studies' raw outputs are the gitignored .npz files in results/close_studies_2026-10-03/*/_work/
(never committed).  This script writes a compact, self-describing copy of everything an analysis
needs into results/close_studies_2026-10-03/forecasts/ (parquet, zstd), and proves that the copy
alone reproduces every study's headline numbers.

Stages:
  python experiments/close_studies_export.py build    read the npz files, the design cache (forecast-row
                                                      dates / targets / column names only), the stored
                                                      master-table tables and the deck; write the export
  python experiments/close_studies_export.py verify   reload the export only (experiments/close_studies_load.py),
                                                      re-score every exported series and every derived
                                                      forecast, compare with the study CSVs (|diff| <= 1e-10
                                                      on QLIKE and Sharpe), compare pred_adj with the npz
                                                      bit for bit where the npz files are present, write
                                                      VERIFY.csv and README.md

Files written (see the generated README.md for every column): forecasts.parquet, targets.parquet,
catalogue.csv, study_arms.csv, linear_coefficients.parquet, linear_rechoices.parquet,
linear_columns.csv, tree_importance.parquet, tree_columns.csv, refits.parquet,
pretune_trials.parquet, VERIFY.csv, README.md.

Light: one process, no model is refitted.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STUDIES = REPO / "results" / "close_studies_2026-10-03"
OUT = STUDIES / "forecasts"
DESIGN_DIR = REPO / "results" / "close_design" / "_work"
DESIGN = DESIGN_DIR / "design_bar1600_all_features.npz"
DESIGN_BASE = DESIGN_DIR / "design_bar1600_baseline.npz"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
SPXW = REPO / "results" / "spxw_pnl"
MASTER = REPO / "results" / "close_master_table" / "master_table.csv"
TREE_START = 130
N_FC = 1469
TOL = 1e-10
ZSTD = 12  # zstd level of every parquet file
DECK_COLS = ("entry", "iv_var", "R", "exit", "ask_c", "ask_p", "bid_c", "bid_p", "signal")
BAR = "bar_end_minute"
LIMIT_MB = 50.0  # GitHub warns above 50 MB; every file must stay below

# the master-table forecasts the studies compared against (results/spxw_pnl/yhat_<key>.parquet)
STORED = {
    "subtree_lgbm_all_features": dict(model="LightGBM", bucket="all_features", refit_every=10, tuning="shipped configuration of specs/causal_tune_trees.py", lightgbm="4.6.0", xgboost=""),
    "subtree_xgb_all_features": dict(model="XGBoost", bucket="all_features", refit_every=10, tuning="shipped configuration of specs/causal_tune_trees.py", lightgbm="", xgboost="3.2.0"),
    "subtree_daily_all_features_lgbm": dict(model="LightGBM", bucket="all_features", refit_every=1, tuning="shipped configuration, refit every session", lightgbm="4.6.0", xgboost=""),
    "subtree_tuned_all_features_lgbm": dict(model="LightGBM", bucket="all_features", refit_every=10, tuning="random search every 250 sessions, MSE-selected on a 125-session validation tail", lightgbm="4.6.0", xgboost=""),
    "subtree_optuna_tp1_all_features_lgbm": dict(model="LightGBM", bucket="all_features", refit_every=1, tuning="Optuna, best of 50 trials, re-tuned every session", lightgbm="4.6.0", xgboost=""),
    "subtree_optuna_tp250_all_features_lgbm": dict(model="LightGBM", bucket="all_features", refit_every=1, tuning="Optuna, best of 50 trials, re-tuned every 250 sessions", lightgbm="4.6.0", xgboost=""),
    "subtree_lgbm_baseline": dict(model="LightGBM", bucket="baseline", refit_every=10, tuning="shipped configuration of specs/causal_tune_trees.py", lightgbm="4.6.0", xgboost=""),
    "subtree_daily_baseline_lgbm": dict(model="LightGBM", bucket="baseline", refit_every=1, tuning="shipped configuration, refit every session", lightgbm="4.6.0", xgboost=""),
    "sub_ridge_all_features": dict(model="ridge", bucket="all_features", refit_every=1, tuning="one penalty, re-chosen every 250 sessions (spec grid)", lightgbm="", xgboost=""),
    "sub_lasso_all_features": dict(model="lasso", bucket="all_features", refit_every=1, tuning="one penalty, re-chosen every 250 sessions (spec grid)", lightgbm="", xgboost=""),
    "sub_enet_all_features": dict(model="elastic net", bucket="all_features", refit_every=1, tuning="one penalty, re-chosen every 250 sessions (spec grid), l1_ratio 0.5", lightgbm="", xgboost=""),
    "sub_ridge_baseline": dict(model="ridge", bucket="baseline", refit_every=1, tuning="one penalty, re-chosen every 250 sessions (spec grid)", lightgbm="", xgboost=""),
    "sub_lasso_baseline": dict(model="lasso", bucket="baseline", refit_every=1, tuning="one penalty, re-chosen every 250 sessions (spec grid)", lightgbm="", xgboost=""),
    "sub_enet_baseline": dict(model="elastic net", bucket="baseline", refit_every=1, tuning="one penalty, re-chosen every 250 sessions (spec grid), l1_ratio 0.5", lightgbm="", xgboost=""),
}
PRETUNE_STORED = {  # close_trees_pretune.MASTER_KEY
    "stored_T10": "subtree_lgbm_all_features",
    "stored_T1": "subtree_daily_all_features_lgbm",
    "stored_RS10_tp250": "subtree_tuned_all_features_lgbm",
    "stored_OP_tp1": "subtree_optuna_tp1_all_features_lgbm",
    "stored_OP_tp250": "subtree_optuna_tp250_all_features_lgbm",
    "stored_lasso_all_features": "sub_lasso_all_features",
    "stored_lasso_baseline": "sub_lasso_baseline",
}
MODEL = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest", "ridge": "ridge", "lasso": "lasso", "enet": "elastic net", "ols": "OLS"}
SPEC_EST = {"ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet", "ols": "least squares (minimum norm)"}
TREE_SHIPPED = "shipped configuration of specs/causal_tune_trees.py (no tuning); leaf minimum scaled to the training rows by the spec's rule"
LINEAR_SPEC = "specs/causal_tune_linear.py RollingTunedLinear: one penalty re-chosen every 250 solves on a fit / 25 embargo / 125 tail split (spec grid), warm rank-one updates between"
EXOG_TUNING = "penalty re-chosen every 250 sessions on the last 125 sessions of the 2000-session window after a 25-session embargo; refit every session (C port of the spec's algorithm)"
PRETUNE_TUNING = {
    "control_r10": "shipped configuration (rounds 392), no tuning",
    "frozen_k10_r10": "pre-tune best after the first 10 trials of each study (Optuna TPE, 4 blocked pre-2020 folds, validation MSE), frozen",
    "frozen_k45_r10": "pre-tune best after all trials (45 + 26, Optuna TPE, 4 blocked pre-2020 folds, validation MSE), frozen",
    "retune_any_r10": "pre-tune best, then a light retune every 250 sessions (15 TPE trials in a box around the incumbent, 2 x 125-session folds); switch on any validation gain",
    "retune_margin_r10": "pre-tune best, then a light retune every 250 sessions; switch only if the HAC t of the squared-error difference is below the Bonferroni-adjusted 5 % critical value",
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO))


# ============================================================================ parquet writer
def write_parquet(df: pd.DataFrame, path: Path, float_cols: list[str] | None = None, dict_cols: list[str] | None = None) -> None:
    """zstd; floating columns byte-stream split (lossless, it compresses floats much better);
    dictionary encoding only on the given string columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.Table.from_pandas(df, preserve_index=False)
    if float_cols is None:
        float_cols = [f.name for f in t.schema if pa.types.is_floating(f.type)]
    dict_cols = dict_cols or []
    enc = {c: "BYTE_STREAM_SPLIT" for c in float_cols}
    pq.write_table(t, path, compression="zstd", compression_level=ZSTD, use_dictionary=dict_cols, column_encoding=enc)


# ============================================================================ sources
def design() -> dict:
    z = np.load(DESIGN, allow_pickle=False)
    W = int(z["W"])
    d = dict(
        W=W,
        names=[str(v) for v in z["names"]],
        date=pd.DatetimeIndex(pd.to_datetime(z["date"][W:])).astype("datetime64[ns]"),
        y=np.asarray(z["y"][W:], float),
        true_raw=np.asarray(z["true_raw"], float),
        baseline=np.asarray(z["baseline"][W:], float),
        X=np.asarray(z["X"], float),
    )
    assert len(d["date"]) == N_FC
    zb = np.load(DESIGN_BASE, allow_pickle=False)
    d["base_names"] = [str(v) for v in zb["names"]]
    # the baseline design (trees_lineartree base arms) has the same forecast rows and target
    assert np.array_equal(zb["date"], z["date"]) and np.array_equal(zb["y"], z["y"]) and np.array_equal(zb["true_raw"], z["true_raw"])
    return d


def stored_pred(key: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """A stored table's 16:00 ET rows on the forecast stamps (as the studies' stored_pred / stored_1600)."""
    t = pd.read_parquet(SPXW / f"yhat_{key}.parquet")
    et = pd.to_datetime(t["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
    s = pd.Series(t.loc[m, "yhat"].to_numpy(float), index=pd.DatetimeIndex(et[m]))
    out = s.reindex(dates).to_numpy(float)
    assert np.isfinite(out).all(), key
    return out


def load_npz(f: Path) -> tuple[dict, dict]:
    z = np.load(f, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    meta = json.loads(str(out.pop("meta"))) if "meta" in out else {}
    return out, meta


def flatten(d: dict, prefix: str = "meta.") -> dict:
    """Nested dicts flattened with '.'; lists and other values kept as JSON text."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


# ============================================================================ inventory
def inventory(D: dict) -> tuple[list[dict], list[dict]]:
    """(exported series, files left out).  Each series: catalogue fields + 'pred' + record arrays."""
    import close_exogpen as XP

    files = sorted(STUDIES.glob("*/_work/**/*.npz"))
    series, skipped = [], []
    mb_reuse = {}
    for m in ("lgbm", "xgb", "ridge", "lasso"):
        mb_reuse[f"{m}_bars1_r2000"] = f"trees_datasize/{m}_bar1600_w2000"
        mb_reuse[f"{m}_bars2_r4000"] = f"trees_datasize/{m}_pool_w4000"
    for f in files:
        rp = rel(f)
        study = f.relative_to(STUDIES).parts[0]
        name = f.stem
        z, meta = load_npz(f)
        base = dict(study=study, source_file=rp, meta=meta, z=z)
        if study == "exog_penalty" and (name.startswith("gate_") or name.startswith("check_")):
            skipped.append(dict(base, arm=name, kind="gate run" if name.startswith("gate_") else "check run"))
            continue
        if study == "trees_lineartree" and "smoke" in f.relative_to(STUDIES).parts:
            skipped.append(dict(base, arm=f"smoke/{name}", kind="smoke run"))
            continue
        if study == "trees_pretune" and name.startswith("pretune_s"):
            skipped.append(dict(base, arm=name, kind="pre-tune trial records"))
            continue
        arm = name[len("walk_"):] if (study == "trees_pretune" and name.startswith("walk_")) else name
        s = dict(base, arm=arm, series_id=f"{study}/{arm}", source_key="pred", pred=np.asarray(z["pred"], float), kind="forecast")
        assert s["pred"].shape == (N_FC,), rp
        describe(s, D, XP)
        series.append(s)
        if study == "exog_penalty" and "exact_pred" in z and np.isfinite(z["exact_pred"]).any():
            e = dict(base, arm=f"{arm}:exact_pred", series_id=f"{study}/{arm}:exact_pred", source_key="exact_pred", pred=np.asarray(z["exact_pred"], float), kind="exact batch solution (every 25th row)")
            describe(e, D, XP)
            e["tuning"] = f"the exact batch solution (_batch_theta) of {arm}'s window at the penalties and mask in force, every 25th session of each block: a check of the warm homotopy path, not a forecast series of record"
            series.append(e)
    for key, info in STORED.items():
        s = dict(study="master_table", arm=key, series_id=f"master_table/{key}", source_file=rel(SPXW / f"yhat_{key}.parquet"), source_key="yhat (16:00 ET rows)",
                 meta={}, z={}, pred=stored_pred(key, D["date"]), kind="stored master-table forecast")
        s.update(
            model=info["model"], estimator={"ridge": "ridge", "lasso": "reclasso", "elastic net": "reclasticnet"}.get(info["model"], ""), family="one-bar (16:00 rows)",
            design=f"16:00-bar design, {info['bucket']}", bucket=info["bucket"], k=1, rows=2000, rows_min=2000, rows_max=2000, sessions_min=2000, sessions_max=2000, window="2000 sessions",
            bar_column="no", seed=42 if info["model"] in ("LightGBM", "XGBoost") else np.nan, refit_every=info["refit_every"], first_forecast_row=0, tuning=info["tuning"],
            versions_lightgbm=info["lightgbm"], versions_xgboost=info["xgboost"], cpu_source="cluster run (not recorded here)",
            penalty="one penalty" if info["model"] in ("ridge", "lasso", "elastic net") else "",
        )
        series.append(s)
    return series, skipped


def describe(s: dict, D: dict, XP) -> None:  # noqa: C901 - one table of facts
    """Catalogue facts of one npz series, from the study's own conventions and the npz meta."""
    study, arm, z, meta = s["study"], s["arm"].split(":")[0], s["z"], s["meta"]
    v = meta.get("versions", {})
    s.update({f"versions_{k}": v.get(k, "") for k in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "optuna")})
    s["cpu_sec"] = float(meta["cpu_sec"]) if "cpu_sec" in meta else (float(z["cpu_sec"]) if "cpu_sec" in z else np.nan)
    s["cpu_source"] = "npz meta cpu_sec" if "cpu_sec" in meta else ("npz cpu_sec" if "cpu_sec" in z else "")
    s["wall_sec"] = float(meta["wall_sec"]) if "wall_sec" in meta else np.nan
    s["params_json"] = json.dumps(meta["params"]) if isinstance(meta.get("params"), dict) else ""
    s["first_forecast_row"] = int(np.flatnonzero(np.isfinite(s["pred"]))[0])
    s["bar_column"] = "no"
    s["seed"] = np.nan
    if study in ("trees_datasize", "trees_morebars", "trees_kfull"):
        m = meta["model"]
        s["model"] = MODEL[m]
        s["estimator"] = SPEC_EST.get(m, m)
        s["bucket"] = meta["bucket"]
        tree = m in ("lgbm", "xgb", "rf")
        if study == "trees_datasize":
            src, win = meta["source"], meta["window"]
            s["family"] = {"bar1600": "one-bar (16:00 rows)", "pool": "last 2 bars ending 16:00 (both last-hour bars)", "h16": "16:00 rows of the last30 design (learning curve)"}[src]
            s["design"] = {"bar1600": "design_bar1600_all_features", "pool": "design_last30_all_features", "h16": "design_last30_all_features, 16:00 rows"}[src]
            s["k"] = 2 if src == "pool" else 1
            nt = z["n_train"]
            s["rows"] = "expanding" if win == "exp" else int(win)
            s["rows_min"], s["rows_max"] = int(nt.min()), int(nt.max())
            s["sessions_min"], s["sessions_max"] = (int(nt.min()) // 2, int(nt.max()) // 2) if src == "pool" else (int(nt.min()), int(nt.max()))
            s["window"] = "expanding (every earlier 16:00 row; linear: re-anchored every 250 solves)" if win == "exp" else (f"{win // 2} sessions ({win} rows)" if src == "pool" else f"{win} sessions")
            if tree:
                s["seed"] = 42
        else:
            bars = int(meta["bars"])
            s["k"] = bars
            s["design"] = f"design_{meta['design']}_all_features" + (", 16:00 rows" if meta.get("rows_from") == "h16" else "")
            if meta.get("rows_from") == "h16":
                s["family"] = "16:00 rows of the lastbars13 design (one-bar, longer history)"
            elif bars == 1:
                s["family"] = "one-bar (16:00 rows)"
            else:
                s["family"] = f"last {bars} bars ending 16:00"
            s["rows"] = int(meta["rows"])
            s["rows_min"], s["rows_max"] = (int(z["n_train"].min()), int(z["n_train"].max())) if tree else (int(meta["rows"]), int(meta["rows"]))
            ns = z.get("n_sessions")
            if ns is not None:
                s["sessions_min"], s["sessions_max"] = int(ns.min()), int(ns.max())
            s["window"] = f"{meta['rows']} rows" + (f" (2000 sessions x {bars} bars)" if meta.get("group") in ("ladder", "design change", "seed replicate") or study == "trees_kfull" else "")
            s["bar_column"] = "yes" if meta.get("cols") == BAR else "no"
            s["n_columns"] = int(meta.get("n_columns", 628))
            if tree:
                s["seed"] = int(meta.get("seed") or 42)
            s["group"] = meta.get("group", "")
        s["refit_every"] = int(meta["refit_every"])
        if tree:
            p = meta["params"]
            s["leaf_min"] = p.get("min_child_samples", p.get("min_child_weight", np.nan))
            s["tuning"] = TREE_SHIPPED
            s["n_refits"] = int(len(z["anchors"]))
        else:
            s["penalty"] = "one penalty"
            s["tuning"] = LINEAR_SPEC
            s["n_refits"] = int(z["n_solves"])
            s["alpha_grid"] = json.dumps(XP.spec_grids()[SPEC_EST[m]])
            s["l1_ratio"] = 1.0 if m == "lasso" else np.nan
            s["seed"] = np.nan
    elif study == "trees_lineartree":
        s["model"] = "LightGBM"
        s["estimator"] = "lgbm"
        s["bucket"] = meta["bucket"]
        s["family"] = "one-bar (16:00 rows)"
        s["design"] = f"design_bar1600_{meta['bucket']}"
        s["k"] = 1
        s["rows"] = s["rows_min"] = s["rows_max"] = int(meta["train_win"])
        s["sessions_min"] = s["sessions_max"] = int(meta["train_win"])
        s["window"] = f"{meta['train_win']} sessions"
        s["seed"] = int(meta["seed"])
        s["refit_every"] = int(meta["refit_every"])
        s["linear_tree"] = "yes" if meta["linear_tree"] else "no"
        s["linear_lambda"] = meta["linear_lambda"] if meta["linear_lambda"] is not None else np.nan
        s["leaf_min"] = meta["params"]["min_child_samples"]
        s["tuning"] = TREE_SHIPPED + ("; linear_tree=True (linear model in every leaf)" if meta["linear_tree"] else "")
        s["n_refits"] = int(len(z["anchors"]))
        s["n_columns"] = len(D["names"]) if meta["bucket"] == "all_features" else len(D["base_names"])
    elif study == "trees_pretune":
        s["model"] = "LightGBM"
        s["estimator"] = "lgbm"
        s["bucket"] = "all_features"
        s["family"] = "one-bar (16:00 rows)"
        s["design"] = "design_bar1600_all_features"
        s["k"] = 1
        s["rows"] = s["rows_min"] = s["rows_max"] = 2000
        s["sessions_min"] = s["sessions_max"] = 2000
        s["window"] = "2000 sessions"
        s["seed"] = 42
        s["refit_every"] = int(meta["refit_every"])
        s["tuning"] = PRETUNE_TUNING[arm]
        s["schedule_json"] = str(z["schedule"])
        sched = json.loads(str(z["schedule"]))
        s["leaf_min"] = sched[0][1]["min_child_samples"]
        s["n_refits"] = int(len(z["rows"]))
        s["cpu_sec"] = float(np.sum(z["fit_sec"]))
        s["cpu_source"] = "sum of the refits' fit seconds (single-threaded; no cpu_sec recorded)"
        s["n_columns"] = len(D["names"])
    elif study == "exog_penalty":
        est = arm.split("_")[0]
        mode = arm.split("_", 1)[1]
        s["model"] = MODEL[est]
        s["estimator"] = SPEC_EST[est]
        s["bucket"] = "baseline" if est == "ols" else "all_features"
        s["family"] = "one-bar (16:00 rows)"
        s["design"] = "design_bar1600_baseline columns of design_bar1600_all_features" if est == "ols" else "design_bar1600_all_features"
        s["k"] = 1
        s["rows"] = s["rows_min"] = s["rows_max"] = 2000
        s["sessions_min"] = s["sessions_max"] = 2000
        s["window"] = "2000 sessions"
        s["refit_every"] = 1
        s["n_refits"] = N_FC
        s["versions_numpy"] = ""
        s["cpu_source"] = s["cpu_source"] + " (C kernel experiments/close_exogpen_kernel.c, gcc -O3, reference LAPACK / BLAS)" if est != "ols" else s["cpu_source"]
        if est == "ols":
            s["penalty"] = "none (HAR + calendar OLS: the master table's sub_ols_baseline, centered window least squares)"
            s["tuning"] = "none; refit every session"
            s["n_columns"] = len(D["base_names"])
        else:
            s["penalty"] = XP.MODE_LABEL[mode]
            s["tuning"] = EXOG_TUNING
            grid = XP.WIDE_GRIDS[SPEC_EST[est]] if mode.endswith("w") else XP.spec_grids()[SPEC_EST[est]]
            s["alpha_grid"] = json.dumps(grid)
            s["r_grid"] = json.dumps(list(XP.R_OF[mode]))
            s["l1_ratio"] = {"ridge": 0.0, "lasso": 1.0, "enet": 0.5}[est]
            s["n_columns"] = len(D["names"]) + 1
            s["npz.n_singular"] = json.dumps(np.asarray(z["n_singular"]).tolist())
            s["npz.n_reseed"] = json.dumps(np.asarray(z["n_reseed"]).tolist())
    for k in ("n_reseed", "n_solves"):
        if k in z and np.ndim(z[k]) == 0:
            s[f"npz.{k}"] = int(z[k])


# ============================================================================ build
def research_clock(D: dict, pred: np.ndarray, start: int) -> np.ndarray:
    import dense_vs_sparse_1530 as dvs

    rows = np.arange(start, N_FC)
    dz = dict(date=D["date"].to_numpy()[rows], true_adj=D["y"][rows], true_raw=D["true_raw"][rows])
    fr = dvs.research_frame(dz, pred[rows])
    out = np.full(N_FC, np.nan)
    pos = pd.DatetimeIndex(D["date"]).get_indexer(fr.index)
    out[pos] = fr["pred_clock"].to_numpy(float)
    return out


def build_targets(D: dict) -> pd.DataFrame:
    dk = pd.read_parquet(DECK).sort_index()
    dk.index = pd.DatetimeIndex(pd.to_datetime(dk.index)).normalize().as_unit("ns")
    day = pd.DatetimeIndex(D["date"]).normalize()
    t = pd.DataFrame(
        dict(
            row=np.arange(N_FC, dtype=np.int16),
            date=D["date"],
            true_adj=D["y"],
            true_raw=D["true_raw"],
            baseline=D["true_raw"] / D["y"] ** 2,
            baseline_design=D["baseline"],
            is_trade_day=day.isin(dk.index),
        )
    )
    assert int(t["is_trade_day"].sum()) == len(dk) == 866
    sel = dk.reindex(day)
    for c in DECK_COLS:
        t[c] = sel[c].to_numpy()  # dtypes kept (ask / bid float32, as the scorer adds them)
    return t


def build() -> None:  # noqa: C901 - one linear driver
    OUT.mkdir(parents=True, exist_ok=True)
    D = design()
    say(f"design: {len(D['names'])} columns, {N_FC} forecast rows {D['date'][0]} .. {D['date'][-1]}")
    T = build_targets(D)
    write_parquet(T, OUT / "targets.parquet", float_cols=["true_adj", "true_raw", "baseline", "baseline_design", *DECK_COLS])
    series, skipped = inventory(D)
    say(f"series: {len(series)} exported, {len(skipped)} npz files left out")

    # ---------------------------------------------------------------- forecasts
    fr, ident = [], {}
    for s in series:
        p = s["pred"]
        fin = np.isfinite(p)
        s["n_forecasts"] = int(fin.sum())
        start = int(np.flatnonzero(fin)[0])
        clock = research_clock(D, p, start)
        s["n_pred_clock"] = int(np.isfinite(clock).sum())
        fr.append(pd.DataFrame(dict(series_id=s["series_id"], study=s["study"], row=np.arange(N_FC, dtype=np.int16), date=D["date"], pred_adj=p, pred_clock=clock)))
        ident.setdefault(np.where(fin, p, np.inf).tobytes(), []).append(s["series_id"])
    F = pd.concat(fr, ignore_index=True)
    write_parquet(F, OUT / "forecasts.parquet", float_cols=["pred_adj", "pred_clock"], dict_cols=["series_id", "study"])
    same = {sid: [o for o in grp if o != sid] for grp in ident.values() for sid in grp}

    # ---------------------------------------------------------------- internals
    layout = build_linear(D, series)
    build_trees(D, series)
    build_refits(D, series)
    build_pretune(skipped)
    # ---------------------------------------------------------------- reported scores (each series' home CSV)
    reported = reported_scores()
    # ---------------------------------------------------------------- catalogue
    rows = []
    for s in series:
        r = {k: v for k, v in s.items() if k not in ("pred", "z", "meta")}
        r["exported"] = "yes"
        r["exclusion_reason"] = ""
        r["is_exact_pred"] = "yes" if s["source_key"] == "exact_pred" else "no"
        r["identical_to"] = ";".join(same.get(s["series_id"], []))
        r["scored_from_row"] = {"exog_penalty": 0, "trees_pretune": 0, "master_table": 0}.get(s["study"], TREE_START)
        r["scoreable"] = "no" if r["is_exact_pred"] == "yes" else "yes"
        rep = reported.get(s["series_id"], {})
        r.update(rep)
        cpu, wall = r.pop("cpu_sec", np.nan), r.pop("wall_sec", np.nan)
        if r["is_exact_pred"] == "yes":  # a check of its parent arm: the CPU is the parent's
            cpu, wall, r["cpu_source"] = np.nan, np.nan, f"part of {s['series_id'].split(':')[0]}"
        r["cpu_min"] = cpu / 60.0
        r["wall_min"] = wall / 60.0
        r["has_coefficients"] = "yes" if ("theta" in s["z"] or "theta_bb" in s["z"]) and r["is_exact_pred"] == "no" else "no"
        r["has_importance"] = "yes" if "imp_gain" in s["z"] else "no"
        r["theta_layout_max_rel_gap"] = layout.get(s["series_id"], np.nan) if r["has_coefficients"] == "yes" else np.nan
        r["npz_keys"] = ";".join(sorted(list(s["z"].keys()) + (["meta"] if s["meta"] else []))) if s["z"] else ""
        r.update(flatten(s["meta"]))
        r["meta_json"] = json.dumps(s["meta"]) if s["meta"] else ""
        rows.append(r)
    exported_pred = {s["series_id"]: s["pred"] for s in series}
    for s in skipped:
        r = dict(study=s["study"], arm=s["arm"], series_id=f"{s['study']}/{s['arm']}", source_file=s["source_file"], kind=s["kind"], exported="no", is_exact_pred="no", scoreable="no")
        r["exclusion_reason"] = skip_reason(s, exported_pred)
        r["npz_keys"] = ";".join(sorted(list(s["z"].keys()) + (["meta"] if s["meta"] else [])))
        r.update(flatten(s["meta"]))
        r["meta_json"] = json.dumps(s["meta"]) if s["meta"] else ""
        rows.append(r)
    C = pd.DataFrame(rows)
    lead = ["series_id", "study", "arm", "kind", "exported", "exclusion_reason", "source_file", "source_key", "model", "estimator", "family", "design", "bucket", "k", "rows", "rows_min", "rows_max",
            "sessions_min", "sessions_max", "window", "bar_column", "seed", "refit_every", "first_forecast_row", "n_forecasts", "n_pred_clock", "n_refits", "n_columns", "penalty", "alpha_grid", "r_grid",
            "l1_ratio", "linear_tree", "linear_lambda", "leaf_min", "tuning", "params_json", "schedule_json", "group", "versions_numpy", "versions_pandas", "versions_sklearn", "versions_lightgbm",
            "versions_xgboost", "versions_optuna", "cpu_min", "cpu_source", "wall_min", "reported_qlike", "reported_sharpe_mid", "reported_sharpe_crossed", "reported_in", "scored_from_row",
            "scoreable", "is_exact_pred", "identical_to", "has_coefficients", "has_importance", "theta_layout_max_rel_gap", "npz_keys"]
    for c in lead:
        if c not in C:
            C[c] = np.nan
    rest = sorted(c for c in C.columns if c not in lead and c != "meta_json")
    C = C[lead + rest + ["meta_json"]]
    C.to_csv(OUT / "catalogue.csv", index=False)
    say(f"catalogue: {len(C)} rows ({(C['exported'] == 'yes').sum()} exported), {C.shape[1]} columns")

    # ---------------------------------------------------------------- recipes (study CSV rows -> series)
    SA = study_arm_rows({s["source_file"]: s["series_id"] for s in series if s["source_key"] == "pred"})
    SA.to_csv(OUT / "study_arms.csv", index=False)
    say(f"study_arms: {len(SA)} rows")

    sizes = {p.name: p.stat().st_size for p in sorted(OUT.iterdir()) if p.is_file()}
    for n, b in sizes.items():
        say(f"  {n:32s} {b / 1e6:8.2f} MB")
        assert b / 1e6 < LIMIT_MB, (n, b)
    say(f"total {sum(sizes.values()) / 1e6:.2f} MB")


def skip_reason(s: dict, exported: dict) -> str:
    p = np.asarray(s["z"].get("pred", np.array([])), float)
    if s["kind"] == "pre-tune trial records":
        return "no forecasts: the pre-tune's trial records (one Optuna study); exported to pretune_trials.parquet"
    if s["kind"] == "smoke run":
        full = exported[f"trees_lineartree/{s['arm'].split('/')[1]}"]
        m = np.isfinite(p)
        same = np.array_equal(p[m], full[m])
        return f"smoke run: first 2 refits only ({int(m.sum())} forecasts, rows {np.flatnonzero(m)[0]}..{np.flatnonzero(m)[-1]}); " + (
            "bit for bit equal to the whole arm on those rows" if same else "differs from the whole arm")
    est = s["arm"].rsplit("_", 1)[1]
    short = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}[est]
    c_arm = exported[f"exog_penalty/{short}_single"]
    st = exported[f"master_table/sub_{short}_all_features"]
    if s["kind"] == "check run":
        return ("check run (the C port's one-penalty arm, gate 2 of exog_penalty/SUMMARY.md): "
                + ("bit for bit equal to exog_penalty/" + short + "_single" if np.array_equal(p, c_arm) else f"max relative difference {np.max(np.abs(p / c_arm - 1)):.1e} to exog_penalty/{short}_single"))
    return (f"gate run (the spec's own Python class on the cached design, gate 1 of exog_penalty/SUMMARY.md): max relative difference {np.max(np.abs(p / st - 1)):.1e} to "
            f"master_table/sub_{short}_all_features and {np.max(np.abs(p / c_arm - 1)):.1e} to exog_penalty/{short}_single")


def reported_scores() -> dict:
    """The home study's own QLIKE / Sharpe for each series."""
    out = {}

    def put(sid, qlike, sh, shx, where):
        out[sid] = dict(reported_qlike=float(qlike), reported_sharpe_mid=float(sh), reported_sharpe_crossed=float(shx) if shx == shx else np.nan, reported_in=where)

    a = pd.read_csv(STUDIES / "trees_datasize" / "arms.csv")
    for r in a.itertuples():
        put(f"trees_datasize/{r.arm}", r.qlike, r.sharpe_mid, r.sharpe_crossed, "trees_datasize/arms.csv")
    a = pd.read_csv(STUDIES / "trees_morebars" / "arms.csv")
    for r in a.itertuples():
        if not isinstance(r.reused_from, str):
            put(f"trees_morebars/{r.arm}", r.qlike, r.sharpe_mid, r.sharpe_crossed, "trees_morebars/arms.csv")
    a = pd.read_csv(STUDIES / "trees_kfull" / "arms.csv")
    for r in a.itertuples():
        if r.exists == "yes" and "/trees_kfull/_work/" in r.path:
            put(f"trees_kfull/{Path(r.path).stem}", r.qlike, r.sharpe_mid, r.sharpe_crossed, "trees_kfull/arms.csv")
    a = pd.read_csv(STUDIES / "trees_lineartree" / "arms.csv")
    for r in a.itertuples():
        if not r.arm.startswith("stored:"):
            put(f"trees_lineartree/{r.arm}", r.qlike, r.sharpe_mid, r.sharpe_crossed, "trees_lineartree/arms.csv")
    a = pd.read_csv(STUDIES / "trees_pretune" / "local_lgbm_arms.csv")
    for r in a.itertuples():
        if not r.arm.startswith("stored_"):
            put(f"trees_pretune/{r.arm}", r.qlike, r.sharpe_mid, r.sharpe_crossed, "trees_pretune/local_lgbm_arms.csv")
    a = pd.read_csv(STUDIES / "exog_penalty" / "headline.csv")
    for r in a.itertuples():
        if not r.key.startswith("stored_"):
            put(f"exog_penalty/{r.key}", r.qlike, r.sharpe_mid, r.sharpe_crossed, "exog_penalty/headline.csv")
    m = pd.read_csv(MASTER).drop_duplicates("key").set_index("key")
    for k in STORED:
        put(f"master_table/{k}", m.loc[k, "qlike_recal"], m.loc[k, "Sharpe_mid"], m.loc[k, "Sharpe_crossed"], rel(MASTER))
    return out


def study_arm_rows(by_file: dict) -> pd.DataFrame:  # noqa: C901 - one table of recipes
    """One row for each row of the study CSVs that verify checks: the series it is, or the recipe
    (mean / weighted, members in order) that rebuilds it from exported series."""
    R = []

    def single(table, label, sid, start, metrics):
        R.append(dict(table=table, label=label, recipe="single", series_id=sid, members="", weights="", score_start=start, metrics=metrics))

    def recipe(table, label, kind, members, start, metrics, weights=None):
        R.append(dict(table=table, label=label, recipe=kind, series_id="", members=json.dumps(members), weights=json.dumps(weights) if weights else "", score_start=start, metrics=metrics))

    QS = "qlike;sharpe_mid;sharpe_crossed"
    # trees_kfull/arms.csv: the file of each arm
    a = pd.read_csv(STUDIES / "trees_kfull" / "arms.csv")
    for r in a.itertuples():
        if r.exists == "yes":
            single("trees_kfull/arms.csv", r.arm, by_file[r.path], TREE_START, QS)
    # trees_morebars/arms.csv: rungs 1 and 2 reuse the data-size arms
    mb_reuse = {}
    for m in ("lgbm", "xgb", "ridge", "lasso"):
        mb_reuse[f"{m}_bars1_r2000"] = f"trees_datasize/{m}_bar1600_w2000"
        mb_reuse[f"{m}_bars2_r4000"] = f"trees_datasize/{m}_pool_w4000"
    a = pd.read_csv(STUDIES / "trees_morebars" / "arms.csv")
    for r in a.itertuples():
        single("trees_morebars/arms.csv", r.arm, mb_reuse.get(r.arm, f"trees_morebars/{r.arm}"), TREE_START, QS)
    # trees_morebars/average.csv (experiments/close_trees_morebars_average.py combos, members in its order)
    lad = [mb_reuse.get(f"lgbm_bars{n}_r{n * 2000}", f"trees_morebars/lgbm_bars{n}_r{n * 2000}") for n in (1, 2, 3, 4, 5, 7, 13)]
    xgb = [mb_reuse.get(f"xgb_bars{n}_r{n * 2000}", f"trees_morebars/xgb_bars{n}_r{n * 2000}") for n in (1, 2, 3, 4, 5, 7, 13)]
    seed = {n: f"trees_morebars/lgbm_bars{n}_r{n * 2000}_seed43" for n in (1, 2, 3, 4, 13)}
    ridge = "exog_penalty/ridge_bb0"
    T = "trees_morebars/average.csv"
    single(T, "lgbm, 1 bar (control)", lad[0], TREE_START, QS)
    single(T, "lgbm, 2 bars", lad[1], TREE_START, QS)
    single(T, "lgbm, 4 bars (best single pool)", lad[3], TREE_START, QS)
    allp = "average of lgbm pools N = 1, 2, 3, 4, 5, 7, 13"
    recipe(T, allp, "mean", lad, TREE_START, QS)
    recipe(T, "average of lgbm pools N = 1 .. 5", "mean", lad[:5], TREE_START, QS)
    recipe(T, "average of lgbm pools N = 1 .. 4", "mean", lad[:4], TREE_START, QS)
    recipe(T, "average of lgbm pools N = 2 .. 5", "mean", lad[1:5], TREE_START, QS)
    recipe(T, "average of lgbm pools N = 3, 4", "mean", lad[2:4], TREE_START, QS)
    recipe(T, "average of lgbm 4 bars, seeds 42 and 43", "mean", [lad[3], seed[4]], TREE_START, QS)
    recipe(T, "average of lgbm 2 bars, seeds 42 and 43", "mean", [lad[1], seed[2]], TREE_START, QS)
    recipe(T, "average of lgbm pools N = 1, 2, 3, 4, 13, seeds 42 and 43", "mean", [*[lad[i] for i in (0, 1, 2, 3, 6)], *[seed[n] for n in (1, 2, 3, 4, 13)]], TREE_START, QS)
    recipe(T, "average of lgbm and xgb pools, all N", "mean", lad + xgb, TREE_START, QS)
    single(T, "ridge, backbone unpenalized (linear reference)", ridge, TREE_START, QS)
    recipe(T, "equal weights: ridge backbone unpenalized + average of lgbm pools, all N", "weighted", [ridge, allp], TREE_START, QS, [0.5, 0.5])
    recipe(T, "equal weights: ridge backbone unpenalized + lgbm 4 bars", "weighted", [ridge, lad[3]], TREE_START, QS, [0.5, 0.5])
    # trees_datasize/arms.csv
    a = pd.read_csv(STUDIES / "trees_datasize" / "arms.csv")
    for r in a.itertuples():
        single("trees_datasize/arms.csv", r.arm, f"trees_datasize/{r.arm}", TREE_START, QS)
    # trees_lineartree/arms.csv
    a = pd.read_csv(STUDIES / "trees_lineartree" / "arms.csv")
    for r in a.itertuples():
        sid = f"master_table/{r.arm.split(':', 1)[1]}" if r.arm.startswith("stored:") else f"trees_lineartree/{r.arm}"
        single("trees_lineartree/arms.csv", r.arm, sid, TREE_START, QS)
    # trees_pretune/local_lgbm_arms.csv (scored on all 1469 rows)
    a = pd.read_csv(STUDIES / "trees_pretune" / "local_lgbm_arms.csv")
    for r in a.itertuples():
        sid = f"master_table/{PRETUNE_STORED[r.arm]}" if r.arm in PRETUNE_STORED else f"trees_pretune/{r.arm}"
        single("trees_pretune/local_lgbm_arms.csv", r.arm, sid, 0, QS)
    # exog_penalty/headline.csv (scored on all 1469 rows)
    a = pd.read_csv(STUDIES / "exog_penalty" / "headline.csv")
    for r in a.itertuples():
        sid = f"master_table/{r.key[len('stored_'):]}" if r.key.startswith("stored_") else f"exog_penalty/{r.key}"
        single("exog_penalty/headline.csv", r.key, sid, 0, QS)
    # kfull_tests/forecasts.csv: singles by file; seed averages over the singles listed before them; family averages over k
    a = pd.read_csv(STUDIES / "kfull_tests" / "forecasts.csv")
    T = "kfull_tests/forecasts.csv"
    QM = "qlike;mse;qlike_2019"
    singles: dict[str, list[str]] = {}
    for r in a.itertuples():
        nm = r.forecast
        if isinstance(r.file, str) and r.file:
            single(T, nm, by_file[r.file], TREE_START, QM)
            if r.kind == "single seed":
                singles.setdefault(nm.rsplit(":s", 1)[0], []).append(nm)
        elif r.kind.startswith("seed average"):
            recipe(T, nm, "mean", singles[nm], TREE_START, QM)
        elif r.kind.startswith("average of the seed averages"):
            fam = nm.split(":")[0]
            ks = sorted(int(x.split(":k")[1]) for x in a["forecast"] if x.startswith(f"{fam}:k") and ":s" not in x)
            recipe(T, nm, "mean", [f"{fam}:k{k}" for k in ks], TREE_START, QM)
        else:
            raise ValueError(nm)
    # kfull_tests/realtime_combination.csv: the ridge + trees combinations (all 866 days: equal weights;
    # trade days 251 .. 866: real-time lambda)
    rt = pd.read_csv(STUDIES / "kfull_tests" / "realtime_combination.csv")
    T2 = "kfull_tests/realtime_combination.csv"
    for nm in rt["forecast"].unique():
        recipe(T2, f"{nm}:eq_ridge", "weighted", [ridge, nm], TREE_START, "equal weights, adjusted scale, rescored", [0.5, 0.5])
        recipe(T2, f"{nm}:variance", "variance", [ridge, nm], TREE_START, "equal weights, variance level", [0.5, 0.5])
        recipe(T2, f"{nm}:realtime", "realtime", [ridge, nm], TREE_START, "real-time lambda (QLIKE on past trade days, expanding)")
    S = pd.DataFrame(R)
    S.insert(0, "study", S["table"].str.split("/").str[0])
    return S


# ---------------------------------------------------------------- linear internals
def build_linear(D: dict, series: list[dict]) -> None:
    import close_exogpen as XP

    names = D["names"]
    bb = np.isin(names, D["base_names"])
    grp = XP.exog_groups(names, bb)
    cols = pd.DataFrame(dict(col=np.arange(len(names) + 1), name=names + ["intercept"],
                             block=["backbone" if b else "exogenous" for b in bb] + ["intercept"],
                             group=[("backbone" if g < 0 else XP.GROUPS[g]) for g in grp] + ["intercept"]))
    cols.to_csv(OUT / "linear_columns.csv", index=False)
    Xaug = np.hstack([D["X"], np.ones((len(D["X"]), 1))])
    W = D["W"]
    parts, rech, checks = [], [], []
    blocks = XP.block_starts(N_FC)
    for s in series:
        if s["study"] != "exog_penalty" or s["source_key"] != "pred":
            continue
        z = s["z"]
        if "theta" in z:
            th = np.asarray(z["theta"], np.float32)
            assert th.shape == (N_FC, len(names) + 1)
            # the C kernel's layout: pred = [X_row, 1] . theta (theta stored as float32)
            recon = np.einsum("ij,ij->i", Xaug[W:], th.astype(np.float64))
            relgap = float(np.max(np.abs(recon - s["pred"]) / np.maximum(np.abs(s["pred"]), 1e-12)))
            checks.append(dict(series_id=s["series_id"], max_rel_gap_pred_vs_X_theta=relgap))
            assert relgap < 1e-4, (s["series_id"], relgap)
            df = pd.DataFrame(th, columns=names + ["intercept"])
            for b, (i0, i1) in enumerate(blocks):
                lk = np.flatnonzero(z["locked"][b]).astype(np.int16)
                mk = np.flatnonzero(z["maskout"][b]).astype(np.int16)
                vm = np.asarray(z["val_mse"][b], float)
                mode = s["arm"].split("_", 1)[1]
                est = s["arm"].split("_")[0]
                grid = XP.WIDE_GRIDS[SPEC_EST[est]] if mode.endswith("w") else XP.spec_grids()[SPEC_EST[est]]
                rech.append(dict(
                    series_id=s["series_id"], block=b, row=i0, date=D["date"][i0], n_rows=i1 - i0,
                    alpha=float(z["alpha_blk"][b]), r=float(z["r_blk"][b]),
                    **{f"penalty_{g}": float(z["pen_g"][b, gi]) for gi, g in enumerate(XP.GROUPS)},
                    mpr_val_mse=float(z["mpr_mse"][b]), val_mse_best=float(np.nanmin(vm)) if np.isfinite(vm).any() else np.nan,
                    val_mse=vm.ravel().tolist(), val_mse_shape=f"{vm.shape[0]} r x {vm.shape[1]} alpha",
                    alpha_grid=list(map(float, grid)), r_grid=list(map(float, XP.R_OF[mode])),
                    n_reseed=int(z["n_reseed"][b]), n_locked=int(len(lk)) - 1, n_masked=int(len(mk)),
                    locked_cols=lk.tolist(), masked_cols=mk.tolist(), n_train=2000, source="exog_penalty npz",
                ))
        elif "theta_bb" in z:
            th = np.asarray(z["theta_bb"], float).astype(np.float32)
            df = pd.DataFrame(np.nan, index=np.arange(N_FC), columns=names + ["intercept"], dtype=np.float32)
            bbi = np.flatnonzero(bb)
            assert th.shape == (N_FC, len(bbi))
            df.iloc[:, bbi] = th
        else:
            continue
        df.insert(0, "date", D["date"])
        df.insert(0, "row", np.arange(N_FC, dtype=np.int16))
        df.insert(0, "series_id", s["series_id"])
        parts.append(df)
    # the data-size / bar-count linear arms: no coefficients stored, only the masked count at each tune
    for s in series:
        z = s["z"]
        if s["study"] in ("trees_datasize", "trees_morebars") and "mask_trace" in z:
            for b, (nt, mk) in enumerate(zip(z["n_train"], z["mask_trace"])):
                rech.append(dict(series_id=s["series_id"], block=b, row=np.nan, date=pd.NaT, n_rows=np.nan, alpha=np.nan, r=np.nan,
                                 n_masked=int(mk), n_train=int(nt), source="spec class trace (block b = solves 250 b .. 250 b + 249 of the series)"))
    L = pd.concat(parts, ignore_index=True)
    fc = [c for c in L.columns if c not in ("series_id", "row", "date")]
    write_parquet(L, OUT / "linear_coefficients.parquet", float_cols=fc, dict_cols=["series_id"])
    Rc = pd.DataFrame(rech)
    for c in ("locked_cols", "masked_cols", "val_mse", "alpha_grid", "r_grid"):
        Rc[c] = Rc[c].apply(lambda v: v if isinstance(v, list) else [])
    write_parquet(Rc, OUT / "linear_rechoices.parquet", dict_cols=["series_id", "val_mse_shape", "source"])
    say(f"linear: {L['series_id'].nunique()} arms x {N_FC} rows; re-choices {len(Rc)}; [X, 1] . theta vs pred max relative gap {max(c['max_rel_gap_pred_vs_X_theta'] for c in checks):.2e}")
    return {c["series_id"]: c["max_rel_gap_pred_vs_X_theta"] for c in checks}


# ---------------------------------------------------------------- tree importance
def build_trees(D: dict, series: list[dict]) -> None:
    names = D["names"] + [BAR]
    pd.DataFrame(dict(col=np.arange(len(names)), name=names)).to_csv(OUT / "tree_columns.csv", index=False)
    parts = []
    for s in series:
        z = s["z"]
        if "imp_gain" not in z:
            continue
        sp, g, k = z["imp_split"], z["imp_gain"], z["imp_kept"]
        p = sp.shape[1]
        assert p == s["n_columns"] and p in (len(D["names"]), len(D["names"]) + 1), (s["series_id"], p)
        # a column the window mask dropped has no split and no gain
        assert not sp[~k].any() and not g[~k].any()
        r, c = np.nonzero(k)
        assert sp.max() < 2**15
        parts.append(pd.DataFrame(dict(series_id=s["series_id"], refit=r.astype(np.int16), row=z["anchors"][r].astype(np.int16), col=c.astype(np.int16),
                                       split=sp[r, c].astype(np.int16), gain=g[r, c].astype(np.float32))))
    imp = pd.concat(parts, ignore_index=True)
    write_parquet(imp, OUT / "tree_importance.parquet", float_cols=["gain"], dict_cols=["series_id"])
    say(f"tree importance: {imp['series_id'].nunique()} arms, {len(imp)} kept (refit, column) cells")


# ---------------------------------------------------------------- refit-level records
def build_refits(D: dict, series: list[dict]) -> None:  # noqa: C901
    rows = []
    for s in series:
        z, sid = s["z"], s["series_id"]
        if s["source_key"] != "pred" or not z:
            continue
        if "anchors" in z or "rows" in z:  # tree refits
            a = np.asarray(z["anchors"] if "anchors" in z else z["rows"], int)
            nxt = np.append(a[1:], N_FC)
            d = dict(series_id=sid, kind="tree refit", refit=np.arange(len(a)), row=a, date=D["date"][a], n_forecast_rows=nxt - a)
            for k in ("fit_sec", "n_train", "n_sessions", "kept_n", "leaf_min", "coef_max", "share_linear", "mean_feat", "fit_lo", "fit_hi"):
                if k in z and len(z[k]) == len(a):
                    d[k] = np.asarray(z[k])
            if "n_kept" in z:
                d["kept_n"] = np.asarray(z["n_kept"])
            if "schedule" in z:  # trees_pretune: the configuration in force at each refit
                sched = json.loads(str(z["schedule"]))
                cur = [[c for c in sched if c[0] <= i][-1] for i in a]
                for ax in sched[0][1]:
                    d[f"cfg_{ax}"] = np.array([c[1][ax] for c in cur], float)
                d["cfg_rounds"] = np.array([c[2] for c in cur], float)
                d["leaf_min"] = d["cfg_min_child_samples"]
            if s["study"] == "trees_lineartree":
                d["n_train"] = np.full(len(a), 2000)
                d["leaf_min"] = np.full(len(a), float(s["leaf_min"]))
            rows.append(pd.DataFrame(d))
        elif "alpha" in z and np.shape(z["alpha"]) == (N_FC,):  # linear spec arms: alpha at every forecast row
            rows.append(pd.DataFrame(dict(series_id=sid, kind="linear solve", refit=np.arange(N_FC), row=np.arange(N_FC), date=D["date"], n_forecast_rows=1, alpha=np.asarray(z["alpha"], float))))
        elif "events" in z:  # exog_penalty: homotopy events of each session's updates; alpha / r of the re-choice in force
            b = np.arange(N_FC) // 250
            rows.append(pd.DataFrame(dict(series_id=sid, kind="linear solve", refit=np.arange(N_FC), row=np.arange(N_FC), date=D["date"], n_forecast_rows=1,
                                          alpha=np.asarray(z["alpha_blk"], float)[b], r=np.asarray(z["r_blk"], float)[b], homotopy_events=np.asarray(z["events"], np.int32))))
    Rf = pd.concat(rows, ignore_index=True)
    for c in ("refit", "row", "n_forecast_rows"):
        Rf[c] = Rf[c].astype(np.int16)
    for c in ("n_train", "n_sessions", "kept_n", "homotopy_events"):
        if c in Rf:
            Rf[c] = Rf[c].astype("Int32")
    write_parquet(Rf, OUT / "refits.parquet", dict_cols=["series_id", "kind"])
    say(f"refits: {len(Rf)} rows, {Rf['series_id'].nunique()} series")


def build_pretune(skipped: list[dict]) -> None:
    rows = []
    for s in skipped:
        if s["kind"] != "pre-tune trial records":
            continue
        z, meta = s["z"], s["meta"]
        axes = meta["axes"]
        folds = meta["folds"]
        n, nf = z["fold_val_mse"].shape
        for t in range(n):
            for f in range(nf):
                fo = folds[f]
                rows.append(dict(study=f"s{meta['seed']}", tpe_seed=meta["tpe_seed_base"] + meta["seed"], trial=t, fold=f, **{ax: float(z["params"][t, i]) for i, ax in enumerate(axes)},
                                 val_mse=float(z["val_mse"][t]), fold_val_mse=float(z["fold_val_mse"][t, f]), fold_val_qlike=float(z["fold_val_qlike"][t, f]),
                                 fold_rounds=float(z["fold_rounds"][t, f]), fold_rounds_max=float(z["fold_rounds_max"][t, f]), fold_sec=float(z["fold_sec"][t, f]),
                                 fold_n_kept=float(z["fold_n_kept"][t, f]), fit_first=fo["fit_first"], val_first=fo["val_first"], val_last=fo["val_last"],
                                 source_file=s["source_file"]))
    P = pd.DataFrame(rows)
    write_parquet(P, OUT / "pretune_trials.parquet", dict_cols=["study", "fit_first", "val_first", "val_last", "source_file"])
    say(f"pretune trials: {P[['study', 'trial']].drop_duplicates().shape[0]} trials x {P['fold'].nunique()} folds")


# ============================================================================ verify
def verify() -> None:  # noqa: C901 - one linear check
    import close_studies_load as L
    from src.evaluation.diebold_mariano import dm_test

    L._read.cache_clear()
    L._recipes.cache_clear()
    rows = []

    def chk(table, label, quantity, value, reference, tol=TOL, relative=False, detail="", check="score vs study CSV"):
        if reference is None or (isinstance(reference, float) and not np.isfinite(reference)):
            return
        diff = abs(value - reference)
        lim = tol * abs(reference) if relative else tol
        rows.append(dict(check=check, table=table, label=label, quantity=quantity, value=value, reference=reference, abs_diff=diff,
                         tolerance=f"{tol:g}{' relative' if relative else ''}", passed=bool(diff <= lim), detail=detail))

    cat = L.catalogue(exported_only=True)
    SA = L.study_arms()
    t0 = time.time()
    # ---------------------------------------------------------------- every exported series scored (both starts)
    scoreable = cat.loc[cat["scoreable"] == "yes", "series_id"].tolist()
    S130 = L.score(scoreable, start=TREE_START)
    S0 = L.score([s for s in scoreable if cat.set_index("series_id").loc[s, "first_forecast_row"] == 0], start=0)
    say(f"scored {len(S130)} series from row {TREE_START} and {len(S0)} from row 0 ({time.time() - t0:.0f}s)")
    assert len(S130) == len(scoreable), set(scoreable) - set(S130.index)
    # the catalogue's reported score = the score from the export, scored from the home study's row
    for r in cat.itertuples():
        if r.scoreable != "yes" or not isinstance(r.reported_in, str):
            continue
        sc = (S0 if r.scored_from_row == 0 else S130).loc[r.series_id]
        chk(r.reported_in, r.series_id, "qlike", float(sc["qlike"]), float(r.reported_qlike), check="catalogue reported score vs export")
        chk(r.reported_in, r.series_id, "sharpe_mid", float(sc["sharpe_mid"]), float(r.reported_sharpe_mid), check="catalogue reported score vs export")
    # ---------------------------------------------------------------- the stored pred_clock column
    import dense_vs_sparse_1530 as dvs

    Tg = L.targets()
    Fa, Fc = L.forecasts(value="pred_adj"), L.forecasts(value="pred_clock")
    Pq = L.panel(Fa[scoreable], start=TREE_START)
    for sid in scoreable:
        p = Fa[sid].to_numpy(float)
        st = int(np.flatnonzero(np.isfinite(p))[0])
        rows_ = np.arange(st, N_FC)
        fr = dvs.research_frame(dict(date=Tg.index.to_numpy()[rows_], true_adj=Tg["true_adj"].to_numpy(float)[rows_], true_raw=Tg["true_raw"].to_numpy(float)[rows_]), p[rows_])
        mine = np.full(N_FC, np.nan)
        mine[rows_] = fr["pred_clock"].to_numpy(float)
        same = bool(np.array_equal(mine, Fc[sid].to_numpy(float), equal_nan=True))
        rows.append(dict(check="pred_clock column vs scorer", table="forecasts.parquet", label=sid, quantity=f"pred_clock recomputed from the export's pred_adj (from row {st})", value=float(not same), reference=0.0, abs_diff=float(not same), tolerance="bitwise", passed=same, detail=""))
        on = Fc[sid].set_axis(Fc.index.normalize()).reindex(Pq["F"].index).to_numpy(float)
        d = float(np.max(np.abs(on / Pq["F"][sid].to_numpy(float) - 1.0)))
        rows.append(dict(check="pred_clock column vs scorer", table="forecasts.parquet", label=sid, quantity=f"pred_clock on the 866 trade days vs score() from row {TREE_START}, max relative difference", value=d, reference=0.0, abs_diff=d, tolerance="1e-12 relative", passed=d <= 1e-12, detail=""))
    # ---------------------------------------------------------------- the study CSVs, row by row
    keys = {"trees_kfull/arms.csv": "arm", "trees_morebars/arms.csv": "arm", "trees_morebars/average.csv": "forecast", "trees_datasize/arms.csv": "arm",
            "trees_lineartree/arms.csv": "arm", "trees_pretune/local_lgbm_arms.csv": "arm", "exog_penalty/headline.csv": "key", "kfull_tests/forecasts.csv": "forecast"}
    cache: dict[tuple, pd.DataFrame] = {}
    for table, key in keys.items():
        ref = pd.read_csv(STUDIES / table)
        ref = ref.drop_duplicates(key).set_index(key)
        sa = SA[SA["table"] == table]
        preds = {}
        for r in sa.itertuples():
            preds[r.label] = L.rebuild(table, r.label)
        start = int(sa["score_start"].iloc[0])
        assert (sa["score_start"] == start).all()
        F = pd.DataFrame(preds)
        sc, P = L.score(F, start=start, daily=True)
        cache[(table, "P")] = P
        for r in sa.itertuples():
            rr = ref.loc[r.label]
            for q in r.metrics.split(";"):
                if q == "mse":
                    chk(table, r.label, q, float(sc.loc[r.label, q]), float(rr[q]), relative=True)
                else:
                    chk(table, r.label, q, float(sc.loc[r.label, q]), float(rr[q]))
        n_ref = len(ref) if table != "trees_kfull/arms.csv" else int((ref["exists"] == "yes").sum())
        rows.append(dict(check="coverage", table=table, label="", quantity="rows checked / rows with a score", value=float(len(sa)), reference=float(n_ref), abs_diff=float(abs(len(sa) - n_ref)), tolerance="0", passed=len(sa) == n_ref, detail=""))
        say(f"{table}: {len(sa)} rows")
    # DM of the average.csv forecasts against the ridge, from the export's daily losses
    P = cache[("trees_morebars/average.csv", "P")]
    ref = pd.read_csv(STUDIES / "trees_morebars" / "average.csv").set_index("forecast")
    rk = "ridge, backbone unpenalized (linear reference)"
    for lab in P["ql"].columns:
        if lab != rk:
            chk("trees_morebars/average.csv", lab, "dm_vs_ridge", float(dm_test(P["ql"][lab].to_numpy(), P["ql"][rk].to_numpy())["dm"]), float(ref.loc[lab, "dm_vs_ridge"]))
    # ---------------------------------------------------------------- kfull_tests ridge + trees combinations
    rt = pd.read_csv(STUDIES / "kfull_tests" / "realtime_combination.csv")
    sa = SA[SA["table"] == "kfull_tests/realtime_combination.csv"]
    T = "kfull_tests/forecasts.csv"
    for r in sa.itertuples():
        ridge, nm = json.loads(r.members)
        trees = L.rebuild(T, nm)
        how = {"weighted": "adjusted", "variance": "variance", "realtime": "realtime"}[r.recipe]
        comb = L.ridge_trees(trees, ridge=ridge, how=how)
        if how == "adjusted":
            res = L.score(comb.rename(r.label)).iloc[0]
            sample = "all 866 trade days"
        else:
            if how == "realtime":
                comb = comb.copy()
                comb.iloc[:250] = np.nan  # the CSV reports trade days 251 .. 866 for the real-time lambda
            res = L.score_clock(comb.rename(r.label)).iloc[0]
            sample = "trade days 251 .. 866" if how == "realtime" else "all 866 trade days"
        for loss in ("qlike", "mse"):
            m = rt[(rt["forecast"] == nm) & (rt["combination"] == r.metrics) & (rt["sample"] == sample) & (rt["loss"] == loss)]
            assert len(m) >= 1, (nm, r.metrics, sample, loss)
            chk("kfull_tests/realtime_combination.csv", r.label, f"{loss} ({sample})", float(res[loss]), float(m["mean_combination"].iloc[0]), relative=(loss == "mse"))
    say(f"kfull_tests combinations: {len(sa)} recipes")
    # ---------------------------------------------------------------- pred_adj against the npz / stored tables, bit for bit
    F_all = L.forecasts(include_exact=True)
    have_work = 0
    for r in cat.itertuples():
        f = REPO / r.source_file
        if not f.is_file():
            rows.append(dict(check="pred_adj bitwise vs source", table=r.source_file, label=r.series_id, quantity="pred_adj", value=np.nan, reference=np.nan, abs_diff=np.nan, tolerance="bitwise", passed=True, detail="source file absent here (gitignored _work/): skipped"))
            continue
        have_work += 1
        if r.study == "master_table":
            src = stored_pred(r.arm, pd.DatetimeIndex(F_all.index))
        else:
            src = np.load(f, allow_pickle=False)[r.source_key]
        mine = F_all[r.series_id].to_numpy(float)
        same = bool(np.array_equal(mine, src, equal_nan=True))
        d = float(np.nanmax(np.abs(mine - src))) if np.isfinite(src).any() else 0.0
        rows.append(dict(check="pred_adj bitwise vs source", table=r.source_file, label=r.series_id, quantity="pred_adj", value=d, reference=0.0, abs_diff=d, tolerance="bitwise", passed=same, detail=f"{int(np.isfinite(src).sum())} forecasts"))
    # internals against the npz (where present)
    if True:
        for r in cat[cat["has_coefficients"] == "yes"].itertuples():
            if not (REPO / r.source_file).is_file():
                continue
            z = np.load(REPO / r.source_file, allow_pickle=False)
            th = L.coefficients(r.series_id)
            if "theta" in z:
                same = bool(np.array_equal(th.to_numpy(np.float32), z["theta"]))
                rows.append(dict(check="coefficients bitwise vs npz", table=r.source_file, label=r.series_id, quantity="theta (float32)", value=float(not same), reference=0.0, abs_diff=float(not same), tolerance="bitwise", passed=same, detail=str(z["theta"].shape)))
            else:
                bb = th.notna().all(axis=0).to_numpy()
                a32, a64 = th.to_numpy(float)[:, bb], np.asarray(z["theta_bb"], float)
                nz = a64 != 0
                d = float(np.max(np.abs(a32[nz] / a64[nz] - 1.0)))
                rows.append(dict(check="coefficients vs npz (float64 -> float32)", table=r.source_file, label=r.series_id, quantity="theta_bb max relative difference", value=d, reference=0.0, abs_diff=d, tolerance="6e-8 relative", passed=d < 6e-8, detail=""))
        for r in cat[cat["has_importance"] == "yes"].itertuples():
            if not (REPO / r.source_file).is_file():
                continue
            z = np.load(REPO / r.source_file, allow_pickle=False)
            sp, gn, kp = (L.importance(r.series_id, k) for k in ("split", "gain", "kept"))
            ok = np.array_equal(sp.to_numpy(), z["imp_split"]) and np.array_equal(kp.to_numpy(), z["imp_kept"]) and np.array_equal(sp.index.to_numpy(), z["anchors"])
            g = z["imp_gain"]
            d = float(np.max(np.abs(gn.to_numpy(float) - g) / np.where(g != 0, np.abs(g), 1.0)))
            rows.append(dict(check="importance vs npz", table=r.source_file, label=r.series_id, quantity="split, kept, anchors bitwise; gain max relative difference (float32)", value=d, reference=0.0, abs_diff=d, tolerance="bitwise; 6e-8 relative", passed=bool(ok and d < 6e-8), detail=""))
    V = pd.DataFrame(rows)
    V.to_csv(OUT / "VERIFY.csv", index=False)
    n_fail = int((~V["passed"]).sum())
    say(f"VERIFY: {len(V)} checks, {n_fail} failed; npz / stored sources present for {have_work} of {len(cat)} series")
    if n_fail:
        print(V[~V["passed"]].head(40).to_string(index=False))
    write_readme(V)
    if n_fail:
        raise SystemExit("VERIFY FAILED")


# ============================================================================ README
COLUMNS = {
    "forecasts.parquet": [
        ("series_id", "string", "`<study>/<arm>`; `exog_penalty/<arm>:exact_pred` for the exact-solution check series; `master_table/<key>` for the stored master-table forecasts"),
        ("study", "string", "folder of results/close_studies_2026-10-03/ the series comes from, or `master_table`"),
        ("row", "int16", "forecast row index 0 .. 1468 (0 = 2018-06-25); trees forecast from row 130"),
        ("date", "timestamp", "16:00 bar-end stamp, naive ET, as in the designs"),
        ("pred_adj", "float64", "the model's own output on the adjusted scale: a forecast of sqrt(RV / B) (dimensionless); NaN where the series has no forecast"),
        ("pred_clock", "float64", "the research scorer's recalibrated variance forecast (pred_adj^2 + s) x B, s = mean squared adjusted-scale error over the previous 250 rows (at least 63), lagged one row, from the series' own first forecast row; units of true_raw; NaN where undefined"),
    ],
    "targets.parquet": [
        ("row", "int16", "forecast row index"),
        ("date", "timestamp", "16:00 bar-end stamp, naive ET"),
        ("true_adj", "float64", "realized target on the adjusted scale, sqrt(RV / B) (the design's y)"),
        ("true_raw", "float64", "realized variance RV of the 15:30-16:00 bar (the target before the diurnal scaling)"),
        ("baseline", "float64", "diurnal scale B = true_raw / true_adj^2, as the scorer computes it"),
        ("baseline_design", "float64", "the design cache's stored B (equal to baseline within 1e-12 relative)"),
        ("is_trade_day", "bool", "one of the 866 trade days 2020-01-03 .. 2024-04-30 of the straddle deck"),
        ("entry", "float64", "trade days: 15:30 mid of call + put (straddle price)"),
        ("iv_var", "float64", "trade days: implied variance of the last 30 minutes from the 15:30 book (iv_30^2); the scorer buys when pred_clock > iv_var, else sells"),
        ("R", "float64", "trade days: straddle return at mid, exit / entry - 1; mid P&L = position x R"),
        ("exit", "float64", "trade days: straddle payoff at the close"),
        ("ask_c, ask_p, bid_c, bid_p", "float32", "trade days: 15:30 call / put quotes (kept float32: the scorer adds them in float32); crossed P&L = position x (exit / ask - 1) when buying, x (exit / bid - 1) when selling"),
        ("signal", "float64", "trade days: the deck's own signal rv_hat - iv_var (not used by the scorer)"),
    ],
    "catalogue.csv": [
        ("series_id, study, arm", "", "identity; arm = the name in the study's own CSVs (pretune walk arms without `walk_`)"),
        ("kind", "", "forecast / exact batch solution (every 25th row) / stored master-table forecast; for files left out: gate run / check run / smoke run / pre-tune trial records"),
        ("exported, exclusion_reason", "", "`no` for the npz files not stored as series, with the reason"),
        ("source_file, source_key", "", "file (relative to the repository) and array the forecast comes from"),
        ("model, estimator", "", "LightGBM / XGBoost / ridge / lasso / elastic net / OLS; spec estimator name"),
        ("family, design, bucket, k", "", "rows the model trained on (one-bar, last k bars ending 16:00, 16:00 rows of a longer design, ...), the design cache file, the feature bucket, bars"),
        ("rows, rows_min, rows_max, sessions_min, sessions_max, window", "", "training rows and distinct sessions in the window at the refits"),
        ("bar_column", "", "`yes` when the design has the added column bar_end_minute"),
        ("seed", "", "LightGBM / XGBoost random_state"),
        ("refit_every, first_forecast_row, n_forecasts, n_pred_clock, n_refits, n_columns", "", "cadence (sessions), first row with a forecast, counts"),
        ("penalty, alpha_grid, r_grid, l1_ratio", "", "linear arms: penalty structure, the alpha grid searched, the backbone : exogenous ratio grid, the l1 share"),
        ("linear_tree, linear_lambda, leaf_min, params_json, schedule_json", "", "tree settings (LightGBM min_child_samples or XGBoost min_child_weight; full parameter dict; trees_pretune: configuration schedule)"),
        ("tuning, group", "", "tuning protocol; the study's arm group"),
        ("versions_*", "", "library versions recorded in the npz meta (stored master-table trees: from the studies' SUMMARY.md)"),
        ("cpu_min, cpu_source, wall_min", "", "CPU minutes of the whole arm and where the figure comes from"),
        ("reported_qlike, reported_sharpe_mid, reported_sharpe_crossed, reported_in", "", "the home study's own numbers for the series (master table: master_table.csv)"),
        ("scored_from_row, scoreable", "", "first forecast row the home study passed to the scorer; `no` for the exact-solution check series"),
        ("is_exact_pred, identical_to", "", "exact-solution flag; other series with bit for bit the same pred_adj"),
        ("has_coefficients, has_importance, npz_keys", "", "which internals are exported; arrays in the npz"),
        ("meta.*, npz.*", "", "every field of the npz `meta` JSON, flattened with `.` (lists as JSON text); scalar npz arrays"),
        ("meta_json", "", "the raw `meta` JSON"),
    ],
    "study_arms.csv": [
        ("study, table, label", "", "a row of a study CSV (`label` = its arm / key / forecast name)"),
        ("recipe", "", "single (an exported series), mean (np.mean of the members in order), weighted (sum of weight x member), variance / realtime (kfull_tests ridge + trees combinations on the variance level)"),
        ("series_id, members, weights", "", "the series, or the members (series ids, or labels of the same table / study) and weights as JSON"),
        ("score_start, metrics", "", "first forecast row the study scored from; the CSV columns verify compares"),
    ],
    "linear_coefficients.parquet": [
        ("series_id, row, date", "", "exog_penalty arm and forecast row"),
        ("<628 design names>, intercept", "float32", "theta at every forecast row, the C kernel's layout: pred = [X_row, 1] . theta, X = the prescaled design_bar1600_all_features row; exactly 0 for columns out of the active set or masked; ols_baseline: the 22 backbone columns only (others NaN, intercept not stored)"),
    ],
    "linear_rechoices.parquet": [
        ("series_id, block, row, date, n_rows", "", "re-choice block (every 250 sessions) and its first forecast row"),
        ("alpha, r, penalty_<group>", "float64", "chosen penalty, backbone : exogenous ratio, the multi-penalty ridge's 8 group penalties"),
        ("val_mse, val_mse_shape, val_mse_best, mpr_val_mse, alpha_grid, r_grid", "", "validation MSE over the (r, alpha) grid (flattened, r major), its minimum, the multi-penalty search's MSE, the grids"),
        ("n_reseed, n_locked, n_masked, locked_cols, masked_cols", "", "cold reseeds in the block; locked (unpenalized, intercept excluded from the count) and masked column sets as indices into linear_columns.csv"),
        ("n_train, source", "", "window rows; the data-size / bar-count linear arms (spec class) have one row for each 250-solve block with n_train and n_masked only"),
    ],
    "linear_columns.csv": [("col, name, block, group", "", "index into theta (0 .. 628), design name, backbone / exogenous / intercept, exogenous feature group")],
    "tree_importance.parquet": [
        ("series_id, refit, row", "", "trees_kfull arm, refit index, refit anchor (forecast row)"),
        ("col", "int16", "index into tree_columns.csv"),
        ("split", "int16", "number of splits on the column in that refit's booster"),
        ("gain", "float32", "total gain of the column in that refit's booster"),
        ("(presence)", "", "a (refit, col) row exists exactly when the window mask kept the column (imp_kept); dropped columns have no split and no gain"),
    ],
    "tree_columns.csv": [("col, name", "", "628 design columns, then bar_end_minute (index 628) for the arms with the bar column")],
    "refits.parquet": [
        ("series_id, kind, refit, row, date, n_forecast_rows", "", "tree refit (anchor row, rows it forecasts) or linear solve (every forecast row)"),
        ("fit_sec, n_train, n_sessions, kept_n, leaf_min", "", "fit seconds, training rows, distinct sessions, columns kept by the window mask, leaf minimum"),
        ("coef_max, share_linear, mean_feat, fit_lo, fit_hi", "float64", "trees_lineartree: largest absolute leaf coefficient, share of linear leaves, mean features in a linear leaf, in-window fitted range"),
        ("cfg_*", "float64", "trees_pretune: configuration in force at the refit (num_leaves, min_child_samples, feature_fraction, bagging_fraction, lambda_l2, learning_rate, rounds)"),
        ("alpha, r, homotopy_events", "", "linear solves: penalty in force (data-size / bar-count arms: recorded at every row; exog_penalty: the block's re-choice), backbone ratio, homotopy events of the session's two updates"),
    ],
    "pretune_trials.parquet": [
        ("study, tpe_seed, trial, fold", "", "trees_pretune pre-tune study (s0 / s1), TPE seed, trial, fold"),
        ("num_leaves .. learning_rate", "float64", "the trial's configuration"),
        ("val_mse, fold_val_mse, fold_val_qlike, fold_rounds, fold_rounds_max, fold_sec, fold_n_kept", "float64", "objective (mean validation MSE over folds) and the fold records"),
        ("fit_first, val_first, val_last", "", "fold dates"),
    ],
    "VERIFY.csv": [("check, table, label, quantity, value, reference, abs_diff, tolerance, passed, detail", "", "one row for each check of the verify stage")],
}


def write_readme(V: pd.DataFrame) -> None:  # noqa: C901
    import close_studies_load as L

    cat = L.catalogue()
    ex = cat[cat["exported"] == "yes"]
    no = cat[cat["exported"] == "no"]
    files = sorted((p for p in OUT.iterdir() if p.is_file() and p.suffix in (".parquet", ".csv")), key=lambda p: (p.suffix != ".parquet", p.name))
    sizes = {p.name: p.stat().st_size for p in files}
    total = sum(sizes.values())
    big = max(sizes, key=sizes.get)
    sc = V[V["check"] == "score vs study CSV"]
    cc = V[V["check"] == "catalogue reported score vs export"]
    pc = V[V["check"] == "pred_clock column vs scorer"]
    bit = V[V["check"] == "pred_adj bitwise vs source"]
    bit_run = bit[~bit["detail"].str.contains("absent", na=False)]
    lay = cat["theta_layout_max_rel_gap"].dropna()
    Ls = []
    a = Ls.append
    a("# Close studies 2026-10-03 / 10-04: forecast export")
    a("")
    a(f"Written by `experiments/close_studies_export.py verify` on {time.strftime('%Y-%m-%d %H:%M')}. A compact, self-describing copy of every forecast series of the close studies "
      "(`results/close_studies_2026-10-03/*/`), the 16:00 targets and trade-day columns, a catalogue, and the model internals the npz files held. "
      "The raw `.npz` outputs in `*/_work/` are gitignored and never committed; this folder replaces them for analysis. Nothing here needs `_work/` or the design cache.")
    a("")
    a("## How to load")
    a("")
    a("```python")
    a("import sys; sys.path.insert(0, 'experiments')")
    a("import close_studies_load as L")
    a("cat = L.catalogue()                                    # one row for each series")
    a("F = L.forecasts(study='trees_kfull')                   # 1469 forecast rows x series (adjusted scale)")
    a("L.score(F)                                             # QLIKE, Sharpe mid / crossed on the 866 trade days")
    a("tab, P = L.score(['exog_penalty/ridge_bb0', 'trees_morebars/lgbm_bars4_r8000'], daily=True)  # P: daily QLIKE / P&L")
    a("L.score(L.rebuild('trees_morebars/average.csv', 'average of lgbm pools N = 1 .. 4'))")
    a("L.ridge_trees(L.rebuild('kfull_tests/forecasts.csv', 'bar:avg'), how='adjusted')  # equal weights, then score()")
    a("L.coefficients('ridge_bb0'); L.rechoices('ridge_bb0'); L.importance('lgbm_bars13_r26000_barmin_seed42', 'gain')")
    a("L.targets(); L.refits('trees_pretune/retune_any_r10'); L.pretune_trials()")
    a("```")
    a("")
    a("Without the loader: `pd.read_parquet('forecasts.parquet').pivot(index='date', columns='series_id', values='pred_adj')`; the targets and trade-day columns are in `targets.parquet`, "
      "so QLIKE and P&L statistics need no other file. The scorer: pred_clock = (pred_adj^2 + s) x B (s from the forecast's own trailing 250-row squared errors, lagged one row), "
      "QLIKE = RV / pred_clock - log(RV / pred_clock) - 1, position = +1 (buy the straddle) when pred_clock > iv_var else -1, mid P&L = position x R, Sharpe = mean / sd x sqrt(252). "
      f"`score()` starts the recalibration at row {TREE_START} by default (the first tree forecast); on the trade days (from row 382) any start at or before row 132 gives the same numbers. "
      "`pred_clock` in forecasts.parquet starts at each series' own first forecast row (row 0 for the linear arms), so it is defined earlier for them; on the trade days it equals what `score()` uses.")
    a("")
    a("## Files")
    a("")
    a("| file | size (MB) | content |")
    a("|---|---|---|")
    desc = {
        "forecasts.parquet": f"long: one row for each (series, forecast row); {len(ex)} series x 1469 rows",
        "targets.parquet": "one row for each 16:00 forecast row: target, B, trade-day straddle columns",
        "catalogue.csv": f"one row for each series ({len(ex)} exported) and each npz file left out ({len(no)})",
        "study_arms.csv": "each row of the study CSVs that verify checks -> the series or the recipe (seed / pool averages, ridge + trees) that rebuilds it",
        "linear_coefficients.parquet": "exog_penalty arms: coefficients at every forecast row (float32, wide)",
        "linear_rechoices.parquet": "every penalty re-choice of the linear arms: alpha, r, group penalties, validation grid, locked / masked sets",
        "linear_columns.csv": "names of the coefficient columns (628 design columns + intercept), block and group",
        "tree_importance.parquet": "trees_kfull arms: split count and gain of every kept column at every refit (long)",
        "tree_columns.csv": "names of the tree importance columns",
        "refits.parquet": "fit-level records: tree refits and linear solves",
        "pretune_trials.parquet": "trees_pretune pre-tune trials, every fold",
        "VERIFY.csv": "the verify stage's checks",
    }
    for n in sizes:
        a(f"| `{n}` | {sizes[n] / 1e6:.2f} | {desc.get(n, '')} |")
    a("")
    a(f"Total TOTAL_MB_ MB (every file here, this README included); largest file `{big}` {sizes[big] / 1e6:.1f} MB (every file is below GitHub's 50 MB warning size, so no Git LFS). "
      "Storage: parquet, zstd level 12, floating columns byte-stream split (lossless); forecasts float64; internals float32. No table needed splitting or float16: "
      "the dense ridge coefficients compress to about 1.8 MB an arm with byte-stream split, and the tree importance is stored long, one row for each (refit, column) the window mask kept "
      "(a dropped column has neither splits nor gain, so nothing is lost).")
    a("")
    a("## Series")
    a("")
    g = ex.groupby(["study", "kind"]).size().reset_index(name="series")
    a("| study | kind | series |")
    a("|---|---|---|")
    for r in g.itertuples():
        a(f"| {r.study} | {r.kind} | {r.series} |")
    a("")
    a(f"{len(ex)} series exported: every forecast npz of the seven studies (trees, linear arms, seed replicates, pools, linear-leaf arms, pretune walk arms, the data-size learning-curve arms, "
      "the exog_penalty arms and the HAR + calendar OLS), the exog_penalty `exact_pred` check series where present (lasso / elastic net arms; ridge arms have none), and the "
      f"{(ex['study'] == 'master_table').sum()} stored master-table forecasts the studies compared against (`results/spxw_pnl/yhat_<key>.parquet`, 16:00 rows). "
      "A reused forecast is one series, stored once under the study that wrote the file (e.g. trees_morebars' 1- and 2-bar rungs are `trees_datasize/lgbm_bar1600_w2000` and "
      "`trees_datasize/lgbm_pool_w4000`); `study_arms.csv` maps every study's own arm names to the series, and `identical_to` in the catalogue lists series that are bit for bit equal "
      "(e.g. `trees_kfull/lgbm_bars13_r26000_barmin_seed42` refitted `trees_morebars/lgbm_bars13_r26000_barmin`).")
    a("")
    a("Left out (listed in the catalogue with `exported = no`):")
    a("")
    for r in no.itertuples():
        a(f"- `{r.source_file}`: {r.exclusion_reason}")
    a("")
    a("Derived forecasts are not stored: the averages and combinations of `trees_morebars/average.csv` and `kfull_tests/` (seed averages for each k, the average of all pools, "
      "ridge + trees with equal weights on the adjusted scale or the variance level, the real-time combination) are rebuilt by `L.rebuild(table, label)` and `L.ridge_trees(...)` "
      "from `study_arms.csv`, exactly as the studies built them (np.mean of the members in the same order).")
    a("")
    a("## Verify")
    a("")
    a(f"`python experiments/close_studies_export.py verify` reloads the export only, re-scores every exported series and every derived forecast, and compares with the study CSVs "
      f"(tolerance {TOL:g} absolute on QLIKE and Sharpe, {TOL:g} relative on MSE): {len(sc)} comparisons, {int((~sc['passed']).sum())} failed, largest absolute difference "
      f"{sc['abs_diff'].max():.1e}.")
    a("")
    a("| table | rows | comparisons | failed | largest abs. difference |")
    a("|---|---|---|---|---|")
    for tb, h in sc.groupby("table", sort=False):
        a(f"| `{tb}` | {h['label'].nunique()} | {len(h)} | {int((~h['passed']).sum())} | {h['abs_diff'].max():.1e} |")
    a("")
    a(f"The catalogue's reported QLIKE / Sharpe mid of each series (its home study's CSV, or master_table.csv) against the export's score from the home study's first row: "
      f"{len(cc)} comparisons, {int((~cc['passed']).sum())} failed, largest absolute difference {cc['abs_diff'].max():.1e}.")
    a(f"The stored pred_clock column: recomputed from the export's pred_adj bit for bit, and within {pc[pc['quantity'].str.contains('trade days')]['abs_diff'].max():.1e} relative of score()'s "
      f"recalibration from row {TREE_START} on the trade days ({len(pc)} checks, {int((~pc['passed']).sum())} failed).")
    a(f"pred_adj against the source npz / stored table, bit for bit: {len(bit_run)} series compared, {int((~bit_run['passed']).sum())} differ"
      + (f"; {len(bit) - len(bit_run)} skipped (source absent)." if len(bit) > len(bit_run) else "."))
    oth = V[V["check"].str.startswith("coefficients") | V["check"].str.startswith("importance")]
    if len(oth):
        a(f"Internals against the npz: {len(oth)} checks, {int((~oth['passed']).sum())} failed (theta bit for bit; split / kept / anchors bit for bit; gain within float32 rounding).")
    if len(lay):
        a(f"Coefficient layout: for every exog_penalty arm, [X_row, 1] . theta (float32 theta, prescaled design_bar1600_all_features row, intercept last) reproduces pred within "
          f"{lay.max():.1e} relative (the C kernel computes pred = sum_k X[t, k] theta[k] over the augmented row, experiments/close_exogpen_kernel.c).")
    a(f"Overall: {'PASS' if V['passed'].all() else 'FAIL'} ({len(V)} checks).")
    a("")
    a("## Columns")
    for n, cols in COLUMNS.items():
        a("")
        a(f"### {n}")
        a("")
        a("| column | type | meaning |")
        a("|---|---|---|")
        for c, t, m in cols:
            a(f"| `{c}` | {t} | {m} |")
    a("")
    a("## Not in the export")
    a("")
    a("- The design matrices (`results/close_design/_work/`, about 1 GB, regenerable by `experiments/capture_design_close.py`): statistics that need the features (SHAP, partial dependence, "
      "refitting) need the design cache.")
    a("- The fitted boosters: no study saved them, so tree structure and leaf values are not available; trees_lineartree kept only summaries of the leaf models (refits.parquet).")
    a("- Coefficients of the trees_datasize / trees_morebars ridge and lasso arms: those runs recorded only the penalty at every row and the masked count at every tune "
      "(refits.parquet, linear_rechoices.parquet).")
    a("- `exact_pred` exists only every 25th session of the lasso / elastic net arms (59 rows), so it is not scoreable on the trade days.")
    a("- The Optuna journals of trees_pretune (`*.journal`; the trials are in pretune_trials.parquet) and the cluster CPU of the stored master-table runs.")
    text = "\n".join(Ls) + "\n"
    text = text.replace("TOTAL_MB_", f"{(total + len(text.encode()) ) / 1e6:.1f}")
    (OUT / "README.md").write_text(text, encoding="utf-8")
    say(f"wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "verify"):
        raise SystemExit(__doc__)
    build() if sys.argv[1] == "build" else verify()
