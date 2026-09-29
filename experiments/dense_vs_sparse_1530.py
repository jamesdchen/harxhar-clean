"""Trees vs linear at the 16:00 bar: is the signal weak-and-dense or sparse?  (checklist C2)

The professor's hypothesis: a weak but DENSE signal (many correlated inputs, each
carrying a little) favours ridge; a SPARSE one (a few strong inputs) favours the
lasso and the trees.  Three tests on the real design (no synthetic data), every one
at the bar ending 16:00 (the forecast the 15:30 last-30-min trade uses), on the
per-bar arms' design, target, 2000-session window and refit cadence.

Stages (``python experiments/dense_vs_sparse_1530.py <stage> [...]``):

capture <bucket>   Re-runs the per-bar ridge arm through the SAME spec class and the
                   SAME run_executor call the research campaign used
                   (specs/causal_tune_linear.py, its estimator section executed read-only,
                   as experiments/model_diagnostics_1530.py does) on a scratch data dir
                   holding only the vendor files (no chain load), and keeps the design
                   matrix the executor hands the model: every 16:00 row (2000 training
                   sessions + 1469 forecast sessions), the target, the column names.
                   GATE: the re-run equals the stored research forecast.
refit              The local refits (at most 4 worker processes):
                   * full models, all columns: per-bar ridge / lasso / elastic net through
                     the spec's own RollingTunedLinear (warm rank-1 updates every session,
                     penalty re-chosen every 250 sessions on the fit / 25 embargo / 125
                     tail split), coefficients recorded at every forecast; LightGBM with
                     the tree spec's shipped config (specs/causal_tune_trees.py, refit every
                     10 sessions).  GATE: each equals its stored research forecast.
                   * the sparsity sweep: the same models on causal top-k input subsets,
                     k in K_GRID.  The subset is re-chosen at every penalty re-choice
                     (every 250 sessions, the spec's own reseed points) from the 2000
                     sessions before the block's first forecast only:
                       rule "screen": |correlation of the column with the target| over
                                      that window (one subset for every model);
                       rule "own":    the full model's own |standardized weight| (beta x
                                      window sd) at the block's first forecast (ridge,
                                      lasso only).
                     Columns constant in the window or byte-copies of an earlier column
                     are never picked (the spec's identifiability mask).
                   * the collapse test: the same blocks; the columns of every group of
                     inputs whose pairwise |correlation| exceeds CORR_GROUP (complete
                     linkage) over the window are replaced by their first principal
                     component (loadings from the window, held for the block); ridge and
                     lasso refit on the collapsed design.
analyze            Scores every run with the research scorer (16:00-bar recalibration of
                   experiments/score_linear_subsection_causal.py + the deck's 15:30 sign(s)
                   straddle of score_linear_subsection.trade_1530) and writes the tables,
                   figures and numbers behind results/dense_vs_sparse/SUMMARY.md.

Environment: DVS_WORK (default results/dense_vs_sparse/_work; never committed) holds the
design and the per-run forecasts.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "dense_vs_sparse"
WORK = Path(os.environ.get("DVS_WORK", str(OUT / "_work")))
SEG = "bar1600"
TW = 2000
BUCKETS = ("live_feasible", "all_features")
STORED_LIN = REPO / "results" / "linear_subsection" / "arms_hoffman2"
STORED_TREES = REPO / "results" / "linear_subsection_trees"
LIN_EST = ("ridge", "reclasso", "reclasticnet")
LABEL = {
    "ridge": "ridge",
    "reclasso": "lasso",
    "reclasticnet": "elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
}
# the sparsity grid: the brief's 1, 2, 4, 8, 16, all, plus 32 and 64 to see the dense end
# of the curve (the designs hold 244 and 640 columns)
K_GRID = (1, 2, 4, 8, 16, 32, 64)
# a group of inputs = columns whose every pairwise |correlation| exceeds this (complete
# linkage): 0.8 is the conventional "highly collinear" line (|corr| 0.8 <=> a shared
# component explains 64 % of each pair's variance)
CORR_GROUP = 0.8
MAX_WORKERS = int(
    os.environ.get("DVS_WORKERS", "4")
)  # the overnight brief's local cap: 4
TREE_REFIT_EVERY = 10  # the tree spec's REFIT_EVERY (asserted against the spec below)
TUNE_PER = 250  # the linear spec's TUNE_PER (asserted against the spec below)


# ============================================================================ capture
def capture(bucket: str) -> None:
    import model_diagnostics_1530 as md

    WORK.mkdir(parents=True, exist_ok=True)
    data = WORK / f"data_{bucket}"
    data.mkdir(parents=True, exist_ok=True)
    files = md.VENDOR_FILES if bucket == md.BUCKET else md.BUCKET_VENDOR_FILES[bucket]
    for f in files:
        if not (data / f).exists():
            shutil.copy2(REPO / "data" / f, data / f)
    os.chdir(REPO)
    ns = md._spec_namespace("ridge", bucket)
    assert ns["TRAIN_WIN"] == TW and ns["SEGMENT"] == SEG and ns["TUNE_PER"] == TUNE_PER
    if bucket != md.BUCKET:
        md._check_vendor_cover(bucket, data, ns["get_bucket"](bucket))
    Base = ns["RollingTunedLinear"]
    Base.grid = ns["ESTIMATOR_GRIDS"]["ridge"]
    box: dict = {}

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        box.update(
            X=np.array(X_chunk, dtype=np.float64, copy=True),
            y=np.array(y_chunk, dtype=np.float64, copy=True),
            W=int(train_win_periods),
            names=[str(c) for c in hyperparams["_feature_names"]],
        )
        Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
        bt = ns["MultiStageBacktest"](
            residualizer=ns["IdentityResidualizer"](),
            regressor_factory=Base,
            refit_frequency=int(hyperparams.get("_refit_frequency", 1)),
        )
        return bt.run(X_chunk, y_chunk, train_win_periods, desc="dvs_capture")

    out_csv = WORK / f"arm_{bucket}" / "results.csv"
    t0 = time.time()
    ns["run_executor"](
        method_name="lin_tuned_ridge",
        fit_predict=fit_predict,
        hyperparams={"_refit_frequency": ns["REFIT_FREQUENCY"]},
        data_path=str(data),
        output_file=str(out_csv),
        horizon=ns["HORIZON"],
        train_window=TW,
        start=0,
        end=-1,
        halo=0,
        exog_cols=ns["get_bucket"](bucket),
        segment=SEG,
        lag_scope="global",
        har_lags=ns["HAR_LAGS"],
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=ns["SEED"],
    )
    res = pd.read_csv(out_csv.with_name(f"results_{SEG}.csv"))
    X, y, W = box["X"], box["y"], box["W"]
    assert W == TW and len(X) - W == len(res), (W, len(X), len(res))
    # the CSV round trip of the target (float text) vs the array handed to the model
    assert np.max(np.abs(y[W:] / res["true_adj"].to_numpy(float) - 1.0)) < 1e-12
    stored = pd.read_csv(
        STORED_LIN / bucket / "ridge" / f"tw{TW}" / f"results_{SEG}.csv"
    )
    assert (
        stored["date"].astype(str).to_numpy() == res["date"].astype(str).to_numpy()
    ).all()
    gate = float(
        np.max(
            np.abs(res["pred_adj"].to_numpy() - stored["pred_adj"].to_numpy())
            / np.abs(stored["pred_adj"].to_numpy())
        )
    )
    print(
        f"GATE capture {bucket}: re-run ridge vs stored research forecast, {len(res)} rows, "
        f"max rel diff {gate:.1e}; {X.shape[1]} columns; {time.time() - t0:.0f}s",
        flush=True,
    )
    assert gate < 1e-8, gate
    np.savez_compressed(
        WORK / f"design_{bucket}.npz",
        X=X,
        y=y,
        W=W,
        names=np.array(box["names"]),
        date=res["date"].astype(str).to_numpy().astype("U19"),
        true_adj=y[W:],
        true_raw=res["true_raw"].to_numpy(float),
        pred_ridge_spec=res["pred_adj"].to_numpy(float),
        gate_rel=gate,
    )


def load_design(bucket: str) -> dict:
    z = np.load(WORK / f"design_{bucket}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


# ============================================================================ helpers
def identifiable(Xw: np.ndarray) -> np.ndarray:
    """The spec's identifiability rule on a window: False for columns constant in the
    window or byte-copies of an earlier kept column (threshold-free, scale-free)."""
    keep = np.ones(Xw.shape[1], dtype=bool)
    seen: set[bytes] = set()
    for j in range(Xw.shape[1]):
        col = np.ascontiguousarray(Xw[:, j])
        if col.max() == col.min():
            keep[j] = False
            continue
        key = col.tobytes()
        if key in seen:
            keep[j] = False
        else:
            seen.add(key)
    return keep


def abs_corr_with(Xw: np.ndarray, yw: np.ndarray) -> np.ndarray:
    Xc = Xw - Xw.mean(axis=0)
    yc = yw - yw.mean()
    den = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc.T @ yc) / den
    return np.abs(np.nan_to_num(r, nan=0.0))


def top_k(
    score: np.ndarray, keep: np.ndarray, k: int, tie: np.ndarray | None = None
) -> np.ndarray:
    """The k eligible columns with the largest score (ties by `tie`, then column order)."""
    idx = np.flatnonzero(keep)
    t = np.zeros(len(idx)) if tie is None else tie[idx]
    order = np.lexsort((np.arange(len(idx)), -t, -score[idx]))
    return np.sort(idx[order[: min(k, len(idx))]])


def corr_groups(
    Xw: np.ndarray, keep: np.ndarray, thr: float = CORR_GROUP
) -> list[np.ndarray]:
    """Complete-linkage groups of the kept columns: every pair inside a group has
    |corr| > thr over the window.  Returns a list of column-index arrays (singletons incl.)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    idx = np.flatnonzero(keep)
    if len(idx) == 1:
        return [idx]
    C = np.corrcoef(Xw[:, idx], rowvar=False)
    D = 1.0 - np.abs(np.nan_to_num(C, nan=0.0))
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0.0, None)
    Z = linkage(squareform(D, checks=False), method="complete")
    # complete linkage: a merge height < 1 - thr <=> every pair |corr| > thr
    lab = fcluster(Z, t=np.nextafter(1.0 - thr, 0.0), criterion="distance")
    return [np.sort(idx[lab == g]) for g in np.unique(lab)]


# ============================================================================ runners
_NS: dict = {}


def spec_ns(kind: str) -> dict:
    """The linear spec's estimator section, or the tree spec's model section (read-only)."""
    if kind in _NS:
        return _NS[kind]
    os.chdir(REPO)
    if kind == "linear":
        import model_diagnostics_1530 as md

        ns = md._spec_namespace("ridge", "live_feasible")
        assert ns["TUNE_PER"] == TUNE_PER and ns["TRAIN_WIN"] == TW
    else:
        os.environ.update(
            {
                "HPC_KW_MODEL": "lgbm",
                "HPC_KW_EXOG_BUCKET": "live_feasible",
                "HPC_KW_SEGMENT": SEG,
                "HPC_KW_TRAIN_WIN": str(TW),
                "HPC_KW_LAG_SCOPE": "global",
                "SLURM_CPUS_PER_TASK": "1",
            }
        )
        src = (REPO / "specs" / "causal_tune_trees.py").read_text(encoding="utf-8")
        a = src.index("import json\nimport time\nimport warnings")
        b = src.index("SIDE: dict = {}")
        ns = {"__name__": "causal_tune_trees_section", "os": os}
        exec(
            compile(src[a:b], str(REPO / "specs" / "causal_tune_trees.py"), "exec"), ns
        )
        assert ns["REFIT_EVERY"] == TREE_REFIT_EVERY and ns["TRAIN_WIN"] == TW
        assert ns["N_THREADS"] == 1
    _NS[kind] = ns
    return ns


def run_linear_blocks(
    est: str, blocks: list[np.ndarray], W: int, y: np.ndarray, want_coef: bool = False
) -> dict:
    """The spec's RollingTunedLinear, one fresh instance per 250-session block.

    blocks[b] holds the block's design rows [t0 - W, W + i1): the W training rows
    before the block's first forecast and the block's forecast rows.  A fresh instance
    at each block start is exactly the spec's continuous run: the spec re-chooses the
    penalty and cold-reseeds at every 250th solve (the block starts), which is what
    init_window does, and the loop below is MultiStageBacktest._run_incremental's."""
    ns = spec_ns("linear")
    Base = ns["RollingTunedLinear"]
    Base.grid = ns["ESTIMATOR_GRIDS"][est]
    Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
    preds, th, mu, sd, alpha = [], [], [], [], []
    i0 = 0
    for Xb in blocks:
        nb = len(Xb) - W
        yb = y[i0 : i0 + W + nb]
        reg = Base()
        for j in range(nb):
            t = W + j
            if j == 0:
                reg.init_window(Xb[0:W], yb[0:W])
            else:
                reg.roll(Xb[t - 1], yb[t - 1], Xb[t - 1 - W], yb[t - 1 - W])
            reg.solve()
            preds.append(reg.predict_one(Xb[t]))
            alpha.append(float(reg.alpha_))
            if want_coef:
                Xw = reg._X[:, :-1]
                th.append(reg._th.copy())
                mu.append(Xw.mean(axis=0))
                sd.append(Xw.std(axis=0, ddof=1))
        i0 += nb
    out = {
        "pred": np.array(preds),
        "alpha": np.array(alpha),
        "n_reseed": len(Base.reseed_trace),
    }
    if want_coef:
        out.update(th=np.array(th), mu=np.array(mu), sd=np.array(sd))
    return out


def run_lgbm_blocks(blocks: list[np.ndarray], W: int, y: np.ndarray) -> dict:
    """The tree spec's LightGBM (make_model, shipped config), refit every 10 sessions on
    the 2000 rows strictly before the refit anchor; 10 divides 250, so a block's refits
    all use the block's columns."""
    ns = spec_ns("tree")
    assert TUNE_PER % TREE_REFIT_EVERY == 0
    preds: list[float] = []
    i0 = 0
    for Xb in blocks:
        nb = len(Xb) - W
        yb = y[i0 : i0 + W + nb]
        for j in range(0, nb, TREE_REFIT_EVERY):
            t = W + j
            k = min(TREE_REFIT_EVERY, nb - j)
            model = ns["make_model"]()
            model.fit(Xb[t - W : t], yb[t - W : t])
            preds.extend(np.asarray(model.predict(Xb[t : t + k]), dtype=np.float64))
        i0 += nb
    return {"pred": np.array(preds), "params": json.dumps(ns["PARAMS"]["lgbm"])}


def block_starts(n_test: int) -> list[tuple[int, int]]:
    return [(i0, min(i0 + TUNE_PER, n_test)) for i0 in range(0, n_test, TUNE_PER)]


def subset_blocks(X: np.ndarray, W: int, cols: list[np.ndarray]) -> list[np.ndarray]:
    n_test = len(X) - W
    return [
        np.ascontiguousarray(X[i0 : W + i1][:, c])
        for (i0, i1), c in zip(block_starts(n_test), cols)
    ]


def collapse_blocks(
    X: np.ndarray, y: np.ndarray, W: int
) -> tuple[list[np.ndarray], list[dict]]:
    """Per block: groups on the training window, each multi-column group -> its first
    principal component (standardized on the window, loadings from the window)."""
    n_test = len(X) - W
    out, info = [], []
    for i0, i1 in block_starts(n_test):
        Xw = X[i0 : i0 + W]
        keep = identifiable(Xw)
        groups = corr_groups(Xw, keep)
        Xr = X[i0 : W + i1]
        cols = []
        pc1_share = []
        for g in groups:
            if len(g) == 1:
                cols.append(Xr[:, g[0]])
                continue
            m, s = Xw[:, g].mean(axis=0), Xw[:, g].std(axis=0, ddof=1)
            Zw = (Xw[:, g] - m) / s
            _, sv, vt = np.linalg.svd(Zw, full_matrices=False)
            v = vt[0] * np.sign(vt[0].sum() or 1.0)
            pc1_share.append(float(sv[0] ** 2 / (sv**2).sum()))
            cols.append(((Xr[:, g] - m) / s) @ v)
        out.append(np.ascontiguousarray(np.column_stack(cols)))
        info.append(
            dict(
                block_start=i0,
                n_kept=int(keep.sum()),
                n_groups=len(groups),
                n_multi=int(sum(len(g) > 1 for g in groups)),
                largest=int(max(len(g) for g in groups)),
                pc1_share_median=float(np.median(pc1_share)) if pc1_share else np.nan,
            )
        )
    return out, info


# ---------------------------------------------------------------------------- jobs
def _job(job: dict) -> dict:
    """One refit (runs in a worker process)."""
    t0 = time.time()
    bucket, model, rule, k = job["bucket"], job["model"], job["rule"], job["k"]
    dz = load_design(bucket)
    X, y, W = dz["X"], dz["y"], int(dz["W"])
    n_test = len(X) - W
    starts = block_starts(n_test)
    extra: dict = {}
    if rule == "collapse":
        blocks, info = collapse_blocks(X, y, W)
        extra["collapse_info"] = json.dumps(info)
    elif k == "all":
        blocks = [np.ascontiguousarray(X[i0 : W + i1]) for i0, i1 in starts]
    else:
        cols = []
        own = None
        if rule == "own":
            own = np.load(WORK / "runs" / bucket / f"{model}_all.npz")
        for b, (i0, i1) in enumerate(starts):
            Xw, yw = X[i0 : i0 + W], y[i0 : i0 + W]
            keep = identifiable(Xw)
            corr = abs_corr_with(Xw, yw)
            if rule == "screen":
                cols.append(top_k(corr, keep, int(k)))
            else:  # own: the full model's |beta x window sd| at the block's first forecast
                assert own is not None
                zw = np.abs(own["th"][i0, :-1] * own["sd"][i0])
                cols.append(top_k(zw, keep, int(k), tie=corr))
        blocks = subset_blocks(X, W, cols)
        extra["cols"] = np.array(
            [np.pad(c, (0, int(k) - len(c)), constant_values=-1) for c in cols]
        )
    if model == "lgbm":
        r = run_lgbm_blocks(blocks, W, y)
    else:
        r = run_linear_blocks(
            model, blocks, W, y, want_coef=(k == "all" and rule == "full")
        )
    assert len(r["pred"]) == n_test
    d = WORK / "runs" / bucket
    d.mkdir(parents=True, exist_ok=True)
    tag = job_tag(job)
    np.savez_compressed(d / f"{tag}.npz", **r, **extra, sec=time.time() - t0)
    return {**job, "tag": tag, "sec": time.time() - t0}


def job_tag(job: dict) -> str:
    if job["rule"] == "full":
        return f"{job['model']}_all"
    if job["rule"] == "collapse":
        return f"{job['model']}_collapse"
    return f"{job['model']}_{job['rule']}_k{job['k']}"


def jobs_for(stage: str) -> list[dict]:
    J: list[dict] = []
    for b in BUCKETS:
        if stage == "full":
            J += [
                dict(bucket=b, model=m, rule="full", k="all")
                for m in (*LIN_EST, "lgbm")
            ]
        elif stage == "sweep":
            J += [
                dict(bucket=b, model=m, rule="screen", k=k)
                for k in K_GRID
                for m in ("ridge", "reclasso", "lgbm")
            ]
            J += [
                dict(bucket=b, model=m, rule="own", k=k)
                for k in K_GRID
                for m in ("ridge", "reclasso")
            ]
            J += [
                dict(bucket=b, model=m, rule="collapse", k="groups")
                for m in ("ridge", "reclasso")
            ]
    return J


def refit(stage: str) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    jobs = [
        j
        for j in jobs_for(stage)
        if not (WORK / "runs" / j["bucket"] / f"{job_tag(j)}.npz").exists()
    ]
    # the slow ones first
    jobs.sort(
        key=lambda j: (
            j["k"] != "all",
            j["bucket"] != "all_features",
            j["model"] != "lgbm",
        )
    )
    print(f"refit {stage}: {len(jobs)} jobs to run, {MAX_WORKERS} workers", flush=True)
    log = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
                print(f"  done {r['bucket']} {r['tag']} {r['sec']:.0f}s", flush=True)
                log.append(r)
            except Exception as e:  # noqa: BLE001 -- report and keep the others running
                print(f"  FAILED {j}: {type(e).__name__}: {e}", flush=True)
    (WORK / f"refit_{stage}.json").write_text(
        json.dumps(log, indent=1, default=str), encoding="utf-8"
    )


# ============================================================================ analyze
BOOT_B = 2000
BOOT_SEED = 20260929
DENSITY_LEVELS = (0.5, 0.8, 0.95)
TREE_MODELS = ("lgbm", "xgb", "rf")
DENSITY_MODELS = (*LIN_EST, *TREE_MODELS)
SWEEP_MODELS = ("ridge", "reclasso", "lgbm")
# validated categorical slots (model_diagnostics_1530_trees.py): blue, violet, orange,
# aqua, yellow; the elastic net takes the sixth, a grey, and lines add a dash pattern
COLOR = {
    "ridge": "#2a78d6",
    "reclasso": "#4a3aa7",
    "reclasticnet": "#7a7a7a",
    "lgbm": "#eb6834",
    "xgb": "#1baf7a",
    "rf": "#eda100",
}
DASH = {
    "ridge": "-",
    "reclasso": (0, (1, 1)),
    "reclasticnet": (0, (2, 2)),
    "lgbm": (0, (5, 2)),
    "xgb": (0, (3, 1, 1, 1)),
    "rf": (0, (8, 3)),
}


def research_frame(dz: dict, pred: np.ndarray) -> pd.DataFrame:
    """A run's 16:00 forecasts through the research scorer's back-transform
    (score_linear_subsection_causal.causal_forecasts, as score_trees_subsection loads a
    results CSV): (yhat^2 + s) x B with s the trailing-250-session mean squared error."""
    import score_linear_subsection_causal as slc

    idx = pd.DatetimeIndex(pd.to_datetime(dz["date"])).as_unit("ns")
    r = pd.DataFrame(
        {"true_adj": dz["true_adj"], "pred_adj": pred, "true_raw": dz["true_raw"]},
        index=idx,
    ).sort_index()
    r = r[(r["true_adj"] > 0) & (r["true_raw"] > 0)].copy()
    r["baseline"] = r["true_raw"] / r["true_adj"] ** 2
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    return slc.causal_forecasts(r)


def deck_frame() -> pd.DataFrame:
    import score_linear_subsection as base

    dk = pd.read_parquet(base.DECK).sort_index()
    dk.index = pd.DatetimeIndex(pd.to_datetime(dk.index)).normalize().as_unit("ns")
    return dk


def deck_panel(frames: list[pd.DataFrame], dk: pd.DataFrame) -> dict:
    """(deck days x runs) arrays: the deck's 15:30 sign(s) trade exactly as trade_1530."""
    F = np.column_stack(
        [
            f["pred_clock"]
            .set_axis(f.index.normalize())
            .reindex(dk.index)
            .to_numpy(float)
            for f in frames
        ]
    )
    RV = np.column_stack(
        [
            f["true_raw"]
            .set_axis(f.index.normalize())
            .reindex(dk.index)
            .to_numpy(float)
            for f in frames
        ]
    )
    assert np.isfinite(F).all() and np.isfinite(RV).all()
    iv = dk["iv_var"].to_numpy(float)[:, None]
    q = np.where(F > iv, 1.0, -1.0)
    R = dk["R"].to_numpy(float)[:, None]
    ex = dk["exit"].to_numpy(float)[:, None]
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)[:, None]
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)[:, None]
    ratio = RV / F
    return dict(
        pnl=q * R,
        pnlx=q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0),
        ql=ratio - np.log(ratio) - 1.0,
        buy=(q > 0).mean(axis=0),
    )


def boot_draws(P: dict, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Per-draw Sharpe (mid, crossed) and mean QLIKE of every run, one set of circular
    block-bootstrap day indices for all runs (paired), block = the research scorer's 21."""
    import atm_straddle_lib as asl
    import score_linear_subsection as base

    n = P["pnl"].shape[0]
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng(seed), n, base.BOOT_BLOCK, B
    )
    Wt = np.stack([np.bincount(i, minlength=n) for i in idx]).astype(float)

    def wsharpe(X: np.ndarray) -> np.ndarray:
        m = Wt @ X / n
        v = (Wt @ (X**2) / n - m**2) * n / (n - 1)
        return m / np.sqrt(v) * base.ANN

    return dict(sh=wsharpe(P["pnl"]), shx=wsharpe(P["pnlx"]), ql=Wt @ P["ql"] / n)


def point(P: dict) -> dict:
    import score_linear_subsection as base

    return dict(
        sh=P["pnl"].mean(0) / P["pnl"].std(0, ddof=1) * base.ANN,
        shx=P["pnlx"].mean(0) / P["pnlx"].std(0, ddof=1) * base.ANN,
        ql=P["ql"].mean(0),
    )


def ci(d: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def load_run(bucket: str, tag: str) -> dict | None:
    f = WORK / "runs" / bucket / f"{tag}.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=False)
    return {k: z[k] for k in z.files}


def tree_shap(bucket: str, model: str) -> dict:
    d = (
        STORED_TREES
        / bucket
        / SEG
        / model
        / f"tw{TW}"
        / "causal_tune_trees"
        / model
        / bucket
    )
    z = np.load(d / f"trees_{SEG}.npz", allow_pickle=True)
    sh = np.asarray(z["shap"], dtype=np.float64)
    return dict(
        names=[str(v) for v in z["feature_names"]],
        phi=sh[:, :-1],
        base=sh[:, -1],
        pred=np.asarray(z["pred_adj"], float),
        date=np.asarray(z["date"]).astype(str),
    )


def contributions(bucket: str, model: str, dz: dict) -> dict:
    """Per forecast row and design column, the input's contribution to the forecast:
    linear beta_j (x_j - window mean_j) (= linear SHAP), trees TreeSHAP."""
    names = [str(v) for v in dz["names"]]
    if model in TREE_MODELS:
        t = tree_shap(bucket, model)
        assert t["names"] == names, (bucket, model)
        assert (t["date"] == dz["date"]).all()
        gap = float(
            np.max(np.abs(t["phi"].sum(1) + t["base"] - t["pred"]) / np.abs(t["pred"]))
        )
        return dict(phi=t["phi"], pred=t["pred"], gap=gap)
    r = load_run(bucket, f"{model}_all")
    assert r is not None, (bucket, model)
    W = int(dz["W"])
    X = dz["X"][W:]
    th, mu = r["th"], r["mu"]
    phi = th[:, :-1] * (X - mu)
    lvl = th[:, -1] + np.einsum("nk,nk->n", th[:, :-1], mu)
    gap = float(np.max(np.abs(phi.sum(1) + lvl - r["pred"]) / np.abs(r["pred"])))
    return dict(phi=phi, pred=r["pred"], gap=gap, th=th, sd=r["sd"])


def density(phi: np.ndarray) -> dict:
    """Signal density of one forecast's contribution matrix (rows x inputs).

    share_j = mean |phi_j| / sum (the usual SHAP importance); N_eff = 1 / sum share^2
    (= p when every input carries the same, 1 when one input carries all).
    k_x (ranked) = the fewest inputs, taken in share order, whose summed contributions
    reproduce x of the variance of the forecast's input-driven part F = sum_j phi_j:
    R^2_k = 1 - var(F - F_k) / var(F).  k_x (greedy) = the same with the inputs added one
    at a time to maximize R^2 (the fewest any ordering needs, up to greedy's myopia)."""
    F = phi.sum(1)
    VF = F.var()
    mabs = np.abs(phi).mean(0)
    share = mabs / mabs.sum()
    order = np.argsort(-share, kind="stable")
    cum = np.cumsum(phi[:, order], axis=1)
    r2 = 1.0 - (F[:, None] - cum).var(axis=0) / VF
    out: dict = dict(
        n_inputs=int(phi.shape[1]),
        n_active=int((mabs > 0).sum()),
        neff=float(1.0 / (share**2).sum()),
        top1_share=float(share[order[0]]),
        r2_curve=r2,
    )
    for x in DENSITY_LEVELS:
        out[f"k{int(100 * x)}"] = int(np.argmax(r2 >= x) + 1)
    # greedy: cov matrix of the contributions; residual r = F - sum of the picked
    C = np.cov(phi, rowvar=False, ddof=0)
    c = C.sum(axis=1)  # cov(F, phi_j)
    vr = VF
    picked = np.zeros(phi.shape[1], dtype=bool)
    need: dict[float, int | None] = {x: None for x in DENSITY_LEVELS}
    for step in range(1, phi.shape[1] + 1):
        gain = 2 * c - np.diag(C)  # var(r) - var(r - phi_j)
        gain[picked] = -np.inf
        j = int(np.argmax(gain))
        picked[j] = True
        vr = vr - gain[j]
        c = c - C[:, j]
        for x in DENSITY_LEVELS:
            if need[x] is None and 1.0 - vr / VF >= x:
                need[x] = step
        if all(v is not None for v in need.values()):
            break
    for x in DENSITY_LEVELS:
        out[f"greedy_k{int(100 * x)}"] = need[x]
    return out


def series_matrix(names: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """(columns x series) 0/1 membership: a series = one source's columns (all HAR lags,
    the is-present / is-nonzero flags); stems = the design's own column names."""
    import model_diagnostics_1530 as md

    grp = [md.parse_feature(x)[0] for x in names]
    gs = sorted(set(grp))
    Mb = np.zeros((len(names), len(gs)))
    for j, g in enumerate(grp):
        Mb[j, gs.index(g)] = 1.0
    stems = [
        md.var_stem(g, names)
        if g != "calendar"
        else "calendar (DOW_*, is_*, hour, days_to_opex)"
        for g in gs
    ]
    return Mb, gs, stems


def run_label(model: str) -> str:
    return LABEL[model]


def analyze() -> None:  # noqa: C901 - one linear report
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import score_linear_subsection as base

    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    NUM: dict = {"gates": {}, "buckets": {}}

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    dk = deck_frame()
    gate_rows = []
    sweep_rows, gap_rows, dens_rows, group_rows, conc_rows, coll_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    pick_rows: list[dict] = []
    curves: dict = {}
    sweep_fig: dict = {}
    for bucket in BUCKETS:
        dz = load_design(bucket)
        names = [str(v) for v in dz["names"]]
        W = int(dz["W"])
        NB: dict = {
            "n_columns": len(names),
            "n_forecasts": int(len(dz["X"]) - W),
            "capture_gate": float(dz["gate_rel"]),
        }
        say(
            f"[{bucket}] {len(names)} design columns; {len(dz['X']) - W} forecast sessions "
            f"{dz['date'][0][:10]} .. {dz['date'][-1][:10]} after a {W}-session window; capture gate "
            f"{float(dz['gate_rel']):.1e}"
        )

        # ---------------- gates: full refits vs the stored research forecasts
        for m in (*LIN_EST, "lgbm"):
            r = load_run(bucket, f"{m}_all")
            if r is None:
                say(f"  missing full run {m}")
                continue
            if m == "lgbm":
                sf = (
                    STORED_TREES
                    / bucket
                    / SEG
                    / m
                    / f"tw{TW}"
                    / "causal_tune_trees"
                    / m
                    / bucket
                    / f"results_{SEG}.csv"
                )
            else:
                sf = STORED_LIN / bucket / m / f"tw{TW}" / f"results_{SEG}.csv"
            s = pd.read_csv(sf)
            assert (s["date"].astype(str).to_numpy() == dz["date"]).all()
            rel = np.abs(r["pred"] / s["pred_adj"].to_numpy(float) - 1.0)
            bad = np.flatnonzero(rel > 1e-6)
            gate_rows.append(
                dict(
                    bucket=bucket,
                    check=f"full {LABEL[m]} refit vs stored research forecast",
                    n=len(rel),
                    max_rel=float(rel.max()),
                    rows_above_1e6=int(len(bad)),
                    first_bad=int(bad.min()) if len(bad) else -1,
                    last_bad=int(bad.max()) if len(bad) else -1,
                )
            )
            if len(bad) and m in LIN_EST:
                # is it this script's block restarts, or the spec itself? the spec's own continuous run
                cache = WORK / f"gate_continuous_{bucket}_{m}.json"
                if not cache.is_file():
                    ns = spec_ns("linear")
                    Base = ns["RollingTunedLinear"]
                    Base.grid = ns["ESTIMATOR_GRIDS"][m]
                    bt = ns["MultiStageBacktest"](
                        residualizer=ns["IdentityResidualizer"](),
                        regressor_factory=Base,
                        refit_frequency=1,
                    )
                    pc = np.asarray(bt.run(dz["X"], dz["y"], W, desc="gate"), float)
                    rc = np.abs(pc / s["pred_adj"].to_numpy(float) - 1.0)
                    cache.write_text(
                        json.dumps(
                            dict(
                                max_rel=float(rc.max()),
                                rows=int((rc > 1e-6).sum()),
                                vs_blocks=float(np.max(np.abs(r["pred"] / pc - 1.0))),
                            )
                        )
                    )
                cg = json.loads(cache.read_text())
                gate_rows.append(
                    dict(
                        bucket=bucket,
                        check=f"spec continuous run of {LABEL[m]} vs stored",
                        n=len(rel),
                        max_rel=cg["max_rel"],
                        rows_above_1e6=cg["rows"],
                        first_bad=-1,
                        last_bad=-1,
                    )
                )
                say(
                    f"  GATE {LABEL[m]}: the spec's own continuous run here vs stored: max rel {cg['max_rel']:.1e} on "
                    f"{cg['rows']} rows; this script's block run vs that continuous run {cg['vs_blocks']:.1e}"
                )
            say(
                f"  GATE {LABEL[m]} full refit vs stored: max rel diff {rel.max():.1e}, rows > 1e-6: {len(bad)}"
                + (f" (forecast rows {bad.min()}..{bad.max()})" if len(bad) else "")
            )

        # ---------------- A1 signal density
        Mser, gser, stems = series_matrix(names)
        NB["n_series"] = len(gser)
        deck_rows = np.asarray(
            pd.DatetimeIndex(pd.to_datetime(dz["date"])).normalize().isin(dk.index)
        )
        curves[bucket] = {}
        contrib: dict = {}
        for m in DENSITY_MODELS:
            try:
                c = contributions(bucket, m, dz)
            except (AssertionError, FileNotFoundError) as e:
                say(f"  density: {m} skipped ({type(e).__name__}: {e})")
                continue
            contrib[m] = c
            gate_rows.append(
                dict(
                    bucket=bucket,
                    check=f"{LABEL[m]} contributions sum to the forecast",
                    n=len(c["pred"]),
                    max_rel=c["gap"],
                )
            )
            assert c["gap"] < 1e-5, (bucket, m, c["gap"])
            for level, phi in (("column", c["phi"]), ("series", c["phi"] @ Mser)):
                for smp, rows in (
                    ("all forecasts", np.ones(len(phi), bool)),
                    ("deck days", deck_rows),
                ):
                    d_ = density(phi[rows])
                    if level == "column" and smp == "all forecasts":
                        curves[bucket][m] = d_["r2_curve"]
                    blk = []
                    if smp == "all forecasts":
                        for i0, i1 in block_starts(len(phi)):
                            if i1 - i0 >= 100:
                                blk.append(density(phi[i0:i1])["k80"])
                    dens_rows.append(
                        dict(
                            bucket=bucket,
                            model=LABEL[m],
                            level=level,
                            sample=smp,
                            n_rows=int(rows.sum()),
                            **{k: v for k, v in d_.items() if k != "r2_curve"},
                            block_k80_median=float(np.median(blk)) if blk else np.nan,
                            block_k80_min=int(min(blk)) if blk else -1,
                            block_k80_max=int(max(blk)) if blk else -1,
                        )
                    )
            if m in LIN_EST:
                nz = (np.abs(c["th"][:, :-1]) > 0).sum(1)
                NB[f"nonzero_weights_{m}"] = [
                    int(nz.min()),
                    float(np.median(nz)),
                    int(nz.max()),
                ]

        # ---------------- A3 groups of correlated inputs (descriptive partition, forecast rows)
        Xo = dz["X"][W:]
        keep = identifiable(Xo)
        groups = corr_groups(Xo, keep)
        gid = np.full(len(names), -1)
        for g_i, g in enumerate(groups):
            gid[g] = g_i
        Mg = np.zeros((len(names), len(groups)))
        for j in range(len(names)):
            if gid[j] >= 0:
                Mg[j, gid[j]] = 1.0
        sizes = np.array([len(g) for g in groups])
        NB["groups"] = dict(
            n_groups=len(groups),
            n_multi=int((sizes > 1).sum()),
            largest=int(sizes.max()),
            columns_in_multi=int(sizes[sizes > 1].sum()),
            n_kept=int(keep.sum()),
        )
        say(
            f"  groups (|corr| > {CORR_GROUP}, complete linkage, the {len(Xo)} forecast rows): {len(groups)} groups "
            f"of the {int(keep.sum())} identifiable columns, {int((sizes > 1).sum())} with 2+ columns "
            f"(largest {sizes.max()}; {int(sizes[sizes > 1].sum())} columns sit in multi-column groups)"
        )
        gshare: dict = {}
        for m, c in contrib.items():
            G = c["phi"] @ Mg
            F = c["phi"].sum(1)
            vshare = (
                np.array([np.cov(G[:, g], F, ddof=0)[0, 1] for g in range(G.shape[1])])
                / F.var()
            )
            ashare = np.abs(G).mean(0) / np.abs(G).mean(0).sum()
            gshare[m] = (vshare, ashare)
            conc_rows.append(
                dict(
                    bucket=bucket,
                    model=LABEL[m],
                    n_groups=len(groups),
                    neff_groups=float(1.0 / (ashare**2).sum()),
                    neff_columns=float(
                        1.0
                        / (
                            (np.abs(c["phi"]).mean(0) / np.abs(c["phi"]).mean(0).sum())
                            ** 2
                        ).sum()
                    ),
                    top_group_var_share=float(vshare.max()),
                    top_group=names[groups[int(np.argmax(vshare))][0]],
                )
            )
        for g_i, g in enumerate(groups):
            row = dict(
                bucket=bucket,
                group=g_i,
                n_columns=len(g),
                members=" ".join(names[j] for j in g),
            )
            for m, (vs, as_) in gshare.items():
                row[f"var_share_{m}"] = float(vs[g_i])
                row[f"abs_share_{m}"] = float(as_[g_i])
            group_rows.append(row)

        # ---------------- score every run with the research scorer
        tags = [f"{m}_all" for m in (*LIN_EST, "lgbm")]
        for rule in ("screen", "own"):
            for m in SWEEP_MODELS if rule == "screen" else ("ridge", "reclasso"):
                tags += [f"{m}_{rule}_k{k}" for k in K_GRID]
        tags += ["ridge_collapse", "reclasso_collapse"]
        loaded = {t: load_run(bucket, t) for t in tags}
        runs: dict[str, dict] = {t: v for t, v in loaded.items() if v is not None}
        have = [t for t in tags if t in runs]
        miss = [t for t in tags if t not in runs]
        if miss:
            say(f"  runs missing (left out): {', '.join(miss)}")
        frames = [research_frame(dz, runs[t]["pred"]) for t in have]
        # GATE: the per-day trade equals trade_1530 on every run
        P = deck_panel(frames, dk)
        pt = point(P)
        g_tr = max(
            abs(base.trade_1530(f["pred_clock"])["Sharpe_mid"] - pt["sh"][i])
            for i, f in enumerate(frames)
        )
        gate_rows.append(
            dict(
                bucket=bucket,
                check="per-day trade vs trade_1530 (Sharpe mid), every run",
                n=len(frames),
                max_rel=float(g_tr),
            )
        )
        assert g_tr < 1e-9, g_tr
        ql_all = {
            t: float(
                (
                    lambda f: (
                        f["true_raw"] / f["pred_clock"]
                        - np.log(f["true_raw"] / f["pred_clock"])
                        - 1
                    ).mean()
                )(f.dropna(subset=["pred_clock"]))
            )
            for t, f in zip(have, frames)
        }
        n_all = int(frames[0]["pred_clock"].notna().sum())
        D = boot_draws(P)
        col = {t: i for i, t in enumerate(have)}

        def stat(t: str) -> dict:
            i = col[t]
            return dict(
                QLIKE_deck=float(pt["ql"][i]),
                QLIKE_all=ql_all[t],
                Sharpe_mid=float(pt["sh"][i]),
                Sharpe_crossed=float(pt["shx"][i]),
                pct_buy=float(100 * P["buy"][i]),
            )

        def contrast(a: str, b: str) -> dict:
            i, j = col[a], col[b]
            return dict(
                dQLIKE=float(pt["ql"][i] - pt["ql"][j]),
                dQLIKE_lo=ci(D["ql"][:, i] - D["ql"][:, j])[0],
                dQLIKE_hi=ci(D["ql"][:, i] - D["ql"][:, j])[1],
                dSharpe=float(pt["sh"][i] - pt["sh"][j]),
                dSharpe_lo=ci(D["sh"][:, i] - D["sh"][:, j])[0],
                dSharpe_hi=ci(D["sh"][:, i] - D["sh"][:, j])[1],
                dSharpe_crossed=float(pt["shx"][i] - pt["shx"][j]),
                dSharpe_crossed_lo=ci(D["shx"][:, i] - D["shx"][:, j])[0],
                dSharpe_crossed_hi=ci(D["shx"][:, i] - D["shx"][:, j])[1],
            )

        def did(a1: str, b1: str, a2: str, b2: str) -> dict:
            """(a1 - b1) - (a2 - b2): how much a gap changes, paired."""
            i1, j1, i2, j2 = col[a1], col[b1], col[a2], col[b2]
            dq = (D["ql"][:, i1] - D["ql"][:, j1]) - (D["ql"][:, i2] - D["ql"][:, j2])
            ds = (D["sh"][:, i1] - D["sh"][:, j1]) - (D["sh"][:, i2] - D["sh"][:, j2])
            return dict(
                did_QLIKE=float(
                    (pt["ql"][i1] - pt["ql"][j1]) - (pt["ql"][i2] - pt["ql"][j2])
                ),
                did_QLIKE_lo=ci(dq)[0],
                did_QLIKE_hi=ci(dq)[1],
                did_Sharpe=float(
                    (pt["sh"][i1] - pt["sh"][j1]) - (pt["sh"][i2] - pt["sh"][j2])
                ),
                did_Sharpe_lo=ci(ds)[0],
                did_Sharpe_hi=ci(ds)[1],
            )

        NB["n_deck"] = int(P["pnl"].shape[0])
        NB["n_qlike_all"] = n_all
        for m in (*LIN_EST, "lgbm"):
            if f"{m}_all" in col:
                NB[f"full_{m}"] = stat(f"{m}_all")
        # A2 sweep table
        for rule in ("screen", "own"):
            for m in SWEEP_MODELS if rule == "screen" else ("ridge", "reclasso"):
                for k in (*K_GRID, "all"):
                    t = f"{m}_all" if k == "all" else f"{m}_{rule}_k{k}"
                    if t not in col:
                        continue
                    row = dict(
                        bucket=bucket, model=LABEL[m], rule=rule, k=str(k), **stat(t)
                    )
                    if k != "all" and f"{m}_all" in col:
                        row.update(contrast(t, f"{m}_all"))
                        cols_ = runs[t].get("cols")
                        if cols_ is not None:
                            row["distinct_columns_used"] = int(
                                len(np.unique(cols_[cols_ >= 0]))
                            )
                    sweep_rows.append(row)
        # the hypothesis: ridge's edge over the lasso and LightGBM, same inputs, as k varies
        for a, b_ in (("ridge", "reclasso"), ("ridge", "lgbm"), ("reclasso", "lgbm")):
            for k in (*K_GRID, "all"):
                ta = f"{a}_all" if k == "all" else f"{a}_screen_k{k}"
                tb = f"{b_}_all" if k == "all" else f"{b_}_screen_k{k}"
                if ta not in col or tb not in col:
                    continue
                row = dict(
                    bucket=bucket,
                    pair=f"{LABEL[a]} - {LABEL[b_]}",
                    k=str(k),
                    **contrast(ta, tb),
                )
                if k != "all" and f"{a}_all" in col and f"{b_}_all" in col:
                    row.update(did(f"{a}_all", f"{b_}_all", ta, tb))
                gap_rows.append(row)
        # which columns the screen picks, block by block (k = 8 run; the smaller k are its prefixes
        # only when the |corr| order is the same, so each k is listed from its own run)
        for k in (1, 2, 4, 8):
            t = f"ridge_screen_k{k}"
            if t in col and "cols" in runs[t]:
                for b_i, cc in enumerate(runs[t]["cols"]):
                    pick_rows.append(
                        dict(
                            bucket=bucket,
                            k=k,
                            block=b_i,
                            first_forecast=str(dz["date"][b_i * TUNE_PER])[:10],
                            columns=" ".join(names[j] for j in cc if j >= 0),
                        )
                    )
        sweep_fig[bucket] = {
            m: [
                (k, stat(f"{m}_all" if k == "all" else f"{m}_screen_k{k}"))
                for k in (*K_GRID, "all")
                if (f"{m}_all" if k == "all" else f"{m}_screen_k{k}") in col
            ]
            for m in SWEEP_MODELS
        }
        # A3 collapse
        for m in ("ridge", "reclasso"):
            if f"{m}_collapse" in col:
                coll_rows.append(
                    dict(
                        bucket=bucket,
                        row=f"{LABEL[m]} collapsed vs full",
                        **stat(f"{m}_collapse"),
                        **contrast(f"{m}_collapse", f"{m}_all"),
                    )
                )
        if {"ridge_collapse", "reclasso_collapse", "ridge_all", "reclasso_all"} <= set(
            col
        ):
            coll_rows.append(
                dict(
                    bucket=bucket,
                    row="ridge - lasso, full design",
                    **contrast("ridge_all", "reclasso_all"),
                )
            )
            coll_rows.append(
                dict(
                    bucket=bucket,
                    row="ridge - lasso, groups collapsed to PC1",
                    **contrast("ridge_collapse", "reclasso_collapse"),
                )
            )
            coll_rows.append(
                dict(
                    bucket=bucket,
                    row="(ridge - lasso) full minus collapsed",
                    **did(
                        "ridge_all",
                        "reclasso_all",
                        "ridge_collapse",
                        "reclasso_collapse",
                    ),
                )
            )
            info = json.loads(str(runs["ridge_collapse"]["collapse_info"]))
            NB["collapse_blocks"] = info
        NUM["buckets"][bucket] = NB

    # ---------------- tables
    GT = pd.DataFrame(gate_rows)
    GT.to_csv(OUT / "gates.csv", index=False)
    DN = pd.DataFrame(dens_rows)
    DN.to_csv(OUT / "density.csv", index=False)
    SW = pd.DataFrame(sweep_rows)
    SW.to_csv(OUT / "sweep.csv", index=False)
    GP = pd.DataFrame(gap_rows)
    GP.to_csv(OUT / "sweep_gaps.csv", index=False)
    GR = pd.DataFrame(group_rows)
    GR.to_csv(OUT / "groups.csv", index=False)
    CC = pd.DataFrame(conc_rows)
    CC.to_csv(OUT / "group_concentration.csv", index=False)
    CL = pd.DataFrame(coll_rows)
    CL.to_csv(OUT / "collapse.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    say("\nA1 signal density (column level, all forecasts):")
    say(
        DN[(DN.level == "column") & (DN["sample"] == "all forecasts")][
            [
                "bucket",
                "model",
                "n_inputs",
                "n_active",
                "neff",
                "top1_share",
                "k50",
                "k80",
                "k95",
                "greedy_k50",
                "greedy_k80",
                "greedy_k95",
                "block_k80_median",
                "block_k80_min",
                "block_k80_max",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    say("A1 signal density (series level, all forecasts):")
    say(
        DN[(DN.level == "series") & (DN["sample"] == "all forecasts")][
            [
                "bucket",
                "model",
                "n_inputs",
                "n_active",
                "neff",
                "top1_share",
                "k50",
                "k80",
                "k95",
                "greedy_k80",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    say(
        "\nA2 sweep (research scorer; QLIKE on the deck days; intervals vs the same model on all columns):"
    )
    swc = [
        "bucket",
        "model",
        "rule",
        "k",
        "QLIKE_deck",
        "Sharpe_mid",
        "Sharpe_crossed",
        "dQLIKE",
        "dQLIKE_lo",
        "dQLIKE_hi",
        "dSharpe",
        "dSharpe_lo",
        "dSharpe_hi",
        "distinct_columns_used",
    ]
    say(SW[[c for c in swc if c in SW.columns]].round(4).to_string(index=False))
    say(
        "\nA2 gaps between models on the same inputs (screen rule), and the change of the gap vs all columns:"
    )
    say(GP.round(4).to_string(index=False))
    say("\nA3 concentration by group:")
    say(CC.round(3).to_string(index=False))
    say("\nA3 collapse:")
    say(CL.round(4).to_string(index=False))

    # ---------------- figures
    ks = [*K_GRID, "all"]
    xpos = {k: i for i, k in enumerate(ks)}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for c_i, bucket in enumerate(BUCKETS):
        for r_i, (key, ylab) in enumerate(
            (
                ("QLIKE_deck", "16:00 QLIKE, 866 deck days"),
                ("Sharpe_mid", "sign(s) Sharpe, mid"),
            )
        ):
            ax = axes[r_i, c_i]
            for m in SWEEP_MODELS:
                pts = sweep_fig.get(bucket, {}).get(m, [])
                if not pts:
                    continue
                ax.plot(
                    [xpos[k] for k, _ in pts],
                    [s[key] for _, s in pts],
                    color=COLOR[m],
                    ls=DASH[m],
                    lw=2,
                    marker="o",
                    ms=5,
                    label=LABEL[m],
                )
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([str(k) for k in ks], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_ylabel(ylab, fontsize=8)
            if r_i == 0:
                n_c = NUM["buckets"][bucket]["n_columns"]
                ax.set_title(f"{bucket} design ({n_c} columns)", fontsize=9)
            if r_i == 1:
                ax.set_xlabel(
                    "k = inputs kept (top-k by |corr with target| over the window before each 250-session block)",
                    fontsize=7,
                )
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_qlike_sharpe_vs_k.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, bucket in zip(axes, BUCKETS):
        for m, r2 in curves.get(bucket, {}).items():
            kk = np.arange(1, len(r2) + 1)
            ax.plot(
                kk, np.clip(r2, 0, 1), color=COLOR[m], ls=DASH[m], lw=2, label=LABEL[m]
            )
        for x in DENSITY_LEVELS:
            ax.axhline(x, color="0.75", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(
            "inputs, in order of their share of mean |contribution|", fontsize=8
        )
        ax.set_title(
            f"{bucket} ({NUM['buckets'][bucket]['n_columns']} columns)", fontsize=9
        )
        ax.tick_params(labelsize=7)
        ax.grid(color="0.92", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("share of the forecast's variance reproduced", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "density_curves.png", dpi=160)
    plt.close(fig)

    NUM["density"] = DN.to_dict("records")
    NUM["sweep"] = SW.to_dict("records")
    NUM["gaps"] = GP.to_dict("records")
    PK = pd.DataFrame(pick_rows)
    PK.to_csv(OUT / "screen_picks.csv", index=False)
    # how many of the paired intervals exclude zero (the hypothesis tests, screen rule)
    cnt = []
    for (bucket, pair), g in GP.groupby(["bucket", "pair"]):
        g = g[g["k"] != "all"]
        for m in ("QLIKE", "Sharpe"):
            if f"did_{m}_lo" not in g or g[f"did_{m}_lo"].isna().all():
                continue
            gg = g.dropna(subset=[f"did_{m}_lo"])
            cnt.append(
                dict(
                    bucket=bucket,
                    pair=pair,
                    measure=m,
                    n_k=len(gg),
                    gap_ci_excl0=int(
                        ((gg[f"d{m}_lo"] > 0) | (gg[f"d{m}_hi"] < 0)).sum()
                    ),
                    did_ci_excl0=int(
                        ((gg[f"did_{m}_lo"] > 0) | (gg[f"did_{m}_hi"] < 0)).sum()
                    ),
                    did_ci_excl0_ks=" ".join(
                        gg.loc[
                            (gg[f"did_{m}_lo"] > 0) | (gg[f"did_{m}_hi"] < 0), "k"
                        ].astype(str)
                    ),
                )
            )
    CN = pd.DataFrame(cnt)
    CN.to_csv(OUT / "sweep_gap_counts.csv", index=False)
    say(
        "\nA2 paired intervals excluding zero (gap at k; change of the gap between k and all columns):"
    )
    say(CN.to_string(index=False))
    NUM["gap_counts"] = CN.to_dict("records")
    NUM["concentration"] = CC.to_dict("records")
    NUM["collapse"] = CL.to_dict("records")
    NUM["gates"] = GT.to_dict("records")
    NUM["constants"] = dict(
        K_GRID=list(K_GRID),
        CORR_GROUP=CORR_GROUP,
        TUNE_PER=TUNE_PER,
        TREE_REFIT_EVERY=TREE_REFIT_EVERY,
        BOOT_B=BOOT_B,
        BOOT_BLOCK=base.BOOT_BLOCK,
        BOOT_SEED=BOOT_SEED,
    )
    (OUT / "numbers.json").write_text(
        json.dumps(NUM, indent=1, default=float), encoding="utf-8"
    )
    (OUT / "analyze_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ summary
def _iv(lo: float, hi: float, nd: int = 4) -> str:
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _sig(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def _members(members: str) -> str:
    """A group's design columns, compressed: stems with a common lag set are listed once."""
    import re

    by: dict[str, list[str]] = {}
    for m in members.split():
        mm = re.fullmatch(r"(.+)_ma_(\d+)", m)
        if mm is None:
            return ", ".join(f"`{x}`" for x in members.split())
        by.setdefault(mm.group(1), []).append(mm.group(2))
    lagsets = {tuple(v) for v in by.values()}
    if len(lagsets) == 1:
        lg = next(iter(lagsets))
        stems = list(by)
        if stems == ["har"]:  # the target's own ladder: har_ma_<lag>
            return ", ".join(f"`har_ma_{x}`" for x in lg)
        return (
            ", ".join(f"`{x}`" for x in stems)
            + " at "
            + ", ".join(f"`_ma_{x}`" for x in lg)
        )
    return ", ".join(f"`{x}`" for x in members.split())


def write_summary() -> None:  # noqa: C901 - one linear report
    """SUMMARY.md from this folder's own outputs; the verdict-bearing claims are asserted
    against the numbers first, so the prose cannot drift from the tables."""
    DN = pd.read_csv(OUT / "density.csv")
    CC = pd.read_csv(OUT / "group_concentration.csv")
    GR = pd.read_csv(OUT / "groups.csv")
    SW = pd.read_csv(OUT / "sweep.csv")
    SW["k"] = SW["k"].astype(str)
    GP = pd.read_csv(OUT / "sweep_gaps.csv")
    GP["k"] = GP["k"].astype(str)
    CN = pd.read_csv(OUT / "sweep_gap_counts.csv")
    CL = pd.read_csv(OUT / "collapse.csv")
    GT = pd.read_csv(OUT / "gates.csv")
    PK = pd.read_csv(OUT / "screen_picks.csv")
    NUM = json.loads((OUT / "numbers.json").read_text(encoding="utf-8"))
    LF, AF = BUCKETS
    NB = NUM["buckets"]
    models = [LABEL[m] for m in DENSITY_MODELS]
    ks = [str(k) for k in K_GRID]

    def den(b: str, m: str, level: str = "column") -> pd.Series:
        r = DN[
            (DN.bucket == b)
            & (DN.model == m)
            & (DN.level == level)
            & (DN["sample"] == "all forecasts")
        ]
        assert len(r) == 1, (b, m, level)
        return r.iloc[0]

    def conc(b: str, m: str) -> pd.Series:
        return CC[(CC.bucket == b) & (CC.model == m)].iloc[0]

    def sw(b: str, m: str, k: str, rule: str = "screen") -> pd.Series:
        r = SW[(SW.bucket == b) & (SW.model == m) & (SW.k == k) & (SW.rule == rule)]
        assert len(r) == 1, (b, m, k, rule)
        return r.iloc[0]

    def gp(b: str, pair: str, k: str) -> pd.Series:
        r = GP[(GP.bucket == b) & (GP.pair == pair) & (GP.k == k)]
        assert len(r) == 1, (b, pair, k)
        return r.iloc[0]

    def cl(b: str, row: str) -> pd.Series:
        return CL[(CL.bucket == b) & (CL.row == row)].iloc[0]

    RL, RG = "ridge - lasso", "ridge - LightGBM"
    others = [m for m in models if m != "ridge"]
    # ---- the claims, checked
    for b in BUCKETS:
        ne = {m: den(b, m)["neff"] for m in models}
        assert ne["ridge"] > max(ne[m] for m in others), ("ridge densest", b, ne)
        assert den(b, "ridge")["k95"] > max(den(b, m)["k95"] for m in others), (
            "ridge k95",
            b,
        )
    k80_max = int(max(den(b, m)["k80"] for b in BUCKETS for m in models))
    k95_other_max = int(max(den(b, m)["k95"] for b in BUCKETS for m in others))
    ne_other = {
        b: (
            min(den(b, m)["neff"] for m in others),
            max(den(b, m)["neff"] for m in others),
        )
        for b in BUCKETS
    }
    g_lf = GR[GR.bucket == LF]
    size_g = g_lf[
        g_lf["members"].str.split().apply(lambda v: "adj_sumabsret_ma_1" in v)
    ].iloc[0]
    har_g = g_lf[g_lf["members"].str.split().apply(lambda v: "har_ma_1" in v)].iloc[0]
    tree_keys = ("lgbm", "xgb", "rf")
    assert size_g["var_share_ridge"] > max(
        size_g[f"var_share_{m}"] for m in ("reclasso", *tree_keys)
    )
    assert har_g["var_share_ridge"] < min(
        har_g[f"var_share_{m}"] for m in ("reclasso", *tree_keys)
    )
    # QLIKE worse than all columns: the largest K such that every model, both designs, every k <= K
    sig_q = {
        (b, m, k): _sig(r["dQLIKE_lo"], r["dQLIKE_hi"]) and r["dQLIKE"] > 0
        for b in BUCKETS
        for m in ("ridge", "lasso", "LightGBM")
        for k in ks
        for r in [sw(b, m, k)]
    }
    K_all = 0
    for k in K_GRID:
        if all(
            sig_q[(b, m, str(k))]
            for b in BUCKETS
            for m in ("ridge", "lasso", "LightGBM")
        ):
            K_all = k
        else:
            break
    K_lf = 0
    for k in K_GRID:
        if all(sig_q[(LF, m, str(k))] for m in ("ridge", "lasso", "LightGBM")):
            K_lf = k
        else:
            break
    assert K_all >= 4, K_all
    below = all(
        sw(b, m, k)["Sharpe_mid"] < sw(b, m, "all")["Sharpe_mid"]
        for b in BUCKETS
        for m in ("ridge", "lasso", "LightGBM")
        for k in ks
        if int(k) <= 16
    )
    assert below
    # the hypothesis
    for b in BUCKETS:
        for kq in ("1", "2"):
            r = gp(b, RG, kq)
            assert _sig(r["dQLIKE_lo"], r["dQLIKE_hi"]) and r["dQLIKE"] < 0, (
                "ridge beats LightGBM sparse",
                b,
                kq,
            )
        r = gp(b, RG, "all")
        assert not _sig(r["dQLIKE_lo"], r["dQLIKE_hi"])
        r = gp(b, RL, "all")
        assert not _sig(r["dQLIKE_lo"], r["dQLIKE_hi"])
    for kq in ("4", "8"):
        r = gp(AF, RL, kq)
        assert _sig(r["dQLIKE_lo"], r["dQLIKE_hi"]) and r["dQLIKE"] > 0, (
            "lasso beats ridge at k",
            kq,
        )
    af_did_ks = [
        k
        for k in ks
        if _sig(gp(AF, RL, k)["did_QLIKE_lo"], gp(AF, RL, k)["did_QLIKE_hi"])
    ]
    lf_did_ks = [
        k
        for k in ks
        if _sig(gp(LF, RL, k)["did_QLIKE_lo"], gp(LF, RL, k)["did_QLIKE_hi"])
    ]
    lf_rg_did = [
        k
        for k in ks
        if _sig(gp(LF, RG, k)["did_QLIKE_lo"], gp(LF, RG, k)["did_QLIKE_hi"])
    ]
    assert len(af_did_ks) >= 2
    ridge_pairs = CN[CN.pair.isin([RL, RG]) & (CN.measure == "Sharpe")]
    n_sh = int(ridge_pairs["n_k"].sum())
    n_sh_gap = int(ridge_pairs["gap_ci_excl0"].sum())
    n_sh_did = int(ridge_pairs["did_ci_excl0"].sum())
    all_sh = CN[CN.measure == "Sharpe"]
    assert n_sh_gap <= 0.1 * n_sh and n_sh_did <= 0.1 * n_sh, (n_sh_gap, n_sh_did, n_sh)
    sh_gap_rows = GP[GP.pair.isin([RL, RG]) & (GP.k != "all")]
    sh_gap_exc = sh_gap_rows[
        (sh_gap_rows.dSharpe_lo > 0) | (sh_gap_rows.dSharpe_hi < 0)
    ]
    for b in BUCKETS:
        for pair in (RL, RG):
            r = gp(b, pair, "all")
            assert not _sig(r["dSharpe_lo"], r["dSharpe_hi"])
        d = cl(b, "(ridge - lasso) full minus collapsed")
        assert not _sig(d["did_Sharpe_lo"], d["did_Sharpe_hi"])
    fullr = {b: sw(b, "ridge", "all") for b in BUCKETS}
    width = [r["dSharpe_hi"] - r["dSharpe_lo"] for _, r in GP[GP.k != "all"].iterrows()]

    L: list[str] = []
    a = L.append
    a(
        "# Trees vs linear at the close: is the 16:00 signal weak-and-dense or sparse? (checklist C2)"
    )
    a("")
    a(
        "Written by `experiments/dense_vs_sparse_1530.py` (stages `capture`, `refit`, `analyze`, which writes this file); "
        "every number below is read from the CSVs and `numbers.json` of this folder, and the claims the verdict rests on are "
        "asserted against those numbers before this file is written."
    )
    a("")
    a(
        f"**What is compared.** The per-bar forecast of the bar ending 16:00 (issued at 15:30; the forecast the last-30-min "
        f"trade uses) on the per-bar arms' own design, target, {TW}-session rolling window and refit cadence: per-bar ridge, "
        f"lasso and elastic net refit every session with the penalty re-chosen every {TUNE_PER} sessions "
        "(`specs/causal_tune_linear.py`, its class run read-only), LightGBM / XGBoost / random forest refit every "
        f"{TREE_REFIT_EVERY} sessions (`specs/causal_tune_trees.py`, shipped configuration). Two designs: `live_feasible` "
        f"({NB[LF]['n_columns']} columns, {NB[LF]['n_series']} source series) and `all_features` ({NB[AF]['n_columns']} "
        f"columns, {NB[AF]['n_series']} series); {NB[LF]['n_forecasts']:,} forecast sessions (2018-06-25 .. 2024-04-30); "
        f"the trade and QLIKE use the {NB[LF]['n_deck']} trade days (2020-01-03 .. 2024-04-30)."
    )
    a("")
    a(
        "**One scorer, the research scorer:** the 16:00 bar recalibrated alone, forecast = (ŷ² + s)·B with s the "
        "forecast's own trailing-250-session mean squared error, lagged one session "
        "(`score_linear_subsection_causal.causal_forecasts`); QLIKE against the per-bar spec's 16:00 target on the "
        f"{NB[LF]['n_deck']} trade days; the trade is the deck's 15:30 **sign(s)** rule (`trade_1530`): buy the "
        "**straddle** (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position) when "
        "the forecast exceeds the 15:30 implied variance, sell it otherwise, hold to the close. Intervals: 95 %, circular "
        f"block bootstrap over the trade days (block {NUM['constants']['BOOT_BLOCK']} sessions, "
        f"{NUM['constants']['BOOT_B']:,} draws, one set of draws for every model, so differences are paired). The per-bar "
        f"ridge `live_feasible` scores QLIKE {fullr[LF]['QLIKE_deck']:.4f} and Sharpe {fullr[LF]['Sharpe_mid']:.2f} mid / "
        f"{fullr[LF]['Sharpe_crossed']:.2f} crossed here, the numbers of the closing-strategy master table."
    )
    a("")
    a(
        "Series names are the design's column stems: `har_ma_*` = the target's own HAR ladder (realized variance, means of "
        "the last 1, 5, 25, 125, 625, 3125 bars), `adj_sumabsret_ma_*` = absolute returns, `adj_sumret4_ma_*` = 4th-power "
        "returns, `adj_sumpret2_ma_*` = upside squared returns, `adj_sumbipow_ma_*` = bipower variation, "
        "`adj_sumvolume_ma_*` = ES volume, `adj_vix_ma_*` / `adj_vvix_ma_*` / `adj_vix3m_ma_*` = VIX, VVIX, VIX3M, "
        "`*_ewstock` / `*_vwstock` = the same statistics on equal- / value-weighted constituent stocks."
    )
    a("")
    # ---- brief
    r_lf, r_af = den(LF, "ridge"), den(AF, "ridge")
    a("## Answer in brief")
    a("")
    a(
        f"1. **Ridge's forecast is the dense one.** Its effective number of inputs is {r_lf['neff']:.0f} of "
        f"{NB[LF]['n_columns']} columns (`live_feasible`) and {r_af['neff']:.0f} of {NB[AF]['n_columns']} (`all_features`), "
        f"against {ne_other[LF][0]:.0f}–{ne_other[LF][1]:.0f} and {ne_other[AF][0]:.0f}–{ne_other[AF][1]:.0f} for the "
        f"lasso, the elastic net and the three tree models; it needs {int(r_lf['k95'])} / {int(r_af['k95'])} columns to "
        f"reproduce 95 % of its forecast's variance, every other model at most {k95_other_max}. But 80 % of every model's "
        f"forecast variance comes from at most {k80_max} columns: the first-order signal is the same few inputs "
        "(`har_ma_1`, `har_ma_5` and the recent return-size columns) for all six models, and ridge's density sits in the "
        "last fifth."
    )
    tr = [size_g[f"var_share_{m}"] for m in tree_keys]
    trh = [har_g[f"var_share_{m}"] for m in tree_keys]
    a(
        f"2. **Where ridge's extra weight goes:** onto the group of recent return-size measures that move with `har_ma` "
        f"({_members(size_g['members'])}): {100 * size_g['var_share_ridge']:.0f} % of ridge's forecast variance vs "
        f"{100 * size_g['var_share_reclasso']:.0f} % of the lasso's and {100 * min(tr):.0f}–{100 * max(tr):.0f} % of the "
        f"trees' (`live_feasible`), while the {_members(har_g['members'])} group carries "
        f"{100 * har_g['var_share_ridge']:.0f} % of ridge's vs {100 * har_g['var_share_reclasso']:.0f} % of the lasso's "
        f"and {100 * min(trh):.0f}–{100 * max(trh):.0f} % of the trees'."
    )
    a(
        f"3. **The forecastable signal is spread over many inputs, for every model.** Restricted to the top-k inputs of a "
        f"causal screen, every model's QLIKE is significantly worse than with all columns at every k ≤ {K_all} on both "
        f"designs (k ≤ {K_lf} on `live_feasible`), and at every k ≤ 16 every model trades below its all-column Sharpe."
    )
    g1, g2 = gp(LF, RL, "all"), gp(LF, RG, "all")
    g3, g4 = gp(AF, RL, "all"), gp(AF, RG, "all")
    a(
        f"4. **The professor's hypothesis, tested directly (the same inputs for all three models, k = "
        f"{', '.join(ks)}):**"
    )
    a(
        f"   * *On the trade:* not testable at this sample size. With identical inputs the ridge-minus-lasso and "
        f"ridge-minus-LightGBM Sharpe gaps have intervals that include zero at {n_sh - n_sh_gap} of the {n_sh} "
        f"(k, design) pairs, and so does the change of the gap between k inputs and all columns ({n_sh - n_sh_did} of "
        f"{n_sh}). Even with all columns ridge's trade edge is inside the noise: {g1['dSharpe']:+.2f} "
        f"{_iv(g1['dSharpe_lo'], g1['dSharpe_hi'], 2)} over the lasso and {g2['dSharpe']:+.2f} "
        f"{_iv(g2['dSharpe_lo'], g2['dSharpe_hi'], 2)} over LightGBM (`live_feasible`); {g3['dSharpe']:+.2f} "
        f"{_iv(g3['dSharpe_lo'], g3['dSharpe_hi'], 2)} and {g4['dSharpe']:+.2f} {_iv(g4['dSharpe_lo'], g4['dSharpe_hi'], 2)} "
        "(`all_features`)."
    )
    q4, q8 = gp(AF, RL, "4"), gp(AF, RL, "8")
    t1, t2 = gp(LF, RG, "1"), gp(LF, RG, "2")
    a(
        f"   * *On QLIKE (the precise measure):* **half confirmed.** *Against the lasso*, as the hypothesis says: with 4–8 "
        f"strong, collinear inputs the lasso's selection wins (ridge minus lasso {q4['dQLIKE']:+.4f} "
        f"{_iv(q4['dQLIKE_lo'], q4['dQLIKE_hi'])} at k = 4, {q8['dQLIKE']:+.4f} {_iv(q8['dQLIKE_lo'], q8['dQLIKE_hi'])} at "
        f"k = 8, `all_features`), and adding the long tail of weak inputs closes the gap ({g3['dQLIKE']:+.4f} "
        f"{_iv(g3['dQLIKE_lo'], g3['dQLIKE_hi'])} with all {NB[AF]['n_columns']} columns); the narrowing is significant at "
        f"k = {', '.join(af_did_ks)} on `all_features`"
        + (
            f" and at k = {', '.join(lf_did_ks)} on `live_feasible`."
            if lf_did_ks
            else ", not on `live_feasible`."
        )
        + f" *Against the trees*, reversed: LightGBM is worst at the sparse end, not best; with 1 or 2 inputs ridge beats it "
        f"({t1['dQLIKE']:+.4f} {_iv(t1['dQLIKE_lo'], t1['dQLIKE_hi'])}, {t2['dQLIKE']:+.4f} "
        f"{_iv(t2['dQLIKE_lo'], t2['dQLIKE_hi'])}), with all columns they tie ({g2['dQLIKE']:+.4f} "
        f"{_iv(g2['dQLIKE_lo'], g2['dQLIKE_hi'])} `live_feasible`, {g4['dQLIKE']:+.4f} "
        f"{_iv(g4['dQLIKE_lo'], g4['dQLIKE_hi'])} `all_features`)"
        + (
            f"; the change is significant at k = {', '.join(lf_rg_did)} on `live_feasible`."
            if lf_rg_did
            else "."
        )
    )
    d_lf, d_af = (
        cl(LF, "(ridge - lasso) full minus collapsed"),
        cl(AF, "(ridge - lasso) full minus collapsed"),
    )
    a(
        f"5. **Collapsing each correlated group to its first principal component** costs both linear models forecast "
        f"accuracy and does not move the ridge-minus-lasso trade gap beyond noise (change {d_lf['did_Sharpe']:+.2f} "
        f"{_iv(d_lf['did_Sharpe_lo'], d_lf['did_Sharpe_hi'], 2)} `live_feasible`, {d_af['did_Sharpe']:+.2f} "
        f"{_iv(d_af['did_Sharpe_lo'], d_af['did_Sharpe_hi'], 2)} `all_features`); there is no significant ridge trade edge "
        "to explain in the first place."
    )
    a("")
    a(
        f'**Verdict.** The data support "the signal is dense" (every model forecasts and trades better with many inputs) '
        f'and "ridge uses it densely" (ridge spreads weight over correlated return-size measures that the lasso and the '
        f'trees mostly leave to `har_ma`). They support "dense favours ridge over the lasso" on forecast accuracy, on the '
        f'full design. They do **not** support "sparse favours trees": trees lose most when inputs are few. None of it '
        f"is visible in the trade's Sharpe, whose model-to-model differences (intervals {min(width):.1f} to "
        f"{max(width):.1f} wide) are larger than every effect measured here; the trade ranking ridge ≥ lasso ≥ trees on "
        f"the {NB[LF]['n_deck']} days is not a significant ranking."
    )
    a("")
    # ---- section 1
    a("## 1. Signal density: how many inputs does each forecast use?")
    a("")
    mx_lin = GT[
        GT.check.str.contains("contributions")
        & GT.check.str.startswith(("ridge", "lasso", "elastic"))
    ]["max_rel"].max()
    mx_tree = GT[
        GT.check.str.contains("contributions")
        & GT.check.str.startswith(("LightGBM", "XGBoost", "random"))
    ]["max_rel"].max()
    a(
        f"For each model, the contribution of every design column to every forecast: linear β_j (x_j − window mean_j) "
        f"(linear SHAP; sums to the forecast minus the window-mean forecast, max relative gap {mx_lin:.1e}); trees: the "
        f"TreeSHAP values the tree campaign stored (additivity ≤ {mx_tree:.1e} relative). Definitions (all "
        f"{NB[LF]['n_forecasts']:,} forecasts; the trade-day values are in `density.csv` and differ little):"
    )
    a("")
    a(
        "* **share_j** = mean |contribution_j| / sum over columns; **N_eff = 1 / Σ share²** (= p if every input carries the "
        "same, 1 if one input carries all);"
    )
    a(
        "* **k_x** = the fewest columns, taken in share order, whose summed contributions reproduce x of the variance of the "
        "input-driven forecast (R² = 1 − var(F − F_k) / var(F)); **greedy k_95** = the same with columns added one at a "
        "time to maximize R²."
    )
    a("")
    a(
        "| design | model | columns with any weight | N_eff (columns) | largest share | k_50 | k_80 | k_95 | greedy k_95 | "
        "N_eff (series) | N_eff (groups, \\|corr\\| > 0.8) |"
    )
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in BUCKETS:
        for i, m in enumerate(models):
            c, s_, g_ = den(b, m), den(b, m, "series"), conc(b, m)
            bold = "**" if m == "ridge" else ""
            a(
                f"| {b if i == 0 else ''} | {m} | {int(c['n_active'])} | {bold}{c['neff']:.1f}{bold} | "
                f"{100 * c['top1_share']:.1f} % | {int(c['k50'])} | {int(c['k80'])} | {bold}{int(c['k95'])}{bold} | "
                f"{int(c['greedy_k95'])} | {bold}{s_['neff']:.1f}{bold} | {bold}{g_['neff_groups']:.1f}{bold} |"
            )
    a("")
    kr = [den(b, "ridge") for b in BUCKETS]
    ko = [den(b, m) for b in BUCKETS for m in others]
    a(
        f"A series is one source (all its HAR lags and its is-present / is-nonzero flags); a group is a set of columns whose "
        f"every pairwise |correlation| over the forecast rows exceeds {CORR_GROUP} (complete linkage; "
        f"{NB[LF]['groups']['n_groups']} groups of the {NB[LF]['groups']['n_kept']} identifiable `live_feasible` columns, "
        f"{NB[AF]['groups']['n_groups']} of {NB[AF]['groups']['n_kept']} for `all_features`). The lasso keeps a median of "
        f"{NB[LF]['nonzero_weights_reclasso'][1]:.0f} (`live_feasible`, range {NB[LF]['nonzero_weights_reclasso'][0]}–"
        f"{NB[LF]['nonzero_weights_reclasso'][2]}) and {NB[AF]['nonzero_weights_reclasso'][1]:.0f} (`all_features`, "
        f"{NB[AF]['nonzero_weights_reclasso'][0]}–{NB[AF]['nonzero_weights_reclasso'][2]}) non-zero weights per refit; "
        f"ridge keeps every identifiable column. k_80 across the six {TUNE_PER}-session blocks: "
        f"{min(int(r['block_k80_min']) for r in kr)}–{max(int(r['block_k80_max']) for r in kr)} for ridge, "
        f"{min(int(r['block_k80_min']) for r in ko)}–{max(int(r['block_k80_max']) for r in ko)} for every other model. "
        "Figure: `density_curves.png`."
    )
    a("")
    a(
        "Share of each model's forecast variance carried by a group (the group's covariance with the forecast / the "
        "forecast's variance; the four largest groups by ridge's share, `groups.csv`):"
    )
    a("")
    a(
        "| design | group (design columns) | ridge | lasso | LightGBM | XGBoost | random forest |"
    )
    a("|---|---|---:|---:|---:|---:|---:|")
    for b in BUCKETS:
        gg = GR[GR.bucket == b].sort_values("var_share_ridge", ascending=False).head(4)
        for i, (_, r) in enumerate(gg.iterrows()):
            a(
                f"| {b if i == 0 else ''} | {_members(r['members'])} | "
                + " | ".join(
                    f"{100 * r[f'var_share_{m}']:.1f} %"
                    for m in ("ridge", "reclasso", "lgbm", "xgb", "rf")
                )
                + " |"
            )
    a("")
    # ---- section 2
    a("## 2. The sparsity sweep: ridge, lasso and LightGBM on the same top-k inputs")
    a("")
    starts = PK[(PK.bucket == LF) & (PK.k == 1)]["first_forecast"].tolist()
    pk = {
        (b, k): PK[(PK.bucket == b) & (PK.k == k)]["columns"].tolist()
        for b in BUCKETS
        for k in (1, 2, 4, 8)
    }
    k1 = sorted(set(pk[(LF, 1)]) | set(pk[(AF, 1)]))
    a(
        f"**Rule (causal):** at each of the six penalty re-choices (every {TUNE_PER} sessions; first forecasts "
        f"{', '.join(starts)}) the k columns with the largest |correlation with the target| over the {TW} sessions before "
        "the block's first forecast are kept for that block (constant or duplicated columns are never picked); the same "
        "columns go to all three models, which then refit exactly as their specs do. The picks (`screen_picks.csv`): "
        f"k = 1 is {', '.join(f'`{x}`' for x in k1)} in every block; k = 2 is "
        f"{' / '.join(sorted({'`' + x.replace(' ', '`, `') + '`' for x in pk[(LF, 2)]}))}; k = 4 is "
        f"{' or '.join(sorted({'`' + x.replace(' ', '`, `') + '`' for x in pk[(LF, 4)]}))}; k = 8 in the last block is "
        f"`{pk[(LF, 8)][-1].replace(' ', '`, `')}` (`live_feasible`) and `{pk[(AF, 8)][-1].replace(' ', '`, `')}` "
        "(`all_features`). A second rule (each linear model's own top-k standardized weights, β × window sd, at the "
        "block start) is in `sweep.csv` (rule `own`)."
    )
    a("")
    a(
        f"QLIKE ({NB[LF]['n_deck']} trade days) and sign(s) Sharpe (mid) against k; * = the paired interval against the "
        "same model on all columns excludes zero (`sweep.csv` has the intervals and the crossed Sharpe; figure "
        "`sweep_qlike_sharpe_vs_k.png`):"
    )
    a("")
    a(
        "| k | ridge QLIKE | lasso QLIKE | LightGBM QLIKE | ridge Sharpe | lasso Sharpe | LightGBM Sharpe |"
    )
    a("|---|---:|---:|---:|---:|---:|---:|")
    for b in BUCKETS:
        a(f"| **{b}** | | | | | | |")
        for kx in (*ks, "all"):
            cells = []
            for key in ("QLIKE", "Sharpe"):
                for m in ("ridge", "lasso", "LightGBM"):
                    r = sw(b, m, kx)
                    v = r["QLIKE_deck"] if key == "QLIKE" else r["Sharpe_mid"]
                    star = ""
                    if kx != "all" and _sig(r[f"d{key}_lo"], r[f"d{key}_hi"]):
                        star = " *"
                    cells.append(
                        f"{v:.4f}{star}" if key == "QLIKE" else f"{v:.2f}{star}"
                    )
            lab = kx if kx != "all" else f"all ({NB[b]['n_columns']})"
            a(f"| {lab} | " + " | ".join(cells) + " |")
    a("")
    a("(k ≤ 4 selects the same columns in both designs, hence the identical rows.)")
    a("")
    a(
        "**The hypothesis test** (`sweep_gaps.csv`, `sweep_gap_counts.csv`): for each pair of models on the same k inputs, "
        "the paired gap and the change of the gap between k and all columns (a difference in differences):"
    )
    a("")
    lf4, lf8 = gp(LF, RL, "4"), gp(LF, RL, "8")
    a(
        f"* ridge − lasso, QLIKE: lasso better at k = 4 and 8 (`live_feasible` {lf4['dQLIKE']:+.4f} "
        f"{_iv(lf4['dQLIKE_lo'], lf4['dQLIKE_hi'])} and {lf8['dQLIKE']:+.4f} {_iv(lf8['dQLIKE_lo'], lf8['dQLIKE_hi'])}; "
        f"`all_features` {q4['dQLIKE']:+.4f} {_iv(q4['dQLIKE_lo'], q4['dQLIKE_hi'])} and {q8['dQLIKE']:+.4f} "
        f"{_iv(q8['dQLIKE_lo'], q8['dQLIKE_hi'])}); tied with all columns ({g1['dQLIKE']:+.4f} "
        f"{_iv(g1['dQLIKE_lo'], g1['dQLIKE_hi'])} `live_feasible`, {g3['dQLIKE']:+.4f} {_iv(g3['dQLIKE_lo'], g3['dQLIKE_hi'])} "
        "`all_features`); on `all_features` the gap narrows significantly from k = "
        + ", ".join(
            f"{k} ({gp(AF, RL, k)['did_QLIKE']:+.4f} {_iv(gp(AF, RL, k)['did_QLIKE_lo'], gp(AF, RL, k)['did_QLIKE_hi'])})"
            for k in af_did_ks
        )
        + " to all columns."
    )
    a(
        f"* ridge − LightGBM, QLIKE: ridge better at k = 1, 2 ({t1['dQLIKE']:+.4f} {_iv(t1['dQLIKE_lo'], t1['dQLIKE_hi'])}, "
        f"{t2['dQLIKE']:+.4f} {_iv(t2['dQLIKE_lo'], t2['dQLIKE_hi'])}); tied with all columns"
        + (
            "; on `live_feasible` the gap closes significantly from k = "
            + ", ".join(
                f"{k} ({gp(LF, RG, k)['did_QLIKE']:+.4f} {_iv(gp(LF, RG, k)['did_QLIKE_lo'], gp(LF, RG, k)['did_QLIKE_hi'])})"
                for k in lf_rg_did
            )
            + "."
            if lf_rg_did
            else "."
        )
    )
    ll = [gp(b, "lasso - LightGBM", k) for b in BUCKETS for k in ("1", "2", "4")]
    ll_sig = [r for r in ll if _sig(r["dQLIKE_lo"], r["dQLIKE_hi"]) and r["dQLIKE"] < 0]
    ll2 = gp(LF, "lasso - LightGBM", "2")
    a(
        f"* lasso − LightGBM, QLIKE: lasso better at k = 1, 2, 4 in {len(ll_sig)} of 6 (k, design) cells (e.g. "
        f"{ll2['dQLIKE']:+.4f} {_iv(ll2['dQLIKE_lo'], ll2['dQLIKE_hi'])} at k = 2); tied with all columns."
    )
    exc = "; ".join(
        f"`{r.bucket}` {r.pair} at k = {r.k}, {r.dSharpe:+.2f} {_iv(r.dSharpe_lo, r.dSharpe_hi, 2)}"
        for r in sh_gap_exc.itertuples()
    )
    a(
        f"* Sharpe: of the {n_sh} ridge-vs-lasso and ridge-vs-LightGBM gaps at k = {ks[0]} … {ks[-1]}, {n_sh_gap} "
        + ("interval excludes zero" if n_sh_gap == 1 else "intervals exclude zero")
        + (f" ({exc})" if exc else "")
        + f"; the change of the gap between k and all columns excludes zero in {n_sh_did} of {n_sh}. Across all {int(all_sh['n_k'].sum())} Sharpe comparisons "
        f"(the lasso − LightGBM pair included) {int(all_sh['did_ci_excl0'].sum())} changes of gap exclude zero, about the "
        "rate expected by chance at 5 %."
    )
    o16, o8 = sw(LF, "ridge", "16", "own"), sw(LF, "ridge", "8", "own")
    a(
        f"* Ridge restricted to its own top 16 standardized weights is within noise of its all-column QLIKE on "
        f"`live_feasible` ({o16['dQLIKE']:+.4f} {_iv(o16['dQLIKE_lo'], o16['dQLIKE_hi'])}); its top 8 are "
        + ("not" if _sig(o8["dQLIKE_lo"], o8["dQLIKE_hi"]) else "also")
        + f" ({o8['dQLIKE']:+.4f} {_iv(o8['dQLIKE_lo'], o8['dQLIKE_hi'])})."
    )
    a("")
    # ---- section 3
    a(
        "## 3. Correlated-input groups: collapse each group to its first principal component"
    )
    a("")
    pcs = [
        blk["pc1_share_median"]
        for b in BUCKETS
        for blk in NB[b].get("collapse_blocks", [])
    ]
    a(
        f"**Rule:** at each {TUNE_PER}-session block, on the {TW} sessions before the block, the identifiable columns are "
        f"grouped by complete linkage at |correlation| > {CORR_GROUP}; each group of two or more columns is replaced by "
        "its first principal component (standardized on the window, loadings from the window, held for the block); ridge "
        "and lasso then refit every session on the collapsed design as their spec does. The first component carries a "
        f"median {100 * min(pcs):.0f}–{100 * max(pcs):.0f} % of a group's window variance (per block and design). The "
        "brief's \"PCA per refit\" is implemented per re-tune block (the spec's own reseed points), so the design only "
        "changes where the penalty is re-chosen."
    )
    a("")
    a("| design | | QLIKE | Sharpe mid | vs the same model on the full design |")
    a("|---|---|---:|---:|---|")
    for b in BUCKETS:
        for i, m in enumerate(("ridge", "lasso")):
            r = cl(b, f"{m} collapsed vs full")
            a(
                f"| {b if i == 0 else ''} | {m}, collapsed | {r['QLIKE_deck']:.4f} | {r['Sharpe_mid']:.2f} | QLIKE "
                f"{r['dQLIKE']:+.4f} {_iv(r['dQLIKE_lo'], r['dQLIKE_hi'])}, Sharpe {r['dSharpe']:+.2f} "
                f"{_iv(r['dSharpe_lo'], r['dSharpe_hi'], 2)} |"
            )
    a("")
    parts = []
    for b in BUCKETS:
        f_, c_, d_ = (
            cl(b, "ridge - lasso, full design"),
            cl(b, "ridge - lasso, groups collapsed to PC1"),
            cl(b, "(ridge - lasso) full minus collapsed"),
        )
        parts.append(
            f"`{b}` {f_['dSharpe']:+.2f} {_iv(f_['dSharpe_lo'], f_['dSharpe_hi'], 2)} on the full design, "
            f"{c_['dSharpe']:+.2f} {_iv(c_['dSharpe_lo'], c_['dSharpe_hi'], 2)} collapsed (change "
            f"{d_['did_Sharpe']:+.2f} {_iv(d_['did_Sharpe_lo'], d_['did_Sharpe_hi'], 2)}); QLIKE gap "
            f"{f_['dQLIKE']:+.4f} → {c_['dQLIKE']:+.4f} (change {d_['did_QLIKE']:+.4f} "
            f"{_iv(d_['did_QLIKE_lo'], d_['did_QLIKE_hi'])})"
        )
    a(
        "Ridge − lasso Sharpe gap: "
        + "; ".join(parts)
        + ". Averaging each correlated group into one input loses forecast "
        "information for both models and leaves the two estimators as far apart as before."
    )
    a("")
    # ---- gates
    a("## Gates and caveats")
    a("")
    gl = GT[GT.check.str.startswith("full")]
    lin_ok = gl[~gl.check.str.contains("LightGBM") & (gl.rows_above_1e6 == 0)][
        "max_rel"
    ].max()
    en = gl[gl.check.str.contains("elastic") & (gl.bucket == AF)].iloc[0]
    lg = gl[gl.check.str.contains("LightGBM")]
    tr_gate = GT[GT.check.str.startswith("per-day")]["max_rel"].max()
    st = {}
    for b in BUCKETS:
        tt = (
            pd.read_csv(STORED_TREES / "rescore_local" / "trees_trade_1530.csv")
            if (STORED_TREES / "rescore_local" / "trees_trade_1530.csv").is_file()
            else None
        )
        if tt is not None:
            rr = tt[
                (tt.bucket == b) & (tt.model == "lgbm") & (tt.forecast == "tree lgbm")
            ]
            st[b] = float(rr["Sharpe_mid"].iloc[0]) if len(rr) else float("nan")
    cont = GT[
        GT.check.str.startswith("spec continuous run of elastic net")
        & (GT.bucket == AF)
    ]
    a(
        f"* The design re-run through the spec's own executor call equals the stored research forecast (max relative "
        f"difference {NB[LF]['capture_gate']:.1e} `live_feasible`, {NB[AF]['capture_gate']:.1e} `all_features`); the local "
        f"full refits equal the stored per-bar ridge / lasso / elastic net forecasts to ≤ {lin_ok:.1e} relative, except "
        f"the `all_features` elastic net: up to {en['max_rel']:.1e} relative on {int(en['rows_above_1e6'])} rows of one "
        f"block (forecast rows {int(en['first_bad'])}–{int(en['last_bad'])})"
        + (
            f"; the spec's own continuous run on this machine differs from the stored run by up to "
            f"{cont['max_rel'].iloc[0]:.1e} on {int(cont['rows_above_1e6'].iloc[0])} rows (the warm homotopy is float-path "
            "dependent in that block)"
            if len(cont)
            else ""
        )
        + ". The elastic net enters only the density table."
    )
    a(
        f"* LightGBM refit locally (1 thread) differs from the stored cluster run (max {lg['max_rel'].min():.1e} / "
        f"{lg['max_rel'].max():.1e} relative; another platform and thread count), so the sweep compares LightGBM with its "
        f"own local all-column refit (Sharpe {sw(LF, 'LightGBM', 'all')['Sharpe_mid']:.2f} / "
        f"{sw(AF, 'LightGBM', 'all')['Sharpe_mid']:.2f} locally"
        + (f" vs {st[LF]:.2f} / {st[AF]:.2f} stored" if st else "")
        + "); the density table uses the stored runs' TreeSHAP."
    )
    a(
        f"* The per-day trade reproduces `trade_1530` to {tr_gate:.0e} for every run; near-zero adjusted-scale forecasts "
        "need no treatment (the recalibrated forecast is at least s·B)."
    )
    a(
        f"* Sample: {NB[LF]['n_deck']} trade days. Sharpe differences between variants carry intervals "
        f"{min(width):.1f}–{max(width):.1f} wide, so the trade cannot separate the estimators at this size; QLIKE "
        "differences of about 0.005 are resolvable."
    )
    a(
        "* The screen is one named rule (univariate |correlation| over the training window); its picks are dominated by "
        "the HAR ladder and absolute returns. A screen that decorrelates its picks would test a different notion of "
        "sparsity and was not run."
    )
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'SUMMARY.md'} ({len(L)} lines)")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if stage == "capture":
        capture(sys.argv[2])
    elif stage == "refit":
        refit(sys.argv[2] if len(sys.argv) > 2 else "full")
    elif stage == "analyze":
        analyze()
        write_summary()
    elif stage == "summary":
        write_summary()
    else:
        raise SystemExit(f"unknown stage {stage}")
