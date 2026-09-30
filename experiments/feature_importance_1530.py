"""Feature importance of the 16:00-bar forecast (the 15:30 forecast) with the professor's measures.

Stages (``python experiments/feature_importance_1530.py <stage> [args]``):

design     Re-builds the exact per-bar design the tree campaign fitted
           (specs/causal_tune_trees.py: run_executor, SEGMENT bar1600, global lags,
           2000-session window, prescale, impute_indicate) for the buckets baseline
           (HAR + calendar), live_feasible and all_features, on a scratch data dir holding
           only the vendor files the bucket reads (never the option chain).  The
           fit_predict handed to run_executor only records X, y and the column names.
           Gate: the out-of-sample stamps, targets and column names equal the stored tree
           runs'.
inputs     One input file per bucket for the tree tasks and the linear stage: the design,
           the realized variance and diurnal baseline of every forecast row (QLIKE), the
           permutation UNITS (every design column; every source series = all lags / flags
           of one input; every correlated cluster = complete-linkage groups of columns with
           |corr| >= CLUSTER_CORR), the thread counts and forecasts of the stored tree runs.
trees      (on the cluster; ``trees <bucket> <models> <k0> <k1>``) refits k0..k1-1 of
           LightGBM / XGBoost / random forest exactly as the untuned per-bar tree spec (same
           params, leaf minimums scaled to the 2000-row window, seed 42, the stored run's
           thread count, refit every 10 sessions on the 2000 rows before the anchor).  At
           every refit: (1) MDI = LightGBM gain / XGBoost total_gain / RF impurity decrease;
           (2) split count; (3) permutation importance on the CAUSAL held-out tail (the <= 10
           sessions this refit's model forecasts, up to the next refit): P1 shuffles the
           unit's values among the tail rows, P2 replaces them with rows drawn from the
           refit's own training window; loss = QLIKE (realized bar variance vs the plain
           back-transform yhat^2 x diurnal baseline) and squared error in the fit space;
           N_REPEATS draws per unit and scheme; (4) TreeSHAP on the same tail.  On every
           PROBE_EVERY-th refit a second fit adds three pure-noise columns (continuous,
           PROBE_LEVELS-valued, binary) and records the four measures for them.
linear     Per-bar ridge and lasso (specs/causal_tune_linear.py; the coefficient captures
           of experiments/model_diagnostics_1530.py, one coefficient vector per forecast,
           re-solved every session).  On the same tails with the SAME draws: permutation
           importance through the exact every-session coefficients; drop-column importance
           (re-solve on the anchor's window without the unit, the penalty in force held);
           the standardized weights |beta x sd| and contributions beta (x - window mean)
           (= linear SHAP with the window as background).
aggregate  Shares averaged over refits (all, by year, by causal VIX tercile), rank
           correlations between measures and across models, top-5 stability, cluster and
           series tables, unique-value counts, the probe table, figures and SUMMARY.md.

Outputs: results/feature_importance_1530/ (CSVs, PNGs, SUMMARY.md); inputs and per-refit
arrays in results/feature_importance_1530/_work/ (npz, not committed).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_importance_1530_trees import (  # noqa: E402
    BUCKETS,
    CLUSTER_CORR,
    OUT,
    PARAMS,
    PROBE_NAMES,
    REFIT_EVERY,
    REPO,
    SEED,
    SEG,
    TREE_MODELS,
    TW,
    WORK,
    _root,
    _tree_npz,
    draws,
    load_input,
    n_refits,
    qlike,
    tail_rows,
)

SCRATCH = OUT / "_scratch"
# the live_feasible / all_features coefficient captures (FEATIMP_CAPTURES_MD re-points a
# re-run at its own captures, made by the `capture` stage; unset = the first pass's)
MD_OUT = _root("FEATIMP_CAPTURES_MD", "results/model_diagnostics_1530")
MD_OUT_FIRST_PASS = REPO / "results" / "model_diagnostics_1530"
# the four vendor files (live/close_signal/forecast.py vendor_root rule) and, for
# all_features, the constituent cross-section and StockTwits files
# (experiments/model_diagnostics_1530.py BUCKET_VENDOR_FILES)
VENDOR_FILES = (
    "core_stats.parquet",
    "vix_and_voldemand.parquet",
    "releases.parquet",
    "time_categories.parquet",
)
BUCKET_FILES = {
    "baseline": VENDOR_FILES,
    "live_feasible": VENDOR_FILES,
    "all_features": VENDOR_FILES
    + ("ewstock_stats.parquet", "vwstock_stats.parquet", "spy_and_sentiment.parquet"),
}


# ============================================================================ design
def design(bucket: str) -> Path:
    """Record X, y, names of the per-bar design through the tree spec's run_executor call."""
    import pandas as pd

    sys.path.insert(0, str(REPO))
    from src.backtest.executor import run_executor
    from src.data.loading import get_bucket

    data = SCRATCH / f"data_{bucket}"
    data.mkdir(parents=True, exist_ok=True)
    for f in BUCKET_FILES[bucket]:
        if not (data / f).exists():
            shutil.copy2(REPO / "data" / f, data / f)
    assert "spxw_chain.parquet" not in {p.name for p in data.iterdir()}
    box: dict = {}

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        box.update(
            X=np.ascontiguousarray(X_chunk, dtype=np.float64),
            y=np.ascontiguousarray(y_chunk, dtype=np.float64),
            names=[str(c) for c in hyperparams["_feature_names"]],
            W=int(train_win_periods),
        )
        return np.asarray(y_chunk[int(train_win_periods) :], dtype=np.float64).copy()

    out_csv = SCRATCH / f"arm_{bucket}" / "results.csv"
    run_executor(
        method_name="featimp_design",
        fit_predict=fit_predict,
        hyperparams={},
        data_path=str(data),
        output_file=str(out_csv),
        horizon=1,
        train_window=TW,
        start=0,
        end=-1,
        halo=0,
        exog_cols=get_bucket(bucket),
        segment=SEG,
        lag_scope="global",
        har_lags=None,
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=SEED,
    )
    res = pd.read_csv(out_csv.with_name(f"results_{SEG}.csv"))
    W = box["W"]
    assert W == TW, (W, TW)
    z = np.load(_tree_npz(bucket, "lgbm"), allow_pickle=True)
    d_st = np.asarray(z["date"]).astype(str)
    d_me = res["date"].astype(str).str.slice(0, 19).to_numpy()
    assert len(d_st) == len(d_me) and (d_st == d_me).all(), (
        bucket,
        len(d_st),
        len(d_me),
    )
    gap = float(np.max(np.abs(np.asarray(z["true_adj"], float) - box["y"][W:])))
    assert gap < 1e-12, (bucket, gap)
    assert list(np.asarray(z["feature_names"]).astype(str)) == box["names"], bucket
    WORK.mkdir(parents=True, exist_ok=True)
    out_f = WORK / f"design_{bucket}.npz"
    np.savez_compressed(
        out_f,
        X=box["X"],
        y=box["y"],
        names=np.array(box["names"]),
        W=W,
        date_oos=d_me.astype("U19"),
    )
    print(
        f"design {bucket}: X {box['X'].shape}, W {W}, OOS {len(d_me)} "
        f"({d_me[0]} .. {d_me[-1]}); stamps, targets and names = the stored tree run "
        f"(target gap {gap:.1e})",
        flush=True,
    )
    return out_f


# ============================================================================ inputs
def series_of(names: list[str]) -> list[str]:
    sys.path.insert(0, str(REPO / "experiments"))
    import model_diagnostics_1530 as md

    return [md.parse_feature(x)[0] for x in names]


def corr_clusters(X: np.ndarray) -> np.ndarray:
    """Cluster label per column: complete linkage on 1 - |corr|, cut at 1 - CLUSTER_CORR.
    Columns constant over the whole series get their own label (no correlation)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    p = X.shape[1]
    sd = X.std(axis=0)
    var = np.flatnonzero(sd > 0)
    lab = np.zeros(p, dtype=np.int64)
    C = np.corrcoef(X[:, var], rowvar=False)
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0.0, None)  # symmetric, float noise below 0 removed
    Z = linkage(squareform(D, checks=False), method="complete")
    lv = fcluster(Z, t=1.0 - CLUSTER_CORR, criterion="distance")
    lab[var] = lv
    nxt = int(lv.max()) + 1
    for j in np.flatnonzero(sd == 0):
        lab[j] = nxt
        nxt += 1
    return lab


def build_inputs(bucket: str) -> Path:
    import pandas as pd

    D = np.load(WORK / f"design_{bucket}.npz", allow_pickle=True)
    X, y, W = D["X"], D["y"], int(D["W"])
    names = [str(v) for v in D["names"]]
    date_oos = np.asarray(D["date_oos"]).astype(str)
    p = len(names)
    assert np.isfinite(X).all() and np.isfinite(y).all(), bucket
    # realized variance and diurnal baseline of each forecast row (the research tables)
    yh = pd.read_parquet(
        REPO / "results" / "spxw_pnl" / "yhat_subtree_lgbm_live_feasible.parquet"
    )
    t = pd.DatetimeIndex(yh["t"]).tz_convert("America/New_York").tz_localize(None)
    m = (t.hour == 16) & (t.minute == 0)
    s = yh[m].set_index(t[m]).reindex(pd.DatetimeIndex(pd.to_datetime(date_oos)))
    assert s["rv_raw"].notna().all() and s["baseline"].notna().all()
    rv = s["rv_raw"].to_numpy(float)
    base = s["baseline"].to_numpy(float)
    ratio = rv / (y[W:] ** 2 * base)
    # units: columns, then source series with > 1 column, then clusters with > 1 column
    grp = series_of(names)
    lab = corr_clusters(X)
    units: list[tuple[str, str, list[int]]] = [
        ("column", n, [j]) for j, n in enumerate(names)
    ]
    for g in dict.fromkeys(grp):
        cols = [j for j in range(p) if grp[j] == g]
        if len(cols) > 1:
            units.append(("series", g, cols))
    ncl = 0
    for c in sorted(set(lab.tolist())):
        cols = [j for j in range(p) if lab[j] == c]
        if len(cols) > 1:
            ncl += 1
            units.append(("cluster", f"C{ncl:02d}", cols))
    ptr = np.cumsum([0] + [len(u[2]) for u in units])
    stored = {}
    threads = {}
    for mdl in TREE_MODELS:
        z = np.load(_tree_npz(bucket, mdl), allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        assert meta["params"] == PARAMS[mdl], (bucket, mdl, meta["params"])
        assert (np.asarray(z["date"]).astype(str) == date_oos).all()
        stored[mdl] = np.asarray(z["pred_adj"], float)
        threads[mdl] = int(meta["threads"])
    f = WORK / f"input_{bucket}.npz"
    np.savez_compressed(
        f,
        X=X,
        y=y,
        W=W,
        names=np.array(names),
        date_oos=date_oos.astype("U19"),
        rv=rv,
        base=base,
        unit_kind=np.array([u[0] for u in units]),
        unit_name=np.array([u[1] for u in units]),
        unit_cols=np.concatenate([np.asarray(u[2], dtype=np.int64) for u in units]),
        unit_ptr=ptr.astype(np.int64),
        cluster_label=lab,
        series=np.array(grp),
        threads=json.dumps(threads),
        **{f"stored_{k}": v for k, v in stored.items()},  # type: ignore[arg-type]
    )
    kinds = pd.Series([u[0] for u in units]).value_counts().to_dict()
    print(
        f"inputs {bucket}: {p} columns, units {kinds}; rv/(y^2 base) median "
        f"{np.median(ratio):.4f} (1 off the winsorized rows: {int((np.abs(ratio - 1) > 1e-9).sum())} rows); "
        f"stored threads {threads}",
        flush=True,
    )
    return f


# ============================================================================ linear
LIN_EST = {"ridge": "ridge", "lasso": "reclasso"}  # label -> the spec's estimator arm
CAPTURES = WORK / "linear_captures"
STORED_LIN = _root("FEATIMP_STORED_LIN", "results/linear_subsection/arms_hoffman2")


def capture_path(bucket: str, est: str) -> Path:
    """The coefficient capture of (bucket, ridge|lasso): the first pass's for live_feasible
    and all_features, this script's own (same code) for the HAR + calendar design."""
    if bucket == "baseline":
        return CAPTURES / (
            "capture_bar1600_baseline.npz"
            if est == "ridge"
            else "capture_bar1600_baseline_reclasso.npz"
        )
    stem = (
        "capture_bar1600" if bucket == "live_feasible" else f"capture_bar1600_{bucket}"
    )
    return MD_OUT / (f"{stem}.npz" if est == "ridge" else f"{stem}_reclasso.npz")


def capture_baseline(est: str) -> None:
    """experiments/model_diagnostics_1530.py capture, pointed at this script's work dir,
    for the HAR + calendar design (its vendor files = the four of live_feasible)."""
    sys.path.insert(0, str(REPO / "experiments"))
    import model_diagnostics_1530 as md

    CAPTURES.mkdir(parents=True, exist_ok=True)
    md.OUT = CAPTURES
    md.SCRATCH = SCRATCH / "md_capture"
    md.BUCKET_VENDOR_FILES["baseline"] = md.VENDOR_FILES  # type: ignore[assignment]
    md.capture(LIN_EST[est], None, "baseline")


def capture_bucket(bucket: str, est: str) -> Path:
    """The capture of any bucket into the directory capture_path() reads it from (a
    re-run's own FEATIMP_CAPTURES_MD; never the first pass's model-diagnostics captures)."""
    if bucket == "baseline":
        capture_baseline(est)
        return capture_path(bucket, est)
    assert MD_OUT.resolve() != MD_OUT_FIRST_PASS.resolve(), (
        "set FEATIMP_CAPTURES_MD: the first pass's captures are not overwritten"
    )
    sys.path.insert(0, str(REPO / "experiments"))
    import model_diagnostics_1530 as md

    MD_OUT.mkdir(parents=True, exist_ok=True)
    md.OUT = MD_OUT
    md.SCRATCH = SCRATCH / "md_capture"
    md.capture(LIN_EST[est], None, bucket)
    return capture_path(bucket, est)


def _mask_window(Xw: np.ndarray) -> np.ndarray:
    """The spec's identifiability mask (RollingTunedLinear._recompute_mask) of a raw window:
    columns constant in it or byte-identical to an earlier kept column."""
    out = np.zeros(Xw.shape[1], dtype=bool)
    seen: dict = {}
    for j in range(Xw.shape[1]):
        col = np.ascontiguousarray(Xw[:, j])
        if col.max() == col.min():
            out[j] = True
            continue
        key = col.tobytes()
        if key in seen:
            out[j] = True
        else:
            seen[key] = j
    return out


def run_linear(bucket: str, est: str, k0: int = 0, k1: int | None = None) -> Path:  # noqa: C901 - one linear walk
    """Permutation (P1, P2) and drop-column importance of the per-bar ridge / lasso on the
    same tails and draws as the trees, plus |beta x sd| per refit and the contributions."""
    sys.path.insert(0, str(REPO / "experiments"))
    import model_diagnostics_1530 as md

    d = load_input(bucket)
    X, y, W, rv, base = d["X"], d["y"], d["W"], d["rv"], d["base"]
    units = d["units"]
    U = len(units)
    p = X.shape[1]
    z = np.load(capture_path(bucket, est), allow_pickle=True)
    assert [str(v) for v in z["names"]] == d["names"], (bucket, est)
    assert (np.asarray(z["date"]).astype(str) == d["date_oos"]).all(), (bucket, est)
    th, xr, mu, sd, alpha = z["th"], z["x"], z["mu"], z["sd"], z["alpha"]
    pred = np.asarray(z["pred"], float)
    g_x = float(np.max(np.abs(xr - X[W:])))
    g_pred = float(
        np.max(
            np.abs(np.einsum("nk,nk->n", th[:, :-1], xr) + th[:, -1] - pred)
            / np.abs(pred)
        )
    )
    stored = pd.read_csv(
        STORED_LIN / bucket / LIN_EST[est] / f"tw{TW}" / f"results_{SEG}.csv"
    )
    sv = (
        stored.set_index(stored["date"].astype(str).str.slice(0, 19))["pred_adj"]
        .reindex(d["date_oos"])
        .to_numpy(float)
    )
    rel_st = np.abs(pred - sv) / np.abs(sv)
    g_st = float(np.max(rel_st))
    n_st = int((rel_st > 1e-9).sum())
    print(
        f"[{bucket} {est}] GATES: capture rows = design rows (max gap {g_x:.1e}); "
        f"coef.x + intercept = forecast (rel {g_pred:.1e}); capture = stored research forecast "
        f"(rel {g_st:.1e}, {n_st} rows > 1e-9); penalties {sorted({float(v) for v in alpha})}",
        flush=True,
    )
    assert g_x == 0.0 and g_pred < 1e-9, (g_x, g_pred)
    # FEATIMP_LIN_STORED_GATE=report (a re-run whose stored arm ran on another machine: the
    # lasso's warm-homotopy float path differs by CPU architecture) records the gap instead
    if os.environ.get("FEATIMP_LIN_STORED_GATE", "assert") != "report":
        assert g_st < 1e-6, g_st
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    ns = md._spec_namespace(LIN_EST[est], bucket) if est == "lasso" else None
    K = n_refits(d["n_oos"])
    dq = np.zeros((K, 2, U))
    dm = np.zeros((K, 2, U))
    sq = np.zeros((K, 2, U))
    ddq = np.zeros((K, U))
    ddm = np.zeros((K, U))
    bsd = np.zeros((K, p))
    q0 = np.zeros(K)
    m0 = np.zeros(K)
    q0a = np.zeros(K)
    m0a = np.zeros(K)
    anchor_gap = np.zeros(K)
    md5s = [""] * K
    done = np.zeros(K, dtype=bool)
    k1 = K if k1 is None else min(k1, K)
    t0 = time.time()
    for k in range(k0, k1):
        i, t, kk = tail_rows(d, k)
        Xt, yt, Xtr, ytr = X[t : t + kk], y[t : t + kk], X[t - W : t], y[t - W : t]
        rvk, bk = rv[i : i + kk], base[i : i + kk]
        Th = th[i : i + kk, :-1]
        yh0 = pred[i : i + kk]
        q0[k] = qlike(rvk, yh0, bk).mean()
        m0[k] = ((yt - yh0) ** 2).mean()
        p1, p2, h = draws(bucket, k, U, kk, W)
        md5s[k] = h
        live = np.abs(Th).max(axis=0) > 0
        for u in range(U):
            cols = units[u]
            if not live[cols].any():
                continue  # no weight on any tail row: the forecast cannot move
            T = Th[:, cols][None]
            base_x = Xt[:, cols][None]
            for s_, P, src in ((0, p1, Xt), (1, p2, Xtr)):
                xs = src[:, cols][P[u]]  # (R, kk, c)
                yh = yh0[None] + ((xs - base_x) * T).sum(axis=2)
                ql = qlike(rvk[None], yh, bk[None]).mean(axis=1)
                dq[k, s_, u] = ql.mean() - q0[k]
                sq[k, s_, u] = ql.std(ddof=1)
                dm[k, s_, u] = ((yt[None] - yh) ** 2).mean(axis=1).mean() - m0[k]
        # drop-column: the anchor's model (fitted on the window before the tail) re-solved
        # without the unit, the penalty in force held; both sides frozen over the tail
        a_ = float(alpha[i])
        Xa = np.hstack([Xtr, np.ones((W, 1))])
        Xta = np.hstack([Xt, np.ones((kk, 1))])
        locked = np.zeros(p + 1, dtype=bool)
        locked[-1] = True
        if est == "ridge":
            mask = np.append(th[i, :-1] == 0.0, False)
            Xa[:, mask] = 0.0
            G = Xa.T @ Xa
            G[np.diag_indices_from(G)] += np.where(locked, 0.0, a_)
            Kinv = np.linalg.inv(G)
            theta = Kinv @ (Xa.T @ ytr)
        else:
            assert ns is not None
            mask = np.append(_mask_window(Xtr), False)
            Xa[:, mask] = 0.0
            theta = ns["_batch_theta"](Xa, ytr, locked, a_, 1.0)
        anchor_gap[k] = float(np.max(np.abs(theta - th[i])) / np.max(np.abs(th[i])))
        yfa = Xta @ theta
        q0a[k] = qlike(rvk, yfa, bk).mean()
        m0a[k] = ((yt - yfa) ** 2).mean()
        for u in range(U):
            S = units[u]
            if not (theta[S] != 0).any():
                continue  # the unit carries no weight: dropping it leaves the fit unchanged
            if est == "ridge":
                KS = Kinv[:, S]
                th_d = theta - KS @ np.linalg.solve(Kinv[np.ix_(S, S)], theta[S])
                th_d[S] = 0.0
            else:
                assert ns is not None
                Xd = Xa.copy()
                Xd[:, S] = 0.0
                th_d = ns["_batch_theta"](Xd, ytr, locked, a_, 1.0)
            yd = Xta @ th_d
            ddq[k, u] = qlike(rvk, yd, bk).mean() - q0a[k]
            ddm[k, u] = ((yt - yd) ** 2).mean() - m0a[k]
        bsd[k] = np.abs(th[i, :-1] * sd[i])
        done[k] = True
        if k % 25 == 0 or k == K - 1:
            print(
                f"  {bucket} {est} refit {k}/{K}: anchor re-solve vs capture "
                f"{anchor_gap[k]:.1e}; elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
    phi = th[:, :-1] * (xr - mu)  # linear SHAP, window-mean background
    part = "" if (k0, k1) == (0, K) else f"_part{k0:03d}_{k1:03d}"
    out = WORK / "linear" / f"linear_{bucket}_{est}{part}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        dq=dq,
        dm=dm,
        sq=sq,
        ddq=ddq,
        ddm=ddm,
        bsd=bsd,
        q0=q0,
        m0=m0,
        q0a=q0a,
        m0a=m0a,
        anchor_gap=anchor_gap,
        draw_md5=np.array(md5s),
        phi=phi.astype(np.float32),
        pred=pred,
        alpha=alpha,
        gates=json.dumps(dict(rows=g_x, coef=g_pred, stored=g_st, stored_rows=n_st)),
        done=done,
    )
    print(
        f"wrote {out}: anchor re-solve vs capture max {anchor_gap.max():.1e} "
        f"(> 1e-6 on {int((anchor_gap > 1e-6).sum())} of {K}); {time.time() - t0:.0f}s",
        flush=True,
    )
    return out


def merge_linear(bucket: str, est: str) -> Path:
    """Combine the refit-range parts of a linear walk into the one file the aggregate reads."""
    import glob

    fs = sorted(glob.glob(str(WORK / "linear" / f"linear_{bucket}_{est}_part*.npz")))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    done = np.stack([z["done"] for z in zs])
    assert (done.sum(axis=0) == 1).all(), (bucket, est, done.sum(axis=0))
    out: dict = {}
    for key in (
        "dq",
        "dm",
        "sq",
        "ddq",
        "ddm",
        "bsd",
        "q0",
        "m0",
        "q0a",
        "m0a",
        "anchor_gap",
    ):
        out[key] = sum(z[key] for z in zs)
    out["draw_md5"] = np.array(
        [
            next(str(z["draw_md5"][k]) for z in zs if z["done"][k])
            for k in range(done.shape[1])
        ]
    )
    for key in ("phi", "pred", "alpha", "gates"):
        out[key] = zs[0][key]
    out["done"] = done.any(axis=0)
    f = WORK / "linear" / f"linear_{bucket}_{est}.npz"
    np.savez_compressed(f, **out)
    print(
        f"merged {len(fs)} parts -> {f}; anchor re-solve vs capture max {out['anchor_gap'].max():.1e} "
        f"(> 1e-6 on {int((out['anchor_gap'] > 1e-6).sum())} of {done.shape[1]})"
    )
    return f


# ============================================================================ aggregate
MODELS = ("ridge", "lasso", "lgbm", "xgb", "rf")
LABEL = {
    "ridge": "ridge",
    "lasso": "lasso",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
}
# measure key -> (plain label, applies to): the professor's four + the linear analogues
MEASURES = {
    "mdi": "MDI / gain (built-in)",
    "split": "split count",
    "bsd": "|beta x sd| (standardized weight)",
    "perm_p1_qlike": "permutation P1 (shuffle within tail), QLIKE",
    "perm_p2_qlike": "permutation P2 (draw from training window), QLIKE",
    "perm_p1_mse": "permutation P1, squared error",
    "perm_p2_mse": "permutation P2, squared error",
    "drop_qlike": "drop-column (re-solve), QLIKE",
    "drop_mse": "drop-column (re-solve), squared error",
    "shap": "SHAP mean |phi|",
}
TREE_MEASURES = (
    "mdi",
    "split",
    "perm_p1_qlike",
    "perm_p2_qlike",
    "perm_p1_mse",
    "perm_p2_mse",
    "shap",
)
LIN_MEASURES = (
    "bsd",
    "perm_p1_qlike",
    "perm_p2_qlike",
    "perm_p1_mse",
    "perm_p2_mse",
    "drop_qlike",
    "drop_mse",
    "shap",
)
SHARE_MEASURES = ("mdi", "split", "bsd", "shap")  # normalized to shares per refit
LEVELS = ("column", "series", "cluster")
TOP_N = 5  # the top-5 stability count (the professor's question: which inputs lead)
BOOT_B = 2000  # bootstrap draws for the intervals over refits
BOOT_SEED = 20260929
VIX_TERCILES = ("low", "mid", "high")


def _tree_chunks(bucket: str, model: str) -> dict:
    """The fleet's chunk outputs of (bucket, model), concatenated in refit order (the
    canary's short files are not part of the fleet and are left out)."""
    import glob

    fs = sorted(glob.glob(str(WORK / "trees" / f"trees_{bucket}_{model}_*_*.npz")))
    fleet = []
    for f in fs:
        a, b = (int(x) for x in Path(f).stem.split("_")[-2:])
        if (bucket == "baseline" and (a, b) == (0, 147)) or (
            bucket != "baseline" and b - a == 21
        ):
            fleet.append((a, f))
    fleet.sort()
    zs = [np.load(f, allow_pickle=True) for _, f in fleet]
    out: dict = {}
    for key in (
        "k",
        "pred_gap",
        "mdi",
        "split",
        "shap_gap",
        "q0",
        "m0",
        "dq",
        "dm",
        "sq",
        "fit_sec",
        "perm_sec",
        "shap_sec",
        "draw_md5",
        "pred",
        "shap",
        "probe_k",
        "probe_mdi",
        "probe_split",
        "probe_shap_abs",
        "probe_dq",
        "probe_dm",
        "probe_q0",
    ):
        out[key] = np.concatenate([z[key] for z in zs])
    for key in ("n_keep", "kept"):  # the window-masked refits' kept columns
        if all(key in z.files for z in zs):
            out[key] = np.concatenate([z[key] for z in zs])
    out["meta"] = [json.loads(str(z["meta"])) for z in zs]
    return out


def _boot_counts(K: int, rng) -> np.ndarray:
    """(BOOT_B, K) counts of each refit in circular-block bootstrap resamples of the
    refits (block = ceil(K^(1/3)), the first pass's rule)."""
    L = int(np.ceil(K ** (1.0 / 3.0)))
    nb = int(np.ceil(K / L))
    starts = rng.integers(0, K, size=(BOOT_B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT_B, -1)[
        :, :K
    ] % K
    C = np.zeros((BOOT_B, K))
    for b in range(BOOT_B):
        C[b] = np.bincount(idx[b], minlength=K)
    return C


def _rank_desc(v: np.ndarray) -> np.ndarray:
    """1 = largest; ties share the smallest rank."""
    from scipy.stats import rankdata

    return rankdata(-v, method="min").astype(int)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _vix_labels(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per refit: calendar year of the tail's first session and its causal VIX tercile
    (VIX at the 15:30 bar end of that session against the 1/3 and 2/3 quantiles of the
    15:30 VIX over the 2000 sessions before it; nothing after the forecast time)."""
    v = pd.read_parquet(
        REPO / "data" / "vix_and_voldemand.parquet", columns=["endbartime", "vix"]
    )
    t = pd.DatetimeIndex(pd.to_datetime(v["endbartime"]))
    m = (t.hour == 15) & (t.minute == 30)
    vix = pd.Series(v["vix"].to_numpy(float)[m], index=t[m].normalize()).dropna()
    vix = vix[~vix.index.duplicated()]
    K = n_refits(d["n_oos"])
    years = np.zeros(K, dtype=int)
    terc = np.empty(K, dtype=object)
    for k in range(K):
        i, _, _ = tail_rows(d, k)
        day = pd.Timestamp(str(d["date_oos"][i])).normalize()
        years[k] = day.year
        past = vix[vix.index < day].iloc[-TW:]
        now = vix.get(day, np.nan)
        lo, hi = np.quantile(past.to_numpy(), [1 / 3, 2 / 3])
        terc[k] = "low" if now <= lo else ("high" if now > hi else "mid")
    return years, terc


def _unit_maps(d: dict) -> dict:
    """Column -> series / cluster; the permutation unit index of every series and cluster."""
    names = d["names"]
    p = len(names)
    kind, uname = d["unit_kind"].astype(str), d["unit_name"].astype(str)
    ser = [str(s) for s in d["series"]]
    lab = d["cluster_label"]
    series = list(dict.fromkeys(ser))
    s_idx = {g: [j for j in range(p) if ser[j] == g] for g in series}
    s_unit = {}
    for g in series:
        hit = np.flatnonzero((kind == "series") & (uname == g))
        s_unit[g] = int(hit[0]) if len(hit) else int(s_idx[g][0])
    # cluster partition: multi-column clusters keep their unit name, singletons the column
    c_members: dict[str, list[int]] = {}
    c_unit: dict[str, int] = {}
    for u in np.flatnonzero(kind == "cluster"):
        cols = [int(j) for j in d["units"][u]]
        c_members[uname[u]] = cols
        c_unit[uname[u]] = int(u)
    in_multi = {j for cols in c_members.values() for j in cols}
    for j in range(p):
        if j not in in_multi:
            c_members[names[j]] = [j]
            c_unit[names[j]] = j
    assert sorted(j for c in c_members.values() for j in c) == list(range(p))
    assert all(len(set(lab[c].tolist())) == 1 for c in c_members.values())
    return dict(
        series=series,
        s_idx=s_idx,
        s_unit=s_unit,
        clusters=list(c_members),
        c_idx=c_members,
        c_unit=c_unit,
    )


def _level_units(
    d: dict, maps: dict, level: str
) -> tuple[list[str], list[list[int]], list[int]]:
    """(names, member columns, permutation unit) of every unit at the level."""
    if level == "column":
        return (
            d["names"],
            [[j] for j in range(len(d["names"]))],
            list(range(len(d["names"]))),
        )
    if level == "series":
        return (
            maps["series"],
            [maps["s_idx"][g] for g in maps["series"]],
            [maps["s_unit"][g] for g in maps["series"]],
        )
    return (
        maps["clusters"],
        [maps["c_idx"][c] for c in maps["clusters"]],
        [maps["c_unit"][c] for c in maps["clusters"]],
    )


def _per_refit(
    d: dict, maps: dict, R: dict, fam: str, level: str
) -> dict[str, np.ndarray]:
    """Per-refit (K x units) value of every measure of the model at the level."""
    K = n_refits(d["n_oos"])
    names, members, punit = _level_units(d, maps, level)
    out: dict[str, np.ndarray] = {}
    agg = np.zeros((len(d["names"]), len(names)))
    for q, cols in enumerate(members):
        agg[cols, q] = 1.0
    add = ("mdi", "split") if fam == "tree" else ("bsd",)
    for m in add:
        v = R[m] @ agg
        tot = v.sum(axis=1, keepdims=True)
        out[m] = np.divide(v, tot, out=np.zeros_like(v), where=tot > 0)
    # SHAP: |sum of the unit's phi| per row, mean over each refit's tail rows, shares
    phi = R["phi"]
    g = np.abs(phi @ agg)  # (n, units)
    rows = np.arange(len(phi)) // REFIT_EVERY
    sh = np.zeros((K, len(names)))
    np.add.at(sh, rows, g)
    sh /= np.bincount(rows, minlength=K)[:, None]
    out["shap_abs"] = sh
    tot = sh.sum(axis=1, keepdims=True)
    out["shap"] = np.divide(sh, tot, out=np.zeros_like(sh), where=tot > 0)
    out["shap_signed"] = np.zeros((K, len(names)))
    np.add.at(out["shap_signed"], rows, phi @ agg)
    out["shap_signed"] /= np.bincount(rows, minlength=K)[:, None]
    pu = np.asarray(punit)
    out["perm_p1_qlike"] = R["dq"][:, 0, pu]
    out["perm_p2_qlike"] = R["dq"][:, 1, pu]
    out["perm_p1_mse"] = R["dm"][:, 0, pu]
    out["perm_p2_mse"] = R["dm"][:, 1, pu]
    if fam == "linear":
        out["drop_qlike"] = R["ddq"][:, pu]
        out["drop_mse"] = R["ddm"][:, pu]
    return out


def _load_model(bucket: str, model: str, d: dict) -> dict:
    if model in TREE_MODELS:
        T = _tree_chunks(bucket, model)
        K = n_refits(d["n_oos"])
        assert (T["k"] == np.arange(K)).all(), (bucket, model)
        return dict(
            fam="tree",
            mdi=T["mdi"],
            split=T["split"],
            phi=T["shap"][:, :-1].astype(np.float64),
            dq=T["dq"],
            dm=T["dm"],
            sq=T["sq"],
            q0=T["q0"],
            m0=T["m0"],
            pred=T["pred"],
            pred_gap=T["pred_gap"],
            shap_gap=T["shap_gap"],
            draw_md5=T["draw_md5"],
            n_keep=T.get("n_keep"),
            raw=T,
        )
    z = np.load(WORK / "linear" / f"linear_{bucket}_{model}.npz", allow_pickle=True)
    return dict(
        fam="linear",
        bsd=z["bsd"],
        phi=z["phi"].astype(np.float64),
        dq=z["dq"],
        dm=z["dm"],
        sq=z["sq"],
        ddq=z["ddq"],
        ddm=z["ddm"],
        q0=z["q0"],
        m0=z["m0"],
        q0a=z["q0a"],
        m0a=z["m0a"],
        pred=z["pred"],
        anchor_gap=z["anchor_gap"],
        draw_md5=z["draw_md5"],
        gates=json.loads(str(z["gates"])),
    )


def unique_counts(d: dict) -> pd.DataFrame:
    """Distinct values per design column: the whole series and every training window."""
    X, W = d["X"], d["W"]
    K = n_refits(d["n_oos"])
    win = np.zeros((K, X.shape[1]), dtype=int)
    for k in range(K):
        _, t, _ = tail_rows(d, k)
        Xw = np.sort(X[t - W : t], axis=0)
        win[k] = 1 + (np.diff(Xw, axis=0) != 0).sum(axis=0)
    Xs = np.sort(X, axis=0)
    full = 1 + (np.diff(Xs, axis=0) != 0).sum(axis=0)
    med = np.median(win, axis=0)

    def cls(n: float) -> str:
        if n <= 1:
            return "constant"
        if n <= 2:
            return "binary"
        if n <= 25:
            return "3-25 values"
        if n <= 500:
            return "26-500 values"
        return "> 500 values"

    return pd.DataFrame(
        dict(
            column=d["names"],
            series=[str(s) for s in d["series"]],
            unique_full_series=full,
            unique_window_median=med,
            unique_window_min=win.min(axis=0),
            unique_window_max=win.max(axis=0),
            windows_constant=(win <= 1).sum(axis=0),
            cardinality_class=[cls(v) for v in med],
        )
    )


def aggregate(buckets: tuple[str, ...] = BUCKETS) -> None:  # noqa: C901 - one report
    sys.path.insert(0, str(REPO / "experiments"))
    import model_diagnostics_1530 as md

    OUT.mkdir(parents=True, exist_ok=True)
    BY = OUT / "by_measure"
    BY.mkdir(exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    S: dict = {}  # summary numbers for SUMMARY.md / the PDF
    gate_rows, corr_rows, xcorr_rows, strat_rows, card_rows, probe_rows, dil_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    top5_rows, clus_rows, uniq_all = [], [], []
    mean_tabs: dict = {}
    for bucket in buckets:
        d = load_input(bucket)
        K = n_refits(d["n_oos"])
        kk = np.array([tail_rows(d, k)[2] for k in range(K)], dtype=float)
        C = _boot_counts(K, rng)
        CW = C * kk[None]
        maps = _unit_maps(d)
        years, terc = _vix_labels(d)
        U = unique_counts(d)
        U.insert(0, "bucket", bucket)
        uniq_all.append(U)
        stems = {g: md.var_stem(g, d["names"]) for g in maps["series"]}
        stems["calendar"] = "calendar (DOW_*, is_*, ...)"
        # clusters of the bucket
        Xf = d["X"]
        for cname in maps["clusters"]:
            cols = maps["c_idx"][cname]
            if len(cols) < 2:
                continue
            cc = np.abs(np.corrcoef(Xf[:, cols], rowvar=False))
            clus_rows.append(
                dict(
                    bucket=bucket,
                    cluster=cname,
                    n_columns=len(cols),
                    min_abs_corr=float(cc[np.triu_indices(len(cols), 1)].min()),
                    series=",".join(dict.fromkeys(str(d["series"][j]) for j in cols)),
                    members=" ".join(d["names"][j] for j in cols),
                )
            )
        M: dict = {}
        for model in MODELS:
            M[model] = _load_model(bucket, model, d)
        # gates: draws identical for every model (md5 per refit), forecasts, additivity
        ref = M["lgbm"]["draw_md5"]
        for model in MODELS:
            R = M[model]
            same = int((R["draw_md5"] == ref).sum())
            gate_rows.append(
                dict(
                    bucket=bucket,
                    model=model,
                    refits=K,
                    draws_equal_to_lgbm=same,
                    anchor_refits_above_1e6=int((R["anchor_gap"] > 1e-6).sum())
                    if "anchor_gap" in R
                    else -1,
                    refits_gap_above_1e9=int((R["pred_gap"] > 1e-9).sum())
                    if "pred_gap" in R
                    else -1,
                    forecast_gap_max=float(R["pred_gap"].max())
                    if "pred_gap" in R
                    else np.nan,
                    forecast_gap_mean=float(
                        np.mean(np.abs(R["pred"] - d[f"stored_{model}"]))
                    )
                    if model in TREE_MODELS
                    else np.nan,
                    forecast_corr_stored=float(
                        np.corrcoef(R["pred"], d[f"stored_{model}"])[0, 1]
                    )
                    if model in TREE_MODELS
                    else np.nan,
                    shap_additivity_max=float(R["shap_gap"].max())
                    if "shap_gap" in R
                    else np.nan,
                    anchor_resolve_gap_max=float(R["anchor_gap"].max())
                    if "anchor_gap" in R
                    else np.nan,
                    capture_vs_stored_rel=R["gates"]["stored"]
                    if "gates" in R
                    else np.nan,
                    **(
                        dict(capture_vs_stored_rows_1e9=R["gates"]["stored_rows"])
                        if "gates" in R and "stored_rows" in R["gates"]
                        else {}
                    ),
                    tail_qlike_mean=float(np.average(R["q0"], weights=kk)),
                    tail_mse_mean=float(np.average(R["m0"], weights=kk)),
                    **(
                        dict(
                            kept_columns_min=int(R["n_keep"].min()),
                            kept_columns_median=float(np.median(R["n_keep"])),
                            kept_columns_max=int(R["n_keep"].max()),
                        )
                        if R.get("n_keep") is not None
                        else {}
                    ),
                )
            )
            assert same == K, (bucket, model, same)
        # per level, per model, per measure
        for level in LEVELS:
            unames, members, _ = _level_units(d, maps, level)
            nu = len(unames)
            # eligible units: at least one member column varies in at least one window
            varies = U["windows_constant"].to_numpy() < K
            elig = np.array([varies[cols].any() for cols in members])
            if level == "series":
                disp = [stems[g] for g in unames]
            elif level == "cluster":
                disp = [
                    c
                    if len(maps["c_idx"][c]) == 1
                    else f"{c}: {d['names'][maps['c_idx'][c][0]]} +{len(maps['c_idx'][c]) - 1}"
                    for c in unames
                ]
            else:
                disp = list(unames)
            means: dict = {}
            for model in MODELS:
                R = M[model]
                V = _per_refit(d, maps, R, R["fam"], level)
                meas = TREE_MEASURES if R["fam"] == "tree" else LIN_MEASURES
                loss0 = {
                    "qlike": np.average(R["q0"], weights=kk),
                    "mse": np.average(R["m0"], weights=kk),
                }
                if R["fam"] == "linear":
                    loss0_anchor = {
                        "qlike": np.average(R["q0a"], weights=kk),
                        "mse": np.average(R["m0a"], weights=kk),
                    }
                for m in meas:
                    v = V[m]
                    mean = (kk[:, None] * v).sum(axis=0) / kk.sum()
                    boot = (CW @ v) / CW.sum(axis=1, keepdims=True)
                    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
                    rk = np.zeros(nu, dtype=int)
                    rk[elig] = _rank_desc(mean[elig])
                    rk[~elig] = 0
                    # per-refit top-N among eligible units
                    top = np.zeros(nu)
                    ve = v[:, elig]
                    order = np.argsort(-ve, axis=1, kind="stable")[:, :TOP_N]
                    hits = np.zeros(ve.shape[1])
                    for row, o in zip(ve, order):
                        hits[o[row[o] > 0]] += 1
                    top[elig] = hits / K
                    tab = pd.DataFrame(
                        dict(
                            bucket=bucket,
                            model=model,
                            measure=m,
                            level=level,
                            unit=unames,
                            label=disp,
                            n_columns=[len(c) for c in members],
                            eligible=elig,
                            value=mean,
                            lo=lo,
                            hi=hi,
                            rank=rk,
                            top5_share=top,
                        )
                    )
                    if m.startswith(("perm", "drop")):
                        lossk = m.split("_")[-1]
                        base0 = (
                            loss0_anchor[lossk]
                            if m.startswith("drop")
                            else loss0[lossk]
                        )
                        tab["pct_of_loss"] = 100 * mean / base0
                        tab["share_refits_positive"] = (v > 0).mean(axis=0)
                        # robustness: the median refit, and how much of the sum one refit carries
                        # (a linear forecast pushed near zero by a perturbed input makes QLIKE
                        # explode on one tail)
                        tab["median_refit"] = np.median(v, axis=0)
                        wv = kk[:, None] * v
                        tot_ = wv.sum(axis=0)
                        tab["max_refit_share"] = np.divide(
                            wv.max(axis=0),
                            tot_,
                            out=np.full(nu, np.nan),
                            where=tot_ > 0,
                        )
                    if m == "shap":
                        tab["mean_abs_phi"] = (kk[:, None] * V["shap_abs"]).sum(
                            axis=0
                        ) / kk.sum()
                        tab["mean_signed_phi"] = (kk[:, None] * V["shap_signed"]).sum(
                            axis=0
                        ) / kk.sum()
                    for yv in sorted(set(years)):
                        sel = years == yv
                        tab[f"y{yv}"] = (kk[sel, None] * v[sel]).sum(axis=0) / kk[
                            sel
                        ].sum()
                    for tv in VIX_TERCILES:
                        sel = terc == tv
                        tab[f"vix_{tv}"] = (kk[sel, None] * v[sel]).sum(axis=0) / kk[
                            sel
                        ].sum()
                    mean_tabs[(bucket, model, m, level)] = tab
                    means[(model, m)] = mean
                    # stability of the ranking by stratum (eligible units)
                    for sname, labels in (("year", years), ("vix_tercile", terc)):
                        for sv in sorted(set(labels.tolist()), key=str):
                            sel = labels == sv
                            ms = (kk[sel, None] * v[sel]).sum(axis=0) / kk[sel].sum()
                            strat_rows.append(
                                dict(
                                    bucket=bucket,
                                    model=model,
                                    measure=m,
                                    level=level,
                                    stratum=sname,
                                    value=str(sv),
                                    refits=int(sel.sum()),
                                    spearman_vs_all=_spearman(ms[elig], mean[elig]),
                                    leader=disp[
                                        int(np.flatnonzero(elig)[np.argmax(ms[elig])])
                                    ],
                                    leader_all=disp[
                                        int(np.flatnonzero(elig)[np.argmax(mean[elig])])
                                    ],
                                )
                            )
                    if level in ("series", "cluster") or (
                        level == "column" and bucket != "all_features"
                    ):
                        for q in np.flatnonzero(elig):
                            if top[q] > 0:
                                top5_rows.append(
                                    dict(
                                        bucket=bucket,
                                        level=level,
                                        model=model,
                                        measure=m,
                                        unit=disp[q],
                                        top5_share=top[q],
                                    )
                                )
                # within-model rank correlation between measures (eligible units)
                for a_i, a in enumerate(meas):
                    for b in meas[a_i + 1 :]:
                        corr_rows.append(
                            dict(
                                bucket=bucket,
                                level=level,
                                model=model,
                                a=a,
                                b=b,
                                n_units=int(elig.sum()),
                                spearman=_spearman(
                                    means[(model, a)][elig], means[(model, b)][elig]
                                ),
                            )
                        )
            # across models, same measure (and every tree-vs-linear pair of comparable ones)
            keys = list(means)
            for a_i, (ma, ea) in enumerate(keys):
                for mb, eb in keys[a_i + 1 :]:
                    if ma == mb:
                        continue
                    xcorr_rows.append(
                        dict(
                            bucket=bucket,
                            level=level,
                            model_a=ma,
                            measure_a=ea,
                            model_b=mb,
                            measure_b=eb,
                            n_units=int(elig.sum()),
                            spearman=_spearman(
                                means[(ma, ea)][elig], means[(mb, eb)][elig]
                            ),
                        )
                    )
            # the dilution check: a cluster's joint permutation vs the sum of its columns'
            if level == "cluster":
                for model in MODELS:
                    R = M[model]
                    for cname in maps["clusters"]:
                        cols = maps["c_idx"][cname]
                        if len(cols) < 2:
                            continue
                        u = maps["c_unit"][cname]
                        for s_, sch in ((0, "P1"), (1, "P2")):
                            joint = float(np.average(R["dq"][:, s_, u], weights=kk))
                            parts = float(
                                np.average(R["dq"][:, s_, cols].sum(axis=1), weights=kk)
                            )
                            dil_rows.append(
                                dict(
                                    bucket=bucket,
                                    model=model,
                                    scheme=sch,
                                    cluster=cname,
                                    n_columns=len(cols),
                                    members=" ".join(d["names"][j] for j in cols),
                                    joint_dqlike=joint,
                                    sum_of_columns_dqlike=parts,
                                    joint_dmse=float(
                                        np.average(R["dm"][:, s_, u], weights=kk)
                                    ),
                                    sum_of_columns_dmse=float(
                                        np.average(
                                            R["dm"][:, s_, cols].sum(axis=1), weights=kk
                                        )
                                    ),
                                    joint_over_sum=joint / parts
                                    if parts != 0
                                    else np.nan,
                                )
                            )
        # cardinality: does a measure track the number of distinct values (column level)
        uw = U["unique_window_median"].to_numpy(float)
        var = U["windows_constant"].to_numpy() < K
        for model in MODELS:
            meas = TREE_MEASURES if model in TREE_MODELS else LIN_MEASURES
            for m in meas:
                v = mean_tabs[(bucket, model, m, "column")]["value"].to_numpy()
                card_rows.append(
                    dict(
                        bucket=bucket,
                        model=model,
                        measure=m,
                        n_columns=int(var.sum()),
                        spearman_with_unique_count=_spearman(np.log(uw[var]), v[var]),
                    )
                )
                cls_ = U["cardinality_class"].to_numpy()
                tot = v[var].sum()
                for c in ("binary", "3-25 values", "26-500 values", "> 500 values"):
                    sel = var & (cls_ == c)
                    card_rows[-1][f"share_{c}"] = (
                        float(v[sel].sum() / tot) if tot != 0 else np.nan
                    )
                    card_rows[-1][f"n_{c}"] = int(sel.sum())
        # the probes
        p = len(d["names"])
        for model in TREE_MODELS:
            T = M[model]["raw"]
            nk = len(T["probe_k"])
            for j, pn in enumerate(PROBE_NAMES):
                col = p + j
                row = dict(bucket=bucket, model=model, probe=pn, refits=nk)
                for m, arr in (
                    ("mdi", T["probe_mdi"]),
                    ("split", T["probe_split"]),
                    ("shap", T["probe_shap_abs"]),
                ):
                    sh = arr[:, col] / arr.sum(axis=1)
                    ranks = np.array([1 + int((a[:p] > a[col]).sum()) for a in arr])
                    real_var = np.array([int((a[:p] > 0).sum()) for a in arr])
                    row[f"{m}_share_mean"] = float(sh.mean())
                    row[f"{m}_rank_median"] = float(np.median(ranks))
                    row[f"{m}_real_columns_below_share"] = float(
                        np.mean([(a[:p][a[:p] > 0] < a[col]).mean() for a in arr])
                    )
                    row[f"{m}_real_columns_used_median"] = float(np.median(real_var))
                for s_, sch in ((0, "p1"), (1, "p2")):
                    vq = T["probe_dq"][:, s_, j]
                    vm = T["probe_dm"][:, s_, j]
                    bq = rng.choice(vq, size=(BOOT_B, nk), replace=True).mean(axis=1)
                    row[f"perm_{sch}_qlike_mean"] = float(vq.mean())
                    row[f"perm_{sch}_qlike_lo"] = float(np.percentile(bq, 2.5))
                    row[f"perm_{sch}_qlike_hi"] = float(np.percentile(bq, 97.5))
                    row[f"perm_{sch}_mse_mean"] = float(vm.mean())
                row["probe_fit_tail_qlike"] = float(T["probe_q0"].mean())
                probe_rows.append(row)
        S[bucket] = dict(
            K=K,
            p=p,
            n=d["n_oos"],
            first=d["date_oos"][0],
            last=d["date_oos"][-1],
            years={int(y): int((years == y).sum()) for y in sorted(set(years))},
            terciles={t: int((terc == t).sum()) for t in VIX_TERCILES},
            n_series=len(maps["series"]),
            n_clusters=sum(len(c) > 1 for c in maps["c_idx"].values()),
            n_cluster_cols=sum(len(c) for c in maps["c_idx"].values() if len(c) > 1),
        )
        say(
            f"[{bucket}] {K} refits, {p} columns, {len(maps['series'])} series, "
            f"{S[bucket]['n_clusters']} clusters of {S[bucket]['n_cluster_cols']} columns"
        )

    # ---- write the tables
    for (bucket, model, m, level), tab in mean_tabs.items():
        pass
    for bucket in buckets:
        for model in MODELS:
            meas = TREE_MEASURES if model in TREE_MODELS else LIN_MEASURES
            for m in meas:
                tab = pd.concat([mean_tabs[(bucket, model, m, lv)] for lv in LEVELS])
                tab.to_csv(
                    BY / f"imp_{bucket}_{model}_{m}.csv",
                    index=False,
                    float_format="%.6g",
                )
    pd.DataFrame(gate_rows).to_csv(OUT / "gates.csv", index=False, float_format="%.6g")
    pd.DataFrame(corr_rows).to_csv(
        OUT / "rank_corr_within_model.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(xcorr_rows).to_csv(
        OUT / "rank_corr_across_models.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(strat_rows).to_csv(
        OUT / "stability_by_stratum.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(card_rows).to_csv(
        OUT / "cardinality_vs_measure.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(probe_rows).to_csv(
        OUT / "noise_probes.csv", index=False, float_format="%.6g"
    )
    pd.DataFrame(dil_rows).to_csv(
        OUT / "cluster_dilution.csv", index=False, float_format="%.6g"
    )
    pd.DataFrame(clus_rows).to_csv(
        OUT / "clusters.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(top5_rows).to_csv(
        OUT / "top5_stability.csv", index=False, float_format="%.4f"
    )
    pd.concat(uniq_all).to_csv(OUT / "unique_values.csv", index=False)
    # the headline table: series-level, every model x measure, value and rank
    head = pd.concat(
        [t for (b, mo, m, lv), t in mean_tabs.items() if lv == "series"],
        ignore_index=True,
    )
    head.to_csv(OUT / "series_level_all.csv", index=False, float_format="%.6g")
    (OUT / "aggregate.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "aggregate_meta.json").write_text(
        json.dumps(S, indent=1, default=str), encoding="utf-8"
    )
    if tuple(buckets) == BUCKETS:
        tuned_trees(mean_tabs)
    figures(mean_tabs)


def tuned_trees(mean_tabs: dict) -> None:
    """The causally tuned trees' own importance extracts (MDI / gain and TreeSHAP at the
    16:00 bar), series level, beside the untuned trees' -- if the tuned run saved them."""
    T = _root("FEATIMP_TUNED_IMP", "results/linear_subsection_trees_tuned/importance")
    rows = []
    if not T.is_dir():
        pd.DataFrame([dict(note="tuned-tree importance extracts not present")]).to_csv(
            OUT / "tuned_trees_series.csv", index=False
        )
        return
    for bucket in BUCKETS:
        d = load_input(bucket)
        ser = np.array([str(s) for s in d["series"]])
        for model in TREE_MODELS:
            fr = T / f"importance_refits_{bucket}_{model}.parquet"
            fs = T / f"shap_rows_{SEG}_{bucket}_{model}.parquet"
            if not (fr.is_file() and fs.is_file()):
                continue
            imp = pd.read_parquet(fr)
            imp = imp[imp["segment"] == SEG]
            A = imp[d["names"]].to_numpy(float)
            A = A / A.sum(axis=1, keepdims=True)
            sh = pd.read_parquet(fs)
            assert list(sh["date"].astype(str).str.slice(0, 19)) == list(
                d["date_oos"]
            ), (bucket, model)
            phi = sh[d["names"]].to_numpy(float)
            un = mean_tabs[(bucket, model, "mdi", "series")].set_index("unit")
            us = mean_tabs[(bucket, model, "shap", "series")].set_index("unit")
            nrow = np.arange(len(phi)) // REFIT_EVERY
            K = n_refits(d["n_oos"])
            groups = list(un.index)
            G = np.column_stack([np.abs(phi[:, ser == g].sum(axis=1)) for g in groups])
            per = np.zeros((K, len(groups)))
            np.add.at(per, nrow, G)
            per /= per.sum(
                axis=1, keepdims=True
            )  # per-refit shares, as the untuned table
            for q, g in enumerate(groups):
                cols = ser == g
                rows.append(
                    dict(
                        bucket=bucket,
                        model=model,
                        series=g,
                        refits_mdi=len(A),
                        tuned_mdi_share=float(A[:, cols].sum(axis=1).mean()),
                        tuned_shap_share=float(per[:, q].mean()),
                        untuned_mdi_share=float(un.loc[g, "value"]),
                        untuned_shap_share=float(us.loc[g, "value"]),
                    )
                )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "tuned_trees_series.csv", index=False, float_format="%.6g")


def figures(mean_tabs: dict) -> None:  # noqa: C901 - one figure set
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the first pass's validated categorical slots (model_diagnostics_1530_trees.py)
    COLOR = {
        "ridge": "#2a78d6",
        "lasso": "#4a3aa7",
        "lgbm": "#eb6834",
        "xgb": "#1baf7a",
        "rf": "#eda100",
    }
    SLOTS = {
        "tree": ("mdi", "split", "perm_p2_qlike", "shap"),
        "linear": ("bsd", "drop_mse", "perm_p2_mse", "shap"),
    }
    SLOT_TITLE = (
        "built-in: MDI / gain (trees), |beta x sd| (linear), share",
        "split count (trees), drop-column dMSE (linear)",
        "permutation P2 on the tail: dQLIKE (trees), dMSE (linear)",
        "SHAP mean |phi|, share",
    )
    N_BAR = 8
    have = [b for b in BUCKETS if (b, "ridge", "shap", "series") in mean_tabs]
    for bucket in have:
        fig, axes = plt.subplots(
            len(MODELS), 4, figsize=(12.5, 2.05 * len(MODELS) + 0.5)
        )
        for r, model in enumerate(MODELS):
            fam = "tree" if model in TREE_MODELS else "linear"
            for c, m in enumerate(SLOTS[fam]):
                ax = axes[r, c]
                t = mean_tabs[(bucket, model, m, "series")]
                t = t[t["eligible"]].sort_values("value", ascending=False).head(N_BAR)
                y = np.arange(len(t))[::-1]
                share = m in SHARE_MEASURES
                scale = 100.0 if share else 1e3
                val = scale * t["value"].to_numpy()
                ax.barh(y, val, color=COLOR[model], height=0.7)
                if not share or m == "shap":
                    lo = scale * t["lo"].to_numpy()
                    hi = scale * t["hi"].to_numpy()
                    ax.errorbar(
                        val,
                        y,
                        xerr=[val - lo, hi - val],
                        fmt="none",
                        ecolor="0.25",
                        lw=0.7,
                        capsize=1.5,
                    )
                ax.set_yticks(y)
                ax.set_yticklabels(t["label"], fontsize=5.8)
                ax.tick_params(axis="x", labelsize=6)
                ax.axvline(0, color="0.5", lw=0.5)
                for sp_ in ("top", "right"):
                    ax.spines[sp_].set_visible(False)
                ax.grid(axis="x", color="0.9", lw=0.5)
                ax.set_axisbelow(True)
                unit = "%" if share else "x 1e-3"
                ax.set_xlabel(unit, fontsize=6, labelpad=1)
                if r == 0:
                    ax.set_title(SLOT_TITLE[c], fontsize=7)
                if c == 0:
                    ax.set_ylabel(
                        LABEL[model], fontsize=8, color=COLOR[model], weight="bold"
                    )
                # the value at the bar end: the small bars stay readable next to the leader
                for yy, vv in zip(y, val):
                    ax.text(
                        vv,
                        yy,
                        f" {vv:.2g}",
                        va="center",
                        ha="left",
                        fontsize=4.8,
                        color="0.3",
                    )
        fig.tight_layout(rect=(0.02, 0, 1, 1))
        fig.savefig(OUT / f"fig_ranked_{bucket}.png", dpi=150)
        plt.close(fig)

    # heat map of ranks across measures (series level; cluster level for live_feasible)
    HEAT = {
        "tree": ("mdi", "split", "perm_p1_qlike", "perm_p2_qlike", "shap"),
        "linear": ("bsd", "perm_p1_mse", "perm_p2_mse", "drop_mse", "shap"),
    }
    SHORT = {
        "mdi": "MDI",
        "split": "split",
        "bsd": "|b sd|",
        "perm_p1_qlike": "perm P1 QL",
        "perm_p2_qlike": "perm P2 QL",
        "perm_p1_mse": "perm P1 MSE",
        "perm_p2_mse": "perm P2 MSE",
        "drop_mse": "drop MSE",
        "shap": "SHAP",
    }
    for bucket, level, nmax in (
        ("baseline", "series", 30),
        ("live_feasible", "series", 30),
        ("all_features", "series", 30),
        ("live_feasible", "cluster", 30),
        ("all_features", "cluster", 30),
    ):
        if bucket not in have:
            continue
        cols, R = [], []
        for model in MODELS:
            fam = "tree" if model in TREE_MODELS else "linear"
            for m in HEAT[fam]:
                t = mean_tabs[(bucket, model, m, level)]
                cols.append(f"{LABEL[model]}: {SHORT[m]}")
                R.append(t.set_index("label")["rank"])
        H = pd.concat(R, axis=1)
        H.columns = cols
        el = mean_tabs[(bucket, "ridge", "shap", level)].set_index("label")["eligible"]
        H = H[el.reindex(H.index).to_numpy()]
        Hn = H.replace(0, np.nan)
        order = Hn.median(axis=1).sort_values().index[:nmax]
        H = Hn.loc[order]
        n_el = int(el.sum())
        fig, ax = plt.subplots(figsize=(0.33 * len(cols) + 3.2, 0.24 * len(H) + 1.9))
        cap = min(n_el, 20)
        im = ax.imshow(
            np.minimum(H.to_numpy(), cap),
            cmap="Blues_r",
            vmin=1,
            vmax=cap,
            aspect="auto",
        )
        for (i_, j_), v in np.ndenumerate(H.to_numpy()):
            if np.isfinite(v):
                ax.text(
                    j_,
                    i_,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if v <= cap * 0.45 else "0.15",
                )
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(np.arange(len(H)))
        ax.set_yticklabels(H.index, fontsize=6)
        for x in np.cumsum(
            [len(HEAT["tree" if mo in TREE_MODELS else "linear"]) for mo in MODELS]
        )[:-1]:
            ax.axvline(x - 0.5, color="white", lw=2)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(
            f"rank among {n_el} {level} units (1 = most important; >= {cap} shown alike)",
            fontsize=6,
        )
        cb.ax.tick_params(labelsize=6)
        ax.set_title(
            f"{bucket}: rank of every {level} by model x measure (rows sorted by median rank; top {len(H)})",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"fig_rank_heatmap_{bucket}_{level}.png", dpi=150)
        plt.close(fig)

    # cardinality: MDI / split vs permutation against the number of distinct values
    U = pd.read_csv(OUT / "unique_values.csv")
    bucket = "live_feasible"
    u = U[U["bucket"] == bucket].reset_index(drop=True)
    var = u["windows_constant"].to_numpy() < n_refits(1469)
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.4), sharex=True)
    ccol = {
        "binary": "#4a3aa7",
        "3-25 values": "#1baf7a",
        "26-500 values": "#eda100",
        "> 500 values": "#eb6834",
    }
    for c, model in enumerate(TREE_MODELS):
        for r, m in enumerate(("mdi", "perm_p2_qlike", "shap")):
            ax = axes[r, c]
            t = mean_tabs[(bucket, model, m, "column")].reset_index(drop=True)
            x = np.log10(u["unique_window_median"].to_numpy(float))
            yv = t["value"].to_numpy() * (100 if m in SHARE_MEASURES else 1e3)
            for cls_, colr in ccol.items():
                sel = var & (u["cardinality_class"].to_numpy() == cls_)
                ax.scatter(
                    x[sel], yv[sel], s=9, color=colr, label=cls_, alpha=0.85, lw=0
                )
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_yscale("symlog", linthresh=0.01 if m in SHARE_MEASURES else 0.001)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(LABEL[model], fontsize=8, color=COLOR[model])
            if c == 0:
                ax.set_ylabel(
                    {
                        "mdi": "MDI / gain share (%)",
                        "perm_p2_qlike": "perm P2 dQLIKE x 1e-3",
                        "shap": "SHAP share (%)",
                    }[m],
                    fontsize=7,
                )
            if r == 2:
                ax.set_xlabel(
                    "log10 distinct values in the training window (median over refits)",
                    fontsize=6,
                )
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig_cardinality_live_feasible.png", dpi=150)
    plt.close(fig)

    # stability: share of refits in which a series is top-5, live_feasible
    bucket = "live_feasible"
    cols, R = [], []
    for model in MODELS:
        fam = "tree" if model in TREE_MODELS else "linear"
        for m in HEAT[fam]:
            t = mean_tabs[(bucket, model, m, "series")]
            cols.append(f"{LABEL[model]}: {SHORT[m]}")
            R.append(t.set_index("label")["top5_share"])
    H = pd.concat(R, axis=1)
    H.columns = cols
    H = H[(H > 0).any(axis=1)]
    H = H.loc[H.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(0.33 * len(cols) + 3.2, 0.26 * len(H) + 1.9))
    im = ax.imshow(100 * H.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for (i_, j_), v in np.ndenumerate(100 * H.to_numpy()):
        if v >= 1:
            ax.text(
                j_,
                i_,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=5,
                color="white" if v > 55 else "0.15",
            )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(H)))
    ax.set_yticklabels(H.index, fontsize=6)
    for x in np.cumsum(
        [len(HEAT["tree" if mo in TREE_MODELS else "linear"]) for mo in MODELS]
    )[:-1]:
        ax.axvline(x - 0.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% of the 147 refits in which the series is in the top 5", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(
        "live_feasible: top-5 stability over refits (series level)", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_top5_stability_live_feasible.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "design":
        for b in sys.argv[2:] or BUCKETS:
            design(b)
    elif stage == "inputs":
        for b in sys.argv[2:] or BUCKETS:
            build_inputs(b)
    elif stage == "capture_baseline":
        capture_baseline(sys.argv[2])
    elif stage == "capture":  # capture <bucket> <est>
        capture_bucket(sys.argv[2], sys.argv[3])
    elif stage == "aggregate":
        aggregate(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else BUCKETS)
    elif stage == "figures":
        figures({})
    elif stage == "linear_part":  # linear <bucket> <est> <k0> <k1>
        run_linear(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif stage == "merge_linear":
        merge_linear(sys.argv[2], sys.argv[3])
    elif stage == "linear":
        for b in sys.argv[2].split(",") if len(sys.argv) > 2 else BUCKETS:
            for e in sys.argv[3].split(",") if len(sys.argv) > 3 else LIN_EST:
                run_linear(b, e)
    else:
        raise SystemExit(f"unknown stage {stage!r}")
