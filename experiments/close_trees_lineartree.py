"""Linear leaves for the 16:00 LightGBM: one setting changed, ``linear_tree``  (close study, 2026-10-03).

The user's question: do the 16:00-bar trees lose to the linear models because they approximate a
mostly linear signal with step functions?  The HAR ladder carries most of the trees' forecast (the
shipped LightGBM's ``har_ma_1`` and ``har_ma_5`` hold about 56 % of its forecast variance), and a sum
of 392 piecewise-constant trees can only approximate a linear function of it.  LightGBM's
``linear_tree=True`` keeps the usual split search but fits, in every leaf, a linear model in the
numerical features used on that leaf's branch (the first tree keeps constant leaves).  This study
changes that one setting and nothing else.

MODEL: specs/causal_tune_trees.py's LightGBM exactly as shipped -- LGBM_PARAMS (min_child_samples
scaled 98 -> 8 for the 2000-row window by the spec's leaf-minimum rule), random_state 42,
num_threads 1 -- read from the spec through dense_vs_sparse_1530.spec_ns("tree"), on the captured
16:00-bar design (experiments/capture_design_close.py: results/close_design/_work/
design_bar1600_<bucket>.npz), window W = 2000 sessions, the per-window column mask
(src/models/window_mask.window_keep on each fit's own window: the spec's WINDOW_MASK = 1), a refit
every REFIT_EVERY sessions on the 2000 rows strictly before the refit anchor, every row predicted by
the model in force.  The linear-leaf arms add ``linear_tree=True`` and ``linear_lambda`` (default 0;
a positive value only if the default proves numerically unstable).

LightGBM 4.7.0 constraints on linear_tree (include/LightGBM/config.h, src/io/config.cpp,
src/treelearner/linear_tree_learner.cpp of the 4.7.0 source): CPU or GPU device and the serial tree
learner only (the spec's single-threaded CPU fit is both); ``zero_as_missing`` must be false (the
default) and the ``regression_l1`` objective is refused (the spec uses the default L2 regression);
TreeSHAP (``pred_contrib``) is not implemented for linear trees (not used here).  The leaf model is
one Newton step on the leaf's rows, -(X'HX + linear_lambda I)^(-1) X'g with X = [branch features, 1],
solved by a full-pivot LU; a leaf with fewer rows than branch features + 1 keeps its constant.  Rows
with a NaN in any of the leaf's features are left out of that leaf's fit and are predicted with the
leaf's constant; the captured designs hold no NaN (impute_indicate, checked below).  ``lambda_l1`` and
``lambda_l2`` act on the split search and the constant only; ``linear_lambda`` is the only penalty on
the leaf coefficients.  LightGBM recommends features on similar scales: the design is prescaled (the
executor's rolling robust scaling).

ARMS (name = <ctrl|lt>_<all|base>_r<refit>[_lam<value>]):
  ctrl_all_r10   shipped config, all_features, refit every 10 (the local twin of the stored
                 subtree_lgbm_all_features table; a forecast already written by
                 experiments/close_trees_datasize.py, arm lgbm_bar1600_w2000, is reused after its
                 configuration is checked and its first refits are re-fitted here bit for bit)
  lt_all_r10     the same with linear_tree=True, linear_lambda 0
  ctrl_base_r10, lt_base_r10   the same pair on the baseline design (HAR + calendar; 17 of its 22
                 columns survive the mask)
  ctrl_all_r1, lt_all_r1       the all_features pair with a refit every session (if CPU allows)
  lt_*_lam<v>    linear_lambda = v (only if lt_* at 0 is unstable)
Tree forecasts start at forecast row TREE_START = 130 (2019-01), as in close_trees_datasize: a
multiple of every cadence used, so every refit anchor of a cadence-10 arm is an anchor of the stored
run, and early enough that the scorer's recalibration (the previous 250 sessions' squared errors,
lagged one session) is full of forecasts on the first trade day (asserted).

SCORER: the master table's research convention (experiments/dense_vs_sparse_1530.py research_frame
-> deck_panel -> point) on the 866 trade days 2020-01-03 .. 2024-04-30: QLIKE of the recalibrated
16:00 forecast and the sign(s) straddle's Sharpe at mid and crossed.  Point estimates only (the
circular block bootstrap is off, commit b761b28).  Differences: Diebold-Mariano on daily QLIKE
(src.evaluation.diebold_mariano.dm_test) and the HAC t of the paired daily P&L difference
(atm_straddle_lib.newey_west_t), the master table's two calls.

STAGES (``python experiments/close_trees_lineartree.py <stage> [arm ...]``)
  smoke [arm ...]   the first SMOKE_ANCHORS refits of each arm (default ctrl_all_r10 lt_all_r10)
                    into _work/smoke/: timing, leaf-coefficient sizes, forecast range
  run [arm ...]     whole arms (default ORDER), one at a time; a finished arm is skipped
  analyze           gate, scores, CSVs and SUMMARY.md (written from the CSVs)
ENV: CLT_OUT (report folder, default below; tests only), CLT_WORK (default <out>/_work),
CLOSE_DESIGN_DIR, CLT_SMOKE_ANCHORS (default 2).

OUTPUTS: results/close_studies_2026-10-03/trees_lineartree/ -- gate.csv, arms.csv, vs_linear.csv,
extremes.csv, leaves.csv, SUMMARY.md; _work/<arm>.npz (forecasts and fit records, never committed).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["SLURM_CPUS_PER_TASK"] = "1"  # the tree spec's N_THREADS

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(REPO)

import dense_vs_sparse_1530 as dvs  # noqa: E402
from src.models.window_mask import window_keep  # noqa: E402

DESIGN = Path(
    os.environ.get("CLOSE_DESIGN_DIR", str(REPO / "results" / "close_design" / "_work"))
)
OUT = Path(os.environ.get("CLT_OUT", str(REPO / "results" / "close_studies_2026-10-03" / "trees_lineartree")))
WORK = Path(os.environ.get("CLT_WORK", str(OUT / "_work")))
SMOKE_ANCHORS = int(os.environ.get("CLT_SMOKE_ANCHORS", "2"))
TRAIN_WIN = 2000  # sessions: the per-bar arms' window (asserted against the cache)
SPEC_REFIT_EVERY = 10  # the tree spec's default REFIT_EVERY (asserted against the spec)
TREE_START = 130  # first tree forecast row (see the module note; = close_trees_datasize.TREE_START)
BUCKET_OF = {"all": "all_features", "base": "baseline"}
FIRST_TRADE_DAY, LAST_TRADE_DAY = "2020-01-03", "2024-04-30"
ORDER = ["ctrl_base_r10", "lt_base_r10", "lt_all_r10", "ctrl_all_r10", "ctrl_all_r1", "lt_all_r1"]
# a forecast of the same configuration written by the data-size study (reused after checks)
DATASIZE_CTRL = (
    REPO / "results" / "close_studies_2026-10-03" / "trees_datasize" / "_work" / "lgbm_bar1600_w2000.npz"
)
REUSE_CHECK_ANCHORS = 2  # refits re-fitted here to confirm a reused forecast bit for bit
STORED = REPO / "results" / "spxw_pnl"
MASTER = REPO / "results" / "close_master_table" / "master_table.csv"
# master-table rows this study is compared with (key -> bucket)
STORED_TREE = {"all": "subtree_lgbm_all_features", "base": "subtree_lgbm_baseline"}
STORED_TREE_DAILY = {"all": "subtree_daily_all_features_lgbm", "base": "subtree_daily_baseline_lgbm"}
STORED_LINEAR = {
    "all": ("sub_lasso_all_features", "sub_ridge_all_features"),
    "base": ("sub_lasso_baseline", "sub_ridge_baseline"),
}
REPRO_TOL = 1e-9  # the master table stores full-precision floats
ARM_RE = re.compile(r"^(ctrl|lt)_(all|base)_r(\d+)(?:_lam([0-9.eE+-]+))?$")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ arms and data
def parse_arm(arm: str) -> dict:
    m = ARM_RE.match(arm)
    if not m:
        raise SystemExit(f"arm {arm!r} is not <ctrl|lt>_<all|base>_r<refit>[_lam<value>]")
    kind, b, k, lam = m.groups()
    if kind == "ctrl" and lam is not None:
        raise SystemExit(f"{arm}: linear_lambda applies to linear-leaf arms only")
    return dict(
        arm=arm,
        linear=kind == "lt",
        b=b,
        bucket=BUCKET_OF[b],
        refit_every=int(k),
        linear_lambda=(float(lam) if lam is not None else 0.0) if kind == "lt" else None,
    )


_DES: dict = {}


def design(bucket: str) -> dict:
    if bucket not in _DES:
        z = np.load(DESIGN / f"design_bar1600_{bucket}.npz", allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["W"] = int(d["W"])
        assert d["W"] == TRAIN_WIN, (d["W"], TRAIN_WIN)
        assert np.isfinite(d["X"]).all() and np.isfinite(d["y"]).all(), "NaN/inf in the design"
        d["n_fc"] = len(d["X"]) - d["W"]
        _DES[bucket] = d
    return _DES[bucket]


def spec_params() -> tuple[dict, int]:
    ns = dvs.spec_ns("tree")
    assert ns["REFIT_EVERY"] == SPEC_REFIT_EVERY and ns["N_THREADS"] == 1
    assert ns["LGBM_PARAMS"]["min_child_samples"] == 8  # the spec's rule at 2000 rows
    return dict(ns["LGBM_PARAMS"]), int(ns["SEED"])


def make_model(cfg: dict):
    """The spec's make_model for LightGBM; a linear-leaf arm adds linear_tree / linear_lambda."""
    import lightgbm as lgb

    p, seed = spec_params()
    if cfg["linear"]:
        p |= {"linear_tree": True, "linear_lambda": cfg["linear_lambda"]}
    return lgb.LGBMRegressor(**p, num_threads=1, random_state=seed, verbosity=-1)


def leaf_stats(model) -> dict:
    """Leaf-model sizes of a fitted linear-leaf booster, from its model string: the largest
    |leaf coefficient| (as stored, i.e. after the learning-rate shrinkage), the share of leaves with
    a linear part, and the mean number of features in a linear leaf."""
    s = model.booster_.model_to_string()
    coefs = [
        float(v)
        for line in s.splitlines()
        if line.startswith("leaf_coeff=")
        for v in line[len("leaf_coeff=") :].split()
    ]
    nfeat = [
        int(v)
        for line in s.splitlines()
        if line.startswith("num_features=")
        for v in line[len("num_features=") :].split()
    ]
    nf = np.asarray(nfeat, dtype=float)
    return dict(
        coef_max=float(np.max(np.abs(coefs))) if coefs else 0.0,
        share_linear=float(np.mean(nf > 0)) if len(nf) else 0.0,
        mean_feat=float(nf[nf > 0].mean()) if (nf > 0).any() else 0.0,
    )


def walk(cfg: dict, anchors: list[int]) -> dict:
    """Refit at each anchor (a forecast-row index) on the W rows before it; predict the next
    refit_every rows with the model in force."""
    d = design(cfg["bucket"])
    X, y, W, n_fc = d["X"], d["y"], d["W"], d["n_fc"]
    k_every = cfg["refit_every"]
    pred = np.full(n_fc, np.nan)
    rec: dict[str, list] = {k: [] for k in ("fit_sec", "kept_n", "coef_max", "share_linear", "mean_feat", "fit_lo", "fit_hi")}
    t0 = time.time()
    for a, j in enumerate(anchors):
        r = W + j  # series row of the anchor's forecast
        Xw, yw = X[r - W : r], y[r - W : r]
        keep = window_keep(Xw)
        k = min(k_every, n_fc - j)
        m = make_model(cfg)
        s = time.time()
        m.fit(Xw[:, keep], yw)
        rec["fit_sec"].append(time.time() - s)
        pred[j : j + k] = m.predict(X[r : r + k][:, keep])
        rec["kept_n"].append(len(keep))
        ls = leaf_stats(m) if cfg["linear"] else dict(coef_max=0.0, share_linear=0.0, mean_feat=0.0)
        for kk, v in ls.items():
            rec[kk].append(v)
        fit = m.predict(Xw[:, keep])  # in-window fitted range (extrapolation check)
        rec["fit_lo"].append(float(fit.min()))
        rec["fit_hi"].append(float(fit.max()))
        if (a + 1) % 25 == 0 or a + 1 == len(anchors):
            say(
                f"  {cfg['arm']}: refit {a + 1}/{len(anchors)}  fit {np.mean(rec['fit_sec']):.1f}s  "
                f"elapsed {time.time() - t0:.0f}s"
            )
    return dict(pred=pred, anchors=np.asarray(anchors), **{k: np.asarray(v) for k, v in rec.items()})


def anchors_of(cfg: dict) -> list[int]:
    d = design(cfg["bucket"])
    assert TREE_START % cfg["refit_every"] == 0
    return list(range(TREE_START, d["n_fc"], cfg["refit_every"]))


def versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- report what is installed
            out[mod] = "absent"
    return out


def save_arm(path: Path, cfg: dict, out: dict, wall: float, cpu: float, extra: dict | None = None) -> None:
    p, seed = spec_params()
    if cfg["linear"]:
        p |= {"linear_tree": True, "linear_lambda": cfg["linear_lambda"]}
    meta = dict(
        arm=cfg["arm"],
        bucket=cfg["bucket"],
        refit_every=cfg["refit_every"],
        linear_tree=cfg["linear"],
        linear_lambda=cfg["linear_lambda"],
        tree_start=TREE_START,
        train_win=TRAIN_WIN,
        window_mask=1,
        params=p,
        seed=seed,
        threads=1,
        wall_sec=wall,
        cpu_sec=cpu,
        versions=versions(),
        **(extra or {}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out, meta=json.dumps(meta))


# ============================================================================ reuse of the data-size control
def reuse_datasize_control(cfg: dict) -> bool:
    """Write _work/ctrl_all_r10.npz from the data-size study's lgbm_bar1600_w2000 forecast if its
    configuration is this arm's and its first REUSE_CHECK_ANCHORS refits re-fit here bit for bit."""
    if cfg["arm"] != "ctrl_all_r10" or not DATASIZE_CTRL.is_file():
        return False
    z = np.load(DATASIZE_CTRL, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    p, _ = spec_params()
    d = design(cfg["bucket"])
    want = dict(
        model="lgbm", source="bar1600", window=TRAIN_WIN, bucket="all_features", refit_every=10,
        tree_start=TREE_START, chunk="", max_anchors=0,
    )
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if meta.get("params") != p:
        bad["params"] = (meta.get("params"), p)
    if meta.get("versions", {}).get("lightgbm") != versions()["lightgbm"]:
        bad["lightgbm"] = (meta.get("versions", {}).get("lightgbm"), versions()["lightgbm"])
    anchors = anchors_of(cfg)
    if not np.array_equal(z["anchors"], anchors):
        bad["anchors"] = "differ"
    pred = np.asarray(z["pred"], float)
    if len(pred) != d["n_fc"] or not np.isfinite(pred[TREE_START:]).all() or np.isfinite(pred[:TREE_START]).any():
        bad["pred"] = "rows"
    if bad:
        say(f"data-size control not reused: {bad}")
        return False
    chk = walk(cfg, anchors[:REUSE_CHECK_ANCHORS])
    rows = slice(TREE_START, TREE_START + REUSE_CHECK_ANCHORS * cfg["refit_every"])
    same = np.array_equal(chk["pred"][rows], pred[rows])
    if not same:
        say("data-size control not reused: the re-fitted refits differ")
        return False
    out = dict(
        pred=pred,
        anchors=np.asarray(z["anchors"]),
        fit_sec=np.asarray(z["fit_sec"]),
        kept_n=np.asarray(z["kept_n"]),
    )
    save_arm(
        WORK / f"{cfg['arm']}.npz", cfg, out, float(meta["wall_sec"]), float(meta["cpu_sec"]),
        extra=dict(
            reused_from=str(DATASIZE_CTRL.relative_to(REPO)),
            reuse_check=f"first {REUSE_CHECK_ANCHORS} refits re-fitted here: identical forecasts",
        ),
    )
    say(f"{cfg['arm']}: reused {DATASIZE_CTRL.relative_to(REPO)} (checked)")
    return True


# ============================================================================ stages
def stage_run(arms: list[str], smoke: bool = False) -> None:
    for arm in arms:
        cfg = parse_arm(arm)
        dest = (WORK / "smoke" if smoke else WORK) / f"{arm}.npz"
        if dest.is_file():
            say(f"{arm}: done already ({dest})")
            continue
        if not smoke and reuse_datasize_control(cfg):
            continue
        anchors = anchors_of(cfg)
        if smoke:
            anchors = anchors[:SMOKE_ANCHORS]
        say(f"{arm}: bucket={cfg['bucket']} refit_every={cfg['refit_every']} linear_tree={cfg['linear']} "
            f"linear_lambda={cfg['linear_lambda']} refits={len(anchors)}")
        t0, c0 = time.time(), time.process_time()
        out = walk(cfg, anchors)
        wall, cpu = time.time() - t0, time.process_time() - c0
        save_arm(dest, cfg, out, wall, cpu, extra=dict(smoke=smoke))
        f = out["pred"][np.isfinite(out["pred"])]
        say(
            f"{arm}: wrote {dest}  wall {wall:.0f}s cpu {cpu:.0f}s  fit {out['fit_sec'].mean():.2f}s/refit  "
            f"forecast {f.min():.4f}..{f.max():.4f} (n<0: {(f < 0).sum()})  "
            f"max|leaf coef| {out['coef_max'].max():.3g}"
        )


def load_arm(arm: str) -> dict | None:
    f = WORK / f"{arm}.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def stored_pred(key: str, dates: np.ndarray) -> np.ndarray:
    """A stored table's 16:00 forecasts on the given bar-end stamps."""
    d = pd.read_parquet(STORED / f"yhat_{key}.parquet")
    t = pd.to_datetime(d["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    s = pd.Series(d["yhat"].to_numpy(float), index=pd.DatetimeIndex(t))
    s = s[s.index.strftime("%H:%M") == "16:00"]
    out = s.reindex(pd.DatetimeIndex(pd.to_datetime(dates))).to_numpy(float)
    assert np.isfinite(out).all(), key
    return out


def score(bucket: str, preds: dict[str, np.ndarray], rows: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Research scorer on forecast rows `rows` of each series; point table + deck panel."""
    d = design(bucket)
    W = d["W"]
    dz = dict(date=d["date"][W:][rows], true_adj=d["y"][W:][rows], true_raw=d["true_raw"][rows])
    names = list(preds)
    frames = [dvs.research_frame(dz, np.asarray(preds[k], float)[rows]) for k in names]
    dk = dvs.deck_frame()
    dk = dk[(dk.index >= FIRST_TRADE_DAY) & (dk.index <= LAST_TRADE_DAY)]
    P = dvs.deck_panel(frames, dk)
    pt = dvs.point(P)
    tab = pd.DataFrame(
        {
            "series": names,
            "n_days": P["pnl"].shape[0],
            "qlike": pt["ql"],
            "sharpe_mid": pt["sh"],
            "sharpe_crossed": pt["shx"],
            "pct_buy": P["buy"] * 100,
        }
    )
    P["names"] = names
    return tab, P


def compare(P: dict, a: str, b: str) -> dict:
    """a minus b: DM on daily QLIKE, HAC t on the paired daily P&L (mid and crossed)."""
    import atm_straddle_lib as asl
    from src.evaluation.diebold_mariano import dm_test

    ia, ib = P["names"].index(a), P["names"].index(b)
    dm = dm_test(P["ql"][:, ia], P["ql"][:, ib])
    t_mid, lag = asl.newey_west_t(P["pnl"][:, ia] - P["pnl"][:, ib])
    t_x, _ = asl.newey_west_t(P["pnlx"][:, ia] - P["pnlx"][:, ib])
    return dict(
        d_qlike=float(P["ql"][:, ia].mean() - P["ql"][:, ib].mean()),
        dm=dm["dm"],
        dm_p=dm["p"],
        dm_hac_lag=dm.get("hac_lag", np.nan),
        t_hac_mid=t_mid,
        t_hac_crossed=t_x,
        hac_lag=lag,
    )


def stage_analyze() -> None:  # noqa: C901 - one linear report
    import score_linear_subsection_causal as slc

    OUT.mkdir(parents=True, exist_ok=True)
    mt = pd.read_csv(MASTER).set_index("key")
    gate_rows: list[dict] = []
    arm_rows: list[dict] = []
    lin_rows: list[dict] = []
    ext_rows: list[dict] = []
    leaf_rows: list[dict] = []
    arms = sorted({p.stem for p in WORK.glob("*.npz")}, key=lambda a: (ORDER.index(a) if a in ORDER else 99, a))
    for b in ("all", "base"):
        bucket = BUCKET_OF[b]
        d = design(bucket)
        W, n_fc = d["W"], d["n_fc"]
        fc_dates = pd.DatetimeIndex(pd.to_datetime(d["date"][W:]))
        first_trade = int(fc_dates.normalize().get_indexer([pd.Timestamp(FIRST_TRADE_DAY)])[0])
        assert first_trade - TREE_START >= slc.SMEAR_W, (first_trade, TREE_START)
        sub = np.arange(TREE_START, n_fc)
        allrows = np.arange(n_fc)
        # ---- gate: stored tables re-scored against the master table (all rows, and rows TREE_START..)
        stored_keys = [STORED_TREE[b], STORED_TREE_DAILY[b], *STORED_LINEAR[b]]
        sp = {k: stored_pred(k, d["date"][W:]) for k in stored_keys}
        tab_all, _ = score(bucket, sp, allrows)
        tab_sub, P_sub_stored = score(bucket, sp, sub)
        for (_, ra), (_, rs) in zip(tab_all.iterrows(), tab_sub.iterrows()):
            m = mt.loc[ra["series"]]
            for col, mcol in (("qlike", "qlike_recal"), ("sharpe_mid", "Sharpe_mid"), ("sharpe_crossed", "Sharpe_crossed")):
                for rows_txt, val in (("all 1469 forecast rows", ra[col]), (f"forecast rows {TREE_START}..", rs[col])):
                    gate_rows.append(
                        dict(
                            gate=f"stored table re-scored = master table ({rows_txt})",
                            item=f"{ra['series']} {col}",
                            value=val,
                            reference=float(m[mcol]),
                            abs_diff=abs(val - float(m[mcol])),
                            ok=abs(val - float(m[mcol])) < REPRO_TOL,
                        )
                    )
        # ---- the arms of this bucket
        mine = [a for a in arms if parse_arm(a)["b"] == b]
        runs = {a: load_arm(a) for a in mine}
        preds = {a: runs[a]["pred"] for a in mine}
        for a in mine:
            assert np.isfinite(preds[a][sub]).all(), a
        allp = {**preds, **sp}
        tab, P = score(bucket, allp, sub)
        tabd = tab.set_index("series")
        ctrl_of = {a: f"ctrl_{b}_r{parse_arm(a)['refit_every']}" for a in mine}
        # local control vs stored (same cadence)
        for a in mine:
            cfg = parse_arm(a)
            if cfg["linear"]:
                continue
            key = STORED_TREE[b] if cfg["refit_every"] == 10 else STORED_TREE_DAILY[b] if cfg["refit_every"] == 1 else None
            if key is None:
                continue
            c = compare(P, a, key)
            diff = np.abs(preds[a][sub] - sp[key][sub])
            for item, val in (
                ("largest absolute forecast difference (adjusted scale)", float(diff.max())),
                ("median absolute forecast difference (adjusted scale)", float(np.median(diff))),
                ("QLIKE local - stored", float(tabd.loc[a, "qlike"] - tabd.loc[key, "qlike"])),
                ("Sharpe mid local - stored", float(tabd.loc[a, "sharpe_mid"] - tabd.loc[key, "sharpe_mid"])),
                ("Sharpe crossed local - stored", float(tabd.loc[a, "sharpe_crossed"] - tabd.loc[key, "sharpe_crossed"])),
                ("DM (daily QLIKE)", c["dm"]),
                ("HAC t (daily P&L mid)", c["t_hac_mid"]),
            ):
                gate_rows.append(
                    dict(
                        gate=f"local control (LightGBM {versions()['lightgbm']}) vs stored {key} (cluster run)",
                        item=f"{a}: {item}",
                        value=val,
                        reference=np.nan,
                        abs_diff=np.nan,
                        ok=True,
                    )
                )
        for a in mine:
            cfg = parse_arm(a)
            r = runs[a]
            meta = r["meta"]
            row = dict(
                arm=a,
                bucket=bucket,
                refit_every=cfg["refit_every"],
                linear_tree=cfg["linear"],
                linear_lambda=cfg["linear_lambda"],
                n_days=int(tabd.loc[a, "n_days"]),
                qlike=float(tabd.loc[a, "qlike"]),
                sharpe_mid=float(tabd.loc[a, "sharpe_mid"]),
                sharpe_crossed=float(tabd.loc[a, "sharpe_crossed"]),
                pct_buy=float(tabd.loc[a, "pct_buy"]),
                control=ctrl_of[a],
            )
            ctrl = ctrl_of[a]
            if cfg["linear"] and ctrl in preds:
                c = compare(P, a, ctrl)
                row |= dict(
                    d_qlike_vs_control=c["d_qlike"],
                    dm_vs_control=c["dm"],
                    dm_p_vs_control=c["dm_p"],
                    d_sharpe_mid_vs_control=row["sharpe_mid"] - float(tabd.loc[ctrl, "sharpe_mid"]),
                    t_hac_mid_vs_control=c["t_hac_mid"],
                    d_sharpe_crossed_vs_control=row["sharpe_crossed"] - float(tabd.loc[ctrl, "sharpe_crossed"]),
                    t_hac_crossed_vs_control=c["t_hac_crossed"],
                    hac_lag=c["hac_lag"],
                    dm_hac_lag=c["dm_hac_lag"],
                )
            row |= dict(
                refits=len(r["fit_sec"]),
                fit_sec_mean=float(np.mean(r["fit_sec"])),
                fit_sec_total=float(np.sum(r["fit_sec"])),
                wall_sec=float(meta["wall_sec"]),
                cpu_sec=float(meta["cpu_sec"]),
                kept_min=int(np.min(r["kept_n"])),
                kept_max=int(np.max(r["kept_n"])),
                lightgbm=meta["versions"]["lightgbm"],
                reused_from=meta.get("reused_from", ""),
            )
            arm_rows.append(row)
            # forecast extremes (adjusted scale = the model's target scale, sqrt of diurnally
            # adjusted variance; the scorer squares it, so a negative forecast loses its sign)
            f = preds[a][sub]
            # the target range of each forecast's own training window (rows [j, j + W) of the series
            # for the refit at forecast row j): a forecast outside it extrapolates
            k_every = cfg["refit_every"]
            anchor = TREE_START + (sub - TREE_START) // k_every * k_every
            ys = pd.Series(d["y"])
            wmax = ys.rolling(W).max().to_numpy()[anchor + W - 1]
            wmin = ys.rolling(W).min().to_numpy()[anchor + W - 1]
            ext = dict(
                arm=a,
                min_forecast=float(f.min()),
                max_forecast=float(f.max()),
                max_abs_forecast=float(np.abs(f).max()),
                date_max_abs=str(fc_dates[TREE_START + int(np.argmax(np.abs(f)))].date()),
                n_negative=int((f < 0).sum()),
                n_forecast_rows=len(f),
                q001=float(np.quantile(f, 0.001)),
                q999=float(np.quantile(f, 0.999)),
                n_above_window_max=int((f > wmax).sum()),
                n_below_window_min=int((f < wmin).sum()),
                largest_window_max=float(wmax.max()),
                max_pred_clock=float(
                    dvs.research_frame(
                        dict(date=d["date"][W:][sub], true_adj=d["y"][W:][sub], true_raw=d["true_raw"][sub]), preds[a][sub]
                    )["pred_clock"].max()
                ),
            )
            if cfg["linear"] and ctrl in preds:
                dd = np.abs(preds[a][sub] - preds[ctrl][sub])
                ext |= dict(max_abs_diff_vs_control=float(dd.max()), median_abs_diff_vs_control=float(np.median(dd)))
            ext_rows.append(ext)
            if cfg["linear"]:
                leaf_rows.append(
                    dict(
                        arm=a,
                        refits=len(r["coef_max"]),
                        max_abs_leaf_coef_max=float(np.max(r["coef_max"])),
                        max_abs_leaf_coef_median=float(np.median(r["coef_max"])),
                        share_linear_leaves_mean=float(np.mean(r["share_linear"])),
                        mean_features_in_linear_leaf=float(np.mean(r["mean_feat"])),
                        in_window_fit_min=float(np.min(r["fit_lo"])),
                        in_window_fit_max=float(np.max(r["fit_hi"])),
                    )
                )
            # against the master table's linear rows (stored forecasts, same days)
            for lk in STORED_LINEAR[b]:
                c = compare(P, a, lk)
                lin_rows.append(
                    dict(
                        arm=a,
                        linear_row=lk,
                        qlike_arm=row["qlike"],
                        qlike_linear=float(tabd.loc[lk, "qlike"]),
                        d_qlike=c["d_qlike"],
                        dm=c["dm"],
                        dm_p=c["dm_p"],
                        sharpe_mid_arm=row["sharpe_mid"],
                        sharpe_mid_linear=float(tabd.loc[lk, "sharpe_mid"]),
                        d_sharpe_mid=row["sharpe_mid"] - float(tabd.loc[lk, "sharpe_mid"]),
                        t_hac_mid=c["t_hac_mid"],
                    )
                )
        # stored reference rows for the report (scored on rows TREE_START.., = all rows on trade days)
        for k in stored_keys:
            arm_rows.append(
                dict(
                    arm=f"stored:{k}",
                    bucket=bucket,
                    n_days=int(tabd.loc[k, "n_days"]),
                    qlike=float(tabd.loc[k, "qlike"]),
                    sharpe_mid=float(tabd.loc[k, "sharpe_mid"]),
                    sharpe_crossed=float(tabd.loc[k, "sharpe_crossed"]),
                    pct_buy=float(tabd.loc[k, "pct_buy"]),
                )
            )
    pd.DataFrame(gate_rows).to_csv(OUT / "gate.csv", index=False)
    pd.DataFrame(arm_rows).to_csv(OUT / "arms.csv", index=False)
    pd.DataFrame(lin_rows).to_csv(OUT / "vs_linear.csv", index=False)
    pd.DataFrame(ext_rows).to_csv(OUT / "extremes.csv", index=False)
    pd.DataFrame(leaf_rows).to_csv(OUT / "leaves.csv", index=False)
    write_summary()
    g = pd.DataFrame(gate_rows)
    if not g["ok"].all():
        print(g[~g["ok"]].to_string(index=False))
        raise SystemExit("GATE FAILED")
    say(f"wrote {OUT}/gate.csv arms.csv vs_linear.csv extremes.csv leaves.csv SUMMARY.md")


# ============================================================================ summary
def _f(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{nd}f}"


def _table(df: pd.DataFrame, cols: list[tuple[str, str, int]]) -> list[str]:
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c, _, nd in cols:
            v = r.get(c, np.nan)
            cells.append(str(v) if isinstance(v, str) else _f(v, nd))
        body.append("| " + " | ".join(cells) + " |")
    return [head, sep, *body]


def write_summary() -> None:  # noqa: C901 - one linear report
    g = pd.read_csv(OUT / "gate.csv")
    A = pd.read_csv(OUT / "arms.csv")
    L = pd.read_csv(OUT / "vs_linear.csv")
    E = pd.read_csv(OUT / "extremes.csv")
    F = pd.read_csv(OUT / "leaves.csv") if (OUT / "leaves.csv").stat().st_size > 1 else pd.DataFrame()
    local = A[~A["arm"].str.startswith("stored:")].copy()
    local = local.assign(_o=local["bucket"].map({"all_features": 0, "baseline": 1}))
    local = local.sort_values(["_o", "refit_every", "linear_tree", "linear_lambda"], na_position="first")
    stored = A[A["arm"].str.startswith("stored:")].copy()
    lines = [
        "# Linear leaves for the 16:00 LightGBM (`linear_tree`)",
        "",
        "Written by `experiments/close_trees_lineartree.py analyze` from the CSVs in this folder.",
        "",
        "**Question (the user's):** do the 16:00-bar trees lose to the linear models because they approximate a "
        "mostly linear signal (the HAR ladder) with step functions? LightGBM's `linear_tree=True` fits, in every "
        "leaf, a linear model in the numerical features used on that leaf's branch instead of a constant (the first "
        "tree keeps constant leaves). One setting changed; everything else is the shipped configuration "
        "(`specs/causal_tune_trees.py` LGBM_PARAMS, `min_child_samples` 8 for the 2000-row window, seed 42, one "
        "thread), on the captured 16:00-bar design, window 2000 sessions, the per-window column mask on, a refit "
        f"every 10 sessions (or every session where marked), forecasts from forecast row {TREE_START} (2019-01) on.",
        "",
        "**Scorer:** the master table's research convention (`dense_vs_sparse_1530.research_frame` -> `deck_panel` "
        "-> `point`) on the 866 trade days 2020-01-03 .. 2024-04-30: QLIKE of the recalibrated 16:00 forecast, sign(s) "
        "straddle Sharpe at mid and crossed. Point estimates (the circular block bootstrap is off, commit b761b28). "
        "Differences against the local control of the same design and cadence: Diebold-Mariano on daily QLIKE "
        "(negative = linear leaves have lower loss) and the HAC t of the paired daily P&L difference (positive = "
        "linear leaves earn more).",
        "",
        "## Gate",
        "",
    ]
    rep = g[g["gate"].str.startswith("stored table")]
    lines.append(
        f"Stored tables re-scored here against `results/close_master_table/master_table.csv`: "
        f"{int(rep['ok'].sum())} of {len(rep)} checks within {REPRO_TOL:g} (largest difference "
        f"{rep['abs_diff'].max():.2e}), on all 1469 forecast rows and on rows {TREE_START}.. alike."
    )
    lines.append("")
    loc = g[g["gate"].str.startswith("local control")]
    if len(loc):
        lines.append(
            "Local controls against the stored cluster runs (LightGBM 4.6.0 on the cluster, "
            "writeup/CAMPAIGN_16H_2026-09-29.md; same configuration otherwise):"
        )
        lines.append("")
        lines += _table(loc.rename(columns={"item": "item_"}).assign(item_=loc["item"]), [("gate", "comparison", 0), ("item_", "item", 0), ("value", "value", 5)])
        lines.append("")
    lines += ["## Control vs linear leaves", ""]
    lines += _table(
        local,
        [
            ("arm", "arm", 0),
            ("bucket", "design", 0),
            ("refit_every", "refit every", 0),
            ("linear_lambda", "linear_lambda", 3),
            ("qlike", "QLIKE", 4),
            ("sharpe_mid", "Sharpe mid", 2),
            ("sharpe_crossed", "Sharpe crossed", 2),
            ("pct_buy", "% buy", 1),
            ("dm_vs_control", "DM vs control", 2),
            ("dm_p_vs_control", "DM p", 3),
            ("t_hac_mid_vs_control", "HAC t mid vs control", 2),
            ("t_hac_crossed_vs_control", "HAC t crossed vs control", 2),
            ("fit_sec_mean", "fit s / refit", 2),
            ("cpu_sec", "CPU s", 0),
        ],
    )
    lines.append("")
    if local["reused_from"].fillna("").str.len().gt(0).any():
        for _, r in local[local["reused_from"].fillna("").str.len().gt(0)].iterrows():
            lines.append(
                f"`{r['arm']}` is the forecast written by `experiments/close_trees_datasize.py` ({r['reused_from']}): "
                "same configuration, checked field by field, and its first refits re-fitted here gave identical forecasts."
            )
        lines.append("")
    lines += ["## Master-table rows on the same days", ""]
    lines += _table(
        stored.assign(arm=stored["arm"].str.replace("stored:", "", regex=False)),
        [("arm", "stored forecast", 0), ("bucket", "design", 0), ("qlike", "QLIKE", 4), ("sharpe_mid", "Sharpe mid", 2), ("sharpe_crossed", "Sharpe crossed", 2)],
    )
    lines += ["", "## Against the linear rows (each arm minus the stored linear forecast)", ""]
    lines += _table(
        L,
        [
            ("arm", "arm", 0),
            ("linear_row", "linear row", 0),
            ("d_qlike", "QLIKE diff", 4),
            ("dm", "DM", 2),
            ("dm_p", "DM p", 3),
            ("d_sharpe_mid", "Sharpe mid diff", 2),
            ("t_hac_mid", "HAC t mid", 2),
        ],
    )
    lines += ["", "## Forecast extremes (adjusted scale: the model's target, before the scorer squares it)", ""]
    lines += _table(
        E,
        [
            ("arm", "arm", 0),
            ("min_forecast", "min", 4),
            ("max_forecast", "max", 4),
            ("max_abs_forecast", "largest abs forecast", 4),
            ("date_max_abs", "its date", 0),
            ("n_negative", "n < 0", 0),
            ("n_above_window_max", "n > max target of own window", 0),
            ("n_below_window_min", "n < min target of own window", 0),
            ("max_abs_diff_vs_control", "largest abs diff vs control", 4),
            ("median_abs_diff_vs_control", "median abs diff vs control", 4),
        ],
    )
    lines.append("")
    if len(F):
        lines += ["## Leaf models (linear-leaf arms)", ""]
        lines += _table(
            F,
            [
                ("arm", "arm", 0),
                ("refits", "refits", 0),
                ("max_abs_leaf_coef_max", "largest abs leaf coefficient (any refit)", 4),
                ("max_abs_leaf_coef_median", "median over refits of the largest abs leaf coefficient", 4),
                ("share_linear_leaves_mean", "share of leaves with a linear part", 3),
                ("mean_features_in_linear_leaf", "features in a linear leaf (mean)", 2),
                ("in_window_fit_min", "in-window fit min", 4),
                ("in_window_fit_max", "in-window fit max", 4),
            ],
        )
        lines.append("")
        lr = spec_params()[0]["learning_rate"]
        lines.append(
            "Leaf coefficients are as stored in the booster, i.e. after the learning-rate shrinkage "
            f"(learning_rate {lr:.5f}); the design columns are robust-scaled."
        )
        lines.append("")
    tot_cpu = float(local["cpu_sec"].sum())
    lines += [
        "## Compute",
        "",
        f"CPU time of the local arms: {tot_cpu / 3600:.2f} h in all (reused arms count the data-size run's CPU), "
        "one process, single-threaded (num_threads 1; OMP / OpenBLAS / MKL threads 1).",
        "",
        "## Files",
        "",
        "- `gate.csv`: stored tables re-scored against the master table; local controls against the stored runs.",
        "- `arms.csv`: every arm and the stored reference rows: QLIKE, Sharpe mid / crossed, DM and HAC t against "
        "the local control, fit seconds, CPU seconds.",
        "- `vs_linear.csv`: every arm against the master table's linear rows of its design.",
        "- `extremes.csv`: forecast range, negative forecasts, the largest recalibrated forecast (`max_pred_clock`).",
        "- `leaves.csv`: leaf-coefficient sizes and in-window fitted range of the linear-leaf arms.",
        "- `_work/<arm>.npz`: forecasts and refit records (not committed).",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ main
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("smoke", "run", "analyze"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "smoke":
        stage_run(sys.argv[2:] or ["ctrl_all_r10", "lt_all_r10"], smoke=True)
    elif sys.argv[1] == "run":
        stage_run(sys.argv[2:] or ORDER)
    else:
        stage_analyze()
