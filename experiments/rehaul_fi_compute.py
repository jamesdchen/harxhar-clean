"""Feature importance of the paper's per-bar 16:00 models of record, by the measures of the rehaul plan (2026-10-15).

User plan (handwritten notes, transcribed in the task):
  * lasso: aggregate = how often a coefficient is nonzero; time-varying = the magnitudes and the on / off pattern;
  * ridge: aggregate = the sum of |coefficients| (the design is robust-prescaled); plus the magnitudes over time;
  * trees: rank the series at every refit by mean |SHAP| on that refit's held-out rows, then aggregate the
    per-refit rankings into a global one (mean rank; Borda count as a check; Kendall's W for stability);
  * interactions: global = mean |SHAP interaction value| over the scored sample (Lundberg et al. 2020), with
    Friedman-Popescu's H statistic as a cross-check on the top pairs; time-varying = the same interaction values per
    refit on that refit's held-out rows, ranked per refit and aggregated as the main effects are.

Models (the paper's per-bar models of record, all features, p = 628 columns in 49 units = 47 exogenous series in 8
buckets + the HAR ladder + the calendar block):
  * ridge and lasso PRE-TUNED with the block penalty (pf = 0.01 on the HAR + calendar backbone, 1 elsewhere):
    alpha 1e3 (ridge) and 1e-2 (lasso), \\cmPtPen{Ridge,Lasso}All; the light margin retune never fired on all
    features (\\cmPtFired{Ridge,Lasso}All = none), so the model is the frozen-alpha walk, re-solved every session
    (experiments/close_linear_pretune.py, walks ridge_bb_a+3 / lasso_bb_a-2).  The coefficient path is re-run here
    through the same C kernel (it returns theta at every session) and gated against the stored walk forecasts.
  * the per-bar trees on their top dev-span entries: LightGBM lgbm_opt_00, XGBoost xgb_opt_00, random forest trial
    255, window-masked, refit every 10 sessions (= the every-session model on those days), exactly as the de-dup
    feature-importance study (experiments/feature_importance_1530_trees.py, imported read-only).

Refits: 147, one every 10 sessions; refit k is fitted on the 2,000 sessions before its anchor and scored on its
causal held-out tail T_k (<= 10 sessions), 2018-06-25 .. 2024-04-30 (1,469 sessions).

Reuse: the tree main-effect SHAP of every refit is the stored TreeSHAP of the de-dup study's cluster refits
(results/feature_importance_1530_dedup/_work/trees/, LightGBM / XGBoost native, shap 0.51 for the forest); the noise
probes (every 3rd refit) also come from there.  Recomputed here: the linear coefficient paths (not stored) and every
interaction value (needs the fitted trees, which were not stored): the trees are refitted locally with the same code
and settings; LightGBM reproduces the stored refits bit for bit, XGBoost (local 2.1.4 vs 3.2.0 on the cluster) and
scikit-learn (1.8.0 vs 1.9.0) do not, so their interactions are those of the local refits (gaps recorded).
SHAP interaction values: experiments/rehaul_fi_treeshap.py (numba port of shap's path-dependent TreeSHAP with
conditioning; gated here against XGBoost's native pred_interactions and LightGBM's pred_contrib).

Stages:  python experiments/rehaul_fi_compute.py <stage>
  linear   ridge / lasso coefficient paths + window scales + the linear noise probes      -> results/rehaul_fi/linear.npz
  trees    per (model, refit): local refit, main + interaction SHAP, probe fit             -> _work/trees/*.npz
  hstat    Friedman-Popescu H^2 for each model's top pairs on 15 refits                    -> _work/hstat_*.npz
  reduce   every table the generator reads                                                  -> results/rehaul_fi/*.csv, summary.json
  all      in order
Env: RHFI_WORKERS (default 11) processes for trees / hstat.
"""

from __future__ import annotations

import os

if __name__ == "__main__" or os.environ.get("RHFI_CHILD") == "1":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ["FEATIMP_OUT"] = "results/feature_importance_1530_dedup"
os.environ["FEATIMP_WINDOW_MASK"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(REPO)

OUT = REPO / "results" / "rehaul_fi"
WORK = OUT / "_work"
TREE_PARTS = REPO / "results" / "feature_importance_1530_dedup" / "_work" / "trees"
PRETUNE = REPO / "results" / "linear_pretune_2026-10-06"
OLD_LASSO = (
    REPO / "results" / "feature_importance_1530_dedup" / "_work" / "linear_captures"
)
WORKERS = int(os.environ.get("RHFI_WORKERS", "11"))
MODELS = ("lgbm", "xgb", "rf")
PART_TAG = {"lgbm": "lgbm", "xgb": "xgb", "rf": "rf255"}
LINEAR: dict[str, dict] = {
    "ridge": dict(est="ridge", alpha=1e3, walk="ridge_bb_a+3"),
    "lasso": dict(est="reclasso", alpha=1e-2, walk="lasso_bb_a-2"),
}
R_BB = 0.01
BUCKET_ORDER = (
    "backbone",
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "sentiment",
    "implied_vol",
    "vol_demand",
    "fomc",
)
# 2026-10-18 (user, new-notes 22): the feature-importance out-of-sample sample starts on 3 January 2020, the first
# session after the lambda validation folds (6 Jan 2016 .. 2 Jan 2020), so no FI day overlaps them. reduce() keeps the
# refits whose tail reaches this date (refit 38 on, anchored 2019-12-31), drops refit 38's tail rows before it, and reads
# the linear coefficients at max(anchor, start). The linear / trees / hstat stages still compute all 147 refits.
# Was: no start restriction -- every forecast row from the design's first OOS row, 2018-06-25 (147 refits, 1,469 sessions).
FI_OOS_START = "2020-01-03"
H_REFITS = tuple(range(0, 147, 10))  # 15 refits for the H statistic (reduce() uses those inside the FI sample)
H_ROWS = 100  # evaluation = background points per refit: the 100 sessions from the refit's anchor on
H_TOP_SERIES, H_TOP_COLS = 10, 5
HDIR = "hstat_v2"  # v2 (2026-10-15) also stores sum PD_B^2 of the partner (hstat/ = v1, H^2 sums only)
TOPK_W_PAIRS = (
    50  # Kendall's W of the pair rankings over each model's global top-50 pairs
)
NZ_TOL = 0.0  # a lasso coefficient counts as nonzero if |beta| > NZ_TOL (the homotopy zeroes exactly)


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "log.txt", "a", encoding="utf-8") as h:
        h.write(line + "\n")


# ============================================================================ shared inputs
_D: dict = {}


def data() -> dict:
    """The de-dup study's input file (design of record, 628 columns) + series / bucket maps."""
    if not _D:
        import feature_importance_1530_trees as FT

        d = FT.load_input("all_features")
        from src.data import loading as LD

        series = [str(s) for s in d["series"]]
        order = []
        for s in series:
            if s not in order:
                order.append(s)
        bmap = {"har_ma": "backbone", "calendar": "backbone"}
        for b in BUCKET_ORDER[1:]:
            for s in LD.SUBGROUPS[b]:
                assert s not in bmap, s
                bmap[s] = b
        assert set(order) == set(bmap), set(order) ^ set(bmap)
        units = sorted(
            order, key=lambda s: (BUCKET_ORDER.index(bmap[s]), order.index(s))
        )
        assert len(units) == 49 and sum(bmap[s] != "backbone" for s in units) == 47
        sidx = np.array([units.index(s) for s in series])
        G = np.zeros((len(series), len(units)))
        G[np.arange(len(series)), sidx] = 1.0
        d.update(series=series, units=units, bmap=bmap, sidx=sidx, G=G, FT=FT)
        _D.update(d)
    return _D


# ============================================================================ linear
def linear() -> None:
    import close_linear_pretune as CLP

    D = data()
    A = CLP.arrays("all_features")
    Xa, y, W = A["bar_Xa"], A["bar_y"], A["W"]
    assert np.abs(Xa[:, :-1] - D["X"]).max() < 1e-9 and np.array_equal(y, D["y"]), (
        "pre-tune arrays = FI design"
    )
    meta = json.loads((CLP.ARR / "meta.json").read_text())["all_features"]
    assert meta["names"] == list(D["names"])
    n_test = len(Xa) - W
    bl = CLP.blocks_of(n_test)
    m = Xa.shape[1]
    lk = np.zeros((len(bl), 1, m), np.uint8)
    mk = np.zeros((len(bl), 1, m), np.uint8)
    pf = np.ones((len(bl), 1, m))
    for k, (i0, _) in enumerate(bl):
        lk[k, 0], mk[k, 0], pf[k, 0] = CLP.structure(Xa[i0 : i0 + W], A["bb_aug"], R_BB)
    bstart = np.array([i0 for i0, _ in bl] + [n_test], np.int32)
    out: dict = {}
    gates = []
    for name, spec in LINEAR.items():
        t0 = time.time()
        o = CLP.krun(Xa, y, W, spec["est"], bstart, lk, mk, pf, [spec["alpha"]])
        stored = np.load(
            PRETUNE / "_work" / "walk" / "all_features" / f"{spec['walk']}.npz"
        )
        gap = float(np.abs(o["pred"] - stored["pred"]).max())
        rec = np.einsum("ij,ij->i", Xa[W:], o["theta"])
        gap_th = float(np.abs(rec - o["pred"]).max())
        gates.append(
            dict(
                gate=f"{name}: kernel re-run == stored walk {spec['walk']} (pred, max abs)",
                value=gap,
                ok=gap <= 1e-9,
            )
        )
        gates.append(
            dict(
                gate=f"{name}: x_t' theta_t == pred (max abs)",
                value=gap_th,
                ok=gap_th <= 1e-9,
            )
        )
        assert np.allclose(stored["alpha_blk"], spec["alpha"]), stored["alpha_blk"]
        out[f"{name}_theta"] = o["theta"]
        out[f"{name}_pred"] = o["pred"]
        say(
            f"linear {name}: alpha {spec['alpha']}, walk {time.time() - t0:.0f}s, pred gap vs stored {gap:.1e}, "
            f"theta reproduces pred to {gap_th:.1e}"
        )
    # window scales at the anchors (is sum |beta| comparable across columns?)
    K = 147
    sd = np.zeros((K, m - 1))
    iqr = np.zeros((K, m - 1))
    for k in range(K):
        Xw = D["X"][10 * k : 10 * k + W]
        sd[k] = Xw.std(axis=0)
        q75, q25 = np.percentile(Xw, [75, 25], axis=0)
        iqr[k] = q75 - q25
    out["win_sd"], out["win_iqr"] = sd, iqr
    # linear noise probes at the probe refits (every 3rd, as the trees): three noise columns, robust-scaled on the
    # window like the design's value ladders (median / IQR), exogenous (pf 1); batch solution at the walk's alpha
    # with the block-start structure of the walk (probes neither locked nor masked).
    FT = D["FT"]
    pk = [k for k in range(K) if k % FT.PROBE_EVERY == 0]
    probe_beta = {n: np.zeros((len(pk), 3)) for n in LINEAR}
    probe_theta = {n: np.zeros((len(pk), m + 3)) for n in LINEAR}
    batch_gap = {n: np.zeros(len(pk)) for n in LINEAR}
    for q, k in enumerate(pk):
        i = 10 * k
        rng = np.random.default_rng([FT.PROBE_SEED, k])
        noise = np.column_stack(
            [
                rng.random(W),
                np.floor(rng.random(W) * FT.PROBE_LEVELS),
                (rng.random(W) < 0.5).astype(np.float64),
            ]
        )
        med = np.median(noise, axis=0)
        q75, q25 = np.percentile(noise, [75, 25], axis=0)
        noise = (noise - med) / np.where(q75 > q25, q75 - q25, 1.0)
        b = i // CLP.TUNE_PER
        Xw = np.hstack([Xa[i : i + W, :-1], noise, Xa[i : i + W, -1:]])
        lk_p = np.concatenate([lk[b, 0, :-1], np.zeros(3, np.uint8), lk[b, 0, -1:]])
        mk_p = np.concatenate([mk[b, 0, :-1], np.zeros(3, np.uint8), mk[b, 0, -1:]])
        pf_p = np.concatenate([pf[b, 0, :-1], np.ones(3), pf[b, 0, -1:]])
        for name, spec in LINEAR.items():
            th = CLP.batch_theta(
                np.ascontiguousarray(Xw),
                np.ascontiguousarray(y[i : i + W]),
                spec["est"],
                spec["alpha"],
                (lk_p, mk_p, pf_p),
            )
            probe_beta[name][q] = th[m - 1 : m + 2]
            probe_theta[name][q] = th
            th0 = CLP.batch_theta(
                np.ascontiguousarray(Xa[i : i + W]),
                np.ascontiguousarray(y[i : i + W]),
                spec["est"],
                spec["alpha"],
                (lk[b, 0], mk[b, 0], pf[b, 0]),
            )
            batch_gap[name][q] = float(np.abs(Xa[W + i] @ th0 - out[f"{name}_pred"][i]))
    for name in LINEAR:
        out[f"{name}_probe_beta"] = probe_beta[name]
        out[f"{name}_probe_theta"] = probe_theta[name]
        out[f"{name}_batch_gap"] = batch_gap[name]
        gates.append(
            dict(
                gate=f"{name}: batch solution at the probe anchors == walk forecast (max abs; reported)",
                value=float(batch_gap[name].max()),
                ok=True,
            )
        )
        say(
            f"linear probes {name}: batch vs walk forecast max gap {batch_gap[name].max():.1e}; "
            f"probe |beta| median {np.median(np.abs(probe_beta[name]), axis=0)}"
        )
    out["probe_k"] = np.array(pk)
    # the old study's linear arm (re-tuned one-penalty lasso, spec arm a), for the reconciliation
    oc = np.load(OLD_LASSO / "capture_bar1600_all_features_reclasso.npz")
    out["old_lasso_theta"] = oc["th"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "linear.npz", **out)
    pd.DataFrame(gates).to_csv(WORK / "gates_linear.csv", index=False)


# ============================================================================ trees
def fit_forest(mdl: str, Xtr: np.ndarray, ytr: np.ndarray, p: int):
    import rehaul_fi_treeshap as TS

    FT = data()["FT"]
    m = FT.fit_tree(mdl, 1, Xtr, ytr)
    if mdl == "lgbm":
        F = TS.from_lightgbm(m.booster_, p)
    elif mdl == "xgb":
        F = TS.from_xgboost(m.get_booster(), p)
    else:
        F = TS.from_sklearn_forest(m, p)
    return m, F


def stored_part(mdl: str, k: int) -> dict:
    """The de-dup study's stored refit k of model mdl (parts of 21 refits)."""
    k0 = (k // 21) * 21
    f = (
        TREE_PARTS
        / f"trees_all_features_{PART_TAG[mdl]}_{k0:03d}_{min(k0 + 21, 147):03d}.npz"
    )
    z = np.load(f, allow_pickle=True)
    j = list(z["k"]).index(k)
    rows = np.cumsum([0] + [10 if kk < 146 else 9 for kk in z["k"]])
    a, b = rows[j], rows[j + 1]
    out = dict(
        pred=np.asarray(z["pred"][a:b], np.float64),
        shap=np.asarray(z["shap"][a:b], np.float64),
        kept=z["kept"][j],
    )
    if k in list(z["probe_k"]):
        q = list(z["probe_k"]).index(k)
        out["probe_shap_abs"] = z["probe_shap_abs"][q]
    return out


def tree_task(job: tuple[str, int]) -> dict:
    mdl, k = job
    f = WORK / "trees" / f"{mdl}_{k:03d}.npz"
    if f.is_file():
        return dict(job=job, cached=True)
    import rehaul_fi_treeshap as TS

    D = data()
    FT = D["FT"]
    X, y, W = D["X"], D["y"], D["W"]
    p = X.shape[1]
    G = D["G"]
    i, t, kk = FT.tail_rows(D, k)
    Xtr, ytr, Xt = X[t - W : t], y[t - W : t], X[t : t + kk]
    keep = FT.window_cols(Xtr)
    a = time.time()
    m, F = fit_forest(mdl, FT.sub(Xtr, keep), ytr, len(keep))
    fit_sec = time.time() - a
    Xs = FT.sub(Xt, keep)
    pred = m.predict(Xs)
    Xin = TS.inputs_for(mdl, Xs)
    a = time.time()
    main, inter, ev = TS.tree_shap(F, Xin, interactions=True)
    shap_sec = time.time() - a
    st = stored_part(mdl, k)
    kept = np.zeros(p, bool)
    kept[keep] = True
    assert np.array_equal(kept, st["kept"]), (mdl, k)
    # scatter to the 628 columns
    main_full = np.zeros((kk, p))
    main_full[:, keep] = main
    inter_full = np.zeros((kk, p, p))
    inter_full[np.ix_(np.arange(kk), keep, keep)] = inter
    ser = (
        G.T[None] @ inter_full @ G[None]
    )  # series-level Phi^S_{t,AB} = sum_{j in A, k in B} Phi_{t,jk}
    pure = np.einsum("tjj->tj", inter_full) @ G  # sum_{j in A} Phi_jj
    rec = dict(
        k=k,
        kk=kk,
        kept=kept,
        fit_sec=fit_sec,
        shap_sec=shap_sec,
        ev=ev,
        pred=pred,
        pred_stored=st["pred"],
        forest_pred_gap=float(np.abs(F.predict(Xin) - pred).max()),
        additivity=float(np.abs(main.sum(1) + ev - pred).max()),
        rowsum=float(np.abs(inter.sum(2) - main).max()),
        main=main_full.astype(np.float32),
        main_stored=st["shap"][:, :-1].astype(np.float32),
        inter_ser=ser,
        pure_ser=pure,
        inter_col_absmean=np.abs(inter_full).mean(axis=0).astype(np.float32),
    )
    if mdl == "xgb":
        import xgboost as xgb

        ri = m.get_booster().predict(xgb.DMatrix(Xs), pred_interactions=True)
        rec["xgb_native_gap"] = float(np.abs(ri[:, :-1, :-1] - inter).max())
    if mdl == "lgbm":
        pc = np.asarray(m.predict(Xs, pred_contrib=True))
        rec["lgbm_native_gap"] = float(np.abs(pc[:, :-1] - main).max())
    if k % FT.PROBE_EVERY == 0:
        # the probe fit (the de-dup study's FT.probe: same noise, same mask rule), with interactions
        rng = np.random.default_rng([FT.PROBE_SEED, k])
        n = len(Xtr) + len(Xt)
        noise = np.column_stack(
            [
                rng.random(n),
                np.floor(rng.random(n) * FT.PROBE_LEVELS),
                (rng.random(n) < 0.5).astype(np.float64),
            ]
        )
        Ptr = np.hstack([Xtr, noise[: len(Xtr)]])
        Pt = np.hstack([Xt, noise[len(Xtr) :]])
        pp = Ptr.shape[1]
        pkeep = FT.window_cols(Ptr)
        mp, Fp = fit_forest(mdl, FT.sub(Ptr, pkeep), ytr, len(pkeep))
        pin = TS.inputs_for(mdl, FT.sub(Pt, pkeep))
        pm, pint, _ = TS.tree_shap(Fp, pin, interactions=True)
        pm_full = np.zeros((kk, pp))
        pm_full[:, pkeep] = pm
        pi_full = np.zeros((kk, pp, pp))
        pi_full[np.ix_(np.arange(kk), pkeep, pkeep)] = pint
        rec["probe_main_absmean"] = np.abs(pm_full).mean(axis=0)
        rec["probe_main_stored"] = st["probe_shap_abs"]
        # pair totals 2|Phi_{probe,j}| for every column j, and the probe's series-level pair totals
        rec["probe_pair_col"] = (2 * np.abs(pi_full[:, p:, :])).mean(axis=0)  # (3, 631)
        Gp = np.vstack([G, np.zeros((3, G.shape[1]))])
        rec["probe_pair_ser"] = np.abs(2 * (pi_full[:, p:, :] @ Gp[None])).mean(
            axis=0
        )  # (3, 49)
        # the real pairs of the probe fit (what a probe pair is compared with)
        ser_p = G.T[None] @ pi_full[:, :p, :p] @ G[None]
        rec["probe_fit_inter_ser"] = ser_p
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **{key: np.asarray(v) for key, v in rec.items()})
    return dict(job=job, cached=False, fit_sec=fit_sec, shap_sec=shap_sec)


def _child_init() -> None:
    os.environ["RHFI_CHILD"] = "1"


def run_pool(fn, jobs: list, label: str) -> None:
    import multiprocessing as mp

    t0 = time.time()
    done = 0
    with mp.get_context("spawn").Pool(WORKERS, initializer=_child_init) as pool:
        for res in pool.imap_unordered(fn, jobs):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                say(
                    f"{label}: {done}/{len(jobs)} ({time.time() - t0:.0f}s) last {res.get('job')}"
                )


def trees() -> None:
    jobs = [(m, k) for m in ("rf", "xgb", "lgbm") for k in range(147)]
    todo = [
        j for j in jobs if not (WORK / "trees" / f"{j[0]}_{j[1]:03d}.npz").is_file()
    ]
    say(f"trees: {len(todo)} of {len(jobs)} tasks to run on {WORKERS} workers")
    run_pool(tree_task, todo, "trees")


# ============================================================================ H statistic
def top_pairs() -> dict:
    """Each model's global top series pairs (A != B) and column pairs (j != k) by mean |pair interaction|."""
    D = data()
    U = len(D["units"])
    p = D["X"].shape[1]
    out = {}
    for mdl in MODELS:
        S = np.zeros((U, U))
        C = np.zeros((p, p))
        n = 0
        for k in range(147):
            z = np.load(WORK / "trees" / f"{mdl}_{k:03d}.npz")
            S += np.abs(2 * z["inter_ser"]).sum(axis=0)
            C += z["inter_col_absmean"].astype(np.float64) * int(z["kk"])
            n += int(z["kk"])
        S /= n
        C /= n
        iu = np.triu_indices(U, 1)
        o = np.argsort(-S[iu])[:H_TOP_SERIES]
        jc = np.triu_indices(p, 1)
        cv = 2 * C[jc]
        oc = np.argsort(-cv)[:H_TOP_COLS]
        out[mdl] = dict(
            series=[(int(iu[0][q]), int(iu[1][q])) for q in o],
            cols=[(int(jc[0][q]), int(jc[1][q])) for q in oc],
        )
    return out


def hstat_task(job: tuple[str, int, dict]) -> dict:
    mdl, k, pairs = job
    f = WORK / HDIR / f"{mdl}_{k:03d}.npz"
    if f.is_file():
        return dict(job=(mdl, k), cached=True)
    D = data()
    FT = D["FT"]
    X, y, W = D["X"], D["y"], D["W"]
    i, t, kk = FT.tail_rows(D, k)
    Xtr, ytr = X[t - W : t], y[t - W : t]
    keep = FT.window_cols(Xtr)
    pos = -np.ones(X.shape[1], np.int64)
    pos[keep] = np.arange(len(keep))
    m = FT.fit_tree(mdl, 1, FT.sub(Xtr, keep), ytr)
    E = FT.sub(X[t : t + H_ROWS], keep)  # evaluation = background points
    n = len(E)
    cols_of = {}
    for a in {u for pr in pairs["series"] for u in pr}:
        cols_of[("s", a)] = pos[np.flatnonzero(D["sidx"] == a)]
    for j in {c for pr in pairs["cols"] for c in pr}:
        cols_of[("c", j)] = pos[[j]]
    cols_of = {key: v[v >= 0] for key, v in cols_of.items()}

    def pd_of(cols: np.ndarray) -> np.ndarray:
        if len(cols) == 0:
            return np.zeros(n)
        Z = np.repeat(
            E[None], n, axis=0
        )  # Z[i, b] = background row b with x_i's values on cols
        Z[:, :, cols] = E[:, None, cols]
        v = m.predict(Z.reshape(n * n, -1)).reshape(n, n).mean(axis=1)
        return v - v.mean()

    single = {key: pd_of(c) for key, c in cols_of.items()}
    res = {}
    for kind, prs in (("s", pairs["series"]), ("c", pairs["cols"])):
        num, den, va, vb = [], [], [], []
        for a, b in prs:
            ca, cb = cols_of[(kind, a)], cols_of[(kind, b)]
            joint = pd_of(np.concatenate([ca, cb]))
            num.append(
                float(((joint - single[(kind, a)] - single[(kind, b)]) ** 2).sum())
            )
            den.append(float((joint**2).sum()))
            va.append(float((single[(kind, a)] ** 2).sum()))
            vb.append(float((single[(kind, b)] ** 2).sum()))
        res[f"{kind}_num"], res[f"{kind}_den"] = np.array(num), np.array(den)
        res[f"{kind}_va"], res[f"{kind}_vb"] = np.array(va), np.array(vb)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, n=n, **res)
    return dict(job=(mdl, k), cached=False)


def hstat() -> None:
    tp = top_pairs()
    (WORK / "top_pairs.json").write_text(json.dumps(tp))
    jobs = [(m, k, tp[m]) for m in ("rf", "xgb", "lgbm") for k in H_REFITS]
    todo = [j for j in jobs if not (WORK / HDIR / f"{j[0]}_{j[1]:03d}.npz").is_file()]
    say(f"hstat: {len(todo)} of {len(jobs)} tasks")
    run_pool(hstat_task, todo, "hstat")


# ============================================================================ reduce helpers
def ranks_desc(v: np.ndarray) -> np.ndarray:
    """Rank 1 = largest, ties averaged (scipy rankdata on -v)."""
    from scipy.stats import rankdata

    return rankdata(-np.asarray(v, float), method="average")


def kendall_w(R: np.ndarray) -> float:
    """Kendall's W of an (m raters x n objects) matrix of ranks, with the tie correction."""
    m, n = R.shape
    Rs = R.sum(axis=0)
    S = float(((Rs - Rs.mean()) ** 2).sum())
    T = 0.0
    for row in R:
        _, c = np.unique(row, return_counts=True)
        T += float((c**3 - c).sum())
    return 12.0 * S / (m**2 * (n**3 - n) - m * T)


def borda_top(R: np.ndarray, top: int = 10) -> np.ndarray:
    """Truncated Borda count: top + 1 - rank points for ranks <= top (a full Borda count is an affine function of
    the mean rank, so it cannot disagree with it)."""
    return np.clip(top + 1 - R, 0, None).sum(axis=0)


def rank_table(
    imp: np.ndarray, labels: list[str], extra: dict | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """imp: (refits x units) importance; per-refit ranks, mean rank, truncated Borda, top-5 share, ..."""
    R = np.vstack([ranks_desc(r) for r in imp])
    df = pd.DataFrame(
        dict(
            unit=labels,
            mean_rank=R.mean(axis=0),
            median_rank=np.median(R, axis=0),
            borda10=borda_top(R, 10),
            top5_share=(R <= 5).mean(axis=0),
            top10_share=(R <= 10).mean(axis=0),
            nonzero_share=(imp > 0).mean(axis=0),
            mean_value=imp.mean(axis=0),
        )
    )
    if extra:
        for key, v in extra.items():
            df[key] = v
    df["rank_mean_rank"] = df["mean_rank"].rank(method="min")
    df["rank_borda10"] = (-df["borda10"]).rank(method="min")
    df["rank_mean_value"] = (-df["mean_value"]).rank(method="min")
    return df.sort_values("mean_rank").reset_index(drop=True), R


# ============================================================================ reduce
def reduce() -> None:  # noqa: C901 -- one reducer
    from scipy.stats import spearmanr

    D = data()
    units, bmap, G = D["units"], D["bmap"], D["G"]
    names = list(D["names"])
    # 2026-10-18 (user, new-notes 22): restricted to the FI sample from FI_OOS_START. Was:
    #     U, p, K = len(units), len(names), 147
    #     FT = D["FT"]
    #     anchors = np.array([10 * k for k in range(K)])
    #     dates = pd.to_datetime(pd.Series(D["date_oos"]).str[:10]).to_numpy()
    U, p, K_ALL = len(units), len(names), 147
    FT = D["FT"]
    dates_all = pd.to_datetime(pd.Series(D["date_oos"]).str[:10]).to_numpy()
    i0 = int((dates_all < np.datetime64(FI_OOS_START)).sum())  # first OOS row on/after FI_OOS_START
    KS = np.arange(i0 // 10, K_ALL)  # refits (on-disk index) whose tail reaches FI_OOS_START
    K = len(KS)
    anchors = 10 * KS  # refit anchors (OOS row index)
    r0 = np.maximum(0, i0 - anchors)  # tail rows dropped at the head of each refit (> 0 only for the first)
    lanch = anchors + r0  # first scored row of each refit; the linear coefficients are read here
    rdates = dates_all[lanch]
    dates = dates_all[i0:]
    HK = tuple(k for k in H_REFITS if k >= KS[0])  # H-statistic refits inside the FI sample
    S: dict = dict(
        n_refits=K,
        n_oos=int(len(dates)),
        p=p,
        n_units=U,
        n_exog=47,
        window=int(D["W"]),
        refit_every=int(FT.REFIT_EVERY),
        r_bb=R_BB,
        n_buckets=len(BUCKET_ORDER) - 1,
        old_block=250,
        n_pairs=U * (U - 1) // 2,
        n_probe_cols=p + 3,
        top_w_pairs=TOPK_W_PAIRS,
        probe_levels=int(FT.PROBE_LEVELS),
        first=str(pd.Timestamp(dates[0]).date()),
        last=str(pd.Timestamp(dates[-1]).date()),
    )
    S["fi_oos_start"], S["first_refit"] = FI_OOS_START, int(KS[0])  # 2026-10-18 (user, new-notes 22)
    gates = pd.read_csv(WORK / "gates_linear.csv").to_dict("records")
    ncol = G.sum(axis=0).astype(int)
    base_cols = dict(unit=units, bucket=[bmap[s] for s in units], n_columns=ncol)

    # ------------------------------------------------------------ linear
    Z = np.load(OUT / "linear.npz")
    lin_rank = {}
    for name in LINEAR:
        # 2026-10-18 (user, new-notes 22): was th = Z[f"{name}_theta"][anchors, :-1]
        th = Z[f"{name}_theta"][lanch, :-1]  # (K, p), intercept dropped
        absb = np.abs(th)
        ser_abs = absb @ G  # (K, U): sum_{j in A} |beta_j|
        nz = absb > NZ_TOL
        df_s, R = rank_table(ser_abs, units)
        df_s = df_s.merge(pd.DataFrame(base_cols), on="unit")
        df_s["sum_abs"] = df_s["unit"].map(dict(zip(units, ser_abs.sum(axis=0))))
        df_s["sum_abs_share"] = df_s["sum_abs"] / ser_abs.sum()
        # 2026-10-18 (user, new-notes 22): was sd = Z["win_sd"]
        sd = Z["win_sd"][KS]
        df_s["sum_abs_sd"] = df_s["unit"].map(
            dict(zip(units, ((absb * sd) @ G).sum(axis=0)))
        )
        df_s["refits_any_nonzero"] = df_s["unit"].map(
            dict(zip(units, (nz.astype(float) @ G > 0).sum(axis=0)))
        )
        df_s["mean_active_cols"] = df_s["unit"].map(
            dict(zip(units, (nz.astype(float) @ G).mean(axis=0)))
        )
        df_s["W"] = kendall_w(R)
        df_s.to_csv(OUT / f"{name}_series.csv", index=False)
        lin_rank[name] = df_s
        dcol = pd.DataFrame(
            dict(
                column=names,
                series=D["series"],
                bucket=[bmap[s] for s in D["series"]],
                refits_nonzero=nz.sum(axis=0),
                sum_abs=absb.sum(axis=0),
                mean_abs=absb.mean(axis=0),
                sum_abs_sd=(absb * sd).sum(axis=0),
            )
        )
        dcol.to_csv(OUT / f"{name}_columns.csv", index=False)
        pd.DataFrame(ser_abs, columns=units).assign(
            # 2026-10-18 (user, new-notes 22): was date=dates[anchors], k=np.arange(K)
            date=rdates, k=KS
        ).to_csv(OUT / f"{name}_series_by_refit.csv", index=False)
        pd.DataFrame((nz.astype(float) @ G), columns=units).assign(
            # 2026-10-18 (user, new-notes 22): was date=dates[anchors], k=np.arange(K)
            date=rdates, k=KS
        ).to_csv(OUT / f"{name}_active_by_refit.csv", index=False)
        tot = absb.sum(axis=1)
        S[f"{name}"] = dict(
            alpha=LINEAR[name]["alpha"],
            W=float(kendall_w(R)),
            total_nonzero_median=float(np.median(nz.sum(axis=1))),
            total_nonzero_min=int(nz.sum(axis=1).min()),
            total_nonzero_max=int(nz.sum(axis=1).max()),
            exog_nonzero_median=float(
                np.median(
                    nz[:, [bmap[s] != "backbone" for s in D["series"]]].sum(axis=1)
                )
            ),
            series_ever=int(((nz.astype(float) @ G) > 0).any(axis=0).sum()),
            series_always=int(((nz.astype(float) @ G) > 0).all(axis=0).sum()),
            exog_series_ever=int(
                sum(
                    ((nz.astype(float) @ G) > 0).any(axis=0)[u]
                    for u in range(U)
                    if bmap[units[u]] != "backbone"
                )
            ),
            exog_series_majority=int(
                sum(
                    ((nz.astype(float) @ G) > 0).mean(axis=0)[u] >= 0.5
                    for u in range(U)
                    if bmap[units[u]] != "backbone"
                )
            ),
            exog_series_always=int(
                sum(
                    ((nz.astype(float) @ G) > 0).all(axis=0)[u]
                    for u in range(U)
                    if bmap[units[u]] != "backbone"
                )
            ),
            cols_ever=int(nz.any(axis=0).sum()),
            cols_majority=int((nz.mean(axis=0) >= 0.5).sum()),
            backbone_share_median=float(
                np.median(
                    (absb[:, [bmap[s] == "backbone" for s in D["series"]]].sum(1)) / tot
                )
            ),
            spearman_abs_vs_abs_sd=float(
                spearmanr(df_s["sum_abs"], df_s["sum_abs_sd"])[0]
            ),
            spearman_meanrank_vs_sum=float(
                spearmanr(df_s["mean_rank"], -df_s["sum_abs"])[0]
            ),
        )
        # 2026-10-18 (user, new-notes 22): probe refits inside the FI sample. Was: pb = Z[f"{name}_probe_beta"]
        pmask = np.isin(Z["probe_k"], KS)
        pb = Z[f"{name}_probe_beta"][pmask]
        S[name]["probe_nonzero"] = [
            int((np.abs(pb[:, q]) > NZ_TOL).sum()) for q in range(3)
        ]
        S[name]["probe_n"] = int(len(pb))
        S[name]["probe_abs_median"] = [
            float(np.median(np.abs(pb[:, q]))) for q in range(3)
        ]
        # columns whose |beta| beats the continuous probe's at the same probe refit, in most probe refits
        thp = np.abs(
            # 2026-10-18 (user, new-notes 22): was Z[f"{name}_probe_theta"][:, :p]
            Z[f"{name}_probe_theta"][pmask][:, :p]
        )  # the probe fit's own real-column coefficients
        beat = (thp > np.abs(pb[:, [0]])).mean(axis=0) >= 0.5
        S[name]["cols_beat_probe"] = int(beat.sum())
        S[name]["series_beat_probe"] = int(
            len(
                {
                    D["series"][j]
                    for j in np.flatnonzero(beat)
                    if bmap[D["series"][j]] != "backbone"
                }
            )
        )
        # selection rate: columns selected more often than the continuous probe (lasso; ridge never zeroes)
        rate = S[name]["probe_nonzero"][0] / len(pb)
        sel = nz.mean(axis=0) > rate
        S[name]["probe_rate"] = float(rate)
        S[name]["cols_above_probe_rate"] = int(sel.sum())
        S[name]["series_above_probe_rate"] = int(
            len(
                {
                    D["series"][j]
                    for j in np.flatnonzero(sel)
                    if bmap[D["series"][j]] != "backbone"
                }
            )
        )
        S[name]["top_columns"] = dcol.sort_values(
            ["refits_nonzero", "sum_abs"], ascending=False
        )["column"][:5].tolist()
        # 2026-10-18 (user, new-notes 22): was S[name]["batch_gap_max"] = float(Z[f"{name}_batch_gap"].max())
        S[name]["batch_gap_max"] = float(Z[f"{name}_batch_gap"][pmask].max())
    # scales of the robust-prescaled inputs (exogenous columns that are not constant in the window)
    # 2026-10-18 (user, new-notes 22): was sd, iqr = Z["win_sd"], Z["win_iqr"]
    sd, iqr = Z["win_sd"][KS], Z["win_iqr"][KS]
    import re

    kind = np.array(
        [
            "har"
            if re.fullmatch(r"har_ma_\d+", c)
            else "value"
            if re.fullmatch(r"adj_.+_ma_\d+", c)
            else "flag"
            if re.fullmatch(r".+_(avail|active)_ma_\d+", c)
            else "calendar"
            for c in names
        ]
    )
    S["scale"] = {}
    for kd in ("value", "flag", "har", "calendar"):
        live = (sd > 0) & (kind == kd)[None]
        S["scale"][kd] = dict(
            n_cols=int((kind == kd).sum()),
            sd_median=float(np.median(sd[live])),
            sd_p05=float(np.percentile(sd[live], 5)),
            sd_p95=float(np.percentile(sd[live], 95)),
            iqr_median=float(np.median(iqr[live])),
            iqr_p05=float(np.percentile(iqr[live], 5)),
            iqr_p95=float(np.percentile(iqr[live], 95)),
        )
    pd.DataFrame(
        dict(
            column=names,
            kind=kind,
            sd_median=np.median(sd, axis=0),
            iqr_median=np.median(iqr, axis=0),
        )
    ).to_csv(OUT / "column_scales.csv", index=False)
    # the old (re-tuned, one-penalty) lasso: the same count measure
    # 2026-10-18 (user, new-notes 22): [anchors, :-1] -> [lanch, :-1] in the two theta reads below
    oth = np.abs(Z["old_lasso_theta"][lanch, :-1]) > NZ_TOL
    old_any = ((oth.astype(float) @ G) > 0).mean(axis=0)
    new_any = (
        (np.abs(Z["lasso_theta"][lanch, :-1]) > NZ_TOL).astype(float) @ G > 0
    ).mean(axis=0)
    S["old_lasso"] = dict(
        spearman_any_share=float(spearmanr(old_any, new_any)[0]),
        total_nonzero_median=float(np.median(oth.sum(axis=1))),
        exog_series_majority=int(
            sum(old_any[u] >= 0.5 for u in range(U) if bmap[units[u]] != "backbone")
        ),
    )

    # ------------------------------------------------------------ trees: main effects (stored SHAP)
    tree_tabs = {}
    for mdl in MODELS:
        I_st = np.zeros((K, U))
        I_loc = np.zeros((K, U))
        I_col = np.zeros((K, p))
        glob_rows_st = []
        gaps: dict[str, list[float]] = dict(
            pred=[], add=[], rowsum=[], native=[], main_vs_stored=[], forest=[]
        )
        probe_rank_col, probe_cols_beat = [], []
        probe_pair_floor_ser, probe_pair_floor_col = [], []
        for k in range(K):  # k indexes the FI refits; the refit on disk is KS[k]
            # 2026-10-18 (user, new-notes 22): FI-sample refits, rows from FI_OOS_START. Was:
            #     z = np.load(WORK / "trees" / f"{mdl}_{k:03d}.npz")
            #     st = z["main_stored"].astype(np.float64)
            #     I_loc[k] = np.abs(z["main"].astype(np.float64) @ G).mean(axis=0)
            #     gaps["pred"].append(float(np.abs(z["pred"] - z["pred_stored"]).max()))
            z = np.load(WORK / "trees" / f"{mdl}_{KS[k]:03d}.npz")
            h0 = int(r0[k])
            st = z["main_stored"][h0:].astype(np.float64)
            I_st[k] = np.abs(st @ G).mean(axis=0)
            I_loc[k] = np.abs(z["main"][h0:].astype(np.float64) @ G).mean(axis=0)
            I_col[k] = np.abs(st).mean(axis=0)
            glob_rows_st.append(st @ G)
            gaps["pred"].append(float(np.abs(z["pred"][h0:] - z["pred_stored"][h0:]).max()))
            gaps["add"].append(float(z["additivity"]))
            gaps["rowsum"].append(float(z["rowsum"]))
            gaps["forest"].append(float(z["forest_pred_gap"]))
            gaps["main_vs_stored"].append(
                float(np.abs(z["main"] - z["main_stored"]).max())
            )
            if "xgb_native_gap" in z.files:
                gaps["native"].append(float(z["xgb_native_gap"]))
            if "lgbm_native_gap" in z.files:
                gaps["native"].append(float(z["lgbm_native_gap"]))
            if "probe_main_absmean" in z.files:
                ps = z["probe_main_stored"]  # the stored probe fit's mean |phi| (631)
                rk = ranks_desc(ps)
                probe_rank_col.append(rk[p])
                probe_cols_beat.append(ps[:p] > ps[p])
                probe_pair_floor_ser.append(z["probe_pair_ser"][0])
                probe_pair_floor_col.append(z["probe_pair_col"][0][:p])
        df_s, R = rank_table(I_st, units)
        df_s = df_s.merge(pd.DataFrame(base_cols), on="unit")
        allrows = np.vstack(glob_rows_st)
        glob = dict(zip(units, np.abs(allrows).mean(axis=0)))
        df_s["global_mean_abs"] = df_s["unit"].map(glob)
        df_s["rank_global_mean_abs"] = (-df_s["global_mean_abs"]).rank(method="min")
        beat = np.array(probe_cols_beat)  # (n_probe, p)
        beat_cols = beat.mean(axis=0) >= 0.5
        ser_beat = {s: bool(beat_cols[np.array(D["series"]) == s].any()) for s in units}
        df_s["clears_probe"] = df_s["unit"].map(ser_beat)
        _, R_loc = rank_table(I_loc, units)
        df_s["W"] = kendall_w(R)
        df_s.to_csv(OUT / f"{mdl}_series.csv", index=False)
        pd.DataFrame(R, columns=units).assign(
            # 2026-10-18 (user, new-notes 22): was date=dates[anchors], k=np.arange(K)
            date=rdates, k=KS
        ).to_csv(OUT / f"{mdl}_series_rank_by_refit.csv", index=False)
        dfc, Rc = rank_table(I_col, names)
        dfc["series"] = dfc["unit"].map(dict(zip(names, D["series"])))
        dfc["beats_probe_share"] = dfc["unit"].map(dict(zip(names, beat.mean(axis=0))))
        dfc.to_csv(OUT / f"{mdl}_columns.csv", index=False)
        tree_tabs[mdl] = df_s
        S[mdl] = dict(
            W=float(kendall_w(R)),
            W_columns=float(kendall_w(Rc)),
            W_local=float(kendall_w(R_loc)),
            spearman_meanrank_vs_borda=float(
                spearmanr(df_s["mean_rank"], -df_s["borda10"])[0]
            ),
            spearman_meanrank_vs_global=float(
                spearmanr(df_s["mean_rank"], -df_s["global_mean_abs"])[0]
            ),
            spearman_local_vs_stored=float(
                spearmanr(R.mean(axis=0), R_loc.mean(axis=0))[0]
            ),
            refit_pred_gap_max=float(max(gaps["pred"])),
            refit_pred_gap_median=float(np.median(gaps["pred"])),
            refits_bitwise=int(sum(g <= 1e-9 for g in gaps["pred"])),
            additivity_max=float(max(gaps["add"])),
            rowsum_max=float(max(gaps["rowsum"])),
            native_gap_max=float(max(gaps["native"])) if gaps["native"] else None,
            forest_pred_gap_max=float(max(gaps["forest"])),
            main_local_vs_stored_max=float(max(gaps["main_vs_stored"])),
            probe_n=int(len(probe_rank_col)),
            probe_col_rank_median=float(np.median(probe_rank_col)),
            cols_beat_probe=int(beat_cols.sum()),
            series_clear_probe=int(sum(ser_beat.values())),
            exog_series_clear_probe=int(
                sum(v for s, v in ser_beat.items() if bmap[s] != "backbone")
            ),
            top_series=list(df_s["unit"][:10]),
            top_columns=list(dfc["unit"][:5]),
            top_columns_rank=[float(v) for v in dfc["mean_rank"][:5]],
        )
        S[mdl]["probe_pair_floor_ser_median"] = float(
            np.median(np.max(np.array(probe_pair_floor_ser), axis=1))
        )
        S[mdl]["probe_pair_floor_col_median"] = float(
            np.median(np.max(np.array(probe_pair_floor_col), axis=1))
        )
        S[mdl]["fit_sec_total"] = None

    # ------------------------------------------------------------ trees: interactions (local refits)
    iu = np.triu_indices(U, 1)
    pair_labels = [f"{units[a]} x {units[b]}" for a, b in zip(*iu)]
    hz = {}
    tp = (
        json.loads((WORK / "top_pairs.json").read_text())
        if (WORK / "top_pairs.json").is_file()
        else None
    )
    cpu = {}
    for mdl in MODELS:
        Ssum = np.zeros((U, U))
        Psum = np.zeros(U)
        Csum = np.zeros((p, p))
        main_loc = np.zeros(U)
        n = 0
        I_pair = np.zeros((K, len(iu[0])))
        within = np.zeros((K, U))
        fit_s, shap_s = 0.0, 0.0
        floor_pairs = []
        for k in range(K):  # k indexes the FI refits; the refit on disk is KS[k]
            # 2026-10-18 (user, new-notes 22): FI-sample refits, rows from FI_OOS_START. Was:
            #     z = np.load(WORK / "trees" / f"{mdl}_{k:03d}.npz")
            #     kk = int(z["kk"])
            #     Sk = z["inter_ser"]  # (kk, U, U)
            #     pk = z["pure_ser"]  # (kk, U)
            #     main_loc += np.abs(z["main"].astype(np.float64) @ G).sum(axis=0)
            z = np.load(WORK / "trees" / f"{mdl}_{KS[k]:03d}.npz")
            h0 = int(r0[k])
            kk = int(z["kk"]) - h0
            Sk = z["inter_ser"][h0:]  # (kk, U, U)
            pk = z["pure_ser"][h0:]  # (kk, U)
            tot = 2 * Sk  # pair totals for A != B
            Ssum += np.abs(tot).sum(axis=0)
            dg = (
                np.einsum("taa->ta", Sk) - pk
            )  # within-series off-diagonal interactions
            Psum += np.abs(dg).sum(axis=0)
            # 2026-10-18 (user, new-notes 22): inter_col_absmean is stored as a mean over the refit's full tail, so for the first
            # FI refit (tail trimmed to FI_OOS_START) it still includes the dropped rows (column-pair table only)
            Csum += z["inter_col_absmean"].astype(np.float64) * kk
            main_loc += np.abs(z["main"][h0:].astype(np.float64) @ G).sum(axis=0)
            n += kk
            I_pair[k] = np.abs(tot).mean(axis=0)[iu]
            within[k] = np.abs(dg).mean(axis=0)
            fit_s += float(z["fit_sec"])
            shap_s += float(z["shap_sec"])
            if "probe_pair_ser" in z.files:
                floor_pairs.append(float(z["probe_pair_ser"][0].max()))
        cpu[mdl] = dict(fit=fit_s, shap=shap_s)
        Sg = Ssum / n
        np.fill_diagonal(
            Sg, Psum / n
        )  # diagonal = within-series cross-column interactions
        pd.DataFrame(Sg, index=units, columns=units).to_csv(
            OUT / f"{mdl}_inter_series_matrix.csv"
        )
        pd.DataFrame(dict(unit=units, main_mean_abs_local=main_loc / n)).to_csv(
            OUT / f"{mdl}_main_local.csv", index=False
        )
        dfp, Rp = rank_table(I_pair, pair_labels)
        dfp["global_mean_abs"] = dfp["unit"].map(dict(zip(pair_labels, Sg[iu])))
        dfp["rank_global"] = (-dfp["global_mean_abs"]).rank(method="min")
        top50 = np.argsort(-Sg[iu])[:TOPK_W_PAIRS]
        Rsub = np.vstack([ranks_desc(r) for r in I_pair[:, top50]])
        dfp["W_top50"] = kendall_w(Rsub)
        dfp.to_csv(OUT / f"{mdl}_inter_pairs.csv", index=False)
        Cg = Csum / n
        jc = np.triu_indices(p, 1)
        cv = 2 * Cg[jc]
        oc = np.argsort(-cv)[:30]
        pd.DataFrame(
            dict(
                col_a=[names[jc[0][q]] for q in oc],
                col_b=[names[jc[1][q]] for q in oc],
                series_a=[D["series"][jc[0][q]] for q in oc],
                series_b=[D["series"][jc[1][q]] for q in oc],
                mean_abs_pair=cv[oc],
            )
        ).to_csv(OUT / f"{mdl}_inter_top_columns.csv", index=False)
        diag = np.diag(Cg)
        S[mdl]["inter"] = dict(
            W_top50=float(kendall_w(Rsub)),
            spearman_meanrank_vs_global=float(
                spearmanr(dfp["mean_rank"], -dfp["global_mean_abs"])[0]
            ),
            total_pair_over_main=float(Sg[iu].sum() / (main_loc / n).sum()),
            col_pair_total_over_diag=float(cv.sum() / diag.sum()),
            top_pair=pair_labels[int(np.argmax(Sg[iu]))],
            top_pair_value=float(Sg[iu].max()),
            floor_ser_median=float(np.median(floor_pairs)) if floor_pairs else None,
            pairs_above_floor=int((Sg[iu] > np.median(floor_pairs)).sum())
            if floor_pairs
            else None,
            within_top=units[int(np.argmax(np.diag(Sg)))],
            within_top_value=float(np.diag(Sg).max()),
        )
        # H statistic
        # 2026-10-18 (user, new-notes 22): H_REFITS -> HK (the H refits inside the FI sample) here and in the loop below
        if tp is not None and (WORK / HDIR / f"{mdl}_{HK[0]:03d}.npz").is_file():
            num_s = np.zeros(H_TOP_SERIES)
            den_s = np.zeros(H_TOP_SERIES)
            va_s = np.zeros(H_TOP_SERIES)
            vb_s = np.zeros(H_TOP_SERIES)
            vb_c = np.zeros(H_TOP_COLS)
            va_c = np.zeros(H_TOP_COLS)
            num_c = np.zeros(H_TOP_COLS)
            den_c = np.zeros(H_TOP_COLS)
            per_s = []
            for k in HK:
                h = np.load(WORK / HDIR / f"{mdl}_{k:03d}.npz")
                num_s += h["s_num"]
                den_s += h["s_den"]
                va_s += h["s_va"]
                vb_s += h["s_vb"]
                va_c += h["c_va"]
                vb_c += h["c_vb"]
                num_c += h["c_num"]
                den_c += h["c_den"]
                per_s.append(h["s_num"] / np.where(h["s_den"] > 0, h["s_den"], np.nan))
            per_arr = np.array(per_s)
            hz[mdl] = pd.DataFrame(
                dict(
                    kind=["series"] * H_TOP_SERIES + ["column"] * H_TOP_COLS,
                    a=[units[a] for a, _ in tp[mdl]["series"]]
                    + [names[a] for a, _ in tp[mdl]["cols"]],
                    b=[units[b] for _, b in tp[mdl]["series"]]
                    + [names[b] for _, b in tp[mdl]["cols"]],
                    H2_pooled=list(num_s / den_s) + list(num_c / den_c),
                    H2_median=list(np.nanmedian(per_arr, axis=0))
                    + [np.nan] * H_TOP_COLS,
                    # the interaction's PD variance relative to the smaller of the two partial dependences' variances
                    H2_min=list(num_s / np.minimum(va_s, vb_s))
                    + list(num_c / np.minimum(va_c, vb_c)),
                    shap_pair=[Sg[a, b] for a, b in tp[mdl]["series"]]
                    + [2 * Cg[a, b] for a, b in tp[mdl]["cols"]],
                )
            )
            hz[mdl].to_csv(OUT / f"{mdl}_hstat.csv", index=False)
    S["cpu_sec"] = cpu
    import re

    walls = {}
    for line in (WORK / "log.txt").read_text(encoding="utf-8").splitlines():
        mm = re.search(r"\] (\w+) done in (\d+)s", line)
        if mm:
            walls[mm.group(1)] = int(mm.group(2))  # the last run of each stage
    S["wall_sec"] = walls
    S["workers"] = WORKERS
    # 2026-10-18 (user, new-notes 22): was refits=list(H_REFITS)
    S["H"] = dict(
        refits=list(HK), rows=H_ROWS, top_series=H_TOP_SERIES, top_cols=H_TOP_COLS
    )
    # agreement across measures / models (series mean ranks)
    mr = {m: tree_tabs[m].set_index("unit")["mean_rank"] for m in MODELS}
    mr.update({n_: lin_rank[n_].set_index("unit")["mean_rank"] for n_ in LINEAR})
    keys = list(mr)
    S["spearman_meanrank"] = {
        f"{a}~{b}": float(spearmanr(mr[a][units], mr[b][units])[0])
        for i_, a in enumerate(keys)
        for b in keys[i_ + 1 :]
    }
    S["gates"] = gates
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    say("reduce: wrote results/rehaul_fi/*.csv and summary.json")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()
    if stage in ("linear", "all"):
        linear()
    if stage in ("trees", "all"):
        trees()
    if stage in ("hstat", "all"):
        hstat()
    if stage in ("reduce", "all"):
        reduce()
    if stage not in ("linear", "trees", "hstat", "reduce", "all"):
        raise SystemExit("stage: linear | trees | hstat | reduce | all")
    say(f"{stage} done in {time.time() - t0:.0f}s")
