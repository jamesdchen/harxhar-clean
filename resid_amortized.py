"""Amortized full-OOS residualized (Ridge->tree) tuning scorer.

Naive ``MultiStage(Residualizer(Ridge) + tree)`` refits sklearn Ridge EVERY bar
(231k x / trial; ~825s/eval even for the small implied_vol/250d cell) -> infeasible
to tune at scale. But the Ridge base is CONFIG-INDEPENDENT within a cell, so we
precompute it ONCE and each trial then fits only trees on ``y - ridge``.

Exact-amortization (reproduces MultiStage bit-for-bit, see ``selfcheck``):
  * every-bar Ridge OOS preds  -> incremental RollingLeastSquares (fast + exact)
  * Ridge coef/intercept at each tree-refit cadence point t_r (sklearn Ridge on the
    window ending at t_r == the every-bar Ridge AT t_r) -> used to form the in-sample
    residual ``y_train - X_train@beta_tr`` the tree trains on at that refit.
  final[t] = ridge_oos[t] + tree_{t_r}.predict(X[t])     (tree from the block's refit)

modes:
  prep      MODEL BUCKET TWD ALPHA TREE_REFIT        -> cache to results/resid_prep/<cell>/
  trial     CELL ARM [OUTNAME]   (TREE_CFG env json) -> load cache, score one config
  selfcheck MODEL BUCKET TWD ALPHA TREE_REFIT NSEG   -> assert amortized == naive MultiStage
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, os.getcwd())
from src.evaluation.metrics import apply_duan_smearing
from src.features.transforms.scaling import rolling_robust_scale
from src.features.transforms.target import PERIODS_PER_DAY
from src.models.ridge import fit_predict_ridge

CACHE_ROOT = "results/resid_prep"
PCA_REFIT = 2000  # cadence for the rolling-PCA basis (stable -> coarse is fine + cheap)


def cell_id(model, bucket, twd, alpha, refit, pipe="slim", base_kind="ridge"):
    tag = f"a{alpha:g}" if base_kind == "ridge" else base_kind  # ridge: "a100" (back-compat); enet: "enet"
    return f"{model}_{bucket}_tw{twd}_{tag}_rf{refit}_{pipe}"


def _split_cols(feats):
    """Partition feature columns into (HAR, exog, indicator) index lists. PCA compresses only
    the dense-weak EXOG block; HAR (strong linear core) and the availability indicators pass raw."""
    har = [i for i, f in enumerate(feats) if "har_ma" in f]
    ind = [i for i, f in enumerate(feats) if "_avail" in f or "_active" in f]
    skip = set(har) | set(ind)
    exog = [i for i in range(len(feats)) if i not in skip]
    return har, exog, ind


# Clock-anchored session-transition gates for the OPEN/CLOSE x HAR regime interaction (base_kind=
# "enetreg"). The auction / session-transition microstructure localizes the HAR vol-persistence
# sign-flip at the 09:30 ET OPEN (hour==9) and the 16:00 CLOSE + after-hours, hours 16-19 (capped
# below 20 -- the -0.21 reversal lobe peaks at 17:00, dead overnight bars excluded) -- per tests123.py.
# These are ACTUAL clock hours, NOT hour-quantile buckets (bucket-0-of-6 is overnight, not the open).
# Crossing them with the 6 HAR columns lets the linear enet base ABSORB the regime the residualized
# EBM otherwise has to rediscover, so the downstream EBM digs into the structure that remains.
_HAR_COLS = ("har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125")


def _regime_interactions(Xs, feats):
    """HAR x {open, close} clock-anchored interaction columns. Returns (INT[N,k], names[k]).
    open = (hour==9), the 09:30 ET opening bar; close = (16<=hour<=19), the 16:00 close auction +
    after-hours (capped below 20 to exclude dead overnight bars; AH trading ends ~20:00 ET).
    Constant products (std<=1e-9, e.g. a gate empty in this cache) drop out."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    gates = (("open", (hour == 9).astype(np.float64)), ("close", ((hour >= 16) & (hour <= 19)).astype(np.float64)))
    cols, names = [], []
    for hn in _HAR_COLS:
        if hn not in feats:
            continue
        h = Xs[:, feats.index(hn)].astype(np.float64)
        for gname, g in gates:
            col = h * g
            if col.std() > 1e-9:
                cols.append(col)
                names.append(f"{hn}_x_{gname}")
    INT = np.ascontiguousarray(np.column_stack(cols)) if cols else np.empty((len(Xs), 0), dtype=np.float64)
    return INT, names


def _load_matrix(bucket, train_win, pipe="slim"):
    """pipe="slim": raw per-slot-rank matrix, NO robust-scale (ablation C_norob_all winner:
    drop semantic + drop robust-scale -> per-slot-rank -> indicators -> Ridge).
    pipe="pca<K>": slim, but the EXOG block compressed to K causal rolling-PCA factors
    (HAR + indicators kept raw) -> [HAR, K factors, indicators]. Tests the dense-weak
    low-rank hypothesis (what colsample->0.1 bagging gropes toward).
    pipe="std": production divide+semantic matrix WITH robust-scale (the old baseline)."""
    if pipe.startswith("pca"):
        from src.features.transforms.rolling_pca import rolling_pca
        k = int(pipe[3:])
        b = f"results/covid_imp_rank/{bucket}"
        X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
        y = np.load(f"{b}/y.npy"); base = np.load(f"{b}/base.npy")
        har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
        factors = rolling_pca(X[:, exog], train_win, k, PCA_REFIT)
        Xs = np.ascontiguousarray(np.hstack([X[:, har], factors, X[:, ind]]))
        return Xs, y, base
    if pipe == "slim":
        b = f"results/covid_imp_rank/{bucket}"
        X = np.load(f"{b}/X_imp.npy"); y = np.load(f"{b}/y.npy"); base = np.load(f"{b}/base.npy")
        Xs = np.ascontiguousarray(np.asarray(X, dtype=np.float64))  # raw rank features; no scaling
        return Xs, y, base
    b = f"results/covid_imp/{bucket}"
    X = np.load(f"{b}/X_imp.npy"); y = np.load(f"{b}/y.npy"); base = np.load(f"{b}/base.npy")
    ref_iqr = np.load(f"{b}/ref_iqr.npy"); meta = json.load(open(f"{b}/meta.json"))
    Xs = np.ascontiguousarray(
        rolling_robust_scale(np.asarray(X, dtype=np.float64), train_win, ref_iqr=ref_iqr, fixed_cols=meta["fixed_cols"])
    )
    return Xs, y, base


def _cadence_ridge(Xs, y, train_win, alpha, refit):
    """sklearn Ridge coef/intercept at each refit point t_r (the in-sample basis for the tree)."""
    n = len(Xs)
    starts = list(range(train_win, n, refit))
    coefs = np.empty((len(starts), Xs.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(starts), dtype=np.float64)
    for i, t_r in enumerate(starts):
        rg = Ridge(alpha=alpha).fit(Xs[t_r - train_win:t_r], y[t_r - train_win:t_r])
        coefs[i] = rg.coef_; intercepts[i] = rg.intercept_
    return np.asarray(starts, dtype=np.int64), coefs, intercepts


def _cadence_predict(Xs, y, train_win, make_est, refit):
    """Cadence-refit OOS preds for ANY sklearn estimator (Ridge/ElasticNet/PLS): fit on the
    trailing window at each refit point, predict the block. Generalizes _cadence_ridge to
    estimators without a clean coef/intercept (PLS)."""
    n = len(Xs)
    oos = np.empty(n - train_win, dtype=np.float64)
    starts = list(range(train_win, n, refit))
    for i, t_r in enumerate(starts):
        est = make_est(); est.fit(Xs[t_r - train_win:t_r], y[t_r - train_win:t_r])
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        oos[t_r - train_win:t_end - train_win] = np.asarray(est.predict(Xs[t_r:t_end])).ravel()
    return oos


def _dr_configs():
    """Linear/DR bases to compare against Ridge-a100 (0.12565): elastic net (sparse supervised
    selection) + PLS (supervised components) — the SUPERVISED DR that variance-PCA isn't."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet, Ridge
    cfgs = [("ridge_a100", lambda: Ridge(alpha=100))]
    for a in (1e-3, 1e-2, 1e-1):
        for l1 in (0.2, 0.5, 0.8):
            cfgs.append((f"enet_a{a:g}_l1{l1:g}", lambda a=a, l1=l1: ElasticNet(alpha=a, l1_ratio=l1, max_iter=1000, tol=1e-3)))
    for nc in (10, 20, 40, 80):
        cfgs.append((f"pls_nc{nc}", lambda nc=nc: PLSRegression(n_components=nc)))
    return cfgs


def do_dimreduce_cell(model, bucket, idx, refit=8000):
    """One config of the supervised-DR comparison (SLURM-array task). Full slim inputs (same as
    raw-slim Ridge), cadence-refit, fixed OOS. Caveat: enet/Ridge L1/L2 see UNstandardized slim
    (HAR raw, exog ~N(0,1), ind 0/1) for apples-to-apples with the Ridge baseline."""
    name, make_est = _dr_configs()[idx]
    b = f"results/covid_imp_rank/{bucket}"
    Xs = np.ascontiguousarray(np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64))
    y = np.load(f"{b}/y.npy"); base = np.load(f"{b}/base.npy")
    tw = 1000 * PERIODS_PER_DAY; oos_start = tw
    t0 = time.time()
    preds = _cadence_predict(Xs, y, tw, make_est, refit)
    pr, tr = apply_duan_smearing(preds[oos_start - tw:], y[oos_start:], base[oos_start:])
    m = (tr > 0) & (pr > 0); r = tr[m] / pr[m]
    print("DR idx=%-2d %-18s qlike=%.5f %.0fs" % (idx, name, float(np.mean(r - np.log(r) - 1)), time.time() - t0), flush=True)


def _cadence_enet(Xs, y, train_win, refit, alpha=0.001, l1=0.2):
    """Cadence-refit ELASTIC-NET base coef/intercept (the winning linear base, 0.12530). Warm-started
    coordinate descent chains the rolling refits (no exact rank-1 for L1; warm-start is the cheap route)."""
    from sklearn.linear_model import ElasticNet
    n = len(Xs)
    starts = list(range(train_win, n, refit))
    en = ElasticNet(alpha=alpha, l1_ratio=l1, warm_start=True, max_iter=1000, tol=1e-3)
    coefs = np.empty((len(starts), Xs.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(starts), dtype=np.float64)
    for i, t_r in enumerate(starts):
        en.fit(Xs[t_r - train_win:t_r], y[t_r - train_win:t_r])
        coefs[i] = en.coef_; intercepts[i] = float(en.intercept_)
    return np.asarray(starts, dtype=np.int64), coefs, intercepts


def _cadence_ridge_oos(Xs, train_win, starts, coefs, intercepts):
    """Cadence-refit Ridge OOS preds: within block [t_r, t_{r+1}) use that block's coef
    (a matvec) — free from the cadence coefs, no slow every-bar incremental. Ridge and the
    tree then refit on the SAME cadence (internally consistent baseline)."""
    n = len(Xs)
    oos = np.empty(n - train_win, dtype=np.float64)
    for i, t_r in enumerate(starts):
        t_r = int(t_r); t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        oos[t_r - train_win:t_end - train_win] = Xs[t_r:t_end] @ coefs[i] + intercepts[i]
    return oos


def _tree_factory(model, cfg):
    if model in ("lgbm", "lightgbm"):
        from lightgbm import LGBMRegressor
        kw = dict(cfg); kw.setdefault("n_jobs", 4); kw.setdefault("verbose", -1); kw.setdefault("random_state", 42)
        return lambda: LGBMRegressor(**kw)
    if model == "ebm":  # Microsoft EBM: bagged boosted GAM (additive + pairwise), interpretable
        from interpret.glassbox import ExplainableBoostingRegressor
        kw = dict(cfg); kw.setdefault("n_jobs", 4); kw.setdefault("random_state", 42)
        return lambda: ExplainableBoostingRegressor(**kw)
    from xgboost import XGBRegressor
    kw = dict(cfg); kw.setdefault("n_jobs", 4); kw.setdefault("tree_method", "hist"); kw.setdefault("random_state", 42)
    return lambda: XGBRegressor(**kw)


def _residualized_preds(Xs, y, train_win, ridge_oos, starts, coefs, intercepts, make_tree):
    """final[t] = ridge_oos[t] + tree_{t_r}.predict(X[t]); tree fits y_train - X_train@beta_tr."""
    n = len(Xs)
    preds = np.array(ridge_oos, dtype=np.float64, copy=True)  # OOS preds indexed by k = t - train_win
    for i, t_r in enumerate(starts):
        Xtr = Xs[t_r - train_win:t_r]
        r_train = y[t_r - train_win:t_r] - (Xtr @ coefs[i] + intercepts[i])
        tree = make_tree(); tree.fit(Xtr, r_train)
        t_end = min(t_r + (starts[i + 1] - t_r if i + 1 < len(starts) else n - t_r), n)
        preds[t_r - train_win:t_end - train_win] += tree.predict(Xs[t_r:t_end])
    return preds


def _qlike(preds, y, base, train_win):
    yo, bo = y[train_win:], base[train_win:]
    pr, tr = apply_duan_smearing(preds, yo, bo)
    m = (tr > 0) & (pr > 0); r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def do_prep(model, bucket, twd, alpha, refit, pipe="slim", base_kind="ridge"):
    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    feats_aug = None
    if base_kind == "enetreg":
        # Fold the distilled OPEN/CLOSE x HAR regime interaction into the linear base. Appended
        # BEFORE the constant-drop so it flows through enet/masks/tree exactly like a native column.
        feats0 = json.load(open(f"results/covid_imp_rank/{bucket}/meta.json"))["feats"]
        INT, int_names = _regime_interactions(Xs, feats0)
        Xs = np.ascontiguousarray(np.hstack([Xs, INT]))
        feats_aug = list(feats0) + int_names
    # Drop dead (zero-variance) columns before fitting/caching. ~31% of the all_buckets
    # slim matrix is constant always-on availability indicators (_avail/_active flags for
    # series that are never missing -> pinned at 1). They are inert to enet (collinear with
    # the intercept -> zero/absorbed) and to trees (no split on a constant), so dropping
    # them leaves QLIKE unchanged but shrinks the cached matrix, makes the effective
    # feature count honest, and removes the divide-by-zero landmine for the robust-scale
    # pipe. Done post-assembly so the _split_cols/fixed_cols column-index machinery is
    # untouched. (No effect on an already-cached cell; takes effect on the next prep.)
    keep = Xs.std(axis=0) > 1e-9
    n_dropped = int((~keep).sum())
    Xs = np.ascontiguousarray(Xs[:, keep])
    if feats_aug is not None:
        feats_aug = [f for f, k in zip(feats_aug, keep) if k]
    t0 = time.time()
    if base_kind in ("enet", "enetreg"):
        starts, coefs, intercepts = _cadence_enet(Xs, y, train_win, refit)
    else:
        starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, alpha, refit)
    ridge_oos = _cadence_ridge_oos(Xs, train_win, starts, coefs, intercepts)  # base OOS preds (base-agnostic matvec)
    cid = cell_id(model, bucket, twd, alpha, refit, pipe, base_kind)
    d = f"{CACHE_ROOT}/{cid}"; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/Xs.npy", Xs); np.save(f"{d}/y.npy", y); np.save(f"{d}/base.npy", base)
    np.save(f"{d}/ridge_oos.npy", ridge_oos)
    np.savez(f"{d}/cadence.npz", starts=starts, coefs=coefs, intercepts=intercepts)
    n_int = 0 if feats_aug is None else sum(1 for f in feats_aug if "_x_open" in f or "_x_close" in f)
    if feats_aug is not None:  # augmented column names (enetreg) so ebm_interpret can label the interactions
        json.dump(feats_aug, open(f"{d}/feats.json", "w"))
    json.dump({"model": model, "bucket": bucket, "twd": twd, "alpha": alpha, "refit": refit, "pipe": pipe,
               "base_kind": base_kind, "train_win": train_win, "n": int(len(Xs)), "n_refits": int(len(starts)),
               "p": int(Xs.shape[1]), "n_dropped_const": n_dropped, "n_regime_int": n_int},
              open(f"{d}/cell.json", "w"), indent=2)
    print("PREP %s n=%d n_refits=%d base=%s base_alone_qlike=%.5f %.0fs" % (
        cid, len(Xs), len(starts), base_kind, _qlike(ridge_oos, y, base, train_win), time.time() - t0), flush=True)


def load_cache(cid):
    """Load the per-cell amortized cache ONCE (worker reuses across trials)."""
    d = f"{CACHE_ROOT}/{cid}"
    cad = np.load(f"{d}/cadence.npz")
    out = {"cell": json.load(open(f"{d}/cell.json")),
           "Xs": np.load(f"{d}/Xs.npy"), "y": np.load(f"{d}/y.npy"), "base": np.load(f"{d}/base.npy"),
           "ridge_oos": np.load(f"{d}/ridge_oos.npy"),
           "starts": cad["starts"], "coefs": cad["coefs"], "intercepts": cad["intercepts"]}
    if os.path.exists(f"{d}/masks.npy"):  # per-cadence-block enet survivor masks (for arm=resid_subset)
        out["masks"] = np.load(f"{d}/masks.npy")
    if os.path.exists(f"{d}/prunable.npy"):  # safe-prune (224-signalless) mask (for arm=resid_pruned)
        out["prunable"] = np.load(f"{d}/prunable.npy")
    # Live availability-indicator mask (arm=resid_subset_ind): _avail/_active columns that VARY.
    # enet-survivor selection (resid_subset) filters these out (~5% survival) because their
    # signal is interaction-only (zero linear main effect), so the residual tree never sees the
    # event channel. This mask lets resid_subset_ind union them back in to TEST that channel.
    out["live_ind"] = None
    try:
        feats = json.load(open(f"results/covid_imp_rank/{out['cell']['bucket']}/meta.json"))["feats"]
        if len(feats) == out["Xs"].shape[1]:
            isind = np.array([("_avail" in f or "_active" in f) for f in feats])
            out["live_ind"] = isind & (out["Xs"].std(axis=0) > 1e-9)
    except Exception:
        pass
    return out


def _preds(cache, arm, cfg):
    """OOS preds for one config. arm in {residualized, raw_tree, ridge_alone}."""
    c = cache; tw = c["cell"]["train_win"]; model = c["cell"]["model"]
    if arm == "ridge_alone":
        return c["ridge_oos"]
    mk = _tree_factory(model, cfg)
    if arm == "raw_tree":  # tree on y directly (control); ridge base zeroed
        return _residualized_preds(c["Xs"], c["y"], tw, np.zeros_like(c["ridge_oos"]),
                                   c["starts"], np.zeros_like(c["coefs"]), np.zeros_like(c["intercepts"]), mk)
    return _residualized_preds(c["Xs"], c["y"], tw, c["ridge_oos"], c["starts"], c["coefs"], c["intercepts"], mk)


def score(cache, arm, cfg):
    """QLIKE for one config against a preloaded cache."""
    c = cache
    return _qlike(_preds(cache, arm, cfg), c["y"], c["base"], c["cell"]["train_win"])


def run_to_csv(cid, arm, cfg, out_csv):
    """Score one config and write a results.csv (true_raw/pred_raw) for the campaign tell."""
    import pandas as pd
    c = load_cache(cid); tw = c["cell"]["train_win"]
    yo, bo = c["y"][tw:], c["base"][tw:]
    pr, tr = apply_duan_smearing(_preds(c, arm, cfg), yo, bo)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame({"true_raw": tr, "pred_raw": pr}).to_csv(out_csv, index=False)
    return out_csv


def preds_chunk(cache, arm, cfg, blk0, blk1):
    """Residualized OOS preds for the rows covered by cadence blocks [blk0, blk1).

    Chunks partition the cadence blocks, so each tree-refit point is fit in exactly ONE
    chunk (no repeated work) and the trial's full preds are the ordered concat of chunks.
    Returns (k0, k1, preds) with k = t - train_win (OOS index)."""
    c = cache; tw = c["cell"]["train_win"]; model = c["cell"]["model"]; n = len(c["Xs"])
    starts = c["starts"]
    k0 = int(starts[blk0]) - tw
    k1 = (int(starts[blk1]) if blk1 < len(starts) else n) - tw
    if arm == "ridge_alone":
        return k0, k1, np.array(c["ridge_oos"][k0:k1], copy=True)
    raw = arm == "raw_tree"
    out = np.zeros(k1 - k0) if raw else np.array(c["ridge_oos"][k0:k1], copy=True)
    if arm == "resid_subset":      # tree sees only the block's rolling enet survivors (~120)
        def colsel(i): return c["masks"][i]
    elif arm == "resid_subset_ind":  # survivors UNION live availability indicators (event channel)
        li = c["live_ind"]
        def colsel(i): return c["masks"][i] if li is None else (c["masks"][i] | li)
    elif arm == "resid_pruned":    # tree sees all-but-the-signalless (the 224-indicator prune)
        keep = ~c["prunable"]
        def colsel(i): return keep
    else:                          # residualized / raw_tree: tree sees all features
        def colsel(i): return slice(None)
    mk = _tree_factory(model, cfg)
    for i in range(blk0, blk1):
        t_r = int(starts[i]); Xtr = c["Xs"][t_r - tw:t_r]; cols = colsel(i)
        r_train = c["y"][t_r - tw:t_r] if raw else c["y"][t_r - tw:t_r] - (Xtr @ c["coefs"][i] + c["intercepts"][i])
        tree = mk(); tree.fit(Xtr[:, cols], r_train)
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        out[t_r - tw - k0:t_end - tw - k0] += tree.predict(c["Xs"][t_r:t_end][:, cols]).ravel()
    return k0, k1, out


def run_chunk_to_csv(cid, arm, cfg, blk0, blk1, out_csv):
    """Score one cadence-block chunk; write its ADJ-space slice (k, pred_adj, y_true, base).
    Smearing is DEFERRED to collect-time (global) so the concatenated QLIKE equals the whole-
    backtest QLIKE — per-chunk smearing would use a per-chunk retransformation factor."""
    import pandas as pd
    c = load_cache(cid); tw = c["cell"]["train_win"]
    k0, k1, preds = preds_chunk(c, arm, cfg, blk0, blk1)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame({"k": np.arange(k0, k1), "pred_adj": preds,
                  "y_true": c["y"][tw + k0:tw + k1], "base": c["base"][tw + k0:tw + k1]}).to_csv(out_csv, index=False)
    return out_csv


def do_enet_masks(cid, alpha=0.001, l1=0.2):
    """Augment a built cache with (1) per-cadence-block rolling enet survivor masks (arm=resid_subset)
    and (2) the safe-prune mask (dead to BOTH enet-all-windows AND a deep residual tree). Warm-started
    coordinate descent chains the cadence enet fits so the rolling refits are cheap."""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    d = f"{CACHE_ROOT}/{cid}"
    c = load_cache(cid); Xs, y, starts = c["Xs"], c["y"], c["starts"]
    tw = c["cell"]["train_win"]; p = Xs.shape[1]
    en = ElasticNet(alpha=alpha, l1_ratio=l1, warm_start=True, max_iter=1000, tol=1e-3)
    masks = np.zeros((len(starts), p), dtype=bool)
    t0 = time.time()
    for i, t_r in enumerate(starts):
        t_r = int(t_r); en.fit(Xs[t_r - tw:t_r], y[t_r - tw:t_r]); masks[i] = en.coef_ != 0.0
    np.save(f"{d}/masks.npy", masks)
    rg = Ridge(alpha=float(c["cell"]["alpha"])).fit(Xs, y)
    lg = LGBMRegressor(n_estimators=500, max_depth=8, num_leaves=127, learning_rate=0.05,
                       min_child_samples=20, n_jobs=8, verbose=-1, importance_type="split").fit(Xs, y - rg.predict(Xs))
    prunable = (masks.sum(axis=0) == 0) & (lg.feature_importances_ == 0)
    np.save(f"{d}/prunable.npy", prunable)
    print("ENETMASKS %s avg_survivors=%.0f/%d prunable=%d %.0fs" % (
        cid, masks.sum(axis=1).mean(), p, int(prunable.sum()), time.time() - t0), flush=True)


def do_chunk_task(cid, arm, idx, nchunks):
    """One chunk of a FIXED-config chunked eval (SLURM/SGE array task). Tree cfg via env TREE_CFG."""
    import math
    nb = int(json.load(open(f"{CACHE_ROOT}/{cid}/cell.json"))["n_refits"])
    sz = max(1, math.ceil(nb / nchunks)); ranges = [(b, min(b + sz, nb)) for b in range(0, nb, sz)]
    if idx >= len(ranges):
        print("CHUNK idx=%d >= effective_chunks=%d (skip)" % (idx, len(ranges)), flush=True); return
    blk0, blk1 = ranges[idx]
    cfg = json.loads(os.environ.get("TREE_CFG", "{}"))
    out = run_chunk_to_csv(cid, arm, cfg, blk0, blk1, f"results/resid_ab/{cid}/{arm}/chunk_{blk0}_{blk1}.csv")
    print("CHUNK %s %s idx=%d blk[%d:%d] -> %s" % (cid, arm, idx, blk0, blk1, out), flush=True)


def do_chunk_collect(cid, arm):
    """Concat a chunked eval's ADJ-space CSVs, apply GLOBAL Duan smearing, report full-OOS QLIKE."""
    import glob

    import pandas as pd
    cell = json.load(open(f"{CACHE_ROOT}/{cid}/cell.json")); n_oos = int(cell["n"] - cell["train_win"])
    files = sorted(glob.glob(f"results/resid_ab/{cid}/{arm}/chunk_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("k").sort_values("k")
    if len(df) != n_oos:
        print("COLLECT %s %s INCOMPLETE %d/%d (%d files)" % (cid, arm, len(df), n_oos, len(files)), flush=True)
        return
    pr, tr = apply_duan_smearing(df["pred_adj"].to_numpy(), df["y_true"].to_numpy(), df["base"].to_numpy())
    m = (tr > 0) & (pr > 0); r = tr[m] / pr[m]
    print("COLLECT %s %-13s qlike=%.5f n=%d chunks=%d" % (cid, arm, float(np.mean(r - np.log(r) - 1)), len(df), len(files)), flush=True)


def do_trial(cid, arm, outname=None):
    cache = load_cache(cid)
    cfg = json.loads(os.environ.get("TREE_CFG", "{}"))
    t0 = time.time()
    q = score(cache, arm, cfg); dt = time.time() - t0
    print("TRIAL %s %-12s qlike=%.5f %.0fs" % (cid, arm, q, dt), flush=True)
    if outname:
        os.makedirs("results/resid_tree", exist_ok=True)
        json.dump({"cid": cid, "arm": arm, "qlike": q, "secs": dt, "cfg": cfg}, open(f"results/resid_tree/{outname}.json", "w"), indent=2)
    return q


def do_selfcheck(model, bucket, twd, alpha, refit, nseg, pipe="slim"):
    """Assert amortized == naive MultiStage(Residualizer(Ridge)+tree) on the first NSEG OOS bars."""
    from src.backtest.multi_stage import MultiStageBacktest
    from src.features.transforms.residualizer import Residualizer
    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    seg = train_win + nseg
    Xs, y, base = Xs[:seg], y[:seg], base[:seg]
    cfg = {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05,
           "num_leaves": 15, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.5}
    make_tree = _tree_factory(model, cfg)
    # amortized
    ridge_oos = fit_predict_ridge(Xs, y, train_win, {"alpha": alpha, "_refit_frequency": 1, "_incremental": True})
    starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, alpha, refit)
    amort = _residualized_preds(Xs, y, train_win, ridge_oos, starts, coefs, intercepts, make_tree)
    # naive MultiStage
    bt = MultiStageBacktest(Residualizer(lambda: Ridge(alpha=alpha)), make_tree, refit_frequency=refit)
    naive = bt.run(np.ascontiguousarray(Xs), y, train_win, desc="naive")
    d = float(np.max(np.abs(amort - naive)))
    qa = _qlike(amort, y, base, train_win); qn = _qlike(naive, y, base, train_win)
    print("SELFCHECK %s/%s rf%d nseg%d: max|amort-naive|=%.3e  qlike amort=%.6f naive=%.6f  %s"
          % (model, bucket, refit, nseg, d, qa, qn, "OK" if d < 1e-6 else "MISMATCH"), flush=True)
    return d


def do_alphascan(model, bucket, twd, pipe, alphas, refit=2000):
    """Cadence-refit ridge_alone QLIKE over a grid of alpha (coarse refit -> fast) to pick alpha*."""
    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    for a in alphas:
        starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, a, refit)
        ro = _cadence_ridge_oos(Xs, train_win, starts, coefs, intercepts)
        print("ALPHASCAN %s %s tw%d %s a%g ridge_alone_qlike=%.5f" % (model, bucket, twd, pipe, a, _qlike(ro, y, base, train_win)), flush=True)


def do_chunkcheck(model, bucket, twd, alpha, refit, pipe, nchunks):
    """Verify chunked == whole: concat of preds_chunk over a CHUNKS-partition equals _preds."""
    import math
    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, alpha, refit)
    ridge_oos = _cadence_ridge_oos(Xs, train_win, starts, coefs, intercepts)
    cache = {"cell": {"train_win": train_win, "model": model}, "Xs": Xs, "y": y, "base": base,
             "ridge_oos": ridge_oos, "starts": starts, "coefs": coefs, "intercepts": intercepts}
    cfg = {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05, "num_leaves": 15,
           "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.5}
    whole = _preds(cache, "residualized", cfg)
    nb = len(starts); sz = max(1, math.ceil(nb / nchunks)); parts = []
    for b0 in range(0, nb, sz):
        k0, _k1, p = preds_chunk(cache, "residualized", cfg, b0, min(b0 + sz, nb))
        parts.append((k0, p))
    cat = np.concatenate([p for _, p in sorted(parts)])
    d = float(np.max(np.abs(whole - cat)))
    print("CHUNKCHECK %s/%s rf%d nchunks%d: max|whole-chunked|=%.3e %s" % (model, bucket, refit, nchunks, d, "OK" if d < 1e-9 else "MISMATCH"), flush=True)


def do_pcatest(model, bucket, twd, k, nrows):
    """Quick PCA validation on a REAL slice (first NROWS) of the all_buckets walk-forward:
      1. causality  — perturb the LAST exog row, recompute; earlier OOS factors must be identical;
      2. var_expl   — fraction of exog variance the top-K PCs capture on the last window;
      3. QLIKE      — cadence-Ridge (alpha=1) on full slim [HAR,exog,ind] vs PCA-compressed
                      [HAR,K factors,ind]. qp approx qf at K<<n_exog => the cross-section is low-rank."""
    from src.features.transforms.rolling_pca import rolling_pca
    train_win = twd * PERIODS_PER_DAY
    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy")[:nrows], dtype=np.float64)
    y = np.load(f"{b}/y.npy")[:nrows]; base = np.load(f"{b}/base.npy")[:nrows]
    har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
    Xe = X[:, exog]
    f = rolling_pca(Xe, train_win, k, PCA_REFIT)
    xp = Xe.copy(); xp[-1] += 1e3
    fp = rolling_pca(xp, train_win, k, PCA_REFIT)
    leak = float(np.max(np.abs(f[train_win:-PCA_REFIT] - fp[train_win:-PCA_REFIT])))
    w = Xe[nrows - train_win:nrows]; sv = np.linalg.svd(w - w.mean(0), compute_uv=False)
    var_expl = float((sv[:k] ** 2).sum() / (sv ** 2).sum())

    def ql(xs):
        st, co, ic = _cadence_ridge(np.ascontiguousarray(xs), y, train_win, 1.0, PCA_REFIT)
        return _qlike(_cadence_ridge_oos(np.ascontiguousarray(xs), train_win, st, co, ic), y, base, train_win)

    qf = ql(np.hstack([X[:, har], Xe, X[:, ind]]))
    qp = ql(np.hstack([X[:, har], f, X[:, ind]]))
    print("PCATEST %s/%s tw%d K%d rows%d n_exog=%d: leak=%.3e (%s) var_expl=%.3f | ridge_qlike full=%.5f pca=%.5f (d=%+.5f)" % (
        model, bucket, twd, k, nrows, len(exog), leak, "CAUSAL" if leak < 1e-9 else "LEAK!", var_expl, qf, qp, qp - qf), flush=True)


def do_pcawindowsweep(model, bucket, k_list, pca_window_days, pca_refit=2000):
    """Decouple test: LONG (pca_window_days) PCA basis + SHORT model window. Build factors ONCE
    at max(k_list) (PCs are ordered -> factors[:, :K] is the top-K set), then sweep K x model
    window on the low-dim set. ALL windows scored on a FIXED OOS region (1000d start) for
    fairness (Part-6 FIXED_OOS). Controls: raw slim (no PCA) at {250,1000}d. Ridge base alpha=1."""
    from src.features.transforms.rolling_pca import rolling_pca
    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.load(f"{b}/y.npy"); base = np.load(f"{b}/base.npy")
    har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
    oos_start = 1000 * PERIODS_PER_DAY  # fixed OOS [oos_start, n) for every window
    Xraw = np.ascontiguousarray(np.hstack([X[:, har], X[:, exog], X[:, ind]]))
    t0 = time.time()
    maxk = max(k_list)
    factors = rolling_pca(X[:, exog], pca_window_days * PERIODS_PER_DAY, maxk, pca_refit)  # built once
    print("SWEEP built pca-maxK%d (pca_window=%dd, refit=%d) %.0fs; n_exog=%d" % (
        maxk, pca_window_days, pca_refit, time.time() - t0, len(exog)), flush=True)

    def ql(Xs, tw_days):
        tw = tw_days * PERIODS_PER_DAY
        st, co, ic = _cadence_ridge(Xs, y, tw, 1.0, 480)
        preds = _cadence_ridge_oos(Xs, tw, st, co, ic)  # indexed from tw
        pr, tr = apply_duan_smearing(preds[oos_start - tw:], y[oos_start:], base[oos_start:])
        m = (tr > 0) & (pr > 0); r = tr[m] / pr[m]
        return float(np.mean(r - np.log(r) - 1))

    for tw_days in (250, 1000):
        print("SWEEP raw_slim       model_tw%-5dd qlike=%.5f" % (tw_days, ql(Xraw, tw_days)), flush=True)
    for k in k_list:
        Xpca = np.ascontiguousarray(np.hstack([X[:, har], factors[:, :k], X[:, ind]]))
        for tw_days in (60, 125, 250, 500, 1000):
            print("SWEEP pca%-2d_w%dd model_tw%-5dd qlike=%.5f" % (k, pca_window_days, tw_days, ql(Xpca, tw_days)), flush=True)


def do_pca_alpha_compare(model, bucket, pca_window_days=1000, pca_refit=2000, refit=4000):
    """Each feature set at its OWN alpha*: raw-slim-1000d vs pcaK-500d (K=10/20/40), alpha-scanned,
    fair fixed-OOS. Settles 'does PCA beat raw' at PROPER shrinkage (the alpha=1 sweep was unfair:
    raw wants ~100). refit coarse (rank-robust) for speed; absolute level shifts, ordering holds."""
    from src.features.transforms.rolling_pca import rolling_pca
    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.load(f"{b}/y.npy"); base = np.load(f"{b}/base.npy")
    har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
    oos_start = 1000 * PERIODS_PER_DAY
    factors = rolling_pca(X[:, exog], pca_window_days * PERIODS_PER_DAY, 40, pca_refit)

    def ql(Xs, tw_days, alpha):
        tw = tw_days * PERIODS_PER_DAY
        Xs = np.ascontiguousarray(Xs)
        st, co, ic = _cadence_ridge(Xs, y, tw, alpha, refit)
        preds = _cadence_ridge_oos(Xs, tw, st, co, ic)
        pr, tr = apply_duan_smearing(preds[oos_start - tw:], y[oos_start:], base[oos_start:])
        m = (tr > 0) & (pr > 0); r = tr[m] / pr[m]
        return float(np.mean(r - np.log(r) - 1))

    Xraw = np.hstack([X[:, har], X[:, exog], X[:, ind]])
    for a in (10, 100, 1000):
        print("ACMP raw_slim tw1000 a%-5g qlike=%.5f" % (a, ql(Xraw, 1000, a)), flush=True)
    for k in (10, 20, 40):
        Xp = np.hstack([X[:, har], factors[:, :k], X[:, ind]])
        for a in (1, 10, 100):
            print("ACMP pca%-2d   tw500  a%-5g qlike=%.5f" % (k, a, ql(Xp, 500, a)), flush=True)


def do_signalless_scan(bucket, alpha=0.001, l1=0.2, nwin=12):
    """ABSOLUTELY signalless exog = enet zeros it in EVERY rolling window (linearly dead) AND a
    deep lgbm NEVER splits on it in the residual (nonlinearly dead). Reports raw-exog families
    that are FULLY dead (all derived features prunable) — those raw exog can be dropped."""
    import re
    from collections import Counter

    from lightgbm import LGBMRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64); y = np.load(f"{b}/y.npy")
    feats = json.load(open(f"{b}/meta.json"))["feats"]
    p = X.shape[1]; tw = 1000 * PERIODS_PER_DAY; n = len(X)

    def fam(name):
        return re.sub(r"_(avail|active)$", "", re.sub(r"_ma_\d+$", "", name))

    nz = np.zeros(p, dtype=int)  # per-feature count of rolling windows with coef != 0
    for t in np.linspace(tw, n, nwin, dtype=int):
        en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=2000, tol=1e-3).fit(X[t - tw:t], y[t - tw:t])
        nz += en.coef_ != 0.0
    dead_enet = nz == 0

    rg = Ridge(alpha=100).fit(X, y)  # full-sample residual (in-sample; this is a usefulness probe, not a forecast)
    lg = LGBMRegressor(n_estimators=500, max_depth=8, num_leaves=127, learning_rate=0.05,
                       min_child_samples=20, n_jobs=8, verbose=-1, importance_type="split").fit(X, y - rg.predict(X))
    dead_tree = lg.feature_importances_ == 0
    prunable = dead_enet & dead_tree

    print("SIGNALLESS p=%d enet_always_dead=%d tree_never_split=%d PRUNABLE(both)=%d" % (
        p, int(dead_enet.sum()), int(dead_tree.sum()), int(prunable.sum())), flush=True)
    by_fam = {}
    for i, f in enumerate(feats):
        by_fam.setdefault(fam(f), []).append(i)
    full_dead = sorted(k for k, idx in by_fam.items() if all(prunable[j] for j in idx))
    live = sorted(k for k, idx in by_fam.items() if any(not prunable[j] for j in idx))
    print("SIGNALLESS FULLY-DEAD families (%d, prune these): %s" % (len(full_dead), ", ".join(full_dead)), flush=True)
    print("SIGNALLESS SURVIVING families (%d): %s" % (len(live), ", ".join(live)), flush=True)


def do_enet_sparsity(bucket, alpha=0.001, l1=0.2):
    """How many features does the winning enet actually kill? Fit on early/mid/late windows of
    the slim matrix, count zeroed coefficients (the realized sparsity behind the -0.00035 gain)."""
    from sklearn.linear_model import ElasticNet
    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64); y = np.load(f"{b}/y.npy")
    tw = 1000 * PERIODS_PER_DAY; p = X.shape[1]
    for label, t in (("early", tw), ("mid", len(X) // 2), ("late", len(X))):
        en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=3000, tol=1e-4).fit(X[t - tw:t], y[t - tw:t])
        nnz = int(np.sum(en.coef_ != 0.0))
        print("ENET_SPARSITY %-5s a%g l1%g: nonzero=%d/%d  killed=%d (%.0f%%)" % (
            label, alpha, l1, nnz, p, p - nnz, 100.0 * (p - nnz) / p), flush=True)


def do_ebmsmoke(bucket, nblocks=3):
    """Verify EBM runs in the residual pipeline + time one fit (EBM is slow -> per-fit time
    decides chunked-campaign feasibility). Reuses the built lgbm cache (model-independent),
    forces the EBM factory, runs the first nblocks cadence blocks of the residualized backtest."""
    cid = cell_id("lgbm", bucket, 1000, 100.0, 480, "slim")
    cache = load_cache(cid); cache["cell"]["model"] = "ebm"
    cfg = {"learning_rate": 0.02, "max_leaves": 3, "interactions": 10, "max_bins": 256, "max_rounds": 500, "outer_bags": 4}
    t0 = time.time()
    k0, k1, preds = preds_chunk(cache, "residualized", cfg, 0, nblocks)
    dt = time.time() - t0
    print("EBMSMOKE blocks[0:%d] rows[%d:%d] finite=%s total=%.0fs per_fit~%.1fs (407 blocks -> ~%.1f min/trial serial)" % (
        nblocks, k0, k1, bool(np.all(np.isfinite(preds))), dt, dt / nblocks, dt / nblocks * 407 / 60), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "enet_masks":
        do_enet_masks(sys.argv[2])
    elif mode == "chunk_task":
        do_chunk_task(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif mode == "chunk_collect":
        do_chunk_collect(sys.argv[2], sys.argv[3])
    elif mode == "signalless_scan":
        do_signalless_scan(sys.argv[2])
    elif mode == "enet_sparsity":
        do_enet_sparsity(sys.argv[2])
    elif mode == "ebmsmoke":
        do_ebmsmoke(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif mode == "dimreduce_cell":
        do_dimreduce_cell(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif mode == "pca_alpha_compare":
        do_pca_alpha_compare(sys.argv[2], sys.argv[3])
    elif mode == "pcawindowsweep":
        do_pcawindowsweep(sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")], int(sys.argv[5]),
                          int(sys.argv[6]) if len(sys.argv) > 6 else 2000)
    elif mode == "pcatest":
        do_pcatest(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    elif mode == "alphascan":
        rf = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
        do_alphascan(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], [float(a) for a in sys.argv[6].split(",")], rf)
    elif mode == "chunkcheck":
        do_chunkcheck(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7], int(sys.argv[8]))
    elif mode == "prep":
        pipe = sys.argv[7] if len(sys.argv) > 7 else "slim"
        base_kind = sys.argv[8] if len(sys.argv) > 8 else "ridge"
        do_prep(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), pipe, base_kind)
    elif mode == "trial":
        do_trial(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif mode == "selfcheck":
        pipe = sys.argv[8] if len(sys.argv) > 8 else "slim"
        do_selfcheck(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), pipe)
    else:
        raise SystemExit(f"unknown mode {mode}")
