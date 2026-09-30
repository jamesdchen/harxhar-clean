"""The cluster half of experiments/feature_importance_1530.py: the tree refits.

``python experiments/feature_importance_1530_trees.py trees <bucket> <models> <k0> <k1>``
refits k0..k1-1 of LightGBM / XGBoost / random forest exactly as the untuned per-bar
tree spec (specs/causal_tune_trees.py: same params, leaf minimums scaled to the
2000-row window, seed 42, the stored run's thread count, refit every 10 sessions on
the 2000 rows before the anchor) on the design shipped in
results/feature_importance_1530/_work/input_<bucket>.npz, and at every refit records
  (1) MDI: LightGBM gain / XGBoost total_gain / random-forest impurity decrease;
  (2) split count (LightGBM split, XGBoost weight, random-forest internal nodes);
  (3) permutation importance on the CAUSAL held-out tail (the <= 10 sessions this
      refit's model forecasts, up to the next refit): P1 shuffles the unit's values
      among the tail rows, P2 replaces them with rows drawn from the refit's own
      training window; loss = QLIKE (the bar's realized variance vs the plain
      back-transform yhat^2 x diurnal baseline) and squared error in the fit space;
      N_REPEATS draws per unit and scheme, the draws shared by every model of the
      bucket (the linear models of the local stage use the same draws);
  (4) TreeSHAP of the tail rows (path-dependent, as the spec).
On every PROBE_EVERY-th refit a second fit appends three pure-noise columns
(continuous, PROBE_LEVELS-valued, binary) and records the four measures for them.
``... gate <bucket> <models> <k0> <k1>`` prints the refit-vs-stored forecast gaps.
Imports nothing from src/ and reads no data/ file: the input file is the whole input.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _root(env: str, default: str) -> Path:
    """A result root: the env var if set (relative = to the repo), else the first pass's."""
    q = Path(os.environ.get(env, default))
    return q if q.is_absolute() else REPO / q


# FEATIMP_OUT / FEATIMP_TREES re-point a re-run (the de-duplicated per-bar design:
# experiments/feature_importance_1530_dedup.py) at its own root and stored tree runs;
# unset, they are the first pass's.
OUT = _root("FEATIMP_OUT", "results/feature_importance_1530")
WORK = Path(os.environ.get("FEATIMP_WORK", str(OUT / "_work")))
TREES = _root("FEATIMP_TREES", "results/linear_subsection_trees")
# FEATIMP_WINDOW_MASK=1: every tree refit (and probe fit) uses only the columns
# src/models/window_mask.window_keep keeps on its own training window [t - W, t) -- the
# per-bar linear arms' identifiability mask applied to the trees (user decision
# 2026-09-29); MDI, split count and TreeSHAP of the kept columns are scattered back to all
# p columns with 0 for a dropped column (the model never saw it), and permuting a dropped
# column leaves the forecast unchanged (0, exactly).  Unset / 0 = the first pass (no mask).
WINDOW_MASK = os.environ.get("FEATIMP_WINDOW_MASK", "0") == "1"
SEG = "bar1600"
TW = 2000  # sessions in every training window (the per-bar campaign's TRAIN_WIN)
# sessions between tree refits (specs/causal_tune_trees.py REFIT_EVERY)
REFIT_EVERY = 10
BUCKETS = ("baseline", "live_feasible", "all_features")
BUCKET_ID = {b: i for i, b in enumerate(BUCKETS)}  # seeds the shared permutation draws
TREE_MODELS = ("lgbm", "xgb", "rf")
SEED = 42  # the tree spec's model seed
# The professor's example threshold for "highly correlated"; complete linkage makes every
# PAIR inside a cluster at least this correlated (not just a chain of neighbours).
CLUSTER_CORR = 0.8
# Draws per unit and scheme on each tail.  The tail holds <= 10 rows, so the tail itself,
# not the number of draws, is the binding noise; 10 draws per refit x 147 refits average
# 1,470 draws per unit.  The within-refit spread of the draws is saved and reported.
N_REPEATS = 10
PERM_SEED = 20260929  # the permutation draws (shared by every model of a bucket)
# The probe fit doubles a refit's cost; every third refit (49 of 147) is enough to
# average the probes' ranks and is spread over the whole sample.
PROBE_EVERY = 3
PROBE_SEED = 20260930
PROBE_LEVELS = 20  # the probe with a few values: as many as days_to_opex takes (0..19)
PROBE_NAMES = ("probe_continuous", f"probe_{PROBE_LEVELS}_levels", "probe_binary")
MAX_BATCH_ROWS = 20000  # rows per predict call when scoring the permuted tails (memory)
# the untuned per-bar tree spec's hyperparameters (specs/causal_tune_trees.py, copied from
# the tree_expert_00 / tree_expert_16 chunk meta; the leaf minimums scaled there from the
# pooled 24,000-row window to the 2,000-row per-bar window).  The trees stage asserts
# they equal the 'params' saved in each stored run's npz meta.
POOLED_WINDOW_ROWS = 24000
LGBM_PARAMS: dict = {
    "num_leaves": 20,
    "learning_rate": 0.01296404565731986,
    "n_estimators": 392,
    "min_child_samples": max(1, round(98 * TW / POOLED_WINDOW_ROWS)),
    "feature_fraction": 0.7017000121697539,
    "bagging_fraction": 0.5636236714089857,
    "lambda_l1": 1.0413823540826713e-08,
    "lambda_l2": 0.015950303832251992,
    "bagging_freq": 1,
}
XGB_PARAMS: dict = {
    "max_depth": 4,
    "learning_rate": 0.010853165384309853,
    "n_estimators": 339,
    "min_child_weight": 24.86134158090589 * TW / POOLED_WINDOW_ROWS,
    "subsample": 0.6086526390529718,
    "colsample_bytree": 0.6975888053713192,
    "reg_alpha": 1.087850849987857,
    "reg_lambda": 0.00041206887024929935,
    "gamma": 1.1082753230505144e-05,
    "tree_method": "hist",
}
RF_PARAMS: dict = {"n_estimators": 100}
PARAMS = {"lgbm": LGBM_PARAMS, "xgb": XGB_PARAMS, "rf": RF_PARAMS}

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _tree_npz(bucket: str, model: str) -> Path:
    return (
        TREES
        / bucket
        / SEG
        / model
        / f"tw{TW}"
        / "causal_tune_trees"
        / model
        / bucket
        / f"trees_{SEG}.npz"
    )


def n_refits(n_oos: int) -> int:
    return -(-n_oos // REFIT_EVERY)


# ============================================================================ shared
def load_input(bucket: str) -> dict:
    z = np.load(WORK / f"input_{bucket}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["W"] = int(d["W"])
    d["names"] = [str(v) for v in d["names"]]
    d["threads"] = json.loads(str(d["threads"]))
    ptr = d["unit_ptr"]
    d["units"] = [d["unit_cols"][ptr[u] : ptr[u + 1]] for u in range(len(ptr) - 1)]
    d["n_oos"] = len(d["date_oos"])
    return d


def tail_rows(d: dict, k: int) -> tuple[int, int, int]:
    """(i, t, kk): OOS index of refit k's first forecast, its design row, tail length."""
    i = k * REFIT_EVERY
    return i, d["W"] + i, min(REFIT_EVERY, d["n_oos"] - i)


def draws(
    bucket: str, k: int, n_units: int, kk: int, W: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """The permutation draws of refit k, identical for every model of the bucket.
    P1: a permutation of the kk tail rows per (unit, repeat); P2: kk row offsets into the
    refit's training window per (unit, repeat).  Built from Generator.random() only (the
    float stream is the part of numpy's Generator kept stable across versions)."""
    rng = np.random.default_rng([PERM_SEED, BUCKET_ID[bucket], k])
    p1 = np.argsort(rng.random((n_units, N_REPEATS, kk)), axis=2, kind="stable")
    p2 = np.floor(rng.random((n_units, N_REPEATS, kk)) * W).astype(np.int64)
    h = hashlib.md5(
        p1.astype(np.int16).tobytes() + p2.astype(np.int32).tobytes()
    ).hexdigest()
    return p1, p2, h[:12]


def qlike(rv: np.ndarray, yhat: np.ndarray, base: np.ndarray) -> np.ndarray:
    """QLIKE of the variance forecast yhat^2 x base against the realized variance rv."""
    r = rv / (yhat**2 * base)
    return r - np.log(r) - 1.0


# ============================================================================ trees
def make_model(model: str, threads: int):
    if model == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            **LGBM_PARAMS, num_threads=threads, random_state=SEED, verbosity=-1
        )
    if model == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            **XGB_PARAMS, n_jobs=threads, random_state=SEED, verbosity=0
        )
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**RF_PARAMS, random_state=SEED, n_jobs=threads)


def native(model_name: str, m, p: int) -> tuple[np.ndarray, np.ndarray]:
    """(MDI / total gain, split count) per column."""
    if model_name == "lgbm":
        b = m.booster_
        return (
            b.feature_importance(importance_type="gain").astype(np.float64),
            b.feature_importance(importance_type="split").astype(np.float64),
        )
    if model_name == "xgb":
        bst = m.get_booster()
        gain, split = np.zeros(p), np.zeros(p)
        for key, v in bst.get_score(importance_type="total_gain").items():
            gain[int(key[1:])] = v
        for key, v in bst.get_score(importance_type="weight").items():
            split[int(key[1:])] = v
        return gain, split
    split = np.zeros(p)
    for est in m.estimators_:
        f = est.tree_.feature
        split += np.bincount(f[f >= 0], minlength=p)
    return np.asarray(m.feature_importances_, dtype=np.float64), split


def tree_shap(model_name: str, m, X: np.ndarray) -> np.ndarray:
    """TreeSHAP (path-dependent) per row; last column = the expected value (as the spec)."""
    if model_name == "lgbm":
        return np.asarray(m.predict(X, pred_contrib=True), dtype=np.float64)
    if model_name == "xgb":
        import xgboost as xgb

        return np.asarray(
            m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True),
            dtype=np.float64,
        )
    import shap

    ex = shap.TreeExplainer(m)
    sv = np.asarray(ex.shap_values(X, check_additivity=True), dtype=np.float64)
    ev = float(np.ravel(ex.expected_value)[0])
    return np.column_stack([sv, np.full(len(X), ev)])


def window_cols(Xtr: np.ndarray) -> np.ndarray:
    """The columns a refit on the window Xtr may use: all of them without the mask, else
    src/models/window_mask.window_keep (constant columns and exact copies removed)."""
    if not WINDOW_MASK:
        return np.arange(Xtr.shape[1], dtype=np.int64)
    sys.path.insert(0, str(REPO))
    from src.models.window_mask import window_keep

    return window_keep(Xtr)


def sub(A: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """The kept columns of A (A itself when every column is kept: the first pass's arrays)."""
    return A if len(keep) == A.shape[1] else A[:, keep]


def spread(values: np.ndarray, keep: np.ndarray, p: int) -> np.ndarray:
    """Per-kept-column values back on all p columns (0 for dropped ones; a trailing
    expected-value column is carried over): src/models/window_mask.scatter."""
    if len(keep) == p:
        return np.asarray(values, dtype=np.float64)
    sys.path.insert(0, str(REPO))
    from src.models.window_mask import scatter

    return scatter(values, keep, p)


def perm_losses(
    predict,
    Xt: np.ndarray,
    Xtrain: np.ndarray,
    units: list[np.ndarray],
    todo: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    rv: np.ndarray,
    base: np.ndarray,
    yt: np.ndarray,
    q0: float,
    m0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean increase in tail QLIKE and squared error per (scheme, unit) over N_REPEATS
    draws, and the across-draw sd of the QLIKE increase; units not in `todo` stay 0."""
    kk = len(Xt)
    U = len(units)
    dq = np.zeros((2, U))
    dm = np.zeros((2, U))
    sq = np.zeros((2, U))
    per = N_REPEATS * kk
    step = max(1, MAX_BATCH_ROWS // per)
    for s, P in ((0, p1), (1, p2)):
        for a in range(0, len(todo), step):
            us = todo[a : a + step]
            Xb = np.repeat(Xt[None], len(us) * N_REPEATS, axis=0).reshape(
                len(us), N_REPEATS, kk, -1
            )
            for q, u in enumerate(us):
                cols = units[u]
                # (R, kk, |cols|): the tail rows in shuffled order (P1) or rows drawn
                # from the training window (P2); a group's columns move together
                src = (Xt if s == 0 else Xtrain)[:, cols][P[u]]
                Xb[q][:, :, cols] = src
            yh = predict(Xb.reshape(-1, Xt.shape[1])).reshape(len(us), N_REPEATS, kk)
            ql = qlike(rv[None, None], yh, base[None, None]).mean(axis=2)  # (u, R)
            se = ((yt[None, None] - yh) ** 2).mean(axis=2)
            dq[s, us] = ql.mean(axis=1) - q0
            sq[s, us] = ql.std(axis=1, ddof=1)
            dm[s, us] = se.mean(axis=1) - m0
    return dq, dm, sq


def run_trees(bucket: str, models: list[str], k0: int, k1: int) -> None:
    d = load_input(bucket)
    X, y, W, rv, base = d["X"], d["y"], d["W"], d["rv"], d["base"]
    p = X.shape[1]
    units = d["units"]
    U = len(units)
    K = n_refits(d["n_oos"])
    k1 = min(k1, K)
    for mdl in models:
        thr = d["threads"][mdl]
        out = WORK / "trees" / f"trees_{bucket}_{mdl}_{k0:03d}_{k1:03d}.npz"
        if out.exists():
            print(f"{out.name} exists, skipped", flush=True)
            continue
        rec: dict[str, list] = {
            key: []
            for key in (
                "k",
                "pred",
                "pred_gap",
                "mdi",
                "split",
                "shap",
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
                "probe_k",
                "probe_mdi",
                "probe_split",
                "probe_shap_abs",
                "probe_dq",
                "probe_dm",
                "probe_q0",
                "n_keep",
                "kept",
            )
        }
        t_all = time.time()
        for k in range(k0, k1):
            i, t, kk = tail_rows(d, k)
            Xtr, ytr = X[t - W : t], y[t - W : t]
            Xt, yt = X[t : t + kk], y[t : t + kk]
            keep = window_cols(Xtr)  # all p columns unless FEATIMP_WINDOW_MASK=1
            kept = np.zeros(p, dtype=bool)
            kept[keep] = True
            rec["n_keep"].append(len(keep))
            rec["kept"].append(kept)
            a = time.time()
            m = make_model(mdl, thr)
            m.fit(sub(Xtr, keep), ytr)
            rec["fit_sec"].append(time.time() - a)
            pred = m.predict(sub(Xt, keep))
            rec["pred"].append(pred)
            rec["pred_gap"].append(
                float(np.max(np.abs(pred - d[f"stored_{mdl}"][i : i + kk])))
            )
            gain, split = native(mdl, m, len(keep))
            gain, split = spread(gain, keep, p), spread(split, keep, p)
            rec["mdi"].append(gain)
            rec["split"].append(split)
            a = time.time()
            sh = spread(tree_shap(mdl, m, sub(Xt, keep)), keep, p)
            rec["shap_sec"].append(time.time() - a)
            rec["shap"].append(sh.astype(np.float32))
            rec["shap_gap"].append(float(np.max(np.abs(sh.sum(axis=1) - pred))))
            q0 = float(qlike(rv[i : i + kk], pred, base[i : i + kk]).mean())
            m0 = float(((yt - pred) ** 2).mean())
            rec["q0"].append(q0)
            rec["m0"].append(m0)
            p1, p2, h = draws(bucket, k, U, kk, W)
            rec["draw_md5"].append(h)
            used = split > 0
            todo = np.array(
                [u for u in range(U) if used[units[u]].any()], dtype=np.int64
            )
            a = time.time()
            dq, dm, sq = perm_losses(
                (lambda Z, m=m, keep=keep: m.predict(Z[:, keep]))
                if len(keep) < p
                else m.predict,
                Xt,
                Xtr,
                units,
                todo,
                p1,
                p2,
                rv[i : i + kk],
                base[i : i + kk],
                yt,
                q0,
                m0,
            )
            rec["perm_sec"].append(time.time() - a)
            rec["dq"].append(dq)
            rec["dm"].append(dm)
            rec["sq"].append(sq)
            rec["k"].append(k)
            if k % PROBE_EVERY == 0:
                probe(
                    rec,
                    mdl,
                    thr,
                    k,
                    Xtr,
                    ytr,
                    Xt,
                    yt,
                    rv[i : i + kk],
                    base[i : i + kk],
                    W,
                )
            if (k - k0) % 5 == 0 or k == k1 - 1:
                print(
                    f"  {bucket} {mdl} refit {k} (tail {kk}): fit {rec['fit_sec'][-1]:.1f}s "
                    f"perm {rec['perm_sec'][-1]:.1f}s ({len(todo)}/{U} units) "
                    f"shap {rec['shap_sec'][-1]:.1f}s; forecast vs stored {rec['pred_gap'][-1]:.1e}, "
                    f"SHAP additivity {rec['shap_gap'][-1]:.1e}; elapsed {time.time() - t_all:.0f}s",
                    flush=True,
                )
        out.parent.mkdir(parents=True, exist_ok=True)
        versions = {}
        for mod in ("numpy", "sklearn", "lightgbm", "xgboost", "shap"):
            try:
                versions[mod] = __import__(mod).__version__
            except Exception:  # noqa: BLE001 -- a library the model does not use may be absent
                versions[mod] = "absent"
        np.savez_compressed(
            out,
            k=np.array(rec["k"]),
            pred=np.concatenate(rec["pred"]),
            pred_gap=np.array(rec["pred_gap"]),
            mdi=np.array(rec["mdi"]),
            split=np.array(rec["split"]),
            shap=np.concatenate(rec["shap"]),
            shap_gap=np.array(rec["shap_gap"]),
            q0=np.array(rec["q0"]),
            m0=np.array(rec["m0"]),
            dq=np.array(rec["dq"]),
            dm=np.array(rec["dm"]),
            sq=np.array(rec["sq"]),
            fit_sec=np.array(rec["fit_sec"]),
            perm_sec=np.array(rec["perm_sec"]),
            shap_sec=np.array(rec["shap_sec"]),
            draw_md5=np.array(rec["draw_md5"]),
            probe_k=np.array(rec["probe_k"], dtype=np.int64),
            probe_mdi=np.array(rec["probe_mdi"]).reshape(-1, p + len(PROBE_NAMES)),
            probe_split=np.array(rec["probe_split"]).reshape(-1, p + len(PROBE_NAMES)),
            probe_shap_abs=np.array(rec["probe_shap_abs"]).reshape(
                -1, p + len(PROBE_NAMES)
            ),
            probe_dq=np.array(rec["probe_dq"]).reshape(-1, 2, len(PROBE_NAMES)),
            probe_dm=np.array(rec["probe_dm"]).reshape(-1, 2, len(PROBE_NAMES)),
            probe_q0=np.array(rec["probe_q0"]),
            n_keep=np.array(rec["n_keep"], dtype=np.int64),
            kept=np.array(rec["kept"], dtype=bool),
            meta=json.dumps(
                dict(
                    bucket=bucket,
                    model=mdl,
                    k0=k0,
                    k1=k1,
                    threads=thr,
                    params=PARAMS[mdl],
                    n_repeats=N_REPEATS,
                    perm_seed=PERM_SEED,
                    probe_every=PROBE_EVERY,
                    probe_seed=PROBE_SEED,
                    probe_names=PROBE_NAMES,
                    window_mask=WINDOW_MASK,
                    versions=versions,
                    wall_sec=time.time() - t_all,
                )
            ),
        )
        print(
            f"wrote {out} ({k1 - k0} refits, {time.time() - t_all:.0f}s; max forecast gap vs "
            f"stored {max(rec['pred_gap']):.1e})",
            flush=True,
        )


def probe(rec, mdl, thr, k, Xtr, ytr, Xt, yt, rv, base, W) -> None:
    """A second fit with three pure-noise columns appended: what each measure gives them."""
    rng = np.random.default_rng([PROBE_SEED, k])
    n = len(Xtr) + len(Xt)
    noise = np.column_stack(
        [
            rng.random(n),  # continuous: every value distinct
            np.floor(rng.random(n) * PROBE_LEVELS),  # PROBE_LEVELS values
            (rng.random(n) < 0.5).astype(np.float64),  # binary
        ]
    )
    Ptr = np.hstack([Xtr, noise[: len(Xtr)]])
    Pt = np.hstack([Xt, noise[len(Xtr) :]])
    p = Ptr.shape[1]
    keep = window_cols(
        Ptr
    )  # the probe fit's own window mask (the noise is always kept)
    m = make_model(mdl, thr)
    m.fit(sub(Ptr, keep), ytr)
    pred = m.predict(sub(Pt, keep))
    gain, split = native(mdl, m, len(keep))
    gain, split = spread(gain, keep, p), spread(split, keep, p)
    sh = spread(tree_shap(mdl, m, sub(Pt, keep)), keep, p)[:, :-1]
    shap_abs = np.abs(sh).mean(axis=0)
    q0 = float(qlike(rv, pred, base).mean())
    m0 = float(((yt - pred) ** 2).mean())
    npb = len(PROBE_NAMES)
    units = [np.array([p - npb + j]) for j in range(npb)]
    r2 = np.random.default_rng([PROBE_SEED, k, 1])
    kk = len(Pt)
    p1 = np.argsort(r2.random((npb, N_REPEATS, kk)), axis=2, kind="stable")
    p2 = np.floor(r2.random((npb, N_REPEATS, kk)) * W).astype(np.int64)
    dq, dm, _ = perm_losses(
        (lambda Z: m.predict(Z[:, keep])) if len(keep) < p else m.predict,
        Pt,
        Ptr,
        units,
        np.arange(npb),
        p1,
        p2,
        rv,
        base,
        yt,
        q0,
        m0,
    )

    rec["probe_k"].append(k)
    rec["probe_mdi"].append(gain)
    rec["probe_split"].append(split)
    rec["probe_shap_abs"].append(shap_abs)
    rec["probe_dq"].append(dq)
    rec["probe_dm"].append(dm)
    rec["probe_q0"].append(q0)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "trees":
        b, ms, a, e = (
            sys.argv[2],
            sys.argv[3].split(","),
            int(sys.argv[4]),
            int(sys.argv[5]),
        )
        run_trees(b, ms, a, e)
    elif stage == "gate":  # the canary's report: refit forecasts vs the stored ones
        b, ms, a, e = (
            sys.argv[2],
            sys.argv[3].split(","),
            int(sys.argv[4]),
            int(sys.argv[5]),
        )
        bad = []
        for mdl in ms:
            z = np.load(
                WORK / "trees" / f"trees_{b}_{mdl}_{a:03d}_{e:03d}.npz",
                allow_pickle=True,
            )
            g = float(z["pred_gap"].max())
            print(
                f"GATE {b} {mdl}: max |refit forecast - stored forecast| {g:.2e}; "
                f"SHAP additivity {float(z['shap_gap'].max()):.1e}; "
                f"versions {json.loads(str(z['meta']))['versions']}"
            )
            if not g <= 1e-9:
                bad.append(mdl)
        print("GATE " + ("DIFF " + ",".join(bad) if bad else "OK"))
    else:
        raise SystemExit(f"unknown stage {stage!r} (trees | gate)")
