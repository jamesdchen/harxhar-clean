"""Stage 1b (replay): the walk's own coefficients at every origin.

The batch re-estimation of fi13_fit.py matches the walk's forecast to 1.8e-12 (ridge) but not everywhere for the
lasso (31 of 151 origins above 1e-10 relative, max 5.4e-3) and the elastic net (4 of 151, max 8.5e-5): between
tunes the walk slides warm (two Garrigues homotopy updates a row), and its path can end on another solution than
the batch solver's.  The coefficients behind the published forecasts are the walk's, so this stage replays the walk
itself from the last tune boundary g <= t (a cold seed with the mask recomputed: the walk's state at g depends only
on the window at g, the chunk seam the walk was run and gated on) through the origin row t, with the spec's
MultiStageBacktest fast path and pool_linear_pretune.make_class (imported read-only), and takes the regressor's
theta after its solve at t.  Gate: every replayed forecast row g .. t equals the published pred_adj.

Also: the KKT violation (relative to max |X'y|) of the walk's and of the batch theta on the masked window problem
at every origin, to say which one is the exact optimum where they differ.

Usage: python fi13_replay.py [--workers N] [--limit K]
"""

from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fi13_fit as F  # noqa: E402

_C: dict = {}


def init_worker() -> None:
    F.init_worker()
    _C["batch"] = np.load(F.SCR / "fit.npz")


def kkt_violation(Xa, yr, mask, th, pf, a, l1) -> float:
    """Max KKT violation of theta on the masked window problem (sklearn ElasticNet units, intercept free),
    relative to max |c|: r = c - G b - lam2 b; active |r_j - mu_j sign b_j|, inactive max(|r_j| - mu_j, 0)."""
    Xm = Xa.copy()
    Xm[:, mask] = 0.0
    n = len(yr)
    G = Xm.T @ Xm
    c = Xm.T @ yr
    locked = np.zeros(len(th), dtype=bool)
    locked[-1] = True
    mu = np.where(locked, 0.0, n * a * l1 * pf)
    lam2 = np.where(locked, 0.0, n * a * (1.0 - l1) * pf)
    r = c - G @ th - lam2 * th
    on = th != 0.0
    v = np.zeros(len(th))
    v[on] = np.abs(r[on] - mu[on] * np.sign(th[on]))
    v[~on] = np.maximum(np.abs(r[~on]) - mu[~on], 0.0)
    return float(v.max() / max(float(np.abs(c).max()), 1.0))


def job(arg: tuple[str, int, int]) -> dict:
    m, q, i = arg
    S = F._S
    ns, PLP, X, y = S["ns"], S["PLP"], S["X"], S["y"]
    t0 = time.time()
    g = (i // F.TUNE_PER) * F.TUNE_PER
    Base = ns["RollingTunedLinear"]
    kind, a, l1 = S["spec"][m]
    Cls = PLP.make_class(ns, F.R_BB, PLP.ConstSched(a), g, [])
    Cls.bb = S["Cls"].bb
    Base.grid = ns["ESTIMATOR_GRIDS"][F.MODELS[m]]
    Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
    holder: list = []

    def factory():
        o = Cls()
        holder.append(o)
        return o

    bt = ns["MultiStageBacktest"](
        residualizer=ns["IdentityResidualizer"](), regressor_factory=factory, refit_frequency=1
    )
    preds = bt.run(X[g : F.W + i + 1], y[g : F.W + i + 1], F.W, desc="replay")
    assert len(holder) == 1 and len(preds) == i - g + 1
    o = holder[0]
    th = np.asarray(o._th, dtype=np.float64).copy()
    mask = o._maskout.copy()
    pred_t = float(o.predict_one(np.asarray(X[F.W + i], dtype=np.float64)))
    assert pred_t == preds[-1]
    # KKT of the walk's and of the batch theta on the masked window problem at t
    Xa = np.hstack([np.asarray(X[i : i + F.W], dtype=np.float64), np.ones((F.W, 1))])
    yr = np.asarray(y[i : i + F.W], dtype=np.float64)
    pf = o._pf
    thb = _C["batch"][f"{m}_theta"][q]
    mb = _C["batch"][f"{m}_mask"][q]
    if kind == "ridge":
        kw = kb = None  # closed form, no KKT question
    else:
        kw = kkt_violation(Xa, yr, mask, th, pf, a, l1)
        kb = kkt_violation(Xa, yr, mb, thb, pf, a, l1)
    return dict(
        m=m,
        q=q,
        i=i,
        g=g,
        preds=preds,
        theta=th,
        mask=mask,
        mask_same_as_batch=bool(np.array_equal(mask, mb)),
        reseeds=[int(s) + g for s in Base.reseed_trace],
        kkt_walk=kw,
        kkt_batch=kb,
        sec=time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    O = pd.read_csv(F.SCR / "origins.csv")
    if a.limit:
        O = O.iloc[: a.limit]
    jobs = [(m, int(q), int(i)) for m in ("lasso", "enet", "ridge") for q, i in zip(O["k"], O["oos_row"])]
    t0 = time.time()
    res: dict = {}
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for n, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            res[(r["m"], r["q"])] = r
            if n % 25 == 0 or n == len(futs):
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(futs)} replays, {time.time() - t0:.0f}s", flush=True)
    fc = pd.read_parquet(F.MAIN_RUN / "forecasts.parquet", columns=["model", "predictor_set", "date", "pred_adj"])
    d = pd.read_csv(F.DZ / "dates_oos.csv", parse_dates=["date"])["date"]
    z: dict = {}
    gates = {}
    batch = np.load(F.SCR / "fit.npz")
    for m in ("ridge", "lasso", "enet"):
        f = fc[(fc.model == m) & (fc.predictor_set == "all_features")].set_index("date")["pred_adj"]
        rows = [res[(m, int(q))] for q in O["k"]]
        ad_all, ad_t, rel_t, nrows = 0.0, [], [], 0
        for r in rows:
            walk = f.reindex(d.iloc[r["g"] : r["i"] + 1]).to_numpy()
            assert np.isfinite(walk).all()
            dif = np.abs(r["preds"] - walk)
            ad_all = max(ad_all, float(dif.max()))
            nrows += len(dif)
            ad_t.append(float(dif[-1]))
            rel_t.append(float(dif[-1] / abs(walk[-1])))
        th = np.vstack([r["theta"] for r in rows])
        z[f"{m}_theta"] = th
        z[f"{m}_mask"] = np.vstack([r["mask"] for r in rows])
        z[f"{m}_pred"] = np.array([r["preds"][-1] for r in rows])
        thb = batch[f"{m}_theta"][: len(rows)]
        pb, pw = batch[f"{m}_pred"][: len(rows)], z[f"{m}_pred"]
        nzw, nzb = th[:, :-1] != 0.0, thb[:, :-1] != 0.0
        gates[m] = dict(
            replay_vs_published_all_rows_max_abs=ad_all,
            replay_rows_checked=nrows,
            replay_vs_published_origin_max_abs=float(max(ad_t)),
            replay_vs_published_origin_max_rel=float(max(rel_t)),
            batch_vs_walk_origin_pred_max_rel=float(np.max(np.abs(pb - pw) / np.abs(pw))),
            batch_vs_walk_origin_pred_n_rel_gt_1e10=int((np.abs(pb - pw) / np.abs(pw) > 1e-10).sum()),
            batch_vs_walk_theta_max_abs=float(np.abs(thb - th).max()),
            batch_vs_walk_support_differs_origins=int((nzw != nzb).any(axis=1).sum()),
            batch_vs_walk_support_differs_cells=int((nzw != nzb).sum()),
            mask_same_as_batch_all=bool(all(r["mask_same_as_batch"] for r in rows)),
            kkt_walk_max=None if m == "ridge" else float(max(r["kkt_walk"] for r in rows)),
            kkt_batch_max=None if m == "ridge" else float(max(r["kkt_batch"] for r in rows)),
            kkt_walk_gt_1e8_origins=None if m == "ridge" else int(sum(r["kkt_walk"] > 1e-8 for r in rows)),
            kkt_batch_gt_1e8_origins=None if m == "ridge" else int(sum(r["kkt_batch"] > 1e-8 for r in rows)),
            replay_sec_total=float(sum(r["sec"] for r in rows)),
        )
        z[f"{m}_kkt_walk"] = np.array([np.nan if r["kkt_walk"] is None else r["kkt_walk"] for r in rows])
        z[f"{m}_kkt_batch"] = np.array([np.nan if r["kkt_batch"] is None else r["kkt_batch"] for r in rows])
        print(m, json.dumps(gates[m]), flush=True)
    tag = "" if not a.limit else "_probe"
    np.savez_compressed(F.SCR / f"walk_theta{tag}.npz", **z)
    (F.SCR / f"gates_replay{tag}.json").write_text(
        json.dumps(dict(gates=gates, wall_sec=time.time() - t0, workers=a.workers), indent=1)
    )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
