"""Feature importance of the pooled 13-bar linear models, monthly re-estimations (2026-10-22).

Stage 1 (fit): at every origin (the first row of every 21st session from the first 2012-01-03 row) the full-set
ridge (1e3), lasso (1e-3) and elastic net (1e-2, l1_ratio 0.5) are solved on the window rows [t - 13,000, t)
exactly as the walk's cold seed does (experiments/pool_linear_pretune.py make_class -> PFTuned._seed, imported
read-only from the ls_wt worktree): same prescaled design cache, adjusted target, augmented unpenalized intercept,
penalty factor 0.01 on the 12 HAR/calendar columns, and the walk's identifiability mask in force at t (the mask
recomputed at the last 250-solve tune boundary g <= t by PFTuned._recompute_mask on the raw window at g, plus, for
the lasso, the columns the walk added between tunes at its recorded reseeds s in (g, t]).

Gate: x_t' theta_t == the walk's pred_adj (forecasts.parquet) at every origin.

Usage: python fi13_fit.py [--workers N] [--limit K]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from argparse import Namespace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LS = Path(r"C:\Users\james\AppData\Local\Temp\ls_wt")
RUN = "pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel"
LS_RUN = LS / "results" / RUN
DZ = LS_RUN / "_work" / "design_all_features"
MAIN = Path(r"C:\Users\james\CC Allowed\harxhar-0dte-professor")
MAIN_RUN = MAIN / "results" / RUN
SCR = Path(__file__).resolve().parent
W = 13000
TUNE_PER = 250
R_BB = 0.01
FIRST_DAY = "2012-01-03"
EVERY = 21  # sessions between re-estimations ("monthly")
MODELS = {"ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet"}  # paper name -> spec grid
CFG = Namespace(
    sample="rth13",
    har_base="5",
    drop_fomc=False,
    keep_fomc_release=False,
    window_days=1000,
    ex_flag=[
        "DROP_ACTIVITY_MA=1",
        "CALENDAR_THROUGH_OVERNIGHT=1",
        "DROP_OVERNIGHT_INDICATOR=1",
        "LEAN_GLOBAL_LAGS=1",
        "LATE_START_WARMUP=1",
    ],
)

_S: dict = {}


def init_worker() -> None:
    for p in (str(LS / "experiments"), str(LS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import pool_linear_pretune as PLP

    PLP.configure(CFG)
    assert PLP.W == W, PLP.W
    meta = json.loads((DZ / "design.json").read_text())
    PLP.check_meta(meta, "all_features")
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        ns = PLP.spec_section("reclasso", "all_features", 0, -1, 0)
    fr = json.loads((MAIN_RUN / "frozen_penalties.json").read_text())
    Cls = PLP.make_class(ns, R_BB, PLP.ConstSched(fr["lasso"]["all_features"]), 0, [])
    Cls.bb = PLP.backbone_mask(meta["names"])
    assert int(Cls.bb.sum()) == 12, int(Cls.bb.sum())
    spec = {}
    for m, est in MODELS.items():
        a = float(fr[m]["all_features"])
        hit = [g for g in ns["ESTIMATOR_GRIDS"][est] if g[1] == a]
        assert len(hit) == 1, (m, a)
        spec[m] = hit[0]
    X = np.load(DZ / "X.npy", mmap_mode="r")
    assert bool(X.flags.f_contiguous) == meta["f_order"]
    y = np.load(DZ / "y.npy")
    reseed = np.sort(
        np.concatenate(
            [np.load(f)["reseed"] for f in glob.glob(str(LS_RUN / "_work" / "walk" / "all_features" / "lasso" / "c*" / "trace.npz"))]
        )
    )
    _S.update(PLP=PLP, ns=ns, Cls=Cls, spec=spec, X=X, y=y, reseed=reseed, names=meta["names"])


def new_obj(X_win, y_win, mask, kind, alpha, l1):
    """The state RollingTunedLinear.init_window + PFTuned.init_window set up, without the tune (the mask and the
    penalty are given)."""
    Cls = _S["Cls"]
    o = Cls.__new__(Cls)
    Xw = np.asarray(X_win, dtype=np.float64)
    o._X = np.hstack([Xw, np.ones((Xw.shape[0], 1))])
    o._y = np.asarray(y_win, dtype=np.float64).copy()
    o._ptr = 0
    o._n_solve = 0
    o._locked = np.zeros(o._X.shape[1], dtype=bool)
    o._locked[-1] = True
    o._maskout = mask
    pf = np.ones(Xw.shape[1] + 1)
    pf[:-1][Cls.bb] = R_BB
    o._pf = pf
    o.kind_, o.alpha_, o.l1_ = kind, alpha, l1
    return o


def degenerate(Xs_raw: np.ndarray) -> np.ndarray:
    """RollingTunedLinear._degenerate_live's test on the whole window: a column constant on every row, or equal on
    every row to an earlier column (intercept appended; locked / masked filtering left to the caller)."""
    Xs = np.hstack([np.asarray(Xs_raw, dtype=np.float64), np.ones((Xs_raw.shape[0], 1))])
    p = Xs.shape[1]
    const = Xs.max(axis=0) == Xs.min(axis=0)
    dup = np.zeros(p, dtype=bool)
    seen: dict = {}
    for j in range(p):
        key = (Xs[:, j] + 0.0).tobytes()
        if key in seen and any(np.array_equal(Xs[:, j], Xs[:, k]) for k in seen[key]):
            dup[j] = True
        seen.setdefault(key, []).append(j)
    return const | dup


def job(i: int) -> dict:
    PLP, X, y = _S["PLP"], _S["X"], _S["y"]
    t0 = time.time()
    g = (i // TUNE_PER) * TUNE_PER
    kind0, a0, l10 = _S["spec"]["ridge"]
    og = new_obj(X[g : g + W], y[g : g + W], None, kind0, a0, l10)
    og._recompute_mask(og._window()[0])
    m_g = og._maskout.copy()
    del og
    Xw, yw = X[i : i + W], y[i : i + W]
    sd = np.asarray(Xw, dtype=np.float64).std(axis=0)
    out = dict(i=i, g=g, n_mask_tune=int(m_g.sum()), sd=sd)
    xt = np.asarray(X[W + i], dtype=np.float64)
    for m in MODELS:
        t1 = time.time()
        mask = m_g.copy()
        added = []
        if m == "lasso":
            locked = np.zeros(len(mask), dtype=bool)
            locked[-1] = True
            for s in _S["reseed"][(_S["reseed"] > g) & (_S["reseed"] <= i)]:
                gone = degenerate(X[s : s + W]) & ~locked & ~mask
                assert gone.any(), (i, s)  # the walk reseeded there, so some live column degenerated
                mask |= gone
                added.append((int(s), [int(j) for j in np.flatnonzero(gone)]))
        rep0 = dict(PLP.REPAIR)
        kind, a, l1 = _S["spec"][m]
        o = new_obj(Xw, yw, mask, kind, a, l1)
        o._seed()
        th = np.asarray(o._th, dtype=np.float64).copy()
        out[m] = dict(
            theta=th,
            pred=float(o.predict_one(xt)),
            mask=mask.copy(),
            added=added,
            repair={k: PLP.REPAIR[k] - rep0[k] for k in rep0},
            sec=time.time() - t1,
        )
        del o
    out["sec"] = time.time() - t0
    return out


def origins() -> pd.DataFrame:
    d = pd.read_csv(DZ / "dates_oos.csv", parse_dates=["date"])["date"]
    i0 = int(np.flatnonzero(d >= FIRST_DAY)[0])
    day = d.dt.normalize().to_numpy()
    first_rows = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    first_rows = first_rows[first_rows >= i0]
    assert first_rows[0] == i0
    sess = np.arange(0, len(first_rows), EVERY)
    rows = first_rows[sess]
    nxt = np.r_[rows[1:], len(d)]
    return pd.DataFrame(
        dict(
            k=np.arange(len(rows)),
            session_index=sess,
            oos_row=rows,
            design_row=rows + W,
            origin=d.iloc[rows].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
            block_rows=nxt - rows,
            block_sessions=[len(np.unique(day[a:b])) for a, b in zip(rows, nxt)],
            block_last=d.iloc[nxt - 1].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    O = origins()
    if a.limit:
        O = O.iloc[: a.limit]
    t0 = time.time()
    res = {}
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker) as ex:
        futs = {ex.submit(job, int(i)): int(i) for i in O["oos_row"]}
        for n, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            res[r["i"]] = r
            if n % 10 == 0 or n == len(futs):
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(futs)} origins, {time.time() - t0:.0f}s", flush=True)
    rows = [res[int(i)] for i in O["oos_row"]]
    fc = pd.read_parquet(MAIN_RUN / "forecasts.parquet", columns=["model", "predictor_set", "date", "pred_adj"])
    dates = pd.to_datetime(O["origin"])
    z: dict = dict(
        oos_row=O["oos_row"].to_numpy(),
        g=np.array([r["g"] for r in rows]),
        n_mask_tune=np.array([r["n_mask_tune"] for r in rows]),
        sd=np.vstack([r["sd"] for r in rows]),
        sec=np.array([r["sec"] for r in rows]),
    )
    gates = {}
    for m in MODELS:
        f = fc[(fc.model == m) & (fc.predictor_set == "all_features")].set_index("date")["pred_adj"]
        walk = f.reindex(dates).to_numpy()
        assert np.isfinite(walk).all(), m
        pred = np.array([r[m]["pred"] for r in rows])
        ad = np.abs(pred - walk)
        rd = ad / np.abs(walk)
        z[f"{m}_theta"] = np.vstack([r[m]["theta"] for r in rows])
        z[f"{m}_pred"] = pred
        z[f"{m}_walk"] = walk
        z[f"{m}_mask"] = np.vstack([r[m]["mask"] for r in rows])
        z[f"{m}_sec"] = np.array([r[m]["sec"] for r in rows])
        rep = {k: int(sum(r[m]["repair"][k] for r in rows)) for k in rows[0][m]["repair"]}
        gates[m] = dict(
            max_abs=float(ad.max()),
            max_rel=float(rd.max()),
            median_rel=float(np.median(rd)),
            n_rel_gt_1e10=int((rd > 1e-10).sum()),
            worst_origin=str(O["origin"].iloc[int(rd.argmax())]),
            at_tune_boundary_max_rel=float(rd[(z["g"] == z["oos_row"])].max()) if (z["g"] == z["oos_row"]).any() else None,
            n_at_tune_boundary=int((z["g"] == z["oos_row"]).sum()),
            repair=rep,
            reseed_mask_additions=[
                dict(origin=str(O["origin"].iloc[q]), added=r[m]["added"]) for q, r in enumerate(rows) if r[m]["added"]
            ],
        )
        print(m, json.dumps({k: v for k, v in gates[m].items() if k != "reseed_mask_additions"}), flush=True)
    np.savez_compressed(SCR / ("fit.npz" if not a.limit else "fit_probe.npz"), **z)
    O.to_csv(SCR / ("origins.csv" if not a.limit else "origins_probe.csv"), index=False)
    (SCR / ("gates.json" if not a.limit else "gates_probe.json")).write_text(
        json.dumps(dict(gates=gates, wall_sec=time.time() - t0, workers=a.workers), indent=1)
    )
    print(f"done {len(rows)} origins in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
