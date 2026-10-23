"""Replay the full-set ridge / lasso / elastic-net coefficients of one window W of the 2014-start window sweep.

RULE_fiw.txt (written before this code ran).  For W = --wdays (1,000 .. 3,500 days):

  * origins: the first row of every 21st session counted from the first 2014-01-02 row (the FI cadence restarted at the
    sweep's first evaluation day), through the end of the data;
  * replay (primary): the walk's own coefficients at each origin t -- the walk replayed with the CURRENT runner code
    (experiments/pool_linear_pretune.py: STANDARDIZE_PENALTY in scaled coordinates, the binomial availability rule,
    WALK_KKT_RESEED, all at their defaults, as the sweep ran) from the last 250-solve tune boundary g <= t through t,
    through the spec's MultiStageBacktest fast path, at the W's frozen penalty (as results/rehaul_fi_pooled13_2026-10-22
    fi13_replay.py, with the 2014-start sweep's design cache and walk layout: OOS row i <-> design row W + i, tune
    boundaries at OOS rows i % 250 == 0, every chunk of the walk starting at one);
  * batch (secondary S1): the cold-seed fit at t at a common penalty -- a fresh PFTuned on [t - W, t) through the same
    fast path with one forecast (identifiability mask at t, standardized seed): the walk's state had t been a tune
    boundary -- at lasso 1e-4 / 1e-3, elastic net 1e-3 / 1e-2, ridge 1e3 / 1e4 / 1e4 x W / 1,000.

Stored per origin: theta (the coefficient vector of the walk's state, intercept last), the scales s and offsets m of the
scaled coordinates (lasso / elastic net: held since the last seed; ridge: s = the window s.d. of its solve at t, from
the walk's own moments), the mask, the forecasts g .. t (replay) and the raw window s.d. of every column at t.
gamma_j (the coefficient on the standardized column): lasso / enet theta_j; ridge theta_j s_j.

Gate (here, and again by the reducer on the committed file): every replayed forecast row equals forecasts.parquet
pred_adj (model, all_features) of this W.

Usage (CARC; env as the sweep's stage.sh): python fiw_replay.py --wdays W [--workers N] [--limit K] [--what replay,batch]
"""

from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_v] = "1"

import argparse
import concurrent.futures as cf
import contextlib
import io
import json
import pickle
import platform
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("FIW_BASE", "/scratch1/jc_905/fi_window"))
CODE = Path(os.environ.get("FIW_CODE", str(BASE / "code")))
TUNE_PER = 250
R_BB = 0.01  # penalty factor on the 12 HAR + calendar columns (all_features)
FIRST_DAY = "2014-01-02"
EVERY = 21
MODELS = {
    "ridge": "ridge",
    "lasso": "reclasso",
    "enet": "reclasticnet",
}  # paper name -> runner / spec grid name
EX_FLAGS = [
    "DROP_ACTIVITY_MA=1",
    "CALENDAR_THROUGH_OVERNIGHT=1",
    "DROP_OVERNIGHT_INDICATOR=1",
    "LEAN_GLOBAL_LAGS=1",
    "LATE_START_WARMUP=1",
]  # the orchestrator's CFG + the sweep's stage.sh --ex-flag
COMMON: dict[str, list[float | str]] = {
    "lasso": [1e-4, 1e-3],
    "enet": [1e-3, 1e-2],
    "ridge": [1e3, 1e4, "rel"],
}  # S1; "rel" = 1e4 x W / 1,000

_S: dict = {}


def wdir(wd: int) -> Path:
    return BASE / f"w{wd}"


def common_alpha(m: str, a, wd: int) -> float:
    return 1e4 * wd / 1000.0 if a == "rel" else float(a)


def init_worker(wd: int) -> None:
    os.environ["PLP_ROOT"] = str(CODE)
    for p in (str(CODE / "experiments"), str(CODE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import pool_linear_pretune as PLP

    PLP.configure(
        Namespace(
            sample="rth13",
            har_base="5",
            drop_fomc=False,
            keep_fomc_release=False,
            window_days=wd,
            ex_flag=EX_FLAGS,
        )
    )
    assert PLP.STANDARDIZE_PENALTY is True
    dz = wdir(wd) / "design_all_features"
    meta = json.loads((dz / "design.json").read_text())
    PLP.check_meta(meta, "all_features")
    names = list(meta["names"])
    ns = {}
    for m, est in MODELS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            ns[m] = PLP.spec_section(est, "all_features", 0, -1, 0)
    import src.models.reclasso_har as RH

    assert RH.WALK_KKT_RESEED is True
    X = np.load(dz / "X.npy", mmap_mode="r")
    assert bool(X.flags.f_contiguous) == meta["f_order"]
    y = np.load(dz / "y.npy")
    fr = json.loads((wdir(wd) / "frozen_penalties.json").read_text())
    bb = PLP.backbone_mask(names)
    assert int(bb.sum()) == 12, int(bb.sum())
    _S.update(
        PLP=PLP,
        RH=RH,
        ns=ns,
        X=X,
        y=y,
        W=PLP.W,
        names=names,
        bb=bb,
        ev_ref=PLP.evid_ref(names, bb),
        av=PLP.avail_k(names, bb),
        frozen={m: float(fr[m]["all_features"]) for m in MODELS},
    )


def grid_for(m: str, a: float) -> list:
    """The walk's Base.grid: the spec's grid if a is on it, else (the orchestrator's --grid-alphas) [(kind, a, l1)]."""
    spec = _S["ns"][m]["ESTIMATOR_GRIDS"][MODELS[m]]
    if any(g[1] == a for g in spec):
        return spec
    kind, _, l1 = spec[0]
    return [(kind, float(a), l1)]


def run_from(m: str, a: float, g: int, i: int):
    """The walk from OOS row g (a cold start: init_window = mask + seed at g) through OOS row i at penalty a, the runner's
    walk() set-up; returns (forecasts g .. i, the regressor after its solve at i, the class)."""
    PLP, ns, X, y, W = _S["PLP"], _S["ns"][m], _S["X"], _S["y"], _S["W"]
    Base = ns["RollingTunedLinear"]
    Cls = PLP.make_class(ns, R_BB, PLP.ConstSched(a), g, [])
    Base.grid = grid_for(m, a)
    Cls.bb, Cls.ev_ref, Cls.av = _S["bb"], _S["ev_ref"], _S["av"]
    Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
    holder: list = []

    def factory():
        o = Cls()
        holder.append(o)
        return o

    bt = ns["MultiStageBacktest"](
        residualizer=ns["IdentityResidualizer"](),
        regressor_factory=factory,
        refit_frequency=1,
    )
    preds = bt.run(X[g : W + i + 1], y[g : W + i + 1], W, desc="replay")
    assert len(holder) == 1 and len(preds) == i - g + 1
    o = holder[0]
    assert o.alpha_ == a, (o.alpha_, a)
    return preds, o, Cls


def state(o, m: str) -> dict:
    """theta, s, m, mask of the regressor's state after its solve (STANDARDIZE_PENALTY's scaled coordinates)."""
    th = np.asarray(o._th, dtype=np.float64).copy()
    if m == "ridge":
        n = len(o._y)
        var = (
            o._sd_s2 / n - (o._sd_s1 / n) ** 2
        )  # _std_ridge_solve's s, from the walk's own window moments
        s = np.sqrt(np.maximum(var, 0.0))
        s[o._sd_run >= n] = 0.0
        sc = s
    else:
        sc = np.asarray(o._sc, dtype=np.float64).copy()
    return {
        "theta": th,
        "sc": sc,
        "ctr": np.asarray(o._ctr, dtype=np.float64).copy(),
        "mask": o._maskout.copy(),
    }


def job(arg: tuple) -> dict:
    what, m, a, q, i = arg
    t0 = time.time()
    X, W = _S["X"], _S["W"]
    if what == "replay":
        a = _S["frozen"][m]
        g = (i // TUNE_PER) * TUNE_PER
    else:
        g = i
    preds, o, Cls = run_from(m, float(a), g, i)
    pred_t = float(o.predict_one(np.asarray(X[W + i], dtype=np.float64)))
    assert pred_t == preds[-1], (pred_t, preds[-1])
    out = dict(
        what=what,
        m=m,
        a=float(a),
        q=q,
        i=i,
        g=g,
        preds=preds if what == "replay" else preds[-1:],
        **state(o, m),
    )
    out["kkt_reseeds"] = len(getattr(Cls, "kkt_reseeds", []))
    out["binom_reseeds"] = len(getattr(Cls, "std_binom_reseeds", []))
    out["evid_reseeds"] = len(getattr(Cls, "std_evid_reseeds", []))
    if what == "replay" and m == "ridge":
        out["sd_raw"] = np.asarray(X[i : i + W], dtype=np.float64).std(
            axis=0
        )  # the FI convention's window s.d.
    out["sec"] = time.time() - t0
    return out


def origins(wd: int) -> pd.DataFrame:
    d = pd.read_csv(
        wdir(wd) / "design_all_features" / "dates_oos.csv", parse_dates=["date"]
    )["date"]
    i0 = int(np.flatnonzero(d >= FIRST_DAY)[0])
    day = d.dt.normalize().to_numpy()
    first_rows = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    first_rows = first_rows[first_rows >= i0]
    assert first_rows[0] == i0
    sess = np.arange(0, len(first_rows), EVERY)
    rows = first_rows[sess]
    nxt = np.r_[rows[1:], len(d)]
    return pd.DataFrame(
        {
            "k": np.arange(len(rows)),
            "session_index": sess,
            "oos_row": rows,
            "origin": d.iloc[rows].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
            "tune_boundary": (rows // TUNE_PER) * TUNE_PER,
            "block_rows": nxt - rows,
            "block_sessions": [len(np.unique(day[a:b])) for a, b in zip(rows, nxt)],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wdays", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--what", default="replay,batch")
    ap.add_argument(
        "--shard",
        default="",
        help="k/n: only jobs j with j % n == k, results to out/parts",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="assemble the shards of this W, gate, write the npz",
    )
    a = ap.parse_args()
    wd = a.wdays
    O = origins(wd)
    if a.limit:
        O = O.iloc[: a.limit]
    whats = a.what.split(",")
    fr = json.loads((wdir(wd) / "frozen_penalties.json").read_text())
    jobs: list[tuple] = []
    for m in ("lasso", "enet", "ridge"):  # the heaviest first
        if "replay" in whats:
            jobs += [
                ("replay", m, None, int(q), int(i))
                for q, i in zip(O["k"], O["oos_row"])
            ]
        if "batch" in whats:
            for c in COMMON[m]:
                jobs += [
                    ("batch", m, common_alpha(m, c, wd), int(q), int(i))
                    for q, i in zip(O["k"], O["oos_row"])
                ]
    t0 = time.time()
    res: dict = {}
    parts = BASE / "out" / "parts"
    tag = "" if not a.limit else "_probe"
    if a.merge:  # the shards' results (--shard k/n) of this W, every job exactly once
        for pth in sorted(parts.glob(f"w{wd}{tag}_s*.pkl")):
            for r in pickle.loads(pth.read_bytes()):
                rkey = (
                    r["what"],
                    r["m"],
                    r["a"] if r["what"] == "batch" else None,
                    r["q"],
                )
                if (
                    rkey in res
                ):  # W = 1,000: ridge "rel" (1e4 x W / 1,000) is the 1e4 job twice
                    assert np.array_equal(res[rkey]["theta"], r["theta"]), rkey
                    continue
                res[rkey] = r
        want = {(w_, m_, a_, q_) for w_, m_, a_, q_, _ in jobs}
        assert set(res) == want, (len(res), len(want), list(want - set(res))[:5])
        print(
            f"[{time.strftime('%H:%M:%S')}] W {wd}: merged {len(res)} jobs", flush=True
        )
    else:
        if a.shard:
            ks, ns_ = (int(v) for v in a.shard.split("/"))
            jobs = [j for n_, j in enumerate(jobs) if n_ % ns_ == ks]
        print(
            f"[{time.strftime('%H:%M:%S')}] W {wd}: {len(O)} origins {O['origin'].iloc[0]} .. "
            f"{O['origin'].iloc[-1]}, {len(jobs)} jobs (shard {a.shard or 'all'}), {a.workers} workers, "
            f"host {platform.node()}",
            flush=True,
        )
        with cf.ProcessPoolExecutor(
            max_workers=a.workers, initializer=init_worker, initargs=(wd,)
        ) as ex:
            futs = [ex.submit(job, j) for j in jobs]
            for n, f in enumerate(cf.as_completed(futs), 1):
                r = f.result()
                res[
                    (
                        r["what"],
                        r["m"],
                        r["a"] if r["what"] == "batch" else None,
                        r["q"],
                    )
                ] = r
                if n % 50 == 0 or n == len(futs):
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {n}/{len(futs)} jobs, {time.time() - t0:.0f}s",
                        flush=True,
                    )
        if a.shard:
            parts.mkdir(parents=True, exist_ok=True)
            ks = int(a.shard.split("/")[0])
            (parts / f"w{wd}{tag}_s{ks:03d}.pkl").write_bytes(
                pickle.dumps(list(res.values()))
            )
            print(
                f"shard {a.shard} of W {wd} done in {time.time() - t0:.0f}s", flush=True
            )
            return
    # gate: the replayed rows against the published walk (forecasts.parquet of this W)
    fc = pd.read_parquet(
        wdir(wd) / "forecasts.parquet",
        columns=["model", "predictor_set", "date", "pred_adj"],
    )
    d = pd.read_csv(
        wdir(wd) / "design_all_features" / "dates_oos.csv", parse_dates=["date"]
    )["date"]
    z: dict = {
        "k": O["k"].to_numpy(),
        "oos_row": O["oos_row"].to_numpy(),
        "origin": O["origin"].to_numpy().astype(str),
    }
    gates: dict = {}
    for m in MODELS:
        if "replay" in whats:
            f = fc[(fc.model == m) & (fc.predictor_set == "all_features")].set_index(
                "date"
            )["pred_adj"]
            rows = [res[("replay", m, None, int(q))] for q in O["k"]]
            mx, nrows, mx_t = 0.0, 0, 0.0
            for r in rows:
                walk = f.reindex(d.iloc[r["g"] : r["i"] + 1]).to_numpy()
                assert np.isfinite(walk).all(), (m, r["q"])
                dif = np.abs(r["preds"] - walk)
                mx, nrows, mx_t = (
                    max(mx, float(dif.max())),
                    nrows + len(dif),
                    max(mx_t, float(dif[-1])),
                )
            gates[m] = {
                "frozen_alpha": _frozen(fr, m),
                "replay_rows": nrows,
                "max_abs_all_rows": mx,
                "max_abs_at_origins": mx_t,
                "pass_1e12": bool(mx <= 1e-12),
                "sec": float(sum(r["sec"] for r in rows)),
                "kkt_reseeds": int(sum(r["kkt_reseeds"] for r in rows)),
                "binom_reseeds": int(sum(r["binom_reseeds"] for r in rows)),
                "evid_reseeds": int(sum(r["evid_reseeds"] for r in rows)),
            }
            for key in ("theta", "sc", "ctr", "mask"):
                z[f"replay_{m}_{key}"] = np.vstack([r[key] for r in rows])
            z[f"replay_{m}_g"] = np.array([r["g"] for r in rows])
            z[f"replay_{m}_pred_t"] = np.array([r["preds"][-1] for r in rows])
            z[f"replay_{m}_preds"] = np.concatenate([r["preds"] for r in rows])
            z[f"replay_{m}_preds_len"] = np.array([len(r["preds"]) for r in rows])
            if m == "ridge":
                z["sd_raw"] = np.vstack([r["sd_raw"] for r in rows])
            print(m, json.dumps(gates[m]), flush=True)
        if "batch" in whats:
            for c in COMMON[m]:
                al = common_alpha(m, c, wd)
                rows = [res[("batch", m, al, int(q))] for q in O["k"]]
                btag = f"batch_{m}_{al:g}"
                for key in ("theta", "sc", "ctr", "mask"):
                    z[f"{btag}_{key}"] = np.vstack([r[key] for r in rows])
                z[f"{btag}_pred_t"] = np.array([r["preds"][-1] for r in rows])
                gates[btag] = {
                    "alpha": al,
                    "sec": float(sum(r["sec"] for r in rows)),
                    "kkt_reseeds": int(sum(r["kkt_reseeds"] for r in rows)),
                }
    out = BASE / "out"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"fiw_w{wd}{tag}.npz", **z)
    import numpy as _np

    (out / f"fiw_w{wd}{tag}.json").write_text(
        json.dumps(
            {
                "wdays": wd,
                "W_rows": _S.get("W") or wd * 13,
                "n_origins": len(O),
                "first_origin": str(O["origin"].iloc[0]),
                "last_origin": str(O["origin"].iloc[-1]),
                "gates": gates,
                "wall_sec": time.time() - t0,
                "workers": a.workers,
                "host": platform.node(),
                "numpy": _np.__version__,
                "python": platform.python_version(),
                "env": {
                    k: os.environ.get(k, "")
                    for k in (
                        "OPENBLAS_CORETYPE",
                        "NPY_DISABLE_CPU_FEATURES",
                        "SLURM_JOB_ID",
                    )
                },
            },
            indent=1,
        )
    )
    O.to_csv(out / f"origins_w{wd}{tag}.csv", index=False)
    print(f"done W {wd} in {time.time() - t0:.0f}s", flush=True)


def _frozen(fr: dict, m: str) -> float:
    return float(fr[m]["all_features"])


if __name__ == "__main__":
    main()
