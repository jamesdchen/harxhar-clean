"""Linear rungs of the feature-count ladder (RULE.txt): ridge, lasso, elastic net.

The walk's own state at an origin on a column subset: the walk (experiments/pool_linear_pretune.py through
results/fi_by_window_2014start_2026-10-22/code/fiw_replay.py's set-up) replayed from the last 250-solve tune boundary
g <= i through i at the FULL SET's frozen penalty, r = 0.01 on the 12 HAR-benchmark columns, the backbone mask /
evidence references / availability columns recomputed from the subset's names; then that state's predict_one on every
row of the origin's block.

    python ladder_linear.py rank [--workers 3]
        the full-set state at r_0 (2013-01-02 10:00) for the three models; beta_j = f(e_j) - f(0) of its predict_one;
        I_s = sum over the series' 12 columns of |beta_j| x sd_j (s.d. over the 2013 rows) -> WORK/rank_linear.json
    python ladder_linear.py run --model ridge|lasso|enet [--workers N] [--gate-arms 1]
        every distinct rung set (full, har, P<k>, S<k>) x 126 origins (+ the GL2 arms add_iv / add_fomc)
        -> WORK/linear/<model>/<rung>/k###.npz (atomic; resumable)

CARC (fi_by_window's fiw_job.sh environment): python /scratch1/jc_905/wsweep/venv, OPENBLAS_CORETYPE=Prescott, AVX512
numpy targets off, the runner code at /scratch1/jc_905/fi_window/code (FIW_CODE).
"""

from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from argparse import Namespace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

import ladder_common as C  # noqa: E402

CODE = Path(os.environ.get("FIW_CODE", "/scratch1/jc_905/fi_window/code"))
TUNE_PER = 250
EST_OF = {"ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet"}
EX_FLAGS = [
    "DROP_ACTIVITY_MA=1",
    "CALENDAR_THROUGH_OVERNIGHT=1",
    "DROP_OVERNIGHT_INDICATOR=1",
    "LEAN_GLOBAL_LAGS=1",
    "LATE_START_WARMUP=1",
]
PLP_SHA256 = "83ecf62810a204a4927fce4fdbaceebba8040f854021aa51c70951a4c7ef84c2"
_S: dict[str, Any] = {}


def init_worker() -> None:
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
            window_days=C.W_DAYS,
            ex_flag=EX_FLAGS,
        )
    )
    assert PLP.STANDARDIZE_PENALTY is True and PLP.W == C.W
    meta = json.loads((C.LIN_D / "design.json").read_text())
    PLP.check_meta(meta, "all_features")
    ns = {}
    for m, est in EST_OF.items():
        with contextlib.redirect_stdout(io.StringIO()):
            ns[m] = PLP.spec_section(est, "all_features", 0, -1, 0)
    import src.models.reclasso_har as RH

    assert RH.WALK_KKT_RESEED is True
    X = np.load(C.LIN_D / "X.npy", mmap_mode="r")
    assert bool(X.flags.f_contiguous) == meta["f_order"] and X.shape == (C.N_ROWS, 564)
    _S.update(PLP=PLP, ns=ns, X=X, y=np.load(C.LIN_D / "y.npy"), names=list(meta["names"]), cache=(None, None))


def subset(rid: str, cols: np.ndarray) -> np.ndarray:
    """The rung's design: the full cache itself for the full set; else a column subset in the cache's (F) order, as the
    ablation run's subset caches (one held per worker)."""
    X = _S["X"]
    if len(cols) == X.shape[1]:
        return X
    key, arr = _S["cache"]
    if key != rid:
        arr = np.asfortranarray(X[:, cols]) if _S["X"].flags.f_contiguous else np.ascontiguousarray(X[:, cols])
        _S["cache"] = (rid, arr)
    return arr


def grid_for(m: str, a: float) -> list:
    spec = _S["ns"][m]["ESTIMATOR_GRIDS"][EST_OF[m]]
    if any(g[1] == a for g in spec):
        return spec
    kind, _, l1 = spec[0]
    return [(kind, float(a), l1)]


def replay(m: str, rid: str, cols: np.ndarray, i: int):
    """The walk on the rung's columns from the tune boundary g = (i // 250) x 250 through OOS row i; returns (forecasts
    g .. i, the regressor after its solve at i, g, the class)."""
    PLP, ns, y = _S["PLP"], _S["ns"][m], _S["y"]
    X = subset(rid, cols)
    nm = [_S["names"][j] for j in cols]
    W, a = C.W, C.FROZEN[m]
    g = (i // TUNE_PER) * TUNE_PER
    bb = PLP.backbone_mask(nm)
    assert int(bb.sum()) == 12, int(bb.sum())
    Base = ns["RollingTunedLinear"]
    Cls = PLP.make_class(ns, C.R_BB, PLP.ConstSched(a), g, [])
    Base.grid = grid_for(m, a)
    Cls.bb, Cls.ev_ref, Cls.av = bb, PLP.evid_ref(nm, bb), PLP.avail_k(nm, bb)
    Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
    holder: list = []

    def factory():
        o = Cls()
        holder.append(o)
        return o

    bt = ns["MultiStageBacktest"](
        residualizer=ns["IdentityResidualizer"](), regressor_factory=factory, refit_frequency=1
    )
    preds = bt.run(X[g : W + i + 1], y[g : W + i + 1], W, desc="replay")
    assert len(holder) == 1 and len(preds) == i - g + 1
    o = holder[0]
    assert o.alpha_ == a, (o.alpha_, a)
    return np.asarray(preds, dtype=np.float64), o, g, Cls


def job(arg: tuple) -> dict:
    m, rid, cols, k, r, r_end, keep_replay = arg
    t0 = time.time()
    X = subset(rid, cols)
    i = r - C.W
    preds, o, g, Cls = replay(m, rid, cols, i)
    block = np.array([float(o.predict_one(np.asarray(X[j], dtype=np.float64))) for j in range(r, r_end)])
    assert block[0] == preds[-1], (block[0], preds[-1])
    rec = dict(
        m=m,
        rid=rid,
        k=k,
        r=r,
        r_end=r_end,
        g=g,
        pred=block,
        n_cols=len(cols),
        kkt_reseeds=len(getattr(Cls, "kkt_reseeds", [])),
        binom_reseeds=len(getattr(Cls, "std_binom_reseeds", [])),
        evid_reseeds=len(getattr(Cls, "std_evid_reseeds", [])),
        sec=time.time() - t0,
        host=platform.node(),
    )
    if keep_replay:
        rec["replay"] = preds
    out = C.WORK / "linear" / m / rid
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"k{k:03d}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **rec)
    os.replace(tmp, out / f"k{k:03d}.npz")
    return dict(m=m, rid=rid, k=k, sec=rec["sec"])


def rank_job(m: str) -> dict:
    nm = _S["names"]
    cols = np.arange(len(nm))
    r0 = C.rank_origin_row(C.LIN_D)
    preds, o, g, Cls = replay(m, "full", cols, r0 - C.W)
    M = len(nm)
    f0 = float(o.predict_one(np.zeros(M)))
    beta = np.array([float(o.predict_one(np.eye(1, M, j).ravel())) - f0 for j in range(M)])
    X = _S["X"]
    X13 = np.asarray(X[r0 : C.R_FIRST], dtype=np.float64)
    fit = X13 @ beta + f0
    direct = np.array([float(o.predict_one(x)) for x in X13])
    sd = X13.std(axis=0)
    order = C.series_list(nm)
    ser = np.array([C.series_of(c) for c in nm], dtype=object)
    contrib = np.abs(beta) * sd
    imp = {s: float(contrib[ser == s].sum()) for s in order}
    return dict(
        m=m,
        r0=r0,
        g0=g,
        importance=imp,
        linear_recon_maxabs=float(np.abs(fit - direct).max()),
        n_nonzero_beta_exog=int(((beta != 0) & (ser != None)).sum()),  # noqa: E711
        har_contrib=float(contrib[ser == None].sum()),  # noqa: E711
        live_series=int(sum(bool((sd[ser == s] > 0).any()) for s in order)),
        kkt_reseeds=len(getattr(Cls, "kkt_reseeds", [])),
    )


def rank(a) -> None:
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker) as ex:
        res = list(ex.map(rank_job, C.LINEAR))
    out = dict(
        importance={r["m"]: r["importance"] for r in res},
        detail={r["m"]: {k: v for k, v in r.items() if k != "importance"} for r in res},
        frozen=C.FROZEN,
        r_bb=C.R_BB,
        sec=time.time() - t0,
        host=platform.node(),
        numpy=np.__version__,
        env={k: os.environ.get(k, "") for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "SLURM_JOB_ID")},
    )
    C.WORK.mkdir(parents=True, exist_ok=True)
    tmp = C.WORK / "rank_linear.tmp.json"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, C.WORK / "rank_linear.json")
    for r in res:
        top = C.order_from_importance(
            r["importance"], C.series_list(json.loads((C.LIN_D / "design.json").read_text())["names"])
        )
        C.say(f"rank {r['m']}: recon {r['linear_recon_maxabs']:.1e}, live {r['live_series']}; top 10 {top[:10]}")


def run(a) -> None:
    import hashlib

    assert hashlib.sha256((CODE / "experiments" / "pool_linear_pretune.py").read_bytes()).hexdigest() == PLP_SHA256
    d = C.LIN_D
    nm = C.names(d)
    order = C.series_list(nm)
    o = C.origins(d)
    sets = C.rung_sets(C.primary_order(a.model, order), C.secondary_order(a.model, order))
    fit, alias = C.unique_sets(sets)
    if a.gate_arms:
        fit.update({f"gate_{arm}": ss for arm, ss in C.GATE_ARMS.items()})
    out = C.WORK / "linear" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "sets.json").write_text(json.dumps(dict(sets=sets, alias=alias, fitted=list(fit)), indent=1))
    jobs = []
    for rid, ss in fit.items():  # the full set first (heaviest), rung by rung (one subset array held per worker)
        cols = C.columns_for(nm, ss)
        for row in o.itertuples():
            if (out / rid / f"k{int(row.k):03d}.npz").is_file():
                continue
            jobs.append((a.model, rid, cols, int(row.k), int(row.r), int(row.r_end), rid == "full"))
    C.say(
        f"{a.model}: {len(fit)} fitted sets {list(fit)}; {len(jobs)} jobs, {a.workers} workers, host {platform.node()}"
    )
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for n, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            if n % 50 == 0 or n == len(futs):
                C.say(f"{n}/{len(futs)} jobs, {time.time() - t0:.0f}s (last {r['rid']} k={r['k']} {r['sec']:.1f}s)")
    (out / "run_info.json").write_text(
        json.dumps(
            dict(
                model=a.model,
                frozen=C.FROZEN[a.model],
                r_bb=C.R_BB,
                n_jobs=len(jobs),
                wall=time.time() - t0,
                host=platform.node(),
                numpy=np.__version__,
                python=platform.python_version(),
                env={
                    k: os.environ.get(k, "") for k in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "SLURM_JOB_ID")
                },
            ),
            indent=1,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rank", "run"))
    ap.add_argument("--model", choices=C.LINEAR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate-arms", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "rank":
        rank(a)
    else:
        assert a.model
        run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
