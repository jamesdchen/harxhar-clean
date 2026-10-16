"""Feature Model Confidence Set (2026-10-08): the candidate 16:00-bar elastic nets, the runner.

User (2026-10-08, the professor's suggestion, replacing the paper's lean set): which exogenous inputs
matter for the 16:00 forecast, answered by a Model Confidence Set (Hansen, Lunde & Nason 2011) over
elastic nets on different input sets, instead of feature importance.

Model: the paper's pre-tuned per-bar elastic net, arm (f) of results/linear_pretune_2026-10-06/
(experiments/close_linear_pretune.py, imported read-only; its functions are called, not copied):
l1_ratio 0.5 on the 16:00-bar design, window 2000 sessions, refit every session through the C port
(experiments/close_exogpen_kernel.c), 250-session blocks with a cold seed at each block start, the HAR +
calendar backbone penalized at r = 1/100 of the exogenous penalty (on HAR + calendar alone every column is
backbone: one penalty, r = 1), alpha pre-tuned on 4 blocked folds x 250 sessions before 2020-01-03 (the
pre-2020 16:00 rows of the last-hour design; criterion = mean validation MSE), then the margin-guarded
light retune at the four block starts in the trade period (the grid neighbours challenge the incumbent on
the two 125-session folds before the block start; switch only if the HAC t of the paired squared-error
difference < -z(0.05 / challengers)).  Every candidate runs that whole procedure on its own design.

Candidates (designs = order-preserving column subsets of the all_features 16:00-bar design of record,
results/close_studies_2026-10-03/forecasts/design_bar1600_all_features.parquet, as every bucket of the
2026-10-06 study is -- checked by `inputs`):
  baseline         HAR + calendar (22 columns)                                anchor
  all_features     every column (628)                                         anchor
  add_<X>          HAR + calendar + every column of series X                  47 models
  loo_<X>          all features minus every column of series X                47 models
  addb_<B>         HAR + calendar + every series of bucket B                  8 models  (2026-10-12)
  loob_<B>         all features minus every series of bucket B                8 models  (2026-10-12)
B runs over the 8 buckets of src/data/loading.py SUBGROUPS that partition the 47 series (moments, liquidity,
market_ew, market_vw, sentiment, implied_vol, vol_demand, fomc; checked by `inputs`).  The bucket designs are
appended after the 96 (kinds addb / loob): the first 96 rows of models.csv are unchanged, and the MCS keeps
to the 96 (the bucket designs are studied in experiments/feature_mcs_report.py's bucket analysis).
X runs over the 49 source series of the feature-importance study (experiments/feature_importance_1530.py
series_of = model_diagnostics_1530.parse_feature; the units of
results/feature_importance_1530_dedup/across_time_refits_series.parquet) minus har_ma and calendar.

Modes (python experiments/feature_mcs_runner.py <mode> ...):
  inputs   (local) the shipped inputs under results/feature_mcs_2026-10-08/_work/inputs/: the all_features
           arrays of the 2026-10-06 study (16:00-bar design of record, pre-2020 16:00 rows of the last-hour
           design), inputs.json, models.csv, ref_armf.npz (the stored arm-f forecasts of the two anchors)
  build    (Linux) compile the kernel unchanged: gcc -O3 -march=haswell -ffp-contract=off (the 2026-10-06
           build flags) against the reference LAPACK / BLAS 3.12 (conda-forge *netlib builds, SONAMEs made
           unique by the ship script so no other BLAS can stand in for them)
  run      --models a,b,.. | --taskfile F --line I ; --workers N: every listed model -> _work/models/<m>.npz
  gate     the anchors vs the stored arm-f forecasts -> _work/gate.csv, GATE_OK (bit for bit, or within the
           documented warm-homotopy tolerance 0.65 % relative with no trade-day position change)
Forecast scale: pred_adj (the adjusted target scale); the plain forecast is pred_adj^2 x B (report).
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import close_linear_pretune as CLP  # noqa: E402

OUT = REPO / "results" / "feature_mcs_2026-10-08"
WORK = OUT / "_work"
INP = WORK / "inputs"
MODELS = (
    WORK / "models"
)  # the elastic net's (the first run); ridge / lasso: models_ridge / models_lasso
KDIR = WORK / "kernel"
EST = "reclasticnet"  # the default estimator (the first run, 2026-10-08)
ESTS = ("reclasticnet", "ridge", "reclasso")  # ridge and lasso added 2026-10-08 (user)
ANCHORS = ("baseline", "all_features")
BUCKETS8 = (
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "sentiment",
    "implied_vol",
    "vol_demand",
    "fomc",
)


def models_dir(est: str) -> Path:
    return MODELS if est == "reclasticnet" else WORK / f"models_{CLP.SHORT[est]}"


def ref_sid(est: str, m: str) -> str:
    """The stored arm-f series of an anchor (results/linear_pretune_2026-10-06/forecasts.parquet)."""
    return f"linear_pretune/{CLP.SHORT[est]}_{m}_f_retune_margin" + (
        "" if m == "baseline" else "_bb"
    )


def gate_names(est: str) -> tuple[Path, Path]:
    sfx = "" if est == "reclasticnet" else f"_{CLP.SHORT[est]}"
    return WORK / f"gate{sfx}.csv", WORK / f"GATE_OK{sfx}"


REF_SID = {m: ref_sid(EST, m) for m in ANCHORS}
TOL_REL = (
    6.5e-3  # the 2026-10-06 study's lasso / elastic net gate tolerance (warm homotopy)
)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ inputs (local)
def inputs() -> None:
    """Copy the all_features arrays of the 2026-10-06 study, check every 2026-10-06 bucket is an
    order-preserving column subset of them, write the candidate list and the anchors' stored arm f."""
    import model_diagnostics_1530 as md

    INP.mkdir(parents=True, exist_ok=True)
    meta = json.loads((CLP.ARR / "meta.json").read_text())
    names = meta["all_features"]["names"]
    arr = {
        k: np.load(CLP.ARR / f"all_features_{k}.npy")
        for k in ("bar_X", "bar_y", "pre_X", "pre_y")
    }
    for b in CLP.BUCKETS:  # the property the candidate designs rely on
        nb = meta[b]["names"]
        idx = [names.index(n) for n in nb]
        assert idx == sorted(idx), b
        assert np.array_equal(
            np.load(CLP.ARR / f"{b}_bar_X.npy"), arr["bar_X"][:, idx]
        ), b
        assert np.array_equal(
            np.load(CLP.ARR / f"{b}_pre_X.npy"), arr["pre_X"][:, idx]
        ), b
        assert meta[b]["backbone"] == meta["all_features"]["backbone"], b
    for k, v in arr.items():
        np.save(INP / f"{k}.npy", np.ascontiguousarray(v))
    bb = meta["all_features"]["backbone"]
    grp = [md.parse_feature(n)[0] for n in names]
    assert {g for n, g in zip(names, grp) if n in bb} == {"har_ma", "calendar"}
    assert [n for n, g in zip(names, grp) if g in ("har_ma", "calendar")] == [
        n for n in names if n in bb
    ]
    series = [g for g in dict.fromkeys(grp) if g not in ("har_ma", "calendar")]
    fi = pd.read_parquet(
        REPO
        / "results"
        / "feature_importance_1530_dedup"
        / "across_time_refits_series.parquet",
        columns=["unit"],
    )
    assert set(fi["unit"].unique()) == set(series) | {"har_ma", "calendar"}, (
        "series differ from the FI study"
    )
    rows = [
        dict(
            model="baseline",
            kind="anchor",
            series="",
            cols=[n for n in names if n in bb],
        ),
        dict(model="all_features", kind="anchor", series="", cols=list(names)),
    ]
    for x in series:
        rows.append(
            dict(
                model=f"add_{x}",
                kind="add",
                series=x,
                cols=[n for n, g in zip(names, grp) if n in bb or g == x],
            )
        )
    for x in series:
        rows.append(
            dict(
                model=f"loo_{x}",
                kind="loo",
                series=x,
                cols=[n for n, g in zip(names, grp) if g != x],
            )
        )
    # the 8 buckets of src/data/loading.py SUBGROUPS (2026-10-12): they must partition the 47 series
    from src.data.loading import SUBGROUPS

    bset = {b: [x for x in series if x in SUBGROUPS[b]] for b in BUCKETS8}
    flat = sum(bset.values(), [])
    assert sorted(flat) == sorted(series) and len(flat) == len(set(flat)), (
        "the 8 buckets do not partition the 47"
    )
    for b in BUCKETS8:
        assert sorted(bset[b]) == sorted(SUBGROUPS[b]), b
    for b in BUCKETS8:
        rows.append(
            dict(
                model=f"addb_{b}",
                kind="addb",
                series=b,
                cols=[n for n, g in zip(names, grp) if n in bb or g in bset[b]],
            )
        )
    for b in BUCKETS8:
        rows.append(
            dict(
                model=f"loob_{b}",
                kind="loob",
                series=b,
                cols=[n for n, g in zip(names, grp) if g not in bset[b]],
            )
        )
    assert rows[0]["cols"] == meta["baseline"]["names"]
    mdf = pd.DataFrame(rows)
    mdf["n_cols"] = mdf["cols"].map(len)
    mdf["n_series_cols"] = [
        0
        if r["kind"] == "anchor"
        else sum(g in bset[r["series"]] for g in grp)
        if r["kind"] in ("addb", "loob")
        else sum(g == r["series"] for g in grp)
        for _, r in mdf.iterrows()
    ]
    mdf["cols"] = mdf["cols"].map(json.dumps)
    mdf[["model", "kind", "series", "n_cols", "n_series_cols", "cols"]].to_csv(
        INP / "models.csv", index=False
    )
    grid = CLP.spec_grid(EST)
    grids = {e: CLP.spec_grid(e) for e in ESTS}
    (INP / "inputs.json").write_text(
        json.dumps(
            dict(
                names=names,
                backbone=bb,
                W=int(meta["_W"]),
                dates=meta["_dates"],
                pre_n=int(meta["all_features"]["pre_n"]),
                grid=grid,
                grids=grids,
                buckets=bset,
                source="results/linear_pretune_2026-10-06/_work/arrays/all_features_*.npy (export design of record)",
            )
        )
    )
    f = pd.read_parquet(
        CLP.OUT / "forecasts.parquet", columns=["series_id", "row", "pred_adj"]
    )
    cat = pd.read_csv(CLP.OUT / "catalogue.csv").set_index("series_id")
    ref = {}
    for e in ESTS:  # <short>_<anchor>_pred / _alphas; the elastic net's also unprefixed (the first run's keys)
        for m in ANCHORS:
            sid = ref_sid(e, m)
            s = f[f["series_id"] == sid].sort_values("row")
            assert len(s) == len(meta["_dates"]) - meta["_W"], sid
            ref[f"{CLP.SHORT[e]}_{m}_pred"] = s["pred_adj"].to_numpy(np.float64)
            ref[f"{CLP.SHORT[e]}_{m}_alphas"] = np.array(
                json.loads(cat.loc[sid, "alphas_by_block"]), float
            )
            if e == "reclasticnet":
                ref[f"{m}_pred"] = ref[f"enet_{m}_pred"]
                ref[f"{m}_alphas"] = ref[f"enet_{m}_alphas"]
    # the trade days for the gate's position check: forecast row of each of the 866 days, the 16:00 diurnal
    # baseline B and the 15:30 implied variance (close_linear_pretune.Scorer: plain = pred^2 B, q = sign(plain - iv))
    S = CLP.Scorer()
    fday = pd.DatetimeIndex(CLP.as_day(pd.to_datetime(meta["_dates"][meta["_W"] :])))
    ref["day_row"] = fday.get_indexer(S.days).astype(np.int64)
    assert (ref["day_row"] >= 0).all() and len(ref["day_row"]) == 866
    ref["day_B"] = S.B.reindex(S.days).to_numpy(float)
    ref["day_iv"] = S.iv
    pl = S.plain_of(ref["all_features_pred"])
    assert np.allclose(
        pl,
        ref["all_features_pred"][ref["day_row"]] ** 2 * ref["day_B"],
        rtol=1e-15,
        atol=0,
    )
    np.savez(INP / "ref_armf.npz", **ref)
    say(
        f"inputs: {len(mdf)} models ({len(series)} series), arrays {arr['bar_X'].shape} / {arr['pre_X'].shape}"
    )


# ============================================================================ kernel
def kernel_path() -> Path:
    if (
        os.name == "nt"
    ):  # the 2026-10-06 study's own build (MinGW gcc 15, -ffp-contract=off, netlib DLLs)
        return CLP.NETLIB / "close_exogpen_kernel_nofma.dll"
    return KDIR / "close_exogpen_kernel_nofma.so"


def build() -> None:
    """The kernel source unchanged, the 2026-10-06 flags, against the renamed reference libraries."""
    import subprocess

    nl = KDIR / "netlib"
    for f in ("libnetlib_lapack.so", "libnetlib_blas.so"):
        assert (nl / f).is_file(), (
            f"{nl / f} missing (cluster/slurm/ship_feature_mcs_carc.sh prepares it)"
        )
    cmd = [
        "gcc",
        "-O3",
        "-march=haswell",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        "-o",
        str(kernel_path()),
        str(CLP.KERNEL_SRC),
        f"-L{nl}",
        "-lnetlib_lapack",
        "-lnetlib_blas",
        f"-Wl,-rpath,{nl}",
        "-lm",
    ]
    subprocess.run(cmd, check=True)
    ver = subprocess.run(
        ["gcc", "--version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    (KDIR / "build.json").write_text(
        json.dumps(
            dict(
                cmd=cmd,
                gcc=ver,
                host=socket.gethostname(),
                date=time.strftime("%Y-%m-%d %H:%M"),
            )
        )
    )
    say("compiled: " + " ".join(cmd) + f"  ({ver})")


_LIB = None


def kernel():
    """Load the kernel (RTLD_DEEPBIND on Linux: its BLAS / LAPACK calls bind to its own dependencies)
    with close_linear_pretune.kernel()'s signature; installed as CLP.kernel so CLP.krun uses it."""
    global _LIB
    if _LIB is not None:
        return _LIB
    import ctypes

    from numpy.ctypeslib import ndpointer

    p = kernel_path()
    if os.name == "nt":
        os.add_dll_directory(str(CLP.NETLIB))
        lib = ctypes.CDLL(str(p))
    else:
        lib = ctypes.CDLL(str(p), mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    f8 = ndpointer(np.float64, flags="C_CONTIGUOUS")
    i4 = ndpointer(np.int32, flags="C_CONTIGUOUS")
    u1 = ndpointer(np.uint8, flags="C_CONTIGUOUS")
    il = ndpointer(
        np.dtype(f"i{ctypes.sizeof(ctypes.c_long)}"), flags="C_CONTIGUOUS"
    )  # C long
    ci = ctypes.c_int
    lib.exogpen_run.restype = ci
    lib.exogpen_run.argtypes = [
        ci,
        ci,
        f8,
        f8,
        ci,
        ci,
        ci,
        ci,
        ci,
        i4,
        ci,
        u1,
        u1,
        f8,
        ci,
        f8,
        ci,
        ci,
        i4,
        ci,
        f8,
        ci,
        ci,
        f8,
        f8,
        i4,
        f8,
        i4,
        f8,
        f8,
        i4,
        il,
        f8,
    ]
    _LIB = lib
    return lib


def smoke() -> None:
    """Load the kernel first, then one batch elastic net through it (run under LD_DEBUG=bindings by the
    canary: the kernel's BLAS / LAPACK symbols must bind to the renamed reference libraries)."""
    kernel()
    CLP.kernel = kernel
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(
        np.hstack([rng.standard_normal((300, 8)), np.ones((300, 1))])
    )
    y = np.ascontiguousarray(X[:, :3].sum(axis=1) + rng.standard_normal(300))
    st = CLP.structure(X, np.zeros(9, bool), 1.0)
    th = CLP.batch_theta(X, y, EST, 1e-2, st)
    say(f"smoke: theta {np.round(th, 4).tolist()}; libs {kernel_libs()}")


# the BLAS / LAPACK routines the compiled kernel references (dgemm_ is declared in the source but unused)
KERNEL_SYMS = ("dsyrk_", "dgesv_", "dgelsd_", "dgesdd_", "dpotrf_", "dpotrs_")


def bindings(files: list[str], tag: str = "") -> bool:
    """Parse LD_DEBUG=bindings output of `smoke`: every BLAS / LAPACK symbol the kernel calls must bind to
    the renamed reference libraries, and nothing the kernel or those libraries reference may bind to an
    OpenBLAS (or any other BLAS) object.  -> _work/bindings.csv; True when clean."""
    import re

    pat = re.compile(
        r"binding file (\S+) \[\d+\] to (\S+) \[\d+\]: normal symbol `([^']+)'"
    )
    rows = []
    for f in files:
        for ln in Path(f).read_text(errors="replace").splitlines():
            m = pat.search(ln)
            if m and ("close_exogpen" in m.group(1) or "libnetlib_" in m.group(1)):
                rows.append(dict(src=m.group(1), dst=m.group(2), sym=m.group(3)))
    df = pd.DataFrame(rows, columns=["src", "dst", "sym"])
    WORK.mkdir(parents=True, exist_ok=True)
    df.to_csv(WORK / f"bindings{tag}.csv", index=False)
    kern = df[df["src"].str.contains("close_exogpen")]
    ok_k = all(
        ((kern["sym"] == s) & kern["dst"].str.contains("libnetlib_")).any()
        and not ((kern["sym"] == s) & ~kern["dst"].str.contains("libnetlib_")).any()
        for s in KERNEL_SYMS
    )
    blas_like = df["sym"].str.fullmatch(
        r"[sdcz][a-z0-9]+_|lsame_|xerbla_|ilaenv_|[sd]lamch_"
    )
    bad = df[blas_like & ~df["dst"].str.contains("libnetlib_")]
    say(
        f"bindings: {len(df)} from the kernel / reference libraries; kernel symbols "
        f"{'all' if ok_k else 'NOT all'} bound to libnetlib_*; BLAS-like symbols bound elsewhere: {len(bad)}"
    )
    if len(bad):
        print(bad.head(20).to_string(index=False))
    return bool(ok_k and len(bad) == 0 and len(df) > 0)


def kernel_libs() -> list[str]:
    """The BLAS / LAPACK shared objects mapped into this process (Linux; recorded with each model)."""
    try:
        with open("/proc/self/maps") as fh:
            libs = {
                ln.split()[-1]
                for ln in fh
                if any(k in ln for k in ("lapack", "blas", "close_exogpen"))
            }
        return sorted(libs)
    except OSError:
        return []


# ============================================================================ one candidate
_IN: dict = {}


def load_inputs() -> dict:
    if not _IN:
        meta = json.loads((INP / "inputs.json").read_text())
        _IN.update(meta)
        for k in ("bar_X", "bar_y", "pre_X", "pre_y"):
            _IN[k] = np.load(INP / f"{k}.npy")
        md = pd.read_csv(INP / "models.csv")
        _IN["models"] = {
            r["model"]: dict(
                kind=r["kind"],
                series=r["series"] if isinstance(r["series"], str) else "",
                cols=json.loads(r["cols"]),
            )
            for _, r in md.iterrows()
        }
        # spec_grid() reads the shipped grids, not the 2026-10-06 cache
        for e, g in meta.get("grids", {EST: meta["grid"]}).items():
            CLP._GRIDS[e] = list(g)
    return _IN


def model_arrays(cols: list[str]) -> dict:
    """close_linear_pretune.arrays() for a column subset of the all_features arrays."""
    inp = load_inputs()
    names = inp["names"]
    idx = [names.index(c) for c in cols]
    assert idx == sorted(idx)
    Xb = inp["bar_X"][:, idx]
    Xp = inp["pre_X"][:, idx]
    bb = np.isin(cols, inp["backbone"])
    return dict(
        bar_Xa=np.ascontiguousarray(np.hstack([Xb, np.ones((len(Xb), 1))])),
        bar_y=np.ascontiguousarray(inp["bar_y"]),
        pre_Xa=np.ascontiguousarray(np.hstack([Xp, np.ones((len(Xp), 1))])),
        pre_y=np.ascontiguousarray(inp["pre_y"]),
        bb_aug=np.append(bb, False),
        W=int(inp["W"]),
        pre_n=int(inp["pre_n"]),
        dates=pd.to_datetime(inp["dates"]),
    )


def fit_val(A: dict, Xa, y, fit, val, alpha: float, r: float, est: str = EST):
    """close_linear_pretune.fold_task's fit_val: the spec's batch solution on the fit rows (structure of
    the fit block), squared errors / MSE / QLIKE on the validation rows."""
    a_, b_ = fit
    st = CLP.structure(Xa[a_:b_], A["bb_aug"], r)
    th = CLP.batch_theta(
        np.ascontiguousarray(Xa[a_:b_]), np.ascontiguousarray(y[a_:b_]), est, alpha, st
    )
    v0, v1 = val
    pv = Xa[v0:v1] @ th
    yv = y[v0:v1]
    mse, ql = CLP.val_losses(pv, yv)
    return (pv - yv) ** 2, mse, ql


def walk(A: dict, r: float, alpha: float, est: str = EST) -> dict:
    """close_linear_pretune.walk_task's one-alpha walk (structure at every block start, cold seed)."""
    Xa, y, W = A["bar_Xa"], A["bar_y"], A["W"]
    n_test = len(Xa) - W
    bl = CLP.blocks_of(n_test)
    m = Xa.shape[1]
    lk = np.zeros((len(bl), 1, m), np.uint8)
    mk = np.zeros((len(bl), 1, m), np.uint8)
    pf = np.ones((len(bl), 1, m))
    for k, (i0, _) in enumerate(bl):
        lk[k, 0], mk[k, 0], pf[k, 0] = CLP.structure(Xa[i0 : i0 + W], A["bb_aug"], r)
    bstart = np.array([i0 for i0, _ in bl] + [n_test], np.int32)
    o = CLP.krun(Xa, y, W, est, bstart, lk, mk, pf, [alpha])
    return dict(
        pred=o["pred"],
        n_reseed=o["n_reseed"],
        n_singular=o["n_singular"].astype(np.int64),
        events=o["events"],
        masked=np.array([int(mk[k, 0].sum()) for k in range(len(bl))]),
        cpu_sec=o["cpu_sec"],
    )


def run_model(name: str, est: str = EST) -> dict:
    """Arm f of the 2026-10-06 study on one candidate's design -> models_dir(est)/<name>.npz (+ .json)."""
    MD = models_dir(est)
    out = MD / f"{name}.npz"
    if out.is_file():
        return dict(model=name, cached=True)
    t0, c0 = time.time(), time.process_time()
    inp = load_inputs()
    spec = inp["models"][name]
    A = model_arrays(spec["cols"])
    r = CLP.R_ONE if name == "baseline" else CLP.R_BB
    grid = CLP.spec_grid(est)
    na = len(grid)
    # ---- pre-tune: every grid alpha on the 4 pre-2020 folds (close_linear_pretune.fold_task)
    pf_ = CLP.pre_folds(A["pre_n"])
    pre = [
        [
            fit_val(A, A["pre_Xa"], A["pre_y"], fd["fit"], fd["val"], a, r, est)
            for fd in pf_
        ]
        for a in grid
    ]
    T = dict(
        pre_mse=np.array([[x[1] for x in row] for row in pre]),
        pre_qlike=np.array([[x[2] for x in row] for row in pre]),
    )
    c_pre = time.process_time() - c0
    P = CLP.pick(T)
    i_pick = P["i_mse"]
    # ---- light retune, margin rule: the two 125-session folds before each retune row, fitted only for the
    # incumbent and its grid neighbours (the only entries close_linear_pretune.light_retune reads)
    W = A["W"]
    rows = CLP.retune_rows(A["dates"], W)
    T["rt_rows"] = np.array(rows)
    T["rt_e2"] = np.full((na, len(rows), 2 * CLP.VAL_TAIL), np.nan)
    fitted = np.zeros((na, len(rows)), bool)
    inc = i_pick
    for k, row in enumerate(rows):
        cands = [i for i in (inc - 1, inc + 1) if 0 <= i < na]
        for j in [inc, *cands]:
            parts = [
                fit_val(
                    A, A["bar_Xa"], A["bar_y"], fd["fit"], fd["val"], grid[j], r, est
                )
                for fd in CLP.retune_folds(W, row)
            ]
            T["rt_e2"][j, k] = np.concatenate([p_[0] for p_ in parts])
            fitted[j, k] = True
        # the decision, exactly as light_retune takes it, to know the next row's incumbent
        _, lg = CLP.light_retune(
            dict(rt_rows=T["rt_rows"][k : k + 1], rt_e2=T["rt_e2"][:, k : k + 1]),
            grid,
            inc,
            "margin",
        )
        inc = grid.index(lg[0]["in_force"])
    sched, rlog = CLP.light_retune(
        T, grid, i_pick, "margin"
    )  # the study's own function on the fitted entries
    assert grid[sched[-1]] == grid[inc]
    bl = CLP.blocks_of(len(A["bar_Xa"]) - W)
    alphas = []
    for i0, _ in bl:  # close_linear_pretune.plan()
        done = [k for k, row in enumerate(T["rt_rows"]) if row <= i0]
        alphas.append(grid[sched[done[-1]]] if done else grid[i_pick])
    c_fold = time.process_time() - c0 - c_pre
    # ---- the one-alpha walks the schedule uses, stitched block by block (close_linear_pretune.stitch)
    walks = {a: walk(A, r, a, est) for a in sorted(set(alphas))}
    pred = np.empty(bl[-1][1])
    for (i0, i1), a in zip(bl, alphas):
        pred[i0:i1] = walks[a]["pred"][i0:i1]
    c_walk = sum(w["cpu_sec"] for w in walks.values())
    MD.mkdir(parents=True, exist_ok=True)
    tmp = MD / f"{name}.tmp.npz"
    np.savez_compressed(
        tmp,
        pred=pred,
        alphas_blk=np.array(alphas),
        r=np.array(r),
        grid=np.array(grid),
        i_pick=np.array(i_pick),
        pre_mse=T["pre_mse"],
        pre_qlike=T["pre_qlike"],
        rt_rows=T["rt_rows"],
        rt_e2=T["rt_e2"],
        rt_fitted=fitted,
        walk_alphas=np.array(sorted(walks)),
        **{
            f"walk{k}_{f}": w[f]
            for k, (_, w) in enumerate(sorted(walks.items()))
            for f in ("pred", "n_reseed", "n_singular", "masked")
        },
    )
    os.replace(tmp, out)
    info = dict(
        model=name,
        est=est,
        kind=spec["kind"],
        series=spec["series"],
        n_cols=len(spec["cols"]),
        r=r,
        pick=grid[i_pick],
        alphas=alphas,
        retune_log=rlog,
        n_fold_fits=int(len(grid) * len(pf_) + 2 * fitted.sum()),
        cpu_pre=c_pre,
        cpu_retune=c_fold,
        cpu_walk=c_walk,
        cpu_total=time.process_time() - c0,
        wall=time.time() - t0,
        host=socket.gethostname(),
        cpu=cpu_model(),
        numpy=np.__version__,
        python=platform.python_version(),
        kernel=str(kernel_path()),
        libs=kernel_libs(),
    )
    (MD / f"{name}.json").write_text(json.dumps(info, default=float))
    return info


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _init_worker() -> None:
    kernel()  # before anything imports scipy.linalg
    CLP.kernel = kernel


def run(names: list[str], workers: int, est: str = EST) -> None:
    load_inputs()
    _init_worker()
    unknown = [n for n in names if n not in _IN["models"]]
    assert not unknown, unknown
    todo = [n for n in names if not (models_dir(est) / f"{n}.npz").is_file()]
    todo.sort(key=lambda n: -len(_IN["models"][n]["cols"]))  # widest designs first
    say(
        f"run {CLP.SHORT[est]}: {len(names)} models, {len(todo)} to do, {workers} processes; kernel {kernel_path()}"
    )
    t0 = time.time()
    if workers <= 1:
        for n in todo:
            i = run_model(n, est)
            say(
                f"  {n}: alphas {i['alphas']} pick {i['pick']} cpu {i['cpu_total']:.0f}s"
            )
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as ex:
        futs = {ex.submit(run_model, n, est): n for n in todo}
        for k, fu in enumerate(as_completed(futs)):
            i = fu.result()
            say(
                f"  [{k + 1}/{len(todo)}] {i['model']}: alphas {i.get('alphas')} pick {i.get('pick')} "
                f"cpu {i.get('cpu_total', 0):.0f}s ({time.time() - t0:.0f}s)"
            )


# ============================================================================ gate
def gate(est: str = EST) -> bool:
    """The anchors run here vs the stored arm-f forecasts of results/linear_pretune_2026-10-06/."""
    sh = CLP.SHORT[est]
    MD = models_dir(est)
    inp = load_inputs()
    ref = np.load(INP / "ref_armf.npz")
    n_test = len(inp["dates"]) - int(inp["W"])
    rows, ok_all = [], True
    for m in ANCHORS:
        z = np.load(MD / f"{m}.npz")
        info = json.loads((MD / f"{m}.json").read_text())
        mine, st = z["pred"], ref[f"{sh}_{m}_pred"]
        assert len(mine) == len(st) == n_test
        rel = np.abs(mine / st - 1)
        same_alpha = bool(np.array_equal(z["alphas_blk"], ref[f"{sh}_{m}_alphas"]))
        bitwise = bool(np.array_equal(mine, st))
        blk = sorted({int(i // CLP.TUNE_PER) for i in np.flatnonzero(rel > 1e-9)})
        dr = ref["day_row"]
        q_mine = np.where(mine[dr] ** 2 * ref["day_B"] > ref["day_iv"], 1, -1)
        q_ref = np.where(st[dr] ** 2 * ref["day_B"] > ref["day_iv"], 1, -1)
        pos_diff = int((q_mine != q_ref).sum())
        tol = 1e-8 if est == "ridge" else TOL_REL  # the 2026-10-06 study's tolerances
        ok = same_alpha and (bitwise or (float(rel.max()) < tol and pos_diff == 0))
        ok_all &= ok
        rows.append(
            dict(
                model=m,
                est=sh,
                ref=ref_sid(est, m),
                bitwise=bitwise,
                max_rel=float(rel.max()),
                sessions_gt_1e9=int((rel > 1e-9).sum()),
                blocks=json.dumps(blk),
                position_diff_days=pos_diff,
                same_alphas=same_alpha,
                alphas=json.dumps([float(a) for a in z["alphas_blk"]]),
                host=info["host"],
                cpu=info["cpu"],
                libs=json.dumps(info["libs"]),
                ok=ok,
            )
        )
    df = pd.DataFrame(rows)
    gfile, flag = gate_names(est)
    df.to_csv(gfile, index=False)
    print(df.drop(columns=["libs"]).to_string(index=False))
    if ok_all:
        flag.write_text(df.to_json(orient="records"))
    elif flag.is_file():
        flag.unlink()
    say(f"gate {sh} {'PASSED' if ok_all else 'FAILED'}")
    return ok_all


CANARY = (
    "baseline",
    "all_features",
    "loo_vix",
    "add_vix",
)  # the gate's anchors + one cost probe of each kind


def tasks(n_lines: int) -> None:
    """cluster/feature_mcs_tasks_canary.txt (one line) and _fleet.txt (n_lines lines of comma-separated
    models): the leave-one-out designs (wide, slow) dealt round-robin, then the add-one designs."""
    md = pd.read_csv(INP / "models.csv")
    rest = [m for m in md["model"] if m not in CANARY]
    loo = [m for m in rest if m.startswith("loo_")]
    add = [m for m in rest if not m.startswith("loo_")]
    lines: list[list[str]] = [[] for _ in range(n_lines)]
    for k, m in enumerate(loo + add):
        lines[k % n_lines].append(m)
    cl = REPO / "cluster"
    (cl / "feature_mcs_tasks_canary.txt").write_bytes(
        (",".join(CANARY) + "\n").encode()
    )
    (cl / "feature_mcs_tasks_fleet.txt").write_bytes(
        "".join(",".join(x) + "\n" for x in lines).encode()
    )
    assert sorted(sum(lines, []) + list(CANARY)) == sorted(md["model"])
    say(
        f"tasks: canary {len(CANARY)} models; fleet {n_lines} lines of {[len(x) for x in lines]}"
    )


def tasks2() -> None:
    """Ridge and lasso (2026-10-08): cluster/feature_mcs_tasks2_canary.txt (per estimator one line
    "<est> <models>": the anchors + the two cost probes) and _fleet.txt (per estimator one line: the other 92)."""
    md = pd.read_csv(INP / "models.csv")
    rest = [m for m in md["model"] if m not in CANARY]
    rest = [m for m in rest if m.startswith("loo_")] + [
        m for m in rest if not m.startswith("loo_")
    ]
    cl = REPO / "cluster"
    can = "".join(f"{e} {','.join(CANARY)}\n" for e in ("ridge", "reclasso"))
    flt = "".join(f"{e} {','.join(rest)}\n" for e in ("ridge", "reclasso"))
    (cl / "feature_mcs_tasks2_canary.txt").write_bytes(can.encode())
    (cl / "feature_mcs_tasks2_fleet.txt").write_bytes(flt.encode())
    say(
        f"tasks2: canary {len(CANARY)} and fleet {len(rest)} models per estimator (ridge, lasso)"
    )


def tasks3() -> None:
    """The 16 bucket designs (2026-10-12) for the three linear estimators: cluster/feature_mcs_tasks3_fleet.txt
    (one line per estimator) and the canary file: each estimator's two anchors (re-run and gated in the job)."""
    md = pd.read_csv(INP / "models.csv")
    bk = [m for m, k in zip(md["model"], md["kind"]) if k in ("addb", "loob")]
    bk = [m for m in bk if m.startswith("loob_")] + [
        m for m in bk if m.startswith("addb_")
    ]  # wide ones first
    cl = REPO / "cluster"
    (
        cl / "feature_mcs_tasks3_canary.txt"
    ).write_bytes(  # the anchors again: the gate in a fresh root
        "".join(
            f"{e} {','.join(ANCHORS)}\n" for e in ("ridge", "reclasso", "reclasticnet")
        ).encode()
    )
    (cl / "feature_mcs_tasks3_fleet.txt").write_bytes(
        "".join(
            f"{e} {','.join(bk)}\n" for e in ("ridge", "reclasso", "reclasticnet")
        ).encode()
    )
    say(f"tasks3: {len(bk)} bucket designs x 3 linear estimators")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode",
        choices=(
            "inputs",
            "tasks",
            "tasks2",
            "tasks3",
            "build",
            "smoke",
            "bindings",
            "run",
            "gate",
        ),
    )
    ap.add_argument("--est", default=EST, choices=ESTS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--lines", type=int, default=4)
    ap.add_argument("--taskfile", default="")
    ap.add_argument("--line", type=int, default=0)
    ap.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    a = ap.parse_args()
    if a.mode == "inputs":
        inputs()
    elif a.mode == "tasks":
        tasks(a.lines)
    elif a.mode == "tasks2":
        tasks2()
    elif a.mode == "tasks3":
        tasks3()
    elif a.mode == "build":
        build()
    elif a.mode == "smoke":
        smoke()
    elif a.mode == "bindings":
        sys.exit(0 if bindings(a.files, a.tag) else 1)
    elif a.mode == "run":
        names = [n for n in a.models.split(",") if n]
        est = a.est
        if (
            a.taskfile
        ):  # a line: "m1,m2,..." (the default estimator) or "<est> m1,m2,..."
            line = Path(a.taskfile).read_text().splitlines()[a.line - 1].split()
            if len(line) == 2:
                est = line[0]
            names += [n for n in line[-1].split(",") if n]
        run(names, a.workers, est)
    elif a.mode == "gate":
        sys.exit(0 if gate(a.est) else 1)


if __name__ == "__main__":
    main()
