"""The 16:00-bar linear models tuned like the trees: one pre-2020 pre-tune, then a sparse light retune.

User (2026-10-06): "it feels like causally retuning the linear models is introducing a bunch of
noise. can we do the same thing as the trees are doing where we tune on the training data before
the trade starts and then light retuning sparsely?"  And: the HAR + calendar backbone is penalized
SEPARATELY and much more lightly than the other inputs (backbone penalty = 1/100 of the exogenous
penalty; the 2026-10-03 exog_penalty study's ``bbfix`` arm).

Models and walk (unchanged from the spec, specs/causal_tune_linear.py, through the C port of the
2026-10-03 study, experiments/close_exogpen_kernel.c, imported / compiled read-only): ridge, lasso,
elastic net (l1_ratio 0.5) on the 16:00-bar design, window 2000 sessions, refit every session,
intercept unpenalized, the identifiability mask at every re-choice, 250-session blocks (a re-choice
and a cold seed at every block start), the spec's grids (ridge 1e-2 .. 1e3, lasso / elastic net
1e-6 .. 1e-2).  Inside a block the forecast path depends only on (alpha, r), so every arm is a
block-by-block stitch of one-alpha walks (gate: the stitch of the causal choices reproduces the
causal run).

Arms (every arm at the 16:00 bar):
  a_causal          one penalty, alpha re-chosen every 250 sessions on the spec's 125-session tail
                    (the spec; gate: the stored forecasts results/spxw_pnl/yhat_sub_<est>_<bucket>)
  b_causal_bb       backbone at r = 1/100 of the exogenous penalty, causal (= exog_penalty bbfix;
                    gate: its export for all_features)
  c_frozen          one penalty, pre-tuned, frozen
  d_frozen_bb       backbone 1/100, pre-tuned, frozen
  e_retune_any_bb   backbone 1/100, pre-tuned, then a light retune at every block start from the
                    first trade day on, switching to the better grid neighbour on ANY validation gain
  f_retune_margin_bb  the same, switching only when the HAC t of the paired daily squared-error
                    difference is below -z(0.05 / number of challengers) (the tree study's rule)
  *_w               supplement: the same arm on the grid widened one decade beyond the edge its
                    pre-tune pick sits at (only where the spec-grid pick is at an edge)
On ``baseline`` (HAR + calendar only) every column is backbone, so r is moot: arms a, c, e, f with
one penalty.

Pre-tune (mirrors experiments/close_trees_pretune.py): 4 blocked walk-forward folds tiling the last
4 x 250 16:00 sessions before 2020-01-03 (the first trade day), each with a 25-session embargo and
the 2000 sessions before it as the fit block; data = the 16:00 rows of the last-hour design (back to
2004; the 16:00-bar design starts 2010-07 and holds one such fold only); every grid alpha fitted by
the spec's batch solution (ridge solve; FWL + batch homotopy) on each fit block (identifiability
mask from that block), criterion = mean over folds of the validation MSE on the adjusted scale; the
QLIKE-criterion pick (Duan back-transform, the tree study's val_losses) is recorded.  Light retune:
at the block starts on or after the first trade day (2020-06-23, 2021-06-21, 2022-06-17,
2023-06-16), candidates = the incumbent and its grid neighbours, validation = the tree study's two
125-session folds before the block start, each with its own 2000-session fit block and 25-session
embargo.

Scoring (the master-table GUI's plain scorer, writeup/master_table_gui/rows.py): plain forecast
f^2 * B (no recalibration) on the 866 trade days 2020-01-03 .. 2024-04-30; QLIKE; DM (daily QLIKE,
src.evaluation.diebold_mariano.dm_test) vs the HAR + calendar OLS (master table sub_ols_baseline)
and vs the same model's causal arm; sign(s) Sharpe mid / crossed; Newey-West t (lag 14) of the daily
mid P&L difference; days with a different position; robust clip and rank of |s| Sharpe on the 614
days after a 252-day warm-up (rows._score_sizes).  Point estimates only.

Stages:  python experiments/close_linear_pretune.py <stage>
  prep      the per-bucket arrays (16:00-bar design: the export's design of record where the
            bucket's columns equal its columns; pre-2020 16:00 rows of the last-hour design)
  compute   every fold fit and walk (CLP_WORKERS processes, default 6), resumable (npz cache)
  report    gates, scores, logs, CSVs, SUMMARY.md, forecasts.parquet
  all       prep + compute + report
Designs: experiments/close_linear_pretune_design.py (capture_design_close.py without reading
data/spxw_chain.parquet).  Kernel: MinGW gcc -O3 -march=haswell -ffp-contract=off (CLP_MINGW) against
the reference LAPACK 3.12 / BLAS DLLs of conda-forge's liblapack / libblas 3.12.1 *netlib builds, copied
to _work/netlib/ (CLP_LAPACK=openblas: MSYS2's OpenBLAS instead; see kernel()).
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

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import NormalDist  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "linear_pretune_2026-10-06"
WORK = OUT / "_work"
DESIGN_DIR = WORK / "design"
ARR = WORK / "arrays"
WALK = WORK / "walk"
FOLDS = WORK / "folds"
EXPORT = REPO / "results" / "close_studies_2026-10-03" / "forecasts"
SPXW = REPO / "results" / "spxw_pnl"
MASTER = REPO / "results" / "close_master_table" / "master_table.csv"
DAILY = REPO / "results" / "close_master_table" / "master_table_daily.parquet"
QUOTES = REPO / "results" / "atm_straddle_0dte_1530" / "daily_a0.parquet"
KERNEL_SRC = REPO / "experiments" / "close_exogpen_kernel.c"
KERNEL_DLL = WORK / "close_exogpen_kernel.dll"
MINGW = Path(os.environ.get("CLP_MINGW", "C:/msys64/mingw64"))

# priority order (compute queue); moments_vol_vix and lean_vol_vix added 2026-10-06 (user: before
# live_feasible, which is no longer in the paper)
BUCKETS = (
    "all_features",
    "moments_vol_vix",
    "lean_vol_vix",
    "baseline",
    "moments_vix",
    "live_feasible",
)
EST = ("ridge", "reclasso", "reclasticnet")
SHORT = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
LABEL = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "elastic net"}
TW, TUNE_PER, VAL_TAIL, EMBARGO = 2000, 250, 125, 25
R_ONE, R_BB = 1.0, 0.01
RTAG = {R_ONE: "one", R_BB: "bb"}
FIRST_TRADE_DAY = "2020-01-03"
LAST_TRADE_DAY = "2024-04-30"
PRE_FOLDS, PRE_FOLD_LEN = 4, 250  # the tree pre-tune's CTP_FOLDS x CTP_FOLD_LEN
RETUNE_FOLDS = 2  # x VAL_TAIL: the tree study's light-retune validation
MARGIN_ALPHA = 0.05
WARMUP = 252
WORKERS = int(os.environ.get("CLP_WORKERS", "6"))
DESIGN_META = ("row", "date", "forecast_row", "y", "baseline", "true_raw")
ARMS = (
    "a_causal",
    "b_causal_bb",
    "c_frozen",
    "d_frozen_bb",
    "e_retune_any_bb",
    "f_retune_margin_bb",
)
ARM_LABEL = {
    "a_causal": "one penalty, causal re-choice every 250 (spec)",
    "b_causal_bb": "backbone 1/100, causal re-choice every 250",
    "c_frozen": "one penalty, pre-tuned, frozen",
    "d_frozen_bb": "backbone 1/100, pre-tuned, frozen",
    "e_retune_any_bb": "backbone 1/100, pre-tuned + light retune (any gain)",
    "f_retune_margin_bb": "backbone 1/100, pre-tuned + light retune (margin)",
    "e_retune_any": "one penalty, pre-tuned + light retune (any gain)",
    "f_retune_margin": "one penalty, pre-tuned + light retune (margin)",
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ grids
_GRIDS: dict = {}


def spec_grid(est: str) -> list[float]:
    """The spec's grid (specs/causal_tune_linear.py ESTIMATOR_GRIDS through close_exogpen, read-only;
    the cache file keeps pool workers from re-executing the spec section)."""
    if est not in _GRIDS:
        f = ARR / "spec_grids.json"
        if f.is_file():
            _GRIDS.update(json.loads(f.read_text()))
        else:
            import close_exogpen as CE

            g = CE.spec_grids()
            ns = CE.spec_ns()
            assert (ns["TUNE_PER"], ns["VAL_TAIL"], ns["EMBARGO"], ns["TRAIN_WIN"]) == (
                TUNE_PER,
                VAL_TAIL,
                EMBARGO,
                TW,
            )
            ARR.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(g))
            _GRIDS.update(g)
    return [float(a) for a in _GRIDS[est]]


def widened(est: str, side: str) -> list[float]:
    g = spec_grid(est)
    return [g[0] / 10.0] + g if side == "low" else g + [g[-1] * 10.0]


def atag(alpha: float) -> str:
    return f"a{int(round(np.log10(alpha))):+d}"


def blocks_of(n_test: int) -> list[tuple[int, int]]:
    return [(i0, min(i0 + TUNE_PER, n_test)) for i0 in range(0, n_test, TUNE_PER)]


# ============================================================================ prep
def prep() -> None:
    """Arrays for the workers.  16:00-bar design: all_features = the export's design of record
    (results/close_studies_2026-10-03/forecasts/design_bar1600_all_features.parquet, built on the
    AVX-512 host that wrote the stored forecasts); another bucket = those columns of it when the
    bucket's design rebuilt here equals the same columns of all_features rebuilt here, else the
    rebuilt design.  Pre-tune rows: the 16:00 rows of the last-hour design before 2020-01-03."""
    ARR.mkdir(parents=True, exist_ok=True)
    spec_grid("ridge")
    exp = pd.read_parquet(EXPORT / "design_bar1600_all_features.parquet")
    ex_names = [c for c in exp.columns if c not in DESIGN_META]
    Xexp = exp[ex_names].to_numpy(np.float64)
    zall = np.load(DESIGN_DIR / "design_bar1600_all_features.npz")
    all_names = [str(v) for v in zall["names"]]
    gate = []
    dX = np.abs(Xexp - zall["X"])
    gate.append(
        dict(
            gate="design: all_features 16:00 bar rebuilt here vs the export (design of record)",
            item="max abs diff / columns differing / y max abs diff",
            value=f"{dX.max():.2e} / {int((dX.max(0) > 0).sum())} of {dX.shape[1]} / "
            f"{np.abs(exp['y'].to_numpy() - zall['y']).max():.1e}",
            ok=bool(
                all_names == ex_names
                and dX.max() < 1e-9
                and np.array_equal(exp["y"].to_numpy(), zall["y"])
            ),
        )
    )
    zb = np.load(DESIGN_DIR / "design_bar1600_baseline.npz")
    bb_names = [str(v) for v in zb["names"]]
    dates = pd.to_datetime(exp["date"]).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    assert (dates == zall["date"]).all()
    W = int(zall["W"])
    meta = {}
    for b in BUCKETS:
        z = np.load(DESIGN_DIR / f"design_bar1600_{b}.npz")
        names = [str(v) for v in z["names"]]
        assert set(bb_names) <= set(names), b
        same = False
        if set(names) <= set(all_names):
            idx = [all_names.index(n) for n in names]
            same = bool(
                np.array_equal(z["X"], zall["X"][:, idx])
                and np.array_equal(z["y"], zall["y"])
                and (z["date"] == zall["date"]).all()
            )
        X = Xexp[:, idx] if same else np.asarray(z["X"], np.float64)
        src = (
            "export design of record"
            if b == "all_features"
            else "export columns (rebuilt bucket design == the same columns of rebuilt all_features)"
            if same
            else "rebuilt here (AVX2 host)"
        )
        gate.append(
            dict(
                gate=f"design: {b} 16:00 bar source",
                item=src,
                value=f"{X.shape[1]} columns",
                ok=True,
            )
        )
        np.save(ARR / f"{b}_bar_X.npy", np.ascontiguousarray(X))
        np.save(
            ARR / f"{b}_bar_y.npy", np.ascontiguousarray(exp["y"].to_numpy(np.float64))
        )
        q = np.load(DESIGN_DIR / f"design_last30_{b}.npz")
        assert [str(v) for v in q["names"]] == names, b
        d30 = pd.to_datetime(q["date"])
        pre = np.asarray((d30.hour == 16) & (d30 < pd.Timestamp(FIRST_TRADE_DAY)))
        np.save(
            ARR / f"{b}_pre_X.npy", np.ascontiguousarray(q["X"][pre], dtype=np.float64)
        )
        np.save(
            ARR / f"{b}_pre_y.npy", np.ascontiguousarray(q["y"][pre], dtype=np.float64)
        )
        # the two caches' 16:00 rows on their common sessions (recorded; scaling windows differ)
        d16 = pd.DatetimeIndex(d30[np.asarray(d30.hour == 16)])
        common = pd.DatetimeIndex(pd.to_datetime(z["date"])).intersection(d16)
        ia = pd.DatetimeIndex(pd.to_datetime(z["date"])).get_indexer(common)
        ib = pd.DatetimeIndex(d30).get_indexer(common)
        dd = np.abs(np.asarray(z["X"])[ia] - np.asarray(q["X"])[ib])
        meta[b] = dict(
            names=names,
            backbone=bb_names,
            W=W,
            n_bar=int(len(X)),
            source=src,
            pre_n=int(pre.sum()),
            pre_first=str(d30[pre][0])[:10],
            pre_last=str(d30[pre][-1])[:10],
            pre_dates=[str(v)[:10] for v in d30[pre]],
            cache_compare=dict(
                common_sessions=int(len(common)),
                columns_identical=int((dd.max(0) == 0).sum()),
                y_max_abs_diff=float(
                    np.abs(np.asarray(z["y"])[ia] - np.asarray(q["y"])[ib]).max()
                ),
            ),
        )
    meta["_dates"] = [str(v) for v in dates]
    meta["_W"] = W
    (ARR / "meta.json").write_text(json.dumps(meta))
    pd.DataFrame(gate).to_csv(WORK / "gate_design.csv", index=False)
    say(
        "prep done: "
        + "; ".join(
            f"{b}: {meta[b]['source']}, {meta[b]['pre_n']} pre-2020 16:00 rows"
            for b in BUCKETS
        )
    )


# ============================================================================ kernel
_LIB = None
# LAPACK / BLAS behind the kernel: "netlib" = the reference LAPACK 3.12 / BLAS DLLs of conda-forge's
# liblapack / libblas 3.12.1 *netlib builds (flang 21, generic x86-64), copied to _work/netlib/: the
# kind of library the 2026-10-03 runs linked (reference liblapack3 / libblas3); "openblas" = MSYS2's
# OpenBLAS (reproduces the stored lasso only to 0.17 % in blocks 3-4; kept as an option)
NETLIB = WORK / "netlib"
LAPACK = os.environ.get(
    "CLP_LAPACK", "netlib" if (NETLIB / "liblapack.dll").is_file() else "openblas"
)


def kernel():
    """experiments/close_exogpen_kernel.c (unchanged) built as a DLL with MinGW gcc -O3
    -march=haswell (no -ffast-math), rebuilt when the .c is newer than the DLL."""
    global _LIB
    if _LIB is not None:
        return _LIB
    import ctypes
    import subprocess

    from numpy.ctypeslib import ndpointer

    # -ffp-contract=off: no FMA contraction.  With it, this gcc (15.1) on the reference LAPACK reproduces
    # the 2026-10-03 C port (all_features lasso to 7.7e-13, and the elastic net's block-4 branch, 6.5e-3
    # from the stored table as documented there); with contraction the elastic net takes another branch
    # in block 4 (1.8e-2 from the stored table).  Checked 2026-10-06 over six build variants.
    dll = (
        NETLIB / "close_exogpen_kernel_nofma.dll" if LAPACK == "netlib" else KERNEL_DLL
    )
    if not dll.is_file() or dll.stat().st_mtime < KERNEL_SRC.stat().st_mtime:
        dll.parent.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            PATH=str(MINGW / "bin") + os.pathsep + os.environ.get("PATH", ""),
        )
        libs = (
            [str(NETLIB / "liblapack.dll"), str(NETLIB / "libblas.dll")]
            if LAPACK == "netlib"
            else [f"-L{MINGW / 'lib'}", "-lopenblas"]
        )
        cmd = [
            str(MINGW / "bin" / "gcc"),
            "-O3",
            "-march=haswell",
            "-ffp-contract=off",
            "-shared",
            "-o",
            str(dll),
            str(KERNEL_SRC),
            *libs,
            "-lm",
        ]
        subprocess.run(cmd, check=True, env=env)
        say("compiled: " + " ".join(cmd))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(NETLIB if LAPACK == "netlib" else MINGW / "bin"))
    lib = ctypes.CDLL(str(dll))
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


def krun(
    Xa, y, W, est, bstart, lk, mk, pf, grid, val_tail=VAL_TAIL, embargo=EMBARGO
) -> dict:
    """One exogpen_run call (single-penalty mode); arrays sized to the last block end."""
    import ctypes

    lib = kernel()
    nb, n_s, m = lk.shape
    n_out = int(bstart[-1])
    o = dict(
        pred=np.zeros(n_out),
        theta=np.zeros((n_out, m)),
        events=np.zeros(n_out, np.int32),
        val_mse=np.full((nb, n_s, len(grid)), np.nan),
        choice=np.zeros((nb, 2), np.int32),
        pen_g=np.zeros((nb, 8)),
        mpr_mse=np.zeros(nb),
        n_reseed=np.zeros(nb, np.int32),
        n_singular=np.zeros(4, np.dtype(f"i{ctypes.sizeof(ctypes.c_long)}")),
        exact_pred=np.zeros(n_out),
    )
    t0 = time.process_time()
    rc = lib.exogpen_run(
        len(Xa),
        m,
        Xa,
        y,
        W,
        {"ridge": 0, "reclasso": 1, "reclasticnet": 2}[est],
        0,
        1 if est == "reclasso" else 0,
        nb,
        np.ascontiguousarray(bstart, dtype=np.int32),
        n_s,
        np.ascontiguousarray(lk),
        np.ascontiguousarray(mk),
        np.ascontiguousarray(pf),
        len(grid),
        np.ascontiguousarray(grid, dtype=np.float64),
        int(val_tail),
        int(embargo),
        np.zeros(m, np.int32),
        8,
        np.zeros(nb),
        2,
        0,
        o["pred"],
        o["theta"],
        o["events"],
        o["val_mse"],
        o["choice"],
        o["pen_g"],
        o["mpr_mse"],
        o["n_reseed"],
        o["n_singular"],
        o["exact_pred"],
    )
    o["cpu_sec"] = time.process_time() - t0
    if rc != 0:
        raise RuntimeError(f"kernel returned {rc}")
    return o


# ============================================================================ worker data
_A: dict = {}
_ST: dict = {}


def arrays(b: str) -> dict:
    if b not in _A:
        meta = json.loads((ARR / "meta.json").read_text())
        mb = meta[b]
        Xb = np.load(ARR / f"{b}_bar_X.npy")
        Xp = np.load(ARR / f"{b}_pre_X.npy")
        bb = np.isin(mb["names"], mb["backbone"])
        _A[b] = dict(
            bar_Xa=np.ascontiguousarray(np.hstack([Xb, np.ones((len(Xb), 1))])),
            bar_y=np.ascontiguousarray(np.load(ARR / f"{b}_bar_y.npy")),
            pre_Xa=np.ascontiguousarray(np.hstack([Xp, np.ones((len(Xp), 1))])),
            pre_y=np.ascontiguousarray(np.load(ARR / f"{b}_pre_y.npy")),
            bb_aug=np.append(bb, False),
            W=int(mb["W"]),
            pre_n=int(mb["pre_n"]),
            dates=pd.to_datetime(meta["_dates"]),
        )
    return _A[b]


def structure(Xa_window: np.ndarray, bb_aug: np.ndarray, r: float):
    """close_exogpen.structure (read-only): locked set, mask, penalty factors of a window."""
    import close_exogpen as CE

    lk, mk, pf = CE.structure(Xa_window, bb_aug, r)
    return lk.astype(np.uint8), mk.astype(np.uint8), np.asarray(pf, np.float64)


def struct3(st) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lk, mk, pf = st
    m = len(lk)
    return lk.reshape(1, 1, m), mk.reshape(1, 1, m), pf.reshape(1, 1, m)


def batch_theta(
    Xa: np.ndarray, y: np.ndarray, est: str, alpha: float, st
) -> np.ndarray:
    """The spec's batch solution on the rows Xa (the kernel's cold seed: ridge K c; lasso / enet FWL
    + batch homotopy) at alpha with the given structure: a one-block, one-session kernel call whose
    window is Xa, so theta[0] is the seed.  The re-choice step of that call is fed a 2-row fit
    block (val_tail = n - 2, no embargo) and discarded: the seed does not depend on it."""
    n = len(Xa)
    Xk = np.ascontiguousarray(np.vstack([Xa, Xa[-1:]]))
    yk = np.ascontiguousarray(np.append(y, y[-1]))
    lk, mk, pf = struct3(st)
    o = krun(
        Xk,
        yk,
        n,
        est,
        np.array([0, 1], np.int32),
        lk,
        mk,
        pf,
        [alpha],
        val_tail=n - 2,
        embargo=0,
    )
    return o["theta"][0]


def pre_folds(pre_n: int) -> list[dict]:
    """close_trees_pretune.pretune_folds: blocked folds tiling the last PRE_FOLDS x PRE_FOLD_LEN
    pre-2020 sessions, each with a 25-session embargo and the 2000 sessions before it."""
    out = []
    for f in range(PRE_FOLDS):
        v1 = pre_n - (PRE_FOLDS - 1 - f) * PRE_FOLD_LEN
        v0 = v1 - PRE_FOLD_LEN
        a, b = v0 - EMBARGO - TW, v0 - EMBARGO
        assert a >= 0, (pre_n, f)
        out.append(dict(fold=f, fit=(a, b), val=(v0, v1)))
    return out


def retune_rows(dates, W: int) -> list[int]:
    n_test = len(dates) - W
    return [
        i0
        for i0, _ in blocks_of(n_test)
        if dates[W + i0] >= pd.Timestamp(FIRST_TRADE_DAY)
    ]


def retune_folds(W: int, row: int) -> list[dict]:
    """close_trees_pretune.retune_folds: RETUNE_FOLDS folds of VAL_TAIL sessions tiling the sessions
    before design row W + row, each with its own 2000-session fit block and 25-session embargo."""
    t = W + row
    out = []
    for f in range(RETUNE_FOLDS):
        v1 = t - (RETUNE_FOLDS - 1 - f) * VAL_TAIL
        v0 = v1 - VAL_TAIL
        a, b = v0 - EMBARGO - TW, v0 - EMBARGO
        assert a >= 0
        out.append(dict(fold=f, fit=(a, b), val=(v0, v1)))
    return out


def val_losses(pv, yv) -> tuple[float, float]:
    import causal_tune_trees_tuned_jobs as J

    return J.val_losses(np.asarray(pv, float), np.asarray(yv, float))


# ============================================================================ tasks
def walk_file(b: str, est: str, r: float, tag: str) -> Path:
    return WALK / b / f"{SHORT[est]}_{RTAG[r]}_{tag}.npz"


def fold_file(b: str, est: str, r: float, alpha: float) -> Path:
    return FOLDS / b / f"{SHORT[est]}_{RTAG[r]}_{atag(alpha)}.npz"


def walk_task(job: dict) -> dict:
    """The 16:00 walk on the bucket's design: the spec grid (tag 'grid': the causal arm) or one
    alpha (tag a<log10 alpha>: that alpha in every block)."""
    b, est, r, alphas, tag = (
        job["bucket"],
        job["est"],
        job["r"],
        job["alphas"],
        job["tag"],
    )
    f = walk_file(b, est, r, tag)
    if f.is_file():
        return dict(job=job, cached=True, cpu_sec=float(np.load(f)["cpu_sec"]))
    A = arrays(b)
    Xa, y, W = A["bar_Xa"], A["bar_y"], A["W"]
    n_test = len(Xa) - W
    bl = blocks_of(n_test)
    m = Xa.shape[1]
    lk = np.zeros((len(bl), 1, m), np.uint8)
    mk = np.zeros((len(bl), 1, m), np.uint8)
    pf = np.ones((len(bl), 1, m))
    for k, (i0, _) in enumerate(bl):
        key = (b, "bar", i0, r)
        if key not in _ST:
            _ST[key] = structure(Xa[i0 : i0 + W], A["bb_aug"], r)
        lk[k, 0], mk[k, 0], pf[k, 0] = _ST[key]
    bstart = np.array([i0 for i0, _ in bl] + [n_test], np.int32)
    o = krun(Xa, y, W, est, bstart, lk, mk, pf, alphas)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f,
        pred=o["pred"],
        val_mse=o["val_mse"][:, 0, :],
        alphas=np.array(alphas),
        alpha_blk=np.array([alphas[i] for i in o["choice"][:, 1]]),
        n_reseed=o["n_reseed"],
        events=o["events"],
        n_singular=o["n_singular"].astype(np.int64),
        cpu_sec=o["cpu_sec"],
        masked=np.array([int(mk[k, 0].sum()) for k in range(len(bl))]),
    )
    return dict(job=job, cached=False, cpu_sec=o["cpu_sec"])


def fold_task(job: dict) -> dict:
    """Every validation fit of one (bucket, estimator, r, alpha): the 4 pre-tune folds (last-hour
    design), the last pre-tune fold on the 16:00-bar design (cache check), the two light-retune
    folds at each retune row, and the spec's own 125-session tail at each block (gate vs the walk's
    validation MSE).  Squared errors of every validation session are stored."""
    b, est, r, alpha = job["bucket"], job["est"], job["r"], job["alpha"]
    f = fold_file(b, est, r, alpha)
    if f.is_file():
        return dict(job=job, cached=True, cpu_sec=float(np.load(f)["cpu_sec"]))
    A = arrays(b)
    t0 = time.process_time()
    out: dict = {}

    def fit_val(Xa, y, fit, val, st=None):
        a_, b_ = fit
        if st is None:
            st = structure(Xa[a_:b_], A["bb_aug"], r)
        th = batch_theta(
            np.ascontiguousarray(Xa[a_:b_]),
            np.ascontiguousarray(y[a_:b_]),
            est,
            alpha,
            st,
        )
        v0, v1 = val
        pv = Xa[v0:v1] @ th
        yv = y[v0:v1]
        mse, ql = val_losses(pv, yv)
        return (pv - yv) ** 2, mse, ql

    pf_ = pre_folds(A["pre_n"])
    res = [fit_val(A["pre_Xa"], A["pre_y"], fd["fit"], fd["val"]) for fd in pf_]
    out["pre_e2"] = np.array([x[0] for x in res])
    out["pre_mse"] = np.array([x[1] for x in res])
    out["pre_qlike"] = np.array([x[2] for x in res])
    # the last pre-tune fold on the 16:00-bar design (the sessions before 2020-01-03)
    W = A["W"]
    E = W + int(np.searchsorted(A["dates"][W:], pd.Timestamp(FIRST_TRADE_DAY)))
    e2, mse, ql = fit_val(
        A["bar_Xa"],
        A["bar_y"],
        (E - PRE_FOLD_LEN - EMBARGO - TW, E - PRE_FOLD_LEN - EMBARGO),
        (E - PRE_FOLD_LEN, E),
    )
    out["prebar_mse"], out["prebar_qlike"] = np.array(mse), np.array(ql)
    rows = retune_rows(A["dates"], W)
    rt_e2, rt_ql = [], []
    for row in rows:
        parts = [
            fit_val(A["bar_Xa"], A["bar_y"], fd["fit"], fd["val"])
            for fd in retune_folds(W, row)
        ]
        rt_e2.append(np.concatenate([p[0] for p in parts]))
        rt_ql.append(np.mean([p[2] for p in parts]))
    out["rt_rows"], out["rt_e2"], out["rt_qlike"] = (
        np.array(rows),
        np.array(rt_e2),
        np.array(rt_ql),
    )
    # the spec's own re-choice tail at every block: fit [i0, i0 + 1850) with the window's structure
    n_test = len(A["bar_Xa"]) - W
    tail = []
    for i0, _ in blocks_of(n_test):
        key = (b, "bar", i0, r)
        if key not in _ST:
            _ST[key] = structure(A["bar_Xa"][i0 : i0 + W], A["bb_aug"], r)
        fit_n = W - VAL_TAIL - EMBARGO
        tail.append(
            fit_val(
                A["bar_Xa"],
                A["bar_y"],
                (i0, i0 + fit_n),
                (i0 + W - VAL_TAIL, i0 + W),
                st=_ST[key],
            )[0]
        )
    out["tail_e2"] = np.array(tail)
    out["cpu_sec"] = time.process_time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return dict(job=job, cached=False, cpu_sec=out["cpu_sec"])


def run_task(job: dict) -> dict:
    t = time.time()
    try:
        res = walk_task(job) if job["kind"] == "walk" else fold_task(job)
    except Exception as e:  # noqa: BLE001 -- reported and re-raised by the caller
        return dict(job=job, error=repr(e))
    res["wall"] = time.time() - t
    return res


def run_pool(jobs: list[dict], label: str) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    todo = [
        j
        for j in jobs
        if not (
            walk_file(j["bucket"], j["est"], j["r"], j["tag"])
            if j["kind"] == "walk"
            else fold_file(j["bucket"], j["est"], j["r"], j["alpha"])
        ).is_file()
    ]
    say(f"{label}: {len(jobs)} tasks, {len(todo)} to run on {WORKERS} processes")
    out = []
    if not todo:
        return out
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(run_task, j): j for j in todo}
        for k, fu in enumerate(as_completed(futs)):
            r = fu.result()
            if "error" in r:
                raise RuntimeError(f"task {r['job']} failed: {r['error']}")
            out.append(r)
            j = r["job"]
            say(
                f"  [{k + 1}/{len(todo)}] {j['kind']} {j['bucket']} {SHORT[j['est']]} {RTAG[j['r']]} "
                f"{j.get('tag', atag(j.get('alpha', 1.0)))}: {r['cpu_sec']:.1f} cpu-s ({time.time() - t0:.0f}s)"
            )
    log = WORK / "task_times.csv"
    rows = [
        dict(
            kind=r["job"]["kind"],
            bucket=r["job"]["bucket"],
            est=r["job"]["est"],
            r=r["job"]["r"],
            tag=r["job"].get("tag", atag(r["job"].get("alpha", 1.0))),
            cpu_sec=r["cpu_sec"],
            wall_sec=r["wall"],
            stage=label,
        )
        for r in out
    ]
    pd.DataFrame(rows).to_csv(log, mode="a", header=not log.is_file(), index=False)
    return out


# ============================================================================ plans
def rs_of(b: str) -> tuple[float, ...]:
    return (R_ONE,) if b == "baseline" else (R_ONE, R_BB)


def fold_table(b: str, est: str, r: float, grid: list[float]) -> dict:
    zs = [np.load(fold_file(b, est, r, a)) for a in grid]
    return {
        k: np.stack([z[k] for z in zs])
        for k in (
            "pre_e2",
            "pre_mse",
            "pre_qlike",
            "prebar_mse",
            "prebar_qlike",
            "rt_e2",
            "rt_qlike",
            "tail_e2",
        )
    } | {"rt_rows": zs[0]["rt_rows"]}


def argmin_first(v) -> int:
    v = np.asarray(v, float)
    return int(np.flatnonzero(v == np.nanmin(v))[0])


def pick(T: dict) -> dict:
    mse = T["pre_mse"].mean(axis=1)
    ql = T["pre_qlike"].mean(axis=1)
    return dict(i_mse=argmin_first(mse), i_qlike=argmin_first(ql), mse=mse, qlike=ql)


def light_retune(
    T: dict, grid: list[float], i0: int, rule: str
) -> tuple[list[int], list[dict]]:
    """Incumbent index i0 from the pre-tune; at each retune row the grid neighbours challenge it on
    the two 125-session folds before the row.  Returns the incumbent in force at each retune row
    (after the decision) and the log."""
    inc, sched, log = i0, [], []
    for k, row in enumerate(T["rt_rows"]):
        e2 = T["rt_e2"][:, k, :]
        mse = e2.mean(axis=1)
        cands = [i for i in (inc - 1, inc + 1) if 0 <= i < len(grid)]
        best = min(cands, key=lambda i: (mse[i], i))
        d = e2[best] - e2[inc]
        from src.evaluation.diebold_mariano import dm_test

        t = float(dm_test(d, np.zeros_like(d))["dm"])
        z = NormalDist().inv_cdf(1 - MARGIN_ALPHA / len(cands))
        sw_any, sw_margin = bool(mse[best] < mse[inc]), bool(t < -z)
        sw = sw_any if rule == "any" else sw_margin
        log.append(
            dict(
                row=int(row),
                incumbent=grid[inc],
                challengers=json.dumps([grid[i] for i in cands]),
                best_challenger=grid[best],
                incumbent_mse=float(mse[inc]),
                best_mse=float(mse[best]),
                rel_gain=float(1 - mse[best] / mse[inc]),
                hac_t=t,
                z_crit=z,
                switch_any=sw_any,
                switch_margin=sw_margin,
                switched=sw,
            )
        )
        if sw:
            inc = best
        log[-1]["in_force"] = grid[inc]
        sched.append(inc)
    return sched, log


def plan(b: str, est: str, causal: dict) -> dict:
    """Every arm of (bucket, estimator): r, grid, the alpha in force in each block, logs.
    causal: {r: alpha_blk of the full-grid walk}."""
    meta = json.loads((ARR / "meta.json").read_text())
    bl = blocks_of(len(meta["_dates"]) - int(meta["_W"]))
    arms: dict = {}
    picks, rlogs = [], []
    for r in rs_of(b):
        bbx = "_bb" if r == R_BB else ""
        a_name = "a_causal" if r == R_ONE else "b_causal_bb"
        arms[a_name] = dict(r=r, grid="spec", alphas=[float(a) for a in causal[r]])
        for gk in ("spec", "wide"):
            if gk == "spec":
                grid = spec_grid(est)
            else:
                side = picks[-1]["edge"]
                if not side:
                    continue
                grid = widened(est, side)
            T = fold_table(b, est, r, grid)
            p = pick(T)
            i = p["i_mse"]
            edge = "low" if i == 0 else "high" if i == len(grid) - 1 else ""
            # fold stability: pick on 2 folds, score on the other 2
            per_fold = [
                grid[argmin_first(T["pre_mse"][:, f])] for f in range(PRE_FOLDS)
            ]
            splits = []
            for S in combinations(range(PRE_FOLDS), PRE_FOLDS // 2):
                Sc = [f for f in range(PRE_FOLDS) if f not in S]
                iS = argmin_first(T["pre_mse"][:, list(S)].mean(axis=1))
                mc = T["pre_mse"][:, Sc].mean(axis=1)
                splits.append(
                    dict(
                        S=list(S),
                        pick_S=grid[iS],
                        pick_Sc=grid[argmin_first(mc)],
                        regret_pct=float(100 * (mc[iS] / mc.min() - 1)),
                        regret_vs_full_pick_pct=float(100 * (mc[iS] / mc[i] - 1)),
                    )
                )
            picks.append(
                dict(
                    bucket=b,
                    est=est,
                    r=r,
                    grid_kind=gk,
                    grid=json.dumps(grid),
                    pick_mse=grid[i],
                    pick_qlike=grid[p["i_qlike"]],
                    edge=edge,
                    val_mse_by_alpha=json.dumps([round(float(v), 6) for v in p["mse"]]),
                    val_qlike_by_alpha=json.dumps(
                        [round(float(v), 6) for v in p["qlike"]]
                    ),
                    fold_picks=json.dumps(per_fold),
                    last_fold_pick_last30=grid[argmin_first(T["pre_mse"][:, -1])],
                    last_fold_pick_bar1600=grid[argmin_first(T["prebar_mse"])],
                    splits=splits,
                )
            )
            sfx = "" if gk == "spec" else "_w"
            frozen = "c_frozen" if r == R_ONE else "d_frozen_bb"
            arms[frozen + sfx] = dict(
                r=r, grid=gk, alphas=[grid[i]] * len(bl), pick=grid[i]
            )
            if r == R_BB or b == "baseline":
                for rule, nm in (
                    ("any", "e_retune_any"),
                    ("margin", "f_retune_margin"),
                ):
                    sched, log = light_retune(T, grid, i, rule)
                    al = []
                    for i0, _ in bl:
                        done = [k for k, row in enumerate(T["rt_rows"]) if row <= i0]
                        al.append(grid[sched[done[-1]]] if done else grid[i])
                    arms[nm + bbx + sfx] = dict(r=r, grid=gk, alphas=al, pick=grid[i])
                    for x in log:
                        rlogs.append(
                            dict(bucket=b, est=est, r=r, grid_kind=gk, rule=rule, **x)
                        )
    return dict(arms=arms, picks=picks, retune_log=rlogs, blocks=bl)


def causal_choices(b: str, est: str) -> dict:
    return {r: np.load(walk_file(b, est, r, "grid"))["alpha_blk"] for r in rs_of(b)}


# ============================================================================ compute
def compute() -> None:
    """Every walk and fold fit of the buckets in CLP_BUCKETS (default all, in priority order)."""
    if not (ARR / "meta.json").is_file():
        prep()
    t0 = time.time()
    todo_b = [
        b for b in os.environ.get("CLP_BUCKETS", ",".join(BUCKETS)).split(",") if b
    ]
    assert set(todo_b) <= set(BUCKETS), todo_b
    hi = {e: widened(e, "high")[-1] for e in EST}
    lo = {e: widened(e, "low")[0] for e in EST}
    jobs = []
    for b in todo_b:
        for est in EST:
            for r in rs_of(b):
                jobs.append(
                    dict(
                        kind="walk",
                        bucket=b,
                        est=est,
                        r=r,
                        alphas=spec_grid(est),
                        tag="grid",
                    )
                )
    for b in todo_b:
        for est in EST:
            for r in rs_of(b):
                for a in spec_grid(est) + [hi[est]]:
                    jobs.append(dict(kind="fold", bucket=b, est=est, r=r, alpha=a))
    # the longest tasks first: all_features lasso / enet walks
    jobs.sort(
        key=lambda j: (
            BUCKETS.index(j["bucket"]),
            j["kind"] != "walk",
            j["est"] == "ridge",
        )
    )
    run_pool(
        jobs,
        f"walks (spec grid) + folds (spec grid + one decade up): {','.join(todo_b)}",
    )
    # the low edge one decade down only where a spec pick sits there
    low = []
    for b in todo_b:
        for est in EST:
            for r in rs_of(b):
                T = fold_table(b, est, r, spec_grid(est))
                if pick(T)["i_mse"] == 0:
                    low.append(dict(kind="fold", bucket=b, est=est, r=r, alpha=lo[est]))
    if low:
        run_pool(low, "folds one decade down (low-edge picks)")
    # one-alpha walks for every alpha an arm uses (and the causal choices: stitch gate, jumps)
    need = set()
    for b in todo_b:
        for est in EST:
            P = plan(b, est, causal_choices(b, est))
            for arm in P["arms"].values():
                for a in arm["alphas"]:
                    need.add((b, est, arm["r"], float(a)))
    wj = [
        dict(kind="walk", bucket=b, est=est, r=r, alphas=[a], tag=atag(a))
        for b, est, r, a in sorted(need)
    ]
    wj.sort(key=lambda j: (BUCKETS.index(j["bucket"]), j["est"] == "ridge"))
    run_pool(wj, "one-alpha walks")
    tf = WORK / "stage_wall.csv"
    pd.DataFrame(
        [
            dict(
                stage="compute",
                wall_sec=time.time() - t0,
                workers=WORKERS,
                finished=time.strftime("%Y-%m-%d %H:%M"),
            )
        ]
    ).to_csv(tf, mode="a", header=not tf.is_file(), index=False)
    say(f"compute done in {time.time() - t0:.0f}s")


# ============================================================================ scoring
def as_day(values) -> np.ndarray:
    t = pd.to_datetime(pd.Series(values))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return t.dt.normalize().to_numpy()


class Scorer:
    """The GUI's plain scorer (writeup/master_table_gui/rows.py, read-only): plain = pred_adj^2 x B,
    target / iv_var of the master table's sub_ols_baseline rows, the daily_a0 quote book.
    2026-10-15 (user, data audit #10): ``target`` is the ACTUAL realized variance (rv_unclipped), not the
    winsorized training target (``target_trimmed``); ``y_adj`` is sqrt(target / B), the fit-space truth."""

    def __init__(self):
        from master_table_gui import rows as R

        self.R = R
        tg = pd.read_parquet(EXPORT / "targets.parquet")
        self.dates = pd.to_datetime(tg["date"]).to_numpy()
        self.B = pd.Series(tg["baseline"].to_numpy(float), index=as_day(tg["date"]))
        md = pd.read_parquet(
            DAILY,
            columns=[
                "key",
                "day",
                "plain",
                "target",
                "rv_unclipped",
                "iv_var",
                "q",
                "ret_mid",
            ],
        )
        md["day"] = as_day(md["day"])
        self.md = md
        base = md[md["key"] == "sub_ols_baseline"].set_index("day").sort_index()
        qb = pd.read_parquet(
            QUOTES, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
        )
        qb.index = pd.to_datetime(qb.index).normalize()
        j = base[["target", "rv_unclipped", "iv_var", "plain"]].join(qb, how="inner")
        assert len(j) == len(base) == 866, (len(j), len(base))
        self.days = j.index
        # self.target = j["target"].to_numpy(float)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        # the scoring truth: the ACTUAL 15:30-16:00 realized variance (the master table's rv_unclipped); the models still
        # fit the winsorized target, kept as target_trimmed for diagnostics only
        self.target = j["rv_unclipped"].to_numpy(float)
        self.target_trimmed = j["target"].to_numpy(float)
        # the actual RV on the fit scale, sqrt(RV / B): the truth of the fit-space (adjusted-scale) out-of-sample losses
        self.y_adj = np.sqrt(self.target / self.B.reindex(self.days).to_numpy(float))
        self.iv = j["iv_var"].to_numpy(float)
        self.ret = j["R"].to_numpy(float)
        self.ex = j["exit"].to_numpy(float)
        self.bid = (j["bid_c"] + j["bid_p"]).to_numpy(float)
        self.ask = (j["ask_c"] + j["ask_p"]).to_numpy(float)
        self.ols_plain = j["plain"].to_numpy(float)
        self.ols = self.daily_from_plain(self.ols_plain)

    def plain_of(self, pred_adj: np.ndarray) -> np.ndarray:
        s = pd.Series(np.asarray(pred_adj, float), index=as_day(self.dates))
        return (s**2 * self.B).reindex(self.days).to_numpy(float)

    def daily_from_plain(self, plain: np.ndarray) -> dict:
        ratio = self.target / plain
        q = np.where(plain > self.iv, 1.0, -1.0)
        return dict(
            plain=plain,
            ql=ratio - np.log(ratio) - 1.0,
            q=q,
            mid=q * self.ret,
            x=self.R._crossed_return(q, self.ex, self.bid, self.ask),
        )

    def daily(self, pred_adj) -> dict:
        return self.daily_from_plain(self.plain_of(pred_adj))

    def stats(self, d: dict) -> dict:
        sz = self.R._score_sizes(
            d["plain"] - self.iv, self.ret, self.ex, self.bid, self.ask
        )
        return dict(
            qlike=float(d["ql"].mean()),
            sharpe_mid=self.R._sharpe_ann(d["mid"]),
            sharpe_crossed=self.R._sharpe_ann(d["x"]),
            pct_buy=float(100 * (d["q"] > 0).mean()),
            clip_sharpe_mid=sz["clip"]["sharpe_mid"],
            clip_sharpe_crossed=sz["clip"]["sharpe_crossed"],
            rank_sharpe_mid=sz["rank"]["sharpe_mid"],
            rank_sharpe_crossed=sz["rank"]["sharpe_crossed"],
            sign614_sharpe_mid=sz["sign"]["sharpe_mid"],
            sized_days=sz["sign"]["days"],
        )

    @staticmethod
    def contrast(d: dict, ref: dict) -> dict:
        import atm_straddle_lib as asl

        from src.evaluation.diebold_mariano import dm_test

        dm = dm_test(d["ql"], ref["ql"])
        t, lag = asl.newey_west_t(d["mid"] - ref["mid"])
        return dict(
            dq_pct=float(100 * (d["ql"].mean() / ref["ql"].mean() - 1)),
            dm=float(dm["dm"]),
            dm_p=float(dm["p"]),
            hac_t_mid=float(t),
            hac_lag=int(lag),
            n_diff_position=int((d["q"] != ref["q"]).sum()),
        )


# ============================================================================ report
def stitch(b: str, est: str, r: float, alphas: list[float], bl) -> np.ndarray:
    pred = np.empty(bl[-1][1])
    for (i0, i1), a in zip(bl, alphas):
        pred[i0:i1] = np.load(walk_file(b, est, r, atag(a)))["pred"][i0:i1]
    return pred


def stored_1600(key: str, dates) -> np.ndarray:
    t = pd.read_parquet(SPXW / f"yhat_{key}.parquet", columns=["t", "yhat"])
    et = pd.to_datetime(t["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
    s = pd.Series(t.loc[m, "yhat"].to_numpy(float), index=pd.DatetimeIndex(et[m]))
    return s.reindex(pd.DatetimeIndex(dates)).to_numpy(float)


def complete(b: str) -> bool:
    """Every walk and fold file the bucket's arms need is on disk."""
    try:
        for est in EST:
            P = plan(b, est, causal_choices(b, est))
            for arm in P["arms"].values():
                for a in arm["alphas"]:
                    if not walk_file(b, est, arm["r"], atag(a)).is_file():
                        return False
    except (FileNotFoundError, OSError):
        return False
    return True


def report() -> None:  # noqa: C901 -- one linear report
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    RB = [b for b in BUCKETS if complete(b)]
    say(f"report on {RB} (incomplete: {[b for b in BUCKETS if b not in RB]})")
    meta = json.loads((ARR / "meta.json").read_text())
    W = meta["_W"]
    dates = pd.to_datetime(meta["_dates"])
    fdates = dates[W:]
    n_test = len(fdates)
    S = Scorer()
    assert (pd.DatetimeIndex(S.dates) == fdates).all()
    gates = pd.read_csv(WORK / "gate_design.csv").to_dict("records")
    ex = pd.read_parquet(
        EXPORT / "forecasts.parquet", columns=["series_id", "date", "pred_adj"]
    )
    exw = ex.pivot(index="date", columns="series_id", values="pred_adj").reindex(fdates)
    master = pd.read_csv(MASTER).set_index("key")

    # ---- gate: the scorer reproduces the master table's plain numbers
    for key in (
        "sub_ols_baseline",
        "sub_ridge_all_features",
        "sub_lasso_all_features",
        "sub_enet_all_features",
        "sub_ridge_baseline",
        "sub_lasso_baseline",
        "sub_enet_baseline",
        "sub_lasso_live_feasible",
    ):
        mk = S.md[S.md["key"] == key].set_index("day").reindex(S.days)
        st = S.stats(S.daily_from_plain(mk["plain"].to_numpy(float)))
        ref = master.loc[key]
        dif = max(
            abs(st["qlike"] - ref["qlike_raw"]),
            abs(st["sharpe_mid"] - ref["Sharpe_mid"]),
            abs(st["sharpe_crossed"] - ref["Sharpe_crossed"]),
        )
        gates.append(
            dict(
                gate="scorer: master daily plain -> master_table.csv (QLIKE, Sharpe mid, crossed)",
                item=key,
                value=f"{st['qlike']:.4f} / {st['sharpe_mid']:.2f} / {st['sharpe_crossed']:.2f}"
                f" (max abs diff {dif:.1e})",
                ok=bool(dif < 1e-9),
            )
        )
        sid = f"master_table/{key}"
        if sid in exw:
            pl = S.plain_of(exw[sid].to_numpy())
            rel = float(np.max(np.abs(pl / mk["plain"].to_numpy(float) - 1)))
            gates.append(
                dict(
                    gate="scorer: stored forecast^2 x B == master daily plain",
                    item=key,
                    value=f"max rel {rel:.1e}",
                    ok=bool(rel < 1e-9),
                )
            )

    # ---- arms, gates on the walks
    series: dict[str, dict] = {}
    pick_rows, split_rows, rlog_rows, switch_rows, fold_rows, path_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for b in RB:
        for est in EST:
            sh = SHORT[est]
            P = plan(b, est, causal_choices(b, est))
            bl = P["blocks"]
            for arm, spec in P["arms"].items():
                r = spec["r"]
                if arm in ("a_causal", "b_causal_bb"):
                    z = np.load(walk_file(b, est, r, "grid"))
                    pred = z["pred"]
                    st_ = stitch(b, est, r, spec["alphas"], bl)
                    rel = float(np.max(np.abs(st_ / pred - 1)))
                    gates.append(
                        dict(
                            gate="stitch: one-alpha walks at the causal choices == the causal run",
                            item=f"{b} {sh} {arm}",
                            value=f"max rel {rel:.1e}",
                            ok=bool(rel < 1e-12),
                        )
                    )
                    # the spec's tail via batch fits == the walk's own validation MSE
                    T = fold_table(b, est, r, spec_grid(est))
                    mse_fit = T["tail_e2"].mean(axis=2).T  # blocks x alphas
                    vm = z["val_mse"]
                    ok = np.isfinite(vm) & (vm < 1e6)
                    rr = float(np.max(np.abs(mse_fit[ok] / vm[ok] - 1)))
                    gates.append(
                        dict(
                            gate="batch fits: spec tail MSE == the walk's re-choice MSE (finite cells)",
                            item=f"{b} {sh} {arm}",
                            value=f"max rel {rr:.1e} ({int(ok.sum())} of {vm.size})",
                            ok=bool(rr < 1e-6),
                        )
                    )
                else:
                    pred = stitch(b, est, r, spec["alphas"], bl)
                sid = f"linear_pretune/{sh}_{b}_{arm}"
                series[sid] = dict(
                    bucket=b,
                    est=est,
                    arm=arm,
                    r=r,
                    grid=spec["grid"],
                    pred=pred,
                    alphas=spec["alphas"],
                )
                for k, (i0, _) in enumerate(bl):
                    path_rows.append(
                        dict(
                            series_id=sid,
                            block=k,
                            first_forecast=str(fdates[i0])[:10],
                            alpha=spec["alphas"][k],
                            r=r,
                        )
                    )
            for p in P["picks"]:
                splits = p.pop("splits")
                pick_rows.append(
                    p
                    | dict(
                        split_agree=int(
                            sum(s["pick_S"] == s["pick_Sc"] for s in splits)
                        ),
                        split_regret_pct_mean=float(
                            np.mean([s["regret_pct"] for s in splits])
                        ),
                        split_regret_pct_max=float(
                            np.max([s["regret_pct"] for s in splits])
                        ),
                    )
                )
                for s in splits:
                    split_rows.append(
                        dict(
                            bucket=b,
                            est=est,
                            r=p["r"],
                            grid_kind=p["grid_kind"],
                            S=json.dumps(s["S"]),
                            pick_S=s["pick_S"],
                            pick_Sc=s["pick_Sc"],
                            regret_pct=s["regret_pct"],
                            regret_vs_full_pick_pct=s["regret_vs_full_pick_pct"],
                        )
                    )
            for x in P["retune_log"]:
                x["date"] = str(fdates[x["row"]])[:10]
                rlog_rows.append(x)
            # pre-tune fold table (long)
            for r in rs_of(b):
                grid = sorted(
                    {
                        a
                        for a in spec_grid(est)
                        + widened(est, "high")
                        + (
                            [widened(est, "low")[0]]
                            if fold_file(b, est, r, widened(est, "low")[0]).is_file()
                            else []
                        )
                    }
                )
                T = fold_table(b, est, r, grid)
                for ia, a in enumerate(grid):
                    for f in range(PRE_FOLDS):
                        fold_rows.append(
                            dict(
                                bucket=b,
                                est=est,
                                r=r,
                                alpha=a,
                                fold=f,
                                val_mse=T["pre_mse"][ia, f],
                                val_qlike=T["pre_qlike"][ia, f],
                            )
                        )
                # ---- noise on the causal arms: switches, edges, forecast jumps, validation gain
                arm = "a_causal" if r == R_ONE else "b_causal_bb"
                al = P["arms"][arm]["alphas"]
                g = spec_grid(est)
                Ts = fold_table(b, est, r, g)
                for k, (i0, i1) in enumerate(bl):
                    row = dict(
                        bucket=b,
                        est=est,
                        arm=arm,
                        block=k,
                        first_forecast=str(fdates[i0])[:10],
                        alpha=al[k],
                        at_edge=bool(al[k] in (g[0], g[-1])),
                        prev_alpha=al[k - 1] if k else np.nan,
                        switched=bool(k and al[k] != al[k - 1]),
                    )
                    if row["switched"]:
                        new = np.load(walk_file(b, est, r, atag(al[k])))["pred"][i0:i1]
                        old = np.load(walk_file(b, est, r, atag(al[k - 1])))["pred"][
                            i0:i1
                        ]
                        rel = np.abs(new / old - 1)
                        dn, do = (
                            S.daily(
                                np.r_[
                                    np.full(i0, np.nan),
                                    new,
                                    np.full(n_test - i1, np.nan),
                                ]
                            ),
                            S.daily(
                                np.r_[
                                    np.full(i0, np.nan),
                                    old,
                                    np.full(n_test - i1, np.nan),
                                ]
                            ),
                        )
                        okd = np.isfinite(dn["plain"])
                        e2 = Ts["tail_e2"][:, k, :]
                        ia, ib = g.index(al[k]), g.index(al[k - 1])
                        from src.evaluation.diebold_mariano import dm_test

                        d = e2[ia] - e2[ib]
                        t = float(dm_test(d, np.zeros_like(d))["dm"])
                        z = NormalDist().inv_cdf(1 - MARGIN_ALPHA / (len(g) - 1))
                        row.update(
                            jump_first_rel=float(rel[0]),
                            jump_first_var_rel=float(abs((new[0] / old[0]) ** 2 - 1)),
                            block_median_abs_rel=float(np.median(rel)),
                            block_mean_abs_rel=float(rel.mean()),
                            block_trade_days=int(okd.sum()),
                            block_position_changes=int(
                                (dn["q"][okd] != do["q"][okd]).sum()
                            ),
                            tail_mse_new=float(e2[ia].mean()),
                            tail_mse_prev=float(e2[ib].mean()),
                            tail_rel_gain=float(1 - e2[ia].mean() / e2[ib].mean()),
                            tail_hac_t=t,
                            margin_z=z,
                            passes_margin=bool(t < -z),
                        )
                    switch_rows.append(row)

    # ---- gates against the stored forecasts and the 2026-10-03 export.  Tolerance: ridge 1e-8
    # relative; lasso / elastic net 0.65 %, the gap the 2026-10-03 study documented between three runs
    # of the same warm homotopy (spec class, C port, stored table: exog_penalty/SUMMARY.md gates 1-2),
    # which branches on the floating-point path in ill-conditioned blocks.  Recorded beside it: the
    # blocks with a gap, trade days with a different sign(s) position, and the plain QLIKE difference.
    def path_gate(mine: np.ndarray, ref: np.ndarray, est: str) -> tuple[str, bool]:
        rel = np.abs(mine / ref - 1)
        blk = sorted({int(i // TUNE_PER) for i in np.flatnonzero(rel > 1e-9)})
        dm_, dr_ = S.daily(mine), S.daily(ref)
        val = (
            f"max rel {rel.max():.1e}; sessions > 1e-9: {int((rel > 1e-9).sum())} (blocks {blk}); "
            f"positions differ on {int((dm_['q'] != dr_['q']).sum())} trade days; "
            f"QLIKE diff {dm_['ql'].mean() - dr_['ql'].mean():+.1e}"
        )
        return val, bool(rel.max() < (1e-8 if est == "ridge" else 6.5e-3))

    for b in RB:
        for est in EST:
            sh = SHORT[est]
            a = series[f"linear_pretune/{sh}_{b}_a_causal"]["pred"]
            st = stored_1600(f"sub_{sh}_{b}", fdates)
            if np.isfinite(st).all():
                val, ok = path_gate(a, st, est)
                gates.append(
                    dict(
                        gate="arm a == stored forecast yhat_sub_<est>_<bucket> (16:00 rows)",
                        item=f"{b} {sh}",
                        value=val,
                        ok=ok,
                    )
                )
            else:
                gates.append(
                    dict(
                        gate="arm a == stored forecast",
                        item=f"{b} {sh}",
                        value=f"stored table missing / {int((~np.isfinite(st)).sum())} rows NaN",
                        ok=False,
                    )
                )
            if b == "all_features":
                for arm, src in (
                    ("a_causal", f"exog_penalty/{sh}_single"),
                    ("b_causal_bb", f"exog_penalty/{sh}_bbfix"),
                ):
                    mine = series[f"linear_pretune/{sh}_{b}_{arm}"]
                    val, ok = path_gate(mine["pred"], exw[src].to_numpy(), est)
                    gates.append(
                        dict(
                            gate=f"arm {arm[0]} == the 2026-10-03 export {src}",
                            item=f"{b} {sh}",
                            value=val,
                            ok=ok,
                        )
                    )
                    rc = pd.read_parquet(
                        EXPORT / "linear_rechoices.parquet",
                        columns=["series_id", "block", "alpha"],
                    )
                    al = (
                        rc[rc["series_id"] == src]
                        .sort_values("block")["alpha"]
                        .to_numpy(float)
                    )
                    gates.append(
                        dict(
                            gate=f"arm {arm[0]}: same alphas as {src}",
                            item=f"{b} {sh}",
                            value=json.dumps(mine["alphas"]),
                            ok=bool(np.array_equal(al, mine["alphas"])),
                        )
                    )

    # ---- scores
    daily = {sid: S.daily(s["pred"]) for sid, s in series.items()}
    gui_ref = {b: ("b_causal_bb" if b == "all_features" else "a_causal") for b in RB}
    rows = []
    for sid, s in series.items():
        b, est, arm = s["bucket"], s["est"], s["arm"]
        sh = SHORT[est]
        d = daily[sid]
        st = S.stats(d)
        base_arm = arm.removesuffix("_w")
        # the causal arm with the same penalty structure (b itself: against a)
        causal = (
            "a_causal"
            if base_arm in ("b_causal_bb", "c_frozen") or b == "baseline"
            else "b_causal_bb"
        )
        row = dict(
            series_id=sid,
            bucket=b,
            estimator=LABEL[est],
            arm=arm,
            label=ARM_LABEL[base_arm],
            widened=arm.endswith("_w"),
            r=s["r"],
            grid=s["grid"],
            alphas=json.dumps(s["alphas"]),
            n_alpha_changes=int(
                sum(
                    s["alphas"][k] != s["alphas"][k - 1]
                    for k in range(1, len(s["alphas"]))
                )
            ),
            **st,
        )
        c = S.contrast(d, S.ols)
        row.update(dq_pct_vs_ols=c["dq_pct"], dm_vs_ols=c["dm"], dm_p_vs_ols=c["dm_p"])
        if arm != "a_causal":
            ref = f"linear_pretune/{sh}_{b}_{causal}"
            c = S.contrast(d, daily[ref])
            row.update(
                ref_causal=causal,
                **{f"{k}_vs_causal": v for k, v in c.items()},
                dsharpe_mid_vs_causal=st["sharpe_mid"]
                - S.R._sharpe_ann(daily[ref]["mid"]),
            )
        gr = f"linear_pretune/{sh}_{b}_{gui_ref[b]}"
        if sid != gr:
            c = S.contrast(d, daily[gr])
            row.update(
                ref_gui=gui_ref[b],
                **{f"{k}_vs_gui": v for k, v in c.items()},
                dsharpe_mid_vs_gui=st["sharpe_mid"] - S.R._sharpe_ann(daily[gr]["mid"]),
            )
        rows.append(row)
    arms_df = pd.DataFrame(rows)
    # the master-table cross-check: arm a on all_features, plain numbers
    want = {
        "ridge": (0.1164, 1.71),
        "reclasso": (0.0973, 2.00),
        "reclasticnet": (0.0986, 2.04),
    }
    for est, (q, s_) in want.items():
        r_ = arms_df[
            (
                arms_df["series_id"]
                == f"linear_pretune/{SHORT[est]}_all_features_a_causal"
            )
        ].iloc[0]
        gates.append(
            dict(
                gate="cross-check: arm a all_features plain QLIKE / Sharpe mid == master table",
                item=SHORT[est],
                value=f"{r_['qlike']:.4f} / {r_['sharpe_mid']:.2f} (want {q} / {s_})",
                ok=bool(
                    round(r_["qlike"], 4) == q and round(r_["sharpe_mid"], 2) == s_
                ),
            )
        )

    # ---- write
    gdf = pd.DataFrame(gates)
    gdf.to_csv(OUT / "gates.csv", index=False)
    arms_df.to_csv(OUT / "arms.csv", index=False)
    for b in RB:
        arms_df[arms_df["bucket"] == b].to_csv(OUT / f"arms_{b}.csv", index=False)
    pd.DataFrame(pick_rows).to_csv(OUT / "penalty_picks.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "fold_stability.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "pretune_folds.csv", index=False)
    pd.DataFrame(rlog_rows).to_csv(OUT / "retune_log.csv", index=False)
    pd.DataFrame(switch_rows).to_csv(OUT / "causal_switch_log.csv", index=False)
    pd.DataFrame(path_rows).to_csv(OUT / "penalty_paths.csv", index=False)
    export_forecasts(series, fdates, meta)
    cpu = cpu_table()
    cpu.to_csv(OUT / "cpu_seconds.csv", index=False)
    write_summary(
        arms_df,
        gdf,
        pd.DataFrame(pick_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(rlog_rows),
        pd.DataFrame(switch_rows),
        cpu,
        meta,
    )
    say(
        f"report done in {time.time() - t_start:.0f}s; gates {int(gdf['ok'].sum())} / {len(gdf)} ok"
    )


def export_forecasts(series: dict, fdates, meta: dict) -> None:
    """forecasts.parquet in the export format of results/close_studies_2026-10-03/forecasts/."""
    import dense_vs_sparse_1530 as dvs

    tg = pd.read_parquet(EXPORT / "targets.parquet")
    dz = dict(
        date=tg["date"].to_numpy(),
        true_adj=tg["true_adj"].to_numpy(float),
        true_raw=tg["true_raw"].to_numpy(float),
    )
    parts = []
    for sid, s in series.items():
        fr = dvs.research_frame(dz, s["pred"])
        clock = fr["pred_clock"].reindex(pd.DatetimeIndex(fdates)).to_numpy(float)
        parts.append(
            pd.DataFrame(
                dict(
                    series_id=sid,
                    study="linear_pretune",
                    row=np.arange(len(fdates), dtype=np.int16),
                    date=pd.DatetimeIndex(fdates).astype("datetime64[ns]"),
                    pred_adj=np.asarray(s["pred"], np.float64),
                    pred_clock=clock,
                )
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT / "forecasts.parquet", index=False, compression="zstd")
    cat = pd.DataFrame(
        [
            dict(
                series_id=sid,
                bucket=s["bucket"],
                estimator=LABEL[s["est"]],
                arm=s["arm"],
                r=s["r"],
                grid=s["grid"],
                alphas_by_block=json.dumps(s["alphas"]),
                design_source=meta[s["bucket"]]["source"],
            )
            for sid, s in series.items()
        ]
    )
    cat.to_csv(OUT / "catalogue.csv", index=False)


def cpu_table() -> pd.DataFrame:
    rows = []
    for kind, root in (("walk", WALK), ("fold", FOLDS)):
        for f in sorted(root.glob("*/*.npz")):
            rows.append(
                dict(
                    kind=kind,
                    bucket=f.parent.name,
                    file=f.stem,
                    cpu_sec=float(np.load(f)["cpu_sec"]),
                )
            )
    return pd.DataFrame(rows)


# ============================================================================ summary
def _f(v, nd=4, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_summary(arms, gates, picks, splits, rlog, sw, cpu, meta) -> None:  # noqa: C901
    L = []
    L.append(
        "# The 16:00-bar linear models pre-tuned like the trees, then lightly retuned"
    )
    L.append("")
    L.append(
        f"Written by `experiments/close_linear_pretune.py report` on {time.strftime('%Y-%m-%d %H:%M')}; "
        "every number below is read from the CSVs in this folder."
    )
    L.append("")
    L.append("## Question and design")
    L.append("")
    L.append(
        "User (2026-10-06): causally retuning the linear models seems to add noise; tune them as the trees "
        "are tuned (pre-tune on the data before the trade period, then a light retune now and then), with "
        "the HAR + calendar backbone penalized separately at 1/100 of the other inputs' penalty."
    )
    L.append("")
    L.append(
        "Every arm: the spec's ridge / lasso / elastic net (l1 0.5) on the 16:00-bar design, window 2000 "
        "sessions, refit every session, 250-session blocks (cold seed at each block start), the spec's "
        "grids (ridge 1e-2 .. 1e3, lasso / elastic net 1e-6 .. 1e-2), through the C port of the 2026-10-03 "
        "study (`experiments/close_exogpen_kernel.c`, unchanged; built here with MinGW gcc 15 -O3 "
        "-march=haswell -ffp-contract=off against the reference LAPACK 3.12 / BLAS of conda-forge's netlib "
        "build, the kind of library the 2026-10-03 runs linked; with FMA contraction, or against OpenBLAS, "
        "the one-penalty lasso / elastic net warm path branches differently in block 3 or 4). Only the "
        "penalty schedule differs between arms. Input sets: all_features, moments_vol_vix (moments + "
        "sumvolume + vix) and lean_vol_vix (sumret, sumabsret, sumret4, sumautocov, sumvolume, vix), both "
        "added on 2026-10-06, baseline (HAR + calendar), moments_vix, live_feasible; every 16:00-bar design "
        "equals the corresponding columns of the exported all_features design of record (gate rows 1-7)."
    )
    L.append("")
    L.append(
        "Gate tolerance against the stored forecasts: ridge 1e-8 relative; lasso / elastic net 0.65 %, the "
        "gap the 2026-10-03 study documented between three runs of the same warm homotopy (the spec's "
        "class, its C port, the stored table), which branches on the floating-point path in an "
        "ill-conditioned block; every such gap here is confined to one block and changes no trade-day "
        "position."
    )
    L.append("")
    L.append("| arm | penalty structure | alpha schedule |")
    L.append("|---|---|---|")
    L.append(
        "| a | one penalty (spec) | re-chosen every 250 sessions on the spec's 125-session tail |"
    )
    L.append(
        "| b | backbone at 1/100 of the exogenous penalty (= exog_penalty `bbfix`) | the same causal re-choice |"
    )
    L.append("| c | one penalty | pre-tuned on pre-2020 data, frozen |")
    L.append("| d | backbone 1/100 | pre-tuned, frozen |")
    L.append(
        "| e | backbone 1/100 | pre-tuned; at each block start from the first trade day the two grid "
        "neighbours challenge the incumbent; switch on any validation gain |"
    )
    L.append(
        "| f | backbone 1/100 | the same; switch only if the HAC t of the paired squared-error difference "
        "< -z(0.05 / challengers) (the tree study's margin rule) |"
    )
    L.append("")
    L.append(
        "On `baseline` every column is backbone, so arms b and d do not exist there and e / f use one penalty. "
        "`*_w`: the arm on the grid widened one decade beyond the edge its spec-grid pre-tune pick sits at."
    )
    L.append("")
    m0 = meta["all_features"]
    L.append(
        f"Pre-tune: 4 folds x 250 sessions tiling the last 1000 16:00 sessions before 2020-01-03, each fit on "
        f"the 2000 sessions before a 25-session embargo (the tree pre-tune's folds), on the 16:00 rows of the "
        f"last-hour design ({m0['pre_n']} pre-2020 16:00 sessions from {m0['pre_first']}); the spec's batch "
        "solution on each fit block; criterion = mean validation MSE on the adjusted scale (QLIKE pick "
        "recorded). Light retune at 2020-06-23, 2021-06-21, 2022-06-17, 2023-06-16 (the block starts in the "
        "trade period), validated on the two 125-session folds before the block start, each with its own "
        "2000-session fit block (the tree study's retune folds)."
    )
    L.append("")
    L.append(
        "Scoring: plain forecast f^2 x B, no recalibration, the 866 trade days 2020-01-03 .. 2024-04-30 "
        "(the master-table GUI's scorer); DM on daily QLIKE; HAC t = Newey-West (lag 14) t of the daily mid "
        "P&L difference; robust clip / rank of |s| Sharpe on the 614 days after a 252-day warm-up. Point "
        "estimates only."
    )
    L.append("")
    L.append("## Gates")
    L.append("")
    g = gates.copy()
    g["ok"] = g["ok"].map({True: "yes", False: "NO"})
    L.append(_md(g[["gate", "item", "value", "ok"]]))
    L.append("")
    L.append(f"{int((gates['ok']).sum())} of {len(gates)} gates pass.")
    L.append("")
    # ---- noise
    L.append("## Is the causal re-choice noisy? (arms a and b)")
    L.append("")
    sw = sw.copy()
    tab = []
    for (b, est, arm), grp in sw.groupby(["bucket", "est", "arm"], sort=False):
        s = grp[grp["switched"]]
        tab.append(
            dict(
                bucket=b,
                estimator=LABEL[est],
                arm=arm[0],
                alphas=" ".join(f"{a:g}" for a in grp["alpha"]),
                switches=f"{len(s)} of {len(grp) - 1}",
                at_edge=f"{int(grp['at_edge'].sum())} of {len(grp)}",
                median_jump_pct=_f(100 * s["jump_first_rel"].median(), 1)
                if len(s)
                else "",
                block_mean_abs_pct=_f(100 * s["block_mean_abs_rel"].mean(), 1)
                if len(s)
                else "",
                position_changes=f"{int(s['block_position_changes'].sum())} of {int(s['block_trade_days'].sum())}"
                if len(s)
                else "",
                tail_gain_pct=" ".join(_f(100 * v, 1) for v in s["tail_rel_gain"])
                if len(s)
                else "",
                tail_t=" ".join(_f(v, 2) for v in s["tail_hac_t"]) if len(s) else "",
                pass_margin=f"{int(s['passes_margin'].sum())} of {len(s)}"
                if len(s)
                else "",
            )
        )
    tab = pd.DataFrame(tab)
    L.append(
        "alphas = the penalty chosen at the six re-choices (2018-06-25 .. 2023-06-16); jump = relative change "
        "of the forecast (adjusted scale) at the switch session, chosen vs the previous alpha's path; "
        "block mean = mean |relative change| over the switched block; position changes = trade days in the "
        "switched blocks whose sign(s) position differs; tail gain / t = the validation gain of the switch on "
        "the spec's 125-session tail and its HAC t; pass margin = switches whose t clears the margin rule "
        "(Bonferroni over the grid's other alphas)."
    )
    L.append("")
    L.append(_md(tab))
    L.append("")
    nsw = int(sw["switched"].sum())
    ntr = int(len(sw) - sw.groupby(["bucket", "est", "arm"]).ngroups)
    L.append(
        f"Over all causal arms: {nsw} switches in {ntr} re-choices; {int(sw['at_edge'].sum())} of {len(sw)} "
        f"choices at a grid edge; {int(sw['passes_margin'].fillna(False).astype(bool).sum())} of {nsw} switches "
        "pass the margin rule on their own validation tail."
    )
    L.append("")
    # ---- pre-tune picks
    L.append("## Pre-tune picks")
    L.append("")
    pk = picks.copy()
    pk["estimator"] = pk["est"].map(LABEL)
    pk["r"] = pk["r"].map(lambda v: "1/100" if v == R_BB else "1")
    pk = pk[
        [
            "bucket",
            "estimator",
            "r",
            "grid_kind",
            "pick_mse",
            "pick_qlike",
            "edge",
            "fold_picks",
            "last_fold_pick_last30",
            "last_fold_pick_bar1600",
            "split_agree",
            "split_regret_pct_mean",
            "split_regret_pct_max",
        ]
    ]
    pk["split_regret_pct_mean"] = pk["split_regret_pct_mean"].map(lambda v: _f(v, 2))
    pk["split_regret_pct_max"] = pk["split_regret_pct_max"].map(lambda v: _f(v, 2))
    L.append(
        "pick = argmin of the mean validation MSE over the 4 folds (pick_qlike: of the mean QLIKE); "
        "edge = the pick sits at that end of its grid; fold_picks = each fold's own argmin; last fold on "
        "the last-hour cache vs on the 16:00-bar cache (same sessions, different scaling windows); "
        "split_agree = of the 6 ways to pick on 2 folds and score on the other 2, how often both halves pick "
        "the same alpha; regret = the half-pick's MSE on the other half above that half's best (%)."
    )
    L.append("")
    L.append(_md(pk))
    L.append("")
    ne = int((picks[picks["grid_kind"] == "spec"]["edge"] != "").sum())
    L.append(
        f"{ne} of {int((picks['grid_kind'] == 'spec').sum())} spec-grid picks sit at a grid edge; "
        f"widened one decade, {int((picks[picks['grid_kind'] == 'wide']['edge'] != '').sum())} of "
        f"{int((picks['grid_kind'] == 'wide').sum())} sit at the new edge."
    )
    L.append("")
    # ---- retune
    L.append("## Light retunes")
    L.append("")
    if len(rlog):
        rl = rlog.copy()
        rl["estimator"] = rl["est"].map(LABEL)
        rl = rl[(rl["grid_kind"] == "spec")]
        tab = []
        for (b, est, rule), grp in rl.groupby(["bucket", "est", "rule"], sort=False):
            tab.append(
                dict(
                    bucket=b,
                    estimator=LABEL[est],
                    rule=rule,
                    in_force=" ".join(f"{a:g}" for a in grp["in_force"]),
                    switches=int(grp["switched"].sum()),
                    best_gain_pct=" ".join(_f(100 * v, 2) for v in grp["rel_gain"]),
                    hac_t=" ".join(_f(v, 2) for v in grp["hac_t"]),
                )
            )
        L.append(
            "At each of the four retunes: the alpha in force after the decision, the best challenger's "
            "validation gain over the incumbent (%) and its HAC t (spec grid; full log in `retune_log.csv`)."
        )
        L.append("")
        L.append(_md(pd.DataFrame(tab)))
        L.append("")
    # ---- scores
    L.append("## Scores")
    L.append("")
    L.append(
        "QLIKE (plain), DM vs the HAR + calendar OLS, Sharpe mid / crossed of sign(s); vs causal = against the "
        "same estimator's causal arm with the same penalty structure (c vs a; b, d, e, f vs a for b and vs b "
        "otherwise; on baseline every arm vs a): DM on QLIKE, HAC t of the mid P&L difference, days with a "
        "different position; robust clip / rank of |s| Sharpe mid on the 614 sized days."
    )
    L.append("")
    for b in sorted(set(arms["bucket"]), key=BUCKETS.index):
        sub = arms[arms["bucket"] == b]
        L.append(f"### {b} ({meta[b]['source']}; {len(meta[b]['names'])} columns)")
        L.append("")
        t = pd.DataFrame(
            dict(
                estimator=sub["estimator"],
                arm=sub["arm"],
                alphas=sub["alphas"].map(
                    lambda s: " ".join(f"{a:g}" for a in json.loads(s))
                ),
                QLIKE=sub["qlike"].map(lambda v: _f(v)),
                DM_vs_OLS=sub["dm_vs_ols"].map(lambda v: _f(v, 2, True)),
                Sharpe=sub["sharpe_mid"].map(lambda v: _f(v, 2))
                + " / "
                + sub["sharpe_crossed"].map(lambda v: _f(v, 2)),
                DM_vs_causal=sub.get("dm_vs_causal", pd.Series(dtype=float)).map(
                    lambda v: _f(v, 2, True)
                ),
                HAC_t_vs_causal=sub.get(
                    "hac_t_mid_vs_causal", pd.Series(dtype=float)
                ).map(lambda v: _f(v, 2, True)),
                pos_diff=sub.get(
                    "n_diff_position_vs_causal", pd.Series(dtype=float)
                ).map(lambda v: "" if not np.isfinite(v) else str(int(v))),
                clip_rank=sub["clip_sharpe_mid"].map(lambda v: _f(v, 2))
                + " / "
                + sub["rank_sharpe_mid"].map(lambda v: _f(v, 2)),
            )
        )
        L.append(_md(t.fillna("")))
        L.append("")
    # ---- verdict (generated from the numbers)
    L.append("## Reading")
    L.append("")
    nb = arms[~arms["widened"] & (arms["bucket"] != "baseline")]
    for arm in (
        "b_causal_bb",
        "c_frozen",
        "d_frozen_bb",
        "e_retune_any_bb",
        "f_retune_margin_bb",
    ):
        t = nb[nb["arm"] == arm]
        if not len(t):
            continue
        L.append(
            f"- {arm} vs {t['ref_causal'].iloc[0]} over {len(t)} estimator x input-set pairs (baseline "
            f"excluded): QLIKE lower in {int((t['dq_pct_vs_causal'] < 0).sum())} (mean "
            f"{t['dq_pct_vs_causal'].mean():+.1f} %), DM below -1.96 in {int((t['dm_vs_causal'] < -1.96).sum())}"
            f", above +1.96 in {int((t['dm_vs_causal'] > 1.96).sum())}; Sharpe mid higher in "
            f"{int((t['dsharpe_mid_vs_causal'] > 0).sum())} (mean {t['dsharpe_mid_vs_causal'].mean():+.2f}), "
            f"HAC |t| above 1.96 in {int((t['hac_t_mid_vs_causal'].abs() > 1.96).sum())}."
        )
    L.append("")
    L.append(
        "The lowest-QLIKE pre-tuned arm of each estimator (picked after seeing the scores; for orientation "
        "only) against its causal arm:"
    )
    L.append("")
    for b in sorted(set(arms["bucket"]), key=BUCKETS.index):
        sub = arms[(arms["bucket"] == b) & ~arms["widened"]].set_index(
            ["estimator", "arm"]
        )
        bits = []
        for est in EST:
            lab = LABEL[est]
            ref = "b_causal_bb" if b != "baseline" else "a_causal"
            cand = [
                a
                for a in ("d_frozen_bb", "e_retune_any_bb", "f_retune_margin_bb")
                if (lab, a) in sub.index
            ] or [
                a
                for a in ("c_frozen", "e_retune_any", "f_retune_margin")
                if (lab, a) in sub.index
            ]
            q_ref = sub.loc[(lab, ref), "qlike"]
            best = min(cand, key=lambda a: sub.loc[(lab, a), "qlike"])
            rb = sub.loc[(lab, best)]
            bits.append(
                f"{lab}: causal {q_ref:.4f} -> {best} {rb['qlike']:.4f} "
                f"(DM {rb.get('dm_vs_causal', np.nan):+.2f}, HAC t {rb.get('hac_t_mid_vs_causal', np.nan):+.2f}, "
                f"Sharpe {rb['sharpe_mid']:.2f})"
            )
        L.append(f"- {b}: " + "; ".join(bits) + ".")
    L.append("")
    L.append("## Compute")
    L.append("")
    if len(cpu):
        L.append(
            f"Kernel CPU: {cpu['cpu_sec'].sum() / 3600:.2f} CPU-hours over {len(cpu)} walks and fold sets "
            f"(walks {cpu[cpu['kind'] == 'walk']['cpu_sec'].sum() / 60:.1f} min, fold fits "
            f"{cpu[cpu['kind'] == 'fold']['cpu_sec'].sum() / 60:.1f} min); `cpu_seconds.csv`, "
            "`_work/task_times.csv`, `_work/stage_wall.csv`."
        )
    L.append("")
    L.append("## Files")
    L.append("")
    L.append(
        "- `arms.csv` (and `arms_<bucket>.csv`): every series, its alpha in each block and every score."
    )
    L.append(
        "- `penalty_picks.csv`, `pretune_folds.csv`, `fold_stability.csv`: the pre-tune."
    )
    L.append(
        "- `causal_switch_log.csv`: every causal re-choice, switch, edge hit, forecast jump and validation gain."
    )
    L.append(
        "- `retune_log.csv`: every light-retune decision (both rules, spec and widened grids)."
    )
    L.append(
        "- `penalty_paths.csv`: the alpha in force in each block for every series."
    )
    L.append(
        "- `forecasts.parquet`: every series in the export format of results/close_studies_2026-10-03/"
        "forecasts/ (series_id `linear_pretune/<est>_<bucket>_<arm>`, date, pred_adj; pred_clock = the "
        "research scorer's recalibration, as there); `catalogue.csv` describes each series."
    )
    L.append("- `gates.csv`, `cpu_seconds.csv`.")
    L.append(
        "- Code: `experiments/close_linear_pretune.py` (prep / compute / report), "
        "`experiments/close_linear_pretune_design.py` (the design cache without reading data/spxw_chain.parquet)."
    )
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ============================================================================ smoke
def smoke() -> None:
    """Kernel check: one batch fit equals the seed of a normal call; ridge all_features block 0."""
    A = arrays("all_features")
    Xa, y, W = A["bar_Xa"], A["bar_y"], A["W"]
    for est in EST:
        st = structure(Xa[0:W], A["bb_aug"], R_ONE)
        a = spec_grid(est)[-1]
        th = batch_theta(
            np.ascontiguousarray(Xa[0:W]), np.ascontiguousarray(y[0:W]), est, a, st
        )
        lk, mk, pf = struct3(st)
        o = krun(
            np.ascontiguousarray(Xa[0 : W + 1]),
            np.ascontiguousarray(y[0 : W + 1]),
            W,
            est,
            np.array([0, 1], np.int32),
            lk,
            mk,
            pf,
            [a],
        )
        say(
            f"smoke {est}: batch_theta vs normal seed max abs {np.abs(th - o['theta'][0]).max():.2e}, "
            f"pred {float(Xa[W] @ th):.6f} vs {o['pred'][0]:.6f}, {o['cpu_sec']:.2f}s"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "prep":
        prep()
    elif stage == "compute":
        compute()
    elif stage == "report":
        report()
    elif stage == "smoke":
        smoke()
    elif stage == "all":
        prep()
        compute()
        report()
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    say(f"stage {stage} done in {time.time() - t0:.0f}s")
