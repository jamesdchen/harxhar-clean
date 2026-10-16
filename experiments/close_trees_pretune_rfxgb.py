"""Random forest and XGBoost on the 16:00 bar: heavy pre-2020 pre-tune, light retune every 250 sessions.

The 2026-10-03 trees pre-tune study (experiments/close_trees_pretune.py, LightGBM only; results/
close_studies_2026-10-03/trees_pretune/SUMMARY.md) extended to the random forest and XGBoost, which
were never tuned on the period before the trades.  Same protocol, same constants, same spec code:

* model and design: specs/causal_tune_trees.py's 16:00-bar random forest (RandomForestRegressor,
  100 trees) / XGBoost (the shipped tree_expert_16 configuration, leaf minimum scaled to the
  window) on the 16:00-bar design of one input set (RXP_BUCKET: all_features, baseline,
  lean_vol_vix = the six columns sumret, sumabsret, sumret4, sumautocov, sumvolume, vix of
  src/data/loading.py with the HAR + calendar backbone; moments_vix was dropped by the user on
  2026-10-06 and is not run), 2000-session window, the window column mask on (src/models/window_mask.py), every
  fit single-threaded.  Search space, round rules and model construction are the Optuna spec's
  (specs/causal_tune_trees_optuna_jobs.py), imported, not copied.  Random forest axes:
  max_features [0.02, 1], min_samples_leaf int-log [1, 256], max_depth {2, 4, 8, 16, 32, None}
  (no rounds: 100 trees, no early stopping).  XGBoost axes: max_depth [1, 8], min_child_weight
  log [0.1, 256], subsample [0.15, 1], colsample_bytree [0.05, 1], reg_lambda log [1e-4, 1e4],
  learning_rate log [0.005, 0.1] (early stopping, patience ceil(1 / lr), the 4 x shipped
  shrinkage cap; a configuration's rounds = the median of its folds' early-stopped rounds).
* design rows: the cached designs of experiments/capture_design_close.py, copied into
  _work/design by the `design` stage (the 16:00-bar cache for the walk and the retunes; the 16:00
  rows of the last-hour cache, back to 2004-04-13, for the pre-tune: the 16:00-bar cache starts
  2010-07-12 and leaves too few pre-2020 sessions for 4 folds after one window).

STAGES (``python experiments/close_trees_pretune_rfxgb.py <stage>``; one model x input set per call,
RXP_MODEL / RXP_BUCKET)

design   (local) copy the six design caches (bar1600 / last30 x the three input sets) into
         _work/design; sha256 of every array (design_sha256.json, checked by every later stage
         wherever it runs) and the comparison with results/close_studies_2026-10-03/
         design_hashes.json (design_check.csv)
gate     (a) the fold fitter equals the Optuna spec's fit_trial on the last pre-tune fold (shipped
         configuration: same validation MSE, QLIKE, rounds); (b) the shipped configuration's refit
         (control) at sample rows against the stored masked cluster tables results/spxw_pnl/
         yhat_subtree_<model>_<bucket>.parquet (refit every 10) and yhat_subtree_daily_<bucket>_
         <model>.parquet (every session); writes gate_<model>_<bucket>.csv and GATE_OK_<model>_
         <bucket> in the run folder when every check passes (max relative forecast difference <=
         GATE_TOL)
pretune  ONE Optuna TPE study (TPESampler(seed = TPE_SEED_BASE + RXP_SEED), trial 0 = the shipped
         configuration, trials sequential; the folds of a trial run in a pool of RXP_WORKERS
         processes and are combined in fold order, so the pool size changes nothing); objective =
         the mean over RXP_FOLDS blocked walk-forward folds of the validation MSE on the
         transformed target; each fold validates RXP_FOLD_LEN sessions after a 25-session embargo
         and the 2000 sessions before it; the folds tile the last RXP_FOLDS x RXP_FOLD_LEN sessions
         before 2020-01-03.  Stops at RXP_TRIALS trials or RXP_TIMEOUT seconds; resumable from its
         Optuna journal (a resumed study reseeds its sampler with seed + RESUME_SALT x completed
         trials, recorded).
plan     merges the studies of RXP_SEEDS (interleaved: trial j of every study before trial j + 1 of
         any) and fixes every arm's configuration schedule:
           control        the shipped configuration (the local twin of the stored masked table)
           frozen_k<k>    the best of the first k trials of every study, frozen (k in
                          RXP_CHECKPOINTS, refit every 10 only)
           frozen_all     the best of all trials, frozen
           retune_any     frozen_all, then a light retune every RETUNE_EVERY = 250 sessions from
                          the refit row at or before the first trade day: RXP_RETUNE_TRIALS
                          trials (trial 0 = the incumbent; TPE draws its first 10 at random) of a
                          TPE study confined to a box around the incumbent (NEIGHBOURHOOD_FRACTION
                          of each numeric axis's range, log units on log axes; the categorical
                          max_depth: the incumbent and its neighbours in the ordered choice
                          list), validated on the 250 sessions before the retune row as 2 folds of
                          125 sessions (each its own 2000-session fit block and 25-session
                          embargo), on the 16:00-bar design; switch on ANY lower validation MSE
           retune_margin  the same, switching only when the HAC t of the paired daily squared-
                          error difference (challenger - incumbent) is below -z(MARGIN_ALPHA /
                          number of challengers)
         The retune trials are asked from Optuna one after another and then fitted in parallel:
         with RXP_RETUNE_TRIALS <= 10 every challenger is a random draw of TPE's start-up phase, so
         the configurations equal a sequential study's (checked in the canary).
refit    one slice (RXP_SLICE = "i/n": job j belongs to slice j mod n) of the distinct refits of
         RXP_PHASE = control (the control arm alone: needs no pre-tune) or arms (every other arm,
         from plan.json; refits the control already makes are not repeated).  A refit at row i of
         one configuration fits [W + i - 2000, W + i) and predicts the rows up to the next
         multiple of 10 when i is one (serving the refit-every-10 arm) and row i alone otherwise
         (the refit-every-session arm): the same fit serves both cadences.  Saved in parts of
         PART_JOBS jobs; a restarted slice skips its finished parts.
assemble the arms' forecast rows from the refit parts: walk_<arm>_r<cadence>.npz (1469 rows each)
report   (local) experiments/close_trees_pretune_rfxgb_report.py over every model x input set found

ENV (RXP_*): MODEL (rf | xgb), BUCKET, TAG (run sub-directory, default local), SEED (0), SEEDS
(plan / assemble: the studies to merge, "0,1,..."), TRIALS (cap, 1000), TIMEOUT (seconds, 0 = none),
FOLDS (4), FOLD_LEN (250), WORKERS (pool size, 4), RETUNE_TRIALS (10), CHECKPOINTS ("10,100"),
CADENCES ("10,1"), R1_ARMS (arms also refitted every session: "control,frozen_all,retune_any,
retune_margin"), WALK_ROWS ("lo,hi": restrict refit and retune rows, smoke tests), SLICE ("0/1"),
PHASE (control | arms), WORK (default results/trees_pretune_2026-10-06/_work), GATE_TOL (1e-9).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "specs", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RXP_{name}", default)


MODEL = _env("MODEL", "rf")
if MODEL not in ("rf", "xgb"):
    raise SystemExit(f"RXP_MODEL must be rf or xgb, got {MODEL!r}")
BUCKETS = (
    "all_features",
    "baseline",
    "lean_vol_vix",
)  # moments_vix dropped (user, 2026-10-06)
BUCKET = _env("BUCKET", "all_features")
if BUCKET not in BUCKETS:
    raise SystemExit(f"RXP_BUCKET must be one of {BUCKETS}, got {BUCKET!r}")
TRAIN_WIN = 2000  # sessions: the 16:00-bar arms' window
# the tree spec's env axes, read by specs/causal_tune_trees.py when tree_setup executes its head
# (spawned workers inherit them): the model, the bucket, the window (leaf-minimum scaling), mask on
os.environ["HPC_KW_MODEL"] = MODEL
os.environ["HPC_KW_EXOG_BUCKET"] = BUCKET
os.environ["HPC_KW_TRAIN_WIN"] = str(TRAIN_WIN)
os.environ["HPC_KW_WINDOW_MASK"] = "1"
os.environ["SLURM_CPUS_PER_TASK"] = "1"

TAG = _env("TAG", "local")
SEED = int(_env("SEED", "0"))
SEEDS = tuple(int(v) for v in re.split("[,:]", _env("SEEDS", str(SEED))) if v.strip())
TRIALS = int(_env("TRIALS", "1000"))
TIMEOUT = float(_env("TIMEOUT", "0"))
FOLDS = int(_env("FOLDS", "4"))
FOLD_LEN = int(_env("FOLD_LEN", "250"))
WORKERS = int(_env("WORKERS", "4"))
RETUNE_TRIALS = int(_env("RETUNE_TRIALS", "10"))
CHECKPOINTS = tuple(
    int(v) for v in re.split("[,:]", _env("CHECKPOINTS", "10,100")) if v.strip()
)
CADENCES = tuple(
    int(v) for v in re.split("[,:]", _env("CADENCES", "10,1")) if v.strip()
)
R1_ARMS = tuple(
    v
    for v in re.split(
        "[,:]", _env("R1_ARMS", "control,frozen_all,retune_any,retune_margin")
    )
    if v.strip()
)
WALK_ROWS = _env("WALK_ROWS", "")
SLICE = _env("SLICE", "0/1")
PHASE = _env("PHASE", "control")
GATE_TOL = float(_env("GATE_TOL", "1e-9"))

OUT = REPO / "results" / "trees_pretune_2026-10-06"
WORK = Path(_env("WORK", str(OUT / "_work")))
DESIGN_DIR = WORK / "design"
DESIGN_SRC = Path(
    _env(
        "DESIGN_SRC",
        str(REPO / "results" / "linear_pretune_2026-10-06" / "_work" / "design"),
    )
)
CELL = f"{MODEL}_{BUCKET}"
RUN = WORK / TAG / CELL
ARR = WORK / "arrays" / BUCKET

FIRST_TRADE_DAY = (
    "2020-01-03"  # the deck's first day; the pre-tune sees sessions before it only
)
TPE_SEED_BASE = (
    20261006  # the day this study was designed; + RXP_SEED (+ the retune row)
)
RESUME_SALT = 7919  # as experiments/close_trees_pretune.py
RETUNE_EVERY = 250  # sessions between light retunes (the LightGBM study's)
RETUNE_FOLDS = (
    2  # x the linear spec's 125-session tail = the 250 sessions before the retune row
)
NEIGHBOURHOOD_FRACTION = 0.25  # the LightGBM study's retune box
MARGIN_ALPHA = (
    0.05  # one-sided level of the retune_margin switch test (Bonferroni-adjusted)
)
PART_JOBS = 40  # refits saved together in one part file
REFIT_BLOCK = (
    10  # a refit at a multiple of 10 predicts the block of the refit-every-10 arm
)
SAMPLE_ROWS_T1 = (
    0,
    1,
    2,
    733,
    1468,
)  # gate (b): every-session control rows checked against T1


def linear_constants(names: tuple[str, ...]) -> dict:
    """The first top-level literal assignment of each name in specs/causal_tune_linear.py."""
    src = (REPO / "specs" / "causal_tune_linear.py").read_text(encoding="utf-8")
    out: dict = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names and tgt.id not in out:
                    out[tgt.id] = ast.literal_eval(node.value)
    assert set(out) == set(names), out
    return out


_LIN = linear_constants(("VAL_TAIL", "EMBARGO"))
VAL_TAIL: int = _LIN["VAL_TAIL"]  # 125
EMBARGO: int = _LIN["EMBARGO"]  # 25


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {CELL} {msg}", flush=True)


def versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "optuna"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- a library the model does not use may be absent
            out[mod] = "absent"
    return out


def host() -> dict:
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "host": os.uname().nodename
        if hasattr(os, "uname")
        else os.environ.get("COMPUTERNAME", ""),
        "cpu": cpu,
    }


# ============================================================================ design
def sha_arrays(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        k: hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()
        for k in z.files
    }


def stage_design() -> None:
    """Copy the design caches into _work/design and record their array hashes."""
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    ref = json.loads(
        (
            REPO / "results" / "close_studies_2026-10-03" / "design_hashes.json"
        ).read_text()
    )["designs"]
    rows, sums = [], {}
    for b in BUCKETS:
        for seg in ("bar1600", "last30"):
            name = f"design_{seg}_{b}.npz"
            src, dst = DESIGN_SRC / name, DESIGN_DIR / name
            if not dst.is_file():
                shutil.copy2(src, dst)
            h = sha_arrays(dst)
            assert h == sha_arrays(src), f"{dst} differs from its source {src}"
            sums[name] = h
            r = ref.get(f"{seg}_{b}")
            for k, v in h.items():
                rows.append(
                    dict(
                        design=name,
                        array=k,
                        sha256=v,
                        reference_sha256=(r["arrays"][k]["sha256"] if r else ""),
                        matches_reference=(
                            r is not None and r["arrays"][k]["sha256"] == v
                        ),
                    )
                )
    # the bar1600 all_features design against the 2026-10-03 export of the stored cache (every row)
    ex = pd.read_parquet(
        REPO
        / "results"
        / "close_studies_2026-10-03"
        / "forecasts"
        / "design_bar1600_all_features.parquet"
    )
    z = np.load(DESIGN_DIR / "design_bar1600_all_features.npz", allow_pickle=False)
    names = [str(n) for n in z["names"]]
    d = np.abs(ex[names].to_numpy(float) - z["X"])
    rows.append(
        dict(
            design="design_bar1600_all_features.npz",
            array="X vs the 2026-10-03 export (stored cache)",
            sha256=f"max abs diff {d.max():.3e}; columns differing {(d.max(axis=0) > 0).sum()} of {d.shape[1]}",
            reference_sha256="",
            matches_reference=bool(d.max() == 0),
        )
    )
    (DESIGN_DIR / "design_sha256.json").write_text(json.dumps(sums, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "design_check.csv", index=False)
    say(
        f"design: {len(sums)} caches copied, hashes in {DESIGN_DIR / 'design_sha256.json'}"
    )


def check_design(name: str) -> Path:
    path = DESIGN_DIR / name
    want = json.loads((DESIGN_DIR / "design_sha256.json").read_text())[name]
    if sha_arrays(path) != want:
        raise SystemExit(f"{path}: array hashes differ from design_sha256.json")
    return path


# ============================================================================ data
def _save_atomic(path: Path, a: np.ndarray) -> None:
    if path.is_file():
        return
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, path)


def prepare_arrays() -> dict:
    """The two designs as .npy files the workers memory-map, plus the row bookkeeping."""
    ARR.mkdir(parents=True, exist_ok=True)
    bar = check_design(f"design_bar1600_{BUCKET}.npz")
    l30 = check_design(f"design_last30_{BUCKET}.npz")
    z = np.load(bar, allow_pickle=False)
    W = int(z["W"])
    assert W == TRAIN_WIN, (W, TRAIN_WIN)
    info = dict(
        W=W,
        n=len(z["y"]),
        date=np.asarray(z["date"]),
        y=np.asarray(z["y"], float),
        names=np.asarray(z["names"]),
    )
    paths = {"bar_X": ARR / "bar_X.npy", "bar_y": ARR / "bar_y.npy"}
    _save_atomic(paths["bar_X"], np.ascontiguousarray(z["X"], dtype=np.float64))
    _save_atomic(paths["bar_y"], np.ascontiguousarray(z["y"], dtype=np.float64))
    q = np.load(l30, allow_pickle=False)
    d30 = pd.to_datetime(q["date"])
    pre = (d30.hour == 16) & (d30 < pd.Timestamp(FIRST_TRADE_DAY))
    assert (np.asarray(q["names"]) == info["names"]).all(), (
        "column names differ between the caches"
    )
    paths |= {"pre_X": ARR / "pre_X.npy", "pre_y": ARR / "pre_y.npy"}
    _save_atomic(paths["pre_X"], np.ascontiguousarray(q["X"][pre], dtype=np.float64))
    _save_atomic(paths["pre_y"], np.ascontiguousarray(q["y"][pre], dtype=np.float64))
    info["pre_date"] = np.asarray(q["date"][pre])
    info["pre_n"] = int(pre.sum())
    info["paths"] = {k: str(v) for k, v in paths.items()}
    return info


def pretune_folds(
    info: dict, n_folds: int = FOLDS, fold_len: int = FOLD_LEN
) -> list[dict]:
    """Blocked walk-forward folds tiling the last n_folds x fold_len pre-2020 16:00 sessions."""
    E = info["pre_n"]
    out = []
    for f in range(n_folds):
        v1 = E - (n_folds - 1 - f) * fold_len
        v0 = v1 - fold_len
        a, b = v0 - EMBARGO - TRAIN_WIN, v0 - EMBARGO
        if a < 0:
            raise SystemExit(
                f"{n_folds} folds x {fold_len} sessions do not fit in {E} pre-2020 sessions"
            )
        out.append(
            dict(
                fold=f,
                fit=(a, b),
                val=(v0, v1),
                fit_first=str(info["pre_date"][a])[:10],
                val_first=str(info["pre_date"][v0])[:10],
                val_last=str(info["pre_date"][v1 - 1])[:10],
            )
        )
    return out


def retune_folds(info: dict, row: int) -> list[dict]:
    """RETUNE_FOLDS folds of VAL_TAIL sessions tiling the sessions before OOS row `row`."""
    t = info["W"] + row
    out = []
    for f in range(RETUNE_FOLDS):
        v1 = t - (RETUNE_FOLDS - 1 - f) * VAL_TAIL
        v0 = v1 - VAL_TAIL
        a, b = v0 - EMBARGO - TRAIN_WIN, v0 - EMBARGO
        assert a >= 0 and v1 <= t, (row, a, v1, t)
        out.append(dict(fold=f, fit=(a, b), val=(v0, v1)))
    return out


# ============================================================================ pool
_POOL = None


def pool():
    global _POOL
    if _POOL is None and WORKERS > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        _POOL = ProcessPoolExecutor(
            WORKERS, mp_context=multiprocessing.get_context("spawn")
        )
    return _POOL


def run_jobs(fn, jobs: list[dict]) -> list[dict]:
    p = pool()
    return list(p.map(fn, jobs)) if p is not None else [fn(j) for j in jobs]


# ============================================================================ helpers
def spaces() -> dict:
    import causal_tune_trees_optuna_jobs as OJ

    return OJ.SPACES[MODEL]


def shipped() -> dict:
    import causal_tune_trees_optuna_jobs as OJ

    return OJ.shipped_config(MODEL)


def shipped_rounds() -> int | None:
    import causal_tune_trees_optuna_jobs as OJ

    return OJ.shipped_rounds(MODEL)


def encode(cfg: dict) -> np.ndarray:
    import causal_tune_trees_optuna_jobs as OJ

    return OJ.encode(MODEL, cfg)


def decode(vec) -> dict:
    import causal_tune_trees_optuna_jobs as OJ

    return OJ.decode(MODEL, vec)


def cfg_key(cfg: dict, rounds: int | None) -> str:
    """A short exact key of a (configuration, rounds) pair."""
    return hashlib.sha1(encode(cfg).tobytes() + str(rounds).encode()).hexdigest()[:16]


def fold_jobs(info: dict, folds: list[dict], cfg: dict, src: str) -> list[dict]:
    P = info["paths"]
    return [
        dict(X=P[f"{src}_X"], y=P[f"{src}_y"], fit=f["fit"], val=f["val"], cfg=cfg)
        for f in folds
    ]


def config_rounds(fold_rounds) -> int | None:
    """A configuration's refit rounds: the median of its folds' early-stopped rounds (XGBoost);
    None for the random forest (100 trees, no rounds)."""
    if MODEL == "rf":
        return None
    return int(np.round(np.median(np.asarray(fold_rounds, dtype=float))))


def journal(path: Path):
    import optuna

    # the open()-based lock (not the default symlink lock): one writer per journal, and it works on
    # Windows without the symlink privilege
    try:  # Optuna >= 4.0
        J = optuna.storages.journal
        backend = J.JournalFileBackend(
            str(path), lock_obj=J.JournalFileOpenLock(str(path))
        )
    except AttributeError:  # older Optuna
        backend = optuna.storages.JournalFileStorage(
            str(path), lock_obj=optuna.storages.JournalFileOpenLock(str(path))
        )
    return optuna.storages.JournalStorage(backend)


def oos_rows(info: dict) -> int:
    return info["n"] - info["W"]


def walk_bounds(info: dict) -> tuple[int, int]:
    if WALK_ROWS:
        lo, hi = (int(v) for v in re.split("[,:]", WALK_ROWS))
        return lo, hi
    return 0, oos_rows(info)


# ============================================================================ gate
def load_stored(info: dict, fname: str) -> np.ndarray:
    """A stored table's 16:00 forecasts on the 1469 forecast rows."""
    d = pd.read_parquet(REPO / "results" / "spxw_pnl" / fname)
    t = pd.to_datetime(d["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    s = pd.Series(d["yhat"].to_numpy(float), index=t)
    s = s[s.index.strftime("%H:%M") == "16:00"]
    want = pd.to_datetime(info["date"][info["W"] :])
    assert len(s) == len(want) and (s.index == want).all(), (fname, len(s))
    return s.to_numpy(float)


def stage_gate() -> None:
    import causal_tune_trees_optuna_jobs as OJ
    import causal_tune_trees_tuned_jobs as J
    import close_trees_pretune_rfxgb_jobs as CJ

    from src.models.window_mask import window_keep

    info = prepare_arrays()
    RUN.mkdir(parents=True, exist_ok=True)
    flag = RUN / f"GATE_OK_{CELL}"
    flag.unlink(missing_ok=True)
    rows = []
    # (a) the fold fitter == the Optuna spec's fit_trial (shipped configuration, last pre-tune fold)
    f = pretune_folds(info)[-1]
    job = fold_jobs(info, [f], shipped(), "pre")[0]
    mine = CJ.fit_fold(job)
    X, y = J.arr(job["X"]), J.arr(job["y"])
    (a, b), (c, d) = f["fit"], f["val"]
    keep = window_keep(np.asarray(X[a:b]))
    ref = OJ.fit_trial(
        np.ascontiguousarray(np.asarray(X[a:b])[:, keep]),
        np.ascontiguousarray(y[a:b]),
        np.ascontiguousarray(np.asarray(X[c:d])[:, keep]),
        np.ascontiguousarray(y[c:d]),
        shipped(),
        True,
    )
    for k in ("val_mse", "val_qlike", "rounds"):
        rows.append(
            dict(
                gate="fold fitter == Optuna spec fit_trial (shipped configuration, last pre-tune fold)",
                item=k,
                value=float(mine[k]),
                reference=float(ref[k]),
                max_rel_diff=abs(float(mine[k]) / float(ref[k]) - 1),
                ok=float(mine[k]) == float(ref[k]),
            )
        )
    # (b) control refits against the stored masked cluster tables
    W, n_oos = info["W"], oos_rows(info)
    P = info["paths"]
    checks = [
        (
            "T10",
            f"yhat_subtree_{MODEL}_{BUCKET}.parquet",
            i,
            min(REFIT_BLOCK, n_oos - i),
        )
        for i in (0, 730, 1460)
    ]
    checks += [
        ("T1", f"yhat_subtree_daily_{BUCKET}_{MODEL}.parquet", i, 1)
        for i in SAMPLE_ROWS_T1
    ]
    jobs = [
        dict(
            X=P["bar_X"],
            y=P["bar_y"],
            t=W + i,
            W=W,
            k=k,
            cfg=shipped(),
            rounds=shipped_rounds(),
            qcfg=None,
            shap=False,
        )
        for _, _, i, k in checks
    ]
    outs = run_jobs(CJ.refit, jobs)
    for (rung, fname, i, k), o in zip(checks, outs):
        if not (REPO / "results" / "spxw_pnl" / fname).is_file():
            rows.append(
                dict(
                    gate=f"control refit == stored {rung} table",
                    item=f"{fname} absent",
                    value=np.nan,
                    reference=np.nan,
                    max_rel_diff=np.nan,
                    ok=True,
                )
            )
            continue
        st = load_stored(info, fname)[i : i + k]
        rel = float(np.max(np.abs(o["preds"] / st - 1)))
        rows.append(
            dict(
                gate=f"control refit == stored {rung} table ({fname})",
                item=f"rows {i}..{i + k - 1} (fit {o['fit_sec']:.1f}s, {o['n_kept']} columns kept)",
                value=float(o["preds"][0]),
                reference=float(st[0]),
                max_rel_diff=rel,
                ok=rel <= GATE_TOL,
            )
        )
    g = pd.DataFrame(rows)
    g["model"], g["bucket"], g["tag"] = MODEL, BUCKET, TAG
    g["versions"] = json.dumps(versions())
    g["host"] = json.dumps(host())
    g.to_csv(RUN / f"gate_{CELL}.csv", index=False)
    print(g[["gate", "item", "max_rel_diff", "ok"]].to_string(index=False))
    if not g["ok"].all():
        raise SystemExit("GATE FAILED")
    flag.write_text(time.ctime() + "\n")
    say(f"gate OK; fold fit {mine['sec']:.1f}s")


# ============================================================================ pretune
def stage_pretune() -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    info = prepare_arrays()
    folds = pretune_folds(info)
    RUN.mkdir(parents=True, exist_ok=True)
    jpath = RUN / f"pretune_s{SEED}.journal"
    name = f"pretune_{CELL}_s{SEED}_f{FOLDS}x{FOLD_LEN}"
    study = optuna.create_study(
        study_name=name,
        storage=journal(jpath),
        direction="minimize",
        load_if_exists=True,
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    seed_eff = TPE_SEED_BASE + SEED + RESUME_SALT * len(done)
    study.sampler = optuna.samplers.TPESampler(seed=seed_eff)
    if not study.trials:
        study.enqueue_trial(shipped())
    meta_path = RUN / f"pretune_s{SEED}_resumes.json"
    resumes = json.loads(meta_path.read_text()) if meta_path.is_file() else []
    resumes.append(
        dict(
            completed_before=len(done),
            sampler_seed=seed_eff,
            start=time.ctime(),
            **host(),
        )
    )
    meta_path.write_text(json.dumps(resumes, indent=1))
    say(
        f"pre-tune {name}: {len(done)} trials complete, cap {TRIALS}, timeout {TIMEOUT:.0f}s; folds "
        + "; ".join(
            f"{f['val_first']}..{f['val_last']} (fit from {f['fit_first']})"
            for f in folds
        )
    )
    space = spaces()
    t0 = time.time()

    def objective(trial) -> float:
        import causal_tune_trees_optuna_jobs as OJ
        import close_trees_pretune_rfxgb_jobs as CJ

        cfg = OJ.suggest(trial, space)
        outs = run_jobs(CJ.fit_fold, fold_jobs(info, folds, cfg, "pre"))
        for k in ("val_mse", "val_qlike", "rounds", "sec", "n_kept", "rounds_max"):
            trial.set_user_attr(f"fold_{k}", [float(o[k]) for o in outs])
        v = float(np.mean([o["val_mse"] for o in outs]))
        if trial.number % 25 == 0 or trial.number < 3:
            say(
                f"  trial {trial.number}: val MSE {v:.5f} rounds {[o['rounds'] for o in outs]} "
                f"{sum(o['sec'] for o in outs):.0f} fit-s, elapsed {time.time() - t0:.0f}s"
            )
        return v

    left = TRIALS - len(done)
    if left > 0:
        study.optimize(
            objective,
            n_trials=left,
            timeout=(TIMEOUT or None),
            n_jobs=1,
            show_progress_bar=False,
            gc_after_trial=False,
        )
    export_pretune(study, folds, info)


def export_pretune(study, folds: list[dict], info: dict) -> None:
    import optuna

    tr = sorted(
        (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        key=lambda t: t.number,
    )
    nums = [t.number for t in tr]
    assert nums == list(range(len(tr))), "pre-tune trials are not contiguous"
    space = spaces()
    arrs = dict(
        params=np.vstack([encode({a: t.params[a] for a in space}) for t in tr]),
        val_mse=np.array([t.value for t in tr]),
    )
    for k in ("val_mse", "val_qlike", "rounds", "sec", "n_kept", "rounds_max"):
        arrs[f"fold_{k}"] = np.array(
            [t.user_attrs[f"fold_{k}"] for t in tr], dtype=float
        )
    meta = dict(
        model=MODEL,
        bucket=BUCKET,
        seed=SEED,
        tpe_seed_base=TPE_SEED_BASE,
        folds=folds,
        fold_len=FOLD_LEN,
        embargo=EMBARGO,
        train_win=TRAIN_WIN,
        axes=list(space),
        shipped=shipped(),
        pre_sessions=info["pre_n"],
        pre_first=str(info["pre_date"][0]),
        pre_last=str(info["pre_date"][-1]),
        versions=versions(),
        trials_cap=TRIALS,
        timeout=TIMEOUT,
        **host(),
    )
    tmp = RUN / f"pretune_s{SEED}.tmp.npz"
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrs)
    os.replace(tmp, RUN / f"pretune_s{SEED}.npz")
    b = int(np.argmin(arrs["val_mse"]))
    say(
        f"pre-tune export: {len(tr)} trials; best trial {b} val MSE {arrs['val_mse'][b]:.5f} "
        f"(shipped {arrs['val_mse'][0]:.5f}), {arrs['fold_sec'].sum() / 3600:.2f} core-h of fold fits"
    )


def load_pretune(seeds=SEEDS) -> list[dict]:
    recs = []
    for s in seeds:
        f = RUN / f"pretune_s{s}.npz"
        if not f.is_file():
            raise SystemExit(f"{f} missing (pretune stage with RXP_SEED={s})")
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files if k != "meta"}
        d["meta"] = json.loads(str(z["meta"]))
        d["seed"] = s
        recs.append(d)
    return recs


def merge_seeds(by_seed: list[dict]) -> dict:
    """Independent seeded studies merged by interleaving (experiments/close_trees_pretune.py)."""
    order = [
        (s, j)
        for j in range(max(len(r["val_mse"]) for r in by_seed))
        for s, r in enumerate(by_seed)
        if j < len(r["val_mse"])
    ]
    keys = [k for k in by_seed[0] if isinstance(by_seed[0][k], np.ndarray)]
    out: dict = {k: np.stack([by_seed[s][k][j] for s, j in order]) for k in keys}
    out["study"] = np.array([by_seed[s]["seed"] for s, _ in order], dtype=np.int64)
    out["trial"] = np.array([j for _, j in order], dtype=np.int64)
    out["meta"] = by_seed[0]["meta"]
    return out


def best_after(rec: dict, k: int) -> int:
    j = rec.get("trial", np.arange(len(rec["val_mse"])))
    v = np.where((j < k) & ~np.isnan(rec["val_mse"]), rec["val_mse"], np.inf)
    return int(np.argmin(v))


# ============================================================================ plan
def box_space(inc: dict) -> dict:
    """The spec's space confined to a box around the incumbent: NEIGHBOURHOOD_FRACTION of each
    numeric axis's range (log units on log axes), centred, clipped; a categorical axis keeps the
    incumbent and its neighbours in the ordered choice list."""
    out: dict = {}
    for a, s in spaces().items():
        if s[0] == "cat":
            ch = list(s[1])
            i = ch.index(inc[a])
            out[a] = ("cat", tuple(ch[max(0, i - 1) : i + 2]))
            continue
        kind, lo, hi, log = s
        v = float(inc[a])
        if log:
            h = NEIGHBOURHOOD_FRACTION * (math.log(hi) - math.log(lo)) / 2
            blo, bhi = max(lo, v * math.exp(-h)), min(hi, v * math.exp(h))
        else:
            h = NEIGHBOURHOOD_FRACTION * (hi - lo) / 2
            blo, bhi = max(lo, v - h), min(hi, v + h)
        if kind == "int":
            out[a] = (
                "int",
                int(max(lo, math.floor(blo))),
                int(min(hi, math.ceil(bhi))),
                log,
            )
        else:
            out[a] = ("float", blo, bhi, log)
    return out


def hac_t(d: np.ndarray) -> float:
    """Mean / Newey-West standard error of a paired loss difference (dm_test's statistic)."""
    from src.evaluation.diebold_mariano import dm_test

    return float(dm_test(d, np.zeros_like(d))["dm"])


_RETUNE_CACHE: dict = {}


def retune(info: dict, row: int, inc: dict, inc_rounds: int | None) -> dict:
    """One light retune at OOS row `row` around the incumbent: RETUNE_TRIALS trials asked from a
    TPE study (trial 0 = the incumbent), every fold of every trial fitted in one parallel batch."""
    import optuna

    import causal_tune_trees_optuna_jobs as OJ
    import close_trees_pretune_rfxgb_jobs as CJ

    key = (row, cfg_key(inc, inc_rounds))
    if key in _RETUNE_CACHE:
        return _RETUNE_CACHE[key]
    assert RETUNE_TRIALS <= 10, (
        "parallel retune trials need TPE's random start-up phase (10 trials)"
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = retune_folds(info, row)
    space = box_space(inc)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED_BASE + SEEDS[0] + row),
    )
    study.enqueue_trial({a: inc[a] for a in space})
    trials = [study.ask() for _ in range(RETUNE_TRIALS)]
    cfgs = [OJ.suggest(t, space) for t in trials]
    assert cfgs[0] == {a: inc[a] for a in space}, "trial 0 is not the incumbent"
    # the same configurations as a sequential study of the same seed (start-up draws ignore values)
    seq_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED_BASE + SEEDS[0] + row),
    )
    seq_study.enqueue_trial({a: inc[a] for a in space})
    seq: list[dict] = []
    seq_study.optimize(
        lambda t: float(len(seq.append(OJ.suggest(t, space)) or seq)),
        n_trials=RETUNE_TRIALS,
    )
    assert seq == cfgs, "parallel retune trials differ from a sequential study's"
    outs = run_jobs(
        CJ.fit_fold, [j for c in cfgs for j in fold_jobs(info, folds, c, "bar")]
    )
    recs = []
    for i, (t, c) in enumerate(zip(trials, cfgs)):
        o = outs[i * RETUNE_FOLDS : (i + 1) * RETUNE_FOLDS]
        e2 = np.concatenate([x["e2"] for x in o])
        recs.append(
            dict(
                cfg=c,
                val_mse=float(np.mean(e2)),
                fold_mse=[x["val_mse"] for x in o],
                val_qlike=float(np.mean([x["val_qlike"] for x in o])),
                rounds=[x["rounds"] for x in o],
                sec=sum(x["sec"] for x in o),
                e2=e2,
            )
        )
        study.tell(t, float(np.mean(e2)))
    mse = np.array([r["val_mse"] for r in recs])
    b = 1 + int(np.argmin(mse[1:]))  # the best challenger
    t_stat = hac_t(recs[b]["e2"] - recs[0]["e2"])
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - MARGIN_ALPHA / (len(recs) - 1))
    out = dict(
        row=row,
        date=str(info["date"][info["W"] + row])[:10],
        trials=recs,
        best=b,
        inc_mse=float(mse[0]),
        best_mse=float(mse[b]),
        t=t_stat,
        z=z,
        switch_any=bool(mse[b] < mse[0]),
        switch_margin=bool(t_stat < -z),
        new_cfg=recs[b]["cfg"],
        new_rounds=config_rounds(recs[b]["rounds"]),
        sec=sum(r["sec"] for r in recs),
    )
    _RETUNE_CACHE[key] = out
    say(
        f"  retune row {row} ({out['date']}): incumbent {mse[0]:.5f}, best challenger (trial {b}) {mse[b]:.5f}, "
        f"HAC t {t_stat:+.2f} vs -{z:.2f}: any={out['switch_any']} margin={out['switch_margin']}"
    )
    return out


def jsonable(cfg: dict) -> dict:
    return {
        k: (
            None
            if v is None
            else (int(v) if isinstance(v, (int, np.integer)) else float(v))
        )
        for k, v in cfg.items()
    }


def control_schedule() -> list:
    return [[0, jsonable(shipped()), shipped_rounds()]]


def stage_plan() -> None:
    info = prepare_arrays()
    n_oos = oos_rows(info)
    recs = load_pretune()
    rec = merge_seeds(recs)
    n_tr = int(rec["trial"].max()) + 1  # trials of the longest study
    arms: dict[str, list] = {"control": control_schedule()}
    picks = []
    for k in sorted({k for k in CHECKPOINTS if k < n_tr}) + [n_tr]:
        b = best_after(rec, k)
        cfg = decode(rec["params"][b])
        r = config_rounds(rec["fold_rounds"][b])
        name = "frozen_all" if k == n_tr else f"frozen_k{k}"
        arms[name] = [[0, jsonable(cfg), r]]
        picks.append(
            dict(
                arm=name,
                k=k,
                trials_total=int(np.sum(rec["trial"] < k)),
                study=int(rec["study"][b]),
                trial=int(rec["trial"][b]),
                val_mse=float(rec["val_mse"][b]),
                rounds=r,
                **{f"cfg_{a}": v for a, v in cfg.items()},
            )
        )
    W = info["W"]
    trade0 = int(
        np.searchsorted(pd.to_datetime(info["date"][W:]), pd.Timestamp(FIRST_TRADE_DAY))
    )
    row0 = (
        trade0 // REFIT_BLOCK
    ) * REFIT_BLOCK  # the refit row at or before the first trade day
    lo, hi = walk_bounds(info)
    rt_rows = [r for r in range(row0, n_oos, RETUNE_EVERY) if lo <= r < hi]
    cfg0, r0 = arms["frozen_all"][0][1], arms["frozen_all"][0][2]
    rt_log, rt_trials = [], []
    for path in ("retune_any", "retune_margin"):
        sched = [[0, cfg0, r0]]
        inc, inc_r = decode(encode(cfg0)), r0
        for row in rt_rows:
            res = retune(info, row, inc, inc_r)
            sw = res["switch_any"] if path == "retune_any" else res["switch_margin"]
            if sw:
                inc, inc_r = res["new_cfg"], res["new_rounds"]
                sched.append([row, jsonable(inc), inc_r])
            rt_log.append(
                dict(
                    path=path,
                    row=row,
                    date=res["date"],
                    incumbent_val_mse=res["inc_mse"],
                    best_challenger_val_mse=res["best_mse"],
                    best_trial=res["best"],
                    rel_gain=1 - res["best_mse"] / res["inc_mse"],
                    hac_t=res["t"],
                    z_crit=res["z"],
                    switched=sw,
                    rounds_in_force=inc_r,
                    retune_fit_sec=res["sec"],
                    **{f"cfg_{a}": inc[a] for a in spaces()},
                )
            )
        arms[path] = sched
    for (row, _), res in _RETUNE_CACHE.items():
        for j, r in enumerate(res["trials"]):
            rt_trials.append(
                dict(
                    row=row,
                    date=res["date"],
                    incumbent_key=cfg_key(res["trials"][0]["cfg"], None),
                    trial=j,
                    val_mse=r["val_mse"],
                    fold0_mse=r["fold_mse"][0],
                    fold1_mse=r["fold_mse"][1],
                    val_qlike=r["val_qlike"],
                    rounds_median=config_rounds(r["rounds"]),
                    sec=r["sec"],
                    **{f"cfg_{a}": v for a, v in r["cfg"].items()},
                )
            )
    RUN.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rt_log).to_csv(RUN / "retune_log.csv", index=False)
    pd.DataFrame(rt_trials).to_csv(RUN / "retune_trials.csv", index=False)
    pd.DataFrame(picks).to_csv(RUN / "frozen_picks.csv", index=False)
    plan = dict(
        model=MODEL,
        bucket=BUCKET,
        seeds=list(SEEDS),
        trials_longest_study=n_tr,
        trials_total=int(len(rec["val_mse"])),
        walk_rows=[lo, hi],
        arms=arms,
        r1_arms=[a for a in R1_ARMS if a in arms],
        cadences=list(CADENCES),
        versions=versions(),
        **host(),
    )
    (RUN / "plan.json").write_text(json.dumps(plan, indent=1))
    say(
        f"plan: {len(arms)} arms; retunes at rows {rt_rows}; {sum(len(v) for v in arms.values())} schedule entries"
    )


# ============================================================================ refit / assemble
def arm_rows(info: dict, arm: str, r1_arms) -> tuple[list[int], list[int]]:
    """(refit-every-10 rows, refit-every-session rows) an arm needs, inside WALK_ROWS."""
    lo, hi = walk_bounds(info)
    n_oos = oos_rows(info)
    r10 = (
        [i for i in range(0, n_oos, REFIT_BLOCK) if lo <= i < hi]
        if 10 in CADENCES
        else []
    )
    r1 = (
        [i for i in range(n_oos) if lo <= i < hi]
        if (1 in CADENCES and arm in r1_arms)
        else []
    )
    return r10, r1


def in_force(sched: list, i: int) -> tuple[dict, int | None]:
    _, cfg, r = [s for s in sched if s[0] <= i][-1]
    return cfg, r


def job_list(info: dict, arms: dict, r1_arms) -> list[dict]:
    """Distinct (row, configuration, rounds) refits of the given arms, in a fixed order."""
    n_oos = oos_rows(info)
    seen: dict[tuple, int] = {}
    jobs: list[dict] = []
    for arm in arms:
        r10, r1 = arm_rows(info, arm, r1_arms)
        for i in sorted(set(r10) | set(r1)):
            cfg, r = in_force(arms[arm], i)
            key = (i, cfg_key(decode(encode(cfg)), r))
            if key in seen:
                continue
            seen[key] = len(jobs)
            k = min(REFIT_BLOCK, n_oos - i) if i % REFIT_BLOCK == 0 else 1
            jobs.append(dict(row=i, key=f"{i}|{key[1]}", cfg=cfg, rounds=r, k=k))
    return jobs


def phase_arms(phase: str) -> tuple[dict, tuple]:
    if phase == "control":
        return {"control": control_schedule()}, R1_ARMS
    plan = json.loads((RUN / "plan.json").read_text())
    return {a: s for a, s in plan["arms"].items() if a != "control"}, tuple(
        plan["r1_arms"]
    )


def stage_refit() -> None:
    import close_trees_pretune_rfxgb_jobs as CJ

    info = prepare_arrays()
    W = info["W"]
    i_sl, n_sl = (int(v) for v in SLICE.split("/"))
    arms, r1_arms = phase_arms(PHASE)
    jobs = job_list(info, arms, r1_arms)
    if PHASE == "arms":  # refits the control phase makes are not repeated
        ctrl = {
            j["key"] for j in job_list(info, {"control": control_schedule()}, R1_ARMS)
        }
        jobs = [j for j in jobs if j["key"] not in ctrl]
    mine = [j for q, j in enumerate(jobs) if q % n_sl == i_sl]
    P = info["paths"]
    out_dir = RUN / "refits"
    out_dir.mkdir(parents=True, exist_ok=True)
    say(
        f"refit {PHASE} slice {i_sl}/{n_sl}: {len(mine)} of {len(jobs)} distinct fits, pool {WORKERS}"
    )
    t0, fit_sum = time.time(), 0.0
    for p0 in range(0, len(mine), PART_JOBS):
        part = out_dir / f"{PHASE}_s{i_sl}of{n_sl}_p{p0 // PART_JOBS}.npz"
        if part.is_file():
            continue
        chunk = mine[p0 : p0 + PART_JOBS]
        outs = run_jobs(
            CJ.refit,
            [
                dict(
                    X=P["bar_X"],
                    y=P["bar_y"],
                    t=W + j["row"],
                    W=W,
                    k=j["k"],
                    cfg=decode(encode(j["cfg"])),
                    rounds=j["rounds"],
                    qcfg=None,
                    shap=False,
                )
                for j in chunk
            ],
        )
        fit_sum += sum(o["fit_sec"] for o in outs)
        tmp = part.with_name(part.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            keys=np.array([j["key"] for j in chunk]),
            k=np.array([j["k"] for j in chunk]),
            preds=np.concatenate([o["preds"] for o in outs]),
            fit_sec=np.array([o["fit_sec"] for o in outs]),
            n_kept=np.array([o["n_kept"] for o in outs]),
            meta=json.dumps(dict(versions=versions(), **host())),
        )
        os.replace(tmp, part)
    say(
        f"refit {PHASE} slice {i_sl}/{n_sl} done: {fit_sum / 3600:.2f} core-h this run, wall {time.time() - t0:.0f}s"
    )


def stage_assemble() -> None:
    info = prepare_arrays()
    n_oos = oos_rows(info)
    lo, hi = walk_bounds(info)
    got: dict[str, tuple[np.ndarray, float, int]] = {}
    for f in sorted((RUN / "refits").glob("*_p*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f, allow_pickle=False)
        off = 0
        for key, k, s, nk in zip(z["keys"], z["k"], z["fit_sec"], z["n_kept"]):
            got[str(key)] = (z["preds"][off : off + int(k)], float(s), int(nk))
            off += int(k)
    plan = (
        json.loads((RUN / "plan.json").read_text())
        if (RUN / "plan.json").is_file()
        else None
    )
    arms = dict(plan["arms"]) if plan else {"control": control_schedule()}
    r1_arms = tuple(plan["r1_arms"]) if plan else R1_ARMS
    missing, written = [], []
    for arm, sched in arms.items():
        r10, r1 = arm_rows(info, arm, r1_arms)
        for cad, rows in ((10, r10), (1, r1)):
            if not rows:
                continue
            pred = np.full(n_oos, np.nan)
            sec, kept = [], []
            for i in rows:
                cfg, r = in_force(sched, i)
                key = f"{i}|{cfg_key(decode(encode(cfg)), r)}"
                if key not in got:
                    missing.append((arm, cad, i))
                    continue
                p, s, nk = got[key]
                if cad == 10:
                    pred[i : i + len(p)] = p
                else:
                    pred[i] = p[0]
                sec.append(s)
                kept.append(nk)
            if any(m[0] == arm and m[1] == cad for m in missing):
                continue
            sl = pred[lo:hi]
            assert np.isfinite(sl).all(), (arm, cad)
            tmp = RUN / f"walk_{arm}_r{cad}.tmp.npz"
            np.savez_compressed(
                tmp,
                pred=pred,
                rows=np.array(rows),
                fit_sec=np.array(sec),
                n_kept=np.array(kept),
                schedule=json.dumps(sched),
                meta=json.dumps(
                    dict(
                        refit_every=cad, walk_rows=[lo, hi], model=MODEL, bucket=BUCKET
                    )
                ),
            )
            os.replace(tmp, RUN / f"walk_{arm}_r{cad}.npz")
            written.append(f"{arm}_r{cad}")
    say(f"assemble: wrote {written}; missing refits {len(missing)}")
    if missing:
        print("first missing:", missing[:10])
        raise SystemExit(1)


# ============================================================================ main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    stage = sys.argv[1]
    t_start = time.time()
    try:
        if stage == "design":
            stage_design()
        elif stage == "gate":
            stage_gate()
        elif stage == "pretune":
            stage_pretune()
        elif stage == "plan":
            stage_plan()
        elif stage == "refit":
            stage_refit()
        elif stage == "assemble":
            stage_assemble()
        elif stage == "report":
            import close_trees_pretune_rfxgb_report as R

            R.main()
        else:
            raise SystemExit(f"unknown stage {stage!r}")
    finally:
        if _POOL is not None:
            _POOL.shutdown()
    say(f"stage {stage} done in {time.time() - t_start:.0f}s")
