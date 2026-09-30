"""causal_tune_trees_optuna -- per-bar trees tuned by Optuna (TPE) at EVERY session, refit every session.

PURPOSE: the 2026-09-29 user decisions (writeup/CAMPAIGN_16H_2026-09-29.md, checklist I3):
replace the tuned spec's random search (specs/causal_tune_trees_tuned.py: 32 fixed
candidates every 250 sessions, refit every 10) with Optuna's TPE on each tuning point's
validation tail, 50 trials ("let's see if 50 is enough"), trees refitted EVERY session
as the per-bar linear arms are, and ablate the tuning cadence TUNE_PER in {1, 5, 25, 250}.
Design, target, window and executor call are the untuned spec's (specs/causal_tune_trees.py:
SEGMENT = one regular-hours bar, TRAIN_WIN = 2000 sessions, the de-duplicated per-bar design
of commit 47f7f9c), whose setup (the shipped parameter dicts after the leaf-minimum scaling,
make_model, native_importance, contributions) is EXECUTED from its source by
specs/causal_tune_trees_tuned_jobs.tree_setup, so the untuned arm's code is the only copy.

TWO STAGES (env STAGE), both through the untuned spec's run_executor call and its
START / END / HALO chunk seam (HALO = the training window W):

STAGE 1 -- "tune": a tuning point at EVERY forecast row t of the chunk.  The window
  [t - W, t) is split by src.models.reclasso_har.forward_window_split(t, W, VAL_TAIL = 125,
  EMBARGO = 25) (VAL_TAIL and EMBARGO READ from specs/causal_tune_linear.py): fit block
  [t - W, t - 150), 25-row embargo, validation tail [t - 125, t); the split is asserted to
  tile the window, and ONLY the window's rows are handed to the worker.  A fresh Optuna
  study per point (specs/causal_tune_trees_optuna_jobs.tune_point): TPESampler(seed =
  TPE_SEED_BASE + the whole-series OOS row), N_TRIALS = 50 trials run SEQUENTIALLY in one
  process (deterministic), trial 0 enqueued = the shipped configuration, no warm start from
  any other point -- so the points are independent (they run in parallel across days) and
  every TUNE_PER below reuses them.  Objective = validation MSE on the transformed target
  (the linear arms' criterion, the rule of record); recorded per trial: validation QLIKE of
  the executor's Duan back-transform (the tuned spec's val_losses), early-stopped rounds,
  the round cap, seconds, parameters.  TPE's first k trials of a 50-trial study are exactly
  a k-trial study (the sampler conditions only on the trials before), so best-of-10 /
  best-of-25 come for free (gated).  Every trial of every point is saved.

STAGE 2 -- "refit": every forecast row t is forecast by a model fitted on the full window
  [t - W, t) with the configuration in force (and its early-stopped rounds) -- a refit
  EVERY session -- along each PATH:
      tp1, tp5, tp25, tp250    best-of-50 (validation MSE) of the latest tuning point at a
                               multiple of TUNE_PER (counted from the whole series' first
                               forecast row) <= t
      tp1_k10, tp1_k25,        best-of-10 / best-of-25 at TUNE_PER 1 and 25 (the trial-
      tp25_k10, tp25_k25       budget question)
      tp25_q                   best-of-50 by validation QLIKE at TUNE_PER 25 (second rule)
  Paths that agree on (configuration, rounds) at a row share one fit (the fit is a pure
  function of them).  Native importance at every IMPORTANCE_EVERY-th row (whole-series row
  index) on every path, TreeSHAP at the same rows on the SHAP_PATH (tp1) only.

ROUNDS / SEARCH SPACES: see specs/causal_tune_trees_optuna_jobs.py (patience ceil(1/lr);
cap = the tuned spec's 4 x shipped rounds held as a total-shrinkage budget; continuous
spaces covering the random-search grid).  The random forest keeps n_estimators = 100.

IDENTITY (env IDENTITY=1): N_TRIALS = 1 (the shipped configuration) and early stopping off
(rounds = the shipped n_estimators): stage 2's every path must then reproduce
specs/causal_tune_trees.py with REFIT_EVERY = 1 bit for bit (experiments/gate_trees_optuna.py).

WINDOW MASK (env WINDOW_MASK=1; user decision 2026-09-29, writeup/CAMPAIGN_16H_2026-09-29.md
"Restart with a per-window mask"): every fit uses only src.models.window_mask.window_keep of
its window -- the columns not constant on the window and not exact copies of an earlier kept
column, the linear arms' identifiability rule.  A tuning point computes the kept set once from
its full window [t - W, t) and applies it to the fit block and the validation tail of every
trial; a refit computes it from its own window and applies it to the forecast row; native
importance and TreeSHAP are mapped back to all p columns (0 for a dropped column).  The kept
count is recorded per tuning point and per refit.  WINDOW_MASK=0 (default) is the run before
that decision, unchanged.

PARALLELISM: a pool of N_WORKERS = SLURM_CPUS_PER_TASK spawned processes (stage 1: one
tuning point per job; stage 2: one refit per job); every fit is single-threaded
(causal_tune_trees_tuned_jobs.MODEL_THREADS = 1), so every number is independent of the
pool size (gate: one process == pooled).  Chunks: any START >= W with HALO = W (the
tuning points are independent and stage 2 reads the merged stage-1 records), gate: two
chunks == the unchunked run.  Stage 1 saves each finished point (points/pt_<row>.npz) as
it completes; with RESUME=1 a re-run chunk skips the saved points.

PERSISTED under $HPC_RESULT_DIR:
  stage 1  causal_tune_trees/<model>/<bucket>/
             trials_<seg>.npz   per point: row, seed, split, every trial's params / val_mse /
                                val_qlike / rounds / rounds_max / patience / seconds, the
                                best-of-k picks; meta (space, sampler, versions)
             trials_<seg>.csv   the same, one row per (point, trial)
             stage1_<seg>.csv   the executor's table of the tuning rows (NaN forecasts: stage
                                1 forecasts nothing; it carries the stamps and targets)
  stage 2  <path>/causal_tune_trees/<model>/<bucket>/   (the untuned layout, per path)
             results_<seg>.csv  the executor's table format (the SHAP path's is the
                                executor's own; the others carry the same columns, pred_raw
                                = the run's Duan back-transform as the tuned spec's qsel CSV)
             trees_<seg>.npz    the untuned npz keys (importance / shap NaN rows between the
                                recorded rows) + the configuration in force per row

ENV AXES: those of specs/causal_tune_trees.py (MODEL, EXOG_BUCKET, SEGMENT, TRAIN_WIN,
LAG_SCOPE, HAR_LAGS / HAR_BASE, START / END / HALO, SHAP_SEGMENTS) plus STAGE (tune |
refit), N_TRIALS (default 50), IDENTITY (0 | 1), WINDOW_MASK (0 | 1, default 0),
IMPORTANCE_EVERY (default 10), STAGE1_TRIALS (stage 2: the arm's merged trials_<seg>.npz),
RESUME (0 | 1, default 1).
"""

# %%
import sys
from pathlib import Path

import os

_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _START
while not (_ROOT / "src").is_dir() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if (_ROOT / "src").is_dir():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)

# %%
import ast  # noqa: E402 -- after the path bootstrap above
import json  # noqa: E402
import multiprocessing  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.reclasso_har import forward_window_split  # noqa: E402

if str(_ROOT / "specs") not in sys.path:
    sys.path.insert(0, str(_ROOT / "specs"))
import causal_tune_trees_optuna_jobs as OJ  # noqa: E402 -- the worker-side jobs
import causal_tune_trees_tuned_jobs as J  # noqa: E402 -- tree_setup / refit_block

LINEAR_SPEC = _ROOT / "specs" / "causal_tune_linear.py"


def load_linear_constants(names: tuple[str, ...]) -> dict:
    """The first top-level literal assignment of each name in specs/causal_tune_linear.py."""
    out: dict = {}
    for node in ast.parse(LINEAR_SPEC.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names and tgt.id not in out:
                    out[tgt.id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    assert not missing, f"causal_tune_linear.py no longer assigns {sorted(missing)}"
    return out


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


T = J.tree_setup(refresh=True)  # specs/causal_tune_trees.py above its run cell
_LIN = load_linear_constants(("VAL_TAIL", "EMBARGO"))
VAL_TAIL: int = _LIN["VAL_TAIL"]  # 125-row validation tail inside the window
EMBARGO: int = _LIN["EMBARGO"]  # 25-row gap between fit block and validation tail

MODEL: str = T["MODEL"]
EXOG_BUCKET: str = T["EXOG_BUCKET"]
SEGMENT: str = T["SEGMENT"]
TRAIN_WIN: int = T["TRAIN_WIN"]
SEED: int = T["SEED"]
START, END, HALO = int(T["START"]), int(T["END"]), int(T["HALO"])
WANT_SHAP: bool = T["WANT_SHAP"]
N_WORKERS: int = int(
    os.environ.get("SLURM_CPUS_PER_TASK", "1")
)  # pool size; fits are 1-thread

STAGE = _env("STAGE", "tune")
if STAGE not in ("tune", "refit"):
    raise SystemExit(f"STAGE must be tune or refit, got {STAGE!r}")
IDENTITY = _env("IDENTITY", "0") == "1"
WINDOW_MASK = (
    _env("WINDOW_MASK", "0") == "1"
)  # the per-window column mask (see the header)
N_TRIALS_DEFAULT = 50  # the user's first budget ("let's see if 50 is enough")
N_TRIALS = 1 if IDENTITY else int(_env("N_TRIALS", str(N_TRIALS_DEFAULT)))
EARLY_STOP = MODEL in ("lgbm", "xgb") and not IDENTITY
BUDGETS = (
    10,
    25,
    50,
)  # best-of-first-k trial budgets recorded (k <= N_TRIALS used as is)
# native importance / TreeSHAP at every IMPORTANCE_EVERY-th row (user decision: every 10th refit)
IMPORTANCE_EVERY = int(_env("IMPORTANCE_EVERY", "10"))
RESUME = _env("RESUME", "1") == "1"
STAGE1_TRIALS = _env("STAGE1_TRIALS", "")
# path -> (TUNE_PER, selection rule, trial budget k)
PATHS: dict[str, tuple[int, str, int]] = {
    "tp1": (1, "mse", 50),
    "tp5": (5, "mse", 50),
    "tp25": (25, "mse", 50),
    "tp250": (250, "mse", 50),
    "tp1_k10": (1, "mse", 10),
    "tp1_k25": (1, "mse", 25),
    "tp25_k10": (25, "mse", 10),
    "tp25_k25": (25, "mse", 25),
    "tp25_q": (25, "qlike", 50),
}
SHAP_PATH = "tp1"  # TreeSHAP on the TUNE_PER = 1 best-of-50 path only
AXES = list(OJ.SPACES[MODEL])
RESULT_DIR = os.environ.get("HPC_RESULT_DIR", "results")
STAGE1_DIR = T["OUT_DIR"]  # $HPC_RESULT_DIR/causal_tune_trees/<model>/<bucket>
if IMPORTANCE_EVERY < 1 or N_TRIALS < 1:
    raise SystemExit(
        f"IMPORTANCE_EVERY and N_TRIALS must be >= 1, got {IMPORTANCE_EVERY}, {N_TRIALS}"
    )


def path_dir(path: str) -> str:
    return os.path.join(RESULT_DIR, path, "causal_tune_trees", MODEL, EXOG_BUCKET)


print(
    f"optuna trees: stage={STAGE} model={MODEL} bucket={EXOG_BUCKET} segment={SEGMENT} "
    f"tw={TRAIN_WIN} trials={N_TRIALS} identity={IDENTITY} early_stop={EARLY_STOP} "
    f"val_tail={VAL_TAIL} embargo={EMBARGO} importance_every={IMPORTANCE_EVERY} "
    f"workers={N_WORKERS} slice=({START},{END},{HALO}) resume={RESUME}"
)


# %% ---------------------------------------------------------------- shared helpers
def best_of(losses: np.ndarray, k: int) -> np.ndarray:
    """Per point (rows of losses), the index of the smallest loss among the first k trials;
    NaN counts as +inf; ties -> the lower index (trial 0 = the shipped configuration)."""
    v = np.asarray(losses, dtype=np.float64)[:, : min(k, np.shape(losses)[1])]
    return np.argmin(np.where(np.isnan(v), np.inf, v), axis=1).astype(np.int64)


def chunk_offset(W: int) -> int:
    """Whole-series OOS index of this run's first forecast row (START = 0: the whole series;
    a chunk replays exactly the training window as halo)."""
    if START == 0:
        assert HALO == 0, f"START 0 with HALO {HALO}"
        return 0
    assert HALO == W, f"chunk halo must be the training window: HALO {HALO} != W {W}"
    off = START - W
    assert off >= 0, f"chunk START {START} < W {W}"
    return off


def pool_for(n: int):
    if n > 1:  # spawn: never fork a process that already holds OpenMP / BLAS threads
        return ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"))
    return None


def point_meta() -> dict:
    return {
        "model": MODEL,
        "bucket": EXOG_BUCKET,
        "segment": SEGMENT,
        "train_win": TRAIN_WIN,
        "n_trials": N_TRIALS,
        "early_stop": EARLY_STOP,
        "axes": AXES,
        "window_mask": WINDOW_MASK,
    }


# %% ---------------------------------------------------------------- stage 1
SIDE: dict = {}


def _points_dir() -> str:
    return os.path.join(STAGE1_DIR, "points")


def _load_partial(row: int) -> dict | None:
    p = os.path.join(_points_dir(), f"pt_{row}.npz")
    if not (RESUME and os.path.isfile(p)):
        return None
    with np.load(p, allow_pickle=False) as z:
        if json.loads(str(z["meta"])) != point_meta():
            return None
        return {k: z[k] for k in z.files if k != "meta"}


def _save_partial(rec: dict) -> None:
    if not RESUME:
        return
    os.makedirs(_points_dir(), exist_ok=True)
    tmp = os.path.join(_points_dir(), f"pt_{rec['row']}.tmp.npz")
    np.savez(
        tmp,
        meta=json.dumps(point_meta()),
        **{k: np.asarray(v) for k, v in rec.items()},  # type: ignore[arg-type]
    )
    os.replace(tmp, os.path.join(_points_dir(), f"pt_{rec['row']}.npz"))


def fit_predict_tune(X_chunk, y_chunk, train_win_periods, hyperparams):
    """A tuning point at every forecast row of the chunk; returns NaN forecasts (stage 1
    forecasts nothing -- the executor's table then records the rows' stamps and targets)."""
    X = np.ascontiguousarray(X_chunk, dtype=np.float64)
    y = np.ascontiguousarray(y_chunk, dtype=np.float64)
    W = int(train_win_periods)
    n_test = len(X) - W
    off = chunk_offset(W)
    recs: dict[int, dict] = {}
    jobs = []
    for i in range(n_test):
        t = W + i
        row = off + i
        got = _load_partial(row)
        if got is not None:
            recs[row] = {
                k: (v.item() if np.ndim(v) == 0 else v) for k, v in got.items()
            }
            continue
        fit_lo, fit_hi, val_lo, val_hi = forward_window_split(t, W, VAL_TAIL, EMBARGO)
        assert fit_lo == t - W and val_hi == t, (fit_lo, val_hi, t, W)
        assert fit_hi + EMBARGO == val_lo and val_hi - val_lo == VAL_TAIL, (
            fit_hi,
            val_lo,
        )
        assert fit_lo < fit_hi < val_lo < val_hi, (fit_lo, fit_hi, val_lo, val_hi)
        lo = t - W
        jobs.append(
            dict(
                X=X[lo:t],  # the window, and nothing at or after the forecast row
                y=y[lo:t],
                fit=(fit_lo - lo, fit_hi - lo),
                val=(val_lo - lo, val_hi - lo),
                row=row,
                seed=OJ.TPE_SEED_BASE + row,
                n_trials=N_TRIALS,
                early_stop=EARLY_STOP,
                mask=WINDOW_MASK,
            )
        )
    print(
        f"  stage 1: {n_test} tuning points (rows {off} .. {off + n_test - 1}), "
        f"{len(recs)} resumed from points/, {len(jobs)} to run",
        flush=True,
    )
    t0 = time.time()
    pool = pool_for(N_WORKERS)
    try:
        if pool is None:
            done_iter = (OJ.tune_point(j) for j in jobs)
        else:
            futs = [pool.submit(OJ.tune_point, j) for j in jobs]
            done_iter = (f.result() for f in as_completed(futs))
        for n_done, rec in enumerate(done_iter, 1):
            rec["split"] = (
                np.array(  # whole-series design rows (OOS row r <-> design row r + W)
                    [
                        rec["row"],
                        rec["row"] + W - VAL_TAIL - EMBARGO,
                        rec["row"] + W - VAL_TAIL,
                        rec["row"] + W,
                    ],
                    dtype=np.int64,
                )
            )
            _save_partial(rec)
            recs[rec["row"]] = rec
            b = int(best_of(rec["val_mse"][None, :], N_TRIALS)[0])
            print(
                f"  point {n_done}/{len(jobs)} row {rec['row']}: best trial {b} val_mse "
                f"{rec['val_mse'][b]:.5f} (shipped {rec['val_mse'][0]:.5f}), "
                f"{rec['study_sec']:.0f}s ({rec['sec'].mean():.2f}s/trial), elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()
    rows = off + np.arange(n_test)
    assert sorted(recs) == list(rows), "tuning records do not tile the chunk's rows"
    SIDE.update(
        records=[recs[int(r)] for r in rows],
        oos_offset=off,
        n_features=X.shape[1],
        wall_sec=time.time() - t0,
    )
    return np.full(n_test, np.nan)


def stack_records(records: list[dict]) -> dict:
    """The per-point records -> arrays (points x trials), plus the best-of-k picks."""
    out = {
        "row": np.array([r["row"] for r in records], dtype=np.int64),
        "seed": np.array([r["seed"] for r in records], dtype=np.int64),
        "split": np.array([r["split"] for r in records], dtype=np.int64).reshape(
            len(records), 4
        ),
        "params": np.array([r["params"] for r in records], dtype=np.float64),
        "study_sec": np.array([r["study_sec"] for r in records], dtype=np.float64),
    }
    for k in ("val_mse", "val_qlike", "sec"):
        out[k] = np.array([r[k] for r in records], dtype=np.float64)
    for k in ("rounds", "rounds_max", "patience"):
        out[k] = np.array([r[k] for r in records], dtype=np.int64)
    out["complete"] = np.array([r["complete"] for r in records], dtype=bool)
    out["n_kept"] = np.array([r["n_kept"] for r in records], dtype=np.int64)
    for budget in BUDGETS:
        out[f"best_mse_k{budget}"] = best_of(out["val_mse"], budget)
    out["best_qlike_k50"] = best_of(out["val_qlike"], 50)
    return out


def trials_long(st: dict, dates: list[str]) -> pd.DataFrame:
    P, N = st["val_mse"].shape
    d = {
        "model": MODEL,
        "bucket": EXOG_BUCKET,
        "segment": SEGMENT,
        "row": np.repeat(st["row"], N),
        "forecast_date": np.repeat(np.asarray(dates), N),
        "trial": np.tile(np.arange(N), P),
    }
    for j, a in enumerate(AXES):
        d[a] = st["params"][:, :, j].ravel()
    d["n_kept"] = np.repeat(st["n_kept"], N)
    for k in (
        "val_mse",
        "val_qlike",
        "rounds",
        "rounds_max",
        "patience",
        "sec",
        "complete",
    ):
        d[k] = st[k].ravel()
    tab = pd.DataFrame(d)
    tab["cap_hit"] = (tab["rounds_max"] > 0) & (tab["rounds"] >= tab["rounds_max"])
    return tab


# %% ---------------------------------------------------------------- stage 2
def load_stage1(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        st = {k: z[k] for k in z.files}
    meta = json.loads(str(st.pop("meta")))
    for k in ("model", "bucket", "segment", "train_win"):
        assert meta[k] == point_meta()[k], (
            f"stage-1 records are for {k}={meta[k]!r}, not {point_meta()[k]!r}"
        )
    assert meta["axes"] == AXES, (meta["axes"], AXES)
    assert meta["early_stop"] == EARLY_STOP, (
        f"stage 1 early_stop {meta['early_stop']} != {EARLY_STOP}"
    )
    assert meta.get("window_mask", False) == WINDOW_MASK, (
        f"stage 1 window_mask {meta.get('window_mask', False)} != {WINDOW_MASK}"
    )
    st["meta"] = meta
    return st


def path_choice(st: dict, rows: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray]:
    """(tuning-point row in force, trial index) at each whole-series OOS row of `rows`."""
    tp, rule, k = PATHS[path]
    idx = {int(r): j for j, r in enumerate(st["row"])}
    points = (rows // tp) * tp
    missing = sorted({int(p) for p in points if int(p) not in idx})
    assert not missing, (
        f"path {path}: stage-1 records lack tuning points {missing[:5]}..."
    )
    losses = st["val_mse"] if rule == "mse" else st["val_qlike"]
    picks = best_of(losses, k)
    return points, np.array([picks[idx[int(p)]] for p in points], dtype=np.int64)


def fit_predict_refit(X_chunk, y_chunk, train_win_periods, hyperparams):
    """A refit every row on [t - W, t) with each path's configuration in force; returns
    the SHAP_PATH's forecasts (the executor's own table), the others go to SIDE."""
    assert STAGE1_TRIALS, (
        "STAGE=refit needs STAGE1_TRIALS (the arm's merged trials npz)"
    )
    st = load_stage1(STAGE1_TRIALS)
    X = np.ascontiguousarray(X_chunk, dtype=np.float64)
    y = np.ascontiguousarray(y_chunk, dtype=np.float64)
    W = int(train_win_periods)
    n_test = len(X) - W
    p = X.shape[1]
    off = chunk_offset(W)
    rows = off + np.arange(n_test)
    idx = {int(r): j for j, r in enumerate(st["row"])}
    choice = {path: path_choice(st, rows, path) for path in PATHS}
    # unique (row, configuration, rounds) fits
    key_of: dict[tuple, int] = {}
    jobs, job_key = [], []
    use: dict[str, list[int]] = {path: [] for path in PATHS}
    tmp = tempfile.mkdtemp(prefix="trees_optuna_", dir=os.environ.get("TMPDIR") or None)
    xp, yp = os.path.join(tmp, "chunk_X.npy"), os.path.join(tmp, "chunk_y.npy")
    np.save(xp, X)
    np.save(yp, y)
    for i in range(n_test):
        record = int(rows[i]) % IMPORTANCE_EVERY == 0
        for path in PATHS:
            pt, g = choice[path][0][i], choice[path][1][i]
            j = idx[int(pt)]
            vec = st["params"][j, g]
            rounds = int(st["rounds"][j, g]) if EARLY_STOP else None
            key = (i, vec.tobytes(), rounds)
            if key not in key_of:
                key_of[key] = len(jobs)
                jobs.append(
                    dict(
                        X=xp,
                        y=yp,
                        t=W + i,
                        W=W,
                        k=1,
                        cfg=OJ.decode(MODEL, vec),
                        rounds=rounds,
                        qcfg=None,
                        qrounds=None,
                        shap=False,
                    )
                )
                job_key.append(key)
            q = key_of[key]
            if WANT_SHAP and record and path == SHAP_PATH:
                jobs[q]["shap"] = True
            use[path].append(q)
    print(
        f"  stage 2: {n_test} rows x {len(PATHS)} paths = {n_test * len(PATHS)} forecasts from "
        f"{len(jobs)} distinct fits (rows {off} .. {off + n_test - 1})",
        flush=True,
    )
    t0 = time.time()
    pool = pool_for(N_WORKERS)
    outs: list[dict] = []
    try:
        run = pool.map if pool is not None else map
        refit = OJ.refit_masked if WINDOW_MASK else J.refit_block
        for n_done, out in enumerate(run(refit, jobs), 1):
            outs.append(out)
            if n_done % 200 == 0:
                print(
                    f"  fits {n_done}/{len(jobs)}  elapsed {time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    paths: dict[str, dict] = {}
    add_gap = 0.0
    for path in PATHS:
        qs = use[path]
        pts, gs = choice[path]
        preds = np.array([outs[q]["preds"][0] for q in qs], dtype=np.float64)
        rec_rows = (rows % IMPORTANCE_EVERY) == 0
        imp = np.full((n_test, p), np.nan, dtype=np.float32)
        for i in np.flatnonzero(rec_rows):
            imp[i] = outs[qs[i]]["importance"]
        shap_mat = None
        if WANT_SHAP and path == SHAP_PATH:
            shap_mat = np.full((n_test, p + 1), np.nan, dtype=np.float32)
            for i in np.flatnonzero(rec_rows):
                o = outs[qs[i]]
                shap_mat[i] = o["shap"][0]
                add_gap = max(add_gap, float(o["gap"]))
        jrow = np.array([idx[int(pt)] for pt in pts], dtype=np.int64)
        paths[path] = dict(
            preds=preds,
            importance=imp,
            shap=shap_mat,
            fit_sec=np.array([outs[q]["fit_sec"] for q in qs]),
            n_kept=np.array([outs[q].get("n_kept", p) for q in qs], dtype=np.int64),
            shap_sec=np.array(
                [outs[q]["shap_sec"] for q in qs if outs[q]["shap_sec"] is not None]
            ),
            cfg_point=pts.astype(np.int64),
            cfg_trial=gs,
            cfg_rounds=np.array(
                [
                    int(st["rounds"][jj, g]) if EARLY_STOP else -1
                    for jj, g in zip(jrow, gs)
                ],
                dtype=np.int64,
            ),
            cfg_params=st["params"][jrow, gs],
            fit_key=np.array(qs, dtype=np.int64),
        )
    SIDE.update(
        paths=paths,
        oos_offset=off,
        n_features=p,
        n_fits=len(jobs),
        shap_additivity_gap=add_gap,
        wall_sec=time.time() - t0,
        feature_names=np.array(
            [
                str(c)
                for c in hyperparams.get("_feature_names", [f"f{j}" for j in range(p)])
            ]
        ),
    )
    return paths[SHAP_PATH]["preds"]


def _versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "shap", "optuna"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- a library the model does not use may be absent
            out[mod] = "absent"
    return out


def space_doc() -> dict:
    return {
        a: (
            list(s[1])
            if s[0] == "cat"
            else {"kind": s[0], "low": s[1], "high": s[2], "log": s[3]}
        )
        for a, s in OJ.SPACES[MODEL].items()
    }


def run_meta(extra: dict) -> dict:
    return {
        "model": MODEL,
        "bucket": EXOG_BUCKET,
        "segment": SEGMENT,
        "lag_scope": T["LAG_SCOPE"],
        "train_win": TRAIN_WIN,
        "stage": STAGE,
        "n_trials": N_TRIALS,
        "identity": IDENTITY,
        "window_mask": WINDOW_MASK,
        "early_stop": EARLY_STOP,
        "axes": AXES,
        "space": space_doc(),
        "shipped_config": OJ.shipped_config(MODEL),
        "shipped_params": T["PARAMS"][MODEL],
        "provenance": T["PROVENANCE"][MODEL]
        + "; Optuna TPE: specs/causal_tune_trees_optuna.py",
        "sampler": {
            "name": "TPESampler",
            "seed": f"{OJ.TPE_SEED_BASE} + whole-series OOS row",
            "defaults": True,
        },
        "rounds_cap_mult": OJ.ROUNDS_CAP_MULT,
        "val_tail": VAL_TAIL,
        "embargo": EMBARGO,
        "budgets": list(BUDGETS),
        "paths": {k: list(v) for k, v in PATHS.items()},
        "shap_path": SHAP_PATH,
        "importance_every": IMPORTANCE_EVERY,
        "threads": J.MODEL_THREADS,
        "workers": N_WORKERS,
        "seed": SEED,
        "har_lags": T["HAR_LAGS"],
        "slice": [START, END, HALO],
        "versions": _versions(),
    } | extra


# %% ---- RUN ----
if __name__ == "__main__":
    from src.backtest.executor import run_executor
    from src.data.loading import get_bucket

    if STAGE == "tune":
        out_dir, stem, fp = STAGE1_DIR, "stage1", fit_predict_tune
    else:
        out_dir, stem, fp = path_dir(SHAP_PATH), "results", fit_predict_refit
    out_csv = os.path.join(out_dir, f"{stem}.csv")
    out_read = os.path.join(out_dir, f"{stem}_{SEGMENT}.csv")
    t_run = time.time()
    run_executor(
        method_name=f"trees_optuna_{STAGE}_{MODEL}",
        fit_predict=fp,
        hyperparams={},
        data_path=T["DATA_PATH"],
        output_file=out_csv,
        horizon=T["HORIZON"],
        train_window=TRAIN_WIN,
        start=T["START"],
        end=T["END"],
        halo=T["HALO"],
        exog_cols=get_bucket(EXOG_BUCKET),
        segment=SEGMENT,
        lag_scope=T["LAG_SCOPE"],
        har_lags=T["HAR_LAGS"],
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=SEED,
    )
    # round_trip: the executor writes repr floats; the default parser is off by an ULP at times
    res = pd.read_csv(out_read, parse_dates=["date"], float_precision="round_trip")
    n_oos = len(res)
    dates = res["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ok = (res["true_adj"] > 0) & (res["true_raw"] > 0)
    baseline = np.where(ok, res["true_raw"] / res["true_adj"].where(ok) ** 2, np.nan)

    if STAGE == "tune":
        st = stack_records(SIDE["records"])
        assert len(st["row"]) == n_oos, (len(st["row"]), n_oos)
        meta = run_meta(
            {
                "oos_offset": SIDE["oos_offset"],
                "n_features": SIDE["n_features"],
                "walk_sec": SIDE["wall_sec"],
                "run_sec": time.time() - t_run,
            }
        )
        np.savez_compressed(
            os.path.join(out_dir, f"trials_{SEGMENT}.npz"),
            date=dates.to_numpy().astype("U19"),
            oos_offset=SIDE["oos_offset"],
            meta=json.dumps(meta),
            **st,
        )
        trials_long(st, list(dates)).to_csv(
            os.path.join(out_dir, f"trials_{SEGMENT}.csv"), index=False
        )
        sec = st["sec"]
        print(
            f"wrote {out_dir}/trials_{SEGMENT}.npz: {len(st['row'])} tuning points x {N_TRIALS} trials "
            f"({dates.iloc[0]} .. {dates.iloc[-1]}), {sec.mean():.2f}s/trial mean ({sec.sum() / 3600:.2f} "
            f"core-h of fits), best at trial 0 in {np.mean(st['best_mse_k50'] == 0):.0%} of points, "
            f"walk {SIDE['wall_sec']:.0f}s, run {time.time() - t_run:.0f}s"
        )
    else:
        meta = run_meta(
            {
                "oos_offset": SIDE["oos_offset"],
                "n_features": SIDE["n_features"],
                "n_fits": SIDE["n_fits"],
                "stage1_trials": STAGE1_TRIALS,
                "walk_sec": SIDE["wall_sec"],
                "run_sec": time.time() - t_run,
            }
        )
        for path, d in SIDE["paths"].items():
            pdir = path_dir(path)
            os.makedirs(pdir, exist_ok=True)
            pr = d["preds"]
            if path == SHAP_PATH:  # the executor's own table
                assert np.array_equal(res["pred_adj"].to_numpy(float), pr), (
                    "executor table != SHAP path"
                )
            else:  # the executor's columns, pred_raw = this path's Duan back-transform over the run
                smear = float(np.mean((res["true_adj"].to_numpy(float) - pr) ** 2))
                tab = res[["date", "horizon", "true_adj"]].copy()
                tab["pred_adj"] = pr
                tab["true_raw"] = res["true_raw"]
                tab["pred_raw"] = (pr**2 + smear) * baseline
                tab.to_csv(os.path.join(pdir, f"results_{SEGMENT}.csv"), index=False)
            shap = d["shap"]
            np.savez_compressed(
                os.path.join(pdir, f"trees_{SEGMENT}.npz"),
                date=dates.to_numpy().astype("U19"),
                pred_adj=pr,
                true_adj=res["true_adj"].to_numpy(float),
                true_raw=res["true_raw"].to_numpy(float),
                baseline=baseline,
                e2_adj=(res["true_adj"].to_numpy(float) - pr) ** 2,
                refit_row=np.arange(
                    n_oos, dtype=np.int64
                ),  # + oos_offset = whole series
                oos_offset=SIDE["oos_offset"],
                importance=d["importance"],
                fit_sec=d["fit_sec"],
                shap_sec=d["shap_sec"],
                feature_names=SIDE["feature_names"],
                shap=shap if shap is not None else np.zeros((0, 0), np.float32),
                shap_additivity_gap=SIDE["shap_additivity_gap"]
                if shap is not None
                else 0.0,
                cfg_point=d["cfg_point"],
                cfg_trial=d["cfg_trial"],
                cfg_rounds=d["cfg_rounds"],
                cfg_params=d["cfg_params"],
                fit_key=d["fit_key"],
                n_kept=d["n_kept"],
                meta=json.dumps(meta | {"path": path, "path_spec": list(PATHS[path])}),
            )
        e2 = (res["true_adj"] - res["pred_adj"]) ** 2
        print(
            f"wrote {len(PATHS)} paths under {RESULT_DIR} ({n_oos} OOS rows, {dates.iloc[0]} .. "
            f"{dates.iloc[-1]}); {SIDE['n_fits']} distinct fits; walk {SIDE['wall_sec']:.0f}s, "
            f"run {time.time() - t_run:.0f}s; {SHAP_PATH} fit-space MSE {e2.mean():.5f}, p={SIDE['n_features']}"
        )
