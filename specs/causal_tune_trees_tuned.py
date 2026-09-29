"""causal_tune_trees_tuned -- per-bar trees with CAUSAL, PERIODIC hyperparameter tuning.

PURPOSE: the per-bar tree campaign (specs/causal_tune_trees.py) ran LightGBM /
XGBoost / random forest with hyperparameters copied from the POOLED tree bank
(tuned on a 24,000-row pooled window, leaf minimums scaled to 2000 rows; RF =
sklearn defaults) and lost to the per-bar linear arms.  This spec re-selects the
tree hyperparameters CAUSALLY and PERIODICALLY on the per-bar design, with the
scheme of the per-bar linear arms (specs/causal_tune_linear.py) copied exactly:

  * every TUNE_PER = 250 rows (sessions) of the one-bar series -- the linear
    spec's TUNE_PER solves at REFIT_FREQUENCY = 1, i.e. every 250 rows there too
    -- the CURRENT training window [t - W, t) (W = TRAIN_WIN = 2000 rows, t = the
    first row forecast under the new choice) is split by
    src.models.reclasso_har.forward_window_split(t, W, VAL_TAIL = 125, EMBARGO =
    25): fit block [t - W, t - 150), 25-row embargo, validation tail
    [t - 125, t);
  * every candidate of the model's grid is fitted on the fit block and scored on
    the validation tail; the argmin of validation MSE on the transformed target
    (the linear spec's criterion) is held until the next tuning point;
  * between tuning points the model is refitted every REFIT_EVERY = 10 rows on
    the full window [t - W, t) with the chosen configuration (the untuned
    campaign's cadence), predicting each block of rows with the model in force.
TUNE_PER, VAL_TAIL and EMBARGO are READ from the source of
specs/causal_tune_linear.py; REFIT_EVERY, SEED, the shipped parameter dicts
(after the leaf-minimum scaling), make_model, native_importance, contributions
and fit_predict_tree are EXECUTED from the source of specs/causal_tune_trees.py
(everything above its run cell), so the untuned arm's code is the only copy.

BOOSTING ROUNDS: learning_rate stays at the shipped value; the number of rounds
is chosen per candidate by early stopping on the validation tail (validation MSE,
patience PATIENCE rounds, cap ROUNDS_MAX); the refits then use that round count
on the full window.  The random forest keeps n_estimators = 100 (more trees only
lower variance; not a tuning axis).

SECOND RULE, RECORDED (not the arm of record): every candidate's validation
QLIKE is recorded next to its MSE -- QLIKE of the executor's own Duan back-
transform on the validation tail (src.evaluation.metrics.apply_duan_smearing:
variance = (f^2 + mean val squared error) x B; B cancels in the QLIKE ratio, so
it is computed on the transformed scale exactly; rows with y <= 0 dropped, as
the scorer drops them).  The QLIKE-argmin configuration is walked forward too
(QSEL=1, default): when it equals the MSE choice its forecasts are the MSE
path's; otherwise it gets its own refits (no TreeSHAP).  Written as
results_qsel_<seg>.csv, so a QLIKE-selected variant is scored post hoc without
re-running.

GRIDS (named choices below; each is centred on the shipped configuration on a
log scale where the parameter is a scale, and stops at a parameter bound where
the shipped value sits on one).  The full products are 420 (LightGBM), 1260
(XGBoost) and 84 (RF) configurations; a FIXED subset of N_CANDIDATES = 32 --
the shipped configuration plus 31 drawn once with GRID_SEED -- is used at every
tuning point of every arm.  TUNE_IDENTITY=1 reduces the grid to the shipped
configuration with early stopping off (rounds = the shipped n_estimators): the
identity gate, which must reproduce the untuned arm bit for bit.

CAUSALITY: the tuning step writes only the window X[t - W : t], y[t - W : t]
(rows strictly before the forecast row t) for the workers that score the
candidates, and asserts that the fit block, embargo and validation tail tile
exactly that window.  experiments/gate_trees_tuned.py adds the perturbation test.

PARALLELISM (specs/causal_tune_trees_tuned_jobs.py): the candidates of a tuning
point, and then the refits of its tuning period, run in a pool of N_WORKERS =
SLURM_CPUS_PER_TASK spawned processes; every fit is single-threaded, so every
number is independent of the pool size (gate: one process == pooled, bit for
bit).  TIME CHUNKS: run_executor's START / END / HALO seam with HALO = W and
START - W a multiple of TUNE_PER (asserted), so a chunk's tuning and refit rows
are the whole series' and its first tuning window is exactly its halo (gate: two
chunks == the unchunked run, bit for bit); experiments/reduce_trees_tuned_chunks.py
merges the chunks of an arm into the unchunked layout.

PERSISTED (under $HPC_RESULT_DIR/causal_tune_trees/<model>/<bucket>/, the
untuned layout, so the untuned scorer and join gates apply unchanged):
  results_<seg>.csv          the executor's results table (MSE rule = the arm of record)
  trees_<seg>.npz            the untuned npz keys + pred_adj_qsel, the tuning
                             trace and every candidate's validation MSE / QLIKE /
                             rounds / fit seconds at every tuning point
  results_qsel_<seg>.csv     the QLIKE-rule path, same columns
  tune_trace_<seg>.csv       one row per (tuning point, rule): chosen config,
                             rounds, losses, edge flags per axis
  tune_candidates_<seg>.csv  one row per (tuning point, candidate)
  grid_<seg>.json            the grid, its natural bounds, the candidate subset

ENV AXES: those of specs/causal_tune_trees.py (MODEL, EXOG_BUCKET, SEGMENT,
TRAIN_WIN, LAG_SCOPE, HAR_LAGS / HAR_BASE, START / END / HALO, SHAP_SEGMENTS)
plus TUNE_IDENTITY (0 | 1, default 0) and QSEL (0 | 1, default 1).
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
import ast
import itertools
import json
import math
import multiprocessing
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.models.reclasso_har import forward_window_split

if str(_ROOT / "specs") not in sys.path:
    sys.path.insert(0, str(_ROOT / "specs"))
import causal_tune_trees_tuned_jobs as J  # noqa: E402 -- the worker-side jobs

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


T = J.tree_setup(refresh=True)  # specs/causal_tune_trees.py above its run cell
_LIN = load_linear_constants(("TUNE_PER", "VAL_TAIL", "EMBARGO"))
TUNE_PER: int = _LIN["TUNE_PER"]  # 250 rows between tunings (the linear arms' cadence)
VAL_TAIL: int = _LIN["VAL_TAIL"]  # 125-row validation tail inside the window
EMBARGO: int = _LIN["EMBARGO"]  # 25-row gap between fit block and validation tail

MODEL: str = T["MODEL"]
EXOG_BUCKET: str = T["EXOG_BUCKET"]
SEGMENT: str = T["SEGMENT"]
TRAIN_WIN: int = T["TRAIN_WIN"]
REFIT_EVERY: int = T["REFIT_EVERY"]
SEED: int = T["SEED"]
# pool size = the task's cores; every fit inside a worker is single-threaded
# (J.MODEL_THREADS), so results do not depend on it
N_WORKERS: int = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
START, END, HALO = int(T["START"]), int(T["END"]), int(T["HALO"])
WANT_SHAP: bool = T["WANT_SHAP"]
SHIPPED: dict = dict(T["PARAMS"][MODEL])  # after the tree spec's leaf-minimum scaling
assert TUNE_PER % REFIT_EVERY == 0, (TUNE_PER, REFIT_EVERY)  # tunings fall on refit rows


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


TUNE_IDENTITY = _env("TUNE_IDENTITY", "0") == "1"
QSEL = _env("QSEL", "1") == "1"

# ---------------------------------------------------------------- grids (named choices)
# Leaf-size ladders are powers of two around the shipped value: a factor-2 step
# halves / doubles the rows per leaf (or the leaf count), the coarsest step that
# still separates neighbouring trees; the span is x1/8 .. x8 for the per-leaf row
# minimums (the pooled bank's value was scaled 12x down to this window, so the
# per-bar optimum may sit far from it) and x1/4 .. x4 for leaf counts / depth.
LADDER_ROWS = tuple(2.0**k for k in range(-3, 4))  # x1/8 .. x8 around the shipped rows-per-leaf
LADDER_LEAVES = tuple(2.0**k for k in range(-2, 3))  # x1/4 .. x4 around the shipped leaf count
# Sampling fractions: half the shipped fraction, the shipped fraction, and the
# bound 1.0 (no subsampling) -- a fraction cannot exceed 1.
FRACTION_STEPS = (0.5, 1.0)
FRACTION_BOUND = 1.0
# L2 on leaf values is in hessian units = rows under squared error (leaf value =
# G / (H + lambda)); the shipped values (0.016 LightGBM, 0.0004 XGBoost) are ~0
# rows, so the axis runs from the shipped value up by decades: lambda = 1, 10,
# 100 rows shrinks a 10-100-row leaf by roughly 1-90 %.  Its low end (the shipped
# ~0) is flagged as the no-penalty bound in the edge table, not a grid choice.
L2_ROWS = (1.0, 10.0, 100.0)
# RF: max_features steps x1/3 down from the shipped 1.0 (bagged trees): 1/3 is
# Breiman's regression default (p/3), 1/9 one step further; max_depth halves from
# unbounded (the shipped) to 16 / 8 / 4 levels; min_samples_leaf climbs the row
# ladder from the shipped bound 1.
# (every axis list is ascending; None = unbounded depth sorts last)
RF_MAX_FEATURES = (1.0 / 9.0, 1.0 / 3.0, 1.0)
RF_MAX_DEPTH = (4, 8, 16, None)
RF_MIN_LEAF = (1, 2, 4, 8, 16, 32, 64)
# Candidate subset: the argmin is taken over 125 validation rows, so more
# candidates increasingly select validation noise (the linear arms choose among
# 5-6 alphas); 32 = 4 waves of an 8-process pool, and in the local smoke
# (live_feasible bar1600, LightGBM) one tuning point of 32 early-stopped
# candidates cost ~400 process-seconds, a third of the 25 refits of its tuning
# period under the chosen configuration -- the grid is not the cost driver.
N_CANDIDATES = 32
GRID_SEED = 20260924  # the day the subset was drawn; fixed for every arm
# Boosting rounds: early stopping on the validation tail with patience = the
# number of rounds whose shrunken steps add up to one full step (ceil(1 / lr));
# cap = ROUNDS_CAP_MULT x the shipped rounds (the shipped count was tuned on a
# 12x larger pooled window; fewer rows should stop earlier, so x4 is headroom and
# cap hits are counted as edge picks).
ROUNDS_CAP_MULT = 4


def _uniq(vals):
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def build_grid() -> tuple[dict, dict]:
    """{axis: values in grid order} and {axis: {"lo": natural?, "hi": natural?}}."""
    s = SHIPPED
    if MODEL == "lgbm":
        axes = {
            "num_leaves": _uniq([max(2, round(s["num_leaves"] * f)) for f in LADDER_LEAVES]),
            "min_child_samples": _uniq(
                [max(1, round(s["min_child_samples"] * f)) for f in LADDER_ROWS]
            ),
            "feature_fraction": _uniq(
                [s["feature_fraction"] * f for f in FRACTION_STEPS] + [FRACTION_BOUND]
            ),
            "lambda_l2": _uniq([s["lambda_l2"], *L2_ROWS]),
        }
        natural = {
            "num_leaves": {"lo": axes["num_leaves"][0] == 2, "hi": False},
            "min_child_samples": {"lo": axes["min_child_samples"][0] == 1, "hi": False},
            "feature_fraction": {"lo": False, "hi": True},
            "lambda_l2": {"lo": True, "hi": False},  # the shipped ~0 = no penalty
        }
    elif MODEL == "xgb":
        d = s["max_depth"]
        axes = {
            "max_depth": _uniq([max(1, d + k) for k in range(-2, 3)]),  # leaves x1/4 .. x4
            "min_child_weight": _uniq([s["min_child_weight"] * f for f in LADDER_ROWS]),
            "subsample": _uniq([s["subsample"] * f for f in FRACTION_STEPS] + [FRACTION_BOUND]),
            "colsample_bytree": _uniq(
                [s["colsample_bytree"] * f for f in FRACTION_STEPS] + [FRACTION_BOUND]
            ),
            "reg_lambda": _uniq([s["reg_lambda"], *L2_ROWS]),
        }
        natural = {
            "max_depth": {"lo": axes["max_depth"][0] == 1, "hi": False},
            "min_child_weight": {"lo": False, "hi": False},
            "subsample": {"lo": False, "hi": True},
            "colsample_bytree": {"lo": False, "hi": True},
            "reg_lambda": {"lo": True, "hi": False},  # the shipped ~0 = no penalty
        }
    else:
        axes = {
            "max_features": list(RF_MAX_FEATURES),
            "min_samples_leaf": list(RF_MIN_LEAF),
            "max_depth": list(RF_MAX_DEPTH),
        }
        natural = {
            "max_features": {"lo": False, "hi": True},  # 1.0 = every feature
            "min_samples_leaf": {"lo": True, "hi": False},  # 1 row
            "max_depth": {"lo": False, "hi": True},  # None = unbounded
        }
    for a, vals in axes.items():  # "lo" / "hi" below mean the list's first / last value
        key = [math.inf if v is None else v for v in vals]
        assert key == sorted(key), f"grid axis {a} is not ascending: {vals}"
    return axes, natural


def shipped_config(axes: dict) -> dict:
    if MODEL == "rf":
        return {"max_features": 1.0, "min_samples_leaf": 1, "max_depth": None}  # sklearn defaults
    return {a: SHIPPED[a] for a in axes}


def edge_of(axes: dict, natural: dict, axis: str, val) -> tuple[str, bool]:
    """('lo' | 'hi' | '', is that end a parameter bound?) of val on the ascending axis."""
    vals = axes[axis]
    if len(vals) < 2:
        return "", False
    end = "lo" if val == vals[0] else ("hi" if val == vals[-1] else "")
    return end, bool(end and natural[axis][end])


def draw_candidates(axes: dict) -> list[dict]:
    """The shipped configuration + a fixed random subset of the full product."""
    ship = shipped_config(axes)
    if TUNE_IDENTITY:
        return [ship]
    names = list(axes)
    full = [dict(zip(names, combo)) for combo in itertools.product(*(axes[a] for a in names))]
    assert ship in full, f"shipped config {ship} not on the grid"
    rest = [c for c in full if c != ship]
    k = min(N_CANDIDATES - 1, len(rest))
    pick = np.random.default_rng(GRID_SEED).choice(len(rest), size=k, replace=False)
    return [ship] + [rest[j] for j in sorted(pick)]


AXES, NATURAL = build_grid()
CANDIDATES = draw_candidates(AXES)
GRID_SIZE = math.prod(len(v) for v in AXES.values())
GB = MODEL in ("lgbm", "xgb")
EARLY_STOP = GB and not TUNE_IDENTITY
if GB:
    PATIENCE = math.ceil(1.0 / SHIPPED["learning_rate"])
    ROUNDS_MAX = ROUNDS_CAP_MULT * SHIPPED["n_estimators"]
else:
    PATIENCE = ROUNDS_MAX = None
print(
    f"tuned trees: {MODEL} grid {GRID_SIZE} configs, {len(CANDIDATES)} candidates "
    f"(seed {GRID_SEED}, identity={TUNE_IDENTITY}), tune every {TUNE_PER} rows "
    f"(val tail {VAL_TAIL}, embargo {EMBARGO}), refit every {REFIT_EVERY}, "
    f"early stopping={EARLY_STOP} patience={PATIENCE} cap={ROUNDS_MAX}, qsel={QSEL}"
)


# %%
TRACE: list[dict] = []  # one entry per tuning point (evidence; reset per walk)


def chunk_offset(W: int) -> int:
    """Whole-series OOS index of this run's first forecast row.

    START = 0 (HALO = 0) is the whole series.  A chunk (run_executor's
    START / END / HALO seam) must replay exactly the training window as halo and
    start on a tuning row of the whole series, so its tuning and refit rows are
    the unchunked run's and it needs nothing before its halo: the window of the
    chunk's first tuning point IS the halo."""
    if START == 0:
        assert HALO == 0, f"START 0 with HALO {HALO}"
        return 0
    off = START - W
    assert HALO == W, f"chunk halo must be the training window: HALO {HALO} != W {W}"
    assert off >= 0 and off % TUNE_PER == 0, (
        f"chunk start {START}: OOS offset {off} is not a tuning row"
    )
    return off


def refit_rounds(rec: dict, g: int) -> int | None:
    """Rounds the full-window refits use: the candidate's early-stopped count
    (boosting), None = the constructor's own n_estimators otherwise."""
    return int(rec["rounds"][g]) if EARLY_STOP else None


def tune(X, y, t: int, W: int, i: int, off: int, tmp: str, run) -> dict:
    """Score every candidate on the forward split of the window before row t.

    Only the window X[t - W : t], y[t - W : t] -- rows strictly before the
    forecast row t -- is written for the workers; the split is asserted to tile
    it exactly (fit | embargo | validation tail)."""
    assert t - W >= 0 and t <= len(X), (t, W, len(X))
    fit_lo, fit_hi, val_lo, val_hi = forward_window_split(t, W, VAL_TAIL, EMBARGO)
    assert fit_lo == t - W and val_hi == t, (fit_lo, val_hi, t, W)
    assert fit_hi + EMBARGO == val_lo and val_hi - val_lo == VAL_TAIL, (fit_hi, val_lo)
    assert fit_lo < fit_hi < val_lo < val_hi, (fit_lo, fit_hi, val_lo, val_hi)
    xp, yp = os.path.join(tmp, f"win{i}_X.npy"), os.path.join(tmp, f"win{i}_y.npy")
    np.save(xp, X[t - W : t])  # the workers see the window and nothing else
    np.save(yp, y[t - W : t])
    lo = t - W  # window-relative rows below
    jobs = [
        dict(
            X=xp,
            y=yp,
            fit=(fit_lo - lo, fit_hi - lo),
            val=(val_lo - lo, val_hi - lo),
            cfg=cfg,
            early_stop=EARLY_STOP,
            patience=PATIENCE,
            rounds_max=ROUNDS_MAX,
        )
        for cfg in CANDIDATES
    ]
    a0 = time.time()
    got = list(run(J.fit_candidate, jobs))
    mse = np.array([g["val_mse"] for g in got])
    qlk = np.array([g["val_qlike"] for g in got])
    rec = dict(
        i=i,
        t=t,
        split=tuple(int(v + off) for v in (fit_lo, fit_hi, val_lo, val_hi)),  # whole-series rows
        val_mse=mse,
        val_qlike=qlk,
        rounds=np.array([g["rounds"] for g in got], dtype=np.int64),
        fit_sec=np.array([g["sec"] for g in got]),
        pick_mse=int(np.argmin(mse)),  # ties -> the lower index; the shipped config is index 0
        pick_qlike=int(np.argmin(qlk)),
        tune_sec=time.time() - a0,
    )
    TRACE.append(rec)
    return rec


SIDE: dict = {}


def fit_predict_tuned(X_chunk, y_chunk, train_win_periods, hyperparams):
    """Walk-forward on the one-bar series: tune every TUNE_PER rows (counted
    from the whole series' first forecast) on the current window, refit every
    REFIT_EVERY rows on the trailing W rows (strictly before the anchor) with
    the configuration in force.  The candidates of a tuning point, and the
    refits of a tuning period, run in a pool of N_WORKERS processes."""
    X = np.ascontiguousarray(X_chunk, dtype=np.float64)
    y = np.ascontiguousarray(y_chunk, dtype=np.float64)
    W = int(train_win_periods)
    n_test = len(X) - W
    p = X.shape[1]
    off = chunk_offset(W)
    preds = np.empty(n_test)
    preds_q = np.empty(n_test)
    shap_mat = np.full((n_test, p + 1), np.nan, dtype=np.float32) if WANT_SHAP else None
    refit_row, imp, fit_sec, shap_sec, qsel_sec, add_gap = [], [], [], [], [], 0.0
    TRACE.clear()
    tmp = tempfile.mkdtemp(prefix="trees_tuned_", dir=os.environ.get("TMPDIR") or None)
    xp, yp = os.path.join(tmp, "chunk_X.npy"), os.path.join(tmp, "chunk_y.npy")
    np.save(xp, X)
    np.save(yp, y)
    pool = None
    if N_WORKERS > 1:  # spawn: never fork a process that already holds OpenMP / BLAS threads
        pool = ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    run = pool.map if pool is not None else map
    t0 = time.time()
    try:
        # off % TUNE_PER == 0, so these are the whole series' tuning rows
        for i0 in range(0, n_test, TUNE_PER):
            rec = tune(X, y, W + i0, W, i0, off, tmp, run)
            gm, gq = rec["pick_mse"], rec["pick_qlike"]
            rm, rq = refit_rounds(rec, gm), refit_rounds(rec, gq)
            same = CANDIDATES[gq] == CANDIDATES[gm] and rq == rm
            print(
                f"  tune @row {off + i0}: {len(CANDIDATES)} candidates in "
                f"{rec['tune_sec']:.1f}s; mse pick {gm} {CANDIDATES[gm]} rounds {rm} "
                f"val_mse {rec['val_mse'][gm]:.5f} (shipped {rec['val_mse'][0]:.5f}); "
                f"qlike pick {gq}",
                flush=True,
            )
            blocks = list(range(i0, min(i0 + TUNE_PER, n_test), REFIT_EVERY))
            jobs = [
                dict(
                    X=xp,
                    y=yp,
                    t=W + i,
                    W=W,
                    k=min(REFIT_EVERY, n_test - i),
                    cfg=CANDIDATES[gm],
                    rounds=rm,
                    qcfg=None if (same or not QSEL) else CANDIDATES[gq],
                    qrounds=rq,
                    shap=WANT_SHAP,
                )
                for i in blocks
            ]
            for i, out in zip(blocks, run(J.refit_block, jobs)):
                k = len(out["preds"])
                preds[i : i + k] = out["preds"]
                preds_q[i : i + k] = out["preds_q"]
                refit_row.append(i)
                imp.append(out["importance"])
                fit_sec.append(out["fit_sec"])
                if out["shap"] is not None:
                    shap_mat[i : i + k] = out["shap"]
                    shap_sec.append(out["shap_sec"])
                    add_gap = max(add_gap, out["gap"])
                if out["qsel_sec"] is not None:
                    qsel_sec.append(out["qsel_sec"])
            print(
                f"  refits to row {off + blocks[-1]}: elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    SIDE.update(
        feature_names=np.array(
            [str(c) for c in hyperparams.get("_feature_names", [f"f{j}" for j in range(p)])]
        ),
        refit_row=np.array(refit_row, dtype=np.int64),
        oos_offset=off,
        importance=np.array(imp, dtype=np.float32),
        fit_sec=np.array(fit_sec),
        shap_sec=np.array(shap_sec),
        qsel_sec=np.array(qsel_sec),
        shap=shap_mat,
        shap_additivity_gap=add_gap,
        pred_q=preds_q,
        n_features=p,
        train_rows=W,
        wall_sec=time.time() - t0,
        trace=list(TRACE),
    )
    if WANT_SHAP:
        print(f"  TreeSHAP additivity gap (max |sum contrib - pred|) = {add_gap:.2e}")
    return preds


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def trace_tables(dates: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tune_trace, tune_candidates) from SIDE['trace']; dates = the OOS stamps."""
    keys = {"model": MODEL, "bucket": EXOG_BUCKET, "segment": SEGMENT, "train_win": TRAIN_WIN}
    trace_rows, cand_rows = [], []
    for j, rec in enumerate(SIDE["trace"]):
        stamp = str(dates.iloc[rec["i"]])
        fit_lo, fit_hi, val_lo, val_hi = rec["split"]
        base = keys | {
            "tune_idx": j,
            "tune_row": rec["i"],  # row of this run's results table (+ chunk_offset = whole series)
            "chunk_offset": SIDE["oos_offset"],
            "forecast_date": stamp,
        }
        for g, cfg in enumerate(CANDIDATES):
            cand_rows.append(
                base
                | {"cand_idx": g, "is_shipped": g == 0}
                | {a: ("None" if v is None else v) for a, v in cfg.items()}
                | {
                    "rounds": int(rec["rounds"][g]),
                    "rounds_cap_hit": bool(EARLY_STOP and rec["rounds"][g] >= ROUNDS_MAX),
                    "val_mse": float(rec["val_mse"][g]),
                    "val_qlike": float(rec["val_qlike"][g]),
                    "fit_sec": float(rec["fit_sec"][g]),
                    "chosen_mse": g == rec["pick_mse"],
                    "chosen_qlike": g == rec["pick_qlike"],
                }
            )
        for rule in ("mse", "qlike"):
            g = rec[f"pick_{rule}"]
            row = base | {
                "rule": rule,
                "fit_lo": fit_lo,
                "fit_hi": fit_hi,
                "val_lo": val_lo,
                "val_hi": val_hi,
                "cand_idx": g,
                "is_shipped": g == 0,
                "rounds": int(rec["rounds"][g]),
                "rounds_cap_hit": bool(EARLY_STOP and rec["rounds"][g] >= ROUNDS_MAX),
                "val_mse": float(rec["val_mse"][g]),
                "val_qlike": float(rec["val_qlike"][g]),
                "shipped_val_mse": float(rec["val_mse"][0]),
                "shipped_val_qlike": float(rec["val_qlike"][0]),
                "n_candidates": len(CANDIDATES),
                "tune_sec": float(rec["tune_sec"]),
            }
            for a, v in CANDIDATES[g].items():
                end, nat = edge_of(AXES, NATURAL, a, v)
                row |= {a: ("None" if v is None else v), f"{a}_edge": end, f"{a}_edge_natural": nat}
            trace_rows.append(row)
    return pd.DataFrame(trace_rows), pd.DataFrame(cand_rows)


# %% ---- RUN ----
if __name__ == "__main__":
    from src.backtest.executor import run_executor
    from src.data.loading import get_bucket

    OUT_DIR = T["OUT_DIR"]
    out_csv = os.path.join(OUT_DIR, "results.csv")
    out_read = os.path.join(OUT_DIR, f"results_{SEGMENT}.csv")
    t_run = time.time()
    run_executor(
        method_name=f"trees_tuned_{MODEL}",
        fit_predict=fit_predict_tuned,
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
    res = pd.read_csv(out_read, parse_dates=["date"])
    n_oos = len(res)
    assert len(SIDE["refit_row"]) == -(-n_oos // REFIT_EVERY), (len(SIDE["refit_row"]), n_oos)
    assert len(SIDE["trace"]) == -(-n_oos // TUNE_PER), (len(SIDE["trace"]), n_oos)
    assert SIDE["shap"] is None or len(SIDE["shap"]) == n_oos, (len(SIDE["shap"]), n_oos)
    ok = (res["true_adj"] > 0) & (res["true_raw"] > 0)
    baseline = np.where(ok, res["true_raw"] / res["true_adj"].where(ok) ** 2, np.nan)

    # the QLIKE-rule path, in the executor's own table format (its Duan pred_raw)
    if QSEL:
        pq = SIDE["pred_q"]
        smear = float(np.mean((res["true_adj"].to_numpy(float) - pq) ** 2))
        qsel = res[["date", "horizon", "true_adj"]].copy()
        qsel["pred_adj"] = pq
        qsel["true_raw"] = res["true_raw"]
        qsel["pred_raw"] = (pq**2 + smear) * baseline
        qsel.to_csv(os.path.join(OUT_DIR, f"results_qsel_{SEGMENT}.csv"), index=False)

    trace, cands = trace_tables(res["date"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    trace.to_csv(os.path.join(OUT_DIR, f"tune_trace_{SEGMENT}.csv"), index=False)
    cands.to_csv(os.path.join(OUT_DIR, f"tune_candidates_{SEGMENT}.csv"), index=False)
    grid_doc = {
        "model": MODEL,
        "axes": {a: [("None" if v is None else _jsonable(v)) for v in vals] for a, vals in AXES.items()},
        "natural_bounds": NATURAL,
        "grid_size": GRID_SIZE,
        "n_candidates": len(CANDIDATES),
        "grid_seed": GRID_SEED,
        "candidates": [{a: _jsonable(v) for a, v in c.items()} for c in CANDIDATES],
        "shipped_params": {a: _jsonable(v) for a, v in SHIPPED.items()},
        "tune_identity": TUNE_IDENTITY,
        "early_stop": EARLY_STOP,
        "patience": PATIENCE,
        "rounds_max": ROUNDS_MAX,
        "tune_per": TUNE_PER,
        "val_tail": VAL_TAIL,
        "embargo": EMBARGO,
        "refit_every": REFIT_EVERY,
    }
    with open(os.path.join(OUT_DIR, f"grid_{SEGMENT}.json"), "w", encoding="utf-8") as fh:
        json.dump(grid_doc, fh, indent=1)

    versions = {}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "shap"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- a library the model does not use may be absent
            versions[mod] = "absent"

    tr = SIDE["trace"]
    npz_path = os.path.join(OUT_DIR, f"trees_{SEGMENT}.npz")
    np.savez_compressed(
        npz_path,
        date=res["date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype("U19"),
        pred_adj=res["pred_adj"].to_numpy(float),
        true_adj=res["true_adj"].to_numpy(float),
        true_raw=res["true_raw"].to_numpy(float),
        baseline=baseline,
        e2_adj=((res["true_adj"] - res["pred_adj"]) ** 2).to_numpy(float),
        refit_row=SIDE["refit_row"],
        importance=SIDE["importance"],
        fit_sec=SIDE["fit_sec"],
        shap_sec=SIDE["shap_sec"],
        qsel_sec=SIDE["qsel_sec"],
        feature_names=SIDE["feature_names"],
        shap=SIDE["shap"] if SIDE["shap"] is not None else np.zeros((0, 0), np.float32),
        shap_additivity_gap=SIDE["shap_additivity_gap"],
        pred_adj_qsel=SIDE["pred_q"] if QSEL else np.zeros(0),
        tune_row=np.array([r["i"] for r in tr], dtype=np.int64),  # + oos_offset = whole series
        oos_offset=SIDE["oos_offset"],
        tune_split=np.array([r["split"] for r in tr], dtype=np.int64),
        tune_sec=np.array([r["tune_sec"] for r in tr]),
        cand_val_mse=np.array([r["val_mse"] for r in tr]),
        cand_val_qlike=np.array([r["val_qlike"] for r in tr]),
        cand_rounds=np.array([r["rounds"] for r in tr]),
        cand_fit_sec=np.array([r["fit_sec"] for r in tr]),
        chosen_mse=np.array([r["pick_mse"] for r in tr], dtype=np.int64),
        chosen_qlike=np.array([r["pick_qlike"] for r in tr], dtype=np.int64),
        meta=json.dumps(
            {
                "model": MODEL,
                "params": SHIPPED,
                "provenance": T["PROVENANCE"][MODEL]
                + "; tuned causally: specs/causal_tune_trees_tuned.py",
                "bucket": EXOG_BUCKET,
                "segment": SEGMENT,
                "lag_scope": T["LAG_SCOPE"],
                "train_win_days": TRAIN_WIN,
                "train_rows": SIDE["train_rows"],
                "refit_every": REFIT_EVERY,
                "threads": J.MODEL_THREADS,
                "workers": N_WORKERS,
                "oos_offset": SIDE["oos_offset"],
                "seed": SEED,
                "har_lags": T["HAR_LAGS"],
                "slice": [T["START"], T["END"], T["HALO"]],
                "n_features": SIDE["n_features"],
                "shap": WANT_SHAP,
                "versions": versions,
                "walk_sec": SIDE["wall_sec"],
                "run_sec": time.time() - t_run,
                "grid": grid_doc,
                "qsel": QSEL,
            },
            default=_jsonable,
        ),
    )
    e2 = (res["true_adj"] - res["pred_adj"]) ** 2
    ship_share = float(np.mean([r["pick_mse"] == 0 for r in tr]))
    print(
        f"wrote {out_read} ({n_oos} OOS rows, {res['date'].min()} .. {res['date'].max()}) and {npz_path}; "
        f"{len(tr)} tunings ({np.mean([r['tune_sec'] for r in tr]):.1f}s each), "
        f"{len(SIDE['refit_row'])} refits ({SIDE['fit_sec'].mean():.2f}s each), "
        f"{len(SIDE['qsel_sec'])} extra qsel refits, walk {SIDE['wall_sec']:.0f}s, "
        f"run {time.time() - t_run:.0f}s; shipped chosen at {ship_share:.0%} of tunings; "
        f"fit-space MSE {e2.mean():.5f}, p={SIDE['n_features']}"
    )
