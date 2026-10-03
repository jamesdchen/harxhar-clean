"""Do the 16:00 trees lose to the linear models for lack of training rows?  (close study, 2026-10-03)

The user's note: "not enough training data for trees? -> last hour (2 x 30-min bars)?".  The
16:00-bar trees (specs/causal_tune_trees.py) fit 2000 rows, one a session; the pooled 48-bar
LightGBM of the paper fitted a 24,000-bar window.  Two tests on the captured design
(experiments/capture_design_close.py; results/close_design/_work/design_<segment>_all_features.npz):

1. POOL THE LAST HOUR ("pool"): train on both last-hour bars (segment last30: the bars ending
   15:30 and 16:00, told apart by the calendar column ``hour``), 2000 sessions = 4000 rows, and
   forecast and score the 16:00 rows only.  The model in force for the 16:00 row of day d was fitted
   on the rows strictly before that row, so the 15:30 row of day d (its target realized at 15:30,
   when the 16:00 forecast is issued) may enter it; nothing later does.
2. LEARNING CURVE ON THE 16:00 ROWS ("h16"): the 16:00 rows of the last30 design (they reach back
   to 2004-04-13; the one-bar design starts in 2010-07 because the executor drops its HAR warm-up
   from the segment's own rows), windows of 500, 1000, 2000 and 3000 sessions and an expanding
   window (every 16:00 row before the refit anchor), on the same forecast rows.

Every arm forecasts the same 16:00 rows (the one-bar design's forecast rows, 2018-06-25 ..
2024-04-30) and is scored on the 866 trade days by the master table's research convention
(experiments/dense_vs_sparse_1530.py research_frame / deck_frame / deck_panel / point).

TREES: the tree spec's shipped configs and construction (make_model of specs/causal_tune_trees.py,
read through dense_vs_sparse_1530.spec_ns), refit every REFIT_EVERY = 10 sessions, single-threaded,
with the window mask (src.models.window_mask.window_keep on each fit's own training window: the
spec's WINDOW_MASK = 1).  The leaf minimum follows the spec's scaling rule for the rows the fit
sees: LightGBM min_child_samples = max(1, round(98 x n / 24000)) (8 at 2000 rows, 16 at 4000),
XGBoost min_child_weight = 24.86 x (n / 24000); for the expanding window n is the row count at
each refit.  Tree forecasts start at forecast row TREE_START (see the constant).

LINEAR CONTROL: the linear spec's ridge and lasso (specs/causal_tune_linear.py RollingTunedLinear,
grids ESTIMATOR_GRIDS["ridge"] and ["reclasso"]) on the same rows and windows, exactly as
dense_vs_sparse_1530.run_linear_blocks drives it: a solve and a forecast at every row of the
series from the first forecast row on (for "pool" this includes the 15:30 rows, whose forecasts
are not scored), warm rank-1 updates between tunings, the penalty re-chosen and the state
cold-reseeded every TUNE_PER = 250 solves on the fit / 25 embargo / 125 tail split of the window
(the spec's constants, in rows).  Expanding window: at each 250-solve block start the window is
every row before it; inside the block it slides at that length (the rank-1 machinery has a fixed
window), so it trails the trees' expanding window by at most 249 rows.

Stages:
  python experiments/close_trees_datasize.py run [arm ...]   fit arms (default: ORDER), one at a
                                                             time; an arm already in _work/ is skipped
  python experiments/close_trees_datasize.py analyze         score, CSVs, figure, SUMMARY.md

Outputs: results/close_studies_2026-10-03/trees_datasize/ (CSVs, figure, SUMMARY.md written from
the CSVs); _work/<arm>.npz holds each arm's forecasts (never committed).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["SLURM_CPUS_PER_TASK"] = "1"  # the tree spec's N_THREADS

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(REPO)

import dense_vs_sparse_1530 as dvs  # noqa: E402
from src.models.window_mask import window_keep  # noqa: E402

DESIGN = Path(
    os.environ.get("CLOSE_DESIGN_DIR", str(REPO / "results" / "close_design" / "_work"))
)
OUT = REPO / "results" / "close_studies_2026-10-03" / "trees_datasize"
BUCKET = "all_features"
SPEC_REFIT_EVERY = 10  # the tree spec's default REFIT_EVERY (asserted against the spec)
# Env axis TDS_REFIT_EVERY (default the spec's 10): 1 = a refit every session (the cadence of the
# linear arms and of the stored subtree_daily_* run); far beyond this container's CPU, it is what the
# Hoffman2 task (cluster/close_trees_datasize_h2_task.sh) is for.  Another cadence writes its own
# report folder, so the default run's files are never overwritten.
REFIT_EVERY = int(os.environ.get("TDS_REFIT_EVERY", str(SPEC_REFIT_EVERY)))
REPORT = OUT if REFIT_EVERY == SPEC_REFIT_EVERY else OUT / f"refit{REFIT_EVERY}"
WORK = Path(os.environ.get("TDS_WORK", str(REPORT / "_work")))
TUNE_PER = 250  # the linear spec's TUNE_PER (asserted against the spec)
# First tree forecast, as an index into the 1469 one-bar forecast rows (0 = 2018-06-25): a multiple
# of REFIT_EVERY, so every refit anchor is an anchor of the stored run, and early enough that the
# scorer's recalibration window (the previous SMEAR_W = 250 sessions' squared errors, lagged one
# session) is full of forecasts on the first trade day (asserted in analyze).  Saves 13 of the 147
# refits; the linear arms run every forecast row (they are cheap) so their tuning points stay the
# stored arms'.
TREE_START = 130
# Hand-check of the scaled leaf minimum: the spec's rule at 2000 rows
LGBM_LEAF_AT_2000 = 8

# arm name -> (model, source, window): window = training rows (int) or "exp" (expanding)
ARMS: dict[str, tuple[str, str, int | str]] = {}
for _m in ("lgbm", "xgb", "rf", "ridge", "lasso"):
    ARMS[f"{_m}_bar1600_w2000"] = (_m, "bar1600", 2000)
    ARMS[f"{_m}_pool_w4000"] = (_m, "pool", 4000)
    for _w in (500, 1000, 2000, 3000, "exp"):
        ARMS[f"{_m}_h16_w{_w}"] = (_m, "h16", _w)
LINEAR = ("ridge", "lasso")
TREES = ("lgbm", "xgb", "rf")
EST_OF = {"ridge": "ridge", "lasso": "reclasso"}  # the linear spec's estimator names
ORDER = [
    *(f"{m}_bar1600_w2000" for m in LINEAR),
    *(f"{m}_pool_w4000" for m in LINEAR),
    *(f"{m}_h16_w{w}" for w in (2000, "exp", 500, 1000, 3000) for m in LINEAR),
    "lgbm_bar1600_w2000",
    "lgbm_pool_w4000",
    "lgbm_h16_w2000",
    "lgbm_h16_wexp",
    "lgbm_h16_w500",
    "lgbm_h16_w1000",
    "lgbm_h16_w3000",
    "xgb_bar1600_w2000",
    "xgb_pool_w4000",
]


# ============================================================================ data
def _load(segment: str) -> dict:
    z = np.load(DESIGN / f"design_{segment}_{BUCKET}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


_SRC: dict = {}


def sources() -> dict:
    """The three row sets.  fc = series index of each of the 1469 one-bar forecast rows."""
    if _SRC:
        return _SRC
    b, ll = _load("bar1600"), _load("last30")
    w1 = int(b["W"])
    n_fc = len(b["X"]) - w1
    db = pd.DatetimeIndex(pd.to_datetime(b["date"]))
    dl = pd.DatetimeIndex(pd.to_datetime(ll["date"]))
    assert list(b["names"]) == list(ll["names"])
    fc_dates = db[w1:]
    pos30 = dl.get_indexer(fc_dates)
    assert (pos30 >= 0).all()
    # the same target on the 16:00 rows of both designs (the scaling of X differs)
    assert np.array_equal(ll["y"][pos30], b["y"][w1:])
    i16 = np.flatnonzero(np.asarray((dl.hour == 16) & (dl.minute == 0)))
    pos16 = pd.Index(dl[i16]).get_indexer(fc_dates)
    assert (pos16 >= 0).all()
    assert np.array_equal(ll["y"][i16][pos16], b["y"][w1:])
    _SRC.update(
        bar1600=dict(X=b["X"], y=b["y"], fc=w1 + np.arange(n_fc), date=db),
        pool=dict(X=ll["X"], y=ll["y"], fc=pos30, date=dl),
        h16=dict(
            X=np.ascontiguousarray(ll["X"][i16]), y=ll["y"][i16], fc=pos16, date=dl[i16]
        ),
        dz=dict(date=b["date"][w1:], true_adj=b["y"][w1:], true_raw=b["true_raw"]),
        names=[str(c) for c in b["names"]],
        n_fc=n_fc,
    )
    return _SRC


# ============================================================================ runners
def tree_params(model: str, n_rows: int) -> dict:
    """The spec's shipped config with the leaf minimum scaled to the fit's row count."""
    ns = dvs.spec_ns("tree")
    if model == "lgbm":
        p = dict(ns["LGBM_PARAMS"])
        p["min_child_samples"] = max(
            1, round(ns["POOLED_MIN_CHILD_SAMPLES"] * (n_rows / ns["POOLED_WINDOW_ROWS"]))
        )
    elif model == "rf":  # the spec's sklearn defaults: no leaf minimum to scale
        p = dict(ns["RF_PARAMS"])
    else:
        p = dict(ns["XGB_PARAMS"])
        p["min_child_weight"] = ns["POOLED_MIN_CHILD_WEIGHT"] * (
            n_rows / ns["POOLED_WINDOW_ROWS"]
        )
    return p


def make_tree(model: str, n_rows: int):
    """Built as the spec's make_model (src/unification.py::_walk_tree), single-threaded."""
    ns = dvs.spec_ns("tree")
    p = tree_params(model, n_rows)
    if model == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(**p, num_threads=1, random_state=ns["SEED"], verbosity=-1)
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**p, random_state=ns["SEED"], n_jobs=1)
    import xgboost as xgb

    return xgb.XGBRegressor(**p, n_jobs=1, random_state=ns["SEED"], verbosity=0)


def check_spec() -> None:
    ns = dvs.spec_ns("tree")
    assert ns["REFIT_EVERY"] == SPEC_REFIT_EVERY and ns["N_THREADS"] == 1
    assert REFIT_EVERY >= 1 and TREE_START % REFIT_EVERY == 0
    assert tree_params("lgbm", 2000) == ns["LGBM_PARAMS"]  # the spec's own 2000-row config
    assert ns["LGBM_PARAMS"]["min_child_samples"] == LGBM_LEAF_AT_2000
    assert tree_params("xgb", 2000) == ns["XGB_PARAMS"]
    lin = dvs.spec_ns("linear")
    assert lin["TUNE_PER"] == TUNE_PER


def run_tree(model: str, src: dict, win: int | str, n_fc: int) -> dict:
    X, y, fc = src["X"], src["y"], src["fc"]
    pred = np.full(n_fc, np.nan)
    fit_sec, n_train, kept_n, leaf = [], [], [], []
    anchors = list(range(TREE_START, n_fc, REFIT_EVERY))
    t0 = time.time()
    for a, j in enumerate(anchors):
        r = int(fc[j])  # series row of the anchor's 16:00 forecast
        lo = 0 if win == "exp" else r - int(win)
        assert lo >= 0, (j, r, win)
        Xw, yw = X[lo:r], y[lo:r]
        keep = window_keep(Xw)  # the spec's WINDOW_MASK = 1, on this fit's own window
        k = min(REFIT_EVERY, n_fc - j)
        rows = fc[j : j + k]
        assert rows.min() >= r  # every forecast row is at or after the anchor
        m = make_tree(model, r - lo)
        s = time.time()
        m.fit(Xw[:, keep], yw)
        fit_sec.append(time.time() - s)
        pred[j : j + k] = m.predict(X[rows][:, keep])
        n_train.append(r - lo)
        kept_n.append(len(keep))
        p = m.get_params()
        leaf.append(
            {"lgbm": "min_child_samples", "xgb": "min_child_weight", "rf": "min_samples_leaf"}
            .get(model) and p[{"lgbm": "min_child_samples", "xgb": "min_child_weight", "rf": "min_samples_leaf"}[model]]
        )
        if (a + 1) % 25 == 0:
            print(
                f"    refit {a + 1}/{len(anchors)}  fit {np.mean(fit_sec):.1f}s  "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
    return dict(
        pred=pred,
        fit_sec=np.array(fit_sec),
        n_train=np.array(n_train),
        kept_n=np.array(kept_n),
        leaf_min=np.array(leaf, dtype=float),
        anchors=np.array(anchors),
    )


def run_linear(model: str, src: dict, win: int | str, n_fc: int) -> dict:
    """dense_vs_sparse_1530.run_linear_blocks' loop on a series with a solve at every row from
    the first forecast row: a fresh RollingTunedLinear (tune + cold reseed) every TUNE_PER solves."""
    ns = dvs.spec_ns("linear")
    Base = ns["RollingTunedLinear"]
    Base.grid = ns["ESTIMATOR_GRIDS"][EST_OF[model]]
    Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
    X, y, fc = src["X"], src["y"], src["fc"]
    first, last = int(fc[0]), int(fc[-1])
    pred_s = np.full(len(X), np.nan)
    alpha_s = np.full(len(X), np.nan)
    n_train = []
    t_start = time.time()
    for t0 in range(first, last + 1, TUNE_PER):
        t1 = min(t0 + TUNE_PER, last + 1)
        wb = t0 if win == "exp" else int(win)
        assert t0 - wb >= 0
        n_train.append(wb)
        reg = Base()
        for t in range(t0, t1):
            if t == t0:
                reg.init_window(X[t0 - wb : t0], y[t0 - wb : t0])
            else:
                reg.roll(X[t - 1], y[t - 1], X[t - 1 - wb], y[t - 1 - wb])
            reg.solve()
            pred_s[t] = reg.predict_one(X[t])
            alpha_s[t] = float(reg.alpha_)
        print(
            f"    block {t0 - first}..{t1 - first} of {last + 1 - first} solves, "
            f"window {wb}, alpha {reg.alpha_:.2e}, elapsed {time.time() - t_start:.0f}s",
            flush=True,
        )
    return dict(
        pred=pred_s[fc],
        alpha=alpha_s[fc],
        n_train=np.array(n_train),
        n_reseed=len(Base.reseed_trace),
        n_solves=last + 1 - first,
        mask_trace=np.array(Base.mask_trace),
    )


def versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- report what is installed
            out[mod] = "absent"
    return out


def run(arms: list[str]) -> None:
    check_spec()
    S = sources()
    WORK.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        f = WORK / f"{arm}.npz"
        if f.is_file():
            print(f"{arm}: done already ({f})", flush=True)
            continue
        model, srcname, win = ARMS[arm]
        print(f"{arm}: model={model} rows={srcname} window={win}", flush=True)
        t0, c0 = time.time(), time.process_time()
        if model in TREES:
            out = run_tree(model, S[srcname], win, S["n_fc"])
        else:
            out = run_linear(model, S[srcname], win, S["n_fc"])
        wall, cpu = time.time() - t0, time.process_time() - c0
        meta = dict(
            arm=arm,
            model=model,
            source=srcname,
            window=win,
            bucket=BUCKET,
            refit_every=REFIT_EVERY if model in TREES else 1,
            tree_start=TREE_START if model in TREES else 0,
            params=tree_params(model, 2000 if win == "exp" else int(win))
            if model in TREES
            else EST_OF[model],
            wall_sec=wall,
            cpu_sec=cpu,
            versions=versions(),
        )
        np.savez_compressed(
            f,
            **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
            **{k: np.array(v) for k, v in out.items() if not isinstance(v, np.ndarray)},
            meta=json.dumps(meta),
        )
        print(f"{arm}: wrote {f}  wall {wall:.0f}s  cpu {cpu:.0f}s", flush=True)


# ============================================================================ main
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "analyze"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "run":
        run(sys.argv[2:] or ORDER)
    else:
        from close_trees_datasize_analyze import analyze  # noqa: E402

        analyze()
