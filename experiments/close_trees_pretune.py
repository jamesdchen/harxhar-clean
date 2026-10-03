"""Per-bar trees at the 16:00 bar: one heavy pre-2020 pre-tune, then a light retune every 250 sessions.

The user's note (2026-10-03): trees do worse than the linear arms -> hyperparameter tuning; "go ham
on the pre-2020 pre-tune, then do very light hyperparameter tuning every year (~250 sessions)".
The stored tuned arms (random search every 250; Optuna with 50 trials on a 125-session validation
tail, every session or every 250) did not beat the shipped configuration; the suspected cause is a
noisy selection on a 125-session tail.  This study replaces that tail with a large pre-2020
validation set for one heavy search, and keeps only a small, guarded retune afterwards.

Model and design: specs/causal_tune_trees.py's per-bar LightGBM (XGBoost with CTP_MODEL=xgb) on
the all_features design of the 16:00 bar, 2000-session window, the per-window column mask on
(src/models/window_mask.py), every fit single-threaded.  Search space, round rules (early stopping
with patience ceil(1 / lr), the 4 x shipped shrinkage cap) and model construction are the Optuna
spec's (specs/causal_tune_trees_optuna_jobs.py), imported, not copied.

STAGES (``python experiments/close_trees_pretune.py <stage>``; sizes from CTP_* env, below)

gate      (a) the stored shipped-config forecast table results/spxw_pnl/
          yhat_subtree_lgbm_all_features.parquet through this script's scorer must reproduce
          its master-table QLIKE and sign(s) Sharpe (mid); (b) the fold fitter equals the Optuna
          spec's fit_trial on one fold (same validation MSE, QLIKE, rounds).
pretune   ONE Optuna TPE study (TPESampler(seed = TPE_SEED_BASE + CTP_SEED), trial 0 = the
          shipped configuration, trials sequential; the folds of a trial run in a pool of
          CTP_WORKERS processes and are combined in fold order, so the pool size changes
          nothing).  Objective = the mean over CTP_FOLDS blocked walk-forward folds of the
          validation MSE on the transformed target (the rule of record); each fold = the
          CTP_FOLD_LEN sessions it validates on, a 25-session embargo (the linear spec's
          EMBARGO) before them, and the 2000 sessions before the embargo as the fit block.
          Folds tile the last CTP_FOLDS x CTP_FOLD_LEN sessions before 2020-01-03 (the first
          trade day).  Validation QLIKE, early-stopped rounds and seconds are recorded for every
          fold of every trial.  Data: the 16:00 rows of the last-hour design cache
          (design_last30_all_features.npz: 16:00 sessions back to 2004-04-13), because the
          16:00-bar cache starts 2010-07-12 and leaves only 357 pre-2020 sessions after one fit
          window and the embargo.  Those 16:00 rows have the same target and columns as the
          16:00-bar design; their rolling robust scaling uses the last-hour window (4000 rows =
          2000 sessions x 2 bars), so column values differ (recorded in the gate CSV).  The
          configuration's refit rounds = the median of its folds' early-stopped rounds.
          Storage: an Optuna JournalFileStorage in _work (resumable: a resumed study reseeds its
          sampler with seed + RESUME_SALT x completed trials, recorded).
walk      Forecasts of every 16:00 forecast row (2018-06-25 .. 2024-04-30, 1469 rows) on the
          16:00-bar design, a refit every CTP_REFIT_EVERY sessions on the 2000 sessions before
          the refit (the spec's stage-2 masked refit, reused), along each ARM:
            control       the shipped configuration (rounds 392): the local twin of the stored
                          subtree_lgbm_all_features table (refit every 10, W 2000, mask on)
            frozen_k<k>   the pre-tune's best configuration after its first k trials (TPE's first
                          k trials ARE a k-trial study), frozen; k in CTP_CHECKPOINTS + all trials
            retune_any    pre-tune best, then a light retune every RETUNE_EVERY = 250 sessions
                          from the refit row at or before the first trade day: CTP_RETUNE_TRIALS
                          trials (trial 0 = the incumbent) of a TPE study confined to a box around
                          the incumbent (each axis: NEIGHBOURHOOD_FRACTION of the spec's range, in
                          log units for log axes), validated on the 250 sessions before the retune
                          row as 2 folds of the spec's 125-session tail (each with its own 2000-
                          session fit block and 25-session embargo); the best trial replaces the
                          incumbent if its validation MSE is lower at all
            retune_margin the same, but the best challenger replaces the incumbent only if the
                          HAC t statistic of the paired daily squared-error difference
                          (challenger - incumbent) over the 250 validation sessions is below
                          -z(MARGIN_ALPHA / number of challengers): a one-sided test at 5 %,
                          Bonferroni-adjusted for picking the best of the challengers
          A configuration's refit rounds: the pre-tune's median fold rounds; a retune winner's =
          the median of its two folds' rounds; a kept incumbent keeps its rounds.
report    Scores every arm and the stored comparison tables with the research scorer
          (experiments/dense_vs_sparse_1530.py research_frame / deck_panel / point: the 16:00-bar
          recalibration and the deck's 15:30 sign(s) straddle), point estimates only (the circular
          block bootstrap is off, commit b761b28); differences against the local control by a
          Diebold-Mariano test on daily QLIKE and a HAC t on the paired daily P&L (mid) difference
          (src/evaluation/diebold_mariano.dm_test on the two series).  Writes the CSVs, a figure
          and SUMMARY.md (generated from the CSVs) under results/close_studies_2026-10-03/
          trees_pretune/.

ENV (CTP_*): MODEL (lgbm | xgb), TAG (output sub-directory, default local), SEED (0), TRIALS
(200), FOLDS (4), FOLD_LEN (250), WORKERS (2), TIMEOUT (pre-tune seconds, 0 = none), RETUNE_TRIALS
(15), REFIT_EVERY (10), CHECKPOINTS ("25,50,100"), SEEDS (walk: the pre-tune seeds to merge, default
SEED), WALK_ROWS ("lo,hi": restrict the walk's refit rows, for a smoke test), WORK (default
<out>/_work), DESIGN_DIR (default results/close_design/_work).
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "specs", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _env(name: str, default: str) -> str:
    return os.environ.get(f"CTP_{name}", default)


MODEL = _env("MODEL", "lgbm")
if MODEL not in ("lgbm", "xgb"):
    raise SystemExit(f"CTP_MODEL must be lgbm or xgb, got {MODEL!r}")
BUCKET = "all_features"
TRAIN_WIN = 2000  # sessions: the per-bar arms' window
# the tree spec's env axes, read by specs/causal_tune_trees.py when tree_setup executes its head
# (spawned workers inherit them): the model, the bucket, the window (leaf-minimum scaling), mask on
os.environ["HPC_KW_MODEL"] = MODEL
os.environ["HPC_KW_EXOG_BUCKET"] = BUCKET
os.environ["HPC_KW_TRAIN_WIN"] = str(TRAIN_WIN)
os.environ["HPC_KW_WINDOW_MASK"] = "1"
os.environ["SLURM_CPUS_PER_TASK"] = "1"

TAG = _env("TAG", "local")
SEED = int(_env("SEED", "0"))
TRIALS = int(_env("TRIALS", "200"))
FOLDS = int(_env("FOLDS", "4"))
FOLD_LEN = int(_env("FOLD_LEN", "250"))
WORKERS = int(_env("WORKERS", "2"))
TIMEOUT = float(_env("TIMEOUT", "0"))
RETUNE_TRIALS = int(_env("RETUNE_TRIALS", "15"))
REFIT_EVERY = int(_env("REFIT_EVERY", "10"))
# lists accept "," or ":" (qsub -v splits its value on commas)
CHECKPOINTS = tuple(int(v) for v in re.split("[,:]", _env("CHECKPOINTS", "25,50,100")) if v.strip())
SEEDS = tuple(int(v) for v in re.split("[,:]", _env("SEEDS", str(SEED))) if v.strip())
WALK_ROWS = _env("WALK_ROWS", "")

OUT = REPO / "results" / "close_studies_2026-10-03" / "trees_pretune"
WORK = Path(_env("WORK", str(OUT / "_work")))
DESIGN_DIR = Path(_env("DESIGN_DIR", str(REPO / "results" / "close_design" / "_work")))
RUN = WORK / TAG / MODEL
ARR = WORK / "arrays"

FIRST_TRADE_DAY = "2020-01-03"  # the deck's first day; the pre-tune sees sessions before it only
TPE_SEED_BASE = 20261003  # the day this study was designed; + CTP_SEED (+ the retune row)
RESUME_SALT = 7919  # a resumed pre-tune reseeds TPE with seed + RESUME_SALT x completed trials
RETUNE_EVERY = 250  # sessions between light retunes (the user's "every year (~250 sessions)")
RETUNE_FOLDS = 2  # x the spec's 125-session tail = the 250 sessions before the retune row
# the retune's box: each axis spans this share of the spec's range (log units for log axes),
# centred on the incumbent and clipped to the range -- (1/4)^6 of the six-axis space
NEIGHBOURHOOD_FRACTION = 0.25
MARGIN_ALPHA = 0.05  # one-sided level of the retune_margin switch test (Bonferroni-adjusted)

STORED = {  # the master table's rows this study is compared with (results/spxw_pnl/)
    "stored_T10": "yhat_subtree_lgbm_all_features.parquet",
    "stored_T1": "yhat_subtree_daily_all_features_lgbm.parquet",
    "stored_RS10_tp250": "yhat_subtree_tuned_all_features_lgbm.parquet",
    "stored_OP_tp1": "yhat_subtree_optuna_tp1_all_features_lgbm.parquet",
    "stored_OP_tp250": "yhat_subtree_optuna_tp250_all_features_lgbm.parquet",
    "stored_lasso_all_features": "yhat_sub_lasso_all_features.parquet",
    "stored_lasso_baseline": "yhat_sub_lasso_baseline.parquet",
}
MASTER_KEY = {  # the same rows' keys in results/close_master_table/master_table.csv
    "stored_T10": "subtree_lgbm_all_features",
    "stored_T1": "subtree_daily_all_features_lgbm",
    "stored_RS10_tp250": "subtree_tuned_all_features_lgbm",
    "stored_OP_tp1": "subtree_optuna_tp1_all_features_lgbm",
    "stored_OP_tp250": "subtree_optuna_tp250_all_features_lgbm",
    "stored_lasso_all_features": "sub_lasso_all_features",
    "stored_lasso_baseline": "sub_lasso_baseline",
}


def linear_constants(names: tuple[str, ...]) -> dict:
    """The first top-level literal assignment of each name in specs/causal_tune_linear.py
    (as specs/causal_tune_trees_optuna.py reads VAL_TAIL and EMBARGO)."""
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
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ data
def prepare_arrays() -> dict:
    """The two designs as .npy files the workers memory-map, plus the row bookkeeping."""
    ARR.mkdir(parents=True, exist_ok=True)
    bar = DESIGN_DIR / f"design_bar1600_{BUCKET}.npz"
    l30 = DESIGN_DIR / f"design_last30_{BUCKET}.npz"
    for f in (bar, l30):
        if not f.is_file():
            raise SystemExit(f"design cache {f} missing (experiments/capture_design_close.py)")
    z = np.load(bar, allow_pickle=False)
    W = int(z["W"])
    assert W == TRAIN_WIN, (W, TRAIN_WIN)
    date = pd.to_datetime(z["date"])
    info = dict(
        W=W,
        n=len(z["y"]),
        date=np.asarray(z["date"]),
        y=np.asarray(z["y"], float),
        true_raw=np.asarray(z["true_raw"], float),
        names=np.asarray(z["names"]),
    )
    paths = {"bar_X": ARR / "bar_X.npy", "bar_y": ARR / "bar_y.npy"}
    if not paths["bar_X"].is_file():
        np.save(paths["bar_X"], np.ascontiguousarray(z["X"], dtype=np.float64))
        np.save(paths["bar_y"], np.ascontiguousarray(z["y"], dtype=np.float64))
    q = np.load(l30, allow_pickle=False)
    d30 = pd.to_datetime(q["date"])
    pre = (d30.hour == 16) & (d30 < pd.Timestamp(FIRST_TRADE_DAY))
    assert (np.asarray(q["names"]) == info["names"]).all(), "column names differ between the caches"
    paths |= {"pre_X": ARR / "pre_X.npy", "pre_y": ARR / "pre_y.npy"}
    if not paths["pre_X"].is_file():
        np.save(paths["pre_X"], np.ascontiguousarray(q["X"][pre], dtype=np.float64))
        np.save(paths["pre_y"], np.ascontiguousarray(q["y"][pre], dtype=np.float64))
    info["pre_date"] = np.asarray(q["date"][pre])
    info["pre_n"] = int(pre.sum())
    # how the two caches' 16:00 rows compare on their common sessions (recorded, not used)
    common = date.intersection(d30[d30.hour == 16])
    ia = date.get_indexer(common)
    ib = pd.DatetimeIndex(d30).get_indexer(common)
    dX = np.abs(np.asarray(z["X"])[ia] - np.asarray(q["X"])[ib])
    info["cache_compare"] = dict(
        common_sessions=len(common),
        y_max_abs_diff=float(np.max(np.abs(np.asarray(z["y"])[ia] - np.asarray(q["y"])[ib]))),
        columns_identical=int((dX.max(axis=0) == 0).sum()),
        columns=int(dX.shape[1]),
        x_max_abs_diff=float(dX.max()),
    )
    info["paths"] = {k: str(v) for k, v in paths.items()}
    return info


def pretune_folds(info: dict, n_folds: int = FOLDS, fold_len: int = FOLD_LEN) -> list[dict]:
    """Blocked walk-forward folds tiling the last n_folds x fold_len pre-2020 16:00 sessions."""
    E = info["pre_n"]
    out = []
    for f in range(n_folds):
        v1 = E - (n_folds - 1 - f) * fold_len
        v0 = v1 - fold_len
        a, b = v0 - EMBARGO - TRAIN_WIN, v0 - EMBARGO
        if a < 0:
            raise SystemExit(
                f"{n_folds} folds x {fold_len} sessions need {n_folds * fold_len + EMBARGO + TRAIN_WIN} "
                f"pre-2020 sessions; the cache holds {E}"
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
    """The light retune at OOS row `row` (design row t = W + row): RETUNE_FOLDS folds of VAL_TAIL
    sessions tiling the RETUNE_FOLDS x VAL_TAIL sessions before t, each with its own fit block."""
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

        _POOL = ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn"))
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


def encode(cfg: dict) -> np.ndarray:
    import causal_tune_trees_optuna_jobs as OJ

    return OJ.encode(MODEL, cfg)


def decode(vec) -> dict:
    import causal_tune_trees_optuna_jobs as OJ

    return OJ.decode(MODEL, vec)


def fold_jobs(info: dict, folds: list[dict], cfg: dict, src: str) -> list[dict]:
    P = info["paths"]
    return [
        dict(X=P[f"{src}_X"], y=P[f"{src}_y"], fit=f["fit"], val=f["val"], cfg=cfg) for f in folds
    ]


def median_rounds(r) -> int:
    return int(np.round(np.median(np.asarray(r, dtype=float))))


def journal(path: Path):
    import optuna

    try:  # Optuna >= 4.0
        backend = optuna.storages.journal.JournalFileBackend(str(path))
    except AttributeError:  # older Optuna (e.g. a cluster's conda env)
        backend = optuna.storages.JournalFileStorage(str(path))
    return optuna.storages.JournalStorage(backend)


# ============================================================================ gate
def score_table(info: dict, preds: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    """Research scorer on each forecast series (1469 forecast rows); returns the point table and
    the deck panel (days x runs) for the paired tests."""
    import dense_vs_sparse_1530 as dvs

    W = info["W"]
    dz = dict(date=info["date"][W:], true_adj=info["y"][W:], true_raw=info["true_raw"])
    names = list(preds)
    frames = [dvs.research_frame(dz, np.asarray(preds[k], float)) for k in names]
    dk = dvs.deck_frame()
    dk = dk[(dk.index >= FIRST_TRADE_DAY) & (dk.index <= "2024-04-30")]
    P = dvs.deck_panel(frames, dk)
    pt = dvs.point(P)
    tab = pd.DataFrame(
        {
            "arm": names,
            "n_days": P["pnl"].shape[0],
            "qlike": pt["ql"],
            "sharpe_mid": pt["sh"],
            "sharpe_crossed": pt["shx"],
            "pct_buy": P["buy"] * 100,
        }
    )
    P["names"] = names
    return tab, P


def load_stored(info: dict, key: str) -> np.ndarray:
    """A stored table's 16:00 forecasts (per-bar linear tables stack every bar: the 16:00 rows)."""
    d = pd.read_parquet(REPO / "results" / "spxw_pnl" / STORED[key])
    W = info["W"]
    t = pd.to_datetime(d["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    s = pd.Series(d["yhat"].to_numpy(float), index=t)
    s = s[s.index.strftime("%H:%M") == "16:00"]
    want = pd.to_datetime(info["date"][W:])
    assert len(s) == len(want) and (s.index == want).all(), (key, len(s))
    return s.to_numpy(float)


def stage_gate() -> None:
    import causal_tune_trees_optuna_jobs as OJ
    import close_trees_pretune_jobs as CJ

    OUT.mkdir(parents=True, exist_ok=True)
    info = prepare_arrays()
    rows = []
    mt = pd.read_csv(REPO / "results" / "close_master_table" / "master_table.csv").set_index("key")
    preds = {k: load_stored(info, k) for k in STORED}
    tab, _ = score_table(info, preds)
    for _, r in tab.iterrows():
        m = mt.loc[MASTER_KEY[r["arm"]]]
        for col, mcol in (("qlike", "qlike_recal"), ("sharpe_mid", "Sharpe_mid")):
            rows.append(
                dict(
                    gate=f"scorer reproduces the master table ({col})",
                    item=r["arm"],
                    value=r[col],
                    reference=float(m[mcol]),
                    abs_diff=abs(r[col] - float(m[mcol])),
                    ok=abs(r[col] - float(m[mcol])) < 1e-9,
                )
            )
    # (b) the fold fitter == the Optuna spec's fit_trial (shipped configuration, last pre-tune fold)
    f = pretune_folds(info)[-1]
    job = fold_jobs(info, [f], shipped(), "pre")[0]
    mine = CJ.fit_fold(job)
    import causal_tune_trees_tuned_jobs as J
    from src.models.window_mask import window_keep

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
                gate="fold fitter == Optuna spec fit_trial (shipped config, last pre-tune fold)",
                item=k,
                value=float(mine[k]),
                reference=float(ref[k]),
                abs_diff=abs(float(mine[k]) - float(ref[k])),
                ok=float(mine[k]) == float(ref[k]),
            )
        )
    cc = info["cache_compare"]
    for k, v in cc.items():
        rows.append(
            dict(
                gate="16:00 rows: last-hour cache vs 16:00-bar cache (common sessions)",
                item=k,
                value=v,
                reference=np.nan,
                abs_diff=np.nan,
                ok=True,
            )
        )
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "gate.csv", index=False)
    print(g.to_string(index=False))
    if not g["ok"].all():
        raise SystemExit("GATE FAILED")
    say(f"gate OK; fold fit {mine['sec']:.1f}s, rounds {mine['rounds']}")


# ============================================================================ pretune
def stage_pretune() -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    info = prepare_arrays()
    folds = pretune_folds(info)
    RUN.mkdir(parents=True, exist_ok=True)
    jpath = RUN / f"pretune_s{SEED}.journal"
    name = f"pretune_{MODEL}_s{SEED}_f{FOLDS}x{FOLD_LEN}"
    study = optuna.create_study(
        study_name=name, storage=journal(jpath), direction="minimize", load_if_exists=True
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    seed_eff = TPE_SEED_BASE + SEED + RESUME_SALT * len(done)
    study.sampler = optuna.samplers.TPESampler(seed=seed_eff)
    if not study.trials:
        study.enqueue_trial(shipped())
    meta_path = RUN / f"pretune_s{SEED}_resumes.json"
    resumes = json.loads(meta_path.read_text()) if meta_path.is_file() else []
    resumes.append(dict(completed_before=len(done), sampler_seed=seed_eff, start=time.ctime()))
    meta_path.write_text(json.dumps(resumes, indent=1))
    say(
        f"pre-tune {name}: {len(done)} trials already complete, target {TRIALS}; folds "
        + "; ".join(f"{f['val_first']}..{f['val_last']} (fit from {f['fit_first']})" for f in folds)
    )
    space = spaces()
    t0 = time.time()

    def objective(trial) -> float:
        import causal_tune_trees_optuna_jobs as OJ
        import close_trees_pretune_jobs as CJ

        cfg = OJ.suggest(trial, space)
        outs = run_jobs(CJ.fit_fold, fold_jobs(info, folds, cfg, "pre"))
        for k in ("val_mse", "val_qlike", "rounds", "sec", "n_kept", "rounds_max"):
            trial.set_user_attr(f"fold_{k}", [float(o[k]) for o in outs])
        v = float(np.mean([o["val_mse"] for o in outs]))
        if trial.number % 10 == 0 or trial.number < 3:
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
    assert [t.number for t in tr] == list(range(len(tr))), "pre-tune trials are not contiguous"
    space = spaces()
    arrs = dict(
        params=np.vstack([encode({a: t.params[a] for a in space}) for t in tr]),
        val_mse=np.array([t.value for t in tr]),
    )
    for k in ("val_mse", "val_qlike", "rounds", "sec", "n_kept", "rounds_max"):
        arrs[f"fold_{k}"] = np.array([t.user_attrs[f"fold_{k}"] for t in tr], dtype=float)
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
    )
    np.savez_compressed(RUN / f"pretune_s{SEED}.npz", meta=json.dumps(meta), **arrs)
    b = int(np.argmin(arrs["val_mse"]))
    say(
        f"pre-tune export: {len(tr)} trials; best trial {b} val MSE {arrs['val_mse'][b]:.5f} "
        f"(shipped {arrs['val_mse'][0]:.5f}), {arrs['fold_sec'].sum() / 3600:.2f} core-h of fold fits"
    )


def versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "optuna"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- a library the model does not use may be absent
            out[mod] = "absent"
    return out


def load_pretune(seeds=SEEDS) -> dict:
    """The pre-tune trials of the given seeds (independent studies), merged in seed order."""
    recs = []
    for s in seeds:
        f = RUN / f"pretune_s{s}.npz"
        if not f.is_file():
            raise SystemExit(f"{f} missing (run the pretune stage with CTP_SEED={s})")
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files if k != "meta"}
        d["meta"] = json.loads(str(z["meta"]))
        d["seed"] = s
        recs.append(d)
    return {"by_seed": recs}


def best_after(rec: dict, k: int) -> int:
    v = rec["val_mse"][: min(k, len(rec["val_mse"]))]
    return int(np.argmin(np.where(np.isnan(v), np.inf, v)))


# ============================================================================ walk
def box_space(inc: dict) -> dict:
    """The spec's space confined to a box around the incumbent: NEIGHBOURHOOD_FRACTION of each
    axis's range (log units on log axes), centred on the incumbent, clipped to the range."""
    out = {}
    for a, s in spaces().items():
        kind, lo, hi, log = s
        v = float(inc[a])
        if log:
            h = NEIGHBOURHOOD_FRACTION * (math.log(hi) - math.log(lo)) / 2
            blo, bhi = max(lo, v * math.exp(-h)), min(hi, v * math.exp(h))
        else:
            h = NEIGHBOURHOOD_FRACTION * (hi - lo) / 2
            blo, bhi = max(lo, v - h), min(hi, v + h)
        if kind == "int":
            blo, bhi = max(lo, math.floor(blo)), min(hi, math.ceil(bhi))
            out[a] = ("int", int(blo), int(bhi), log)
        else:
            out[a] = ("float", blo, bhi, log)
    return out


def hac_t(d: np.ndarray) -> float:
    """Mean / Newey-West standard error of a paired loss difference (dm_test's statistic)."""
    from src.evaluation.diebold_mariano import dm_test

    return float(dm_test(d, np.zeros_like(d))["dm"])


_RETUNE_CACHE: dict = {}


def retune(info: dict, row: int, inc: dict, inc_rounds: int) -> dict:
    """One light retune at OOS row `row` around the incumbent; returns every trial and both
    decisions (any improvement / Bonferroni-guarded margin)."""
    import optuna

    import causal_tune_trees_optuna_jobs as OJ
    import close_trees_pretune_jobs as CJ

    key = (row, encode(inc).tobytes())
    if key in _RETUNE_CACHE:
        return _RETUNE_CACHE[key]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = retune_folds(info, row)
    space = box_space(inc)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=TPE_SEED_BASE + SEED + row)
    )
    study.enqueue_trial({a: inc[a] for a in space})
    recs: list[dict] = []

    def objective(trial) -> float:
        cfg = OJ.suggest(trial, space)
        outs = run_jobs(CJ.fit_fold, fold_jobs(info, folds, cfg, "bar"))
        e2 = np.concatenate([o["e2"] for o in outs])
        recs.append(
            dict(
                cfg=cfg,
                val_mse=float(np.mean(e2)),
                fold_mse=[o["val_mse"] for o in outs],
                val_qlike=float(np.mean([o["val_qlike"] for o in outs])),
                rounds=[o["rounds"] for o in outs],
                sec=sum(o["sec"] for o in outs),
                e2=e2,
            )
        )
        return float(np.mean(e2))

    study.optimize(objective, n_trials=RETUNE_TRIALS, n_jobs=1, show_progress_bar=False)
    assert recs[0]["cfg"] == {a: inc[a] for a in space}, "trial 0 is not the incumbent"
    mse = np.array([r["val_mse"] for r in recs])
    b = 1 + int(np.argmin(mse[1:]))  # the best challenger
    d = recs[b]["e2"] - recs[0]["e2"]
    t = hac_t(d)
    from statistics import NormalDist

    n_ch = len(recs) - 1
    z = NormalDist().inv_cdf(1 - MARGIN_ALPHA / n_ch)
    out = dict(
        row=row,
        date=str(info["date"][info["W"] + row])[:10],
        trials=recs,
        best=b,
        inc_mse=float(mse[0]),
        best_mse=float(mse[b]),
        t=t,
        z=z,
        switch_any=bool(mse[b] < mse[0]),
        switch_margin=bool(t < -z),
        new_cfg=recs[b]["cfg"],
        new_rounds=median_rounds(recs[b]["rounds"]),
        inc_rounds=inc_rounds,
        sec=sum(r["sec"] for r in recs),
    )
    _RETUNE_CACHE[key] = out
    say(
        f"  retune row {row} ({out['date']}): incumbent {mse[0]:.5f}, best challenger (trial {b}) "
        f"{mse[b]:.5f}, HAC t {t:+.2f} vs -{z:.2f}: any={out['switch_any']} margin={out['switch_margin']}"
    )
    return out


def stage_walk() -> None:
    import causal_tune_trees_optuna_jobs as OJ
    import close_trees_pretune_jobs as CJ

    info = prepare_arrays()
    W = info["W"]
    n_oos = info["n"] - W
    pt = load_pretune()
    n_st = len(pt["by_seed"])
    rec = pt["by_seed"][0] if n_st == 1 else merge_seeds(pt["by_seed"])
    n_tr = len(rec["val_mse"]) // n_st  # trials of each study in the merged sequence
    # arms: (configuration in force from row 0, rounds) plus the retune paths; a checkpoint k =
    # the best of the first k trials of EVERY study (k x studies trials in all)
    arms: dict[str, list[tuple[int, dict, int | None]]] = {}
    arms["control"] = [(0, shipped(), None)]
    ks = sorted({k for k in CHECKPOINTS if k < n_tr} | {n_tr})
    frozen_of: dict[int, tuple[dict, int]] = {}
    for k in ks:
        b = best_after(rec, k * n_st)
        cfg = decode(rec["params"][b])
        r = median_rounds(rec["fold_rounds"][b])
        frozen_of[k] = (cfg, r)
        arms[f"frozen_k{k}"] = [(0, cfg, r)]
    trade0 = int(np.searchsorted(pd.to_datetime(info["date"][W:]), pd.Timestamp(FIRST_TRADE_DAY)))
    row0 = (trade0 // 10) * 10  # the 10-session refit row at or before the first trade day
    rt_rows = list(range(row0, n_oos, RETUNE_EVERY))
    lo_hi = [int(v) for v in re.split("[,:]", WALK_ROWS)] if WALK_ROWS else [0, n_oos]
    rt_rows = [r for r in rt_rows if lo_hi[0] <= r < lo_hi[1]]
    cfg0, r0 = frozen_of[n_tr]
    rt_log = []
    for path in ("retune_any", "retune_margin"):
        sched = [(0, cfg0, r0)]
        inc, inc_r = cfg0, r0
        for row in rt_rows:
            res = retune(info, row, inc, inc_r)
            sw = res["switch_any"] if path == "retune_any" else res["switch_margin"]
            if sw:
                inc, inc_r = res["new_cfg"], res["new_rounds"]
                sched.append((row, inc, inc_r))
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
                    **{f"cfg_{a}": inc[a] for a in spaces()},
                    retune_fit_sec=res["sec"],
                )
            )
        arms[path] = sched
    RUN.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rt_log).to_csv(RUN / f"retune_log_r{REFIT_EVERY}.csv", index=False)
    save_retune_trials()
    # refits: one job per distinct (refit row, configuration, rounds)
    refit_rows = [i for i in range(0, n_oos, REFIT_EVERY) if lo_hi[0] <= i < lo_hi[1]]
    key_of: dict[tuple, int] = {}
    jobs: list[dict] = []
    use: dict[str, list[int]] = {}
    P = info["paths"]
    for arm, sched in arms.items():
        use[arm] = []
        for i in refit_rows:
            _, cfg, r = [s for s in sched if s[0] <= i][-1]
            key = (i, encode(cfg).tobytes(), r)
            if key not in key_of:
                key_of[key] = len(jobs)
                k = min(REFIT_EVERY, n_oos - i)
                jobs.append(
                    dict(X=P["bar_X"], y=P["bar_y"], t=W + i, W=W, k=k, cfg=cfg, rounds=r, qcfg=None, shap=False)
                )
            use[arm].append(key_of[key])
    say(f"walk: {len(arms)} arms x {len(refit_rows)} refits = {len(jobs)} distinct fits")
    t0 = time.time()
    outs = run_jobs(CJ.refit, jobs)
    for arm in arms:
        pred = np.concatenate([outs[q]["preds"] for q in use[arm]])
        fit_sec = np.array([outs[q]["fit_sec"] for q in use[arm]])
        n_kept = np.array([outs[q]["n_kept"] for q in use[arm]])
        np.savez_compressed(
            RUN / f"walk_{arm}_r{REFIT_EVERY}.npz",
            pred=pred,
            rows=np.array(refit_rows),
            fit_sec=fit_sec,
            n_kept=n_kept,
            schedule=json.dumps([[s[0], s[1], s[2]] for s in arms[arm]]),
            meta=json.dumps(dict(refit_every=REFIT_EVERY, walk_rows=lo_hi, versions=versions())),
        )
    say(
        f"walk done: {len(jobs)} fits, {sum(o['fit_sec'] for o in outs) / 3600:.2f} core-h, "
        f"wall {time.time() - t0:.0f}s"
    )
    del OJ


def merge_seeds(by_seed: list[dict]) -> dict:
    """Independent seeded studies merged: their trials concatenated in seed order, so 'best after
    k trials' means after k trials of EACH study (k x seeds in all)."""
    n = min(len(r["val_mse"]) for r in by_seed)
    keys = [k for k in by_seed[0] if isinstance(by_seed[0][k], np.ndarray)]
    out = {}
    for k in keys:  # interleave: trial j of every seed before trial j + 1 of any
        out[k] = np.stack([r[k][:n] for r in by_seed], axis=1).reshape(n * len(by_seed), *by_seed[0][k].shape[1:])
    out["meta"] = by_seed[0]["meta"]
    return out


def save_retune_trials() -> None:
    rows = []
    for (row, _), res in _RETUNE_CACHE.items():
        for j, r in enumerate(res["trials"]):
            rows.append(
                dict(
                    row=row,
                    date=res["date"],
                    trial=j,
                    val_mse=r["val_mse"],
                    fold0_mse=r["fold_mse"][0],
                    fold1_mse=r["fold_mse"][1],
                    val_qlike=r["val_qlike"],
                    rounds_median=median_rounds(r["rounds"]),
                    sec=r["sec"],
                    incumbent=res["trials"][0]["cfg"] == r["cfg"] and j == 0,
                    **{f"cfg_{a}": v for a, v in r["cfg"].items()},
                )
            )
    pd.DataFrame(rows).to_csv(RUN / f"retune_trials_r{REFIT_EVERY}.csv", index=False)


# ============================================================================ main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    stage = sys.argv[1]
    t_start = time.time()
    try:
        if stage == "gate":
            stage_gate()
        elif stage == "pretune":
            stage_pretune()
        elif stage == "walk":
            stage_walk()
        elif stage == "report":
            import close_trees_pretune_report as R

            R.main()
        else:
            raise SystemExit(f"unknown stage {stage!r}")
    finally:
        if _POOL is not None:
            _POOL.shutdown()
    say(f"stage {stage} done in {time.time() - t_start:.0f}s")
