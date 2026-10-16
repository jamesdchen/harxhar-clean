"""Dev-span tree menu study + freeze -- the RANDOM FOREST family (user decision 2026-10-06).

WHY.  The per-bar 16:00 LightGBM and XGBoost settings (specs/causal_tune_trees.py LGBM_PARAMS /
XGB_PARAMS, provenance lgbm_opt_00 / xgb_opt_06) come from experiments/tree_menu_dev.py: one Optuna
study per family over a box, on the DEV SPAN ONLY (panel rows before 2003-01-01), objective =
rolling-backtest mean sqrt-space MSE, window 24,000 bars, refit every 250 bars in the screen, then a
freeze to top-8 unique completed trials + 2 Sobol draws per family.  The random forest was never in
that search (it runs sklearn defaults).  This script puts the forest through the same procedure so
all three trees' settings come from one identical, look-ahead-free process.

WHAT IS IDENTICAL (imported from experiments/tree_menu_dev.py, not copied): DEV_END_DATE, WINDOW,
SCREEN_REFIT, the screen arithmetic (blocks t0 = WINDOW, WINDOW+250, ... ; sse accumulated block by
block in time order; value = sse / bars), the model seed (random_state 42), _suggest (the box ->
trial.suggest_* mapping), _sha, _storage (Optuna JournalStorage on a file backend), _sobol_draws
(scrambled Sobol, seed 20260807), N_TOP_PER_FAMILY = 8, N_SOBOL_PER_FAMILY = 2; the sampler (Optuna's
default TPE, unseeded, as tree_menu_dev's create_study); the worker layout of jobs/slurm/tree_menu.sbatch
(8 workers sharing one journal, --n-trials 100 split over two families = 50 trials of a family per
worker -> a nominal budget of 400 trials a family).  The freeze below is tree_menu_dev.freeze's
per-family loop for the one family (tree_menu_dev.freeze itself writes experiments/tree_menu.json,
which this study must not touch).

WHAT DIFFERS (disclosed).  (1) Fits are single-threaded and the 50 refits of one trial run in a pool
of processes (the original ran them one after another with 8 threads a fit); the block sums are
added in the same order, so the value does not depend on the pool, and a random forest's fit does
not depend on n_jobs.  (2) The dev matrices: tree_menu_dev._dev_matrices built them through
src.unification._load_panel, i.e. from today's panel code, which is no longer the panel the bank
was tuned on (FOMC columns etc. were added later).  This study reads rows [0, dev_end) of the
bank-era panel cache itself (harxhar-clean/results/prep_cache_all_features_b2.npz: 300,317 x 1144,
the shape every bank chunk records), dev_end = the first row stamped on or after 2003-01-01 in the
bank chunks (np.searchsorted, as _dev_matrices); y is checked equal to the bank chunks' y_fit on
every dev row they cover, and the gate below re-scores stored menu entries on these matrices.
(3) Each trial records its fit seconds, host and CPU model as user attributes (read-only extras).

THE RF BOX (name, low, high, log, is_int; bootstrap fixed True, the forest's resampling, which
max_samples needs):
  n_estimators      50 .. 500   linear int  -- the LightGBM / XGBoost range
  max_depth          2 .. 32    log int     -- 32 is effectively unlimited at a 24,000-row window
  min_samples_leaf   1 .. 1000  log int     -- sklearn's default 1 up to ~4 % of the window
                                               (LightGBM's min_child_samples box is 20 .. 500)
  max_features    0.02 .. 1.0   log float   -- fraction of the 1144 columns (23 .. all; 1.0 is
                                               sklearn's regression default)
  max_samples     0.05 .. 1.0   log float   -- bootstrap draw per tree as a fraction of the window
                                               (1.0 is sklearn's default)
The two fractions span more than a decade, hence log scales (the GBM fraction boxes span less than
an octave, where linear and log hardly differ).

GATE.  The study's objective re-run (this script's pool screen) for the stored CARC menu entry
lgbm_opt_00 (the per-bar LightGBM config) must reproduce its stored dev_screen_mse (relative
difference below GATE_RTOL; REQUIRED before any worker runs).  xgb_opt_06 (the per-bar XGBoost
config) and xgb_opt_00 are re-scored too and recorded, each in a fresh pool (informational).

MODES
  extract  (local) dev matrices from the bank-era cache + bank chunks -> _work/dev/{X,y,names}.npy
  gate     re-score the stored entries; writes _work/<tag>/gate.json
  cost     time one fit at 16 Sobol points of the box (cost estimate); writes _work/<tag>/cost.json
  worker   one Optuna worker: --n-trials trials of the RF family into --journal
  freeze   menu (top 8 + 2 Sobol), rf_config.json, SUMMARY.md, RF_CONFIG_DONE  (local)
  smoke    synthetic panel-free check of the pool screen and the RF model
"""

import _bootstrap  # noqa: F401  (sys.path: repo root + experiments/)

import argparse
import glob
import hashlib
import json
import os
import platform
import socket
import sys
import time

import numpy as np

import tree_menu_dev as TMD

OUT = os.path.join("results", "rf_devmenu_2026-10-06")
WORK = os.path.join(OUT, "_work")
DEV_DIR = os.path.join(WORK, "dev")
JOURNAL = os.path.join(WORK, "tree_menu_journal_rf.log")
STUDY = "tree_menu_dev_rf"
FAMILY = "rf"
SEED = 42  # tree_menu_dev._make_model's random_state
PER_WORKER_TRIALS = 50  # jobs/slurm/tree_menu.sbatch: --n-trials 100 // 2 families
N_WORKERS = 8  # jobs/slurm/tree_menu.sbatch: --array=0-7
BUDGET = (
    N_WORKERS * PER_WORKER_TRIALS
)  # 400 trials a family (nominal; LightGBM completed it)
XGB_RULE_INDEX = (
    6  # the per-bar XGBoost config is xgb_opt_06, the 7th-ranked XGBoost dev entry
)
GATE_RTOL = (
    1e-7  # well below the 4e-6 relative spacing of the top LightGBM menu entries
)
COST_DRAWS, COST_SEED, COST_BLOCK = (
    16,
    1,
    25,
)  # cost probe: Sobol points, seed, refit block

RF_BOX: list[tuple[str, float, float, bool, bool]] = [
    ("n_estimators", 50, 500, False, True),
    ("max_depth", 2, 32, True, True),
    ("min_samples_leaf", 1, 1000, True, True),
    ("max_features", 0.02, 1.0, True, False),
    ("max_samples", 0.05, 1.0, True, False),
]
RF_FIXED = {"bootstrap": True}
# register the family with tree_menu_dev so its own _suggest / _sobol_draws serve the forest
TMD.BOXES.setdefault(FAMILY, RF_BOX)
TMD.FIXED.setdefault(FAMILY, RF_FIXED)

# The CARC bank's menu entries the gate re-scores: copied from the 'tree_config' of
# harxhar-clean/results/unification_carc/tree_expert_{00,10,16}/chunk_000.npz (every chunk of an
# arm carries the same sha); the sha is re-derived from the params below before any use.
STORED = {
    "lgbm_opt_00": {
        "family": "lgbm",
        "sha": "abab811bd4b6",
        "dev_screen_mse": 0.10744949046943114,
        "params": {
            "bagging_fraction": 0.5636236714089857,
            "bagging_freq": 1,
            "feature_fraction": 0.7017000121697539,
            "lambda_l1": 1.0413823540826713e-08,
            "lambda_l2": 0.015950303832251992,
            "learning_rate": 0.01296404565731986,
            "min_child_samples": 98,
            "n_estimators": 392,
            "num_leaves": 20,
        },
    },
    "xgb_opt_00": {
        "family": "xgb",
        "sha": "88bec9f385c7",
        "dev_screen_mse": 0.10789407028049709,
        "params": {
            "colsample_bytree": 0.6538448390071842,
            "gamma": 1.5147505105071128e-06,
            "learning_rate": 0.01233734017817195,
            "max_depth": 5,
            "min_child_weight": 50.20121979133419,
            "n_estimators": 367,
            "reg_alpha": 0.021478436574914658,
            "reg_lambda": 1.15908122770414e-05,
            "subsample": 0.6551492349018447,
            "tree_method": "hist",
        },
    },
    "xgb_opt_06": {
        "family": "xgb",
        "sha": "195b74503d9d",
        "dev_screen_mse": 0.10807554130151135,
        "params": {
            "colsample_bytree": 0.6975888053713192,
            "gamma": 1.1082753230505144e-05,
            "learning_rate": 0.010853165384309853,
            "max_depth": 4,
            "min_child_weight": 24.86134158090589,
            "n_estimators": 339,
            "reg_alpha": 1.087850849987857,
            "reg_lambda": 0.00041206887024929935,
            "subsample": 0.6086526390529718,
            "tree_method": "hist",
        },
    },
}
GATE_ENTRIES = ("lgbm_opt_00", "xgb_opt_06", "xgb_opt_00")
GATE_REQUIRED = ("lgbm_opt_00",)  # the task's gate: the frozen LightGBM config


# ── pool screen ───────────────────────────────────────────────────────────────

_X: np.ndarray | None = None
_Y: np.ndarray | None = None


def _init_pool(dev_dir: str) -> None:
    global _X, _Y
    import faulthandler
    import warnings

    faulthandler.enable()  # a worker killed by a signal leaves a traceback in the job log

    # LightGBM names numpy columns Column_i at fit; predicting on a bare array then trips
    # sklearn's feature-name check (harmless; silenced as in specs/causal_tune_trees.py)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    if dev_dir == "__synthetic__":
        rng = np.random.default_rng(0)
        _X = rng.standard_normal((1200, 8))
        _Y = _X[:, 0] * 0.5 + rng.standard_normal(1200) * 0.1
        return
    _X = np.load(os.path.join(dev_dir, "X.npy"), mmap_mode="r")
    _Y = np.load(os.path.join(dev_dir, "y.npy"), mmap_mode="r")


def _model(family: str, params: dict):
    if family == FAMILY:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params, n_jobs=1, random_state=SEED)
    return TMD._make_model(family, params, n_threads=1)


def _fit_block(job: tuple) -> tuple[int, float, int, float]:
    """One screen block: fit on the trailing window, predict the next <= refit bars.

    Same arithmetic as tree_menu_dev._screen_score's loop body."""
    family, params, t0, window, refit = job
    assert _X is not None and _Y is not None
    n = len(_Y)
    t1 = min(t0 + refit, n)
    a = time.perf_counter()
    model = _model(family, params)
    model.fit(np.asarray(_X[t0 - window : t0]), np.asarray(_Y[t0 - window : t0]))
    pred = model.predict(np.asarray(_X[t0:t1]))
    sse = float(np.sum((np.asarray(_Y[t0:t1]) - pred) ** 2))
    return t0, sse, t1 - t0, time.perf_counter() - a


def _pool(procs: int, dev_dir: str):
    """A process pool of single-threaded fitters.  concurrent.futures, not multiprocessing.Pool:
    a worker that dies (signal, OOM kill) raises BrokenProcessPool in the caller instead of
    leaving the map waiting forever on the lost task (the first canary, CARC job 12776003, hung
    that way in its XGBoost gate entry)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    return ProcessPoolExecutor(
        max_workers=procs, mp_context=ctx, initializer=_init_pool, initargs=(dev_dir,)
    )


def screen(pool, family: str, params: dict, n: int, window: int = 0, refit: int = 0):
    """Rolling-backtest mean sqrt-space MSE on the dev span (tree_menu_dev's screen), the
    blocks fitted in parallel and summed in time order.  Returns (value, per-block seconds)."""
    window = window or TMD.WINDOW
    refit = refit or TMD.SCREEN_REFIT
    jobs = [(family, params, t0, window, refit) for t0 in range(window, n, refit)]
    res = list(pool.map(_fit_block, jobs))
    sse, cnt = 0.0, 0
    for t0, s, c, _sec in sorted(res):
        sse += s
        cnt += c
    return sse / cnt, [r[3] for r in sorted(res)]


def _dev_n(dev_dir: str) -> int:
    return int(np.load(os.path.join(dev_dir, "y.npy"), mmap_mode="r").shape[0])


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "sklearn", "lightgbm", "xgboost", "optuna", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- recorded as absent
            out[mod] = "absent"
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _procs(arg: int) -> int:
    return arg or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


# ── extract (local) ───────────────────────────────────────────────────────────


def _zip_member_memmap(path: str, name: str) -> np.memmap:
    """Memory-map one STORED (uncompressed) .npy member of an .npz without loading it."""
    import zipfile
    from numpy.lib import format as npf

    info = zipfile.ZipFile(path).getinfo(name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise SystemExit(f"{path}:{name} is compressed; cannot memory-map")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        h = fh.read(30)
        off = info.header_offset + 30 + int.from_bytes(h[26:28], "little")
        off += int.from_bytes(h[28:30], "little")
        fh.seek(off)
        ver = npf.read_magic(fh)
        shape, fortran, dtype = npf._read_array_header(fh, ver)
        data_off = fh.tell()
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=data_off,
        shape=shape,
        order="F" if fortran else "C",
    )


def extract(cache: str, bank: str) -> None:
    files = sorted(glob.glob(os.path.join(bank, "tree_expert_00", "chunk_*.npz")))
    if not files:
        raise SystemExit(f"no bank chunks under {bank}/tree_expert_00")
    rows, ts, yf, meta0 = [], [], [], None
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta0 = meta0 or json.loads(str(z["meta"]))
        rows.append(np.asarray(z["row_id"]))
        ts.append(np.asarray(z["t"]))
        yf.append(np.asarray(z["y_fit"], dtype=np.float64))
        if ts[-1][-1] >= TMD.DEV_END_DATE:
            break
    row, t, y_fit = np.concatenate(rows), np.concatenate(ts), np.concatenate(yf)
    if not np.all(np.diff(row) == 1) or row[0] != TMD.WINDOW:
        raise SystemExit(
            "bank chunks are not one contiguous run of rows from the window"
        )
    i = int(np.searchsorted(t, TMD.DEV_END_DATE))
    dev_end = int(row[i])
    X = _zip_member_memmap(cache, "X.npy")
    y = _zip_member_memmap(cache, "y.npy")
    if X.shape[1] != meta0["n_design_cols"]:
        raise SystemExit(
            f"cache has {X.shape[1]} columns, the bank {meta0['n_design_cols']}"
        )
    Xd = np.ascontiguousarray(X[:dev_end], dtype=np.float64)
    yd = np.array(y[:dev_end], dtype=np.float64)
    sel = row < dev_end
    ydiff = float(np.max(np.abs(yd[row[sel]] - y_fit[sel])))
    if ydiff != 0.0:
        raise SystemExit(
            f"cache y differs from the bank chunks' y_fit on dev rows: {ydiff}"
        )
    if not (np.isfinite(Xd).all() and np.isfinite(yd).all()):
        raise SystemExit("non-finite values in the dev matrices")
    names = np.load(cache, allow_pickle=True)["names"]
    os.makedirs(DEV_DIR, exist_ok=True)
    np.save(os.path.join(DEV_DIR, "X.npy"), Xd)
    np.save(os.path.join(DEV_DIR, "y.npy"), yd)
    np.save(os.path.join(DEV_DIR, "names.npy"), np.asarray([str(s) for s in names]))
    man = {
        "source_cache": os.path.abspath(cache),
        "source_cache_bytes": os.path.getsize(cache),
        "source_cache_mtime": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(cache))
        ),
        "cache_shape": list(X.shape),
        "bank": os.path.abspath(bank),
        "bank_n_design_cols": meta0["n_design_cols"],
        "dev_end_row": dev_end,
        "dev_end_rule": f"first bank row stamped >= {TMD.DEV_END_DATE} (np.searchsorted)",
        "last_dev_stamp": str(t[i - 1]),
        "first_post_dev_stamp": str(t[i]),
        "eval_bars": dev_end - TMD.WINDOW,
        "screen_blocks": len(range(TMD.WINDOW, dev_end, TMD.SCREEN_REFIT)),
        "y_vs_bank_y_fit_max_abs_diff": ydiff,
        "y_rows_checked": int(sel.sum()),
        "sha256": {
            k: _sha256_file(os.path.join(DEV_DIR, f"{k}.npy"))
            for k in ("X", "y", "names")
        },
    }
    with open(os.path.join(DEV_DIR, "dev_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps(man, indent=1))


def _check_dev(dev_dir: str) -> dict:
    with open(os.path.join(dev_dir, "dev_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    for k, want in man["sha256"].items():
        got = _sha256_file(os.path.join(dev_dir, f"{k}.npy"))
        if got != want:
            raise SystemExit(f"dev matrix {k}.npy sha256 {got} != manifest {want}")
    return man


# ── gate / cost ───────────────────────────────────────────────────────────────


def gate(tag: str, procs: int, entries: tuple[str, ...]) -> bool:
    man = _check_dev(DEV_DIR)
    n = _dev_n(DEV_DIR)
    out = {
        "tag": tag,
        "host": socket.gethostname(),
        "cpu": _cpu_model(),
        "procs": procs,
    }
    out["versions"], out["dev_end_row"], out["entries"] = (
        _versions(),
        man["dev_end_row"],
        {},
    )
    ok = True
    for name in entries:
        e = STORED[name]
        if TMD._sha(e["family"], e["params"]) != e["sha"]:
            raise SystemExit(f"{name}: params do not hash to the stored sha {e['sha']}")
        a = time.time()
        rec: dict = {
            "family": e["family"],
            "sha": e["sha"],
            "stored": e["dev_screen_mse"],
        }
        try:
            with _pool(procs, DEV_DIR) as pool:  # a fresh pool per entry
                value, secs = screen(pool, e["family"], e["params"], n)
        except Exception as exc:  # noqa: BLE001 -- recorded; fatal only for a required entry
            rec.update(error=repr(exc), rerun=None, rel_diff=None, pass_=False)
            passed = False
            print(f"gate {name}: ERROR {exc!r}", flush=True)
        else:
            rel = abs(value - e["dev_screen_mse"]) / e["dev_screen_mse"]
            passed = bool(rel < GATE_RTOL)
            rec.update(
                rerun=value,
                abs_diff=value - e["dev_screen_mse"],
                rel_diff=rel,
                fit_sec_sum=float(sum(secs)),
            )
            print(
                f"gate {name}: stored {e['dev_screen_mse']!r} rerun {value!r} rel {rel:.2e} "
                f"{'PASS' if passed else 'FAIL'}",
                flush=True,
            )
        rec.pop("pass_", None)
        rec.update(**{"pass": passed, "required": name in GATE_REQUIRED})
        rec["wall_sec"] = time.time() - a
        out["entries"][name] = rec
        if name in GATE_REQUIRED:
            ok &= passed
        d = os.path.join(WORK, tag)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "gate.json"), "w", encoding="utf-8") as fh:
            json.dump({**out, "pass": ok, "partial": True}, fh, indent=1)
    # None = no required entry in this run (informational re-scores only)
    out["pass"] = ok if any(nm in GATE_REQUIRED for nm in entries) else None
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "gate.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return ok


def _cost_draws() -> list[dict]:
    from scipy.stats import qmc

    draws = []
    for row in qmc.Sobol(d=len(RF_BOX), scramble=True, seed=COST_SEED).random(
        COST_DRAWS
    ):
        p = {}
        for (name, lo, hi, log, is_int), v in zip(RF_BOX, row):
            x = (
                float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))))
                if log
                else lo + v * (hi - lo)
            )
            p[name] = int(round(x)) if is_int else float(x)
        p.update(RF_FIXED)
        draws.append(p)
    return draws


def cost(tag: str, procs: int) -> None:
    _check_dev(DEV_DIR)
    t0 = TMD.WINDOW + COST_BLOCK * TMD.SCREEN_REFIT
    jobs = [(FAMILY, p, t0, TMD.WINDOW, TMD.SCREEN_REFIT) for p in _cost_draws()]
    with _pool(procs, DEV_DIR) as pool:
        res = list(pool.map(_fit_block, jobs))
    n_blocks = len(range(TMD.WINDOW, _dev_n(DEV_DIR), TMD.SCREEN_REFIT))
    rows = [{"params": j[1], "fit_sec": r[3]} for j, r in zip(jobs, res)]
    mean = float(np.mean([r["fit_sec"] for r in rows]))
    out = {
        "tag": tag,
        "host": socket.gethostname(),
        "cpu": _cpu_model(),
        "procs": procs,
        "draws": rows,
        "mean_fit_sec": mean,
        "est_trial_cpu_h": mean * n_blocks / 3600,
        "est_budget_cpu_h": mean * n_blocks * BUDGET / 3600,
    }
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cost.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(
        f"cost: mean fit {mean:.0f}s, est {out['est_trial_cpu_h']:.2f} CPU-h a trial, "
        f"{out['est_budget_cpu_h']:.0f} CPU-h for {BUDGET} trials ({out['cpu']})",
        flush=True,
    )


# ── worker ────────────────────────────────────────────────────────────────────


def run_worker(journal: str, n_trials: int, procs: int, cap: int = 0) -> None:
    """One TPE worker.  cap > 0 (top-up workers added 2026-10-07 to reach the 400 budget): a
    trial numbered >= cap is pruned before it samples or fits anything and the worker stops, so
    no compute goes to trials beyond the budget (the original 8 workers run without a cap)."""
    import optuna

    man = _check_dev(DEV_DIR)
    n = _dev_n(DEV_DIR)
    host, cpu = socket.gethostname(), _cpu_model()
    print(
        f"dev span: rows {n} (dev_end {man['dev_end_row']}), eval bars {n - TMD.WINDOW}, "
        f"pool {procs} single-threaded fits, {host} [{cpu}]",
        flush=True,
    )
    with _pool(procs, DEV_DIR) as pool:

        def objective(trial) -> float:
            if cap and trial.number >= cap:
                study.stop()
                raise optuna.TrialPruned(f"trial {trial.number} >= budget cap {cap}")
            params = TMD._suggest(trial, FAMILY)
            a = time.time()
            value, secs = screen(pool, FAMILY, params, n)
            trial.set_user_attr("fit_sec_sum", float(sum(secs)))
            trial.set_user_attr("fit_sec_max", float(max(secs)))
            trial.set_user_attr("wall_sec", time.time() - a)
            trial.set_user_attr("host", host)
            trial.set_user_attr("cpu", cpu)
            trial.set_user_attr("procs", procs)
            return value

        study = optuna.create_study(
            study_name=STUDY,
            storage=TMD._storage(journal),
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)


# ── freeze + report (local) ───────────────────────────────────────────────────


def _load_trials(journal: str, study: str = STUDY):
    import optuna

    st = optuna.load_study(study_name=study, storage=TMD._storage(journal))
    return st.trials


def _menu(trials, limit: int, order: str = "completion") -> tuple[list[dict], list]:
    """tree_menu_dev.freeze's per-family loop for the forest, plus its 2 Sobol draws, over the
    budget: order "completion" = the first `limit` completed trials in completion order (the
    87-trial freeze); order "number" = the completed trials numbered < `limit`, i.e. the first
    `limit` trials started (the 400-trial freeze; every one of them must be complete)."""
    import optuna

    done = [
        t
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if order == "number" and limit:
        have = {t.number for t in done}
        missing = [
            t.number for t in trials if t.number < limit and t.number not in have
        ]
        if missing:
            raise SystemExit(f"trials numbered < {limit} not complete: {missing}")
        considered = [t for t in done if t.number < limit]
    else:
        done.sort(key=lambda t: t.datetime_complete)
        considered = done[:limit] if limit else done
    ranked = sorted(sorted(considered, key=lambda t: t.number), key=lambda t: t.value)
    menu, seen, fam_n = [], set(), 0
    for t in ranked:
        params = dict(t.params)
        params.update(TMD.FIXED[FAMILY])
        h = TMD._sha(FAMILY, params)
        if h in seen:
            continue
        seen.add(h)
        menu.append(
            {
                "name": f"{FAMILY}_opt_{fam_n:02d}",
                "family": FAMILY,
                "params": params,
                "sha": h,
                "dev_screen_mse": float(t.value),
                "trial": t.number,
            }
        )
        fam_n += 1
        if fam_n == TMD.N_TOP_PER_FAMILY:
            break
    for k, params in enumerate(TMD._sobol_draws(FAMILY)):
        menu.append(
            {
                "name": f"{FAMILY}_sobol_{k:02d}",
                "family": FAMILY,
                "params": params,
                "sha": TMD._sha(FAMILY, params),
            }
        )
    return menu, considered


def _bank_table(bank: str) -> list[dict]:
    """The CARC bank's 20 entries (chunk meta) with per-arm sqrt-space MSE by span, computed from
    the stored forecasts (the per-bar refit bank, not the screen)."""
    import pandas as pd

    rows = []
    for k in range(20):
        fs = sorted(
            glob.glob(os.path.join(bank, f"tree_expert_{k:02d}", "chunk_*.npz"))
        )
        if not fs:
            continue
        T, E2, cfg = [], [], None
        for f in fs:
            z = np.load(f, allow_pickle=True)
            cfg = cfg or json.loads(str(z["meta"]))["tree_config"]
            T.append(np.asarray(z["t"]))
            E2.append(
                (np.asarray(z["y_fit"], float) - np.asarray(z["yhat"], float)) ** 2
            )
        t, e2 = pd.to_datetime(np.concatenate(T)), np.concatenate(E2)
        ok = np.isfinite(e2)
        post = ok & (t >= "2003-01-01")
        win = ok & (t >= "2020-01-01") & (t < "2025-01-01")
        b16 = win & (t.hour == 16) & (t.minute == 0)
        rows.append(
            {
                "arm": f"tree_expert_{k:02d}",
                "name": cfg["name"],
                "family": cfg["family"],
                "sha": cfg["sha"],
                "dev_screen_mse": cfg.get("dev_screen_mse"),
                "chunks": len(fs),
                "mse_2003_on": float(e2[post].mean()),
                "mse_2020_24": float(e2[win].mean()),
                "mse_1600_2020_24": float(e2[b16].mean()),
            }
        )
    return rows


def _rank_within(rows: list[dict], key: str, family: str, name: str) -> tuple[int, int]:
    opt = [r for r in rows if r["family"] == family and "_opt_" in r["name"]]
    order = sorted(opt, key=lambda r: r[key])
    return [r["name"] for r in order].index(name) + 1, len(order)


def _h2_counts(h2_dir: str) -> dict:
    import collections

    out = {}
    for fam in ("lgbm", "xgb"):
        p = os.path.join(h2_dir, f"tree_menu_journal_{fam}.log")
        if not os.path.exists(p):
            continue
        trials = _load_trials(p, f"tree_menu_dev_{fam}")
        c = collections.Counter(t.state.name for t in trials)
        starts = [t.datetime_start for t in trials if t.datetime_start]
        ends = [t.datetime_complete for t in trials if t.datetime_complete]
        out[fam] = {
            "states": dict(c),
            "first_start": str(min(starts)),
            "last_complete": str(max(ends)),
        }
    return out


def _sacct_rows(path: str) -> list[dict]:
    """Parse `sacct -X -P --format=JobID,JobName,State,Elapsed,NCPUS,Start,End,NodeList` text."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    head = lines[0].split("|")
    return [dict(zip(head, ln.split("|"))) for ln in lines[1:]]


def _hours(elapsed: str) -> float:
    d = 0
    if "-" in elapsed:
        dd, elapsed = elapsed.split("-", 1)
        d = int(dd)
    parts = [int(p) for p in elapsed.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return d * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _qlike_ranks(
    bank: str, arm: str, prefixes: tuple[str, ...], bank_rows: list[dict]
) -> str:
    """The paper scorer's 2003+ QLIKE of `arm` and its rank among the Optuna arms of its family
    (harxhar-clean/results/unification_scores.csv beside the bank folder), when present."""
    import csv

    if not bank or not bank_rows:
        return ""
    p = os.path.join(os.path.dirname(os.path.abspath(bank)), "unification_scores.csv")
    if not os.path.exists(p):
        return ""
    root = os.path.basename(os.path.abspath(bank))
    names = {r["arm"]: r["name"] for r in bank_rows}
    with open(p, encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r["root"] == root and names.get(r["arm"], "").startswith(prefixes)
        ]
    order = sorted(rows, key=lambda r: float(r["qlike"]))
    mine = [r for r in rows if r["arm"] == arm]
    if not mine:
        return ""
    k = [r["arm"] for r in order].index(arm) + 1
    return (
        f"paper-scorer QLIKE {float(mine[0]['qlike']):.5f}, rank {k}/{len(order)}, "
        f"{os.path.basename(p)}"
    )


def _node_classes() -> str:
    rows = _sacct_rows(os.path.join(WORK, "carc_tree_menu_node_classes.txt"))
    menu = _sacct_rows(os.path.join(WORK, "sacct_tree_menu_carc_2026-08-07.txt"))
    cls = {r["node"]: r["features"].split(",")[0] for r in rows}
    if not cls or not menu:
        return "classes not recorded"
    c: dict[str, int] = {}
    for r in menu:
        k = cls.get(r["NodeList"], "?")
        c[k] = c.get(k, 0) + 1
    return ", ".join(f"{v} x {k}" for k, v in sorted(c.items()))


def freeze(
    journal: str,
    limit: int,
    bank: str,
    h2_dir: str,
    tag: str,
    out: str = OUT,
    note: str = "",
    order: str = "completion",
) -> None:
    trials = _load_trials(journal)
    menu, considered = _menu(trials, limit, order)
    menu_all, done_all = _menu(trials, 0)  # every completed trial at freeze time
    states = {}
    for t in trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    top = menu[0]
    alt = menu[XGB_RULE_INDEX] if len(menu) > XGB_RULE_INDEX else None
    # the XGBoost realized budget on Hoffman2 (the only surviving journal of the procedure)
    h2 = _h2_counts(h2_dir)
    xgb_h2_complete = h2.get("xgb", {}).get("states", {}).get("COMPLETE")
    menu_small = (
        _menu(trials, xgb_h2_complete)[0]
        if xgb_h2_complete and len(done_all) >= xgb_h2_complete
        else None
    )
    fit_core_h = sum(float(t.user_attrs.get("fit_sec_sum", 0.0)) for t in trials) / 3600
    cpus = sorted(
        {str(t.user_attrs.get("cpu")) for t in trials if t.user_attrs.get("cpu")}
    )
    gate_res = _read_json(os.path.join(WORK, tag, "gate.json"))
    cost_res = _read_json(os.path.join(WORK, tag, "cost.json"))
    man = _check_dev(DEV_DIR)
    sacct_rf = _sacct_rows(os.path.join(WORK, "sacct_rf_devmenu.txt"))
    alloc_cpu_h = sum(_hours(r["Elapsed"]) * int(r["NCPUS"]) for r in sacct_rf)
    sacct_menu = _sacct_rows(os.path.join(WORK, "sacct_tree_menu_carc_2026-08-07.txt"))
    bank_rows = _bank_table(bank) if bank and os.path.isdir(bank) else []

    os.makedirs(out, exist_ok=True)
    with open(
        os.path.join(out, "menu_rf.json"), "w", encoding="utf-8", newline=""
    ) as fh:
        json.dump(menu, fh, indent=1)
    leaf_scale = (
        2000 / TMD.WINDOW
    )  # specs/causal_tune_trees.py: TRAIN_WIN / POOLED_WINDOW_ROWS
    cfg = {
        "family": FAMILY,
        "name": top["name"],
        "sha": top["sha"],
        "params": top["params"],
        "dev_screen_mse": top["dev_screen_mse"],
        "trial": top["trial"],
        "selection_rule": "top dev-span entry of the frozen menu (the rule behind lgbm_opt_00)",
        "trials": {
            "complete_considered": len(considered),
            "budget_rule": note or f"the first {limit} completed trials",
            "nominal_budget": BUDGET,
            "journal_states": states,
            "complete_at_freeze": len(done_all),
        },
        "all_complete_top_entry": menu_all[0],
        "xgb_rule_entry": alt,
        "xgb_h2_budget_top_entry": menu_small[0] if menu_small else None,
        "xgb_h2_budget": xgb_h2_complete,
        "compare_dev_screen_mse": {k: v["dev_screen_mse"] for k, v in STORED.items()},
        "per_bar_note": (
            "params are tuned on the 24,000-bar pooled window; specs/causal_tune_trees.py scales "
            "LightGBM's min_child_samples and XGBoost's min_child_weight by TRAIN_WIN / 24000 for "
            "its per-bar window; the forest's analogue is min_samples_leaf (fractions and depth "
            "are scale-free or absolute). NOT applied here."
        ),
        "min_samples_leaf_scaled_to_2000_rows": max(
            1, round(top["params"]["min_samples_leaf"] * leaf_scale)
        ),
        "provenance": {
            "script": "experiments/tree_menu_dev_rf.py",
            "procedure": "experiments/tree_menu_dev.py (objective, window, refit, dev span, "
            "seed, sampler, journal, Sobol, freeze rule imported)",
            "study": STUDY,
            "journal_sha256": _sha256_file(journal),
            "box": [list(b) for b in RF_BOX],
            "fixed": RF_FIXED,
            "window": TMD.WINDOW,
            "screen_refit": TMD.SCREEN_REFIT,
            "dev_end_date": str(TMD.DEV_END_DATE),
            "dev_end_row": man["dev_end_row"],
            "dev_matrices_sha256": man["sha256"],
            "dev_source_cache": man["source_cache"],
            "random_state": SEED,
            "cpu_models": cpus,
            "fit_core_hours": fit_core_h,
            "allocated_cpu_hours": alloc_cpu_h,
            "versions_freeze": _versions(),
            "gate": gate_res,
        },
    }
    with open(
        os.path.join(out, "rf_config.json"), "w", encoding="utf-8", newline=""
    ) as fh:
        json.dump(cfg, fh, indent=1)

    L: list[str] = []
    w = L.append
    w("# Random forest settings from the dev-span tree menu procedure (2026-10-06)\n")
    w(
        "Written by `experiments/tree_menu_dev_rf.py freeze`; every number below is read from the "
        "files it names.\n"
    )
    w("## 1. The procedure that gave LightGBM and XGBoost their settings\n")
    w(
        f"`experiments/tree_menu_dev.py`: Optuna (default TPE, unseeded) per family on the dev span "
        f"(panel rows before {TMD.DEV_END_DATE}: rows 0..{man['dev_end_row'] - 1}), objective = "
        f"rolling-backtest mean sqrt-space MSE, window {TMD.WINDOW:,} bars, refit every "
        f"{TMD.SCREEN_REFIT} bars ({man['screen_blocks']} fits a trial over {man['eval_bars']:,} "
        f"bars), design = the {man['bank_n_design_cols']}-column all_features basis.  Freeze: top "
        f"{TMD.N_TOP_PER_FAMILY} unique completed trials + {TMD.N_SOBOL_PER_FAMILY} Sobol draws a "
        f"family -> the 20-arm bank.\n"
    )
    w("### Trial budgets\n")
    w(
        f"* Nominal: {N_WORKERS} workers x `--n-trials 100` split over 2 families = "
        f"{PER_WORKER_TRIALS} trials a family a worker = **{BUDGET} a family** "
        "(`jobs/slurm/tree_menu.sbatch`; the Hoffman2 run used the same split)."
    )
    if sacct_menu:
        st = {}
        for r in sacct_menu:
            st[r["State"]] = st.get(r["State"], 0) + 1
        ch = sum(_hours(r["Elapsed"]) * int(r["NCPUS"]) for r in sacct_menu)
        w(
            f"* CARC run (the bank of record, `unification_carc`): job {sacct_menu[0]['JobID'].split('_')[0]}, "
            f"{len(sacct_menu)} workers x {sacct_menu[0]['NCPUS']} CPUs, states {st}, "
            f"{ch:.0f} allocated CPU-hours (sacct).  Its journals were on `/scratch1/jc_905/harxhar-clean`, "
            "which no longer exists: **the CARC trial counts are not recoverable**.  Every worker ran its "
            "50 LightGBM trials before any XGBoost trial, and the frozen menu has 8 XGBoost entries with "
            "dev values, so LightGBM had its full nominal budget on the workers that reached XGBoost; "
            "XGBoost was cut by the 6 h wall (7 of 8 workers TIMEOUT)."
        )
    for fam, d in h2.items():
        w(
            f"* Hoffman2 replica (same code, same split, separate journal; menu NOT shipped): "
            f"{fam} states {d['states']} ({d['first_start'][:16]} .. {d['last_complete'][:16]})."
        )
    w("")
    if bank_rows:
        w(
            "### The CARC menu (from the bank chunks' meta) and the post-2002 ranks of the shipped picks\n"
        )
        w(
            "| arm | entry | sha | dev screen MSE | MSE 2003+ | MSE 2020-24 | MSE 16:00 bar 2020-24 |"
        )
        w("|---|---|---|---|---|---|---|")
        for r in bank_rows:
            dv = (
                f"{r['dev_screen_mse']:.6f}"
                if r["dev_screen_mse"] is not None
                else "(Sobol)"
            )
            w(
                f"| {r['arm']} | {r['name']} | {r['sha']} | {dv} | {r['mse_2003_on']:.6f} | "
                f"{r['mse_2020_24']:.6f} | {r['mse_1600_2020_24']:.6f} |"
            )
        w("")
        names = {r["name"] for r in bank_rows}
        for nm in ("lgbm_opt_00", "xgb_opt_06"):
            if nm not in names:
                continue
            fam = nm.split("_")[0]
            parts = []
            for key, lab in (
                ("dev_screen_mse", "dev"),
                ("mse_2003_on", "2003+"),
                ("mse_2020_24", "2020-24"),
                ("mse_1600_2020_24", "16:00 bar 2020-24"),
            ):
                k, m = _rank_within(bank_rows, key, fam, nm)
                parts.append(f"{lab} {k}/{m}")
            w(
                f"* `{nm}` rank among the {fam} Optuna entries (1 = lowest MSE): "
                + ", ".join(parts)
                + "."
            )
        w("")
    w("### How the per-bar XGBoost config was picked\n")
    q16 = _qlike_ranks(bank, "tree_expert_16", ("xgb_opt",), bank_rows)
    if bank_rows:
        r16 = {
            lab: _rank_within(bank_rows, key, "xgb", "xgb_opt_06")
            for key, lab in (("mse_2003_on", "2003+"), ("mse_2020_24", "2020-24"))
        }
        r00 = _rank_within(bank_rows, "mse_2003_on", "lgbm", "lgbm_opt_00")
        perf = (
            f"arm 16 ranks {r16['2003+'][0]}/{r16['2003+'][1]} on 2003+ MSE and "
            f"{r16['2020-24'][0]}/{r16['2020-24'][1]} on 2020-24 MSE among the Optuna XGBoost arms"
            + (f" ({q16})" if q16 else "")
            + f", and arm 00 ranks {r00[0]}/{r00[1]} on 2003+ MSE among the LightGBM arms"
        )
    else:
        perf = "(bank chunks not available to this run: no post-2002 ranks)"
    w(
        "The per-bar LightGBM config is `lgbm_opt_00`, the top dev entry (the freeze rule).  The per-bar "
        "XGBoost config is `xgb_opt_06` = `tree_expert_16`, the 7th-ranked XGBoost dev entry, NOT the top "
        "one (`xgb_opt_00`, `tree_expert_10`).  It was inherited from the 15:30 deck's `yhat_tree16.parquet` "
        "(`specs/causal_tune_trees.py` HYPERPARAMETERS), whose two tree columns (arms 00 and 16) were "
        "hand-picked by 2026-08-14 (first appearance: harxhar-clean commit cd4d3cd, "
        "`experiments/spxw_complete_table.py`); the basis of the pick is recorded nowhere "
        "(`results/spxw_pnl/MANIFEST.md`).  The pick was made after the bank had been scored on "
        "2003+ data (tree-verdict commits of 2026-08-12/13), so it was not made under the dev-span-only "
        "rule and cannot be certified free of look-ahead.  It was not a selection FOR post-2002 "
        f"performance, though: {perf}.  Verdict: a deviation from the freeze rule with an unrecorded "
        "basis; nothing indicates that post-2002 data favoured it.\n"
    )
    w("## 2. The random forest through the same procedure\n")
    w(
        "Box (name, low, high, log, int); fixed "
        + json.dumps(RF_FIXED)
        + "; fits single-threaded "
        f"(n_jobs 1, random_state {SEED}), the {man['screen_blocks']} fits of a trial in a process pool:\n"
    )
    w("| param | low | high | log | int |")
    w("|---|---|---|---|---|")
    for b in RF_BOX:
        w(f"| {b[0]} | {b[1]} | {b[2]} | {b[3]} | {b[4]} |")
    w("")
    w(
        f"Dev matrices: rows [0, {man['dev_end_row']}) of `{man['source_cache']}` "
        f"({man['source_cache_mtime']}, shape {man['cache_shape']}); y equals the bank chunks' y_fit on "
        f"{man['y_rows_checked']:,} dev rows (max |diff| {man['y_vs_bank_y_fit_max_abs_diff']}); "
        f"sha256 X {man['sha256']['X'][:16]}..., y {man['sha256']['y'][:16]}....\n"
    )
    w(
        f"Trials: journal states at freeze time {states}; the freeze considers the first "
        f"{len(considered)} completed trials "
        f"({'numbered < ' + str(limit) if order == 'number' else 'completion order'}) -- "
        f"{note or 'the budget rule'} (nominal budget {BUDGET}).  CPU classes: {', '.join(cpus) or 'n/a'}.  Fit core-hours (sum of "
        f"per-fit seconds): {fit_core_h:.0f}; allocated CPU-hours (sacct, elapsed x CPUs, canary + "
        f"workers): {alloc_cpu_h:.0f}.\n"
    )
    if cost_res:
        w(
            f"Cost probe ({cost_res['cpu']}): mean single fit {cost_res['mean_fit_sec']:.0f} s over "
            f"{len(cost_res['draws'])} Sobol points of the box.\n"
        )
    w("### Frozen menu (top 8 + 2 Sobol)\n")
    w("| entry | trial | dev screen MSE | params |")
    w("|---|---|---|---|")
    for e in menu:
        dv = f"{e['dev_screen_mse']:.8f}" if "dev_screen_mse" in e else "(Sobol)"
        w(
            f"| {e['name']} | {e.get('trial', '')} | {dv} | `{json.dumps(e['params'], sort_keys=True)}` |"
        )
    w("")
    w("### The frozen random forest\n")
    w(
        f"* **LightGBM rule (top dev entry): `{top['name']}`** sha {top['sha']}, dev screen MSE "
        f"{top['dev_screen_mse']:.8f}, params `{json.dumps(top['params'], sort_keys=True)}`."
    )
    if alt:
        w(
            f"* The rule that picked XGBoost (7th-ranked entry, index {XGB_RULE_INDEX}) would give "
            f"`{alt['name']}`, dev {alt['dev_screen_mse']:.8f}, params "
            f"`{json.dumps(alt['params'], sort_keys=True)}` -- that rule has no recorded basis, so it is "
            "reported, not used."
        )
    if len(done_all) > len(considered):
        s = menu_all[0]
        w(
            f"* Over all {len(done_all)} trials completed at freeze time (beyond the budget rule) the "
            f"top entry is trial {s['trial']}, dev {s['dev_screen_mse']:.8f}, params "
            f"`{json.dumps(s['params'], sort_keys=True)}` (reported, not used)."
        )
    if menu_small:
        s = menu_small[0]
        w(
            f"* At XGBoost's realized Hoffman2 budget ({xgb_h2_complete} completed trials, completion "
            f"order) the top entry would be trial {s['trial']}, dev {s['dev_screen_mse']:.8f}, params "
            f"`{json.dumps(s['params'], sort_keys=True)}`."
        )
    w(
        "* Dev screen MSE of the shipped GBM entries: "
        + ", ".join(f"{k} {v['dev_screen_mse']:.8f}" for k, v in STORED.items())
        + f"; forest {top['dev_screen_mse']:.8f}."
    )
    w(
        f"* Per-bar use: the leaf minimum scaled by 2000/{TMD.WINDOW} as the spec does for the GBMs "
        f"would be min_samples_leaf {cfg['min_samples_leaf_scaled_to_2000_rows']} (not applied).\n"
    )
    w("## 3. Gate\n")
    gates = sorted(glob.glob(os.path.join(WORK, "*", "gate.json")))
    if not gates:
        w("No gate.json found.\n")
    for gp in gates:
        g = _read_json(gp)
        assert g is not None
        w(
            f"`{gp.replace(os.sep, '/')}`: pool screen re-run on {g['host']} [{g['cpu']}], {g['procs']} single-threaded "
            f"processes, versions {json.dumps(g['versions'])}:\n"
        )
        w("| entry | stored dev screen MSE | re-run | rel diff | pass |")
        w("|---|---|---|---|---|")
        for k, v in g["entries"].items():
            rel = "n/a" if v.get("rel_diff") is None else f"{v['rel_diff']:.2e}"
            rerun = v.get("rerun")
            rerun_s = repr(rerun) if rerun is not None else f"error: {v.get('error')}"
            req = " (required)" if v.get("required", k in GATE_REQUIRED) else ""
            w(f"| {k}{req} | {v['stored']!r} | {rerun_s} | {rel} | {v['pass']} |")
        w(
            f"\nTolerance: relative {GATE_RTOL:g}; required entries: {', '.join(GATE_REQUIRED)}.  "
            f"Overall: **{'informational' if g['pass'] is None else 'PASS' if g['pass'] else 'FAIL'}**"
            + (" (partial: the run did not finish)" if g.get("partial") else "")
            + ".\n"
        )
    w(
        "The stored values came from 8-thread fits on a mix of CPU classes (the CARC menu workers ran on "
        + _node_classes()
        + "); the re-runs use single-threaded fits.\n"
    )
    with open(
        os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("\n".join(L) + "\n")
    with open(
        os.path.join(out, "RF_CONFIG_DONE"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write(
            f"{top['name']} sha {top['sha']} dev {top['dev_screen_mse']!r} "
            f"trials {len(considered)} {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
    print(
        f"froze {len(menu)} entries -> {out}/menu_rf.json; config {top['name']} "
        f"dev {top['dev_screen_mse']:.8f} ({len(considered)} trials)"
    )


# ── smoke ─────────────────────────────────────────────────────────────────────


def smoke() -> None:
    params = {
        "n_estimators": 20,
        "max_depth": 4,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "max_samples": 0.5,
        **RF_FIXED,
    }
    _init_pool("__synthetic__")
    assert _X is not None and _Y is not None
    # sequential reference with tree_menu_dev's own loop (window 500, refit 250)
    sse, cnt = 0.0, 0
    for t0 in range(500, len(_Y), 250):
        t1 = min(t0 + 250, len(_Y))
        m = _model(FAMILY, params)
        m.fit(_X[t0 - 500 : t0], _Y[t0 - 500 : t0])
        sse += float(np.sum((_Y[t0:t1] - m.predict(_X[t0:t1])) ** 2))
        cnt += t1 - t0
    with _pool(2, "__synthetic__") as pool:
        v, _ = screen(pool, FAMILY, params, len(_Y), window=500, refit=250)
    assert v == sse / cnt, (v, sse / cnt)
    draws = TMD._sobol_draws(FAMILY)
    assert len(draws) == TMD.N_SOBOL_PER_FAMILY and all(d["bootstrap"] for d in draws)
    print(f"smoke OK: pool screen == sequential screen ({v:.6f}); Sobol draws {draws}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "mode", choices=["extract", "gate", "cost", "worker", "freeze", "smoke"]
    )
    ap.add_argument("--cache", default="")
    ap.add_argument("--bank", default="")
    ap.add_argument("--h2-journals", default=os.path.join(WORK, "h2_journals"))
    ap.add_argument("--journal", default=JOURNAL)
    ap.add_argument("--n-trials", type=int, default=PER_WORKER_TRIALS)
    ap.add_argument(
        "--cap", type=int, default=0, help="worker: prune trials numbered >= cap"
    )
    ap.add_argument("--order", choices=["completion", "number"], default="completion")
    ap.add_argument("--limit", type=int, default=BUDGET)
    ap.add_argument("--budget-note", default="", help="why this --limit (recorded)")
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--tag", default="carc")
    ap.add_argument("--entries", default=",".join(GATE_ENTRIES))
    ap.add_argument(
        "--out", default=OUT, help="freeze output folder (tests: elsewhere)"
    )
    a = ap.parse_args()
    if a.mode == "extract":
        extract(a.cache, a.bank)
    elif a.mode == "gate":
        if not gate(a.tag, _procs(a.procs), tuple(a.entries.split(","))):
            raise SystemExit("GATE FAILED")
    elif a.mode == "cost":
        cost(a.tag, _procs(a.procs))
    elif a.mode == "worker":
        run_worker(a.journal, a.n_trials, _procs(a.procs), a.cap)
    elif a.mode == "freeze":
        freeze(
            a.journal,
            a.limit,
            a.bank,
            a.h2_journals,
            a.tag,
            a.out,
            a.budget_note,
            a.order,
        )
    else:
        smoke()


if __name__ == "__main__":
    main()
