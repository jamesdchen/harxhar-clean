"""Dev-span tree menu study + freeze — MIXED FAMILY (LightGBM + XGBoost).

Two Optuna studies over the SAME design / dev prefix / objective (author
expansion 2026-08-07): the LightGBM box and an XGBoost box (tree_method
'hist'). Objective = rolling-backtest mean sqrt-space MSE (screen metric —
it only RANKS candidates; final arms are scored under the full contract) on
the DEV SPAN ONLY: panel rows [0, first bar of 2003). Window 24000 bars,
refit every 250 bars IN THE SCREEN ONLY (cost containment, disclosed — the
final expert arms refit per bar). Design: IDENTICAL to the tuned penalized
linear arms' wide all_features basis (author correction 2026-08-07 — same
information set, richer hypothesis class; NO product block). Target: the
same y_fit winsorized sqrt-scale series every arm fits.

Workers run BOTH families sequentially, half the trial budget each (simplest
split; the two studies use SEPARATE journal files). Storage is Optuna
JournalStorage on a file backend (scratch-safe — NEVER sqlite on NFS).

Freeze (--freeze): experiments/tree_menu.json = per family, top-8 completed
trials by dev screen value (unique configs) + 2 deterministic Sobol draws
over that family's box (scipy qmc, scrambled, seed 20260807) -> K = 20
entries tagged {"family": "lgbm"|"xgb"}; sha = sha256 of the sorted-JSON
{family + params}.

Usage:
    python experiments/tree_menu_dev.py --n-trials 16     # worker (8 lgbm + 8 xgb)
    python experiments/tree_menu_dev.py --freeze          # emit the menu
    python experiments/tree_menu_dev.py --smoke           # synthetic CI
"""

import _bootstrap  # noqa: F401  (sys.path: repo root + experiments/)

import argparse
import hashlib
import json
import os

import numpy as np

DEV_END_DATE = np.datetime64("2003-01-01")  # dev span = rows strictly before
WINDOW = 24_000
SCREEN_REFIT = 250  # screen-only cadence (final arms are per-bar; disclosed)
MENU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_menu.json")
JOURNAL_DEFAULTS = {
    "lgbm": os.path.join("results", "tree_menu_journal_lgbm.log"),
    "xgb": os.path.join("results", "tree_menu_journal_xgb.log"),
}
N_TOP_PER_FAMILY = 8
N_SOBOL_PER_FAMILY = 2
SOBOL_SEED = 20260807

# Boxes (directives 2026-08-07). (name, low, high, log, is_int)
BOXES: dict[str, list[tuple[str, float, float, bool, bool]]] = {
    "lgbm": [
        ("num_leaves", 15, 255, True, True),
        ("learning_rate", 1e-2, 0.3, True, False),
        ("n_estimators", 50, 500, False, True),
        ("min_child_samples", 20, 500, True, True),
        ("feature_fraction", 0.5, 1.0, False, False),
        ("bagging_fraction", 0.5, 1.0, False, False),
        ("lambda_l1", 1e-8, 10.0, True, False),
        ("lambda_l2", 1e-8, 10.0, True, False),
    ],
    "xgb": [
        ("max_depth", 3, 10, False, True),
        ("learning_rate", 1e-2, 0.3, True, False),
        ("n_estimators", 50, 500, False, True),
        ("min_child_weight", 1, 100, True, False),
        ("subsample", 0.5, 1.0, False, False),
        ("colsample_bytree", 0.5, 1.0, False, False),
        ("reg_alpha", 1e-8, 10.0, True, False),
        ("reg_lambda", 1e-8, 10.0, True, False),
        ("gamma", 1e-8, 1.0, True, False),
    ],
}
# family-fixed params outside the search box
FIXED = {"lgbm": {"bagging_freq": 1}, "xgb": {"tree_method": "hist"}}


def _suggest(trial, family: str) -> dict:
    params = {}
    for name, lo, hi, log, is_int in BOXES[family]:
        if is_int:
            params[name] = trial.suggest_int(name, int(lo), int(hi), log=log)
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=log)
    params.update(FIXED[family])
    return params


def _sha(family: str, params: dict) -> str:
    blob = json.dumps({"family": family, **params}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _make_model(family: str, params: dict, n_threads: int):
    if family == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            **params, num_threads=n_threads, random_state=42, verbosity=-1
        )
    if family == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            **params, n_jobs=n_threads, random_state=42, verbosity=0
        )
    raise KeyError(f"unknown tree family '{family}'")


def _dev_matrices() -> tuple[np.ndarray, np.ndarray]:
    """(X, y) on the dev span only — the wide linear design + y_fit target."""
    from src.unification import _load_panel

    p = _load_panel()
    dev_end = int(np.searchsorted(p.t, DEV_END_DATE))
    if dev_end <= WINDOW:
        raise SystemExit(
            f"dev span too short: {dev_end} rows before {DEV_END_DATE} "
            f"(need > window {WINDOW})"
        )
    return (
        np.ascontiguousarray(p.X[:dev_end], dtype=np.float64),
        np.asarray(p.y[:dev_end], dtype=np.float64),
    )


def _screen_score(
    X: np.ndarray, y: np.ndarray, family: str, params: dict, n_threads: int
) -> float:
    """Rolling-backtest mean sqrt-space MSE on the dev span, refit-every-250."""
    n = len(y)
    sse, cnt = 0.0, 0
    for t0 in range(WINDOW, n, SCREEN_REFIT):
        t1 = min(t0 + SCREEN_REFIT, n)
        model = _make_model(family, params, n_threads)
        model.fit(X[t0 - WINDOW : t0], y[t0 - WINDOW : t0])
        pred = model.predict(X[t0:t1])
        sse += float(np.sum((y[t0:t1] - pred) ** 2))
        cnt += t1 - t0
    return sse / cnt


def _storage(journal_path: str):
    """Optuna JournalStorage on a file backend (NFS/scratch-safe, unlike sqlite)."""
    from optuna.storages import JournalStorage

    try:  # optuna >= 4 layout
        from optuna.storages.journal import JournalFileBackend as _Backend
    except ImportError:  # optuna 3.x name
        from optuna.storages import JournalFileStorage as _Backend
    os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
    return JournalStorage(_Backend(journal_path))


def run_worker(journals: dict[str, str], n_trials: int) -> None:
    import optuna

    n_threads = int(
        os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("NSLOTS", "1"))
    )
    X, y = _dev_matrices()
    print(f"dev span: X {X.shape}, eval bars {len(y) - WINDOW}, threads {n_threads}")
    per_family = max(1, n_trials // len(BOXES))
    for family in BOXES:  # both families, half budget each (simple split)
        study = optuna.create_study(
            study_name=f"tree_menu_dev_{family}",
            storage=_storage(journals[family]),
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(
            lambda tr, fam=family: _screen_score(
                X, y, fam, _suggest(tr, fam), n_threads
            ),
            n_trials=per_family,
            gc_after_trial=True,
        )


def _sobol_draws(family: str) -> list[dict]:
    """Deterministic Sobol draws over one family's box (seeded, reproducible)."""
    from scipy.stats import qmc

    box = BOXES[family]
    sampler = qmc.Sobol(d=len(box), scramble=True, seed=SOBOL_SEED)
    draws = []
    for row in sampler.random(N_SOBOL_PER_FAMILY):
        params = {}
        for (name, lo, hi, log, is_int), v in zip(box, row):
            if log:
                x = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))))
            else:
                x = float(lo + v * (hi - lo))
            params[name] = int(round(x)) if is_int else x
        params.update(FIXED[family])
        draws.append(params)
    return draws


def freeze(journals: dict[str, str]) -> None:
    """experiments/tree_menu.json: per family top-8 + 2 Sobol -> K=20 tagged."""
    import optuna

    menu = []
    for family in BOXES:
        study = optuna.load_study(
            study_name=f"tree_menu_dev_{family}", storage=_storage(journals[family])
        )
        done = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        done.sort(key=lambda t: t.value)
        fam_n, seen = 0, set()
        for t in done:
            params = dict(t.params)
            params.update(FIXED[family])
            h = _sha(family, params)
            if h in seen:
                continue
            seen.add(h)
            menu.append(
                {
                    "name": f"{family}_opt_{fam_n:02d}",
                    "family": family,
                    "params": params,
                    "sha": h,
                    "dev_screen_mse": float(t.value),
                }
            )
            fam_n += 1
            if fam_n == N_TOP_PER_FAMILY:
                break
        if fam_n < N_TOP_PER_FAMILY:
            print(
                f"WARNING: {family}: only {fam_n} unique trials (< {N_TOP_PER_FAMILY})"
            )
        for k, params in enumerate(_sobol_draws(family)):
            menu.append(
                {
                    "name": f"{family}_sobol_{k:02d}",
                    "family": family,
                    "params": params,
                    "sha": _sha(family, params),
                }
            )
    with open(MENU_PATH, "w", encoding="utf-8") as fh:
        json.dump(menu, fh, indent=1)
    print(f"froze {len(menu)} experts -> {MENU_PATH}")
    for e in menu:
        print(
            f"  {e['name']:14s} {e['family']}  sha={e['sha']}  dev={e.get('dev_screen_mse', '')}"
        )


def smoke() -> None:
    """Panel-free synthetic sanity: tiny random panel, both families."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1200, 8))
    y = X[:, 0] * 0.5 + rng.standard_normal(1200) * 0.1
    global WINDOW, SCREEN_REFIT
    win_orig, refit_orig = WINDOW, SCREEN_REFIT
    WINDOW, SCREEN_REFIT = 500, 250
    try:
        probes = {
            "lgbm": {
                "num_leaves": 15,
                "learning_rate": 0.1,
                "n_estimators": 50,
                "min_child_samples": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                **FIXED["lgbm"],
            },
            "xgb": {
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 50,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                **FIXED["xgb"],
            },
        }
        for family, params in probes.items():
            score = _screen_score(X, y, family, params, n_threads=1)
            assert np.isfinite(score), (family, score)
            print(f"synthetic screen OK [{family}]: mse {score:.5f}")
    finally:
        WINDOW, SCREEN_REFIT = win_orig, refit_orig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal-lgbm", default=JOURNAL_DEFAULTS["lgbm"])
    ap.add_argument("--journal-xgb", default=JOURNAL_DEFAULTS["xgb"])
    ap.add_argument("--n-trials", type=int, default=16, help="total (split per family)")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    journals = {"lgbm": args.journal_lgbm, "xgb": args.journal_xgb}
    if args.smoke:
        smoke()
    elif args.freeze:
        freeze(journals)
    else:
        run_worker(journals, args.n_trials)


if __name__ == "__main__":
    main()
