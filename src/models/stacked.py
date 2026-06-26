"""Residualized stacking model — Ridge base + a non-negative stack of tree learners.

The "elastic" model: a dense linear core (Ridge) captures the weak-but-dense
signal (KNS-style), and a **non-negative ridge stack** over heterogeneous tree
learners on the *residual* supplies the nonlinear correction — with the blend
weights chosen by the data, so it interpolates between the tail-chasing (boosting)
and variance-controlled (RF) ends instead of committing to one.

    final = ridge.predict(X)  +  Σ_k w_k · base_k(X → y − ridge)        w_k ≥ 0

Architecture (reuses the existing pipeline unchanged):
    MultiStageBacktest(
        residualizer = Residualizer(Ridge),          # the linear core
        regressor    = StackingResidualRegressor,    # the elastic residual stack
    )

The stack is **causal and leakage-free by construction**:

* The walk-forward already gives out-of-sample base predictions (each base is
  trained on ``[t−train_win, t)`` and predicts ``t``), so the *backtest* never
  leaks. Within a single ``fit`` the meta-weights are learned on a held-out
  *tail* of the training window (bases trained on the earlier part, predicting
  the tail) — the classic stacking out-of-fold requirement — with an optional
  embargo gap to blunt autocorrelation bleed across the split.
* Non-negativity (NNLS on a ridge-augmented design = non-negative ridge) keeps
  the meta from taking unstable extrapolating combinations, and lets it shrink a
  base to weight 0 — including *all* bases to 0, i.e. fall back to pure Ridge.
  That zero-floor is the elastic knob: the data decides how much nonlinear
  correction (and which flavor) to apply.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.data.loading import parse_exog_cols
from src.evaluation.metrics import calculate_metrics
from src.features.transforms.residualizer import Residualizer

# Defaults reflect what the COVID-slice sweeps actually chose: the tuner drives
# boosting hard toward the robust/RF-like end (low lr, heavy column subsampling,
# large leaves), so the lgbm base starts there and the rf base complements it.
DEFAULT_STACKED_PARAMS: dict = dict(
    ridge_alpha=1.0,
    holdout_frac=0.25,
    embargo=0,
    meta_alpha=1e-3,
    bases=["lgbm", "rf"],
    lgbm=dict(
        n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.5,
        n_jobs=4, verbose=-1, subsample_freq=1, random_state=42,
    ),
    rf=dict(
        n_estimators=300, max_depth=12, min_samples_leaf=20, max_features=0.5,
        n_jobs=4, random_state=42,
    ),
    xgb=dict(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.5, tree_method="hist", n_jobs=4,
    ),
)


def _make_base_factories(names: list[str], cfgs: dict) -> list[Callable[[], Any]]:
    """Build zero-arg factories for the named base learners (c bound by value)."""
    facs: list[Callable[[], Any]] = []
    for nm in names:
        c = dict(cfgs.get(nm, {}))
        if nm == "lgbm":
            from lightgbm import LGBMRegressor

            facs.append(lambda c=c: LGBMRegressor(**c))
        elif nm == "rf":
            from sklearn.ensemble import RandomForestRegressor

            facs.append(lambda c=c: RandomForestRegressor(**c))
        elif nm == "xgb":
            from xgboost import XGBRegressor

            facs.append(lambda c=c: XGBRegressor(**c))
        else:
            raise ValueError(f"unknown base learner: {nm!r}")
    return facs


class StackingResidualRegressor:
    """Sklearn-style non-negative ridge stack over base learners (time-series-safe).

    ``fit(X, y)`` trains each base on the head of the window, gets out-of-fold
    predictions on the held-out tail, solves non-negative ridge meta-weights on
    those (NNLS on a ridge-augmented design), then refits the bases on the full
    window for deployment. ``predict(X)`` returns the non-negative weighted blend.

    Parameters
    ----------
    base_factories
        Zero-arg callables each returning a fresh sklearn-style regressor.
    holdout_frac
        Fraction of the (time-ordered) training window held out as the tail on
        which the meta-weights are fit.
    embargo
        Rows skipped between the base-training head and the meta-fit tail, to
        blunt autocorrelation bleed across the split.
    meta_alpha
        L2 penalty on the meta-weights (ridge-augmentation of the NNLS design).
    weight_log
        Optional list; if given, each ``fit`` appends its solved weight vector
        (for diagnostics — "where does rf earn weight?"). Off the hot path.
    """

    def __init__(
        self,
        base_factories: list[Callable[[], Any]],
        holdout_frac: float = 0.25,
        embargo: int = 0,
        meta_alpha: float = 1e-3,
        weight_log: list | None = None,
    ) -> None:
        self.base_factories = base_factories
        self.holdout_frac = float(holdout_frac)
        self.embargo = int(embargo)
        self.meta_alpha = float(meta_alpha)
        self.weight_log = weight_log
        self.bases_: list[Any] = []
        self.w_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingResidualRegressor":
        from scipy.optimize import nnls

        n = len(X)
        k = len(self.base_factories)
        h = max(1, int(round(n * (1.0 - self.holdout_frac))))
        Xho, yho = X[h + self.embargo :], y[h + self.embargo :]

        if len(Xho) < max(8, k + 1):
            # Window too short for an honest meta-fit: equal weights, full refit.
            self.bases_ = [f().fit(X, y) for f in self.base_factories]
            self.w_ = np.full(k, 1.0 / k)
        else:
            oof = np.column_stack([f().fit(X[:h], y[:h]).predict(Xho) for f in self.base_factories])
            # Non-negative ridge meta: NNLS on [oof; sqrt(α)·I] → [yho; 0].
            A = np.vstack([oof, np.sqrt(self.meta_alpha) * np.eye(k)])
            b = np.concatenate([yho, np.zeros(k)])
            self.w_, _ = nnls(A, b)
            # Refit the bases on the full window for the actual prediction.
            self.bases_ = [f().fit(X, y) for f in self.base_factories]

        if self.weight_log is not None:
            self.weight_log.append(self.w_.copy())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = np.column_stack([b.predict(X) for b in self.bases_])
        return P @ self.w_


def fit_predict_stacked(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward residualized-stack via :class:`MultiStageBacktest`.

    ``Residualizer(Ridge)`` supplies the linear base every step;
    ``StackingResidualRegressor`` supplies the non-negative tree-stack correction
    on the residual every ``hyperparams['_refit_frequency']`` steps. Features are
    rolling-robust-scaled whole-series upstream (``prescale=True``); the Ridge
    base needs the scaling, the trees are scale-invariant. Internal control keys
    (``_*``) are stripped; an optional ``_weight_log`` list collects meta-weights.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", 1))
    weight_log = hyperparams.get("_weight_log")
    hp = {k: v for k, v in hyperparams.items() if not k.startswith("_")}

    ridge_alpha = float(hp.get("ridge_alpha", 1.0))
    holdout_frac = float(hp.get("holdout_frac", 0.25))
    embargo = int(hp.get("embargo", 0))
    meta_alpha = float(hp.get("meta_alpha", 1e-3))
    bases = list(hp.get("bases", ["lgbm", "rf"]))
    base_factories = _make_base_factories(bases, {nm: hp.get(nm, {}) for nm in bases})

    backtest = MultiStageBacktest(
        residualizer=Residualizer(lambda: Ridge(alpha=ridge_alpha)),
        regressor_factory=lambda: StackingResidualRegressor(
            base_factories, holdout_frac, embargo, meta_alpha, weight_log
        ),
        refit_frequency=refit_frequency,
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="stacked")


def run(
    horizon: int = 1,
    train_window: int = 500,
    refit_frequency: int | None = None,
    exog_cols: str = "",
    seed: int = 42,
    data_path: str = "data",
    output_file: str = "results/stacked/run.json",
    params_file: str = "",
    overnight_fill: bool = True,
    impute_indicate: bool = True,
    diurnal_mode: str = "divide",
) -> dict:
    """Residualized-stack walk-forward volatility backtest.

    Returns a metrics dict; the per-row prediction table is written next to
    ``output_file`` as ``results.csv``. Like Ridge: calendar features on,
    diurnal-adjusted RV winsorized at 240, ``prescale=True`` (the Ridge base
    needs scaled features). ``impute_indicate=True`` by default so the Ridge
    base sees no NaN while keeping the full sample.
    """
    hyperparams: dict = dict(DEFAULT_STACKED_PARAMS)
    if params_file:
        with open(params_file) as fh:
            hyperparams.update(json.load(fh))
    hyperparams["_refit_frequency"] = refit_frequency if refit_frequency is not None else 1

    results_csv = str(Path(output_file).with_name("results.csv"))
    run_executor(
        method_name="stacked",
        fit_predict=fit_predict_stacked,
        hyperparams=hyperparams,
        data_path=data_path,
        output_file=results_csv,
        horizon=horizon,
        train_window=train_window,
        start=0,
        end=-1,
        halo=0,
        exog_cols=parse_exog_cols(exog_cols or None),
        segment=None,
        lag_scope="global",
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=True,
        overnight_fill=overnight_fill,
        impute_indicate=impute_indicate,
        diurnal_mode=diurnal_mode,
        prescale=True,
        seed=seed,
    )
    metrics = calculate_metrics(pd.read_csv(results_csv))
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
