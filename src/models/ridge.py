import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.data.loading import parse_exog_cols
from src.evaluation.metrics import calculate_metrics
from src.features.transforms.residualizer import IdentityResidualizer
from src.models.rolling_least_squares import RollingLeastSquares

DEFAULT_RIDGE_PARAMS: dict = dict(alpha=1.0)


def fit_predict_ridge(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward Ridge regression via :class:`MultiStageBacktest`.

    Plain single-stage model: ``IdentityResidualizer`` + no feature transform
    + ``Ridge`` regressor. The feature matrix is rolling-robust-scaled
    whole-series upstream (``prescale=True`` in :func:`run_executor`).
    Refit cadence comes from ``hyperparams['_refit_frequency']``.
    Internal control keys (``_*``) are stripped before forwarding to ``Ridge``.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", 1))
    incremental = bool(hyperparams.get("_incremental", True))
    model_kwargs = {k: v for k, v in hyperparams.items() if not k.startswith("_")}

    if incremental:
        # Rank-1 sliding-window solver: predictions identical to a per-window
        # Ridge refit, at O(p²)/bar instead of O(W·p²)/bar. MultiStageBacktest
        # detects the rolling protocol (init_window + roll) and takes its
        # incremental fast path. Falls back to sklearn Ridge for any model_kwarg
        # the rolling solver doesn't model (set incremental=False then).
        alpha = float(model_kwargs.get("alpha", 1.0))
        fit_intercept = bool(model_kwargs.get("fit_intercept", True))

        def make_regressor() -> Any:
            return RollingLeastSquares(alpha=alpha, fit_intercept=fit_intercept)
    else:

        def make_regressor() -> Any:
            return Ridge(**model_kwargs)

    backtest = MultiStageBacktest(
        residualizer=IdentityResidualizer(),
        regressor_factory=make_regressor,
        refit_frequency=refit_frequency,
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="ridge")


def run(
    horizon: int = 1,
    train_window: int = 500,
    refit_frequency: int | None = None,
    exog_cols: str = "",
    segment: str = "",
    lag_scope: str = "global",
    alpha: float = 1.0,
    seed: int = 42,
    data_path: str = "data",
    output_file: str = "results/ridge/run.json",
    params_file: str = "",
    overnight_fill: bool = True,
    impute_indicate: bool = False,
    diurnal_mode: str = "divide",
    incremental: bool = True,
) -> dict:
    """Ridge regression walk-forward volatility backtest.

    Returns a metrics dict. The per-row prediction table is written next to
    ``output_file`` as ``results.csv`` by the shared backtest scaffold.

    Ridge data-prep invariants are inline here: calendar features on,
    diurnal-adjusted RV target winsorized at a 240-period window, a
    leading-edge NaN drop (the closed-form solver rejects NaN), and
    ``prescale=True`` — the feature matrix is rolling-robust-scaled
    whole-series.
    """
    if segment:
        raise NotImplementedError(
            "ridge run(): segmented backtests are a pending follow-up; pass segment=''"
        )

    hyperparams: dict = dict(DEFAULT_RIDGE_PARAMS, alpha=alpha)
    if params_file:
        with open(params_file) as fh:
            hyperparams.update(json.load(fh))
    hyperparams["_refit_frequency"] = (
        refit_frequency if refit_frequency is not None else 1
    )
    hyperparams["_incremental"] = incremental

    results_csv = str(Path(output_file).with_name("results.csv"))
    run_executor(
        method_name="ridge",
        fit_predict=fit_predict_ridge,
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
        lag_scope=lag_scope,
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
