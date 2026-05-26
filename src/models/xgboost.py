import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.data.loading import parse_exog_cols
from src.evaluation.metrics import calculate_metrics
from src.features.transforms.residualizer import IdentityResidualizer

DEFAULT_XGB_PARAMS: dict = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=-1,
)


def fit_predict_xgb(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward XGBoost regression via :class:`MultiStageBacktest`.

    Plain single-stage model: ``IdentityResidualizer`` + no feature transform
    + ``XGBRegressor``. Refit cadence from ``hyperparams['_refit_frequency']``.
    Internal control keys (``_*``) are stripped before forwarding to the
    model constructor.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", 1))
    model_kwargs = {k: v for k, v in hyperparams.items() if not k.startswith("_")}

    backtest = MultiStageBacktest(
        residualizer=IdentityResidualizer(),
        regressor_factory=lambda: XGBRegressor(**model_kwargs),
        refit_frequency=refit_frequency,
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="xgboost")


def run(
    horizon: int = 1,
    train_window: int = 500,
    refit_frequency: int | None = None,
    exog_cols: str = "",
    seed: int = 42,
    data_path: str = "data",
    output_file: str = "results/xgb/run.json",
    params_file: str = "",
) -> dict:
    """XGBoost walk-forward volatility backtest.

    Returns a metrics dict. The per-row prediction table is written next to
    ``output_file`` as ``results.csv`` by the shared backtest scaffold.
    Data-prep invariants are inline literals below.
    """
    hyperparams: dict = dict(DEFAULT_XGB_PARAMS)
    if params_file:
        with open(params_file) as fh:
            hyperparams.update(json.load(fh))
    hyperparams["_refit_frequency"] = refit_frequency if refit_frequency is not None else 1

    results_csv = str(Path(output_file).with_name("results.csv"))
    run_executor(
        method_name="xgboost",
        fit_predict=fit_predict_xgb,
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
        dropna_with_exog=False,
        seed=seed,
    )
    metrics = calculate_metrics(pd.read_csv(results_csv))
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
