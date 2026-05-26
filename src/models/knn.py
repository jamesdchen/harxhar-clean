import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.data.loading import parse_exog_cols
from src.evaluation.metrics import calculate_metrics
from src.features.transforms.residualizer import IdentityResidualizer

DEFAULT_KNN_PARAMS: dict = dict(n_neighbors=25, weighting="gaussian")


def gaussian_weights(distances: np.ndarray) -> np.ndarray:
    """sklearn-compatible `weights` callable.

    ``distances`` arrives as shape ``(n_queries, k)``. We use a self-tuning
    bandwidth equal to the median neighbour distance per query, so the
    bandwidth adapts row-by-row to the local density.
    """
    sigma = np.median(distances, axis=1, keepdims=True) + 1e-12
    return np.exp(-(distances**2) / (2 * sigma**2))


def _resolve_weights(weighting: str) -> str | Callable[[np.ndarray], np.ndarray]:
    """Map the config's ``weighting`` string to a sklearn ``weights`` argument."""
    if weighting == "gaussian":
        return gaussian_weights
    if weighting in ("uniform", "distance"):
        return weighting
    raise ValueError(f"unknown kNN weighting: {weighting!r}")


def fit_predict_knn(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward kNN regression via :class:`MultiStageBacktest`.

    Plain single-stage model: ``IdentityResidualizer`` + no feature transform
    + ``KNeighborsRegressor``. The feature matrix is rolling-robust-scaled
    whole-series upstream (``prescale=True`` in :func:`run_executor`) so
    distances are sensible. Refit cadence from
    ``hyperparams['_refit_frequency']``; kNN's "refit" is just rebuilding the
    KDTree, so refitting every step is cheap. Internal control keys (``_*``)
    are stripped before forwarding to :class:`KNeighborsRegressor`.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", 1))
    model_kwargs = {k: v for k, v in hyperparams.items() if not k.startswith("_")}
    weighting = model_kwargs.pop("weighting", "gaussian")
    weights_arg = _resolve_weights(weighting)

    backtest = MultiStageBacktest(
        residualizer=IdentityResidualizer(),
        regressor_factory=lambda: KNeighborsRegressor(weights=weights_arg, **model_kwargs),
        refit_frequency=refit_frequency,
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="knn")


def run(
    horizon: int = 1,
    train_window: int = 500,
    refit_frequency: int | None = None,
    exog_cols: str = "",
    n_neighbors: int = 25,
    weighting: str = "gaussian",
    seed: int = 42,
    data_path: str = "data",
    output_file: str = "results/knn/run.json",
    params_file: str = "",
) -> dict:
    """Generic kNN walk-forward volatility backtest on HAR features.

    Returns a metrics dict. The per-row prediction table is written next to
    ``output_file`` as ``results.csv`` by the shared backtest scaffold.

    Data-prep invariants are inline literals below: calendar features on,
    diurnal-adjusted RV target winsorized at a 240-period window, leading-
    edge NaN drop, ``prescale=True`` so Euclidean distances in HAR space
    have comparable per-dimension scale.
    """
    hyperparams: dict = dict(DEFAULT_KNN_PARAMS, n_neighbors=n_neighbors, weighting=weighting)
    if params_file:
        with open(params_file) as fh:
            hyperparams.update(json.load(fh))
    hyperparams["_refit_frequency"] = refit_frequency if refit_frequency is not None else 1

    results_csv = str(Path(output_file).with_name("results.csv"))
    run_executor(
        method_name="knn",
        fit_predict=fit_predict_knn,
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
        prescale=True,
        seed=seed,
    )
    metrics = calculate_metrics(pd.read_csv(results_csv))
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
