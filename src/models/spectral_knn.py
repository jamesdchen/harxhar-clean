"""Ridge residualizer + spectral embedding of residuals + kNN regression.

A non-trivial :class:`MultiStageBacktest` configuration — same shape as every
other ``src/models/<x>.py`` (the simple models just plug ``IdentityResidualizer``
+ ``None`` feature), only with non-degenerate residualizer and feature stages.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.evaluation.metrics import calculate_metrics
from src.features.extractors.spectral_embedding import build_embedding
from src.features.transforms.residualizer import Residualizer
from src.models.knn import gaussian_weights

DEFAULT_SPECTRAL_KNN_PARAMS: dict = dict(
    ridge_alpha=1.0,
    view_window=960,
    embedding_dim=8,
    graph_k=10,
    neighbor_k=25,
    refit_frequency=240,
)


def fit_predict_spectral_knn(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Spectral-kNN backtest as a MultiStageBacktest composition.

    Reads as the sentence "Ridge residualizer + spectral_embedding feature +
    Gaussian-weighted kNN regressor":

        residualizer  = Residualizer(Ridge)
        feature_fit   = build_embedding(views, d, k_graph)
        regressor     = KNeighborsRegressor(n_neighbors, weights=gaussian)

    Internal control keys (``_*``) are stripped before forwarding.
    """
    hp = {k: v for k, v in hyperparams.items() if not k.startswith("_")}
    seed = int(hp.get("seed", 42))
    backtest = MultiStageBacktest(
        residualizer=Residualizer(lambda: Ridge(alpha=float(hp["ridge_alpha"]))),
        regressor_factory=lambda: KNeighborsRegressor(
            n_neighbors=int(hp["neighbor_k"]), weights=gaussian_weights
        ),
        refit_frequency=int(hp["refit_frequency"]),
        feature_fit=lambda views: build_embedding(
            views,
            d=int(hp["embedding_dim"]),
            k_graph=int(hp["graph_k"]),
            seed=seed,
        ),
        view_window=int(hp["view_window"]),
    )
    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="spectral_knn")


def run(
    horizon: int = 1,
    train_window: int = 500,
    ridge_alpha: float = 1.0,
    view_window: int = 960,
    embedding_dim: int = 8,
    graph_k: int = 10,
    neighbor_k: int = 25,
    refit_frequency: int = 240,
    exog_cols: str = "",
    seed: int = 42,
    data_path: str = "data",
    output_file: str = "results/spectral_knn/run.json",
    params_file: str = "",
) -> dict:
    """Spectral-kNN walk-forward volatility backtest.

    Composition: Ridge residualizer + spectral embedding of residual windows
    + Gaussian-weighted kNN regression in embedding space. Returns a metrics
    dict. The per-row prediction table is written next to ``output_file`` as
    ``results.csv`` by the shared backtest scaffold.

    Data-prep invariants match the linear-method path (calendar features on,
    diurnal-adjusted RV target winsorized at a 240-period window, leading-edge
    NaN drop, ``prescale=True`` so view residuals are on a sane scale).
    """
    from src.data.loading import parse_exog_cols

    hp: dict = dict(
        DEFAULT_SPECTRAL_KNN_PARAMS,
        ridge_alpha=ridge_alpha,
        view_window=view_window,
        embedding_dim=embedding_dim,
        graph_k=graph_k,
        neighbor_k=neighbor_k,
        refit_frequency=refit_frequency,
        seed=seed,
    )
    if params_file:
        with open(params_file) as fh:
            hp.update(json.load(fh))

    results_csv = str(Path(output_file).with_name("results.csv"))
    run_executor(
        method_name="spectral_knn",
        fit_predict=fit_predict_spectral_knn,
        hyperparams=hp,
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
