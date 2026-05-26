import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.evaluation.metrics import calculate_metrics
from src.backtest.executor import run_executor
from src.data.loading import parse_exog_cols
from src.features.scaling import run_backtest

DEFAULT_RIDGE_PARAMS: dict = dict(alpha=1.0)


def fit_predict_ridge(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward backtest with Ridge. Returns OOS predictions.

    Wraps :func:`src.features.scaling.run_backtest` with ``use_scaling=False``: the
    feature matrix is rolling-robust-scaled *whole-series* upstream (the
    shared scaffold calls ``rolling_robust_scale`` when ``prescale=True``),
    so the scaler's look-back is absorbed into data prep and the chunked
    walk-forward stays fungible with a single ``train_win`` halo. Refit
    cadence comes from ``hyperparams['_refit_frequency']``.

    Ridge's default solver is closed-form, so ``random_state`` is irrelevant
    for reproducibility. Internal control keys (``_*``) are stripped before
    forwarding to ``Ridge(**...)``.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", 1))
    model_kwargs = {k: v for k, v in hyperparams.items() if not k.startswith("_")}

    def model_fn():
        return Ridge(**model_kwargs)

    return run_backtest(
        model_fn,
        X_chunk,
        y_chunk,
        train_win=train_win_periods,
        refit_frequency=refit_frequency,
        use_scaling=False,
    )


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
        raise NotImplementedError("ridge run(): segmented backtests are a pending follow-up; pass segment=''")

    hyperparams: dict = dict(DEFAULT_RIDGE_PARAMS, alpha=alpha)
    if params_file:
        with open(params_file) as fh:
            hyperparams.update(json.load(fh))
    hyperparams["_refit_frequency"] = refit_frequency if refit_frequency is not None else 1

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
        prescale=True,
        seed=seed,
    )
    metrics = calculate_metrics(pd.read_csv(results_csv))
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
