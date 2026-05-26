import numpy as np
import pandas as pd

from src.evaluation.metrics import apply_duan_smearing, calculate_metrics
from src.backtest.executor import load_and_transform
from src.features.transforms import (
    PERIODS_PER_DAY,
    apply_horizon_shift,
    generate_har_features,
    resolve_har_lags,
)


def run(
    horizon: int = 1,
    train_window: int = 500,
    data_path: str = "data",
    output_file: str = "results/baseline/run.json",
) -> dict:
    """Naive HAR-MA(125) baseline volatility forecast.

    The forecast is just the 125-period moving-average HAR feature; there is
    no fitted model, so no seed. Returns a metrics dict; writes the per-row
    ``results.csv`` next to ``output_file``.
    """
    train_win_periods = train_window * PERIODS_PER_DAY

    df, _ = load_and_transform(
        data_path,
        exog_cols=[],
        target_use_diurnal=False,
        target_winsor_window=None,
        dropna_with_exog=False,
    )
    df, feature_names = generate_har_features(df, target_col="adj_RV")
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon)

    if train_win_periods >= len(X):
        raise ValueError(f"train_window ({train_win_periods} periods) >= series length ({len(X)})")

    lag_125_index = feature_names.index("har_ma_125")
    oos_start = train_win_periods
    X_oos = X[oos_start:]
    y_oos = y[oos_start:]
    dates_oos = dates.iloc[oos_start:].values
    baselines_oos = baselines[oos_start:]
    preds = X_oos[:, lag_125_index]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )
    results_csv = Path(output_file).with_name("results.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_csv, index=False)
    metrics = calculate_metrics(results)
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
