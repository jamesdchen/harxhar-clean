"""Single-experiment entry point.

Usage:
    python run.py --config configs/ridge.yaml
    python run.py --config configs/xgboost.yaml --override train_window=750 seed=7
    python run.py --config configs/spectral_knn.yaml

The YAML may have either:

* ``model: <name>`` — dispatch to ``src.models.<name>``'s ``run(...)`` (ML) or
  ``compute(args)`` (DL). All other top-level keys are forwarded as kwargs.
* ``pipeline: <name>`` — dispatch to one of the composed pipelines defined in
  this file (currently: ``spectral_knn``). Used when the experiment glues
  multiple ``src/`` modules together rather than calling a single model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _coerce(value: str) -> Any:
    """Parse a CLI override value (yaml-style) into a Python scalar."""
    return yaml.safe_load(value)


def _apply_overrides(params: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        params[key.strip()] = _coerce(raw.strip())
    return params


def _load_config(path: Path) -> dict:
    with path.open() as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: must be a mapping")
    if "model" not in config and "pipeline" not in config:
        raise SystemExit(f"{path}: must have a 'model' or 'pipeline' key")
    return config


# ─── Composed pipelines ─────────────────────────────────────────────────────


def _spectral_knn_pipeline(config: dict) -> dict:
    """Ridge residualizer + spectral embedding of residuals + kNN regression.

    Each piece is its own module — the composition is explicit here::

        residualizer  = Residualizer(Ridge)                    (src.features.residualizer)
        feature       = build_embedding                        (src.features.spectral_embedding)
        regressor     = KNeighborsRegressor (Gaussian weights) (src.models.knn helper)

    :class:`src.backtest.multi_stage.MultiStageBacktest` runs the walk-forward
    loop + refit bookkeeping; this function only does data prep + Duan smearing
    + result-saving around it.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    from src.backtest.executor import _build_har_and_calendar, load_and_transform
    from src.backtest.multi_stage import MultiStageBacktest
    from src.evaluation.metrics import calculate_metrics
    from src.features.transforms.residualizer import Residualizer
    from src.features.transforms.scaling import rolling_robust_scale
    from src.features.extractors.spectral_embedding import build_embedding
    from src.features.transforms.target import PERIODS_PER_DAY
    from src.models.knn import gaussian_weights

    # ── 1. Data prep ────────────────────────────────────────────────────────
    df, _ = load_and_transform(
        config["data_path"],
        exog_cols=[],
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=True,
    )
    df, feature_names = _build_har_and_calendar(df, exog_cols=[], add_calendar=True)
    df["target"] = df["adj_RV"].shift(-config["horizon"])
    df = df.dropna(subset=["target"] + feature_names).reset_index(drop=True)

    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64)
    baselines = df["baseline"].to_numpy(dtype=np.float64)
    dates = df["t"].to_numpy()

    train_win = int(config["train_window"]) * PERIODS_PER_DAY
    X_scaled = rolling_robust_scale(X, train_win)

    # ── 2. Composition: Ridge residualizer -> spectral_embedding -> kNN ─────
    seed = int(config["seed"])
    backtest = MultiStageBacktest(
        residualizer=Residualizer(lambda: Ridge(alpha=float(config["ridge_alpha"]))),
        feature_fit=lambda views: build_embedding(
            views,
            d=int(config["embedding_dim"]),
            k_graph=int(config["graph_k"]),
            seed=seed,
        ),
        regressor_factory=lambda: KNeighborsRegressor(
            n_neighbors=int(config["neighbor_k"]), weights=gaussian_weights
        ),
        view_window=int(config["view_window"]),
        refit_frequency=int(config["refit_frequency"]),
    )
    predictions = backtest.run(X_scaled, y, train_win, desc="spectral_knn")

    # ── 3. Duan smearing + save ─────────────────────────────────────────────
    y_test = y[train_win:]
    baselines_test = baselines[train_win:]
    dates_test = dates[train_win:]

    smear = float(np.mean((y_test - predictions) ** 2))
    pred_raw = (predictions**2 + smear) * baselines_test
    true_raw = (y_test**2) * baselines_test

    results = pd.DataFrame(
        {
            "date": dates_test,
            "horizon": config["horizon"],
            "true_adj": y_test,
            "pred_adj": predictions,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )
    results_csv = Path(config["output_file"]).with_name("results.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_csv, index=False)

    metrics = calculate_metrics(results)
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}


PIPELINES = {
    "spectral_knn": _spectral_knn_pipeline,
}


# ─── Entry point ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="experiment YAML")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values (parsed as YAML scalars)",
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    params = _apply_overrides(config, args.override)

    if "pipeline" in params:
        name = params.pop("pipeline")
        if name not in PIPELINES:
            raise SystemExit(f"unknown pipeline: {name!r} (known: {sorted(PIPELINES)})")
        result = PIPELINES[name](params)
        print(json.dumps(result, indent=2, default=str))
        return 0

    model = params.pop("model")
    module = importlib.import_module(f"src.models.{model}")

    if hasattr(module, "run"):
        result = module.run(**params)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if hasattr(module, "compute"):
        module.compute(SimpleNamespace(**params))
        return 0
    raise SystemExit(f"src.models.{model} exposes neither run(**params) nor compute(args)")


if __name__ == "__main__":
    raise SystemExit(main())
