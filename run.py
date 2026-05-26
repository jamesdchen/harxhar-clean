"""Single-experiment entry point.

Usage:
    python run.py --config configs/ridge.yaml
    python run.py --config configs/xgboost.yaml --override train_window=750 seed=7
    python run.py --config configs/spectral_knn.yaml

The YAML must have a top-level ``model:`` field naming a module under
``src/models/`` (e.g. ``ridge``, ``spectral_knn``, ``patchts``). All other
top-level keys are passed as kwargs to that module's entry function:
``run(...)`` for the ML / composed models, ``compute(args)`` for the DL
models. ``--override key=value ...`` patches the config from the command
line; values are parsed as YAML scalars.
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
    if not isinstance(config, dict) or "model" not in config:
        raise SystemExit(f"{path}: must be a mapping with a 'model' key")
    return config


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
    model = config.pop("model")
    params = _apply_overrides(config, args.override)

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
