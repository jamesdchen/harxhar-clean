# harxhar — pre-hpc-agent-consumption snapshot

Realized-volatility forecasting experiment. Notebook-authored, runnable today via `run.py`; ready to be consumed by `hpc-agent` for chunked HPC execution.

## Layout

```
src/
├── data/               Iterate on data source / NaN policy
│   └── loading.py        load_raw_data, parse_exog_cols
├── features/           Iterate on lag scales / scaler variants
│   ├── transforms.py     diurnal_adjust, winsorize, generate_har_features
│   └── scaling.py        RollingRobustScaler, rolling_robust_scale
├── models/             Iterate on a model in isolation
│   ├── ridge.py          Ridge (closed-form, refit every step)
│   ├── xgboost.py        XGBoost
│   ├── lightgbm.py       LightGBM
│   ├── random_forest.py  Random forest
│   ├── pcr.py            PCA + Ridge
│   ├── baseline.py       Naive HAR-MA(125)
│   ├── patchts.py        PatchTST transformer (GPU)
│   └── ae_ridge.py       Autoencoder + Ridge (GPU)
├── backtest/           Wire data → features → model → eval
│   ├── executor.py       run_executor() — ML scaffold
│   ├── dl_executor.py    DL scaffold
│   └── tune_tree.py      Optuna loop
└── evaluation/         Scoring + strategy
    ├── metrics.py        Duan smearing, QLIKE / MSE / MAE
    └── strategy.py       PnL evaluation

notebooks/              Scratchpad mirroring src/ (data / features / models / …)
configs/                One YAML per model run
all30min/               Input parquets (30-min bar data)
data/                   Output staging
results/                Tracked summary CSVs
writeup/                LaTeX paper + figures
run.py                  Entry point: python run.py --config configs/<model>.yaml
```

## Usage

```bash
pip install -r requirements.txt
python run.py --config configs/ridge.yaml
python run.py --config configs/xgboost.yaml --override train_window=750 seed=7
```

`run.py` reads the YAML's `model:` field (e.g. `ridge`), imports
`src.models.<model>`, and calls its entry function (`run(**params)` for ML,
`compute(args)` for DL) with the remaining YAML keys as parameters.
`--override key=value ...` patches the config from the command line; values
are parsed as YAML scalars.

## hpc-agent consumption

This snapshot is the state *before* `hpc-agent` consumes it. After consumption,
`hpc-agent` will add:

- `.hpc/axes.yaml` — classified data axis (currently bounded-halo with
  `train_window * 48` look-back for all six `ml_*` executors)
- `.hpc/tasks.py`, `.hpc/cli.py` — generated chunk plan + dispatcher
- A vendored `hpc_agent` wheel (optional, for cluster pin)

The `src/models/<x>.py` executors already carry the inlined `hpc_agent.template`
runtime (`@register_run`, `current_slice()`, etc.) so they run standalone today
(whole-series; `current_slice()` defaults to the canonical 0/-1/0 slice) and
chunk-aware tomorrow under `hpc-agent`'s dispatcher.
