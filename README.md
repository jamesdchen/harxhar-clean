# harxhar — pre-hpc-agent-consumption snapshot

Realized-volatility forecasting experiment. Notebook-authored, runnable today via `run.py`; ready to be consumed by `hpc-agent` for chunked HPC execution.

## Layout

```
src/                    Engine (data loading, transforms, scaling, evaluation, models)
  loading.py            Parquet load + market-hours filter + NaN policy
  transforms.py         Diurnal adjust, semantic transform, winsorize, HAR lags
  evaluation.py         Duan smearing, QLIKE / MSE / MAE
  scaling.py            Rolling robust scaler, walk-forward backtest kernel
  executor.py           Shared ML walk-forward scaffold
  dl_executor.py        DL helpers
  ml_ridge.py           Ridge executor
  ml_xgboost.py         XGBoost executor
  ml_lightgbm.py        LightGBM executor
  ml_pcr.py             PCA + Ridge executor
  ml_random_forest.py   Random forest executor
  ml_baseline.py        Naive HAR-MA(125) baseline
  dl_patchts.py         PatchTST transformer (GPU)
  dl_ae_ridge.py        Autoencoder + Ridge (GPU)
  strategy_eval.py      Strategy / PnL evaluation
  tune_tree.py          Optuna tuning loop for tree models

notebooks/              Scratchpad: pipeline / executors / audits / scripts
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

`run.py` reads the YAML's `model:` field (e.g. `ml_ridge`), imports `src.<model>`,
and calls its entry function (`run(**params)` for ML, `compute(args)` for DL) with
the remaining YAML keys as parameters. `--override key=value ...` patches the
config from the command line; values are parsed as YAML scalars.

## hpc-agent consumption

This snapshot is the state *before* `hpc-agent` consumes it. After consumption,
`hpc-agent` will add:

- `.hpc/axes.yaml` — classified data axis (currently bounded-halo with
  `train_window * 48` look-back for all six `ml_*` executors)
- `.hpc/tasks.py`, `.hpc/cli.py` — generated chunk plan + dispatcher
- A vendored `hpc_agent` wheel (optional, for cluster pin)

The `src/ml_*.py` executors already carry the inlined `hpc_agent.template`
runtime (`@register_run`, `current_slice()`, etc.) so they run standalone today
(whole-series; `current_slice()` defaults to the canonical 0/-1/0 slice) and
chunk-aware tomorrow under `hpc-agent`'s dispatcher.
