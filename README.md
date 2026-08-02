# harxhar — realized-volatility forecasting

Forecasting next-30-min realized variance (`adj_RV`) from a HAR baseline plus optional
exogenous feature buckets, evaluated by walk-forward QLIKE. Runnable today via `run.py`;
`hpc-agent` consumes the same modules for chunked HPC execution.

**`src/` is the source of truth.** The modules under `src/` are the canonical, runnable
code — imported by `run.py` and on the cluster. The notebooks are *views* onto that code
(they `import` from `src/`, never the reverse); there is no notebook→`src` export step.

## Layout

```
src/                      canonical code (the source of truth)
├── data/loading.py         load_raw_data, parse_exog_cols, apply_overnight_fills
├── features/
│   ├── extractors/         har.py, calendar.py, spectral_embedding.py
│   └── transforms/         target.py (diurnal_adjust, winsorize, robust_transform),
│                           scaling.py (RollingRobustScaler), residualizer.py
├── models/                 ridge.py, xgboost.py, lightgbm.py, random_forest.py, knn.py,
│                           pcr.py, baseline.py, spectral_knn.py, patchts.py, ae_ridge.py,
│                           rolling_least_squares.py (incremental sliding-window Ridge)
├── backtest/               executor.py (run_executor scaffold), multi_stage.py
│                           (MultiStageBacktest), tune_tree.py (Optuna), dl_executor.py
└── evaluation/             metrics.py (Duan smearing, QLIKE/MSE/MAE, MZ), strategy.py

notebooks/results/         verification + weekly-findings notebooks (see below)
configs/                   one YAML per experiment (model + train_window + exog_cols + …)
data/                      input parquets (30-min bar data)
results/                   tracked summary CSVs; results/repro/ is local scratch (gitignored)
writeup/                   LaTeX paper + session notes
run.py                     entry point: python run.py --config configs/<name>.yaml
```

## Notebooks — verify, then interpret

The notebooks in `notebooks/results/` are self-contained, auditable views onto `src/`.
Each **shows the real `src/` code** (live `inspect.getsource`, collapsible) and **runs it**
— no pre-computed CSVs — and is structured **verify first, interpret second**:

- `ridge_pipeline_throughline.ipynb` — the full pipeline traced code→result: every working
  `src/` function shown and proven (incremental solver ≡ sklearn, no-hidden-code coverage,
  MZ calibration). The machinery, proven once.
- `bucket_sweep.ipynb` — each feature bucket as a real run reproducing the cluster, then
  the ranking interpretation.
- `window_ablation.ipynb` — the training-window sweep (HAR grid re-run live), then the
  curves and the ≈250 / ≈1000-day findings.

## Usage

```bash
pip install -r requirements.txt
python run.py --config configs/ridge.yaml
python run.py --config configs/xgboost.yaml --override train_window=750 seed=7
```

Every YAML has a top-level `model: <name>` naming a module under `src/models/`; `run.py`
imports it and calls `run(**params)` (ML / composed models) or `compute(args)` (DL). The
simple and composed models all configure the same `MultiStageBacktest` harness — only
their residualizer / feature / regressor stages differ. `--override key=value …` patches
the config (values parsed as YAML scalars).

## Key invariants

- **Target:** `winsorize( sqrt( RV / rolling-per-slot-diurnal-mean ) )`, forecast one bar
  ahead — the `.shift(1)` lives in feature construction (`horizon=1`).
- **QLIKE is raw-space** (Duan smearing, `evaluation/metrics.py`); MSE/MAE are sqrt-space —
  they do not move together.
- **Walk-forward, refit every bar** via the incremental `RollingLeastSquares` — exact to
  ~1e-11 vs a per-window sklearn refit (verified live in the through-line notebook).
- Features and target use the **intersection** of available timestamps; calendar + diurnal
  features are on for every model (Ridge included). **`SEED=42`** (DL pins numpy + torch +
  cudnn + PYTHONHASHSEED).

## hpc-agent consumption

`src/models/<x>.py` runs as plain Python today (whole-series, no chunking). `hpc-agent`
re-injects its runtime (`@register_run` + `current_slice()`) to make the same modules
chunk-aware without changing model code; `.hpc/` holds the generated `tasks.py` / wrappers.
